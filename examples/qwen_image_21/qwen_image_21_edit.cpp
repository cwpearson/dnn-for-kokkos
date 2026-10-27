// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 one-shot reference-image editing driver.
//
// Wires the Qwen-Image-2.1-Turbo edit pipeline stage by stage, mirroring the
// diffusers QwenImage21Pipeline reference (text-to-image + image-conditioned
// __call__ with use_kv_cache=True):
//
//   load RGBA condition image -> derive geometry from its aspect (reference
//   calculate_dimensions at --output-resolution, /32 rounding) -> resized
//   vision copy (float Lanczos) + bit-exact VAE copy (PIL-fixed-point
//   Lanczos replica of preprocess()) -> joint text+vision
//   encode (edit template below) -> the VAE-encoded condition latents
//   (--ref-latents-in override, else in-process VAE encode of the shared
//   resized condition image, else zeros) are staged as packed cond tokens
//   alongside the target noise -> FlowEuler sample with the DiT
//   image-condition path ([cond|target] + img_shapes/img_mask) -> VAE decode.
//
// ONE-SHOT ONLY: no interactive loop; one edit per invocation (unlike
// qwen_image_21_generate.cpp, which loops on a TTY without --prompt).
//
// Joint text+vision encode is wired in-process: --tokenizer (processor dir
// with tokenizer.json) + --text-encoder (single-shard model.safetensors)
// + --image runs the raw-text BPE of the edit template (the slot's single
// image_pad placeholder expanded in place to Nv vision ids, processor
// convention), the Qwen3-VL vision tower over the packed pixel rows, and the
// text tower forward_joint (slot replacement, true M-RoPE, DeepStack),
// dropping the system block. --prompt-embeds-in (joint torch dump) and
// null zeros remain as fallbacks with logged reasons.
//   * VAE encode (image -> (mode-mean)/std latents) runs in-process via
//     QwenImage21VaeModel::encode when --vae-weights is given:
//     --ref-latents-in carries torch/golden-encoded reference latents
//     ([C,H,W] chw or torch [H*W,C] token, validated against the condition
//     grid) as an override, else zeros. Vision-slot rows come from the
//     joint build (minus drop_idx) or --vis-rows-in; without either, the
//     run falls back to target-only sampling with a logged reason.
//
// Shared implementation (EncoderStage, sample loop, DiT adapters, VAE
// decode, image writer, parity dumps) comes from
// qwen_image_21_common.hpp, so the two binaries cannot drift apart.
//
//   qwen_image_21_edit --image in.png --prompt "make it photorealistic"
//                       [--output-resolution 1024]
//                       [--height H --width W]  (override output; /32 snap)
//                       [--tokenizer tok_dir --text-encoder enc_dir]
//                       [--prompt-embeds-in pe.f32]
//                       [--ref-latents-in ref.f32
//                        [--ref-latents-layout chw|token]]
//                       [--weights transformer_dir] [--init-latents-in n.f32]
//                       [--steps 8] [--cfg 1.0] [--shift 3.0]
//                       [--sigmas driver|turbo] [--seed 42]
//                       [--vae-weights vae.safetensors [--no-vae-decode]]
//                       [--out edit.png] [--staged] [--dump-steps DIR]
//                       [--tiny]
#include "qwen_image_21_common.hpp"

#ifdef KOKKOSDNN_ENABLE_IMAGE
#if __has_include("KokkosDNN_Image.hpp")
#include "KokkosDNN_Image.hpp"
#elif __has_include("images/KokkosDNN_Image.hpp")
#include "images/KokkosDNN_Image.hpp"
#else
#error "KOKKOSDNN_ENABLE_IMAGE is set but no KokkosDNN_Image header was found"
#endif
#endif

namespace {

// ---- Reference edit wiring (diffusers QwenImage21Pipeline) -----------------
// Tokenizer-tied template builders: only consumed by the joint in-process
// encode, so they are compiled out when the BPE tables are unavailable.
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER) && defined(KOKKOSDNN_ENABLE_IMAGE)
// System prompt shared with the T2I template (models/qwen_image_21*
// encode the same fixed system block; drop_idx is its token count).
inline std::string edit_sys_prompt() {
  return "Comprehend and analyze the provided prompt.";
}

// One vision slot, exactly as the reference template spells it.
inline std::string edit_vision_slot(int i) {
  return "<image" + std::to_string(i) +
         "><|vision_start|><|image_pad|><|vision_end|>";
}

// Edit head: system block + user block with n_images vision slots, matching
// the reference prompt_template_ti2i (slots joined by single spaces).
inline std::string qwen21_edit_head(int n_images) {
  if (n_images <= 0)
    throw std::invalid_argument("qwen21_edit_head: need >= 1 image");
  std::string slots = edit_vision_slot(1);
  for (int i = 2; i <= n_images; ++i) slots += " " + edit_vision_slot(i);
  return "<|im_start|>system\n" + edit_sys_prompt() +
         "<|im_end|>\n<|im_start|>user\n" + slots;
}

inline std::string qwen21_edit_tail() {
  return "<|im_end|>\n<|im_start|>assistant\n";
}
#endif  // HAVE_ENCODER && HAVE_TOKENIZER && IMAGE (joint path)

// Reference smart_resize (transformers Qwen2VLImageProcessor, reused by the
// Qwen3-VL processor: image_processor_type Qwen2VLImageProcessorFast):
// snap each side to a `factor` multiple (factor = patch_size * merge_size
// = 32 here), then clamp the pixel count to [min_pixels, max_pixels] from
// processor/processor.json (65536 / 16777216 for this checkpoint). The
// pipeline derives BOTH the vision/VAE input size and the default output
// size from this. Image path only, compiled out when the image component
// is off.
#ifdef KOKKOSDNN_ENABLE_IMAGE
inline void edit_calculate_dimensions(int target_area, double ratio,
                                       int& out_w, int& out_h) {
  if (target_area <= 0)
    throw std::invalid_argument("edit geometry: target_area must be > 0");
  if (!(ratio > 0.0) || !std::isfinite(ratio))
    throw std::invalid_argument("edit geometry: aspect ratio must be > 0");
  // Python round(): round-half-to-even, matching the reference exactly.
  auto py_round = [](double v) {
    const double f = std::floor(v), d = v - f;
    if (d < 0.5) return (int)f;
    if (d > 0.5) return (int)f + 1;
    return ((long long)f % 2 == 0) ? (int)f : (int)f + 1;
  };
  const double w = std::sqrt((double)target_area * ratio);
  const double h = w / ratio;
  out_w = py_round(w / 32.0) * 32;
  out_h = py_round(h / 32.0) * 32;
}

// PIL-matching Lanczos-3 separable resize of an HWC float image.
// Mirrors Pillow's Resample.c precompute_coeffs for LANCZOS (support 3,
// stretched by max(scale,1) on downsample): center (d+0.5)*scale, taps at
// (i-center+0.5)/filterscale, clipped (not replicated) edges renormalized
// per output. This matches the reference VaeImageProcessor resize
// (PIL LANCZOS, no reducing_gap); the processor BICUBIC stage is a
// same-size no-op. Host-side, no file I/O. (The edit_bicubic_* name is
// kept for call-site stability; the kernel is Lanczos, not Keys bicubic.)
inline KokkosDNN::Image::HostImageF32 edit_bicubic_resize_hwc(
    const KokkosDNN::Image::HostImageF32& in, int out_h, int out_w) {
  namespace KI = KokkosDNN::Image;
  const size_t ih = in.extent(0), iw = in.extent(1), ch = in.extent(2);
  if (ih == 0 || iw == 0 || ch == 0 || out_h <= 0 || out_w <= 0)
    throw std::invalid_argument("edit lanczos: bad dims");
  constexpr double kPi = 3.14159265358979323846;
  const auto lanczos3 = [kPi](double x) {
    if (x <= -3.0 || x >= 3.0) return 0.0;
    if (x == 0.0) return 1.0;
    const double px = kPi * x;
    const double q = px / 3.0;
    return (std::sin(px) / px) * (std::sin(q) / q);
  };
  struct AxisTaps {
    std::vector<int> lo;     // per-output first input index
    std::vector<int> count;  // per-output tap count
    std::vector<double> w;   // per-output ksize weights
    int ksize = 0;
  };
  const auto build_axis = [&lanczos3](size_t in_n, int out_n) {
    AxisTaps ax;
    const double scale = (double)in_n / (double)out_n;
    const double fscale = scale < 1.0 ? 1.0 : scale;
    const double support = 3.0 * fscale;
    ax.ksize = (int)std::ceil(support) * 2 + 1;
    ax.lo.assign((size_t)out_n, 0);
    ax.count.assign((size_t)out_n, 0);
    ax.w.assign((size_t)out_n * (size_t)ax.ksize, 0.0);
    const double inv = 1.0 / fscale;
    for (int d = 0; d < out_n; ++d) {
      const double center = ((double)d + 0.5) * scale;
      int xmin = (int)(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      int xmax = (int)(center + support + 0.5);
      if (xmax > (int)in_n) xmax = (int)in_n;
      xmax -= xmin;
      double ww = 0.0;
      for (int k = 0; k < xmax; ++k) {
        const double wv = lanczos3(((double)(k + xmin) - center + 0.5) * inv);
        ax.w[(size_t)d * (size_t)ax.ksize + (size_t)k] = wv;
        ww += wv;
      }
      if (ww != 0.0)
        for (int k = 0; k < xmax; ++k)
          ax.w[(size_t)d * (size_t)ax.ksize + (size_t)k] /= ww;
      ax.lo[(size_t)d] = xmin;
      ax.count[(size_t)d] = xmax;
    }
    return ax;
  };
  const AxisTaps ax = build_axis(iw, out_w);
  const AxisTaps ay = build_axis(ih, out_h);
  KI::HostImageF32 tmp("qwen21edit_lanczos_tmp", ih, (size_t)out_w, ch);
  for (size_t y = 0; y < ih; ++y)
    for (int x = 0; x < out_w; ++x) {
      const int xmin = ax.lo[(size_t)x], n = ax.count[(size_t)x];
      const double* kw = &ax.w[(size_t)x * (size_t)ax.ksize];
      for (size_t c = 0; c < ch; ++c) {
        double acc = 0.0;
        for (int k = 0; k < n; ++k)
          acc += kw[k] * (double)in(y, (size_t)(xmin + k), c);
        tmp(y, (size_t)x, c) = (float)acc;
      }
    }
  KI::HostImageF32 out("qwen21edit_lanczos", (size_t)out_h, (size_t)out_w, ch);
  for (int y = 0; y < out_h; ++y) {
    const int ymin = ay.lo[(size_t)y], n = ay.count[(size_t)y];
    const double* kw = &ay.w[(size_t)y * (size_t)ay.ksize];
    for (int x = 0; x < out_w; ++x)
      for (size_t c = 0; c < ch; ++c) {
        double acc = 0.0;
        for (int k = 0; k < n; ++k)
          acc += kw[k] * (double)tmp((size_t)(ymin + k), (size_t)x, c);
        out((size_t)y, (size_t)x, c) = (float)acc;
      }
  }
  return out;
}

// Bit-exact replica of Pillow's LANCZOS separable RGBA resize (Pillow
// src/libImaging/Resample.c: precompute_coeffs + normalize_coeffs_8bpc +
// _ImagingResampleHorizontal/Vertical_8bpc, 4-band branch). The reference
// VAE preprocess() resizes the RGBA condition image with PIL LANCZOS in
// the uint8 domain (no reducing_gap), so its pixels are uint8-exact; the
// float-domain Lanczos above lands within ~2 LSB, which the VAE encoder
// amplifies past the gate (rel 8.25e-2 vs 3e-2). The VAE copy is routed
// through this replica (uint8 in/out, [H,W,4]) to make the pre-encode
// pixels bit-identical to the reference. Verified: zero differing bytes
// vs the pipeline resize on the 3200x1120 -> 1728x608 boat fit.
// Pillow details mirrored here: double filter weights renormalized per
// output, quantized to 22-bit fixed point (round-half-away-from-zero),
// per-pass accumulation from a 2^21 rounding offset with arithmetic
// shift + clamp to uint8 (so each pass rounds exactly like PIL), and the
// horizontal-first/vertical-first order rule. Host-side, no file I/O.
inline KokkosDNN::Image::HostImageU8 edit_pil_resize_rgba_u8(
    const KokkosDNN::Image::HostImageU8& in, int out_h, int out_w) {
  namespace KI = KokkosDNN::Image;
  const size_t ih = in.extent(0), iw = in.extent(1);
  if (in.extent(2) != 4 || ih == 0 || iw == 0 || out_h <= 0 || out_w <= 0)
    throw std::invalid_argument("edit pil resize: bad dims");
  constexpr double kPi = 3.14159265358979323846;
  constexpr int kFixBits = 22;
  constexpr int64_t kFixRound = int64_t{1} << (kFixBits - 1);
  const auto lanczos3 = [kPi](double x) {
    if (x <= -3.0 || x >= 3.0) return 0.0;
    if (x == 0.0) return 1.0;
    const double px = kPi * x;
    const double q = px / 3.0;
    return (std::sin(px) / px) * (std::sin(q) / q);
  };
  struct AxisTaps {
    std::vector<size_t> lo;     // per-output first input index
    std::vector<size_t> count;  // per-output tap count
    std::vector<int32_t> kvals;  // per-output fixed-point weights, row-major
    size_t ksize = 0;
  };
  const auto build_axis = [&lanczos3](size_t in_n, int out_n) {
    AxisTaps ax;
    const double scale =
        static_cast<double>(in_n) / static_cast<double>(out_n);
    const double fscale = scale < 1.0 ? 1.0 : scale;
    const double support = 3.0 * fscale;
    ax.ksize = static_cast<size_t>(std::ceil(support)) * 2 + 1;
    ax.lo.assign(static_cast<size_t>(out_n), 0);
    ax.count.assign(static_cast<size_t>(out_n), 0);
    ax.kvals.assign(static_cast<size_t>(out_n) * ax.ksize, 0);
    const double inv = 1.0 / fscale;
    const double fix = static_cast<double>(int64_t{1} << 22);
    std::vector<double> wsum;
    for (int d = 0; d < out_n; ++d) {
      const double center = (static_cast<double>(d) + 0.5) * scale;
      int xmin = static_cast<int>(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      int xmax = static_cast<int>(center + support + 0.5);
      if (xmax > static_cast<int>(in_n)) xmax = static_cast<int>(in_n);
      xmax -= xmin;
      wsum.assign(static_cast<size_t>(xmax), 0.0);
      double ww = 0.0;
      for (int k = 0; k < xmax; ++k) {
        const double wv = lanczos3((static_cast<double>(k + xmin) - center +
                                    0.5) *
                                   inv);
        wsum[static_cast<size_t>(k)] = wv;
        ww += wv;
      }
      if (ww != 0.0)
        for (int k = 0; k < xmax; ++k) wsum[static_cast<size_t>(k)] /= ww;
      for (int k = 0; k < xmax; ++k) {
        const double wk = wsum[static_cast<size_t>(k)] * fix;
        ax.kvals[static_cast<size_t>(d) * ax.ksize + static_cast<size_t>(k)] =
            static_cast<int32_t>(wk >= 0.0 ? 0.5 + wk : -0.5 + wk);
      }
      ax.lo[static_cast<size_t>(d)] = static_cast<size_t>(xmin);
      ax.count[static_cast<size_t>(d)] = static_cast<size_t>(xmax);
    }
    return ax;
  };
  const AxisTaps ax = build_axis(iw, out_w);
  const AxisTaps ay = build_axis(ih, out_h);
  const size_t oh = static_cast<size_t>(out_h), ow = static_cast<size_t>(out_w);
  // Pillow runs the vertical pass first only when height shrinks more than
  // twice as fast as width; otherwise horizontal first. Either order is
  // exact here (uint8 rounding per pass matches in both), this mirrors it.
  const long long dH =
      static_cast<long long>(ih) - static_cast<long long>(out_h);
  const long long dW =
      static_cast<long long>(iw) - static_cast<long long>(out_w);
  const bool horiz_first = !(dH > 0 && dH > dW * 2);
  const auto horiz_pass = [&ax](const KI::HostImageU8& src,
                                KI::HostImageU8& dst) {
    const size_t h = src.extent(0);
    const size_t w = dst.extent(1);
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x) {
        const size_t xmin = ax.lo[x], n = ax.count[x];
        const int32_t* kw = &ax.kvals[x * ax.ksize];
        for (size_t c = 0; c < 4; ++c) {
          int64_t acc = kFixRound;
          for (size_t k = 0; k < n; ++k)
            acc += static_cast<int64_t>(src(y, xmin + k, c)) *
                   static_cast<int64_t>(kw[k]);
          const int64_t v = acc >> kFixBits;
          dst(y, x, c) =
              static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
      }
  };
  const auto vert_pass = [&ay](const KI::HostImageU8& src,
                               KI::HostImageU8& dst) {
    const size_t h = dst.extent(0);
    const size_t w = dst.extent(1);
    for (size_t y = 0; y < h; ++y) {
      const size_t ymin = ay.lo[y], n = ay.count[y];
      const int32_t* kw = &ay.kvals[y * ay.ksize];
      for (size_t x = 0; x < w; ++x)
        for (size_t c = 0; c < 4; ++c) {
          int64_t acc = kFixRound;
          for (size_t k = 0; k < n; ++k)
            acc += static_cast<int64_t>(src(ymin + k, x, c)) *
                   static_cast<int64_t>(kw[k]);
          const int64_t v = acc >> kFixBits;
          dst(y, x, c) =
              static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
  };
  if (horiz_first) {
    KI::HostImageU8 tmp("qwen21edit_pil_tmp", ih, ow, 4);
    horiz_pass(in, tmp);
    KI::HostImageU8 out("qwen21edit_pil", oh, ow, 4);
    vert_pass(tmp, out);
    return out;
  }
  KI::HostImageU8 tmp("qwen21edit_pil_tmp", oh, iw, 4);
  vert_pass(in, tmp);
  KI::HostImageU8 out("qwen21edit_pil", oh, ow, 4);
  horiz_pass(tmp, out);
  return out;
}

// RGBA -> RGB flattened over white (the reference composites the vision
// copy over white before the processor; the VAE keeps RGBA). In-place on a
// copy: out_rgb = rgb*a + (1-a).
inline KokkosDNN::Image::HostImageF32 edit_flatten_white_rgb(
    const KokkosDNN::Image::HostImageF32& rgba) {
  namespace KI = KokkosDNN::Image;
  if (rgba.extent(2) != 4)
    throw std::invalid_argument("edit flatten: need RGBA input");
  const size_t h = rgba.extent(0), w = rgba.extent(1);
  KI::HostImageF32 rgb("qwen21edit_rgb", h, w, 3);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      const float a = rgba(y, x, 3);
      for (size_t c = 0; c < 3; ++c)
        rgb(y, x, c) = rgba(y, x, c) * a + (1.0f - a);
    }
  return rgb;
}

// Patch packing for the vision tower: HWC RGB [-1,1] -> rows
// [C*T*P*P] in spatial-merge-block-major order (block_row, block_col,
// in_row, in_col). Each patch row is a (C,T,Ph,Pw) row-major flatten with
// strides C:512, T:256, py:16, px:1, i.e. offset ((c*T)+t)*P*P + py*P + px
// (transformers qwen2_vl image_processing + modeling patchify temporal
// expand, T=2x duplicate for single images). Channel-outermost inner loops
// match that offset; row order and the tower's pos/rope block-major layout
// are unchanged.
inline std::vector<float> edit_pack_patches_blockmajor(
    const KokkosDNN::Image::HostImageF32& rgb, int P, int Tmerge) {
  const size_t h = rgb.extent(0), w = rgb.extent(1);
  if (rgb.extent(2) != 3 || h == 0 || w == 0 || P <= 0 || Tmerge <= 0)
    throw std::invalid_argument("edit pack: bad dims");
  if (h % (size_t)P != 0 || w % (size_t)P != 0)
    throw std::invalid_argument("edit pack: image must tile the patch");
  const size_t gh = h / (size_t)P, gw = w / (size_t)P;
  constexpr int kMerge = 2;
  if (gh % kMerge != 0 || gw % kMerge != 0)
    throw std::invalid_argument("edit pack: grid must tile the 2x2 merge");
  std::vector<float> out;
  out.reserve(gh * gw * 3 * (size_t)Tmerge * (size_t)P * (size_t)P);
  for (size_t bh = 0; bh < gh / kMerge; ++bh)
    for (size_t bw = 0; bw < gw / kMerge; ++bw)
      for (int ih = 0; ih < kMerge; ++ih)
        for (int iw = 0; iw < kMerge; ++iw)
          for (size_t c = 0; c < 3; ++c)
            for (int t = 0; t < Tmerge; ++t)
              for (int py = 0; py < P; ++py)
                for (int px = 0; px < P; ++px) {
                  const size_t y =
                      (bh * kMerge + (size_t)ih) * (size_t)P + (size_t)py;
                  const size_t x =
                      (bw * kMerge + (size_t)iw) * (size_t)P + (size_t)px;
                  out.push_back(rgb(y, x, c));
                }
  return out;
}

// M-RoPE positions for the B=1 joint sequence, mirroring get_rope_index:
// text runs advance all three rows equally; the vision block holds T at the
// running offset with H/W grids, then the offset advances by
// max(llm_h, llm_w). Returns 3*S ints row-major (t/h/w rows). Tokenizer-tied
// (joint path only): compiled out with the template builders above.
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER) && defined(KOKKOSDNN_ENABLE_IMAGE)
inline std::vector<int> edit_rope_positions(size_t head_len, int llm_h,
                                            int llm_w, size_t tail_len) {
  if (llm_h <= 0 || llm_w <= 0)
    throw std::invalid_argument("edit rope: bad vision grid");
  const size_t Nv = (size_t)llm_h * (size_t)llm_w;
  const size_t S = head_len + Nv + tail_len;
  std::vector<int> pos(3 * S, 0);
  int cur = 0;
  for (size_t s = 0; s < head_len; ++s) {
    pos[s] = cur + (int)s;
    pos[S + s] = cur + (int)s;
    pos[2 * S + s] = cur + (int)s;
  }
  cur += (int)head_len;
  for (int hr = 0; hr < llm_h; ++hr)
    for (int wc = 0; wc < llm_w; ++wc) {
      const size_t k = (size_t)hr * (size_t)llm_w + (size_t)wc;
      pos[head_len + k] = cur;
      pos[S + head_len + k] = cur + hr;
      pos[2 * S + head_len + k] = cur + wc;
    }
  cur += std::max(llm_h, llm_w);
  for (size_t s = 0; s < tail_len; ++s) {
    pos[head_len + Nv + s] = cur + (int)s;
    pos[S + head_len + Nv + s] = cur + (int)s;
    pos[2 * S + head_len + Nv + s] = cur + (int)s;
  }
  return pos;
}
#endif  // HAVE_ENCODER && HAVE_TOKENIZER && IMAGE (joint path)
#endif  // KOKKOSDNN_ENABLE_IMAGE

// Raw int32 LE vision-slot rows (embed-space, post-drop): e.g. converted
// from the golden image_pad_rows.npy (int64 + 128-byte header: skip the
// header with `python3 -c "import numpy,sys;
// numpy.load(f).astype('<i4').tofile(...)"`).
inline std::vector<int> load_vis_rows_raw(const std::string& path, size_t S) {
  static_assert(sizeof(int) == 4, "vis rows need 32-bit int");
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("qwen21: cannot open '" + path + "'");
  const long long n = (long long)f.tellg();
  if (n <= 0 || n % 4 != 0)
    throw std::runtime_error("qwen21: bad int32 file '" + path + "'");
  f.seekg(0, std::ios::beg);
  std::vector<int> rows((size_t)n / 4);
  f.read(reinterpret_cast<char*>(rows.data()), n);
  if (!f) throw std::runtime_error("qwen21: short read of '" + path + "'");
  for (int r : rows)
    if (r < 0 || (size_t)r >= S)
      throw std::runtime_error("qwen21: vis row out of range in '" + path +
                               "' (need embed-space rows in [0, S))");
  return rows;
}

void edit_usage() {
  std::printf(
      "qwen_image_21_edit: one-shot Qwen-Image-2.1 reference-image editing\n"
      "Reference: Qwen-Image-2.1-Turbo edit pipeline (diffusers "
      "QwenImage21Pipeline\n"
      "with image=..., use_kv_cache=True). One edit per invocation; no "
      "REPL.\n"
      "\n"
      "  --image PATH            condition image (required; exactly one in "
      "v1;\n"
      "                          RGBA via the image component, needs "
      "KOKKOSDNN_ENABLE_IMAGE)\n"
      "  --prompt TEXT           edit instruction\n"
      "  --output-resolution N   reference area side (default 1024): input "
      "HxW\n"
      "                          derives from the condition aspect (both snap "
      "to\n"
      "                          /32); explicit --height/--width override only "
      "the\n"
      "                          output size\n"
      "  --height/--width N      output pixels (override the derived size)\n"
      "  --tokenizer D           processor dir (tokenizer.json); with "
      "--text-encoder\n"
      "                          runs the in-process joint Qwen forward\n"
      "  --text-encoder F        Qwen text-encoder weights (single-shard "
      "model.safetensors)\n"
      "  --prompt-embeds-in F    joint text+vision embeddings [L,Cc] f32 LE "
      "(torch\n"
      "                          reference dump); else null zeros\n"
      "  --text-weights F        alias of --prompt-embeds-in\n"
      "  --ref-latents-in F      VAE-encoded condition latents ([C,Hc,Wc] "
      "chw,\n"
      "                          or torch [Hc*Wc,C] with --ref-latents-layout "
      "token);\n"
      "                          override; default is in-process VAE encode "
      "(needs\n"
      "                          --vae-weights), else zeros\n"
      "  --ref-latents-layout L  chw (default) | token\n"
      "  --vis-rows-in F       vision-slot rows (raw int32 LE, embed-space);\n"
      "                          joint build provides them, else this file\n"
      "                          (e.g. golden image_pad_rows.npy converted)\n"
      "                          enables the [cond|target] DiT path\n"
      "  --weights D             DiT shard dir or file (needs the model "
      "headers;\n"
      "                          --tiny runs the random-weights smoke model)\n"
      "  --init-latents-in F     target noise [C,H,W] f32 LE (else mt19937 "
      "seed)\n"
      "  --steps N               Euler steps (default 8 Turbo; --tiny "
      "forces 2)\n"
      "  --cfg/--guidance G      CFG scale (default 1.0, no CFG branch)\n"
      "  --shift/--flow-shift S  sigma shift, driver schedule only "
      "(default 3.0)\n"
      "  --sigmas M              driver (default) | turbo (checkpoint "
      "sample_sigmas)\n"
      "  --seed N                default 42\n"
      "  --vae-weights F         VAE checkpoint (encode + decode need it)\n"
      "  --vae-decode / --no-vae-decode  real decode (default) vs preview\n"
      "  --out F                 default qwen_image_21_edit.png\n"
      "  --staged / --no-staged  fenced staged release (default on)\n"
      "  --dump-steps DIR        parity tensors + manifest.json\n"
      "  --tiny                  4x4 smoke latents, 2 steps, random weights\n"
      "\n"
      "Run fallbacks (logged per run): without vision-slot rows (joint\n"
      "build or --vis-rows-in), the run falls back to target-only DiT\n"
      "sampling.\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string image_path, prompt = "turn the sketch into a photorealistic photo";
  std::string text_weights, prompt_embeds_in;
  std::string ref_latents_in, ref_latents_layout = "chw";
  std::string vis_rows_in;
  std::string init_latents_in, dump_dir;
  std::string weights = "transformer";
  std::string tokenizer_dir, text_encoder;  // joint Qwen forward needs both
                                                 // together (enforced below)
  std::string vae_weights, out = "qwen_image_21_edit.png";
#ifdef KOKKOSDNN_ENABLE_IMAGE
  int output_resolution = 1024;
#endif
  int height = 0, width = 0;  // 0 = derive from the condition aspect
  int steps = 8;
  bool steps_given = false;
  float guidance = 1.0f, flow_shift = 3.0f;
  std::string sigmas_mode = "driver";
  uint64_t seed = 42;
  bool tiny = false;
  bool staged = true, vae_decode = true;
  int n_image_flags = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto need = [&](std::string& dst) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
        return false;
      }
      dst = argv[++i];
      return true;
    };
    if (a == "--image") {
      if (!need(image_path)) return 1;
      ++n_image_flags;
    } else if (a == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
#ifdef KOKKOSDNN_ENABLE_IMAGE
    } else if (a == "--output-resolution" && i + 1 < argc) {
      output_resolution = std::stoi(argv[++i]);
#endif
    } else if (a == "--text-weights" && i + 1 < argc) {
      text_weights = argv[++i];
    } else if (a == "--prompt-embeds-in" && i + 1 < argc) {
      prompt_embeds_in = argv[++i];
    } else if (a == "--ref-latents-in" && i + 1 < argc) {
      ref_latents_in = argv[++i];
    } else if (a == "--ref-latents-layout" && i + 1 < argc) {
      ref_latents_layout = argv[++i];
      if (ref_latents_layout != "chw" && ref_latents_layout != "token") {
        std::fprintf(stderr,
                     "error: --ref-latents-layout must be 'chw' or 'token' "
                     "(got '%s')\n",
                     ref_latents_layout.c_str());
        return 1;
      }
    } else if (a == "--vis-rows-in" && i + 1 < argc) {
      vis_rows_in = argv[++i];
    } else if (a == "--init-latents-in" && i + 1 < argc) {
      init_latents_in = argv[++i];
    } else if (a == "--dump-steps" && i + 1 < argc) {
      dump_dir = argv[++i];
    } else if (a == "--weights") {
      if (!need(weights)) return 1;
      tiny = false;
    } else if (a == "--tokenizer" && i + 1 < argc) {
      tokenizer_dir = argv[++i];
    } else if (a == "--text-encoder" && i + 1 < argc) {
      text_encoder = argv[++i];
    } else if (a == "--vae-weights" && i + 1 < argc) {
      vae_weights = argv[++i];
    } else if (a == "--vae-decode") {
      vae_decode = true;
    } else if (a == "--no-vae-decode" || a == "--no-vae_decode") {
      vae_decode = false;
    } else if (a == "--height" && i + 1 < argc) {
      height = std::stoi(argv[++i]);
    } else if (a == "--width" && i + 1 < argc) {
      width = std::stoi(argv[++i]);
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
      steps_given = true;
    } else if ((a == "--cfg" || a == "--guidance") && i + 1 < argc) {
      guidance = std::stof(argv[++i]);
    } else if ((a == "--shift" || a == "--flow-shift") && i + 1 < argc) {
      flow_shift = std::stof(argv[++i]);
    } else if (a == "--sigmas" && i + 1 < argc) {
      sigmas_mode = argv[++i];
      if (sigmas_mode != "driver" && sigmas_mode != "turbo") {
        std::fprintf(stderr,
                     "error: --sigmas must be 'driver' or 'turbo' (got '%s')\n",
                     sigmas_mode.c_str());
        return 1;
      }
    } else if (a == "--seed" && i + 1 < argc) {
      seed = (uint64_t)std::stoull(argv[++i]);
    } else if (a == "--tiny") {
      tiny = true;
    } else if (a == "--staged") {
      staged = true;
    } else if (a == "--no-staged" || a == "--no-stage") {
      staged = false;
    } else if (a == "--help" || a == "-h") {
      edit_usage();
      return 0;
    } else {
      std::fprintf(stderr, "warning: ignoring unknown arg '%s'\n", a.c_str());
    }
  }

  if (n_image_flags == 0 && !tiny) {
    std::fprintf(stderr, "error: --image PATH is required (omit only with --tiny,"
                         " which uses a synthesized test card)\n");
    return 1;
  }
  if (n_image_flags > 1) {
    std::fprintf(stderr,
                 "error: v1 supports exactly one --image (the reference "
                 "accepts up to 10; multi-image edit is a follow-up)\n");
    return 1;
  }
  if ((!tokenizer_dir.empty()) != (!text_encoder.empty())) {
    std::fprintf(stderr,
                 "error: --tokenizer and --text-encoder must be given together "
                 "for the in-process joint Qwen forward\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    KokkosDNN::print_configuration(std::cout);
    std::printf("[backend dit=%s encoder=%s vae=%s]\n",
#ifdef QWEN21_DRIVER_HAVE_DIT
                "qwen-image-21-dit",
#else
                "placeholder",
#endif
#ifdef QWEN21_DRIVER_HAVE_ENCODER
                "qwen",
#else
                "file/null",
#endif
#ifdef QWEN21_DRIVER_HAVE_VAE
                "qwen-image-21-vae"
#else
                "preview"
#endif
    );
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::QwenImage21::Config cfg =
        tiny ? KokkosDNN::QwenImage21::Config::tiny_test()
             : KokkosDNN::QwenImage21::Config();
    cfg.validate();
#else
    QW::Config cfg = tiny ? QW::Config::tiny_test() : QW::Config::qwen_image_21();
    cfg.validate();
#endif
    const int steps_eff = (!steps_given && tiny) ? 2 : steps;
    const size_t C = be_latent_channels(cfg);

    // ---- Condition image: load + reference geometry ----
    // One resize feeds both the vision copy and the VAE copy (reference:
    // input_images for the text encoder, vae_images for the VAE). The
    // pipeline reads RGBA (non-RGBA inputs are converted up); the vision
    // copy is flattened over white for the encoder while the VAE keeps all
    // four channels. Input size comes from the reference
    // calculate_dimensions (output_resolution^2 area at the condition
    // aspect, /32 rounding); explicit --height/--width override only the
    // OUTPUT size (floored to /32 like the reference).
    int cond_w = 0, cond_h = 0;  // resized condition pixels
    int in_lat_h = 0, in_lat_w = 0;
    std::vector<float> vis_pixels;  // packed vision rows, [-1,1]
    std::vector<float> vae_chw;     // resized RGBA [-1,1] CHW for VAE encode
    {
#ifdef KOKKOSDNN_ENABLE_IMAGE
      namespace KI = KokkosDNN::Image;
      KI::HostImageF32 img("qwen21edit_testcard", 0, 0, 0);
      KI::HostImageU8 img_u8("qwen21edit_rgba_u8", 0, 0, 0);
      if (image_path.empty()) {
        // Weight-free smoke card (tiny only): 64x64 RGBA gradient.
        img = KI::HostImageF32("qwen21edit_testcard", 64, 64, 4);
        for (size_t y = 0; y < 64; ++y)
          for (size_t x = 0; x < 64; ++x) {
            img(y, x, 0) = (float)x / 63.0f;
            img(y, x, 1) = (float)y / 63.0f;
            img(y, x, 2) = 0.5f;
            img(y, x, 3) = 1.0f;
          }
        std::printf("[image synthesized test card] 64x64 RGBA (--tiny, no "
                    "--image)\n");
      } else {
        img = KI::load_image_f32(image_path, /*desired_channels=*/4);
        img_u8 = KI::load_image_u8(image_path, /*desired_channels=*/4);
      }
      const size_t iw = img.extent(1), ih = img.extent(0);
      std::printf("[image %s] %zux%zu RGBA\n", image_path.c_str(), iw, ih);
      if (tiny) {
        cond_w = cond_h = 4 * kPixelScale;
      } else {
        if (output_resolution <= 0)
          throw std::runtime_error("--output-resolution must be > 0");
        edit_calculate_dimensions(output_resolution * output_resolution,
                                  (double)iw / (double)ih, cond_w, cond_h);
        if (cond_w < 32 || cond_h < 32)
          throw std::runtime_error(
              "edit geometry: derived condition size too small");
      }
      KI::HostImageF32 rgb = edit_flatten_white_rgb(img);
      if ((size_t)cond_w != iw || (size_t)cond_h != ih) {
        rgb = edit_bicubic_resize_hwc(rgb, cond_h, cond_w);
        std::printf("[image] resized %zux%zu -> %dx%d (float lanczos vision "
                    "copy; PIL-exact uint8 VAE copy)\n",
                    iw, ih, cond_w, cond_h);
      }
      // VAE copy: bit-exact replica of the reference preprocess() resize
      // (PIL LANCZOS on the uint8 RGBA image, no white flatten, no RGB
      // drop) then u8/255 normalized to [-1,1] like pil_to_numpy +
      // normalize. A float-domain resize here leaves ~2 LSB of mismatch
      // that the VAE encoder amplifies past the gate, so the uint8-exact
      // path above is required.
      if (!tiny) {
        KI::HostImageU8 vrgba = img_u8;
        if ((size_t)cond_w != iw || (size_t)cond_h != ih)
          vrgba = edit_pil_resize_rgba_u8(img_u8, cond_h, cond_w);
        vae_chw.resize(4 * (size_t)cond_h * (size_t)cond_w);
        size_t vk = 0;
        for (size_t c = 0; c < 4; ++c)
          for (int y = 0; y < cond_h; ++y)
            for (int x = 0; x < cond_w; ++x)
              vae_chw[vk++] = static_cast<float>(vrgba((size_t)y, (size_t)x,
                                                        c)) /
                              255.0f * 2.0f - 1.0f;
      }
      // Processor normalize: (x - 0.5) / 0.5 over RGB.
      for (size_t y = 0; y < rgb.extent(0); ++y)
        for (size_t x = 0; x < rgb.extent(1); ++x)
          for (size_t c = 0; c < 3; ++c)
            rgb(y, x, c) = rgb(y, x, c) * 2.0f - 1.0f;
      vis_pixels = edit_pack_patches_blockmajor(rgb, 16, 2);
      std::printf("[vision] %dx%d px -> %zu patches (16px, block-major, "
                  "temporal x2)\n",
                  cond_w, cond_h, vis_pixels.size() / 1536);
      in_lat_h = cond_h / kPixelScale;
      in_lat_w = cond_w / kPixelScale;
      std::printf("[condition] %dx%d px -> VAE grid %dx%d (x%d)\n", cond_w,
                  cond_h, in_lat_h, in_lat_w, kPixelScale);
#else
      (void)cond_w;
      (void)cond_h;
      (void)in_lat_h;
      (void)in_lat_w;
      (void)vis_pixels;
      throw std::runtime_error(
          "--image needs the image component "
          "(configure with -DKokkosDNN_ENABLE_IMAGE=ON)");
#endif
    }

    // Target geometry: explicit H/W override the derived condition size;
    // --tiny pins 64px. Driver stride snap is a no-op on /32 sizes.
    PixelGeometry geo;
    if (tiny) {
      geo.px_h = geo.px_w = 4 * kPixelScale;
      geo.lat_h = geo.lat_w = 4;
    } else {
      const int req_h = (height > 0) ? height : cond_h;
      const int req_w = (width > 0) ? width : cond_w;
      geo = resolve_pixel_geometry(req_h, req_w, be_patch(cfg));
      if (geo.px_h != req_h || geo.px_w != req_w)
        std::printf("[size] rounded %dx%d px -> %dx%d px (latent %dx%d, "
                    "stride %dpx)\n",
                    req_w, req_h, geo.px_w, geo.px_h, geo.lat_w, geo.lat_h,
                    kPixelScale * be_patch(cfg));
    }
    const int H = geo.lat_h, W = geo.lat_w;
    const int pxH = geo.px_h, pxW = geo.px_w;
    std::printf("[model %s: %s]\n", tiny ? "tiny_test" : "qwen-image-2.1",
                be_label(cfg));

    // Caption source: --prompt-embeds-in is an alias of --text-weights and
    // wins when both are given. It is resolved before the joint attempt so
    // the fallback path below is already known.
    std::string caption_src = text_weights;
    if (!prompt_embeds_in.empty()) {
      if (!text_weights.empty() && prompt_embeds_in != text_weights)
        std::fprintf(stderr,
                     "warning: --prompt-embeds-in wins over --text-weights\n");
      caption_src = prompt_embeds_in;
    }

    // Joint processor inputs for the in-process edit encode (token ids with
    // the image_pad expansion, vision rows, M-RoPE positions, pixel rows).
    // Built here (host-side, needs the BPE tables); consumed by
    // EncoderStage::run_qwen_joint below. Empty unless the build succeeds.
    std::vector<int> joint_ids, joint_rows, joint_pos3;
    std::vector<float> joint_pixels;
    int joint_Hp = 0, joint_Wp = 0;
    size_t joint_drop = 0;
    bool have_joint_inputs = false;
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER) && defined(KOKKOSDNN_ENABLE_IMAGE)
    if (!tiny && !tokenizer_dir.empty() && !text_encoder.empty() &&
        !image_path.empty()) {
      try {
        namespace QE = KokkosDNN::QwenImage21Encoder;
        const std::string tok_json = tokenizer_dir + "/tokenizer.json";
        if (!file_exists(tok_json))
          throw std::runtime_error("--tokenizer '" + tokenizer_dir +
                                   "' has no tokenizer.json (need the "
                                   "Qwen-Image-2.1 processor/ dir)");
        const RawQwenTables tabs = load_raw_qwen_tables(tok_json);
        // No 512-row T2I cap on the joint path (real edits carry thousands
        // of vision tokens); bound only against absurd sequences.
        constexpr size_t kJointMax = 32768;
        const size_t max_len = kJointMax;
        // System block (drop prefix): same fixed block the T2I template
        // uses (qwen21_t2i_head minus the user prefix), so drop_idx matches
        // the reference _drop_idx.
        const std::string kUserPrefix = "<|im_start|>user\n";
        const std::string full_head = QE::qwen21_t2i_head();
        if (full_head.size() < kUserPrefix.size() ||
            full_head.compare(full_head.size() - kUserPrefix.size(),
                              kUserPrefix.size(), kUserPrefix) != 0)
          throw std::runtime_error(
              "joint encode: T2I head lost its user-role prefix");
        const std::string sys_block =
            full_head.substr(0, full_head.size() - kUserPrefix.size());
        const std::string head_text = qwen21_edit_head(1);
        // Tail follows the slot's own <|vision_end|>: the reference
        // prompt_template_ti2i closes each vision slot exactly once, so no
        // second close is prepended here.
        const std::string tail_text = prompt + qwen21_edit_tail();
        const std::vector<int> sys_ids =
            encode_qwen_raw(tabs, sys_block, max_len);
        const std::vector<int> head_ids =
            encode_qwen_raw(tabs, head_text, max_len);
        const std::vector<int> tail_ids =
            encode_qwen_raw(tabs, tail_text, max_len);
        if (sys_ids.empty() || head_ids.size() <= sys_ids.size() ||
            !std::equal(sys_ids.begin(), sys_ids.end(), head_ids.begin()))
          throw std::runtime_error(
              "joint encode: head lost its system-block prefix");
        const auto pad_it = tabs.vocab.find("<|image_pad|>");
        int pad_id = -1;
        if (pad_it != tabs.vocab.end()) {
          pad_id = pad_it->second;
        } else {
          // Added tokens live in specials, not vocab: longest-match split
          // emits their ids directly (see encode_qwen_raw).
          for (const auto& kv : tabs.specials)
            if (kv.first == "<|image_pad|>") pad_id = kv.second;
        }
        if (pad_id < 0)
          throw std::runtime_error(
              "joint encode: tokenizer.json lacks <|image_pad|>");
        joint_Hp = cond_h / 16;
        joint_Wp = cond_w / 16;
        if (joint_Hp <= 0 || joint_Wp <= 0 || joint_Hp % 2 != 0 ||
            joint_Wp % 2 != 0)
          throw std::runtime_error(
              "joint encode: patch grid must tile the 2x2 merge");
        const size_t Nv =
            (size_t)(joint_Hp / 2) * (size_t)(joint_Wp / 2);
        const size_t Np = (size_t)joint_Hp * (size_t)joint_Wp;
        if (vis_pixels.size() != Np * 1536)
          throw std::runtime_error(
              "joint encode: pixel rows != patches*1536");
        // Expand the slot's single <|image_pad|> placeholder in place to Nv
        // vision ids, mirroring the processor convention (one pad token in
        // the raw text expands to Nv image tokens). Total pads == Nv and
        // vis_rows covers all of them.
        size_t pad_pos = head_ids.size();
        size_t pad_count = 0;
        for (size_t k = 0; k < head_ids.size(); ++k)
          if (head_ids[k] == pad_id) {
            if (pad_count == 0) pad_pos = k;
            ++pad_count;
          }
        if (pad_count != 1)
          throw std::runtime_error(
              "joint encode: head must hold exactly one <|image_pad|> "
              "placeholder");
        joint_ids.assign(head_ids.begin(),
                         head_ids.begin() + (long)pad_pos);
        joint_ids.insert(joint_ids.end(), Nv, pad_id);
        joint_ids.insert(joint_ids.end(),
                         head_ids.begin() + (long)pad_pos + 1, head_ids.end());
        joint_ids.insert(joint_ids.end(), tail_ids.begin(), tail_ids.end());
        // No 512-row T2I cap here: real edits carry thousands of vision
        // tokens (2048px -> 4096). Bound only against absurd sequences.
        if (joint_ids.size() > 32768)
          throw std::runtime_error(
              "joint encode: sequence exceeds 32768 tokens");
        for (size_t k = 0; k < Nv; ++k)
          joint_rows.push_back((int)(pad_pos + k));
        // edit_rope_positions lays out [head text][Nv vision][tail text]
        // with S = head_len + Nv + tail_len, so head_len must be the
        // PRE-vision text only: tokens before the image_pad placeholder.
        // The head remainder after the placeholder (e.g. <|vision_end|>)
        // joins the tail side, placing the vision block at pad_pos per
        // get_rope_index. Passing head_ids.size()-1+Nv here double-counted
        // Nv (pos3 held 3*(S+Nv) ints, tripping the 3*S check in
        // EncoderStage::run_qwen_joint).
        const size_t joint_head_len = pad_pos;
        const size_t joint_tail_len =
            (head_ids.size() - pad_pos - 1) + tail_ids.size();
        joint_pos3 = edit_rope_positions(joint_head_len, joint_Hp / 2,
                                         joint_Wp / 2, joint_tail_len);
        joint_pixels = vis_pixels;
        joint_drop = sys_ids.size();
        have_joint_inputs = true;
        std::printf("[edit template] %zu ids (%zu head + %zu vision + %zu "
                    "tail), drop_idx=%zu, tok_vocab=%zu\n",
                    joint_ids.size(), joint_head_len, Nv, joint_tail_len,
                    joint_drop, tabs.vocab_size);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: joint processor build failed (%s); falling "
                     "back to %s\n",
                     e.what(), caption_src.c_str());
      }
    }
#endif  // HAVE_ENCODER && HAVE_TOKENIZER && IMAGE (joint path)

    // ---- Reference latents (VAE-encoded condition image) ----
    // Precedence: --ref-latents-in (torch/golden (latents-mean)/std,
    // driver chw or torch token layout) -> in-process VAE encode of the
    // shared resized condition image (needs --vae-weights + the VAE
    // header) -> zeros with a logged reason. Either way they are CARRIED
    // (validated, dumped, reported), then staged below as packed cond
    // tokens for the DiT image-condition path ([cond|target] +
    // img_shapes/img_mask); without vision-slot rows the run falls back
    // to target-only sampling (still dumped + reported).
    std::vector<float> ref_lat;
    if (!ref_latents_in.empty()) {
      if (tiny)
        throw std::runtime_error(
            "--ref-latents-in needs real geometry; drop --tiny");
      ref_lat = load_decode_latents(ref_latents_in, C, (size_t)in_lat_h,
                                    (size_t)in_lat_w,
                                    ref_latents_layout == "token");
      std::printf("[ref-latents file %s] %zux%dx%d floats (%s layout)\n",
                  ref_latents_in.c_str(), C, in_lat_h, in_lat_w,
                  ref_latents_layout.c_str());
    } else if (!tiny && !vae_chw.empty() && !vae_weights.empty()) {
#ifdef QWEN21_DRIVER_HAVE_VAE
      try {
        ref_lat = try_vae_encode_image(vae_chw, (size_t)cond_h, (size_t)cond_w,
                                       vae_weights);
        std::printf("[ref-latents vae-encode] %zux%dx%d floats (in-process "
                    "RGBA [-1,1] -> (mode-mean)/std)\n",
                    C, in_lat_h, in_lat_w);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: in-process VAE encode failed (%s); using "
                     "zeros\n",
                     e.what());
        ref_lat.assign(C * (size_t)in_lat_h * (size_t)in_lat_w, 0.0f);
        std::printf("[ref-latents null] VAE encode failed: zeros %zux%dx%d\n",
                    C, in_lat_h, in_lat_w);
      }
#else
      ref_lat.assign(C * (size_t)in_lat_h * (size_t)in_lat_w, 0.0f);
      std::printf("[ref-latents null] VAE encode needs "
                  "KokkosDNN_QwenImage21Vae.hpp in this build: zeros "
                  "%zux%dx%d\n",
                  C, in_lat_h, in_lat_w);
#endif
    } else {
      ref_lat.assign(C * (size_t)in_lat_h * (size_t)in_lat_w, 0.0f);
      const char* why = tiny ? "--tiny skips VAE encode (smoke model)"
                             : vae_chw.empty()
                                   ? "no --image pixels to encode"
                                   : "need --vae-weights for VAE encode";
      std::printf("[ref-latents null] %s: zeros %zux%dx%d\n", why, C, in_lat_h,
                  in_lat_w);
    }

    // Caption source (resolved above): --prompt-embeds-in wins over
    // --text-weights (joint text+vision embeddings for edit; L is inferred
    // from the file). Uncond stays zeros (the reference re-encodes the
    // negative prompt with the images under CFG; file/null parity stays
    // zeros like the generate driver).

    // ---- Phase 1: encode (no DiT weights are loaded yet) ----
    // Precedence: in-process joint Qwen forward (--tokenizer dir +
    // --text-encoder weights + --image) -> --prompt-embeds-in file -> null
    // zeros. A throw inside the joint attempt falls back to file/null with
    // a logged reason instead of failing.
    EncoderStage enc;
    bool joint_ok = false;  // true only when the joint embeds below are live
    if (have_joint_inputs) {
      try {
        EncoderStage::run_qwen_joint(enc, cfg, joint_ids, joint_rows,
                                     joint_pos3, joint_pixels, joint_Hp,
                                     joint_Wp, joint_drop, text_encoder);
        joint_ok = true;
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: in-process joint Qwen encode failed (%s); "
                     "falling back to %s\n",
                     e.what(), caption_src.empty() ? "null" : "file");
        enc = EncoderStage::run(cfg, prompt, caption_src, "", "", tiny,
                                     /*max_seq_override=*/32768);
      }
    } else {
      if (!tiny && !tokenizer_dir.empty() && !text_encoder.empty() &&
          !image_path.empty())
        std::printf("[encode] joint Qwen forward needs the encoder + BPE "
                    "tables headers in this build; falling back to %s\n",
                    caption_src.empty() ? "null" : "file");
      else if (tiny && (!tokenizer_dir.empty() || !text_encoder.empty()))
        std::printf("[encode] --tiny skips the joint Qwen forward "
                    "(random-weights smoke model); falling back to %s\n",
                    caption_src.empty() ? "null" : "file");
      enc = EncoderStage::run(cfg, prompt, caption_src, "", "", tiny,
                                     /*max_seq_override=*/32768);
    }
    log_encode_memory(enc);
    if (guidance != 1.0f)
      std::printf("[cfg] uncond stays zeros (the reference re-encodes the "
                  "negative prompt with images under CFG; file/null parity "
                  "stays zeros like the generate driver)\n");

    if (staged) {
      Kokkos::fence();
      QW::print_phase_memory("encode-released", enc.device_bytes(), 0);
      std::printf("[staged] keeping %.1f MiB of embeds resident on device "
                  "across the DiT load (no disk round-trip)\n",
                  (double)enc.device_bytes() / (1024.0 * 1024.0));
    } else {
      std::printf("[note] single-phase run (--no-staged): encoder embeds stay\n"
                  "                 live across the DiT load (no fenced, logged\n"
                  "                 encoder-release point)\n");
    }

    // ---- Phase 2: DiT load ----
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar> dit_model(
        cfg);
    {
      auto& w = dit_model.weights();
      if (tiny) {
        w.random_init(seed);
        std::printf("[tiny random weights] layers=%d hidden=%d latent=%dx%d\n",
                    cfg.num_layers, cfg.hidden_dim, H, W);
      } else {
        std::printf("[loading %s]\n", weights.c_str());
        try {
          const std::vector<std::string> shards =
              resolve_shard_list(weights, "--weights");
          if (shards.size() > 1)
            std::printf("[loading %zu shards]\n", shards.size());
          w.load_safetensors(shards);
        } catch (const std::exception& e) {
          throw std::runtime_error(std::string(e.what()) +
                                   " (need Qwen-Image-2.1 weights or --tiny)");
        }
        std::printf("[loaded] layers=%d hidden=%d\n", cfg.num_layers,
                    cfg.hidden_dim);
      }
      QW::print_phase_memory("sample", w.resident_bytes(),
                             dit_model.scratch_bytes());
    }
    std::printf("[budget] DiT transient scratch grows during sampling "
                "(see README fit table)\n");
#else
    QW::Weights<Exec, DScalar> w;
    if (tiny) {
      QW::random_init(cfg, w, seed);
      std::printf("[tiny random weights] hidden=%d depth=%d latent=%dx%d\n",
                  cfg.hidden_size, cfg.depth, H, W);
    } else {
      throw std::runtime_error(
          "weights file '" + weights +
          "' needs the true Qwen-Image-2.1 DiT "
          "(KokkosDNN_QwenImage21Dit.hpp is not in this build); pass --tiny "
          "for the random-weights smoke run. See README for the expected "
          "checkpoint layout.");
    }
    QW::print_phase_memory("sample", w.resident_bytes(), w.resident_bytes());
#endif
    std::printf("[budget] full-checkpoint projection: DiT %.1f GB + VAE %.3f "
                "GB + activations vs 24GB (see README fit table)\n",
                QW::kDitBudgetBytes / 1e9, QW::kVaeBudgetBytes / 1e9);

    if (!vae_weights.empty()) {
      std::error_code ec0;
      const bool vae_is_dir = std::filesystem::is_directory(vae_weights, ec0);
      if (!vae_is_dir && !file_exists(vae_weights))
        throw std::runtime_error("vae weights file '" + vae_weights +
                                 "' not found");
#ifdef QWEN21_DRIVER_HAVE_VAE
      std::printf("[vae %s] validated%s\n", vae_weights.c_str(),
                  vae_decode ? "; decode enabled (default)"
                             : "; preview (pass --vae-decode to decode)");
#else
      std::printf("[vae %s] validated, but KokkosDNN_QwenImage21Vae.hpp is "
                  "not in this build; writing a grayscale preview instead\n",
                  vae_weights.c_str());
#endif
    } else if (vae_decode) {
      std::printf("[vae] VAE decode needs --vae-weights (else grayscale "
                  "preview); disabling decode for this run\n");
      vae_decode = false;
    }

    std::vector<float> x_init;
    if (!init_latents_in.empty()) {
      x_init = load_init_latents_raw(init_latents_in, C, (size_t)H, (size_t)W);
      std::printf("[init-latents %s] %zux%dx%d floats (fp32 LE [C,H,W])\n",
                  init_latents_in.c_str(), C, H, W);
    } else {
      x_init = QW::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    }
    std::vector<float> turbo_sigmas;
    if (sigmas_mode == "turbo") {
#ifdef QWEN21_DRIVER_HAVE_OPS
      for (double s : cfg.sample_sigmas) turbo_sigmas.push_back((float)s);
      turbo_sigmas.push_back(0.0f);
      std::printf("[schedule turbo] checkpoint sample_sigmas verbatim "
                  "(mu-shifted already; --shift/--steps ignored except for "
                  "this note)\n");
#else
      throw std::runtime_error(
          "--sigmas turbo needs the true QwenImage21::Config "
          "(KokkosDNN_QwenImage21Config.hpp is not in this build)");
#endif
    }
    const int steps_run =
        (sigmas_mode == "turbo") ? (int)turbo_sigmas.size() - 1 : steps_eff;
    if (sigmas_mode == "turbo" && steps_given && steps != steps_run)
      std::printf("[schedule turbo] ignoring --steps %d (checkpoint schedule "
                  "has %d steps)\n",
                  steps, steps_run);
    std::printf("[sample] edit %dx%d px (target latent %dx%d, condition grid "
                "%dx%d) steps=%d cfg=%.3g shift=%.3g seed=%llu\n",
                pxH, pxW, H, W, in_lat_h, in_lat_w, steps_run,
                (double)guidance, (double)flow_shift,
                (unsigned long long)seed);
    // ---- Edit image-condition state ([cond|target] DiT path) ----
    // Vision-slot rows in embed (post-drop) coordinates: the joint build
    // emits full-sequence positions, so subtract joint_drop; --vis-rows-in
    // is already in embed coordinates (range-checked at load). Staged cond
    // tokens pack ref_lat host chw -> token rows (row t = h*in_lat_w + w,
    // the same order DitForward packs the target grid); img_shapes is
    // [[(1,in_lat_h,in_lat_w),(1,H,W)]] (cond then target) and img_mask is
    // vision-slots True + text False, then N/4 target True.
    Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> cond_tok;
    std::vector<std::array<int, 3>> img_blocks;
    std::vector<char> img_mask;
    std::string img_shapes_str;
    size_t img_mask_true = 0;
    {
      std::vector<int> slot_rows;
      if (joint_ok && !joint_rows.empty()) {
        for (int r : joint_rows) slot_rows.push_back(r - (int)joint_drop);
      } else if (!vis_rows_in.empty() && !tiny) {
        slot_rows =
            load_vis_rows_raw(vis_rows_in, (size_t)enc.device.extent(0));
      }
      const size_t Senc = (size_t)enc.device.extent(0);
      const size_t Nc = (size_t)in_lat_h * (size_t)in_lat_w;
      const size_t Nt = (size_t)H * (size_t)W;
      if (!tiny && !slot_rows.empty()) {
        if (in_lat_h <= 0 || in_lat_w <= 0)
          throw std::runtime_error("edit cond: condition grid not derived");
        // Each VLM slot stands for a 2x2 latent group: both cond latent
        // dims must tile the merge, and the target grid must hold N/4
        // slots (in_lat dims are /32-rounded, so even in practice).
        if (in_lat_h % 2 != 0 || in_lat_w % 2 != 0 || Nt % 4 != 0)
          throw std::runtime_error(
              "edit cond: latent grids must tile the 2x2 slots "
              "(in_lat_h/w even, N % 4 == 0)");
        if (slot_rows.size() * 4 != Nc)
          throw std::runtime_error(
              "edit cond: " + std::to_string(slot_rows.size()) +
              " vision slots != Nc/4 = " + std::to_string(Nc / 4));
        for (size_t k = 1; k < slot_rows.size(); ++k)
          if (slot_rows[k] <= slot_rows[k - 1])
            throw std::runtime_error(
                "edit cond: vision-slot rows must be strictly increasing");
        if (slot_rows.front() < 0 || slot_rows.back() >= (int)Senc)
          throw std::runtime_error(
              "edit cond: vision-slot row outside the staged S");
        cond_tok = Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec>(
            "qwen21edit_cond", 1, Nc, C);
        {
          auto hc = Kokkos::create_mirror_view(cond_tok);
          for (int hh = 0; hh < in_lat_h; ++hh)
            for (int ww = 0; ww < in_lat_w; ++ww)
              for (size_t c = 0; c < C; ++c)
                hc(0, (size_t)(hh * in_lat_w + ww), c) =
                    static_cast<DScalar>(ref_lat[(c * (size_t)in_lat_h +
                                                  (size_t)hh) *
                                                     (size_t)in_lat_w +
                                                 (size_t)ww]);
          Kokkos::deep_copy(cond_tok, hc);
        }
        Kokkos::fence();
        img_blocks = {{1, in_lat_h, in_lat_w}, {1, H, W}};
        img_mask.assign(Senc + Nt / 4, 0);
        for (int r : slot_rows) img_mask[(size_t)r] = 1;
        for (size_t v = Senc; v < img_mask.size(); ++v) img_mask[v] = 1;
        for (char c : img_mask) img_mask_true += c ? (size_t)1 : (size_t)0;
        std::ostringstream oss;
        oss << "[[(1," << in_lat_h << "," << in_lat_w << "),(1," << H << ","
            << W << ")]]";
        img_shapes_str = oss.str();
        std::printf("[dit] cond path: Nc=%zu (in-lat %dx%d) N=%zu S=%zu "
                    "T=%zu slots=%zu mask true=%zu/%zu img_shapes %s\n",
                    Nc, in_lat_h, in_lat_w, Nt, Senc,
                    Senc - slot_rows.size() + Nc + Nt, slot_rows.size(),
                    img_mask_true, img_mask.size(), img_shapes_str.c_str());
      } else {
        std::printf("[dit] target-only sampling (no vision-slot rows%s)\n",
                    tiny ? " under --tiny"
                         : (joint_ok || !vis_rows_in.empty()
                                ? "; slot/geometry mismatch (see above)"
                                : "; need the joint build or --vis-rows-in "
                                  "for the cond path"));
      }
    }
    DumpWriter dump(dump_dir);
    std::vector<float> dump_sigmas;
    bool dump_use_cfg = false;
    if (dump.active()) {
      dump_sigmas = (sigmas_mode == "turbo")
                        ? turbo_sigmas
                        : build_sigmas_nearest_grid(steps_eff, flow_shift);
      dump.write("initial_latents", x_init);
      // Reference latents exactly as staged (zeros when --ref-latents-in
      // was absent): the packed tensor the cond DiT path consumes.
      dump.write("ref_latents", ref_lat);
      if (!img_mask.empty()) {
        std::vector<float> mf(img_mask.size());
        for (size_t i = 0; i < img_mask.size(); ++i)
          mf[i] = img_mask[i] ? 1.0f : 0.0f;
        dump.write("img_mask", mf);
      }
    }
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::DNNHandle<Exec> dit_handle;
    DitForward dit_forward{&dit_model, &dit_handle, H, W};
    if (cond_tok.is_allocated()) {
      dit_forward.cond_tok = cond_tok;
      dit_forward.img_blocks = img_blocks;
      dit_forward.img_mask = img_mask;
    }
#else
    QW::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
    StepDumpHook dump_hook;
    if (dump.active()) {
      dump_hook.want_branches = (guidance != 1.0f);
      dump_hook.use_cfg_out = &dump_use_cfg;
      dump_hook.on_step_done =
          [&dump](int i, const std::vector<float>& x_step,
                  const std::vector<float>& v_step,
                  const std::vector<float>& v_cond,
                  const std::vector<float>& v_uncond) {
            char name[64];
            std::snprintf(name, sizeof(name), "step%02d_noise_pred", i);
            dump.write(name, v_step);
            if (!v_cond.empty()) {
              std::snprintf(name, sizeof(name), "step%02d_noise_pred_cond", i);
              dump.write(name, v_cond);
              std::snprintf(name, sizeof(name), "step%02d_noise_pred_uncond",
                            i);
              dump.write(name, v_uncond);
            }
            std::snprintf(name, sizeof(name), "step%02d_latents", i);
            dump.write(name, x_step);
          };
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps_run,
        guidance, flow_shift, H, W,
        [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        },
        (sigmas_mode == "turbo") ? &turbo_sigmas : nullptr, dump_hook,
        /*max_seq_override=*/32768);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("[done] %.2fs total, %.2fs/step\n", secs,
                secs / (double)std::max(steps_run, 1));

    double mean = 0.0, var = 0.0, mn = lat[0], mx = lat[0];
    for (float v : lat) {
      mean += v;
      mn = std::min(mn, (double)v);
      mx = std::max(mx, (double)v);
    }
    mean /= (double)lat.size();
    for (float v : lat) var += (v - mean) * (v - mean);
    var /= (double)lat.size();
    std::printf("[latents %zux%zux%zu] mean=%.6f std=%.6f min=%.4f max=%.4f\n",
                C, (size_t)H, (size_t)W, mean, std::sqrt(var), mn, mx);

    if (dump.active()) {
      DumpManifest dm;
      dm.prompt = prompt;
      dm.prompt_source = std::string("edit:") + enc.source;
      dm.model_label = be_label(cfg);
      dm.sigmas_mode = sigmas_mode;
      dm.sigmas = dump_sigmas;
      dm.height = pxH;
      dm.width = pxW;
      dm.latent_h = H;
      dm.latent_w = W;
      dm.steps = (int)dump_sigmas.size() - 1;
      dm.latent_channels = C;
      dm.prompt_len = enc.device.extent(0);
      dm.prompt_dim = enc.device.extent(1);
      dm.seed = (unsigned long long)seed;
      dm.guidance = (double)guidance;
      dm.use_cfg = dump_use_cfg;
      dm.staged = staged;
      dm.img_shapes = img_shapes_str;
      dm.img_mask_len = img_mask.size();
      dm.img_mask_true = img_mask_true;
      if (dm.steps != steps_run)
        std::printf("[dump] note: snapped schedule has %d steps (driver "
                    "reported %d)\n",
                    dm.steps, steps_run);
      write_dump_manifest(dump, dm);
    }

    enc.release();
    // ---- Phase 3: decode (the DiT is released first, then fenced) ----
#ifdef QWEN21_DRIVER_HAVE_DIT
    dit_model.weights().release();
    dit_model.release_scratch();
#else
    w.release();
#endif
    Kokkos::fence();
    QW::print_phase_memory("dit-released", 0, 0);
    size_t oH = 0, oW = 0;
    [[maybe_unused]] int oCh = 1;
    std::vector<unsigned char> pix;
    bool decoded = false;
    if (vae_decode) {
#ifdef QWEN21_DRIVER_HAVE_VAE
      try {
        decoded = try_vae_decode_rgba(lat, C, (size_t)H, (size_t)W,
                                      vae_weights, pix, oH, oW, oCh);
        std::printf("[decode] VAE RGBA %zux%zux%d ready for output\n", oH, oW,
                    oCh);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: real VAE decode failed (%s); writing a "
                     "grayscale preview instead\n",
                     e.what());
      }
#else
      std::printf("[vae] VAE decode requested (default) but "
                  "KokkosDNN_QwenImage21Vae.hpp is not in this build; writing "
                  "a grayscale preview instead (need the VAE worktree "
                  "header + --vae-weights)\n");
#endif
    }
    if (!decoded) {
      pix = render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
      QW::print_phase_memory("decode-preview", 0, 0);
    }
    write_output_image(out, pix, oH, oW, oCh, decoded);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
