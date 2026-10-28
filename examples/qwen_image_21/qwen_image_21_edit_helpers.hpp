// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 edit helpers shared by the example driver and the
// qwen-image-edit-ui sidecar (internal include, NOT installed).
//
// Pure move of the anonymous-namespace helpers formerly defined in
// qwen_image_21_edit.cpp (reference edit wiring: templates, smart-
// resize geometry, PIL-exact + float Lanczos resizes, white-flatten,
// patch packing, M-RoPE positions, raw row loading, CLI usage), so the
// driver and the sidecar cannot drift apart. Needs KOKKOSDNN_ENABLE_IMAGE
// for the image-path helpers, like the driver.
#ifndef KOKKOSDNN_QWEN_IMAGE_21_EDIT_HELPERS_HPP
#define KOKKOSDNN_QWEN_IMAGE_21_EDIT_HELPERS_HPP

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
  const auto lanczos3 = [](double x) {
    constexpr double kPi = 3.14159265358979323846;
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
  constexpr int kFixBits = 22;
  constexpr int64_t kFixRound = int64_t{1} << (kFixBits - 1);
  const auto lanczos3 = [](double x) {
    constexpr double kPi = 3.14159265358979323846;
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

// Multi-image M-RoPE: same get_rope_index walk over the fully expanded
// joint sequence (length S), where vis_rows holds the Nv_total vision rows
// in ascending order and grids/Nvs describe each image's llm block in
// --image order. Text runs advance cur per token; each vision block holds
// T at its block start with H/W grids, then cur advances by
// max(llm_h, llm_w). Inter-slot text (e.g. the space between slots) is
// handled naturally by the scan. Returns 3*S ints row-major.
inline std::vector<int> edit_rope_positions_multi(
    size_t S, const std::vector<int>& vis_rows,
    const std::vector<std::pair<int, int>>& grids,
    const std::vector<size_t>& Nvs) {
  if (grids.size() != Nvs.size())
    throw std::invalid_argument("edit rope: grid/count mismatch");
  size_t Nv_total = 0;
  for (size_t n : Nvs) Nv_total += n;
  if (vis_rows.size() != Nv_total)
    throw std::invalid_argument("edit rope: rows != total vision slots");
  for (size_t i = 0; i < grids.size(); ++i) {
    if (grids[i].first <= 0 || grids[i].second <= 0)
      throw std::invalid_argument("edit rope: bad vision grid");
    if (Nvs[i] != (size_t)grids[i].first * (size_t)grids[i].second)
      throw std::invalid_argument("edit rope: slots != llm_h*llm_w");
  }
  for (size_t k = 1; k < vis_rows.size(); ++k)
    if (vis_rows[k] <= vis_rows[k - 1])
      throw std::invalid_argument("edit rope: vision rows must ascend");
  for (int r : vis_rows)
    if (r < 0 || (size_t)r >= S)
      throw std::invalid_argument("edit rope: vision row out of range");
  std::vector<int> pos(3 * S, 0);
  size_t pi = 0;     // index into vis_rows
  size_t img = 0;    // current image block
  size_t k_in = 0;   // index within the current image block
  int cur = 0;
  int block_start = 0;
  for (size_t s = 0; s < S; ++s) {
    const bool is_vis = (pi < vis_rows.size() && vis_rows[pi] == (int)s);
    if (is_vis) {
      if (img >= grids.size())
        throw std::runtime_error("edit rope: vision row beyond images");
      if (k_in == 0) block_start = cur;
      const int llm_w = grids[img].second;
      const int hr = (int)(k_in / (size_t)llm_w);
      const int wc = (int)(k_in % (size_t)llm_w);
      pos[s] = block_start;
      pos[S + s] = block_start + hr;
      pos[2 * S + s] = block_start + wc;
      ++pi;
      ++k_in;
      if (k_in == Nvs[img]) {
        cur = block_start + std::max(grids[img].first, grids[img].second);
        ++img;
        k_in = 0;
      }
    } else {
      pos[s] = cur;
      pos[S + s] = cur;
      pos[2 * S + s] = cur;
      ++cur;
    }
  }
  if (pi != vis_rows.size() || img != grids.size())
    throw std::runtime_error("edit rope: vision rows do not match images");
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

[[maybe_unused]] void edit_usage() {
  std::printf(
      "qwen_image_21_edit: one-shot Qwen-Image-2.1 reference-image editing\n"
      "Reference: Qwen-Image-2.1-Turbo edit pipeline (diffusers "
      "QwenImage21Pipeline\n"
      "with image=..., use_kv_cache=True). One edit per invocation; no "
      "REPL.\n"
      "\n"
      "  --image PATH            condition image (repeatable, up to 10, in\n"
      "                          --image order like the reference image=[...]\n"
      "                          list; required unless --tiny; RGBA via the\n"
      "                          image component, needs KOKKOSDNN_ENABLE_IMAGE)\n"
      "  --prompt TEXT           edit instruction\n"
      "  --output-resolution N   reference area side (default 1024): each input\n"
      "                          HxW derives from its own aspect (both snap "
      "to\n"
      "                          /32); the output size derives from the LAST\n"
      "                          image's aspect; explicit --height/--width\n"
      "                          override only the output size\n"
      "  --height/--width N      output pixels (override the derived size)\n"
      "  --ckpt DIR              canonical Qwen-Image-2.1-Turbo checkpoint\n"
      "                          root holding transformer/, text_encoder/,\n"
      "                          processor/tokenizer.json, and vae/\n"
      "                          (required unless --tiny; drives the joint\n"
      "                          Qwen forward, DiT, and VAE encode/decode)\n"
      "  --prompt-embeds-in F    joint text+vision embeddings [L,Cc] f32 LE "
      "(torch\n"
      "                          reference dump); else null zeros\n"
      "  --text-weights F        alias of --prompt-embeds-in\n"
      "  --ref-latents-in F      VAE-encoded condition latents ([C,Hc,Wc] "
      "chw,\n"
      "                          or torch [Hc*Wc,C] with --ref-latents-layout "
      "token);\n"
      "                          repeatable: one per --image in the same "
      "order;\n"
      "                          override; default is in-process VAE encode "
      "(from\n"
      "                          <ckpt>/vae), else zeros\n"
      "  --ref-latents-layout L  chw (default) | token\n"
      "  --vis-rows-in F       vision-slot rows (raw int32 LE, embed-space);\n"
      "                          joint build provides them, else this file\n"
      "                          (e.g. golden image_pad_rows.npy converted)\n"
      "                          enables the [cond|target] DiT path\n"
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

#endif  // KOKKOSDNN_QWEN_IMAGE_21_EDIT_HELPERS_HPP
