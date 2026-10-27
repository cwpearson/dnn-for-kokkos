// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// MOSS-Image-SFT text-to-image driver: true-pipeline phase order
// (encode -> release+fence -> DiT sample -> release+fence -> VAE decode)
// with FlowEuler sampling, CFG, and positive-branch SLG.
//
// Backend selection is a compile-time compatibility shim: when the
// true-pipeline headers land in top-level src/ from the parallel worktrees
// (KokkosDNN_MossImageDit.hpp, KokkosDNN_MossImageEncoder.hpp,
// KokkosDNN_MossImageVae.hpp) they are picked up via __has_include and the
// corresponding phase runs the real model; otherwise the phase falls back
// to the placeholder Moss::* geometry / file / null path with a logged
// reason, so the tree always builds. Macros MOSS_DRIVER_HAVE_{DIT,ENCODER,
// VAE} gate every use (see the shim below). Until those headers land, all
// three are 0: encode resolves --prompt-embeds-in files or null, sampling
// runs the placeholder DiT, decode writes a grayscale preview.
//
//   moss_image_generate [--tiny] [--weights transformer_dir]
//                       [--prompt "text"] [--negative-prompt "text"]
//                       [--tokenizer tok_dir] [--text-encoder enc_dir]
//                       [--text-weights feats.f32]
//                       [--prompt-embeds-in feats.f32]
//                       [--height H] [--width W]
//                       [--steps 50] [--cfg 3.0] [--shift 5.0]
//                       [--skip-layers 22]
//                       [--slg-layers 24 --slg-scale 3.0
//                        --slg-start 0.0 --slg-stop 1.0]
//                       [--seed 42]
//                       [--vae-weights vae.safetensors [--no-vae-decode]]
//                       [--out img.png | img.jpg]
//                       [--no-staged]
//
// Text conditioning precedence: in-process Qwen forward (--tokenizer dir
// with tokenizer.json + --text-encoder weights: BPE + chat template +
// on-device Qwen35 forward for prompt and negative prompt, weights
// released before the DiT load) -> --prompt-embeds-in file (alias
// --text-weights) -> null (zeros) with the prompt logged. --tokenizer /
// --text-encoder must be given together. --tiny skips the Qwen forward.
// --vae-weights is validated whenever given; VAE decode runs by default
// (the real VAE decode when the VAE header + weights are present, else a
// channel-mean grayscale preview with a clear log; pass --no-vae-decode
// for the preview).
//
// Staged execution (24GB fit, default): encode -> releases the encoder
// weights (the MiB-scale embeds stay resident on the device) ->
// Kokkos::fence() -> loads the DiT -> samples straight from the
// device-resident embeds -> releases the DiT -> Kokkos::fence() -> decodes,
// so the 16.8GB encoder, the 10.2GB DiT, and the VAE are never resident at
// once. Pass --no-staged for the legacy single-phase run (encoder embeds
// stay live across the DiT load, no fenced encoder-release point). No disk
// round-trip and no second process: the embeds are separate
// (MiB) allocations from the (GiB) encoder weights. --prompt-embeds-in
// remains for loading precomputed file embeds (e.g. from
// scripts/moss_encode.py). Each phase logs its resident estimate (weights
// bytes + activation estimate).
//
// --tiny runs a random-weights smoke model on 4x4 latents (no downloads).
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_MossConfig.hpp"
#include "KokkosDNN_MossModel.hpp"
#include "KokkosDNN_MossSampler.hpp"
#include "KokkosDNN_MossWeights.hpp"

// ---- True-pipeline compatibility shim ------------------------------------
//
// The real MOSS-Image-SFT pipeline interfaces live in models/moss_image/
// as:
//   models/moss_image/KokkosDNN_MossImageDit.hpp      (MossImageDit::Model/Weights)
//   models/moss_image/KokkosDNN_MossImageEncoder.hpp  (MossImageEncoder::encode_prompt)
//   models/moss_image/KokkosDNN_MossImageVae.hpp      (MossImageVae::decode_latents)
//   (+ GroupNorm for the VAE decoder)
// True geometry/ops already merged in models/moss_image/KokkosDNN_MossImage.hpp
// (MossImage::Config: dim 3072, 42 layers, 24 heads, text_dim 4096,
// latent_channels 128) is picked up the same way.
//
// Every true-pipeline use below is gated on its MOSS_DRIVER_HAVE_* macro,
// so this TU builds with or without any of those headers: missing pieces
// fall back to the placeholder Moss::* / file / null path with a logged
// reason. There is intentionally NO hard dependency here (and none in
// CMakeLists.txt): the driver probes, never requires.
#if defined(__has_include)
#if __has_include("KokkosDNN_MossImage.hpp")
#include "KokkosDNN_MossImage.hpp"
#define MOSS_DRIVER_HAVE_MOSSIMAGE_OPS 1
#endif
#if __has_include("KokkosDNN_MossImageDit.hpp")
#include "KokkosDNN_MossImageDit.hpp"
#define MOSS_DRIVER_HAVE_DIT 1
#endif
#if __has_include("KokkosDNN_MossImageEncoder.hpp")
#include "KokkosDNN_MossImageEncoder.hpp"
#define MOSS_DRIVER_HAVE_ENCODER 1
#endif
#if __has_include("KokkosDNN_MossImageVae.hpp")
#include "KokkosDNN_MossImageVae.hpp"
#define MOSS_DRIVER_HAVE_VAE 1
#endif
#endif

// In-repo host-side BPE tokenizer for the Qwen text encoder (always
// available; no backend probe needed).
#include "KokkosDNN_QwenTokenizer.hpp"

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

namespace MO = KokkosDNN::Moss;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = MO::DefaultScalar;  // float (host-side / latent dtype)
#ifdef MOSS_DRIVER_HAVE_DIT
// DiT + encoder checkpoints are bf16 on disk; running them in bhalf_t
// (fp32 accumulation in every kernel) halves residency vs fp32 and is what
// the 24GB fit table assumes. Host-side embeds and latents stay float.
using DScalar = Kokkos::Experimental::bhalf_t;
#else
using DScalar = Scalar;  // placeholder model is float-only
#endif

// ---- Backend geometry traits ---------------------------------------------
// The placeholder (Moss::Config: 24x1536/16ch/cap2048/maxlen512) and the
// true pipeline (MossImage::Config: dim3072/42L/24h/128ch/text4096) expose
// different member names; these overloads let the phase code below stay
// backend-generic. The driver truncation length for the true config is 512
// (the pipeline allows max_sequence_length up to 4096; longer prompts need
// precomputed file embeds at this geometry — see README).
// Backend-generic geometry helpers. The MO::Config (placeholder) overloads
// are only reachable in non-DIT builds (cfg is MossImage::Config when the
// DiT header is present); guard them so toolchain-strict -Wunused-function
// (Clang) stays clean in DIT builds. The MossImage::Config overloads below
// are likewise gated on MOSS_DRIVER_HAVE_MOSSIMAGE_OPS.
#ifndef MOSS_DRIVER_HAVE_DIT
inline size_t be_latent_channels(const MO::Config& c) {
  return (size_t)c.in_channels;
}
inline size_t be_caption_dim(const MO::Config& c) {
  return (size_t)c.caption_channels;
}
inline size_t be_max_seq(const MO::Config& c) {
  return (size_t)c.model_max_length;
}
inline int be_depth(const MO::Config& c) { return c.depth; }
inline int be_patch(const MO::Config& c) { return c.patch_size; }
inline const char* be_label(const MO::Config&) {
  return "moss-sft (placeholder 24x1536/16ch/cap2048 geometry)";
}
#endif  // !MOSS_DRIVER_HAVE_DIT
#ifdef MOSS_DRIVER_HAVE_MOSSIMAGE_OPS
inline size_t be_latent_channels(const KokkosDNN::MossImage::Config& c) {
  return (size_t)c.latent_channels;
}
inline size_t be_caption_dim(const KokkosDNN::MossImage::Config& c) {
  return (size_t)c.text_dim;
}
inline size_t be_max_seq(const KokkosDNN::MossImage::Config&) { return 512; }
inline int be_depth(const KokkosDNN::MossImage::Config& c) {
  return c.num_layers;
}
inline int be_patch(const KokkosDNN::MossImage::Config&) { return 1; }
inline const char* be_label(const KokkosDNN::MossImage::Config&) {
  return "moss-image-sft (true dim3072/42L/24h/128ch/text4096 geometry)";
}
#endif

// Host-side skip-layer validation (negative-branch uncond skips and
// positive-branch SLG skips): every index must sit in [0, depth), mirroring
// the reference _validate_skip_layers (order-free set semantics).
inline std::vector<int> check_skip_layers(const std::vector<int>& layers,
                                          int depth, const char* what) {
  if (depth <= 0)
    throw std::invalid_argument("moss check_skip_layers: depth must be > 0");
  for (int l : layers)
    if (l < 0 || l >= depth)
      throw std::invalid_argument(std::string("moss ") + what +
                                  ": layer index out of range");
  return layers;
}

// Grayscale preview: channel-mean -> min/max normalize -> u8 -> nearest 8x.
std::vector<unsigned char> render_preview(const std::vector<float>& lat,
                                          size_t C, size_t H, size_t W,
                                          size_t& oH, size_t& oW) {
  std::vector<float> gray(H * W, 0.0f);
  for (size_t h = 0; h < H; ++h)
    for (size_t w = 0; w < W; ++w) {
      double acc = 0.0;
      for (size_t c = 0; c < C; ++c)
        acc += lat[(c * H + h) * W + w];
      gray[h * W + w] = (float)(acc / (double)C);
    }
  float mn = gray[0], mx = gray[0];
  for (float v : gray) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  const float span = (mx > mn) ? (mx - mn) : 1.0f;
  constexpr size_t kUp = 8;  // nominal VAE downsample stand-in (preview only)
  oH = H * kUp;
  oW = W * kUp;
  std::vector<unsigned char> img(oH * oW);
  for (size_t h = 0; h < oH; ++h)
    for (size_t w = 0; w < oW; ++w) {
      const float v = gray[(h / kUp) * W + (w / kUp)];
      img[h * oW + w] =
          (unsigned char)(std::lround(255.0f * (v - mn) / span));
    }
  return img;
}

#ifndef KOKKOSDNN_ENABLE_IMAGE
// PPM fallback writers: only the non-image build path (below) uses these;
// guard them so image-enabled builds do not see unused functions.
void save_ppm_rgb(const std::string& path,
                  const std::vector<unsigned char>& rgb, size_t H, size_t W) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f << "P6\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char*>(rgb.data()),
          (long long)rgb.size());
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

void save_ppm(const std::string& path, const std::vector<unsigned char>& gray,
              size_t H, size_t W) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f << "P5\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char*>(gray.data()),
          (long long)gray.size());
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}
#endif

[[maybe_unused]] std::string with_extension(const std::string& path,
    const std::string& ext) {
  const size_t dot = path.find_last_of('.');
  const size_t sep = path.find_last_of("/\\");
  if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
    return path + ext;
  return path.substr(0, dot) + ext;
}

#ifdef KOKKOSDNN_ENABLE_IMAGE
// Output format follows the --out extension: ".jpg"/".jpeg" (case
// insensitive) selects lossy JPEG, anything else selects lossless PNG.
bool out_is_jpeg(const std::string& path) {
  const size_t dot = path.find_last_of('.');
  const size_t sep = path.find_last_of("/\\");
  if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
    return false;
  std::string ext = path.substr(dot);
  for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
  return ext == ".jpg" || ext == ".jpeg";
}
#endif

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

// Weights source: a single .safetensors file or a directory of shards
// (sorted). Shared by --text-encoder and --weights.
inline std::vector<std::string> resolve_shard_list(const std::string& path,
                                                   const std::string& flag) {
  std::vector<std::string> shards;
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    for (const auto& e : std::filesystem::directory_iterator(path, ec)) {
      if (e.is_regular_file() && e.path().extension() == ".safetensors")
        shards.push_back(e.path().string());
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty())
      throw std::runtime_error(flag + " dir '" + path +
                               "' holds no *.safetensors shards");
  } else {
    if (!file_exists(path))
      throw std::runtime_error(flag + " file '" + path + "' not found");
    shards.push_back(path);
  }
  return shards;
}

// Encode stage: owns the caption-embedding state that the text encoder
// produces and hands to the DiT sample phase (cond + uncond/negative).
//
// Conditioning precedence (first hit wins):
//   1. In-process Qwen forward: --tokenizer (dir with tokenizer.json) +
//      --text-encoder (weights file or shard dir), not --tiny. The repo's
//      byte-level BPE (chat template + truncation, reference encode_prompt)
//      tokenizes the prompt and --negative-prompt (default ''), then the
//      Qwen35::EncoderModel forward runs on-device for both branches,
//      matching the reference which encodes the negative the same way
//      whenever CFG is active. Weights are released before the DiT load.
//   2. Raw float32 LE file (--text-weights / --prompt-embeds-in, e.g.
//      scripts/moss_encode.py output): cond from file, uncond zeros.
//   3. Null (zeros) for both, prompt logged only.
// All three paths device-stage both branches into EncoderStage::device /
// device_neg; the sample phase reads straight from those device views, so
// the --staged handoff never touches disk (see main).
//
struct EncoderStage {
  std::vector<float> embeds;      // host [L*Cc] row-major cond features
  std::vector<float> neg_embeds;  // host [L*Cc] row-major uncond features
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> device;  // [L,Cc] cond
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> device_neg;  // [L,Cc] uncond
  size_t L = 0, Cc = 0;

  // Device-stage both cond and uncond embeds (HostSpace + one deep_copy
  // each), the same handoff an on-device encoder output uses. The two views
  // are ~L*Cc floats each (MiB-scale: 512x4096 fp32 ~= 8 MiB) and are
  // separate allocations from the GiB-scale encoder weights, so they can
  // stay resident on the device while the encoder weights are released and
  // the DiT is loaded.
  void stage_device() {
    device = Kokkos::View<float**, Kokkos::LayoutRight, Exec>(
        "moss_prompt_embeds", L, Cc);
    device_neg = Kokkos::View<float**, Kokkos::LayoutRight, Exec>(
        "moss_neg_embeds", L, Cc);
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> stage(
        "moss_prompt_embeds_host", L, Cc);
    std::memcpy(stage.data(), embeds.data(), embeds.size() * sizeof(float));
    Kokkos::deep_copy(device, stage);
    std::memcpy(stage.data(), neg_embeds.data(),
                neg_embeds.size() * sizeof(float));
    Kokkos::deep_copy(device_neg, stage);
    Kokkos::fence();
  }

// In-process Qwen forward (needs MOSS_DRIVER_HAVE_ENCODER): tokenize the
// prompt and negative prompt with the repo's byte-level BPE
// (KokkosDNN_QwenTokenizer, verified against transformers 5.12.1), run the
// batched Qwen35::EncoderModel forward on-device, and fill zero-padded
// [L*Cc] host embeds for both branches. The encoder weights are released
// before returning; only the embeds survive (staged to device by the
// caller), so the encoder and DiT are never resident together.
#ifdef MOSS_DRIVER_HAVE_ENCODER
  static void run_qwen(EncoderStage& st, const std::string& prompt,
                       const std::string& negative_prompt,
                       const std::string& tokenizer_dir,
                       const std::string& encoder_path) {
    namespace QT = KokkosDNN::QwenTokenizer;
    const std::string tok_json = tokenizer_dir + "/tokenizer.json";
    if (!file_exists(tok_json))
      throw std::runtime_error(
          "--tokenizer '" + tokenizer_dir +
          "' has no tokenizer.json (need the MOSS-Image-SFT tokenizer/ dir; "
          "see README)");
    QT::Tokenizer tok;
    tok.load(tok_json);
    const std::vector<int> ids0 = tok.encode(prompt, true, st.L);
    const std::vector<int> ids1 =
        tok.encode(negative_prompt, true, st.L);
    std::printf("[tokenize] prompt seq_len=%zu neg seq_len=%zu (thinking, "
                "truncated to L=%zu)\n",
                ids0.size(), ids1.size(), st.L);

    KokkosDNN::Qwen35::Config qcfg =
        KokkosDNN::Qwen35::Config::moss_image_sft();
    qcfg.validate();
    if ((size_t)qcfg.hidden != st.Cc)
      throw std::runtime_error(
          "Qwen hidden (" + std::to_string(qcfg.hidden) +
          ") != caption dim (" + std::to_string(st.Cc) + ")");
    KokkosDNN::Qwen35::EncoderModel<Exec, Kokkos::Experimental::bhalf_t> enc(
        qcfg);
    const std::vector<std::string> shards =
        resolve_shard_list(encoder_path, "--text-encoder");
    std::printf("[loading %s]%s\n", encoder_path.c_str(),
                shards.size() > 1 ? " (shards)" : "");
    enc.load_safetensors(shards);
    std::printf("[encode] Qwen weights resident: %.1f GB\n",
                (double)enc.resident_bytes() / 1e9);

    // Batched forward over (prompt, negative): one unpadded forward per row
    // with explicit valid lengths, matching the reference encode_prompt
    // (zero-padded batch + bool mask).
    const size_t S = std::max(ids0.size(), ids1.size());
    Kokkos::View<int**, Kokkos::LayoutRight, Exec> ids("moss_qwen_ids", 2, S);
    {
      auto hm = Kokkos::create_mirror_view(ids);
      for (size_t b = 0; b < 2; ++b) {
        const std::vector<int>& src = (b == 0) ? ids0 : ids1;
        for (size_t s = 0; s < S; ++s)
          hm(b, s) = (s < src.size()) ? src[s] : 0;
      }
      Kokkos::deep_copy(ids, hm);
    }
    // bhalf device out (matches the encoder model); converted to float embeds.
    Kokkos::View<Kokkos::Experimental::bhalf_t***, Kokkos::LayoutRight, Exec>
        out("moss_qwen_out", 2, S, (size_t)qcfg.hidden);
    Kokkos::deep_copy(out, Kokkos::Experimental::bhalf_t(0));
    KokkosDNN::DNNHandle<Exec> handle;
    enc.forward(handle, ids, {ids0.size(), ids1.size()}, out);
    Kokkos::fence();
    auto hout = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    st.embeds.assign(st.L * st.Cc, 0.0f);
    st.neg_embeds.assign(st.L * st.Cc, 0.0f);
    for (size_t t = 0; t < ids0.size(); ++t)
      for (size_t h = 0; h < st.Cc; ++h)
        st.embeds[t * st.Cc + h] = (float)hout(0, t, h);
    for (size_t t = 0; t < ids1.size(); ++t)
      for (size_t h = 0; h < st.Cc; ++h)
        st.neg_embeds[t * st.Cc + h] = (float)hout(1, t, h);
    const size_t wbytes = enc.resident_bytes();
    enc.release();
    Kokkos::fence();
    std::printf("[encode] released %.1f GB Qwen weights; %.1f MiB embeds "
                "survive for the sample phase\n",
                (double)wbytes / 1e9,
                (double)(2 * st.L * st.Cc * sizeof(float)) / (1024.0 * 1024.0));
  }
#endif

  template <class CfgT>
  static EncoderStage run(const CfgT& cfg, const std::string& prompt,
                          const std::string& negative_prompt,
                          const std::string& caption_path,
                          const std::string& tokenizer_dir,
                          const std::string& encoder_path, bool tiny) {
    EncoderStage st;
    st.L = be_max_seq(cfg);
    st.Cc = be_caption_dim(cfg);
#ifdef MOSS_DRIVER_HAVE_ENCODER
    // In-process Qwen forward when --tokenizer + --text-encoder are given
    // (and not --tiny: the smoke model uses random weights at test geometry,
    // which real token ids cannot address).
    if (!tiny && (!tokenizer_dir.empty() || !encoder_path.empty())) {
      if (tokenizer_dir.empty() || encoder_path.empty())
        throw std::runtime_error(
            "--tokenizer and --text-encoder must be given together for the "
            "in-process Qwen forward");
      run_qwen(st, prompt, negative_prompt, tokenizer_dir, encoder_path);
      st.stage_device();
      return st;
    }
    if (tiny && (!tokenizer_dir.empty() || !encoder_path.empty()))
      std::printf("[encode] --tiny skips the Qwen forward (random-weights "
                  "smoke model); falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#else
    if (!tokenizer_dir.empty() || !encoder_path.empty())
      std::printf("[encode] --tokenizer/--text-encoder need "
                  "KokkosDNN_MossImageEncoder.hpp, not present in this "
                  "build; falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#endif
    if (!caption_path.empty()) {
      try {
        st.embeds = MO::load_caption_raw(caption_path, (int)st.L, (int)st.Cc);
        std::printf("[caption %s] %zux%zu features\n", caption_path.c_str(),
                    st.L, st.Cc);
      } catch (const std::exception& e) {
        if (tiny) {
          // A full-geometry file cannot condition the tiny smoke model;
          // fall back to null instead of failing the smoke run.
          std::fprintf(stderr,
                       "warning: caption file '%s' unusable here (%s); using "
                       "null embedding\n",
                       caption_path.c_str(), e.what());
          st.embeds.assign(st.L * st.Cc, 0.0f);
        } else {
          throw;
        }
      }
    } else {
      st.embeds.assign(st.L * st.Cc, 0.0f);
      std::printf("[caption null] prompt \"%s\" has no features here; using "
                  "zeros (%zux%zu)\n",
                  prompt.c_str(), st.L, st.Cc);
    }
    st.neg_embeds.assign(st.L * st.Cc, 0.0f);
    if (!negative_prompt.empty())
      std::printf("[caption null] --negative-prompt ignored without the Qwen "
                  "encoder (uncond=zeros)\n");
    st.stage_device();
    return st;
  }

  size_t device_bytes() const {
    return (device.extent(0) * device.extent(1) +
            device_neg.extent(0) * device_neg.extent(1)) *
           sizeof(float);
  }

  // Full teardown: mirrors Weights::release (device + host, cond + neg).
  void release() {
    device = {};
    device_neg = {};
    embeds.clear();
    embeds.shrink_to_fit();
    neg_embeds.clear();
    neg_embeds.shrink_to_fit();
    L = Cc = 0;
  }
};

void log_encode_memory(const EncoderStage& st) {
  MO::print_phase_memory("encode", st.device_bytes(), st.device_bytes());
  std::printf("[budget] full Qwen encoder projects to %.1f GB; it is "
              "released before the DiT load, so the staged peak stays under "
              "24GB (see README fit table)\n",
              MO::kEncoderBudgetBytes / 1e9);
  std::fflush(stdout);
}

// ---- Nearest-grid sigma schedule -----------------------------------------
// Host mirror of MOSSImageScheduler.set_timesteps (reference
// pipeline_moss_image.py): the inference grid with `shift` is snapped to
// the nearest training-grid (shift 1.0, 1000 points) entries, then a
// terminal 0 is appended. unique_consecutive may shrink the count, so the
// caller must use sigmas.size()-1 as the step count (the reference reads
// len(scheduler.timesteps) the same way). Model timestep inputs are
// 1000*sig, and the Euler update is x += (sig[i+1]-sig[i]) * v.
inline std::vector<float> build_sigmas_nearest_grid(int steps, float shift) {
  if (steps <= 0)
    throw std::invalid_argument("moss schedule: steps must be > 0");
  if (!(shift > 0.0f) || !std::isfinite(shift))
    throw std::invalid_argument("moss schedule: shift must be > 0");
  constexpr int kTrain = 1000;
  std::vector<double> infer((size_t)steps);
  {
    const double s = (double)shift;
    const double lower = s / ((double)kTrain + s - 1.0);
    for (int i = 0; i < steps; ++i) {
      const double u = (steps == 1)
                           ? 1.0
                           : 1.0 + (lower - 1.0) * ((double)i /
                                                    (double)(steps - 1));
      infer[(size_t)i] = s * u / (1.0 + (s - 1.0) * u);
    }
  }
  // Ascending training grid: shift 1.0 is the identity map over
  // linspace(1, 0.001), i.e. linspace(0.001, 1) ascending.
  std::vector<double> train((size_t)kTrain);
  for (int j = 0; j < kTrain; ++j)
    train[(size_t)j] = 0.001 + 0.999 * ((double)j / (double)(kTrain - 1));
  std::vector<float> sig;
  sig.reserve((size_t)steps + 1);
  for (double v : infer) {
    const auto up = std::lower_bound(train.begin(), train.end(), v);
    const size_t hi = (up == train.end()) ? (size_t)kTrain - 1
                                          : (size_t)(up - train.begin());
    const size_t lo = (hi == 0) ? 0 : hi - 1;
    // Ties round down, matching the reference torch.where(<=, lower, upper).
    const size_t pick = ((v - train[lo]) <= (train[hi] - v)) ? lo : hi;
    const float snapped = (float)train[pick];
    if (sig.empty() || sig.back() != snapped) sig.push_back(snapped);
  }
  sig.push_back(0.0f);
  return sig;
}

// ---- Backend-generic FlowEuler sample loop --------------------------------
// Reference __call__ semantics (pipeline_moss_image.py):
//   do_cfg = (guidance != 1.0)            (so 0 <= g < 1 takes the CFG branch)
//   per step: v_cond = DiT(x, t, cond)
//             v_uncond = DiT(x, t, neg, skip=uncond_skips)   if do_cfg
//             v = v_uncond + g*(v_cond - v_uncond)           (CFG combine, fp32)
//             v_skip = DiT(x, t, cond, skip=slg_layers)      if SLG active
//             v += slg_scale*(v_cond - v_skip)               (SLG correction)
//             x += (sig[i+1]-sig[i]) * v                     (Euler step)
// Uncond skipping and SLG are both gated on the same window:
//   skip_active = slg_start <= i/n < slg_stop (denoising order); outside it
// the step uses plain CFG with no skipping. With guidance == 1 the SLG-only
// form is v_skip + slg_scale*(v_cond - v_skip).
//
// `forward(x, t, y, out, skips)` is caller-supplied so both backends work:
// the placeholder Model takes a single skip int (adapter below uses the
// first entry), while the true MossImageDit::Model takes the whole set
// (ADAPT: conform the call site to the landed forward signature).
template <class CfgT, class Forward>
std::vector<float> run_sample_loop(
    const CfgT& cfg, Forward&& forward, const std::vector<float>& x_init,
    const std::vector<float>& cond, const std::vector<float>& uncond,
    int steps, float guidance, float shift,
    const std::vector<int>& uncond_skips, const std::vector<int>& slg_layers,
    float slg_scale, float slg_start, float slg_stop, int H, int W,
    const std::function<void(int, float, float)>& on_step = {}) {
  if (steps <= 0)
    throw std::invalid_argument("moss sample: steps must be > 0");
  if (!std::isfinite(guidance) || guidance < 0.0f)
    throw std::invalid_argument(
        "moss sample: guidance_scale must be finite and >= 0");
  if (!(shift > 0.0f) || !std::isfinite(shift))
    throw std::invalid_argument("moss sample: shift must be > 0");
  if (!std::isfinite(slg_scale) || slg_scale < 0.0f)
    throw std::invalid_argument("moss sample: slg_scale must be finite >= 0");
  if (!(0.0f <= slg_start && slg_start <= slg_stop && slg_stop <= 1.0f))
    throw std::invalid_argument(
        "moss sample: require 0 <= slg_start <= slg_stop <= 1");
  const int depth = be_depth(cfg);
  check_skip_layers(uncond_skips, depth, "uncond_skip_layers");
  check_skip_layers(slg_layers, depth, "slg_layers");
  const size_t C = be_latent_channels(cfg);
  const size_t Cc = be_caption_dim(cfg);
  const size_t HH = (size_t)H, WW = (size_t)W;
  if (HH == 0 || WW == 0)
    throw std::invalid_argument("moss sample: bad latent geometry");
  const size_t T = HH * WW;
  if (x_init.size() != C * HH * WW)
    throw std::invalid_argument("moss sample: x_init size mismatch");
  if (cond.size() % Cc != 0 || cond.empty())
    throw std::invalid_argument("moss sample: cond size mismatch");
  const size_t L = cond.size() / Cc;
  if (L == 0 || L > be_max_seq(cfg))
    throw std::invalid_argument("moss sample: caption length out of range");
  const bool do_cfg = (guidance != 1.0f);
  const std::vector<float>& unc = uncond.empty() ? cond : uncond;
  if (unc.size() != cond.size())
    throw std::invalid_argument("moss sample: uncond size mismatch");
  // Identical cond/uncond make guidance a no-op: run a single branch.
  const bool same = (unc.data() == cond.data()) || (unc == cond);
  const bool need_cfg = do_cfg && !same;
  const bool want_slg = !slg_layers.empty() && slg_scale > 0.0f;

  using LatR = Kokkos::View<DScalar****, Kokkos::LayoutRight, Exec>;
  using TokR = Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec>;
  using TVec = Kokkos::View<float*, Exec>;
  LatR x_d("moss_loop_x", 1, C, HH, WW);
  LatR vc_d("moss_loop_vc", 1, C, HH, WW);
  LatR vu_d("moss_loop_vu", 1, C, HH, WW);
  LatR vs_d("moss_loop_vs", 1, C, HH, WW);
  TVec t_d("moss_loop_t", 1);
  TokR y_cond_d("moss_loop_y_cond", 1, L, Cc);
  TokR y_unc_d("moss_loop_y_unc", 1, L, Cc);
  // Device residency: copy evolving state + captions once, keep x on device
  // across steps (only the scalar timestep crosses host->device per step).
  {
    auto xh = Kokkos::create_mirror_view(x_d);
    size_t k = 0;
    for (size_t c = 0; c < C; ++c)
      for (size_t h = 0; h < HH; ++h)
        for (size_t w = 0; w < WW; ++w) xh(0, c, h, w) = (DScalar)x_init[k++];
    Kokkos::deep_copy(x_d, xh);
  }
  auto fill_caption = [](TokR& dst, const std::vector<float>& y, size_t Lc,
                         size_t Cc_) {
    auto yh = Kokkos::create_mirror_view(dst);
    size_t k = 0;
    for (size_t l = 0; l < Lc; ++l)
      for (size_t c = 0; c < Cc_; ++c) yh(0, l, c) = (DScalar)y[k++];
    Kokkos::deep_copy(dst, yh);
  };
  fill_caption(y_cond_d, cond, L, Cc);
  fill_caption(y_unc_d, unc, L, Cc);

  const std::vector<float> sigmas = build_sigmas_nearest_grid(steps, shift);
  std::vector<float> tvals(sigmas.size());
  for (size_t i = 0; i < sigmas.size(); ++i) tvals[i] = 1000.0f * sigmas[i];
  const int n = (int)sigmas.size() - 1;
  const float g = guidance, s = slg_scale;
  const std::vector<int> no_skip;
  auto t_h = Kokkos::create_mirror_view(t_d);
  for (int i = 0; i < n; ++i) {
    if (on_step) on_step(i, tvals[(size_t)i], sigmas[(size_t)i]);
    // Scalar timestep only: single float host->device per step.
    t_h(0) = tvals[(size_t)i];
    Kokkos::deep_copy(t_d, t_h);
    const float dt = sigmas[(size_t)i + 1] - sigmas[(size_t)i];
    const float frac = (float)i / (float)n;
    const bool skip_active =
        (slg_start <= frac) && (frac < slg_stop);  // gates uncond skip + SLG
    // No fences inside the step: every forward, deep_copy, and kernel below
    // runs on the default execution space stream (the cached cuBLAS/cuDNN
    // handles re-select that stream per call), so launch order already
    // guarantees correctness. The host consumes device data only after the
    // loop, where a single fence guards the final device->host copy.
    // Conditional branch (full depth).
    forward(x_d, t_d, y_cond_d, vc_d, no_skip);
    const bool slg_now = want_slg && skip_active;
    if (need_cfg) {
      // Unconditional branch with negative-branch skipping (window-gated).
      // Same-stream ordering covers the forward -> combine read-after-write.
      forward(x_d, t_d, y_unc_d, vu_d,
              skip_active ? uncond_skips : no_skip);
      if (slg_now) {
        // Positive-branch skipped forward + SLG correction.
        forward(x_d, t_d, y_cond_d, vs_d, slg_layers);
      }
      Kokkos::parallel_for(
          "moss_loop_cfg_slg_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float vcond = static_cast<float>(vc_d(b, c, h, w));
            const float vun = static_cast<float>(vu_d(b, c, h, w));
            float v = vun + g * (vcond - vun);
            if (slg_now)
              v += s * (vcond - static_cast<float>(vs_d(b, c, h, w)));
            x_d(b, c, h, w) = static_cast<DScalar>(xc + dt * v);
          });
    } else if (slg_now) {
      forward(x_d, t_d, y_cond_d, vs_d, slg_layers);
      Kokkos::parallel_for(
          "moss_loop_slg_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float vcond = static_cast<float>(vc_d(b, c, h, w));
            const float vsk = static_cast<float>(vs_d(b, c, h, w));
            x_d(b, c, h, w) =
                static_cast<DScalar>(xc + dt * (vsk + s * (vcond - vsk)));
          });
    } else {
      Kokkos::parallel_for(
          "moss_loop_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float v = static_cast<float>(vc_d(b, c, h, w));
            x_d(b, c, h, w) = static_cast<DScalar>(xc + dt * v);
          });
    }
  }
  // Single fence for the whole loop: the host consumes device data only at
  // the final device->host copy below.
  Kokkos::fence();
  // Single device->host copy of the final latents.
  auto xh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_d);
  std::vector<float> x(C * T);
  size_t k = 0;
  for (size_t c = 0; c < C; ++c)
    for (size_t h = 0; h < HH; ++h)
      for (size_t w = 0; w < WW; ++w) x[k++] = static_cast<float>(xh(0, c, h, w));
  return x;
}

// ---- Device-resident variant -----------------------------------------------
// Same reference CFG/SLG/nearest-grid semantics as run_sample_loop, but the
// captions are sourced from the EncoderStage device views (rank-2 [L,Cc])
// that stayed resident on the device across the encoder-release / DiT-load
// boundary. The only copies are device->device (one parallel_for per
// branch into the [1,L,Cc] loop views); nothing round-trips through host
// or disk. Falls back to the host-vector path is intentionally NOT done
// here: callers must guarantee cond_d/uncond_d are allocated.
template <class CfgT, class Forward>
std::vector<float> run_sample_loop_from_device(
    const CfgT& cfg, Forward&& forward, const std::vector<float>& x_init,
    const Kokkos::View<float**, Kokkos::LayoutRight, Exec>& cond_d,
    const Kokkos::View<float**, Kokkos::LayoutRight, Exec>& uncond_d,
    int steps, float guidance, float shift,
    const std::vector<int>& uncond_skips, const std::vector<int>& slg_layers,
    float slg_scale, float slg_start, float slg_stop, int H, int W,
    const std::function<void(int, float, float)>& on_step = {}) {
  if (steps <= 0)
    throw std::invalid_argument("moss sample: steps must be > 0");
  if (!std::isfinite(guidance) || guidance < 0.0f)
    throw std::invalid_argument(
        "moss sample: guidance_scale must be finite and >= 0");
  if (!(shift > 0.0f) || !std::isfinite(shift))
    throw std::invalid_argument("moss sample: shift must be > 0");
  if (!std::isfinite(slg_scale) || slg_scale < 0.0f)
    throw std::invalid_argument("moss sample: slg_scale must be finite >= 0");
  if (!(0.0f <= slg_start && slg_start <= slg_stop && slg_stop <= 1.0f))
    throw std::invalid_argument(
        "moss sample: require 0 <= slg_start <= slg_stop <= 1");
  if (!cond_d.is_allocated() || !uncond_d.is_allocated())
    throw std::invalid_argument("moss sample: device captions not staged");
  const int depth = be_depth(cfg);
  check_skip_layers(uncond_skips, depth, "uncond_skip_layers");
  check_skip_layers(slg_layers, depth, "slg_layers");
  const size_t C = be_latent_channels(cfg);
  const size_t Cc = be_caption_dim(cfg);
  const size_t HH = (size_t)H, WW = (size_t)W;
  if (HH == 0 || WW == 0)
    throw std::invalid_argument("moss sample: bad latent geometry");
  const size_t T = HH * WW;
  if (x_init.size() != C * HH * WW)
    throw std::invalid_argument("moss sample: x_init size mismatch");
  const size_t L = cond_d.extent(0);
  if (cond_d.extent(1) != Cc || uncond_d.extent(0) != L ||
      uncond_d.extent(1) != Cc)
    throw std::invalid_argument("moss sample: device caption dim mismatch");
  if (L == 0 || L > be_max_seq(cfg))
    throw std::invalid_argument("moss sample: caption length out of range");
  const bool do_cfg = (guidance != 1.0f);
  // Same no-op rule as the host path: identical cond/uncond make guidance
  // a no-op. The captions are MiB-scale, so mirror once for the equality
  // test (the sample loop itself never touches the host).
  bool same = false;
  {
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cond_d);
    auto hu =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uncond_d);
    same = true;
    for (size_t l = 0; same && l < L; ++l)
      for (size_t c = 0; c < Cc; ++c)
        if (hc(l, c) != hu(l, c)) {
          same = false;
          break;
        }
  }
  const bool use_cfg = do_cfg && !same;
  const bool want_slg = !slg_layers.empty() && slg_scale > 0.0f;

  using LatR = Kokkos::View<DScalar****, Kokkos::LayoutRight, Exec>;
  using TokR = Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec>;
  using TVec = Kokkos::View<float*, Exec>;
  LatR x_d("moss_loop_x", 1, C, HH, WW);
  LatR vc_d("moss_loop_vc", 1, C, HH, WW);
  LatR vu_d("moss_loop_vu", 1, C, HH, WW);
  LatR vs_d("moss_loop_vs", 1, C, HH, WW);
  TVec t_d("moss_loop_t", 1);
  TokR y_cond_d("moss_loop_y_cond", 1, L, Cc);
  TokR y_unc_d("moss_loop_y_unc", 1, L, Cc);
  {
    auto xh = Kokkos::create_mirror_view(x_d);
    size_t k = 0;
    for (size_t c = 0; c < C; ++c)
      for (size_t h = 0; h < HH; ++h)
        for (size_t w = 0; w < WW; ++w) xh(0, c, h, w) = (DScalar)x_init[k++];
    Kokkos::deep_copy(x_d, xh);
  }
  // Device->device caption staging: no host or disk round-trip.
  Kokkos::parallel_for(
      "moss_stage_cond",
      Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>({0, 0}, {L, Cc}),
      KOKKOS_LAMBDA(size_t l, size_t c) {
        y_cond_d(0, l, c) = static_cast<DScalar>(cond_d(l, c));
        y_unc_d(0, l, c) = static_cast<DScalar>(uncond_d(l, c));
      });
  Kokkos::fence();

  const std::vector<float> sigmas = build_sigmas_nearest_grid(steps, shift);
  std::vector<float> tvals(sigmas.size());
  for (size_t i = 0; i < sigmas.size(); ++i) tvals[i] = 1000.0f * sigmas[i];
  const int n = (int)sigmas.size() - 1;
  const float g = guidance, s = slg_scale;
  const std::vector<int> no_skip;
  auto t_h = Kokkos::create_mirror_view(t_d);
  for (int i = 0; i < n; ++i) {
    if (on_step) on_step(i, tvals[(size_t)i], sigmas[(size_t)i]);
    t_h(0) = tvals[(size_t)i];
    Kokkos::deep_copy(t_d, t_h);
    const float dt = sigmas[(size_t)i + 1] - sigmas[(size_t)i];
    const float frac = (float)i / (float)n;
    const bool skip_active =
        (slg_start <= frac) && (frac < slg_stop);
    forward(x_d, t_d, y_cond_d, vc_d, no_skip);
    const bool slg_now = want_slg && skip_active;
    if (use_cfg) {
      forward(x_d, t_d, y_unc_d, vu_d,
              skip_active ? uncond_skips : no_skip);
      if (slg_now) {
        forward(x_d, t_d, y_cond_d, vs_d, slg_layers);
      }
      Kokkos::parallel_for(
          "moss_loop_cfg_slg_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float vcond = static_cast<float>(vc_d(b, c, h, w));
            const float vun = static_cast<float>(vu_d(b, c, h, w));
            float v = vun + g * (vcond - vun);
            if (slg_now)
              v += s * (vcond - static_cast<float>(vs_d(b, c, h, w)));
            x_d(b, c, h, w) = static_cast<DScalar>(xc + dt * v);
          });
    } else if (slg_now) {
      forward(x_d, t_d, y_cond_d, vs_d, slg_layers);
      Kokkos::parallel_for(
          "moss_loop_slg_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float vcond = static_cast<float>(vc_d(b, c, h, w));
            const float vsk = static_cast<float>(vs_d(b, c, h, w));
            x_d(b, c, h, w) =
                static_cast<DScalar>(xc + dt * (vsk + s * (vcond - vsk)));
          });
    } else {
      Kokkos::parallel_for(
          "moss_loop_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float v = static_cast<float>(vc_d(b, c, h, w));
            x_d(b, c, h, w) = static_cast<DScalar>(xc + dt * v);
          });
    }
  }
  // Single fence for the whole loop: the host consumes device data only at
  // the final device->host copy below.
  Kokkos::fence();
  auto xh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_d);
  std::vector<float> x(C * T);
  size_t k = 0;
  for (size_t c = 0; c < C; ++c)
    for (size_t h = 0; h < HH; ++h)
      for (size_t w = 0; w < WW; ++w) x[k++] = static_cast<float>(xh(0, c, h, w));
  return x;
}

#ifdef MOSS_DRIVER_HAVE_DIT
// True DiT forward: (x_d [1,C,H,W], t_d [1], y [1,L,Cc]) -> out. The loop
// captions are always all-valid, so this uses the mask-free
// forward_all_valid entry: no per-forward mask allocation + fill and no
// mask device->host copy + host scan in the preamble (previously
// `mask("moss_loop_mask")` + deep_copy per forward). Cc == text_dim and
// C == latent_channels by construction of the backend traits above.
// Namespace scope: NVCC rejects function-local
// closure types as the Forward template argument above extended device
// lambdas, so the forward callable is a named functor.
struct DitForward {
  KokkosDNN::MossImageDit::MossImageDitModel<Exec, DScalar>* model;
  template <class X, class T, class Y, class Out>
  void operator()(const X& x, const T& t, const Y& y, Out& out,
                  const std::vector<int>& skips) const {
    model->forward_all_valid(x, t, y, out, skips);
  }
};
#else
// Placeholder-model forward (single skip per forward). Same namespace-scope
// requirement as above.
struct DitForward {
  MO::Model<Exec, DScalar>* model;
  mutable bool slg_multi_warned = false;
  template <class X, class T, class Y, class Out>
  void operator()(const X& x, const T& t, const Y& y, Out& out,
                  const std::vector<int>& skips) const {
    const int skip = skips.empty() ? -1 : skips.front();
    if (skips.size() > 1 && !slg_multi_warned) {
      slg_multi_warned = true;
      std::fprintf(stderr,
                   "warning: placeholder Model applies one skip per "
                   "forward; using the first of %zu --slg-layers\n",
                   skips.size());
    }
    model->forward(x, t, y, out, skip);
  }
};
#endif

void usage() {
  std::printf(
      "usage: moss_image_generate [--tiny] [--weights transformer_dir]\n"
      "                    [--prompt \"text\"] [--negative-prompt \"text\"]\n"
      "                    [--tokenizer tok_dir] [--text-encoder enc_dir]\n"
      "                    [--text-weights feats.f32]\n"
      "                    [--prompt-embeds-in feats.f32]\n"
      "                    [--height H] [--width W]\n"
      "                    [--steps N] [--cfg G] [--shift S]\n"
      "                    [--skip-layers N]\n"
      "                    [--slg-layers A,B --slg-scale S\n"
      "                     --slg-start F --slg-stop F]\n"
      "                    [--seed N]\n"
      "                    [--vae-weights vae.safetensors [--no-vae-decode]]\n"
      "                    [--out img.png|img.jpg]\n"
      "                    [--no-staged]\n"
      "\n"
      "  --tiny         random-weights smoke model, 4x4 latents (no files)\n"
      "  --weights      DiT transformer checkpoint: transformer/ shard dir\n"
      "                 or single .safetensors (default: moss_image_sft.safetensors)\n"
      "  --prompt       caption text (Qwen-encoded in-process with\n"
      "                 --tokenizer + --text-encoder, else needs\n"
      "                 --text-weights, else null)\n"
      "  --negative-prompt negative caption (encoded in-process alongside\n"
      "                 the prompt, else zeros)\n"
      "  --tokenizer    dir holding tokenizer.json (Qwen2Tokenizer files);\n"
      "                 with --text-encoder runs the in-process Qwen forward\n"
      "  --text-weights raw float32 LE caption features [L,Cc]\n"
      "                 (L <= maxlen, Cc == caption dim)\n"
      "  --text-encoder Qwen text-encoder weights: text_encoder/ shard dir\n"
      "                 or single .safetensors\n"
      "  --prompt-embeds-in alias of --text-weights (wins if both differ)\n"
      "  --staged       (default ON) encode -> release encoder weights (MiB\n"
      "                 embeds stay resident on device) -> fence -> load DiT\n"
      "                 -> sample from device embeds -> release DiT -> fence\n"
      "                 -> decode (one component resident at a time, no disk\n"
      "                 round-trip). Pass --no-staged for the legacy\n"
      "                 single-phase run\n"
      "  --height/--width latent H/W (default 64; x16 = pixels, so 64\n"
      "                 means 1024px; placeholder path needs multiples of 16)\n"
      "  --steps        Euler steps (default 50)\n"
      "  --cfg/--guidance CFG scale (!=1 takes the CFG branch; default 3.0)\n"
      "  --shift/--flow-shift flow-matching sigma shift (default 5.0)\n"
      "  --skip-layers/--skip-layer uncond-branch skip layer(s)\n"
      "                 (single id; -1 disables; default 22)\n"
      "  --slg-layers   comma-separated block ids for a positive-branch\n"
      "                 skipped forward, e.g. 24 (empty disables)\n"
      "  --slg-scale    SLG correction scale (default 0 = off)\n"
      "  --slg-start/--slg-stop SLG+uncond-skip window as a fraction of\n"
      "                 steps in denoising order (default 0..1)\n"
      "  --seed         latent init seed (default 42)\n"
      "  --vae-weights  VAE checkpoint (validated whenever given)\n"
      "  --vae-decode   (default ON) run the real VAE decode when the VAE\n"
      "                 header + weights are present (else grayscale preview).\n"
      "                 Pass --no-vae-decode for the preview\n"
      "  --out          PNG (.jpg/.jpeg writes JPEG) with "
      "KokkosDNN_ENABLE_IMAGE=ON, else PPM\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "moss_image_sft.safetensors";
  std::string prompt = "a red panda reading a newspaper";
  std::string negative_prompt;
  std::string text_weights, prompt_embeds_in;
  std::string tokenizer_dir, text_encoder;
  std::string vae_weights, out = "moss_out.png";
  int height = 64, width = 64;
  int steps = 50;
  float guidance = 3.0f, flow_shift = 5.0f;
  int skip_layers = 22;
  std::string slg_layers_str;
  float slg_scale = 0.0f, slg_start = 0.0f, slg_stop = 1.0f;
  uint64_t seed = 42;
  bool tiny = false;
  bool staged = true, vae_decode = true;

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
    if (a == "--weights") {
      if (!need(weights)) return 1;
      tiny = false;
    } else if (a == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (a == "--negative-prompt" && i + 1 < argc) {
      negative_prompt = argv[++i];
    } else if (a == "--tokenizer" && i + 1 < argc) {
      tokenizer_dir = argv[++i];
    } else if (a == "--text-encoder" && i + 1 < argc) {
      text_encoder = argv[++i];
    } else if (a == "--text-weights" && i + 1 < argc) {
      text_weights = argv[++i];
    } else if (a == "--prompt-embeds-in" && i + 1 < argc) {
      prompt_embeds_in = argv[++i];
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
    } else if ((a == "--cfg" || a == "--guidance") && i + 1 < argc) {
      guidance = std::stof(argv[++i]);
    } else if ((a == "--shift" || a == "--flow-shift") && i + 1 < argc) {
      flow_shift = std::stof(argv[++i]);
    } else if ((a == "--skip-layers" || a == "--skip-layer") && i + 1 < argc) {
      skip_layers = std::stoi(argv[++i]);
    } else if (a == "--slg-layers" && i + 1 < argc) {
      slg_layers_str = argv[++i];
    } else if (a == "--slg-scale" && i + 1 < argc) {
      slg_scale = std::stof(argv[++i]);
    } else if (a == "--slg-start" && i + 1 < argc) {
      slg_start = std::stof(argv[++i]);
    } else if (a == "--slg-stop" && i + 1 < argc) {
      slg_stop = std::stof(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      seed = (uint64_t)std::stoull(argv[++i]);
    } else if (a == "--tiny") {
      tiny = true;
    } else if (a == "--staged") {
      staged = true;
    } else if (a == "--no-staged" || a == "--no-stage") {
      staged = false;
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "warning: ignoring unknown arg '%s'\n", a.c_str());
    }
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    KokkosDNN::print_configuration(std::cout);
    std::printf("[backend dit=%s encoder=%s vae=%s]\n",
#ifdef MOSS_DRIVER_HAVE_DIT
                "moss-image-dit",
#else
                "placeholder",
#endif
#ifdef MOSS_DRIVER_HAVE_ENCODER
                "qwen",
#else
                "file/null",
#endif
#ifdef MOSS_DRIVER_HAVE_VAE
                "moss-image-vae"
#else
                "preview"
#endif
    );
#ifdef MOSS_DRIVER_HAVE_DIT
    // True-pipeline DiT header is present: use the merged MossImage geometry
    // (dim3072/42L/24h/128ch/text4096) with the Dit model/weights.
    // ADAPT: conform the MossImageDit:: names below to the landed
    // KokkosDNN_MossImageDit.hpp (assumed mirror of the Moss::* API: Config
    // from MossImage, Weights with random_init/load_safetensors/
    // load_torch_pth/resident_bytes/release, Model(cfg, w) with a
    // skip-layer-set forward).
    KokkosDNN::MossImage::Config cfg =
        tiny ? KokkosDNN::MossImage::Config::tiny_test()
             : KokkosDNN::MossImage::Config::moss_image_sft();
    cfg.validate();
#else
    MO::Config cfg = tiny ? MO::Config::tiny_test() : MO::Config::moss_sft();
    cfg.validate();
#endif
    const int H = tiny ? 4 : height, W = tiny ? 4 : width;
    if (!tiny) {
      // Driver H/W are latent dims (x16 = pixels, so 64 means 1024px, the
      // reference default). The placeholder path additionally requires
      // multiples of 16 (historical); the true path only needs positive
      // dims that tile the patch.
      if (H <= 0 || W <= 0)
        throw std::runtime_error("--height/--width must be positive (got " +
                                 std::to_string(height) + "x" +
                                 std::to_string(width) + ")");
#ifndef MOSS_DRIVER_HAVE_DIT
      if (H % 16 != 0 || W % 16 != 0)
        throw std::runtime_error(
            "--height/--width must be positive multiples of 16 (got " +
            std::to_string(height) + "x" + std::to_string(width) + ")");
#endif
      if (H % be_patch(cfg) != 0 || W % be_patch(cfg) != 0)
        throw std::runtime_error("--height/--width must tile patch_size=" +
                                 std::to_string(be_patch(cfg)));
    }
    const size_t C = be_latent_channels(cfg);
    std::printf("[model %s: %s]\n", tiny ? "tiny_test" : "moss-sft",
                be_label(cfg));
    if (tiny && skip_layers >= be_depth(cfg)) {
      std::fprintf(stderr,
                   "warning: --skip-layers %d >= tiny depth %d; disabling "
                   "uncond skipping (-1) for the smoke run\n",
                   skip_layers, be_depth(cfg));
      skip_layers = -1;
    }
    // Positive-branch SLG layer set (--slg-layers A,B,...; empty disables).
    std::vector<int> slg_layers;
    if (!slg_layers_str.empty()) {
      const char* p = slg_layers_str.c_str();
      while (*p) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p)
          throw std::runtime_error("cannot parse --slg-layers '" +
                                   slg_layers_str + "'");
        slg_layers.push_back((int)v);
        p = end;
      }
      check_skip_layers(slg_layers, be_depth(cfg), "slg_layers");
      if (tiny) {
        std::vector<int> kept;
        for (int l : slg_layers)
          if (l < be_depth(cfg))
            kept.push_back(l);
          else
            std::fprintf(stderr,
                         "warning: dropping SLG layer %d >= tiny depth %d\n",
                         l, be_depth(cfg));
        slg_layers.swap(kept);
      }
    }
    const std::vector<int> uncond_skips =
        (skip_layers < 0) ? std::vector<int>{} : std::vector<int>{skip_layers};
    if (!uncond_skips.empty())
      check_skip_layers(uncond_skips, be_depth(cfg), "uncond_skip_layers");

    // Resolve the caption source: --prompt-embeds-in is an alias of
    // --text-weights and wins when both are given with different values.
    std::string caption_src = text_weights;
    if (!prompt_embeds_in.empty()) {
      if (!text_weights.empty() && prompt_embeds_in != text_weights)
        std::fprintf(stderr,
                     "warning: --prompt-embeds-in wins over --text-weights\n");
      caption_src = prompt_embeds_in;
    }

    // ---- Phase 1: encode (no DiT weights are loaded yet) ----
    EncoderStage enc = EncoderStage::run(cfg, prompt, negative_prompt,
                                         caption_src, tokenizer_dir,
                                         text_encoder, tiny);
    log_encode_memory(enc);

    // In-memory handoff: the embeds (~L*Cc floats per branch, MiB-scale) are
    // separate device allocations from the GiB-scale encoder weights, so they
    // stay resident on the device while the encoder weights are released and
    // the DiT is loaded. No disk round-trip and no second process.
    if (staged) {
      // When the real Qwen encoder lands in this TU, its EncoderModel weights
      // are released here; the EncoderStage embeds are independent Views and
      // survive. Today the weights live in scripts/moss_encode_qwen.py, so
      // this is a fence + accounting point only.
      Kokkos::fence();
      MO::print_phase_memory("encode-released", enc.device_bytes(), 0);
      std::printf("[staged] keeping %.1f MiB of embeds resident on device "
                  "across the DiT load (no disk round-trip)\n",
                  (double)enc.device_bytes() / (1024.0 * 1024.0));
    } else {
      std::printf("[note] single-phase run (--no-staged): encoder embeds stay\n"
                  "                 live across the DiT load (no fenced, logged\n"
                  "                 encoder-release point)\n");
    }

    // ---- Phase 2: DiT load ----
#ifdef MOSS_DRIVER_HAVE_DIT
    // True-pipeline DiT (MossImageDit::MossImageDitModel owns its weights).
    KokkosDNN::MossImageDit::MossImageDitModel<Exec, DScalar> dit_model(cfg);
    {
      auto& w = dit_model.weights();
      if (tiny) {
        w.random_init(seed);
        std::printf("[tiny random weights] dim=%d depth=%d latent=%dx%dx%zu\n",
                    cfg.dim, cfg.num_layers, H, W, C);
      } else {
        std::printf("[loading %s]\n", weights.c_str());
        try {
          const std::vector<std::string> shards =
              resolve_shard_list(weights, "--weights");
          if (shards.size() > 1)
            std::printf("[loading %zu shards]\n", shards.size());
          bool is_st = false;
          for (const auto& s : shards)
            if (s.size() >= 12 &&
                s.compare(s.size() - 12, 12, ".safetensors") == 0)
              is_st = true;
          if (!is_st)
            throw std::runtime_error(
                "only .safetensors transformer shards are supported "
                "(convert .pth via the README layout)");
          w.load_safetensors(shards);
        } catch (const std::exception& e) {
          throw std::runtime_error(std::string(e.what()) +
                                   " (need MOSS weights or --tiny)");
        }
        std::printf("[loaded] depth=%d dim=%d\n", cfg.num_layers, cfg.dim);
      }
      MO::print_phase_memory("sample", w.resident_bytes(), 0);
    }
    std::printf("[budget] DiT transient activations run ~2-4 GiB at 1024px "
                "with CFG 2x (see README fit table); exact per-config "
                "estimate lands with the Dit header\n");
#else
    MO::Weights<Exec, DScalar> w;
    if (tiny) {
      MO::random_init(cfg, w, seed);
      std::printf("[tiny random weights] hidden=%d depth=%d latent=%dx%d\n",
                  cfg.hidden_size, cfg.depth, H, W);
    } else {
      std::printf("[loading %s]\n", weights.c_str());
      if (!file_exists(weights))
        throw std::runtime_error("weights file '" + weights +
                                 "' not found (need a MOSS checkpoint or "
                                 "--tiny). See README for the expected key "
                                 "layout.");
      try {
        if (weights.size() >= 11 &&
            weights.compare(weights.size() - 11, 11, ".safetensors") == 0)
          MO::load_safetensors(cfg, weights, w);
        else
          MO::load_torch_pth(cfg, weights, w);
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) +
                                 " (need MOSS weights or --tiny)");
      }
      std::printf("[loaded] depth=%d hidden=%d\n", cfg.depth, cfg.hidden_size);
    }
    MO::print_phase_memory("sample", w.resident_bytes(),
                           MO::dit_activation_bytes(cfg, H, W,
                                                    sizeof(DScalar)));
#endif
    std::printf("[budget] full-checkpoint projection: DiT %.1f GB + VAE %.2f "
                "GB + activations vs 24GB (see README fit table)\n",
                MO::kDitBudgetBytes / 1e9, MO::kVaeBudgetBytes / 1e9);

    if (!vae_weights.empty()) {
      if (!file_exists(vae_weights))
        throw std::runtime_error("vae weights file '" + vae_weights +
                                 "' not found");
#ifdef MOSS_DRIVER_HAVE_VAE
      std::printf("[vae %s] validated%s\n", vae_weights.c_str(),
                  vae_decode ? "; decode enabled (default)"
                             : "; preview (pass --vae-decode to decode)");
#else
      std::printf("[vae %s] validated, but KokkosDNN_MossImageVae.hpp is not "
                  "in this build; writing a grayscale preview instead\n",
                  vae_weights.c_str());
#endif
    } else if (vae_decode) {
      std::printf("[vae] VAE decode needs --vae-weights (else grayscale "
                  "preview); disabling decode for this run\n");
      vae_decode = false;
    }

    std::vector<float> x_init = MO::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    std::printf("[sample] H=%d W=%d steps=%d cfg=%.3g shift=%.3g "
                "skip_layers=%d slg_layers=%s slg_scale=%.3g window=[%.3g,%.3g) "
                "seed=%llu\n",
                H, W, steps, (double)guidance, (double)flow_shift, skip_layers,
                slg_layers_str.empty() ? "-" : slg_layers_str.c_str(),
                (double)slg_scale, (double)slg_start, (double)slg_stop,
                (unsigned long long)seed);
#ifdef MOSS_DRIVER_HAVE_DIT
    DitForward dit_forward{&dit_model};
#else
    MO::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
    const auto t0 = std::chrono::steady_clock::now();
    // Sample straight from the device-resident embeds staged above: no host
    // reload, no disk re-read. The host vectors (enc.embeds/neg_embeds) stay
    // as the fallback for the --prompt-embeds-in file path, which also lands
    // staged on-device via EncoderStage::run.
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps,
        guidance, flow_shift, uncond_skips, slg_layers, slg_scale, slg_start,
        slg_stop, H, W, [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        });
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("[done] %.2fs total, %.2fs/step\n", secs,
                secs / (double)std::max(steps, 1));

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

    // Captions are no longer needed: drop host + device copies before decode.
    enc.release();
    // ---- Phase 3: decode (the DiT is released first, then fenced) ----
#ifdef MOSS_DRIVER_HAVE_DIT
    dit_model.weights().release();
    dit_model.release_scratch();
#else
    w.release();
#endif
    Kokkos::fence();
    MO::print_phase_memory("dit-released", 0, 0);
    size_t oH = 0, oW = 0;
    // Only consumed by the PNG path (KOKKOSDNN_ENABLE_IMAGE); the PPM
    // fallback below ignores it.
    [[maybe_unused]] int oCh = 1;
    std::vector<unsigned char> pix;
    bool decoded = false;
    if (vae_decode) {
#ifdef MOSS_DRIVER_HAVE_VAE
      std::printf("[vae decode %s] full VAE decode of %zux%zux%zu latents\n",
                  vae_weights.c_str(), C, (size_t)H, (size_t)W);
      {
        using VaeExec = Exec;
        KokkosDNN::MossImage::Vae::Config vcfg =
            tiny ? KokkosDNN::MossImage::Vae::Config::tiny_test()
                 : KokkosDNN::MossImage::Vae::Config();
        vcfg.validate();
        KokkosDNN::MossImage::Vae::Weights<VaeExec> vw;
        vw.allocate(vcfg);
        if (tiny) {
          KokkosDNN::MossImage::Vae::random_init(vcfg, vw, seed);
        } else {
          KokkosDNN::MossImage::Vae::load_safetensors(vcfg, vae_weights, vw);
        }
        KokkosDNN::MossImage::Vae::DecoderModel<VaeExec> dec(vcfg, vw);
        Kokkos::View<float****, Kokkos::LayoutRight, VaeExec> z("vae_z", 1, C,
                                                                (size_t)H,
                                                                (size_t)W);
        {
          auto zh = Kokkos::create_mirror_view(z);
          size_t k = 0;
          for (size_t c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
              for (int ww = 0; ww < W; ++ww)
                zh(0, c, h, ww) = lat[k++];
          Kokkos::deep_copy(z, zh);
        }
        oH = (size_t)vcfg.scale_factor * (size_t)H;
        oW = (size_t)vcfg.scale_factor * (size_t)W;
        Kokkos::View<float****, Kokkos::LayoutRight, VaeExec> rgb("vae_rgb", 1,
                                                                  3, oH, oW);
        dec.decode(z, rgb);
        Kokkos::fence();
        auto rh = Kokkos::create_mirror_view(rgb);
        Kokkos::deep_copy(rh, rgb);
        // Reference postprocess: VAE output clamped to [-1,1], then
        // (x*0.5+0.5) to [0,1] (VaeImageProcessor denormalize). NOT a
        // per-image min-max stretch (that hides exposure errors and shifts
        // colors with outliers).
        pix.resize(oH * oW * 3);
        for (size_t h = 0; h < oH; ++h)
          for (size_t ww = 0; ww < oW; ++ww)
            for (size_t c = 0; c < 3; ++c) {
              float v = rh(0, c, h, ww);
              v = std::min(1.0f, std::max(-1.0f, v)) * 0.5f + 0.5f;
              v = std::min(1.0f, std::max(0.0f, v));
              pix[(h * oW + ww) * 3 + c] =
                  (unsigned char)(std::lround(255.0f * v));
            }
        vw.release();
        Kokkos::fence();
      }
      oCh = 3;
      decoded = true;
      MO::print_phase_memory("decode", (size_t)MO::kVaeBudgetBytes, 0);
#else
      std::printf("[vae] VAE decode requested (default) but "
                  "KokkosDNN_MossImageVae.hpp is not in this build; writing "
                  "a grayscale preview instead (need the VAE worktree "
                  "header + --vae-weights)\n");
#endif
    }
    if (!decoded) {
      pix = render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
      MO::print_phase_memory("decode-preview", 0, 0);
    }
#ifdef KOKKOSDNN_ENABLE_IMAGE
    {
      Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::HostSpace>
          img("moss_out", oH, oW, (size_t)oCh);
      for (size_t h = 0; h < oH; ++h)
        for (size_t ww = 0; ww < oW; ++ww)
          for (int c = 0; c < oCh; ++c)
            img(h, ww, (size_t)c) = pix[(h * oW + ww) * (size_t)oCh + (size_t)c];
      if (out_is_jpeg(out)) {
        KokkosDNN::Image::save_image_jpg(out, img);
        std::printf("[wrote %s] %zux%zu %s (JPEG)\n", out.c_str(), oH, oW,
                    decoded ? "VAE RGB" : "grayscale preview");
      } else {
        KokkosDNN::Image::save_image_png(out, img);
        std::printf("[wrote %s] %zux%zu %s (PNG)\n", out.c_str(), oH, oW,
                    decoded ? "VAE RGB" : "grayscale preview");
      }
    }
#else
    {
      const std::string ppm = with_extension(out, ".ppm");
      if (ppm != out)
        std::printf("[note] built without KokkosDNN_ENABLE_IMAGE: writing PPM "
                    "to %s instead of %s\n",
                    ppm.c_str(), out.c_str());
      if (decoded) {
        save_ppm_rgb(ppm, pix, oH, oW);
      } else {
        save_ppm(ppm, pix, oH, oW);
      }
      // oCh is only consumed by the image path above.
      std::printf("[wrote %s] %zux%zu %s (PPM)\n", ppm.c_str(), oH, oW,
                  decoded ? "VAE RGB" : "grayscale preview");
    }
#endif
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
