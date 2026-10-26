// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 text-to-image driver: true-pipeline phase order
// (encode -> release+fence -> DiT sample -> release+fence -> VAE decode)
// with FlowEuler sampling and CFG.
//
// TEXT-TO-IMAGE ONLY: this driver conditions on text alone. Qwen-Image-Edit
// (reference-image conditioning) is out of scope and is rejected nowhere
// here simply because no such flag exists; do not add one in this TU.
//
// Backend selection is a compile-time compatibility shim: when the
// true-pipeline headers are present in models/qwen_image_21/
// (KokkosDNN_QwenImage21Dit.hpp, KokkosDNN_QwenImage21Encoder.hpp,
// KokkosDNN_QwenImage21Vae.hpp, plus geometry in
// KokkosDNN_QwenImage21Config.hpp) they are picked up via __has_include and
// the corresponding phase runs the real model (encode: raw-text BPE +
// EncoderModel forward; decode: runtime-affine denormalize +
// VaeModel::decode to RGBA); otherwise the phase falls
// back to the placeholder Qwen21:: geometry / file / null path with a
// logged reason, so the tree always builds. The BPE tables come from
// models/moss_image/KokkosDNN_QwenTokenizer.hpp (probed, same shim).
// QWEN21_DRIVER_HAVE_{OPS,DIT,ENCODER,VAE,TOKENIZER} gate every use (see
// the shim below). Status: Config geometry + the DiT are usable when
// present (the driver samples with the real DiT, including --tiny on
// random weights); encode runs the in-process raw-text BPE +
// EncoderModel forward when --tokenizer + --text-encoder are given (else
// --prompt-embeds-in file or null), and decode runs the real VAE decode
// to RGBA by default when --vae-weights are given (else a grayscale
// preview; pass --no-vae-decode for the preview), each fallback with a
// logged reason.
//
// Interactive loop (default): without --prompt on a TTY (and without the
// file/parity handoff flags) the driver asks for a prompt, asks for a
// filename, generates, and repeats until EOF or quit/exit. Giving --prompt
// (or any of --prompt-embeds-in/--text-weights, --init-latents-in,
// --dump-steps, --decode-latents-in, or non-TTY stdin) runs ONE generation
// and exits. The per-iteration seed is --seed + iteration; an empty
// filename auto-numbers qwen_image_21_out_<n>.png.
//
//   qwen_image_21_generate [--tiny] [--weights transformer_dir]
//                          [--prompt "text"]   (one-off; omit to loop)
//                          [--tokenizer tok_dir] [--text-encoder enc_dir]
//                          [--text-weights feats.f32]
//                          [--prompt-embeds-in feats.f32]
//                          [--height H] [--width W]
//                          [--steps 8] [--cfg 1.0] [--shift 3.0]
//                          [--sigmas driver|turbo]
//                          [--seed 42] [--init-latents-in lat.f32]
//                          [--vae-weights vae.safetensors [--no-vae-decode]]
//                          [--decode-latents-in lat.f32
//                           [--decode-latents-layout chw|token]
//                           [--decode-out-f32 file]]
//                          [--out img.png]
//                          [--no-staged] [--dump-steps DIR]
//// Text conditioning precedence: in-process Qwen forward (--tokenizer dir
// with tokenizer.json + --text-encoder weights, raw T2I template, no chat
// template) -> --prompt-embeds-in file (alias
// --text-weights) -> null (zeros) with the prompt logged. --tokenizer /
// --text-encoder must be given together. --tiny skips the Qwen forward.
// --vae-weights is validated whenever given; VAE decode runs by default
// (latents denormalized by the runtime vae/config.json affine, then
// QwenImage21VaeModel::decode to RGBA; pass --no-vae-decode for the
// channel-mean grayscale preview). Raw fp32
// latents (<stem>.f32, [C,H,W] row-major) are always written.
//
// Staged execution (24GB fit, default): encode -> releases the encoder
// weights (the MiB-scale embeds stay resident on the device) ->
// Kokkos::fence() -> loads the DiT -> samples straight from the
// device-resident embeds -> releases the DiT -> Kokkos::fence() -> decodes,
// so the 17.5GB encoder, the 14.2GB DiT, and the VAE are never resident at
// once. Pass --no-staged for the legacy single-phase run. No disk round-trip
// and no second process: the embeds are separate
// (MiB) allocations from the (GiB) encoder weights. In the interactive loop
// the encoder and transformer additionally live in HOST memory (loaded once
// before the first prompt) and are copied host->device per iteration, so
// staging (default) keeps the same one-component-resident property every
// iteration
// without re-reading the ~32 GB checkpoint from disk. Each phase logs its
// resident estimate (weights bytes + activation estimate).
//
// --tiny runs a random-weights smoke model on 4x4 latents, 2 steps
// (no downloads).
//
// --dump-steps DIR writes golden-parity tensors as raw little-endian float32
// (driver layout: latents [C,H,W] row-major C-major lat[(c*H+h)*W+w],
// prompt embeds [L,Cc] row-major) plus manifest.json: prompt_embeds.f32
// (and prompt_embeds_neg.f32 when --cfg != 1), initial_latents.f32, then
// stepNN_noise_pred.f32 / stepNN_latents.f32 per step (with
// stepNN_noise_pred_cond.f32 / _uncond.f32 when the CFG branch runs), each
// logged with its f64 sum. Off by default: the sample loop only takes the
// extra device->host copies when the flag is set.
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

#include "KokkosDNN.hpp"

// ---- True-pipeline compatibility shim ------------------------------------
//
// The real Qwen-Image-2.1 pipeline interfaces live in models/qwen_image_21/
// as:
//
//   models/qwen_image_21/KokkosDNN_QwenImage21Config.hpp  (QwenImage21::Config)
//   models/qwen_image_21/KokkosDNN_QwenImage21Dit.hpp     (QwenImage21Dit::Model/Weights)
//   models/qwen_image_21/KokkosDNN_QwenImage21Encoder.hpp (QwenImage21Encoder::Model/Weights)
//   models/qwen_image_21/KokkosDNN_QwenImage21Vae.hpp     (QwenImage21Vae::Model/Weights)
//
// Status: Config geometry + the DiT are usable when present (the driver
// samples with the real DiT, including --tiny on random weights); the
// landed encoder forward() and VAE decode() are wired below (raw-text BPE
// + EncoderModel forward for encode, runtime-affine denormalize +
// VaeModel::decode to RGBA for decode), each with a file/null + preview
// fallback (with a logged reason) when their flags/headers are absent.
// Every true-pipeline use is gated on its QWEN21_DRIVER_HAVE_* macro, so
// this TU builds with or without any of those headers. There is
// intentionally NO hard dependency here (and none in CMakeLists.txt): the
// driver probes, never requires.
//
// The in-process Qwen encode additionally needs the byte-level BPE tables
// (models/moss_image/KokkosDNN_QwenTokenizer.hpp, Unicode tables +
// runtime tokenizer.json load: structurally compatible with the Qwen-Image
// processor tokenizer.json, no table regen). It is probed on the include
// path first, then relative to this file for in-tree builds, and gated on
// QWEN21_DRIVER_HAVE_TOKENIZER.
#if defined(__has_include)
#if __has_include("KokkosDNN_QwenImage21Config.hpp")
#include "KokkosDNN_QwenImage21Config.hpp"
#define QWEN21_DRIVER_HAVE_OPS 1
#endif
#if __has_include("KokkosDNN_QwenImage21Dit.hpp")
#include "KokkosDNN_QwenImage21Dit.hpp"
#define QWEN21_DRIVER_HAVE_DIT 1
#endif
#if __has_include("KokkosDNN_QwenImage21Encoder.hpp")
#include "KokkosDNN_QwenImage21Encoder.hpp"
#define QWEN21_DRIVER_HAVE_ENCODER 1
#endif
#if __has_include("KokkosDNN_QwenImage21Vae.hpp")
#include "KokkosDNN_QwenImage21Vae.hpp"
#define QWEN21_DRIVER_HAVE_VAE 1
#endif
#if __has_include("KokkosDNN_QwenTokenizer.hpp")
#include "KokkosDNN_QwenTokenizer.hpp"
#define QWEN21_DRIVER_HAVE_TOKENIZER 1
#elif __has_include("../../models/moss_image/KokkosDNN_QwenTokenizer.hpp")
#include "../../models/moss_image/KokkosDNN_QwenTokenizer.hpp"
#define QWEN21_DRIVER_HAVE_TOKENIZER 1
#endif
#endif

#ifdef KOKKOSDNN_ENABLE_IMAGE
#if __has_include("KokkosDNN_Image.hpp")
#include "KokkosDNN_Image.hpp"
#elif __has_include("images/KokkosDNN_Image.hpp")
#include "images/KokkosDNN_Image.hpp"
#else
#error "KOKKOSDNN_ENABLE_IMAGE is set but no KokkosDNN_Image header was found"
#endif
#endif

// ---- Placeholder backend (always available, file scope) -------------------
// Stand-in geometry/weights/model used when the true-pipeline headers are
// absent. Labeled "placeholder" in every log; never silently mistaken for
// the Qwen-Image-2.1 checkpoint. File scope (NOT inside the anonymous
// namespace below): the full-header build never instantiates these, and
// internal linkage would trip nvcc #177-D unused warnings under -Werror.
namespace Qwen21 {

struct Config {
  int depth = 30;             // transformer blocks
  int hidden_size = 2048;     // token dim D
  int num_heads = 16;         // joint attention heads (head_dim = D / heads)
  int patch_size = 2;         // latent patch (2x2 conv == per-patch linear)
  double mlp_ratio = 4.0;     // FFN hidden = D * ratio
  int caption_channels = 3584;  // text-encoder embedding dim (Qwen2.5-VL)
  int model_max_length = 512;   // caption token capacity
  int in_channels = 16;         // latent channels
  int out_channels = 16;        // noise prediction channels (== in_channels)
  int t_freq_dim = 256;         // sinusoidal timestep embedding dim

  static Config qwen_image_21() { return Config{}; }

  static Config tiny_test() {
    Config c;
    c.depth = 2;
    c.hidden_size = 32;
    c.num_heads = 4;  // head dim 8
    c.patch_size = 1;
    c.mlp_ratio = 2.0;  // hidden 64
    c.caption_channels = 16;
    c.model_max_length = 8;
    c.in_channels = 4;
    c.out_channels = 4;
    c.t_freq_dim = 16;
    return c;
  }

  int mlp_hidden() const {
    return (int)std::llround((double)hidden_size * mlp_ratio);
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Qwen21::Config::validate: " + m);
    };
    if (depth <= 0) bad("depth must be > 0");
    if (hidden_size <= 0) bad("hidden_size must be > 0");
    if (patch_size <= 0) bad("patch_size must be > 0");
    if (num_heads <= 0) bad("num_heads must be > 0");
    if (!(mlp_ratio > 0) || !std::isfinite(mlp_ratio))
      bad("mlp_ratio must be positive and finite");
    if (caption_channels <= 0) bad("caption_channels must be > 0");
    if (model_max_length <= 0) bad("model_max_length must be > 0");
    if (in_channels <= 0) bad("in_channels must be > 0");
    if (out_channels <= 0) bad("out_channels must be > 0");
    if (t_freq_dim <= 0 || t_freq_dim % 2 != 0)
      bad("t_freq_dim must be a positive even number");
    if (hidden_size % num_heads != 0)
      bad("hidden_size must be divisible by num_heads");
    if (out_channels != in_channels)
      bad("out_channels must equal in_channels (noise prediction)");
  }
};

template <class ExecSpace, class Scalar = float>
struct BlockWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_qkv_w;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_cap_kv_w;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_proj_w;
  Kokkos::View<Scalar*, ExecSpace> attn_proj_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mlp_fc1_w;
  Kokkos::View<Scalar*, ExecSpace> mlp_fc1_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mlp_fc2_w;
  Kokkos::View<Scalar*, ExecSpace> mlp_fc2_b;

  // Drop every device allocation (staged execution: release the DiT weights
  // before the VAE decode is loaded so a 24GB GPU never holds both at
  // once). Assigning {} releases the View's reference; the caller must
  // Kokkos::fence() before the next phase is loaded.
  void release() {
    attn_qkv_w = {};
    attn_cap_kv_w = {};
    attn_proj_w = {};
    attn_proj_b = {};
    mlp_fc1_w = {};
    mlp_fc1_b = {};
    mlp_fc2_w = {};
    mlp_fc2_b = {};
  }
};

template <class ExecSpace, class Scalar = float>
struct Weights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x_proj_w;
  Kokkos::View<Scalar*, ExecSpace> x_proj_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> t_mlp0_w, t_mlp2_w;
  Kokkos::View<Scalar*, ExecSpace> t_mlp0_b, t_mlp2_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> cap_proj_w;
  Kokkos::View<Scalar*, ExecSpace> cap_proj_b;
  Kokkos::View<Scalar*, ExecSpace> cap_norm_w;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> final_w;
  Kokkos::View<Scalar*, ExecSpace> final_b;

  void allocate(const Config& cfg) {
    cfg.validate();
    const size_t D = (size_t)cfg.hidden_size;
    const size_t Hh = (size_t)cfg.mlp_hidden();
    const size_t P = (size_t)cfg.patch_size;
    x_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "qwen21_x_proj_w", D, (size_t)cfg.in_channels * P * P);
    x_proj_b = Kokkos::View<Scalar*, ExecSpace>("qwen21_x_proj_b", D);
    t_mlp0_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "qwen21_t_mlp0_w", D, (size_t)cfg.t_freq_dim);
    t_mlp0_b = Kokkos::View<Scalar*, ExecSpace>("qwen21_t_mlp0_b", D);
    t_mlp2_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "qwen21_t_mlp2_w", D, D);
    t_mlp2_b = Kokkos::View<Scalar*, ExecSpace>("qwen21_t_mlp2_b", D);
    cap_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "qwen21_cap_proj_w", D, (size_t)cfg.caption_channels);
    cap_proj_b = Kokkos::View<Scalar*, ExecSpace>("qwen21_cap_proj_b", D);
    cap_norm_w = Kokkos::View<Scalar*, ExecSpace>("qwen21_cap_norm_w", D);
    blocks.resize((size_t)cfg.depth);
    for (size_t i = 0; i < blocks.size(); ++i) {
      auto& b = blocks[i];
      b.attn_qkv_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "qwen21_qkv", 3 * D, D);
      b.attn_cap_kv_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "qwen21_cap_kv", 2 * D, D);
      b.attn_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "qwen21_attn_proj", D, D);
      b.attn_proj_b =
          Kokkos::View<Scalar*, ExecSpace>("qwen21_attn_proj_b", D);
      b.mlp_fc1_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "qwen21_fc1", Hh, D);
      b.mlp_fc1_b = Kokkos::View<Scalar*, ExecSpace>("qwen21_fc1_b", Hh);
      b.mlp_fc2_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "qwen21_fc2", D, Hh);
      b.mlp_fc2_b = Kokkos::View<Scalar*, ExecSpace>("qwen21_fc2_b", D);
    }
    final_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "qwen21_final_w", (size_t)cfg.out_channels * P * P, D);
    final_b = Kokkos::View<Scalar*, ExecSpace>(
        "qwen21_final_b", (size_t)cfg.out_channels * P * P);
  }

  void release() {
    x_proj_w = {};
    x_proj_b = {};
    t_mlp0_w = {};
    t_mlp2_w = {};
    t_mlp0_b = {};
    t_mlp2_b = {};
    cap_proj_w = {};
    cap_proj_b = {};
    cap_norm_w = {};
    for (auto& b : blocks) b.release();
    blocks.clear();
    blocks.shrink_to_fit();
    final_w = {};
    final_b = {};
  }

  size_t resident_bytes() const {
    size_t n = 0;
    auto acc2 = [&](const auto& v) {
      n += v.extent(0) * v.extent(1);
    };
    auto acc1 = [&](const auto& v) { n += v.extent(0); };
    acc2(x_proj_w);
    acc1(x_proj_b);
    acc2(t_mlp0_w);
    acc2(t_mlp2_w);
    acc1(t_mlp0_b);
    acc1(t_mlp2_b);
    acc2(cap_proj_w);
    acc1(cap_proj_b);
    acc1(cap_norm_w);
    for (const auto& b : blocks) {
      n += b.attn_qkv_w.extent(0) * b.attn_qkv_w.extent(1);
      n += b.attn_cap_kv_w.extent(0) * b.attn_cap_kv_w.extent(1);
      n += b.attn_proj_w.extent(0) * b.attn_proj_w.extent(1);
      n += b.attn_proj_b.extent(0);
      n += b.mlp_fc1_w.extent(0) * b.mlp_fc1_w.extent(1);
      n += b.mlp_fc1_b.extent(0);
      n += b.mlp_fc2_w.extent(0) * b.mlp_fc2_w.extent(1);
      n += b.mlp_fc2_b.extent(0);
    }
    acc2(final_w);
    acc1(final_b);
    return n * sizeof(Scalar);
  }
};

template <class ExecSpace, class Scalar = float>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed) {
  w.allocate(cfg);
  std::mt19937_64 eng(seed ^ 0x9e3779b97f4a7c15ULL);
  std::uniform_real_distribution<double> dist(-0.02, 0.02);
  auto fill2 = [&](auto& v, double scale) {
    auto hm = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < hm.extent(0); ++i)
      for (size_t j = 0; j < hm.extent(1); ++j)
        hm(i, j) = (Scalar)(dist(eng) * scale);
    Kokkos::deep_copy(v, hm);
  };
  auto fill1 = [&](auto& v, double val, bool jitter) {
    auto hm = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < hm.extent(0); ++i)
      hm(i) = (Scalar)(jitter ? dist(eng) : val);
    Kokkos::deep_copy(v, hm);
  };
  fill2(w.x_proj_w, 1.0);
  fill1(w.x_proj_b, 0.0, false);
  fill2(w.t_mlp0_w, 1.0);
  fill1(w.t_mlp0_b, 0.0, false);
  fill2(w.t_mlp2_w, 1.0);
  fill1(w.t_mlp2_b, 0.0, false);
  fill2(w.cap_proj_w, 1.0);
  fill1(w.cap_proj_b, 0.0, false);
  fill1(w.cap_norm_w, 1.0, false);
  for (auto& b : w.blocks) {
    fill2(b.attn_qkv_w, 1.0);
    fill2(b.attn_cap_kv_w, 1.0);
    fill2(b.attn_proj_w, 1.0);
    fill1(b.attn_proj_b, 0.0, false);
    fill2(b.mlp_fc1_w, 1.0);
    fill1(b.mlp_fc1_b, 0.0, false);
    fill2(b.mlp_fc2_w, 1.0);
    fill1(b.mlp_fc2_b, 0.0, false);
  }
  fill2(w.final_w, 1.0);
  fill1(w.final_b, 0.0, false);
  Kokkos::fence();
}

// Placeholder velocity head: out = a*x + b*mean(y). Cheap, finite, and
// caption-dependent, so the --tiny smoke run exercises the full
// CFG/nearest-grid loop with real conditioning signal.
template <class ExecSpace, class Scalar = float>
class Model {
 public:
  using LatR = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using TVec = Kokkos::View<float*, ExecSpace>;
  using TokR = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;

  Model(const Config& cfg, Weights<ExecSpace, Scalar>& weights)
      : cfg_(cfg), w_(weights) {
    cfg_.validate();
  }

  void forward(LatR x, TVec t, TokR y, LatR out) {
    const size_t B = x.extent(0), C = x.extent(1), H = x.extent(2),
                 W = x.extent(3);
    const size_t L = y.extent(1), Cc = y.extent(2);
    double mean = 0.0;
    Kokkos::parallel_reduce(
        "qwen21_capmean",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                         {B, L, Cc}),
        KOKKOS_LAMBDA(size_t b, size_t l, size_t c, double& acc) {
          acc += static_cast<double>(y(b, l, c));
        },
        mean);
    const double n = (double)(B * L * Cc);
    const float cap = (n > 0.0) ? (float)(mean / n) : 0.0f;
    float t0 = 0.0f;
    {
      auto th = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t);
      t0 = th(0);
    }
    const float tscale = t0 / 1000.0f;
    Kokkos::parallel_for(
        "qwen21_placeholder_fwd",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                         {B, C, H, W}),
        KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
          const float xv = static_cast<float>(x(b, c, h, w));
          out(b, c, h, w) =
              static_cast<Scalar>(0.02f * xv + 0.01f * cap + 0.001f * tscale);
        });
  }

 private:
  Config cfg_;
  Weights<ExecSpace, Scalar>& w_;
};

// Gaussian init latents (Box-Muller, splitmix-style seed decorrelation).
inline std::vector<float> sample_gaussian(size_t n, uint64_t seed) {
  std::mt19937 eng((uint32_t)seed ^ (uint32_t)(seed >> 32) ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::vector<float> out(n);
  for (size_t i = 0; i < n; i += 2) {
    double u1 = 0.0;
    while (u1 <= 0.0) u1 = uni(eng);
    const double u2 = uni(eng);
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = 6.28318530717958647692 * u2;
    out[i] = (float)(r * std::cos(a));
    if (i + 1 < n) out[i + 1] = (float)(r * std::sin(a));
  }
  return out;
}

// Load raw float32 caption features [L, Cc] (row-major). Accepts [L,Cc],
// [1,L,Cc], [1,1,L,Cc] flattened (takes the trailing block). With
// expect_L <= 0 the row count is inferred from the file size, so a golden
// [19,4096] prompt-embed dump loads as-is instead of being forced to the
// backend maximum sequence length.
inline std::vector<float> load_caption_raw(const std::string& path, int expect_L,
                                           int expect_Cc) {
  if (expect_Cc <= 0)
    throw std::runtime_error("qwen21: caption dim must be > 0");
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("qwen21: cannot open '" + path + "'");
  const long long n = (long long)f.tellg();
  if (n <= 0 || n % 4 != 0)
    throw std::runtime_error("qwen21: bad float32 file '" + path + "'");
  f.seekg(0, std::ios::beg);
  std::vector<float> out((size_t)n / 4);
  f.read(reinterpret_cast<char*>(out.data()), n);
  if (!f) throw std::runtime_error("qwen21: short read of '" + path + "'");
  if (expect_L <= 0) {
    if (out.size() % (size_t)expect_Cc != 0)
      throw std::runtime_error(
          "qwen21: caption file '" + path + "' holds " +
          std::to_string(out.size()) + " floats, not a multiple of Cc=" +
          std::to_string(expect_Cc) + " ([L,Cc])");
    return out;
  }
  const size_t want = (size_t)expect_L * (size_t)expect_Cc;
  if (out.size() == want) return out;
  if (out.size() > want && out.size() % want == 0)
    return std::vector<float>(out.end() - (long long)want, out.end());
  throw std::runtime_error("qwen21: caption file '" + path + "' holds " +
                           std::to_string(out.size()) + " floats, expected " +
                           std::to_string(want) + " ([L,Cc])");
}

// ---- staged-execution memory budget --------------------------------------
// Real-checkpoint resident budgets (bf16, documented in the README 24GB fit
// table). The encode phase holds the text encoder, the sample phase holds
// the DiT; the VAE decode (~0.7GB) joins the sample phase. Staging keeps
// the peak at max(17.5, 14.2) + 0.7 + activations instead of ~32+ GB.
inline constexpr double kEncoderBudgetBytes = 17.5e9;  // Qwen text encoder
inline constexpr double kDitBudgetBytes = 14.2e9;      // Qwen-Image DiT (Turbo)
inline constexpr double kVaeBudgetBytes = 0.676e9;     // Qwen-Image VAE

// One-line per-phase resident estimate used by the driver between staged
// phases (call after Kokkos::fence()).
inline void print_phase_memory(const char* phase, size_t weights_bytes,
                               size_t activation_bytes) {
  const double w = (double)weights_bytes / 1e9;
  const double a = (double)activation_bytes / 1e9;
  const double t = w + a;
  std::printf("[memory phase=%s] weights=%.3f GB + activations~%.3f GB = %.3f "
              "GB resident (24GB budget: %s)\n",
              phase, w, a, t, t < 24.0 ? "FITS" : "EXCEEDS");
  std::fflush(stdout);
}

}  // namespace Qwen21

namespace {

using Exec = Kokkos::DefaultExecutionSpace;

namespace QW = Qwen21;
using Scalar = float;
#ifdef QWEN21_DRIVER_HAVE_DIT
// DiT checkpoints are bf16 on disk; running them in bhalf_t (fp32
// accumulation in every kernel) halves residency vs fp32 and is what the
// 24GB fit table assumes. Host-side embeds and latents stay float.
using DScalar = Kokkos::Experimental::bhalf_t;
#else
using DScalar = Scalar;  // placeholder model is float-only
#endif

// ---- Backend geometry traits ---------------------------------------------
#ifndef QWEN21_DRIVER_HAVE_DIT
inline size_t be_latent_channels(const QW::Config& c) {
  return (size_t)c.in_channels;
}
inline size_t be_caption_dim(const QW::Config& c) {
  return (size_t)c.caption_channels;
}
inline size_t be_max_seq(const QW::Config& c) {
  return (size_t)c.model_max_length;
}
inline int be_patch(const QW::Config& c) { return c.patch_size; }
inline const char* be_label(const QW::Config&) {
  return "qwen-image-2.1 (placeholder 30x2048/16ch/cap3584 geometry)";
}
#endif  // !QWEN21_DRIVER_HAVE_DIT
#ifdef QWEN21_DRIVER_HAVE_OPS
inline size_t be_latent_channels(const KokkosDNN::QwenImage21::Config& c) {
  return (size_t)c.latent_channels;
}
inline size_t be_caption_dim(const KokkosDNN::QwenImage21::Config& c) {
  return (size_t)c.context_dim;
}
inline size_t be_max_seq(const KokkosDNN::QwenImage21::Config&) { return 512; }
inline int be_patch(const KokkosDNN::QwenImage21::Config&) { return 1; }
inline const char* be_label(const KokkosDNN::QwenImage21::Config&) {
  return "qwen-image-2.1 (true 32x4096/64ch/ctx4096 geometry)";
}
#endif

void save_raw_f32(const std::string& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f.write(reinterpret_cast<const char*>(v.data()),
          (long long)(v.size() * sizeof(float)));
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

// Raw fp32 LE initial latents [C,H,W] row-major, flattened C-major as
// lat[(c*H+h)*W+w] (the driver's native layout: matches the sampler's
// x_init pack/unpack order and the <stem>.f32 files it writes).
inline std::vector<float> load_init_latents_raw(const std::string& path,
                                                size_t C, size_t H,
                                                size_t W) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("qwen21: cannot open '" + path + "'");
  const long long n = (long long)f.tellg();
  if (n <= 0 || n % 4 != 0)
    throw std::runtime_error("qwen21: bad float32 file '" + path + "'");
  const size_t want = C * H * W;
  if ((size_t)n / 4 != want)
    throw std::runtime_error(
        "qwen21: init latents file '" + path + "' holds " +
        std::to_string(n / 4) + " floats, expected " + std::to_string(want) +
        " ([C,H,W]=" + std::to_string(C) + "x" + std::to_string(H) + "x" +
        std::to_string(W) + ")");
  f.seekg(0, std::ios::beg);
  std::vector<float> out(want);
  f.read(reinterpret_cast<char*>(out.data()), n);
  if (!f) throw std::runtime_error("qwen21: short read of '" + path + "'");
  return out;
}

// ---- Parity dumps (--dump-steps) -----------------------------------------
// Golden-parity tensor dumps: every tensor is written as raw float32 in the
// driver's own native (little-endian) layout, logged with its f64 sum, and
// recorded in a hand-rolled manifest.json (no JSON dependency). Nothing here
// touches the device: the caller hands over host copies taken only when the
// flag is set.
struct DumpTensor {
  std::string name;  // tensor name (also the .f32 file stem)
  size_t numel = 0;
  double f64sum = 0.0;
};

struct DumpWriter {
  std::string dir;
  std::vector<DumpTensor> tensors;

  explicit DumpWriter(const std::string& d) : dir(d) {
    if (dir.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (!std::filesystem::is_directory(dir))
      throw std::runtime_error("--dump-steps: cannot create directory '" + dir +
                               "'");
    std::printf("[dump] writing parity tensors to %s\n", dir.c_str());
    std::fflush(stdout);
  }

  bool active() const { return !dir.empty(); }

  // One tensor: DIR/<name>.f32 plus "[dump] <name> <numel> floats
  // f64sum=<%.17g>". The sum accumulates in double over the float32 values.
  void write(const std::string& name, const std::vector<float>& v) {
    if (!active()) return;
    save_raw_f32(dir + "/" + name + ".f32", v);
    double s = 0.0;
    for (float x : v) s += (double)x;
    tensors.push_back(DumpTensor{name, v.size(), s});
    std::printf("[dump] %s %zu floats f64sum=%.17g\n", name.c_str(), v.size(),
                s);
    std::fflush(stdout);
  }
};

// Manifest metadata: exactly what produced the dumped tensors.
struct DumpManifest {
  std::string prompt, prompt_source, model_label, sigmas_mode;
  std::vector<float> sigmas;  // the schedule actually used, incl. terminal 0
  int height = 0, width = 0, steps = 0;
  size_t latent_channels = 0;
  size_t prompt_len = 0, prompt_dim = 0;  // staged conditioning [L,Cc]
  unsigned long long seed = 0;
  double guidance = 1.0;
  bool use_cfg = false, staged = false;
};

inline std::string json_escape(const std::string& s) {
  std::string o;
  for (char ch : s) {
    const unsigned char c = (unsigned char)ch;
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
          o += buf;
        } else {
          o += ch;
        }
    }
  }
  return o;
}

inline std::string json_num(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

// Hand-rolled JSON: metadata + per-tensor f64 sums, so a golden harness can
// check that the files it read are the ones logged.
inline void write_dump_manifest(const DumpWriter& dw, const DumpManifest& m) {
  if (!dw.active()) return;
  const std::string path = dw.dir + "/manifest.json";
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f << "{\n";
  f << "  \"format\": \"kokkosdnn-qwen21-dump-v1\",\n";
  f << "  \"model\": \"" << json_escape(m.model_label) << "\",\n";
  f << "  \"prompt\": \"" << json_escape(m.prompt) << "\",\n";
  f << "  \"prompt_source\": \"" << json_escape(m.prompt_source) << "\",\n";
  f << "  \"height\": " << m.height << ",\n";
  f << "  \"width\": " << m.width << ",\n";
  f << "  \"latent_channels\": " << m.latent_channels << ",\n";
  f << "  \"prompt_embeds_shape\": [" << m.prompt_len << ", "
    << m.prompt_dim << "],\n";
  f << "  \"steps\": " << m.steps << ",\n";
  f << "  \"sigmas\": [";
  for (size_t i = 0; i < m.sigmas.size(); ++i)
    f << (i ? ", " : "") << json_num((double)m.sigmas[i]);
  f << "],\n";
  f << "  \"seed\": " << m.seed << ",\n";
  f << "  \"cfg\": " << json_num(m.guidance) << ",\n";
  f << "  \"guidance\": " << json_num(m.guidance) << ",\n";
  f << "  \"use_cfg\": " << (m.use_cfg ? "true" : "false") << ",\n";
  f << "  \"staged\": " << (m.staged ? "true" : "false") << ",\n";
  f << "  \"sigmas_mode\": \"" << json_escape(m.sigmas_mode) << "\",\n";
  f << "  \"layout\": \"float32 little-endian: latents [C,H,W] row-major "
       "C-major lat[(c*H+h)*W+w]; prompt_embeds [L,Cc] row-major\",\n";
#ifdef QWEN21_DRIVER_HAVE_DIT
  f << "  \"latent_storage\": \"bfloat16 DiT state stored per step, dumped as "
       "float32 (initial_latents.f32 is the fp32 x_init before that "
       "quantization)\",\n";
#else
  f << "  \"latent_storage\": \"float32 (placeholder DScalar)\",\n";
#endif
  f << "  \"tensors\": [\n";
  for (size_t i = 0; i < dw.tensors.size(); ++i) {
    const DumpTensor& t = dw.tensors[i];
    f << "    {\"name\": \"" << json_escape(t.name) << "\", \"file\": \""
      << json_escape(t.name) << ".f32\", \"numel\": " << t.numel
      << ", \"f64sum\": " << json_num(t.f64sum) << "}"
      << (i + 1 < dw.tensors.size() ? "," : "") << "\n";
  }
  f << "  ]\n";
  f << "}\n";
  f.close();
  if (!f) throw std::runtime_error("short write to '" + path + "'");
  std::printf("[dump] wrote %zu tensors to %s\n", dw.tensors.size(),
              dw.dir.c_str());
  std::fflush(stdout);
}

// Raw fp32 LE latents for --decode-latents-in. `chw` is the driver layout
// lat[(c*H+h)*W+w]; `token` is torch's packed [H*W, C] (row t = h*W+w,
// channel inner), i.e. the golden final-latents .f32. Both are validated
// against C*H*W.
inline std::vector<float> load_decode_latents(const std::string& path,
                                              size_t C, size_t H, size_t W,
                                              bool token_major) {
  std::vector<float> raw = load_init_latents_raw(path, C, H, W);
  if (!token_major) return raw;
  std::vector<float> out(raw.size());
  for (size_t h = 0; h < H; ++h)
    for (size_t w = 0; w < W; ++w)
      for (size_t c = 0; c < C; ++c)
        out[(c * H + h) * W + w] = raw[(h * W + w) * C + c];
  return out;
}

// Grayscale preview: channel-mean -> min/max normalize -> u8 -> nearest 16x.
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
  constexpr size_t kUp = 16;  // nominal VAE downsample stand-in (preview only)
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
void save_ppm_rgb(const std::string& path, const std::vector<unsigned char>& rgb,
                  size_t H, size_t W) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f << "P6\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char*>(rgb.data()), (long long)rgb.size());
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

void save_ppm(const std::string& path, const std::vector<unsigned char>& gray,
              size_t H, size_t W) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f << "P5\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char*>(gray.data()), (long long)gray.size());
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}
#endif

std::string with_extension(const std::string& path, const std::string& ext) {
  const size_t dot = path.find_last_of('.');
  const size_t sep = path.find_last_of("/\\");
  if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
    return path + ext;
  return path.substr(0, dot) + ext;
}

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

// Weights source: a single .safetensors file or a directory of shards
// (sorted). Shared by --text-encoder, --weights, and --vae-weights. Only
// needed when a true-pipeline header is present; guarded so header-less
// builds see no unused-function warning.
#if defined(QWEN21_DRIVER_HAVE_DIT) || defined(QWEN21_DRIVER_HAVE_ENCODER) || \
    defined(QWEN21_DRIVER_HAVE_VAE)
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
#endif

#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
// ---- Raw-text Qwen2 BPE (no chat template) --------------------------------
// The in-tree Tokenizer only exposes the MOSS single-turn chat-template
// path (apply_chat_template + encode); the Qwen-Image-2.1 T2I prompt is a
// RAW template (qwen21_format_prompt_t2i: system block + user + assistant
// prefix, never the MOSS template), so the driver loads processor/
// tokenizer.json into local tables and runs the same host-side pipeline
// (added-special split -> NFC -> Qwen2 pre-tokenizer -> BPE ranks) through
// the header's public Impl primitives. No table regeneration: the .inc
// tables are Unicode data; vocab/merges/added come from the JSON at
// runtime. Logic mirrors Tokenizer::parse_*/encode (same checks, same
// longest-match split, same right truncation).
struct RawQwenTables {
  std::unordered_map<std::string, int> vocab;
  std::unordered_map<std::string, int> ranks;
  std::vector<std::pair<std::string, int>> specials;  // longest-first
  size_t max_special_len = 0;
  long long top_id = -1;
  size_t vocab_size = 0;
};

namespace RawQwenDetail {
namespace QT = KokkosDNN::QwenTokenizer;

inline void parse_vocab(QT::Impl::JsonCursor& cur, RawQwenTables& t) {
  cur.ws();
  if (cur.peek() != '{') cur.fail("vocab must be an object");
  ++cur.p;
  t.vocab.reserve(160000);
  while (true) {
    cur.ws();
    if (cur.peek() == '}') {
      ++cur.p;
      break;
    }
    std::string tok = cur.parse_string();
    cur.ws();
    if (cur.peek() != ':') cur.fail("expected ':'");
    ++cur.p;
    const long long id = cur.parse_int();
    if (id < 0 || id > 10000000) cur.fail("vocab id out of range");
    t.vocab[tok] = (int)id;
    t.top_id = std::max(t.top_id, id);
    cur.ws();
    const char d = cur.peek();
    if (d == ',') {
      ++cur.p;
      continue;
    }
    if (d == '}') {
      ++cur.p;
      break;
    }
    cur.fail("expected ',' or '}'");
  }
}

inline void parse_merges(QT::Impl::JsonCursor& cur, RawQwenTables& t) {
  cur.ws();
  if (cur.peek() != '[') cur.fail("merges must be an array");
  ++cur.p;
  int rank = 0;
  while (true) {
    cur.ws();
    if (cur.peek() == ']') {
      ++cur.p;
      break;
    }
    if (cur.peek() != '[') cur.fail("merge entry must be an array");
    ++cur.p;
    cur.ws();
    const std::string left = cur.parse_string();
    cur.ws();
    if (cur.peek() != ',') cur.fail("expected ',' in merge entry");
    ++cur.p;
    const std::string right = cur.parse_string();
    cur.ws();
    if (cur.peek() != ']') cur.fail("expected ']' in merge entry");
    ++cur.p;
    if (left.empty() || right.empty() || left.find(' ') != std::string::npos ||
        right.find(' ') != std::string::npos)
      cur.fail("malformed merge entry");
    t.ranks[left + " " + right] = rank++;
    cur.ws();
    const char d = cur.peek();
    if (d == ',') {
      ++cur.p;
      continue;
    }
    if (d == ']') {
      ++cur.p;
      break;
    }
    cur.fail("expected ',' or ']'");
  }
}

inline void parse_model(QT::Impl::JsonCursor& cur, RawQwenTables& t) {
  cur.ws();
  if (cur.peek() != '{') cur.fail("model must be an object");
  ++cur.p;
  while (true) {
    cur.ws();
    if (cur.peek() == '}') {
      ++cur.p;
      break;
    }
    std::string key = cur.parse_string();
    cur.ws();
    if (cur.peek() != ':') cur.fail("expected ':'");
    ++cur.p;
    if (key == "vocab") {
      parse_vocab(cur, t);
    } else if (key == "merges") {
      parse_merges(cur, t);
    } else {
      cur.skip_value();
    }
    cur.ws();
    const char d = cur.peek();
    if (d == ',') {
      ++cur.p;
      continue;
    }
    if (d == '}') {
      ++cur.p;
      break;
    }
    cur.fail("expected ',' or '}'");
  }
}

inline void parse_added(QT::Impl::JsonCursor& cur, RawQwenTables& t) {
  cur.ws();
  if (cur.peek() != '[') cur.fail("added_tokens must be an array");
  ++cur.p;
  while (true) {
    cur.ws();
    if (cur.peek() == ']') {
      ++cur.p;
      break;
    }
    if (cur.peek() != '{') cur.fail("added token must be an object");
    ++cur.p;
    std::string content;
    long long id = -1;
    while (true) {
      cur.ws();
      if (cur.peek() == '}') {
        ++cur.p;
        break;
      }
      std::string key = cur.parse_string();
      cur.ws();
      if (cur.peek() != ':') cur.fail("expected ':'");
      ++cur.p;
      if (key == "content") {
        content = cur.parse_string();
      } else if (key == "id") {
        id = cur.parse_int();
      } else {
        cur.skip_value();
      }
      cur.ws();
      const char d = cur.peek();
      if (d == ',') {
        ++cur.p;
        continue;
      }
      if (d == '}') {
        ++cur.p;
        break;
      }
      cur.fail("expected ',' or '}'");
    }
    if (content.empty() || id < 0) cur.fail("bad added token entry");
    t.specials.emplace_back(content, (int)id);
    t.top_id = std::max(t.top_id, id);
    cur.ws();
    const char d = cur.peek();
    if (d == ',') {
      ++cur.p;
      continue;
    }
    if (d == ']') {
      ++cur.p;
      break;
    }
    cur.fail("expected ',' or ']'");
  }
}
}  // namespace RawQwenDetail

inline RawQwenTables load_raw_qwen_tables(const std::string& path) {
  namespace QT = KokkosDNN::QwenTokenizer;
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("raw Qwen tables: cannot open '" + path + "'");
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string js = ss.str();
  QT::Impl::JsonCursor cur{js.data(), js.data() + js.size(), js.data()};
  cur.ws();
  if (cur.peek() != '{') cur.fail("top level must be an object");
  ++cur.p;
  RawQwenTables t;
  bool have_model = false;
  while (true) {
    cur.ws();
    if (cur.peek() == '}') {
      ++cur.p;
      break;
    }
    std::string key = cur.parse_string();
    cur.ws();
    if (cur.peek() != ':') cur.fail("expected ':'");
    ++cur.p;
    if (key == "model") {
      RawQwenDetail::parse_model(cur, t);
      have_model = true;
    } else if (key == "added_tokens") {
      RawQwenDetail::parse_added(cur, t);
    } else {
      cur.skip_value();
    }
    cur.ws();
    const char d = cur.peek();
    if (d == ',') {
      ++cur.p;
      continue;
    }
    if (d == '}') {
      ++cur.p;
      break;
    }
    cur.fail("expected ',' or '}'");
  }
  if (!have_model || t.vocab.empty())
    throw std::runtime_error("raw Qwen tables: no model.vocab in '" + path +
                             "'");
  if (t.ranks.empty())
    throw std::runtime_error("raw Qwen tables: no model.merges in '" + path +
                             "'");
  if (t.specials.empty())
    throw std::runtime_error("raw Qwen tables: no added_tokens in '" + path +
                             "'");
  std::sort(t.specials.begin(), t.specials.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
  t.vocab_size = (size_t)(t.top_id + 1);
  for (const auto& kv : t.specials)
    t.max_special_len = std::max(t.max_special_len, kv.first.size());
  return t;
}

// Raw-text encode: added-special longest-match split over the RAW text,
// then (per gap) NFC -> Qwen2 pre-tokenizer -> BPE ranks, right-truncated
// to max_len. No template is applied; the caller passes the already
// formatted T2I string (qwen21_format_prompt_t2i).
inline std::vector<int> encode_qwen_raw(const RawQwenTables& t,
                                        const std::string& text,
                                        size_t max_len) {
  namespace QT = KokkosDNN::QwenTokenizer;
  if (max_len == 0)
    throw std::invalid_argument("encode_qwen_raw: max_len == 0");
  std::vector<int> ids;
  size_t i = 0;
  const size_t n = text.size();
  size_t seg_start = 0;
  auto flush_gap = [&](size_t end) {
    if (end == seg_start) return;
    const std::string gap = text.substr(seg_start, end - seg_start);
    std::vector<uint32_t> cps = QT::Impl::utf8_decode(gap);
    QT::Impl::nfc_inplace(cps);
    for (const auto& pre : QT::Impl::pretokenize(cps))
      QT::Impl::bpe_encode_pretoken(pre, t.vocab, t.ranks, ids);
  };
  while (i < n) {
    size_t hit_len = 0;
    int hit_id = -1;
    for (const auto& kv : t.specials) {
      const size_t L = kv.first.size();
      if (L > hit_len && i + L <= n && text.compare(i, L, kv.first) == 0) {
        hit_len = L;
        hit_id = kv.second;
        if (L == t.max_special_len) break;
      }
    }
    if (hit_len) {
      flush_gap(i);
      ids.push_back(hit_id);
      i += hit_len;
      seg_start = i;
    } else {
      ++i;
    }
  }
  flush_gap(n);
  if (ids.size() > max_len) ids.resize(max_len);
  return ids;
}
#endif  // QWEN21_DRIVER_HAVE_ENCODER + QWEN21_DRIVER_HAVE_TOKENIZER

// Encode stage: owns the caption-embedding state that the text encoder
// produces and hands to the DiT sample phase (cond + null uncond).
//
// Conditioning precedence (first hit wins):
//   1. In-process Qwen forward: --tokenizer (dir with tokenizer.json) +
//      --text-encoder (weights file or shard dir), not --tiny. Tokenizes
//      the prompt and runs the encoder forward on-device. Weights are
//      released before the DiT load.
//   2. Raw float32 LE file (--text-weights / --prompt-embeds-in): cond from
//      file, uncond zeros.
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
  // Conditioning source actually used: "qwen" (in-process forward), "file"
  // (--text-weights / --prompt-embeds-in), or "null" (zeros). Recorded in the
  // --dump-steps manifest.
  const char* source = "null";

  // Device-stage both cond and uncond embeds (HostSpace + one deep_copy
  // each), the same handoff an on-device encoder output uses. The two views
  // are ~L*Cc floats each (MiB-scale: 512x3584 fp32 ~= 7 MiB) and are
  // separate allocations from the GiB-scale encoder weights, so they can
  // stay resident on the device while the encoder weights are released and
  // the DiT is loaded.
  void stage_device() {
    device = Kokkos::View<float**, Kokkos::LayoutRight, Exec>(
        "qwen21_prompt_embeds", L, Cc);
    device_neg = Kokkos::View<float**, Kokkos::LayoutRight, Exec>(
        "qwen21_neg_embeds", L, Cc);
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> stage(
        "qwen21_prompt_embeds_host", L, Cc);
    std::memcpy(stage.data(), embeds.data(), embeds.size() * sizeof(float));
    Kokkos::deep_copy(device, stage);
    std::memcpy(stage.data(), neg_embeds.data(),
                neg_embeds.size() * sizeof(float));
    Kokkos::deep_copy(device_neg, stage);
    Kokkos::fence();
  }

// No in-process Qwen forward yet: the landed QwenImage21EncoderModel::
// forward is a skeleton TODO (it throws), so even with the encoder header
// present the encode phase falls back to file/null with a logged reason
// (see the HAVE_ENCODER note in run() below).
// (ADAPT: implement the tokenize + on-device forward here once the landed
// encoder forward() works.)

#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
  // In-process Qwen forward: raw-text BPE (no chat template) of the T2I
  // prompt, then the landed QwenImage21EncoderModel forward on-device.
  // The pipeline drop_idx counts ONLY the system block
  // ("<|im_start|>system\n<sys><|im_end|>\n", 14 ids for the fixed T2I
  // prompt, matching diffusers'
  // len(apply_chat_template(sys-only message))); the surviving rows START
  // with the "<|im_start|>user\n" prefix (golden prompt_embeds row 0 is
  // <|im_start|>). Using the full-head count would eat those live rows, so
  // the user prefix is split off here (models/ is frozen) and the split is
  // verified to re-encode id-for-id. The uncond branch stays zeros,
  // matching the file/null path (guidance 1.0 takes no CFG branch anyway).
  // The encoder weights are released before returning; only the [L,Cc]
  // embeds survive (staged to device here), so the encoder and DiT are
  // never resident together.
  template <class CfgT>
  static void run_qwen(EncoderStage& st, const CfgT& cfg,
                       const std::string& prompt,
                       const std::string& tokenizer_dir,
                       const std::string& encoder_path) {
    namespace QE = KokkosDNN::QwenImage21Encoder;
    const std::string tok_json = tokenizer_dir + "/tokenizer.json";
    if (!file_exists(tok_json))
      throw std::runtime_error("--tokenizer '" + tokenizer_dir +
                               "' has no tokenizer.json (need the "
                               "Qwen-Image-2.1 processor/ dir)");
    const RawQwenTables tabs = load_raw_qwen_tables(tok_json);
    const size_t max_len = be_max_seq(cfg);
    // Pipeline drop_idx = system block only (see above): qwen21_t2i_head()
    // is that block plus the "<|im_start|>user\n" prefix, which stays in
    // the conditioning sequence.
    const std::string kUserPrefix = "<|im_start|>user\n";
    const std::string full_head = QE::qwen21_t2i_head();
    if (full_head.size() < kUserPrefix.size() ||
        full_head.compare(full_head.size() - kUserPrefix.size(),
                          kUserPrefix.size(), kUserPrefix) != 0)
      throw std::runtime_error(
          "in-process Qwen encode: T2I head lost its user-role prefix");
    const std::string sys_block =
        full_head.substr(0, full_head.size() - kUserPrefix.size());
    const std::vector<int> sys_ids = encode_qwen_raw(tabs, sys_block, max_len);
    const std::vector<int> prefix_ids =
        encode_qwen_raw(tabs, kUserPrefix, max_len);
    const std::vector<int> head_ids =
        encode_qwen_raw(tabs, full_head, max_len);
    if (sys_ids.empty() || prefix_ids.empty() || head_ids.empty() ||
        head_ids.size() != sys_ids.size() + prefix_ids.size() ||
        !std::equal(sys_ids.begin(), sys_ids.end(), head_ids.begin()) ||
        !std::equal(prefix_ids.begin(), prefix_ids.end(),
                    head_ids.begin() + (long)sys_ids.size()))
      throw std::runtime_error(
          "in-process Qwen encode: T2I head/prefix split mismatch");
    const std::vector<int> ids =
        encode_qwen_raw(tabs, QE::qwen21_format_prompt_t2i(prompt), max_len);
    if (head_ids.empty() || ids.size() <= head_ids.size())
      throw std::runtime_error(
          "in-process Qwen encode: empty conditioning sequence");
    const size_t drop = sys_ids.size(), S = ids.size();
    QE::Config ecfg;
    ecfg.validate();
    if ((size_t)ecfg.hidden_dim != st.Cc)
      throw std::runtime_error(
          "Qwen encoder hidden (" + std::to_string(ecfg.hidden_dim) +
          ") != caption dim (" + std::to_string(st.Cc) + ")");
    QE::QwenImage21EncoderModel<Exec, Kokkos::Experimental::bhalf_t> enc(ecfg);
    const std::vector<std::string> shards =
        resolve_shard_list(encoder_path, "--text-encoder");
    if (shards.size() != 1)
      throw std::runtime_error(
          "--text-encoder must be the single-shard text_encoder/ "
          "model.safetensors (got " +
          std::to_string(shards.size()) + " shards)");
    std::printf("[loading %s]\n", encoder_path.c_str());
    enc.weights().load_safetensors(shards[0]);
    std::printf("[encode] Qwen weights resident: %.1f GB\n",
                (double)enc.weights().resident_bytes() / 1e9);
    Kokkos::View<int**, Kokkos::LayoutRight, Exec> toks("qwen21enc_toks", 1, S);
    {
      auto hm = Kokkos::create_mirror_view(toks);
      for (size_t s = 0; s < S; ++s) hm(0, s) = ids[s];
      Kokkos::deep_copy(toks, hm);
    }
    Kokkos::View<Kokkos::Experimental::bhalf_t***, Kokkos::LayoutRight, Exec>
        hid("qwen21enc_hid", 1, S, (size_t)ecfg.hidden_dim);
    KokkosDNN::DNNHandle<Exec> handle;
    enc.forward(handle, toks, hid, (int)drop);
    Kokkos::fence();
    auto hhid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hid);
    const size_t L = S - drop, Cc = st.Cc;
    st.L = L;
    st.embeds.assign(L * Cc, 0.0f);
    st.neg_embeds.assign(L * Cc, 0.0f);
    st.source = "qwen";
    for (size_t t = 0; t < L; ++t)
      for (size_t h = 0; h < Cc; ++h)
        st.embeds[t * Cc + h] = (float)hhid(0, drop + t, h);
    const size_t wbytes = enc.weights().resident_bytes();
    enc.weights().release();
    Kokkos::fence();
    std::printf("[encode] released %.1f GB Qwen weights; %.1f MiB embeds "
                "survive for the sample phase\n",
                (double)wbytes / 1e9,
                (double)(2 * L * Cc * sizeof(float)) / (1024.0 * 1024.0));
    st.stage_device();
    std::printf("[caption qwen] %zux%zu features (raw T2I template, "
                "drop_idx=%zu, tok_vocab=%zu)\n",
                st.L, st.Cc, drop, tabs.vocab_size);
  }
#endif

  template <class CfgT>
  static EncoderStage run(const CfgT& cfg, const std::string& prompt,
                          const std::string& caption_path,
                          const std::string& tokenizer_dir,
                          const std::string& encoder_path, bool tiny) {
    EncoderStage st;
    st.L = be_max_seq(cfg);
    st.Cc = be_caption_dim(cfg);
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
    // In-process Qwen forward when --tokenizer + --text-encoder are given
    // (and not --tiny: the smoke model uses random weights at test geometry,
    // which real token ids cannot address). A throw inside the attempt
    // falls back to file/null with a logged reason instead of failing.
    if (!tiny && (!tokenizer_dir.empty() || !encoder_path.empty())) {
      if (tokenizer_dir.empty() || encoder_path.empty())
        throw std::runtime_error(
            "--tokenizer and --text-encoder must be given together for the "
            "in-process Qwen forward");
      try {
        run_qwen(st, cfg, prompt, tokenizer_dir, encoder_path);
        return st;
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: in-process Qwen encode failed (%s); falling "
                     "back to %s\n",
                     e.what(), caption_path.empty() ? "null" : "file");
        // run_qwen only mutates st on success, but reset the dims anyway so
        // the file/null path below always sizes from the backend maximum.
        st.L = be_max_seq(cfg);
        st.Cc = be_caption_dim(cfg);
      }
    }
    if (tiny && (!tokenizer_dir.empty() || !encoder_path.empty()))
      std::printf("[encode] --tiny skips the Qwen forward (random-weights "
                  "smoke model); falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#elif defined(QWEN21_DRIVER_HAVE_ENCODER)
    // The encoder header is present but the BPE tables header is not, so
    // the in-process forward cannot tokenize: fall through to file/null.
    if (!tiny && (!tokenizer_dir.empty() || !encoder_path.empty()))
      std::printf("[encode] --tokenizer/--text-encoder need "
                  "KokkosDNN_QwenTokenizer.hpp, not present in this "
                  "build; falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
    if (tiny && (!tokenizer_dir.empty() || !encoder_path.empty()))
      std::printf("[encode] --tiny skips the Qwen forward (random-weights "
                  "smoke model); falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#else
    if (!tokenizer_dir.empty() || !encoder_path.empty())
      std::printf("[encode] --tokenizer/--text-encoder need "
                  "KokkosDNN_QwenImage21Encoder.hpp, not present in this "
                  "build; falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#endif
    if (!caption_path.empty()) {
      try {
        // Infer L from the file: golden prompt-embed dumps are [19,4096],
        // not the backend's 512-row maximum. Validate against max_seq so a
        // full-geometry file cannot overflow the tiny smoke model.
        st.embeds = QW::load_caption_raw(caption_path, 0, (int)st.Cc);
        const size_t L = st.embeds.size() / st.Cc;
        if (L == 0 || L > be_max_seq(cfg))
          throw std::runtime_error(
              "caption file has " + std::to_string(L) +
              " rows, max_seq is " + std::to_string(be_max_seq(cfg)));
        st.L = L;
        st.source = "file";
        std::printf("[caption %s] %zux%zu features (L inferred from file)\n",
                    caption_path.c_str(), st.L, st.Cc);
      } catch (const std::exception& e) {
        if (tiny) {
          // A full-geometry file cannot condition the tiny smoke model;
          // fall back to null instead of failing the smoke run.
          std::fprintf(stderr,
                       "warning: caption file '%s' unusable here (%s); using "
                       "null embedding\n",
                       caption_path.c_str(), e.what());
          st.L = be_max_seq(cfg);
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
  QW::print_phase_memory("encode", st.device_bytes(), st.device_bytes());
  std::printf("[budget] full Qwen text encoder projects to %.1f GB; it is "
              "released before the DiT load, so the staged peak stays under "
              "24GB (see README fit table)\n",
              QW::kEncoderBudgetBytes / 1e9);
  std::fflush(stdout);
}

// ---- Nearest-grid sigma schedule -----------------------------------------
// Inference grid with `shift`, snapped to the nearest training-grid
// (shift 1.0, 1000 points) entries, then a terminal 0 is appended.
// unique_consecutive may shrink the count, so the caller must use
// sigmas.size()-1 as the step count. The sample-loop `t` view carries
// 1000*sig (placeholder-model and step-logging convention); the true DiT
// forward takes the RAW [0, 1] sigma (scales by 1000 internally), so its
// adapter divides back by 1000 at the call site. The Euler update is
// x += (sig[i+1]-sig[i]) * v.
inline std::vector<float> build_sigmas_nearest_grid(int steps, float shift) {
  if (steps <= 0)
    throw std::invalid_argument("qwen21 schedule: steps must be > 0");
  if (!(shift > 0.0f) || !std::isfinite(shift))
    throw std::invalid_argument("qwen21 schedule: shift must be > 0");
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
    const size_t pick = ((v - train[lo]) <= (train[hi] - v)) ? lo : hi;
    const float snapped = (float)train[pick];
    if (sig.empty() || sig.back() != snapped) sig.push_back(snapped);
  }
  sig.push_back(0.0f);
  return sig;
}

// ---- Device-resident FlowEuler sample loop --------------------------------
// Reference CFG semantics:
//
//   do_cfg = (guidance != 1.0)            (so 0 <= g < 1 takes the CFG branch)
//   per step: v_cond = DiT(x, t, cond)
//             v_uncond = DiT(x, t, neg)                     if do_cfg
//             v = v_uncond + g*(v_cond - v_uncond)           (CFG combine, fp32)
//             x += (sig[i+1]-sig[i]) * v                    (Euler step)
//
// `forward(x, t, y, out)` is caller-supplied so both backends work: the
// placeholder Qwen21::Model takes no skip set, while the true DiT model
// takes the landed forward signature
// (ADAPT: conform the call site when the Dit header lands).
//
// The captions are sourced from the EncoderStage device views (rank-2
// [L,Cc]) that stayed resident on the device across the encoder-release /
// DiT-load boundary. The only copies are device-to-device (one
// parallel_for per branch into the [1,L,Cc] loop views); nothing
// round-trips through host or disk except through the optional parity dump
// hook below (off by default, so the default loop is copy-free).
//
// Per-step parity-dump hook (--dump-steps): `on_step_done(step, x, v,
// v_cond, v_uncond)` receives host fp32 copies of the post-Euler-update
// latents `x` and of the model output `v` actually used for that step, both
// in the driver's [C,H,W] C-major layout. With the CFG branch active the
// cond/uncond model outputs are passed too; otherwise both stay empty.
struct StepDumpHook {
  bool want_branches = false;
  std::function<void(int, const std::vector<float>&, const std::vector<float>&,
                     const std::vector<float>&, const std::vector<float>&)>
      on_step_done;
  // Optional out: whether the CFG branch actually ran (guidance != 1 and the
  // cond/uncond captions differ).
  bool* use_cfg_out = nullptr;
};

template <class CfgT, class Forward>
std::vector<float> run_sample_loop_from_device(
    const CfgT& cfg, Forward&& forward, const std::vector<float>& x_init,
    const Kokkos::View<float**, Kokkos::LayoutRight, Exec>& cond_d,
    const Kokkos::View<float**, Kokkos::LayoutRight, Exec>& uncond_d,
    int steps, float guidance, float shift, int H, int W,
    const std::function<void(int, float, float)>& on_step = {},
    const std::vector<float>* sigmas_override = nullptr,
    const StepDumpHook& step_dump = {}) {
  if (steps <= 0)
    throw std::invalid_argument("qwen21 sample: steps must be > 0");
  if (!std::isfinite(guidance) || guidance < 0.0f)
    throw std::invalid_argument(
        "qwen21 sample: guidance_scale must be finite and >= 0");
  if (!(shift > 0.0f) || !std::isfinite(shift))
    throw std::invalid_argument("qwen21 sample: shift must be > 0");
  if (!cond_d.is_allocated() || !uncond_d.is_allocated())
    throw std::invalid_argument("qwen21 sample: device captions not staged");
  const size_t C = be_latent_channels(cfg);
  const size_t Cc = be_caption_dim(cfg);
  const size_t HH = (size_t)H, WW = (size_t)W;
  if (HH == 0 || WW == 0)
    throw std::invalid_argument("qwen21 sample: bad latent geometry");
  const size_t T = HH * WW;
  if (x_init.size() != C * HH * WW)
    throw std::invalid_argument("qwen21 sample: x_init size mismatch");
  const size_t L = cond_d.extent(0);
  if (cond_d.extent(1) != Cc || uncond_d.extent(0) != L ||
      uncond_d.extent(1) != Cc)
    throw std::invalid_argument("qwen21 sample: device caption dim mismatch");
  if (L == 0 || L > be_max_seq(cfg))
    throw std::invalid_argument("qwen21 sample: caption length out of range");
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
  if (step_dump.use_cfg_out) *step_dump.use_cfg_out = use_cfg;

  using LatR = Kokkos::View<DScalar****, Kokkos::LayoutRight, Exec>;
  using TokR = Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec>;
  using TVec = Kokkos::View<float*, Exec>;
  LatR x_d("qwen21_loop_x", 1, C, HH, WW);
  LatR vc_d("qwen21_loop_vc", 1, C, HH, WW);
  LatR vu_d("qwen21_loop_vu", 1, C, HH, WW);
  TVec t_d("qwen21_loop_t", 1);
  TokR y_cond_d("qwen21_loop_y_cond", 1, L, Cc);
  TokR y_unc_d("qwen21_loop_y_unc", 1, L, Cc);
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
      "qwen21_stage_cond",
      Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>({0, 0}, {L, Cc}),
      KOKKOS_LAMBDA(size_t l, size_t c) {
        y_cond_d(0, l, c) = static_cast<DScalar>(cond_d(l, c));
        y_unc_d(0, l, c) = static_cast<DScalar>(uncond_d(l, c));
      });
  Kokkos::fence();

  const std::vector<float> sigmas =
      (sigmas_override && !sigmas_override->empty())
          ? *sigmas_override
          : build_sigmas_nearest_grid(steps, shift);
  // AUDIT(a): turbo passes a non-empty sigmas_override, so this ternary
  // bypasses build_sigmas_nearest_grid entirely and shift/flow_shift is
  // unused for the schedule (still range-validated above, never applied).
  std::vector<float> tvals(sigmas.size());
  for (size_t i = 0; i < sigmas.size(); ++i) tvals[i] = 1000.0f * sigmas[i];
  // AUDIT(d): the loop `t` view carries 1000*sigma (placeholder timestep
  // and step-logging convention); the true-DiT adapter divides by 1000 at
  // the call site so forward() sees the RAW [0,1] sigma (scaled to
  // time_factor 1000 internally) — no double-scaling.
  const int n = (int)sigmas.size() - 1;
  const float g = guidance;
  auto t_h = Kokkos::create_mirror_view(t_d);
  for (int i = 0; i < n; ++i) {
    if (on_step) on_step(i, tvals[(size_t)i], sigmas[(size_t)i]);
    // Scalar timestep only: single float host->device per step.
    t_h(0) = tvals[(size_t)i];
    Kokkos::deep_copy(t_d, t_h);
    // AUDIT(b): Euler x += (sig[i+1]-sig[i]) * v with descending sigmas
    // (plus terminal 0), so dt <= 0 stepping from noise toward data,
    // matching FlowMatchEulerDiscreteScheduler / the reference __call__ loop.
    const float dt = sigmas[(size_t)i + 1] - sigmas[(size_t)i];
    // No fences inside the step: every forward, deep_copy, and kernel below
    // runs on the default execution space stream, so launch order already
    // guarantees correctness. The host consumes device data only after the
    // loop, where a single fence guards the final device->host copy.
    // AUDIT(c): guidance == 1.0 forces use_cfg false (see do_cfg/same
    // above), so this step runs exactly ONE forward (cond only, no
    // unconditional branch) plus the plain Euler update.
    forward(x_d, t_d, y_cond_d, vc_d);
    if (use_cfg) {
      forward(x_d, t_d, y_unc_d, vu_d);
      Kokkos::parallel_for(
          "qwen21_loop_cfg_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float vcond = static_cast<float>(vc_d(b, c, h, w));
            const float vun = static_cast<float>(vu_d(b, c, h, w));
            const float v = vun + g * (vcond - vun);
            x_d(b, c, h, w) = static_cast<DScalar>(xc + dt * v);
          });
    } else {
      Kokkos::parallel_for(
          "qwen21_loop_step",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>(
              {(size_t)0, (size_t)0, (size_t)0, (size_t)0},
              {(size_t)1, C, HH, WW}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
            const float xc = static_cast<float>(x_d(b, c, h, w));
            const float v = static_cast<float>(vc_d(b, c, h, w));
            x_d(b, c, h, w) = static_cast<DScalar>(xc + dt * v);
          });
    }
    if (step_dump.on_step_done) {
      // Parity-dump only: fence, then take the host copies for this step.
      // The default sample path never enters this block (no extra
      // allocations and no device->host traffic).
      Kokkos::fence();
      auto vc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      vc_d);
      auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_d);
      decltype(vc_h) vu_h;  // populated only when the CFG branch ran
      if (use_cfg)
        vu_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vu_d);
      std::vector<float> x_step(C * T), v_step(C * T), v_cond, v_uncond;
      if (use_cfg && step_dump.want_branches) {
        v_cond.resize(C * T);
        v_uncond.resize(C * T);
      }
      size_t kk = 0;
      for (size_t c = 0; c < C; ++c)
        for (size_t h = 0; h < HH; ++h)
          for (size_t w = 0; w < WW; ++w) {
            const float a = static_cast<float>(vc_h(0, c, h, w));
            float vv = a;
            if (use_cfg) {
              const float b = static_cast<float>(vu_h(0, c, h, w));
              // Same fp32 CFG combine the Euler kernel above applies.
              vv = b + g * (a - b);
              if (!v_cond.empty()) {
                v_cond[kk] = a;
                v_uncond[kk] = b;
              }
            }
            v_step[kk] = vv;
            x_step[kk] = static_cast<float>(x_h(0, c, h, w));
            ++kk;
          }
      step_dump.on_step_done(i, x_step, v_step, v_cond, v_uncond);
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

#ifdef QWEN21_DRIVER_HAVE_DIT
// True DiT forward: the sample loop holds latents as [1,C,H,W] grids and
// captions as float [1,L,Cc]; the landed QwenImage21DitModel::forward takes
// token-sequence [B,N,C] views plus a double timestep, so this adapter
// packs/unpacks the grid <-> token transpose around each forward.
// Same-stream ordering covers the pack -> forward -> unpack read-after-write
// (no extra fence). The caption extents already match the model's [B,S,Td]
// (Cc == context_dim); only the Scalar (float -> bhalf_t) converts.
// Namespace scope: NVCC rejects function-local closure types as the Forward
// template argument above extended device lambdas, so the forward callable
// is a named functor. Only locals/params/views are touched inside the
// device lambdas (no member access, hence no `this` capture).
struct DitForward {
  KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar>* model;
  KokkosDNN::DNNHandle<Exec>* handle;
  int H, W;  // latent grid (N = H*W tokens)
  template <class X, class T, class Y, class Out>
  void operator()(const X& x, const T& t, const Y& y, Out& out) const {
    const size_t C = x.extent(1);
    const size_t HH = (size_t)H, WW = (size_t)W;
    const size_t N = HH * WW;
    const size_t L = y.extent(1), Cc = y.extent(2);  // y is [1,L,Cc]
    Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> xtok("qwen21_fwd_x",
                                                             1, N, C);
    Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> vtok("qwen21_fwd_v",
                                                             1, N, C);
    Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> ytok("qwen21_fwd_y",
                                                             1, L, Cc);
    Kokkos::parallel_for(
        "qwen21_fwd_pack",
        Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>({0, 0, 0}, {HH, WW, C}),
        KOKKOS_LAMBDA(size_t h, size_t w, size_t c) {
          xtok(0, h * WW + w, c) = static_cast<DScalar>(x(0, c, h, w));
        });
    Kokkos::parallel_for(
        "qwen21_fwd_pack_y",
        Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>({0, 0}, {L, Cc}),
        KOKKOS_LAMBDA(size_t l, size_t c) {
          ytok(0, l, c) = static_cast<DScalar>(y(0, l, c));
        });
    float ts = 0.0f;
    {
      auto th = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t);
      ts = th(0);
    }
    // The sample-loop `t` view carries 1000*sigma (placeholder path and
    // step logging convention); the real DiT forward takes the RAW [0, 1]
    // sigma and scales by time_factor 1000 internally, so divide back here.
    model->forward(*handle, xtok, ytok, (double)ts / 1000.0, vtok, HH, WW);
    Kokkos::parallel_for(
        "qwen21_fwd_unpack",
        Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>({0, 0, 0}, {HH, WW, C}),
        KOKKOS_LAMBDA(size_t h, size_t w, size_t c) {
          out(0, c, h, w) = static_cast<DScalar>(vtok(0, h * WW + w, c));
        });
  }
};
#else
// Placeholder-model forward. Same namespace-scope requirement as above.
struct DitForward {
  QW::Model<Exec, DScalar>* model;
  template <class X, class T, class Y, class Out>
  void operator()(const X& x, const T& t, const Y& y, Out& out) const {
    model->forward(x, t, y, out);
  }
};
#endif

#ifdef QWEN21_DRIVER_HAVE_VAE
// Real VAE decode: raw DiT latents [C,H,W] host float -> device [1,C,H,W]
// -> pipeline denormalize (x*std+mean per channel, runtime affine from
// vae/config.json) -> QwenImage21VaeModel::decode -> RGBA [oH,oW,4] u8
// (x*0.5+0.5 by the pipeline, PNG alpha = ch 3). Returns true on success
// (pix/oH/oW/oCh set, decoded); throws with a reason otherwise so the
// caller can fall back to the grayscale preview.
inline bool try_vae_decode_rgba(const std::vector<float>& lat, size_t C,
                                size_t H, size_t W,
                                const std::string& vae_weights,
                                std::vector<unsigned char>& pix, size_t& oH,
                                size_t& oW, int& oCh,
                                std::vector<float>* rgba_f32 = nullptr) {
  namespace QV = KokkosDNN::QwenImage21Vae;
  QV::Config vcfg;
  {
    std::error_code ec;
    std::string cfg_path;
    if (std::filesystem::is_directory(vae_weights, ec)) {
      cfg_path = vae_weights + "/config.json";
    } else {
      const std::filesystem::path p(vae_weights);
      cfg_path = p.has_parent_path()
                     ? (p.parent_path() / "config.json").string()
                     : "config.json";
    }
    QV::qwen21_load_vae_config_json(cfg_path, vcfg);
  }
  vcfg.validate();
  std::printf("[vae affine %zuch] mean[0]=%.4g std[0]=%.4g (runtime "
              "vae/config.json)\n",
              (size_t)vcfg.z_dim, (double)vcfg.latents_mean[0],
              (double)vcfg.latents_std[0]);
  QV::QwenImage21VaeModel<Exec, DScalar> dec(vcfg);
  {
    std::error_code ec;
    std::vector<std::string> shards =
        std::filesystem::is_directory(vae_weights, ec)
            ? resolve_shard_list(vae_weights, "--vae-weights")
            : std::vector<std::string>{vae_weights};
    std::printf("[loading %s]%s\n", vae_weights.c_str(),
                shards.size() > 1 ? " (shards)" : "");
    dec.weights().load_safetensors(shards);
    std::printf("[vae] weights resident: %.3f GB\n",
                (double)dec.weights().resident_bytes() / 1e9);
  }
  using V4 = Kokkos::View<DScalar****, Kokkos::LayoutRight, Exec>;
  V4 z("qwen21vae_z", 1, C, H, W);
  {
    auto zh = Kokkos::create_mirror_view(z);
    size_t k = 0;
    for (size_t c = 0; c < C; ++c)
      for (size_t h = 0; h < H; ++h)
        for (size_t w = 0; w < W; ++w)
          zh(0, c, h, w) = (DScalar)lat[k++];
    Kokkos::deep_copy(z, zh);
  }
  KokkosDNN::DNNHandle<Exec> handle;
  V4 zden = QV::QwenImage21VaeModel<Exec, DScalar>::denormalize_latents(z,
                                                                       vcfg);
  oH = (size_t)vcfg.compression * H;
  oW = (size_t)vcfg.compression * W;
  V4 rgba("qwen21vae_rgba", 1, (size_t)vcfg.out_channels, oH, oW);
  dec.decode(handle, zden, rgba);
  Kokkos::fence();
  auto rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rgba);
  pix.resize(oH * oW * (size_t)vcfg.out_channels);
  if (rgba_f32) rgba_f32->resize(oH * oW * (size_t)vcfg.out_channels);
  for (size_t h = 0; h < oH; ++h)
    for (size_t w = 0; w < oW; ++w)
      for (int c = 0; c < vcfg.out_channels; ++c) {
        const float clamped =
            std::min(1.0f, std::max(-1.0f, (float)rh(0, (size_t)c, h, w)));
        float v = clamped * 0.5f + 0.5f;
        v = std::min(1.0f, std::max(0.0f, v));
        const size_t idx = (h * oW + w) * (size_t)vcfg.out_channels +
                           (size_t)c;
        pix[idx] = (unsigned char)(std::lround(255.0f * v));
        if (rgba_f32) (*rgba_f32)[idx] = clamped;
      }
  oCh = vcfg.out_channels;
  dec.weights().release();
  Kokkos::fence();
  QW::print_phase_memory("decode", 0, 0);
  return true;
}
#endif  // QWEN21_DRIVER_HAVE_VAE

// Shared image writer: PNG when the image component is built in, otherwise
// the PPM fallback (which drops alpha). Used by the normal pipeline tail and
// by the VAE-only parity path.
inline void write_output_image(const std::string& out,
                                const std::vector<unsigned char>& pix,
                                size_t oH, size_t oW, int oCh, bool decoded) {
#ifdef KOKKOSDNN_ENABLE_IMAGE
  Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::HostSpace> img(
      "qwen21_out", oH, oW, (size_t)oCh);
  for (size_t h = 0; h < oH; ++h)
    for (size_t ww = 0; ww < oW; ++ww)
      for (int c = 0; c < oCh; ++c)
        img(h, ww, (size_t)c) = pix[(h * oW + ww) * (size_t)oCh + (size_t)c];
  KokkosDNN::Image::save_image_png(out, img);
  std::printf("[wrote %s] %zux%zu %s (PNG)\n", out.c_str(), oH, oW,
              decoded ? "VAE RGB" : "grayscale preview");
#else
  const std::string ppm = with_extension(out, ".ppm");
  if (ppm != out)
    std::printf("[note] built without KokkosDNN_ENABLE_IMAGE: writing PPM "
                "to %s instead of PNG %s\n",
                ppm.c_str(), out.c_str());
  if (decoded) {
    if (oCh == 4) {
      // PPM has no alpha: drop it (PNG keeps RGBA).
      std::vector<unsigned char> rgb(oH * oW * 3);
      for (size_t k = 0; k < oH * oW; ++k) {
        rgb[3 * k] = pix[4 * k];
        rgb[3 * k + 1] = pix[4 * k + 1];
        rgb[3 * k + 2] = pix[4 * k + 2];
      }
      save_ppm_rgb(ppm, rgb, oH, oW);
    } else {
      save_ppm_rgb(ppm, pix, oH, oW);
    }
  } else {
    save_ppm(ppm, pix, oH, oW);
  }
  std::printf("[wrote %s] %zux%zu %s (PPM)\n", ppm.c_str(), oH, oW,
              decoded ? "VAE RGB" : "grayscale preview");
#endif
}

// VAE-only parity path (--decode-latents-in): golden latents -> runtime
// affine -> QwenImage21VaeModel::decode -> image. No encoder, no DiT, no
// sampler, so a VAE change can be checked in seconds.
inline void run_vae_only(const std::string& lat_path, bool token_major,
                         size_t C, size_t H, size_t W,
                         const std::string& vae_weights, const std::string& out,
                         const std::string& f32_out) {
#ifdef QWEN21_DRIVER_HAVE_VAE
  std::vector<float> lat = load_decode_latents(lat_path, C, H, W, token_major);
  std::printf("[decode-latents %s] %zux%zux%zu floats (%s layout)\n",
              lat_path.c_str(), C, H, W, token_major ? "token" : "chw");
  std::vector<float> rgba_f32;
  size_t oH = 0, oW = 0;
  int oCh = 1;
  std::vector<unsigned char> pix;
  const bool ok = try_vae_decode_rgba(lat, C, H, W, vae_weights, pix, oH, oW,
                                      oCh,
                                      f32_out.empty() ? nullptr : &rgba_f32);
  if (!ok) throw std::runtime_error("VAE-only decode failed");
  if (!f32_out.empty()) {
    save_raw_f32(f32_out, rgba_f32);
    std::printf("[wrote %s] post-clamp RGBA %zux%zux%d f32\n", f32_out.c_str(),
                oH, oW, oCh);
  }
  write_output_image(out, pix, oH, oW, oCh, /*decoded=*/true);
#else
  (void)lat_path;
  (void)token_major;
  (void)C;
  (void)H;
  (void)W;
  (void)vae_weights;
  (void)out;
  (void)f32_out;
  throw std::runtime_error(
      "--decode-latents-in needs KokkosDNN_QwenImage21Vae.hpp in this build");
#endif
}

void usage();
template <class CfgT>
int run_interactive_loop(const CfgT& cfg, const struct LoopFixedOpts& o);

// ---- Interactive generation loop (default) --------------------------------
// When --prompt is absent (and no parity/file handoff flags are given) the
// driver runs as a REPL on a TTY: it asks for a prompt, asks for an output
// filename, generates, and repeats indefinitely until EOF or quit/exit.
// The encoder and transformer weights are loaded ONCE into host memory
// (HostSpace mirrors) before the first prompt; every iteration copies the
// needed component host->device, runs the phase, and releases the device
// copy -- so staging (default) keeps the same one-component-resident property
// per iteration without re-reading the ~32 GB checkpoint from disk.
// File/parity modes (--prompt-embeds-in/--text-weights, --init-latents-in,
// --dump-steps, --decode-latents-in) and non-TTY stdin force the legacy
// one-off run. In the loop the latent seed is base --seed + iteration, so
// successive images differ even for the same prompt.

inline bool stdin_is_tty() { return ::isatty(STDIN_FILENO) != 0; }

// Print msg, flush, read one line. Returns false on EOF/error. Strips a
// trailing CR so CRLF consoles behave.
inline bool ask_line(const char* msg, std::string& out) {
  std::fputs(msg, stdout);
  std::fflush(stdout);
  if (!std::getline(std::cin, out)) return false;
  if (!out.empty() && out.back() == '\r') out.pop_back();
  return true;
}

inline bool is_quit_word(const std::string& s) {
  return s == "quit" || s == "exit" || s == ":q";
}

#ifdef QWEN21_DRIVER_HAVE_DIT
// Host<->device Dit weight copy (both directions: instantiate with
// DstExec/SrcExec as needed). Both sides must be allocated from the same
// config; extents are checked by deep_copy.
template <class DstExec, class SrcExec>
void copy_dit_weights(
    KokkosDNN::QwenImage21Dit::QwenImage21DitWeights<DstExec, DScalar>& dst,
    const KokkosDNN::QwenImage21Dit::QwenImage21DitWeights<SrcExec, DScalar>&
        src) {
  if (dst.blocks.size() != src.blocks.size())
    throw std::runtime_error("copy_dit_weights: block count mismatch");
  Kokkos::deep_copy(dst.txt_norm, src.txt_norm);
  Kokkos::deep_copy(dst.txt_in_W, src.txt_in_W);
  Kokkos::deep_copy(dst.txt_out_W, src.txt_out_W);
  Kokkos::deep_copy(dst.img_in_W, src.img_in_W);
  Kokkos::deep_copy(dst.time1_W, src.time1_W);
  Kokkos::deep_copy(dst.time2_W, src.time2_W);
  Kokkos::deep_copy(dst.mod_W, src.mod_W);
  Kokkos::deep_copy(dst.norm_out_W, src.norm_out_W);
  Kokkos::deep_copy(dst.proj_out_W, src.proj_out_W);
  for (size_t i = 0; i < dst.blocks.size(); ++i) {
    auto& d = dst.blocks[i];
    const auto& s = src.blocks[i];
    Kokkos::deep_copy(d.to_q, s.to_q);
    Kokkos::deep_copy(d.to_k, s.to_k);
    Kokkos::deep_copy(d.to_v, s.to_v);
    Kokkos::deep_copy(d.to_o, s.to_o);
    Kokkos::deep_copy(d.norm_q, s.norm_q);
    Kokkos::deep_copy(d.norm_k, s.norm_k);
    Kokkos::deep_copy(d.proj, s.proj);
    Kokkos::deep_copy(d.gate, s.gate);
    Kokkos::deep_copy(d.out, s.out);
  }
  Kokkos::fence();
}
#endif

#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
// Host<->device encoder weight copy (bhalf_t, matching run_qwen's
// EncoderModel<Exec, bhalf_t>).
template <class DstExec, class SrcExec>
void copy_enc_weights(
    KokkosDNN::QwenImage21Encoder::QwenImage21EncoderWeights<
        DstExec, Kokkos::Experimental::bhalf_t>& dst,
    const KokkosDNN::QwenImage21Encoder::QwenImage21EncoderWeights<
        SrcExec, Kokkos::Experimental::bhalf_t>& src) {
  if (dst.layers.size() != src.layers.size())
    throw std::runtime_error("copy_enc_weights: layer count mismatch");
  Kokkos::deep_copy(dst.embed, src.embed);
  Kokkos::deep_copy(dst.final_norm, src.final_norm);
  for (size_t i = 0; i < dst.layers.size(); ++i) {
    auto& d = dst.layers[i];
    const auto& s = src.layers[i];
    Kokkos::deep_copy(d.q_W, s.q_W);
    Kokkos::deep_copy(d.k_W, s.k_W);
    Kokkos::deep_copy(d.v_W, s.v_W);
    Kokkos::deep_copy(d.o_W, s.o_W);
    Kokkos::deep_copy(d.qn, s.qn);
    Kokkos::deep_copy(d.kn, s.kn);
    Kokkos::deep_copy(d.in_norm, s.in_norm);
    Kokkos::deep_copy(d.post_norm, s.post_norm);
    Kokkos::deep_copy(d.gate_W, s.gate_W);
    Kokkos::deep_copy(d.up_W, s.up_W);
    Kokkos::deep_copy(d.down_W, s.down_W);
  }
  Kokkos::fence();
}
#endif

// Fixed generation options shared by every loop iteration (prompt, output
// path, and per-iteration seed vary).
struct LoopFixedOpts {
  std::string weights = "transformer";
  std::string tokenizer_dir;
  std::string text_encoder;
  int H = 64, W = 64;
  int steps_eff = 8;
  float guidance = 1.0f, flow_shift = 3.0f;
  std::string sigmas_mode = "driver";
  uint64_t seed = 42;
  std::string vae_weights;
  bool vae_decode = true;
  std::string out_default = "qwen_image_21_out.png";
  bool staged = true;
  bool tiny = false;
};

void usage() {
  std::printf(
      "usage: qwen_image_21_generate [--tiny] [--weights transformer_dir]\n"
      "                    [--prompt \"text\"]\n"
      "                    [--tokenizer tok_dir] [--text-encoder enc_dir]\n"
      "                    [--text-weights feats.f32]\n"
      "                    [--prompt-embeds-in feats.f32]\n"
      "                    [--height H] [--width W]\n"
      "                    [--steps N] [--cfg G] [--shift S]\n"
      "                    [--sigmas driver|turbo]\n"
      "                    [--seed N] [--init-latents-in lat.f32]\n"
      "                    [--vae-weights vae.safetensors [--no-vae-decode]]\n"
      "                    [--decode-latents-in lat.f32\n"
      "                     [--decode-latents-layout chw|token]\n"
      "                     [--decode-out-f32 file]]\n"
      "                    [--out img.png]\n"
      "                    [--no-staged] [--dump-steps DIR]\n"
      "\n"
      "  --tiny         random-weights smoke model, 4x4 latents, 2 steps\n"
      "                 (no files)\n"
      "  --weights      DiT transformer checkpoint: transformer/ shard dir\n"
      "                 or single .safetensors (default: transformer)\n"
      "  --prompt       caption text (Qwen-encoded in-process with\n"
      "                 --tokenizer + --text-encoder, else needs\n"
      "                 --text-weights, else null). Giving --prompt runs ONE\n"
      "                 generation and exits; without it (TTY stdin, no\n"
      "                 --prompt-embeds-in/--text-weights/--init-latents-in/\n"
      "                 --dump-steps/--decode-latents-in) the driver loops:\n"
      "                 it asks for a prompt, asks for a filename, generates,\n"
      "                 and repeats until EOF or quit/exit (empty prompt\n"
      "                 re-asks; empty filename auto-numbers). The per-iteration\n"
      "                 seed is --seed + iteration.\n"
      "  --tokenizer    dir holding tokenizer.json;\n"
      "                 with --text-encoder runs the in-process Qwen forward\n"
      "                 (raw T2I template, no chat template)\n"
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
      "  --steps        Euler steps (default 8 Turbo; --tiny forces 2\n"
      "                 unless --steps is given)\n"
      "  --cfg/--guidance CFG scale (!=1 takes the CFG branch; default 1.0)\n"
      "  --shift/--flow-shift flow-matching sigma shift (default 3.0;\n"
      "                 ignored under --sigmas turbo)\n"
      "  --sigmas       'driver' (default): nearest-grid schedule from\n"
      "                 --steps/--shift; 'turbo': checkpoint sample_sigmas\n"
      "                 verbatim (mu-shifted already, 8 steps, shift=1.0)\n"
      "  --seed         latent init seed (default 42; ignored when\n"
      "                 --init-latents-in is given)\n"
      "  --init-latents-in raw float32 LE initial latents [C,H,W]\n"
      "                 row-major, flattened C-major lat[(c*H+h)*W+w]\n"
      "                 (C*H*W floats, PRE-denormalization latent noise:\n"
      "                 no mean/std applied); replaces the mt19937 sample\n"
      "  --vae-weights  VAE checkpoint (validated whenever given)\n"
      "  --vae-decode   (default ON) run the real VAE decode (denormalize by\n"
      "                 the runtime vae/config.json affine, decode to RGBA)\n"
      "                 when the VAE header + weights are present, else\n"
      "                 grayscale preview. Pass --no-vae-decode for the preview\n"
      "  --decode-latents-in raw fp32 LE latents to decode directly (skips\n"
      "                 encode, DiT load and sampling; needs --vae-weights;\n"
      "                 --height/--width are the latent dims)\n"
      "  --decode-latents-layout 'chw' (default: lat[(c*H+h)*W+w]) or\n"
      "                 'token' (torch packed [H*W,C], golden final latents)\n"
      "  --decode-out-f32 also write the post-clamp RGBA decode as raw f32\n"
      "  --out          PNG with KokkosDNN_ENABLE_IMAGE=ON, else PPM;\n"
      "                 raw F32 latents always written to <stem>.f32\n"
      "  --dump-steps   write golden-parity tensors (raw float32 LE + f64\n"
      "                 sums + manifest.json) into DIR: prompt_embeds\n"
      "                 (+ _neg when --cfg != 1), initial_latents, then\n"
      "                 stepNN_noise_pred / stepNN_latents per Euler step;\n"
      "                 off by default (no extra device->host copies)\n");
}

// ---- Interactive loop driver (both backends) -------------------------------
// CfgT is KokkosDNN::QwenImage21::Config (true pipeline) or QW::Config
// (placeholder); be_* traits select geometry. Parity/file handoff modes are
// excluded by construction (main forces one-off when --prompt-embeds-in,
// --text-weights, --init-latents-in, --dump-steps, or --decode-latents-in is
// given), so this path never dumps and always samples fresh latents.
template <class CfgT>
int run_interactive_loop(const CfgT& cfg, const LoopFixedOpts& o) {
  const int H = o.tiny ? 4 : o.H, W = o.tiny ? 4 : o.W;
  if (!o.tiny) {
    if (H <= 0 || W <= 0)
      throw std::runtime_error("--height/--width must be positive (got " +
                               std::to_string(o.H) + "x" +
                               std::to_string(o.W) + ")");
#ifndef QWEN21_DRIVER_HAVE_DIT
    if (H % 16 != 0 || W % 16 != 0)
      throw std::runtime_error(
          "--height/--width must be positive multiples of 16 (got " +
          std::to_string(o.H) + "x" + std::to_string(o.W) + ")");
#endif
    if (H % be_patch(cfg) != 0 || W % be_patch(cfg) != 0)
      throw std::runtime_error("--height/--width must tile patch_size=" +
                               std::to_string(be_patch(cfg)));
  }
  const size_t C = be_latent_channels(cfg);
  std::printf("[model %s: %s]\n", o.tiny ? "tiny_test" : "qwen-image-2.1",
              be_label(cfg));

  // VAE checkpoint validated once (decoded per iteration from disk; 0.7GB).
  // VAE decode is ON by default whenever --vae-weights are given.
  bool vae_decode = o.vae_decode;
  if (!o.vae_weights.empty()) {
    std::error_code ec0;
    const bool vae_is_dir = std::filesystem::is_directory(o.vae_weights, ec0);
    if (!vae_is_dir && !file_exists(o.vae_weights))
      throw std::runtime_error("vae weights file '" + o.vae_weights +
                               "' not found");
#ifdef QWEN21_DRIVER_HAVE_VAE
    std::printf("[vae %s] validated%s\n", o.vae_weights.c_str(),
                vae_decode ? "; decode enabled (default)"
                           : "; preview (pass --vae-decode to decode)");
#else
    std::printf("[vae %s] validated, but KokkosDNN_QwenImage21Vae.hpp is "
                "not in this build; writing a grayscale preview instead\n",
                o.vae_weights.c_str());
#endif
  } else if (vae_decode) {
    std::printf("[vae] VAE decode needs --vae-weights (else grayscale "
                "preview); disabling decode for this run\n");
    vae_decode = false;
  }

#ifdef QWEN21_DRIVER_HAVE_DIT
  // DiT weights live in host memory from here on; each iteration copies
  // host->device, samples, and releases the device copy.
  KokkosDNN::QwenImage21Dit::QwenImage21DitWeights<Kokkos::HostSpace, DScalar>
      dit_host;
  bool have_dit_host = false;
  if (!o.tiny) {
    const std::vector<std::string> shards =
        resolve_shard_list(o.weights, "--weights");
    KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar> tmp(cfg);
    if (shards.size() > 1)
      std::printf("[loading %zu shards]\n", shards.size());
    else
      std::printf("[loading %s]\n", o.weights.c_str());
    try {
      tmp.weights().load_safetensors(shards);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string(e.what()) +
                               " (need Qwen-Image-2.1 weights or --tiny)");
    }
    dit_host.allocate(cfg);
    copy_dit_weights(dit_host, tmp.weights());
    have_dit_host = true;
    std::printf("[loop] DiT weights resident in host memory: %.1f GB "
                "(copied host->device each iteration%s)\n",
                (double)tmp.weights().resident_bytes() / 1e9,
                o.staged ? ", staged" : "");
  }
#endif

#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
  // In-process Qwen inputs preloaded once: BPE tables + host encoder
  // weights. The per-iteration forward copies host->device into a fresh
  // device model and drops it when the embeds are staged.
  RawQwenTables loop_tabs;
  std::vector<int> loop_head_ids;
  size_t loop_drop = 0;
  KokkosDNN::QwenImage21Encoder::Config loop_ecfg;
  KokkosDNN::QwenImage21Encoder::QwenImage21EncoderWeights<
      Kokkos::HostSpace, Kokkos::Experimental::bhalf_t>
      enc_host;
  bool have_enc_host = false;
  if (!o.tiny && (!o.tokenizer_dir.empty() || !o.text_encoder.empty())) {
    if (o.tokenizer_dir.empty() || o.text_encoder.empty())
      throw std::runtime_error(
          "--tokenizer and --text-encoder must be given together for the "
          "in-process Qwen forward");
    namespace QELoop = KokkosDNN::QwenImage21Encoder;
    const std::string tok_json = o.tokenizer_dir + "/tokenizer.json";
    if (!file_exists(tok_json))
      throw std::runtime_error("--tokenizer '" + o.tokenizer_dir +
                               "' has no tokenizer.json (need the "
                               "Qwen-Image-2.1 processor/ dir)");
    loop_tabs = load_raw_qwen_tables(tok_json);
    const size_t max_len = be_max_seq(cfg);
    const std::string kUserPrefix = "<|im_start|>user\n";
    const std::string full_head = QELoop::qwen21_t2i_head();
    if (full_head.size() < kUserPrefix.size() ||
        full_head.compare(full_head.size() - kUserPrefix.size(),
                          kUserPrefix.size(), kUserPrefix) != 0)
      throw std::runtime_error(
          "in-process Qwen encode: T2I head lost its user-role prefix");
    const std::string sys_block =
        full_head.substr(0, full_head.size() - kUserPrefix.size());
    const std::vector<int> sys_ids =
        encode_qwen_raw(loop_tabs, sys_block, max_len);
    const std::vector<int> prefix_ids =
        encode_qwen_raw(loop_tabs, kUserPrefix, max_len);
    loop_head_ids = encode_qwen_raw(loop_tabs, full_head, max_len);
    if (sys_ids.empty() || prefix_ids.empty() || loop_head_ids.empty() ||
        loop_head_ids.size() != sys_ids.size() + prefix_ids.size() ||
        !std::equal(sys_ids.begin(), sys_ids.end(), loop_head_ids.begin()) ||
        !std::equal(prefix_ids.begin(), prefix_ids.end(),
                    loop_head_ids.begin() + (long)sys_ids.size()))
      throw std::runtime_error(
          "in-process Qwen encode: T2I head/prefix split mismatch");
    loop_drop = sys_ids.size();
    loop_ecfg.validate();
    if ((size_t)loop_ecfg.hidden_dim != be_caption_dim(cfg))
      throw std::runtime_error(
          "Qwen encoder hidden (" + std::to_string(loop_ecfg.hidden_dim) +
          ") != caption dim (" + std::to_string(be_caption_dim(cfg)) + ")");
    KokkosDNN::QwenImage21Encoder::QwenImage21EncoderModel<
        Exec, Kokkos::Experimental::bhalf_t>
        enc_tmp(loop_ecfg);
    const std::vector<std::string> eshards =
        resolve_shard_list(o.text_encoder, "--text-encoder");
    if (eshards.size() != 1)
      throw std::runtime_error(
          "--text-encoder must be the single-shard text_encoder/ "
          "model.safetensors (got " +
          std::to_string(eshards.size()) + " shards)");
    std::printf("[loading %s]\n", o.text_encoder.c_str());
    enc_tmp.weights().load_safetensors(eshards[0]);
    enc_host.allocate(loop_ecfg);
    copy_enc_weights(enc_host, enc_tmp.weights());
    have_enc_host = true;
    std::printf("[loop] Qwen encoder weights resident in host memory: %.1f "
                "GB (copied host->device each iteration)\n",
                (double)enc_tmp.weights().resident_bytes() / 1e9);
  }
  if (o.tiny && (!o.tokenizer_dir.empty() || !o.text_encoder.empty()))
    std::printf("[encode] --tiny skips the Qwen forward (random-weights "
                "smoke model); using null embeddings\n");
#else
  if (!o.tiny && (!o.tokenizer_dir.empty() || !o.text_encoder.empty()))
    std::printf("[encode] --tokenizer/--text-encoder need "
                "KokkosDNN_QwenImage21Encoder.hpp + "
                "KokkosDNN_QwenTokenizer.hpp, not present in this build; "
                "using null embeddings\n");
#endif

  // Sigma schedule is prompt-independent: build once, reuse every iteration.
  std::vector<float> base_sigmas;
  if (o.sigmas_mode == "turbo") {
#ifdef QWEN21_DRIVER_HAVE_OPS
    for (double s : cfg.sample_sigmas)
      base_sigmas.push_back(static_cast<float>(s));
    base_sigmas.push_back(0.0f);
    std::printf("[schedule turbo] checkpoint sample_sigmas verbatim "
                "(mu-shifted already)\n");
#else
    throw std::runtime_error(
        "--sigmas turbo needs the true QwenImage21::Config "
        "(KokkosDNN_QwenImage21Config.hpp is not in this build)");
#endif
  } else {
    base_sigmas = build_sigmas_nearest_grid(o.steps_eff, o.flow_shift);
  }
  const int steps_run = static_cast<int>(base_sigmas.size()) - 1;

  std::printf(
      "[loop] interactive generation: prompt, then output filename, per "
      "iteration (EOF or quit/exit stops; empty prompt re-asks; empty "
      "filename auto-numbers qwen_image_21_out_<n>.png; seed = %llu + "
      "iteration)\n",
      (unsigned long long)o.seed);
  std::fflush(stdout);
  size_t iter = 0;
  while (true) {
    std::string cur_prompt;
    if (!ask_line("prompt> ", cur_prompt)) break;
    if (is_quit_word(cur_prompt)) break;
    if (cur_prompt.empty()) continue;
    std::string cur_out_raw;
    if (!ask_line("output> ", cur_out_raw)) break;
    if (is_quit_word(cur_out_raw)) continue;  // cancel this one
    const std::string cur_out =
        cur_out_raw.empty()
            ? ("qwen_image_21_out_" + std::to_string(iter) + ".png")
            : cur_out_raw;
    const uint64_t cur_seed = o.seed + iter;
    std::printf("[iter %zu] seed=%llu out=%s\n", iter,
                (unsigned long long)cur_seed, cur_out.c_str());
    std::fflush(stdout);

    // ---- Phase 1: encode (device encoder is a per-iteration copy) ----
    EncoderStage enc;
    bool encoded = false;
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
    if (have_enc_host) {
      try {
        namespace QEIter = KokkosDNN::QwenImage21Encoder;
        const size_t max_len = be_max_seq(cfg);
        const std::vector<int> ids = encode_qwen_raw(
            loop_tabs, QEIter::qwen21_format_prompt_t2i(cur_prompt), max_len);
        if (loop_head_ids.empty() || ids.size() <= loop_head_ids.size())
          throw std::runtime_error(
              "in-process Qwen encode: empty conditioning sequence");
        KokkosDNN::QwenImage21Encoder::QwenImage21EncoderModel<
            Exec, Kokkos::Experimental::bhalf_t>
            enc_model(loop_ecfg);
        copy_enc_weights(enc_model.weights(), enc_host);
        Kokkos::View<int**, Kokkos::LayoutRight, Exec> toks("qwen21enc_toks",
                                                            1, ids.size());
        {
          auto hm = Kokkos::create_mirror_view(toks);
          for (size_t s = 0; s < ids.size(); ++s) hm(0, s) = ids[s];
          Kokkos::deep_copy(toks, hm);
        }
        Kokkos::View<Kokkos::Experimental::bhalf_t***, Kokkos::LayoutRight,
                      Exec>
            hid("qwen21enc_hid", 1, ids.size(),
                (size_t)loop_ecfg.hidden_dim);
        KokkosDNN::DNNHandle<Exec> enc_handle;
        enc_model.forward(enc_handle, toks, hid, (int)loop_drop);
        Kokkos::fence();
        auto hhid =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hid);
        const size_t L = ids.size() - loop_drop;
        const size_t Cc = be_caption_dim(cfg);
        enc.L = L;
        enc.Cc = Cc;
        enc.embeds.assign(L * Cc, 0.0f);
        enc.neg_embeds.assign(L * Cc, 0.0f);
        enc.source = "qwen";
        for (size_t t = 0; t < L; ++t)
          for (size_t h = 0; h < Cc; ++h)
            enc.embeds[t * Cc + h] = (float)hhid(0, loop_drop + t, h);
        std::printf("[caption qwen] %zux%zu features (host->device copy, "
                    "drop_idx=%zu)\n",
                    enc.L, enc.Cc, loop_drop);
        enc.stage_device();
        encoded = true;
        // enc_model goes out of scope here: the device encoder copy is
        // released while the host cache above survives for the next prompt.
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "warning: in-process Qwen encode failed (%s); using "
                     "null embedding\n",
                     e.what());
      }
    }
#endif
    if (!encoded) {
      enc.L = be_max_seq(cfg);
      enc.Cc = be_caption_dim(cfg);
      enc.embeds.assign(enc.L * enc.Cc, 0.0f);
      enc.neg_embeds.assign(enc.L * enc.Cc, 0.0f);
      enc.source = "null";
      std::printf("[caption null] prompt \"%s\" has no features here; using "
                  "zeros (%zux%zu)\n",
                  cur_prompt.c_str(), enc.L, enc.Cc);
      enc.stage_device();
    }
    log_encode_memory(enc);
    if (o.staged) {
      Kokkos::fence();
      QW::print_phase_memory("encode-released", enc.device_bytes(), 0);
      std::printf("[staged] keeping %.1f MiB of embeds resident on device "
                  "across the DiT load (no disk round-trip)\n",
                  (double)enc.device_bytes() / (1024.0 * 1024.0));
    }

    // ---- Phase 2: DiT (device copy restored from the host cache) ----
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::QwenImage21Dit::QwenImage21DitModel<Exec, DScalar> dit_model(
        cfg);
    {
      auto& w = dit_model.weights();
      if (have_dit_host) {
        copy_dit_weights(w, dit_host);
        std::printf("[loading host->device %.1f GB]\n",
                    (double)w.resident_bytes() / 1e9);
      } else if (o.tiny) {
        w.random_init(o.seed);
        std::printf("[tiny random weights] layers=%d hidden=%d latent=%dx%d\n",
                    cfg.num_layers, cfg.hidden_dim, H, W);
      } else {
        throw std::runtime_error(
            "need Qwen-Image-2.1 weights or --tiny (unreachable: host load "
            "would have failed before the loop)");
      }
      QW::print_phase_memory("sample", w.resident_bytes(),
                             dit_model.scratch_bytes());
    }
#else
    QW::Weights<Exec, DScalar> w;
    if (o.tiny) {
      QW::random_init(cfg, w, o.seed);
      std::printf("[tiny random weights] hidden=%d depth=%d latent=%dx%d\n",
                  cfg.hidden_size, cfg.depth, H, W);
    } else {
      throw std::runtime_error(
          "weights file '" + o.weights +
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

    std::vector<float> x_init =
        QW::sample_gaussian(C * (size_t)H * (size_t)W, cur_seed);
    std::printf("[sample] H=%d W=%d steps=%d cfg=%.3g shift=%.3g seed=%llu\n",
                H, W, steps_run, (double)o.guidance, (double)o.flow_shift,
                (unsigned long long)cur_seed);
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::DNNHandle<Exec> dit_handle;
    DitForward dit_forward{&dit_model, &dit_handle, H, W};
#else
    QW::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps_run,
        o.guidance, o.flow_shift, H, W,
        [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        },
        &base_sigmas, StepDumpHook{});
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

    const std::string lat_path = with_extension(cur_out, ".f32");
    save_raw_f32(lat_path, lat);
    std::printf("[wrote %s] raw fp32 latents\n", lat_path.c_str());

    // Captions are no longer needed: drop host + device copies before decode.
    enc.release();
    // ---- Phase 3: decode (the DiT device copy is released first) ----
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
                                      o.vae_weights, pix, oH, oW, oCh);
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
    write_output_image(cur_out, pix, oH, oW, oCh, decoded);
    ++iter;
  }
  std::printf("[loop] done after %zu generation%s\n", iter,
              iter == 1 ? "" : "s");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "transformer";
  std::string prompt = "a red panda reading a newspaper";
  std::string text_weights, prompt_embeds_in, init_latents_in, dump_dir;
  std::string tokenizer_dir, text_encoder;
  std::string vae_weights, out = "qwen_image_21_out.png";
  std::string decode_latents_in, decode_latents_layout = "chw",
              decode_out_f32;
  int height = 64, width = 64;
  int steps = 8;
  bool steps_given = false;
  float guidance = 1.0f, flow_shift = 3.0f;
  std::string sigmas_mode = "driver";
  uint64_t seed = 42;
  bool tiny = false;
  bool staged = true, vae_decode = true;
  bool prompt_given = false;

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
      prompt_given = true;
    } else if (a == "--tokenizer" && i + 1 < argc) {
      tokenizer_dir = argv[++i];
    } else if (a == "--text-encoder" && i + 1 < argc) {
      text_encoder = argv[++i];
    } else if (a == "--text-weights" && i + 1 < argc) {
      text_weights = argv[++i];
    } else if (a == "--prompt-embeds-in" && i + 1 < argc) {
      prompt_embeds_in = argv[++i];
    } else if (a == "--init-latents-in" && i + 1 < argc) {
      init_latents_in = argv[++i];
    } else if (a == "--dump-steps" && i + 1 < argc) {
      dump_dir = argv[++i];
    } else if (a == "--vae-weights" && i + 1 < argc) {
      vae_weights = argv[++i];
    } else if (a == "--vae-decode") {
      vae_decode = true;
    } else if (a == "--no-vae-decode" || a == "--no-vae_decode") {
      vae_decode = false;
    } else if (a == "--decode-latents-in" && i + 1 < argc) {
      decode_latents_in = argv[++i];
    } else if (a == "--decode-latents-layout" && i + 1 < argc) {
      decode_latents_layout = argv[++i];
      if (decode_latents_layout != "chw" && decode_latents_layout != "token") {
        std::fprintf(stderr,
                     "error: --decode-latents-layout must be 'chw' or "
                     "'token' (got '%s')\n",
                     decode_latents_layout.c_str());
        return 1;
      }
    } else if (a == "--decode-out-f32" && i + 1 < argc) {
      decode_out_f32 = argv[++i];
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
        std::fprintf(stderr, "error: --sigmas must be 'driver' or 'turbo' "
                             "(got '%s')\n",
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
    // True-pipeline DiT header is present: use the landed geometry
    // (QwenImage21::Config: the default ctor is the full 32x4096/64ch/ctx4096
    // config; tiny_test() is the 2x32/4ch/ctx16 smoke config).
    KokkosDNN::QwenImage21::Config cfg =
        tiny ? KokkosDNN::QwenImage21::Config::tiny_test()
             : KokkosDNN::QwenImage21::Config();
    if (tiny) {
      // tiny_test() is the 2x32/4ch/ctx16 smoke config (now validate-clean).
    }
    cfg.validate();
#else
    QW::Config cfg = tiny ? QW::Config::tiny_test() : QW::Config::qwen_image_21();
    cfg.validate();
#endif
    const int steps_eff = (!steps_given && tiny) ? 2 : steps;
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
#ifndef QWEN21_DRIVER_HAVE_DIT
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
    std::printf("[model %s: %s]\n", tiny ? "tiny_test" : "qwen-image-2.1",
                be_label(cfg));

    // ---- VAE-only parity path (--decode-latents-in) ----
    // Bypasses encode + DiT load + sampler: golden latents go straight
    // through the runtime affine + VAE decode. Needs the real VAE.
    if (!decode_latents_in.empty()) {
      if (vae_weights.empty())
        throw std::runtime_error(
            "--decode-latents-in needs --vae-weights (implies decode)");
      if (tiny)
        throw std::runtime_error(
            "--decode-latents-in needs the real VAE; drop --tiny");
      run_vae_only(decode_latents_in, decode_latents_layout == "token", C,
                   (size_t)H, (size_t)W, vae_weights, out, decode_out_f32);
      Kokkos::finalize();
      return 0;
    }

    // Resolve the caption source: --prompt-embeds-in is an alias of
    // --text-weights and wins when both are given with different values.
    std::string caption_src = text_weights;
    if (!prompt_embeds_in.empty()) {
      if (!text_weights.empty() && prompt_embeds_in != text_weights)
        std::fprintf(stderr,
                     "warning: --prompt-embeds-in wins over --text-weights\n");
      caption_src = prompt_embeds_in;
    }

    // Default interactive loop: no --prompt (and no file/parity handoff) on
    // a TTY means prompt -> filename -> generate, indefinitely. Weights are
    // preloaded once into host memory and copied host->device per iteration
    // (see run_interactive_loop). Anything script-like -- an explicit
    // --prompt, a caption/latent file, --dump-steps, or non-TTY stdin --
    // takes the legacy one-off path below, so golden/CI runs are unchanged.
    const bool loop_mode =
        !prompt_given && caption_src.empty() && init_latents_in.empty() &&
        dump_dir.empty() && stdin_is_tty();
    if (loop_mode) {
      LoopFixedOpts lo;
      lo.weights = weights;
      lo.tokenizer_dir = tokenizer_dir;
      lo.text_encoder = text_encoder;
      lo.H = H;
      lo.W = W;
      lo.steps_eff = steps_eff;
      lo.guidance = guidance;
      lo.flow_shift = flow_shift;
      lo.sigmas_mode = sigmas_mode;
      lo.seed = seed;
      lo.vae_weights = vae_weights;
      lo.vae_decode = vae_decode;
      lo.out_default = out;
      lo.staged = staged;
      lo.tiny = tiny;
      const int lrc = run_interactive_loop(cfg, lo);
      Kokkos::finalize();
      return lrc;
    }

    // ---- Phase 1: encode (no DiT weights are loaded yet) ----
    EncoderStage enc = EncoderStage::run(cfg, prompt, caption_src,
                                         tokenizer_dir, text_encoder, tiny);
    log_encode_memory(enc);

    // In-memory handoff: the embeds (~L*Cc floats per branch, MiB-scale) are
    // separate device allocations from the GiB-scale encoder weights, so they
    // stay resident on the device while the encoder weights are released and
    // the DiT is loaded. No disk round-trip and no second process.
    if (staged) {
      // When the real Qwen encoder lands in this TU, its EncoderModel weights
      // are released here; the EncoderStage embeds are independent Views and
      // survive. Until then this is a fence + accounting point only.
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
    // True-pipeline DiT (QwenImage21DitModel owns its weights + scratch).
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
      // --init-latents-in: raw float32 LE initial latents in the driver's
      // native layout ([C,H,W] row-major, flattened C-major as
      // lat[(c*H+h)*W+w], the same order the sampler packs into
      // x_d(0,c,h,w) and the <stem>.f32 files it writes). Values are
      // PRE-denormalization latent noise (no mean/std applied); validated
      // against C*H*W, deep-copied to device by the sample loop, and used
      // in place of the mt19937 sample.
      x_init = load_init_latents_raw(init_latents_in, C, (size_t)H, (size_t)W);
      std::printf("[init-latents %s] %zux%dx%d floats (fp32 LE [C,H,W])\n",
                  init_latents_in.c_str(), C, H, W);
    } else {
      x_init = QW::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    }
    // Sigma schedule: 'driver' (default, current behavior) builds the
    // nearest-grid schedule from --steps/--shift; 'turbo' uses the
    // checkpoint sample_sigmas verbatim (mu-shifted already, so shift is
    // 1.0/no-shift in that mode) plus a terminal 0, i.e. 8 Euler steps.
    std::vector<float> turbo_sigmas;
    if (sigmas_mode == "turbo") {
#ifdef QWEN21_DRIVER_HAVE_OPS
      for (double s : cfg.sample_sigmas)
        turbo_sigmas.push_back((float)s);
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
    std::printf("[sample] H=%d W=%d steps=%d cfg=%.3g shift=%.3g seed=%llu\n",
                H, W, steps_run, (double)guidance, (double)flow_shift,
                (unsigned long long)seed);
    // ---- Parity dumps (--dump-steps DIR): inert unless the flag is set ----
    DumpWriter dump(dump_dir);
    std::vector<float> dump_sigmas;  // exact schedule, incl. terminal 0
    bool dump_use_cfg = false;
    if (dump.active()) {
      // The driver-mode schedule is rebuilt here because the nearest-grid
      // snap can drop repeated entries: the real step count is
      // sigmas.size()-1, not necessarily --steps.
      dump_sigmas = (sigmas_mode == "turbo")
                        ? turbo_sigmas
                        : build_sigmas_nearest_grid(steps_eff, flow_shift);
      // Conditioning exactly as staged for the DiT ([L,Cc] row-major).
      {
        auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      enc.device);
        const size_t L = enc.device.extent(0), Cc = enc.device.extent(1);
        std::vector<float> pe(L * Cc);
        size_t k = 0;
        for (size_t l = 0; l < L; ++l)
          for (size_t c = 0; c < Cc; ++c) pe[k++] = (float)hc(l, c);
        dump.write("prompt_embeds", pe);
        if (guidance != 1.0f) {
          auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        enc.device_neg);
          std::vector<float> ne(L * Cc);
          k = 0;
          for (size_t l = 0; l < L; ++l)
            for (size_t c = 0; c < Cc; ++c) ne[k++] = (float)hu(l, c);
          dump.write("prompt_embeds_neg", ne);
        }
      }
      // The sampler's starting latents, exactly as handed to it (no
      // additional scaling).
      dump.write("initial_latents", x_init);
    }
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::DNNHandle<Exec> dit_handle;
    DitForward dit_forward{&dit_model, &dit_handle, H, W};
#else
    QW::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
    // Per-step parity dump (--dump-steps): writes stepNN_noise_pred.f32 /
    // stepNN_latents.f32 (+ the cond/uncond branches under CFG) from the
    // host copies the loop takes only when this hook is set.
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
    // Sample straight from the device-resident embeds staged above: no host
    // reload, no disk re-read. The host vectors (enc.embeds/neg_embeds) stay
    // as the fallback for the --prompt-embeds-in file path, which also lands
    // staged on-device via EncoderStage::run.
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps_run,
        guidance, flow_shift, H, W,
        [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        },
        (sigmas_mode == "turbo") ? &turbo_sigmas : nullptr, dump_hook);
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

    std::string lat_path = with_extension(out, ".f32");
    save_raw_f32(lat_path, lat);
    std::printf("[wrote %s] raw fp32 latents\n", lat_path.c_str());

    if (dump.active()) {
      DumpManifest dm;
      dm.prompt = prompt;
      dm.prompt_source = enc.source;
      dm.model_label = be_label(cfg);
      dm.sigmas_mode = sigmas_mode;
      dm.sigmas = dump_sigmas;
      dm.height = H;
      dm.width = W;
      dm.steps = (int)dump_sigmas.size() - 1;
      dm.latent_channels = C;
      dm.prompt_len = enc.device.extent(0);
      dm.prompt_dim = enc.device.extent(1);
      dm.seed = (unsigned long long)seed;
      dm.guidance = (double)guidance;
      dm.use_cfg = dump_use_cfg;
      dm.staged = staged;
      if (dm.steps != steps_run)
        std::printf("[dump] note: snapped schedule has %d steps (driver "
                    "reported %d)\n",
                    dm.steps, steps_run);
      write_dump_manifest(dump, dm);
    }

    // Captions are no longer needed: drop host + device copies before decode.
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
    // Only consumed by the PNG path (KOKKOSDNN_ENABLE_IMAGE); the PPM
    // fallback below ignores it.
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
