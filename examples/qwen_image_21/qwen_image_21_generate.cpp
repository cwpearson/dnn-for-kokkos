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
// the corresponding phase runs the real model; otherwise the phase falls
// back to the placeholder Qwen21:: geometry / file / null path with a
// logged reason, so the tree always builds. Macros
// QWEN21_DRIVER_HAVE_{OPS,DIT,ENCODER,VAE} gate every use (see the shim
// below). Status: Config geometry + the DiT are usable when present (the
// driver samples with the real DiT, including --tiny on random weights);
// the landed encoder forward() and VAE decode() are still skeleton TODOs
// (they throw), so encode resolves --prompt-embeds-in files or null and
// decode writes a grayscale preview, each with a logged reason.
//
//   qwen_image_21_generate [--tiny] [--weights transformer_dir]
//                          [--prompt "text"]
//                          [--tokenizer tok_dir] [--text-encoder enc_dir]
//                          [--text-weights feats.f32]
//                          [--prompt-embeds-in feats.f32]
//                          [--height H] [--width W]
//                          [--steps 8] [--cfg 1.0] [--shift 3.0]
//                          [--seed 42]
//                          [--vae-weights vae.safetensors [--vae-decode]]
//                          [--out img.png]
//                          [--staged]
//
// Text conditioning precedence: in-process Qwen forward (--tokenizer dir
// with tokenizer.json + --text-encoder weights, once the landed encoder
// forward() is implemented) -> --prompt-embeds-in file (alias
// --text-weights) -> null (zeros) with the prompt logged. --tokenizer /
// --text-encoder must be given together. --tiny skips the Qwen forward.
// --vae-weights is validated whenever given; --vae-decode additionally runs
// the real VAE decode once the landed decode() is implemented, else a
// channel-mean grayscale preview is written with a clear log. Raw fp32
// latents (<stem>.f32, [C,H,W] row-major) are always written.
//
// Staged execution (24GB fit): --staged runs encode -> releases the encoder
// weights (the MiB-scale embeds stay resident on the device) ->
// Kokkos::fence() -> loads the DiT -> samples straight from the
// device-resident embeds -> releases the DiT -> Kokkos::fence() -> decodes,
// so the 17.5GB encoder, the 14.2GB DiT, and the VAE are never resident at
// once. No disk round-trip and no second process: the embeds are separate
// (MiB) allocations from the (GiB) encoder weights. Each phase logs its
// resident estimate (weights bytes + activation estimate).
//
// --tiny runs a random-weights smoke model on 4x4 latents, 2 steps
// (no downloads).
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
#include <string>
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
// Status: Config geometry + the DiT (Model/Weights/random_init/
// load_safetensors/resident_bytes/release + release_scratch) are usable
// and consumed below when present. The landed QwenImage21EncoderModel::
// forward and QwenImage21VaeModel::decode are still skeleton TODOs (they
// throw), so the encode and decode phases fall back to file/null + preview
// with a logged reason even when those headers are present. Every
// true-pipeline use is gated on its QWEN21_DRIVER_HAVE_* macro, so this TU
// builds with or without any of those headers. There is intentionally NO
// hard dependency here (and none in CMakeLists.txt): the driver probes,
// never requires.
//
// ADAPT: enable the in-process Qwen forward in EncoderStage::run and the
// real VAE decode in main once the landed forward()/decode() are
// implemented (see the skeleton notes at those sites).
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
// [1,L,Cc], [1,1,L,Cc] flattened (takes the trailing block).
inline std::vector<float> load_caption_raw(const std::string& path, int expect_L,
                                           int expect_Cc) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("qwen21: cannot open '" + path + "'");
  const long long n = (long long)f.tellg();
  if (n <= 0 || n % 4 != 0)
    throw std::runtime_error("qwen21: bad float32 file '" + path + "'");
  f.seekg(0, std::ios::beg);
  std::vector<float> out((size_t)n / 4);
  f.read(reinterpret_cast<char*>(out.data()), n);
  if (!f) throw std::runtime_error("qwen21: short read of '" + path + "'");
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
// (sorted). Shared by --text-encoder and --weights. Only needed when a
// true-pipeline header is present; guarded so header-less builds see no
// unused-function warning.
#if defined(QWEN21_DRIVER_HAVE_DIT) || defined(QWEN21_DRIVER_HAVE_ENCODER)
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

  template <class CfgT>
  static EncoderStage run(const CfgT& cfg, const std::string& prompt,
                          const std::string& caption_path,
                          const std::string& tokenizer_dir,
                          const std::string& encoder_path, bool tiny) {
    EncoderStage st;
    st.L = be_max_seq(cfg);
    st.Cc = be_caption_dim(cfg);
#ifdef QWEN21_DRIVER_HAVE_ENCODER
    // The landed QwenImage21EncoderModel::forward is still a skeleton (it
    // throws TODO), so the in-process forward is unavailable even with the
    // header present: fall through to file/null with a logged reason.
    // (ADAPT: tokenize + run the encoder forward here once it lands.)
    if (!tiny && (!tokenizer_dir.empty() || !encoder_path.empty()))
      std::printf("[encode] --tokenizer/--text-encoder need a working "
                  "QwenImage21EncoderModel::forward (still a skeleton TODO "
                  "in this build); falling back to %s\n",
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
        st.embeds = QW::load_caption_raw(caption_path, (int)st.L, (int)st.Cc);
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
// sigmas.size()-1 as the step count. Model timestep inputs are 1000*sig,
// and the Euler update is x += (sig[i+1]-sig[i]) * v.
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
// round-trips through host or disk.
template <class CfgT, class Forward>
std::vector<float> run_sample_loop_from_device(
    const CfgT& cfg, Forward&& forward, const std::vector<float>& x_init,
    const Kokkos::View<float**, Kokkos::LayoutRight, Exec>& cond_d,
    const Kokkos::View<float**, Kokkos::LayoutRight, Exec>& uncond_d,
    int steps, float guidance, float shift, int H, int W,
    const std::function<void(int, float, float)>& on_step = {}) {
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

  const std::vector<float> sigmas = build_sigmas_nearest_grid(steps, shift);
  std::vector<float> tvals(sigmas.size());
  for (size_t i = 0; i < sigmas.size(); ++i) tvals[i] = 1000.0f * sigmas[i];
  const int n = (int)sigmas.size() - 1;
  const float g = guidance;
  auto t_h = Kokkos::create_mirror_view(t_d);
  for (int i = 0; i < n; ++i) {
    if (on_step) on_step(i, tvals[(size_t)i], sigmas[(size_t)i]);
    // Scalar timestep only: single float host->device per step.
    t_h(0) = tvals[(size_t)i];
    Kokkos::deep_copy(t_d, t_h);
    const float dt = sigmas[(size_t)i + 1] - sigmas[(size_t)i];
    // No fences inside the step: every forward, deep_copy, and kernel below
    // runs on the default execution space stream, so launch order already
    // guarantees correctness. The host consumes device data only after the
    // loop, where a single fence guards the final device->host copy.
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
    model->forward(*handle, xtok, ytok, (double)ts, vtok, HH, WW);
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

void usage() {
  std::printf(
      "usage: qwen_image_21_generate [--tiny] [--weights transformer_dir]\n"
      "                    [--prompt \"text\"]\n"
      "                    [--tokenizer tok_dir] [--text-encoder enc_dir]\n"
      "                    [--text-weights feats.f32]\n"
      "                    [--prompt-embeds-in feats.f32]\n"
      "                    [--height H] [--width W]\n"
      "                    [--steps N] [--cfg G] [--shift S]\n"
      "                    [--seed N]\n"
      "                    [--vae-weights vae.safetensors [--vae-decode]]\n"
      "                    [--out img.png]\n"
      "                    [--staged]\n"
      "\n"
      "  --tiny         random-weights smoke model, 4x4 latents, 2 steps\n"
      "                 (no files)\n"
      "  --weights      DiT transformer checkpoint: transformer/ shard dir\n"
      "                 or single .safetensors (default: transformer)\n"
      "  --prompt       caption text (Qwen-encoded in-process with\n"
      "                 --tokenizer + --text-encoder, else needs\n"
      "                 --text-weights, else null)\n"
      "  --tokenizer    dir holding tokenizer.json;\n"
      "                 requests the in-process Qwen forward with\n"
      "                 --text-encoder (skeleton: falls back, see log)\n"
      "  --text-weights raw float32 LE caption features [L,Cc]\n"
      "                 (L <= maxlen, Cc == caption dim)\n"
      "  --text-encoder Qwen text-encoder weights: text_encoder/ shard dir\n"
      "                 or single .safetensors\n"
      "  --prompt-embeds-in alias of --text-weights (wins if both differ)\n"
      "  --staged       encode -> release encoder weights (MiB embeds stay\n"
      "                 resident on device) -> fence -> load DiT -> sample\n"
      "                 from device embeds -> release DiT -> fence -> decode\n"
      "                 (one component resident at a time, no disk round-trip)\n"
      "  --height/--width latent H/W (default 64; x16 = pixels, so 64\n"
      "                 means 1024px; placeholder path needs multiples of 16)\n"
      "  --steps        Euler steps (default 8 Turbo; --tiny forces 2\n"
      "                 unless --steps is given)\n"
      "  --cfg/--guidance CFG scale (!=1 takes the CFG branch; default 1.0)\n"
      "  --shift/--flow-shift flow-matching sigma shift (default 3.0)\n"
      "  --seed         latent init seed (default 42)\n"
      "  --vae-weights  VAE checkpoint (validated whenever given)\n"
      "  --vae-decode   run the real VAE decode when the VAE header +\n"
      "                 weights are present (else grayscale preview)\n"
      "  --out          PNG with KokkosDNN_ENABLE_IMAGE=ON, else PPM;\n"
      "                 raw F32 latents always written to <stem>.f32\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "transformer";
  std::string prompt = "a red panda reading a newspaper";
  std::string text_weights, prompt_embeds_in;
  std::string tokenizer_dir, text_encoder;
  std::string vae_weights, out = "qwen_image_21_out.png";
  int height = 64, width = 64;
  int steps = 8;
  bool steps_given = false;
  float guidance = 1.0f, flow_shift = 3.0f;
  uint64_t seed = 42;
  bool tiny = false;
  bool staged = false, vae_decode = false;

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
    } else if (a == "--seed" && i + 1 < argc) {
      seed = (uint64_t)std::stoull(argv[++i]);
    } else if (a == "--tiny") {
      tiny = true;
    } else if (a == "--staged") {
      staged = true;
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
      std::printf("[note] single-phase run: encoder embeds stay live "
                  "across the DiT load (pass --staged for the fenced, logged "
                  "encoder-release point)\n");
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
      if (!file_exists(vae_weights))
        throw std::runtime_error("vae weights file '" + vae_weights +
                                 "' not found");
#ifdef QWEN21_DRIVER_HAVE_VAE
      std::printf("[vae %s] validated%s\n", vae_weights.c_str(),
                  vae_decode ? "; decode enabled" : "; pass --vae-decode "
                               "to decode (else grayscale preview)");
#else
      std::printf("[vae %s] validated, but KokkosDNN_QwenImage21Vae.hpp is "
                  "not in this build; writing a grayscale preview instead\n",
                  vae_weights.c_str());
#endif
    } else if (vae_decode) {
      std::printf("[vae] --vae-decode needs --vae-weights; writing a "
                  "grayscale preview instead\n");
      vae_decode = false;
    }

    std::vector<float> x_init =
        QW::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    std::printf("[sample] H=%d W=%d steps=%d cfg=%.3g shift=%.3g seed=%llu\n",
                H, W, steps_eff, (double)guidance, (double)flow_shift,
                (unsigned long long)seed);
#ifdef QWEN21_DRIVER_HAVE_DIT
    KokkosDNN::DNNHandle<Exec> dit_handle;
    DitForward dit_forward{&dit_model, &dit_handle, H, W};
#else
    QW::Model<Exec, DScalar> model(cfg, w);
    DitForward dit_forward{&model};
#endif
    const auto t0 = std::chrono::steady_clock::now();
    // Sample straight from the device-resident embeds staged above: no host
    // reload, no disk re-read. The host vectors (enc.embeds/neg_embeds) stay
    // as the fallback for the --prompt-embeds-in file path, which also lands
    // staged on-device via EncoderStage::run.
    std::vector<float> lat = run_sample_loop_from_device(
        cfg, dit_forward, x_init, enc.device, enc.device_neg, steps_eff,
        guidance, flow_shift, H, W, [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        });
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("[done] %.2fs total, %.2fs/step\n", secs,
                secs / (double)std::max(steps_eff, 1));

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
      std::printf("[vae] --vae-decode requested but the landed "
                  "QwenImage21VaeModel::decode is still a skeleton (TODO); "
                  "writing a grayscale preview instead\n");
      // (ADAPT: run the real VAE decode here once the landed decode() is
      // implemented: allocate QwenImage21VaeWeights, load_safetensors, scale
      // latents by the runtime affine, QwenImage21VaeModel::decode to RGB.)
#else
      std::printf("[vae] --vae-decode requested but "
                  "KokkosDNN_QwenImage21Vae.hpp is not in this build; writing "
                  "a grayscale preview instead (need the VAE worktree "
                  "header + --vae-weights)\n");
#endif
    }
    if (!decoded) {
      pix = render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
      QW::print_phase_memory("decode-preview", 0, 0);
    }
#ifdef KOKKOSDNN_ENABLE_IMAGE
    {
      Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::HostSpace>
          img("qwen21_out", oH, oW, (size_t)oCh);
      for (size_t h = 0; h < oH; ++h)
        for (size_t ww = 0; ww < oW; ++ww)
          for (int c = 0; c < oCh; ++c)
            img(h, ww, (size_t)c) = pix[(h * oW + ww) * (size_t)oCh + (size_t)c];
      KokkosDNN::Image::save_image_png(out, img);
      std::printf("[wrote %s] %zux%zu %s (PNG)\n", out.c_str(), oH, oW,
                  decoded ? "VAE RGB" : "grayscale preview");
    }
#else
    {
      const std::string ppm = with_extension(out, ".ppm");
      if (ppm != out)
        std::printf("[note] built without KokkosDNN_ENABLE_IMAGE: writing PPM "
                    "to %s instead of PNG %s\n",
                    ppm.c_str(), out.c_str());
      if (decoded) {
        save_ppm_rgb(ppm, pix, oH, oW);
      } else {
        save_ppm(ppm, pix, oH, oW);
      }
      // oCh is only consumed by the PNG path above.
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
