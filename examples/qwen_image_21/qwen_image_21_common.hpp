// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 driver shared components (internal include, NOT installed).
//
// Included by both qwen_image_21_generate.cpp (text-to-image REPL/one-off)
// and qwen_image_21_edit.cpp (one-shot reference-image editing). Holds the
// backend shim (placeholder Qwen21:: + true-pipeline probing), pixel
// geometry, parity dumps, raw-text BPE, EncoderStage, the FlowEuler sample
// loop, DiT forward adapters, VAE decode, and the image writer, so the two
// binaries share one implementation. The interactive loop and main() stay in
// the generate TU; edit-specific helpers live in qwen_image_21_edit.cpp.
#ifndef KOKKOSDNN_QWEN_IMAGE_21_COMMON_HPP
#define KOKKOSDNN_QWEN_IMAGE_21_COMMON_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
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
#include <string_view>
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
#if __has_include("KokkosDNN_QwenImage21Vision.hpp")
#include "KokkosDNN_QwenImage21Vision.hpp"
#define QWEN21_DRIVER_HAVE_VISION 1
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

// ---- Pixel-space geometry -------------------------------------------------
// --height/--width are output-image pixels (default 1024). The DiT samples a
// latent grid at 1/16 of the pixels (VAE compression 16; the placeholder
// preview upscales by the same nominal 16x); each latent dim must tile the
// DiT patch (be_patch: 1 for the true pipeline, 2 for the placeholder), so
// the pixel stride is 16*patch and each request rounds to the nearest stride
// multiple (ties round up, minimum one stride). The latent grid is then
// pixels/16; the actual pixel dims (what the VAE/preview emits) are
// latent*16.
inline constexpr int kPixelScale = 16;

inline int snap_pixels_to_stride(int px, int stride) {
  if (px <= 0)
    throw std::invalid_argument("pixel dims must be positive (got " +
                                std::to_string(px) + ")");
  if (stride <= 0)
    throw std::invalid_argument("pixel stride must be positive (got " +
                                std::to_string(stride) + ")");
  long long r = std::lround((double)px / (double)stride);
  if (r < 1) r = 1;
  return (int)(r * (long long)stride);
}

struct PixelGeometry {
  int px_h = 0, px_w = 0;    // actual output pixels (what the image holds)
  int lat_h = 0, lat_w = 0;  // DiT latent grid (pixels/16, tiles be_patch)
};

inline PixelGeometry resolve_pixel_geometry(int req_h_px, int req_w_px,
                                            int patch) {
  if (patch <= 0)
    throw std::invalid_argument("patch_size must be > 0 (got " +
                                std::to_string(patch) + ")");
  PixelGeometry g;
  const int stride = kPixelScale * patch;
  g.px_h = snap_pixels_to_stride(req_h_px, stride);
  g.px_w = snap_pixels_to_stride(req_w_px, stride);
  g.lat_h = g.px_h / kPixelScale;
  g.lat_w = g.px_w / kPixelScale;
  return g;
}

void save_raw_f32(const std::string& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f.write(reinterpret_cast<const char*>(v.data()),
          (long long)(v.size() * sizeof(float)));
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

// Raw fp32 LE initial latents [C,H,W] row-major, flattened C-major as
// lat[(c*H+h)*W+w] (the driver's native layout: matches the sampler's
// x_init pack/unpack order).
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
  int height = 0, width = 0;  // actual output pixels (rounded, see --height)
  int latent_h = 0, latent_w = 0;  // DiT latent grid (pixels/16)
  int steps = 0;
  size_t latent_channels = 0;
  size_t prompt_len = 0, prompt_dim = 0;  // staged conditioning [L,Cc]
  unsigned long long seed = 0;
  double guidance = 1.0;
  bool use_cfg = false, staged = false;
  // Edit image-condition layout (empty/zero when sampling target-only):
  // img_shapes as the reference spells it, e.g. "[[(1,38,108),(1,38,108]]]".
  std::string img_shapes;
  size_t img_mask_len = 0, img_mask_true = 0;
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
  f << "  \"latent_h\": " << m.latent_h << ",\n";
  f << "  \"latent_w\": " << m.latent_w << ",\n";
  f << "  \"latent_shape\": [" << m.latent_channels << ", " << m.latent_h
    << ", " << m.latent_w << "],\n";
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
       "C-major lat[(c*H+h)*W+w]\",\n";
  if (!m.img_shapes.empty()) {
    f << "  \"img_shapes\": \"" << json_escape(m.img_shapes) << "\",\n";
    f << "  \"img_mask_len\": " << m.img_mask_len << ",\n";
    f << "  \"img_mask_true\": " << m.img_mask_true << ",\n";
  }
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

[[maybe_unused]] std::string with_extension(const std::string& path,
    const std::string& ext) {
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
// (sorted). Shared by the --ckpt transformer/text_encoder/vae subdirs. Only
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

// ---- Canonical checkpoint dir (--ckpt) ------------------------------------
// The drivers aim at one canonical Qwen-Image-2.1-Turbo checkpoint root
// (e.g. `hf download Qwen/Qwen-Image-2.1-Turbo --local-dir <ckpt>`) instead
// of one flag per component:
//
//   <ckpt>/transformer/  DiT shards
//   <ckpt>/text_encoder/ Qwen text encoder, single-shard model.safetensors
//   <ckpt>/processor/    dir holding tokenizer.json (+ configs)
//   <ckpt>/vae/          VAE shards + config.json
//
// Unguarded so both the encode and decode paths share it (and so the
// header-less placeholder build still reports the same layout).
inline std::string ckpt_sub(const std::string& ckpt, const char* sub) {
  std::string r = ckpt;
  while (r.size() > 1 && r.back() == '/') r.pop_back();
  return r + "/" + sub;
}

inline void require_ckpt_dir(const std::string& ckpt) {
  std::error_code ec;
  if (ckpt.empty() || !std::filesystem::is_directory(ckpt, ec))
    throw std::runtime_error(
        "--ckpt '" + ckpt +
        "' is not a directory (need the canonical Qwen-Image-2.1 "
        "checkpoint dir)");
}

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
//   1. In-process Qwen forward: --ckpt (canonical checkpoint dir), not
//      --tiny. Tokenizes the prompt (processor/tokenizer.json) and runs
//      the text_encoder forward on-device. Weights are released before
//      the DiT load.
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
                       const std::string& ckpt_dir) {
    namespace QE = KokkosDNN::QwenImage21Encoder;
    require_ckpt_dir(ckpt_dir);
    const std::string tok_json =
        ckpt_sub(ckpt_dir, "processor") + "/tokenizer.json";
    const std::string encoder_path = ckpt_sub(ckpt_dir, "text_encoder");
    if (!file_exists(tok_json))
      throw std::runtime_error("--ckpt '" + ckpt_dir +
                               "' has no processor/tokenizer.json (need the "
                               "canonical Qwen-Image-2.1 checkpoint dir)");
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
        resolve_shard_list(encoder_path, "--ckpt");
    if (shards.size() != 1)
      throw std::runtime_error(
          "--ckpt text_encoder must be the single-shard text_encoder/ "
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

#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER) && defined(QWEN21_DRIVER_HAVE_VISION)
  // Joint in-process encode for edit conditioning: vision tower forward on
  // the driver-packed pixel rows, then the text tower forward_joint over the
  // pre-tokenized edit sequence (image_pad slots at vis_rows, true M-RoPE
  // pos3). Drops the first `drop` rows (system block). The vision (~1.2GB)
  // and text (~17.5GB) weights are both released before returning; only the
  // [L,Cc] embeds survive (staged to device here).
  template <class CfgT>
  static void run_qwen_joint(EncoderStage& st, const CfgT& cfg,
                             const std::vector<int>& ids,
                             const std::vector<int>& vis_rows,
                             const std::vector<int>& pos3,
                             const std::vector<float>& pixels, int Hp, int Wp,
                             size_t drop, const std::string& ckpt_dir) {
    namespace QE = KokkosDNN::QwenImage21Encoder;
    namespace QV = KokkosDNN::QwenImage21Vision;
    require_ckpt_dir(ckpt_dir);
    const std::string encoder_path = ckpt_sub(ckpt_dir, "text_encoder");
    const size_t S = ids.size(), Nv = vis_rows.size();
    if (S == 0 || Nv == 0)
      throw std::runtime_error(
          "in-process joint encode: empty token/pixel sequence");
    if (pos3.size() != 3 * S)
      throw std::runtime_error(
          "in-process joint encode: pos3 must hold 3*S ints");
    if (drop > S)
      throw std::runtime_error("in-process joint encode: drop out of range");
    const size_t Cc = be_caption_dim(cfg);
    if (Hp <= 0 || Wp <= 0 || Hp % 2 != 0 || Wp % 2 != 0)
      throw std::runtime_error(
          "in-process joint encode: patch grid must tile the 2x2 merge");
    const size_t Np = (size_t)Hp * (size_t)Wp;
    if (Nv != Np / 4)
      throw std::runtime_error(
          "in-process joint encode: vision slots != patches/4");
    QV::Config vcfg;
    vcfg.validate();
    if ((size_t)vcfg.out_hidden_size != Cc)
      throw std::runtime_error(
          "in-process joint encode: vision out dim != caption dim");
    if (pixels.size() != Np * (size_t)vcfg.patch_dim())
      throw std::runtime_error(
          "in-process joint encode: pixel rows != Np*patch_dim");
    // Vision tower first (small): pixel rows -> LLM vision tokens +
    // deepstack taps.
    QV::QwenImage21VisionModel<Exec, Kokkos::Experimental::bhalf_t> vis(vcfg);
    {
      const std::vector<std::string> shards =
          resolve_shard_list(encoder_path, "--ckpt");
      if (shards.size() != 1)
        throw std::runtime_error(
            "--ckpt text_encoder must be the single-shard text_encoder/ "
            "model.safetensors (got " +
            std::to_string(shards.size()) + " shards)");
      std::printf("[loading %s]\n", encoder_path.c_str());
      vis.weights().load_safetensors(shards[0]);
      std::printf("[encode] vision weights resident: %.3f GB\n",
                  (double)vis.weights().resident_bytes() / 1e9);
    }
    using DScalar2 = Kokkos::Experimental::bhalf_t;
    Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec> d_pix(
        "qwen21edit_pix", Np, (size_t)vcfg.patch_dim());
    {
      auto h = Kokkos::create_mirror_view(d_pix);
      size_t k = 0;
      for (size_t p = 0; p < Np; ++p)
        for (int d = 0; d < vcfg.patch_dim(); ++d)
          h(p, (size_t)d) = (DScalar2)pixels[k++];
      Kokkos::deep_copy(d_pix, h);
    }
    Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec> d_vis(
        "qwen21edit_vis", Nv, Cc);
    std::vector<Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec>> d_deep;
    for (int k = 0; k < 3; ++k)
      d_deep.emplace_back("qwen21edit_deep", Nv, Cc);
    KokkosDNN::DNNHandle<Exec> vhandle;
    vis.forward(vhandle, d_pix, Hp, Wp, d_vis, d_deep);
    Kokkos::fence();
    std::printf("[encode] vision tokens %zu (grid %dx%d -> %zu patches), "
                "deepstack taps staged\n",
                Nv, Hp, Wp, Np);
    QE::Config ecfg;
    ecfg.validate();
    if ((size_t)ecfg.hidden_dim != Cc)
      throw std::runtime_error(
          "Qwen encoder hidden (" + std::to_string(ecfg.hidden_dim) +
          ") != caption dim (" + std::to_string(Cc) + ")");
    QE::QwenImage21EncoderModel<Exec, Kokkos::Experimental::bhalf_t> enc(ecfg);
    {
      const std::vector<std::string> shards =
          resolve_shard_list(encoder_path, "--ckpt");
      enc.weights().load_safetensors(shards[0]);
      std::printf("[encode] Qwen weights resident: %.1f GB\n",
                  (double)enc.weights().resident_bytes() / 1e9);
    }
    Kokkos::View<int**, Kokkos::LayoutRight, Exec> toks("qwen21encj_toks", 1,
                                                         S);
    Kokkos::View<int*, Exec> d_rows("qwen21encj_visrows", Nv);
    Kokkos::View<int*, Exec> d_pos3("qwen21encj_pos3", 3 * S);
    {
      auto ht = Kokkos::create_mirror_view(toks);
      for (size_t s = 0; s < S; ++s) ht(0, s) = ids[s];
      Kokkos::deep_copy(toks, ht);
      auto hr = Kokkos::create_mirror_view(d_rows);
      for (size_t k = 0; k < Nv; ++k) hr(k) = vis_rows[k];
      Kokkos::deep_copy(d_rows, hr);
      auto hp = Kokkos::create_mirror_view(d_pos3);
      for (size_t i = 0; i < 3 * S; ++i) hp(i) = pos3[i];
      Kokkos::deep_copy(d_pos3, hp);
    }
    Kokkos::View<Kokkos::Experimental::bhalf_t***, Kokkos::LayoutRight, Exec>
        hid("qwen21encj_hid", 1, S, (size_t)ecfg.hidden_dim);
    KokkosDNN::DNNHandle<Exec> handle;
    // forward_joint consumes rank-3 [1,Nv,Cc] views; the vision tower wrote
    // rank-2 rows, so stage them through [1,Nv,Cc] allocations (one subview
    // deep_copy each: MiB-scale, negligible next to the encoder pass).
    using JV3 = typename QE::QwenImage21EncoderModel<
        Exec, Kokkos::Experimental::bhalf_t>::V3;
    JV3 vis3("qwen21encj_vis3", 1, Nv, Cc);
    Kokkos::deep_copy(
        Kokkos::subview(vis3, 0, Kokkos::ALL, Kokkos::ALL), d_vis);
    JV3 deep3[3] = {JV3("qwen21encj_deep0", 1, Nv, Cc),
                    JV3("qwen21encj_deep1", 1, Nv, Cc),
                    JV3("qwen21encj_deep2", 1, Nv, Cc)};
    for (int k = 0; k < 3; ++k)
      Kokkos::deep_copy(
          Kokkos::subview(deep3[k], 0, Kokkos::ALL, Kokkos::ALL),
          d_deep[(size_t)k]);
    enc.forward_joint(handle, toks, d_rows, vis3, deep3, d_pos3, hid,
                      (int)drop);
    Kokkos::fence();
    {
      auto hhid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hid);
      const size_t L = S - drop;
      st.L = L;
      st.Cc = Cc;
      st.embeds.assign(L * Cc, 0.0f);
      st.neg_embeds.assign(L * Cc, 0.0f);
      st.source = "qwen-joint";
      for (size_t t = 0; t < L; ++t)
        for (size_t h = 0; h < Cc; ++h)
          st.embeds[t * Cc + h] = (float)hhid(0, drop + t, h);
    }
    const size_t wbytes =
        enc.weights().resident_bytes() + vis.weights().resident_bytes();
    enc.weights().release();
    vis.weights().release();
    Kokkos::fence();
    std::printf("[encode] released %.1f GB joint Qwen+vision weights; %.1f "
                "MiB embeds survive for the sample phase\n",
                (double)wbytes / 1e9,
                (double)(2 * st.L * st.Cc * sizeof(float)) / (1024.0 * 1024.0));
    st.stage_device();
    std::printf("[caption qwen-joint] %zux%zu features (edit template + 1 "
                "vision slot, drop_idx=%zu)\n",
                st.L, st.Cc, drop);
  }

  // Multi-image joint encode: one vision-tower forward per condition image
  // (the reference packs one image per call), concatenating the LLM vision
  // tokens and DeepStack taps in --image order, then a single text-tower
  // forward_joint over the multi-slot edit sequence. Hps/Wps/pixels_list
  // are parallel vectors (pixels_list[i] holds Np_i*patch_dim floats for
  // the Hps[i] x Wps[i] patch grid). Drops the first `drop` rows.
  template <class CfgT>
  static void run_qwen_joint_multi(
      EncoderStage& st, const CfgT& cfg, const std::vector<int>& ids,
      const std::vector<int>& vis_rows, const std::vector<int>& pos3,
      const std::vector<std::vector<float>>& pixels_list,
      const std::vector<int>& Hps, const std::vector<int>& Wps, size_t drop,
      const std::string& ckpt_dir) {
    namespace QE = KokkosDNN::QwenImage21Encoder;
    namespace QV = KokkosDNN::QwenImage21Vision;
    require_ckpt_dir(ckpt_dir);
    const std::string encoder_path = ckpt_sub(ckpt_dir, "text_encoder");
    const size_t S = ids.size(), Nv = vis_rows.size();
    const size_t n_imgs = pixels_list.size();
    if (S == 0 || Nv == 0 || n_imgs == 0)
      throw std::runtime_error(
          "in-process joint encode: empty token/pixel sequence");
    if (Hps.size() != n_imgs || Wps.size() != n_imgs)
      throw std::runtime_error(
          "in-process joint encode: grid count != image count");
    if (pos3.size() != 3 * S)
      throw std::runtime_error(
          "in-process joint encode: pos3 must hold 3*S ints");
    if (drop > S)
      throw std::runtime_error("in-process joint encode: drop out of range");
    const size_t Cc = be_caption_dim(cfg);
    QV::Config vcfg;
    vcfg.validate();
    if ((size_t)vcfg.out_hidden_size != Cc)
      throw std::runtime_error(
          "in-process joint encode: vision out dim != caption dim");
    size_t Nv_expect = 0;
    for (size_t i = 0; i < n_imgs; ++i) {
      const int Hp = Hps[i], Wp = Wps[i];
      if (Hp <= 0 || Wp <= 0 || Hp % 2 != 0 || Wp % 2 != 0)
        throw std::runtime_error(
            "in-process joint encode: patch grid must tile the 2x2 merge");
      const size_t Np = (size_t)Hp * (size_t)Wp;
      Nv_expect += Np / 4;
      if (pixels_list[i].size() != Np * (size_t)vcfg.patch_dim())
        throw std::runtime_error(
            "in-process joint encode: pixel rows != Np*patch_dim");
    }
    if (Nv != Nv_expect)
      throw std::runtime_error(
          "in-process joint encode: vision slots != sum(patches/4)");
    QV::QwenImage21VisionModel<Exec, Kokkos::Experimental::bhalf_t> vis(vcfg);
    {
      const std::vector<std::string> shards =
          resolve_shard_list(encoder_path, "--ckpt");
      if (shards.size() != 1)
        throw std::runtime_error(
            "--ckpt text_encoder must be the single-shard text_encoder/ "
            "model.safetensors (got " +
            std::to_string(shards.size()) + " shards)");
      std::printf("[loading %s]\n", encoder_path.c_str());
      vis.weights().load_safetensors(shards[0]);
      std::printf("[encode] vision weights resident: %.3f GB\n",
                  (double)vis.weights().resident_bytes() / 1e9);
    }
    using DScalar2 = Kokkos::Experimental::bhalf_t;
    std::vector<float> host_vis;
    std::vector<float> host_deep[3];
    host_vis.reserve(Nv * Cc);
    for (int k = 0; k < 3; ++k) host_deep[k].reserve(Nv * Cc);
    KokkosDNN::DNNHandle<Exec> vhandle;
    for (size_t i = 0; i < n_imgs; ++i) {
      const int Hp = Hps[i], Wp = Wps[i];
      const size_t Np = (size_t)Hp * (size_t)Wp;
      const size_t Nv_i = Np / 4;
      Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec> d_pix(
          "qwen21edit_pix", Np, (size_t)vcfg.patch_dim());
      {
        auto h = Kokkos::create_mirror_view(d_pix);
        size_t kk = 0;
        const std::vector<float>& px = pixels_list[i];
        for (size_t p = 0; p < Np; ++p)
          for (int d = 0; d < vcfg.patch_dim(); ++d)
            h(p, (size_t)d) = (DScalar2)px[kk++];
        Kokkos::deep_copy(d_pix, h);
      }
      Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec> d_vis(
          "qwen21edit_vis", Nv_i, Cc);
      std::vector<Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec>> d_deep;
      for (int k = 0; k < 3; ++k)
        d_deep.emplace_back("qwen21edit_deep", Nv_i, Cc);
      vis.forward(vhandle, d_pix, Hp, Wp, d_vis, d_deep);
      Kokkos::fence();
      auto h_vis =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_vis);
      for (size_t r = 0; r < Nv_i; ++r)
        for (size_t c = 0; c < Cc; ++c)
          host_vis.push_back((float)h_vis(r, c));
      for (int k = 0; k < 3; ++k) {
        auto h_deep = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           d_deep[(size_t)k]);
        for (size_t r = 0; r < Nv_i; ++r)
          for (size_t c = 0; c < Cc; ++c)
            host_deep[k].push_back((float)h_deep(r, c));
      }
      std::printf("[encode] vision image %zu/%zu tokens %zu (grid %dx%d -> "
                  "%zu patches)\n",
                  i + 1, n_imgs, Nv_i, Hp, Wp, Np);
    }
    Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec> d_vis_all(
        "qwen21edit_vis_all", Nv, Cc);
    std::vector<Kokkos::View<DScalar2**, Kokkos::LayoutRight, Exec>> d_deep_all;
    for (int k = 0; k < 3; ++k)
      d_deep_all.emplace_back("qwen21edit_deep_all", Nv, Cc);
    {
      auto h = Kokkos::create_mirror_view(d_vis_all);
      size_t kk = 0;
      for (size_t r = 0; r < Nv; ++r)
        for (size_t c = 0; c < Cc; ++c) h(r, c) = (DScalar2)host_vis[kk++];
      Kokkos::deep_copy(d_vis_all, h);
      for (int k = 0; k < 3; ++k) {
        auto hd = Kokkos::create_mirror_view(d_deep_all[(size_t)k]);
        size_t kk2 = 0;
        for (size_t r = 0; r < Nv; ++r)
          for (size_t c = 0; c < Cc; ++c)
            hd(r, c) = (DScalar2)host_deep[k][kk2++];
        Kokkos::deep_copy(d_deep_all[(size_t)k], hd);
      }
    }
    QE::Config ecfg;
    ecfg.validate();
    if ((size_t)ecfg.hidden_dim != Cc)
      throw std::runtime_error(
          "Qwen encoder hidden (" + std::to_string(ecfg.hidden_dim) +
          ") != caption dim (" + std::to_string(Cc) + ")");
    QE::QwenImage21EncoderModel<Exec, Kokkos::Experimental::bhalf_t> enc(ecfg);
    {
      const std::vector<std::string> shards =
          resolve_shard_list(encoder_path, "--ckpt");
      enc.weights().load_safetensors(shards[0]);
      std::printf("[encode] Qwen weights resident: %.1f GB\n",
                  (double)enc.weights().resident_bytes() / 1e9);
    }
    Kokkos::View<int**, Kokkos::LayoutRight, Exec> toks("qwen21encj_toks", 1,
                                                         S);
    Kokkos::View<int*, Exec> d_rows("qwen21encj_visrows", Nv);
    Kokkos::View<int*, Exec> d_pos3("qwen21encj_pos3", 3 * S);
    {
      auto ht = Kokkos::create_mirror_view(toks);
      for (size_t s = 0; s < S; ++s) ht(0, s) = ids[s];
      Kokkos::deep_copy(toks, ht);
      auto hr = Kokkos::create_mirror_view(d_rows);
      for (size_t k = 0; k < Nv; ++k) hr(k) = vis_rows[k];
      Kokkos::deep_copy(d_rows, hr);
      auto hp = Kokkos::create_mirror_view(d_pos3);
      for (size_t i = 0; i < 3 * S; ++i) hp(i) = pos3[i];
      Kokkos::deep_copy(d_pos3, hp);
    }
    Kokkos::View<Kokkos::Experimental::bhalf_t***, Kokkos::LayoutRight, Exec>
        hid("qwen21encj_hid", 1, S, (size_t)ecfg.hidden_dim);
    KokkosDNN::DNNHandle<Exec> handle;
    using JV3 = typename QE::QwenImage21EncoderModel<
        Exec, Kokkos::Experimental::bhalf_t>::V3;
    JV3 vis3("qwen21encj_vis3", 1, Nv, Cc);
    Kokkos::deep_copy(
        Kokkos::subview(vis3, 0, Kokkos::ALL, Kokkos::ALL), d_vis_all);
    JV3 deep3[3] = {JV3("qwen21encj_deep0", 1, Nv, Cc),
                    JV3("qwen21encj_deep1", 1, Nv, Cc),
                    JV3("qwen21encj_deep2", 1, Nv, Cc)};
    for (int k = 0; k < 3; ++k)
      Kokkos::deep_copy(
          Kokkos::subview(deep3[k], 0, Kokkos::ALL, Kokkos::ALL),
          d_deep_all[(size_t)k]);
    enc.forward_joint(handle, toks, d_rows, vis3, deep3, d_pos3, hid,
                      (int)drop);
    Kokkos::fence();
    {
      auto hhid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hid);
      const size_t L = S - drop;
      st.L = L;
      st.Cc = Cc;
      st.embeds.assign(L * Cc, 0.0f);
      st.neg_embeds.assign(L * Cc, 0.0f);
      st.source = "qwen-joint";
      for (size_t t = 0; t < L; ++t)
        for (size_t h = 0; h < Cc; ++h)
          st.embeds[t * Cc + h] = (float)hhid(0, drop + t, h);
    }
    const size_t wbytes =
        enc.weights().resident_bytes() + vis.weights().resident_bytes();
    enc.weights().release();
    vis.weights().release();
    Kokkos::fence();
    std::printf("[encode] released %.1f GB joint Qwen+vision weights; %.1f "
                "MiB embeds survive for the sample phase\n",
                (double)wbytes / 1e9,
                (double)(2 * st.L * st.Cc * sizeof(float)) / (1024.0 * 1024.0));
    st.stage_device();
    std::printf("[caption qwen-joint] %zux%zu features (edit template + %zu "
                "vision slots across %zu images, drop_idx=%zu)\n",
                st.L, st.Cc, Nv, n_imgs, drop);
  }
#else
  // Joint encode needs the encoder + tokenizer + vision headers; the edit
  // driver falls back to file/null with a logged reason when absent.
  template <class CfgT>
  static void run_qwen_joint(EncoderStage&, const CfgT&,
                             const std::vector<int>&, const std::vector<int>&,
                             const std::vector<int>&, const std::vector<float>&,
                             int, int, size_t, const std::string&) {
    throw std::runtime_error(
        "in-process joint encode needs KokkosDNN_QwenImage21Encoder.hpp + "
        "KokkosDNN_QwenImage21Vision.hpp in this build");
  }
  template <class CfgT>
  static void run_qwen_joint_multi(
      EncoderStage&, const CfgT&, const std::vector<int>&,
      const std::vector<int>&, const std::vector<int>&,
      const std::vector<std::vector<float>>&, const std::vector<int>&,
      const std::vector<int>&, size_t, const std::string&) {
    throw std::runtime_error(
        "in-process joint encode needs KokkosDNN_QwenImage21Encoder.hpp + "
        "KokkosDNN_QwenImage21Vision.hpp in this build");
  }
#endif
  // (vision tower + text tower joint forward; stub above throws when the
  // headers are absent)

  template <class CfgT>
  static EncoderStage run(const CfgT& cfg, const std::string& prompt,
                          const std::string& caption_path,
                          const std::string& ckpt_dir, bool tiny,
                          size_t max_seq_override = 0) {
    EncoderStage st;
    st.L = be_max_seq(cfg);
    st.Cc = be_caption_dim(cfg);
#if defined(QWEN21_DRIVER_HAVE_ENCODER) && \
    defined(QWEN21_DRIVER_HAVE_TOKENIZER)
    // In-process Qwen forward when --ckpt is given (and not --tiny: the
    // smoke model uses random weights at test geometry, which real token
    // ids cannot address). A throw inside the attempt falls back to
    // file/null with a logged reason instead of failing.
    if (!tiny && !ckpt_dir.empty()) {
      try {
        run_qwen(st, cfg, prompt, ckpt_dir);
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
    if (tiny && !ckpt_dir.empty())
      std::printf("[encode] --tiny skips the Qwen forward (random-weights "
                  "smoke model); falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#elif defined(QWEN21_DRIVER_HAVE_ENCODER)
    // The encoder header is present but the BPE tables header is not, so
    // the in-process forward cannot tokenize: fall through to file/null.
    if (!tiny && !ckpt_dir.empty())
      std::printf("[encode] --ckpt needs "
                  "KokkosDNN_QwenTokenizer.hpp, not present in this "
                  "build; falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
    if (tiny && !ckpt_dir.empty())
      std::printf("[encode] --tiny skips the Qwen forward (random-weights "
                  "smoke model); falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#else
    if (!ckpt_dir.empty())
      std::printf("[encode] --ckpt needs "
                  "KokkosDNN_QwenImage21Encoder.hpp, not present in this "
                  "build; falling back to %s\n",
                  caption_path.empty() ? "null" : "file");
#endif
    if (!caption_path.empty()) {
      try {
        // Infer L from the file: golden prompt-embed dumps are [19,4096],
        // not the backend's 512-row maximum. Validate against max_seq so a
        // full-geometry file cannot overflow the tiny smoke model. Edit
        // golden dumps carry thousands of joint text+vision rows, so the
        // edit driver passes a wider bound via max_seq_override.
        const size_t max_seq =
            (max_seq_override > 0) ? max_seq_override : be_max_seq(cfg);
        st.embeds = QW::load_caption_raw(caption_path, 0, (int)st.Cc);
        const size_t L = st.embeds.size() / st.Cc;
        if (L == 0 || L > max_seq)
          throw std::runtime_error(
              "caption file has " + std::to_string(L) +
              " rows, max_seq is " + std::to_string(max_seq));
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
    const StepDumpHook& step_dump = {},
    size_t max_seq_override = 0) {
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
  // The T2I bound (backend max_seq) does not fit edit conditioning, whose
  // joint sequences carry thousands of vision tokens; the edit driver
  // passes its wider bound (validated at encode time).
  const size_t max_seq =
      (max_seq_override > 0) ? max_seq_override : be_max_seq(cfg);
  if (L == 0 || L > max_seq)
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
  // Edit image-condition state (unallocated when sampling target-only):
  // cond_tok is the staged reference latents [1,Nc,C] in token order
  // (row t = h*in_lat_w + w, matching the pack order below); img_blocks
  // are the (frame, h, w) latent grids (cond first, target last, shared
  // row-0 layout); img_mask is the shared [V] row with V = S + N/4.
  Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> cond_tok;
  std::vector<std::array<int, 3>> img_blocks;
  std::vector<char> img_mask;
  DitForward() = default;
  DitForward(decltype(model) m, decltype(handle) h, int hh, int ww)
      : model(m), handle(h), H(hh), W(ww) {}
  bool has_cond() const { return cond_tok.is_allocated(); }
  template <class X, class T, class Y, class Out>
  void operator()(const X& x, const T& t, const Y& y, Out& out) const {
    const size_t C = x.extent(1);
    const size_t HH = (size_t)H, WW = (size_t)W;
    const size_t N = HH * WW;
    const size_t L = y.extent(1), Cc = y.extent(2);  // y is [1,L,Cc]
    Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> xtok("qwen21_fwd_x",
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
    if (has_cond()) {
      // Edit path: packed cond tokens + [cond|target] joint sequence with
      // per-block RoPE and block-causal segment attention; the model
      // slices the target rows internally, so vtok is target-only already.
      if (img_mask.size() != L + N / 4)
        throw std::runtime_error(
            "DitForward: img_mask length must equal S + N/4");
      Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> vtok(
          "qwen21_fwd_vc", 1, N, C);
      model->forward_cond(*handle, xtok, cond_tok, ytok, (double)ts / 1000.0,
                          vtok, img_blocks, img_mask, {}, HH, WW);
      Kokkos::parallel_for(
          "qwen21_fwd_unpack_c",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>({0, 0, 0}, {HH, WW, C}),
          KOKKOS_LAMBDA(size_t h, size_t w, size_t c) {
            out(0, c, h, w) = static_cast<DScalar>(vtok(0, h * WW + w, c));
          });
      return;
    }
    Kokkos::View<DScalar***, Kokkos::LayoutRight, Exec> vtok("qwen21_fwd_v",
                                                             1, N, C);
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
// Real VAE encode: pipeline-preprocessed RGBA image (host CHW in [-1,1],
// same pixels the reference preprocess() feeds vae_images) -> normalized
// ref latents (host chw [C,Hc,Wc], the (mode-mean)/std tensor the edit
// pipeline carries). `ckpt_dir` is the canonical checkpoint root (the VAE
// lives in <ckpt>/vae/). Returns the latents (C = z_dim, Hc = H/16);
// throws with a reason so the caller can fall back to file/null.
inline std::vector<float> try_vae_encode_image(
    const std::vector<float>& rgba_chw, size_t H, size_t W,
    const std::string& ckpt_dir) {
  namespace QV = KokkosDNN::QwenImage21Vae;
  require_ckpt_dir(ckpt_dir);
  const std::string vae_dir = ckpt_sub(ckpt_dir, "vae");
  QV::Config vcfg;
  {
    std::error_code ec;
    if (!std::filesystem::is_directory(vae_dir, ec))
      throw std::runtime_error("--ckpt '" + ckpt_dir +
                               "' has no vae/ subdir (need the canonical "
                               "Qwen-Image-2.1 checkpoint dir)");
    QV::qwen21_load_vae_config_json(vae_dir + "/config.json", vcfg);
  }
  vcfg.validate();
  if (rgba_chw.size() != (size_t)vcfg.in_channels * H * W)
    throw std::runtime_error(
        "VAE encode: RGBA pixels hold " + std::to_string(rgba_chw.size()) +
        " floats, expected " + std::to_string(vcfg.in_channels) + "x" +
        std::to_string(H) + "x" + std::to_string(W));
  QV::QwenImage21VaeModel<Exec, DScalar> enc(vcfg);
  {
    const std::vector<std::string> shards =
        resolve_shard_list(vae_dir, "--ckpt");
    std::printf("[loading %s]%s\n", vae_dir.c_str(),
                shards.size() > 1 ? " (shards)" : "");
    enc.weights().load_safetensors(shards);
    std::printf("[vae-encode] weights resident: %.3f GB\n",
                (double)enc.weights().resident_bytes() / 1e9);
  }
  using V4 = Kokkos::View<DScalar****, Kokkos::LayoutRight, Exec>;
  V4 img("qwen21vae_enc_img", 1, (size_t)vcfg.in_channels, H, W);
  {
    auto hm = Kokkos::create_mirror_view(img);
    size_t k = 0;
    for (int c = 0; c < vcfg.in_channels; ++c)
      for (size_t h = 0; h < H; ++h)
        for (size_t w = 0; w < W; ++w) hm(0, (size_t)c, h, w) = (DScalar)rgba_chw[k++];
    Kokkos::deep_copy(img, hm);
  }
  KokkosDNN::DNNHandle<Exec> handle;
  V4 lat = enc.encode(handle, img);
  Kokkos::fence();
  auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lat);
  const size_t Z = (size_t)vcfg.z_dim, Hc = H / (size_t)vcfg.compression,
               Wc = W / (size_t)vcfg.compression;
  std::vector<float> out(Z * Hc * Wc);
  size_t k = 0;
  for (size_t c = 0; c < Z; ++c)
    for (size_t h = 0; h < Hc; ++h)
      for (size_t w = 0; w < Wc; ++w)
        out[k++] = static_cast<float>(hl(0, c, h, w));
  enc.weights().release();
  Kokkos::fence();
  return out;
}

// Real VAE decode: raw DiT latents [C,H,W] host float -> device [1,C,H,W]
// -> pipeline denormalize (x*std+mean per channel, runtime affine from
// <ckpt>/vae/config.json) -> QwenImage21VaeModel::decode -> RGBA [oH,oW,4]
// u8 (x*0.5+0.5 by the pipeline, PNG alpha = ch 3). `ckpt_dir` is the
// canonical checkpoint root. Returns true on success (pix/oH/oW/oCh set,
// decoded); throws with a reason otherwise so the caller can fall back to
// the grayscale preview.
inline bool try_vae_decode_rgba(const std::vector<float>& lat, size_t C,
                                size_t H, size_t W,
                                const std::string& ckpt_dir,
                                std::vector<unsigned char>& pix, size_t& oH,
                                size_t& oW, int& oCh,
                                std::vector<float>* rgba_f32 = nullptr) {
  namespace QV = KokkosDNN::QwenImage21Vae;
  require_ckpt_dir(ckpt_dir);
  const std::string vae_dir = ckpt_sub(ckpt_dir, "vae");
  QV::Config vcfg;
  {
    std::error_code ec;
    if (!std::filesystem::is_directory(vae_dir, ec))
      throw std::runtime_error("--ckpt '" + ckpt_dir +
                               "' has no vae/ subdir (need the canonical "
                               "Qwen-Image-2.1 checkpoint dir)");
    QV::qwen21_load_vae_config_json(vae_dir + "/config.json", vcfg);
  }
  vcfg.validate();
  std::printf("[vae affine %zuch] mean[0]=%.4g std[0]=%.4g (runtime "
              "vae/config.json)\n",
              (size_t)vcfg.z_dim, (double)vcfg.latents_mean[0],
              (double)vcfg.latents_std[0]);
  QV::QwenImage21VaeModel<Exec, DScalar> dec(vcfg);
  {
    const std::vector<std::string> shards =
        resolve_shard_list(vae_dir, "--ckpt");
    std::printf("[loading %s]%s\n", vae_dir.c_str(),
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
// sampler, so a VAE change can be checked in seconds. `ckpt_dir` is the
// canonical checkpoint root (the VAE lives in <ckpt>/vae/).
inline void run_vae_only(const std::string& lat_path, bool token_major,
                         size_t C, size_t H, size_t W,
                         const std::string& ckpt_dir, const std::string& out,
                         const std::string& f32_out) {
#ifdef QWEN21_DRIVER_HAVE_VAE
  std::vector<float> lat = load_decode_latents(lat_path, C, H, W, token_major);
  std::printf("[decode-latents %s] %zux%zux%zu floats (%s layout)\n",
              lat_path.c_str(), C, H, W, token_major ? "token" : "chw");
  std::vector<float> rgba_f32;
  size_t oH = 0, oW = 0;
  int oCh = 1;
  std::vector<unsigned char> pix;
  const bool ok = try_vae_decode_rgba(lat, C, H, W, ckpt_dir, pix, oH, oW,
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
  (void)ckpt_dir;
  (void)out;
  (void)f32_out;
  throw std::runtime_error(
      "--decode-latents-in needs KokkosDNN_QwenImage21Vae.hpp in this build");
#endif
}


// Closes the anonymous driver-internals namespace opened above (backend
// geometry traits, EncoderStage, sampler, DiT adapters, VAE decode, image
// writer): every including TU sees balanced, self-contained internals, and
// main() stays at global scope.
}  // namespace

#endif  // KOKKOSDNN_QWEN_IMAGE_21_COMMON_HPP
