#ifndef KOKKOSDNN_MOSSIMAGEDIT_HPP
#define KOKKOSDNN_MOSSIMAGEDIT_HPP

// TRUE MOSS-Image flow transformer (MOSSImageTransformer) + weights loader +
// flow-grid scheduler, on top of src/KokkosDNN_MossImage.hpp ops.
//
// Ground truth (retrieved, not guessed): OpenMOSS-Team/MOSS-Image-SFT
// transformer/config.json (dim=3072, ffn_dim=8192, num_heads=24,
// num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128,
// rope_base=10000.0), scheduler/scheduler_config.json (shift=5.0,
// training_shift=1.0, 1000 train timesteps), pipeline_moss_image.py
// (MOSSImageTransformer / _Block / _TextBlock / _TextProjection / _RoPE /
// MOSSImageScheduler::set_timesteps), and the in-tree sections 2/3/6 notes in
// examples/moss_image/MOSS_ARCH_INVESTIGATION.md. The checkpoint holds 463
// tensors: video_embedder.{weight,bias}, text_embedder.proj.{weight,bias},
// text_embedder.blocks.0..1.{qkv,proj,ffn_gate,ffn_up,ffn_down}.{weight,bias}
// + {norm1,norm2,q_norm,k_norm}.weight, time_embedding.{0,2}.{weight,bias},
// shared_block_modulation.{weight,bias}, video_head_modulation.{weight,bias},
// layers.0..41.{norm1,norm2,q_norm,k_norm}.weight +
// proj.{q,k,v,o,ffn_up_gate,ffn_down}.weight (no bias),
// final_norm_video.weight, final_linear_video.weight (no bias), rope.bands
// (derived fp32 constant, skipped by the loader).
//
// Model structure (verified from the pipeline source):
//   video path: [B,128,H,W] latents -> flatten/transpose -> video_embedder.
//   text path: Linear 4096->3072, then 2x _TextBlock (RMSNorm -> fused qkv
//     GEMM + bias -> per-head QK RMSNorm -> joint SDPA with text-slice RoPE
//     -> proj + residual -> RMSNorm -> SiTU FFN (separate gate/up) +
//     residual). Padding is truncated per sample via the boolean mask.
//   time path: float64 sinusoid (freq = 10000^(-arange(half)/half),
//     cos|sin) -> float32 -> SiLU MLP -> shared_block_modulation [B,6D] +
//     video_head_modulation [B,2D].
//   per sample: x = concat(video[HW], text[S_valid]); 42x _Block (adaLN
//     6-lane + gated residuals, split q/k/v/o GEMMs without bias,
//     ffn_up_gate/ffn_down without bias, joint image+text self-attention
//     with 4D RoPE; skip_layers blocks are exact identity); head: video
//     rows only -> final RMSNorm -> head adaLN -> final Linear (no bias)
//     -> [B,128,H,W].
//
// Dispatch contract (no hand-written cublas/cudnn calls anywhere here):
//   * Every Linear (video/text embedders, time MLP, modulations, q/k/v/o,
//     qkv, ffn, final) goes through KokkosDNN::gemm / gemm_bias, which own
//     the cuBLAS/hipBLAS ladder. On CUDA+CUBLAS / HIP+HIPBLAS builds the
//     handle enables TPL dispatch automatically; elsewhere (or on TPL
//     failure) the portable native kernels run.
//   * Joint attention goes through attention_mha_bidirectional, which owns
//     the cuDNN ladder (float/bhalf_t on CUDA+CUDNN, native streaming
//     FlashAttention-style kernel otherwise).
//   * Norms / modulations / residuals / RoPE / SiTU / SiLU go through the
//     core entry points (rmsnorm, adaln_modulate_forward, silu_forward,
//     moss_* ops). Same FP32-accumulate/cast-back convention throughout.
//   * The only device code added here (Impl::dit_* helpers) is layout-only
//     data movement with no arithmetic: BCHW<->row transposes, mask gather,
//     column splits. Rank reshuffles reuse Kokkos::deep_copy on flat
//     unmanaged aliases (identical row-major element order), and strided
//     row-range copies reuse cross-layout Kokkos::deep_copy: never a new
//     math kernel. Same rationale as the native elementwise ops: no TPL
//     expresses these, so native == dispatched by construction.
//
// Precision: time sinusoid phases are computed in float64 on the host
// (exactly the reference: double arange/half exponent, double times),
// cast to float32 for the MLP, matching the reference embedding bit path
// before the learned MLP. RoPE stays FP32 (moss_rope_build_forward).
// Activations/weights are Scalar (float or bhalf_t); latents/encoder inputs
// arrive as Scalar (callers cast fp32 pipeline tensors first, mirroring the
// reference .to(weight.dtype) casts).
//
// All ops are templated <ExecSpace, Scalar>. Internals live in
// KokkosDNN::MossImageDit::Impl. No backend #ifdefs in native paths.

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN_Activation.hpp"
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Diffusion.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_MossImage.hpp"
#include "KokkosDNN_RMSNorm.hpp"

namespace KokkosDNN {
namespace MossImageDit {

// Model/sampler constants: reuse the grounded MossImage::Config verbatim
// (dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096,
// freq_dim=256, latent_channels=128, rope_base=10000, shift=5.0,
// guidance 3.0, skip {22}; plus tiny_test() for unit tests).
using Config = MossImage::Config;

// Host-side flow-grid scheduler: MOSSImageScheduler::set_timesteps, pure C++.
// grid(count, s) places `count` linspace points in [1, s/(total+s-1)] then
// maps v -> s*v/(1+(s-1)*v); the inference grid (shift) is snapped to the
// nearest training-grid (training_shift) points (lower_bound searchsorted,
// ties round down, matching torch.searchsorted + the <= in torch.where),
// then unique_consecutive. Returns the snapped sigmas, descending.
// Throws for steps < 1, non-positive/non-finite shifts, total < 1.
inline std::vector<double> moss_flow_grid_timesteps(
    int num_inference_steps, double shift = 5.0, double training_shift = 1.0,
    int num_train_timesteps = 1000) {
  if (num_inference_steps < 1)
    throw std::invalid_argument(
        "moss_flow_grid_timesteps: num_inference_steps must be >= 1");
  if (!(shift > 0.0) || !std::isfinite(shift))
    throw std::invalid_argument("moss_flow_grid_timesteps: shift must be > 0");
  if (!(training_shift > 0.0) || !std::isfinite(training_shift))
    throw std::invalid_argument(
        "moss_flow_grid_timesteps: training_shift must be > 0");
  if (num_train_timesteps < 1)
    throw std::invalid_argument(
        "moss_flow_grid_timesteps: num_train_timesteps must be >= 1");
  const int total = num_train_timesteps;
  auto grid = [&](int count, double s) {
    const double lower = s / (total + s - 1.0);
    std::vector<double> g((size_t)count);
    for (int i = 0; i < count; ++i) {
      const double v = (count == 1)
                           ? 1.0
                           : 1.0 + (lower - 1.0) * (double)i / (count - 1);
      g[(size_t)i] = s * v / (1.0 + (s - 1.0) * v);
    }
    return g;
  };
  const std::vector<double> sigmas = grid(num_inference_steps, shift);
  std::vector<double> training = grid(total, training_shift);
  std::reverse(training.begin(), training.end());  // ascending, like .flip(0)
  std::vector<double> snapped;
  snapped.reserve(sigmas.size());
  for (double sg : sigmas) {
    int u = (int)(std::lower_bound(training.begin(), training.end(), sg) -
                  training.begin());
    if (u > total - 1) u = total - 1;
    const int l = (u > 0) ? u - 1 : 0;
    snapped.push_back((std::fabs(sg - training[(size_t)l]) <=
                       std::fabs(training[(size_t)u] - sg))
                          ? training[(size_t)l]
                          : training[(size_t)u]);
  }
  std::vector<double> out;
  for (double v : snapped)
    if (out.empty() || v != out.back()) out.push_back(v);  // unique_consecutive
  return out;
}

// One expected checkpoint tensor: bare (suffix-matched) name + shape.
// Matrices are torch [out, in] row-major, matching the [N, K] GEMM operand
// layout, so file bytes land verbatim (no host transpose).
struct DitTensorSpec {
  std::string name;
  int64_t d0 = 0;  // rows (matrices) or length (vectors)
  int64_t d1 = 1;  // cols (matrices) or 1 (vectors)
  bool matrix = false;
};

// Canonical 462-weight tensor list for a Config (rope.bands excluded: it is
// a derived fp32 constant rebuilt by _RoPE, not a weight). Order: embedders,
// text refiner, time path, modulations, per-layer blocks, head.
inline std::vector<DitTensorSpec> moss_dit_expected_tensors(const Config& cfg) {
  cfg.validate();
  const int64_t D = cfg.dim, F = cfg.ffn_dim, Td = cfg.text_dim,
                Fr = cfg.freq_dim, C = cfg.latent_channels;
  const int64_t Hd = D / cfg.num_heads;
  std::vector<DitTensorSpec> ts;
  auto mat = [&](const std::string& n, int64_t r, int64_t c) {
    ts.push_back({n, r, c, true});
  };
  auto vec = [&](const std::string& n, int64_t r) {
    ts.push_back({n, r, 1, false});
  };
  mat("video_embedder.weight", D, C);
  vec("video_embedder.bias", D);
  mat("text_embedder.proj.weight", D, Td);
  vec("text_embedder.proj.bias", D);
  for (int i = 0; i < 2; ++i) {
    const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
    mat(p + "qkv.weight", 3 * D, D);
    vec(p + "qkv.bias", 3 * D);
    mat(p + "proj.weight", D, D);
    vec(p + "proj.bias", D);
    mat(p + "ffn_gate.weight", F, D);
    vec(p + "ffn_gate.bias", F);
    mat(p + "ffn_up.weight", F, D);
    vec(p + "ffn_up.bias", F);
    mat(p + "ffn_down.weight", D, F);
    vec(p + "ffn_down.bias", D);
    vec(p + "norm1.weight", D);
    vec(p + "norm2.weight", D);
    vec(p + "q_norm.weight", Hd);
    vec(p + "k_norm.weight", Hd);
  }
  mat("time_embedding.0.weight", D, Fr);
  vec("time_embedding.0.bias", D);
  mat("time_embedding.2.weight", D, D);
  vec("time_embedding.2.bias", D);
  mat("shared_block_modulation.weight", 6 * D, D);
  vec("shared_block_modulation.bias", 6 * D);
  mat("video_head_modulation.weight", 2 * D, D);
  vec("video_head_modulation.bias", 2 * D);
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    vec(p + "norm1.weight", D);
    vec(p + "norm2.weight", D);
    vec(p + "q_norm.weight", Hd);
    vec(p + "k_norm.weight", Hd);
    mat(p + "proj.q.weight", D, D);
    mat(p + "proj.k.weight", D, D);
    mat(p + "proj.v.weight", D, D);
    mat(p + "proj.o.weight", D, D);
    mat(p + "proj.ffn_up_gate.weight", 2 * F, D);
    mat(p + "proj.ffn_down.weight", D, F);
  }
  vec("final_norm_video.weight", D);
  mat("final_linear_video.weight", C, D);
  return ts;
}

namespace Impl {

// ---- layout-only device glue (no arithmetic; exact copies/splits) ----

template <class ExecSpace, class Scalar>
void dit_transpose_bchw_to_rows(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& src,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst,
    const std::string& label = "KokkosDNN::dit_transpose_in") {
  const size_t B = src.extent(0), C = src.extent(1), H = src.extent(2),
               W = src.extent(3);
  if (dst.extent(0) != B * H * W || dst.extent(1) != C)
    throw std::invalid_argument("dit_transpose_bchw_to_rows: dim mismatch");
  if (B == 0 || C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                       {B, C, H, W}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
        dst(b * H * W + h * W + w, c) = src(b, c, h, w);
      });
}

template <class ExecSpace, class Scalar>
void dit_transpose_rows_to_chw(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src, size_t H,
    size_t W,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& dst,
    const std::string& label = "KokkosDNN::dit_transpose_out") {
  const size_t C = dst.extent(0);
  if (H == 0 || W == 0)
    throw std::invalid_argument("dit_transpose_rows_to_chw: bad geometry");
  if (src.extent(0) != H * W || src.extent(1) != C || dst.extent(1) != H ||
      dst.extent(2) != W)
    throw std::invalid_argument("dit_transpose_rows_to_chw: dim mismatch");
  if (C == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {C, H, W}),
      KOKKOS_LAMBDA(size_t c, size_t h, size_t w) {
        dst(c, h, w) = src(h * W + w, c);
      });
}

// Order-preserving masked gather: out[i] = enc[idx[i]] rows, i in [0, Sv).
// idx (device, [Sv]) is built on the host from the mask row (masks are tiny:
// B*S integers), so no device atomics are needed.
template <class ExecSpace, class Scalar>
void dit_gather_rows(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& enc,
    const Kokkos::View<int*, ExecSpace>& idx,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::dit_gather") {
  const size_t Sv = idx.extent(0), Td = enc.extent(1);
  if (out.extent(0) != Sv || out.extent(1) != Td)
    throw std::invalid_argument("dit_gather_rows: dim mismatch");
  if (Sv == 0 || Td == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Sv, Td}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        out(i, j) = enc((size_t)idx(i), j);
      });
}

// Column splits of a fused GEMM output: [R,2N]->[R,N]x2, [R,3N]->3x,
// [R,6N]->6x (modulation lanes). Exact copies.
template <class ExecSpace, class Scalar>
void dit_split2(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& a,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& b,
    const std::string& label = "KokkosDNN::dit_split2") {
  const size_t R = src.extent(0), N = a.extent(1);
  if (src.extent(1) != 2 * N || b.extent(0) != R || b.extent(1) != N ||
      a.extent(0) != R)
    throw std::invalid_argument("dit_split2: dim mismatch");
  if (R == 0 || N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        a(i, j) = src(i, j);
        b(i, j) = src(i, j + N);
      });
}

template <class ExecSpace, class Scalar>
void dit_split3(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& a,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& c,
    const std::string& label = "KokkosDNN::dit_split3") {
  const size_t R = src.extent(0), N = a.extent(1);
  if (src.extent(1) != 3 * N || b.extent(0) != R || b.extent(1) != N ||
      c.extent(0) != R || c.extent(1) != N || a.extent(0) != R)
    throw std::invalid_argument("dit_split3: dim mismatch");
  if (R == 0 || N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        a(i, j) = src(i, j);
        b(i, j) = src(i, j + N);
        c(i, j) = src(i, j + 2 * N);
      });
}

template <class ExecSpace, class Scalar>
void dit_split6(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& l0,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& l1,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& l2,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& l3,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& l4,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& l5,
    const std::string& label = "KokkosDNN::dit_split6") {
  const size_t R = src.extent(0), N = l0.extent(1);
  auto ok = [&](const auto& l) {
    return l.extent(0) == R && l.extent(1) == N;
  };
  if (src.extent(1) != 6 * N || !ok(l0) || !ok(l1) || !ok(l2) || !ok(l3) ||
      !ok(l4) || !ok(l5))
    throw std::invalid_argument("dit_split6: dim mismatch");
  if (R == 0 || N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        l0(i, j) = src(i, j);
        l1(i, j) = src(i, j + N);
        l2(i, j) = src(i, j + 2 * N);
        l3(i, j) = src(i, j + 3 * N);
        l4(i, j) = src(i, j + 4 * N);
        l5(i, j) = src(i, j + 5 * N);
      });
}

// Flat copy between same-space buffers (rank reshuffles with identical
// row-major element order, e.g. [S,D] <-> [1,S,H,Dh] with D=H*Dh).
// Pure Kokkos::deep_copy on unmanaged rank-1 aliases: no new math kernel.
template <class ExecSpace, class Scalar>
void dit_copy_flat(Scalar* dst, const Scalar* src, size_t n) {
  if (n == 0) return;
  Kokkos::View<Scalar*, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      d(dst, n);
  Kokkos::View<const Scalar*, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s(src, n);
  Kokkos::deep_copy(d, s);
}

// ---- host-side safetensors parsing (F32/BF16, header-only) ----

inline uint64_t dit_ld64(const char* p) {
  uint64_t v = 0;
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

inline float dit_bf16_to_float(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round-to-nearest-even float32 -> bf16, preserving Inf/NaN payloads.
inline uint16_t dit_float_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7F800000u) == 0x7F800000u) {
    uint16_t h = (uint16_t)(u >> 16);
    if ((u & 0x007FFFFFu) != 0) h |= (uint16_t)0x0040u;  // keep NaN quiet
    return h;
  }
  const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)((u + bias) >> 16);
}

struct DitSafeEntry {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // absolute file offsets of the data slice
};

inline std::string dit_parse_json_string(const std::string& s, size_t& pos) {
  ++pos;  // opening quote
  std::string out;
  while (pos < s.size()) {
    char c = s[pos++];
    if (c == '"') return out;
    if (c == '\\' && pos < s.size()) {
      out += s[pos++];
    } else {
      out += c;
    }
  }
  throw std::runtime_error("MossImageDit: malformed safetensors header");
}

// Minimal header-JSON scanner: only the torch-saved subset
// {"__metadata__":{...},"name":{"dtype":..,"shape":[..],"data_offsets":[a,b]}}.
inline std::map<std::string, DitSafeEntry> dit_parse_header(
    const std::string& js, uint64_t data_base) {
  std::map<std::string, DitSafeEntry> out;
  size_t pos = 0;
  auto skip_ws = [&]() {
    while (pos < js.size() && (js[pos] == ' ' || js[pos] == '\t' ||
                               js[pos] == '\n' || js[pos] == '\r'))
      ++pos;
  };
  skip_ws();
  if (pos >= js.size() || js[pos] != '{')
    throw std::runtime_error("MossImageDit: malformed safetensors header");
  ++pos;
  while (true) {
    skip_ws();
    if (pos >= js.size())
      throw std::runtime_error("MossImageDit: truncated safetensors header");
    if (js[pos] == '}') break;
    if (js[pos] != '"')
      throw std::runtime_error("MossImageDit: bad header key");
    std::string name = dit_parse_json_string(js, pos);
    skip_ws();
    if (pos >= js.size() || js[pos] != ':')
      throw std::runtime_error("MossImageDit: bad header separator");
    ++pos;
    skip_ws();
    if (name == "__metadata__") {
      if (pos >= js.size() || js[pos] != '{')
        throw std::runtime_error("MossImageDit: bad __metadata__");
      int depth = 0;
      bool in_str = false;
      while (pos < js.size()) {
        char c = js[pos++];
        if (in_str) {
          if (c == '\\') {
            if (pos < js.size()) ++pos;
          } else if (c == '"') {
            in_str = false;
          }
        } else {
          if (c == '"') in_str = true;
          if (c == '{') ++depth;
          if (c == '}') {
            if (--depth == 0) break;
          }
        }
      }
    } else {
      if (pos >= js.size() || js[pos] != '{')
        throw std::runtime_error("MossImageDit: bad entry for " + name);
      ++pos;
      DitSafeEntry e;
      bool have_dtype = false, have_shape = false, have_off = false;
      while (true) {
        skip_ws();
        if (pos >= js.size())
          throw std::runtime_error("MossImageDit: truncated entry " + name);
        if (js[pos] == '}') {
          ++pos;
          break;
        }
        if (js[pos] != '"')
          throw std::runtime_error("MossImageDit: bad entry key " + name);
        std::string k = dit_parse_json_string(js, pos);
        skip_ws();
        if (pos >= js.size() || js[pos] != ':')
          throw std::runtime_error("MossImageDit: bad entry sep " + name);
        ++pos;
        skip_ws();
        if (k == "dtype") {
          if (pos >= js.size() || js[pos] != '"')
            throw std::runtime_error("MossImageDit: bad dtype " + name);
          e.dtype = dit_parse_json_string(js, pos);
          have_dtype = true;
        } else if (k == "shape") {
          if (pos >= js.size() || js[pos] != '[')
            throw std::runtime_error("MossImageDit: bad shape " + name);
          ++pos;
          e.shape.clear();
          while (true) {
            skip_ws();
            if (pos < js.size() && js[pos] == ']') {
              ++pos;
              break;
            }
            size_t st = pos;
            while (pos < js.size() &&
                   (isdigit((unsigned char)js[pos]) || js[pos] == '-'))
              ++pos;
            if (st == pos)
              throw std::runtime_error("MossImageDit: bad shape num " + name);
            e.shape.push_back(std::stoll(js.substr(st, pos - st)));
            skip_ws();
            if (pos < js.size() && js[pos] == ',') ++pos;
          }
          have_shape = true;
        } else if (k == "data_offsets") {
          if (pos >= js.size() || js[pos] != '[')
            throw std::runtime_error("MossImageDit: bad offsets " + name);
          ++pos;
          auto num = [&]() {
            skip_ws();
            size_t st = pos;
            while (pos < js.size() && isdigit((unsigned char)js[pos])) ++pos;
            if (st == pos)
              throw std::runtime_error("MossImageDit: bad offset " + name);
            uint64_t v = std::stoull(js.substr(st, pos - st));
            skip_ws();
            return v;
          };
          uint64_t a = num();
          if (pos >= js.size() || js[pos] != ',')
            throw std::runtime_error("MossImageDit: bad offsets " + name);
          ++pos;
          uint64_t b = num();
          skip_ws();
          if (pos >= js.size() || js[pos] != ']')
            throw std::runtime_error("MossImageDit: bad offsets " + name);
          ++pos;
          if (b < a)
            throw std::runtime_error("MossImageDit: bad offsets " + name);
          e.begin = data_base + a;
          e.end = data_base + b;
          have_off = true;
        } else {
          if (pos < js.size() && js[pos] == '"') {
            (void)dit_parse_json_string(js, pos);
          } else {
            while (pos < js.size() && js[pos] != ',' && js[pos] != '}') ++pos;
          }
        }
        skip_ws();
        if (pos < js.size() && js[pos] == ',') ++pos;
      }
      if (!have_dtype || !have_shape || !have_off)
        throw std::runtime_error("MossImageDit: incomplete entry " + name);
      out[name] = e;
    }
    skip_ws();
    if (pos < js.size() && js[pos] == ',') ++pos;
  }
  return out;
}

// Grow-only view cache: exact-size managed views keyed by (tag, extents).
// Entries are never shrunk or evicted, so steady-state forward() performs
// no allocation; distinct shapes seen over the process lifetime bound memory.
template <class V>
struct DitViewCache {
  std::map<std::pair<std::string, std::vector<size_t>>, V> m;
  template <class... Dims>
  V get(const std::string& tag, Dims... d) {
    std::vector<size_t> key{(size_t)d...};
    auto& v = m[std::make_pair(tag, key)];
    bool fit = (v.data() != nullptr);
    if (fit) {
      size_t i = 0;
      for (size_t e : key) {  // rank is fixed by V; compare each extent
        if (v.extent(i++) != e) {
          fit = false;
          break;
        }
      }
    }
    if (!fit) v = V(tag, d...);
    return v;
  }
  size_t bytes(size_t scalar_bytes) const {
    size_t n = 0;
    for (const auto& kv : m) n += kv.second.size() * scalar_bytes;
    return n;
  }
  void clear() { m.clear(); }
};

}  // namespace Impl

// ---- per-block weights (device Views in Scalar) ----

template <class ExecSpace, class Scalar>
struct MossImageDitTextBlockWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> qkv_W;   // [3D,D]
  Kokkos::View<Scalar*, ExecSpace> qkv_b;                         // [3D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> proj_W;  // [D,D]
  Kokkos::View<Scalar*, ExecSpace> proj_b;                        // [D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> gate_W;  // [F,D]
  Kokkos::View<Scalar*, ExecSpace> gate_b;                        // [F]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> up_W;    // [F,D]
  Kokkos::View<Scalar*, ExecSpace> up_b;                          // [F]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> down_W;  // [D,F]
  Kokkos::View<Scalar*, ExecSpace> down_b;                        // [D]
  Kokkos::View<Scalar*, ExecSpace> norm1, norm2;                  // [D]
  Kokkos::View<Scalar*, ExecSpace> qn, kn;                        // [Hd]
};

template <class ExecSpace, class Scalar>
struct MossImageDitBlockWeights {
  Kokkos::View<Scalar*, ExecSpace> norm1, norm2;  // [D]
  Kokkos::View<Scalar*, ExecSpace> qn, kn;        // [Hd]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> q_W, k_W, v_W,
      o_W;                                                     // [D,D] x4
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> up_gate_W;  // [2F,D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> down_W;     // [D,F]
};

// Full transformer weights: video/text embedders, time MLP, modulations,
// 2 text-refiner blocks, N main blocks, final norm + head. BF16-capable:
// Scalar is float or Kokkos::Experimental::bhalf_t.
template <class ExecSpace, class Scalar>
struct MossImageDitWeights {
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;

  Config cfg;
  V2 video_W;  // [D,C]
  V1 video_b;  // [D]
  V2 text_proj_W;  // [D,Td]
  V1 text_proj_b;  // [D]
  std::vector<MossImageDitTextBlockWeights<ExecSpace, Scalar>> text_blocks;
  V2 time0_W;  // [D,Freq]
  V1 time0_b;  // [D]
  V2 time2_W;  // [D,D]
  V1 time2_b;  // [D]
  V2 shared_W;  // [6D,D]
  V1 shared_b;  // [6D]
  V2 head_W;    // [2D,D]
  V1 head_b;    // [2D]
  std::vector<MossImageDitBlockWeights<ExecSpace, Scalar>> blocks;
  V1 final_norm;  // [D]
  V2 final_W;     // [C,D]

  void allocate(const Config& c) {
    c.validate();
    cfg = c;
    const size_t D = (size_t)c.dim, F = (size_t)c.ffn_dim,
                 Td = (size_t)c.text_dim, Fr = (size_t)c.freq_dim,
                 C = (size_t)c.latent_channels;
    const size_t Hd = D / (size_t)c.num_heads;
    auto mat = [&](const char* lb, size_t r, size_t k) {
      return V2(lb, r, k);
    };
    auto vec = [&](const char* lb, size_t n) { return V1(lb, n); };
    video_W = mat("dit_video_W", D, C);
    video_b = vec("dit_video_b", D);
    text_proj_W = mat("dit_text_proj_W", D, Td);
    text_proj_b = vec("dit_text_proj_b", D);
    text_blocks.resize(2);
    for (auto& t : text_blocks) {
      t.qkv_W = mat("dit_tqkv_W", 3 * D, D);
      t.qkv_b = vec("dit_tqkv_b", 3 * D);
      t.proj_W = mat("dit_tproj_W", D, D);
      t.proj_b = vec("dit_tproj_b", D);
      t.gate_W = mat("dit_tgate_W", F, D);
      t.gate_b = vec("dit_tgate_b", F);
      t.up_W = mat("dit_tup_W", F, D);
      t.up_b = vec("dit_tup_b", F);
      t.down_W = mat("dit_tdown_W", D, F);
      t.down_b = vec("dit_tdown_b", D);
      t.norm1 = vec("dit_tnorm1", D);
      t.norm2 = vec("dit_tnorm2", D);
      t.qn = vec("dit_tqn", Hd);
      t.kn = vec("dit_tkn", Hd);
    }
    time0_W = mat("dit_time0_W", D, Fr);
    time0_b = vec("dit_time0_b", D);
    time2_W = mat("dit_time2_W", D, D);
    time2_b = vec("dit_time2_b", D);
    shared_W = mat("dit_shared_W", 6 * D, D);
    shared_b = vec("dit_shared_b", 6 * D);
    head_W = mat("dit_head_W", 2 * D, D);
    head_b = vec("dit_head_b", 2 * D);
    blocks.resize((size_t)c.num_layers);
    for (auto& b : blocks) {
      b.norm1 = vec("dit_norm1", D);
      b.norm2 = vec("dit_norm2", D);
      b.qn = vec("dit_qn", Hd);
      b.kn = vec("dit_kn", Hd);
      b.q_W = mat("dit_q_W", D, D);
      b.k_W = mat("dit_k_W", D, D);
      b.v_W = mat("dit_v_W", D, D);
      b.o_W = mat("dit_o_W", D, D);
      b.up_gate_W = mat("dit_ug_W", 2 * F, D);
      b.down_W = mat("dit_down_W", D, F);
    }
    final_norm = vec("dit_final_norm", D);
    final_W = mat("dit_final_W", C, D);
  }

  // Drop every device allocation (assigning {} releases the View reference;
  // clears the block vectors so resident bytes return to ~0).
  void release() {
    video_W = {};
    video_b = {};
    text_proj_W = {};
    text_proj_b = {};
    text_blocks.clear();
    text_blocks.shrink_to_fit();
    time0_W = {};
    time0_b = {};
    time2_W = {};
    time2_b = {};
    shared_W = {};
    shared_b = {};
    head_W = {};
    head_b = {};
    blocks.clear();
    blocks.shrink_to_fit();
    final_norm = {};
    final_W = {};
  }

  // Resident device bytes currently held (rank sizes times sizeof(Scalar);
  // 0 for released views). Host-side estimate only.
  size_t resident_bytes() const {
    size_t n = 0;
    auto acc2 = [&](const auto& v) {
      n += v.extent(0) * v.extent(1);
    };
    auto acc1 = [&](const auto& v) { n += v.extent(0); };
    acc2(video_W);
    acc1(video_b);
    acc2(text_proj_W);
    acc1(text_proj_b);
    for (const auto& t : text_blocks) {
      acc2(t.qkv_W);
      acc1(t.qkv_b);
      acc2(t.proj_W);
      acc1(t.proj_b);
      acc2(t.gate_W);
      acc1(t.gate_b);
      acc2(t.up_W);
      acc1(t.up_b);
      acc2(t.down_W);
      acc1(t.down_b);
      acc1(t.norm1);
      acc1(t.norm2);
      acc1(t.qn);
      acc1(t.kn);
    }
    acc2(time0_W);
    acc1(time0_b);
    acc2(time2_W);
    acc1(time2_b);
    acc2(shared_W);
    acc1(shared_b);
    acc2(head_W);
    acc1(head_b);
    for (const auto& b : blocks) {
      acc1(b.norm1);
      acc1(b.norm2);
      acc1(b.qn);
      acc1(b.kn);
      acc2(b.q_W);
      acc2(b.k_W);
      acc2(b.v_W);
      acc2(b.o_W);
      acc2(b.up_gate_W);
      acc2(b.down_W);
    }
    acc1(final_norm);
    acc2(final_W);
    return n * sizeof(Scalar);
  }

  // Deterministic small-random fill for shape/forward tests without a
  // checkpoint (norms are ones, biases zero; needs allocate() first).
  void random_init(uint64_t seed = 42) {
    if (video_W.extent(0) == 0)
      throw std::runtime_error("MossImageDitWeights::random_init: allocate?");
    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    auto rnd2 = [&](auto& v) {
      auto h = Kokkos::create_mirror_view(Kokkos::HostSpace(), v);
      for (size_t i = 0; i < v.extent(0); ++i)
        for (size_t j = 0; j < v.extent(1); ++j)
          h(i, j) = static_cast<Scalar>(dist(eng));
      Kokkos::deep_copy(v, h);
    };
    auto rnd1 = [&](auto& v) {
      auto h = Kokkos::create_mirror_view(Kokkos::HostSpace(), v);
      for (size_t i = 0; i < v.extent(0); ++i)
        h(i) = static_cast<Scalar>(dist(eng));
      Kokkos::deep_copy(v, h);
    };
    auto ones1 = [&](auto& v) {
      Kokkos::deep_copy(v, static_cast<Scalar>(1.0f));
    };
    auto zero1 = [&](auto& v) {
      Kokkos::deep_copy(v, static_cast<Scalar>(0.0f));
    };
    rnd2(video_W);
    zero1(video_b);
    rnd2(text_proj_W);
    zero1(text_proj_b);
    for (auto& t : text_blocks) {
      rnd2(t.qkv_W);
      zero1(t.qkv_b);
      rnd2(t.proj_W);
      zero1(t.proj_b);
      rnd2(t.gate_W);
      zero1(t.gate_b);
      rnd2(t.up_W);
      zero1(t.up_b);
      rnd2(t.down_W);
      zero1(t.down_b);
      ones1(t.norm1);
      ones1(t.norm2);
      ones1(t.qn);
      ones1(t.kn);
    }
    rnd2(time0_W);
    zero1(time0_b);
    rnd2(time2_W);
    zero1(time2_b);
    rnd2(shared_W);
    zero1(shared_b);
    rnd2(head_W);
    zero1(head_b);
    for (auto& b : blocks) {
      ones1(b.norm1);
      ones1(b.norm2);
      ones1(b.qn);
      ones1(b.kn);
      rnd2(b.q_W);
      rnd2(b.k_W);
      rnd2(b.v_W);
      rnd2(b.o_W);
      rnd2(b.up_gate_W);
      rnd2(b.down_W);
    }
    ones1(final_norm);
    rnd2(final_W);
  }

  // Load the REAL 463-tensor checkpoint layout (possibly sharded across
  // several files; the combined shard set must cover every expected tensor).
  // Tensor names are suffix-matched (training/export prefixes allowed);
  // rope.bands is skipped (derived fp32 constant, rebuilt by _RoPE).
  // File dtypes F32 and BF16 are accepted (converted to Scalar on the host).
  // Bytes stream per tensor through bounded HostSpace chunks into a host
  // mirror with a single deep_copy per tensor (no 2x device residency).
  // Throws on missing tensors, unknown tensor names, non-F32/BF16 dtypes,
  // or shape/numel/byte-size mismatch. Needs allocate() first.
  void load_safetensors(const std::vector<std::string>& shard_paths) {
    if (shard_paths.empty())
      throw std::invalid_argument(
          "MossImageDitWeights::load_safetensors: no shards given");
    if (video_W.extent(0) == 0)
      throw std::runtime_error(
          "MossImageDitWeights::load_safetensors: allocate first");
    static_assert(std::is_same_v<Scalar, float> ||
                      std::is_same_v<Scalar,
                                     Kokkos::Experimental::bhalf_t>,
                  "MossImageDitWeights::load_safetensors: Scalar must be "
                  "float or bhalf_t");
    const std::vector<DitTensorSpec> specs = moss_dit_expected_tensors(cfg);
    std::map<std::string, bool> found;
    for (const auto& s : specs) found[s.name] = false;

    for (const std::string& path : shard_paths) {
      std::ifstream f(path, std::ios::binary | std::ios::ate);
      if (!f)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: cannot open '" + path +
            "'");
      std::streamoff e = f.tellg();
      if (e < 8)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: too small '" + path + "'");
      const size_t fsize = (size_t)e;
      f.seekg(0, std::ios::beg);
      char hlen_raw[8];
      f.read(hlen_raw, 8);
      if (!f)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: short read '" + path +
            "'");
      const uint64_t hn = Impl::dit_ld64(hlen_raw);
      if (8 + hn > fsize)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: bad header in '" + path +
            "'");
      std::string js((size_t)hn, '\0');
      f.read(js.data(), (std::streamsize)hn);
      if (!f)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: short header '" + path +
            "'");
      const uint64_t data_base = 8 + hn;
      auto entries = Impl::dit_parse_header(js, data_base);
      for (const auto& kv : entries) {
        const std::string& full = kv.first;
        const Impl::DitSafeEntry& en = kv.second;
        if (full == "rope.bands" ||
            (full.size() > 11 &&
             full.compare(full.size() - 11, 11, ".rope.bands") == 0))
          continue;  // derived constant, not a weight
        const DitTensorSpec* spec = nullptr;
        for (const auto& s : specs) {
          if (full.size() >= s.name.size() &&
              full.compare(full.size() - s.name.size(), s.name.size(),
                           s.name) == 0 &&
              (full.size() == s.name.size() ||
               full[full.size() - s.name.size() - 1] == '.')) {
            if (!spec || s.name.size() > spec->name.size()) spec = &s;
          }
        }
        if (!spec)
          throw std::runtime_error(
              "MossImageDitWeights::load_safetensors: unknown tensor '" +
              full + "' in '" + path + "'");
        if (found[spec->name])
          throw std::runtime_error(
              "MossImageDitWeights::load_safetensors: duplicate tensor '" +
              spec->name + "'");
        load_one_tensor(f, full, en, *spec);
        found[spec->name] = true;
      }
    }
    for (const auto& s : specs) {
      if (!found[s.name])
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: missing tensor '" +
            s.name + "'");
    }
  }

  void load_safetensors(const std::string& shard_path) {
    load_safetensors(std::vector<std::string>{shard_path});
  }

 private:
  // Stream one file tensor into its device view (shape-checked).
  void load_one_tensor(std::ifstream& f, const std::string& full,
                       const Impl::DitSafeEntry& en, const DitTensorSpec& s) {
    size_t elsize = 0;
    if (en.dtype == "F32") {
      elsize = 4;
    } else if (en.dtype == "BF16") {
      elsize = 2;
    } else {
      throw std::runtime_error(
          "MossImageDitWeights::load_safetensors: tensor '" + full +
          "' has dtype " + en.dtype + " (only F32/BF16 supported)");
    }
    int64_t numel = 1;
    for (int64_t d : en.shape) {
      if (d < 0)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: bad shape '" + full + "'");
      numel *= d;
    }
    const size_t want =
        s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
    if ((size_t)numel != want)
      throw std::runtime_error(
          "MossImageDitWeights::load_safetensors: tensor '" + full +
          "' has numel " + std::to_string(numel) + ", expected " +
          std::to_string(want) + " (" + s.name + ")");
    if (en.end < en.begin || en.end - en.begin != want * elsize)
      throw std::runtime_error(
          "MossImageDitWeights::load_safetensors: tensor '" + full +
          "' slice size mismatch");
    V2* dst2 = nullptr;
    V1* dst1 = nullptr;
    resolve_tensor(s.name, dst2, dst1);
    if (s.matrix) {
      if (!dst2 || (*dst2).extent(0) != (size_t)s.d0 ||
          (*dst2).extent(1) != (size_t)s.d1)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: view mismatch '" + s.name +
            "'");
      stream_into(f, en, want, elsize, (*dst2).data(), want);
    } else {
      if (!dst1 || (*dst1).extent(0) != (size_t)s.d0)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: view mismatch '" + s.name +
            "'");
      stream_into(f, en, want, elsize, (*dst1).data(), want);
    }
  }

  // File bytes -> host mirror in bounded chunks, then one deep_copy.
  void stream_into(std::ifstream& f, const Impl::DitSafeEntry& en, size_t n,
                   size_t elsize, Scalar* dev_ptr, size_t n_elems) {
    using HostV = Kokkos::View<Scalar*, Kokkos::HostSpace>;
    HostV host("dit_load", n_elems);
    constexpr size_t kChunk = size_t{1} << 18;
    std::vector<char> raw;
    raw.reserve(std::min(n, kChunk) * elsize);
    const bool fast_f32 = std::is_same_v<Scalar, float> && elsize == 4;
    for (size_t o = 0; o < n;) {
      const size_t m = std::min(n - o, kChunk);
      raw.resize(m * elsize);
      f.seekg((std::streamoff)(en.begin + o * elsize), std::ios::beg);
      f.read(raw.data(), (std::streamsize)(m * elsize));
      if (!f)
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: truncated tensor data");
      if (fast_f32) {
        std::memcpy(host.data() + o, raw.data(), m * 4);
      } else if (elsize == 4) {  // F32 file -> Scalar (float or bhalf_t)
        const float* src = reinterpret_cast<const float*>(raw.data());
        for (size_t i = 0; i < m; ++i)
          host(o + i) = static_cast<Scalar>(src[i]);
      } else {  // BF16 file -> Scalar via FP32
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < m; ++i)
          host(o + i) =
              static_cast<Scalar>(Impl::dit_bf16_to_float(src[i]));
      }
      o += m;
    }
    Kokkos::View<Scalar*, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        dev(dev_ptr, n_elems);
    Kokkos::deep_copy(dev, host);
  }

  void resolve_tensor(const std::string& name, V2*& out2, V1*& out1) {
    out2 = nullptr;
    out1 = nullptr;
    if (name == "video_embedder.weight") {
      out2 = &video_W;
      return;
    }
    if (name == "video_embedder.bias") {
      out1 = &video_b;
      return;
    }
    if (name == "text_embedder.proj.weight") {
      out2 = &text_proj_W;
      return;
    }
    if (name == "text_embedder.proj.bias") {
      out1 = &text_proj_b;
      return;
    }
    if (name == "time_embedding.0.weight") {
      out2 = &time0_W;
      return;
    }
    if (name == "time_embedding.0.bias") {
      out1 = &time0_b;
      return;
    }
    if (name == "time_embedding.2.weight") {
      out2 = &time2_W;
      return;
    }
    if (name == "time_embedding.2.bias") {
      out1 = &time2_b;
      return;
    }
    if (name == "shared_block_modulation.weight") {
      out2 = &shared_W;
      return;
    }
    if (name == "shared_block_modulation.bias") {
      out1 = &shared_b;
      return;
    }
    if (name == "video_head_modulation.weight") {
      out2 = &head_W;
      return;
    }
    if (name == "video_head_modulation.bias") {
      out1 = &head_b;
      return;
    }
    if (name == "final_norm_video.weight") {
      out1 = &final_norm;
      return;
    }
    if (name == "final_linear_video.weight") {
      out2 = &final_W;
      return;
    }
    const std::string tb = "text_embedder.blocks.";
    if (name.compare(0, tb.size(), tb) == 0) {
      size_t dot = name.find('.', tb.size());
      int bi = std::stoi(name.substr(tb.size(), dot - tb.size()));
      if (bi < 0 || (size_t)bi >= text_blocks.size())
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: bad text block '" + name +
            "'");
      auto& t = text_blocks[(size_t)bi];
      std::string tail = name.substr(dot + 1);
      if (tail == "qkv.weight") {
        out2 = &t.qkv_W;
        return;
      }
      if (tail == "qkv.bias") {
        out1 = &t.qkv_b;
        return;
      }
      if (tail == "proj.weight") {
        out2 = &t.proj_W;
        return;
      }
      if (tail == "proj.bias") {
        out1 = &t.proj_b;
        return;
      }
      if (tail == "ffn_gate.weight") {
        out2 = &t.gate_W;
        return;
      }
      if (tail == "ffn_gate.bias") {
        out1 = &t.gate_b;
        return;
      }
      if (tail == "ffn_up.weight") {
        out2 = &t.up_W;
        return;
      }
      if (tail == "ffn_up.bias") {
        out1 = &t.up_b;
        return;
      }
      if (tail == "ffn_down.weight") {
        out2 = &t.down_W;
        return;
      }
      if (tail == "ffn_down.bias") {
        out1 = &t.down_b;
        return;
      }
      if (tail == "norm1.weight") {
        out1 = &t.norm1;
        return;
      }
      if (tail == "norm2.weight") {
        out1 = &t.norm2;
        return;
      }
      if (tail == "q_norm.weight") {
        out1 = &t.qn;
        return;
      }
      if (tail == "k_norm.weight") {
        out1 = &t.kn;
        return;
      }
    }
    const std::string lb = "layers.";
    if (name.compare(0, lb.size(), lb) == 0) {
      size_t dot = name.find('.', lb.size());
      int li = std::stoi(name.substr(lb.size(), dot - lb.size()));
      if (li < 0 || (size_t)li >= blocks.size())
        throw std::runtime_error(
            "MossImageDitWeights::load_safetensors: bad layer '" + name + "'");
      auto& b = blocks[(size_t)li];
      std::string tail = name.substr(dot + 1);
      if (tail == "norm1.weight") {
        out1 = &b.norm1;
        return;
      }
      if (tail == "norm2.weight") {
        out1 = &b.norm2;
        return;
      }
      if (tail == "q_norm.weight") {
        out1 = &b.qn;
        return;
      }
      if (tail == "k_norm.weight") {
        out1 = &b.kn;
        return;
      }
      if (tail == "proj.q.weight") {
        out2 = &b.q_W;
        return;
      }
      if (tail == "proj.k.weight") {
        out2 = &b.k_W;
        return;
      }
      if (tail == "proj.v.weight") {
        out2 = &b.v_W;
        return;
      }
      if (tail == "proj.o.weight") {
        out2 = &b.o_W;
        return;
      }
      if (tail == "proj.ffn_up_gate.weight") {
        out2 = &b.up_gate_W;
        return;
      }
      if (tail == "proj.ffn_down.weight") {
        out2 = &b.down_W;
        return;
      }
    }
    throw std::runtime_error(
        "MossImageDitWeights::load_safetensors: cannot resolve '" + name +
        "'");
  }
};

// Forward scratch: grow-only view cache (see Impl::DitViewCache).
template <class ExecSpace, class Scalar>
struct MossImageDitScratch {
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using V3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using V4 = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using F2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using F3 = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;
  using I1 = Kokkos::View<int*, ExecSpace>;
  Impl::DitViewCache<V1> c1;
  Impl::DitViewCache<V2> c2;
  Impl::DitViewCache<V3> c3;
  Impl::DitViewCache<V4> c4;
  Impl::DitViewCache<F2> f2;
  Impl::DitViewCache<F3> f3;
  Impl::DitViewCache<I1> i1;
  size_t bytes() const {
    return c1.bytes(sizeof(Scalar)) + c2.bytes(sizeof(Scalar)) +
           c3.bytes(sizeof(Scalar)) + c4.bytes(sizeof(Scalar)) +
           f2.bytes(4) + f3.bytes(4) + i1.bytes(4);
  }
  void clear() {
    c1.clear();
    c2.clear();
    c3.clear();
    c4.clear();
    f2.clear();
    f3.clear();
    i1.clear();
  }
};

// TRUE MOSS-Image DiT model: owns Weights + grow-only scratch, forward()
// follows pipeline_moss_image.py MOSSImageTransformer::forward exactly
// (per-sample batch loop with mask truncation, skip_layers identity).
template <class ExecSpace, class Scalar>
class MossImageDitModel {
 public:
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using V3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using V4 = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using F1 = Kokkos::View<float*, ExecSpace>;
  using F2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using F3 = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;
  using L4 = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using I1 = Kokkos::View<int*, ExecSpace>;
  using Weights = MossImageDitWeights<ExecSpace, Scalar>;

  explicit MossImageDitModel(const Config& cfg = Config::moss_image_sft()) {
    cfg.validate();
    cfg_ = cfg;
    weights_.allocate(cfg_);
  }

  const Config& config() const { return cfg_; }
  Weights& weights() { return weights_; }
  const Weights& weights() const { return weights_; }
  size_t scratch_bytes() const { return scratch_.bytes(); }
  void release_scratch() { scratch_.clear(); }

  // DiT forward: latents [B,C,H,W] + per-sample timestep (already-shifted
  // scheduler sigmas, [B]) + encoder_hidden [B,S,Td] + binary mask [B,S]
  // (any integer view type; nonzero = valid) -> out [B,C,H,W] velocity.
  // skip_layers holds zero-based block indices evaluated as exact identity
  // (the default negative-branch skip is {22} at the pipeline level).
  // No cross-sample interaction: each sample is truncated to its valid
  // text length before concat, exactly like the reference batch loop.
  template <class MaskView>
  void forward(const L4& latents, const F1& timestep, const V3& encoder,
               const MaskView& mask, const L4& out,
               const std::vector<int>& skip_layers = {},
               const std::string& label = "KokkosDNN::moss_dit") {
    static_assert(MaskView::rank == 2, "mask must be rank-2 [B,S]");
    const std::vector<int> skips =
        MossImage::moss_checked_skip_layers(skip_layers, cfg_.num_layers);
    const size_t B = latents.extent(0), C = latents.extent(1),
                 H = latents.extent(2), Wd = latents.extent(3);
    const size_t S = encoder.extent(1), Td = encoder.extent(2);
    if (C != (size_t)cfg_.latent_channels)
      throw std::invalid_argument("MossImageDitModel::forward: latents must be"
                                  " [B, latent_channels, H, W]");
    if (H == 0 || Wd == 0)
      throw std::invalid_argument(
          "MossImageDitModel::forward: H and W must be > 0");
    if (encoder.extent(0) != B || Td != (size_t)cfg_.text_dim)
      throw std::invalid_argument(
          "MossImageDitModel::forward: encoder must be [B, S, text_dim]");
    if (mask.extent(0) != B || mask.extent(1) != S)
      throw std::invalid_argument(
          "MossImageDitModel::forward: mask must be [B, S]");
    if (timestep.extent(0) != B)
      throw std::invalid_argument(
          "MossImageDitModel::forward: timestep must be [B]");
    if (out.extent(0) != B || out.extent(1) != C || out.extent(2) != H ||
        out.extent(3) != Wd)
      throw std::invalid_argument(
          "MossImageDitModel::forward: out must be [B, C, H, W]");
    if (B == 0) return;

    const size_t D = (size_t)cfg_.dim, F = (size_t)cfg_.ffn_dim,
                 Fr = (size_t)cfg_.freq_dim;
    const size_t Nh = (size_t)cfg_.num_heads, Hd = D / Nh;
    const size_t HW = H * Wd;
    const float attn_scale = 1.0f / std::sqrt((float)Hd);

    // Host copies of the tiny per-sample metadata (mask + timestep).
    auto hmask = Kokkos::create_mirror_view(mask);
    Kokkos::deep_copy(hmask, mask);
    auto htime = Kokkos::create_mirror_view(timestep);
    Kokkos::deep_copy(htime, timestep);
    std::vector<size_t> nvalid(B, 0);
    for (size_t b = 0; b < B; ++b) {
      size_t n = 0;
      for (size_t s = 0; s < S; ++s)
        if (hmask(b, s)) ++n;
      if (n == 0)
        throw std::invalid_argument(
            "MossImageDitModel::forward: each sample needs >= 1 valid token");
      nvalid[b] = n;
    }

    auto& W = weights_;
    auto& Sc = scratch_;

    // Time path (batched): float64 sinusoid on the host (reference training
    // path: double freq/phase, cos|sin, float32), cast to Scalar for the
    // SiLU MLP (reference casts the embedding to the MLP weight dtype).
    V2 temb = Sc.c2.get("temb", B, Fr);
    {
      auto htemb = Kokkos::create_mirror_view(temb);
      const size_t Hh = Fr / 2;
      for (size_t b = 0; b < B; ++b) {
        const double tb = (double)htime(b);
        for (size_t j = 0; j < Hh; ++j) {
          const double freq = std::pow(10000.0, -(double)j / (double)Hh);
          const double ph = tb * freq;
          htemb(b, j) = static_cast<Scalar>((float)std::cos(ph));
          htemb(b, j + Hh) = static_cast<Scalar>((float)std::sin(ph));
        }
      }
      Kokkos::deep_copy(temb, htemb);
    }
    V2 thid = Sc.c2.get("thid", B, D);
    gemm_bias<ExecSpace, Scalar>(handle_, temb, W.time0_W, W.time0_b, thid,
                                 label + "_t0");
    silu_forward<ExecSpace, Scalar>(thid, thid, label + "_silu");
    V2 thid2 = Sc.c2.get("thid2", B, D);
    gemm_bias<ExecSpace, Scalar>(handle_, thid, W.time2_W, W.time2_b, thid2,
                                 label + "_t2");
    V2 mod6 = Sc.c2.get("mod6", B, 6 * D);
    gemm_bias<ExecSpace, Scalar>(handle_, thid2, W.shared_W, W.shared_b, mod6,
                                 label + "_mod");
    V2 mod2 = Sc.c2.get("mod2", B, 2 * D);
    gemm_bias<ExecSpace, Scalar>(handle_, thid2, W.head_W, W.head_b, mod2,
                                 label + "_headmod");
    V2 sh_a = Sc.c2.get("sh_a", B, D), sc_a = Sc.c2.get("sc_a", B, D),
       gt_a = Sc.c2.get("gt_a", B, D), sh_f = Sc.c2.get("sh_f", B, D),
       sc_f = Sc.c2.get("sc_f", B, D), gt_f = Sc.c2.get("gt_f", B, D);
    Impl::dit_split6<ExecSpace, Scalar>(mod6, sh_a, sc_a, gt_a, sh_f, sc_f,
                                       gt_f, label + "_split6");
    V2 hd_sh = Sc.c2.get("hd_sh", B, D), hd_sc = Sc.c2.get("hd_sc", B, D);
    Impl::dit_split2<ExecSpace, Scalar>(mod2, hd_sh, hd_sc, label + "_split2");
    V2 sh_a1 = Sc.c2.get("sh_a1", 1, D), sc_a1 = Sc.c2.get("sc_a1", 1, D),
       gt_a1 = Sc.c2.get("gt_a1", 1, D), sh_f1 = Sc.c2.get("sh_f1", 1, D),
       sc_f1 = Sc.c2.get("sc_f1", 1, D), gt_f1 = Sc.c2.get("gt_f1", 1, D),
       hd_sh1 = Sc.c2.get("hd_sh1", 1, D), hd_sc1 = Sc.c2.get("hd_sc1", 1, D);
    V2 ones = Sc.c2.get("ones", 1, D);
    Kokkos::deep_copy(ones, static_cast<Scalar>(1.0f));

    // Video path (batched): flatten/transpose + video_embedder.
    V2 vid_in = Sc.c2.get("vidin", B * HW, C);
    Impl::dit_transpose_bchw_to_rows<ExecSpace, Scalar>(latents, vid_in,
                                                       label + "_vin");
    V2 vidf = Sc.c2.get("vidf", B * HW, D);
    gemm_bias<ExecSpace, Scalar>(handle_, vid_in, W.video_W, W.video_b, vidf,
                                 label + "_vemb");

    for (size_t b = 0; b < B; ++b) {
      const size_t Sv = nvalid[b], Sj = HW + Sv;
      copy_lane_row(sh_a, b, sh_a1);
      copy_lane_row(sc_a, b, sc_a1);
      copy_lane_row(gt_a, b, gt_a1);
      copy_lane_row(sh_f, b, sh_f1);
      copy_lane_row(sc_f, b, sc_f1);
      copy_lane_row(gt_f, b, gt_f1);
      copy_lane_row(hd_sh, b, hd_sh1);
      copy_lane_row(hd_sc, b, hd_sc1);

      // Text path: gather valid rows, project, 2x TextBlock.
      V3 txt3 = Sc.c3.get("txt3", 1, Sv, D);
      {
        I1 gidx = Sc.i1.get("gidx", Sv);
        auto hgidx = Kokkos::create_mirror_view(gidx);
        size_t n = 0;
        for (size_t s = 0; s < S; ++s)
          if (hmask(b, s)) hgidx(n++) = (int)s;
        Kokkos::deep_copy(gidx, hgidx);
        auto enc_b =
            Kokkos::subview(encoder, b, Kokkos::ALL(), Kokkos::ALL());
        V2 txtin = Sc.c2.get("txtin", Sv, Td);
        Impl::dit_gather_rows<ExecSpace, Scalar>(enc_b, gidx, txtin,
                                                 label + "_gather");
        V2 txn = Sc.c2.get("txn", Sv, D);
        gemm_bias<ExecSpace, Scalar>(handle_, txtin, W.text_proj_W,
                                     W.text_proj_b, txn, label + "_tproj");
        Impl::dit_copy_flat<ExecSpace, Scalar>(row2(txt3).data(), txn.data(), Sv * D);
        F3 rope = Sc.f3.get("rope", 1, HW + Sv, Hd);
        MossImage::moss_rope_build_forward<ExecSpace>(
            (int)H, (int)Wd, (int)Sv, (float)cfg_.rope_base, rope,
            label + "_rope");
        F3 trope = Sc.f3.get("trope", 1, Sv, Hd);
        auto rope_tail = Kokkos::subview(
            rope, Kokkos::ALL(), std::pair<size_t, size_t>(HW, HW + Sv),
            Kokkos::ALL());
        Kokkos::deep_copy(trope, rope_tail);
        for (size_t ti = 0; ti < 2; ++ti)
          text_block_forward(txt3, trope, W.text_blocks[ti], attn_scale,
                             label);
        // Joint sequence: [video HW | text Sv].
        V3 joint3 = Sc.c3.get("joint", 1, Sj, D);
        auto jv = Kokkos::subview(joint3, 0,
                                  std::pair<size_t, size_t>(0, HW),
                                  Kokkos::ALL());
        auto vsrc = Kokkos::subview(
            vidf, std::pair<size_t, size_t>(b * HW, (b + 1) * HW),
            Kokkos::ALL());
        Kokkos::deep_copy(jv, vsrc);
        auto jt = Kokkos::subview(joint3, 0,
                                  std::pair<size_t, size_t>(HW, HW + Sv),
                                  Kokkos::ALL());
        Kokkos::deep_copy(jt, row2(txt3));
        main_blocks_forward(joint3, rope, skips, sh_a1, sc_a1, gt_a1, sh_f1,
                            sc_f1, gt_f1, attn_scale, label);
        // Head: video rows only -> final RMSNorm -> head adaLN -> Linear.
        V2 nr = Sc.c2.get("nr", HW, D);
        rmsnorm<ExecSpace, Scalar>(jv_head(joint3, HW), W.final_norm,
                                              kNormEps, nr, label + "_fnorm");
        auto hdsh_b = Kokkos::subview(hd_sh, b, Kokkos::ALL());
        auto hdsc_b = Kokkos::subview(hd_sc, b, Kokkos::ALL());
        adaln_modulate_forward<ExecSpace, Scalar>(nr, hdsh_b, hdsc_b, nr,
                                                 label + "_fmod");
        V2 fout = Sc.c2.get("fout", HW, C);
        gemm<ExecSpace, Scalar>(handle_, static_cast<Scalar>(1), nr,
                                W.final_W, static_cast<Scalar>(0), fout,
                                label + "_fout");
        auto out_b = Kokkos::subview(out, b, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL());
        Impl::dit_transpose_rows_to_chw<ExecSpace, Scalar>(fout, H, Wd, out_b,
                                                           label + "_vout");
      }
    }
  }

 private:
  Config cfg_;
  Weights weights_;
  DNNHandle<ExecSpace> handle_;
  MossImageDitScratch<ExecSpace, Scalar> scratch_;
  static constexpr float kNormEps = 1e-6f;  // reference _RMSNorm eps

  // [1,D] copy of batch row b (modulation lanes are [B,D]).
  static void copy_lane_row(const V2& src, size_t b, const V2& dst1) {
    auto r = Kokkos::subview(src, std::pair<size_t, size_t>(b, b + 1),
                             Kokkos::ALL());
    Kokkos::deep_copy(dst1, r);
  }

  // Rank-2 [S,D] view of an exact [1,S,D] stream buffer (leading-dim fix
  // of LayoutRight stays LayoutRight, so core signatures bind directly).
  static V2 row2(const V3& v) {
    return Kokkos::subview(v, 0, Kokkos::ALL(), Kokkos::ALL());
  }

  // Video head rows [HW,D] of the joint stream (leading fix + leading range
  // of LayoutRight stays LayoutRight).
  static V2 jv_head(const V3& joint, size_t HW) {
    return Kokkos::subview(joint, 0, std::pair<size_t, size_t>(0, HW),
                           Kokkos::ALL());
  }

  // One _TextBlock over txt [1,Sv,D] with text-slice rope [1,Sv,Hd].
  void text_block_forward(
      const V3& txt, const F3& trope,
      const MossImageDitTextBlockWeights<ExecSpace, Scalar>& T,
      float attn_scale, const std::string& label) {
    auto& Sc = scratch_;
    const size_t Sv = txt.extent(1), D = txt.extent(2);
    const size_t Nh = (size_t)cfg_.num_heads, Hd = D / Nh, F = (size_t)cfg_.ffn_dim;
    V2 txn = Sc.c2.get("txn2", Sv, D);
    rmsnorm<ExecSpace, Scalar>(row2(txt), T.norm1, kNormEps, txn,
                                          label + "_tnorm1");
    V2 tqkv = Sc.c2.get("tqkv2", Sv, 3 * D);
    gemm_bias<ExecSpace, Scalar>(handle_, txn, T.qkv_W, T.qkv_b, tqkv,
                                 label + "_tqkv");
    V2 q2 = Sc.c2.get("tq", Sv, D), k2 = Sc.c2.get("tk", Sv, D),
       v2 = Sc.c2.get("tv", Sv, D);
    Impl::dit_split3<ExecSpace, Scalar>(tqkv, q2, k2, v2, label + "_tsplit");
    V4 q4 = Sc.c4.get("tq4", 1, Sv, Nh, Hd),
       k4 = Sc.c4.get("tk4", 1, Sv, Nh, Hd);
    V3 vv3 = Sc.c3.get("tv3", Sv, Nh, Hd),
       ao3 = Sc.c3.get("tao3", Sv, Nh, Hd);
    Impl::dit_copy_flat<ExecSpace, Scalar>(q4.data(), q2.data(), Sv * D);
    Impl::dit_copy_flat<ExecSpace, Scalar>(k4.data(), k2.data(), Sv * D);
    Impl::dit_copy_flat<ExecSpace, Scalar>(vv3.data(), v2.data(), Sv * D);
    MossImage::moss_qk_rmsnorm_forward<ExecSpace, Scalar>(q4, T.qn, kNormEps,
                                                          q4, label + "_tqn");
    MossImage::moss_qk_rmsnorm_forward<ExecSpace, Scalar>(k4, T.kn, kNormEps,
                                                          k4, label + "_tkn");
    MossImage::moss_rope_apply_forward<ExecSpace, Scalar>(q4, trope, q4,
                                                          label + "_trq");
    MossImage::moss_rope_apply_forward<ExecSpace, Scalar>(k4, trope, k4,
                                                          label + "_trk");
    attention_mha_bidirectional<ExecSpace, Scalar>(
        handle_,
        Kokkos::subview(q4, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
        Kokkos::subview(k4, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
        vv3, ao3, attn_scale, label + "_tattn");
    V2 ao2 = Sc.c2.get("tao", Sv, D);
    Impl::dit_copy_flat<ExecSpace, Scalar>(ao2.data(), ao3.data(), Sv * D);
    V2 prao = Sc.c2.get("tprao", Sv, D);
    gemm_bias<ExecSpace, Scalar>(handle_, ao2, T.proj_W, T.proj_b, prao,
                                 label + "_tproj");
    V3 prao3 = Sc.c3.get("tprao3", 1, Sv, D);
    Impl::dit_copy_flat<ExecSpace, Scalar>(prao3.data(), prao.data(), Sv * D);
    V2 ones = Sc.c2.get("ones", 1, D);
    MossImage::moss_gate_residual_forward<ExecSpace, Scalar>(
        txt, prao3, ones, txt, label + "_tres");
    rmsnorm<ExecSpace, Scalar>(row2(txt), T.norm2, kNormEps, txn,
                                          label + "_tnorm2");
    V2 fh = Sc.c2.get("tfh", Sv, F), fg = Sc.c2.get("tfg", Sv, F);
    gemm_bias<ExecSpace, Scalar>(handle_, txn, T.gate_W, T.gate_b, fh,
                                 label + "_tgate");
    gemm_bias<ExecSpace, Scalar>(handle_, txn, T.up_W, T.up_b, fg,
                                 label + "_tup");
    MossImage::moss_situ_forward<ExecSpace, Scalar>(fh, fg, fh,
                                                    label + "_tsitu");
    V2 fd = Sc.c2.get("tfd", Sv, D);
    gemm_bias<ExecSpace, Scalar>(handle_, fh, T.down_W, T.down_b, fd,
                                 label + "_tdown");
    V3 fd3 = Sc.c3.get("tfd3", 1, Sv, D);
    Impl::dit_copy_flat<ExecSpace, Scalar>(fd3.data(), fd.data(), Sv * D);
    MossImage::moss_gate_residual_forward<ExecSpace, Scalar>(
        txt, fd3, ones, txt, label + "_tres2");
  }

  // All main _Blocks over joint [1,Sj,D] (skipped layers: exact identity).
  void main_blocks_forward(const V3& joint, const F3& rope,
                           const std::vector<int>& skips, const V2& sh_a1,
                           const V2& sc_a1, const V2& gt_a1, const V2& sh_f1,
                           const V2& sc_f1, const V2& gt_f1, float attn_scale,
                           const std::string& label) {
    auto& Sc = scratch_;
    auto& W = weights_;
    const size_t Sj = joint.extent(1), D = joint.extent(2);
    const size_t Nh = (size_t)cfg_.num_heads, Hd = D / Nh, F = (size_t)cfg_.ffn_dim;
    V3 nrm3 = Sc.c3.get("nrm", 1, Sj, D);
    V2 nrm2 = row2(nrm3);
    V2 q2 = Sc.c2.get("qq", Sj, D), k2 = Sc.c2.get("kk", Sj, D),
       v2 = Sc.c2.get("vv", Sj, D), ao2 = Sc.c2.get("ao", Sj, D),
       prao = Sc.c2.get("prao", Sj, D), fd = Sc.c2.get("fd", Sj, D);
    V4 q4 = Sc.c4.get("q4", 1, Sj, Nh, Hd),
       k4 = Sc.c4.get("k4", 1, Sj, Nh, Hd);
    V3 vv3 = Sc.c3.get("vv3", Sj, Nh, Hd),
       ao3 = Sc.c3.get("ao3", Sj, Nh, Hd);
    V3 prao3 = Sc.c3.get("prao3", 1, Sj, D);
    V3 fd3 = Sc.c3.get("fd3", 1, Sj, D);
    V2 ug = Sc.c2.get("ug", Sj, 2 * F);
    V2 fh = Sc.c2.get("fh", Sj, F), fg = Sc.c2.get("fg", Sj, F);
    for (int li = 0; li < cfg_.num_layers; ++li) {
      if (std::find(skips.begin(), skips.end(), li) != skips.end()) continue;
      const auto& L = W.blocks[(size_t)li];
      MossImage::moss_rmsnorm_modulate_forward<ExecSpace, Scalar>(
          joint, L.norm1, kNormEps, sh_a1, sc_a1, nrm3, label + "_norm1");
      gemm<ExecSpace, Scalar>(handle_, static_cast<Scalar>(1), nrm2, L.q_W,
                              static_cast<Scalar>(0), q2, label + "_q");
      gemm<ExecSpace, Scalar>(handle_, static_cast<Scalar>(1), nrm2, L.k_W,
                              static_cast<Scalar>(0), k2, label + "_k");
      gemm<ExecSpace, Scalar>(handle_, static_cast<Scalar>(1), nrm2, L.v_W,
                              static_cast<Scalar>(0), v2, label + "_v");
      Impl::dit_copy_flat<ExecSpace, Scalar>(q4.data(), q2.data(), Sj * D);
      Impl::dit_copy_flat<ExecSpace, Scalar>(k4.data(), k2.data(), Sj * D);
      Impl::dit_copy_flat<ExecSpace, Scalar>(vv3.data(), v2.data(), Sj * D);
      MossImage::moss_qk_rmsnorm_forward<ExecSpace, Scalar>(q4, L.qn, kNormEps,
                                                            q4, label + "_qn");
      MossImage::moss_qk_rmsnorm_forward<ExecSpace, Scalar>(k4, L.kn, kNormEps,
                                                            k4, label + "_kn");
      MossImage::moss_rope_apply_forward<ExecSpace, Scalar>(q4, rope, q4,
                                                            label + "_rq");
      MossImage::moss_rope_apply_forward<ExecSpace, Scalar>(k4, rope, k4,
                                                            label + "_rk");
      attention_mha_bidirectional<ExecSpace, Scalar>(
          handle_,
          Kokkos::subview(q4, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
          Kokkos::subview(k4, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
          vv3, ao3, attn_scale, label + "_attn");
      Impl::dit_copy_flat<ExecSpace, Scalar>(ao2.data(), ao3.data(), Sj * D);
      gemm<ExecSpace, Scalar>(handle_, static_cast<Scalar>(1), ao2, L.o_W,
                              static_cast<Scalar>(0), prao, label + "_o");
      Impl::dit_copy_flat<ExecSpace, Scalar>(prao3.data(), prao.data(), Sj * D);
      MossImage::moss_gate_residual_forward<ExecSpace, Scalar>(
          joint, prao3, gt_a1, joint, label + "_res");
      MossImage::moss_rmsnorm_modulate_forward<ExecSpace, Scalar>(
          joint, L.norm2, kNormEps, sh_f1, sc_f1, nrm3, label + "_norm2");
      gemm<ExecSpace, Scalar>(handle_, static_cast<Scalar>(1), nrm2,
                              L.up_gate_W, static_cast<Scalar>(0), ug,
                              label + "_ug");
      Impl::dit_split2<ExecSpace, Scalar>(ug, fh, fg, label + "_ugs");
      MossImage::moss_situ_forward<ExecSpace, Scalar>(fh, fg, fh,
                                                      label + "_situ");
      gemm<ExecSpace, Scalar>(handle_, static_cast<Scalar>(1), fh, L.down_W,
                              static_cast<Scalar>(0), fd, label + "_down");
      Impl::dit_copy_flat<ExecSpace, Scalar>(fd3.data(), fd.data(), Sj * D);
      MossImage::moss_gate_residual_forward<ExecSpace, Scalar>(
          joint, fd3, gt_f1, joint, label + "_res2");
    }
  }
};

}  // namespace MossImageDit
}  // namespace KokkosDNN

#endif
