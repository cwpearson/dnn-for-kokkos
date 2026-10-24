// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21DIT_HPP
#define KOKKOSDNN_QWENIMAGE21DIT_HPP

// Qwen-Image-2.1-Turbo flow transformer (DiT) weights + forward skeleton +
// FlowMatchEuler sampler, mirroring the models/moss_image structural pattern
// (Config, Weights with device Views + release() + resident_bytes() +
// load_safetensors with chunked host->device streaming, Model with
// grow-only scratch). TEXT-TO-IMAGE ONLY (no image-condition path).
//
// Checkpoint layout (all `.weight`, NO biases; torch [out, in] row-major,
// matching the [N, K] GEMM operand layout so file bytes land verbatim):
//   transformer_blocks.{0..31}.attn.{to_q,to_k,to_v,to_out.0}  [D, D]
//   transformer_blocks.{i}.attn.{norm_q,norm_k}                 [Hd]
//   transformer_blocks.{i}.img_mlp.{proj,gate_layer}           [H, D]
//   transformer_blocks.{i}.img_mlp.out                        [D, H]
//   txt_in.{text_norm}            [Td]  (ZeroCenterRMSNorm weight)
//   txt_in.{in_layer,out_layer}   [D, Td] / [D, D]
//   img_in                        [D, C]
//   time_text_embed.timestep_embedder.{linear_1 [D, F], linear_2 [D, D]}
//   modulation.1                  [4D, D] (ONE shared modulation)
//   norm_out.linear               [D, D] (temb -> scale for final AdaLN)
//   proj_out                      [C, D]
// (D = 4096, Hd = 128, H = gate_dim = mlp_dim, F = 256, C = 64,
// Td = context_dim.) Sharded over 2 safetensors files.
//
// Dispatch contract (no backend-gated code anywhere here):
//   * Every Linear goes through KokkosDNN::gemm (cuBLAS/hipBLAS ladder).
//   * Joint attention is exact prefill segmentation mirroring
//     QwenImage21AttnProcessor: the text prefix [0, S) runs causal
//     attention_gqa_single (q_off = 0), the target rows [S, T) run full
//     attention_mha_bidirectional over all keys (cuDNN SDPA ladder, else
//     native streaming kernels). Pure T2I has a single text segment, so no
//     per-segment loop is needed.
//   * Block norms are affine-free LayerNorm (layernorm_noaffine_forward);
//     RMSNorm survives only where the reference has it (QK norm_q/norm_k,
//     ZeroCenterRMSNorm in txt_in).
//   * SiLU / AdaLN scales go through the core entry points (silu_forward,
//     layernorm_noaffine_forward).
//   * The only device code added here (Impl::qwen21dit_* helpers) is
//     small elementwise/layout glue with no TPL equivalent: zero-centered
//     norm, GELU-tanh, SwiGLU multiply, per-head QK RMSNorm, 3-axis RoPE
//     with text-shared positions + zero-centered image grids, per-region
//     scale/gate application, BCHW<->row transposes, column splits.
// Forward() is shape-correct and follows the reference numerics lane by
// lane (see GOLDEN notes below); KV-cache decode is out of scope.

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cmath>
#include <cstddef>
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
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_LayerNorm.hpp"
#include "KokkosDNN_RMSNorm.hpp"

#include "KokkosDNN_QwenImage21Config.hpp"

namespace KokkosDNN {
namespace QwenImage21Dit {

// One expected checkpoint tensor: bare (suffix-matched) name + shape.
// Matrices are torch [out, in] row-major, matching the [N, K] GEMM operand
// layout, so file bytes land verbatim (no host transpose).
struct DitTensorSpec {
  std::string name;
  int64_t d0 = 0;  // rows (matrices) or length (vectors)
  int64_t d1 = 1;  // cols (matrices) or 1 (vectors)
  bool matrix = false;
};

// Canonical weight list for a Config (297 tensors at full size:
// 32 blocks x 9 + 9 globals). Rope tables are derived constants rebuilt
// from rope_theta, not weights.
inline std::vector<DitTensorSpec> qwen21_dit_expected_tensors(
    const QwenImage21::Config& cfg) {
  cfg.validate();
  const int64_t D = cfg.hidden_dim, G = cfg.gate_dim(), F = 256,
                C = cfg.latent_channels, Td = cfg.context_dim,
                Hd = cfg.head_dim();
  std::vector<DitTensorSpec> ts;
  auto mat = [&](const std::string& n, int64_t r, int64_t c) {
    ts.push_back({n, r, c, true});
  };
  auto vec = [&](const std::string& n, int64_t r) {
    ts.push_back({n, r, 1, false});
  };
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string p =
        "transformer_blocks." + std::to_string(i) + ".attn.";
    mat(p + "to_q.weight", D, D);
    mat(p + "to_k.weight", D, D);
    mat(p + "to_v.weight", D, D);
    mat(p + "to_out.0.weight", D, D);
    vec(p + "norm_q.weight", Hd);
    vec(p + "norm_k.weight", Hd);
    const std::string m =
        "transformer_blocks." + std::to_string(i) + ".img_mlp.";
    mat(m + "proj.weight", G, D);
    mat(m + "gate_layer.weight", G, D);
    mat(m + "out.weight", D, G);
  }
  vec("txt_in.text_norm.weight", Td);
  mat("txt_in.in_layer.weight", D, Td);
  mat("txt_in.out_layer.weight", D, D);
  mat("img_in.weight", D, C);
  mat("time_text_embed.timestep_embedder.linear_1.weight", D, F);
  mat("time_text_embed.timestep_embedder.linear_2.weight", D, D);
  mat("modulation.1.weight", 4 * D, D);
  mat("norm_out.linear.weight", D, D);
  mat("proj_out.weight", C, D);
  return ts;
}

// Turbo default sigmas (FlowMatchEuler, 8 steps).
inline std::vector<double> qwen21_turbo_sigmas() {
  return {1.0, 0.978453, 0.95418, 0.926626,
          0.89508, 0.845148, 0.704534, 0.414568};
}

// SD3-style static time shift of one sigma: s' = sh*s / (1 + (sh-1)*s).
// Dynamic-mu selection for (max_shift, max_seq_len, image_seq_len) is a
// TODO (golden scheduler test); pass shift = 1.0 for no shift.
inline double qwen21_time_shift(double sigma, double shift) {
  if (!(shift > 0.0) || !std::isfinite(shift))
    throw std::invalid_argument("qwen21_time_shift: shift must be > 0");
  if (!std::isfinite(sigma) || sigma < 0.0)
    throw std::invalid_argument("qwen21_time_shift: sigma bad");
  return shift * sigma / (1.0 + (shift - 1.0) * sigma);
}

namespace Impl {

// ---- layout-only device glue (no arithmetic beyond elementwise) ----

// ZeroCenterRMSNorm (matches QwenImage21ZeroCenterRMSNorm, whose stored
// weight is scale - 1, i.e. the effective scale is weight + 1, in fp32):
//   y = x / sqrt(mean(x^2) + eps) * (w + 1).
// NOTE: this is a plain RMSNorm with a shifted weight, NOT a mean-centered
// norm; the reference subtracts no mean.
template <class ExecSpace, class Scalar>
void qwen21dit_zero_center_norm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::qwen21dit_zcnorm") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (y.extent(0) != R || y.extent(1) != D || w.extent(0) != D)
    throw std::invalid_argument("qwen21dit_zero_center_norm: dim mismatch");
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(i, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrtf(ss / (float)D + eps);
        y(i, j) = static_cast<Scalar>(static_cast<float>(x(i, j)) * inv *
                                      (static_cast<float>(w(j)) + 1.0f));
      });
}

// In-place GELU-tanh: y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
template <class ExecSpace, class Scalar>
void qwen21dit_gelu_tanh_inplace(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::qwen21dit_gelu") {
  const size_t R = y.extent(0), C = y.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float v = static_cast<float>(y(i, j));
        float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        y(i, j) = static_cast<Scalar>(0.5f * v * (1.0f + tanhf(c)));
      });
}

// SwiGLU multiply: out = silu(gate) * up, elementwise over [R, G].
template <class ExecSpace, class Scalar>
void qwen21dit_swiglu(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::qwen21dit_swiglu") {
  const size_t R = gate.extent(0), G = gate.extent(1);
  if (up.extent(0) != R || up.extent(1) != G || out.extent(0) != R ||
      out.extent(1) != G)
    throw std::invalid_argument("qwen21dit_swiglu: dim mismatch");
  if (R == 0 || G == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, G}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float g = static_cast<float>(gate(i, j));
        float s = g / (1.0f + expf(-g));
        out(i, j) =
            static_cast<Scalar>(s * static_cast<float>(up(i, j)));
      });
}

// Per-head QK RMSNorm over Dh with weight w ([Dh]), in place on [T, H, Dh].
template <class ExecSpace, class Scalar, class V3>
void qwen21dit_qk_rmsnorm(
    const V3& qk, const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const std::string& label = "KokkosDNN::qwen21dit_qknorm") {
  const size_t T = qk.extent(0), H = qk.extent(1), Dh = qk.extent(2);
  if (w.extent(0) != Dh)
    throw std::invalid_argument("qwen21dit_qk_rmsnorm: dim mismatch");
  if (T == 0 || H == 0 || Dh == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < Dh; ++d) {
          float v = static_cast<float>(qk(t, h, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrtf(ss / (float)Dh + eps);
        for (size_t d = 0; d < Dh; ++d)
          qk(t, h, d) = static_cast<Scalar>(static_cast<float>(qk(t, h, d)) *
                        inv * static_cast<float>(w(d)));
      });
}

// 3-axis RoPE over head_dim Dh = a0 + a1 + a2, in place on [T, H, Dh],
// matching QwenImage21Rope for pure T2I (one text prefix + target block):
//   * text rows [0, S) advance a SHARED position on all three axes:
//     p0 = p1 = p2 = row index r.
//   * target rows [S, T) freeze the frame axis at the position reached by
//     the preceding text (p0 = S) and lay tokens out on height/width grids
//     centred on zero: p1 = h - (H - H/2), p2 = w - (W - W/2) with
//     j = r - S, h = j / W, w = j % W (negative indices are negative
//     angles, matching the reference neg_index freq table).
// Frequencies are theta^(-2k/ds) per section (theta from Config, 10000 per
// the reference), i.e. rotation by angle p * theta^(-j/ds) on pair j.
template <class ExecSpace, class Scalar, class V3>
void qwen21dit_rope_3axis(
    const V3& x, size_t text_len, size_t grid_h, size_t grid_w, int a0, int a1,
    int a2, double theta,
    const std::string& label = "KokkosDNN::qwen21dit_rope") {
  const size_t T = x.extent(0), H = x.extent(1), Dh = x.extent(2);
  if ((size_t)(a0 + a1 + a2) != Dh)
    throw std::invalid_argument("qwen21dit_rope_3axis: axes/head_dim mismatch");
  if (T == 0 || H == 0 || Dh == 0) return;
  if (grid_h == 0 || grid_w == 0)
    throw std::invalid_argument("qwen21dit_rope_3axis: grid dims must be > 0");
  if (text_len + grid_h * grid_w != T)
    throw std::invalid_argument("qwen21dit_rope_3axis: S + H*W must equal T");
  const long h_off = (long)(grid_h - grid_h / 2);
  const long w_off = (long)(grid_w - grid_w / 2);
  const long frame = (long)text_len;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        long p0, p1, p2;
        if (t < text_len) {
          p0 = p1 = p2 = (long)t;
        } else {
          const size_t j = t - text_len;
          const long hh = (long)(j / grid_w);
          const long ww = (long)(j % grid_w);
          p0 = frame;
          p1 = hh - h_off;
          p2 = ww - w_off;
        }
        const long ps[3] = {p0, p1, p2};
        const int as[3] = {a0, a1, a2};
        size_t base = 0;
        for (int s = 0; s < 3; ++s) {
          const int ds = as[s];
          for (int j = 0; j < ds; j += 2) {
            float inv =
                powf((float)theta, -((float)j) / (float)ds);
            float ang = (float)ps[s] * inv;
            float co = cosf(ang), si = sinf(ang);
            float x0 = static_cast<float>(x(t, h, base + (size_t)j));
            float x1 = static_cast<float>(x(t, h, base + (size_t)j + 1));
            x(t, h, base + (size_t)j) = static_cast<Scalar>(x0 * co - x1 * si);
            x(t, h, base + (size_t)j + 1) =
                static_cast<Scalar>(x0 * si + x1 * co);
          }
          base += (size_t)ds;
        }
      });
}

// Scale application: y = x * (1 + scale), row-broadcast over [R, D].
template <class ExecSpace, class Scalar>
void qwen21dit_apply_scale(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& scale,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::qwen21dit_scale") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (y.extent(0) != R || y.extent(1) != D || scale.extent(0) != D)
    throw std::invalid_argument("qwen21dit_apply_scale: dim mismatch");
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        y(i, j) = static_cast<Scalar>(static_cast<float>(x(i, j)) *
                                      (1.0f + static_cast<float>(scale(j))));
      });
}

// Per-region scale application for causal_condition modulation:
// rows [0, split) use s_txt (the t=0 row), rows [split, R) use s_img
// (the sampled-timestep row). Matches _select_modulation_rows with
// target_token_mask (text takes the trailing t=0 row).
template <class ExecSpace, class Scalar>
void qwen21dit_apply_scale_split(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& s_txt,
    const Kokkos::View<Scalar*, ExecSpace>& s_img, size_t split,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::qwen21dit_scale_split") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (y.extent(0) != R || y.extent(1) != D || s_txt.extent(0) != D ||
      s_img.extent(0) != D || split > R)
    throw std::invalid_argument("qwen21dit_apply_scale_split: dim mismatch");
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float s = (i < split) ? static_cast<float>(s_txt(j))
                              : static_cast<float>(s_img(j));
        y(i, j) = static_cast<Scalar>(static_cast<float>(x(i, j)) * (1.0f + s));
      });
}
// Tanh-gated residual: x = x + tanh(gate) * upd, gate [D] broadcast.
template <class ExecSpace, class Scalar>
void qwen21dit_gated_residual(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& upd,
    const Kokkos::View<Scalar*, ExecSpace>& gate,
    const std::string& label = "KokkosDNN::qwen21dit_gate") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (upd.extent(0) != R || upd.extent(1) != D || gate.extent(0) != D)
    throw std::invalid_argument("qwen21dit_gated_residual: dim mismatch");
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float g = tanhf(static_cast<float>(gate(j)));
        x(i, j) = static_cast<Scalar>(static_cast<float>(x(i, j)) +
                                      g * static_cast<float>(upd(i, j)));
      });
}

// Per-region tanh-gated residual for causal_condition modulation:
// rows [0, split) use g_txt (the t=0 row), rows [split, R) use g_img.
template <class ExecSpace, class Scalar>
void qwen21dit_gated_residual_split(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& upd,
    const Kokkos::View<Scalar*, ExecSpace>& g_txt,
    const Kokkos::View<Scalar*, ExecSpace>& g_img, size_t split,
    const std::string& label = "KokkosDNN::qwen21dit_gate_split") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (upd.extent(0) != R || upd.extent(1) != D || g_txt.extent(0) != D ||
      g_img.extent(0) != D || split > R)
    throw std::invalid_argument(
        "qwen21dit_gated_residual_split: dim mismatch");
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float g = tanhf((i < split) ? static_cast<float>(g_txt(j))
                                    : static_cast<float>(g_img(j)));
        x(i, j) = static_cast<Scalar>(static_cast<float>(x(i, j)) +
                                      g * static_cast<float>(upd(i, j)));
      });
}

// FlowMatchEuler step: x = x + (s_to - s_from) * v.
template <class ExecSpace, class Scalar>
void qwen21dit_euler_step(
    const Kokkos::View<Scalar*, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& v, double s_from, double s_to,
    const std::string& label = "KokkosDNN::qwen21dit_euler") {
  if (x.extent(0) != v.extent(0))
    throw std::invalid_argument("qwen21dit_euler_step: dim mismatch");
  const size_t N = x.extent(0);
  if (N == 0) return;
  const float ds = (float)(s_to - s_from);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t i) {
        x(i) = static_cast<Scalar>(static_cast<float>(x(i)) +
                                   ds * static_cast<float>(v(i)));
      });
}

}  // namespace Impl

// ---- per-block weights (device Views in Scalar) ----

template <class ExecSpace, class Scalar>
struct QwenImage21DitBlockWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> to_q, to_k, to_v,
      to_o;                                                     // [D, D] x4
  Kokkos::View<Scalar*, ExecSpace> norm_q, norm_k;              // [Hd]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> proj;  // [G, D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> gate;  // [G, D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> out;   // [D, G]
};

// Full transformer weights. BF16-capable: Scalar is float or bhalf_t.
template <class ExecSpace, class Scalar>
struct QwenImage21DitWeights {
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;

  QwenImage21::Config cfg;
  V1 txt_norm;    // [Td] ZeroCenterRMSNorm weight
  V2 txt_in_W;    // [D, Td]
  V2 txt_out_W;   // [D, D]
  V2 img_in_W;    // [D, C]
  V2 time1_W;     // [D, F=256]
  V2 time2_W;     // [D, D]
  V2 mod_W;       // [4D, D] shared modulation
  V2 norm_out_W;  // [D, D] temb -> scale (AdaLayerNormContinuous)
  V2 proj_out_W;  // [C, D]
  std::vector<QwenImage21DitBlockWeights<ExecSpace, Scalar>> blocks;

  void allocate(const QwenImage21::Config& c) {
    c.validate();
    cfg = c;
    const size_t D = (size_t)c.hidden_dim, G = (size_t)c.gate_dim(),
                 Td = (size_t)c.context_dim, C = (size_t)c.latent_channels,
                 Hd = (size_t)c.head_dim();
    constexpr size_t F = 256;  // timestep sinusoidal dim
    auto mat = [&](const char* lb, size_t r, size_t k) {
      return V2(lb, r, k);
    };
    auto vec = [&](const char* lb, size_t n) { return V1(lb, n); };
    txt_norm = vec("qwen21_txt_norm", Td);
    txt_in_W = mat("qwen21_txt_in", D, Td);
    txt_out_W = mat("qwen21_txt_out", D, D);
    img_in_W = mat("qwen21_img_in", D, C);
    time1_W = mat("qwen21_time1", D, F);
    time2_W = mat("qwen21_time2", D, D);
    mod_W = mat("qwen21_mod", 4 * D, D);
    norm_out_W = mat("qwen21_norm_out", D, D);
    proj_out_W = mat("qwen21_proj_out", C, D);
    blocks.resize((size_t)c.num_layers);
    for (auto& b : blocks) {
      b.to_q = mat("qwen21_to_q", D, D);
      b.to_k = mat("qwen21_to_k", D, D);
      b.to_v = mat("qwen21_to_v", D, D);
      b.to_o = mat("qwen21_to_o", D, D);
      b.norm_q = vec("qwen21_nq", Hd);
      b.norm_k = vec("qwen21_nk", Hd);
      b.proj = mat("qwen21_mlp_proj", G, D);
      b.gate = mat("qwen21_mlp_gate", G, D);
      b.out = mat("qwen21_mlp_out", D, G);
    }
  }

  // Drop every device allocation (assigning {} releases the View reference;
  // clears the block vector so resident bytes return to ~0).
  void release() {
    txt_norm = {};
    txt_in_W = {};
    txt_out_W = {};
    img_in_W = {};
    time1_W = {};
    time2_W = {};
    mod_W = {};
    norm_out_W = {};
    proj_out_W = {};
    blocks.clear();
    blocks.shrink_to_fit();
  }

  // Resident device bytes currently held (rank sizes times sizeof(Scalar);
  // 0 for released views). Host-side estimate only.
  size_t resident_bytes() const {
    size_t n = 0;
    auto acc2 = [&](const auto& v) {
      n += v.extent(0) * v.extent(1);
    };
    auto acc1 = [&](const auto& v) { n += v.extent(0); };
    acc1(txt_norm);
    acc2(txt_in_W);
    acc2(txt_out_W);
    acc2(img_in_W);
    acc2(time1_W);
    acc2(time2_W);
    acc2(mod_W);
    acc2(norm_out_W);
    acc2(proj_out_W);
    for (const auto& b : blocks) {
      acc2(b.to_q);
      acc2(b.to_k);
      acc2(b.to_v);
      acc2(b.to_o);
      acc1(b.norm_q);
      acc1(b.norm_k);
      acc2(b.proj);
      acc2(b.gate);
      acc2(b.out);
    }
    return n * sizeof(Scalar);
  }

  // Deterministic small-random fill for shape/forward tests without a
  // checkpoint (norms are ones; needs allocate() first).
  void random_init(uint64_t seed = 42) {
    if (img_in_W.extent(0) == 0)
      throw std::runtime_error("QwenImage21DitWeights::random_init: allocate?");
    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    auto rnd2 = [&](auto& v) {
      auto h = Kokkos::create_mirror_view(Kokkos::HostSpace(), v);
      for (size_t i = 0; i < v.extent(0); ++i)
        for (size_t j = 0; j < v.extent(1); ++j)
          h(i, j) = static_cast<Scalar>(dist(eng));
      Kokkos::deep_copy(v, h);
    };
    auto ones1 = [&](auto& v) {
      Kokkos::deep_copy(v, static_cast<Scalar>(1.0f));
    };
    rnd2(txt_in_W);
    rnd2(txt_out_W);
    rnd2(img_in_W);
    rnd2(time1_W);
    rnd2(time2_W);
    rnd2(mod_W);
    rnd2(norm_out_W);
    rnd2(proj_out_W);
    ones1(txt_norm);
    for (auto& b : blocks) {
      rnd2(b.to_q);
      rnd2(b.to_k);
      rnd2(b.to_v);
      rnd2(b.to_o);
      rnd2(b.proj);
      rnd2(b.gate);
      rnd2(b.out);
      ones1(b.norm_q);
      ones1(b.norm_k);
    }
  }

  // Load the REAL checkpoint layout (possibly sharded across several files;
  // the combined shard set must cover every expected tensor). Tensor names
  // are suffix-matched (training/export prefixes allowed). File dtypes F32
  // and BF16 are accepted (converted to Scalar on the host). Bytes stream
  // per tensor through bounded HostSpace chunks into a host mirror with a
  // single deep_copy per tensor (no 2x device residency). Throws on missing
  // tensors, unknown tensor names, non-F32/BF16 dtypes, or shape/numel/
  // byte-size mismatch. Needs allocate() first.
  void load_safetensors(const std::vector<std::string>& shard_paths) {
    if (shard_paths.empty())
      throw std::invalid_argument(
          "QwenImage21DitWeights::load_safetensors: no shards given");
    if (img_in_W.extent(0) == 0)
      throw std::runtime_error(
          "QwenImage21DitWeights::load_safetensors: allocate first");
    static_assert(std::is_same_v<Scalar, float> ||
                      std::is_same_v<Scalar,
                                     Kokkos::Experimental::bhalf_t>,
                  "QwenImage21DitWeights::load_safetensors: Scalar must be "
                  "float or bhalf_t");
    const std::vector<DitTensorSpec> specs =
        qwen21_dit_expected_tensors(cfg);
    std::map<std::string, bool> found;
    for (const auto& s : specs) found[s.name] = false;

    for (const std::string& path : shard_paths) {
      std::ifstream f;
      auto entries = QwenImage21::Impl::qwen21_open_shard(f, path);
      for (const auto& kv : entries) {
        const std::string& full = kv.first;
        const QwenImage21::Impl::Qwen21SafeEntry& en = kv.second;
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
              "QwenImage21DitWeights::load_safetensors: unknown tensor '" +
              full + "' in '" + path + "'");
        if (found[spec->name])
          throw std::runtime_error(
              "QwenImage21DitWeights::load_safetensors: duplicate tensor '" +
              spec->name + "'");
        load_one_tensor(f, full, en, *spec);
        found[spec->name] = true;
      }
    }
    for (const auto& s : specs) {
      if (!found[s.name])
        throw std::runtime_error(
            "QwenImage21DitWeights::load_safetensors: missing tensor '" +
            s.name + "'");
    }
  }

  void load_safetensors(const std::string& shard_path) {
    load_safetensors(std::vector<std::string>{shard_path});
  }

 private:
  // Stream one file tensor into its device view (shape-checked).
  void load_one_tensor(std::ifstream& f, const std::string& full,
                       const QwenImage21::Impl::Qwen21SafeEntry& en,
                       const DitTensorSpec& s) {
    const size_t elsize = QwenImage21::Impl::qwen21_entry_elsize(en, full);
    const int64_t numel = QwenImage21::Impl::qwen21_entry_numel(en, full);
    const size_t want =
        s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
    if ((size_t)numel != want)
      throw std::runtime_error(
          "QwenImage21DitWeights::load_safetensors: tensor '" + full +
          "' has numel " + std::to_string(numel) + ", expected " +
          std::to_string(want) + " (" + s.name + ")");
    if (en.end < en.begin || en.end - en.begin != want * elsize)
      throw std::runtime_error(
          "QwenImage21DitWeights::load_safetensors: tensor '" + full +
          "' slice size mismatch");
    V2* dst2 = nullptr;
    V1* dst1 = nullptr;
    resolve_tensor(s.name, dst2, dst1);
    if (s.matrix) {
      if (!dst2 || (*dst2).extent(0) != (size_t)s.d0 ||
          (*dst2).extent(1) != (size_t)s.d1)
        throw std::runtime_error(
            "QwenImage21DitWeights::load_safetensors: view mismatch '" +
            s.name + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          f, en, want, elsize, (*dst2).data());
    } else {
      if (!dst1 || (*dst1).extent(0) != (size_t)s.d0)
        throw std::runtime_error(
            "QwenImage21DitWeights::load_safetensors: view mismatch '" +
            s.name + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          f, en, want, elsize, (*dst1).data());
    }
  }

  void resolve_tensor(const std::string& name, V2*& out2, V1*& out1) {
    out2 = nullptr;
    out1 = nullptr;
    if (name == "txt_in.text_norm.weight") {
      out1 = &txt_norm;
      return;
    }
    if (name == "txt_in.in_layer.weight") {
      out2 = &txt_in_W;
      return;
    }
    if (name == "txt_in.out_layer.weight") {
      out2 = &txt_out_W;
      return;
    }
    if (name == "img_in.weight") {
      out2 = &img_in_W;
      return;
    }
    if (name == "time_text_embed.timestep_embedder.linear_1.weight") {
      out2 = &time1_W;
      return;
    }
    if (name == "time_text_embed.timestep_embedder.linear_2.weight") {
      out2 = &time2_W;
      return;
    }
    if (name == "modulation.1.weight") {
      out2 = &mod_W;
      return;
    }
    if (name == "norm_out.linear.weight") {
      out2 = &norm_out_W;
      return;
    }
    if (name == "proj_out.weight") {
      out2 = &proj_out_W;
      return;
    }
    const std::string tb = "transformer_blocks.";
    if (name.compare(0, tb.size(), tb) == 0) {
      size_t dot = name.find('.', tb.size());
      int bi = std::stoi(name.substr(tb.size(), dot - tb.size()));
      if (bi < 0 || (size_t)bi >= blocks.size())
        throw std::runtime_error(
            "QwenImage21DitWeights::load_safetensors: bad block '" + name +
            "'");
      auto& b = blocks[(size_t)bi];
      std::string tail = name.substr(dot + 1);
      if (tail == "attn.to_q.weight") {
        out2 = &b.to_q;
        return;
      }
      if (tail == "attn.to_k.weight") {
        out2 = &b.to_k;
        return;
      }
      if (tail == "attn.to_v.weight") {
        out2 = &b.to_v;
        return;
      }
      if (tail == "attn.to_out.0.weight") {
        out2 = &b.to_o;
        return;
      }
      if (tail == "attn.norm_q.weight") {
        out1 = &b.norm_q;
        return;
      }
      if (tail == "attn.norm_k.weight") {
        out1 = &b.norm_k;
        return;
      }
      if (tail == "img_mlp.proj.weight") {
        out2 = &b.proj;
        return;
      }
      if (tail == "img_mlp.gate_layer.weight") {
        out2 = &b.gate;
        return;
      }
      if (tail == "img_mlp.out.weight") {
        out2 = &b.out;
        return;
      }
    }
    throw std::runtime_error(
        "QwenImage21DitWeights::load_safetensors: cannot resolve '" + name +
        "'");
  }
};

// Forward scratch: grow-only view cache (see QwenImage21::Impl).
template <class ExecSpace, class Scalar>
struct QwenImage21DitScratch {
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using F2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  QwenImage21::Impl::Qwen21ViewCache<V1> c1;
  QwenImage21::Impl::Qwen21ViewCache<V2> c2;
  QwenImage21::Impl::Qwen21ViewCache<F2> f2;
  size_t bytes() const {
    return c1.bytes(sizeof(Scalar)) + c2.bytes(sizeof(Scalar)) +
           f2.bytes(4);
  }
  void clear() {
    c1.clear();
    c2.clear();
    f2.clear();
  }
};

// Qwen-Image-2.1 DiT model: owns Weights + grow-only scratch. forward()
// follows the reference pipeline stage by stage (txt_in projection with
// ZeroCenterRMSNorm, timestep x1000 + shared modulation with the t=0 row
// split, 32 single-stream blocks over the joint [text | target] sequence
// with exact block-causal prefill segmentation, final scale-only
// AdaLayerNormContinuous + proj_out over the target rows).
//
// GOLDEN notes (deviations from the reference, verified against
// transformer_qwenimage21.py):
//   * FIXED: joint layout is [text | target] (was [image | text]); the
//     head slices the target rows [S, T) for proj_out. The driver keeps
//     passing (img, context) separately, so NO driver change is needed
//     for the layout; only the internal concatenation order changed.
//   * FIXED: block input norms + final norm are affine-free LayerNorm
//     (layernorm_noaffine_forward), matching img_norm1/img_norm2/norm_out.
//   * FIXED: ZeroCenterRMSNorm is RMS (no mean subtraction) with effective
//     scale (w + 1), matching QwenImage21ZeroCenterRMSNorm.
//   * FIXED: timestep is scaled by time_factor 1000 BEFORE the sinusoid
//     ([cos|sin] halves, 256-d), matching QwenImage21TemporalTimesteps.
//     CONTRACT NOTE: forward() therefore takes the RAW [0, 1] sigma as
//     `timestep`; the driver currently passes 1000*sigma (tvals), so it
//     must divide by 1000 at the call site (follow-up, NOT done here).
//   * FIXED: modulation lanes are (attn_scale, attn_gate, mlp_scale,
//     mlp_gate) = mod1.scale, mod1.gate, mod2.scale, mod2.gate, matching
//     the reference chunk order; text rows read the t=0 row, target rows
//     the sampled row (causal_condition split via _select_modulation_rows
//     semantics, implemented as per-region scale/gate application).
//   * FIXED: 3-axis RoPE matches QwenImage21Rope for T2I (text advances a
//     shared position on all axes; target frame frozen at S with
//     zero-centered H/W grids; theta 10000, confirmed by the reference).
//   * FIXED: block-causal mask is exact for T2I: text prefix runs causal
//     attention_gqa_single, target rows run full
//     attention_mha_bidirectional over all keys — the two-segment form of
//     QwenImage21AttnProcessor (single text segment + target block).
//   * OPEN (decode-mode parity): no KV cache; cached decode steps are
//     future work. Prefill (step 0, no-cache) is the parity target.
//   * OPEN: fp16 clip to [-65504, 65504] is skipped (Scalar here is float
//     or bhalf_t, whose max is 65504 already).
template <class ExecSpace, class Scalar>
class QwenImage21DitModel {
 public:
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using V3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using Weights = QwenImage21DitWeights<ExecSpace, Scalar>;

  explicit QwenImage21DitModel(const QwenImage21::Config& cfg = QwenImage21::Config()) {
    cfg.validate();
    cfg_ = cfg;
    weights_.allocate(cfg_);
  }

  const QwenImage21::Config& config() const { return cfg_; }
  Weights& weights() { return weights_; }
  const Weights& weights() const { return weights_; }
  size_t scratch_bytes() const { return scratch_.bytes(); }
  void release_scratch() { scratch_.clear(); }

  // DiT forward: img [B, N, C] latent tokens + context [B, S, Td] encoder
  // embeddings + scalar timestep (RAW [0, 1] sigma; scaled by time_factor
  // 1000 internally) -> out [B, N, C] velocity. latent_h/w default to cfg
  // (N must equal latent_h * latent_w). TEXT-TO-IMAGE ONLY: the joint
  // sequence is [text (S rows) | target (N rows)], T = S + N; proj_out
  // reads the target rows [S, T).
  void forward(const DNNHandle<ExecSpace>& handle, const V3& img,
               const V3& context, double timestep, V3& out,
               size_t latent_h = 0, size_t latent_w = 0) {
    cfg_.validate();
    const size_t B = img.extent(0), N = img.extent(1),
                 C = img.extent(2), S = context.extent(1),
                 Td = context.extent(2);
    const size_t D = (size_t)cfg_.hidden_dim, Hd = (size_t)cfg_.head_dim(),
                 H = (size_t)cfg_.num_heads, G = (size_t)cfg_.gate_dim();
    if (latent_h == 0) latent_h = (size_t)cfg_.latent_h;
    if (latent_w == 0) latent_w = (size_t)cfg_.latent_w;
    if (context.extent(0) != B)
      throw std::invalid_argument("QwenImage21DitModel::forward: batch mismatch");
    if (C != (size_t)cfg_.latent_channels || Td != (size_t)cfg_.context_dim)
      throw std::invalid_argument("QwenImage21DitModel::forward: dim mismatch");
    if (N != latent_h * latent_w || N == 0)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward: N must equal latent_h * latent_w");
    if (out.extent(0) != B || out.extent(1) != N || out.extent(2) != C)
      throw std::invalid_argument("QwenImage21DitModel::forward: out mismatch");
    if (S == 0)
      throw std::invalid_argument("QwenImage21DitModel::forward: empty context");
    if (!cfg_.causal_condition)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward: causal_condition=false unsupported "
          "(t=0 row split required)");
    const size_t T = S + N;
    const float eps = (float)cfg_.norm_eps;
    const float attn_scale =
        1.0f / std::sqrt(static_cast<float>(Hd));  // SDPA default 1/sqrt(E)

    using U2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using U1 = Kokkos::View<Scalar*, ExecSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using U3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Timestep sinusoid on host (float64 phases, 256-d [cos|sin] halves).
    // Reference (QwenImage21TemporalTimesteps): timestep is scaled by
    // time_factor 1000 BEFORE the sinusoid.
    constexpr size_t F = 256;
    std::vector<float> sinus_s(F), sinus_0(F);
    {
      for (size_t i = 0; i < F / 2; ++i) {
        const double inv =
            std::exp(-std::log(10000.0) * (double)i / (double)(F / 2));
        const double a_s = 1000.0 * timestep * inv;
        sinus_s[i] = (float)std::cos(a_s);
        sinus_s[i + F / 2] = (float)std::sin(a_s);
        sinus_0[i] = 1.0f;  // t = 0: cos(0) = 1 ...
        sinus_0[i + F / 2] = 0.0f;  // ... sin(0) = 0
      }
    }

    for (size_t b = 0; b < B; ++b) {
      U2 img_b(const_cast<Scalar*>(img.data()) + b * N * C, N, C);
      U2 ctx_b(const_cast<Scalar*>(context.data()) + b * S * Td, S, Td);
      U2 out_b(out.data() + b * N * C, N, C);

      // --- txt_in: ZeroCenterRMSNorm -> Linear -> GELU-tanh -> Linear ---
      V2 txt = scratch_.c2.get("txt", S, D);
      V2 tmpT = scratch_.c2.get("tmpT", S, (Td > D ? Td : D));
      {
        U2 tmpT_td(tmpT.data(), S, Td);
        Impl::qwen21dit_zero_center_norm<ExecSpace, Scalar>(
            ctx_b, weights_.txt_norm, eps, tmpT_td);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), tmpT_td,
                         weights_.txt_in_W, static_cast<Scalar>(0.0f), txt);
        Impl::qwen21dit_gelu_tanh_inplace<ExecSpace, Scalar>(txt);
        U2 tmpT_d(tmpT.data(), S, D);
        Kokkos::deep_copy(tmpT_d, txt);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), tmpT_d,
                         weights_.txt_out_W, static_cast<Scalar>(0.0f), txt);
      }

      // --- img_in: Linear C -> D ---
      V2 img_t = scratch_.c2.get("img", N, D);
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), img_b,
                       weights_.img_in_W, static_cast<Scalar>(0.0f), img_t);

      // --- timestep MLP (TimestepEmbedder: Linear -> SiLU -> Linear) ---
      // Two rows: sampled timestep + t = 0 (causal_condition). Text rows
      // modulate from the t=0 row, target rows from the sampled row.
      V2 s3 = scratch_.c2.get("s3", 2, D);
      {
        V2 s1 = scratch_.c2.get("s1", 2, F);
        V2 s2 = scratch_.c2.get("s2", 2, D);
        auto hs = Kokkos::create_mirror_view(Kokkos::HostSpace(), s1);
        for (size_t i = 0; i < F; ++i) {
          hs(0, i) = static_cast<Scalar>(sinus_s[i]);
          hs(1, i) = static_cast<Scalar>(sinus_0[i]);
        }
        Kokkos::deep_copy(s1, hs);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), s1, weights_.time1_W,
                         static_cast<Scalar>(0.0f), s2);
        KokkosDNN::silu_forward<ExecSpace, Scalar>(s2, s2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), s2, weights_.time2_W,
                         static_cast<Scalar>(0.0f), s3);
      }

      // --- shared modulation: SiLU(temb rows) -> Linear -> 2 x 4[D] ---
      // Row 0 = sampled timestep (target rows), row 1 = t=0 (text rows).
      // Lane order matches the reference chunking: mod1 = (attn_scale,
      // attn_gate), mod2 = (mlp_scale, mlp_gate).
      V2 mod4 = scratch_.c2.get("mod4", 2, 4 * D);
      U1 a_s, g1_s, b_s, g2_s, a_z, g1_z, b_z, g2_z;
      {
        V2 t1 = scratch_.c2.get("t1", 2, D);
        Kokkos::deep_copy(t1, s3);
        KokkosDNN::silu_forward<ExecSpace, Scalar>(t1, t1);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), t1, weights_.mod_W,
                         static_cast<Scalar>(0.0f), mod4);
        Scalar* mp = mod4.data();
        a_s = U1(mp, D);
        g1_s = U1(mp + D, D);
        b_s = U1(mp + 2 * D, D);
        g2_s = U1(mp + 3 * D, D);
        Scalar* mz = mp + 4 * D;
        a_z = U1(mz, D);
        g1_z = U1(mz + D, D);
        b_z = U1(mz + 2 * D, D);
        g2_z = U1(mz + 3 * D, D);
      }

      // --- joint sequence [text | target] ---
      V2 joint = scratch_.c2.get("joint", T, D);
      {
        Kokkos::parallel_for(
            "KokkosDNN::qwen21dit_concat",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
            KOKKOS_LAMBDA(size_t r, size_t d) {
              joint(r, d) = (r < S) ? txt(r, d) : img_t(r - S, d);
            });
      }
      V2 hn = scratch_.c2.get("hn", T, D);
      V2 q2 = scratch_.c2.get("q", T, D);
      V2 k2 = scratch_.c2.get("k", T, D);
      V2 v2 = scratch_.c2.get("v", T, D);
      V2 o2 = scratch_.c2.get("o", T, D);
      V2 p2 = scratch_.c2.get("p", T, G);
      V2 g2 = scratch_.c2.get("g", T, G);
      V2 h2 = scratch_.c2.get("h", T, G);

      for (size_t li = 0; li < weights_.blocks.size(); ++li) {
        const auto& bw = weights_.blocks[li];
        // Attention path: affine-free LayerNorm, then per-region scale
        // (text rows <- t=0 lane, target rows <- sampled lane).
        KokkosDNN::layernorm_noaffine_forward<ExecSpace, Scalar>(joint, eps,
                                                                 hn);
        Impl::qwen21dit_apply_scale_split<ExecSpace, Scalar>(hn, a_z, a_s, S,
                                                            hn);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_q,
                         static_cast<Scalar>(0.0f), q2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_k,
                         static_cast<Scalar>(0.0f), k2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_v,
                         static_cast<Scalar>(0.0f), v2);
        U3 q_txt(q2.data(), S, H, Hd), k_pre(k2.data(), S, H, Hd),
            v_pre(v2.data(), S, H, Hd), o_txt(o2.data(), S, H, Hd);
        U3 q_tgt(q2.data() + S * H * Hd, N, H, Hd),
            k_full(k2.data(), T, H, Hd), v_full(v2.data(), T, H, Hd),
            o_tgt(o2.data() + S * H * Hd, N, H, Hd);
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(q_txt, bw.norm_q, eps);
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(
            U3(q2.data() + S * H * Hd, N, H, Hd), bw.norm_q, eps);
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(k_pre, bw.norm_k, eps);
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(
            U3(k2.data() + S * H * Hd, N, H, Hd), bw.norm_k, eps);
        U3 q_full(q2.data(), T, H, Hd), k_full3(k2.data(), T, H, Hd);
        Impl::qwen21dit_rope_3axis<ExecSpace, Scalar>(
            q_full, S, latent_h, latent_w, cfg_.rope_axes[0],
            cfg_.rope_axes[1], cfg_.rope_axes[2], cfg_.rope_theta);
        Impl::qwen21dit_rope_3axis<ExecSpace, Scalar>(
            k_full3, S, latent_h, latent_w, cfg_.rope_axes[0],
            cfg_.rope_axes[1], cfg_.rope_axes[2], cfg_.rope_theta);
        // Exact block-causal prefill segmentation (the T2I form of
        // QwenImage21AttnProcessor): the single text segment attends
        // causally over its own prefix; the target block attends over all
        // keys (prefix + its own internally-bidirectional block).
        KokkosDNN::attention_gqa_single<ExecSpace, Scalar>(
            handle, q_txt, k_pre, v_pre, (size_t)0, o_txt, attn_scale,
            "KokkosDNN::qwen21dit_attn_txt");
        KokkosDNN::attention_mha_bidirectional<ExecSpace, Scalar>(
            handle, q_tgt, k_full, v_full, o_tgt, attn_scale,
            "KokkosDNN::qwen21dit_attn_img");
        U2 o2f(o2.data(), T, D);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), o2f, bw.to_o,
                         static_cast<Scalar>(0.0f), hn);
        Impl::qwen21dit_gated_residual_split<ExecSpace, Scalar>(joint, hn,
                                                               g1_z, g1_s, S);
        // SwiGLU FFN path.
        KokkosDNN::layernorm_noaffine_forward<ExecSpace, Scalar>(joint, eps,
                                                                 hn);
        Impl::qwen21dit_apply_scale_split<ExecSpace, Scalar>(hn, b_z, b_s, S,
                                                            hn);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.proj,
                         static_cast<Scalar>(0.0f), p2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.gate,
                         static_cast<Scalar>(0.0f), g2);
        Impl::qwen21dit_swiglu<ExecSpace, Scalar>(g2, p2, h2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), h2, bw.out,
                         static_cast<Scalar>(0.0f), hn);
        Impl::qwen21dit_gated_residual_split<ExecSpace, Scalar>(joint, hn,
                                                               g2_z, g2_s, S);
      }

      // --- head: AdaLayerNormContinuous (temb-conditioned scale) + proj ---
      // Reference: scale = norm_out_linear(silu(temb)); rows selected by
      // target_token_mask (text <- t=0 row, target <- sampled row);
      // joint = layernorm_affinefree(joint) * (1 + scale); proj_out reads
      // the target rows only.
      {
        V2 t1 = scratch_.c2.get("t1n", 2, D);
        Kokkos::deep_copy(t1, s3);
        KokkosDNN::silu_forward<ExecSpace, Scalar>(t1, t1);
        V2 ns2 = scratch_.c2.get("ns2", 2, D);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), t1,
                         weights_.norm_out_W, static_cast<Scalar>(0.0f), ns2);
        Scalar* np = ns2.data();
        U1 ns_s(np, D), ns_z(np + D, D);
        KokkosDNN::layernorm_noaffine_forward<ExecSpace, Scalar>(joint, eps,
                                                                 hn);
        Impl::qwen21dit_apply_scale_split<ExecSpace, Scalar>(hn, ns_z, ns_s,
                                                            S, hn);
      }
      {
        U2 hn_tgt(hn.data() + S * D, N, D);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn_tgt,
                         weights_.proj_out_W, static_cast<Scalar>(0.0f), out_b);
      }
    }
  }

 private:
  QwenImage21::Config cfg_;
  Weights weights_;
  QwenImage21DitScratch<ExecSpace, Scalar> scratch_;
};

}  // namespace QwenImage21Dit
}  // namespace KokkosDNN

#endif
