// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21DIT_HPP
#define KOKKOSDNN_QWENIMAGE21DIT_HPP

// Qwen-Image-2.1-Turbo flow transformer (DiT) weights + forward skeleton +
// FlowMatchEuler sampler, mirroring the models/moss_image structural pattern
// (Config, Weights with device Views + release() + resident_bytes() +
// load_safetensors with chunked host->device streaming, Model with
// grow-only scratch). T2I via forward(); single-reference-image editing via
// forward_cond() ([cond|target] joint sequence, per-block RoPE, block-causal
// segment attention, target-only output slice). Prefill only: no KV-cache.
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
//     QwenImage21AttnProcessor: text segments run causal attention_gqa_single
//     (q_off = segment start), non-target image (cond) segments and the
//     target block run full attention_mha_bidirectional over their visible
//     prefix (cuDNN SDPA ladder, else native streaming kernels). Pure T2I has
//     a single text segment, so no per-segment loop is needed there.
//   * Block norms are affine-free LayerNorm (layernorm_noaffine_forward);
//     RMSNorm survives only where the reference has it (QK norm_q/norm_k,
//     ZeroCenterRMSNorm in txt_in).
//   * SiLU / AdaLN scales go through the core entry points (silu_forward,
//     layernorm_noaffine_forward).
//   * The only device code added here (Impl::qwen21dit_* helpers) is
//     small elementwise/layout glue with no TPL equivalent: zero-centered
//     norm, GELU-tanh, SwiGLU multiply, per-head QK RMSNorm, 3-axis RoPE
//     with text-shared positions + zero-centered image grids (single-block
//     form, or explicit per-row position tables for multi-block edit
//     sequences), per-region scale/gate application (split-point or
//     target-mask form), BCHW<->row transposes, column splits.
// Forward() is shape-correct and follows the reference numerics lane by
// lane (see GOLDEN notes below); KV-cache decode is out of scope.

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
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
// 1D flattened RangePolicy (adjacent threads touch adjacent j: coalesced);
// per-output-element ops are unchanged (bitwise-identical).
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
      label, Kokkos::RangePolicy<ExecSpace>(0, R * G),
      KOKKOS_LAMBDA(size_t n) {
        const size_t i = n / G, j = n % G;
        float g = static_cast<float>(gate(i, j));
        float s = g / (1.0f + expf(-g));
        out(i, j) =
            static_cast<Scalar>(s * static_cast<float>(up(i, j)));
      });
}

// Team functor for qwen21dit_qk_rmsnorm: one team per (t, h) row, with a
// TeamThreadRange reduction over Dh (coalesced: adjacent threads read
// adjacent d) followed by the in-place scale pass in the same launch.
// Replaces the fiber-per-thread MDRange form, whose per-thread sequential
// Dh loop made adjacent threads stride by Dh (~15 GB/s effective); the
// fused affine-free LayerNorm functor below already runs this pattern at
// ~0.2ms/call. The tree reduction sums in a different order than the old
// sequential loop, so this is tolerance-gated, not bitwise-identical.
template <class ExecSpace, class Scalar, class V3, class W1>
struct Qwen21ditQkRmsnormFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  V3 qk;
  W1 w;
  float eps;
  size_t H, Dh;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto X = qk;
    const auto W = w;
    const float E = eps;
    const size_t DD = Dh;
    const size_t idx = static_cast<size_t>(member.league_rank());
    const size_t t = idx / H, h = idx % H;
    float ss = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, DD),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          float v = static_cast<float>(X(t, h, d));
          acc += v * v;
        },
        ss);
    const float inv = 1.0f / sqrtf(ss / (float)DD + E);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, DD), KOKKOS_LAMBDA(size_t d) {
          X(t, h, d) = static_cast<Scalar>(static_cast<float>(X(t, h, d)) *
                                           inv * static_cast<float>(W(d)));
        });
  }
};

// Per-head QK RMSNorm over Dh with weight w ([Dh]), in place on [T, H, Dh].
template <class ExecSpace, class Scalar, class V3>
void qwen21dit_qk_rmsnorm(
    const V3& qk, const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const std::string& label = "KokkosDNN::qwen21dit_qknorm") {
  const size_t T = qk.extent(0), H = qk.extent(1), Dh = qk.extent(2);
  if (w.extent(0) != Dh)
    throw std::invalid_argument("qwen21dit_qk_rmsnorm: dim mismatch");
  if (T == 0 || H == 0 || Dh == 0) return;
  Qwen21ditQkRmsnormFunctor<ExecSpace, Scalar, V3,
                            Kokkos::View<Scalar*, ExecSpace>>
      f{qk, w, eps, H, Dh};
  Kokkos::parallel_for(label, Kokkos::TeamPolicy<ExecSpace>(T * H, Kokkos::AUTO),
                       f);
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
// RoPE inv-freq table: inv[p] = theta^(-j/ds) for all (section s, pair j)
// pairs, laid out [s0 pairs][s1 pairs][s2 pairs] (np = Dh/2 entries).
// Computed ONCE per DiT forward on device with the same device powf
// expression as the per-fiber loop below, so the table values are
// bitwise-identical to the hoisted powf calls. Axes must be positive
// even (matches Config::validate).
template <class ExecSpace>
Kokkos::View<float*, ExecSpace> qwen21dit_rope_inv_table(
    int a0, int a1, int a2, double theta,
    const std::string& label = "KokkosDNN::qwen21dit_rope_inv") {
  const int as[3] = {a0, a1, a2};
  size_t base[3] = {0, 0, 0};
  size_t np = 0;
  for (int s = 0; s < 3; ++s) {
    if (as[s] <= 0 || as[s] % 2 != 0)
      throw std::invalid_argument(
          "qwen21dit_rope_inv_table: axes must be positive even numbers");
    base[s] = np;
    np += (size_t)as[s] / 2;
  }
  if (!(theta > 0) || !std::isfinite(theta))
    throw std::invalid_argument(
        "qwen21dit_rope_inv_table: theta must be positive and finite");
  Kokkos::View<float*, ExecSpace> inv(
      Kokkos::ViewAllocateWithoutInitializing(label), np);
  const size_t n0 = base[1], n1 = base[2];
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, np),
      KOKKOS_LAMBDA(size_t p) {
        int ds = a0;
        size_t j = 2 * p;
        if (p >= n1) {
          ds = a2;
          j = 2 * (p - n1);
        } else if (p >= n0) {
          ds = a1;
          j = 2 * (p - n0);
        }
        // Same expression as the per-fiber loop (was evaluated per fiber).
        inv(p) = powf((float)theta, -((float)j) / (float)ds);
      });
  return inv;
}

// Per-(row, head) fiber of the 3-axis RoPE rotation below (shared by the
// single- and dual-tensor launches so the angle math exists exactly once).
// inv is the qwen21dit_rope_inv_table for (a0, a1, a2, theta); angle math
// (pos * inv, cosf/sinf, rotation) is unchanged.
template <class Scalar, class V3>
KOKKOS_INLINE_FUNCTION void qwen21dit_rope_fiber(
    const V3& x, size_t t, size_t h, size_t text_len, size_t grid_w,
    long h_off, long w_off, long frame, int a0, int a1, int a2,
    const float* inv) {
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
  size_t base = 0;   // element offset into the head dim
  size_t pbase = 0;  // pair offset into the inv table
  for (int s = 0; s < 3; ++s) {
    const int ds = as[s];
    for (int j = 0; j < ds; j += 2) {
      float ang = (float)ps[s] * inv[pbase + (size_t)(j / 2)];
      float co = cosf(ang), si = sinf(ang);
      float x0 = static_cast<float>(x(t, h, base + (size_t)j));
      float x1 = static_cast<float>(x(t, h, base + (size_t)j + 1));
      x(t, h, base + (size_t)j) = static_cast<Scalar>(x0 * co - x1 * si);
      x(t, h, base + (size_t)j + 1) = static_cast<Scalar>(x0 * si + x1 * co);
    }
    base += (size_t)ds;
    pbase += (size_t)ds / 2;
  }
}

template <class ExecSpace, class Scalar, class V3>
void qwen21dit_rope_3axis(
    const V3& x, const Kokkos::View<float*, ExecSpace>& inv, size_t text_len,
    size_t grid_h, size_t grid_w, int a0, int a1, int a2,
    const std::string& label = "KokkosDNN::qwen21dit_rope") {
  const size_t T = x.extent(0), H = x.extent(1), Dh = x.extent(2);
  if ((size_t)(a0 + a1 + a2) != Dh)
    throw std::invalid_argument("qwen21dit_rope_3axis: axes/head_dim mismatch");
  if (inv.extent(0) != Dh / 2)
    throw std::invalid_argument("qwen21dit_rope_3axis: inv table dim mismatch");
  if (T == 0 || H == 0 || Dh == 0) return;
  if (grid_h == 0 || grid_w == 0)
    throw std::invalid_argument("qwen21dit_rope_3axis: grid dims must be > 0");
  if (text_len + grid_h * grid_w != T)
    throw std::invalid_argument("qwen21dit_rope_3axis: S + H*W must equal T");
  const long h_off = (long)(grid_h - grid_h / 2);
  const long w_off = (long)(grid_w - grid_w / 2);
  const long frame = (long)text_len;
  const float* invp = inv.data();
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        qwen21dit_rope_fiber<Scalar>(x, t, h, text_len, grid_w, h_off, w_off,
                                     frame, a0, a1, a2, invp);
      });
}

// Dual-tensor 3-axis RoPE: the qwen21dit_rope_3axis rotation applied to q
// and k in ONE launch (same bytes, half the launches, better occupancy).
// Per-fiber math is unchanged (one shared-helper call per tensor), so this
// is bitwise-identical to two separate calls. q and k must share geometry.
// inv is the qwen21dit_rope_inv_table for (a0, a1, a2, theta), built ONCE
// per forward by the caller (the table depends only on geometry/theta,
// which are block-invariant). 1D flattened over (fiber, pair): adjacent
// threads rotate adjacent pairs (coalesced); the old fiber-per-thread
// MDRange form strided adjacent threads by Dh (~30 GB/s effective).
// Per-pair ops (angle, sincos, rotation) are unchanged, so this is
// bitwise-identical to the fiber form.
template <class ExecSpace, class Scalar, class V3>
void qwen21dit_rope_3axis_dual(
    const V3& q, const V3& k, const Kokkos::View<float*, ExecSpace>& inv,
    size_t text_len, size_t grid_h, size_t grid_w, int a0, int a1, int a2,
    const std::string& label = "KokkosDNN::qwen21dit_rope_dual") {
  const size_t T = q.extent(0), H = q.extent(1), Dh = q.extent(2);
  if ((size_t)(a0 + a1 + a2) != Dh)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual: axes/head_dim mismatch");
  if (k.extent(0) != T || k.extent(1) != H || k.extent(2) != Dh)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual: q/k geometry mismatch");
  if (inv.extent(0) != Dh / 2)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual: inv table dim mismatch");
  if (T == 0 || H == 0 || Dh == 0) return;
  if (grid_h == 0 || grid_w == 0)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual: grid dims must be > 0");
  if (text_len + grid_h * grid_w != T)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual: S + H*W must equal T");
  const long h_off = (long)(grid_h - grid_h / 2);
  const long w_off = (long)(grid_w - grid_w / 2);
  const long frame = (long)text_len;
  const size_t P = Dh / 2;  // total pairs
  const size_t p0 = (size_t)a0 / 2, p1 = (size_t)a1 / 2;
  const size_t n0 = p0, n1 = p0 + p1;  // pair-section bases
  const size_t e1 = (size_t)a0, e2 = (size_t)a0 + (size_t)a1;  // elem bases
  const float* invp = inv.data();
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * H * P),
      KOKKOS_LAMBDA(size_t n) {
        const size_t f = n / P, pp = n % P;
        const size_t t = f / H, h = f % H;
        // Section s and in-section pair k from the pair index.
        int s = 0;
        size_t kk = pp;
        if (pp >= n1) {
          s = 2;
          kk = pp - n1;
        } else if (pp >= n0) {
          s = 1;
          kk = pp - n0;
        }
        // Position: text rows advance a shared position; target rows use
        // the frozen frame on axis 0 and zero-centered grids on axes 1-2.
        long pos;
        if (t < text_len) {
          pos = (long)t;
        } else {
          const size_t j = t - text_len;
          if (s == 0) {
            pos = frame;
          } else if (s == 1) {
            pos = (long)(j / grid_w) - h_off;
          } else {
            pos = (long)(j % grid_w) - w_off;
          }
        }
        const float ang = (float)pos * invp[pp];
        const float co = cosf(ang), si = sinf(ang);
        const size_t ebase = (s == 0) ? 2 * kk : (s == 1) ? e1 + 2 * kk
                                                          : e2 + 2 * kk;
        float q0 = static_cast<float>(q(t, h, ebase));
        float q1 = static_cast<float>(q(t, h, ebase + 1));
        q(t, h, ebase) = static_cast<Scalar>(q0 * co - q1 * si);
        q(t, h, ebase + 1) = static_cast<Scalar>(q0 * si + q1 * co);
        float k0 = static_cast<float>(k(t, h, ebase));
        float k1 = static_cast<float>(k(t, h, ebase + 1));
        k(t, h, ebase) = static_cast<Scalar>(k0 * co - k1 * si);
        k(t, h, ebase + 1) = static_cast<Scalar>(k0 * si + k1 * co);
      });
}

// Dual-tensor 3-axis RoPE with explicit per-row positions (edit path):
// the same per-pair rotation as qwen21dit_rope_3axis_dual, but the position
// for each row comes from caller-built [T] int tables (p0/p1/p2) instead of
// the single-block (text_len, grid) rule. The T2I form is the special case
// with one text prefix + one target block; the edit path builds the tables
// on the host from img_shapes + image_pad_mask exactly like
// QwenImage21Rope.forward (per-block frame freeze, zero-centered H/W grids,
// position advancing by max(h, w) per block). Bitwise-identical rotation
// math, so a single-block table reproduces the dual kernel bit-for-bit.
template <class ExecSpace, class Scalar, class V3, class PView>
void qwen21dit_rope_3axis_dual_pos(
    const V3& q, const V3& k, const Kokkos::View<float*, ExecSpace>& inv,
    const PView& p0, const PView& p1, const PView& p2, int a0, int a1, int a2,
    const std::string& label = "KokkosDNN::qwen21dit_rope_dual_pos") {
  const size_t T = q.extent(0), H = q.extent(1), Dh = q.extent(2);
  if ((size_t)(a0 + a1 + a2) != Dh)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual_pos: axes/head_dim mismatch");
  if (k.extent(0) != T || k.extent(1) != H || k.extent(2) != Dh)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual_pos: q/k geometry mismatch");
  if (inv.extent(0) != Dh / 2)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual_pos: inv table dim mismatch");
  if (p0.extent(0) != T || p1.extent(0) != T || p2.extent(0) != T)
    throw std::invalid_argument(
        "qwen21dit_rope_3axis_dual_pos: position table length mismatch");
  if (T == 0 || H == 0 || Dh == 0) return;
  const size_t P = Dh / 2;  // total pairs
  const size_t np0 = (size_t)a0 / 2, np1 = (size_t)a1 / 2;
  const size_t n1 = np0 + np1;  // pair-section bases
  const size_t e1 = (size_t)a0, e2 = (size_t)a0 + (size_t)a1;  // elem bases
  const float* invp = inv.data();
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * H * P),
      KOKKOS_LAMBDA(size_t n) {
        const size_t f = n / P, pp = n % P;
        const size_t t = f / H, h = f % H;
        int s = 0;
        size_t kk = pp;
        if (pp >= n1) {
          s = 2;
          kk = pp - n1;
        } else if (pp >= np0) {
          s = 1;
          kk = pp - np0;
        }
        const long pos =
            (s == 0) ? (long)p0(t) : (s == 1) ? (long)p1(t) : (long)p2(t);
        const float ang = (float)pos * invp[pp];
        const float co = cosf(ang), si = sinf(ang);
        const size_t ebase = (s == 0) ? 2 * kk : (s == 1) ? e1 + 2 * kk
                                                          : e2 + 2 * kk;
        float q0 = static_cast<float>(q(t, h, ebase));
        float q1 = static_cast<float>(q(t, h, ebase + 1));
        q(t, h, ebase) = static_cast<Scalar>(q0 * co - q1 * si);
        q(t, h, ebase + 1) = static_cast<Scalar>(q0 * si + q1 * co);
        float k0 = static_cast<float>(k(t, h, ebase));
        float k1 = static_cast<float>(k(t, h, ebase + 1));
        k(t, h, ebase) = static_cast<Scalar>(k0 * co - k1 * si);
        k(t, h, ebase + 1) = static_cast<Scalar>(k0 * si + k1 * co);
      });
}

// Reference QwenImage21Rope.forward on the host: per-row (frame, h, w)
// positions for the joint sequence, built from the per-image (frame, h, w)
// latent grids and the expanded image_pad_mask. Text rows advance a shared
// position on all three axes; every image block freezes the frame axis at
// the position reached by the preceding text and lays its tokens on
// zero-centered grids; the position then advances by max(h, w). Throws on
// any layout mismatch (blocks must consume exactly the masked rows).
inline void qwen21dit_build_rope_positions(
    const std::vector<std::array<int, 3>>& blocks,
    const std::vector<char>& image_pad_mask, std::vector<int>& p0,
    std::vector<int>& p1, std::vector<int>& p2) {
  const size_t T = image_pad_mask.size();
  p0.assign(T, 0);
  p1.assign(T, 0);
  p2.assign(T, 0);
  size_t cursor = 0;
  long position = 0;
  for (const auto& b : blocks) {
    const int h = b[1], w = b[2];
    if (b[0] <= 0 || h <= 0 || w <= 0)
      throw std::invalid_argument(
          "qwen21dit_build_rope_positions: block dims must be > 0");
    size_t start = cursor;
    while (start < T && !image_pad_mask[start]) ++start;
    for (size_t k = cursor; k < start; ++k) {
      p0[k] = p1[k] = p2[k] = (int)(position + (long)(k - cursor));
    }
    position += (long)(start - cursor);
    const size_t need = (size_t)h * (size_t)w;
    if (start + need > T)
      throw std::invalid_argument(
          "qwen21dit_build_rope_positions: block overruns the sequence");
    for (size_t k = 0; k < need; ++k)
      if (!image_pad_mask[start + k])
        throw std::invalid_argument(
            "qwen21dit_build_rope_positions: block rows must be masked");
    const long ho = (long)(h - h / 2), wo = (long)(w - w / 2);
    for (int hh = 0; hh < h; ++hh)
      for (int ww = 0; ww < w; ++ww) {
        const size_t j = start + (size_t)hh * (size_t)w + (size_t)ww;
        p0[j] = (int)position;
        p1[j] = hh - (int)ho;
        p2[j] = ww - (int)wo;
      }
    position += (long)((h > w) ? h : w);
    cursor = start + need;
  }
  for (size_t k = cursor; k < T; ++k) {
    if (image_pad_mask[k])
      throw std::invalid_argument(
          "qwen21dit_build_rope_positions: trailing masked rows need a block");
    p0[k] = p1[k] = p2[k] = (int)(position + (long)(k - cursor));
  }
}

// Edit-path joint metadata, mirroring transformer_qwenimage21.py: the
// repeat_interleave expansion of img_mask, the block labeling
// (build_token_metadata) and the prefix segmentation
// (_qwenimage21_prefix_segments). img_mask is the shared [V] row with
// V = S + N/4 (VLM slots plus one True slot per 2x2 target group); blocks
// are the per-image (frame, h, w) latent grids, condition images first and
// the target image last. The target block always occupies the trailing N
// rows. Throws exactly where the reference raises (block/mask mismatch).
struct Qwen21CondMeta {
  std::vector<char> image_pad_mask;  // [T] expanded mask (0/1)
  std::vector<long> image_ids;       // [T] -1 at text, block id at images
  std::vector<char> target_mask;     // [T] 1 at target-image rows
  // Prefix runs over [0, T-N): (start, end, is_text).
  std::vector<std::tuple<size_t, size_t, bool>> segments;
  size_t T = 0, S = 0, N = 0, Nc = 0, V = 0, nvision = 0;
  size_t prefix_len = 0;
};

inline Qwen21CondMeta qwen21dit_build_cond_meta(
    const std::vector<std::array<int, 3>>& blocks,
    const std::vector<char>& img_mask, size_t S, size_t N) {
  if (blocks.size() < 2)
    throw std::invalid_argument(
        "qwen21dit_build_cond_meta: need >= 1 cond block + target block");
  if (N == 0 || N % 4 != 0)
    throw std::invalid_argument(
        "qwen21dit_build_cond_meta: N must be a nonzero multiple of 4");
  const size_t V = img_mask.size();
  if (V != S + N / 4)
    throw std::invalid_argument(
        "qwen21dit_build_cond_meta: img_mask must hold S + N/4 entries");
  for (size_t v = S; v < V; ++v)
    if (!img_mask[v])
      throw std::invalid_argument(
          "qwen21dit_build_cond_meta: target slots must all be True");
  Qwen21CondMeta m;
  m.S = S;
  m.N = N;
  m.V = V;
  m.image_pad_mask.reserve(V * 2);
  for (char c : img_mask) {
    const int r = c ? 4 : 1;
    for (int k = 0; k < r; ++k) m.image_pad_mask.push_back(c ? 1 : 0);
  }
  m.T = m.image_pad_mask.size();
  std::vector<size_t> blengths;
  size_t total = 0;
  for (const auto& b : blocks) {
    if (b[0] <= 0 || b[1] <= 0 || b[2] <= 0)
      throw std::invalid_argument(
          "qwen21dit_build_cond_meta: block dims must be > 0");
    const size_t L = (size_t)b[1] * (size_t)b[2];
    blengths.push_back(L);
    total += L;
  }
  size_t pop = 0;
  for (char c : m.image_pad_mask) pop += c ? (size_t)1 : (size_t)0;
  if (total != pop)
    throw std::invalid_argument(
        "qwen21dit_build_cond_meta: img_shapes accounts for " +
        std::to_string(total) + " image tokens but img_mask marks " +
        std::to_string(pop));
  m.Nc = total - N;
  m.image_ids.assign(m.T, -1);
  m.target_mask.assign(m.T, 0);
  {
    size_t bi = 0, left = blengths[0];
    for (size_t t = 0; t < m.T; ++t) {
      if (!m.image_pad_mask[t]) continue;
      m.image_ids[t] = (long)bi;
      if (--left == 0 && bi + 1 < blengths.size()) {
        ++bi;
        left = blengths[bi];
      }
    }
    if (bi + 1 != blengths.size() || left != 0)
      throw std::invalid_argument(
          "qwen21dit_build_cond_meta: block lengths do not tile the mask");
  }
  {
    // Target rows are the last blengths.back() masked positions.
    size_t left = blengths.back();
    for (size_t t = m.T; t-- > 0 && left > 0;) {
      if (m.image_pad_mask[t]) {
        m.target_mask[t] = 1;
        --left;
      }
    }
    if (left != 0)
      throw std::invalid_argument(
          "qwen21dit_build_cond_meta: cannot locate the target block");
  }
  m.prefix_len = m.T - N;
  {
    // Runs of equal image_ids over the prefix (text and cond blocks).
    size_t start = 0;
    while (start < m.prefix_len) {
      size_t end = start + 1;
      while (end < m.prefix_len && m.image_ids[end] == m.image_ids[start])
        ++end;
      m.segments.emplace_back(start, end, m.image_ids[start] < 0);
      start = end;
    }
  }
  m.nvision = 0;
  for (size_t v = 0; v < S; ++v) m.nvision += img_mask[v] ? (size_t)1 : (size_t)0;
  return m;
}

// Joint assembly for the edit path: joint[j] comes from txt, the projected
// cond latents, or the projected target latents per the host-built maps
// (kind 0 = text row idx, 1 = cond row idx, 2 = target row idx). Replaces the
// reference repeat_interleave + substitution with one direct scatter: every
// joint row is written exactly once from its already-projected source.
template <class ExecSpace, class Scalar>
void qwen21dit_scatter_joint(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& txt,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& img,
    const Kokkos::View<int*, ExecSpace>& kind,
    const Kokkos::View<int*, ExecSpace>& idx,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& joint,
    const std::string& label = "KokkosDNN::qwen21dit_scatter") {
  const size_t T = joint.extent(0), D = joint.extent(1);
  if (kind.extent(0) != T || idx.extent(0) != T)
    throw std::invalid_argument("qwen21dit_scatter_joint: map length mismatch");
  if (txt.extent(1) != D || cond.extent(1) != D || img.extent(1) != D)
    throw std::invalid_argument("qwen21dit_scatter_joint: dim mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * D),
      KOKKOS_LAMBDA(size_t n) {
        const size_t j = n / D, d = n % D;
        const int k = kind(j), ii = idx(j);
        joint(j, d) = (k == 0) ? txt((size_t)ii, d)
                               : ((k == 1) ? cond((size_t)ii, d)
                                           : img((size_t)ii, d));
      });
}

// Fused affine-free LayerNorm + per-region scale with a target mask
// (edit form of qwen21dit_layernorm_noaffine_scaled): rows with
// tmask set read the sampled-timestep lane, all other rows (text AND cond
// tokens, matching _select_modulation_rows) read the t=0 lane.
// Same team-per-row two-pass FP32 algorithm as the split form.
template <class ExecSpace, class Scalar>
struct LayernormNoaffineScaledMaskFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x, y;
  Kokkos::View<Scalar*, ExecSpace> s_txt, s_img;
  Kokkos::View<unsigned char*, ExecSpace> tmask;
  float eps;
  size_t D;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto X = x;
    const auto Y = y;
    const auto ST = s_txt;
    const auto SI = s_img;
    const auto M = tmask;
    const size_t t = static_cast<size_t>(member.league_rank());
    const float Dn = static_cast<float>(D);
    float sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          acc += static_cast<float>(X(t, d));
        },
        sum);
    const float mean = sum / Dn;
    float sq = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          float c = static_cast<float>(X(t, d)) - mean;
          acc += c * c;
        },
        sq);
    const float inv = 1.0f / sqrt(sq / Dn + eps);
    const bool tgt = M(t) != 0;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, D), KOKKOS_LAMBDA(size_t d) {
          float n = (static_cast<float>(X(t, d)) - mean) * inv;
          float s =
              tgt ? static_cast<float>(SI(d)) : static_cast<float>(ST(d));
          Y(t, d) = static_cast<Scalar>(n * (1.0f + s));
        });
  }
};

// y = layernorm_noaffine(x) * (1 + per-row lane); tmask rows use s_img
// (sampled timestep), all other rows use s_txt (t=0). TeamPolicy over R rows.
template <class ExecSpace, class Scalar>
void qwen21dit_layernorm_noaffine_scaled_mask(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    float eps, const Kokkos::View<Scalar*, ExecSpace>& s_txt,
    const Kokkos::View<Scalar*, ExecSpace>& s_img,
    const Kokkos::View<unsigned char*, ExecSpace>& tmask,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::qwen21dit_ln_scaled_mask") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (y.extent(0) != R || y.extent(1) != D || s_txt.extent(0) != D ||
      s_img.extent(0) != D || tmask.extent(0) != R)
    throw std::invalid_argument(
        "qwen21dit_layernorm_noaffine_scaled_mask: dim mismatch");
  if (R == 0 || D == 0) return;
  LayernormNoaffineScaledMaskFunctor<ExecSpace, Scalar> f{x, y, s_txt, s_img,
                                                          tmask, eps, D};
  Kokkos::parallel_for(label, Kokkos::TeamPolicy<ExecSpace>(R, Kokkos::AUTO),
                       f);
}

// Per-row tanh-gated residual with a target mask (edit form of
// qwen21dit_gated_residual_split): masked rows use g_img, the rest g_txt.
// Same tanh-hoisted 1D flattened MAC; per-output-element ops unchanged.
template <class ExecSpace, class Scalar>
void qwen21dit_gated_residual_split_mask(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& upd,
    const Kokkos::View<Scalar*, ExecSpace>& g_txt,
    const Kokkos::View<Scalar*, ExecSpace>& g_img,
    const Kokkos::View<unsigned char*, ExecSpace>& tmask,
    const std::string& label = "KokkosDNN::qwen21dit_gate_mask") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (upd.extent(0) != R || upd.extent(1) != D || g_txt.extent(0) != D ||
      g_img.extent(0) != D || tmask.extent(0) != R)
    throw std::invalid_argument(
        "qwen21dit_gated_residual_split_mask: dim mismatch");
  if (R == 0 || D == 0) return;
  Kokkos::View<float*, ExecSpace> tg(
      Kokkos::ViewAllocateWithoutInitializing(label + "_gate_tanh"), 2 * D);
  Kokkos::parallel_for(
      label + "_tanh", Kokkos::RangePolicy<ExecSpace>(0, 2 * D),
      KOKKOS_LAMBDA(size_t k) {
        tg(k) = (k < D) ? tanhf(static_cast<float>(g_txt(k)))
                        : tanhf(static_cast<float>(g_img(k - D)));
      });
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, R * D),
      KOKKOS_LAMBDA(size_t n) {
        const size_t i = n / D, j = n % D;
        float g = (tmask(i) != 0) ? tg(D + j) : tg(j);
        x(i, j) = static_cast<Scalar>(static_cast<float>(x(i, j)) +
                                      g * static_cast<float>(upd(i, j)));
      });
}

// Fused affine-free LayerNorm + per-region scale (replaces
// layernorm_noaffine_forward + qwen21dit_apply_scale_split pairs):
//   y = (x - mean(x)) * inv * (1 + s[row < split ? txt : img]).
// Same team-per-row two-pass FP32 algorithm as
// KokkosDNN::Impl::LayerNormNoAffineFunctor (identical mean/var/inv op
// order: rsqrt first, scale multiply folded into the write-back only),
// saving one full [R, D] pass. In-place safe (x and y may alias: the
// reductions only read x; writes start after the team's final barrier).
// Not bitwise-identical to the two-pass form (one fewer Scalar rounding
// on the write-back), well within tolerance.
// Named functor (not a nested extended lambda): nvcc rejects an extended
// __host__ __device__ lambda defined inside another extended lambda.
template <class ExecSpace, class Scalar>
struct LayernormNoaffineScaledSplitFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x, y;
  Kokkos::View<Scalar*, ExecSpace> s_txt, s_img;
  float eps;
  size_t D, split;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto X = x;
    const auto Y = y;
    const auto ST = s_txt;
    const auto SI = s_img;
    const size_t t = static_cast<size_t>(member.league_rank());
    const size_t sp = split;
    const float Dn = static_cast<float>(D);
    float sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          acc += static_cast<float>(X(t, d));
        },
        sum);
    const float mean = sum / Dn;
    float sq = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          float c = static_cast<float>(X(t, d)) - mean;
          acc += c * c;
        },
        sq);
    const float inv = 1.0f / sqrt(sq / Dn + eps);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, D), KOKKOS_LAMBDA(size_t d) {
          float n = (static_cast<float>(X(t, d)) - mean) * inv;
          float s = (t < sp) ? static_cast<float>(ST(d))
                             : static_cast<float>(SI(d));
          Y(t, d) = static_cast<Scalar>(n * (1.0f + s));
        });
  }
};

// y = layernorm_noaffine(x) * (1 + per-region scale); s_txt covers rows
// [0, split), s_img rows [split, R). Same checker/launch conventions as
// layernorm_noaffine_forward (TeamPolicy over R rows).
template <class ExecSpace, class Scalar>
void qwen21dit_layernorm_noaffine_scaled(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    float eps, const Kokkos::View<Scalar*, ExecSpace>& s_txt,
    const Kokkos::View<Scalar*, ExecSpace>& s_img, size_t split,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::qwen21dit_ln_scaled") {
  const size_t R = x.extent(0), D = x.extent(1);
  if (y.extent(0) != R || y.extent(1) != D || s_txt.extent(0) != D ||
      s_img.extent(0) != D || split > R)
    throw std::invalid_argument(
        "qwen21dit_layernorm_noaffine_scaled: dim mismatch");
  if (R == 0 || D == 0) return;
  LayernormNoaffineScaledSplitFunctor<ExecSpace, Scalar> f{x, y, s_txt,
                                                           s_img, eps, D,
                                                           split};
  Kokkos::parallel_for(label, Kokkos::TeamPolicy<ExecSpace>(R, Kokkos::AUTO),
                       f);
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
// The gate vectors are [D] broadcast over R rows, so the old fiber
// recomputed tanhf(gate(j)) R times per column. The tanh is hoisted into
// a [2*D] float buffer (fully overwritten before use, so allocated
// WithoutInitializing) and the MAC runs as a 1D flattened RangePolicy
// (adjacent threads touch adjacent j: coalesced). Per-output-element ops
// are unchanged, so this is bitwise-identical to the fiber form.
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
  Kokkos::View<float*, ExecSpace> tg(
      Kokkos::ViewAllocateWithoutInitializing(label + "_gate_tanh"), 2 * D);
  Kokkos::parallel_for(
      label + "_tanh", Kokkos::RangePolicy<ExecSpace>(0, 2 * D),
      KOKKOS_LAMBDA(size_t k) {
        tg(k) = (k < D) ? tanhf(static_cast<float>(g_txt(k)))
                        : tanhf(static_cast<float>(g_img(k - D)));
      });
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, R * D),
      KOKKOS_LAMBDA(size_t n) {
        const size_t i = n / D, j = n % D;
        float g = (i < split) ? tg(j) : tg(D + j);
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
  // byte-size mismatch. Needs allocate() first. Pass a warmed
  // QwenImage21::ShardCache to stream staged shards from host RAM instead
  // of disk (repeat loads); the default nullptr reads every shard from
  // disk, so existing one-argument callers are unchanged.
  void load_safetensors(const std::vector<std::string>& shard_paths,
                        QwenImage21::ShardCache* cache = nullptr) {
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
      const QwenImage21::Impl::Qwen21ShardBlob* blob =
          (cache != nullptr) ? cache->find(path) : nullptr;
      const std::map<std::string, QwenImage21::Impl::Qwen21SafeEntry> entries =
          (blob != nullptr)
              ? QwenImage21::Impl::qwen21_parse_shard_blob(blob->bytes, path)
              : QwenImage21::Impl::qwen21_open_shard(f, path);
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
        if (blob != nullptr) {
          load_one_tensor(*blob, full, en, *spec);
        } else {
          load_one_tensor(f, full, en, *spec);
        }
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

  void load_safetensors(const std::string& shard_path,
                        QwenImage21::ShardCache* cache = nullptr) {
    load_safetensors(std::vector<std::string>{shard_path}, cache);
  }

 private:
  // Stream one file tensor into its device view (shape-checked). Source is
  // either a disk stream or a staged shard image (overload resolution on
  // the first argument picks the matching qwen21_stream_into).
  template <class Source>
  void load_one_tensor(Source& src, const std::string& full,
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
          src, en, want, elsize, (*dst2).data());
    } else {
      if (!dst1 || (*dst1).extent(0) != (size_t)s.d0)
        throw std::runtime_error(
            "QwenImage21DitWeights::load_safetensors: view mismatch '" +
            s.name + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          src, en, want, elsize, (*dst1).data());
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
  using I1 = Kokkos::View<int*, ExecSpace>;
  using U1 = Kokkos::View<unsigned char*, ExecSpace>;
  QwenImage21::Impl::Qwen21ViewCache<V1> c1;
  QwenImage21::Impl::Qwen21ViewCache<V2> c2;
  QwenImage21::Impl::Qwen21ViewCache<F2> f2;
  QwenImage21::Impl::Qwen21ViewCache<I1> i1;
  QwenImage21::Impl::Qwen21ViewCache<U1> u1;
  size_t bytes() const {
    return c1.bytes(sizeof(Scalar)) + c2.bytes(sizeof(Scalar)) +
           f2.bytes(4) + i1.bytes(4) + u1.bytes(1);
  }
  void clear() {
    c1.clear();
    c2.clear();
    f2.clear();
    i1.clear();
    u1.clear();
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

    // Timestep sinusoid on host (float32 phases, 256-d [cos|sin] halves).
    // Reference (QwenImage21TemporalTimesteps): the pipeline rounds the
    // timestep to bf16, divides by 1000 (bf16), and only then does the
    // transformer multiply by time_factor 1000 in fp32. Reproduce that lane
    // for bf16 builds so steps 1..7 see the same effective timestep as the
    // torch golden (step 0 at sigma=1.0 is unaffected: both are exact).
    constexpr size_t F = 256;
    std::vector<float> sinus_s(F), sinus_0(F);
    {
      double t_arg = timestep;
      if constexpr (std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
        const float t1000 = (float)(1000.0 * timestep);  // scheduler t, f32
        const Scalar tq = static_cast<Scalar>(t1000);    // bf16(t)
        const Scalar q =
            static_cast<Scalar>((float)tq / 1000.0f);    // bf16(t/1000)
        t_arg = 1000.0 * (double)(float)q;  // time_factor * timestep.float()
      }
      const float half_f = (float)(F / 2);
      for (size_t i = 0; i < F / 2; ++i) {
        // torch builds freqs in float32; keep the angle multiply fp32 too.
        const float inv =
            std::exp(-9.210340371976184f * (float)i / half_f);  // -ln(1e4)
        const float a_s = (float)t_arg * inv;
        sinus_s[i] = std::cos(a_s);
        sinus_s[i + F / 2] = std::sin(a_s);
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

      // RoPE inv-freq table: block-invariant (geometry/theta fixed), so
      // build ONCE per forward and reuse across all 32 blocks.
      Kokkos::View<float*, ExecSpace> rope_inv =
          Impl::qwen21dit_rope_inv_table<ExecSpace>(
              cfg_.rope_axes[0], cfg_.rope_axes[1], cfg_.rope_axes[2],
              cfg_.rope_theta);
      for (size_t li = 0; li < weights_.blocks.size(); ++li) {
        const auto& bw = weights_.blocks[li];
        // Attention path: fused affine-free LayerNorm + per-region scale
        // (text rows <- t=0 lane, target rows <- sampled lane).
        Impl::qwen21dit_layernorm_noaffine_scaled<ExecSpace, Scalar>(
            joint, eps, a_z, a_s, S, hn);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_q,
                         static_cast<Scalar>(0.0f), q2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_k,
                         static_cast<Scalar>(0.0f), k2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_v,
                         static_cast<Scalar>(0.0f), v2);
        U3 q_txt(q2.data(), S, H, Hd), k_pre(k2.data(), S, H, Hd),
            v_pre(v2.data(), S, H, Hd), o_txt(o2.data(), S, H, Hd);
        U3 q_full(q2.data(), T, H, Hd), k_full(k2.data(), T, H, Hd);
        U3 q_tgt(q2.data() + S * H * Hd, N, H, Hd),
            v_full(v2.data(), T, H, Hd),
            o_tgt(o2.data() + S * H * Hd, N, H, Hd);
        // FUSED qk_rmsnorm (was 4 launches over txt/tgt halves): the halves
        // are contiguous halves of q_full/k_full and rows are independent,
        // so per-row math is unchanged.
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(q_full, bw.norm_q, eps);
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(k_full, bw.norm_k, eps);
        // FUSED rope (was 2 launches): one launch covers q_full + k_full
        // with identical per-fiber math.
        Impl::qwen21dit_rope_3axis_dual<ExecSpace, Scalar>(
            q_full, k_full, rope_inv, S, latent_h, latent_w, cfg_.rope_axes[0],
            cfg_.rope_axes[1], cfg_.rope_axes[2]);
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
        // SwiGLU FFN path (fused norm + scale, same lane convention).
        Impl::qwen21dit_layernorm_noaffine_scaled<ExecSpace, Scalar>(
            joint, eps, b_z, b_s, S, hn);
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
        Impl::qwen21dit_layernorm_noaffine_scaled<ExecSpace, Scalar>(
            joint, eps, ns_z, ns_s, S, hn);
      }
      {
        U2 hn_tgt(hn.data() + S * D, N, D);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn_tgt,
                         weights_.proj_out_W, static_cast<Scalar>(0.0f), out_b);
      }
    }
  }

  // DiT edit forward: target tokens [B,N,C] + packed cond tokens [B,Nc,C]
  // + context [B,S,Td] + scalar timestep (RAW [0, 1] sigma, scaled by
  // time_factor 1000 internally) -> out [B,N,C] target-only velocity.
  // Mirrors the reference edit __call__ + forward: hidden latents are
  // [cond | target]; the joint sequence lays text, cond-image and target
  // tokens out via the repeat_interleave-4 + substitution layout; RoPE
  // freezes the frame axis per image block; attention runs the exact
  // block-causal prefill segmentation (text segments causal, image segments
  // including the target block full over their visible prefix); cond rows
  // modulate from t=0 exactly like text (target_token_mask); proj_out reads
  // the trailing target block (the pipeline's [:, -N:] slice, applied here).
  // img_blocks holds the per-image (frame, h, w) latent grids, condition
  // images first and the target image last (shared across samples, like the
  // reference row-0 layout). img_mask is the shared [V] row with
  // V = S + N/4 (VLM slots plus one True slot per 2x2 target group); the
  // trailing N/4 entries must be True. enc_mask (optional) marks valid
  // encoder rows; empty means all-valid (padded-key attention is not
  // implemented: any zero entry throws).
  void forward_cond(const DNNHandle<ExecSpace>& handle, const V3& target,
                    const V3& cond, const V3& context, double timestep,
                    V3& out,
                    const std::vector<std::array<int, 3>>& img_blocks,
                    const std::vector<char>& img_mask,
                    const std::vector<char>& enc_mask = {},
                    size_t latent_h = 0, size_t latent_w = 0) {
    cfg_.validate();
    const size_t B = target.extent(0), N = target.extent(1),
                 C = target.extent(2), Nc = cond.extent(1),
                 S = context.extent(1), Td = context.extent(2);
    const size_t D = (size_t)cfg_.hidden_dim, Hd = (size_t)cfg_.head_dim(),
                 H = (size_t)cfg_.num_heads, G = (size_t)cfg_.gate_dim();
    if (latent_h == 0) latent_h = (size_t)cfg_.latent_h;
    if (latent_w == 0) latent_w = (size_t)cfg_.latent_w;
    if (cond.extent(0) != B || cond.extent(2) != C)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: cond/target batch mismatch");
    if (context.extent(0) != B)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: batch mismatch");
    if (C != (size_t)cfg_.latent_channels || Td != (size_t)cfg_.context_dim)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: dim mismatch");
    if (N != latent_h * latent_w || N == 0)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: N must equal latent_h*latent_w");
    if (Nc == 0)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: empty cond (use forward())");
    if (out.extent(0) != B || out.extent(1) != N || out.extent(2) != C)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: out mismatch");
    if (S == 0)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: empty context");
    if (!cfg_.causal_condition)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: causal_condition=false "
          "unsupported (t=0 row split required)");
    if (!enc_mask.empty()) {
      if (enc_mask.size() != S)
        throw std::invalid_argument(
            "QwenImage21DitModel::forward_cond: enc_mask length != S");
      for (char c : enc_mask)
        if (!c)
          throw std::invalid_argument(
              "QwenImage21DitModel::forward_cond: padded encoder rows "
              "(key_valid) are not implemented; pass an all-valid mask");
    }
    const Impl::Qwen21CondMeta meta =
        Impl::qwen21dit_build_cond_meta(img_blocks, img_mask, S, N);
    if (Nc != meta.Nc)
      throw std::invalid_argument(
          "QwenImage21DitModel::forward_cond: cond holds " +
          std::to_string(Nc) + " tokens, img_shapes expects " +
          std::to_string(meta.Nc));
    {
      const auto& tb = img_blocks.back();
      if (tb[0] != 1 || (size_t)tb[1] * (size_t)tb[2] != N)
        throw std::invalid_argument(
            "QwenImage21DitModel::forward_cond: last img_shapes block must "
            "be the (1, latent_h, latent_w) target grid");
    }
    const size_t T = meta.T;
    const float eps = (float)cfg_.norm_eps;
    const float attn_scale =
        1.0f / std::sqrt(static_cast<float>(Hd));  // SDPA default 1/sqrt(E)

    using U2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using U1 = Kokkos::View<Scalar*, ExecSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using U3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Timestep sinusoid on host (float32 phases, 256-d [cos|sin] halves),
    // with the same bf16 double-rounding lane as forward(): the pipeline
    // rounds the timestep to bf16, divides by 1000 (bf16), and only then
    // does the transformer multiply by time_factor 1000 in fp32.
    constexpr size_t F = 256;
    std::vector<float> sinus_s(F), sinus_0(F);
    {
      double t_arg = timestep;
      if constexpr (std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
        const float t1000 = (float)(1000.0 * timestep);  // scheduler t, f32
        const Scalar tq = static_cast<Scalar>(t1000);    // bf16(t)
        const Scalar q =
            static_cast<Scalar>((float)tq / 1000.0f);  // bf16(t/1000)
        t_arg = 1000.0 * (double)(float)q;  // time_factor * timestep.float()
      }
      const float half_f = (float)(F / 2);
      for (size_t i = 0; i < F / 2; ++i) {
        // torch builds freqs in float32; keep the angle multiply fp32 too.
        const float inv =
            std::exp(-9.210340371976184f * (float)i / half_f);  // -ln(1e4)
        const float a_s = (float)t_arg * inv;
        sinus_s[i] = std::cos(a_s);
        sinus_s[i + F / 2] = std::sin(a_s);
        sinus_0[i] = 1.0f;  // t = 0: cos(0) = 1 ...
        sinus_0[i + F / 2] = 0.0f;  // ... sin(0) = 0
      }
    }

    // Shared joint tables (the layout is sample-invariant, like the
    // reference row-0 layout): scatter maps, target mask, RoPE positions.
    Kokkos::View<int*, ExecSpace> d_kind("qwen21c_kind", T),
        d_idx("qwen21c_idx", T), d_p0("qwen21c_p0", T),
        d_p1("qwen21c_p1", T), d_p2("qwen21c_p2", T);
    Kokkos::View<unsigned char*, ExecSpace> d_tmask("qwen21c_tmask", T);
    {
      std::vector<int> p0, p1, p2;
      Impl::qwen21dit_build_rope_positions(img_blocks, meta.image_pad_mask,
                                           p0, p1, p2);
      // Encoder rows feeding the surviving text rows: the VLM slots expand
      // 4x and are then fully overwritten by cond latents (substitution),
      // so only non-slot encoder rows survive, in order.
      std::vector<int> text_src;
      text_src.reserve(S);
      for (size_t v = 0; v < S; ++v)
        if (!img_mask[v]) text_src.push_back((int)v);
      // Scatter maps: text rows read txt in surviving order; image rows
      // read the packed [cond | target] projections in joint order (cond
      // first: every masked row is substituted, like the reference).
      std::vector<int> kind(T, 0), idx(T, 0);
      size_t ti = 0, li = 0;
      for (size_t j = 0; j < T; ++j) {
        if (!meta.image_pad_mask[j]) {
          if (ti >= text_src.size())
            throw std::logic_error(
                "QwenImage21DitModel::forward_cond: text map overrun");
          kind[j] = 0;
          idx[j] = text_src[ti++];
        } else if (li < Nc) {
          kind[j] = 1;
          idx[j] = (int)(li++);
        } else {
          kind[j] = 2;
          idx[j] = (int)(li++ - Nc);
        }
      }
      if (ti != text_src.size() || li != Nc + N)
        throw std::logic_error(
            "QwenImage21DitModel::forward_cond: scatter map miscount");
      auto hk = Kokkos::create_mirror_view(d_kind);
      auto hi = Kokkos::create_mirror_view(d_idx);
      auto h0 = Kokkos::create_mirror_view(d_p0);
      auto h1 = Kokkos::create_mirror_view(d_p1);
      auto h2 = Kokkos::create_mirror_view(d_p2);
      auto hm = Kokkos::create_mirror_view(d_tmask);
      for (size_t j = 0; j < T; ++j) {
        hk(j) = kind[j];
        hi(j) = idx[j];
        h0(j) = p0[j];
        h1(j) = p1[j];
        h2(j) = p2[j];
        hm(j) = meta.target_mask[j] ? (unsigned char)1 : (unsigned char)0;
      }
      Kokkos::deep_copy(d_kind, hk);
      Kokkos::deep_copy(d_idx, hi);
      Kokkos::deep_copy(d_p0, h0);
      Kokkos::deep_copy(d_p1, h1);
      Kokkos::deep_copy(d_p2, h2);
      Kokkos::deep_copy(d_tmask, hm);
    }

    // RoPE inv-freq table: block-invariant (axes/theta fixed), so build ONCE
    // per forward and reuse across all blocks.
    Kokkos::View<float*, ExecSpace> rope_inv =
        Impl::qwen21dit_rope_inv_table<ExecSpace>(
            cfg_.rope_axes[0], cfg_.rope_axes[1], cfg_.rope_axes[2],
            cfg_.rope_theta);

    for (size_t b = 0; b < B; ++b) {
      U2 tgt_b(const_cast<Scalar*>(target.data()) + b * N * C, N, C);
      U2 cnd_b(const_cast<Scalar*>(cond.data()) + b * Nc * C, Nc, C);
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

      // --- img_in: cond + target Linear C -> D (one GEMM each) ---
      V2 cond_t = scratch_.c2.get("cond", Nc, D);
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), cnd_b,
                       weights_.img_in_W, static_cast<Scalar>(0.0f), cond_t);
      V2 img_t = scratch_.c2.get("img", N, D);
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), tgt_b,
                       weights_.img_in_W, static_cast<Scalar>(0.0f), img_t);

      // --- timestep MLP (TimestepEmbedder: Linear -> SiLU -> Linear) ---
      // Two rows: sampled timestep + t = 0 (causal_condition). Text AND
      // cond rows modulate from the t=0 row, target rows from sampled.
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
      // Row 0 = sampled timestep (target rows), row 1 = t=0 (text + cond).
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

      // --- joint sequence [text | cond | target] (interleaved per mask) ---
      V2 joint = scratch_.c2.get("joint", T, D);
      Impl::qwen21dit_scatter_joint<ExecSpace, Scalar>(txt, cond_t, img_t,
                                                       d_kind, d_idx, joint);
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
        // Attention path: fused affine-free LayerNorm + per-row lane
        // (text and cond rows <- t=0 lane, target rows <- sampled lane).
        Impl::qwen21dit_layernorm_noaffine_scaled_mask<ExecSpace, Scalar>(
            joint, eps, a_z, a_s, d_tmask, hn);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_q,
                         static_cast<Scalar>(0.0f), q2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_k,
                         static_cast<Scalar>(0.0f), k2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.to_v,
                         static_cast<Scalar>(0.0f), v2);
        U3 q_full(q2.data(), T, H, Hd), k_full(k2.data(), T, H, Hd);
        U3 v_full(v2.data(), T, H, Hd);
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(q_full, bw.norm_q, eps);
        Impl::qwen21dit_qk_rmsnorm<ExecSpace, Scalar>(k_full, bw.norm_k, eps);
        Impl::qwen21dit_rope_3axis_dual_pos<ExecSpace, Scalar>(
            q_full, k_full, rope_inv, d_p0, d_p1, d_p2, cfg_.rope_axes[0],
            cfg_.rope_axes[1], cfg_.rope_axes[2]);
        // Exact block-causal prefill segmentation (the general form of
        // QwenImage21AttnProcessor): each text segment attends causally over
        // its visible prefix, each cond image segment attends fully over its
        // visible prefix, and the target block attends over all keys.
        for (const auto& sg : meta.segments) {
          const size_t s0 = std::get<0>(sg), s1 = std::get<1>(sg);
          const bool is_text = std::get<2>(sg);
          const size_t sl = s1 - s0;
          if (is_text) {
            U3 qs(q2.data() + s0 * H * Hd, sl, H, Hd),
                kp(k2.data(), s1, H, Hd), vp(v2.data(), s1, H, Hd),
                os(o2.data() + s0 * H * Hd, sl, H, Hd);
            KokkosDNN::attention_gqa_single<ExecSpace, Scalar>(
                handle, qs, kp, vp, s0, os, attn_scale,
                "KokkosDNN::qwen21dit_attn_txt");
          } else {
            U3 qs(q2.data() + s0 * H * Hd, sl, H, Hd),
                kp(k2.data(), s1, H, Hd), vp(v2.data(), s1, H, Hd),
                os(o2.data() + s0 * H * Hd, sl, H, Hd);
            KokkosDNN::attention_mha_bidirectional<ExecSpace, Scalar>(
                handle, qs, kp, vp, os, attn_scale,
                "KokkosDNN::qwen21dit_attn_cond");
          }
        }
        {
          U3 qt(q2.data() + (T - N) * H * Hd, N, H, Hd);
          U3 ot(o2.data() + (T - N) * H * Hd, N, H, Hd);
          KokkosDNN::attention_mha_bidirectional<ExecSpace, Scalar>(
              handle, qt, k_full, v_full, ot, attn_scale,
              "KokkosDNN::qwen21dit_attn_img");
        }
        U2 o2f(o2.data(), T, D);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), o2f, bw.to_o,
                         static_cast<Scalar>(0.0f), hn);
        Impl::qwen21dit_gated_residual_split_mask<ExecSpace, Scalar>(
            joint, hn, g1_z, g1_s, d_tmask);
        // SwiGLU FFN path (fused norm + lane, same convention).
        Impl::qwen21dit_layernorm_noaffine_scaled_mask<ExecSpace, Scalar>(
            joint, eps, b_z, b_s, d_tmask, hn);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.proj,
                         static_cast<Scalar>(0.0f), p2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), hn, bw.gate,
                         static_cast<Scalar>(0.0f), g2);
        Impl::qwen21dit_swiglu<ExecSpace, Scalar>(g2, p2, h2);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), h2, bw.out,
                         static_cast<Scalar>(0.0f), hn);
        Impl::qwen21dit_gated_residual_split_mask<ExecSpace, Scalar>(
            joint, hn, g2_z, g2_s, d_tmask);
      }

      // --- head: AdaLayerNormContinuous (temb-conditioned scale) + proj ---
      // Reference: scale = norm_out_linear(silu(temb)); rows selected by
      // target_token_mask (text and cond <- t=0 row, target <- sampled);
      // joint = layernorm_affinefree(joint) * (1 + scale); proj_out reads
      // the trailing target rows only (the [:, -N:] slice, applied here).
      {
        V2 t1 = scratch_.c2.get("t1n", 2, D);
        Kokkos::deep_copy(t1, s3);
        KokkosDNN::silu_forward<ExecSpace, Scalar>(t1, t1);
        V2 ns2 = scratch_.c2.get("ns2", 2, D);
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), t1,
                         weights_.norm_out_W, static_cast<Scalar>(0.0f), ns2);
        Scalar* np = ns2.data();
        U1 ns_s(np, D), ns_z(np + D, D);
        Impl::qwen21dit_layernorm_noaffine_scaled_mask<ExecSpace, Scalar>(
            joint, eps, ns_z, ns_s, d_tmask, hn);
      }
      {
        U2 hn_tgt(hn.data() + (T - N) * D, N, D);
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
