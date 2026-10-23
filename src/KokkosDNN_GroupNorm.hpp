// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GROUPNORM_HPP
#define KOKKOSDNN_GROUPNORM_HPP

// GroupNorm over NCHW activations (diffusers ResNet/attention norm).
//
// Covers the FLUX.2-small VAE decoder (AutoencoderKLFlux2, MOSS-Image-SFT
// vae/): every ResNet block normalizes with GroupNorm(32 groups, eps 1e-6,
// affine weight+bias), the mid-block self-attention entry normalizes with
// GroupNorm(32 groups, eps 1e-6, affine), and the decoder head normalizes
// with GroupNorm(32 groups, eps 1e-6, affine) before SiLU + conv_out.
// Reference: diffusers 0.38.0 models/resnet.py (ResnetBlock2D norm1/norm2),
// models/autoencoders/vae.py (Decoder conv_norm_out), and
// models/attention_processor.py (Attention.group_norm).
//
// Math (PyTorch nn.GroupNorm, channels split into G contiguous groups):
//   for each (n, g): mean/var over the Cg*H*W elements
//     (Cg = C / G channels [g*Cg, (g+1)*Cg)),
//   y[n,c,h,w] = (x[n,c,h,w] - mean) / sqrt(var + eps) * w[c] + b[c].
// Scalar-agnostic (float, bhalf_t); statistics and the normalize/scale/
// shift accumulate in FP32, then cast to Scalar.
//
// Dispatch: native portable team-parallel kernel (TeamPolicy +
// TeamThreadRange reductions, no TPL — matches repo policy that norm ops
// are native; cf. layernorm_forward, rmsnorm). No backend #ifdefs in the
// native path.
//
// In-place safety: statistics are fully reduced before any write in the
// same kernel launch, and each output element depends only on the
// same-index input plus its group's (already-final) statistics and the
// affine vectors, so x may alias out.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Impl {

// Three-phase portable GroupNorm. Phases A and C give each (n, g, ch)
// row to one team and fan out over W with TeamThreadRange, so consecutive
// threads always touch consecutive W elements (fully coalesced under
// LayoutRight, auto-vectorized) with no per-element div/mod (row indices
// come from the league rank, once per team):
//
//   A. row partials: one team per row reduces W into an FP32 (sum, sumsq)
//      pair (two TeamThreadRange reductions). Reads each x element twice.
//   B. finalize: RangePolicy over N*G groups; each thread serially combines
//      its CH row pairs in FP64 (E[x^2]-mean^2) into mean and 1/sqrt(var+eps).
//   C. normalize: one team per row; TeamThreadRange over W, fully
//      independent per element.
//
// This replaces the old MDRange-over-(N,G) kernel that serialized up to
// millions of elements in a single thread (~97 ms/launch on CUDA at
// VAE-decode geometry). Outer functors are named (rather than nested
// lambdas) to stay compilable under nvcc. All phases use only portable
// Kokkos (no backend #ifdefs) and run on Serial. In-place safe: phases are
// separate launches with device-wide sync between them.
template <class ExecSpace, class Scalar>
struct GroupNormRowPartialFunctor {
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Policy::member_type;
  Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace> x;
  size_t num_groups;
  size_t Cg;
  size_t H;
  size_t W;
  size_t CH;
  Kokkos::View<float*, ExecSpace> scratch;  // [2*NP + 2*NG]

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    const size_t i = static_cast<size_t>(team.league_rank());
    const size_t ng = i / CH;
    const size_t ch = i % CH;
    const size_t n = ng / num_groups;
    const size_t c = (ng % num_groups) * Cg + ch / H;
    const size_t h = ch % H;
    auto x_ = x;
    const size_t W_ = W;
    float sum = 0.0f;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, W_),
        KOKKOS_LAMBDA(size_t w, float& lsum) {
          lsum += static_cast<float>(x_(n, c, h, w));
        },
        sum);
    float ssq = 0.0f;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, W_),
        KOKKOS_LAMBDA(size_t w, float& lssq) {
          const float v = static_cast<float>(x_(n, c, h, w));
          lssq += v * v;
        },
        ssq);
    scratch[2 * i] = sum;
    scratch[2 * i + 1] = ssq;
  }
};

template <class ExecSpace>
void groupnorm_finalize_stats(
    const Kokkos::View<float*, ExecSpace>& scratch,  // [2*NP + 2*NG]
    size_t NP, size_t NG, size_t CH, size_t M, float eps,
    const std::string& label) {
  const size_t CH_ = CH, NP_ = NP;
  const float eps_ = eps;
  const double Md = static_cast<double>(M);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, NG), KOKKOS_LAMBDA(size_t g) {
        double sum = 0.0, ssq = 0.0;
        const size_t base = g * CH_;
        for (size_t ch = 0; ch < CH_; ++ch) {
          sum += static_cast<double>(scratch[2 * (base + ch)]);
          ssq += static_cast<double>(scratch[2 * (base + ch) + 1]);
        }
        const double mean = sum / Md;
        double var = ssq / Md - mean * mean;
        if (var < 0.0 && var > -1e-12) var = 0.0;  // sumsq roundoff guard
        const float inv = 1.0f / sqrtf(static_cast<float>(var) + eps_);
        scratch[2 * (NP_ + g)] = static_cast<float>(mean);
        scratch[2 * (NP_ + g) + 1] = inv;
      });
}

template <class ExecSpace, class Scalar>
struct GroupNormRowNormFunctor {
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Policy::member_type;
  Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace> x;
  Kokkos::View<Scalar*, ExecSpace> weight;
  Kokkos::View<Scalar*, ExecSpace> bias;
  size_t num_groups;
  size_t Cg;
  size_t H;
  size_t W;
  size_t CH;
  Kokkos::View<float*, ExecSpace> scratch;  // [2*NP + 2*NG]
  size_t NP;
  Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace> y;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    const size_t i = static_cast<size_t>(team.league_rank());
    const size_t ng = i / CH;
    const size_t ch = i % CH;
    const size_t n = ng / num_groups;
    const size_t g = ng % num_groups;
    const size_t c = g * Cg + ch / H;
    const size_t h = ch % H;
    auto x_ = x;
    auto y_ = y;
    const float mean = scratch[2 * (NP + ng)];
    const float inv = scratch[2 * (NP + ng) + 1];
    const float cw = static_cast<float>(weight(c));
    const float cb = static_cast<float>(bias(c));
    const size_t W_ = W;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, W_), KOKKOS_LAMBDA(size_t w) {
          y_(n, c, h, w) = static_cast<Scalar>(
              (static_cast<float>(x_(n, c, h, w)) - mean) * inv * cw + cb);
        });
  }
};

// Forward declaration: defined just below; the allocating entry delegates
// to it so persistent-scratch callers share the exact same phases.
template <class ExecSpace, class Scalar>
void groupnorm_nchw_with_scratch(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, float eps, size_t num_groups,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& y,
    const Kokkos::View<float*, ExecSpace>& scratch,
    const std::string& label);

template <class ExecSpace, class Scalar>
void groupnorm_nchw_impl(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, float eps, size_t num_groups,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label) {
  const size_t N = x.extent(0), C = x.extent(1), H = x.extent(2),
                W = x.extent(3);
  if (N == 0 || C == 0 || H == 0 || W == 0) return;
  const size_t Cg = C / num_groups;
  const size_t CH = Cg * H;  // rows per (n, g) group, > 0
  const size_t NP = N * num_groups * CH;
  const size_t NG = N * num_groups;
  // Write-first scratch (phase A writes all 2*NP row partials, phase B
  // writes all 2*NG group stats before phase C reads them), so no
  // zero-fill is needed.
  Kokkos::View<float*, ExecSpace> scratch(
      Kokkos::ViewAllocateWithoutInitializing("groupnorm_scratch"),
      2 * (NP + NG));
  groupnorm_nchw_with_scratch<ExecSpace, Scalar>(x, weight, bias, eps,
                                                 num_groups, y, scratch,
                                                 label);
}

// Same as groupnorm_nchw_impl but the caller provides the FP32 scratch
// (a contiguous [2*(NP+NG)] view with NP = N*num_groups*(C/num_groups)*H
// and NG = N*num_groups; managed or unmanaged). Lets steady-state
// callers (e.g. the VAE decoder) reuse a persistent buffer instead of
// allocating per call. Same kernels, same order, same numerics.
template <class ExecSpace, class Scalar>
void groupnorm_nchw_with_scratch(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, float eps, size_t num_groups,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& y,
    const Kokkos::View<float*, ExecSpace>& scratch,
    const std::string& label) {
  const size_t N = x.extent(0), C = x.extent(1), H = x.extent(2),
                W = x.extent(3);
  if (N == 0 || C == 0 || H == 0 || W == 0) return;
  const size_t Cg = C / num_groups;
  const size_t CH = Cg * H;  // rows per (n, g) group, > 0
  const size_t NP = N * num_groups * CH;
  const size_t NG = N * num_groups;
  using Partial = GroupNormRowPartialFunctor<ExecSpace, Scalar>;
  Kokkos::parallel_for(
      label + "_rows",
      typename Partial::Policy(static_cast<int>(NP), Kokkos::AUTO),
      Partial{x, num_groups, Cg, H, W, CH, scratch});
  groupnorm_finalize_stats<ExecSpace>(scratch, NP, NG, CH, CH * W, eps,
                                      label + "_stats");
  using Norm = GroupNormRowNormFunctor<ExecSpace, Scalar>;
  Kokkos::parallel_for(
      label + "_norm",
      typename Norm::Policy(static_cast<int>(NP), Kokkos::AUTO),
      Norm{x, weight, bias, num_groups, Cg, H, W, CH, scratch, NP, y});
}

}  // namespace Impl

// GroupNorm forward (affine):
//   y[n,c,h,w] = (x[n,c,h,w] - mean[n,g]) / sqrt(var[n,g] + eps)
//                * weight[c] + bias[c],  g = c / (C / num_groups).
// x, y: [N, C, H, W] (LayoutRight); weight, bias: [C] (diffusers affine
// GroupNorm always carries both). num_groups must be > 0 and divide C;
// eps must be finite and >= 0 (diffusers VAE uses 1e-6). In-place safe
// (x may alias y). Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void groupnorm_forward(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, size_t num_groups,
    float eps,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::groupnorm_forward") {
  const size_t N = x.extent(0), C = x.extent(1), H = x.extent(2),
               W = x.extent(3);
  if (y.extent(0) != N || y.extent(1) != C || y.extent(2) != H ||
      y.extent(3) != W || weight.extent(0) != C || bias.extent(0) != C)
    throw std::invalid_argument("groupnorm_forward: dimension mismatch");
  if (num_groups == 0)
    throw std::invalid_argument("groupnorm_forward: num_groups must be > 0");
  if (C % num_groups != 0)
    throw std::invalid_argument(
        "groupnorm_forward: channels must be divisible by num_groups");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("groupnorm_forward: bad eps");
  Impl::groupnorm_nchw_impl<ExecSpace, Scalar>(x, weight, bias, eps,
                                               num_groups, y, label);
}

}  // namespace KokkosDNN

#endif
