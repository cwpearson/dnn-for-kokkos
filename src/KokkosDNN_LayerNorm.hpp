// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_LAYERNORM_HPP
#define KOKKOSDNN_LAYERNORM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

namespace Impl {

// Team kernels below use one team per normalized row. Both accumulate passes
// run as TeamThreadRange parallel reductions (coalesced accesses along the
// contiguous LayoutRight dimension, full GPU occupancy), followed by a
// team-parallel normalized write-back in the same kernel launch. Same
// two-pass FP32 algorithm and eps placement as the original serial-row code:
//   mean = sum(x)/D; var = sum((x-mean)^2)/D; inv = 1/sqrt(var+eps).
// In-place safe: reductions only read x; writes start after the team's final
// barrier, and each (row, col) write reads only its own input element.
//
// Named functors (not nested extended lambdas): nvcc rejects an extended
// __host__ __device__ lambda defined inside another extended lambda, so the
// team body lives in operator() where inner KOKKOS_LAMBDA levels are legal.

template <class ExecSpace, class Scalar>
struct LayerNormAffineFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x, y;
  Kokkos::View<Scalar*, ExecSpace> weight, bias;
  float eps;
  size_t D;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    // Local handles: inner lambdas capture by value without an implicit
    // `this` capture (quiets -Wdeprecated under C++20).
    const auto X = x;
    const auto Y = y;
    const auto W = weight;
    const auto Bc = bias;
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
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, D), KOKKOS_LAMBDA(size_t d) {
          Y(t, d) = static_cast<Scalar>(
              (static_cast<float>(X(t, d)) - mean) * inv *
                  static_cast<float>(W(d)) +
              static_cast<float>(Bc(d)));
        });
  }
};

template <class ExecSpace, class Scalar>
struct LayerNormNoAffineFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x, y;
  float eps;
  size_t D;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto X = x;
    const auto Y = y;
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
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, D), KOKKOS_LAMBDA(size_t d) {
          Y(t, d) =
              static_cast<Scalar>((static_cast<float>(X(t, d)) - mean) * inv);
        });
  }
};

template <class ExecSpace, class Scalar>
struct LayerNormNoAffine3DFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> x, y;
  float eps;
  size_t T, D;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto X = x;
    const auto Y = y;
    const size_t row = static_cast<size_t>(member.league_rank());
    const size_t b = row / T;
    const size_t t = row % T;
    const float Dn = static_cast<float>(D);
    float sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          acc += static_cast<float>(X(b, t, d));
        },
        sum);
    const float mean = sum / Dn;
    float sq = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          float c = static_cast<float>(X(b, t, d)) - mean;
          acc += c * c;
        },
        sq);
    const float inv = 1.0f / sqrt(sq / Dn + eps);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, D), KOKKOS_LAMBDA(size_t d) {
          Y(b, t, d) = static_cast<Scalar>(
              (static_cast<float>(X(b, t, d)) - mean) * inv);
        });
  }
};

}  // namespace Impl

// LayerNorm (biased variant used by ViT/DINOv3, no RMSNorm substitution):
//   y[t,:] = (x[t,:] - mean(x[t,:])) / sqrt(var(x[t,:]) + eps) * w[:] + b[:]
// x, y: [tokens, dim], w, b: [dim]. Scalar-agnostic; statistics and the
// normalize/scale/shift are accumulated in FP32, then cast to Scalar.
// In-place safe (x and y may alias): each output depends only on the
// same-row input plus w/b.
template <class ExecSpace, class Scalar>
void layernorm_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::layernorm_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D || weight.extent(0) != D ||
      bias.extent(0) != D)
    throw std::invalid_argument("layernorm_forward: dimension mismatch");
  if (T == 0 || D == 0) return;
  Impl::LayerNormAffineFunctor<ExecSpace, Scalar> f{x, y, weight, bias, eps, D};
  Kokkos::parallel_for(label, Kokkos::TeamPolicy<ExecSpace>(T, Kokkos::AUTO),
                        f);
}

// Single-row variant for decode.
template <class ExecSpace, class Scalar>
void layernorm_row(
    const Kokkos::View<Scalar*, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, float eps,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::layernorm_row") {
  const size_t D = x.extent(0);
  if (y.extent(0) != D || weight.extent(0) != D || bias.extent(0) != D)
    throw std::invalid_argument("layernorm_row: dimension mismatch");
  if (D == 0) return;
  // Two reductions (mean, then variance) followed by the elementwise
  // normalize/scale/shift. Reductions keep the row variant correct on
  // device execution spaces.
  float sum = 0;
  Kokkos::parallel_reduce(
      label + "_mean", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        acc += static_cast<float>(x(d));
      },
      sum);
  const float mean = sum / static_cast<float>(D);
  float sq = 0;
  Kokkos::parallel_reduce(
      label + "_var", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        float c = static_cast<float>(x(d)) - mean;
        acc += c * c;
      },
      sq);
  const float inv =
      1.0f / std::sqrt(sq / static_cast<float>(D) + eps);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        y(d) = static_cast<Scalar>((static_cast<float>(x(d)) - mean) * inv *
                                       static_cast<float>(weight(d)) +
                                   static_cast<float>(bias(d)));
      });
}

// LayerNorm without affine parameters:
//   y[i,:] = (x[i,:] - mean(x[i,:])) / sqrt(var(x[i,:]) + eps)
// Scalar-agnostic; statistics accumulated in FP32. In-place safe.
// Rank-2 variant ([tokens, dim]).
template <class ExecSpace, class Scalar>
void layernorm_noaffine_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::layernorm_noaffine_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D)
    throw std::invalid_argument("layernorm_noaffine_forward: dimension mismatch");
  if (T == 0 || D == 0) return;
  Impl::LayerNormNoAffineFunctor<ExecSpace, Scalar> f{x, y, eps, D};
  Kokkos::parallel_for(label, Kokkos::TeamPolicy<ExecSpace>(T, Kokkos::AUTO),
                        f);
}

// Rank-3 variant ([batch, tokens, dim]).
template <class ExecSpace, class Scalar>
void layernorm_noaffine_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::layernorm_noaffine_forward") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D)
    throw std::invalid_argument("layernorm_noaffine_forward: dimension mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  Impl::LayerNormNoAffine3DFunctor<ExecSpace, Scalar> f{x, y, eps, T, D};
  Kokkos::parallel_for(label,
                        Kokkos::TeamPolicy<ExecSpace>(B * T, Kokkos::AUTO), f);
}

}  // namespace KokkosDNN

#endif
