#ifndef KOKKOSDNN_LAYERNORM_HPP
#define KOKKOSDNN_LAYERNORM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

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
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float mean = 0;
        for (size_t d = 0; d < D; ++d)
          mean += static_cast<float>(x(t, d));
        mean /= static_cast<float>(D);
        float var = 0;
        for (size_t d = 0; d < D; ++d) {
          float c = static_cast<float>(x(t, d)) - mean;
          var += c * c;
        }
        var /= static_cast<float>(D);
        float inv = 1.0f / sqrt(var + eps);
        for (size_t d = 0; d < D; ++d)
          y(t, d) = static_cast<Scalar>(
              (static_cast<float>(x(t, d)) - mean) * inv *
                  static_cast<float>(weight(d)) +
              static_cast<float>(bias(d)));
      });
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
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float mean = 0;
        for (size_t d = 0; d < D; ++d)
          mean += static_cast<float>(x(t, d));
        mean /= static_cast<float>(D);
        float var = 0;
        for (size_t d = 0; d < D; ++d) {
          float c = static_cast<float>(x(t, d)) - mean;
          var += c * c;
        }
        var /= static_cast<float>(D);
        float inv = 1.0f / sqrt(var + eps);
        for (size_t d = 0; d < D; ++d)
          y(t, d) = static_cast<Scalar>(
              (static_cast<float>(x(t, d)) - mean) * inv);
      });
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
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, T}),
      KOKKOS_LAMBDA(size_t b, size_t t) {
        float mean = 0;
        for (size_t d = 0; d < D; ++d)
          mean += static_cast<float>(x(b, t, d));
        mean /= static_cast<float>(D);
        float var = 0;
        for (size_t d = 0; d < D; ++d) {
          float c = static_cast<float>(x(b, t, d)) - mean;
          var += c * c;
        }
        var /= static_cast<float>(D);
        float inv = 1.0f / sqrt(var + eps);
        for (size_t d = 0; d < D; ++d)
          y(b, t, d) = static_cast<Scalar>(
              (static_cast<float>(x(b, t, d)) - mean) * inv);
      });
}

}  // namespace KokkosDNN

#endif
