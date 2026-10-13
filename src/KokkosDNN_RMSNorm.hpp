#ifndef KOKKOSDNN_RMSNORM_HPP
#define KOKKOSDNN_RMSNORM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// RMSNorm (Llama variant, no mean-centering, no bias):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * w[:]
// x, y: [tokens, dim], w: [dim]. Scalar-agnostic; norm accumulated in FP32.
template <class ExecSpace, class Scalar>
void rmsnorm(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
             const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
             const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
             const std::string& label = "KokkosDNN::rmsnorm") {
  const size_t T = x.extent(0), D = x.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          y(t, d) = static_cast<Scalar>(static_cast<float>(x(t, d)) * inv *
                                        static_cast<float>(w(d)));
      });
}

// Batched variant: x, y: [batch, tokens, dim], w: [dim].
// Same per-row math as the rank-2 overload, applied independently to each
// (batch, token) row. Scalar-agnostic; norm accumulated in FP32.
// In-place safe (x and y may alias).
template <class ExecSpace, class Scalar>
void rmsnorm(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("rmsnorm: dimension mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, T}),
      KOKKOS_LAMBDA(size_t b, size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(b, t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          y(b, t, d) = static_cast<Scalar>(
              static_cast<float>(x(b, t, d)) * inv * static_cast<float>(w(d)));
      });
}

// Single-row variant for decode.
template <class ExecSpace, class Scalar>
void rmsnorm_row(const Kokkos::View<Scalar*, ExecSpace>& x,
                 const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
                 const Kokkos::View<Scalar*, ExecSpace>& y,
                 const std::string& label = "KokkosDNN::rmsnorm_row") {
  const size_t D = x.extent(0);
  // Two-pass: norm then scale. Single thread-range is fine for D=4096.
  float ss = 0;
  Kokkos::parallel_reduce(
      label + "_reduce", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        float v = static_cast<float>(x(d));
        acc += v * v;
      },
      ss);
  float inv = 1.0f / std::sqrt(ss / (float)D + eps);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        y(d) = static_cast<Scalar>(static_cast<float>(x(d)) * inv *
                                   static_cast<float>(w(d)));
      });
}

// Scaleless RMSNorm (e.g. Gemma4 `v_norm`, `with_scale=False`):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps)
// No weight multiply. Scalar-agnostic; norm accumulated in FP32.
// In-place safe (x and y may alias).
template <class ExecSpace, class Scalar>
void rmsnorm_noscale(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_noscale") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D)
    throw std::invalid_argument("rmsnorm_noscale: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          y(t, d) =
              static_cast<Scalar>(static_cast<float>(x(t, d)) * inv);
      });
}

// Gemma2 RMSNorm variant (zero-initialized weights multiply as (1 + w)):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * (1 + w[:])
// Matches HF Gemma2RMSNorm (norm in FP32, then scale, then cast).
// Scalar-agnostic; norm accumulated in FP32.
// In-place safe (x and y may alias).
// No TPL dispatch: pure elementwise dedup shared with the Sana Gemma2 text
// encoder (same justification as rmsnorm_noscale above).
template <class ExecSpace, class Scalar>
void rmsnorm_one_plus(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_one_plus") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D || w.extent(0) != D)
    throw std::invalid_argument("rmsnorm_one_plus: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          y(t, d) = static_cast<Scalar>(static_cast<float>(x(t, d)) * inv *
                                        (1.0f + static_cast<float>(w(d))));
      });
}

// Single-row scaleless variant for decode.
template <class ExecSpace, class Scalar>
void rmsnorm_row_noscale(
    const Kokkos::View<Scalar*, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_row_noscale") {
  const size_t D = x.extent(0);
  if (y.extent(0) != D)
    throw std::invalid_argument("rmsnorm_row_noscale: dimension mismatch");
  if (D == 0) return;
  float ss = 0;
  Kokkos::parallel_reduce(
      label + "_reduce", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        float v = static_cast<float>(x(d));
        acc += v * v;
      },
      ss);
  float inv = 1.0f / std::sqrt(ss / (float)D + eps);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        y(d) = static_cast<Scalar>(static_cast<float>(x(d)) * inv);
      });
}

}  // namespace KokkosDNN

#endif
