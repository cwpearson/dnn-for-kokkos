#ifndef KOKKOSDNN_RMSNORM_HPP
#define KOKKOSDNN_RMSNORM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

namespace KokkosDNN {

// RMSNorm (Llama variant, no mean-centering, no bias):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * w[:]
// x, y: [tokens, dim], w: [dim].
template <class ExecSpace>
void rmsnorm(const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
             const Kokkos::View<float*, ExecSpace>& w, float eps,
             const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y,
             const std::string& label = "KokkosDNN::rmsnorm") {
  const size_t T = x.extent(0), D = x.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) ss += x(t, d) * x(t, d);
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d) y(t, d) = x(t, d) * inv * w(d);
      });
}

// Single-row variant for decode.
template <class ExecSpace>
void rmsnorm_row(const Kokkos::View<float*, ExecSpace>& x,
                 const Kokkos::View<float*, ExecSpace>& w, float eps,
                 const Kokkos::View<float*, ExecSpace>& y,
                 const std::string& label = "KokkosDNN::rmsnorm_row") {
  const size_t D = x.extent(0);
  // Two-pass: norm then scale. Single thread-range is fine for D=4096.
  float ss = 0;
  Kokkos::parallel_reduce(
      label + "_reduce", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d, float& acc) { acc += x(d) * x(d); }, ss);
  float inv = 1.0f / std::sqrt(ss / (float)D + eps);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d) { y(d) = x(d) * inv * w(d); });
}

}  // namespace KokkosDNN

#endif
