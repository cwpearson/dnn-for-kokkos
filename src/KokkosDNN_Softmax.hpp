#ifndef KOKKOSDNN_SOFTMAX_HPP
#define KOKKOSDNN_SOFTMAX_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

namespace KokkosDNN {

// Row-wise softmax: y[i,:] = softmax(x[i,:]). Separate views allowed
// (in-place safe: pass same view for x and y).
template <class ExecSpace>
void softmax_rows(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::softmax") {
  const size_t R = x.extent(0), C = x.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, R), KOKKOS_LAMBDA(size_t r) {
        float m = x(r, 0);
        for (size_t c = 1; c < C; ++c) m = fmaxf(m, x(r, c));
        float s = 0;
        for (size_t c = 0; c < C; ++c) {
          float e = expf(x(r, c) - m);
          y(r, c) = e;
          s += e;
        }
        float inv = 1.0f / s;
        for (size_t c = 0; c < C; ++c) y(r, c) *= inv;
      });
}

// Causal variant over the last dim with logical length `len`:
// positions >= len are set to 0, softmax over [0, len).
template <class ExecSpace>
void softmax_rows_causal(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y, size_t len,
    const std::string& label = "KokkosDNN::softmax_causal") {
  const size_t R = x.extent(0), C = x.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, R), KOKKOS_LAMBDA(size_t r) {
        float m = x(r, 0);
        for (size_t c = 1; c < len; ++c) m = fmaxf(m, x(r, c));
        float s = 0;
        for (size_t c = 0; c < len; ++c) {
          float e = expf(x(r, c) - m);
          y(r, c) = e;
          s += e;
        }
        float inv = 1.0f / s;
        for (size_t c = 0; c < len; ++c) y(r, c) *= inv;
        for (size_t c = len; c < C; ++c) y(r, c) = 0;
      });
}

}  // namespace KokkosDNN

#endif
