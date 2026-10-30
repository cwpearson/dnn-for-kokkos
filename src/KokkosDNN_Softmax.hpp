// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SOFTMAX_HPP
#define KOKKOSDNN_SOFTMAX_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

namespace KokkosDNN {

// Row-wise softmax: y[i,:] = softmax(x[i,:]). Separate views allowed
// (in-place safe: pass same view for x and y). Scalar-agnostic; computed
// in FP32 and cast back to Scalar.
template <class ExecSpace, class Scalar>
void softmax_rows(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::softmax") {
  const size_t R = x.extent(0), C = x.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, R), KOKKOS_LAMBDA(size_t r) {
        float m = static_cast<float>(x(r, 0));
        for (size_t c = 1; c < C; ++c)
          m = Kokkos::fmaxf(m, static_cast<float>(x(r, c)));
        float s = 0;
        for (size_t c = 0; c < C; ++c) {
          float e = Kokkos::expf(static_cast<float>(x(r, c)) - m);
          y(r, c) = static_cast<Scalar>(e);
          s += e;
        }
        float inv = 1.0f / s;
        for (size_t c = 0; c < C; ++c)
          y(r, c) = static_cast<Scalar>(static_cast<float>(y(r, c)) * inv);
      });
}

// Causal variant over the last dim with logical length `len`:
// positions >= len are set to 0, softmax over [0, len).
template <class ExecSpace, class Scalar>
void softmax_rows_causal(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t len,
    const std::string& label = "KokkosDNN::softmax_causal") {
  const size_t R = x.extent(0), C = x.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, R), KOKKOS_LAMBDA(size_t r) {
        float m = static_cast<float>(x(r, 0));
        for (size_t c = 1; c < len; ++c)
          m = Kokkos::fmaxf(m, static_cast<float>(x(r, c)));
        float s = 0;
        for (size_t c = 0; c < len; ++c) {
          float e = Kokkos::expf(static_cast<float>(x(r, c)) - m);
          y(r, c) = static_cast<Scalar>(e);
          s += e;
        }
        float inv = 1.0f / s;
        for (size_t c = 0; c < len; ++c)
          y(r, c) = static_cast<Scalar>(static_cast<float>(y(r, c)) * inv);
        for (size_t c = len; c < C; ++c) y(r, c) = static_cast<Scalar>(0);
      });
}

}  // namespace KokkosDNN

#endif
