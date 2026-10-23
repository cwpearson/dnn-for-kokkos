// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CONV1D_HPP
#define KOKKOSDNN_CONV1D_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

// Causal depthwise 1D convolution (e.g. Mamba-style / Qwen3-Next /
// Gemma-SSM sequence mixers): one length-K kernel per channel, left zero
// padding of K-1 so output[t] only sees inputs[<=t] (causal).
//   y[t,c] = b[c] + sum_{j=0}^{K-1} w[c,j] * x_pad[t-(K-1)+j, c]
// with x_pad[s,c] = 0 for s < 0. Arbitrary K >= 1 (K=3 typical).
// Two entry points share the math:
//   depthwise_conv1d_causal_forward : full-sequence prefill, x/y [T, C].
//   depthwise_conv1d_causal_step    : single-token streaming decode with an
//     explicit [C, K-1] history cache (oldest-first); the cache is updated
//     in place, so repeated steps equal one forward pass.
// Scalar-agnostic; FP32 accumulation; Kokkos-portable (MDRange/RangePolicy
// only, no backend-specific code). Host-friendly (T=1 decode runs on any
// execution space, including Serial).

namespace KokkosDNN {

// Full-sequence causal depthwise 1D conv (prefill path).
// x [T, C] inputs, w [C, K] per-channel kernels (K >= 1), b [C] bias or
// empty (extent 0) for none, y [T, C] outputs.
template <class ExecSpace, class Scalar>
void depthwise_conv1d_causal_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::depthwise_conv1d_causal_forward") {
  const size_t T = x.extent(0), C = x.extent(1), K = w.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (w.extent(0) != C || y.extent(0) != T || y.extent(1) != C ||
      (biased && b.extent(0) != C))
    throw std::invalid_argument(
        "depthwise_conv1d_causal_forward: dimension mismatch");
  if (K < 1)
    throw std::invalid_argument(
        "depthwise_conv1d_causal_forward: kernel length K must be >= 1");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (size_t j = 0; j < K; ++j) {
          // Causal tap: kernel position j reads input t-(K-1)+j;
          // negative source indices are the left zero padding.
          const long s = static_cast<long>(t) -
                         static_cast<long>(K - 1) + static_cast<long>(j);
          if (s < 0) continue;
          acc += static_cast<float>(w(c, j)) *
                 static_cast<float>(x(static_cast<size_t>(s), c));
        }
        y(t, c) = static_cast<Scalar>(acc);
      });
}

// Single-token streaming decode step.
// x_t [C] current input, w [C, K] kernels, b [C] bias or empty,
// cache [C, K-1] history of the K-1 previous inputs, oldest-first
//   (cache[c,j] == x[t-(K-1)+j] for j = 0..K-2; zeros before position 0).
//   For K == 1 the cache has extent(1) == 0 and is untouched.
// y_t [C] output. On return the cache is shifted left by one and x_t is
// appended, ready for the next step. Stepping over a whole sequence with a
// zero-initialized cache reproduces depthwise_conv1d_causal_forward exactly.
// x_t and y_t must not alias (the cache update reads x_t after use).
template <class ExecSpace, class Scalar>
void depthwise_conv1d_causal_step(
    const Kokkos::View<Scalar*, ExecSpace>& x_t,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cache,
    const Kokkos::View<Scalar*, ExecSpace>& y_t,
    const std::string& label = "KokkosDNN::depthwise_conv1d_causal_step") {
  const size_t C = x_t.extent(0), K = w.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (w.extent(0) != C || y_t.extent(0) != C || (biased && b.extent(0) != C))
    throw std::invalid_argument(
        "depthwise_conv1d_causal_step: dimension mismatch");
  if (K < 1)
    throw std::invalid_argument(
        "depthwise_conv1d_causal_step: kernel length K must be >= 1");
  if (cache.extent(0) != C || cache.extent(1) != K - 1)
    throw std::invalid_argument(
        "depthwise_conv1d_causal_step: cache must be [C, K-1]");
  if (C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C), KOKKOS_LAMBDA(size_t c) {
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (size_t j = 0; j + 1 < K; ++j)
          acc += static_cast<float>(w(c, j)) * static_cast<float>(cache(c, j));
        acc += static_cast<float>(w(c, K - 1)) * static_cast<float>(x_t(c));
        y_t(c) = static_cast<Scalar>(acc);
        // Shift history left, append current input (no-op for K == 1).
        for (size_t j = 0; j + 1 < K - 1; ++j) cache(c, j) = cache(c, j + 1);
        if (K > 1) cache(c, K - 2) = x_t(c);
      });
}

}  // namespace KokkosDNN

#endif
