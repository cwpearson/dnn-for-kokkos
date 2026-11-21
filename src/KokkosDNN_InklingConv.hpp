// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLINGCONV_HPP
#define KOKKOSDNN_INKLINGCONV_HPP

// Inkling2 loop short causal convolutions (thinkingmachines/Inkling-2-Mini
// text decoder): static depthwise convolution with residual normalization,
// and the dynamic key/value variant whose taps add per-token projected
// weights to static per-channel weights. All accumulation is FP32 with the
// result cast back to Scalar. Backend-portable: only parallel_for kernels;
// no backend-specific code.
//
// Weight layout convention (torch cross-correlation order):
//   w[d, 0] weights the oldest tap, w[d, K-1] the current token, i.e.
//     conv(t, d) = sum_{k=0..K-1} w[d, k] * x[t-(K-1)+k, d]
// with zero padding for out-of-range indices. This matches
// torch.nn.Conv1d (padding K-1) used by Inkling2LoopShortConvolution.

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// Static depthwise causal short convolution with residual normalization:
//   y[t, d] = (conv(t, d) + x[t, d]) / sqrt(K + 1)
// x, y: [T, D]; weight: [D, K] (torch order, see above). Matches
// Inkling2LoopShortConvolution. In-place safe for x/y.
template <class ExecSpace, class Scalar>
void inkling_short_conv_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& weight,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_short_conv_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  const size_t K = weight.extent(1);
  if (weight.extent(0) != D || y.extent(0) != T || y.extent(1) != D)
    throw std::invalid_argument("inkling_short_conv_forward: dim mismatch");
  if (K == 0)
    throw std::invalid_argument("inkling_short_conv_forward: K must be > 0");
  if (T == 0 || D == 0) return;
  const float inv = 1.0f / Kokkos::sqrtf(static_cast<float>(K + 1));
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k) {
          const int s = static_cast<int>(t) - static_cast<int>(K - 1) +
                        static_cast<int>(k);
          if (s < 0) continue;
          acc += static_cast<float>(weight(d, k)) *
                 static_cast<float>(x(static_cast<size_t>(s), d));
        }
        acc = (acc + static_cast<float>(x(t, d))) * inv;
        y(t, d) = static_cast<Scalar>(acc);
      });
}

// Single-token stateful static short convolution (decode):
//   window = [state[d, 0..K-2], x_new[d]] (torch tap order),
//   y[d] = (sum_k w[d,k]*window[k] + x_new[d]) / sqrt(K+1),
// then state shifts left by one and appends x_new.
// x_new, y: [D]; state: [D, K-1] holding the previous K-1 inputs with
// state[d, 0] the oldest; weight: [D, K]. Matches the cache path of
// Inkling2LoopShortConvolution (causal_conv1d_update).
template <class ExecSpace, class Scalar>
void inkling_short_conv_update_state(
    const Kokkos::View<Scalar*, ExecSpace>& x_new,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& state,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_short_conv_update_state") {
  const size_t D = x_new.extent(0);
  const size_t K = weight.extent(1);
  if (weight.extent(0) != D || y.extent(0) != D ||
      state.extent(0) != D || state.extent(1) != (K > 0 ? K - 1 : 0))
    throw std::invalid_argument(
        "inkling_short_conv_update_state: dimension mismatch");
  if (K == 0)
    throw std::invalid_argument(
        "inkling_short_conv_update_state: K must be > 0");
  if (D == 0) return;
  const float inv = 1.0f / Kokkos::sqrtf(static_cast<float>(K + 1));
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        const float xn = static_cast<float>(x_new(d));
        float acc = 0;
        for (size_t k = 0; k < K - 1; ++k)
          acc += static_cast<float>(weight(d, k)) *
                 static_cast<float>(state(d, k));
        acc += static_cast<float>(weight(d, K - 1)) * xn;
        y(d) = static_cast<Scalar>((acc + xn) * inv);
        for (size_t k = 0; k + 1 < K - 1; ++k) state(d, k) = state(d, k + 1);
        // Append the snapshot, not a re-read: x_new may alias y (the Model
        // calls these in place), and y(d) is already overwritten here.
        if (K > 1) state(d, K - 2) = static_cast<Scalar>(xn);
      });
}

// Dynamic short convolution (full-attention K/V path):
// taps are the sum of static per-channel weights and per-token weights
// projected from `dynamic_input`, shared within each key/value head:
//   tap(t,h,k) = static_w[k, c] + dyn[t, h*K+k],  c = head-major channel
//   conv(t,c)  = sum_k tap(t,h,k) * x[t-(K-1)+k, c]
//   y[t,c]     = (conv(t,c) + x[t,c]) / sqrt(2K+1)
// x, y: [T, D] with D = nheads * head_dim; static_w: [K, D] (torch
// Inkling2LoopDynamicShortConvolution.static_weight layout); dyn: [T, H*K]
// (the weight_proj output). Matches Inkling2LoopDynamicShortConvolution.
// In-place safe for x/y.
template <class ExecSpace, class Scalar, class WView>
void inkling_dynamic_short_conv_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const WView& static_w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dyn,
    size_t nheads, size_t head_dim,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label =
        "KokkosDNN::inkling_dynamic_short_conv_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  const size_t K = static_w.extent(0);
  if (K == 0)
    throw std::invalid_argument(
        "inkling_dynamic_short_conv_forward: K must be > 0");
  if (static_w.extent(1) != D || dyn.extent(0) != T ||
      dyn.extent(1) != nheads * K || y.extent(0) != T || y.extent(1) != D ||
      D != nheads * head_dim)
    throw std::invalid_argument(
        "inkling_dynamic_short_conv_forward: dimension mismatch");
  if (T == 0 || D == 0) return;
  const float inv = 1.0f / Kokkos::sqrtf(static_cast<float>(2 * K + 1));
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const size_t h = c / head_dim;
        float acc = 0;
        for (size_t k = 0; k < K; ++k) {
          const int s = static_cast<int>(t) - static_cast<int>(K - 1) +
                        static_cast<int>(k);
          if (s < 0) continue;
          const float tap = static_cast<float>(static_w(k, c)) +
                            static_cast<float>(dyn(t, h * K + k));
          acc += tap * static_cast<float>(x(static_cast<size_t>(s), c));
        }
        acc = (acc + static_cast<float>(x(t, c))) * inv;
        y(t, c) = static_cast<Scalar>(acc);
      });
}

// Single-token stateful dynamic short convolution (decode):
// window[d] = [state[d, 0..K-2], x_new[d]], tap(h,k) = static_w[k,c] +
// dyn_row[h*K+k], y[c] = (sum_k tap*window[k] + x_new[c]) / sqrt(2K+1);
// state shifts left and appends x_new. Layouts mirror the full-sequence
// overload with T = 1.
template <class ExecSpace, class Scalar, class SWView>
void inkling_dynamic_short_conv_update_state(
    const Kokkos::View<Scalar*, ExecSpace>& x_new,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& state,
    const SWView& static_w,
    const Kokkos::View<Scalar*, ExecSpace>& dyn_row, size_t nheads,
    size_t head_dim, const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label =
        "KokkosDNN::inkling_dynamic_short_conv_update_state") {
  const size_t D = x_new.extent(0);
  const size_t K = static_w.extent(0);
  if (K == 0)
    throw std::invalid_argument(
        "inkling_dynamic_short_conv_update_state: K must be > 0");
  if (static_w.extent(1) != D || dyn_row.extent(0) != nheads * K ||
      y.extent(0) != D || state.extent(0) != D || state.extent(1) != K - 1 ||
      D != nheads * head_dim)
    throw std::invalid_argument(
        "inkling_dynamic_short_conv_update_state: dimension mismatch");
  if (D == 0) return;
  const float inv = 1.0f / Kokkos::sqrtf(static_cast<float>(2 * K + 1));
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t c) {
        const size_t h = c / head_dim;
        const float xn = static_cast<float>(x_new(c));
        float acc = 0;
        for (size_t k = 0; k < K - 1; ++k) {
          const float tap = static_cast<float>(static_w(k, c)) +
                            static_cast<float>(dyn_row(h * K + k));
          acc += tap * static_cast<float>(state(c, k));
        }
        const float tap_last = static_cast<float>(static_w(K - 1, c)) +
                               static_cast<float>(dyn_row(h * K + K - 1));
        acc += tap_last * xn;
        y(c) = static_cast<Scalar>((acc + xn) * inv);
        for (size_t k = 0; k + 1 < K - 1; ++k) state(c, k) = state(c, k + 1);
        // Append the snapshot (x_new may alias y; see above).
        if (K > 1) state(c, K - 2) = static_cast<Scalar>(xn);
      });
}

}  // namespace KokkosDNN

#endif
