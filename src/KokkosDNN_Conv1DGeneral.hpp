// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CONV1D_GENERAL_HPP
#define KOKKOSDNN_CONV1D_GENERAL_HPP

// Dense 1D convolution and transposed 1D convolution over time-major
// activations [T, C] (LayoutRight).
//
// Covers the StyleTTS2/Kokoro convolutional stack: plain Conv1d (TextEncoder
// k=5/pad=2, F0/N proj k=1, noise_convs k=12/s=6 and k=1, asr_res k=1),
// weight-norm Conv1d (all AdainResBlk1d conv1/conv2/conv1x1, F0_conv/N_conv,
// asr_res, conv_post k=7) with dilation (ISTFTNet AdaINResBlock1 k=3,7,11
// x d=1,3,5), ConvTranspose1d (generator ups k=20/s=10, k=12/s=6; Adain
// depthwise pool k=3/s=2/groups=C), where weight-norm is fused on the host
// at load time so every kernel below sees plain dense weights.
//
//   conv1d_forward:
//     y[t,o] = b[o] + sum_{i in group(o)} sum_k
//              w[o,i,k] * x[t*s + k*d - p, i]
//     Lout = floor((Lin + 2p - d*(K-1) - 1)/s + 1).
//     w [Cout,Cg,K] with Cg = Cin/groups (groups=1: [Cout,Cin,K]).
//   conv_transpose1d_forward:
//     Lout = (Lin-1)*s - 2p + K + op.
//     y[t,o] = b[o] + sum_{s,k: s*s-p+k(+op?) == t} w[i,o',k] * x[s,i]
//     (output_padding folded as a trailing zero-insert: out-of-range reads
//     contribute nothing, so op only extends Lout with bias/zeros).
//     w [Cin,Cout/groups,K] (PyTorch ConvTranspose1d layout).
//
// Scalar-agnostic; FP32 accumulation. Bias is [Cout] or empty (extent 0)
// for none. Portable parallel_for only. Throws on dimension mismatch.

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// Dense 1D convolution, stride s, symmetric zero padding p, dilation d,
// grouped (groups=1 dense; groups=Cin=Cout depthwise with w [C,1,K]).
template <class ExecSpace, class Scalar>
void conv1d_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    size_t stride = 1, size_t padding = 0, size_t dilation = 1,
    size_t groups = 1,
    const std::string& label = "KokkosDNN::conv1d_forward") {
  const size_t Lin = x.extent(0), Cin = x.extent(1);
  const size_t Cout = w.extent(0), Cg = w.extent(1), K = w.extent(2);
  const bool biased = (b.extent(0) != 0);
  if (Cout % groups != 0 || Cin != groups * Cg || K == 0 || stride == 0 ||
      dilation == 0)
    throw std::invalid_argument("conv1d_forward: bad geometry");
  if ((biased && b.extent(0) != Cout))
    throw std::invalid_argument("conv1d_forward: bias mismatch");
  const size_t Lout =
      (Lin + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
  if (y.extent(0) != Lout || y.extent(1) != Cout)
    throw std::invalid_argument("conv1d_forward: output shape mismatch");
  if (Lin == 0 || Cout == 0) return;
  const size_t Co = Cout / groups;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {Lout, Cout}),
      KOKKOS_LAMBDA(size_t t, size_t o) {
        float acc = biased ? static_cast<float>(b(o)) : 0.0f;
        const size_t g = o / Co;
        for (size_t i = 0; i < Cg; ++i) {
          for (size_t k = 0; k < K; ++k) {
            const long s =
                static_cast<long>(t * stride + k * dilation) -
                static_cast<long>(padding);
            if (s < 0 || static_cast<size_t>(s) >= Lin) continue;
            acc += static_cast<float>(w(o, i, k)) *
                   static_cast<float>(x(static_cast<size_t>(s), g * Cg + i));
          }
        }
        y(t, o) = static_cast<Scalar>(acc);
      });
}

// Transposed 1D convolution (PyTorch ConvTranspose1d semantics).
// w [Cin, Cout/groups, K]; groups=1 dense, groups=Cin depthwise
// (w [Cin,1,K]). output_padding extends Lout with trailing bias-only rows.
template <class ExecSpace, class Scalar>
void conv_transpose1d_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    size_t stride = 1, size_t padding = 0, size_t output_padding = 0,
    size_t groups = 1,
    const std::string& label = "KokkosDNN::conv_transpose1d_forward") {
  const size_t Lin = x.extent(0), Cin = x.extent(1);
  const size_t Co = w.extent(1), K = w.extent(2);
  if (w.extent(0) != Cin || K == 0 || stride == 0)
    throw std::invalid_argument("conv_transpose1d_forward: bad geometry");
  const size_t Cout = Co * groups;
  if (Cin % groups != 0)
    throw std::invalid_argument(
        "conv_transpose1d_forward: Cin must divide by groups");
  const bool biased = (b.extent(0) != 0);
  if (biased && b.extent(0) != Cout)
    throw std::invalid_argument("conv_transpose1d_forward: bias mismatch");
  const size_t Cg = Cin / groups;
  const size_t Lout =
      (Lin - 1) * stride - 2 * padding + K + output_padding;
  if (y.extent(0) != Lout || y.extent(1) != Cout)
    throw std::invalid_argument(
        "conv_transpose1d_forward: output shape mismatch");
  if (Lin == 0 || Cout == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {Lout, Cout}),
      KOKKOS_LAMBDA(size_t t, size_t o) {
        float acc = biased ? static_cast<float>(b(o)) : 0.0f;
        const size_t g = o / Co;
        const size_t oo = o % Co;
        for (size_t i = 0; i < Cg; ++i) {
          for (size_t k = 0; k < K; ++k) {
            // t = s*stride - padding + k  =>  s = (t + padding - k)/stride.
            const long num = static_cast<long>(t) +
                             static_cast<long>(padding) -
                             static_cast<long>(k);
            if (num < 0 || num % (long)stride != 0) continue;
            const size_t s = static_cast<size_t>(num) / stride;
            if (s >= Lin) continue;
            acc += static_cast<float>(w(g * Cg + i, oo, k)) *
                   static_cast<float>(x(s, g * Cg + i));
          }
        }
        y(t, o) = static_cast<Scalar>(acc);
      });
}

}  // namespace KokkosDNN

#endif
