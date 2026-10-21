#ifndef KOKKOSDNN_CONV1DGENERIC_HPP
#define KOKKOSDNN_CONV1DGENERIC_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

// Generic 1D convolution / transposed convolution over [B, C, T]
// (LayoutRight, channel-middle) as used by TTS vocoders (StyleTTS2 /
// ISTFTNet). Native portable kernels only (no TPL dispatch); FP32
// accumulation, scalar-agnostic.
//
//   conv1d_forward: y[b, o, to] = b[o] + sum_{i,k} w[o, i, k] *
//       x[b, i + g*?, to*stride - pad + k*dilation]
//     w [Cout, Cin/groups, K], x [B, Cin, Tin], y [B, Cout, Tout] with
//     Tout = (Tin + 2*pad - dilation*(K-1) - 1) / stride + 1.
//     groups divides Cin and Cout; group g owns outputs
//     [g*Cog, (g+1)*Cog) and inputs [g*Cig, (g+1)*Cig).
//
//   convtranspose1d_forward: y[b, o, to] with
//     Tout = (Tin-1)*stride - 2*pad + dilation*(K-1) + output_pad + 1.
//     w [Cin, Cout/groups, K] (PyTorch ConvTranspose1d layout), groups
//     divides Cin and Cout.
//
// Bias may be empty (extent 0) for no bias. Throws on dimension mismatch.

namespace KokkosDNN {

template <class ExecSpace, class Scalar>
void conv1d_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& bias,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    size_t stride = 1, size_t pad = 0, size_t dilation = 1, size_t groups = 1,
    const std::string& label = "KokkosDNN::conv1d_forward") {
  const size_t B = x.extent(0), Cin = x.extent(1), Tin = x.extent(2);
  const size_t Cout = w.extent(0), Cpg = w.extent(1), K = w.extent(2);
  const bool biased = bias.extent(0) != 0;
  if (groups == 0) throw std::invalid_argument("conv1d_forward: groups==0");
  if (Cout % groups != 0 || Cin != groups * Cpg)
    throw std::invalid_argument("conv1d_forward: group mismatch");
  if (biased && bias.extent(0) != Cout)
    throw std::invalid_argument("conv1d_forward: bias mismatch");
  const size_t keff = dilation * (K - 1) + 1;
  if (Tin + 2 * pad < keff)
    throw std::invalid_argument("conv1d_forward: input too short");
  const size_t Tout = (Tin + 2 * pad - keff) / stride + 1;
  if (y.extent(0) != B || y.extent(1) != Cout || y.extent(2) != Tout)
    throw std::invalid_argument("conv1d_forward: output mismatch");
  if (B == 0 || Cout == 0 || Tout == 0) return;
  const size_t Cog = Cout / groups;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, Cout, Tout}),
      KOKKOS_LAMBDA(size_t b, size_t o, size_t to) {
        const size_t g = o / Cog;
        float acc = biased ? static_cast<float>(bias(o)) : 0.0f;
        for (size_t k = 0; k < K; ++k) {
          const long ti = static_cast<long>(to * stride + k * dilation) -
                          static_cast<long>(pad);
          if (ti < 0 || static_cast<size_t>(ti) >= Tin) continue;
          for (size_t i = 0; i < Cpg; ++i)
            acc += static_cast<float>(w(o, i, k)) *
                   static_cast<float>(x(b, g * Cpg + i, (size_t)ti));
        }
        y(b, o, to) = static_cast<Scalar>(acc);
      });
}

template <class ExecSpace, class Scalar>
void convtranspose1d_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& bias,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    size_t stride = 1, size_t pad = 0, size_t output_pad = 0,
    size_t groups = 1, size_t dilation = 1,
    const std::string& label = "KokkosDNN::convtranspose1d_forward") {
  const size_t B = x.extent(0), Cin = x.extent(1), Tin = x.extent(2);
  const size_t Cpg = w.extent(1), K = w.extent(2);
  const size_t Wcin = w.extent(0);
  const bool biased = bias.extent(0) != 0;
  if (groups == 0)
    throw std::invalid_argument("convtranspose1d_forward: groups==0");
  if (Wcin != Cin)
    throw std::invalid_argument(
        "convtranspose1d_forward: weight in-channels mismatch");
  const size_t Cout = y.extent(1);
  if (Cpg * groups != Cout)
    throw std::invalid_argument(
        "convtranspose1d_forward: weight out-channels mismatch");
  if (Cout % groups != 0 || Cin % groups != 0)
    throw std::invalid_argument("convtranspose1d_forward: group mismatch");
  if (biased && bias.extent(0) != Cout)
    throw std::invalid_argument("convtranspose1d_forward: bias mismatch");
  const size_t Tout =
      (Tin - 1) * stride - 2 * pad + dilation * (K - 1) + output_pad + 1;
  if (y.extent(0) != B || y.extent(2) != Tout)
    throw std::invalid_argument("convtranspose1d_forward: output mismatch");
  if (B == 0 || Cout == 0 || Tout == 0) return;
  const size_t Cig = Cin / groups, Cog = Cout / groups;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, Cout, Tout}),
      KOKKOS_LAMBDA(size_t b, size_t o, size_t to) {
        const size_t g = o / Cog;
        const size_t oc = o % Cog;
        float acc = biased ? static_cast<float>(bias(o)) : 0.0f;
        for (size_t k = 0; k < K; ++k) {
          const long num =
              static_cast<long>(to + pad) - static_cast<long>(k * dilation);
          if (num < 0) continue;
          if (static_cast<size_t>(num) % stride != 0) continue;
          const size_t ti = static_cast<size_t>(num) / stride;
          if (ti >= Tin) continue;
          for (size_t i = 0; i < Cig; ++i)
            acc += static_cast<float>(x(b, g * Cig + i, ti)) *
                   static_cast<float>(w(g * Cig + i, oc, k));
        }
        y(b, o, to) = static_cast<Scalar>(acc);
      });
}

}  // namespace KokkosDNN

#endif
