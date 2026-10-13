#ifndef KOKKOSDNN_DIFFUSION_HPP
#define KOKKOSDNN_DIFFUSION_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// Generic DiT diffusion conditioning ops: adaptive layer-norm modulation
// (AdaLN) and sinusoidal timestep embeddings. These apply to any diffusion
// transformer that conditions per-token features on a per-sample
// shift/scale pair and embeds integer/float timesteps into sinusoids.
//
// Historical note: these ops were first introduced alongside Sana support
// in KokkosDNN_SanaCore.hpp and are promoted here as model-agnostic
// primitives.

// AdaLN modulation:
//   y = x * (1 + scale) + shift
// Computed in FP32 and cast back to Scalar. In-place safe with respect to
// x (each output depends only on the same-index input plus the broadcast
// scale/shift row).
// Rank-2 variant: x, y: [tokens, dim]; shift, scale: [dim].
template <class ExecSpace, class Scalar>
void adaln_modulate_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& shift,
    const Kokkos::View<Scalar*, ExecSpace>& scale,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::adaln_modulate_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D || shift.extent(0) != D ||
      scale.extent(0) != D)
    throw std::invalid_argument("adaln_modulate_forward: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float v = static_cast<float>(x(t, d));
        y(t, d) = static_cast<Scalar>(
            v * (1.0f + static_cast<float>(scale(d))) +
            static_cast<float>(shift(d)));
      });
}

// Rank-3 variant: x, y: [batch, tokens, dim]; shift, scale: [batch, dim]
// broadcast over the token axis (per-sample conditioning).
template <class ExecSpace, class Scalar>
void adaln_modulate_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& shift,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& scale,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::adaln_modulate_forward") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D ||
      shift.extent(0) != B || shift.extent(1) != D || scale.extent(0) != B ||
      scale.extent(1) != D)
    throw std::invalid_argument("adaln_modulate_forward: dimension mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, D}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t d) {
        float v = static_cast<float>(x(b, t, d));
        y(b, t, d) = static_cast<Scalar>(
            v * (1.0f + static_cast<float>(scale(b, d))) +
            static_cast<float>(shift(b, d)));
      });
}

// Rank-3 variant with [batch, 1, dim] scale/shift broadcast over tokens.
template <class ExecSpace, class Scalar>
void adaln_modulate_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& shift,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& scale,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::adaln_modulate_forward") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D ||
      shift.extent(0) != B || shift.extent(1) != 1 || shift.extent(2) != D ||
      scale.extent(0) != B || scale.extent(1) != 1 || scale.extent(2) != D)
    throw std::invalid_argument("adaln_modulate_forward: dimension mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, D}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t d) {
        float v = static_cast<float>(x(b, t, d));
        y(b, t, d) = static_cast<Scalar>(
            v * (1.0f + static_cast<float>(scale(b, 0, d))) +
            static_cast<float>(shift(b, 0, d)));
      });
}

// Timestep sinusoidal embedding: t [N] -> out [N, F].
//   half = F / 2
//   freq[j] = exp(-ln(10000) * j / half), j in [0, half)
//   out[i, j]        = cos(t[i] * freq[j])
//   out[i, j + half] = sin(t[i] * freq[j])
// FP32 throughout; frequencies are recomputed per column inside the
// kernel so no temporary allocation is needed. F must be positive and even.
template <class ExecSpace>
void timestep_sinusoidal(
    const Kokkos::View<float*, ExecSpace>& t,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::timestep_sinusoidal") {
  const size_t N = t.extent(0), F = out.extent(1);
  if (out.extent(0) != N)
    throw std::invalid_argument("timestep_sinusoidal: dimension mismatch");
  if (F == 0 || F % 2 != 0)
    throw std::invalid_argument(
        "timestep_sinusoidal: embedding dim must be positive and even");
  if (N == 0) return;
  const size_t H = F / 2;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, H}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float freq = expf(-logf(10000.0f) * static_cast<float>(j) /
                          static_cast<float>(H));
        float arg = t(i) * freq;
        out(i, j) = cosf(arg);
        out(i, j + H) = sinf(arg);
      });
}

}  // namespace KokkosDNN

#endif
