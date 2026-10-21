#ifndef KOKKOSDNN_TTSACTIVATIONS_HPP
#define KOKKOSDNN_TTSACTIVATIONS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// LeakyReLU: y = x > 0 ? x : slope * x. Rank-3 [B,C,T] channel-time layout
// used by TTS conv stacks, computed in FP32. In-place safe (each output
// depends only on the same-index input).
template <class ExecSpace, class Scalar>
void leaky_relu_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    float slope = 0.2f,
    const std::string& label = "KokkosDNN::leaky_relu") {
  if (y.extent(0) != x.extent(0) || y.extent(1) != x.extent(1) ||
      y.extent(2) != x.extent(2))
    throw std::invalid_argument("leaky_relu_forward: dimension mismatch");
  const size_t B = x.extent(0), C = x.extent(1), T = x.extent(2);
  if (B == 0 || C == 0 || T == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, C, T}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t t) {
        const float v = static_cast<float>(x(b, c, t));
        y(b, c, t) = static_cast<Scalar>(v > 0.0f ? v : slope * v);
      });
}

// Rank-2 variant ([rows, cols], e.g. flattened token MLPs).
template <class ExecSpace, class Scalar>
void leaky_relu_forward_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    float slope = 0.2f,
    const std::string& label = "KokkosDNN::leaky_relu_2d") {
  if (y.extent(0) != x.extent(0) || y.extent(1) != x.extent(1))
    throw std::invalid_argument("leaky_relu_forward_2d: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        const float v = static_cast<float>(x(i, j));
        y(i, j) = static_cast<Scalar>(v > 0.0f ? v : slope * v);
      });
}

// SnakeBeta activation (StyleTTS2 AdaINResBlock1, per-channel alpha):
//   y = x + (1/alpha) * sin(alpha * x)^2
// x/y [B,C,T], alpha [1,C,1] (broadcast over B,T). FP32, in-place safe.
template <class ExecSpace, class Scalar>
void snake_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& alpha,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::snake") {
  const size_t B = x.extent(0), C = x.extent(1), T = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != C || y.extent(2) != T ||
      alpha.extent(0) != 1 || alpha.extent(1) != C || alpha.extent(2) != 1)
    throw std::invalid_argument("snake_forward: dimension mismatch");
  if (B == 0 || C == 0 || T == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, C, T}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t t) {
        const float v = static_cast<float>(x(b, c, t));
        const float a = static_cast<float>(alpha(0, c, 0));
        const float s = sinf(a * v);
        y(b, c, t) = static_cast<Scalar>(v + (1.0f / a) * s * s);
      });
}

}  // namespace KokkosDNN

#endif
