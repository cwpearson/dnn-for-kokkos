#ifndef KOKKOSDNN_ACTIVATION_HPP
#define KOKKOSDNN_ACTIVATION_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// GELU variant selector (pins down the HuggingFace `ACT2FN` ambiguity):
//   HF "gelu"               -> GeluVariant::ExactErf (exact erf version,
//                              0.5*x*(1+erf(x/sqrt(2)))); this is the default.
//   HF "gelu_new" /
//      "gelu_pytorch_tanh"  -> GeluVariant::Tanh (tanh approximation,
//                              0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))).
// DINOv3 (`dinov3_vit` config) uses plain `gelu`, i.e. the exact erf form.
enum class GeluVariant { ExactErf, Tanh };

namespace Impl {

// Scalar GELU kernels (FP32 in, FP32 out; callers cast to/from Scalar).
// Device-callable (erff/tanhf resolve on host and CUDA device).
KOKKOS_INLINE_FUNCTION float gelu_exact(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

KOKKOS_INLINE_FUNCTION float gelu_tanh_approx(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  const float u = c * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(u));
}

KOKKOS_INLINE_FUNCTION float gelu_apply(float x, GeluVariant variant) {
  return variant == GeluVariant::Tanh ? gelu_tanh_approx(x) : gelu_exact(x);
}

}  // namespace Impl

// GELU forward: y = gelu(x), elementwise. Out-of-place-capable and
// in-place safe (x and y may alias; each output depends only on the
// same-index input). Computed in FP32 and cast back to Scalar.
// Rank-2 variant (e.g. [tokens, dim] MLP activations).
template <class ExecSpace, class Scalar>
void gelu_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    GeluVariant variant = GeluVariant::ExactErf,
    const std::string& label = "KokkosDNN::gelu_forward") {
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1))
    throw std::invalid_argument("gelu_forward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float v = static_cast<float>(x(i, j));
        y(i, j) = static_cast<Scalar>(Impl::gelu_apply(v, variant));
      });
}

// Rank-1 variant (e.g. single-row / decode path).
template <class ExecSpace, class Scalar>
void gelu_forward(
    const Kokkos::View<Scalar*, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    GeluVariant variant = GeluVariant::ExactErf,
    const std::string& label = "KokkosDNN::gelu_forward") {
  if (x.extent(0) != y.extent(0))
    throw std::invalid_argument("gelu_forward: dimension mismatch");
  const size_t N = x.extent(0);
  if (N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t i) {
        float v = static_cast<float>(x(i));
        y(i) = static_cast<Scalar>(Impl::gelu_apply(v, variant));
      });
}

// Elementwise activations for small training MLPs (e.g. the XOR example).
// All ops are out-of-place-capable and in-place safe (x and y may alias;
// each output element depends only on the same-index input). Computed in
// FP32 and cast back to Scalar.
//
// Forward: y = f(x). Backward: dX = dY * f'(.), where the derivative is
// expressed through the cached forward output (sigmoid/tanh) so the
// backward pass needs no re-evaluation of exp:
//   sigmoid: dX = dY * Y * (1 - Y)
//   tanh:    dX = dY * (1 - Y*Y)
//   relu:    dX = dY * (X > 0)  (needs the pre-activation X, not Y)

template <class ExecSpace, class Scalar>
void sigmoid_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::sigmoid_forward") {
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1))
    throw std::invalid_argument("sigmoid_forward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float v = static_cast<float>(x(i, j));
        y(i, j) = static_cast<Scalar>(1.0f / (1.0f + expf(-v)));
      });
}

template <class ExecSpace, class Scalar>
void sigmoid_backward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dy,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dx,
    const std::string& label = "KokkosDNN::sigmoid_backward") {
  if (y.extent(0) != dy.extent(0) || y.extent(1) != dy.extent(1) ||
      y.extent(0) != dx.extent(0) || y.extent(1) != dx.extent(1))
    throw std::invalid_argument("sigmoid_backward: dimension mismatch");
  const size_t R = y.extent(0), C = y.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float s = static_cast<float>(y(i, j));
        dx(i, j) =
            static_cast<Scalar>(static_cast<float>(dy(i, j)) * s * (1.0f - s));
      });
}

template <class ExecSpace, class Scalar>
void tanh_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::tanh_forward") {
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1))
    throw std::invalid_argument("tanh_forward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        y(i, j) = static_cast<Scalar>(tanhf(static_cast<float>(x(i, j))));
      });
}

template <class ExecSpace, class Scalar>
void tanh_backward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dy,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dx,
    const std::string& label = "KokkosDNN::tanh_backward") {
  if (y.extent(0) != dy.extent(0) || y.extent(1) != dy.extent(1) ||
      y.extent(0) != dx.extent(0) || y.extent(1) != dx.extent(1))
    throw std::invalid_argument("tanh_backward: dimension mismatch");
  const size_t R = y.extent(0), C = y.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float t = static_cast<float>(y(i, j));
        dx(i, j) = static_cast<Scalar>(static_cast<float>(dy(i, j)) *
                                       (1.0f - t * t));
      });
}

template <class ExecSpace, class Scalar>
void relu_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::relu_forward") {
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1))
    throw std::invalid_argument("relu_forward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float v = static_cast<float>(x(i, j));
        y(i, j) = static_cast<Scalar>(v > 0.0f ? v : 0.0f);
      });
}

template <class ExecSpace, class Scalar>
void relu_backward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dy,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dx,
    const std::string& label = "KokkosDNN::relu_backward") {
  if (x.extent(0) != dy.extent(0) || x.extent(1) != dy.extent(1) ||
      x.extent(0) != dx.extent(0) || x.extent(1) != dx.extent(1))
    throw std::invalid_argument("relu_backward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float gate = static_cast<float>(x(i, j)) > 0.0f ? 1.0f : 0.0f;
        dx(i, j) =
            static_cast<Scalar>(static_cast<float>(dy(i, j)) * gate);
      });
}

}  // namespace KokkosDNN

#endif
