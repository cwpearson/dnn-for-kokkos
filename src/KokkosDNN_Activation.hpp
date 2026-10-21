#ifndef KOKKOSDNN_ACTIVATION_HPP
#define KOKKOSDNN_ACTIVATION_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "KokkosDNN_SwiGLU.hpp"

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

// SiLU forward: y = x / (1 + exp(-x)), elementwise. Computed in FP32 and
// cast back to Scalar via ::KokkosDNN::silu (see KokkosDNN_SwiGLU.hpp).
// In-place safe (x and y may alias; each output depends only on the
// same-index input).
// Rank-2 variant (e.g. [tokens, dim] MLP activations).
template <class ExecSpace, class Scalar>
void silu_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::silu_forward") {
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1))
    throw std::invalid_argument("silu_forward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float v = static_cast<float>(x(i, j));
        y(i, j) = static_cast<Scalar>(::KokkosDNN::silu(v));
      });
}

// Rank-1 variant (e.g. single-row / decode path).
template <class ExecSpace, class Scalar>
void silu_forward(
    const Kokkos::View<Scalar*, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::silu_forward") {
  if (x.extent(0) != y.extent(0))
    throw std::invalid_argument("silu_forward: dimension mismatch");
  const size_t N = x.extent(0);
  if (N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t i) {
        float v = static_cast<float>(x(i));
        y(i) = static_cast<Scalar>(::KokkosDNN::silu(v));
      });
}

// Gated-GeLU multiply (e.g. Gemma MLP): out = gelu(gate) * up,
// elementwise over [tokens, ffn_dim]. NOT SwiGLU (no SiLU). The gate is
// activated with the selected GELU variant (Gemma uses GeluVariant::Tanh)
// in FP32 and multiplied by the up projection; the result is cast back to
// Scalar. In-place safe with respect to gate and up (each output depends
// only on the same-index inputs, so out may alias gate and/or up).
template <class ExecSpace, class Scalar>
void gated_gelu_multiply(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    GeluVariant variant = GeluVariant::Tanh,
    const std::string& label = "KokkosDNN::gated_gelu_multiply") {
  if (gate.extent(0) != up.extent(0) || gate.extent(1) != up.extent(1) ||
      gate.extent(0) != out.extent(0) || gate.extent(1) != out.extent(1))
    throw std::invalid_argument("gated_gelu_multiply: dimension mismatch");
  const size_t R = gate.extent(0), C = gate.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float g = static_cast<float>(gate(i, j));
        out(i, j) = static_cast<Scalar>(
            Impl::gelu_apply(g, variant) * static_cast<float>(up(i, j)));
      });
}

// Final-logit softcapping (e.g. Gemma `final_logit_softcapping`):
//   y = cap * tanh(x / cap), elementwise.
// Rank-2 variant (e.g. [tokens, vocab]). Computed in FP32 and cast back to
// Scalar. In-place safe (x and y may alias). Throws for cap <= 0.
template <class ExecSpace, class Scalar>
void logit_softcap_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float cap,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::logit_softcap") {
  if (cap <= 0.0f)
    throw std::invalid_argument("logit_softcap_forward: cap must be > 0");
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1))
    throw std::invalid_argument("logit_softcap_forward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float v = static_cast<float>(x(i, j));
        y(i, j) = static_cast<Scalar>(cap * tanhf(v / cap));
      });
}

// Rank-1 variant (e.g. single-row decode head).
template <class ExecSpace, class Scalar>
void logit_softcap_forward(
    const Kokkos::View<Scalar*, ExecSpace>& x, float cap,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::logit_softcap") {
  if (cap <= 0.0f)
    throw std::invalid_argument("logit_softcap_forward: cap must be > 0");
  if (x.extent(0) != y.extent(0))
    throw std::invalid_argument("logit_softcap_forward: dimension mismatch");
  const size_t N = x.extent(0);
  if (N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t i) {
        float v = static_cast<float>(x(i));
        y(i) = static_cast<Scalar>(cap * tanhf(v / cap));
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

// LeakyReLU forward: y = x >= 0 ? x : slope * x, elementwise. Computed
// in FP32 and cast back to Scalar. In-place safe (x and y may alias).
// Rank-2 variant (e.g. [frames, channels] conv activations).
template <class ExecSpace, class Scalar>
void leaky_relu_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    float slope,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::leaky_relu_forward") {
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1))
    throw std::invalid_argument("leaky_relu_forward: dimension mismatch");
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float v = static_cast<float>(x(i, j));
        y(i, j) = static_cast<Scalar>(v >= 0.0f ? v : slope * v);
      });
}

// Rank-1 variant.
template <class ExecSpace, class Scalar>
void leaky_relu_forward(
    const Kokkos::View<Scalar*, ExecSpace>& x, float slope,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::leaky_relu_forward") {
  if (x.extent(0) != y.extent(0))
    throw std::invalid_argument("leaky_relu_forward: dimension mismatch");
  const size_t N = x.extent(0);
  if (N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t i) {
        float v = static_cast<float>(x(i));
        y(i) = static_cast<Scalar>(v >= 0.0f ? v : slope * v);
      });
}

// Snake1D activation (ISTFTNet AdaINResBlock1): y = x + (1/a)*sin(a*x)^2,
// per-channel learnable a [C] (alpha). Time-major [T,C]. Computed in FP32,
// cast back to Scalar. In-place safe.
template <class ExecSpace, class Scalar>
void snake_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& alpha,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::snake_forward") {
  if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1) ||
      alpha.extent(0) != x.extent(1))
    throw std::invalid_argument("snake_forward: dimension mismatch");
  const size_t T = x.extent(0), C = x.extent(1);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const float v = static_cast<float>(x(t, c));
        const float a = static_cast<float>(alpha(c));
        const float s = sinf(a * v);
        y(t, c) = static_cast<Scalar>(v + (1.0f / a) * s * s);
      });
}

}  // namespace KokkosDNN

#endif
