#ifndef KOKKOSDNN_SANACORE_HPP
#define KOKKOSDNN_SANACORE_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "KokkosDNN_SwiGLU.hpp"

namespace KokkosDNN {

// SiLU forward: y = x / (1 + exp(-x)), elementwise. Computed in FP32 and
// cast back to Scalar. In-place safe (x and y may alias; each output
// depends only on the same-index input).
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
        y(i, j) = static_cast<Scalar>(silu(v));
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
        y(i) = static_cast<Scalar>(silu(v));
      });
}

// Sana RMSNorm with a pre-scaled weight (weight init ones * scale_factor
// is already folded into the weights, so this is the standard RMSNorm
// weight multiply):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * w[:]
// Scalar-agnostic; norm accumulated in FP32. In-place safe.
// Rank-2 variant ([tokens, dim]).
template <class ExecSpace, class Scalar>
void rmsnorm_scaled_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_scaled_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D || w.extent(0) != D)
    throw std::invalid_argument("rmsnorm_scaled_forward: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          y(t, d) = static_cast<Scalar>(static_cast<float>(x(t, d)) * inv *
                                        static_cast<float>(w(d)));
      });
}

// Rank-3 variant ([batch, tokens, dim], e.g. Sana [B,T,D] / [B,L,D]).
template <class ExecSpace, class Scalar>
void rmsnorm_scaled_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_scaled_forward") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("rmsnorm_scaled_forward: dimension mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, T}),
      KOKKOS_LAMBDA(size_t b, size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(b, t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          y(b, t, d) = static_cast<Scalar>(
              static_cast<float>(x(b, t, d)) * inv * static_cast<float>(w(d)));
      });
}

// LayerNorm without affine parameters (Sana `LayerNorm` with
// `elementwise_affine=False`):
//   y[i,:] = (x[i,:] - mean(x[i,:])) / sqrt(var(x[i,:]) + eps)
// Scalar-agnostic; statistics accumulated in FP32. In-place safe.
// Rank-2 variant ([tokens, dim]).
template <class ExecSpace, class Scalar>
void layernorm_noaffine_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::layernorm_noaffine_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D)
    throw std::invalid_argument("layernorm_noaffine_forward: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        float mean = 0;
        for (size_t d = 0; d < D; ++d)
          mean += static_cast<float>(x(t, d));
        mean /= static_cast<float>(D);
        float var = 0;
        for (size_t d = 0; d < D; ++d) {
          float c = static_cast<float>(x(t, d)) - mean;
          var += c * c;
        }
        var /= static_cast<float>(D);
        float inv = 1.0f / sqrt(var + eps);
        for (size_t d = 0; d < D; ++d)
          y(t, d) = static_cast<Scalar>(
              (static_cast<float>(x(t, d)) - mean) * inv);
      });
}

// Rank-3 variant ([batch, tokens, dim]).
template <class ExecSpace, class Scalar>
void layernorm_noaffine_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::layernorm_noaffine_forward") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D)
    throw std::invalid_argument("layernorm_noaffine_forward: dimension mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, T}),
      KOKKOS_LAMBDA(size_t b, size_t t) {
        float mean = 0;
        for (size_t d = 0; d < D; ++d)
          mean += static_cast<float>(x(b, t, d));
        mean /= static_cast<float>(D);
        float var = 0;
        for (size_t d = 0; d < D; ++d) {
          float c = static_cast<float>(x(b, t, d)) - mean;
          var += c * c;
        }
        var /= static_cast<float>(D);
        float inv = 1.0f / sqrt(var + eps);
        for (size_t d = 0; d < D; ++d)
          y(b, t, d) = static_cast<Scalar>(
              (static_cast<float>(x(b, t, d)) - mean) * inv);
      });
}

// AdaLN modulation (Sana `t2i_modulate` / final-layer modulate):
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
// broadcast over the token axis (Sana SanaBlock t2i_modulate).
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

// Sana timestep sinusoidal embedding: t [N] -> out [N, F].
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
