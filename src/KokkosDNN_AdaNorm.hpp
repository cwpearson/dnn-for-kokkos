#ifndef KOKKOSDNN_ADANORM_HPP
#define KOKKOSDNN_ADANORM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {

// Channel-wise norms over [B, C, T] (LayoutRight) for TTS blocks.
//
// instancenorm_noaffine_forward: per-(b,c) mean/var over T:
//   y = (x - mean) / sqrt(var + eps). Scalar-agnostic, FP32 stats.
// adain_forward: InstanceNorm (no affine) modulated by a style vector:
//   (gamma, beta) = fc(s) split [2C]; y = (1+gamma)*norm + beta.
//   fc_w [2C, S], fc_b [2C] (bias may be empty).
// adalayernorm_forward: LayerNorm (no affine) over C per-(b,t), modulated:
//   (gamma, beta) = fc(s) split [2C]; y = (1+gamma)*norm + beta.
// NOTE: x and y must not alias (each output depends on a full row).

template <class ExecSpace, class Scalar>
void instancenorm_noaffine_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    float eps = 1e-5f,
    const std::string& label = "KokkosDNN::instancenorm_noaffine") {
  const size_t B = x.extent(0), C = x.extent(1), T = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != C || y.extent(2) != T)
    throw std::invalid_argument("instancenorm_noaffine: dimension mismatch");
  if (B == 0 || C == 0 || T == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, C}),
      KOKKOS_LAMBDA(size_t b, size_t c) {
        float sum = 0;
        for (size_t t = 0; t < T; ++t) sum += static_cast<float>(x(b, c, t));
        const float mean = sum / static_cast<float>(T);
        float sq = 0;
        for (size_t t = 0; t < T; ++t) {
          const float d = static_cast<float>(x(b, c, t)) - mean;
          sq += d * d;
        }
        const float inv = 1.0f / sqrtf(sq / static_cast<float>(T) + eps);
        for (size_t t = 0; t < T; ++t)
          y(b, c, t) = static_cast<Scalar>(
              (static_cast<float>(x(b, c, t)) - mean) * inv);
      });
}

template <class ExecSpace, class Scalar>
void adain_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& fc_w,
    const Kokkos::View<Scalar*, ExecSpace>& fc_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& style,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    float eps = 1e-5f,
    const std::string& label = "KokkosDNN::adain") {
  const size_t B = x.extent(0), C = x.extent(1), T = x.extent(2);
  const size_t S = style.extent(1);
  if (style.extent(0) != B || fc_w.extent(0) != 2 * C ||
      fc_w.extent(1) != S || y.extent(0) != B || y.extent(1) != C ||
      y.extent(2) != T)
    throw std::invalid_argument("adain_forward: dimension mismatch");
  const bool biased = fc_b.extent(0) != 0;
  if (biased && fc_b.extent(0) != 2 * C)
    throw std::invalid_argument("adain_forward: bias mismatch");
  if (B == 0 || C == 0 || T == 0) return;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> gb("adain_gb", B,
                                                             2 * C);
  if (biased) {
    gemm_bias(handle, style, fc_w, fc_b, gb, label + "_fc");
  } else {
    gemm(handle, Scalar(1), style, fc_w, Scalar(0), gb, label + "_fc");
  }
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, C}),
      KOKKOS_LAMBDA(size_t b, size_t c) {
        float sum = 0;
        for (size_t t = 0; t < T; ++t) sum += static_cast<float>(x(b, c, t));
        const float mean = sum / static_cast<float>(T);
        float sq = 0;
        for (size_t t = 0; t < T; ++t) {
          const float d = static_cast<float>(x(b, c, t)) - mean;
          sq += d * d;
        }
        const float inv = 1.0f / sqrtf(sq / static_cast<float>(T) + eps);
        const float gamma = static_cast<float>(gb(b, c));
        const float beta = static_cast<float>(gb(b, C + c));
        for (size_t t = 0; t < T; ++t)
          y(b, c, t) = static_cast<Scalar>(
              (1.0f + gamma) * (static_cast<float>(x(b, c, t)) - mean) * inv +
              beta);
      });
}

template <class ExecSpace, class Scalar>
void adalayernorm_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& fc_w,
    const Kokkos::View<Scalar*, ExecSpace>& fc_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& style,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    float eps = 1e-5f,
    const std::string& label = "KokkosDNN::adalayernorm") {
  const size_t B = x.extent(0), C = x.extent(1), T = x.extent(2);
  const size_t S = style.extent(1);
  if (style.extent(0) != B || fc_w.extent(0) != 2 * C ||
      fc_w.extent(1) != S || y.extent(0) != B || y.extent(1) != C ||
      y.extent(2) != T)
    throw std::invalid_argument("adalayernorm_forward: dimension mismatch");
  const bool biased = fc_b.extent(0) != 0;
  if (biased && fc_b.extent(0) != 2 * C)
    throw std::invalid_argument("adalayernorm_forward: bias mismatch");
  if (B == 0 || C == 0 || T == 0) return;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> gb("adaln_gb", B,
                                                             2 * C);
  if (biased) {
    gemm_bias(handle, style, fc_w, fc_b, gb, label + "_fc");
  } else {
    gemm(handle, Scalar(1), style, fc_w, Scalar(0), gb, label + "_fc");
  }
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, T}),
      KOKKOS_LAMBDA(size_t b, size_t t) {
        float sum = 0;
        for (size_t c = 0; c < C; ++c) sum += static_cast<float>(x(b, c, t));
        const float mean = sum / static_cast<float>(C);
        float sq = 0;
        for (size_t c = 0; c < C; ++c) {
          const float d = static_cast<float>(x(b, c, t)) - mean;
          sq += d * d;
        }
        const float inv = 1.0f / sqrtf(sq / static_cast<float>(C) + eps);
        for (size_t c = 0; c < C; ++c) {
          const float gamma = static_cast<float>(gb(b, c));
          const float beta = static_cast<float>(gb(b, C + c));
          y(b, c, t) = static_cast<Scalar>(
              (1.0f + gamma) * (static_cast<float>(x(b, c, t)) - mean) * inv +
              beta);
        }
      });
}

}  // namespace KokkosDNN

#endif
