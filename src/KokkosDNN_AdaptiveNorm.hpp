#ifndef KOKKOSDNN_ADAPTIVE_NORM_HPP
#define KOKKOSDNN_ADAPTIVE_NORM_HPP

// Channel-wise normalization family for time-major [T, C] activations,
// covering the StyleTTS2/Kokoro style-conditioning blocks:
//
//   instance_norm1d  : y[t,c] = (x[t,c] - mean_c) / sqrt(var_c + eps),
//                     statistics over time per channel (affine-free,
//                     matches nn.InstanceNorm1d(affine=False)).
//   adain1d_forward : AdaIN1d: instance-norm then style modulation
//                     y = (1 + gamma) * n + beta, with [gamma;beta] =
//                     fc(s) split in halves (fc [2C,S], bias [2C]).
//   adaln_forward   : AdaLayerNorm: layer-norm over channels per time step
//                     (eps) then the same style modulation with fc(s).
//
// All Scalar-agnostic (FP32 accumulation, cast back). Portable parallel_for
// only. In-place safe (each output depends only on its own input element
// plus row statistics, reduced before any write in the same launch only
// where needed; adain/adaln need statistics first, so they use two launches
// internally via instance/layer passes into a temp).

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "KokkosDNN_LayerNorm.hpp"

namespace KokkosDNN {
namespace Impl {

template <class ExecSpace, class Scalar>
struct InstanceNormStatsFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x;
  Kokkos::View<float*, ExecSpace> mean, inv;
  float eps;
  size_t T;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto X = x;
    const size_t c = (size_t)member.league_rank();
    float sum = 0.0f;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, T),
        KOKKOS_LAMBDA(size_t t, float& acc) {
          acc += static_cast<float>(X(t, c));
        },
        sum);
    const float mu = sum / (float)T;
    float sq = 0.0f;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, T),
        KOKKOS_LAMBDA(size_t t, float& acc) {
          const float d = static_cast<float>(X(t, c)) - mu;
          acc += d * d;
        },
        sq);
    mean(c) = mu;
    inv(c) = 1.0f / sqrtf(sq / (float)T + eps);
  }
};

}  // namespace Impl

// InstanceNorm1d (affine-free) over time-major [T,C]:
// per-channel mean/var over the T steps.
template <class ExecSpace, class Scalar>
void instance_norm1d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::instance_norm1d") {
  const size_t T = x.extent(0), C = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != C)
    throw std::invalid_argument("instance_norm1d: dimension mismatch");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("instance_norm1d: bad eps");
  if (T == 0 || C == 0) return;
  // Two-pass: per-channel stats (one team per channel), then normalize.
  Kokkos::View<float*, ExecSpace> mean("instnorm::mean", C);
  Kokkos::View<float*, ExecSpace> inv("instnorm::inv", C);
  Impl::InstanceNormStatsFunctor<ExecSpace, Scalar> f{x, mean, inv, eps, T};
  Kokkos::parallel_for(label + "_stats",
                       Kokkos::TeamPolicy<ExecSpace>(C, Kokkos::AUTO), f);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        y(t, c) = static_cast<Scalar>((static_cast<float>(x(t, c)) - mean(c)) *
                                      inv(c));
      });
}

// AdaIN1d: instance-norm x [T,C], then style modulation from s [S]:
//   h = fc_w @ s + fc_b  ([2C]); gamma,beta = split(h)
//   y[t,c] = (1 + gamma[c]) * n[t,c] + beta[c].
// fc_w [2C,S], fc_b [2C] (plain Linear weights, FP32 in practice).
template <class ExecSpace, class Scalar>
void adain1d_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float*, ExecSpace>& style,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& fc_w,
    const Kokkos::View<float*, ExecSpace>& fc_b, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::adain1d_forward") {
  const size_t T = x.extent(0), C = x.extent(1), S = style.extent(0);
  if (fc_w.extent(0) != 2 * C || fc_w.extent(1) != S || fc_b.extent(0) != 2 * C)
    throw std::invalid_argument("adain1d_forward: fc shape mismatch");
  if (y.extent(0) != T || y.extent(1) != C)
    throw std::invalid_argument("adain1d_forward: output shape mismatch");
  if (T == 0 || C == 0) return;
  // h = fc(s) on host-lane kernel (2C rows).
  Kokkos::View<float*, ExecSpace> h("adain::h", 2 * C);
  Kokkos::parallel_for(
      label + "_fc", Kokkos::RangePolicy<ExecSpace>(0, 2 * C),
      KOKKOS_LAMBDA(size_t r) {
        float acc = fc_b(r);
        for (size_t j = 0; j < S; ++j) acc += fc_w(r, j) * style(j);
        h(r) = acc;
      });
  // Instance-norm into a temp, then modulate.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> n("adain::n", T, C);
  instance_norm1d<ExecSpace, Scalar>(x, eps, n, label + "_norm");
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const float g = h(c), b = h(C + c);
        y(t, c) = static_cast<Scalar>((1.0f + g) *
                                     static_cast<float>(n(t, c)) +
                                     b);
      });
}

// AdaLayerNorm (StyleTTS2 DurationEncoder): layer-norm over channels per
// time step (eps), then style modulation with fc(s):
//   y[t,c] = (1 + gamma[c]) * ln[t,c] + beta[c].
// Same fc layout as adain1d_forward. Scalar-agnostic.
template <class ExecSpace, class Scalar>
void adaln_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float*, ExecSpace>& style,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& fc_w,
    const Kokkos::View<float*, ExecSpace>& fc_b, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::adaln_forward") {
  const size_t T = x.extent(0), C = x.extent(1), S = style.extent(0);
  if (fc_w.extent(0) != 2 * C || fc_w.extent(1) != S || fc_b.extent(0) != 2 * C)
    throw std::invalid_argument("adaln_forward: fc shape mismatch");
  if (y.extent(0) != T || y.extent(1) != C)
    throw std::invalid_argument("adaln_forward: output shape mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::View<float*, ExecSpace> h("adaln::h", 2 * C);
  Kokkos::parallel_for(
      label + "_fc", Kokkos::RangePolicy<ExecSpace>(0, 2 * C),
      KOKKOS_LAMBDA(size_t r) {
        float acc = fc_b(r);
        for (size_t j = 0; j < S; ++j) acc += fc_w(r, j) * style(j);
        h(r) = acc;
      });
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> n("adaln::n", T, C);
  layernorm_noaffine_forward<ExecSpace, Scalar>(x, eps, n, label + "_norm");
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const float g = h(c), b = h(C + c);
        y(t, c) = static_cast<Scalar>((1.0f + g) *
                                     static_cast<float>(n(t, c)) +
                                     b);
      });
}

}  // namespace KokkosDNN

#endif
