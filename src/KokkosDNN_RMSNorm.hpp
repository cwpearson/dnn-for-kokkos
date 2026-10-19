#ifndef KOKKOSDNN_RMSNORM_HPP
#define KOKKOSDNN_RMSNORM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// RMSNorm (Llama variant, no mean-centering, no bias):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * w[:]
// x, y: [tokens, dim], w: [dim]. Scalar-agnostic; norm accumulated in FP32.
template <class ExecSpace, class Scalar>
void rmsnorm(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
             const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
             const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
             const std::string& label = "KokkosDNN::rmsnorm") {
  const size_t T = x.extent(0), D = x.extent(1);
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

// Batched variant: x, y: [batch, tokens, dim], w: [dim].
// Same per-row math as the rank-2 overload, applied independently to each
// (batch, token) row. Scalar-agnostic; norm accumulated in FP32.
// In-place safe (x and y may alias).
template <class ExecSpace, class Scalar>
void rmsnorm(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("rmsnorm: dimension mismatch");
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

// Single-row variant for decode.
template <class ExecSpace, class Scalar>
void rmsnorm_row(const Kokkos::View<Scalar*, ExecSpace>& x,
                 const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
                 const Kokkos::View<Scalar*, ExecSpace>& y,
                 const std::string& label = "KokkosDNN::rmsnorm_row") {
  const size_t D = x.extent(0);
  // Two-pass: norm then scale. Single thread-range is fine for D=4096.
  float ss = 0;
  Kokkos::parallel_reduce(
      label + "_reduce", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        float v = static_cast<float>(x(d));
        acc += v * v;
      },
      ss);
  float inv = 1.0f / std::sqrt(ss / (float)D + eps);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        y(d) = static_cast<Scalar>(static_cast<float>(x(d)) * inv *
                                   static_cast<float>(w(d)));
      });
}

// Scaleless RMSNorm (e.g. Gemma4 `v_norm`, `with_scale=False`):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps)
// No weight multiply. Scalar-agnostic; norm accumulated in FP32.
// In-place safe (x and y may alias).
template <class ExecSpace, class Scalar>
void rmsnorm_noscale(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_noscale") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D)
    throw std::invalid_argument("rmsnorm_noscale: dimension mismatch");
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
          y(t, d) =
              static_cast<Scalar>(static_cast<float>(x(t, d)) * inv);
      });
}

namespace Impl {
// Team-parallel row-norm worker: one team per row; the sum-of-squares is
// accumulated sequentially in the same order as the serial form (identical
// summation) and broadcast via Kokkos::single, while the scale pass fans out
// over the team (elementwise, order-free). A RangePolicy over rows alone
// leaves only T threads (256 at encoder geometry); teams expose T*D-way
// parallelism. (The rsqrt uses sqrtf where the serial form used sqrt: at
// most 1 ulp, far inside the 1e-5 test tolerance.) Team size clamps to
// team_size_max (Serial -> 1) via the caller's probe.
template <class ExecSpace, class Scalar>
struct RmsNormOnePlusTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View2 x;
  View1 w;
  View2 y;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = x.extent(1);
    float ss = 0;
    Kokkos::single(
        Kokkos::PerTeam(team),
        [&](float& v) {
          float s = 0;
          for (size_t d = 0; d < D; ++d) {
            const float xvd = static_cast<float>(x(t, d));
            s += xvd * xvd;
          }
          v = s;
        },
        ss);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(D) + eps);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          y(t, d) = static_cast<Scalar>(static_cast<float>(x(t, d)) * inv *
                                        (1.0f + static_cast<float>(w(d))));
        });
  }
};
}  // namespace Impl

// Gemma2 RMSNorm variant (zero-initialized weights multiply as (1 + w)):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * (1 + w[:])
// Matches HF Gemma2RMSNorm (norm in FP32, then scale, then cast).
// Scalar-agnostic; norm accumulated in FP32.
// In-place safe (x and y may alias).
// No TPL dispatch: pure elementwise dedup shared with the Sana Gemma2 text
// encoder (same justification as rmsnorm_noscale above).
template <class ExecSpace, class Scalar>
void rmsnorm_one_plus(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_one_plus") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D || w.extent(0) != D)
    throw std::invalid_argument("rmsnorm_one_plus: dimension mismatch");
  if (T == 0 || D == 0) return;
  // Team per row (same summation order; the serial form exposed only T
  // threads). Probe clamps the team size for backends that cannot field
  // 256 threads (Serial -> 1; strided TeamThreadRange stays correct).
  Impl::RmsNormOnePlusTeam<ExecSpace, Scalar> f{
      x, w, y, eps,
  };
  Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
  int ts = static_cast<int>(D < 256 ? D : 256);
  const int mx = probe.team_size_max(f, Kokkos::ParallelForTag{});
  if (mx < ts) ts = mx;
  if (ts < 1) ts = 1;
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(T), ts);
  Kokkos::parallel_for(label, policy, f);
}

// Single-row scaleless variant for decode.
template <class ExecSpace, class Scalar>
void rmsnorm_row_noscale(
    const Kokkos::View<Scalar*, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_row_noscale") {
  const size_t D = x.extent(0);
  if (y.extent(0) != D)
    throw std::invalid_argument("rmsnorm_row_noscale: dimension mismatch");
  if (D == 0) return;
  float ss = 0;
  Kokkos::parallel_reduce(
      label + "_reduce", Kokkos::RangePolicy<ExecSpace>(0, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        float v = static_cast<float>(x(d));
        acc += v * v;
      },
      ss);
  float inv = 1.0f / std::sqrt(ss / (float)D + eps);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        y(d) = static_cast<Scalar>(static_cast<float>(x(d)) * inv);
      });
}

// Fused reshape + per-head QK RMSNorm (library counterpart of the
// Gemma4 example-local qk_reshape_norm: the plain rmsnorm op works on
// [tokens, dim] rows, while q/k norms run per attention head):
//   out[t,h,d] = flat[t,h*D+d] / rms(flat[t,h*D:(h+1)*D]) * w[d]
// Same FP32 formula as rmsnorm above, applied independently per (t, h).
// Scalar-agnostic; backend-portable MDRange kernel.
template <class ExecSpace, class Scalar>
void qk_rmsnorm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& flat,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::qk_rmsnorm") {
  const size_t T = out.extent(0), H = out.extent(1), D = out.extent(2);
  if (flat.extent(0) != T || flat.extent(1) != H * D || w.extent(0) != D)
    throw std::invalid_argument("qk_rmsnorm: dimension mismatch");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(flat(t, h * D + d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          out(t, h, d) = static_cast<Scalar>(
              static_cast<float>(flat(t, h * D + d)) * inv *
              static_cast<float>(w(d)));
      });
}

// Rank-3 overload for already-shaped q/k: q/out [T, H, D], w [D].
// Same per-(t, h) math as qk_rmsnorm. In-place safe (q and out may alias).
template <class ExecSpace, class Scalar>
void qk_rmsnorm3d(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& q,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::qk_rmsnorm3d") {
  const size_t T = q.extent(0), H = q.extent(1), D = q.extent(2);
  if (out.extent(0) != T || out.extent(1) != H || out.extent(2) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("qk_rmsnorm3d: dimension mismatch");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(q(t, h, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          out(t, h, d) = static_cast<Scalar>(
              static_cast<float>(q(t, h, d)) * inv * static_cast<float>(w(d)));
      });
}

}  // namespace KokkosDNN

#endif
