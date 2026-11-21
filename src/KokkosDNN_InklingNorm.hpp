// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLINGNORM_HPP
#define KOKKOSDNN_INKLINGNORM_HPP

// Inkling2 loop stateless norms (thinkingmachines/Inkling-2-Mini text
// decoder): unweighted RMSNorm (no gamma), per-head-vector L2Clip,
// learned per-head-dim ScaledL2Clip, LogScale position multiplier, and the
// LogScaledL2Clip composition. All accumulation is FP32 with the result
// cast back to Scalar. Backend-portable: only parallel_for /
// parallel_reduce kernels; no backend-specific code.

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Occupancy.hpp"

namespace KokkosDNN {
namespace Impl {

// Team-parallel sum-of-squares over one row, returned as the RMS inverse
// scale. `fetch(d)` supplies element d as float. In-place safe: the reduce
// completes before the caller writes.
template <class Member, class Fetch>
KOKKOS_INLINE_FUNCTION float inkling_rms_row_inv(const Member& team,
                                                 size_t D, float eps,
                                                 Fetch fetch) {
  float ss = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        const float v = fetch(d);
        acc += v * v;
      },
      ss);
  return 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D) + eps);
}

// Team-parallel per-head L2 norm over one row. `fetch(d)` supplies element
// d as float (already scaled by the caller when needed).
template <class Member, class Fetch>
KOKKOS_INLINE_FUNCTION float inkling_l2_row_norm(const Member& team, size_t D,
                                                 Fetch fetch) {
  float ss = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team, D),
      KOKKOS_LAMBDA(size_t d, float& acc) {
        const float v = fetch(d);
        acc += v * v;
      },
      ss);
  return Kokkos::sqrtf(ss);
}

// Launcher shared by the entry points below: one team per row, team size
// min(D, 256) clamped to team_size_max (Serial-safe).
template <class ExecSpace, class Functor>
void inkling_norm_team_launch(Functor f, size_t rows, size_t D,
                              const std::string& label) {
  if (rows == 0 || D == 0) return;
  int ts = static_cast<int>(D < 256 ? D : 256);
  // Clamp to team_size_max (Serial -> 1); the strided TeamThreadRange stays
  // correct at any team size. Probe via the canonical TeamPolicy(1, 1)
  // (see Impl::cached_team_size_max).
  const int mx = Impl::cached_team_size_max<ExecSpace>(f);
  if (mx < ts) ts = mx;
  if (ts < 1) ts = 1;
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(rows), ts);
  Kokkos::parallel_for(label, policy, f);
}

// One team per (token[, head]) row: unweighted RMSNorm over D.
template <class ExecSpace, class Scalar, class XView, class YView>
struct InklingUnweightedRmsTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  XView x;
  YView y;
  size_t rows;
  size_t D;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const XView x_ = x;
    const YView y_ = y;
    const float eps_ = eps;
    const size_t D_ = D;
    const size_t row = static_cast<size_t>(team.league_rank());
    const float inv = inkling_rms_row_inv(team, D_, eps_, KOKKOS_LAMBDA(size_t d) {
      return static_cast<float>(x_(row, d));
    });
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          y_(row, d) = static_cast<Scalar>(static_cast<float>(x_(row, d)) * inv);
        });
  }
};

// One team per (token, head) row: y = x * min(1, threshold / (||x|| + 1e-6)).
template <class ExecSpace, class Scalar, class XView, class YView>
struct InklingL2ClipTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  XView x;
  YView y;
  size_t H;
  size_t D;
  float threshold;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const XView x_ = x;
    const YView y_ = y;
    const float thr_ = threshold;
    const size_t H_ = H, D_ = D;
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t t = id / H_, h = id % H_;
    const float n = inkling_l2_row_norm(team, D_, KOKKOS_LAMBDA(size_t d) {
      return static_cast<float>(x_(t, h, d));
    });
    const float s = Kokkos::fminf(1.0f, thr_ / (n + 1e-6f));
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          y_(t, h, d) = static_cast<Scalar>(static_cast<float>(x_(t, h, d)) * s);
        });
  }
};

// One team per (token, head) row: scale-then-clip,
// y = (x * scale) * min(1, threshold / (||x * scale|| + 1e-6)).
template <class ExecSpace, class Scalar, class XView, class SView, class YView>
struct InklingScaledL2ClipTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  XView x;
  SView scale;
  YView y;
  size_t H;
  size_t D;
  float threshold;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const XView x_ = x;
    const SView s_ = scale;
    const YView y_ = y;
    const float thr_ = threshold;
    const size_t H_ = H, D_ = D;
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t t = id / H_, h = id % H_;
    const float n = inkling_l2_row_norm(team, D_, KOKKOS_LAMBDA(size_t d) {
      return static_cast<float>(x_(t, h, d)) * static_cast<float>(s_(h, d));
    });
    const float c = Kokkos::fminf(1.0f, thr_ / (n + 1e-6f));
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          y_(t, h, d) = static_cast<Scalar>(static_cast<float>(x_(t, h, d)) *
                                           static_cast<float>(s_(h, d)) * c);
        });
  }
};

// One team per (token, head) row: scaled-clip followed by the log-scale
// multiplier tau = 1 + alpha * log(max((pos+1)/floor, 1)).
template <class ExecSpace, class Scalar, class XView, class SView, class PView,
          class YView>
struct InklingLogScaledL2ClipTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  XView x;
  SView scale;
  PView pos;
  YView y;
  size_t H;
  size_t D;
  float threshold;
  float alpha;
  float floor;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const XView x_ = x;
    const SView s_ = scale;
    const PView p_ = pos;
    const YView y_ = y;
    const float thr_ = threshold, alpha_ = alpha, floor_ = floor;
    const size_t H_ = H, D_ = D;
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t t = id / H_, h = id % H_;
    const float n = inkling_l2_row_norm(team, D_, KOKKOS_LAMBDA(size_t d) {
      return static_cast<float>(x_(t, h, d)) * static_cast<float>(s_(h, d));
    });
    const float c = Kokkos::fminf(1.0f, thr_ / (n + 1e-6f));
    const float ratio =
        (static_cast<float>(p_(t)) + 1.0f) / floor_;
    const float tau =
        1.0f + alpha_ * Kokkos::logf(Kokkos::fmaxf(ratio, 1.0f));
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          y_(t, h, d) = static_cast<Scalar>(static_cast<float>(x_(t, h, d)) *
                                           static_cast<float>(s_(h, d)) * c *
                                           tau);
        });
  }
};

}  // namespace Impl

// Unweighted RMSNorm (no gamma), rank-2 rows:
//   y[t,:] = x[t,:] / sqrt(mean(x[t,:]^2) + eps)
// x, y: [T, D]. Scalar-agnostic; norm accumulated in FP32. In-place safe.
template <class ExecSpace, class Scalar>
void inkling_unweighted_rmsnorm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_unweighted_rmsnorm") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D)
    throw std::invalid_argument("inkling_unweighted_rmsnorm: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::InklingUnweightedRmsTeam<
          ExecSpace, Scalar,
          Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>,
          Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>>{x, y, T, D,
                                                                 eps},
      T, D, label);
}

// Unweighted RMSNorm, rank-3 rows: y[t,h,:] = x[t,h,:] / rms(x[t,h,:]).
// x, y: [T, H, D]. Scalar-agnostic; FP32 accumulation. In-place safe.
template <class ExecSpace, class Scalar>
void inkling_unweighted_rmsnorm_3d(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_unweighted_rmsnorm_3d") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (y.extent(0) != T || y.extent(1) != H || y.extent(2) != D)
    throw std::invalid_argument(
        "inkling_unweighted_rmsnorm_3d: dimension mismatch");
  if (T == 0 || H == 0 || D == 0) return;
  // Reinterpret as flat [T*H, D] rows via unmanaged views (LayoutRight
  // rank-3 is contiguous in D, so the reinterpretation is exact).
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  V2 xf(x.data(), T * H, D), yf(y.data(), T * H, D);
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::InklingUnweightedRmsTeam<ExecSpace, Scalar, V2, V2>{xf, yf, T * H,
                                                               D, eps},
      T * H, D, label);
}

// L2Clip per head vector: y[t,h,:] = x[t,h,:] * min(1, thr/(||x||+1e-6)).
// x, y: [T, H, D]. Matches Inkling2LoopL2Clip. In-place safe.
template <class ExecSpace, class Scalar>
void inkling_l2_clip(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    float threshold,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_l2_clip") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (y.extent(0) != T || y.extent(1) != H || y.extent(2) != D)
    throw std::invalid_argument("inkling_l2_clip: dimension mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::InklingL2ClipTeam<
          ExecSpace, Scalar,
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>,
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>>{x, y, H, D,
                                                                  threshold},
      T * H, D, label);
}

// ScaledL2Clip: L2Clip applied after a learned per-head-dim scale.
// x, y: [T, H, D]; scale: [H, D] (FP32 weights). Matches
// Inkling2LoopScaledL2Clip. In-place safe for x/y (scale is read-only).
template <class ExecSpace, class Scalar>
void inkling_scaled_l2_clip(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& scale,
    float threshold,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_scaled_l2_clip") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (scale.extent(0) != H || scale.extent(1) != D || y.extent(0) != T ||
      y.extent(1) != H || y.extent(2) != D)
    throw std::invalid_argument("inkling_scaled_l2_clip: dimension mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::InklingScaledL2ClipTeam<
          ExecSpace, Scalar,
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>,
          Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>,
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>>{x, scale, y,
                                                                  H, D,
                                                                  threshold},
      T * H, D, label);
}

// LogScale multiplier: y[t,h,:] = x[t,h,:] * tau[t] with
// tau = 1 + alpha * log(max((pos[t]+1)/n_floor, 1)).
// x, y: [T, H, D]; pos: [T] integer positions. Matches Inkling2LoopLogScale.
// In-place safe.
template <class ExecSpace, class Scalar, class PView>
void inkling_log_scale(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const PView& pos, float alpha, float n_floor,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_log_scale") {
  static_assert(PView::rank == 1, "inkling_log_scale: pos must be rank 1");
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (pos.extent(0) != T || y.extent(0) != T || y.extent(1) != H ||
      y.extent(2) != D)
    throw std::invalid_argument("inkling_log_scale: dimension mismatch");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {T, H, D}),
      KOKKOS_LAMBDA(size_t t, size_t h, size_t d) {
        const float ratio = (static_cast<float>(pos(t)) + 1.0f) / n_floor;
        const float tau =
            1.0f + alpha * Kokkos::logf(Kokkos::fmaxf(ratio, 1.0f));
        y(t, h, d) =
            static_cast<Scalar>(static_cast<float>(x(t, h, d)) * tau);
      });
}

// LogScaledL2Clip: ScaledL2Clip followed by LogScale (single launch).
// x, y: [T, H, D]; scale: [H, D] FP32; pos: [T] integer positions.
// Matches Inkling2LoopLogScaledL2Clip. In-place safe for x/y.
template <class ExecSpace, class Scalar, class PView>
void inkling_log_scaled_l2_clip(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& scale,
    float threshold, const PView& pos, float alpha, float n_floor,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_log_scaled_l2_clip") {
  static_assert(PView::rank == 1,
                "inkling_log_scaled_l2_clip: pos must be rank 1");
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (scale.extent(0) != H || scale.extent(1) != D || pos.extent(0) != T ||
      y.extent(0) != T || y.extent(1) != H || y.extent(2) != D)
    throw std::invalid_argument(
        "inkling_log_scaled_l2_clip: dimension mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::InklingLogScaledL2ClipTeam<
          ExecSpace, Scalar,
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>,
          Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>, PView,
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>>{
          x, scale, pos, y, H, D, threshold, alpha, n_floor},
      T * H, D, label);
}

}  // namespace KokkosDNN

#endif
