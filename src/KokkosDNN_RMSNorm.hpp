// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_RMSNORM_HPP
#define KOKKOSDNN_RMSNORM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Occupancy.hpp"

namespace KokkosDNN {
namespace Impl {

// Team-parallel RMSNorm row worker: one team per row. The sum-of-squares is
// a team-parallel reduce (FP32 reassociation vs a serial loop is ~1 ulp,
// inside test tolerances), then the scale fans out over the team. The old
// RangePolicy-over-rows left decode (T=1) to a single thread running the
// whole D loop serially; this exposes T*D-way parallelism on every backend.
// Team size clamps to team_size_max (Serial -> 1; strided TeamThreadRange
// stays correct). In-place safe: the reduce completes before any write.
template <class ExecSpace, class Scalar>
struct RmsNormTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View2 x;
  View1 w;
  View2 y;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    // Locals: the lambdas below capture by value, so bind members to
    // locals to avoid an implicit `this` capture (deprecated in C++20).
    const View2 x_ = x, y_ = y;
    const View1 w_ = w;
    const float eps_ = eps;
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = x_.extent(1);
    float ss = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          const float v = static_cast<float>(x_(t, d));
          acc += v * v;
        },
        ss);
    const float inv = 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D) + eps_);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          y_(t, d) = static_cast<Scalar>(static_cast<float>(x_(t, d)) * inv *
                                        static_cast<float>(w_(d)));
        });
  }
};

// Team-parallel scaleless row worker: same reduce-then-scale structure as
// RmsNormTeam without the weight multiply. In-place safe (the reduction
// completes before any write). Used by rmsnorm_noscale, whose old RangePolicy
// -over-rows form left a decode (T=1) row to one thread running the whole D
// loop serially.
template <class ExecSpace, class Scalar>
struct RmsNormNoScaleTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  View2 x;
  View2 y;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View2 x_ = x, y_ = y;
    const float eps_ = eps;
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = x_.extent(1);
    float ss = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          const float v = static_cast<float>(x_(t, d));
          acc += v * v;
        },
        ss);
    const float inv = 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D) + eps_);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          y_(t, d) = static_cast<Scalar>(static_cast<float>(x_(t, d)) * inv);
        });
  }
};

// Launcher shared by rmsnorm and the per-head qk variants below: one team
// per row, team size min(D, 256) clamped to team_size_max.
template <class ExecSpace, class Functor>
void rmsnorm_team_launch(Functor f, size_t rows, size_t D,
                         const std::string& label) {
  if (rows == 0 || D == 0) return;
  int ts = static_cast<int>(D < 256 ? D : 256);
  // Occupancy probe cached per (ExecSpace instance, functor type, scratch):
  // decode calls this kernel once per layer norm and twice per attention
  // layer, i.e. the same probe was re-run ~100x per token.
  const int mx = cached_team_size_max<ExecSpace>(f);
  if (mx < ts) ts = mx;
  if (ts < 1) ts = 1;
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(rows), ts);
  Kokkos::parallel_for(label, policy, f);
}

// Per-head QK worker: one team per (token, head) row of D. Same reduce-then
// -scale structure as RmsNormTeam; the flat [T, H*D] input is indexed at
// h*D+d. Fixes the same single-threaded-decode issue for head_dim-wide
// rows (decode ran H threads serial over D).
template <class ExecSpace, class Scalar>
struct QkRmsNormTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  View2 flat;
  View1 w;
  View3 out;
  size_t H;
  size_t D;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View2 flat_ = flat;
    const View1 w_    = w;
    const View3 out_  = out;
    const float eps_  = eps;
    const size_t H_ = H, D_ = D;
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t t  = id / H_;
    const size_t h  = id % H_;
    const size_t base = h * D_;
    float ss = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, D_),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          const float v = static_cast<float>(flat_(t, base + d));
          acc += v * v;
        },
        ss);
    const float inv = 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D_) + eps_);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          out_(t, h, d) = static_cast<Scalar>(
              static_cast<float>(flat_(t, base + d)) * inv *
              static_cast<float>(w_(d)));
        });
  }
};

// Rank-3 variant: q/out are [T, H, D] (in-place safe: reduce completes
// before any write, same as above).
template <class ExecSpace, class Scalar>
struct QkRmsNorm3dTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  View3 q;
  View1 w;
  View3 out;
  size_t H;
  size_t D;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View3 q_ = q, out_ = out;
    const View1 w_ = w;
    const float eps_ = eps;
    const size_t H_ = H, D_ = D;
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t t  = id / H_;
    const size_t h  = id % H_;
    float ss = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, D_),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          const float v = static_cast<float>(q_(t, h, d));
          acc += v * v;
        },
        ss);
    const float inv = 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D_) + eps_);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          out_(t, h, d) = static_cast<Scalar>(
              static_cast<float>(q_(t, h, d)) * inv *
              static_cast<float>(w_(d)));
        });
  }
};

}  // namespace Impl

// RMSNorm (Llama variant, no mean-centering, no bias):
//   y[i,:] = x[i,:] / sqrt(mean(x[i,:]^2) + eps) * w[:]
// x, y: [tokens, dim], w: [dim]. Scalar-agnostic; norm accumulated in FP32.
// Team-parallel over rows and dim (Impl::RmsNormTeam); in-place safe.
template <class ExecSpace, class Scalar>
void rmsnorm(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
             const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
             const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
             const std::string& label = "KokkosDNN::rmsnorm") {
  const size_t T = x.extent(0), D = x.extent(1);
  Impl::rmsnorm_team_launch<ExecSpace>(
      Impl::RmsNormTeam<ExecSpace, Scalar>{x, w, y, eps}, T, D, label);
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
// Team-parallel over rows and dim (Impl::RmsNormNoScaleTeam; the FP32
// sum-of-squares is reassociated by the reduction tree, a few ulp).
// In-place safe (x and y may alias).
template <class ExecSpace, class Scalar>
void rmsnorm_noscale(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::rmsnorm_noscale") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D)
    throw std::invalid_argument("rmsnorm_noscale: dimension mismatch");
  Impl::rmsnorm_team_launch<ExecSpace>(
      Impl::RmsNormNoScaleTeam<ExecSpace, Scalar>{x, y, eps}, T, D, label);
}

namespace Impl {
// Team-parallel row-norm worker: one team per row. The sum-of-squares is a
// team-parallel reduce (all threads participate) followed by a scale pass
// that fans out over the team; the reduction tree reassociates the FP32
// summation relative to a serial loop (a few ulp, far inside the 1e-5 test
// tolerance). A RangePolicy over rows alone leaves only T threads (256 at
// encoder geometry), and an earlier revision ran the sum serially on one
// thread per team, which dominated decode (T=1 left a single thread running
// the whole D loop while the other 255 idled). Team size clamps to
// team_size_max (Serial -> 1) via the caller's probe; the strided
// TeamThreadRange stays correct at any team size.
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
    // Local handles: the KOKKOS_LAMBDA below captures by value, so bind
    // members to locals to avoid an implicit `this` capture (deprecated
    // in C++20; an error under -Werror on the nvcc path).
    const View2 x_ = x, y_ = y;
    const View1 w_ = w;
    const float eps_ = eps;
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = x_.extent(1);
    float ss = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          const float v = static_cast<float>(x_(t, d));
          acc += v * v;
        },
        ss);
    const float inv = 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D) + eps_);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          y_(t, d) = static_cast<Scalar>(static_cast<float>(x_(t, d)) * inv *
                                        (1.0f + static_cast<float>(w_(d))));
        });
  }
};

// Fused (1+w)-RMSNorm + residual add: h[t,:] += norm(x[t,:]) * (1 + w).
// Same per-row math as RmsNormOnePlusTeam (team-parallel FP32 sum-of-squares,
// i.e. reduction-tree reassociation of a few ulp) with the scale pass adding
// directly into the residual stream instead of writing an intermediate.
// Reassociation vs the two-pass form (rmsnorm_one_plus into tmp, then
// h += tmp): in addition to the sum-of-squares tree, the fused add may
// FMA-contract with the scale multiply (single rounding instead of two), so
// float results can differ by a few ulp. Narrow-Scalar (bhalf) builds are
// within the same narrow-Scalar rounding; they are no longer guaranteed
// bit-identical because the reduction reassociates the norm. Eliminates one
// kernel launch and one full [T, D] tmp round-trip per call. x and h must
// not alias (the model passes proj/down vs h).
template <class ExecSpace, class Scalar>
struct RmsNormOnePlusResidTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View2 x;
  View1 w;
  View2 h;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    // Local handles: the lambdas below capture by value, so bind members
    // to locals to avoid an implicit `this` capture (deprecated in C++20).
    const View2 x_ = x, h_ = h;
    const View1 w_ = w;
    const float eps_ = eps;
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = x_.extent(1);
    float ss = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, D),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          const float v = static_cast<float>(x_(t, d));
          acc += v * v;
        },
        ss);
    const float inv = 1.0f / Kokkos::sqrtf(ss / static_cast<float>(D) + eps_);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          const Scalar n = static_cast<Scalar>(
              static_cast<float>(x_(t, d)) * inv *
              (1.0f + static_cast<float>(w_(d))));
          h_(t, d) += n;
        });
  }
};

// Fused Q/K per-head RMSNorm + V reshape: one team per (token, head-slot)
// with slots [0, T*NH) normalizing q, [T*NH, T*(NH+NKV)) normalizing k,
// and the remainder copying v from flat [T, Hkv*D] to [T, Hkv, D].
// The q/k branches repeat QkRmsNormTeam's team-parallel reduce verbatim
// (same summation, same team-size selection at the call site), so results
// are bit-identical to separate qk_rmsnorm calls; the v copy is exact.
// Replaces three launches (qk q, qk k, reshape_v) with one.
template <class ExecSpace, class Scalar>
struct QkVRmsNormFusedTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  View2 qflat;
  View1 wq;
  View3 qout;
  View2 kflat;
  View1 wk;
  View3 kout;
  View2 vflat;
  View3 vout;
  size_t NH;
  size_t NKV;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View2 qflat_ = qflat, kflat_ = kflat, vflat_ = vflat;
    const View1 wq_ = wq, wk_ = wk;
    const View3 qout_ = qout, kout_ = kout, vout_ = vout;
    const float eps_ = eps;
    const size_t NH_ = NH, NKV_ = NKV;
    const size_t HD_ = wq_.extent(0);
    const size_t T_ = qflat_.extent(0);
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t qn = T_ * NH_;
    const size_t kn = T_ * NKV_;
    if (id < qn) {
      const size_t t = id / NH_;
      const size_t h = id % NH_;
      const size_t base = h * HD_;
      float ss = 0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, HD_),
          KOKKOS_LAMBDA(size_t d, float& acc) {
            const float v = static_cast<float>(qflat_(t, base + d));
            acc += v * v;
          },
          ss);
      const float inv =
          1.0f / Kokkos::sqrtf(ss / static_cast<float>(HD_) + eps_);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, HD_), KOKKOS_LAMBDA(size_t d) {
            qout_(t, h, d) = static_cast<Scalar>(
                static_cast<float>(qflat_(t, base + d)) * inv *
                static_cast<float>(wq_(d)));
          });
    } else if (id < qn + kn) {
      const size_t j = id - qn;
      const size_t t = j / NKV_;
      const size_t h = j % NKV_;
      const size_t base = h * HD_;
      float ss = 0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, HD_),
          KOKKOS_LAMBDA(size_t d, float& acc) {
            const float v = static_cast<float>(kflat_(t, base + d));
            acc += v * v;
          },
          ss);
      const float inv =
          1.0f / Kokkos::sqrtf(ss / static_cast<float>(HD_) + eps_);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, HD_), KOKKOS_LAMBDA(size_t d) {
            kout_(t, h, d) = static_cast<Scalar>(
                static_cast<float>(kflat_(t, base + d)) * inv *
                static_cast<float>(wk_(d)));
          });
    } else {
      const size_t j = id - qn - kn;
      const size_t t = j / NKV_;
      const size_t h = j % NKV_;
      const size_t base = h * HD_;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, HD_), KOKKOS_LAMBDA(size_t d) {
            vout_(t, h, d) = vflat_(t, base + d);
          });
    }
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
  // Team per row: the sum-of-squares is a team-parallel reduce, so decode
  // (T=1) spreads the whole row over the team instead of one serial thread.
  // The probe clamps the team size for backends that cannot field 256
  // threads (Serial -> 1; strided TeamThreadRange stays correct).
  Impl::RmsNormOnePlusTeam<ExecSpace, Scalar> f{
      x, w, y, eps,
  };
  const int mx = Impl::cached_team_size_max<ExecSpace>(f);
  int ts = static_cast<int>(D < 256 ? D : 256);
  if (mx < ts) ts = mx;
  if (ts < 1) ts = 1;
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(T), ts);
  Kokkos::parallel_for(label, policy, f);
}

// Fused Gemma2 (1+w)-RMSNorm + residual add:
//   h[i,:] += x[i,:] / sqrt(mean(x[i,:]^2) + eps) * (1 + w[:])
// Reassociation vs rmsnorm_one_plus + separate add (see
// Impl::RmsNormOnePlusResidTeam: a few ulp in float from the team-parallel
// sum-of-squares tree and FMA contraction of the fused add; bhalf stays
// within narrow-Scalar rounding but is no longer bit-identical): use a tight
// tolerance, not bitwise equality, when comparing against the two-pass form.
// Team-parallel, one team per row; Serial-safe via the team_size_max probe
// clamp. x and h must not alias.
template <class ExecSpace, class Scalar>
void rmsnorm_one_plus_resid(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& h,
    const std::string& label = "KokkosDNN::rmsnorm_one_plus_resid") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (h.extent(0) != T || h.extent(1) != D || w.extent(0) != D)
    throw std::invalid_argument("rmsnorm_one_plus_resid: dimension mismatch");
  if (T == 0 || D == 0) return;
  Impl::RmsNormOnePlusResidTeam<ExecSpace, Scalar> f{
      x, w, h, eps,
  };
  const int mx = Impl::cached_team_size_max<ExecSpace>(f);
  int ts = static_cast<int>(D < 256 ? D : 256);
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
// Same per-row math as rmsnorm above, applied independently to each (t, h).
// Scalar-agnostic; backend-portable team-parallel kernel (one team per
// (token, head) row, D-way parallel within the team).
template <class ExecSpace, class Scalar>
void qk_rmsnorm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& flat,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::qk_rmsnorm") {
  const size_t T = out.extent(0), H = out.extent(1), D = out.extent(2);
  if (flat.extent(0) != T || flat.extent(1) != H * D || w.extent(0) != D)
    throw std::invalid_argument("qk_rmsnorm: dimension mismatch");
  Impl::rmsnorm_team_launch<ExecSpace>(
      Impl::QkRmsNormTeam<ExecSpace, Scalar>{flat, w, out, H, D, eps}, T * H,
      D, label);
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
  Impl::rmsnorm_team_launch<ExecSpace>(
      Impl::QkRmsNorm3dTeam<ExecSpace, Scalar>{q, w, out, H, D, eps}, T * H,
      D, label);
}

// Fused Q/K per-head RMSNorm + V flat-to-3D reshape (see
// Impl::QkVRmsNormFusedTeam): bit-identical to qk_rmsnorm(qflat, wq, eps,
// qout) + qk_rmsnorm(kflat, wk, eps, kout) + a vflat -> vout reshape copy,
// in a single launch. q and k share one eps (the model passes the same
// rms_norm_eps to both). Scalar-agnostic; backend-portable team-parallel.
template <class ExecSpace, class Scalar>
void qk_rmsnorm_qkv_fused(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& qflat,
    const Kokkos::View<Scalar*, ExecSpace>& wq,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& qout,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kflat,
    const Kokkos::View<Scalar*, ExecSpace>& wk,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& kout,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& vflat,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& vout,
    float eps,
    const std::string& label = "KokkosDNN::qk_rmsnorm_qkv_fused") {
  const size_t T = qout.extent(0), NH = qout.extent(1), HD = qout.extent(2);
  const size_t NKV = kout.extent(1);
  if (qflat.extent(0) != T || qflat.extent(1) != NH * HD ||
      wq.extent(0) != HD || kflat.extent(0) != T ||
      kflat.extent(1) != NKV * HD || wk.extent(0) != HD ||
      kout.extent(0) != T || kout.extent(2) != HD || vflat.extent(0) != T ||
      vflat.extent(1) != NKV * HD || vout.extent(0) != T ||
      vout.extent(1) != NKV || vout.extent(2) != HD)
    throw std::invalid_argument("qk_rmsnorm_qkv_fused: dimension mismatch");
  Impl::rmsnorm_team_launch<ExecSpace>(
      Impl::QkVRmsNormFusedTeam<ExecSpace, Scalar>{
          qflat, wq, qout, kflat, wk, kout, vflat, vout, NH, NKV, eps},
      T * (NH + 2 * NKV), HD, label);
}

}  // namespace KokkosDNN

#endif
