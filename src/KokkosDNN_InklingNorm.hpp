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

// Fused 3-stream weighted input norms (layer_forward section 2 when the
// qh/kh/vh streams are already materialized, e.g. the has_msgs gate
// outputs): one team per (stream, token) row, 3*C teams total. Each team
// repeats its stream's rmsnorm row verbatim (same team-parallel reduce,
// same team-size selection at the call site), so the result is
// bit-identical to three separate rmsnorm calls in one launch instead of
// three.
template <class ExecSpace, class Scalar>
struct FusedInputNorms3Team {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View2 qh;
  View2 kh;
  View2 vh;
  View1 wq;
  View1 wk;
  View1 wv;
  View2 hn;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View2 qh_ = qh, kh_ = kh, vh_ = vh, hn_ = hn;
    const View1 wq_ = wq, wk_ = wk, wv_ = wv;
    const float eps_ = eps;
    const size_t C = qh_.extent(0);
    const size_t D = qh_.extent(1);
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t t = id % C;
    const size_t s = id / C;
    const View2 xs = (s == 0) ? qh_ : ((s == 1) ? kh_ : vh_);
    const View1 ws = (s == 0) ? wq_ : ((s == 1) ? wk_ : wv_);
    if (s == 2) {
      // vh is normed in place (same as the separate rmsnorm call).
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  xs(t, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            vh_(t, d) = static_cast<Scalar>(static_cast<float>(vh_(t, d)) *
                                           inv *
                                           static_cast<float>(ws(d)));
          });
    } else if (s == 1) {
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  xs(t, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            kh_(t, d) = static_cast<Scalar>(static_cast<float>(kh_(t, d)) *
                                           inv *
                                           static_cast<float>(ws(d)));
          });
    } else {
      // qh is consumed into hn (out-of-place, as in the separate calls).
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  qh_(t, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            hn_(t, d) = static_cast<Scalar>(static_cast<float>(qh_(t, d)) *
                                           inv *
                                           static_cast<float>(wq_(d)));
          });
    }
  }
};

// Fused sliding copy+norm (layer_forward sections 1+2, no-msg paths):
// the single retrieval row retr(t,0,:) (or the raw stream h for the
// layer-0 skip path, passed as a rank-2 source) is normed once and
// broadcast to hn/kh/vh. The three separate norms reduce the same row
// with the same team size, so one shared reduction is bit-identical to
// all three while saving two reductions and the two redundant
// deep_copies. SourceT is view-generic (rank-2 packed or rank-3 retr
// with a fixed query index); only operator()(t, d) or (t, q, d) extents
// are used.
template <class ExecSpace, class Scalar, class SrcView>
struct SlidingBroadcastNormTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  SrcView src;
  View1 w;
  View2 hn;
  View2 kh;
  View2 vh;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const SrcView src_ = src;
    const View1 w_ = w;
    const View2 hn_ = hn, kh_ = kh, vh_ = vh;
    const float eps_ = eps;
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = hn_.extent(1);
    if constexpr (SrcView::rank == 3) {
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  src_(t, 0, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            const float v = static_cast<float>(src_(t, 0, d));
            const Scalar n =
                static_cast<Scalar>(v * inv * static_cast<float>(w_(d)));
            hn_(t, d) = n;
            kh_(t, d) = n;
            vh_(t, d) = n;
          });
    } else {
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  src_(t, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            const float v = static_cast<float>(src_(t, d));
            const Scalar n =
                static_cast<Scalar>(v * inv * static_cast<float>(w_(d)));
            hn_(t, d) = n;
            kh_(t, d) = n;
            vh_(t, d) = n;
          });
    }
  }
};

// Fused full-kind copy+norm (sections 1+2, no-msg full layers): retr is
// [C,3,D]; one team per (stream, token) row norms retr(t,s,:) with the
// stream's weight into hn/kh/vh. Per-row math repeats the separate
// rmsnorm rows verbatim, so the result is bit-identical to the
// deep_copy x3 + rmsnorm x3 sequence in one launch instead of six.
template <class ExecSpace, class Scalar>
struct FullRetrNormsTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View3 retr;
  View1 wq;
  View1 wk;
  View1 wv;
  View2 hn;
  View2 kh;
  View2 vh;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View3 retr_ = retr;
    const View1 wq_ = wq, wk_ = wk, wv_ = wv;
    const View2 hn_ = hn, kh_ = kh, vh_ = vh;
    const float eps_ = eps;
    const size_t C = hn_.extent(0);
    const size_t D = hn_.extent(1);
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t t = id % C;
    const size_t s = id / C;
    if (s == 0) {
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  retr_(t, 0, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            hn_(t, d) = static_cast<Scalar>(
                static_cast<float>(retr_(t, 0, d)) * inv *
                static_cast<float>(wq_(d)));
          });
    } else if (s == 1) {
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  retr_(t, 1, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            kh_(t, d) = static_cast<Scalar>(
                static_cast<float>(retr_(t, 1, d)) * inv *
                static_cast<float>(wk_(d)));
          });
    } else {
      const float inv = inkling_rms_row_inv(team, D, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  retr_(t, 2, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
            vh_(t, d) = static_cast<Scalar>(
                static_cast<float>(retr_(t, 2, d)) * inv *
                static_cast<float>(wv_(d)));
          });
    }
  }
};

// Fused full-kind alias copy+norm (layer-0 full path, sections 1+2): the
// raw stream h feeds all three norms with per-stream weights. The three
// separate norms reduce the same row, so one shared reduction is
// bit-identical to all three inv values; only the weight multiply
// differs per stream.
template <class ExecSpace, class Scalar>
struct FullAliasNormsTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View2 h;
  View1 wq;
  View1 wk;
  View1 wv;
  View2 hn;
  View2 kh;
  View2 vh;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View2 h_ = h, hn_ = hn, kh_ = kh, vh_ = vh;
    const View1 wq_ = wq, wk_ = wk, wv_ = wv;
    const float eps_ = eps;
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = h_.extent(1);
    const float inv =
        inkling_rms_row_inv(team, D, eps_, KOKKOS_LAMBDA(size_t d) {
          return static_cast<float>(h_(t, d));
        });
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          const float v = static_cast<float>(h_(t, d));
          hn_(t, d) =
              static_cast<Scalar>(v * inv * static_cast<float>(wq_(d)));
          kh_(t, d) =
              static_cast<Scalar>(v * inv * static_cast<float>(wk_(d)));
          vh_(t, d) =
              static_cast<Scalar>(v * inv * static_cast<float>(wv_(d)));
        });
  }
};

// Single-stream norm straight from a Q=1 retrieval (section 12 no-msg
// path and the final retrieval): out(t,:) = retr(t,0,:)/rms * w. Reads
// the strided middle-dim slice directly, so the caller skips the
// deep_copy into a packed temp. Bit-identical to copy-then-rmsnorm (the
// copy is exact).
template <class ExecSpace, class Scalar>
struct Retr1RmsTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using View2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View3 retr;
  View1 w;
  View2 out;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View3 retr_ = retr;
    const View1 w_ = w;
    const View2 out_ = out;
    const float eps_ = eps;
    const size_t t = static_cast<size_t>(team.league_rank());
    const size_t D = out_.extent(1);
    const float inv =
        inkling_rms_row_inv(team, D, eps_, KOKKOS_LAMBDA(size_t d) {
          return static_cast<float>(retr_(t, 0, d));
        });
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          out_(t, d) = static_cast<Scalar>(
              static_cast<float>(retr_(t, 0, d)) * inv *
              static_cast<float>(w_(d)));
        });
  }
};

// Last-row norm straight from a Q=1 retrieval (final head): y[d] =
// retr(C-1,0,d)/rms * w[d]. One team; replaces deep_copy(last, rq_last)
// plus rmsnorm_row (a RangePolicy reduce + scale pair) with a single
// team-parallel launch. The sum-of-squares reduction tree reassociates
// vs the RangePolicy reduce (a few ulp), so compare with a tight
// tolerance, not bitwise equality.
template <class ExecSpace, class Scalar>
struct Retr1LastRowTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  View3 retr;
  size_t row;
  View1 w;
  View1 y;
  float eps;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View3 retr_ = retr;
    const View1 w_ = w, y_ = y;
    const float eps_ = eps;
    const size_t rr = row;
    const size_t D = y_.extent(0);
    const float inv =
        inkling_rms_row_inv(team, D, eps_, KOKKOS_LAMBDA(size_t d) {
          return static_cast<float>(retr_(rr, 0, d));
        });
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D), KOKKOS_LAMBDA(size_t d) {
          y_(d) = static_cast<Scalar>(static_cast<float>(retr_(rr, 0, d)) *
                                     inv * static_cast<float>(w_(d)));
        });
  }
};

// Fused q/k/v head clips (section 6): one team per (token, head-slot)
// with slots [0, C*NH) running the q scaled (or log-scaled) clip,
// [C*NH, C*(NH+NKV)) the plain k L2 clip, and the remainder the
// unweighted v RMSNorm. Each branch repeats its separate kernel's row
// math verbatim (same reductions, same team-size selection), so the
// result is bit-identical to the three separate launches. In-place safe
// for q/k/v (each reduce completes before its writes).
template <class ExecSpace, class Scalar, class SView, class PView>
struct FusedQkvClipsTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  View3 q;
  View3 k;
  View3 v;
  SView qscale;
  float qk_clip;
  PView pos;
  float alpha;
  float floor;
  float eps;
  size_t NH;
  size_t NKV;
  size_t HD;
  size_t C;
  bool is_full;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const View3 q_ = q, k_ = k, v_ = v;
    const SView qs_ = qscale;
    const PView pos_ = pos;
    const float thr_ = qk_clip, alpha_ = alpha, floor_ = floor, eps_ = eps;
    const size_t NH_ = NH, NKV_ = NKV, HD_ = HD;
    const bool full_ = is_full;
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t qn = C * NH_;
    const size_t kn = C * NKV_;
    if (id < qn) {
      const size_t t = id / NH_, h = id % NH_;
      const float n = inkling_l2_row_norm(team, HD_, KOKKOS_LAMBDA(
                                                          size_t d) {
        return static_cast<float>(q_(t, h, d)) * static_cast<float>(qs_(h, d));
      });
      const float c = Kokkos::fminf(1.0f, thr_ / (n + 1e-6f));
      float tau = 1.0f;
      if (full_) {
        const float ratio =
            (static_cast<float>(pos_(t)) + 1.0f) / floor_;
        tau = 1.0f + alpha_ * Kokkos::logf(Kokkos::fmaxf(ratio, 1.0f));
      }
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, HD_), KOKKOS_LAMBDA(size_t d) {
            q_(t, h, d) = static_cast<Scalar>(static_cast<float>(q_(t, h, d)) *
                                             static_cast<float>(qs_(h, d)) *
                                             c * tau);
          });
    } else if (id < qn + kn) {
      const size_t j = id - qn;
      const size_t t = j / NKV_, h = j % NKV_;
      const float n = inkling_l2_row_norm(team, HD_, KOKKOS_LAMBDA(
                                                          size_t d) {
        return static_cast<float>(k_(t, h, d));
      });
      const float s = Kokkos::fminf(1.0f, thr_ / (n + 1e-6f));
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, HD_), KOKKOS_LAMBDA(size_t d) {
            k_(t, h, d) =
                static_cast<Scalar>(static_cast<float>(k_(t, h, d)) * s);
          });
    } else {
      const size_t j = id - qn - kn;
      const size_t t = j / NKV_, h = j % NKV_;
      const float inv = inkling_rms_row_inv(team, HD_, eps_,
                                            KOKKOS_LAMBDA(size_t d) {
                                              return static_cast<float>(
                                                  v_(t, h, d));
                                            });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, HD_), KOKKOS_LAMBDA(size_t d) {
            v_(t, h, d) =
                static_cast<Scalar>(static_cast<float>(v_(t, h, d)) * inv);
          });
    }
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
template <class ExecSpace, class Scalar, class SView>
void inkling_scaled_l2_clip(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const SView& scale,
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
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>, SView,
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
template <class ExecSpace, class Scalar, class SView, class PView>
void inkling_log_scaled_l2_clip(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const SView& scale,
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
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>, SView,
          PView,
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>>{
          x, scale, pos, y, H, D, threshold, alpha, n_floor},
      T * H, D, label);
}

// Fused 3-stream weighted input norms (layer_forward section 2 when the
// streams are already materialized): hn = rms(qh)*wq (out-of-place),
// kh = rms(kh)*wk and vh = rms(vh)*wv in place. One launch instead of
// three; each team's row math repeats the separate rmsnorm rows verbatim,
// so the result is bit-identical. Backend-portable team-parallel kernel.
template <class ExecSpace, class Scalar>
void inkling_fused_input_norms3(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& qh,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kh,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& vh,
    const Kokkos::View<Scalar*, ExecSpace>& wq,
    const Kokkos::View<Scalar*, ExecSpace>& wk,
    const Kokkos::View<Scalar*, ExecSpace>& wv, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hn,
    const std::string& label = "KokkosDNN::inkling_fused_input_norms3") {
  const size_t C = qh.extent(0), D = qh.extent(1);
  if (kh.extent(0) != C || kh.extent(1) != D || vh.extent(0) != C ||
      vh.extent(1) != D || hn.extent(0) != C || hn.extent(1) != D ||
      wq.extent(0) != D || wk.extent(0) != D || wv.extent(0) != D)
    throw std::invalid_argument("inkling_fused_input_norms3: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::FusedInputNorms3Team<ExecSpace, Scalar>{qh, kh, vh, wq, wk, wv,
                                                   hn, eps},
      3 * C, D, label);
}

// Fused sliding copy+norm (sections 1+2, no-msg layer-0 skip path):
// hn/kh/vh = rms(h)*w from the raw stream, one launch instead of one
// copy plus three norms. Bit-identical: the copies are exact and the
// three reductions over the same row agree, so one shared reduction
// reproduces all three inv values.
template <class ExecSpace, class Scalar>
void inkling_sliding_alias_norms(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& h,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hn,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kh,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& vh,
    const std::string& label = "KokkosDNN::inkling_sliding_alias_norms") {
  const size_t C = h.extent(0), D = h.extent(1);
  if (hn.extent(0) != C || hn.extent(1) != D || kh.extent(0) != C ||
      kh.extent(1) != D || vh.extent(0) != C || vh.extent(1) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("inkling_sliding_alias_norms: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::SlidingBroadcastNormTeam<ExecSpace, Scalar,
                                    Kokkos::View<Scalar**, Kokkos::LayoutRight,
                                                 ExecSpace>>{h, w, hn, kh, vh,
                                                             eps},
      C, D, label);
}

// Fused sliding copy+norm from a Q=1 retrieval (no-msg sliding layers):
// hn/kh/vh = rms(retr(:,0,:))*w. One launch instead of three deep_copies
// plus three norms. Bit-identical (copies are exact; shared reduction
// agrees with the three identical reductions).
template <class ExecSpace, class Scalar>
void inkling_sliding_retr_norms(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& retr,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hn,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kh,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& vh,
    const std::string& label = "KokkosDNN::inkling_sliding_retr_norms") {
  const size_t C = retr.extent(0), D = retr.extent(2);
  if (retr.extent(1) != 1 || hn.extent(0) != C || hn.extent(1) != D ||
      kh.extent(0) != C || kh.extent(1) != D || vh.extent(0) != C ||
      vh.extent(1) != D || w.extent(0) != D)
    throw std::invalid_argument("inkling_sliding_retr_norms: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::SlidingBroadcastNormTeam<ExecSpace, Scalar,
                                    Kokkos::View<Scalar***, Kokkos::LayoutRight,
                                                 ExecSpace>>{retr, w, hn, kh,
                                                             vh, eps},
      C, D, label);
}

// Fused full-kind copy+norm from a Q=3 retrieval (no-msg full layers):
// hn/kh/vh = rms(retr(:,q,:))*wq/q/k/w for q = 0/1/2. One launch instead
// of three deep_copies plus three norms; bit-identical (copies exact,
// per-row math verbatim).
template <class ExecSpace, class Scalar>
void inkling_full_retr_norms(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& retr,
    const Kokkos::View<Scalar*, ExecSpace>& wq,
    const Kokkos::View<Scalar*, ExecSpace>& wk,
    const Kokkos::View<Scalar*, ExecSpace>& wv, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hn,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kh,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& vh,
    const std::string& label = "KokkosDNN::inkling_full_retr_norms") {
  const size_t C = retr.extent(0), D = retr.extent(2);
  if (retr.extent(1) != 3 || hn.extent(0) != C || hn.extent(1) != D ||
      kh.extent(0) != C || kh.extent(1) != D || vh.extent(0) != C ||
      vh.extent(1) != D || wq.extent(0) != D || wk.extent(0) != D ||
      wv.extent(0) != D)
    throw std::invalid_argument("inkling_full_retr_norms: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::FullRetrNormsTeam<ExecSpace, Scalar>{retr, wq, wk, wv, hn, kh,
                                                vh, eps},
      3 * C, D, label);
}

// Fused full-kind alias copy+norm (no-msg layer-0 full path):
// hn/kh/vh = rms(h)*wq/q/k/w. One launch instead of one copy plus three
// norms. Bit-identical: one shared reduction reproduces the three
// identical inv values; only the weight multiply differs.
template <class ExecSpace, class Scalar>
void inkling_full_alias_norms(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& h,
    const Kokkos::View<Scalar*, ExecSpace>& wq,
    const Kokkos::View<Scalar*, ExecSpace>& wk,
    const Kokkos::View<Scalar*, ExecSpace>& wv, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hn,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kh,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& vh,
    const std::string& label = "KokkosDNN::inkling_full_alias_norms") {
  const size_t C = h.extent(0), D = h.extent(1);
  if (hn.extent(0) != C || hn.extent(1) != D || kh.extent(0) != C ||
      kh.extent(1) != D || vh.extent(0) != C || vh.extent(1) != D ||
      wq.extent(0) != D || wk.extent(0) != D || wv.extent(0) != D)
    throw std::invalid_argument("inkling_full_alias_norms: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::FullAliasNormsTeam<ExecSpace, Scalar>{h, wq, wk, wv, hn, kh, vh,
                                                 eps},
      C, D, label);
}

// Single-stream norm straight from a Q=1 retrieval (section 12 no-msg
// mlp-input path): out = rms(retr(:,0,:))*w. One launch instead of a
// deep_copy plus rmsnorm; bit-identical (the copy is exact).
template <class ExecSpace, class Scalar>
void inkling_retr1_rmsnorm(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& retr,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::inkling_retr1_rmsnorm") {
  const size_t C = retr.extent(0), D = retr.extent(2);
  if (retr.extent(1) != 1 || out.extent(0) != C || out.extent(1) != D ||
      w.extent(0) != D)
    throw std::invalid_argument("inkling_retr1_rmsnorm: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::Retr1RmsTeam<ExecSpace, Scalar>{retr, w, out, eps}, C, D, label);
}

// Last-row norm straight from a Q=1 retrieval (final head):
// y = rms(retr(row,0,:))*w. One launch instead of deep_copy plus
// rmsnorm_row (two launches). FP reassociation vs rmsnorm_row (team vs
// range reduction, a few ulp): use a tight tolerance, not bitwise
// equality.
template <class ExecSpace, class Scalar>
void inkling_rmsnorm_from_retr_last_row(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& retr,
    size_t row, const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::inkling_rmsnorm_from_retr") {
  const size_t C = retr.extent(0), D = retr.extent(2);
  if (retr.extent(1) != 1 || row >= C || w.extent(0) != D ||
      y.extent(0) != D)
    throw std::invalid_argument(
        "inkling_rmsnorm_from_retr_last_row: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::Retr1LastRowTeam<ExecSpace, Scalar>{retr, row, w, y, eps}, 1, D,
      label);
}

// Fused q/k/v head clips (section 6): q scaled (sliding) or log-scaled
// (full) clip in place, k plain L2 clip in place, v unweighted RMSNorm in
// place. One launch instead of three; per-branch row math repeats the
// separate kernels verbatim, so the result is bit-identical.
// q/k/v: [C, H, HD] head aliases; qscale: [NH, HD]; pos: [C] positions
// (only read when is_full). In-place safe.
template <class ExecSpace, class Scalar, class SView, class PView>
void inkling_fused_qkv_clips(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& q,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& k,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& v,
    const SView& qscale, float qk_clip, const PView& pos, float alpha,
    float n_floor, float eps, bool is_full,
    const std::string& label = "KokkosDNN::inkling_fused_qkv_clips") {
  static_assert(PView::rank == 1, "inkling_fused_qkv_clips: pos rank 1");
  const size_t C = q.extent(0), NH = q.extent(1), HD = q.extent(2);
  const size_t NKV = k.extent(1);
  if (k.extent(0) != C || k.extent(2) != HD || v.extent(0) != C ||
      v.extent(1) != NKV || v.extent(2) != HD ||
      qscale.extent(0) != NH || qscale.extent(1) != HD || pos.extent(0) != C)
    throw std::invalid_argument("inkling_fused_qkv_clips: dim mismatch");
  Impl::inkling_norm_team_launch<ExecSpace>(
      Impl::FusedQkvClipsTeam<ExecSpace, Scalar, SView, PView>{
          q, k, v, qscale, qk_clip, pos, alpha, n_floor, eps, NH, NKV, HD, C,
          is_full},
      C * (NH + 2 * NKV), HD, label);
}

}  // namespace KokkosDNN

#endif
