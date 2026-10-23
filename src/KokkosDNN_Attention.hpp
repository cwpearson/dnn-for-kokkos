// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_ATTENTION_HPP
#define KOKKOSDNN_ATTENTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>  // memcpy (bit tricks in bidir_expf)
#include <stdexcept>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_Macros.hpp"
#ifdef KOKKOSDNN_ENABLE_CUDNN
#include "tpl/cudnn/KokkosDNN_CuDNN.hpp"
#endif
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
#include "tpl/miopen/KokkosDNN_MIOpen.hpp"
#endif

namespace KokkosDNN {
namespace Impl {

// Portable per-team scratch ceiling. The Serial backend aborts above
// 32 KiB of team scratch; CUDA/HIP/SYCL allow more, but every tuner below
// sizes to this minimum so one binary runs on all backends.
inline constexpr size_t kMaxTeamScratchBytes = 32 * 1024;

// Team-scratch footprint (in floats) of GqaStreamingFunctor below: scaled Q
// rows plus FP32 accumulator rows ([nqr x SS] each), one K/V tile row pair
// per (tile slot, distinct-KV-head row), and per-row (m, l) state. SS carries
// the +1 bank pad (plus one more when that lands on a multiple of 32).
// Shared by the device functor (offsets) and the host launcher (request),
// so the two cannot drift apart. Device-callable (used inside operator()).
KOKKOS_INLINE_FUNCTION
size_t gqa_scratch_floats(int tchunk, int hchunk, int ktiles, int kvrows,
                           size_t D) {
  size_t ss = D + 1;
  if (ss % 32 == 0) ++ss;
  return (size_t)(2 * tchunk * hchunk + 2 * kvrows * ktiles) * ss +
         (size_t)2 * tchunk * hchunk;
}

// Max distinct KV heads inside any HC-wide head chunk: the K/V tile only
// needs kvrows rows per key slot (standard GQA shares one KV head per whole
// chunk, so kvrows == 1 there). Host-side loop over <= 32 chunks; exact, so
// no scratch is wasted on never-used tile rows.
inline int gqa_max_kvrows(size_t Hq, size_t Hkv, int hchunk) {
  int kvmax = 1;
  for (size_t hbase = 0; hbase < Hq; hbase += (size_t)hchunk) {
    int n = 0, prev = -1;
    for (int jj = 0; jj < hchunk; ++jj) {
      const int kvh =
          static_cast<int>(((hbase + jj) * Hkv) / Hq);
      if (jj == 0 || kvh != prev) {
        ++n;
        prev = kvh;
      }
    }
    if (n > kvmax) kvmax = n;
  }
  return kvmax;
}

// Team-parallel streaming online-softmax GQA core shared by the plain and
// softcap paths below (UseMask/UseSoftcap select the softcap extras at
// compile time, so both entry points execute an identical FP32 op sequence
// when the mask is all-valid and softcap <= 0).
//
// One team per (token chunk of tchunk tokens, head chunk of hchunk heads)
// covering tchunk*hchunk query rows strided across its threads. Heads in
// a chunk usually share one KV head (standard GQA), and every token in the
// chunk attends nearly the same key range, so each distinct K/V row is
// loaded once per key into team-scratch and reused across the whole token
// x head tile, instead of every (t, h) thread streaming K/V from global
// memory independently.
//
// Keys stream through in tiles of KT (runtime ktiles): one cooperative
// tile load plus two barriers per KT keys instead of per key, and each
// tile load spans distinct-heads x KT x D units so the whole team stays
// busy even for small head dims. Per-team team-scratch holds the scaled
// query rows (loaded once, scale fused), the FP32 output accumulators (keys
// stream through fast scratch; each output row is written once at the end),
// the K/V tile (kvrows rows per key slot), and the per-row (m, l) streaming
// state. Row loads use the whole team (TeamVectorRange), which coalesces
// for D-contiguous (LayoutRight) inputs and still issues D-parallel loads
// otherwise; all per-row work is serial on the owning team thread, so no
// vector-lane coherence hazards exist by construction.
//
// Math is the per-key deferred-normalization streaming rescaling:
//   m_new = max(m, s), l_new = l*exp(m-m_new) + exp(s-m_new),
//   acc_new = acc*exp(m-m_new) + V*exp(s-m_new),
// with a single acc/l write per row at the end.
// The final result equals softmax(QK^T*scale)*V up to FP32 rounding; the dot
// uses eight interleaved accumulators over full-D rows (independent FMA
// chains hide latency and vectorize portably; order differs from a plain
// serial dot only by ~1-ulp reassociation, exactly as between any two
// parallel schedules). Update order over D is serial-blocked, so results
// agree with a sequential reference up to ~1-ulp reassociation. FP32
// accumulation throughout; the only Scalar rounding is the single final
// write. UF blocks the D axis of loads/updates (each iteration handles UF
// consecutive D) so the compiler can issue wide loads/stores for
// D-contiguous inputs and interleave independent chains; UF == 1 keeps the
// plain scalar order.
// Causal key-window ends (half-open [klo, khi)) for token t of a chunk
// starting at absolute position q_off. Free function (not a functor
// member) so the streaming and single-token decode kernels below share one
// definition of the mask; device-callable.
KOKKOS_INLINE_FUNCTION
void gqa_key_window(size_t q_off, size_t Ktot, size_t sliding_window, size_t t,
                    size_t& klo, size_t& khi) {
  khi = q_off + t + 1;
  if (khi > Ktot) khi = Ktot;
  klo = 0;
  if (sliding_window > 0) {
    const size_t p = q_off + t;  // absolute position of query t
    const size_t start =
        (p + 1 >= sliding_window) ? (p + 1 - sliding_window) : 0;
    if (start > klo) klo = start;
  }
}

template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class MView, class OView, int UF, int HC, bool UseMask,
          bool UseSoftcap>
struct GqaStreamingFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using ScratchView =
      Kokkos::View<float*, typename ExecSpace::scratch_memory_space,
                   Kokkos::MemoryUnmanaged>;
  QView q;
  KView k;
  VView v;
  MView key_valid;  // unused unless UseMask (plain path passes an empty view)
  OView out;
  size_t T, Hq, D, Hkv, Ktot, q_off, sliding_window;
  int tchunk;  // tokens covered per team (heads covered = HC, compile-time)
  int ktiles;  // keys streamed per K-tile (runtime KT blocking)
  int kvrows;  // tile rows per key slot (max distinct KV heads over chunks)
  float scale, softcap;  // softcap unused unless UseSoftcap

  // KV head of head (hbase + jj): plain integer arithmetic, callable from
  // team-body scalar code on any backend.
  static KOKKOS_INLINE_FUNCTION int kv_of_head(size_t hbase, size_t Hkv,
                                               size_t Hq, int jj) {
    return static_cast<int>(((hbase + jj) * Hkv) / Hq);
  }
  // Causal key-window ends (half-open [klo, khi)) for token t (shared with
  // the single-token decode kernel below via gqa_key_window).
  static KOKKOS_INLINE_FUNCTION void key_window(size_t q_off, size_t Ktot,
                                                size_t sliding_window,
                                                size_t t, size_t& klo,
                                                size_t& khi) {
    gqa_key_window(q_off, Ktot, sliding_window, t, klo, khi);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    // Local handles: inner lambdas capture by value without an implicit
    // `this` capture (quiets -Wdeprecated under C++20).
    const auto Q = q;
    const auto K = k;
    const auto V = v;
    const auto O = out;
    const auto Valid = key_valid;
    const size_t Dloc = D;
    const size_t Hqloc = Hq;
    const size_t Tloc = T;
    const int TC = tchunk;
    const int KT = ktiles;
    const int NR = kvrows;
    const float scl = scale;
    const float cap = softcap;
    // Team (ct, ch) covers tokens [tbase, tbase+TC) and heads
    // [hbase, hbase+HC). Rows of the tile are strided across team threads
    // (row rr owned by rank rr % team_size), so any team size works: 1 row
    // per thread on CUDA, all rows looped by the single thread on Serial.
    // Per-row work is serial on the owning thread (vector_length == 1, so
    // ThreadVectorRange is just a serial loop); per-row state lives in
    // team-scratch, never in thread-lane stack.
    const int rank = static_cast<int>(member.team_rank());
    const int team_n = static_cast<int>(member.team_size());

    const size_t nchunks_h = Hqloc / HC;
    const size_t chunk = static_cast<size_t>(member.league_rank());
    const size_t tbase = (chunk / nchunks_h) * TC;
    const size_t hbase = (chunk % nchunks_h) * HC;

    // Union key range over the chunk's real tokens (khi/klo are monotone in
    // t, so the extrema sit at the chunk ends; no cross-thread exchange).
    const size_t t_last =
        (tbase + TC - 1 < Tloc) ? tbase + TC - 1 : Tloc - 1;
    size_t klo_u, khi_u, klo_first, khi_last;
    key_window(q_off, Ktot, sliding_window, tbase, klo_u, khi_last);
    key_window(q_off, Ktot, sliding_window, t_last, klo_first, khi_u);
    (void)khi_last;
    (void)klo_first;

    // Scratch layout: Q rows, acc rows (TC*HC each, stride SS), one K/V
    // tile row pair per (key slot, distinct-KV-head row) ([KT x NR], tile
    // slot kki and head row r at (kki*NR + r)), then per-row (m, l) state.
    // SS carries the +1 pad (plus one more on multiples of 32) so the team
    // threads touching the same d in different rows hit distinct banks
    // instead of serializing 32-way. Sizes mirror gqa_scratch_floats.
    const size_t nqr = static_cast<size_t>(TC) * HC;  // query rows per team
    size_t SS = Dloc + 1;
    if (SS % 32 == 0) ++SS;
    ScratchView buf(member.team_scratch(0),
                    gqa_scratch_floats(TC, HC, KT, NR, Dloc));
    const size_t qoff = 0;
    const size_t aoff = nqr * SS;
    const size_t koff = aoff + nqr * SS;
    const size_t voff = koff + static_cast<size_t>(NR) * KT * SS;
    const size_t moff = voff + static_cast<size_t>(NR) * KT * SS;
    const size_t loff = moff + nqr;
    // Distinct-KV-head plan for this head chunk, computed once per team:
    // ku_of_row[r] is the kv head of the r-th leader head (jj == 0 or a
    // head whose kv head differs from its predecessor), myrow_of_head[jj]
    // is head jj's row. Stack arrays are safe to capture here because
    // vector_length == 1 (every inner lambda runs on its calling thread,
    // which reads its own identically-computed copy).
    const size_t Hkbloc = Hkv;
    int ku_of_row[HC];
    int myrow_of_head[HC];
    int nrows = 0;
    for (int jj = 0; jj < HC; ++jj) {
      const int kvh = kv_of_head(hbase, Hkbloc, Hqloc, jj);
      if (jj == 0 || kvh != kv_of_head(hbase, Hkbloc, Hqloc, jj - 1)) {
        ku_of_row[nrows] = kvh;
        myrow_of_head[jj] = nrows;
        ++nrows;
      } else {
        myrow_of_head[jj] = nrows - 1;
      }
    }
    const size_t nbu = Dloc / UF;

    // Scaled Q rows, loaded once per team (rows past T clamp to row 0;
    // their scratch is never used). D is blocked by UF so threads touch
    // consecutive elements. Nested (row, block) loops avoid flattened
    // index division.
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, nqr), KOKKOS_LAMBDA(size_t ii) {
          const size_t tii = tbase + ii / HC;
          const size_t hj = hbase + ii % HC;
          const size_t tq = (tii < Tloc) ? tii : 0;
          for (size_t b = 0; b < nbu; ++b) {
            const size_t bb = b * UF;
            for (int u = 0; u < UF; ++u) {
              const size_t d = bb + u;
              buf(qoff + ii * SS + d) =
                  static_cast<float>(Q(tq, hj, d)) * scl;
              buf(aoff + ii * SS + d) = 0.0f;
            }
          }
        });
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nqr), KOKKOS_LAMBDA(size_t r) {
          buf(moff + r) = -1e30f;  // -inf equivalent (scores are >> this)
          buf(loff + r) = 0.0f;
        });
    // Cross-warp visibility for Q/acc/state before the key loop consumes
    // them on other warps.
    member.team_barrier();

    // Keys stream through in tiles of KT: one cooperative tile load plus
    // two barriers per KT keys instead of per key. Tile loads span nrows x
    // KT x D units (flat (row, slot, block) index, shift-free decode), so
    // the whole team stays busy; tail slots and masked keys are zero-filled
    // so the row loop below only ever reads owned tile rows.
    const size_t KTs = static_cast<size_t>(KT);
    const size_t NRs = static_cast<size_t>(NR);
    for (size_t k0 = klo_u; k0 < khi_u; k0 += KTs) {
      const int kw = (k0 + KTs <= khi_u) ? KT : static_cast<int>(khi_u - k0);
      // Load this tile's distinct K/V rows once per team (one row per
      // leader head; row->kv-head mapping hoisted above). Team-wide
      // collective: every team thread participates.
      for (int r = 0; r < nrows; ++r) {
        const size_t ku = static_cast<size_t>(ku_of_row[r]);
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, KTs * nbu),
            KOKKOS_LAMBDA(size_t u) {
              const int kki = static_cast<int>(u / nbu);
              const size_t b = (u % nbu) * UF;
              const size_t kk = k0 + static_cast<size_t>(kki);
              float* __restrict__ kdst = buf.data() + koff +
                                         (static_cast<size_t>(kki) * NRs + r) *
                                             SS +
                                         b;
              float* __restrict__ vdst = buf.data() + voff +
                                         (static_cast<size_t>(kki) * NRs + r) *
                                             SS +
                                         b;
              bool ok = (kki < kw);
              if (ok && UseMask) ok = static_cast<bool>(Valid(kk));
              if (ok) {
                for (int uu = 0; uu < UF; ++uu) {
                  const size_t d = b + uu;
                  kdst[uu] = static_cast<float>(K(kk, ku, d));
                  vdst[uu] = static_cast<float>(V(kk, ku, d));
                }
              } else {
                for (int uu = 0; uu < UF; ++uu) {
                  kdst[uu] = 0.0f;
                  vdst[uu] = 0.0f;
                }
              }
            });
      }
      // Cross-warp visibility for the freshly loaded tile before any
      // thread's warp consumes it in the row loop below.
      member.team_barrier();
      // Per-row streaming update, serial on the owning team thread. Each
      // row is owned by exactly one team thread; (m, l) ride in registers
      // across the tile's keys and are written back once per tile. Rows of
      // tokens outside a key's window sit out (their state stays put, so no
      // NaN can poison it). With vector_length == 1 there are no lanes: the
      // dots, the scalars, and the updates are plain serial code with no
      // coherence hazards, and skipping rows is always safe.
      for (int rr = rank; rr < (int)nqr; rr += team_n) {
        const size_t t = tbase + rr / HC;
        if (t >= Tloc) continue;  // dummy row of a partial edge chunk
        size_t klo_t = 0, khi_t = 0;
        key_window(q_off, Ktot, sliding_window, t, klo_t, khi_t);
        const int j = rr % HC;
        const int myrow = myrow_of_head[j];
        float m = buf(moff + rr);
        float l = buf(loff + rr);
        // Raw restrict pointers so the compiler can pipeline/vectorize the
        // serial streams (View operator() on a shared base otherwise
        // inhibits aliasing assumptions). Portable C++, no intrinsics.
        const float* __restrict__ qrow = buf.data() + qoff + rr * SS;
        float* __restrict__ accrow = buf.data() + aoff + rr * SS;
        for (int kki = 0; kki < kw; ++kki) {
          const size_t kk = k0 + static_cast<size_t>(kki);
          if (kk < klo_t || kk >= khi_t) continue;
          if constexpr (UseMask) {
            if (!Valid(kk)) continue;
          }
          const float* __restrict__ krow = buf.data() + koff +
                                           (static_cast<size_t>(kki) * NRs +
                                            myrow) *
                                               SS;
          const float* __restrict__ vrow = buf.data() + voff +
                                           (static_cast<size_t>(kki) * NRs +
                                            myrow) *
                                               SS;
          // Full-D dot with eight interleaved accumulators (independent
          // FMA chains hide latency and vectorize portably; reassociation
          // vs a serial dot is ~1 ulp). Q rows carry the fused scale, so
          // no extra multiply here.
          float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
          float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
          size_t dd = 0;
          const size_t dend8 = Dloc & ~size_t(7);
          for (; dd < dend8; dd += 8) {
            a0 += qrow[dd] * krow[dd];
            a1 += qrow[dd + 1] * krow[dd + 1];
            a2 += qrow[dd + 2] * krow[dd + 2];
            a3 += qrow[dd + 3] * krow[dd + 3];
            a4 += qrow[dd + 4] * krow[dd + 4];
            a5 += qrow[dd + 5] * krow[dd + 5];
            a6 += qrow[dd + 6] * krow[dd + 6];
            a7 += qrow[dd + 7] * krow[dd + 7];
          }
          float stail = 0.0f;
          for (; dd < Dloc; ++dd) stail += qrow[dd] * krow[dd];
          const float s =
              ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)) + stail;
          float ss = s;
          if (cap > 0) ss = cap * tanhf(ss / cap);
          // m == -1e30 on the first key => a = exp(-1e30) = 0, as desired.
          // (Runtime cap check: plain passes softcap == 0, so the tanh is
          // skipped there exactly as with if constexpr.)
          // Split on new-maximum so each path needs one expf (the other is
          // expf(0) == 1): a new max rescales acc by aa, otherwise acc only
          // accumulates (aa == 1, bit-identical to the fused form).
          if (ss > m) {
            const float aa = expf(m - ss);
            const float l_new = l * aa + 1.0f;
            for (size_t b = 0; b < nbu; ++b) {
              const size_t bb = b * UF;
              for (int u = 0; u < UF; ++u) {
                const size_t d = bb + u;
                accrow[d] = accrow[d] * aa + vrow[d];
              }
            }
            m = ss;
            l = l_new;
          } else {
            const float gg = expf(ss - m);
            const float l_new = l + gg;
            for (size_t b = 0; b < nbu; ++b) {
              const size_t bb = b * UF;
              for (int u = 0; u < UF; ++u) {
                const size_t d = bb + u;
                accrow[d] += vrow[d] * gg;
              }
            }
            l = l_new;
          }
        }
        buf(moff + rr) = m;
        buf(loff + rr) = l;
      }
      // End-of-tile barrier: the next tile's loads overwrite the K/V tile
      // this tile's update just consumed; without it a fast warp can
      // clobber rows a slower warp is still accumulating from.
      member.team_barrier();
    }
    // Write back: rows with no contributing key (l == 0) keep zeros,
    // matching the previous kernel's zero-init contract.
    for (int rr = rank; rr < (int)nqr; rr += team_n) {
      const size_t t = tbase + rr / HC;
      if (t >= Tloc) continue;  // dummy row of a partial edge chunk: no row
      const size_t h = hbase + rr % HC;
      const float l = buf(loff + rr);
      if (l > 0.0f) {
        const float inv_l = 1.0f / l;
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, nbu), KOKKOS_LAMBDA(size_t m) {
              const size_t b = m * UF;
              for (int u = 0; u < UF; ++u) {
                const size_t d = b + u;
                O(t, h, d) =
                    static_cast<Scalar>(buf(aoff + rr * SS + d) * inv_l);
              }
            });
      } else {
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, nbu), KOKKOS_LAMBDA(size_t m) {
              const size_t b = m * UF;
              for (int u = 0; u < UF; ++u)
                O(t, h, b + u) = static_cast<Scalar>(0);
            });
      }
    }
  }
};

// One team per (token chunk, head chunk) with vector_length == 1 and
// team_size == AUTO (1 on Serial, occupancy-chosen on CUDA): rows stride
// across team threads, so the kernel is correct for every team size and on
// every backend. Token chunks of up to 8 and 4-head chunks let a team load
// each K/V tile once and reuse it across 32 query rows (standard GQA shares
// one KV head per 4 heads); keys stream in tiles of up to 8 so barriers and
// cooperative loads amortize over KT keys. Smaller chunks cover odd head
// counts, small T, and large D (scratch fit). All per-row work is serial
// (no vector lanes exist), so the dot/update order is fixed and identical
// across the plain and softcap instantiations (and backends) for a given D;
// UF == 4 additionally groups D into consecutive quadruples for wide loads.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class MView, class OView, int UF, int HC, bool UseMask,
          bool UseSoftcap>
void launch_gqa_streaming(const QView& q, const KView& k, const VView& v,
                           const MView& key_valid, const OView& out, size_t T,
                           size_t Hq, size_t D, size_t Hkv, size_t Ktot,
                           size_t q_off, float scale, float softcap,
                           size_t sliding_window, int tchunk, int ktiles,
                           int kvrows, const std::string& label) {
  GqaStreamingFunctor<ExecSpace, Scalar, QView, KView, VView, MView, OView, UF,
                      HC, UseMask, UseSoftcap>
      f{q, k,      v,       key_valid, out, T, Hq, D, Hkv, Ktot,
        q_off, sliding_window, tchunk, ktiles, kvrows, scale, softcap};
  const size_t nchunks_t = (T + tchunk - 1) / tchunk;
  Kokkos::TeamPolicy<ExecSpace> policy(nchunks_t * (Hq / HC),
                                       Kokkos::AUTO, 1);
  policy.set_scratch_size(
      0, Kokkos::PerTeam(gqa_scratch_floats(tchunk, HC, ktiles, kvrows, D) *
                         sizeof(float)));
  Kokkos::parallel_for(label, policy, f);
}

template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class MView, class OView, bool UseMask, bool UseSoftcap>
void launch_gqa_streaming_auto(const QView& q, const KView& k, const VView& v,
                               const MView& key_valid, const OView& out,
                               size_t T, size_t Hq, size_t D, size_t Hkv,
                               size_t Ktot, size_t q_off, float scale,
                               float softcap, size_t sliding_window,
                               const std::string& label) {
  // 4-head chunks share KV rows under standard GQA; odd head counts fall
  // back to one head per team. Token chunks of 8 reuse each K/V tile across
  // 32 query rows; keys stream in tiles of up to 8 (one cooperative tile
  // load plus two barriers per tile instead of per key). Smaller chunks and
  // tiles keep team scratch (see gqa_scratch_floats) within the portable
  // per-team budget (kMaxTeamScratchBytes) for large head dims. The K/V
  // tile only carries
  // the head chunk's distinct KV rows (1 under standard GQA), so wide tiles
  // stay affordable where they matter. UF == 8 unrolls D in consecutive
  // octuples (D % 8 == 0), UF == 4 in quadruples (D % 4 == 0); all branches
  // share the dot/update order, so plain-vs-softcap bit-identity holds
  // within each branch.
  const int hchunk = (Hq % 4 == 0) ? 4 : 1;
  const int kvmax = gqa_max_kvrows(Hq, Hkv, hchunk);
  int tchunk = 8;  // (hchunk stays runtime here for the scratch budget)
  if (T < static_cast<size_t>(tchunk)) tchunk = static_cast<int>(T);
  int ktiles = 8;
  // Shrink tokens first (KT drives the barrier/load-width win), then tiles.
  while (tchunk > 1 &&
         gqa_scratch_floats(tchunk, hchunk, ktiles, kvmax, D) * sizeof(float) >
             kMaxTeamScratchBytes)
    tchunk /= 2;
  while (ktiles > 1 &&
         gqa_scratch_floats(tchunk, hchunk, ktiles, kvmax, D) * sizeof(float) >
             kMaxTeamScratchBytes)
    ktiles /= 2;
  const int uf = (D % 8 == 0) ? 8 : ((D % 4 == 0) ? 4 : 1);
  if (hchunk == 4) {
    if (uf == 8) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 8, 4, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, ktiles, kvmax, label);
    } else if (uf == 4) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 4, 4, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, ktiles, kvmax, label);
    } else {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 1, 4, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, ktiles, kvmax, label);
    }
  } else {
    if (uf == 8) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 8, 1, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, ktiles, kvmax, label);
    } else if (uf == 4) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 4, 1, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, ktiles, kvmax, label);
    } else {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 1, 1, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, ktiles, kvmax, label);
    }
  }
}

// ---------------------------------------------------------------------------
// Single-token (T == 1) decode attention: one team per query head.
//
// Why a separate kernel: at T == 1 the streaming functor above hands each of
// the HC heads of a team its own query row, and that row walks the whole key
// axis serially on one thread. With Hq == 32, HC == 4 only 4 of 256 team
// threads do the key loop, so decode attention (a memory-bound GEMV per head:
// ~2 x Ktot x D x sizeof(Scalar) bytes) ran at ~0.1 GB/s -- measured 1.9 ms
// per call at Ktot = 900, D = 64, Hq = 32, i.e. ~19 ms of the ~27 ms
// native-build decode step, and growing with context. Decode is exactly this
// shape, once per layer per token, so it dominated batch-1 decode in both
// builds (the TPL build additionally rebuilt its cuDNN SDPA plan per token;
// see the dispatch note in attention_gqa_streaming).
//
// Structure (league = Hq teams, team = one thread per head dim where the
// backend allows it; every loop strides on the runtime team size, so Serial
// and clamped teams are correct by construction):
//   * keys stream through in tiles of `ktile`; K and V tile rows are staged
//     in team scratch by the whole team, coalesced along D;
//   * scores: thread r owns keys r, r + team, ... of the tile and does the
//     full-D dot (8 interleaved accumulators, same as the streaming kernel)
//     from the staged K row;
//   * one team max per tile folds into the running max m (the same deferred
//     normalization the streaming kernel uses): the shared accumulator row
//     and each thread's partial denominator are rescaled by exp(m - m_new);
//   * the accumulator row is updated by thread d for dim d, so the V
//     multiply is D-way parallel over consecutive addresses (and the staged
//     rows are padded to an odd stride so the score phase's row access is
//     bank-conflict-free);
//   * per-thread partial denominators are combined once, at the end.
// The result equals softmax(Q K^T * scale) V with FP32 accumulation and the
// same per-key rescaling as the streaming kernel, up to FP32 rounding of a
// different summation order (tile-blocked instead of strictly sequential).
// The tile-pair (K + V) plus state is sized by the launcher to stay within
// the portable kMaxTeamScratchBytes budget, so this runs on every backend.

// Team-scratch footprint (in floats) of GqaDecodeFunctor: scaled query row
// and accumulator row, this tile's scores/weights, per-slot partial
// denominators, and the staged K/V tile pair at the padded row stride.
// Device-callable so the functor and the launcher cannot drift apart.
KOKKOS_INLINE_FUNCTION
size_t gqa_decode_scratch_floats(size_t D, int slots, int ktile) {
  const size_t kss = static_cast<size_t>(static_cast<int>(D) | 1);
  return 2 * D + static_cast<size_t>(ktile) + static_cast<size_t>(slots) +
         2 * static_cast<size_t>(ktile) * kss;
}

template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
struct GqaDecodeFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using ScratchView =
      Kokkos::View<float*, typename ExecSpace::scratch_memory_space,
                   Kokkos::MemoryUnmanaged>;
  QView q;
  KView k;
  VView v;
  OView out;
  size_t Hq, D, Hkv, Ktot, q_off, sliding_window;
  int ktile;   // keys staged per tile
  int kss;     // staged-row stride in floats (odd => no bank conflicts)
  int tslots;  // team slots the launcher reserved in scratch (>= team size)
  float scale;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto Q = q;
    const auto K = k;
    const auto V = v;
    const auto O = out;
    const size_t Dloc = D;
    const size_t Kvloc = Hkv;
    const size_t Hqloc = Hq;
    const size_t Ktotl = Ktot;
    const size_t qoff = q_off;
    const size_t win = sliding_window;
    const int KT = ktile;
    const int SS = kss;
    const int slots = tslots;
    const float scl = scale;
    const int rank = static_cast<int>(member.team_rank());
    const int team_n = static_cast<int>(member.team_size());
    const size_t h = static_cast<size_t>(member.league_rank());
    const size_t kvh = (h * Kvloc) / Hqloc;

    // Single query row: one causal key window for the whole team.
    size_t klo = 0, khi = 0;
    gqa_key_window(qoff, Ktotl, win, 0, klo, khi);

    ScratchView buf(member.team_scratch(0),
                    gqa_decode_scratch_floats(Dloc, slots, KT));
    float* base = buf.data();
    const size_t qo = 0;                             // [D] scaled query row
    const size_t ao = Dloc;                          // [D] accumulator row
    const size_t wo = 2 * Dloc;                      // [ktile] scores/weights
    const size_t lo = wo + static_cast<size_t>(KT);  // [slots] partial (l)
    const size_t ko = lo + static_cast<size_t>(slots);  // [ktile][SS] K rows
    const size_t vo = ko + static_cast<size_t>(KT) * static_cast<size_t>(SS);

    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, Dloc), KOKKOS_LAMBDA(size_t d) {
          base[qo + d] = static_cast<float>(Q(0, h, d)) * scl;
          base[ao + d] = 0.0f;
        });
    for (int r = rank; r < slots; r += team_n) base[lo + r] = 0.0f;
    member.team_barrier();

    // Deferred-normalization state: m is the running max (the -1e30 sentinel
    // makes the first tile's rescale exp(-1e30 - m_new) == 0, as in the
    // streaming kernel), lp this thread's rescaled partial denominator.
    float m = -1e30f;
    float lp = 0.0f;
    for (size_t k0 = klo; k0 < khi; k0 += static_cast<size_t>(KT)) {
      const int kw = (k0 + static_cast<size_t>(KT) <= khi)
                         ? KT
                         : static_cast<int>(khi - k0);
      // Cooperative stage of this tile's K and V rows. The inner loop walks
      // the tile's keys so each thread keeps ~kw independent loads in flight
      // (the staged rows differ only in the key index), while consecutive
      // threads hold consecutive d, so every load instruction is one
      // coalesced row segment. Tail slots past kw are never read.
      for (size_t d = static_cast<size_t>(rank); d < Dloc;
           d += static_cast<size_t>(team_n)) {
        float* kd = base + ko + d;
        float* vd = base + vo + d;
        for (int kki = 0; kki < kw; ++kki) {
          const size_t kk = k0 + static_cast<size_t>(kki);
          kd[static_cast<size_t>(kki) * SS] = static_cast<float>(K(kk, kvh, d));
          vd[static_cast<size_t>(kki) * SS] = static_cast<float>(V(kk, kvh, d));
        }
      }
      member.team_barrier();
      // Scores: thread r owns keys r, r + team, ... (full-D dot from the
      // staged row; same eight-chain reduction as the streaming kernel).
      for (int kki = rank; kki < kw; kki += team_n) {
        const float* __restrict__ krow =
            base + ko + static_cast<size_t>(kki) * SS;
        const float* __restrict__ qrow = base + qo;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        size_t dd = 0;
        const size_t dend8 = Dloc & ~static_cast<size_t>(7);
        for (; dd < dend8; dd += 8) {
          a0 += qrow[dd] * krow[dd];
          a1 += qrow[dd + 1] * krow[dd + 1];
          a2 += qrow[dd + 2] * krow[dd + 2];
          a3 += qrow[dd + 3] * krow[dd + 3];
          a4 += qrow[dd + 4] * krow[dd + 4];
          a5 += qrow[dd + 5] * krow[dd + 5];
          a6 += qrow[dd + 6] * krow[dd + 6];
          a7 += qrow[dd + 7] * krow[dd + 7];
        }
        float stail = 0.0f;
        for (; dd < Dloc; ++dd) stail += qrow[dd] * krow[dd];
        base[wo + kki] =
            ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)) + stail;
      }
      member.team_barrier();
      // One team max per tile, folded into the running max.
      float tmax = -1e30f;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(member, static_cast<size_t>(kw)),
          KOKKOS_LAMBDA(size_t i, float& vmax) {
            if (base[wo + i] > vmax) vmax = base[wo + i];
          },
          Kokkos::Max<float>(tmax));
      const float mnew = (tmax > m) ? tmax : m;
      const float aa = expf(m - mnew);
      // Weights for this thread's keys, and its rescaled partial denominator.
      float lsum = 0.0f;
      for (int kki = rank; kki < kw; kki += team_n) {
        const float wt = expf(base[wo + kki] - mnew);
        base[wo + kki] = wt;
        lsum += wt;
      }
      lp = lp * aa + lsum;
      member.team_barrier();  // all weights visible to the accumulator pass
      // Accumulator row: thread owns dims rank, rank + team, ... Each tile's
      // share is summed in registers and folded into the shared row once, so
      // the FP32 output sum runs over Ktot/ktile tile sums instead of one
      // Ktot-long chain. Measured on the 8192-key reference data of
      // tests/Test_Attention (Attention.FloatLongKeys): worst error against
      // an FP64 reference is 4.7e-6 here vs 1.3e-4 with a per-key shared
      // accumulator. (That test's reference accumulates acc/denom in float
      // and is itself ~1.3e-4 off the FP64 answer at 8192 keys, so it only
      // passes kernels whose rounding happens to cancel its own; the kernel
      // being more accurate than the reference is not something to undo.)
      for (size_t d = static_cast<size_t>(rank); d < Dloc;
           d += static_cast<size_t>(team_n)) {
        float acc = 0.0f;
        for (int kki = 0; kki < kw; ++kki)
          acc += base[wo + kki] * base[vo + static_cast<size_t>(kki) * SS + d];
        base[ao + d] = base[ao + d] * aa + acc;
      }
      m = mnew;
      member.team_barrier();  // next tile overwrites the staged K/V rows
    }

    // Combine the per-thread partial denominators (once per call) and write
    // the normalized row. Rows with no contributing key keep zeros.
    base[lo + rank] = lp;
    member.team_barrier();
    float ltot = 0.0f;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, static_cast<size_t>(team_n)),
        KOKKOS_LAMBDA(int r, float& lsum) { lsum += base[lo + r]; }, ltot);
    const float inv = (ltot > 0.0f) ? 1.0f / ltot : 0.0f;
    for (size_t d = static_cast<size_t>(rank); d < Dloc;
         d += static_cast<size_t>(team_n))
      O(0, h, d) = static_cast<Scalar>(base[ao + d] * inv);
  }
};

// Single-token decode launcher: team size one thread per head dim (rounded
// up to a warp, clamped to what the backend can field, e.g. Serial -> 1) and
// a key-tile width that keeps the scratch within kMaxTeamScratchBytes.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void launch_gqa_decode(const QView& q, const KView& k, const VView& v,
                       size_t q_off, const OView& out, float scale,
                       size_t sliding_window, const std::string& label) {
  const size_t Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1), Ktot = k.extent(0);
  if (Hq == 0 || D == 0 || Ktot == 0) return;
  const int kss = static_cast<int>(D) | 1;
  // Slot count for the scratch-sizing team request: one thread per dim,
  // rounded up to a warp, capped so a head still fits a small team.
  int slots = (static_cast<int>(D) + 31) / 32 * 32;
  if (slots < 32) slots = 32;
  if (slots > 256) slots = 256;
  int ktile = 32;
  auto bytes = [&](int kt) {
    return gqa_decode_scratch_floats(D, slots, kt) * sizeof(float);
  };
  while (ktile > 1 && bytes(ktile) > kMaxTeamScratchBytes) ktile /= 2;

  GqaDecodeFunctor<ExecSpace, Scalar, QView, KView, VView, OView> f{
      q, k, v, out, Hq, D, Hkv, Ktot, q_off, sliding_window,
      ktile, kss, slots, scale};

  int team_size = slots;
  {
    Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
    probe.set_scratch_size(0, Kokkos::PerTeam(bytes(ktile)));
    const int mx = probe.team_size_max(f, Kokkos::ParallelForTag{});
    if (mx < team_size) team_size = mx;
    if (team_size < 1) team_size = 1;
  }

  Kokkos::TeamPolicy<ExecSpace> policy(
      static_cast<int>(Hq), team_size, 1);
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes(ktile)));
  Kokkos::parallel_for(label, policy, f);
}

// Streaming online-softmax GQA core shared by both public overloads below.
// See the public docs for contract; Scalar is the I/O dtype, accumulation
// is FP32, views are layout-agnostic (plain Views or subviews alike).
// sliding_window == 0 selects full causal attention; otherwise query t at
// absolute position p = q_off + t attends keys
// [max(0, p + 1 - sliding_window), p] (sliding-window causal mask).
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_gqa_streaming(const DNNHandle<ExecSpace>& handle, const QView& q,
                             const KView& k, const VView& v, size_t q_off,
                             const OView& out, float scale,
                             size_t sliding_window,
                             const std::string& label) {
  const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1);
  const size_t Ktot = k.extent(0);
  if (T == 0 || Hq == 0 || D == 0) return;

  // Single-token decode never uses a TPL attention kernel, and never the
  // multi-row streaming kernel either:
  //   * cuDNN SDPA bakes the KV length into the graph dims, and a decode
  //     step is a new KV length every token. Bottom-right causal alignment
  //     at T == 1 means "attend every key", so the extra keys of a padded
  //     (bucketed) KV view cannot be masked out and the plan genuinely
  //     cannot be reused -- the only options are an exact-length plan per
  //     token or the native kernel. Measured on the LFM2.5 decode shapes
  //     (T=1, Hq=32, Hkv=8, D=64, bf16): one plan rebuild is ~38 ms and a
  //     warm execute ~35 us, and because all attention layers of a token
  //     share the shape, that is exactly one rebuild per token (probe:
  //     sdpa_plan_cache_stats shows 1 miss + 9 hits per step). That single
  //     rebuild was ~2/3 of the TPL build's decode step.
  //   * the multi-row streaming kernel gives a T == 1 team one query row per
  //     thread and walks the whole key axis serially on 4 of 256 threads
  //     (~1.9 ms per layer at Ktot = 900).
  // GqaDecodeFunctor above is the T == 1 kernel for every backend: one team
  // per head with the key axis spread across the team, measured 81 us per
  // layer at the same shape (vs 1.9 ms) and 4.6 us at Ktot = 8, so a decode
  // token's whole attention drops from ~19 ms to ~0.8 ms at ctx 900.
  // Prefill (T > 1) keeps the cuDNN SDPA dispatch below.
  if (T == 1) {
    tpl_note_attn(false);
    launch_gqa_decode<ExecSpace, Scalar, QView, KView, VView, OView>(
        q, k, v, q_off, out, scale, sliding_window, label);
    return;
  }
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
    // Full causal (sliding_window == 0) uses the bottom-right causal cuDNN
    // SDPA graph; the windowed path uses the sliding cuDNN SDPA graph
    // (bottom-right alignment + left band bound == window). Either returning
    // false falls back to the native kernel below.
    if (handle.use_cudnn) {
      if (sliding_window == 0) {
        if (try_cudnn_sdpa_gqa<Scalar>(handle, q, k, v, q_off, out, scale)) {
          tpl_note_attn(true);
          return;
        }
      } else {
        if (try_cudnn_sdpa_gqa_sliding<Scalar>(handle, q, k, v, q_off, out,
                                               scale, sliding_window)) {
          tpl_note_attn(true);
          return;
        }
      }
    }
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
    // HIP/MIOpen branch (mirrors the cuDNN block above): MIOpen exposes
    // no fused SDPA (see tpl/miopen/KokkosDNN_MIOpen_Forward.hpp), so the
    // try_miopen_* entry points decline by construction and execution
    // falls through to the native kernel below. Wired for a future
    // hipBLAS + MIOpen-softmax composition path.
    constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
    if constexpr (is_hip) {
      if (handle.use_miopen) {
        if (sliding_window == 0) {
          if (try_miopen_sdpa_gqa<Scalar>(handle, q, k, v, q_off, out, scale))
            return;
        } else {
          if (try_miopen_sdpa_gqa_sliding<Scalar>(handle, q, k, v, q_off, out,
                                                  scale, sliding_window))
            return;
        }
      }
    }
  } else {
    (void)handle;
  }
#endif
  // One team per (token chunk, head chunk); see GqaStreamingFunctor for
  // the blocked online-softmax structure. No [T, Hq, K] score tensor is
  // materialized; extra memory is O(D) scratch per resident team plus O(1)
  // per (t, h) beyond the outputs, so 128k contexts stay feasible.
  using DummyMask = Kokkos::View<int*, ExecSpace>;
  tpl_note_attn(false);
  launch_gqa_streaming_auto<ExecSpace, Scalar, QView, KView, VView, DummyMask,
                            OView, false, false>(
      q, k, v, DummyMask(), out, T, Hq, D, Hkv, Ktot, q_off, scale, 0.0f,
      sliding_window, label);
}

}  // namespace Impl

// Grouped-query causal attention for one request (prefill or decode).
//   q     : [T, Hq, D]  queries (already RoPE'd)
//   k, v  : [Ktotal, Hkv, D] full KV cache prefix (RoPE'd keys)
//   out   : [T, Hq, D]
//   q_off : absolute position of q[0] inside the KV sequence (prefill
//           offset; 0 for full-prompt prefill, seq_len for decode step)
// Causal mask: query t attends to keys [0, q_off + t].
// GQA: head h uses kv head h * Hkv / Hq. Scale = 1/sqrt(D).
// Scalar-agnostic (float, bhalf_t); scores/softmax accumulate in FP32.
//
// Views are layout-agnostic (any Kokkos layout / subview / memory traits),
// so callers may pass a cache subview directly without repacking. Only
// extents and operator()(t/h/d) indexing are used.
//
// Memory: streaming online softmax (FlashAttention-style) over the key
// axis. No [T, Hq, K] score tensor is materialized; extra memory is O(D)
// team-scratch per resident team plus O(1) per (t, h) beyond the outputs.
// This is what makes 128k contexts feasible: a full-context prefill
// (T=K=128k, Hq=32) would need ~2 PiB as a dense FP32 score tensor, while
// the streaming kernel needs only the [T, Hq, D] output.
//
// Numerics: per-key streaming rescaling
//   m_new = max(m, s), l_new = l*exp(m-m_new) + exp(s-m_new),
//   O_new = (O*l*exp(m-m_new) + V*exp(s-m_new)) / l_new,
// which is algebraically identical to softmax(QK^T*scale)*V in FP32.
// Element order matches a left-to-right softmax reference up to FP32
// rounding (verified in tests/Test_Attention.cpp against a host reference).
//
// When KokkosDNN_ENABLE_CUDNN and ExecSpace is CUDA, float and bhalf_t
// prefill widths (T > 1) dispatch to cuDNN SDPA first (see
// tpl/cudnn/KokkosDNN_CuDNN.hpp for contract). Single-token decode (T == 1)
// always runs the native decode kernel (GqaDecodeFunctor): a cuDNN SDPA
// graph bakes the KV length into its dims, so every decode token would be a
// plan-cache miss/build, and with bottom-right causal alignment at T == 1
// every key is visible, so no padded/bucketed KV view can reuse a plan.
// One narrow, version-gated exception remains inside cudnn_sdpa_forward for
// direct callers: FP32 single-token GQA (T=1, Hq != Hkv) falls back to
// native on cuDNN <= 9.27.0, where the composite path computes only the
// first query head per KV group while reporting success. Other scalars run
// the native kernel. Otherwise native.
//
// Two overloads: explicit <ExecSpace, Scalar> (tests, Llama model) and a
// deduction overload taking Scalar from the query view (benchmarks). Both
// share the Impl core above so behavior cannot diverge.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_gqa_single(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa") {
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, 0, label);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_gqa_single(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, 0, label);
}

// Sliding-window causal GQA (e.g. Gemma4 sliding layers, window 512):
// same contract as attention_gqa_single, except query t at absolute
// position p = q_off + t attends keys [max(0, p + 1 - sliding_window), p]
// instead of the full prefix [0, p]. sliding_window == 0 degrades to full
// causal attention. GQA mapping, FP32 streaming numerics, and O(1) extra
// memory match the full-causal path. On CUDA builds with cuDNN enabled,
// float and bhalf_t prefill widths (T > 1) dispatch to cuDNN SDPA first
// (sliding graph with left band bound == window; falls back to native on
// failure), like the full-causal path; single-token decode (T == 1) runs
// the native decode kernel (same reason: no reusable SDPA plan per token);
// other scalars run the native Kokkos kernel.
//
// Two overloads: explicit <ExecSpace, Scalar> and a deduction overload
// taking Scalar from the query view. Both share the Impl core above so
// behavior cannot diverge from attention_gqa_single.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_gqa_single_sliding(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    size_t sliding_window,
    const std::string& label = "KokkosDNN::attn_gqa_sliding") {
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, sliding_window, label);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_gqa_single_sliding(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    size_t sliding_window,
    const std::string& label = "KokkosDNN::attn_gqa_sliding") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, sliding_window, label);
}

namespace Impl {

// Streaming online-softmax causal GQA core with Gemma2 tanh logit
// softcapping and a per-key validity mask. Same FP32 streaming structure as
// attention_gqa_streaming above, plus:
//   s = softcap * tanh((q.k * scale) / softcap)  [softcap <= 0 disables]
// and pad keys (key_valid == 0) are skipped entirely. Shared by both public
// overloads below so behavior cannot diverge.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class MView, class OView>
void attention_gqa_softcap_streaming(const DNNHandle<ExecSpace>& handle,
                                     const QView& q, const KView& k,
                                     const VView& v, const MView& key_valid,
                                     size_t q_off, const OView& out, float scale,
                                     float softcap, const std::string& label) {
  // No TPL dispatch: the pinned cudnn-frontend (GIT_TAG b8e39555...)
  // exposes no SDPA tanh-softcap attribute — its Attention.md score-mod
  // knobs are only diagonal bands, additive bias, block mask, ALiBi, and a
  // generic score_mod callback — and its padding mask is a per-batch
  // sequence-length truncation, which cannot express interior per-key
  // validity alongside pad query rows. Emulating the mask with a bias
  // tensor would materialize O(T*K) mask state and still leave the tanh
  // unfused, so the native kernel below is the only path (exact by
  // construction, no version-gated fallback to validate). MIOpen likewise
  // exposes no fused SDPA/softcap path (checked miopen.h 3.5.2: no
  // attention/flash/SDPA API), so no HIP/MIOpen dispatch is attempted
  // here either — native-only on every backend by construction.
  (void)handle;
  const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1);
  const size_t Ktot = k.extent(0);
  if (v.extent(0) != Ktot || out.extent(0) != T || out.extent(1) != Hq ||
      out.extent(2) != D || key_valid.extent(0) != Ktot)
    throw std::invalid_argument(
        "attention_gqa_single_softcap: dimension mismatch");
  if (T == 0 || Hq == 0 || D == 0) return;
  // Same team-parallel streaming core as the plain path (mask + tanh
  // enabled); see GqaStreamingFunctor. A query row with no valid key keeps
  // zeros via the l == 0 branch.
  launch_gqa_streaming_auto<ExecSpace, Scalar, QView, KView, VView, MView, OView,
                             true, true>(
      q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap, 0,
      label);
}

}  // namespace Impl

// Causal GQA attention with Gemma2 tanh logit softcapping and a key padding
// mask (e.g. the Sana Gemma2 text encoder: 8 Q / 4 KV heads, head_dim 256,
// scale = 256^-0.5, softcap 50).
//   q         : [T, Hq, D] queries (already RoPE'd)
//   k, v      : [Ktot, Hkv, D] keys / values
//   key_valid : [Ktot] per-key validity (1 = real token, 0 = pad)
//   out       : [T, Hq, D]
//   q_off     : absolute position of q[0] inside the K sequence; query t
//               attends valid keys [0, q_off + t]
//   s         : softcap * tanh((q.k * scale) / softcap); softcap <= 0
//               disables softcapping (pure scaled dot-product scores)
// Pad keys (valid == 0) are skipped, which equals the HF additive -inf mask
// whenever at least one key is valid. A query row with no valid key keeps
// zeros (documented; the Sana path always has key 0 valid, so this never
// triggers there). GQA mapping, FP32 streaming online-softmax numerics, and
// O(1) extra memory match attention_gqa_single; with q_off == 0,
// Ktot == T, and all keys valid the only difference from
// attention_gqa_single is the tanh (identity when softcap <= 0).
//
// Views are layout-agnostic (any Kokkos layout / subview / memory traits),
// like the rest of the attention family; key_valid is any rank-1 view whose
// elements are truthiness-testable (char/int/bool).
//
// TPL: none (see Impl note above) — the native kernel is the only path on
// every backend, so dispatched-vs-native agreement holds trivially.
//
// Two overloads: explicit <ExecSpace, Scalar> and a deduction overload
// taking Scalar from the query view. Both share the Impl core above so
// behavior cannot diverge.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class MView, class OView>
void attention_gqa_single_softcap(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const MView& key_valid, size_t q_off, const OView& out,
    float scale, float softcap,
    const std::string& label = "KokkosDNN::attn_gqa_softcap") {
  Impl::attention_gqa_softcap_streaming<ExecSpace, Scalar>(
      handle, q, k, v, key_valid, q_off, out, scale, softcap, label);
}

template <class ExecSpace, class QView, class KView, class VView, class MView,
          class OView>
void attention_gqa_single_softcap(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const MView& key_valid, size_t q_off, const OView& out,
    float scale, float softcap,
    const std::string& label = "KokkosDNN::attn_gqa_softcap") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_gqa_softcap_streaming<ExecSpace, Scalar>(
      handle, q, k, v, key_valid, q_off, out, scale, softcap, label);
}

namespace Impl {

// Inline single-precision exp for the bidirectional streaming update.
// Cody-Waite range reduction (k = round(x/ln2) with a 12-significant-bit
// c1, so k*c1 is exact) plus a degree-5 Horner polynomial and exact 2^k
// scaling via exponent bits. No function calls, no tables, no branches in
// the hot path — portable C++ callable from any backend. Host-verified max
// relative error vs expf is ~3.3e-6 over [-87, 0], far inside the attention
// test tolerances; args below -87 return 0 (true value is a subnormal <
// 1e-38, immaterial after normalization). Args above 0 never occur here
// (both call sites pass non-positive differences); the guard returns a
// large finite value instead of inf to keep downstream math NaN free if
// that contract ever changes.
KOKKOS_INLINE_FUNCTION float bidir_expf(float x) {
  const float c1 = 0.693359375f;
  const float c2 = -2.1219440717e-4f;
  const float invln2 = 1.4426950408889634f;
  const float k_real = x * invln2;
  const int k = static_cast<int>(k_real + ((k_real >= 0.0f) ? 0.5f : -0.5f));
  const float r = (x - static_cast<float>(k) * c1) - static_cast<float>(k) * c2;
  // Degree-5 Horner (~2.5e-6 max rel err on |r| <= ln2/2, verified on
  // host; far inside attention test tolerances when combined with the
  // exact 2^k scaling).
  const float p =
      1.0f +
      r * (1.0f +
           r * (0.5f +
                r * (0.1666666716337204f +
                     r * (0.0416666679084301f +
                          r * 0.0083333337679505f))));
  if (k < -126) return 0.0f;
  if (k > 127) return 1e30f;
  unsigned u = static_cast<unsigned>(k + 127) << 23;
  float s;
  memcpy(&s, &u, sizeof(float));
  return p * s;
}

// Single-kernel team-blocked bidirectional attention (native path for
// attention_mha_bidirectional). One team covers a tile of TC query tokens x
// HC heads with full-D rows; keys stream through in tiles of KT, with all
// reuse in team scratch:
//
//   - full Q rows ([nqr x SQ], loaded once per team) and full FP32
//     accumulator rows ([nqr x SQ] in scratch, or a per-thread register
//     array when DMAX > 0),
//   - K/V tiles ([nkv x KT x SD] each, cooperatively loaded once per K-tile
//     and reused across every row),
//   - per-row (m, l) online-softmax state in registers; every output row is
//     written exactly once.
//
// Row strides carry a +1 pad (plus one more when that lands on a multiple
// of 32) so same-element accesses across rows never bank-conflict. Cost per
// team is one Q-load plus #K-tiles x (one KV-load + nqr row updates) with
// one barrier per load phase — about two barriers per KT keys instead of
// per key. Dots use full-D rows with eight interleaved FP32 accumulators
// (portable C++, no intrinsics); the exp is an inlined Cody-Waite
// polynomial (~3e-6 vs expf, no function calls). Cooperative loads use
// vectorized 16-wide segments. FP32 accumulation throughout; the only
// Scalar rounding is the single final write. Team size is one thread per
// row (explicit, portable TeamPolicy API; the launcher probes
// team_size_max and clamps, so Serial runs team_size 1 through the same
// rank-strided loops); GQA leader mapping shares KV
// tile rows the way the causal kernel does. K-tail keys and T-tail rows
// are guarded; Ktot == 0 yields zeros per the documented contract.
//
// Numerics equal softmax(QK^T*scale)*V up to FP32 reassociation (verified
// against the host reference in tests/Test_AttentionBidirectional).
template <class ExecSpace, class QP, class KP, class VP, class OView, int HC,
          int DMAX>
struct BidirFlashFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using ScratchView =
      Kokkos::View<float*, typename ExecSpace::scratch_memory_space,
                   Kokkos::MemoryUnmanaged>;
  using Elem = std::remove_const_t<typename OView::value_type>;
  const QP* qbase;
  const KP* kbase;
  const VP* vbase;
  OView out;
  size_t T, Hq, D, Hkv, Ktot;
  size_t qs0, qs1;  // row(t,h) = base + t*s0 + h*s1 (D-contiguous rows)
  size_t ks0, ks1;
  size_t vs0, vs1;
  size_t os0, os1;  // out row(t,h) = obase + t*os0 + h*os1 (+d*os2)
  size_t os2;       // out D stride (out is never repacked: usually 1)
  Elem* obase;      // mutable out base (extracted on host; see launch)
  int tchunk;  // tokens per team (heads per team = HC, compile-time)
  int ktiles;  // keys per K-tile
  float scale;

  static KOKKOS_INLINE_FUNCTION int kv_of_head(size_t hbase, size_t Hkv,
                                               size_t Hq, int jj) {
    return static_cast<int>(((hbase + jj) * Hkv) / Hq);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const size_t Dloc = D;
    const size_t Hqloc = Hq;
    const size_t Tloc = T;
    const size_t Kloc = Ktot;
    const int TC = tchunk;
    const int KT = ktiles;
    const float scl = scale;
    const int rank = static_cast<int>(member.team_rank());
    const int team_n = static_cast<int>(member.team_size());

    const size_t nchunks_h = Hqloc / HC;
    const size_t chunk = static_cast<size_t>(member.league_rank());
    const size_t tbase = (chunk / nchunks_h) * TC;
    const size_t hbase = (chunk % nchunks_h) * HC;
    const size_t nqr = static_cast<size_t>(TC) * HC;

    // Distinct-KV-head plan for this head chunk (computed once per team).
    const size_t Hkbloc = Hkv;
    int ku_of_row[HC];
    int myrow_of_head[HC];
    int nkv = 0;
    for (int jj = 0; jj < HC; ++jj) {
      const int kvh = kv_of_head(hbase, Hkbloc, Hqloc, jj);
      if (jj == 0 || kvh != ku_of_row[nkv - 1]) {
        ku_of_row[nkv] = kvh;
        myrow_of_head[jj] = nkv;
        ++nkv;
      } else {
        myrow_of_head[jj] = nkv - 1;
      }
    }

    // Scratch holds full Q rows ([nqr x SQ]) plus the K/V tiles ([HC x
    // KT x SD] each; nkv <= HC rows used); with DMAX == 0 it additionally
    // holds full FP32 acc rows ([nqr x SQ]). Row strides get +1 pad, plus
    // one more when that lands on a multiple of 32, so same-element
    // accesses across rows never bank-conflict; SD is also at least DMAX
    // so fixed-DMAX register bodies stay in-bounds. Full rows (not
    // D-chunks): the online softmax needs complete dots before any exp,
    // so D is reduced inside the per-key dot, never across passes.
    size_t SQ = Dloc + 1;
    if (SQ % 32 == 0) ++SQ;
    size_t SD = Dloc + 1;
    if (SD % 32 == 0) ++SD;
    if (DMAX > 0 && SD < (size_t)DMAX) SD = DMAX;
    ScratchView buf(member.team_scratch(0),
                    nqr * SQ + (DMAX > 0 ? 0 : nqr * SQ) + 2 * HC * KT * SD);
    const size_t qoff = 0;
    const size_t aoff = nqr * SQ;
    const size_t koff = nqr * SQ + (DMAX > 0 ? 0 : nqr * SQ);
    const size_t voff = koff + HC * KT * SD;
    float* __restrict__ qb = buf.data() + qoff;
    float* accb = buf.data() + aoff;  // unused when DMAX > 0

    // Cooperative full-row Q load, 16-wide vectorized segments (one
    // (row, segment) unit per iteration over floor(D/16) segments) plus a
    // scalar tail for D % 16. Segment stores stay strictly inside the
    // [0, D) row range (never into padding or neighbors).
    const size_t QSEG = 16;
    const size_t qseg = Dloc / QSEG;
    const size_t qunits = nqr * qseg;
    for (size_t u = static_cast<size_t>(rank); u < qunits;
         u += static_cast<size_t>(team_n)) {
      const size_t row = u / qseg;
      const size_t s = u % qseg;
      const size_t t = tbase + row / HC;
      const size_t h = hbase + row % HC;
      float* __restrict__ qdst = qb + row * SQ + s * QSEG;
      if (t < Tloc) {
        const QP* qrowp = qbase + t * qs0 + h * qs1 + s * QSEG;
KOKKOSDNN_PRAGMA_UNROLL()
        for (int w = 0; w < 16; ++w)
          qdst[w] = static_cast<float>(qrowp[w]);
      } else {
KOKKOSDNN_PRAGMA_UNROLL()
        for (int w = 0; w < 16; ++w) qdst[w] = 0.0f;
      }
    }
    for (size_t dd = qseg * QSEG + rank; dd < Dloc;
         dd += static_cast<size_t>(team_n)) {
      for (size_t row = 0; row < nqr; ++row) {
        const size_t t = tbase + row / HC;
        const size_t h = hbase + row % HC;
        float x = 0.0f;
        if (t < Tloc) x = static_cast<float>(qbase[t * qs0 + h * qs1 + dd]);
        qb[row * SQ + dd] = x;
      }
    }
    member.team_barrier();
    // DMAX == 0 (generic path) zeroes the shared acc rows; the DMAX > 0
    // path uses a per-thread register array instead (zeroed below).
    if constexpr (DMAX == 0) {
      for (int rr = rank; rr < (int)nqr; rr += team_n) {
        float* __restrict__ accrow = accb + rr * SQ;
        for (size_t d = 0; d < Dloc; ++d) accrow[d] = 0.0f;
      }
      member.team_barrier();
    }

    for (int rr = rank; rr < (int)nqr; rr += team_n) {
      // Per-row streaming state lives in registers.
      float m = -1e30f;
      float l = 0.0f;
      const size_t t = tbase + rr / HC;
      const bool real = (t < Tloc);
      const int jj0 = real ? myrow_of_head[rr % HC] : 0;
      const float* __restrict__ qrow = qb + rr * SQ;
      float* __restrict__ accrow = accb + rr * SQ;
      float areg[DMAX > 0 ? DMAX : 1];
      if constexpr (DMAX > 0) {
        for (int dd = 0; dd < DMAX; ++dd) areg[dd] = 0.0f;
      }
      for (size_t k0 = 0; k0 < Kloc; k0 += KT) {
        // Cooperative K/V tile load, full D rows, 16-wide vectorized
        // segments (flat units, shift-free decode via division — a few
        // integer divisions per tile, negligible). Scalar tail for D % 16.
        const size_t SEG = 16;
        const size_t nseg = Dloc / SEG;
        const size_t per = (size_t)KT * nseg;
        const size_t nunits = (size_t)nkv * per;
        for (size_t u = static_cast<size_t>(rank); u < nunits;
             u += static_cast<size_t>(team_n)) {
          const size_t r = u / per;
          const size_t rem = u % per;
          const size_t kki = rem / nseg;
          const size_t s = rem % nseg;
          const size_t kh = static_cast<size_t>(ku_of_row[r]);
          const size_t kk = k0 + kki;
          const size_t dd = s * SEG;
          float* __restrict__ kdst = buf.data() + koff + (r * KT + kki) * SD + dd;
          float* __restrict__ vdst = buf.data() + voff + (r * KT + kki) * SD + dd;
          if (kk < Kloc) {
            const KP* krowp = kbase + kk * ks0 + kh * ks1 + dd;
            const VP* vrowp = vbase + kk * vs0 + kh * vs1 + dd;
            for (int w = 0; w < 16; ++w) {
              kdst[w] = static_cast<float>(krowp[w]);
              vdst[w] = static_cast<float>(vrowp[w]);
            }
          } else {
            for (int w = 0; w < 16; ++w) {
              kdst[w] = 0.0f;
              vdst[w] = 0.0f;
            }
          }
        }
        for (size_t dd = nseg * SEG + rank; dd < Dloc;
             dd += static_cast<size_t>(team_n)) {
          for (int r = 0; r < nkv; ++r) {
            const size_t kh = static_cast<size_t>(ku_of_row[r]);
            float* __restrict__ kdst = buf.data() + koff + r * KT * SD + dd;
            float* __restrict__ vdst = buf.data() + voff + r * KT * SD + dd;
            for (int kki = 0; kki < KT; ++kki) {
              const size_t kk = k0 + kki;
              if (kk < Kloc) {
                kdst[kki * SD] = static_cast<float>(
                    kbase[kk * ks0 + kh * ks1 + dd]);
                vdst[kki * SD] = static_cast<float>(
                    vbase[kk * vs0 + kh * vs1 + dd]);
              } else {
                kdst[kki * SD] = 0.0f;
                vdst[kki * SD] = 0.0f;
              }
            }
          }
        }
        member.team_barrier();
          if (real) {
            // Tile row bases hoisted out of the per-key loop.
            const float* __restrict__ kbase_r = buf.data() + koff + jj0 * KT * SD;
            const float* __restrict__ vbase_r = buf.data() + voff + jj0 * KT * SD;
KOKKOSDNN_PRAGMA_UNROLL_N(2)
            for (int kki = 0; kki < KT; ++kki) {
              const size_t kk = k0 + kki;
              if (kk >= Kloc) break;
              const float* __restrict__ krow = kbase_r + kki * SD;
              // Full-D dot with eight interleaved accumulators (D blocked
              // in octets for wide shared loads and independent chains).
              float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
              float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
              size_t dd = 0;
              const size_t dend8 = Dloc & ~size_t(7);
              for (; dd < dend8; dd += 8) {
                a0 += qrow[dd] * krow[dd];
                a1 += qrow[dd + 1] * krow[dd + 1];
                a2 += qrow[dd + 2] * krow[dd + 2];
                a3 += qrow[dd + 3] * krow[dd + 3];
                a4 += qrow[dd + 4] * krow[dd + 4];
                a5 += qrow[dd + 5] * krow[dd + 5];
                a6 += qrow[dd + 6] * krow[dd + 6];
                a7 += qrow[dd + 7] * krow[dd + 7];
              }
              float tail = 0.0f;
              for (; dd < Dloc; ++dd) tail += qrow[dd] * krow[dd];
              const float st =
                  (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)) + tail) *
                  scl;
            const float* __restrict__ vrow = vbase_r + kki * SD;
            // Single-exp streaming update with the inlined polynomial exp
            // (no function calls, so no caller-saved spill traffic). Only
            // one direction is evaluated; the other follows from
            // exp(st - m) == 1/exp(m - st) via one reciprocal (accurate
            // division, ~2e-7 combined error — far inside test
            // tolerances). The branch diverges only while the running
            // max is still settling (early keys) and runs uniformly
            // converged afterwards. m == -1e30 on the first key gives
            // aa = exp(-1e30) = 0, as desired (the unused reciprocal of
            // zero stays a finite-weighted select, never NaN).
            if (st > m) {
                const float aa = bidir_expf(m - st);
                const float l_new = l * aa + 1.0f;
                if constexpr (DMAX > 0) {
KOKKOSDNN_PRAGMA_UNROLL()
                  for (int e = 0; e < DMAX; e += 4) {
                    areg[e] = areg[e] * aa + vrow[e];
                    areg[e + 1] = areg[e + 1] * aa + vrow[e + 1];
                    areg[e + 2] = areg[e + 2] * aa + vrow[e + 2];
                    areg[e + 3] = areg[e + 3] * aa + vrow[e + 3];
                  }
                } else {
                  size_t dd2 = 0;
                  const size_t dend8_2 = Dloc & ~size_t(7);
                  for (; dd2 < dend8_2; dd2 += 8) {
                    accrow[dd2] = accrow[dd2] * aa + vrow[dd2];
                    accrow[dd2 + 1] = accrow[dd2 + 1] * aa + vrow[dd2 + 1];
                    accrow[dd2 + 2] = accrow[dd2 + 2] * aa + vrow[dd2 + 2];
                    accrow[dd2 + 3] = accrow[dd2 + 3] * aa + vrow[dd2 + 3];
                    accrow[dd2 + 4] = accrow[dd2 + 4] * aa + vrow[dd2 + 4];
                    accrow[dd2 + 5] = accrow[dd2 + 5] * aa + vrow[dd2 + 5];
                    accrow[dd2 + 6] = accrow[dd2 + 6] * aa + vrow[dd2 + 6];
                    accrow[dd2 + 7] = accrow[dd2 + 7] * aa + vrow[dd2 + 7];
                  }
                  for (; dd2 < Dloc; ++dd2)
                    accrow[dd2] = accrow[dd2] * aa + vrow[dd2];
                }
                m = st;
                l = l_new;
              } else {
                // exp(st - m) == 1/exp(m - st): reuse via one reciprocal
                // (m - st >= 0 here, so e_big is finite-positive; when it
                // saturates to 1e30 the reciprocal correctly rounds to ~0).
                const float gg = 1.0f / bidir_expf(m - st);
                const float l_new = l + gg;
                if constexpr (DMAX > 0) {
KOKKOSDNN_PRAGMA_UNROLL()
                  for (int e = 0; e < DMAX; e += 4) {
                    areg[e] += vrow[e] * gg;
                    areg[e + 1] += vrow[e + 1] * gg;
                    areg[e + 2] += vrow[e + 2] * gg;
                    areg[e + 3] += vrow[e + 3] * gg;
                  }
                } else {
                  size_t dd2 = 0;
                  const size_t dend8_2 = Dloc & ~size_t(7);
                  for (; dd2 < dend8_2; dd2 += 8) {
                    accrow[dd2] += vrow[dd2] * gg;
                    accrow[dd2 + 1] += vrow[dd2 + 1] * gg;
                    accrow[dd2 + 2] += vrow[dd2 + 2] * gg;
                    accrow[dd2 + 3] += vrow[dd2 + 3] * gg;
                    accrow[dd2 + 4] += vrow[dd2 + 4] * gg;
                    accrow[dd2 + 5] += vrow[dd2 + 5] * gg;
                    accrow[dd2 + 6] += vrow[dd2 + 6] * gg;
                    accrow[dd2 + 7] += vrow[dd2 + 7] * gg;
                  }
                  for (; dd2 < Dloc; ++dd2) accrow[dd2] += vrow[dd2] * gg;
                }
                l = l_new;
              }
            }
          }
        member.team_barrier();
      }
      // Write back full rows (l == 0, i.e. Ktot == 0, keeps zeros).
      // Raw out row pointer: one mult-add per row instead of a full 3-D
      // index computation per element.
      if (real) {
        const size_t hwr = hbase + rr % HC;
        Elem* __restrict__ orow = obase + t * os0 + hwr * os1;
        const size_t ostr = os2;
        if (l > 0.0f) {
          const float inv_l = 1.0f / l;
          if constexpr (DMAX > 0) {
KOKKOSDNN_PRAGMA_UNROLL()
            for (int e = 0; e < DMAX; e += 4) {
              const float w0 = areg[e] * inv_l;
              const float w1 = areg[e + 1] * inv_l;
              const float w2 = areg[e + 2] * inv_l;
              const float w3 = areg[e + 3] * inv_l;
              // Dloc-masked stores: tail lanes never escape.
              if ((size_t)e < Dloc) orow[e * ostr] = static_cast<Elem>(w0);
              if ((size_t)(e + 1) < Dloc)
                orow[(e + 1) * ostr] = static_cast<Elem>(w1);
              if ((size_t)(e + 2) < Dloc)
                orow[(e + 2) * ostr] = static_cast<Elem>(w2);
              if ((size_t)(e + 3) < Dloc)
                orow[(e + 3) * ostr] = static_cast<Elem>(w3);
            }
          } else {
            size_t dd = 0;
            const size_t dend8 = Dloc & ~size_t(7);
            for (; dd < dend8; dd += 8) {
              orow[dd * ostr] = static_cast<Elem>(accrow[dd] * inv_l);
              orow[(dd + 1) * ostr] =
                  static_cast<Elem>(accrow[dd + 1] * inv_l);
              orow[(dd + 2) * ostr] =
                  static_cast<Elem>(accrow[dd + 2] * inv_l);
              orow[(dd + 3) * ostr] =
                  static_cast<Elem>(accrow[dd + 3] * inv_l);
              orow[(dd + 4) * ostr] =
                  static_cast<Elem>(accrow[dd + 4] * inv_l);
              orow[(dd + 5) * ostr] =
                  static_cast<Elem>(accrow[dd + 5] * inv_l);
              orow[(dd + 6) * ostr] =
                  static_cast<Elem>(accrow[dd + 6] * inv_l);
              orow[(dd + 7) * ostr] =
                  static_cast<Elem>(accrow[dd + 7] * inv_l);
            }
            for (; dd < Dloc; ++dd)
              orow[dd * ostr] = static_cast<Elem>(accrow[dd] * inv_l);
          }
        } else {
          for (size_t dd = 0; dd < Dloc; ++dd)
            orow[dd * ostr] = static_cast<Elem>(0);
        }
      }
    }
    member.team_barrier();
  }
};

// Naive obviously-correct fallback for backends that cannot field even a
// single-thread team with the flash kernel's scratch (team_size_max < 1):
// per-(t, h) scalar two-pass softmax over ALL keys in FP32, then normalize.
// Only the fallback path runs this; team-capable backends always run the
// flash kernel above. D-contiguous rows (the only layout the flat launcher
// passes here) stream elementwise through the raw pointers; out is written
// through its own strides. Ktot == 0 yields zeros, matching the flash path.
template <class ExecSpace, class QP, class KP, class VP, class OView>
void launch_bidir_naive(const QP* qbase, const KP* kbase, const VP* vbase,
                        const OView& out, size_t T, size_t Hq, size_t D,
                        size_t Hkv, size_t Ktot, size_t qs0, size_t qs1,
                        size_t ks0, size_t ks1, size_t vs0, size_t vs1,
                        size_t os0, size_t os1, size_t os2,
                        typename std::remove_const_t<
                            typename OView::value_type>* obase,
                        float scale, const std::string& label) {
  using Elem = std::remove_const_t<typename OView::value_type>;
  if (T == 0 || Hq == 0 || D == 0) return;
  (void)out;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> row_m(
      Kokkos::ViewAllocateWithoutInitializing("KokkosDNN::bidir_naive_m"), T,
      Hq);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> acc(
      Kokkos::ViewAllocateWithoutInitializing("KokkosDNN::bidir_naive_acc"), T,
      Hq, D);
  // Pass 1: per-row max score (+ zero the accumulator row).
  Kokkos::parallel_for(
      label + "_naive_max",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const size_t kvh = (h * Hkv) / Hq;
        float m = -1e30f;
        for (size_t kk = 0; kk < Ktot; ++kk) {
          float s = 0.0f;
          for (size_t d = 0; d < D; ++d)
            s += static_cast<float>(qbase[t * qs0 + h * qs1 + d]) *
                 static_cast<float>(kbase[kk * ks0 + kvh * ks1 + d]);
          s *= scale;
          if (s > m) m = s;
        }
        row_m(t, h) = m;
        for (size_t d = 0; d < D; ++d) acc(t, h, d) = 0.0f;
      });
  // Pass 2: weights from the row max, FP32 accumulation, normalize on write.
  Kokkos::parallel_for(
      label + "_naive_update",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const size_t kvh = (h * Hkv) / Hq;
        Elem* orow = obase + t * os0 + h * os1;
        const float m = row_m(t, h);
        if (Ktot == 0 || m <= -1e29f) {
          for (size_t d = 0; d < D; ++d) orow[d * os2] = static_cast<Elem>(0);
          return;
        }
        float l = 0.0f;
        for (size_t kk = 0; kk < Ktot; ++kk) {
          float s = 0.0f;
          for (size_t d = 0; d < D; ++d)
            s += static_cast<float>(qbase[t * qs0 + h * qs1 + d]) *
                 static_cast<float>(kbase[kk * ks0 + kvh * ks1 + d]);
          s *= scale;
          const float w = expf(s - m);
          l += w;
          for (size_t d = 0; d < D; ++d)
            acc(t, h, d) +=
                w * static_cast<float>(vbase[kk * vs0 + kvh * vs1 + d]);
        }
        if (l > 0.0f) {
          const float inv = 1.0f / l;
          for (size_t d = 0; d < D; ++d)
            orow[d * os2] = static_cast<Elem>(acc(t, h, d) * inv);
        } else {
          for (size_t d = 0; d < D; ++d) orow[d * os2] = static_cast<Elem>(0);
        }
      });
}

template <class ExecSpace, class QP, class KP, class VP, class OView, int HC,
          int DMAX>
void launch_bidir_flash(const QP* qbase, const KP* kbase, const VP* vbase,
                        const OView& out, size_t T, size_t Hq, size_t D,
                        size_t Hkv, size_t Ktot, size_t qs0, size_t qs1,
                        size_t ks0, size_t ks1, size_t vs0, size_t vs1,
                        float scale, const std::string& label) {
  // Token chunks (rows = TC x HC) with full-D Q rows plus K/V tiles in
  // scratch (acc lives in registers for D <= 64). TC favors fatter tiles
  // on long sequences (fewer teams, more K/V reuse per tile load) and more
  // teams on short ones (fill the machine): T >= 320 uses 24-token chunks,
  // else 16. KT starts at 12 and shrinks (gradually, then by halves) with
  // TC so scratch stays within the portable per-team budget
  // (kMaxTeamScratchBytes).
  int tchunk = (T < 16) ? static_cast<int>(T) : ((T >= 320) ? 24 : 16);
  int kt = 12;
  // Scratch mirror of the functor layout (Q rows + optional acc rows +
  // K/V tiles); must match exactly. DMAX > 0 drops the shared acc rows.
  // KT shrinks gradually (then by halves) and TC by halves so scratch
  // stays within the portable per-team budget (kMaxTeamScratchBytes).
  auto bidir_scratch_floats = [&](int tc, int kkt) -> size_t {
    size_t sq = D + 1;
    if (sq % 32 == 0) ++sq;
    size_t sd = D + 1;
    if (sd % 32 == 0) ++sd;
    constexpr bool reg = (DMAX > 0);
    if (reg && sd < 64) sd = 64;
    const size_t nq = (size_t)tc * HC;
    return nq * sq + (reg ? 0 : nq * sq) + 2 * HC * kkt * sd;
  };
  while (kt > 8 && bidir_scratch_floats(tchunk, kt) * sizeof(float) >
                        kMaxTeamScratchBytes)
    --kt;
  while (kt > 1 && bidir_scratch_floats(tchunk, kt) * sizeof(float) >
                        kMaxTeamScratchBytes)
    kt /= 2;
  while (tchunk > 1 && bidir_scratch_floats(tchunk, kt) * sizeof(float) >
                            kMaxTeamScratchBytes)
    tchunk /= 2;
  // Mutable out base via a non-const handle copy (avoids a const_cast; the
  // caller's allocation is mutable — out is only ever written here).
  OView out_nc = out;
  using Elem = std::remove_const_t<typename OView::value_type>;
  Elem* obase = out_nc.data();
  const size_t os0 = out_nc.stride(0), os1 = out_nc.stride(1);
  const size_t os2 = out_nc.stride(2);
  BidirFlashFunctor<ExecSpace, QP, KP, VP, OView, HC, DMAX> f{
      qbase, kbase, vbase, out, T, Hq, D, Hkv, Ktot, qs0, qs1, ks0,
      ks1,   vs0,   vs1,   os0, os1, os2, obase, tchunk,  kt,  scale};
  const size_t nqr = static_cast<size_t>(tchunk) * HC;
  const size_t nchunks_t = (T + tchunk - 1) / tchunk;
  // Scratch footprint of the functor layout above (Q rows + optional acc
  // rows + K/V tiles); must match exactly. DMAX > 0 drops the shared acc
  // rows (register array instead). Computed once here so the probe and the
  // launch request identical scratch.
  size_t sqq = D + 1;
  if (sqq % 32 == 0) ++sqq;
  size_t sdd = D + 1;
  if (sdd % 32 == 0) ++sdd;
  if (DMAX > 0 && sdd < (size_t)DMAX) sdd = DMAX;
  const size_t shBytes =
      (nqr * sqq + (DMAX > 0 ? 0 : nqr * sqq) + 2 * HC * kt * sdd) *
      sizeof(float);
  // Team-capability probe FIRST: constructing the TeamPolicy below aborts
  // on backends that cannot field the team (e.g. Serial, max team size 1),
  // so probe max team size from a team_size-1 policy (valid everywhere) and
  // take the obviously-correct naive kernel when even a single-thread team
  // does not fit. The query accounts for functor registers and the per-team
  // scratch.
  {
    Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
    probe.set_scratch_size(0, Kokkos::PerTeam(shBytes));
    const int max_team = probe.team_size_max(f, Kokkos::ParallelForTag{});
    if (max_team < 1) {
      launch_bidir_naive<ExecSpace>(qbase, kbase, vbase, out, T, Hq, D, Hkv,
                                    Ktot, qs0, qs1, ks0, ks1, vs0, vs1, os0,
                                    os1, os2, obase, scale, label);
      return;
    }
    // One thread per row: full warps stay lockstep-simple and barriers stay
    // single-warp cheap. Clamped at runtime to what the backend can field:
    // every loop strides on the runtime team size, so the clamped size stays
    // correct (Serial runs team_size 1); team-capable backends keep nqr
    // unchanged, so the fast path is identical.
    const int team = std::min(static_cast<int>(nqr), max_team);
    Kokkos::TeamPolicy<ExecSpace> policy(nchunks_t * (Hq / HC), team, 1);
    policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));
    Kokkos::parallel_for(label, policy, f);
  }
}

template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void launch_bidir_flat(const QView& q, const KView& k, const VView& v,
                       const OView& out, size_t T, size_t Hq, size_t D,
                       size_t Hkv, size_t Ktot, float scale,
                       const std::string& label) {
  using QE = std::remove_const_t<typename QView::value_type>;
  using KE = std::remove_const_t<typename KView::value_type>;
  using VE = std::remove_const_t<typename VView::value_type>;
  const QE* qb = q.data();
  const KE* kb = k.data();
  const VE* vb = v.data();
  const size_t qs0 = q.stride(0), qs1 = q.stride(1);
  const size_t ks0 = k.stride(0), ks1 = k.stride(1);
  const size_t vs0 = v.stride(0), vs1 = v.stride(1);
  // D <= 64 uses the register-accumulator instantiation (fixed-64 bodies,
  // no local-memory spill); larger head dims fall back to the shared-acc
  // instantiation (always correct, slower).
  const bool use_reg = (D <= 64);
  if ((Hq % 4) == 0) {
    if (use_reg) {
      launch_bidir_flash<ExecSpace, QE, KE, VE, OView, 4, 64>(
          qb, kb, vb, out, T, Hq, D, Hkv, Ktot, qs0, qs1, ks0, ks1, vs0,
          vs1, scale, label);
    } else {
      launch_bidir_flash<ExecSpace, QE, KE, VE, OView, 4, 0>(
          qb, kb, vb, out, T, Hq, D, Hkv, Ktot, qs0, qs1, ks0, ks1, vs0,
          vs1, scale, label);
    }
  } else {
    if (use_reg) {
      launch_bidir_flash<ExecSpace, QE, KE, VE, OView, 1, 64>(
          qb, kb, vb, out, T, Hq, D, Hkv, Ktot, qs0, qs1, ks0, ks1, vs0,
          vs1, scale, label);
    } else {
      launch_bidir_flash<ExecSpace, QE, KE, VE, OView, 1, 0>(
          qb, kb, vb, out, T, Hq, D, Hkv, Ktot, qs0, qs1, ks0, ks1, vs0,
          vs1, scale, label);
    }
  }
}

// D-contiguous fast path gate. The kernel below streams whole rows
// elementwise through raw pointers, so a D stride != 1 (e.g. Kokkos's
// default LayoutLeft on device, where consecutive D elements sit T*H apart
// and every access wastes a full cache sector) collapses effective bandwidth
// ~16x. Repack strided inputs once into LayoutRight temporaries;
// already-contiguous inputs run in place with zero copies. `out` is never
// repacked: every path writes each output element exactly once, so its
// strided cost is one pass regardless.
template <class View>
inline bool bidir_is_d_contig(const View& w) {
  return w.span_is_contiguous() && w.stride(2) == 1;
}

template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void launch_bidir_streaming_auto(const QView& q, const KView& k, const VView& v,
                                 const OView& out, size_t T, size_t Hq,
                                 size_t D, size_t Hkv, size_t Ktot, float scale,
                                 const std::string& label) {
  if (bidir_is_d_contig(q) && bidir_is_d_contig(k) && bidir_is_d_contig(v)) {
    launch_bidir_flat<ExecSpace, Scalar>(q, k, v, out, T, Hq, D, Hkv, Ktot,
                                         scale, label);
    return;
  }
  // Strided inputs are repacked into LayoutRight temporaries (with a
  // transposing copy) so the main kernel streams D-contiguous rows.
  // WithoutInitializing: every element is overwritten by the repack copy
  // immediately below.
  using QE = std::remove_const_t<typename QView::value_type>;
  using KE = std::remove_const_t<typename KView::value_type>;
  using VE = std::remove_const_t<typename VView::value_type>;
  Kokkos::View<QE***, Kokkos::LayoutRight, ExecSpace> qc(
      Kokkos::ViewAllocateWithoutInitializing("KokkosDNN::bidir_q"), T, Hq,
      D);
  Kokkos::View<KE***, Kokkos::LayoutRight, ExecSpace> kc(
      Kokkos::ViewAllocateWithoutInitializing("KokkosDNN::bidir_k"), Ktot,
      Hkv, D);
  Kokkos::View<VE***, Kokkos::LayoutRight, ExecSpace> vc(
      Kokkos::ViewAllocateWithoutInitializing("KokkosDNN::bidir_v"), Ktot,
      Hkv, D);
  Kokkos::deep_copy(qc, q);
  Kokkos::deep_copy(kc, k);
  Kokkos::deep_copy(vc, v);
  launch_bidir_flat<ExecSpace, Scalar>(qc, kc, vc, out, T, Hq, D, Hkv, Ktot,
                                       scale, label);
}

// Streaming online-softmax bidirectional core shared by both public
// overloads below. Same structure as attention_gqa_streaming above, except
// every query attends to ALL keys (no causal truncation).
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
// Optional caller-provided workspace for the single-head FP32 TPL math
// fallback (see try_cudnn_sdpa_bidir): attn_ws[0, attn_ws_elems) replaces
// that fallback's per-call [T,K] scores allocation when sized
// >= (size_t)T * (size_t)K. Defaulted so all existing callers and both
// attention_mha_bidirectional overloads below are unaffected.
void attention_bidirectional_streaming(const DNNHandle<ExecSpace>& handle,
                                       const QView& q, const KView& k,
                                       const VView& v, const OView& out,
                                       float scale, const std::string& label,
                                       float* attn_ws = nullptr,
                                       size_t attn_ws_elems = 0) {
  const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1);
  const size_t Ktot = k.extent(0);
  if (T == 0 || Hq == 0 || D == 0) return;
  // scale <= 0 selects the ViT default 1/sqrt(head_dim); see public docs.
  const float eff =
      (scale > 0.0f) ? scale : 1.0f / std::sqrt(static_cast<float>(D));

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
    if (handle.use_cudnn &&
        try_cudnn_sdpa_bidir<Scalar>(handle, q, k, v, out, eff, attn_ws,
                                    attn_ws_elems))
      return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
  // attn_ws/attn_ws_elems feed only the cuDNN SDPA path above.
  (void)attn_ws;
  (void)attn_ws_elems;
#endif
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
    // HIP/MIOpen branch (mirrors the cuDNN block above): MIOpen exposes
    // no fused SDPA (see tpl/miopen/KokkosDNN_MIOpen_Forward.hpp), so
    // try_miopen_sdpa_bidir declines by construction and execution falls
    // through to the native kernel below. Wired for a future hipBLAS +
    // MIOpen-softmax composition path.
    constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
    if constexpr (is_hip) {
      if (handle.use_miopen &&
          try_miopen_sdpa_bidir<Scalar>(handle, q, k, v, out, eff))
        return;
    }
  } else {
    (void)handle;
  }
#endif

  // Native path: single-kernel team-blocked FlashAttention-style kernel
  // (Q/acc chunks + K/V tiles in team scratch, one normalization at the end
  // per D-chunk); see BidirFlashFunctor for structure. Transient extras are
  // O(D) team-scratch per resident team plus O(1) per (t, h) beyond the
  // outputs — no score tensor is materialized, so long contexts stay
  // feasible. Since every query attends every key, the dense key tiles give
  // every team independent, uniformly-shaped work.
  launch_bidir_streaming_auto<ExecSpace, Scalar>(q, k, v, out, T, Hq, D, Hkv,
                                                 Ktot, eff, label);
}

}  // namespace Impl

// Bidirectional (non-causal) multi-head self/cross attention for inference.
// ViT/DINOv3 encoder blocks attend over the full token sequence
// (1 CLS + 4 registers + Np patches, e.g. 201 @224px); there is no
// causality, no KV-cache, and no sampling, so this replaces the causal
// attention_gqa_single for that use case. The causal API is untouched.
//
//   q     : [T, Hq, Dh] queries (already projected, optionally RoPE'd)
//   k, v  : [K, Hkv, Dh] keys / values (K == T for ViT self-attention)
//   out   : [T, Hq, Dh]
// Bidirectional mask: every query t attends to ALL keys [0, K).
// GQA-compatible: head h uses kv head h * Hkv / Hq (== h for pure MHA such
// as ViT-B/16 with Hq == Hkv == 12, Dh == 64).
//
// Layout convention matches attention_gqa_single ([T, H, Dh] triples,
// layout-agnostic views/subviews; only extents and operator()(t/h/d) are
// used). Callers holding packed [T, D] activations with D == H*Dh may pass
// an unmanaged reinterpreted [T, H, Dh] view instead of repacking. There is
// no bias/mask parameter: ViT attention is dense with no additive bias.
// Scalar-agnostic (float, bhalf_t); scores/softmax accumulate in FP32 with
// the same per-key streaming rescaling numerics as the causal kernel.
//
// scale : softmax scale; pass <= 0 (the default) for 1/sqrt(Dh).
//
// TPL: float and bhalf_t dispatch to the non-causal cuDNN SDPA graph
// first (see tpl/cudnn/); single-head float shapes past the graph's
// head-dim engine limit (D > 128, e.g. the VAE mid-block) take the TPL
// math composition there instead. Any failure falls back to the native
// kernel. Otherwise native.
//
// Two overloads: explicit <ExecSpace, Scalar> and a deduction overload
// taking Scalar from the query view. Both share the Impl core above so
// behavior cannot diverge.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const OView& out, float scale = 0.0f,
    const std::string& label = "KokkosDNN::attn_bidir",
    float* attn_ws = nullptr, size_t attn_ws_elems = 0) {
  Impl::attention_bidirectional_streaming<ExecSpace, Scalar>(
      handle, q, k, v, out, scale, label, attn_ws, attn_ws_elems);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const OView& out, float scale = 0.0f,
    const std::string& label = "KokkosDNN::attn_bidir",
    float* attn_ws = nullptr, size_t attn_ws_elems = 0) {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_bidirectional_streaming<ExecSpace, Scalar>(
      handle, q, k, v, out, scale, label, attn_ws, attn_ws_elems);
}

}  // namespace KokkosDNN

#endif
