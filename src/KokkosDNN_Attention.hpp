#ifndef KOKKOSDNN_ATTENTION_HPP
#define KOKKOSDNN_ATTENTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#ifdef KOKKOSDNN_ENABLE_CUDNN
#include "tpl/cudnn/KokkosDNN_CuDNN.hpp"
#endif

namespace KokkosDNN {
namespace Impl {

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
// Per-team team-scratch holds the scaled query rows (loaded once, scale
// fused), the FP32 output accumulators (keys stream through fast scratch;
// each output row is written once at the end), the distinct K/V rows of
// the current key, and the per-row (m, l) streaming state. Row loads use
// the whole team (TeamVectorRange), which coalesces for D-contiguous
// (LayoutRight) inputs and still issues D-parallel loads otherwise; all
// per-row work is serial on the owning team thread, so no vector-lane
// coherence hazards exist by construction.
//
// Math is the per-key deferred-normalization streaming rescaling:
//   m_new = max(m, s), l_new = l*exp(m-m_new) + exp(s-m_new),
//   acc_new = acc*exp(m-m_new) + V*exp(s-m_new),
// with a single acc/l write per row at the end.
// The final result equals softmax(QK^T*scale)*V up to FP32 rounding; dot and
// update order over D is serial-blocked, so results agree with a
// sequential reference up to ~1-ulp reassociation, exactly as between any
// two parallel schedules. FP32 accumulation throughout; the only Scalar
// rounding is the single final write. UF blocks the D axis (each iteration
// handles UF consecutive D) so the compiler can issue wide loads/stores
// for D-contiguous inputs and interleave independent chains; UF == 1 keeps
// the plain scalar order.
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
  float scale, softcap;  // softcap unused unless UseSoftcap

  // KV head of head (hbase + jj): plain integer arithmetic, callable from
  // team-body scalar code on any backend.
  static KOKKOS_INLINE_FUNCTION int kv_of_head(size_t hbase, size_t Hkv,
                                               size_t Hq, int jj) {
    return static_cast<int>(((hbase + jj) * Hkv) / Hq);
  }
  // Causal key-window ends (half-open [klo, khi)) for token t.
  static KOKKOS_INLINE_FUNCTION void key_window(size_t q_off, size_t Ktot,
                                                size_t sliding_window,
                                                size_t t, size_t& klo,
                                                size_t& khi) {
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

    // Scratch layout: Q rows, acc rows (TC*HC each, stride SS = D + 1),
    // one K row + one V row per distinct KV head (<= HC), then per-row
    // (m, l) state. The +1 pad makes the row stride coprime to the 32
    // shared-memory banks, so the team threads touching the same d in
    // different rows hit distinct banks instead of serializing 32-way.
    const size_t nqr = static_cast<size_t>(TC) * HC;  // query rows per team
    const size_t SS = Dloc + 1;
    ScratchView buf(member.team_scratch(0),
                    (2 * nqr + 2 * HC) * SS + 2 * nqr);
    const size_t qoff = 0;
    const size_t aoff = nqr * SS;
    const size_t koff = 2 * nqr * SS;
    const size_t voff = (2 * nqr + HC) * SS;
    const size_t moff = (2 * nqr + 2 * HC) * SS;
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

    for (size_t kk = klo_u; kk < khi_u; ++kk) {
      if constexpr (UseMask) {
        if (!Valid(kk)) continue;
      }
      // Load this key's distinct K/V rows once per team (one row per
      // leader head; row->kv-head mapping hoisted above). Team-wide
      // collective: every team thread participates.
      for (int r = 0; r < nrows; ++r) {
        const size_t ku = static_cast<size_t>(ku_of_row[r]);
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, nbu), KOKKOS_LAMBDA(size_t m) {
              const size_t b = m * UF;
              for (int u = 0; u < UF; ++u) {
                const size_t d = b + u;
                buf(koff + r * SS + d) = static_cast<float>(K(kk, ku, d));
                buf(voff + r * SS + d) = static_cast<float>(V(kk, ku, d));
              }
            });
      }
      // Cross-warp visibility for the freshly loaded rows before any
      // thread's warp consumes them in the row loop below.
      member.team_barrier();
      // Per-row streaming update, serial on the owning team thread. Each
      // row is owned by exactly one team thread; rows of tokens outside
      // this key's window sit out (their state stays put, so no NaN can
      // poison it). With vector_length == 1 there are no lanes: the dot,
      // the scalars, and the update are plain serial code with no
      // coherence hazards, and skipping rows is always safe.
      for (int rr = rank; rr < nqr; rr += team_n) {
        const size_t t = tbase + rr / HC;
        if (t >= Tloc) continue;  // dummy row of a partial edge chunk
        size_t klo_t = 0, khi_t = 0;
        key_window(q_off, Ktot, sliding_window, t, klo_t, khi_t);
        if (kk < klo_t || kk >= khi_t) continue;
        const int j = rr % HC;
        const int myrow = myrow_of_head[j];
        const size_t myoff = rr;
        const float m = buf(moff + rr);
        const float l = buf(loff + rr);
        // Raw restrict pointers so the compiler can pipeline/vectorize the
        // serial streams (View operator() on a shared base otherwise
        // inhibits aliasing assumptions). Portable C++, no intrinsics.
        const float* __restrict__ qrow = buf.data() + qoff + myoff * SS;
        const float* __restrict__ krow = buf.data() + koff + myrow * SS;
        float* __restrict__ accrow = buf.data() + aoff + myoff * SS;
        const float* __restrict__ vrow = buf.data() + voff + myrow * SS;
        float s = 0.0f;
        for (size_t b = 0; b < nbu; ++b) {
          const size_t bb = b * UF;
          for (int u = 0; u < UF; ++u) s += qrow[bb + u] * krow[bb + u];
        }
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
          buf(moff + rr) = ss;
          buf(loff + rr) = l_new;
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
          buf(loff + rr) = l_new;
        }
      }
      // End-of-key barrier: the next key's row loads overwrite the K/V
      // rows this key's update just consumed; without it a fast warp can
      // clobber rows a slower warp is still accumulating from. (The mask
      // continue above skips uniformly, so all threads reach this barrier
      // together.)
      member.team_barrier();
    }
    // Write back: rows with no contributing key (l == 0) keep zeros,
    // matching the previous kernel's zero-init contract.
    for (int rr = rank; rr < nqr; rr += team_n) {
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
// each K/V row once and reuse it across 32 query rows (standard GQA shares
// one KV head per 4 heads). Smaller chunks cover odd head counts, small T,
// and large D (scratch fit). All per-row work is serial (no vector lanes
// exist), so the dot/update order is fixed and identical across the plain
// and softcap instantiations (and backends) for a given D; UF == 4
// additionally groups D into consecutive quadruples for wide loads.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class MView, class OView, int UF, int HC, bool UseMask,
          bool UseSoftcap>
void launch_gqa_streaming(const QView& q, const KView& k, const VView& v,
                          const MView& key_valid, const OView& out, size_t T,
                          size_t Hq, size_t D, size_t Hkv, size_t Ktot,
                          size_t q_off, float scale, float softcap,
                          size_t sliding_window, int tchunk,
                          const std::string& label) {
  GqaStreamingFunctor<ExecSpace, Scalar, QView, KView, VView, MView, OView, UF,
                      HC, UseMask, UseSoftcap>
      f{q, k,      v,       key_valid, out, T, Hq, D, Hkv, Ktot,
        q_off, sliding_window, tchunk, scale, softcap};
  const size_t nchunks_t = (T + tchunk - 1) / tchunk;
  Kokkos::TeamPolicy<ExecSpace> policy(nchunks_t * (Hq / HC),
                                       Kokkos::AUTO, 1);
  policy.set_scratch_size(
      0, Kokkos::PerTeam(((2 * tchunk * HC + 2 * HC) * (D + 1) +
                          2 * tchunk * HC) *
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
  // back to one head per team. Token chunks of 8 reuse each K/V row across
  // 8 query rows; smaller chunks keep team scratch ((2*TC*HC + 2*HC) rows
  // of D + 1 plus 2*TC*HC (m, l) scalars) under ~40 KiB for large head dims.
  // UF == 8 unrolls D in consecutive octuples (D % 8 == 0), UF == 4 in
  // quadruples (D % 4 == 0); all branches share the serial dot/update
  // order, so plain-vs-softcap bit-identity holds within each branch.
  const int hchunk = (Hq % 4 == 0) ? 4 : 1;
  int tchunk = 8;  // (hchunk stays runtime here for the scratch budget)
  while (tchunk > 1 &&
         ((2 * tchunk * hchunk + 2 * hchunk) * (D + 1) + 2 * tchunk * hchunk) *
                 sizeof(float) >
             size_t(40 * 1024))
    tchunk /= 2;
  if (T < static_cast<size_t>(tchunk)) tchunk = static_cast<int>(T);
  const int uf = (D % 8 == 0) ? 8 : ((D % 4 == 0) ? 4 : 1);
  if (hchunk == 4) {
    if (uf == 8) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 8, 4, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, label);
    } else if (uf == 4) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 4, 4, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, label);
    } else {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 1, 4, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, label);
    }
  } else {
    if (uf == 8) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 8, 1, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, label);
    } else if (uf == 4) {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 4, 1, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, label);
    } else {
      launch_gqa_streaming<ExecSpace, Scalar, QView, KView, VView, MView,
                           OView, 1, 1, UseMask, UseSoftcap>(
          q, k, v, key_valid, out, T, Hq, D, Hkv, Ktot, q_off, scale, softcap,
          sliding_window, tchunk, label);
    }
  }
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
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
    // Full causal (sliding_window == 0) uses the bottom-right causal cuDNN
    // SDPA graph; the windowed path uses the sliding cuDNN SDPA graph
    // (bottom-right alignment + left band bound == window). Either returning
    // false falls back to the native kernel below.
    if (handle.use_cudnn) {
      if (sliding_window == 0) {
        if (try_cudnn_sdpa_gqa<Scalar>(handle, q, k, v, q_off, out, scale))
          return;
      } else {
        if (try_cudnn_sdpa_gqa_sliding<Scalar>(handle, q, k, v, q_off, out,
                                               scale, sliding_window))
          return;
      }
    }
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif
  const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1);
  const size_t Ktot = k.extent(0);
  if (T == 0 || Hq == 0 || D == 0) return;

  // One team per (token chunk, head chunk); see GqaStreamingFunctor for
  // the blocked online-softmax structure. No [T, Hq, K] score tensor is
  // materialized; extra memory is O(D) scratch per resident team plus O(1)
  // per (t, h) beyond the outputs, so 128k contexts stay feasible.
  using DummyMask = Kokkos::View<int*, ExecSpace>;
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
// dispatch to cuDNN SDPA first (see tpl/cudnn/KokkosDNN_CuDNN.hpp for
// contract). One narrow, version-gated exception lives in
// cudnn_sdpa_forward: FP32 single-token GQA decode (T=1, Hq != Hkv) falls
// back to native on cuDNN <= 9.27.0, where the composite path computes
// only the first query head per KV group while reporting success.
// Other scalars run the native kernel. Otherwise native.
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
// float and bhalf_t dispatch to cuDNN SDPA first (sliding graph with left
// band bound == window; falls back to native on failure), like the
// full-causal path; other scalars run the native Kokkos kernel.
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
  // construction, no version-gated fallback to validate).
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

// Streaming online-softmax bidirectional core shared by both public
// overloads below. Same structure as attention_gqa_streaming above, except
// every query attends to ALL keys (no causal truncation).
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_bidirectional_streaming(const DNNHandle<ExecSpace>& handle,
                                       const QView& q, const KView& k,
                                       const VView& v, const OView& out,
                                       float scale, const std::string& label) {
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
        try_cudnn_sdpa_bidir<Scalar>(handle, q, k, v, out, eff))
      return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif

  // One thread per (query token, query head); streams over the full key
  // axis with an online softmax running state (m, l) and accumulates
  // directly into out. No [T, Hq, K] score tensor is materialized.
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const size_t hkv = (h * Hkv) / Hq;
        for (size_t d = 0; d < D; ++d) out(t, h, d) = static_cast<Scalar>(0);
        if (Ktot == 0) return;
        float m = -1e30f;
        float l = 0.0f;
        for (size_t kk = 0; kk < Ktot; ++kk) {
          float s = 0;
          for (size_t d = 0; d < D; ++d)
            s += static_cast<float>(q(t, h, d)) *
                 static_cast<float>(k(kk, hkv, d));
          s *= eff;
          const float m_new = (m > s) ? m : s;
          const float a = expf(m - m_new);
          const float g = expf(s - m_new);
          const float la = l * a;
          const float l_new = la + g;
          const float inv = 1.0f / l_new;
          const float c = la * inv;
          const float e = g * inv;
          for (size_t d = 0; d < D; ++d) {
            const float o_old = static_cast<float>(out(t, h, d));
            const float v_kd = static_cast<float>(v(kk, hkv, d));
            out(t, h, d) = static_cast<Scalar>(o_old * c + v_kd * e);
          }
          m = m_new;
          l = l_new;
        }
      });
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
// first (see tpl/cudnn/); any failure falls back to the native kernel.
// Other scalars run the native kernel. Otherwise native.
//
// Two overloads: explicit <ExecSpace, Scalar> and a deduction overload
// taking Scalar from the query view. Both share the Impl core above so
// behavior cannot diverge.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const OView& out, float scale = 0.0f,
    const std::string& label = "KokkosDNN::attn_bidir") {
  Impl::attention_bidirectional_streaming<ExecSpace, Scalar>(handle, q, k, v,
                                                            out, scale, label);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const OView& out, float scale = 0.0f,
    const std::string& label = "KokkosDNN::attn_bidir") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_bidirectional_streaming<ExecSpace, Scalar>(handle, q, k, v,
                                                            out, scale, label);
}

}  // namespace KokkosDNN

#endif
