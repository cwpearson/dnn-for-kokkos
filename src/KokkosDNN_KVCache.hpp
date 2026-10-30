// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_KVCACHE_HPP
#define KOKKOSDNN_KVCACHE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_Attention.hpp"

namespace KokkosDNN {
namespace Impl {

// Small residency helpers used by the Llama Model's paged pools. These are
// free functions (not private Model members) on purpose: nvcc extended
// lambdas cannot appear inside a private/protected member function, so any
// device kernel must live here or in a public member.
template <class ExecSpace, class TableView>
void fill_identity(const TableView& tbl) {
  const size_t n = tbl.extent(0);
  if (n == 0) return;
  Kokkos::parallel_for(
      "KokkosDNN::kv_table_fill", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(size_t i) { tbl(i) = (int)i; });
}

// Copy the full extents of src into the [0, extents) prefix of dst
// (dst may be larger). Rank-4 variant for KV pools, rank-3 for scales.
template <class ExecSpace, class DstView, class SrcView>
void copy_prefix_4(const DstView& dst, const SrcView& src) {
  const size_t n0 = src.extent(0), n1 = src.extent(1), n2 = src.extent(2),
               n3 = src.extent(3);
  if (n0 == 0 || n1 == 0 || n2 == 0 || n3 == 0) return;
  Kokkos::parallel_for(
      "KokkosDNN::kv_grow_copy",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {n0, n1, n2, n3}),
      KOKKOS_LAMBDA(size_t p, size_t s, size_t h, size_t d) {
        dst(p, s, h, d) = src(p, s, h, d);
      });
}

template <class ExecSpace, class DstView, class SrcView>
void copy_prefix_3(const DstView& dst, const SrcView& src) {
  const size_t n0 = src.extent(0), n1 = src.extent(1), n2 = src.extent(2);
  if (n0 == 0 || n1 == 0 || n2 == 0) return;
  Kokkos::parallel_for(
      "KokkosDNN::kv_grow_copy_s",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                        {n0, n1, n2}),
      KOKKOS_LAMBDA(size_t p, size_t s, size_t h) {
        dst(p, s, h) = src(p, s, h);
      });
}

}  // namespace Impl

// Paged + int8-quantized KV-cache primitives for long-context inference.
//
// Two orthogonal options (combinable):
//   * Paging: logical token position p maps to (page, slot) with
//     page = p / page_size, slot = p % page_size, then through an explicit
//     block table physical = table[page]. Single-sequence runs use the
//     identity table (table[i] == i); the indirection is still exercised by
//     every load/store so future multi-sequence sharing plugs in here.
//     Pools are [n_pages, page_size, Hkv, D] (LayoutRight: D contiguous).
//   * Int8 quantization: per-token per-KV-head symmetric quantization,
//     scale = max_abs / 127 over the head-dim, stored as FP32. K and V have
//     independent scales. Dequantization (float(q) * scale) happens on the
//     fly inside the attention kernels, so no dequantized copy is ever
//     materialized. Exactness is intentionally given up here: expect small
//     numerical drift vs FP (covered by tests), in exchange for ~2x fewer
//     KV bytes plus a small scale table.
//
// Dense (contiguous, FP) remains the default path in attention_gqa_single;
// the kernels below are used by the Llama Model when Config::kv_page_size
// > 0 and/or Config::kv_int8 are set, and by the Glimmer Model in
// quantized mode (full + sliding). All kernels accumulate in FP32 and
// are Scalar-agnostic (float, bhalf_t) like the dense path. Paged/quant
// paths never dispatch to a TPL (no cuDNN dispatch: cuDNN
// SDPA has no paged/int8 form); TPL GEMMs elsewhere are unaffected.
// Single-token decode (T == 1) of the contiguous int8 caches runs a
// split-K kernel mirroring the dense launch_gqa_decode path (same
// partial/combine structure, dequantized on the fly); wider prefill
// widths and the paged-quant path keep the streaming core below.

inline constexpr int KvInt8Max = 127;

// Per-token per-head symmetric quantize+pack of K3/V3 into contiguous
// int8 caches. k3/v3: [C, Hkv, D]; kq/vq: [Ktot, Hkv, D] int8;
// ks/vs: [Ktot, Hkv] FP32 scales. Writes logical rows [cpos, cpos+C).
template <class ExecSpace, class Scalar, class K3View, class KQView,
          class KSView, class V3View, class VQView, class VSView>
void kv_write_quantized_contiguous(const K3View& k3, const V3View& v3,
                                   size_t cpos, const KQView& kq,
                                   const KSView& ks, const VQView& vq,
                                   const VSView& vs,
                                   const std::string& label = "KokkosDNN::kv_wq") {
  const size_t C = k3.extent(0), Hkv = k3.extent(1), D = k3.extent(2);
  if (C == 0 || Hkv == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Hkv}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float kmax = 0.0f, vmax = 0.0f;
        for (size_t d = 0; d < D; ++d) {
          float a = fabsf(static_cast<float>(k3(t, h, d)));
          float b = fabsf(static_cast<float>(v3(t, h, d)));
          if (a > kmax) kmax = a;
          if (b > vmax) vmax = b;
        }
        float ks_ = (kmax > 0.0f) ? kmax / float(KvInt8Max) : 1.0f;
        float vs_ = (vmax > 0.0f) ? vmax / float(KvInt8Max) : 1.0f;
        ks(cpos + t, h) = ks_;
        vs(cpos + t, h) = vs_;
        float kinv = 1.0f / ks_, vinv = 1.0f / vs_;
        for (size_t d = 0; d < D; ++d) {
          float kf = static_cast<float>(k3(t, h, d)) * kinv;
          float vf = static_cast<float>(v3(t, h, d)) * vinv;
          int kqi = int(roundf(kf));
          int vqi = int(roundf(vf));
          if (kqi > KvInt8Max) kqi = KvInt8Max;
          if (kqi < -KvInt8Max) kqi = -KvInt8Max;
          if (vqi > KvInt8Max) vqi = KvInt8Max;
          if (vqi < -KvInt8Max) vqi = -KvInt8Max;
          kq(cpos + t, h, d) = static_cast<int8_t>(kqi);
          vq(cpos + t, h, d) = static_cast<int8_t>(vqi);
        }
      });
}

// Paged dense write: k3/v3 [C, Hkv, D] into pools [P, page, Hkv, D] via
// the block table. Logical row p = cpos + t maps to
// lpage = p / page_size, slot = p % page_size, phys = table[lpage].
template <class ExecSpace, class Scalar, class K3View, class PoolView,
          class TableView, class V3View>
void kv_write_paged_dense(const K3View& k3, const V3View& v3, size_t cpos,
                          const PoolView& kpool, const PoolView& vpool,
                          const TableView& table, size_t page_size,
                          const std::string& label = "KokkosDNN::kv_wp") {
  const size_t C = k3.extent(0), Hkv = k3.extent(1), D = k3.extent(2);
  if (C == 0 || Hkv == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                        {C, Hkv, D}),
      KOKKOS_LAMBDA(size_t t, size_t h, size_t d) {
        size_t p = cpos + t;
        size_t lpage = p / page_size;
        size_t slot = p % page_size;
        size_t phys = static_cast<size_t>(table(lpage));
        kpool(phys, slot, h, d) = k3(t, h, d);
        vpool(phys, slot, h, d) = v3(t, h, d);
      });
}

// Paged quantized write: k3/v3 [C, Hkv, D] into int8 pools
// [P, page, Hkv, D] plus FP32 scale pools [P, page, Hkv].
template <class ExecSpace, class Scalar, class K3View, class KQPoolView,
          class KSPoolView, class TableView, class V3View>
void kv_write_paged_quantized(
    const K3View& k3, const V3View& v3, size_t cpos, const KQPoolView& kpool,
    const KSPoolView& kscales, const KQPoolView& vpool,
    const KSPoolView& vscales, const TableView& table, size_t page_size,
    const std::string& label = "KokkosDNN::kv_wpq") {
  const size_t C = k3.extent(0), Hkv = k3.extent(1), D = k3.extent(2);
  if (C == 0 || Hkv == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Hkv}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float kmax = 0.0f, vmax = 0.0f;
        for (size_t d = 0; d < D; ++d) {
          float a = fabsf(static_cast<float>(k3(t, h, d)));
          float b = fabsf(static_cast<float>(v3(t, h, d)));
          if (a > kmax) kmax = a;
          if (b > vmax) vmax = b;
        }
        float ksc = (kmax > 0.0f) ? kmax / float(KvInt8Max) : 1.0f;
        float vsc = (vmax > 0.0f) ? vmax / float(KvInt8Max) : 1.0f;
        size_t p = cpos + t;
        size_t lpage = p / page_size;
        size_t slot = p % page_size;
        size_t phys = static_cast<size_t>(table(lpage));
        kscales(phys, slot, h) = ksc;
        vscales(phys, slot, h) = vsc;
        float kinv = 1.0f / ksc, vinv = 1.0f / vsc;
        for (size_t d = 0; d < D; ++d) {
          int kqi = int(roundf(static_cast<float>(k3(t, h, d)) * kinv));
          int vqi = int(roundf(static_cast<float>(v3(t, h, d)) * vinv));
          if (kqi > KvInt8Max) kqi = KvInt8Max;
          if (kqi < -KvInt8Max) kqi = -KvInt8Max;
          if (vqi > KvInt8Max) vqi = KvInt8Max;
          if (vqi < -KvInt8Max) vqi = -KvInt8Max;
          kpool(phys, slot, h, d) = static_cast<int8_t>(kqi);
          vpool(phys, slot, h, d) = static_cast<int8_t>(vqi);
        }
      });
}

namespace Impl {

// Loaders resolve a logical key position kk to a (k, v) float pair for the
// streaming core below. Namespace-scope templates (not function-local
// structs): nvcc extended lambdas cannot capture function-local types.
template <class KQView, class KSView, class VQView, class VSView>
struct QuantLoader {
  KQView kq;
  KSView ks;
  VQView vq;
  VSView vs;
  size_t ktot;
  KOKKOS_INLINE_FUNCTION size_t tokens() const { return ktot; }
  KOKKOS_INLINE_FUNCTION float k(size_t kk, size_t h, size_t d) const {
    return static_cast<float>(kq(kk, h, d)) * ks(kk, h);
  }
  KOKKOS_INLINE_FUNCTION float v(size_t kk, size_t h, size_t d) const {
    return static_cast<float>(vq(kk, h, d)) * vs(kk, h);
  }
};

template <class PoolView, class TableView>
struct PagedLoader {
  PoolView kp, vp;
  TableView tab;
  size_t ps, cap;
  KOKKOS_INLINE_FUNCTION size_t tokens() const { return cap; }
  KOKKOS_INLINE_FUNCTION float k(size_t kk, size_t h, size_t d) const {
    size_t phys = static_cast<size_t>(tab(kk / ps));
    return static_cast<float>(kp(phys, kk % ps, h, d));
  }
  KOKKOS_INLINE_FUNCTION float v(size_t kk, size_t h, size_t d) const {
    size_t phys = static_cast<size_t>(tab(kk / ps));
    return static_cast<float>(vp(phys, kk % ps, h, d));
  }
};

template <class KQPoolView, class KSPoolView, class TableView>
struct PagedQuantLoader {
  KQPoolView kq, vq;
  KSPoolView ks, vs;
  TableView tab;
  size_t ps, cap;
  KOKKOS_INLINE_FUNCTION size_t tokens() const { return cap; }
  KOKKOS_INLINE_FUNCTION float k(size_t kk, size_t h, size_t d) const {
    size_t phys = static_cast<size_t>(tab(kk / ps));
    size_t slot = kk % ps;
    return static_cast<float>(kq(phys, slot, h, d)) * ks(phys, slot, h);
  }
  KOKKOS_INLINE_FUNCTION float v(size_t kk, size_t h, size_t d) const {
    size_t phys = static_cast<size_t>(tab(kk / ps));
    size_t slot = kk % ps;
    return static_cast<float>(vq(phys, slot, h, d)) * vs(phys, slot, h);
  }
};

// Shared streaming core shape: one thread per (query token, query head),
// online softmax over keys, O(1) extra memory. Loaders differ per cache
// kind; each loader resolves logical key kk to a (k, v) float pair.
// sliding_window == 0 attends the full causal prefix [0, q_off + t];
// otherwise query t at absolute position p = q_off + t attends the windowed
// range [max(0, p + 1 - sliding_window), q_off + t] (same convention as
// gqa_key_window in KokkosDNN_Attention.hpp).
template <class ExecSpace, class Scalar, class QView, class OView, class Loader>
void attention_streaming_loader(const QView& q, size_t q_off, const OView& out,
                                float scale, size_t Hq, size_t D, size_t Hkv,
                                Loader loader, const std::string& label,
                                size_t sliding_window = 0) {
  const size_t T = q.extent(0);
  if (T == 0 || Hq == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        size_t klen = q_off + t + 1;
        size_t kmax = loader.tokens();
        if (klen > kmax) klen = kmax;
        // Windowed lower clamp (mirrors gqa_key_window): keys below the
        // sliding window are never streamed, so long prefixes stay O(window)
        // in work here too.
        size_t klo = 0;
        if (sliding_window > 0) {
          const size_t p     = q_off + t;  // absolute position of query t
          const size_t start = (p + 1 >= sliding_window) ? (p + 1 - sliding_window) : 0;
          if (start > klo) klo = start;
        }
        const size_t hkv = (h * Hkv) / Hq;
        for (size_t d = 0; d < D; ++d) out(t, h, d) = static_cast<Scalar>(0);
        if (klen == 0) return;
        float m = -1e30f, l = 0.0f;
        for (size_t kk = klo; kk < klen; ++kk) {
          float s = 0;
          for (size_t d = 0; d < D; ++d)
            s += static_cast<float>(q(t, h, d)) * loader.k(kk, hkv, d);
          s *= scale;
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
            out(t, h, d) = static_cast<Scalar>(o_old * c + loader.v(kk, hkv, d) * e);
          }
          m = m_new;
          l = l_new;
        }
      });
}

// ---------------------------------------------------------------------------
// Single-token (T == 1) split-K decode over contiguous int8 KV caches.
//
// Same structure as the dense GqaDecodeSplitFunctor (see
// KokkosDNN_Attention.hpp): the query row's key window [klo, khi) is cut
// into `nsplit` contiguous ranges, one block per (head group, key range),
// with the same scratch layout (scaled Q rows + normalized running means
// [NG x SS] plus this tile's per-head weights [NG x ktile]), the same
// per-tile score/max/weight/accumulate phases, and the same partial-record
// + GqaDecodeCombineFunctor merge. The only difference is the KV source:
// K/V rows are int8 (kq/vq [Ktot, Hkv, D]) dequantized on the fly with
// their per-token per-head FP32 scales (ks/vs [Ktot, Hkv]). Per-key scales
// are hoisted out of the D loops (one scale load per key, reused across
// the NG heads and all D), so the traffic stays one int8 row + one FP32
// scale per key per block. No TPL dispatch (cuDNN SDPA has no int8 form).
//
// Numerics mirror the dense split-K kernel exactly (FP32 accumulation,
// normalized running mean, log-sum-exp combine), so the result equals
// softmax(Q K^T*scale) V over the dequantized rows up to FP32
// reassociation, matching the streaming core above to ~1e-6.
// ---------------------------------------------------------------------------
template <class ExecSpace, class Scalar, class QView, class KQView, class KSView,
          class VQView, class VSView, class PView, int NG>
struct GqaDecodeSplitQuantFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using ScratchView =
      Kokkos::View<float*, typename ExecSpace::scratch_memory_space,
                   Kokkos::MemoryUnmanaged>;
  QView q;
  KQView kq;
  KSView ks;
  VQView vq;
  VSView vs;
  PView partial;
  size_t Hq, D, Hkv, klo, khi, nsplit, ksplit;
  int ktile;  // keys per tile
  float scale;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    // Local handles: inner lambdas capture by value without an implicit
    // `this` capture (quiets -Wdeprecated under C++20).
    const auto Q = q;
    const auto KQ = kq;
    const auto KS = ks;
    const auto VQ = vq;
    const auto VS = vs;
    const auto P = partial;
    const size_t Dloc = D;
    const size_t Hqloc = Hq;
    const size_t Kvloc = Hkv;
    const size_t NS = nsplit;
    const int KT = ktile;
    const float scl = scale;
    const int rank = static_cast<int>(member.team_rank());
    const int team_n = static_cast<int>(member.team_size());

    // Block (g, sp): head group g (kv head shared by the NG query heads)
    // and key split sp.
    const size_t block = static_cast<size_t>(member.league_rank());
    const size_t grp = block / NS;
    const size_t sp = block % NS;
    const size_t hbase = grp * static_cast<size_t>(NG);
    const size_t kvh = (hbase * Kvloc) / Hqloc;
    const size_t k0 = klo + sp * ksplit;
    const size_t k1 = (k0 + ksplit < khi) ? (k0 + ksplit) : khi;

    // Scratch layout mirrors gqa_decode_split_scratch_floats.
    size_t SS = Dloc + 1;
    if (SS % 32 == 0) ++SS;
    ScratchView buf(member.team_scratch(0),
                    gqa_decode_split_scratch_floats(Dloc, NG, KT));
    float* base = buf.data();
    const size_t qo = 0;                                // [NG][SS] scaled Q
    const size_t ao = static_cast<size_t>(NG) * SS;     // [NG][SS] FP32 acc
    const size_t wo = 2 * static_cast<size_t>(NG) * SS;  // [NG][KT] weights

    // Scaled query rows (one per head of the group) and zeroed accumulators.
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, static_cast<size_t>(NG) * SS),
        KOKKOS_LAMBDA(size_t i) {
          const size_t hh = i / SS;
          const size_t dd = i % SS;
          base[qo + i] = (dd < Dloc)
                             ? static_cast<float>(Q(0, hbase + hh, dd)) * scl
                             : 0.0f;
          base[ao + i] = 0.0f;
        });
    member.team_barrier();

    // Dims per thread in the accumulator phase (D <= team size => one).
    const size_t dpt = (Dloc + static_cast<size_t>(team_n) - 1) /
                       static_cast<size_t>(team_n);

    // Streaming state: every thread keeps an identical copy of (m, aa) -- the
    // team reductions broadcast them -- and the dim-owning threads advance
    // the running denominator lacc with their accumulator.
    float m[NG], lacc[NG], aa[NG];
    for (int h = 0; h < NG; ++h) {
      m[h] = -1e30f;
      lacc[h] = 0.0f;
    }

    for (size_t t0 = k0; t0 < k1; t0 += static_cast<size_t>(KT)) {
      const int kw = (t0 + static_cast<size_t>(KT) <= k1)
                         ? KT
                         : static_cast<int>(k1 - t0);
      // Scores: thread r owns keys r, r + team, ... of this tile; the
      // full-D dot is done once per (key, group head) from the dequantized
      // K row. Two keys are in flight per iteration; each key's FP32 scale
      // is loaded once and reused across D and all NG heads.
      int kki = rank;
      for (; kki + team_n < kw; kki += 2 * team_n) {
        const size_t kk0 = t0 + static_cast<size_t>(kki);
        const size_t kk1 = kk0 + static_cast<size_t>(team_n);
        const float f0 = KS(kk0, kvh);
        const float f1 = KS(kk1, kvh);
        float s0[NG], s1[NG];
        for (int h = 0; h < NG; ++h) {
          s0[h] = 0.0f;
          s1[h] = 0.0f;
        }
        for (size_t d = 0; d < Dloc; ++d) {
          const float kv0 = static_cast<float>(KQ(kk0, kvh, d)) * f0;
          const float kv1 = static_cast<float>(KQ(kk1, kvh, d)) * f1;
          for (int h = 0; h < NG; ++h) {
            const float qs = base[qo + static_cast<size_t>(h) * SS + d];
            s0[h] += qs * kv0;
            s1[h] += qs * kv1;
          }
        }
        for (int h = 0; h < NG; ++h) {
          base[wo + static_cast<size_t>(h) * KT + static_cast<size_t>(kki)] =
              s0[h];
          base[wo + static_cast<size_t>(h) * KT +
               static_cast<size_t>(kki + team_n)] = s1[h];
        }
      }
      for (; kki < kw; kki += team_n) {
        const size_t kk = t0 + static_cast<size_t>(kki);
        const float fk = KS(kk, kvh);
        float s[NG];
        for (int h = 0; h < NG; ++h) s[h] = 0.0f;
        for (size_t d = 0; d < Dloc; ++d) {
          const float kv = static_cast<float>(KQ(kk, kvh, d)) * fk;
          for (int h = 0; h < NG; ++h)
            s[h] += base[qo + static_cast<size_t>(h) * SS + d] * kv;
        }
        for (int h = 0; h < NG; ++h)
          base[wo + static_cast<size_t>(h) * KT + static_cast<size_t>(kki)] =
              s[h];
      }
      member.team_barrier();
      // Per-head tile maximum folded into the running max (the -1e30 sentinel
      // makes the first tile's rescale exp(-1e30 - m_new) == 0).
      for (int h = 0; h < NG; ++h) {
        float tmax = -1e30f;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(member, static_cast<size_t>(kw)),
            KOKKOS_LAMBDA(size_t i, float& vmax) {
              const float sv =
                  base[wo + static_cast<size_t>(h) * KT + i];
              if (sv > vmax) vmax = sv;
            },
            Kokkos::Max<float>(tmax));
        const float mnew = (tmax > m[h]) ? tmax : m[h];
        aa[h] = expf(m[h] - mnew);
        m[h] = mnew;
      }
      // Weights for the owner's keys: exp(s - m) in the (possibly updated)
      // running max scale.
      for (int kw_i = rank; kw_i < kw; kw_i += team_n) {
        for (int h = 0; h < NG; ++h)
          base[wo + static_cast<size_t>(h) * KT + static_cast<size_t>(kw_i)] =
              expf(base[wo + static_cast<size_t>(h) * KT +
                         static_cast<size_t>(kw_i)] -
                   m[h]);
      }
      member.team_barrier();
      // Pre-tile denominator in the (possibly updated) max scale. Hoisted
      // out of the per-dim loop below so every owned dim of this thread folds
      // the tile into the same running denominator.
      float lold[NG];
      for (int h = 0; h < NG; ++h) lold[h] = lacc[h] * aa[h];
      // Accumulate: thread r owns dims r, r + team, ...; the NG accumulators
      // of its dims live in registers across the tile, and each V element
      // read feeds all NG heads. Four keys are in flight per iteration with
      // one accumulator set each. The accumulator is the *normalized* running
      // mean (same deferred-normalization argument as the dense kernel), so
      // its magnitude stays at the answer's scale.
      for (size_t u = 0; u < dpt; ++u) {
        const size_t d =
            static_cast<size_t>(rank) + u * static_cast<size_t>(team_n);
        if (d >= Dloc) break;
        float a0[NG], a1[NG], a2[NG], a3[NG], tw[NG];
        for (int h = 0; h < NG; ++h) {
          a0[h] = 0.0f;
          a1[h] = 0.0f;
          a2[h] = 0.0f;
          a3[h] = 0.0f;
          tw[h] = 0.0f;
        }
        int kkj = 0;
        for (; kkj + 4 <= kw; kkj += 4) {
          const size_t kk0 = t0 + static_cast<size_t>(kkj);
          const float v0 =
              static_cast<float>(VQ(kk0, kvh, d)) * VS(kk0, kvh);
          const float v1 =
              static_cast<float>(VQ(kk0 + 1, kvh, d)) * VS(kk0 + 1, kvh);
          const float v2 =
              static_cast<float>(VQ(kk0 + 2, kvh, d)) * VS(kk0 + 2, kvh);
          const float v3 =
              static_cast<float>(VQ(kk0 + 3, kvh, d)) * VS(kk0 + 3, kvh);
          for (int h = 0; h < NG; ++h) {
            const float* __restrict__ wp =
                base + wo + static_cast<size_t>(h) * KT +
                static_cast<size_t>(kkj);
            const float w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3];
            a0[h] += w0 * v0;
            a1[h] += w1 * v1;
            a2[h] += w2 * v2;
            a3[h] += w3 * v3;
            tw[h] += (w0 + w1) + (w2 + w3);
          }
        }
        for (; kkj < kw; ++kkj) {
          const size_t kk = t0 + static_cast<size_t>(kkj);
          const float vv =
              static_cast<float>(VQ(kk, kvh, d)) * VS(kk, kvh);
          for (int h = 0; h < NG; ++h) {
            const float wt =
                base[wo + static_cast<size_t>(h) * KT + static_cast<size_t>(kkj)];
            a0[h] += wt * vv;
            tw[h] += wt;
          }
        }
        // Fold this tile into the running mean. `tw` is the tile's total
        // weight, identical in every owned-dim iteration, so the first one
        // (u == 0) is the one that advances the running denominator.
        for (int h = 0; h < NG; ++h) {
          const float l_new = lold[h] + tw[h];
          const float num =
              base[ao + static_cast<size_t>(h) * SS + d] * lold[h] +
              ((a0[h] + a1[h]) + (a2[h] + a3[h]));
          base[ao + static_cast<size_t>(h) * SS + d] =
              (l_new > 0.0f) ? (num / l_new) : 0.0f;
          if (u == 0) lacc[h] = l_new;
        }
      }
      member.team_barrier();  // tile buffer reusable by the next iteration
    }

    // Write this split's partial record(s): (running max, denominator,
    // normalized mean per dim). A block with no key writes the -1e30 / 0
    // sentinel pair, which the combine weights to zero.
    const size_t rec = Dloc + 2;
    if (rank == 0) {
      for (int h = 0; h < NG; ++h) {
        float* pr =
            P.data() + ((hbase + static_cast<size_t>(h)) * NS + sp) * rec;
        pr[0] = m[h];
        pr[1] = lacc[h];
      }
    }
    for (size_t u = 0; u < dpt; ++u) {
      const size_t d =
          static_cast<size_t>(rank) + u * static_cast<size_t>(team_n);
      if (d >= Dloc) break;
      for (int h = 0; h < NG; ++h)
        P.data()[((hbase + static_cast<size_t>(h)) * NS + sp) * rec + 2 + d] =
            base[ao + static_cast<size_t>(h) * SS + d];
    }
  }
};

// Split-K decode launcher for a fixed group width NG (compile-time so the
// per-head arrays stay in registers). `nsplit`/`ksplit`/`ktile`/`team_size`
// are chosen by launch_gqa_decode_quant below. Partial workspace and merge
// reuse the dense path's helpers (gqa_decode_partial_floats,
// GqaDecodeCombineFunctor).
template <class ExecSpace, class Scalar, class QView, class KQView, class KSView,
          class VQView, class VSView, class OView, int NG>
void launch_gqa_decode_quant_split(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KQView& kq,
    const KSView& ks, const VQView& vq, const VSView& vs, size_t klo,
    size_t khi, size_t nsplit, size_t ksplit, int ktile, int team_size,
    const OView& out, float scale, const std::string& label) {
  const size_t Hq = q.extent(1), D = q.extent(2);
  const size_t ngroups = Hq / static_cast<size_t>(NG);
  Kokkos::View<float*, ExecSpace> partial = handle.attn_decode_scratch(
      gqa_decode_partial_floats(Hq, nsplit, D));

  GqaDecodeSplitQuantFunctor<ExecSpace, Scalar, QView, KQView, KSView, VQView,
                             VSView, Kokkos::View<float*, ExecSpace>, NG>
      f{q, kq, ks, vq, vs, partial, Hq, D, kq.extent(1), klo, khi, nsplit,
        ksplit, ktile, scale};
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(ngroups * nsplit),
                                       team_size, 1);
  policy.set_scratch_size(
      0, Kokkos::PerTeam(gqa_decode_split_scratch_floats(D, NG, ktile) *
                         sizeof(float)));
  Kokkos::parallel_for(label, policy, f);

  GqaDecodeCombineFunctor<ExecSpace, Scalar, OView,
                          Kokkos::View<float*, ExecSpace>>
      cf{out, partial, D, nsplit, team_size};
  Kokkos::TeamPolicy<ExecSpace> cpolicy(static_cast<int>(Hq), team_size, 1);
  cpolicy.set_scratch_size(
      0, Kokkos::PerTeam((static_cast<size_t>(team_size) + nsplit) *
                         sizeof(float)));
  Kokkos::parallel_for(label + "_combine", cpolicy, cf);
}

// Single-token decode launcher over contiguous int8 KV. Mirrors the dense
// launch_gqa_decode: GQA group width, split count, tile width and team size
// from the geometry and the backend's actual scratch budget, then the
// split-K partial kernel plus the combine kernel. Head dims too large for
// even a minimal split-K footprint fall back to the serial streaming core
// (correct, just unparallelized).
template <class ExecSpace, class Scalar, class QView, class KQView, class KSView,
          class VQView, class VSView, class OView>
void launch_gqa_decode_quant(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KQView& kq,
    const KSView& ks, const VQView& vq, const VSView& vs, size_t q_off,
    const OView& out, float scale, size_t sliding_window,
    const std::string& label) {
  const size_t Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = kq.extent(1), Ktot = kq.extent(0);
  if (Hq == 0 || D == 0 || Hkv == 0 || Ktot == 0) return;

  // Causal (plus sliding-window) key window of the single query row.
  size_t klo = 0, khi = 0;
  gqa_key_window(q_off, Ktot, sliding_window, 0, klo, khi);
  if (khi > Ktot) khi = Ktot;
  const size_t len = (khi > klo) ? (khi - klo) : 0;

  // Group width: all query heads sharing one KV head, when that is 1/2/4/8.
  // Anything else runs one head per block with the plain h * Hkv / Hq
  // mapping (e.g. Glimmer's 32 Q / 2 KV heads: group of 16 -> NG == 1).
  size_t ng = 1;
  if (Hq % Hkv == 0) {
    const size_t g = Hq / Hkv;
    if (g == 2 || g == 4 || g == 8) ng = g;
  }
  const int ngroups = static_cast<int>(Hq / ng);
  const int team_req = 128;
  const size_t budget =
      static_cast<size_t>(Kokkos::TeamPolicy<ExecSpace>::scratch_size_max(0));

  // Extreme head dims: one head per block, and below even that the serial
  // streaming core (always correct; the split path handles every geometry
  // that fits).
  if (gqa_decode_split_scratch_floats(D, static_cast<int>(ng), 8) *
          sizeof(float) >
      budget)
    ng = 1;
  if (gqa_decode_split_scratch_floats(D, 1, 8) * sizeof(float) > budget) {
    attention_streaming_loader<ExecSpace, Scalar>(
        q, q_off, out, scale, Hq, D, Hkv,
        QuantLoader<KQView, KSView, VQView, VSView>{kq, ks, vq, vs, Ktot},
        label, sliding_window);
    return;
  }

  int conc = 1;
  {
    // Threads the backend can run at once (1 on Serial/Threads with one
    // thread); only used as an SM-count proxy, never for correctness.
    const int c = ExecSpace().concurrency();
    if (c > 1) conc = c;
  }
  const int nsplit_i = gqa_decode_nsplit(len, static_cast<size_t>(ngroups),
                                         conc);
  const size_t nsplit = static_cast<size_t>(nsplit_i);
  const size_t ksplit = (len + nsplit - 1) / nsplit;  // ceil, >= 1 when len > 0

  // Team size: a requested warp count clamped to what the backend and the
  // kernel's scratch allow (Serial -> 1). Probe with the largest tile so the
  // answer is conservative for whatever tile width is chosen below.
  auto scratch_bytes = [&](int kt) {
    return gqa_decode_split_scratch_floats(D, static_cast<int>(ng), kt) *
           sizeof(float);
  };
  int team_size = team_req;
  {
    Kokkos::View<float*, ExecSpace> probe_ws =
        handle.attn_decode_scratch(gqa_decode_partial_floats(Hq, nsplit, D));
    GqaDecodeSplitQuantFunctor<ExecSpace, Scalar, QView, KQView, KSView, VQView,
                               VSView, Kokkos::View<float*, ExecSpace>, 1>
        probe_f{q, kq,      ks,     vq,    vs, probe_ws, Hq, D, Hkv,
                klo, khi,  nsplit, ksplit, 512, scale};
    Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
    const size_t req = scratch_bytes(512);
    probe.set_scratch_size(0, Kokkos::PerTeam(req < budget ? req : budget));
    const int mx = probe.team_size_max(probe_f, Kokkos::ParallelForTag{});
    if (mx < team_size) team_size = mx;
    if (team_size < 1) team_size = 1;
  }

  // Tile width: 512 keys (four keys per thread at a 128-thread team, which
  // is what the score and accumulator loops are unrolled for; a split shorter
  // than a tile simply runs one partial tile), shrunk until the per-team
  // scratch fits the backend's budget.
  int ktile = 512;
  while (ktile > 8 && scratch_bytes(ktile) > budget) ktile /= 2;

  switch (ng) {
    case 8:
      launch_gqa_decode_quant_split<ExecSpace, Scalar, QView, KQView, KSView,
                                    VQView, VSView, OView, 8>(
          handle, q, kq, ks, vq, vs, klo, khi, nsplit, ksplit, ktile,
          team_size, out, scale, label);
      break;
    case 4:
      launch_gqa_decode_quant_split<ExecSpace, Scalar, QView, KQView, KSView,
                                    VQView, VSView, OView, 4>(
          handle, q, kq, ks, vq, vs, klo, khi, nsplit, ksplit, ktile,
          team_size, out, scale, label);
      break;
    case 2:
      launch_gqa_decode_quant_split<ExecSpace, Scalar, QView, KQView, KSView,
                                    VQView, VSView, OView, 2>(
          handle, q, kq, ks, vq, vs, klo, khi, nsplit, ksplit, ktile,
          team_size, out, scale, label);
      break;
    default:
      launch_gqa_decode_quant_split<ExecSpace, Scalar, QView, KQView, KSView,
                                    VQView, VSView, OView, 1>(
          handle, q, kq, ks, vq, vs, klo, khi, nsplit, ksplit, ktile,
          team_size, out, scale, label);
      break;
  }
}

}  // namespace Impl

// Contiguous int8 KV attention. kq/vq: [Ktot, Hkv, D] int8;
// ks/vs: [Ktot, Hkv] FP32. Causal mask identical to attention_gqa_single:
// query t attends keys [0, q_off + t]. Single-token decode (T == 1) runs
// the split-K kernel above (mirrors the dense T == 1 path); wider widths
// run the streaming core below. Native kernels only (no cuDNN int8 form).
template <class ExecSpace, class Scalar, class QView, class KQView, class KSView,
          class VQView, class VSView, class OView>
void attention_gqa_quantized(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KQView& kq,
    const KSView& ks, const VQView& vq, const VSView& vs, size_t q_off,
    const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa_q") {
  const size_t Hq = q.extent(1), D = q.extent(2), Hkv = kq.extent(1);
  const size_t Ktot = kq.extent(0);
  if (q.extent(0) == 1) {
    Impl::tpl_note_attn(false);
    Impl::launch_gqa_decode_quant<ExecSpace, Scalar>(
        handle, q, kq, ks, vq, vs, q_off, out, scale, 0, label);
    return;
  }
  Impl::tpl_note_attn(false);
  Impl::attention_streaming_loader<ExecSpace, Scalar>(
      q, q_off, out, scale, Hq, D, Hkv,
      Impl::QuantLoader<KQView, KSView, VQView, VSView>{kq, ks, vq, vs, Ktot},
      label);
}

// Sliding-window causal int8 KV attention. Same contract as
// attention_gqa_quantized, except query t at absolute position
// p = q_off + t attends keys [max(0, p + 1 - sliding_window), p] instead of
// the full prefix [0, p] (same convention as attention_gqa_single_sliding;
// sliding_window == 0 degrades to the full-causal form). Native kernel only
// (no cuDNN int8 form), like the full-causal quantized path below.
template <class ExecSpace, class Scalar, class QView, class KQView, class KSView,
          class VQView, class VSView, class OView>
void attention_gqa_quantized_sliding(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KQView& kq,
    const KSView& ks, const VQView& vq, const VSView& vs, size_t q_off,
    const OView& out, float scale, size_t sliding_window,
    const std::string& label = "KokkosDNN::attn_gqa_qs") {
  const size_t Hq = q.extent(1), D = q.extent(2), Hkv = kq.extent(1);
  const size_t Ktot = kq.extent(0);
  if (q.extent(0) == 1) {
    Impl::tpl_note_attn(false);
    Impl::launch_gqa_decode_quant<ExecSpace, Scalar>(
        handle, q, kq, ks, vq, vs, q_off, out, scale, sliding_window, label);
    return;
  }
  Impl::tpl_note_attn(false);
  Impl::attention_streaming_loader<ExecSpace, Scalar>(
      q, q_off, out, scale, Hq, D, Hkv,
      Impl::QuantLoader<KQView, KSView, VQView, VSView>{kq, ks, vq, vs, Ktot},
      label, sliding_window);
}

// Paged dense attention. Pools [P, page, Hkv, D]; table maps logical page
// to physical. Logical capacity = table.extent(0) * page_size; klen is
// capped there (callers only query resident prefixes).
template <class ExecSpace, class Scalar, class QView, class PoolView,
          class TableView, class OView>
void attention_gqa_paged(const DNNHandle<ExecSpace>& handle, const QView& q,
                         const PoolView& kpool, const PoolView& vpool,
                         const TableView& table, size_t page_size, size_t q_off,
                         const OView& out, float scale,
                         const std::string& label = "KokkosDNN::attn_gqa_p") {
  (void)handle;
  const size_t Hq = q.extent(1), D = q.extent(2), Hkv = kpool.extent(2);
  const size_t cap = table.extent(0) * page_size;
  Impl::attention_streaming_loader<ExecSpace, Scalar>(
      q, q_off, out, scale, Hq, D, Hkv,
      Impl::PagedLoader<PoolView, TableView>{kpool, vpool, table, page_size,
                                             cap},
      label);
}

// Paged int8 attention. kpool/vpool: [P, page, Hkv, D] int8;
// kscales/vscales: [P, page, Hkv] FP32.
template <class ExecSpace, class Scalar, class QView, class KQPoolView,
          class KSPoolView, class TableView, class OView>
void attention_gqa_paged_quantized(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KQPoolView& kpool,
    const KSPoolView& kscales, const KQPoolView& vpool,
    const KSPoolView& vscales, const TableView& table, size_t page_size,
    size_t q_off, const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa_pq") {
  (void)handle;
  const size_t Hq = q.extent(1), D = q.extent(2), Hkv = kpool.extent(2);
  const size_t cap = table.extent(0) * page_size;
  Impl::attention_streaming_loader<ExecSpace, Scalar>(
      q, q_off, out, scale, Hq, D, Hkv,
      Impl::PagedQuantLoader<KQPoolView, KSPoolView, TableView>{
          kpool, vpool, kscales, vscales, table, page_size, cap},
      label);
}

}  // namespace KokkosDNN

#endif
