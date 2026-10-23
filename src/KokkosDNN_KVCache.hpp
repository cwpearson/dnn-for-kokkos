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
// > 0 and/or Config::kv_int8 are set. All kernels accumulate in FP32 and
// are Scalar-agnostic (float, bhalf_t) like the dense path. Paged/quant
// paths always run the native streaming kernel (no cuDNN dispatch: cuDNN
// SDPA has no paged/int8 form); TPL GEMMs elsewhere are unaffected.

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
template <class ExecSpace, class Scalar, class QView, class OView, class Loader>
void attention_streaming_loader(const QView& q, size_t q_off, const OView& out,
                                float scale, size_t Hq, size_t D, size_t Hkv,
                                Loader loader,
                                const std::string& label) {
  const size_t T = q.extent(0);
  if (T == 0 || Hq == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        size_t klen = q_off + t + 1;
        size_t kmax = loader.tokens();
        if (klen > kmax) klen = kmax;
        const size_t hkv = (h * Hkv) / Hq;
        for (size_t d = 0; d < D; ++d) out(t, h, d) = static_cast<Scalar>(0);
        if (klen == 0) return;
        float m = -1e30f, l = 0.0f;
        for (size_t kk = 0; kk < klen; ++kk) {
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

}  // namespace Impl

// Contiguous int8 KV attention. kq/vq: [Ktot, Hkv, D] int8;
// ks/vs: [Ktot, Hkv] FP32. Causal mask identical to attention_gqa_single:
// query t attends keys [0, q_off + t].
template <class ExecSpace, class Scalar, class QView, class KQView, class KSView,
          class VQView, class VSView, class OView>
void attention_gqa_quantized(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KQView& kq,
    const KSView& ks, const VQView& vq, const VSView& vs, size_t q_off,
    const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa_q") {
  (void)handle;  // native kernel only (no cuDNN paged/int8 form)
  const size_t Hq = q.extent(1), D = q.extent(2), Hkv = kq.extent(1);
  const size_t Ktot = kq.extent(0);
  Impl::attention_streaming_loader<ExecSpace, Scalar>(
      q, q_off, out, scale, Hq, D, Hkv,
      Impl::QuantLoader<KQView, KSView, VQView, VSView>{kq, ks, vq, vs, Ktot},
      label);
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
