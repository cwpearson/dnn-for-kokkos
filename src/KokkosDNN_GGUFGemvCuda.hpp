// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GGUFGEMVCUDA_HPP
#define KOKKOSDNN_GGUFGEMVCUDA_HPP

// Portable int8-dot GEMV for packed GGUF weights at M == 1 (decode), MMVQ style.
//
// The activation row is quantized once into int8 blocks ({d, per-half sums,
// 32 x int8}) and every output row then dots its packed weights against the
// int8 row with integer arithmetic: one int dot per 16/32-element window plus
// one FP32 rescale per window, instead of the per-element nibble-extract +
// int->float + fp16-scale + FMA chain. Decode is instruction-bound on the
// portable path (per 32 weight elements: nibble unpacks, fp16 superblock
// scales, 32 FMAs for an 18-byte payload), so moving the dots to integers and
// hoisting the scales out of the element loop is the whole win. Weights keep
// their on-disk 4/5/6-bit precision and FP16 superblock scales, and all block
// scaling stays in FP32.
//
// Structure (two Kokkos launches per GEMV):
//   1. QuantizeActQ8Functor: one RangePolicy iteration per 32-element
//      activation block. Block max of |x| -> d = amax/127,
//      q_i = rint(x_i/d) in [-127, 127], plus the sum of each 16-element half
//      (the weight-side min / Q6_K -32 terms, so the dot never re-derives
//      them). The row is quantized ONCE per GEMV and reused by all N rows.
//   2. GemvInt8BlockedFunctor<DT, ROWS>: one team per ROWS-row group with
//      128 threads per group (64 per row at ROWS=2). Each thread strides the
//      row's decode units, loads the unit's activation window once
//      (Int8UnitWindow: 8 int8 words + the two block headers) and dots it
//      against all ROWS packed weight rows, so the activation side is shared
//      across rows instead of re-read per row. The per-row partials reduce
//      through team scratch. Each unit dot is straight-line word-level
//      integer math through Impl::IntDot (one dp4a-class dot per 4-byte word
//      where the backend provides it, exact scalar arithmetic elsewhere).
//      GemvInt8Functor (one team per row, TeamThreadRange reduction) stays as
//      the short-row / backend-cap fallback.
// The unit mapping is EXACTLY the portable one (Impl::quant_unit_count /
// quant_unit_dot: Q4_K unit = block c/8, iter (c%8)/2, half c%2; Q6_K unit =
// block c/8, chunk (c%8)/4, pair (c%4)/2, half c%2; Q5_K/Q8_0 unit = one
// 32-element sub-block), so a row is the same set of terms as before, just
// summed in a different order and with the activations rounded to int8.
//
// Numerics (documented, mirrors llama.cpp's default MMVQ path):
//   * activations: per-32-element symmetric int8, d = amax/127 in FP32.
//     This is the only new error source: ~0.4% of the block amax per
//     element, i.e. a ~1% relative perturbation of the dot product, the
//     same trade llama.cpp's default (Q8_1 activations) makes. Weights keep
//     their on-disk 4/5/6-bit precision and their FP16 superblock scales,
//     and all block scaling stays in FP32.
//   * accumulation: INT32 per 16/32-element dot, FP32 across units/threads.
//
// Everything here is portable Kokkos: the only backend specialization is the
// integer dot itself, factored out into KokkosDNN_IntDot.hpp (Impl::IntDot:
// one dp4a-class dot per 4-byte word on execution spaces with a
// single-instruction int8 dot, exact scalar arithmetic everywhere else). The
// plain portable path in KokkosDNN_GGUF.hpp remains the fallback for every
// unsupported dtype and every non-block-aligned K.
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "KokkosDNN_Macros.hpp"
#include "KokkosDNN_Occupancy.hpp"
#include "KokkosDNN_config.h"
#include "KokkosDNN_IntDot.hpp"

namespace KokkosDNN {
namespace Impl {
namespace GgufMvq {

// Packed weight dtypes this path implements. Anything else declines (the
// caller keeps the portable kernel), so adding a tag here is the only way to
// opt a format in.
enum class Wd : int { Q4K = 0, Q5K = 1, Q6K = 2, Q80 = 3 };

// ggml block layouts (ggml-common.h field order). GGUF.hpp asserts these
// sizes against its own Gguf::BlockXXX definitions, so the two cannot drift.
struct BlockQ4K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};
struct BlockQ5K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};
struct BlockQ6K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};
struct BlockQ80 {
  uint16_t d;
  int8_t qs[32];
};

// ---------------------------------------------------------------------------
// Activation staging layout (one GEMV's worth, caller-owned).
//   [0, K)                    int8 quantized activation row
//   [align16(K), ...)         one ActQ8Hdr per 32-element block
// SoA keeps both sides vector-loadable: every unit's element offset is a
// multiple of 16.
// ---------------------------------------------------------------------------
struct alignas(16) ActQ8Hdr {
  float d;         // block scale: amax / 127
  int32_t sum_lo;  // sum of qs[0..15]
  int32_t sum_hi;  // sum of qs[16..31]
  int32_t pad;     // keeps the header a 16-byte load (and the array aligned)
};

constexpr std::size_t act_q8_blocks(std::size_t K) { return K / 32; }

constexpr std::size_t act_q8_hdr_offset(std::size_t K) {
  return (K + 15) & ~std::size_t(15);
}

constexpr std::size_t act_q8_bytes(std::size_t K) {
  return act_q8_hdr_offset(K) + sizeof(ActQ8Hdr) * (K / 32);
}

// Backend/scalar gate for the portable int8-dot path: any backend, fp32/bf16
// activations. Used both by the launcher and by the public capability query,
// so a caller can decide up front whether staging is worth allocating.
template <class ExecSpace, class Scalar>
inline constexpr bool available =
    (std::is_same_v<Scalar, float> ||
     std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>);

// Block-aligned K for each implemented dtype (the portable unit path's
// rules; see Impl::quant_can_unit).
constexpr bool dtype_ok(Wd dt, std::size_t K) {
  if (K == 0 || K % 32 != 0) return false;
  switch (dt) {
    case Wd::Q4K:
    case Wd::Q5K:
    case Wd::Q6K: return K % 256 == 0;
    case Wd::Q80: return true;
  }
  return false;
}

// Portable fp16 -> float (bit-exact copy of Gguf::Impl::fp16_to_float; kept
// local so this header stays standalone).
KOKKOS_INLINE_FUNCTION float mvq_h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = ((uint32_t)h & 0x7C00u) >> 10;
  uint32_t mant = (uint32_t)h & 0x03FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while (!(mant & 0x0400u)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FFu;
      f = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

// Portable copy of Gguf::Impl::get_scale_min_k4 (ggml-quants.c). One call per
// unit (not per element): each thread unpacks its own unit's pair(s), so no
// cross-lane sharing is needed.
KOKKOS_INLINE_FUNCTION void mvq_scale_min_k4(int j, const uint8_t* scales,
                                             uint8_t& sc, uint8_t& mn) {
  if (j < 4) {
    sc = (uint8_t)(scales[j] & 63u);
    mn = (uint8_t)(scales[j + 4] & 63u);
  } else {
    sc = (uint8_t)((scales[j + 4] & 0xFu) | ((scales[j - 4] >> 6) << 4));
    mn = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
  }
}

// Stage 1: quantize one activation row into int8 blocks. One RangePolicy
// iteration per 32-element block; bit-identical to the old CUDA quantizer
// (d = amax/127, q = rint(x/d) clamped to [-127, 127], half sums).
template <class ExecSpace, class Scalar>
struct QuantizeActQ8Functor {
  const Scalar* x;
  int8_t* xq;
  ActQ8Hdr* xh;
  int nblk;

  KOKKOS_INLINE_FUNCTION void operator()(int b) const {
    const size_t base = (size_t)b * 32;
    float amax        = 0.0f;
    for (int i = 0; i < 32; ++i) {
      const float v = static_cast<float>(x[base + (size_t)i]);
      const float a = v >= 0.0f ? v : -v;
      if (a > amax) amax = a;
    }
    // d = amax/127 keeps every |q| <= 127; an all-zero block quantizes to
    // q == 0 with d == 0 (the weight-side scale, not d, carries the value).
    const float d   = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int sum_lo = 0, sum_hi = 0;
    for (int i = 0; i < 32; ++i) {
      const float v = static_cast<float>(x[base + (size_t)i]);
      // rint: round-to-nearest-even, matching __float2int_rn and the host
      // nearbyint reference bit for bit.
      int q = (int)Kokkos::rint(v * inv);
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      xq[base + (size_t)i] = (int8_t)q;
      if (i < 16)
        sum_lo += q;
      else
        sum_hi += q;
    }
    xh[b].d      = d;
    xh[b].sum_lo = sum_lo;
    xh[b].sum_hi = sum_hi;
    xh[b].pad    = 0;
  }
};

// Stage 2: int8-dot GEMV, one team per output row. The team's threads split
// the row's decode units (TeamThreadRange) and Kokkos reduces them; each
// thread's unit dot is straight-line word-level integer math through
// Impl::IntDot (one dp4a-class dot per 4-byte word where the backend
// provides it, exact scalar arithmetic elsewhere). This mirrors the old
// one-warp-per-row schedule in portable terms -- the old kernel's 32 warp
// lanes are the team's threads here, which is why there is no nested vector
// level, no team scratch and no barrier: a nested ThreadVectorRange per unit
// measured as pure overhead against this layout. Same FP32 terms per row as
// the portable unit path, reassociated across the row's threads only.
//
// Unpack math is word-parallel throughout: every weight byte the dot consumes
// is non-negative (4/5/6-bit quants top out at 0x0F/0x1F/0x3F, high bit
// clear), so signed dp4a lanes agree with the scalar path by construction;
// genuinely signed payloads (int8 activations, Q8_0 weights) are exactly
// what both paths compute.
// One decode unit's activation window, loaded once and shared by several
// output rows. `w[0..3]` is the unit's first 16-element int8 window and
// `w[4..7]` the second (Q4_K/Q6_K: non-contiguous halves; Q5_K/Q8_0: the
// contiguous upper half). d0/d1 are the matching activation block scales and
// sum0/sum1 the precomputed half sums, so a multi-row dot never re-reads or
// re-derives the activation side.
struct Int8UnitWindow {
  int32_t w[8];
  float d0, d1;
  int32_t sum0, sum1;
};

// One aligned 16-byte chunk (four int8 words). The activation staging buffer
// and every Q4_K/Q5_K weight window are 16-byte aligned, so these become one
// vector load instead of four scalar ones.
struct alignas(16) Int8Words4 {
  int32_t w[4];
};

KOKKOS_INLINE_FUNCTION Int8Words4 mvq_load_words4(const void* p) {
  return *static_cast<const Int8Words4*>(p);
}

template <class ExecSpace, Wd DT>
KOKKOS_INLINE_FUNCTION Int8UnitWindow mvq_load_unit_window(const int8_t* xq,
                                                           const ActQ8Hdr* xh,
                                                           int c) {
  Int8UnitWindow x;
  if constexpr (DT == Wd::Q4K) {
    const int blk  = c >> 3;
    const int iter = (c >> 1) & 3;
    const int half = c & 1;
    const int k0   = (blk << 8) + (iter << 6) + (half << 4);
    const int b0   = k0 >> 5;
    const Int8Words4 a = mvq_load_words4(xq + k0);
    const Int8Words4 b = mvq_load_words4(xq + k0 + 32);
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 4; ++i) {
      x.w[i]     = a.w[i];
      x.w[4 + i] = b.w[i];
    }
    x.d0   = xh[b0].d;
    x.d1   = xh[b0 + 1].d;
    x.sum0 = (half != 0) ? xh[b0].sum_hi : xh[b0].sum_lo;
    x.sum1 = (half != 0) ? xh[b0 + 1].sum_hi : xh[b0 + 1].sum_lo;
  } else if constexpr (DT == Wd::Q5K) {
    const int blk  = c >> 3;
    const int sub  = c & 7;
    const int iter = sub >> 1;
    const bool low = (sub & 1) == 0;
    const int k0   = (blk << 8) + (iter << 6) + (low ? 0 : 32);
    const Int8Words4 a = mvq_load_words4(xq + k0);
    const Int8Words4 b = mvq_load_words4(xq + k0 + 16);
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 4; ++i) {
      x.w[i]     = a.w[i];
      x.w[4 + i] = b.w[i];
    }
    const int hb    = k0 >> 5;
    x.d0            = xh[hb].d;
    x.d1            = x.d0;
    x.sum0          = xh[hb].sum_lo + xh[hb].sum_hi;
    x.sum1          = x.sum0;
  } else if constexpr (DT == Wd::Q6K) {
    const int blk   = c >> 3;
    const int chunk = (c >> 2) & 1;
    const int pair  = (c >> 1) & 1;
    const int half  = c & 1;
    const int k0 = (blk << 8) + (chunk << 7) + (pair << 5) + (half << 4);
    const int b0 = k0 >> 5;
    const Int8Words4 a = mvq_load_words4(xq + k0);
    const Int8Words4 b = mvq_load_words4(xq + k0 + 64);
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 4; ++i) {
      x.w[i]     = a.w[i];
      x.w[4 + i] = b.w[i];
    }
    x.d0   = xh[b0].d;
    x.d1   = xh[b0 + 2].d;
    x.sum0 = (half != 0) ? xh[b0].sum_hi : xh[b0].sum_lo;
    x.sum1 = (half != 0) ? xh[b0 + 2].sum_hi : xh[b0 + 2].sum_lo;
  } else {  // Wd::Q80: one 32-element block, int8 payload.
    const int k0 = c << 5;
    const Int8Words4 a = mvq_load_words4(xq + k0);
    const Int8Words4 b = mvq_load_words4(xq + k0 + 16);
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 4; ++i) {
      x.w[i]     = a.w[i];
      x.w[4 + i] = b.w[i];
    }
    x.d0   = xh[c].d;
    x.d1   = x.d0;
    x.sum0 = 0;
    x.sum1 = 0;
  }
  return x;
}

// Dot one packed weight unit against an already-loaded activation window. The
// weight-side scale/min unpack stays per row (it is row data), but the
// activation words, block scales and half sums are shared across the ROWS
// output rows a row-blocked team handles.
template <class ExecSpace, Wd DT>
KOKKOS_INLINE_FUNCTION float gemv_int8_unit_dot_win(
    const uint8_t* row, int c, const Int8UnitWindow& x) {
  using Dot = ::KokkosDNN::Impl::IntDot<ExecSpace>;
  if constexpr (DT == Wd::Q4K) {
    const int blk  = c >> 3;
    const int iter = (c >> 1) & 3;
    const int half = c & 1;
    const BlockQ4K* b =
        reinterpret_cast<const BlockQ4K*>(row) + blk;
    const uint8_t* qw = b->qs + (size_t)iter * 32 + (size_t)half * 16;
    uint8_t sc0, mn0, sc1, mn1;
    mvq_scale_min_k4(2 * iter, b->scales, sc0, mn0);
    mvq_scale_min_k4(2 * iter + 1, b->scales, sc1, mn1);
    const float dw = mvq_h2f(b->d);
    const float dm = mvq_h2f(b->dmin);
    const float fa = dw * (float)sc0 * x.d0;
    const float ma = dm * (float)mn0 * x.d0;
    const float fb = dw * (float)sc1 * x.d1;
    const float mb = dm * (float)mn1 * x.d1;
    int dotA = 0, dotB = 0;
    const Int8Words4 q = mvq_load_words4(qw);
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 4; ++i) {
      const uint32_t w = (uint32_t)q.w[i];
      dotA = Dot::dot4((int32_t)(w & 0x0F0F0F0Fu), x.w[i], dotA);
      dotB = Dot::dot4((int32_t)((w >> 4) & 0x0F0F0F0Fu), x.w[4 + i], dotB);
    }
    return fa * (float)dotA - ma * (float)x.sum0 + fb * (float)dotB -
           mb * (float)x.sum1;
  } else if constexpr (DT == Wd::Q5K) {
    const int blk  = c >> 3;
    const int sub  = c & 7;
    const int iter = sub >> 1;
    const bool low = (sub & 1) == 0;
    const BlockQ5K* b =
        reinterpret_cast<const BlockQ5K*>(row) + blk;
    const uint8_t* ql = b->qs + (size_t)iter * 32;
    const uint8_t* qh = b->qh;
    uint8_t sc, mn;
    mvq_scale_min_k4(sub, b->scales, sc, mn);
    const float dw = mvq_h2f(b->d) * (float)sc * x.d0;
    const float dm = mvq_h2f(b->dmin) * (float)mn * x.d0;
    const uint32_t sh    = low ? 0u : 4u;
    const uint32_t bitsh = (uint32_t)(2 * iter + (low ? 0 : 1));
    const Int8Words4 ql0 = mvq_load_words4(ql);
    const Int8Words4 ql1 = mvq_load_words4(ql + 16);
    const Int8Words4 qh0 = mvq_load_words4(qh);
    const Int8Words4 qh1 = mvq_load_words4(qh + 16);
    int dot = 0;
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 4; ++i) {
      const uint32_t qlw0 = (uint32_t)ql0.w[i];
      const uint32_t qlw1 = (uint32_t)ql1.w[i];
      const uint32_t qhw0 = (uint32_t)qh0.w[i];
      const uint32_t qhw1 = (uint32_t)qh1.w[i];
      const uint32_t w0 =
          ((qlw0 >> sh) & 0x0F0F0F0Fu) | (((qhw0 >> bitsh) & 0x01010101u) << 4);
      const uint32_t w1 =
          ((qlw1 >> sh) & 0x0F0F0F0Fu) | (((qhw1 >> bitsh) & 0x01010101u) << 4);
      dot = Dot::dot4((int32_t)w0, x.w[i], dot);
      dot = Dot::dot4((int32_t)w1, x.w[4 + i], dot);
    }
    return dw * (float)dot - dm * (float)x.sum0;
  } else if constexpr (DT == Wd::Q6K) {
    const int blk   = c >> 3;
    const int chunk = (c >> 2) & 1;
    const int pair  = (c >> 1) & 1;
    const int half  = c & 1;
    const BlockQ6K* b =
        reinterpret_cast<const BlockQ6K*>(row) + blk;
    const uint8_t* ql =
        b->ql + (size_t)chunk * 64 + (size_t)pair * 32 + (size_t)half * 16;
    const uint8_t* qh = b->qh + (size_t)chunk * 32 + (size_t)half * 16;
    const float d     = mvq_h2f(b->d);
    const float s0 =
        d * (float)b->scales[(size_t)chunk * 8 + (size_t)2 * (size_t)pair +
                            (size_t)half] *
        x.d0;
    const float s1 =
        d * (float)b->scales[(size_t)chunk * 8 + (size_t)2 * (size_t)pair +
                            (size_t)half + 4] *
        x.d1;
    const uint32_t sh = (pair != 0) ? 2u : 0u;
    int dotA = 0, dotB = 0;
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 4; ++i) {
      const uint32_t qlw = (uint32_t)Dot::load_u16(ql + i * 4);
      const uint32_t qhw = (uint32_t)Dot::load_u16(qh + i * 4);
      const uint32_t wA =
          (qlw & 0x0F0F0F0Fu) | ((((qhw >> sh) & 0x03030303u)) << 4);
      const uint32_t wB = ((qlw >> 4) & 0x0F0F0F0Fu) |
                          ((((qhw >> (sh + 4u)) & 0x03030303u)) << 4);
      dotA = Dot::dot4((int32_t)wA, x.w[i], dotA);
      dotB = Dot::dot4((int32_t)wB, x.w[4 + i], dotB);
    }
    return s0 * (float)(dotA - (x.sum0 << 5)) +
           s1 * (float)(dotB - (x.sum1 << 5));
  } else {  // Wd::Q80: one 32-element block, one scale, int8 payload.
    const BlockQ80* b = reinterpret_cast<const BlockQ80*>(row) + c;
    const float dw    = mvq_h2f(b->d) * x.d0;
    int dot           = 0;
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 8; ++i)
      dot = Dot::dot4(Dot::load_u16(b->qs + i * 4), x.w[i], dot);
    return dw * (float)dot;
  }
}

template <class ExecSpace, Wd DT>
KOKKOS_INLINE_FUNCTION float gemv_int8_unit_dot(const uint8_t* row, int c,
                                                const int8_t* xq,
                                                const ActQ8Hdr* xh) {
  return gemv_int8_unit_dot_win<ExecSpace, DT>(
      row, c, mvq_load_unit_window<ExecSpace, DT>(xq, xh, c));
}

// Fallback for layouts the blocked kernel cannot field (team-size cap below
// ROWS, or a host backend with a 1-thread team and ROWS > 1): one team per
// output row, decode units reduced over the team's threads. No team scratch
// and no barrier: Kokkos owns the reduction.
template <class ExecSpace, class Scalar, Wd DT>
struct GemvInt8Functor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  const uint8_t* wdata;
  size_t row_bytes;
  const int8_t* xq;
  const ActQ8Hdr* xh;
  Scalar* y;
  int nunits;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const uint8_t* r = wdata + (size_t)team.league_rank() * row_bytes;
    const int8_t* xql   = xq;
    const ActQ8Hdr* xhl = xh;
    const int nu        = nunits;
    float sum           = 0.0f;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nu),
        KOKKOS_LAMBDA(const int c, float& s) {
          s += gemv_int8_unit_dot<ExecSpace, DT>(r, c, xql, xhl);
        },
        sum);
    if (team.team_rank() == 0)
      y[team.league_rank()] = static_cast<Scalar>(sum);
  }
};

// One team per row with a thread count that fits the backend and the row:
// wide enough to keep short rows' reduction inside a few warps, clamped to
// the row's unit count and the backend's team-size cap (host backends run
// team_size 1, i.e. a serial loop -- still correct).
template <class ExecSpace, class Scalar, Wd DT>
inline void launch_gemv_dt(const uint8_t* wdata, size_t row_bytes, size_t rows,
                           const int8_t* xq, const ActQ8Hdr* xh, Scalar* y,
                           int nunits) {
  GemvInt8Functor<ExecSpace, Scalar, DT> f{wdata, row_bytes, xq, xh, y,
                                           nunits};
  const int tmax = ::KokkosDNN::Impl::cached_team_size_max<ExecSpace>(f);
  int ts         = 1;
  while (ts < 128 && ts < nunits) ts *= 2;
  if (ts > tmax) ts = tmax > 0 ? tmax : 1;
  Kokkos::TeamPolicy<ExecSpace> policy((int)rows, ts);
  Kokkos::parallel_for("KokkosDNN::GgufMvq::gemv_int8", policy, f);
}

// Row-blocked int8 GEMV: each thread group handles ROWS output rows and keeps
// ROWS accumulators for the same unit column, so the quantized activation
// window (32 int8 values + its two block headers) is loaded once per ROWS
// weight units instead of once per row. That is the cross-row reuse the
// one-team-per-row kernel lacks; it also amortizes the activation quantize's
// global reads over the whole row block. `Dt` is a compile-time dtype (no
// per-unit switch), tpr/groups are runtime so one instantiation per ROWS
// covers every shape.
template <class ExecSpace, class Scalar, Wd DT, int ROWS>
struct GemvInt8BlockedFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  const uint8_t* wdata;
  size_t row_bytes;
  const int8_t* xq;
  const ActQ8Hdr* xh;
  Scalar* y;
  int nunits;
  size_t nrows;
  int threads_per_row;
  int groups;  // row groups per team (team_size == threads_per_row * groups)

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const int tpr  = threads_per_row;
    const int gs   = groups;
    const int rank = team.team_rank();
    const int g    = (tpr > 0) ? (rank / tpr) : 0;
    const int lane = (tpr > 0) ? (rank - g * tpr) : 0;
    float* red     = static_cast<float*>(
        team.team_shmem().get_shmem(sizeof(float) * (size_t)team.team_size() *
                                    (size_t)ROWS));
    const int8_t* xql   = xq;
    const ActQ8Hdr* xhl = xh;
    const uint8_t* wd   = wdata;
    const int nu        = nunits;
    const size_t rb     = row_bytes;
    const size_t nr     = nrows;
    // Tail teams clamp their row indices; rows past the end recompute the last
    // real row and their result is dropped.
    const size_t first =
        ((size_t)team.league_rank() * (size_t)gs + (size_t)g) * (size_t)ROWS;
    const uint8_t* row[ROWS];
KOKKOSDNN_PRAGMA_UNROLL()
    for (int k = 0; k < ROWS; ++k)
      row[k] = wd + ((first + (size_t)k < nr) ? (first + (size_t)k)
                                              : (nr - 1)) *
                       rb;
    float acc[ROWS];
KOKKOSDNN_PRAGMA_UNROLL()
    for (int k = 0; k < ROWS; ++k) acc[k] = 0.0f;
    for (int c = lane; c < nu; c += tpr) {
      const Int8UnitWindow xw =
          mvq_load_unit_window<ExecSpace, DT>(xql, xhl, c);
KOKKOSDNN_PRAGMA_UNROLL()
      for (int k = 0; k < ROWS; ++k)
        acc[k] += gemv_int8_unit_dot_win<ExecSpace, DT>(row[k], c, xw);
    }
    float* slot = red + (size_t)rank * (size_t)ROWS;
KOKKOSDNN_PRAGMA_UNROLL()
    for (int k = 0; k < ROWS; ++k) slot[k] = acc[k];
    team.team_barrier();
    const int rows_team = gs * ROWS;
    if (rank < rows_team) {
      const int gg   = rank / ROWS;
      const int kk   = rank - gg * ROWS;
      float sum      = 0.0f;
      const float* p = red + (size_t)gg * (size_t)tpr * (size_t)ROWS + kk;
      for (int t = 0; t < tpr; ++t) sum += p[(size_t)t * (size_t)ROWS];
      const size_t rowidx =
          ((size_t)team.league_rank() * (size_t)gs + (size_t)gg) *
              (size_t)ROWS +
          (size_t)kk;
      if (rowidx < nr) y[rowidx] = static_cast<Scalar>(sum);
    }
  }
};

// Explicit schedule knobs for tuning; zero fields mean "pick the tuned
// default for this shape". Kept in Impl so production call sites do not need
// to know the layout, while the bandwidth benchmark can sweep it.
struct Int8Layout {
  int rows_per_group = 0;
  int threads_per_row = 0;
  int groups = 0;
};

template <class ExecSpace, class Scalar, Wd DT, int ROWS>
inline void launch_gemv_int8_blocked(const uint8_t* wdata, size_t row_bytes,
                                     size_t rows, const int8_t* xq,
                                     const ActQ8Hdr* xh, Scalar* y,
                                     int nunits, int threads_per_row,
                                     int groups) {
  if (rows == 0) return;
  GemvInt8BlockedFunctor<ExecSpace, Scalar, DT, ROWS> f{
      wdata, row_bytes, xq, xh, y, nunits, rows, threads_per_row, groups};
  const std::size_t shmem = sizeof(float) * (std::size_t)threads_per_row *
                            (std::size_t)groups * (std::size_t)ROWS;
  const int tmax =
      ::KokkosDNN::Impl::cached_team_size_max<ExecSpace>(f, shmem);
  int tpr = threads_per_row;
  int gs  = groups;
  // Never field more threads than the row has decode units.
  while (tpr > ROWS && (size_t)tpr > (size_t)nunits) tpr /= 2;
  while (tpr > 1 && tpr * gs > tmax) tpr /= 2;
  if (tpr * gs > tmax) gs = (tmax / tpr > 0) ? (tmax / tpr) : 1;
  if (tmax < 1 || tpr < ROWS || tpr * gs > tmax) {
    // No blocked team fits (team-size cap below ROWS, e.g. a 1-thread host
    // backend with ROWS > 1): fall back to the one-team-per-row kernel, which
    // runs correctly with team_size 1.
    launch_gemv_dt<ExecSpace, Scalar, DT>(wdata, row_bytes, rows, xq, xh, y,
                                          nunits);
    return;
  }
  f.threads_per_row = tpr;
  f.groups          = gs;
  const int ts      = tpr * gs;
  const std::size_t shmem_ts =
      sizeof(float) * (std::size_t)ts * (std::size_t)ROWS;
  const std::size_t rows_per_team = (std::size_t)gs * (std::size_t)ROWS;
  const int teams = (int)((rows + rows_per_team - 1) / rows_per_team);
  Kokkos::TeamPolicy<ExecSpace> policy(teams, ts);
  policy.set_scratch_size(0, Kokkos::PerTeam(shmem_ts));
  Kokkos::parallel_for("KokkosDNN::GgufMvq::gemv_int8_blocked", policy, f);
}

// Rows per thread group for the blocked int8 kernel. Two rows per group is
// the measured sweet spot for every wide decode shape: it halves activation
// window re-reads, and four rows no longer buy enough reuse to pay for the
// bigger per-thread accumulator/register footprint (see the per-shape table
// from benchmarks/Bench_GGUFGemv.cpp). Short matrices are team-count bound,
// so they keep one row per group.
inline int gemv_int8_rows_per_group(size_t rows) {
  return rows >= 1024 ? 2 : 1;
}

// 128 threads per ROWS-row group (four warps, i.e. 64 threads per row at the
// default ROWS=2), one row group per team. The launcher shrinks
// threads_per_row for short rows and the backend's team-size cap, so this is
// an upper bound rather than a hard layout.
inline void gemv_int8_threads(size_t /*nunits*/, int& tpr, int& groups) {
  tpr    = 128;
  groups = 1;
}

template <class ExecSpace, class Scalar, Wd DT>
inline void launch_gemv_int8_dispatch(const uint8_t* wdata, size_t row_bytes,
                                      size_t rows, const int8_t* xq,
                                      const ActQ8Hdr* xh, Scalar* y,
                                      int nunits, Int8Layout layout) {
  const int rpg = layout.rows_per_group > 0
                      ? layout.rows_per_group
                      : gemv_int8_rows_per_group(rows);
  int tpr       = layout.threads_per_row;
  int groups    = layout.groups;
  if (tpr <= 0 || groups <= 0) {
    int default_tpr = 0, default_groups = 0;
    gemv_int8_threads((size_t)nunits, default_tpr, default_groups);
    if (tpr <= 0) tpr = default_tpr;
    if (groups <= 0) groups = default_groups;
  }
  if (rpg >= 8)
    launch_gemv_int8_blocked<ExecSpace, Scalar, DT, 8>(
        wdata, row_bytes, rows, xq, xh, y, nunits, tpr, groups);
  else if (rpg >= 4)
    launch_gemv_int8_blocked<ExecSpace, Scalar, DT, 4>(
        wdata, row_bytes, rows, xq, xh, y, nunits, tpr, groups);
  else if (rpg == 2)
    launch_gemv_int8_blocked<ExecSpace, Scalar, DT, 2>(
        wdata, row_bytes, rows, xq, xh, y, nunits, tpr, groups);
  else
    launch_gemv_int8_blocked<ExecSpace, Scalar, DT, 1>(
        wdata, row_bytes, rows, xq, xh, y, nunits, tpr, groups);
}


// Stage 1 + stage 2 with portable Kokkos launches. Returns false when the
// shape/dtype/scratch does not fit this path; the caller then runs the
// portable kernel, so a declined launch can never change results.
template <class ExecSpace, class Scalar>
bool launch_quant_gemv_int8(const Kokkos::View<Scalar*, ExecSpace>& x,
                            const uint8_t* wdata, Wd dt, size_t rows,
                            size_t cols, size_t row_bytes,
                            const Kokkos::View<Scalar*, ExecSpace>& y,
                            uint8_t* scratch, size_t scratch_bytes,
                            Int8Layout layout = Int8Layout{}) {
  constexpr bool is_f = std::is_same_v<Scalar, float>;
  constexpr bool is_b =
      std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>;
  if constexpr (!is_f && !is_b) {
    return false;
  } else {
    if (wdata == nullptr || scratch == nullptr) return false;
    if (!dtype_ok(dt, cols)) return false;
    if (rows == 0 || rows > (size_t)(std::numeric_limits<int>::max()))
      return false;
    if (scratch_bytes < act_q8_bytes(cols)) return false;
    // The kernels index the activation by element (contiguous row).
    if (x.stride(0) != 1 || y.stride(0) != 1) return false;
    // Alignment contract of the loaders below: the staging base and the
    // Q4_K/Q5_K weight windows use 16-byte loads (144/176-byte blocks keep
    // every window 16-aligned), so both need 16-byte alignment. Q6_K/Q8_0
    // weight windows use load_u16 (two 16-bit loads) and only need their
    // even block offsets plus an even wdata base. Decline instead of
    // faulting when a caller hands us a misaligned buffer -- the portable
    // fallback has no alignment requirements. In practice Kokkos allocations
    // always satisfy this, so the check never fires.
    if ((reinterpret_cast<uintptr_t>(scratch) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(wdata) & 1u) != 0u) return false;
    if ((dt == Wd::Q4K || dt == Wd::Q5K) &&
        (reinterpret_cast<uintptr_t>(wdata) & 15u) != 0u)
      return false;
    const size_t nunits = act_q8_blocks(cols);
    if (nunits > (size_t)(std::numeric_limits<int>::max)()) return false;

    uint8_t* const sbase = scratch;
    int8_t* const xq     = reinterpret_cast<int8_t*>(sbase);
    ActQ8Hdr* const xh =
        reinterpret_cast<ActQ8Hdr*>(sbase + act_q8_hdr_offset(cols));
    const int nblk = (int)nunits;
    QuantizeActQ8Functor<ExecSpace, Scalar> qf{x.data(), xq, xh, nblk};
    Kokkos::parallel_for("KokkosDNN::GgufMvq::quantize_act_q8",
                         Kokkos::RangePolicy<ExecSpace>(0, nblk), qf);

    const int nu = (int)nunits;
    switch (dt) {
      case Wd::Q4K:
        launch_gemv_int8_dispatch<ExecSpace, Scalar, Wd::Q4K>(
            wdata, row_bytes, rows, xq, xh, y.data(), nu, layout);
        break;
      case Wd::Q5K:
        launch_gemv_int8_dispatch<ExecSpace, Scalar, Wd::Q5K>(
            wdata, row_bytes, rows, xq, xh, y.data(), nu, layout);
        break;
      case Wd::Q6K:
        launch_gemv_int8_dispatch<ExecSpace, Scalar, Wd::Q6K>(
            wdata, row_bytes, rows, xq, xh, y.data(), nu, layout);
        break;
      case Wd::Q80:
        launch_gemv_int8_dispatch<ExecSpace, Scalar, Wd::Q80>(
            wdata, row_bytes, rows, xq, xh, y.data(), nu, layout);
        break;
    }
    return true;
  }
}

}  // namespace GgufMvq
}  // namespace Impl
}  // namespace KokkosDNN

#endif
