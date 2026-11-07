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
//   2. GemvInt8Functor<DT>: one team per output row (the portable spelling
//      of one warp per row). The team's threads split the row's decode units
//      (TeamThreadRange) under a Kokkos parallel_reduce; each thread's unit
//      dot is straight-line word-level integer math through Impl::IntDot
//      (one dp4a-class dot per 4-byte word where the backend provides it,
//      exact scalar arithmetic elsewhere). No team scratch, no barrier, no
//      nested vector level: the old schedule's 32 warp lanes are the team's
//      threads, and a nested ThreadVectorRange per unit measured as pure
//      overhead against this layout.
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

#include "KokkosDNN_config.h"
#include "KokkosDNN_IntDot.hpp"
#include "KokkosDNN_Occupancy.hpp"

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
template <class ExecSpace, Wd DT>
KOKKOS_INLINE_FUNCTION float gemv_int8_unit_dot(const uint8_t* row, int c,
                                                const int8_t* xq,
                                                const ActQ8Hdr* xh) {
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
    const int k0   = (blk << 8) + (iter << 6) + (half << 4);
    const int b0   = k0 >> 5;
    const float fa = dw * (float)sc0 * xh[b0].d;
    const float ma = dm * (float)mn0 * xh[b0].d;
    const float fb = dw * (float)sc1 * xh[b0 + 1].d;
    const float mb = dm * (float)mn1 * xh[b0 + 1].d;
    const int sumA = (half != 0) ? xh[b0].sum_hi : xh[b0].sum_lo;
    const int sumB = (half != 0) ? xh[b0 + 1].sum_hi : xh[b0 + 1].sum_lo;
    // Two 16-element dots (low / high nibble windows), 4 words each.
    int dotA = 0, dotB = 0;
    for (int i = 0; i < 4; ++i) {
      const uint32_t w = (uint32_t)Dot::load(qw + i * 4);
      dotA = Dot::dot4((int32_t)(w & 0x0F0F0F0Fu),
                       Dot::load(xq + k0 + i * 4), dotA);
      dotB = Dot::dot4((int32_t)((w >> 4) & 0x0F0F0F0Fu),
                       Dot::load(xq + k0 + 32 + i * 4), dotB);
    }
    return fa * (float)dotA - ma * (float)sumA + fb * (float)dotB -
           mb * (float)sumB;
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
    const int k0    = (blk << 8) + (iter << 6) + (low ? 0 : 32);
    const float dw  = mvq_h2f(b->d) * (float)sc * xh[k0 >> 5].d;
    const float dm  = mvq_h2f(b->dmin) * (float)mn * xh[k0 >> 5].d;
    const int sum   = xh[k0 >> 5].sum_lo + xh[k0 >> 5].sum_hi;
    const uint32_t sh    = low ? 0u : 4u;
    const uint32_t bitsh = (uint32_t)(2 * iter + (low ? 0 : 1));
    int dot = 0;
    for (int i = 0; i < 8; ++i) {
      const uint32_t qlw = (uint32_t)Dot::load(ql + i * 4);
      const uint32_t qhw = (uint32_t)Dot::load(qh + i * 4);
      const uint32_t w =
          ((qlw >> sh) & 0x0F0F0F0Fu) | (((qhw >> bitsh) & 0x01010101u) << 4);
      dot = Dot::dot4((int32_t)w, Dot::load(xq + k0 + i * 4),
                      dot);
    }
    return dw * (float)dot - dm * (float)sum;
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
    const int k0 = (blk << 8) + (chunk << 7) + (pair << 5) + (half << 4);
    const int b0 = k0 >> 5;
    // Both windows start at the same offset within their 32-element block
    // (k0 and k0+64 are congruent mod 32), so they read the same half sum.
    const int sumA = (half != 0) ? xh[b0].sum_hi : xh[b0].sum_lo;
    const int sumB = (half != 0) ? xh[b0 + 2].sum_hi : xh[b0 + 2].sum_lo;
    const float s0 =
        d * (float)b->scales[(size_t)chunk * 8 + (size_t)2 * (size_t)pair +
                            (size_t)half] *
        xh[b0].d;
    const float s1 =
        d * (float)b->scales[(size_t)chunk * 8 + (size_t)2 * (size_t)pair +
                            (size_t)half + 4] *
        xh[b0 + 2].d;
    const uint32_t sh = (pair != 0) ? 2u : 0u;
    int dotA = 0, dotB = 0;
    for (int i = 0; i < 4; ++i) {
      const uint32_t qlw = (uint32_t)Dot::load_u(ql + i * 4);
      const uint32_t qhw = (uint32_t)Dot::load_u(qh + i * 4);
      const uint32_t wA =
          (qlw & 0x0F0F0F0Fu) | ((((qhw >> sh) & 0x03030303u)) << 4);
      const uint32_t wB = ((qlw >> 4) & 0x0F0F0F0Fu) |
                          ((((qhw >> (sh + 4u)) & 0x03030303u)) << 4);
      dotA = Dot::dot4((int32_t)wA, Dot::load(xq + k0 + i * 4),
                       dotA);
      dotB = Dot::dot4((int32_t)wB,
                       Dot::load(xq + k0 + 64 + i * 4), dotB);
    }
    // 6-bit quants are stored biased by +32; subtract 32 * sum instead of
    // biasing every weight byte.
    return s0 * (float)(dotA - (sumA << 5)) + s1 * (float)(dotB - (sumB << 5));
  } else {  // Wd::Q80: one 32-element block, one scale, int8 payload.
    const BlockQ80* b = reinterpret_cast<const BlockQ80*>(row) + c;
    const float dw    = mvq_h2f(b->d) * xh[c].d;
    const int k0      = c << 5;
    int dot           = 0;
    for (int i = 0; i < 8; ++i) {
      dot = Dot::dot4(Dot::load_u(b->qs + i * 4),
                      Dot::load(xq + k0 + i * 4), dot);
    }
    return dw * (float)dot;
  }
}

// One team per output row; the row's decode units are reduced over the team's
// threads (the portable spelling of one warp per row). No team scratch and no
// barrier: Kokkos owns the reduction.
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


// Stage 1 + stage 2 with portable Kokkos launches. Returns false when the
// shape/dtype/scratch does not fit this path; the caller then runs the
// portable kernel, so a declined launch can never change results.
template <class ExecSpace, class Scalar>
bool launch_quant_gemv_int8(const Kokkos::View<Scalar*, ExecSpace>& x,
                            const uint8_t* wdata, Wd dt, size_t rows,
                            size_t cols, size_t row_bytes,
                            const Kokkos::View<Scalar*, ExecSpace>& y,
                            uint8_t* scratch, size_t scratch_bytes) {
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
    // IntDot::load needs aligned addresses: the staging base (16-multiple
    // offsets into it) and, for Q4_K/Q5_K whose weight windows also use wide
    // loads, 16-aligned packed rows (144/176-byte blocks keep every window
    // 16-aligned). Q6_K/Q8_0 weight windows use the unaligned-safe loader,
    // so only the staging base matters there. Decline instead of faulting
    // when a caller hands us a misaligned buffer -- the portable fallback
    // has no alignment requirements. In practice Kokkos allocations always
    // satisfy this, so the check never fires.
    if ((reinterpret_cast<uintptr_t>(scratch) & 15u) != 0u) return false;
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
        launch_gemv_dt<ExecSpace, Scalar, Wd::Q4K>(wdata, row_bytes, rows, xq,
                                                  xh, y.data(), nu);
        break;
      case Wd::Q5K:
        launch_gemv_dt<ExecSpace, Scalar, Wd::Q5K>(wdata, row_bytes, rows, xq,
                                                  xh, y.data(), nu);
        break;
      case Wd::Q6K:
        launch_gemv_dt<ExecSpace, Scalar, Wd::Q6K>(wdata, row_bytes, rows, xq,
                                                  xh, y.data(), nu);
        break;
      case Wd::Q80:
        launch_gemv_dt<ExecSpace, Scalar, Wd::Q80>(wdata, row_bytes, rows, xq,
                                                  xh, y.data(), nu);
        break;
    }
    return true;
  }
}

}  // namespace GgufMvq
}  // namespace Impl
}  // namespace KokkosDNN

#endif
