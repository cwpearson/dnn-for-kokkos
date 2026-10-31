// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GGUFGEMVCUDA_HPP
#define KOKKOSDNN_GGUFGEMVCUDA_HPP

// CUDA int8-dot GEMV for packed GGUF weights at M == 1 (decode), llama.cpp
// MMVQ style: the activation row is quantized once into int8 blocks
// ({d, per-half sums, 32 x int8}) and every output row then runs 4 `__dp4a`
// (4-element int8 dot) instructions per 32 weight elements instead of the
// portable path's nibble-extract + int->float + fp16 convert + FMA chain.
//
// Why this exists (measured on the RTX 3090 decode profile): the portable
// unit-dot path spends most of its instructions on code no memory system can
// keep up with -- per 32 weight elements it converts nibbles to ints, then to
// floats, folds in two fp16 superblock scales, and runs 32 FMAs, while the
// packed payload is only 18 bytes. Decode is therefore instruction-bound, not
// bandwidth-bound, and on a power-capped part (180 W here) instruction count
// *is* energy: the SM clock governor lowers clocks to hold the cap (735 ->
// 540 MHz in the A/B below), so wall time tracks instructions per token.
// This kernel's Q4_K loop is 112 SASS instructions per 32 elements (224 per
// 2-unit iteration, ~3.5 per element) and its dots are `__dp4a` (4 int8
// multiply-adds per instruction); decoding Q4_K_M on the same part went from
// 89 to 53 ms/token (11.2 -> 18.4 tok/s), with the clock dropping from 735 to
// 540 MHz at the same 180 W cap. The remaining cost per element is the 6-bit
// (scale, min) unpack (four byte loads + two int->float per unit) and the
// activation window loads, not the dots.
//
// Structure (two launches per GEMV, both on the caller's CUDA stream):
//   1. quantize_act_q8_kernel: one warp per 32-element activation block.
//      Warp max of |x| -> d = amax/127, q_i = rint(x_i/d) in [-127, 127],
//      plus the sum of each 16-element half (the bit the weight-side min /
//      Q6_K -32 terms need, so the dot kernel never re-derives them).
//      The row is quantized ONCE per GEMV and reused by all N output rows.
//   2. gemv_int8_kernel: one warp per output row, 4 rows per 128-thread
//      block (same row/warp mapping as QuantGemvUnitMultiFunctor). Each lane
//      walks the row's 32-element units with stride 32 and warp-shuffles the
//      partial sums at the end.
// The unit mapping is EXACTLY the portable one (Impl::quant_unit_count /
// quant_unit_dot: Q4_K unit = block c/8, iter (c%8)/2, half c%2; Q6_K unit =
// block c/8, chunk (c%8)/4, pair (c%4)/2, half c%2; Q5_K/Q8_0 unit = one
// 32-element sub-block), so a row is the same set of terms as before, just
// summed in a different order and with the activations rounded to int8.
//
// Numerics (documented, mirrors llama.cpp's default CUDA path):
//   * activations: per-32-element symmetric int8, d = amax/127 in FP32.
//     This is the only new error source: ~0.4% of the block amax per
//     element, i.e. a ~1% relative perturbation of the dot product, the
//     same trade llama.cpp's default (Q8_1 activations) makes. Weights keep
//     their on-disk 4/5/6-bit precision and their FP16 superblock scales,
//     and all block scaling stays in FP32 (no fp16 dyads as in llama's
//     `dm` packs, which is *better* than the reference CUDA path).
//   * accumulation: INT32 per 32-element dot, FP32 across units/threads.
//
// Everything here is CUDA-only (dp4a plus warp intrinsics). The portable
// path in KokkosDNN_GGUF.hpp remains the fallback for every other backend,
// every unsupported dtype, and every non-block-aligned K, so this header can
// never change the result of a case it does not claim.
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "KokkosDNN_config.h"

#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDACC__)
#define KOKKOSDNN_IMPL_CUDA_MMVQ_AVAILABLE 1
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

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
// SoA keeps both sides vector-loadable: the 16-byte int4 activation window
// the dot kernel reads is 16-byte aligned because every unit's element offset
// is a multiple of 16.
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

// Backend/scalar gate for the dp4a path: CUDA + fp32/bf16 activations. Used
// both by the launcher and by the public capability query, so a caller can
// decide up front whether staging is worth allocating.
#if defined(KOKKOSDNN_IMPL_CUDA_MMVQ_AVAILABLE)
template <class ExecSpace, class Scalar>
inline constexpr bool available =
    std::is_same_v<ExecSpace, Kokkos::Cuda> &&
    (std::is_same_v<Scalar, float> ||
     std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>);
#else
template <class ExecSpace, class Scalar>
inline constexpr bool available = false;
#endif

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

#if defined(KOKKOSDNN_IMPL_CUDA_MMVQ_AVAILABLE)

namespace Kernels {

__device__ __forceinline__ float h2f(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

// get_scale_min_k4 (ggml-quants.c), branch-free in the block's 8 sub-blocks:
// 12 bytes hold 8 x (6-bit scale, 6-bit min).
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* scales,
                                             int& sc, int& mn) {
  if (j < 4) {
    sc = scales[j] & 63;
    mn = scales[j + 4] & 63;
  } else {
    sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// One 16-element window: `a` = 4 packed weight bytes (one element per byte)
// and `u` = the matching 16 int8 activations.
__device__ __forceinline__ int dot4(const int4& a, const int4& u) {
  const int t0 = __dp4a(a.x, u.x, 0);
  const int t1 = __dp4a(a.y, u.y, 0);
  const int t2 = __dp4a(a.z, u.z, 0);
  const int t3 = __dp4a(a.w, u.w, 0);
  return (t0 + t1) + (t2 + t3);
}

// 16 packed bytes behind eight 2-byte loads. Q6_K/Q5_K/Q8_0 blocks are
// 210/176/34 bytes, so a row's packed payload is only 2-byte aligned per
// block: 16-byte vector loads would fault on odd block indices. Natural
// 2-byte alignment is safe on every device, and the extra instructions are
// still far below the portable path's per-element unpack.
__device__ __forceinline__ int4 load_u16x8(const uint8_t* p) {
  const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
  int4 r;
  r.x = (int)q[0] | ((int)q[1] << 16);
  r.y = (int)q[2] | ((int)q[3] << 16);
  r.z = (int)q[4] | ((int)q[5] << 16);
  r.w = (int)q[6] | ((int)q[7] << 16);
  return r;
}

constexpr int kLaneMask = 0x0F0F0F0F;

// One 32-element unit of one packed row (see the header comment for the
// unit->element mapping, which is the portable one).
template <Wd DT>
__device__ __forceinline__ float unit_dot(const uint8_t* __restrict__ row,
                                          int c,
                                          const int8_t* __restrict__ xq,
                                          const ActQ8Hdr* __restrict__ xh) {
  if constexpr (DT == Wd::Q4K) {
    const int blk  = c >> 3;
    const int iter = (c >> 1) & 3;
    const int half = c & 1;
    const BlockQ4K* b = reinterpret_cast<const BlockQ4K*>(row) + blk;
    const int4 w =
        *reinterpret_cast<const int4*>(b->qs + iter * 32 + half * 16);
    // Unit base: block*256 + iter*64 + half*16. Window A (low nibbles of the
    // 16 packed bytes) starts there; window B (high nibbles) at +32.
    const int k0 = (blk << 8) + (iter << 6) + (half << 4);
    const int b0 = k0 >> 5;
    const ActQ8Hdr h0 = xh[b0], h1 = xh[b0 + 1];
    const int4 a  = *reinterpret_cast<const int4*>(xq + k0);
    const int4 bb = *reinterpret_cast<const int4*>(xq + k0 + 32);
    const int4 lo = {w.x & kLaneMask, w.y & kLaneMask, w.z & kLaneMask,
                     w.w & kLaneMask};
    const int4 hi = {(w.x >> 4) & kLaneMask, (w.y >> 4) & kLaneMask,
                     (w.z >> 4) & kLaneMask, (w.w >> 4) & kLaneMask};
    int sc0, mn0, sc1, mn1;
    scale_min_k4(2 * iter, b->scales, sc0, mn0);
    scale_min_k4(2 * iter + 1, b->scales, sc1, mn1);
    // value = (d*sc)*q - (dmin*mn), x ~= d8*q8: the min term is a constant
    // times the activation block sum, exactly as in the portable kernel, but
    // summed once per window instead of once per element.
    const float dw = h2f(b->d), dm = h2f(b->dmin);
    // Window A starts at k0 (k0 % 32 == 16*half) and window B at k0 + 32,
    // i.e. both are the SAME half of their 32-element activation block; the
    // min term needs that block half's int8 sum.
    const int sumA = half ? h0.sum_hi : h0.sum_lo;
    const int sumB = half ? h1.sum_hi : h1.sum_lo;
    const float fa = dw * (float)sc0 * h0.d;
    const float ma = dm * (float)mn0 * h0.d;
    const float fb = dw * (float)sc1 * h1.d;
    const float mb = dm * (float)mn1 * h1.d;
    const float acc = fmaf(fa, (float)dot4(lo, a), -ma * (float)sumA);
    return fmaf(fb, (float)dot4(hi, bb), acc - mb * (float)sumB);
  } else if constexpr (DT == Wd::Q5K) {
    // Unit = one 32-element sub-block (both nibble halves of the qs window,
    // high bit from qh bit 2*iter + half).
    const int blk  = c >> 3;
    const int sub  = c & 7;
    const int iter = sub >> 1;
    const bool low = (sub & 1) == 0;
    const BlockQ5K* b = reinterpret_cast<const BlockQ5K*>(row) + blk;
    const int4 ql0 = *reinterpret_cast<const int4*>(b->qs + iter * 32);
    const int4 ql1 = *reinterpret_cast<const int4*>(b->qs + iter * 32 + 16);
    const int4 qh0 = load_u16x8(b->qh);
    const int4 qh1 = load_u16x8(b->qh + 16);
    const int sh  = 2 * iter + (low ? 0 : 1);
    const int shn = low ? 0 : 4;  // low nibble half vs high nibble half
    const int4 bit0 = {(qh0.x >> sh) & 0x01010101, (qh0.y >> sh) & 0x01010101,
                       (qh0.z >> sh) & 0x01010101,
                       (qh0.w >> sh) & 0x01010101};
    const int4 bit1 = {(qh1.x >> sh) & 0x01010101, (qh1.y >> sh) & 0x01010101,
                       (qh1.z >> sh) & 0x01010101,
                       (qh1.w >> sh) & 0x01010101};
    const int4 nib0 = {(ql0.x >> shn) & kLaneMask,
                       (ql0.y >> shn) & kLaneMask,
                       (ql0.z >> shn) & kLaneMask,
                       (ql0.w >> shn) & kLaneMask};
    const int4 nib1 = {(ql1.x >> shn) & kLaneMask,
                       (ql1.y >> shn) & kLaneMask,
                       (ql1.z >> shn) & kLaneMask,
                       (ql1.w >> shn) & kLaneMask};
    // 5th bit as +16 per element: 0x10 in the matching byte lanes.
    const int4 w0 = {nib0.x | (bit0.x << 4), nib0.y | (bit0.y << 4),
                     nib0.z | (bit0.z << 4), nib0.w | (bit0.w << 4)};
    const int4 w1 = {nib1.x | (bit1.x << 4), nib1.y | (bit1.y << 4),
                     nib1.z | (bit1.z << 4), nib1.w | (bit1.w << 4)};
    const int k0 = (blk << 8) + (iter << 6) + (low ? 0 : 32);
    const ActQ8Hdr h = xh[k0 >> 5];
    const int4 a = *reinterpret_cast<const int4*>(xq + k0);
    const int4 bb = *reinterpret_cast<const int4*>(xq + k0 + 16);
    int sc, mn;
    scale_min_k4(sub, b->scales, sc, mn);
    const float dw = h2f(b->d) * (float)sc * h.d;
    const float dm = h2f(b->dmin) * (float)mn * h.d;
    const int sum = h.sum_lo + h.sum_hi;
    const int dot = dot4(w0, a) + dot4(w1, bb);
    return fmaf(dw, (float)dot, -dm * (float)sum);
  } else if constexpr (DT == Wd::Q6K) {
    // Unit = 16 elements at k0 (sub-block 2*pair+half, low nibbles) and 16 at
    // k0+64 (sub-block 2*pair+half+4, the same bytes' high nibbles), with the
    // 2 high bits of each quant from qh bit planes.
    const int blk   = c >> 3;
    const int chunk = (c >> 2) & 1;
    const int pair  = (c >> 1) & 1;
    const int half  = c & 1;
    const BlockQ6K* b = reinterpret_cast<const BlockQ6K*>(row) + blk;
    const int4 ql = load_u16x8(b->ql + chunk * 64 + pair * 32 + half * 16);
    const int4 qh = load_u16x8(b->qh + chunk * 32 + half * 16);
    const int sh = pair ? 2 : 0;
    const int4 wA = {(ql.x & kLaneMask) | (((qh.x >> sh) & 0x03030303) << 4),
                     (ql.y & kLaneMask) | (((qh.y >> sh) & 0x03030303) << 4),
                     (ql.z & kLaneMask) | (((qh.z >> sh) & 0x03030303) << 4),
                     (ql.w & kLaneMask) | (((qh.w >> sh) & 0x03030303) << 4)};
    const int4 wB = {((ql.x >> 4) & kLaneMask) |
                         (((qh.x >> (sh + 4)) & 0x03030303) << 4),
                     ((ql.y >> 4) & kLaneMask) |
                         (((qh.y >> (sh + 4)) & 0x03030303) << 4),
                     ((ql.z >> 4) & kLaneMask) |
                         (((qh.z >> (sh + 4)) & 0x03030303) << 4),
                     ((ql.w >> 4) & kLaneMask) |
                         (((qh.w >> (sh + 4)) & 0x03030303) << 4)};
    const int k0 = (blk << 8) + (chunk << 7) + (pair << 5) + (half << 4);
    const int b0 = k0 >> 5;
    const ActQ8Hdr h0 = xh[b0], h1 = xh[b0 + 2];
    const int4 a  = *reinterpret_cast<const int4*>(xq + k0);
    const int4 bb = *reinterpret_cast<const int4*>(xq + k0 + 64);
    // Both windows start at the same offset within their 32-element block
    // (k0 and k0+64 are congruent mod 32), so they read the same half sum.
    const int sumA = half ? h0.sum_hi : h0.sum_lo;
    const int sumB = half ? h1.sum_hi : h1.sum_lo;
    // 6-bit quants are stored biased by +32; subtract 32 * sum instead of
    // biasing every weight byte (llama does the same with a second dp4a).
    const int dotA = dot4(wA, a) - (sumA << 5);
    const int dotB = dot4(wB, bb) - (sumB << 5);
    const float d  = h2f(b->d);
    const float s0 = d * (float)b->scales[chunk * 8 + 2 * pair + half];
    const float s1 = d * (float)b->scales[chunk * 8 + 2 * pair + half + 4];
    return fmaf(s0 * h0.d, (float)dotA, s1 * h1.d * (float)dotB);
  } else {  // Wd::Q80: one 32-element block, one scale, int8 payload.
    const BlockQ80* b = reinterpret_cast<const BlockQ80*>(row) + c;
    const int4 w0 = load_u16x8(reinterpret_cast<const uint8_t*>(b->qs));
    const int4 w1 = load_u16x8(reinterpret_cast<const uint8_t*>(b->qs + 16));
    const int k0 = c << 5;
    const int4 a  = *reinterpret_cast<const int4*>(xq + k0);
    const int4 bb = *reinterpret_cast<const int4*>(xq + k0 + 16);
    const float dw = h2f(b->d) * xh[c].d;
    const int dot = dot4(w0, a) + dot4(w1, bb);
    return dw * (float)dot;
  }
}

template <class Scalar>
__global__ void quantize_act_q8_kernel(const Scalar* __restrict__ x,
                                       int8_t* __restrict__ xq,
                                       ActQ8Hdr* __restrict__ xh, int nblk) {
  const int lane = (int)(threadIdx.x & 31u);
  const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  if (warp >= nblk) return;
  const float v = static_cast<float>(x[(size_t)warp * 32 + (size_t)lane]);
  float amax = fabsf(v);
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));
  // d = amax/127 keeps every |q| <= 127; an all-zero block quantizes to
  // q == 0 with d == 0 (the weight-side scale, not d, carries the value).
  // Both divisions are correctly rounded IEEE ops, so a host reference can
  // reproduce this quantizer bit for bit.
  const float d   = amax / 127.0f;
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  int q = __float2int_rn(v * inv);
  q     = max(-127, min(127, q));
  // 16-element half sums (xor butterfly below lane-of-16 keeps the halves
  // separate): exactly the sums the weight-side min / -32 terms need.
  int s = q;
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  s += __shfl_xor_sync(0xffffffffu, s, 4);
  s += __shfl_xor_sync(0xffffffffu, s, 8);
  xq[(size_t)warp * 32 + (size_t)lane] = (int8_t)q;
  if (lane == 0) {
    xh[warp].d      = d;
    xh[warp].sum_lo = s;
    xh[warp].pad    = 0;
  } else if (lane == 16) {
    xh[warp].sum_hi = s;
  }
}

constexpr int kWarpsPerBlock = 4;

template <class Scalar, Wd DT>
__global__ __launch_bounds__(32 * kWarpsPerBlock) void gemv_int8_kernel(
    const uint8_t* __restrict__ wdata, size_t row_bytes, size_t nrows,
    const int8_t* __restrict__ xq, const ActQ8Hdr* __restrict__ xh,
    Scalar* __restrict__ y, int nunits) {
  const int lane = (int)(threadIdx.x & 31u);
  const int warp = (int)(threadIdx.x >> 5);
  const size_t row = (size_t)blockIdx.x * (size_t)kWarpsPerBlock + (size_t)warp;
  if (row >= nrows) return;
  const uint8_t* r = wdata + row * row_bytes;
  // Two accumulators: the dp4a chains of consecutive units are independent,
  // so the loop keeps both the int and the float pipes busy.
  float acc0 = 0.0f, acc1 = 0.0f;
  int c = lane;
  for (; c + 32 < nunits; c += 64) {
    acc0 += unit_dot<DT>(r, c, xq, xh);
    acc1 += unit_dot<DT>(r, c + 32, xq, xh);
  }
  if (c < nunits) acc0 += unit_dot<DT>(r, c, xq, xh);
  float acc = acc0 + acc1;
  acc += __shfl_xor_sync(0xffffffffu, acc, 16);
  acc += __shfl_xor_sync(0xffffffffu, acc, 8);
  acc += __shfl_xor_sync(0xffffffffu, acc, 4);
  acc += __shfl_xor_sync(0xffffffffu, acc, 2);
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);
  if (lane == 0) y[row] = static_cast<Scalar>(acc);
}

}  // namespace Kernels

// Stage 1 + stage 2 on the caller's (Kokkos) CUDA stream. Returns false when
// the shape/dtype/scratch does not fit this path; the caller then runs the
// portable kernel, so a declined launch can never change results.
template <class ExecSpace, class Scalar>
bool launch_quant_gemv_int8(const Kokkos::View<Scalar*, ExecSpace>& x,
                            const uint8_t* wdata, Wd dt, size_t rows,
                            size_t cols, size_t row_bytes,
                            const Kokkos::View<Scalar*, ExecSpace>& y,
                            uint8_t* scratch, size_t scratch_bytes) {
  constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (!is_cuda) {
    return false;
  } else {
    constexpr bool is_f = std::is_same_v<Scalar, float>;
    constexpr bool is_b =
        std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>;
    if constexpr (!is_f && !is_b) {
      return false;
    } else {
      if constexpr (is_b) {
        static_assert(!KOKKOS_BHALF_T_IS_FLOAT,
                      "int8 GEMV needs hardware bfloat16");
        static_assert(sizeof(Kokkos::Experimental::bhalf_t) == 2 &&
                          alignof(Kokkos::Experimental::bhalf_t) == 2,
                      "bhalf_t must reinterpret as __nv_bfloat16");
      }
      if (wdata == nullptr || scratch == nullptr) return false;
      if (!dtype_ok(dt, cols)) return false;
      if (rows == 0 || rows > (size_t)(std::numeric_limits<int>::max)())
        return false;
      if (scratch_bytes < act_q8_bytes(cols)) return false;
      // The kernels index the activation by element (contiguous row).
      if (x.stride(0) != 1 || y.stride(0) != 1) return false;
      const size_t nunits = act_q8_blocks(cols);
      if (nunits > (size_t)(std::numeric_limits<int>::max)()) return false;

      uint8_t* const sbase = scratch;
      int8_t* const xq = reinterpret_cast<int8_t*>(sbase);
      ActQ8Hdr* const xh = reinterpret_cast<ActQ8Hdr*>(
          sbase + act_q8_hdr_offset(cols));
      const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
      const int nblk = (int)nunits;
      const int qthreads = 256;
      const int qblocks = (nblk + (qthreads / 32) - 1) / (qthreads / 32);
      Kernels::quantize_act_q8_kernel<Scalar>
          <<<qblocks, qthreads, 0, stream>>>(x.data(), xq, xh, nblk);
      if (cudaGetLastError() != cudaSuccess) return false;

      const int gthreads = 32 * Kernels::kWarpsPerBlock;
      const int gblocks =
          (int)((rows + (size_t)Kernels::kWarpsPerBlock - 1) /
                (size_t)Kernels::kWarpsPerBlock);
      switch (dt) {
        case Wd::Q4K:
          Kernels::gemv_int8_kernel<Scalar, Wd::Q4K>
              <<<gblocks, gthreads, 0, stream>>>(wdata, row_bytes, rows, xq, xh,
                                                 y.data(), (int)nunits);
          break;
        case Wd::Q5K:
          Kernels::gemv_int8_kernel<Scalar, Wd::Q5K>
              <<<gblocks, gthreads, 0, stream>>>(wdata, row_bytes, rows, xq, xh,
                                                 y.data(), (int)nunits);
          break;
        case Wd::Q6K:
          Kernels::gemv_int8_kernel<Scalar, Wd::Q6K>
              <<<gblocks, gthreads, 0, stream>>>(wdata, row_bytes, rows, xq, xh,
                                                 y.data(), (int)nunits);
          break;
        case Wd::Q80:
          Kernels::gemv_int8_kernel<Scalar, Wd::Q80>
              <<<gblocks, gthreads, 0, stream>>>(wdata, row_bytes, rows, xq, xh,
                                                 y.data(), (int)nunits);
          break;
      }
      return cudaGetLastError() == cudaSuccess;
    }
  }
}

#else  // !KOKKOS_IMPL_CUDA_MMVQ_AVAILABLE

// Non-CUDA (or non-nvcc) builds: always decline, so every call keeps the
// portable kernel. Unnamed parameters dodge -Wunused-parameter.
template <class ExecSpace, class Scalar>
bool launch_quant_gemv_int8(const Kokkos::View<Scalar*, ExecSpace>&,
                            const uint8_t*, Wd, size_t, size_t, size_t,
                            const Kokkos::View<Scalar*, ExecSpace>&, uint8_t*,
                            size_t) {
  return false;
}

#endif

}  // namespace GgufMvq
}  // namespace Impl
}  // namespace KokkosDNN

#endif
