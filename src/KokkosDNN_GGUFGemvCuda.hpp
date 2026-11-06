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
// (scale, min) unpack and the activation window loads, not the dots: the
// unpack is done once per 8-lane warp batch and shuffled to lanes
// (scales_k4_batch/k4_pair), nibble splits use 64-bit masking
// (split_u4_to_i8), weight windows use alignment-dispatched wide loads
// (load_win16), and d/dmin ride one u32 load.
//
// Structure (two launches per GEMV, both on the caller's CUDA stream):
//   1. quantize_act_q8_kernel: one warp per 32-element activation block.
//      Warp max of |x| -> d = amax/127, q_i = rint(x_i/d) in [-127, 127],
//      plus the sum of each 16-element half (the bit the weight-side min /
//      Q6_K -32 terms need, so the dot kernel never re-derives them).
//      The row is quantized ONCE per GEMV and reused by all N output rows.
//   2. Baseline one-warp-per-row GEMV, 4 rows per 128-thread block (same
//      row/warp mapping as QuantGemvUnitMultiFunctor):
//      gemv_int8_kernel_direct loads activation windows from global memory
//      per row (L1-absorbed reuse across the block's 4 warps);
//      gemv_int8_kernel_sm instead stages the whole quantized activation
//      (xq int8 row + ActQ8Hdr array, contiguous since K % 32 == 0) into
//      __shared__ memory with one coalesced copy by all 128 threads +
//      __syncthreads, and the 4 row-warps dot against shared memory.
//      The sm variant is gated by KOKKOSDNN_IMPL_GEMV_SMEM_STAGING (default 0,
//      i.e. direct): measured 0.61-0.86x of direct on RTX 3090 across
//      ffn_down/gate and lm_head shapes (equal-occupancy 10 KiB staging
//      still 0.84-0.86x, so this is the strided per-lane int4 shared reads'
//      4-way bank conflicts + copy/sync, not just the 30 KiB case's 9->3
//      blocks/SM occupancy drop), while direct already runs at 94-99% of the
//      DRAM roofline there (879/882 GB/s lm_head). Flip to 1 to A/B on other
//      parts. Oversized activations (> 48 KiB staging, no opt-in needed)
//      always use the direct kernel, so every shape stays correct. Skinny shapes (small N, large K: attn_k/v)
//      instead take gemv_int8_splitk_kernel (Kernels::launch_gemv_dt picks):
//      one block per row, 4-8 warps splitting the K-window with an
//      intra-block reduction, so a 256-row GEMV fills the GPU instead of
//      launching ~64 blocks. Unit math is shared (unit_dot), so numerics
//      are identical across schedules.
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

// Toggle for the shared-memory staged baseline (gemv_int8_kernel_sm): 1
// routes baseline shapes through cross-row activation reuse in __shared__,
// 0 keeps the direct per-row global-load kernel. Default 0: the staged
// kernel measured 0.61-0.86x of direct on RTX 3090 (see the header comment),
// so staging is strictly an experiment toggle (e.g. -D...=1 for an H100 A/B
// build), never a default. Always defined so the dispatcher condition below
// compiles unchanged in both settings.
#ifndef KOKKOSDNN_IMPL_GEMV_SMEM_STAGING
#define KOKKOSDNN_IMPL_GEMV_SMEM_STAGING 0
#endif

namespace Kernels {

__device__ __forceinline__ float h2f(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

// get_scale_min_k4 (ggml-quants.c) unpacked ONCE per 8-lane warp batch.
//
// The 8 units of one Q4_K/Q5_K block share 12 scale bytes holding 8 x
// (6-bit scale, 6-bit min); the pre-change code had every lane load and
// unpack its own pair(s) (two branched per-lane unpacks per Q4_K unit).
// Lanes c..c+7 of a warp cover exactly one block's 8 units (unit_dot's
// mapping is blk = c >> 3), so the batch leader loads the 12 bytes once
// (2-byte-safe u16 loads; the scales sit at block+4), unpacks all 8 pairs,
// and broadcasts them as two u64 words holding bytes
// [sc0,mn0,..,sc3,mn3] / [sc4,mn4,..,sc7,mn7]. Each lane then extracts its
// pair(s) with one __shfl_sync per word plus shifts. Amortized cost is ~1/8
// of the unpack plus shuffles/extracts per lane, versus two branched
// unpacks plus ~8 narrow scale loads per lane before.
//
// Deliberately shuffle-based, not shared-memory: the sibling
// activation-staging change owns shared memory in these kernels, and
// cross-lane shuffle keeps the two edits in disjoint helpers.
__device__ __forceinline__ void scales_k4_batch(const uint8_t* scales,
                                                int lane,
                                                unsigned long long& pg0,
                                                unsigned long long& pg1) {
  const int leader = lane & ~7;
  // Group mask, not the full warp: at a K tail some batches sit fully past
  // nunits and skip unit_dot, and nunits is a multiple of 8 for every dtype
  // using this helper (dtype_ok requires K % 256 == 0), so participation is
  // all-or-nothing per batch and the shuffle stays well-defined.
  const unsigned gmask = 0xFFu << (unsigned)leader;
  unsigned long long g0 = 0ULL, g1 = 0ULL;
  if (lane == leader) {
    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(scales);
    const uint32_t w0 = (uint32_t)s16[0] | ((uint32_t)s16[1] << 16);
    const uint32_t w1 = (uint32_t)s16[2] | ((uint32_t)s16[3] << 16);
    const uint32_t w2 = (uint32_t)s16[4] | ((uint32_t)s16[5] << 16);
    for (int j = 0; j < 8; ++j) {
      // Byte k of w0/w1/w2 is scales[k]/scales[k+4]/scales[k+8]; the
      // j<4 / j>=4 cases are get_scale_min_k4 (ggml-quants.c), branch-free
      // in the block's 8 sub-blocks. The unrolled loop sees a constant j,
      // so the branch below resolves at compile time on the leader.
      const int k = j & 3;
      const uint32_t b0 = (w0 >> (k << 3)) & 0xFFu;
      const uint32_t b1 = (w1 >> (k << 3)) & 0xFFu;
      const uint32_t b2 = (w2 >> (k << 3)) & 0xFFu;
      uint32_t sc, mn;
      if (j < 4) {
        sc = b0 & 63u;
        mn = b1 & 63u;
      } else {
        sc = (b2 & 15u) | ((b0 >> 6) << 4);
        mn = (b2 >> 4) | ((b1 >> 6) << 4);
      }
      const unsigned long long pair = (unsigned long long)(sc | (mn << 8));
      if (j < 4) {
        g0 |= pair << (k << 4);
      } else {
        g1 |= pair << (k << 4);
      }
    }
  }
  pg0 = __shfl_sync(gmask, g0, leader);
  pg1 = __shfl_sync(gmask, g1, leader);
}

// One (scale, min) pair out of a scales_k4_batch broadcast (j = 0..7).
__device__ __forceinline__ void k4_pair(unsigned long long pg0,
                                        unsigned long long pg1, int j, int& sc,
                                        int& mn) {
  const unsigned long long g = (j < 4) ? pg0 : pg1;
  const unsigned p = (unsigned)((g >> ((j & 3) << 4)) & 0xFFFFULL);
  sc = (int)(p & 0xFFu);
  mn = (int)((p >> 8) & 0xFFu);
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
// 2-byte alignment is safe on every device. This is the fallback inside
// load_win16 below, which issues 4/8/16-byte vector loads wherever the
// window address allows and only lands here for odd alignments.
__device__ __forceinline__ int4 load_u16x8(const uint8_t* p) {
  const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
  int4 r;
  r.x = (int)q[0] | ((int)q[1] << 16);
  r.y = (int)q[2] | ((int)q[3] << 16);
  r.z = (int)q[4] | ((int)q[5] << 16);
  r.w = (int)q[6] | ((int)q[7] << 16);
  return r;
}

// Wide 16-byte window loads. load_u16x8 above issues 8 narrow loads; where
// the window address allows, 4-byte (u32x4), 8-byte (u64x2) or full 16-byte
// vector loads issue 4, 2 or 1 memory instructions for the same 16 bytes.
// Read-only (__ldg) throughout: weights never change under the kernel, so
// every path may route through the read-only cache. Q4_K/Q5_K rows are
// multiples of 16 bytes (144/176-byte blocks), so a 16-byte-aligned base
// keeps every qs/qh window on the 16-byte path; Q6_K (210) and Q8_0 (34)
// rows only guarantee 2-byte alignment, so those windows dispatch per warp
// at runtime. The branch is uniform: all lanes of a warp share one row, and
// every window offset in unit_dot is a multiple of 16 within its block.
__device__ __forceinline__ int4 load_u32x4(const uint8_t* p) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
  int4 r;
  r.x = (int)__ldg(q);
  r.y = (int)__ldg(q + 1);
  r.z = (int)__ldg(q + 2);
  r.w = (int)__ldg(q + 3);
  return r;
}

__device__ __forceinline__ int4 load_u64x2(const uint8_t* p) {
  const unsigned long long* q =
      reinterpret_cast<const unsigned long long*>(p);
  const unsigned long long v0 = __ldg(q);
  const unsigned long long v1 = __ldg(q + 1);
  int4 r;
  r.x = (int)(v0 & 0xFFFFFFFFULL);
  r.y = (int)(v0 >> 32);
  r.z = (int)(v1 & 0xFFFFFFFFULL);
  r.w = (int)(v1 >> 32);
  return r;
}

__device__ __forceinline__ int4 load_win16(const uint8_t* p) {
  const size_t a = reinterpret_cast<size_t>(p);
  if ((a & 15u) == 0u) return __ldg(reinterpret_cast<const int4*>(p));
  if ((a & 7u) == 0u) return load_u64x2(p);
  if ((a & 3u) == 0u) return load_u32x4(p);
  return load_u16x8(p);
}

// Split 16 packed 4-bit quants (one int4 = 16 bytes) into two dp4a-ready
// int4s. 64-bit masking issues 6 ALU instructions for the 16 bytes instead
// of 12 with per-int32 masking; dp4a reads the bytes as signed int8, and
// 0..15 are unaffected by signedness.
__device__ __forceinline__ void split_u4_to_i8(const int4& w, int4& lo,
                                               int4& hi) {
  const unsigned long long* v =
      reinterpret_cast<const unsigned long long*>(&w);
  unsigned long long* l = reinterpret_cast<unsigned long long*>(&lo);
  unsigned long long* h = reinterpret_cast<unsigned long long*>(&hi);
  constexpr unsigned long long kNib = 0x0F0F0F0F0F0F0F0FULL;
  l[0] = v[0] & kNib;
  l[1] = v[1] & kNib;
  h[0] = (v[0] >> 4) & kNib;
  h[1] = (v[1] >> 4) & kNib;
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
    int4 lo, hi;
    split_u4_to_i8(w, lo, hi);
    // (scale, min) pairs 2*iter / 2*iter+1, unpacked once per 8-lane warp
    // batch (scales_k4_batch) instead of twice per lane; d/dmin ride one
    // u32 load (rows are 16-byte multiples, so block+0 is u32-safe wherever
    // the qs int4 loads above are safe) instead of two u16 loads.
    const int lane = (int)(threadIdx.x & 31u);
    unsigned long long pg0 = 0ULL, pg1 = 0ULL;
    scales_k4_batch(b->scales, lane, pg0, pg1);
    int sc0, mn0, sc1, mn1;
    k4_pair(pg0, pg1, 2 * iter, sc0, mn0);
    k4_pair(pg0, pg1, 2 * iter + 1, sc1, mn1);
    // value = (d*sc)*q - (dmin*mn), x ~= d8*q8: the min term is a constant
    // times the activation block sum, exactly as in the portable kernel, but
    // summed once per window instead of once per element.
    const uint32_t dd = *reinterpret_cast<const uint32_t*>(&b->d);
    const float dw = h2f((uint16_t)(dd & 0xFFFFu));
    const float dm = h2f((uint16_t)(dd >> 16));
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
    // qh is one 32-byte bit-plane per block, shared by the 8-lane warp batch
    // covering the block: alignment-dispatched wide loads (16-byte path when
    // the 176-byte rows keep block+16 aligned) instead of 16 narrow loads.
    const int4 qh0 = load_win16(b->qh);
    const int4 qh1 = load_win16(b->qh + 16);
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
    // (scale, min) unpacked once per 8-lane warp batch (scales_k4_batch);
    // d/dmin ride one u32 load instead of two u16 loads.
    const int lane = (int)(threadIdx.x & 31u);
    unsigned long long pg0 = 0ULL, pg1 = 0ULL;
    scales_k4_batch(b->scales, lane, pg0, pg1);
    int sc, mn;
    k4_pair(pg0, pg1, sub, sc, mn);
    const uint32_t dd = *reinterpret_cast<const uint32_t*>(&b->d);
    const float dw = h2f((uint16_t)(dd & 0xFFFFu)) * (float)sc * h.d;
    const float dm = h2f((uint16_t)(dd >> 16)) * (float)mn * h.d;
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
    // 210-byte blocks only guarantee 2-byte row alignment, so these windows
    // dispatch per warp at runtime (8/4-byte paths on even rows, 2-byte
    // fallback otherwise) instead of always assembling from 2-byte loads.
    const int4 ql = load_win16(b->ql + chunk * 64 + pair * 32 + half * 16);
    const int4 qh = load_win16(b->qh + chunk * 32 + half * 16);
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
    // 34-byte blocks only guarantee 2-byte row alignment: same
    // alignment-dispatched loads as Q6_K (the qs payload sits at block+2, so
    // even rows take the 4-byte path, odd rows the 2-byte fallback).
    const int4 w0 = load_win16(reinterpret_cast<const uint8_t*>(b->qs));
    const int4 w1 = load_win16(reinterpret_cast<const uint8_t*>(b->qs + 16));
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

// Skinny-shape split-K schedule: below these row counts a GEMV cannot fill
// the GPU one-warp-per-row (attn_k / attn_v at N ~= 256 are ~64 blocks on a
// 132-SM H100), so each row is split across WARPS_PER_ROW warps of one
// block with a small intra-block reduction instead of one warp. The
// heuristic only kicks in for small N with enough K to feed every warp (the
// MinUnits gates); large shapes (ffn_down, vocab head) keep the baseline
// kernel above untouched.
constexpr size_t kSplitK8MaxRows = 256;
constexpr size_t kSplitK4MaxRows = 1024;
constexpr size_t kSplitK8MinUnits = 128;
constexpr size_t kSplitK4MinUnits = 64;

// Baseline one-warp-per-row kernel: each row-warp loads its activation
// windows from global memory. Kept as the fallback for activations whose
// staging does not fit the shared-memory budget below; always correct.
template <class Scalar, Wd DT>
__global__ __launch_bounds__(32 * kWarpsPerBlock) void gemv_int8_kernel_direct(
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

// Shared-memory activation-staging budget: 48 KiB is the default per-block
// dynamic limit on every CUDA architecture, so no opt-in attribute is
// needed. All in-model shapes fit (largest: K = 19968 -> ~29.3 KiB);
// anything bigger uses gemv_int8_kernel_direct above.
constexpr std::size_t kMaxSmemStageBytes = 48 * 1024;

// Cross-row activation reuse: same row/warp mapping as the direct kernel,
// but the block first stages the whole quantized activation (xq int8 row +
// ActQ8Hdr array) into __shared__ memory with one coalesced copy by all 128
// threads + __syncthreads, and the 4 row-warps then dot against shared
// memory. Staging runs before the row early-exit so every thread takes part
// in the copy (no __syncthreads divergence); the exit moves after the sync.
// Dynamic shared size is act_q8_bytes(K): K is a multiple of 32 on this path
// (dtype_ok), so xq (K bytes) and the header array (nunits * 16 bytes) are
// each a whole number of int4s and the int4 copy is aligned end to end.
template <class Scalar, Wd DT>
__global__ __launch_bounds__(32 * kWarpsPerBlock) void gemv_int8_kernel_sm(
    const uint8_t* __restrict__ wdata, size_t row_bytes, size_t nrows,
    const int8_t* __restrict__ xq, const ActQ8Hdr* __restrict__ xh,
    Scalar* __restrict__ y, int nunits) {
  extern __shared__ int4 sdata4[];
  const int tid      = (int)threadIdx.x;
  const int nthreads = 32 * kWarpsPerBlock;
  const size_t nq4   = (size_t)nunits * 2;  // K bytes / 16
  const int4* __restrict__ srcq = reinterpret_cast<const int4*>(xq);
  for (size_t i = (size_t)tid; i < nq4; i += (size_t)nthreads)
    sdata4[i] = srcq[i];
  int4* shdr                          = sdata4 + nq4;
  const int4* __restrict__ srch = reinterpret_cast<const int4*>(xh);
  for (int i = tid; i < nunits; i += nthreads) shdr[i] = srch[i];
  __syncthreads();

  const int8_t* __restrict__ sxq = reinterpret_cast<const int8_t*>(sdata4);
  const ActQ8Hdr* __restrict__ sxh = reinterpret_cast<const ActQ8Hdr*>(shdr);
  const int lane                   = tid & 31;
  const int warp                   = tid >> 5;
  const size_t row =
      (size_t)blockIdx.x * (size_t)kWarpsPerBlock + (size_t)warp;
  if (row >= nrows) return;
  const uint8_t* r = wdata + row * row_bytes;
  // Two accumulators: the dp4a chains of consecutive units are independent,
  // so the loop keeps both the int and the float pipes busy.
  float acc0 = 0.0f, acc1 = 0.0f;
  int c = lane;
  for (; c + 32 < nunits; c += 64) {
    acc0 += unit_dot<DT>(r, c, sxq, sxh);
    acc1 += unit_dot<DT>(r, c + 32, sxq, sxh);
  }
  if (c < nunits) acc0 += unit_dot<DT>(r, c, sxq, sxh);
  float acc = acc0 + acc1;
  acc += __shfl_xor_sync(0xffffffffu, acc, 16);
  acc += __shfl_xor_sync(0xffffffffu, acc, 8);
  acc += __shfl_xor_sync(0xffffffffu, acc, 4);
  acc += __shfl_xor_sync(0xffffffffu, acc, 2);
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);
  if (lane == 0) y[row] = static_cast<Scalar>(acc);
}

// Split-K companion to the baseline kernels above for skinny shapes: one
// block per row, WARPS_PER_ROW warps splitting the row's K-window, one
// intra-block reduction. A 256-row GEMV launches 256 blocks of 4-8 warps
// instead of 64 blocks of 4 warps. Separate kernel (own __shared__ partials,
// global activation reads: a block covers a single row, so there is no
// cross-row reuse to stage) so the shared-memory staging in
// gemv_int8_kernel_sm is untouched. No __launch_bounds__: the bound would be
// template-dependent and buys nothing for this streaming kernel.
template <class Scalar, Wd DT, int WARPS_PER_ROW>
__global__ void gemv_int8_splitk_kernel(
    const uint8_t* __restrict__ wdata, size_t row_bytes, size_t nrows,
    const int8_t* __restrict__ xq, const ActQ8Hdr* __restrict__ xh,
    Scalar* __restrict__ y, int nunits) {
  const int lane = (int)(threadIdx.x & 31u);
  const int warp = (int)(threadIdx.x >> 5);
  // Uniform per block (one row per block), so the early return cannot
  // diverge under the __syncthreads() below.
  const size_t row = (size_t)blockIdx.x;
  if (row >= nrows) return;
  const uint8_t* r = wdata + row * row_bytes;
  const int tid    = warp * 32 + lane;
  const int stride = WARPS_PER_ROW * 32;
  float acc0 = 0.0f, acc1 = 0.0f;
  int c = tid;
  for (; c + stride < nunits; c += 2 * stride) {
    acc0 += unit_dot<DT>(r, c, xq, xh);
    acc1 += unit_dot<DT>(r, c + stride, xq, xh);
  }
  if (c < nunits) acc0 += unit_dot<DT>(r, c, xq, xh);
  float acc = acc0 + acc1;
  acc += __shfl_xor_sync(0xffffffffu, acc, 16);
  acc += __shfl_xor_sync(0xffffffffu, acc, 8);
  acc += __shfl_xor_sync(0xffffffffu, acc, 4);
  acc += __shfl_xor_sync(0xffffffffu, acc, 2);
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);
  __shared__ float part[WARPS_PER_ROW];
  if (lane == 0) part[warp] = acc;
  __syncthreads();
  // Warp 0 reduces the per-warp partials; idle lanes contribute 0.
  if (warp == 0) {
    float v = 0.0f;
    if (lane < WARPS_PER_ROW) v = part[lane];
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (lane == 0) y[row] = static_cast<Scalar>(v);
  }
}

// Stage-2 dispatch for one dtype: skinny shapes take the split-K kernel,
// everything else the direct baseline. The shared-staged kernel is selected
// only with KOKKOSDNN_IMPL_GEMV_SMEM_STAGING=1 (and staging within budget);
// it measured slower than direct on RTX 3090 (see the header comment), so
// the default stays on the direct kernel. Grid/block only differ here; unit
// math is shared through unit_dot, so numerics are identical across
// schedules.
template <class Scalar, Wd DT>
bool launch_gemv_dt(const uint8_t* wdata, size_t row_bytes, size_t rows,
                    const int8_t* xq, const ActQ8Hdr* xh, Scalar* y, int nunits,
                    size_t stage_bytes, cudaStream_t stream) {
  const size_t nu = (size_t)nunits;
  const bool use8 = rows <= kSplitK8MaxRows && nu >= kSplitK8MinUnits;
  const bool use4 = !use8 && rows <= kSplitK4MaxRows && nu >= kSplitK4MinUnits;
  if (use8) {
    gemv_int8_splitk_kernel<Scalar, DT, 8>
        <<< (int)rows, 32 * 8, 0, stream>>>(wdata, row_bytes, rows, xq, xh, y,
                                             nunits);
  } else if (use4) {
    gemv_int8_splitk_kernel<Scalar, DT, 4>
        <<< (int)rows, 32 * 4, 0, stream>>>(wdata, row_bytes, rows, xq, xh, y,
                                             nunits);
  } else if (KOKKOSDNN_IMPL_GEMV_SMEM_STAGING != 0 &&
             stage_bytes <= kMaxSmemStageBytes) {
    const int gthreads = 32 * kWarpsPerBlock;
    const int gblocks =
        (int)((rows + (size_t)kWarpsPerBlock - 1) / (size_t)kWarpsPerBlock);
    gemv_int8_kernel_sm<Scalar, DT>
        <<<gblocks, gthreads, stage_bytes, stream>>>(wdata, row_bytes, rows,
                                                      xq, xh, y, nunits);
  } else {
    const int gthreads = 32 * kWarpsPerBlock;
    const int gblocks =
        (int)((rows + (size_t)kWarpsPerBlock - 1) / (size_t)kWarpsPerBlock);
    gemv_int8_kernel_direct<Scalar, DT>
        <<<gblocks, gthreads, 0, stream>>>(wdata, row_bytes, rows, xq, xh, y,
                                           nunits);
  }
  return cudaGetLastError() == cudaSuccess;
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

      const int nu = (int)nunits;
      const size_t stage_bytes = act_q8_bytes(cols);
      bool ok = false;
      switch (dt) {
        case Wd::Q4K:
          ok = Kernels::launch_gemv_dt<Scalar, Wd::Q4K>(
              wdata, row_bytes, rows, xq, xh, y.data(), nu, stage_bytes,
              stream);
          break;
        case Wd::Q5K:
          ok = Kernels::launch_gemv_dt<Scalar, Wd::Q5K>(
              wdata, row_bytes, rows, xq, xh, y.data(), nu, stage_bytes,
              stream);
          break;
        case Wd::Q6K:
          ok = Kernels::launch_gemv_dt<Scalar, Wd::Q6K>(
              wdata, row_bytes, rows, xq, xh, y.data(), nu, stage_bytes,
              stream);
          break;
        case Wd::Q80:
          ok = Kernels::launch_gemv_dt<Scalar, Wd::Q80>(
              wdata, row_bytes, rows, xq, xh, y.data(), nu, stage_bytes,
              stream);
          break;
      }
      return ok;
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
