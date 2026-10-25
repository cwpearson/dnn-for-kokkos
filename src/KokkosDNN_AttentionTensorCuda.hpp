// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_ATTENTIONTENSORCUDA_HPP
#define KOKKOSDNN_ATTENTIONTENSORCUDA_HPP

// CUDA-only tensor-core fused bidirectional attention (native path):
//   O = softmax(Q K^T * scale) V, FP32 accumulation, one kernel launch.
//
// This is the one backend-gated kernel in the attention family, and the
// gating is deliberate: tensor cores (wmma) exist only on NVIDIA SM80+
// hardware, so no portable Kokkos spelling can reach them. Everything else
// in the attention ladder stays portable (see KokkosDNN_Attention.hpp
// design notes). Dispatch order in attention_bidirectional_streaming is
// unchanged: cuDNN SDPA first, then this kernel (CUDA + SM80+ + bhalf_t +
// D in {64, 128} + D-contiguous Q/K/V), then the portable SIMT flash
// kernel, which always remains the correctness fallback (a failed launch
// also falls back -- the launcher returns false). handle.use_tensor_cores
// (default on for CUDA) is the kill switch, shared with the GEMM tensor
// path.
//
// Numerics (read before comparing against FP32 references):
//   * bhalf_t only: exact BF16 tensor-core products (Q and K round to BF16
//     on load, exactly as the portable kernel's bhalf_t inputs do),
//     FP32 accumulation, FP32 online softmax. The only extra rounding vs
//     the portable kernel is the BF16 weight table (weights in [0, 1]
//     rounded to BF16, ~1e-3 absolute on the weight) plus a tile-blocked
//     summation order (up to ~1 ulp vs sequential). Output agreement with
//     the portable kernel is ~1e-3 on O(0.1) outputs; the dedicated tests
//     below pin this envelope.
//   * float is deliberately NOT served here: TF32 tensor products would
//     move outputs ~1e-3, breaking the 1e-4 contract the float attention
//     tests pin. Float shapes stay on the improved portable kernel.
//
// Structure (CTA = 128 threads covering TQ query rows x 1 head, TQ = 32
// or 64):
//   * Q tile [TQ][D], K/V slabs [32][D] (BF16) plus the FP32 score plane
//     [TQ][32], the BF16 weight plane [TQ][32] and the FP32 output
//     accumulator [TQ][D] in dynamic shared (carve-out; static over 48 KB
//     is rejected by ptxas, so the footprint rides in the launch config
//     with a MaxDynamicSharedMemorySize opt-in where needed).
//   * TQ=64 halves the K/V re-read traffic. It is selected only at D=64,
//     where it fits the default budget (~46 KB, occupancy preserved);
//     at D=128 it needs ~79 KB (1 block/SM) and measured slower than
//     TQ=32 on RTX 3090 (40 ms vs 34 ms at Sj=4608), so D=128 stays on
//     the 32-wide engine.
//   * Per K-slab of 32: cooperative float4 K/V load (zero-padded tail),
//     S = Q K^T via wmma m16n16k16 (one 16x16 atom per warp, 4 warps),
//     scale + tail-mask (-inf) in shared, per-row max/rescale, weights in
//     BF16, Oacc rescale by row-threads on shared rows (no fragment
//     row-mapping assumptions anywhere), then P = W V via wmma
//     accumulating directly into fragments loaded from / stored to the
//     shared Oacc (16 atoms, 4 per warp). Denominators and inverse
//     denominators ride in shared; m/l stay in row-thread registers.
//   * K-tail keys are masked to -inf before softmax (zero weight) and the
//     T-tail writes are guarded, so edge tiles need no extra passes.
//     Ktot == 0 declines to the portable kernel (which owns that contract).
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "KokkosDNN_config.h"

#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDACC__) &&                      \
    (defined(KOKKOS_ARCH_AMPERE) || defined(KOKKOS_ARCH_ADA89) ||              \
     defined(KOKKOS_ARCH_HOPPER) || defined(KOKKOS_ARCH_BLACKWELL))
#define KOKKOSDNN_IMPL_ATTN_TENSOR_AVAILABLE 1
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#endif

namespace KokkosDNN {
namespace Impl {

#if defined(KOKKOSDNN_IMPL_ATTN_TENSOR_AVAILABLE)
namespace AttnTensor {

// Cached SM major of the current device (queried once; Kokkos pins the
// device before kernels run, so the current device is the right one).
inline int device_arch_major() {
  static const int major = [] {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 0;
    return prop.major;
  }();
  return major;
}

constexpr int kTq = 32;  // query rows per CTA (one head)
constexpr int kTk = 32;  // keys per K-slab

// Cooperative BF16 tile load: [rows x D] from `src` (row stride `pitch`
// elements) to shared `dst`, zero-padding rows >= bound. All 128 threads
// participate; 16 B chunks when the pitch keeps every row 16 B-aligned
// (the launcher gates this), scalar tail handled by the row guard.
__device__ inline void load_bf16_tile(const __nv_bfloat16* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst,
                                      int rows, int D, int pitch, int bound,
                                      int tid, int nthreads) {
  const int per_row = D / 8;  // float4 chunks per row (D is 64 or 128)
  const int chunks = rows * per_row;
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int l = tid; l < chunks; l += nthreads) {
    const int r = l / per_row, c = l % per_row;
    float4 v = zero;
    if (r < bound) {
      const float4* g =
          reinterpret_cast<const float4*>(src + r * pitch) + c;
      v = *g;
    }
    reinterpret_cast<float4*>(dst)[r * per_row + c] = v;
  }
}

template <int D, int TQ>
__global__ __launch_bounds__(128) void bidir_sdpa_kernel(
    const __nv_bfloat16* __restrict__ qd,
    const __nv_bfloat16* __restrict__ kd,
    const __nv_bfloat16* __restrict__ vd, __nv_bfloat16* __restrict__ od,
    int T, int Hq, int Hkv, int K, int qs0, int qs1, int ks0, int ks1,
    int vs0, int vs1, int os0, int os1, int os2, float scale) {
  static_assert(D == 64 || D == 128, "tensor bidir supports D 64/128 only");
  static_assert(TQ == 32 || TQ == 64, "tensor bidir supports TQ 32/64 only");
  constexpr int kTq = TQ;  // query rows per CTA (one head)
  constexpr int kSteps = D / 16;  // wmma k-steps per atom (4 or 8)

  // Dynamic shared carve-out (static over 48 KB is rejected by ptxas;
  // the wide engine opts into its footprint via
  // cudaFuncAttributeMaxDynamicSharedMemorySize at launch). All segment
  // sizes are multiples of 16 B, so every segment stays 16 B-aligned.
  __align__(16) extern __shared__ unsigned char smem_raw[];
  auto* Qs = reinterpret_cast<__nv_bfloat16(*)[D]>(smem_raw + 0);
  auto* Ks = reinterpret_cast<__nv_bfloat16(*)[D]>(
      smem_raw + size_t(kTq) * D * 2);
  auto* Vs = reinterpret_cast<__nv_bfloat16(*)[D]>(
      smem_raw + size_t(kTq) * D * 2 + size_t(kTk) * D * 2);
  auto* SW = reinterpret_cast<float(*)[kTk]>(
      smem_raw + size_t(kTq) * D * 2 + 2 * size_t(kTk) * D * 2);
  auto* W = reinterpret_cast<__nv_bfloat16(*)[kTk]>(
      smem_raw + size_t(kTq) * D * 2 + 2 * size_t(kTk) * D * 2 +
      size_t(kTq) * kTk * 4);
  auto* Oacc = reinterpret_cast<float(*)[D]>(
      smem_raw + size_t(kTq) * D * 2 + 2 * size_t(kTk) * D * 2 +
      size_t(kTq) * kTk * 4 + size_t(kTq) * kTk * 2);
  auto* Saux = reinterpret_cast<float(*)[kTq]>(
      smem_raw + size_t(kTq) * D * 2 + 2 * size_t(kTk) * D * 2 +
      size_t(kTq) * kTk * 4 + size_t(kTq) * kTk * 2 +
      size_t(kTq) * D * 4);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;

  const int tile = blockIdx.x / Hq;
  const int h = blockIdx.x % Hq;
  const int t0 = tile * kTq;
  const int kvh = (h * Hkv) / Hq;

  using FragA =
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>;
  using FragB =
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::col_major>;
  // V-slab fragment: Vs is naturally [K][N] (row = key), so the PV product
  // loads it row-major (unlike the K fragment above, which needs the
  // transposed view for S = Q K^T).
  using FragBV =
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>;
  using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                                       float>;

  // Q tile (zero-padded past T) then a zeroed output accumulator.
  load_bf16_tile(qd + t0 * qs0 + h * qs1, &Qs[0][0], kTq, D, qs0, T - t0, tid,
                 blockDim.x);
  for (int e = tid; e < kTq * D; e += blockDim.x) Oacc[0][e] = 0.0f;
  __syncthreads();

  // Streaming state for the owning row-threads (tid < kTq).
  float m = -1e30f;
  float l = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kTk) {
    const int kw = (k0 + kTk <= K) ? kTk : (K - k0);
    load_bf16_tile(kd + k0 * ks0 + kvh * ks1, &Ks[0][0], kTk, D, ks0, kw, tid,
                   blockDim.x);
    load_bf16_tile(vd + k0 * vs0 + kvh * vs1, &Vs[0][0], kTk, D, vs0, kw, tid,
                   blockDim.x);
    __syncthreads();

    // S = Q K^T: (kTq/16) x 2 atoms of 16x16, dealt round-robin over
    // warps (1 atom each at TQ=32, 2 each at TQ=64).
    constexpr int kSAtoms = (kTq / 16) * 2;
    for (int a = warp; a < kSAtoms; a += 4) {
      const int rb = a / 2, cb = a % 2;
      FragC sc;
      nvcuda::wmma::fill_fragment(sc, 0.0f);
      for (int ks = 0; ks < kSteps; ++ks) {
        FragA fa;
        FragB fb;
        nvcuda::wmma::load_matrix_sync(fa, &Qs[rb * 16][ks * 16], D);
        nvcuda::wmma::load_matrix_sync(fb, &Ks[cb * 16][ks * 16], D);
        nvcuda::wmma::mma_sync(sc, fa, fb, sc);
      }
      nvcuda::wmma::store_matrix_sync(&SW[rb * 16][cb * 16], sc, kTk,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    // Scale + K-tail mask (tail keys read zero K rows, so mask them out).
    for (int e = tid; e < kTq * kTk; e += blockDim.x) {
      const int r = e / kTk, c = e % kTk;
      float v = SW[r][c] * scale;
      if (c >= kw) v = -1e30f;
      SW[r][c] = v;
    }
    __syncthreads();

    // Per-row tile max folded into the running max (row-threads 0..31).
    if (tid < kTq) {
      float tmax = -1e30f;
      for (int c = 0; c < kTk; ++c) {
        const float v = SW[tid][c];
        if (v > tmax) tmax = v;
      }
      const float mnew = (tmax > m) ? tmax : m;
      Saux[0][tid] = expf(m - mnew);  // aa (m == -1e30 first: 0)
      Saux[1][tid] = mnew;
    }
    __syncthreads();

    // Rescale the accumulator rows + materialize BF16 weights (all
    // threads). expf outputs are rounded to BF16 once; downstream (row
    // sums, tensor A fragments) consumes the rounded values.
    for (int e = tid; e < kTq * D; e += blockDim.x) {
      const int r = e / D;
      Oacc[0][e] *= Saux[0][r];
    }
    for (int e = tid; e < kTq * kTk; e += blockDim.x) {
      const int r = e / kTk, c = e % kTk;
      W[r][c] = static_cast<__nv_bfloat16>(expf(SW[r][c] - Saux[1][r]));
    }
    __syncthreads();

    // Denominator from this slab's weights (row-threads) + inverse.
    if (tid < kTq) {
      float s = 0.0f;
      for (int c = 0; c < kTk; ++c)
        s += static_cast<float>(W[tid][c]);
      l = l * Saux[0][tid] + s;
      m = Saux[1][tid];
      Saux[2][tid] = (l > 0.0f) ? (1.0f / l) : 0.0f;
    }

    // P = W V into Oacc: (kTq/16) row blocks x (D/16) col blocks, dealt
    // round-robin over warps; each loads its Oacc frag, accumulates 2
    // k-steps (kTk/16), stores back.
    constexpr int kColBlocks = D / 16;
    constexpr int kAtoms = (kTq / 16) * kColBlocks;
    for (int a = warp; a < kAtoms; a += 4) {
      const int arb = a / kColBlocks, acb = a % kColBlocks;
      FragC acc;
      nvcuda::wmma::load_matrix_sync(acc, &Oacc[arb * 16][acb * 16], D,
                                     nvcuda::wmma::mem_row_major);
      for (int ks = 0; ks < 2; ++ks) {
        FragA fa;
        FragBV fb;
        nvcuda::wmma::load_matrix_sync(fa, &W[arb * 16][ks * 16], kTk);
        nvcuda::wmma::load_matrix_sync(fb, &Vs[ks * 16][acb * 16], D);
        nvcuda::wmma::mma_sync(acc, fa, fb, acc);
      }
      nvcuda::wmma::store_matrix_sync(&Oacc[arb * 16][acb * 16], acc, D,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
  }

  // Normalized write-back (T-tail rows guarded).
  for (int e = tid; e < kTq * D; e += blockDim.x) {
    const int r = e / D, c = e % D;
    const int t = t0 + r;
    if (t < T) od[t * os0 + h * os1 + c * os2] = static_cast<__nv_bfloat16>(
        Oacc[r][c] * Saux[2][r]);
  }
}

}  // namespace AttnTensor

// Tensor-core fused bidirectional attention attempt. Returns true when the
// kernel was launched (caller returns), false to take the portable kernel
// (also on launch failure). Only bhalf_t D in {64, 128} D-contiguous
// shapes past a small-size floor run here, on the 64-wide engine when the
// device fields its shared memory and the 32-wide engine otherwise;
// everything else (float, odd head dims, strided/unaligned views, tiny
// shapes, pre-SM80, non-CUDA) declines to the portable kernel, which also
// owns the Ktot == 0 contract.
template <class ExecSpace, class QView, class KView, class VView, class OView>
bool try_bidir_sdpa_tensor_cuda(const QView& q, const KView& k, const VView& v,
                                const OView& out, float scale) {
  constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (!is_cuda) {
    (void)q;
    (void)k;
    (void)v;
    (void)out;
    (void)scale;
    return false;
  } else {
    using Scalar = std::remove_const_t<typename QView::value_type>;
    constexpr bool is_b =
        std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>;
    if constexpr (!is_b) {
      (void)q;
      (void)k;
      (void)v;
      (void)out;
      (void)scale;
      return false;
    } else {
      static_assert(!KOKKOS_BHALF_T_IS_FLOAT,
                    "tensor path needs hardware bfloat16");
      static_assert(sizeof(Kokkos::Experimental::bhalf_t) == 2 &&
                        alignof(Kokkos::Experimental::bhalf_t) == 2,
                    "bhalf_t must reinterpret as __nv_bfloat16");
      const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
      const size_t Hkv = k.extent(1), K = k.extent(0);
      if (T == 0 || K == 0 || Hq == 0 || Hkv == 0 || D == 0) return false;
      if (k.extent(2) != D || v.extent(0) != K || v.extent(1) != Hkv ||
          v.extent(2) != D || out.extent(0) != T || out.extent(1) != Hq ||
          out.extent(2) != D)
        return false;
      if (D != 64 && D != 128) return false;
      // One full tile per CTA minimum: tiny shapes stay on the portable
      // kernel (which also keeps the tight-tolerance unit tests exact).
      if (T < 32 || K < 32) return false;
      constexpr size_t kMax =
          static_cast<size_t>((std::numeric_limits<int>::max)());
      if (T > kMax || K > kMax || Hq > kMax || Hkv > kMax) return false;
      if (AttnTensor::device_arch_major() < 8) return false;
      // D-contiguous rows (stride(2) == 1) with 16 B-aligned bases and row
      // pitches so the cooperative float4 loads are always in bounds.
      if (q.stride(2) != 1 || k.stride(2) != 1 || v.stride(2) != 1)
        return false;
      auto aligned16 = [](const void* p) {
        return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
      };
      if (!aligned16(q.data()) || !aligned16(k.data()) ||
          !aligned16(v.data()))
        return false;
      auto pitch_ok = [](size_t s) { return ((s * 2) % 16) == 0; };
      if (!pitch_ok(q.stride(0)) || !pitch_ok(q.stride(1)) ||
          !pitch_ok(k.stride(0)) || !pitch_ok(k.stride(1)) ||
          !pitch_ok(v.stride(0)) || !pitch_ok(v.stride(1)))
        return false;
      using GElem = __nv_bfloat16;
      const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
      // Query-tile width: 64 halves the K/V re-read traffic (arithmetic
      // intensity doubles, the binding roof for this streaming kernel).
      // Measured on RTX 3090: at D=64 the wide tile fits the default
      // budget (~46 KB, 2 blocks/SM) and wins; at D=128 it needs opt-in
      // shared (~79 KB, 1 block/SM) and the lost occupancy outweighs the
      // traffic saved (40 ms vs 34 ms at Sj=4608), so D=128 stays narrow.
      // The launcher falls back to the 32-wide engine whenever the device
      // cannot field the wide tile.
      const size_t dsz = static_cast<size_t>(D);
      auto need_bytes = [&](int tq) {
        return static_cast<size_t>(tq) * dsz * 2 +        // Q
               2 * 32 * dsz * 2 +                         // K, V slabs
               static_cast<size_t>(tq) * 32 * 4 +         // S scores
               static_cast<size_t>(tq) * 32 * 2 +         // W weights
               static_cast<size_t>(tq) * dsz * 4 +         // Oacc
               3 * static_cast<size_t>(tq) * 4;            // Saux
      };
      // Wide engine only where measured faster (see comment above).
      const size_t need64 = need_bytes(64);
      bool wide = (D == 64);
      // Launch helper: one instantiation per (D, TQ); the wide D=128
      // engine opts into the extra shared memory it declares.
      const bool launched = [&] {
        const int Ti = static_cast<int>(T), Hi = static_cast<int>(Hq),
                  Ki = static_cast<int>(K), Hvi = static_cast<int>(Hkv);
        const int qs0i = static_cast<int>(q.stride(0)),
                  qs1i = static_cast<int>(q.stride(1)),
                  ks0i = static_cast<int>(k.stride(0)),
                  ks1i = static_cast<int>(k.stride(1)),
                  vs0i = static_cast<int>(v.stride(0)),
                  vs1i = static_cast<int>(v.stride(1)),
                  os0i = static_cast<int>(out.stride(0)),
                  os1i = static_cast<int>(out.stride(1)),
                  os2i = static_cast<int>(out.stride(2));
        const GElem* qd = reinterpret_cast<const GElem*>(q.data());
        const GElem* kd = reinterpret_cast<const GElem*>(k.data());
        const GElem* vd = reinterpret_cast<const GElem*>(v.data());
        GElem* od = reinterpret_cast<GElem*>(const_cast<Scalar*>(out.data()));
        // Dynamic-shared launch: the footprint rides in the launch config
        // (static over 48 KB is rejected by ptxas); the attribute call is
        // harmless when the footprint fits the default budget.
        if (D == 128) {
          if (wide) {
            const int ntiles = (Ti + 63) / 64;
            cudaFuncSetAttribute(
                (const void*)AttnTensor::bidir_sdpa_kernel<128, 64>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(need64));
            AttnTensor::bidir_sdpa_kernel<128, 64>
                <<<ntiles * Hi, 128, need64, stream>>>(
                    qd, kd, vd, od, Ti, Hi, Hvi, Ki, qs0i, qs1i, ks0i, ks1i,
                    vs0i, vs1i, os0i, os1i, os2i, scale);
          } else {
            const size_t need32 = need_bytes(32);
            const int ntiles = (Ti + 31) / 32;
            AttnTensor::bidir_sdpa_kernel<128, 32>
                <<<ntiles * Hi, 128, need32, stream>>>(
                    qd, kd, vd, od, Ti, Hi, Hvi, Ki, qs0i, qs1i, ks0i, ks1i,
                    vs0i, vs1i, os0i, os1i, os2i, scale);
          }
        } else if (wide) {
          const int ntiles = (Ti + 63) / 64;
          cudaFuncSetAttribute(
              (const void*)AttnTensor::bidir_sdpa_kernel<64, 64>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(need64));
          AttnTensor::bidir_sdpa_kernel<64, 64>
              <<<ntiles * Hi, 128, need64, stream>>>(
                  qd, kd, vd, od, Ti, Hi, Hvi, Ki, qs0i, qs1i, ks0i, ks1i,
                  vs0i, vs1i, os0i, os1i, os2i, scale);
        } else {
          const size_t need32 = need_bytes(32);
          const int ntiles = (Ti + 31) / 32;
          AttnTensor::bidir_sdpa_kernel<64, 32>
              <<<ntiles * Hi, 128, need32, stream>>>(
                  qd, kd, vd, od, Ti, Hi, Hvi, Ki, qs0i, qs1i, ks0i, ks1i,
                  vs0i, vs1i, os0i, os1i, os2i, scale);
        }
        return cudaGetLastError() == cudaSuccess;
      }();
      return launched;
    }
  }
}

#else

// Non-CUDA (or pre-SM80-only) builds: no tensor path, always decline to the
// portable kernel. Unnamed parameters dodge -Wunused-parameter under
// -Wextra/-Werror.
template <class ExecSpace, class QView, class KView, class VView, class OView>
bool try_bidir_sdpa_tensor_cuda(const QView&, const KView&, const VView&,
                                const OView&, float) {
  return false;
}

#endif

}  // namespace Impl
}  // namespace KokkosDNN

#endif
