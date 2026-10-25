// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GEMMTENSORCUDA_HPP
#define KOKKOSDNN_GEMMTENSORCUDA_HPP

// CUDA-only tensor-core GEMM for the native (non-TPL) path:
//   Y = alpha * X * W^T + beta * Y + bias (bias optional, row-broadcast).
//
// This is the one backend-gated kernel in the library, and the gating is
// deliberate: tensor cores (wmma) exist only on NVIDIA SM80+ hardware, so no
// portable Kokkos spelling can reach them. Everything else in the GEMM
// ladder stays portable (see KokkosDNN_Gemm.hpp design notes). Dispatch
// order in gemm()/gemm_bias() is unchanged: cuBLAS/hipBLAS/Accelerate first,
// then this kernel (CUDA + SM80+ + float/bhalf_t + K/N depth), then the
// portable SIMT tiled kernel, which always remains the correctness fallback
// (a failed launch also falls back -- the launcher returns false).
//
// Numerics (read before comparing against FP32 references):
//   * float   : TF32 tensor cores -- inputs rounded to 10-bit mantissa on
//               fragment load (the same precision PyTorch and cuBLAS-TF32
//               mode compute at), FP32 accumulation. Relative error ~1e-3.
//               Disable per-handle with use_tensor_cores = false for bit
//               parity with the portable kernel.
//   * bhalf_t : BF16 tensor cores -- exact BF16 products, FP32 accumulation
//               (same order-of-magnitude accuracy as the portable kernel,
//               summation order differs by ~1 ulp).
// Shapes are fully general: M/N edges are masked (zero-padded shared tiles
// + guarded global stores) and K is streamed in TK slabs with a guarded
// tail, so any M/N/K works as long as it passes the launcher gates
// (alignment for 16 B bulk copies, M >= 64, K >= 128, N >= 64 -- tensor
// cores need K/N depth and a full CTA row-tile to pay off; small shapes
// stay on the portable kernels, which also keeps the tight-tolerance unit
// tests on the exact path).
//
// Structure (CTA 64x64 output tile, 4 warps = 128 threads):
//   * A tile [64][32] + B tile [64][32] in shared (SStrideAB = TK+8 padded
//     stride: consecutive rows start at different banks, the same padding
//     trick as NVIDIA's cudaTensorCoreGemm sample's SKEW_HALF).
//   * Per warp (32x32 outputs = 2x2 wmma atoms): KSteps mma_sync per slab
//     (2 for BF16 k=16, 4 for TF32 k=8), accumulators in registers.
//   * Epilogue stages the 64x64 FP32 accumulators through shared, then one
//     cooperative guarded global write applies alpha/beta/bias. Staging
//     keeps a single code path for interior and edge tiles; the extra
//     shared read/write is bandwidth noise next to the K loop.
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "KokkosDNN_config.h"

#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDACC__) &&                      \
    (defined(KOKKOS_ARCH_AMPERE) || defined(KOKKOS_ARCH_ADA89) ||              \
     defined(KOKKOS_ARCH_HOPPER) || defined(KOKKOS_ARCH_BLACKWELL))
#define KOKKOSDNN_IMPL_CUDA_TENSOR_AVAILABLE 1
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#endif

namespace KokkosDNN {
namespace Impl {

#if defined(KOKKOSDNN_IMPL_CUDA_TENSOR_AVAILABLE)
namespace CudaTensor {

// CTA/warp geometry (see header comment). TN=TM=64 keeps the float shared
// budget (A 64x40 + B 64x40 floats + 64x72 staging) at ~39 KB < 48 KB, so no
// dynamic-shared opt-in is needed and two resident blocks still fit on
// larger-SM parts; tensor cores saturate with few warps (each warp issues
// 4 mma_sync per K-step), so the 4-warp CTA is compute-bound, not
// occupancy-bound.
constexpr int TM = 64;
constexpr int TN = 64;
constexpr int TK = 32;
constexpr int WM = 32;
constexpr int WN = 32;
constexpr int SStrideAB = TK + 8;  // padded shared stride (bank conflicts)
constexpr int SStrideC  = TN + 8;  // FP32 epilogue staging stride

template <class Scalar>
struct Traits;

template <>
struct Traits<float> {
  using Global = float;
  static constexpr int KStep = 8;
  using FragA =
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8,
                             nvcuda::wmma::precision::tf32,
                             nvcuda::wmma::row_major>;
  using FragB =
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8,
                             nvcuda::wmma::precision::tf32,
                             nvcuda::wmma::col_major>;
  using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8,
                                       float>;
};

template <>
struct Traits<Kokkos::Experimental::bhalf_t> {
  // Kokkos bhalf_t is alignas(2) wrapping a single __nv_bfloat16, so the
  // view buffers reinterpret exactly (static_asserted in the launcher).
  using Global = __nv_bfloat16;
  static constexpr int KStep = 16;
  using FragA = nvcuda::wmma::fragment<
      nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
      nvcuda::wmma::row_major>;
  // B is stored [N][K] (HF Linear layout), but the matrix_b fragment wants
  // logical [K][N]: col_major reinterprets the same bytes as the transpose
  // (logical (r,c) = base[c*ldm + r] = sB[n0+c][k0+r]). The origin stays the
  // [n][k] tile corner -- only the layout flips.
  using FragB = nvcuda::wmma::fragment<
      nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
      nvcuda::wmma::col_major>;
  using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                                       float>;
};

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

template <class Scalar>
__global__ __launch_bounds__(128) void gemm_tensor_kernel(
    const typename Traits<Scalar>::Global* __restrict__ Xd,
    const typename Traits<Scalar>::Global* __restrict__ Wd,
    Scalar* __restrict__ Yd, const Scalar* __restrict__ bias, float alpha,
    float beta, int M, int N, int K, int sx0, int sw0, int sy0) {
  using Tr    = Traits<Scalar>;
  using GElem = typename Tr::Global;
  constexpr int KStep  = Tr::KStep;
  constexpr int KSteps = TK / KStep;

  __shared__ GElem sA[TM][SStrideAB];
  __shared__ GElem sB[TN][SStrideAB];
  __shared__ float sC[TM][SStrideC];

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int wr   = warp >> 1;
  const int wc   = warp & 1;

  const int tilesN = (N + TN - 1) / TN;
  const int tile   = blockIdx.x;
  const int i0     = (tile / tilesN) * TM;
  const int j0     = (tile % tilesN) * TN;

  // Accumulators for this warp's 2x2 atoms (32x32 outputs), zero-filled.
  typename Tr::FragC acc[WM / 16][WN / 16];
  for (int ar = 0; ar < WM / 16; ++ar)
    for (int ac = 0; ac < WN / 16; ++ac)
      nvcuda::wmma::fill_fragment(acc[ar][ac], 0.0f);

  // 16 B bulk copies move full slabs; the K-tail slab (and only it) uses
  // guarded scalar loads. M/N edges are masked per chunk/element, so only
  // the slab tail needs the slow path. All branches are uniform per block.
  const float4 kZero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int k0 = 0; k0 < K; k0 += TK) {
    if (k0 + TK <= K) {
      // Bulk: TM*TK (resp. TN*TK) elements as 16 B chunks, row-guarded.
      constexpr int ChunksA = TM * TK / 16 * sizeof(GElem);
      constexpr int ChunksB = TN * TK / 16 * sizeof(GElem);
      constexpr int RowQuadsA =
          TK / 16 * sizeof(GElem);  // 16 B chunks per shared row
      constexpr int RowQuadsB = TK / 16 * sizeof(GElem);
      float4* dA = reinterpret_cast<float4*>(&sA[0][0]);
      float4* dB = reinterpret_cast<float4*>(&sB[0][0]);
      // Elements per 16 B chunk and chunks per global row (in chunks):
      // float: 4 elems/chunk, sx0/4 chunks/row, k0/4 first chunk.
      constexpr int ElemsPerChunk = 16 / sizeof(GElem);
      const int sxq = sx0 / ElemsPerChunk, swq = sw0 / ElemsPerChunk;
      const int kq = k0 / ElemsPerChunk;
      for (int l = tid; l < ChunksA; l += blockDim.x) {
        const int r  = l / RowQuadsA, cq = l % RowQuadsA;
        const int gi = i0 + r;
        float4 v     = kZero;
        if (gi < M) {
          const float4* g =
              reinterpret_cast<const float4*>(Xd) + gi * sxq + kq + cq;
          v = *g;
        }
        dA[r * (SStrideAB * sizeof(GElem) / 16) + cq] = v;
      }
      for (int l = tid; l < ChunksB; l += blockDim.x) {
        const int r  = l / RowQuadsB, cq = l % RowQuadsB;
        const int gj = j0 + r;
        float4 v     = kZero;
        if (gj < N) {
          const float4* g =
              reinterpret_cast<const float4*>(Wd) + gj * swq + kq + cq;
          v = *g;
        }
        dB[r * (SStrideAB * sizeof(GElem) / 16) + cq] = v;
      }
    } else {
      // K-tail slab: guarded scalar loads, zero-padded past K.
      for (int e = tid; e < TM * TK; e += blockDim.x) {
        const int r = e / TK, c = e % TK;
        const int gi = i0 + r, gk = k0 + c;
        float v = 0.0f;
        if (gi < M && gk < K)
          v = static_cast<float>(Xd[gi * sx0 + gk]);
        sA[r][c] = static_cast<GElem>(v);
      }
      for (int e = tid; e < TN * TK; e += blockDim.x) {
        const int r = e / TK, c = e % TK;
        const int gj = j0 + r, gk = k0 + c;
        float v = 0.0f;
        if (gj < N && gk < K)
          v = static_cast<float>(Wd[gj * sw0 + gk]);
        sB[r][c] = static_cast<GElem>(v);
      }
    }
    __syncthreads();
    for (int ks = 0; ks < KSteps; ++ks) {
      typename Tr::FragA fa[WM / 16];
      typename Tr::FragB fb[WN / 16];
      for (int ar = 0; ar < WM / 16; ++ar)
        nvcuda::wmma::load_matrix_sync(
            fa[ar], &sA[wr * WM + ar * 16][ks * KStep], SStrideAB);
      // B is stored [N][K] (HF Linear layout), but the matrix_b fragment wants
      // logical [K][N]: col_major reinterprets the same bytes as the transpose
      // (logical (r,c) = base[c*ldm + r] = sB[n0+c][k0+r]). The origin stays
      // the [n][k] tile corner -- only the layout flips.
      for (int ac = 0; ac < WN / 16; ++ac)
        nvcuda::wmma::load_matrix_sync(
            fb[ac], &sB[wc * WN + ac * 16][ks * KStep], SStrideAB);
      for (int ar = 0; ar < WM / 16; ++ar)
        for (int ac = 0; ac < WN / 16; ++ac)
          nvcuda::wmma::mma_sync(acc[ar][ac], fa[ar], fb[ac], acc[ar][ac]);
    }
    __syncthreads();
  }

  // Epilogue: fragments -> shared staging (always full 64x64, in bounds),
  // then one cooperative guarded global write with alpha/beta/bias.
  for (int ar = 0; ar < WM / 16; ++ar)
    for (int ac = 0; ac < WN / 16; ++ac)
      nvcuda::wmma::store_matrix_sync(
          &sC[wr * WM + ar * 16][wc * WN + ac * 16], acc[ar][ac], SStrideC,
          nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int e = tid; e < TM * TN; e += blockDim.x) {
    const int r = e / TN, c = e % TN;
    const int i = i0 + r, j = j0 + c;
    if (i < M && j < N) {
      float v = alpha * sC[r][c];
      if (beta != 0.0f) v += beta * static_cast<float>(Yd[i * sy0 + j]);
      if (bias != nullptr) v += static_cast<float>(bias[j]);
      Yd[i * sy0 + j] = static_cast<Scalar>(v);
    }
  }
}

}  // namespace CudaTensor

// Tensor-core GEMM attempt for the native path. Returns true when the kernel
// was launched (caller returns), false to take the portable kernel. A failed
// launch also returns false, so the portable kernel doubles as the error
// fallback. Scalars other than float/bhalf_t, small/unaligned shapes, old
// GPUs, and non-CUDA spaces all decline here.
template <class ExecSpace, class Scalar>
bool try_gemm_tensor_cuda(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    float alpha_f, float beta_f,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const Scalar* bias) {
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
                      "tensor path needs hardware bfloat16");
        static_assert(sizeof(Kokkos::Experimental::bhalf_t) == 2 &&
                          alignof(Kokkos::Experimental::bhalf_t) == 2,
                      "bhalf_t must reinterpret as __nv_bfloat16");
      }
      const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
      if (M == 0 || N == 0 || K == 0) return false;
      constexpr size_t kMax =
          static_cast<size_t>((std::numeric_limits<int>::max)());
      if (M > kMax || N > kMax || K > kMax) return false;
      // Tensor cores need K/N depth and a full CTA row-tile to pay off;
      // small shapes stay on the portable kernels (which also keeps the
      // tight-tolerance unit tests on the exact FP32 path).
      if (M < 64 || K < 128 || N < 64) return false;
      if (CudaTensor::device_arch_major() < 8) return false;
      // 16 B bulk copies need 16 B-aligned bases and row pitches.
      using GElem = typename CudaTensor::Traits<Scalar>::Global;
      constexpr size_t kAlign = 16;
      if (reinterpret_cast<uintptr_t>(X.data()) % kAlign != 0) return false;
      if (reinterpret_cast<uintptr_t>(W.data()) % kAlign != 0) return false;
      if ((X.stride(0) * sizeof(GElem)) % kAlign != 0) return false;
      if ((W.stride(0) * sizeof(GElem)) % kAlign != 0) return false;
      const int tilesM = static_cast<int>((M + CudaTensor::TM - 1) /
                                          CudaTensor::TM);
      const int tilesN = static_cast<int>((N + CudaTensor::TN - 1) /
                                          CudaTensor::TN);
      const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
      CudaTensor::gemm_tensor_kernel<Scalar>
          <<<tilesM * tilesN, 128, 0, stream>>>(
              reinterpret_cast<const GElem*>(X.data()),
              reinterpret_cast<const GElem*>(W.data()), Y.data(), bias,
              alpha_f, beta_f, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), static_cast<int>(X.stride(0)),
              static_cast<int>(W.stride(0)), static_cast<int>(Y.stride(0)));
      return cudaGetLastError() == cudaSuccess;
    }
  }
}

#else

// Non-CUDA (or pre-SM80-only) builds: no tensor path, always decline to the
// portable kernel. Unnamed parameters dodge -Wunused-parameter under
// -Wextra/-Werror.
template <class ExecSpace, class Scalar>
bool try_gemm_tensor_cuda(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>&,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>&, float,
    float, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>&,
    const Scalar*) {
  return false;
}

#endif

}  // namespace Impl
}  // namespace KokkosDNN

#endif
