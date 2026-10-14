#ifndef KOKKOSDNN_GEMM_HPP
#define KOKKOSDNN_GEMM_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#ifdef KOKKOSDNN_ENABLE_CUBLAS
#include "tpl/cublas/KokkosDNN_CuBLAS.hpp"
#endif

namespace KokkosDNN {
namespace Impl {

// ---------------------------------------------------------------------------
// CUDA tiled fallback GEMM: design notes (breadcrumbs for future tuners).
//
// Dispatch ladder for gemm() below:
//   1. cuBLAS on CUDA with handle.use_cublas: sgemm for float,
//      cublasGemmEx (R_16BF I/O, FP32 accumulate) for bhalf_t -- TPL.
//   2. Tiled TeamPolicy kernel here (CUDA fallback for other scalars and
//      when cuBLAS is disabled/fails; also the bhalf_t path's safety net
//      on arches without BF16 support) -- ~70% of cuBLAS on Llama shapes.
//   3. Naive MDRange kernel (non-CUDA backends) -- simple, memory-bound.
//
// Why tiling: the naive kernel gives every (i,j) output thread its own
// K-loop over *global* memory -- 2 FP32 loads per FMA, arithmetic intensity
// 0.25 FLOP/byte, 0 bytes of shared memory. Roofline then caps it at
// ~HBM_bw * 0.25 ~= 230 GFLOPS on an RTX 3090, which is exactly what nsys
// showed (229 GFLOPS, 0 B shared, 30 regs/thread) against 1303 GFLOPS for
// cuBLAS's `ampere_sgemm_64x32_sliced1x4_tn` (27 KB shared, 82 regs/thread).
// Tiling the K-loop in shared memory makes each weight byte serve TM
// outputs instead of 1, cutting DRAM traffic ~20x on the benchmark shapes
// (M=16/N=4096/K=4096: ~100 MB vs ~2 GB).
//
// Tile geometry (tuned for skinny-M Llama inference on Ampere):
//   Team (block) = 256 threads owns a TMxTN = 32x128 output tile.
//   Thread mapping: tr = rank>>5 (8 row groups), tc = rank&31 (32 col
//   groups); each thread accumulates an RTxCT = 4x4 register block, i.e.
//   16 FMAs per K-step. K is streamed in TK = 32-wide slabs through shared.
//   Shared budget: (32+128)*33 floats ~= 21 KB (fits 4+ resident blocks/SM).
//   Grid: tilesM*tilesN teams, e.g. 32 teams for M=16/N=4096 -- enough to
//   fill the GPU but small-M waves under-fill; that is inherent to the
//   shape, cuBLAS is launch-bound there too.
//
// Two hard-won pitfalls, do not "simplify" these away:
//   * Shared column stride is TK+1 = 33, not 32. With stride 32, the 32
//     threads of a warp touch 32 rows at one column; row_stride*4 B = 128 B
//     aliases them all onto a single bank -> 32-way conflict on every
//     operand fetch (measured 4x slowdown). 33 = 1 mod 32 spreads them.
//   * The kk slab loop must stay ROLLED (#pragma unroll 1). Fully unrolling
//     keeps hundreds of operands live -> 170+ regs/thread -> 1 block/SM
//     (measured: slower than the naive kernel). Likewise UseVec is a
//     template parameter, not a runtime branch, to keep the live set small
//     (~72 regs/thread -> 2 blocks/SM on shared-memory limits).
//
// Re-tuning cheat sheet: nsys (no root needed) reports grid/block/regs/
// shared/duration per kernel -- check regs/thread first when touching the
// inner loop, and confirm shared use is ~= shBytes (nsys `dynamicShared`).
// ncu needs NVreg_RestrictProfilingToAdminHost=0 (unavailable in most
// containers), so SASS-level work goes through cuobjdump on the bench binary.
// Correctness net: tests/Test_Gemm.cpp covers only tiny odd shapes (scalar
// path). There is no committed test for the vec float4 path or M/N/K tile
// edges -- add one (e.g. M=17/N=130/K=68 vs host reference, alpha/beta !=
// {1,0}) if you change thread mapping or the load paths.
// ---------------------------------------------------------------------------
constexpr int GemmTM = 32;
constexpr int GemmTN = 128;
constexpr int GemmTK = 32;  // K-slab width; see GemmSK for the store stride.
constexpr int GemmSK = GemmTK + 1;  // padded shared stride (bank conflicts).
constexpr int GemmTeamSize = 256;
constexpr int GemmRT = 4;  // rows per thread (GemmTM * GemmTN / team / CT)
constexpr int GemmCT = 4;  // cols per thread

#ifdef KOKKOS_ENABLE_CUDA
// Tiled fallback GEMM for CUDA exec spaces: Y = alpha*X*W^T + beta*Y.
// Geometry/mapping rationale lives in the design-notes block above; the
// code below is the direct transcription: cooperative global->shared load
// (float4 vectorized for aligned float, scalar + on-the-fly to-float for
// bhalf_t/odd shapes), team_barrier, register-blocked outer product over
// the slab, repeat over K, guarded alpha/beta write-back. Slab loads are
// zero-padded so M/N/K edges need no special-case tiles.
template <class ExecSpace, class Scalar, bool UseVec>
struct TiledGemmFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> X, W, Y;
  float alpha_f, beta_f;
  size_t M, N, K, tilesN, nSlabs, shBytes;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const size_t tile = static_cast<size_t>(team.league_rank());
    const size_t i0   = (tile / tilesN) * GemmTM;
    const size_t j0   = (tile % tilesN) * GemmTN;
    const int t       = team.team_rank();
    const int tr      = t >> 5;  // 0..7, row group
    const int tc      = t & 31;  // 0..31, col group

    float* sh =
        static_cast<float*>(team.team_scratch(0).get_shmem(shBytes));
    float* sX = sh;                          // [GemmTM][GemmSK]
    float* sW = sh + size_t(GemmTM) * GemmSK;  // [GemmTN][GemmSK]

    const size_t sx0 = X.stride(0), sw0 = W.stride(0), sy0 = Y.stride(0);
    const Scalar* Xd = X.data();
    const Scalar* Wd = W.data();
    Scalar* Yd       = Y.data();

    float acc[GemmRT][GemmCT] = {};

    for (size_t s = 0; s < nSlabs; ++s) {
      const size_t k0 = s * GemmTK;
      // -- cooperative global -> shared, zero-padded at M/N/K edges.
      // Load layout: linearization is row-major within the slab so
      // consecutive threads read consecutive columns -> fully coalesced
      // 128 B segments (X slab rows and W slab rows are K-contiguous).
      if constexpr (UseVec) {
        // 1 float4 per thread for X (256 threads -> 32x8 quads),
        // 4 float4s per thread for W (256 threads -> 128x8 quads).
        const float4* Xv = reinterpret_cast<const float4*>(Xd);
        const float4* Wv = reinterpret_cast<const float4*>(Wd);
        const size_t Kv4 = K / 4, k04 = k0 / 4;
        {
          const size_t e  = static_cast<size_t>(t);
          const size_t r  = e / 8, c4 = e % 8;
          const size_t gi = i0 + r, gk4 = k04 + c4;
          float4 v = {0, 0, 0, 0};
          if (gi < M && gk4 < Kv4) v = Xv[gi * (sx0 / 4) + gk4];
          float* dst     = &sX[r * GemmSK];
          dst[4 * c4 + 0] = v.x;
          dst[4 * c4 + 1] = v.y;
          dst[4 * c4 + 2] = v.z;
          dst[4 * c4 + 3] = v.w;
        }
        for (int l = 0; l < 4; ++l) {
          const size_t e  = static_cast<size_t>(t) * 4 + l;
          const size_t r  = e / 8, c4 = e % 8;
          const size_t gj = j0 + r, gk4 = k04 + c4;
          float4 v = {0, 0, 0, 0};
          if (gj < N && gk4 < Kv4) v = Wv[gj * (sw0 / 4) + gk4];
          float* dst      = &sW[r * GemmSK];
          dst[4 * c4 + 0] = v.x;
          dst[4 * c4 + 1] = v.y;
          dst[4 * c4 + 2] = v.z;
          dst[4 * c4 + 3] = v.w;
        }
      } else {
        // Scalar path (odd K/strides, and all bhalf_t): 4 X elems and
        // 16 W elems per thread, converted to float in shared.
        for (int l = 0; l < 4; ++l) {
          const size_t e  = static_cast<size_t>(t) * 4 + l;
          const size_t r  = e / GemmTK, c = e % GemmTK;
          const size_t gi = i0 + r, gk = k0 + c;
          float v = 0;
          if (gi < M && gk < K) v = static_cast<float>(Xd[gi * sx0 + gk]);
          sX[r * GemmSK + c] = v;
        }
        for (int l = 0; l < 16; ++l) {
          const size_t e  = static_cast<size_t>(t) * 16 + l;
          const size_t r  = e / GemmTK, c = e % GemmTK;
          const size_t gj = j0 + r, gk = k0 + c;
          float v = 0;
          if (gj < N && gk < K) v = static_cast<float>(Wd[gj * sw0 + gk]);
          sW[r * GemmSK + c] = v;
        }
      }
      team.team_barrier();
      // -- register-blocked outer product over the slab (16 FMA/ thread /
      // K-step; sX row is warp-broadcast, sW column is bank-conflict-free
      // thanks to the GemmSK stride). The kk loop stays rolled: unrolling
      // it explodes the live set (16 acc + operands x 32) into 170+
      // regs/thread and craters occupancy to 1 block/SM.
#pragma unroll 1
      for (int kk = 0; kk < GemmTK; ++kk) {
        float xr[GemmRT];
#pragma unroll
        for (int r = 0; r < GemmRT; ++r)
          xr[r] = sX[(tr * GemmRT + r) * GemmSK + kk];
#pragma unroll
        for (int r = 0; r < GemmRT; ++r) {
#pragma unroll
          for (int c = 0; c < GemmCT; ++c)
            acc[r][c] += xr[r] * sW[(tc * GemmCT + c) * GemmSK + kk];
        }
      }
      team.team_barrier();
    }

    // -- write-back with alpha/beta, guarded at M/N edges --
    for (int r = 0; r < GemmRT; ++r) {
      for (int c = 0; c < GemmCT; ++c) {
        const size_t i = i0 + size_t(tr * GemmRT + r);
        const size_t j = j0 + size_t(tc * GemmCT + c);
        if (i < M && j < N) {
          float yold =
              (beta_f != 0) ? beta_f * static_cast<float>(Yd[i * sy0 + j])
                            : 0.0f;
          Yd[i * sy0 + j] = static_cast<Scalar>(alpha_f * acc[r][c] + yold);
        }
      }
    }
  }
};

template <class ExecSpace, class Scalar>
void gemm_tiled(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
                const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
                float alpha_f, float beta_f,
                const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
                const std::string& label) {
  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
  if (M == 0 || N == 0 || K == 0) return;
  const size_t tilesM = (M + GemmTM - 1) / GemmTM;
  const size_t tilesN = (N + GemmTN - 1) / GemmTN;
  const size_t nSlabs = (K + GemmTK - 1) / GemmTK;
  const size_t shBytes =
      (size_t(GemmTM) * GemmSK + size_t(GemmTN) * GemmSK) * sizeof(float);

  // float4 cooperative loads need 16 B-aligned row starts on a K divisible
  // by 4: row i starts at i*K floats, so K%4==0 keeps every row 16 B
  // aligned whenever the base pointer is (all Llama shapes qualify; odd
  // test shapes take the scalar path). Checked at runtime because views may
  // in principle be strided; all call sites pass contiguous views today.
  bool use_vec = false;
  if constexpr (std::is_same_v<Scalar, float>) {
    const auto ax = reinterpret_cast<uintptr_t>(X.data());
    const auto aw = reinterpret_cast<uintptr_t>(W.data());
    use_vec = (K % 4 == 0) && (ax % 16 == 0) && (aw % 16 == 0) &&
              ((X.stride(0) * sizeof(float)) % 16 == 0);
  }

  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(tilesM * tilesN),
                                       GemmTeamSize);
  policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));

  if (use_vec) {
    TiledGemmFunctor<ExecSpace, Scalar, true> f{X, W, Y, alpha_f, beta_f, M, N,
                                                K, tilesN, nSlabs, shBytes};
    Kokkos::parallel_for(label, policy, f);
  } else {
    TiledGemmFunctor<ExecSpace, Scalar, false> f{X, W, Y, alpha_f, beta_f, M, N,
                                                 K, tilesN, nSlabs, shBytes};
    Kokkos::parallel_for(label, policy, f);
  }
}
#endif  // KOKKOS_ENABLE_CUDA

}  // namespace Impl

// Y = alpha * X * W^T + beta * Y
//   X : [M, K] row-major activations
//   W : [N, K] row-major weights (HF Linear: [out_features, in_features])
//   Y : [M, N]
// This is the only GEMM form the Llama PoC needs (no transposes at call
// sites). Scalar-agnostic (float, bhalf_t); accumulation is FP32 and the
// result is cast back to Scalar. Dispatch ladder on CUDA: float goes to
// cuBLAS sgemm and bhalf_t to cublasGemmEx (BF16 I/O, FP32 accumulate)
// on the handle's cached cuBLAS handle (created once per DNNHandle, stream
// re-selected per call); anything else (or a cuBLAS failure) uses the
// shared-memory tiled kernel in Impl (design notes there);
// non-CUDA backends use the naive MDRange kernel (correct, memory-bound).
template <class ExecSpace, class Scalar>
void gemm(const DNNHandle<ExecSpace>& handle, Scalar alpha,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
          Scalar beta,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
          const std::string& label = "KokkosDNN::gemm") {
  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
  if (W.extent(1) != K || Y.extent(0) != M || Y.extent(1) != N)
    throw std::invalid_argument("gemm: dimension mismatch");

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
  constexpr bool is_cuda =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaSpace> ||
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaUVMSpace>;
  if constexpr (is_cuda) {
    if (handle.use_cublas) {
      void* raw = handle.cublas_raw();
      if (raw != nullptr) {
        // No fence here: cublas_*_with_handle() re-selects the Kokkos CUDA
        // stream on the cached handle every call, so cuBLAS work is ordered
        // with surrounding Kokkos kernels on that stream.
        bool ok = false;
        auto h  = static_cast<cublasHandle_t>(raw);
        if constexpr (std::is_same_v<Scalar, float>) {
          ok = Impl::cublas_gemm_nt_with_handle(
              h, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), alpha, X.data(), W.data(), beta, Y.data());
        } else if constexpr (std::is_same_v<
                                 Scalar, Kokkos::Experimental::bhalf_t>) {
          // FP32 alpha/beta: cublasGemmEx with CUBLAS_COMPUTE_32F matches
          // the native kernel's FP32 accumulation.
          ok = Impl::cublas_gemm_nt_bf16_with_handle(
              h, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), static_cast<float>(alpha), X.data(),
              W.data(), static_cast<float>(beta), Y.data());
        }
        if (ok) return;
        // else fall through to native (unsupported scalar or cuBLAS failure,
        // e.g. no BF16 tensor support on the arch)
      }
    }
  }
#else
  (void)handle;
#endif
#ifdef KOKKOS_ENABLE_CUDA
  // CUDA fallback: shared-memory tiled kernel (design notes at Impl top).
  // The portable MDRange below stays for other backends -- it is the only
  // GEMM they get, so keep it obviously correct; optimize here instead.
  constexpr bool is_cuda_exec =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaSpace> ||
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaUVMSpace>;
  if constexpr (is_cuda_exec) {
    const float alpha_f = static_cast<float>(alpha);
    const float beta_f  = static_cast<float>(beta);
    Impl::gemm_tiled<ExecSpace, Scalar>(X, W, alpha_f, beta_f, Y, label);
    return;
  }
#endif
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f  = static_cast<float>(beta);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {M, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k)
          acc += static_cast<float>(X(i, k)) * static_cast<float>(W(j, k));
        Y(i, j) = static_cast<Scalar>(alpha_f * acc +
                                      beta_f * static_cast<float>(Y(i, j)));
      });
}

// Rank-1 (single token) variant used in decode: y[n] = sum_k x[k]*W[n,k].
// This is GEMM with M=1 at the pointer level, so it shares gemm's full
// dispatch ladder: cuBLAS on the handle's cached handle for float/bhalf_t
// on CUDA (sgemm / cublasGemmEx, FP32 accumulation), native kernel
// otherwise. The Llama lm_head (last-token hidden -> vocab logits) calls
// this, replacing the old naive parallel_for(vocab) head.
template <class ExecSpace, class Scalar>
void gemv(const DNNHandle<ExecSpace>& handle,
          const Kokkos::View<Scalar*, ExecSpace>& x,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
          const Kokkos::View<Scalar*, ExecSpace>& y,
          const std::string& label = "KokkosDNN::gemv") {
  const size_t K = x.extent(0), N = W.extent(0);
  if (W.extent(1) != K || y.extent(0) != N)
    throw std::invalid_argument("gemv: dimension mismatch");
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
  constexpr bool is_cuda =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaSpace> ||
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaUVMSpace>;
  if constexpr (is_cuda) {
    if (handle.use_cublas && N > 0 && K > 0) {
      void* raw = handle.cublas_raw();
      if (raw != nullptr) {
        // No fence here: cublas_*_with_handle() re-selects the Kokkos CUDA
        // stream on the cached handle every call, so cuBLAS work is ordered
        // with surrounding Kokkos kernels on that stream.
        bool ok = false;
        auto h  = static_cast<cublasHandle_t>(raw);
        if constexpr (std::is_same_v<Scalar, float>) {
          ok = Impl::cublas_gemm_nt_with_handle(h, 1, static_cast<int>(N),
                                               static_cast<int>(K), 1.0f,
                                               x.data(), W.data(), 0.0f,
                                               y.data());
        } else if constexpr (std::is_same_v<
                                 Scalar, Kokkos::Experimental::bhalf_t>) {
          ok = Impl::cublas_gemm_nt_bf16_with_handle(
              h, 1, static_cast<int>(N), static_cast<int>(K), 1.0f, x.data(),
              W.data(), 0.0f, y.data());
        }
        if (ok) return;
        // else fall through to native (unsupported scalar or cuBLAS failure)
      }
    }
  }
#else
  (void)handle;
#endif
  if (N == 0 || K == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k)
          acc += static_cast<float>(x(k)) * static_cast<float>(W(j, k));
        y(j) = static_cast<Scalar>(acc);
      });
}

}  // namespace KokkosDNN

#endif
