// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GEMM_HPP
#define KOKKOSDNN_GEMM_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_Macros.hpp"
#include "KokkosDNN_GemmTensorCuda.hpp"
#ifdef KOKKOSDNN_ENABLE_CUBLAS
#include "tpl/cublas/KokkosDNN_CuBLAS.hpp"
#endif
#ifdef KOKKOSDNN_ENABLE_HIPBLAS
#include "tpl/hipblas/KokkosDNN_HIPBLAS.hpp"
#endif
#ifdef KOKKOSDNN_ENABLE_ACCELERATE
#include "tpl/accelerate/KokkosDNN_Accelerate.hpp"
#endif

namespace KokkosDNN {
namespace Impl {

// ---------------------------------------------------------------------------
// Portable tiled GEMM: design notes (breadcrumbs for future tuners).
//
// Dispatch ladder for gemm() below:
//   1. cuBLAS on CUDA with handle.use_cublas: sgemm for float,
//      cublasGemmEx (R_16BF I/O, FP32 accumulate) for bhalf_t -- TPL.
//      hipBLAS on HIP with handle.use_hipblas mirrors this (hipblasSgemm /
//      hipblasGemmEx, HIP_R_16BF I/O, FP32 accumulate) -- TPL.
//   2. Native CUDA tensor-core kernel (SM80+, float/bhalf_t, gated shapes;
//      TF32 for float -- see KokkosDNN_GemmTensorCuda.hpp). The one
//      backend-gated kernel in the library: tensor cores have no portable
//      spelling. Kill switch handle.use_tensor_cores.
//   3. Portable tiled TeamPolicy kernel (all team-capable backends, all
//      scalars) -- TeamPolicy + team scratch, no backend-gated code, no raw
//      CUDA/cub/wmma intrinsics. FP32 accumulation, result cast to Scalar.
//   4. Naive MDRange kernel (backends whose team_size_max cannot field the
//      team, e.g. Serial with max team size 1) -- obviously correct,
//      memory-bound. Chosen at runtime via team_size_max, no backend
//      #ifdefs.
//   (The old CUDA-only `#ifdef` gate around the tiled kernel is gone: every
//   team-capable backend runs it, including host-parallel ones.)
//
// Why tiling: a naive kernel gives every (i,j) output thread its own
// K-loop over *global* memory -- 2 FP32 loads per FMA, arithmetic intensity
// 0.25 FLOP/byte, 0 bytes of shared memory. Roofline then caps it at
// ~HBM_bw * 0.25 ~= 230 GFLOPS on an RTX 3090, which is exactly what nsys
// showed (229 GFLOPS, 0 B shared, 30 regs/thread) against 1303 GFLOPS for
// cuBLAS's `ampere_sgemm_64x32_sliced1x4_tn` (27 KB shared, 82 regs/thread).
// Tiling the K-loop in shared memory makes each weight byte serve TM
// outputs instead of 1, cutting DRAM traffic ~20x on the benchmark shapes
// (M=16/N=4096/K=4096: ~100 MB vs ~2 GB).
//
// Tile geometries (tuned on Ampere; base also covers large-M MOSS shapes):
//   Base     : team (block) = 256 threads owns TMxTN = 64x128 output tile,
//              RTxCT = 8x4 register block (32 FMA/thread/K-step).
//   Mid      : team = 128 threads owns TMxTN = 32x64, 4x4 block (M <= 64;
//              M=64/N=4096: 128 teams).
//   Thin-fine: team = 128 threads owns TMxTN = 16x32, 2x2 block (M <= 16;
//              M=16/N=4096: 128 teams, no padded-row waste).
//   K is streamed in TK-wide slabs through shared (TK=32 base/mid, TK=64
//   thin-fine -- fewer barriers/exposures pay off for the small kernel);
//   shared budget is (TM+TN)*(TK+1) floats: base (64+128)*33 ~= 25 KB, mid
//   (32+64)*33 ~= 12.7 KB, thin-fine (16+32)*65 ~= 12.5 KB (small teams let
//   many more blocks stay resident per SM for latency hiding).
//   Thread mapping: tr = rank/ColGroups (row group), tc = rank%ColGroups
//   (col group), ColGroups = TN/CT (32 base, 16 mid, 16 thin-fine).
//   Grid: tilesM*tilesN teams; finer N-tiling in the small-M kernels is what
//   feeds skinny waves more parallel teams (parallelism comes from N).
//   (Split-K over concurrent chunks was tried: replicating the grid over K
//   with chunks in flight simultaneously HURT by ~20% -- interleaved
//   K-ranges thrash DRAM locality -- while finer N-tiling with the same K
//   range in flight helps. Keep parallelism on the N axis.)
//
// Three hard-won pitfalls, do not "simplify" these away:
//   * Shared column stride is TK+1 = 33, not 32. With stride 32, the 32
//     threads of a warp touch 32 rows at one column; row_stride*4 B = 128 B
//     aliases them all onto a single bank -> 32-way conflict on every
//     operand fetch (measured 4x slowdown). 33 = 1 mod 32 spreads them.
//   * The kk slab loop must stay (near-)ROLLED (UNROLL=4 for the M <= 16
//     kernel, 2 elsewhere). Partial unrolling multiplies the independent
//     LDS->FMA chains per thread, which matters when teams are scarce
//     (M=32/mid: +50% for unroll 2 vs 1; M=16/thin-fine: +4% for unroll 4
//     vs 2, -6% for 8 -- with 1 block/SM there are too few warps to hide
//     LDS latency otherwise). Fully unrolling keeps hundreds of operands
//     live -> 170+ regs/thread -> 1 block/SM (measured: slower than the
//     naive kernel). Likewise UseVec is a template parameter, not a runtime
//     branch, to keep the live set small.
//   * Global->shared loads must map consecutive threads to consecutive
//     slab columns (e = rank + iter*TeamSize, row-major). The old mapping
//     (e = rank*ElemsPerThread + l) put consecutive threads on different
//     rows / strided columns: for the scalar bhalf_t path every 2 B load
//     pulled its own 32 B sector (~16x traffic amplification on the weight
//     stream -- the ~7x small-M bhalf gap vs cuBLAS). The vector path loads
//     consecutive quads per thread (e = rank*quadsPerThread + l, row-major),
//     which keeps full slab-row segments coalesced.
//
// Re-tuning cheat sheet: nsys (no root needed) reports grid/block/regs/
// shared/duration per kernel -- check regs/thread first when touching the
// inner loop, and confirm shared use is ~= shBytes (nsys `dynamicShared`).
// ncu needs NVreg_RestrictProfilingToAdminHost=0 (unavailable in most
// containers), so SASS-level work goes through cuobjdump on the bench binary.
// Correctness net: tests/Test_Gemm.cpp covers tiny odd shapes (scalar path)
// plus tile-edge shapes (M=17/N=130/K=68 mid geometry, M=13/N=70/K=67
// thin-fine geometry, alpha/beta != {1,0}) and a Serial-space instantiation
// of the portable kernel -- extend those if you change thread mapping, the
// load paths, or the geometry thresholds.
// ---------------------------------------------------------------------------
constexpr int GemmTM = 64;
constexpr int GemmTN = 128;
constexpr int GemmTK = 32;  // K-slab width; see GemmSK for the store stride.
constexpr int GemmSK = GemmTK + 1;  // padded shared stride (bank conflicts).
constexpr int GemmTeamSize = 256;
constexpr int GemmRT = 8;  // rows per thread (GemmTM * GemmTN / team / CT)
constexpr int GemmCT = 4;  // cols per thread

// Small-M specializations for inference shapes (thresholds in gemm_tiled;
// sweep history: 2x2/256-team, 4x4/64-team, 2x4, 4x2, split-K variants were
// all measured -- finer N-tiling with the same K range in flight wins;
// concurrent split-K over K hurt ~20% via DRAM locality loss):
//   thin-fine (M <= 16): TM=16/TN=32/2x2/TK=64 on 128-thread teams -- no
//             padded-row waste, 128 teams on M=16/N=4096, ~12.5 KB shared.
//   mid  (M <= 64): TM=32/TN=64/4x4/TK=32 on 128-thread teams -- halves
//             shared per team vs base (more resident blocks/SM), 128 teams
//             on M=64/N=4096, ~12.7 KB shared.
constexpr int GemmThinFineTM = 16;
constexpr int GemmThinFineTN = 32;
constexpr int GemmThinFineTK = 64;  // wider slabs: fewer barriers/exposures
constexpr int GemmThinFineRT = 2;
constexpr int GemmThinFineCT = 2;
constexpr int GemmMidTM = 32;
constexpr int GemmMidTN = 64;
constexpr int GemmMidTK = 32;
constexpr int GemmMidRT = 4;
constexpr int GemmMidCT = 4;

// cuBLAS NT layout guard: the cublasSgemm/cublasGemmEx calls below pass the
// extents as leading dims (lda == K for X/W, N for Y), which is exact only
// for row-pitch == extent views. Any padded/strided view (stride(0) !=
// extent) or > INT_MAX extent must decline TPL and run the native kernel --
// cuBLAS would otherwise silently read/write the wrong elements. All LFM
// call sites pass contiguous views (scratch subviews slice only the row
// axis of LayoutRight buffers, whose row pitch still equals K), so this
// guard never fires there; it protects future/strided callers.
// (LayoutRight guarantees stride(1) == 1, so checking stride(0) fully
// characterizes the leading dim.)
inline bool gemm_nt_cublas_layout_ok(size_t M, size_t N, size_t K, size_t sx0,
                                      size_t sw0, size_t sy0) {
  constexpr size_t kMax =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  return M > 0 && N > 0 && K > 0 && M <= kMax && N <= kMax && K <= kMax &&
         sx0 == K && sw0 == K && sy0 == N;
}

// Rank-1 (M == 1) variant of the guard: x/y are contiguous by construction,
// only the weight row pitch and the int range need checking.
inline bool gemv_nt_cublas_layout_ok(size_t N, size_t K, size_t sw0) {
  constexpr size_t kMax =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  return N > 0 && K > 0 && N <= kMax && K <= kMax && sw0 == K;
}

// Portable 4-wide vector element for coalesced global loads: 4 consecutive
// Scalars at natural alignment (16 B for float, 8 B for bhalf_t). Replaces
// the old CUDA-only float4, which did not compile on host backends.
template <class Scalar>
struct alignas(sizeof(Scalar) * 4) GemmVec4 {
  Scalar v[4];
};

// Portable tiled GEMM functor for Y = alpha*X*W^T + beta*Y.
// Template geometry TM/TN/TK (K-slab width) with an RTxCT register block per
// thread; team size is TM*TN/(RT*CT) (256 base, 128 mid/thin-fine).
// UNROLL sets the kk-loop partial unroll (4 for the small kernel, 2
// elsewhere -- deeper chains help scarce-team waves; more starves occupancy).
// Cooperative global->shared load (4-wide vectorized when UseVec, scalar +
// on-the-fly to-float otherwise), team_barrier, register-blocked outer
// product over the slab, repeat over K, guarded alpha/beta write-back.
// Slab loads are zero-padded so M/N/K edges need no special-case tiles.
template <class ExecSpace, class Scalar, int TM, int TN, int TK, int RT, int CT,
          bool UseVec, int UNROLL>
struct TiledGemmFunctor {
  static constexpr int SK = TK + 1;  // padded shared stride (bank conflicts)
  static constexpr int TeamSize  = TM * TN / (RT * CT);
  static constexpr int ColGroups = TN / CT;
  static constexpr int RowGroups = TM / RT;
  static_assert(TM % RT == 0 && TN % CT == 0, "tile must split over threads");
  static_assert(TeamSize == RowGroups * ColGroups, "mapping mismatch");
  static_assert((TM * TK) % TeamSize == 0 && (TN * TK) % TeamSize == 0,
                "slab must split evenly over the team");
  static_assert(!UseVec || TK % 4 == 0, "vector loads need TK divisible by 4");

  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using Vec4   = GemmVec4<Scalar>;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> X, W, Y;
  float alpha_f, beta_f;
  size_t M, N, K, tilesN, nSlabs, shBytes;
  // Optional row-broadcast bias for fused gemm_bias: bias[j] is added in
  // FP32 at write-back (nullptr = plain GEMM). A raw device pointer keeps
  // the functor trivially copyable and lets every existing aggregate
  // initializer keep working (defaulted to nullptr).
  const Scalar* bias = nullptr;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const size_t tile = static_cast<size_t>(team.league_rank());
    const size_t i0   = (tile / tilesN) * TM;
    const size_t j0   = (tile % tilesN) * TN;
    const int t       = team.team_rank();
    const int tr      = t / ColGroups;  // row group
    const int tc      = t % ColGroups;  // col group

    float* sh =
        static_cast<float*>(team.team_scratch(0).get_shmem(shBytes));
    float* sX = sh;                   // [TM][SK]
    float* sW = sh + size_t(TM) * SK;  // [TN][SK]

    const size_t sx0 = X.stride(0), sw0 = W.stride(0), sy0 = Y.stride(0);
    const Scalar* Xd = X.data();
    const Scalar* Wd = W.data();
    Scalar* Yd       = Y.data();

    float acc[RT][CT] = {};

    // Per-thread load counts. Scalar: slab elems / team (exact split, see
    // static_asserts), linearized row-major (e = rank + iter*team) so
    // consecutive threads cover consecutive columns -> coalesced.
    // Vector: VLX/VLW consecutive quads per thread (row-major: each
    // thread's quads form full slab-row segments -> coalesced). All
    // committed geometries cover at least one quad per thread.
    constexpr int XL  = TM * TK / TeamSize;
    constexpr int WL  = TN * TK / TeamSize;
    constexpr int TQX = TM * TK / 4;
    constexpr int TQW = TN * TK / 4;
    constexpr int VLX = TQX / TeamSize;
    constexpr int VLW = TQW / TeamSize;
    static_assert(TQX % TeamSize == 0 && TQW % TeamSize == 0,
                  "slab quads must split evenly over the team");
    constexpr size_t TQC = TK / 4;  // quads per slab row

    for (size_t s = 0; s < nSlabs; ++s) {
      const size_t k0 = s * TK;
      // -- cooperative global -> shared, zero-padded at M/N/K edges.
      // Linearization is row-major within the slab (e = rank + iter*team)
      // so consecutive threads read consecutive columns -> fully coalesced
      // segments (X/W slab rows are K-contiguous).
      if constexpr (UseVec) {
        const Vec4* Xv = reinterpret_cast<const Vec4*>(Xd);
        const Vec4* Wv = reinterpret_cast<const Vec4*>(Wd);
        const size_t Kv4 = K / 4, k04 = k0 / 4;
        const size_t sx4 = sx0 / 4, sw4 = sw0 / 4;
        for (int l = 0; l < VLX; ++l) {
          const size_t e = size_t(t) * VLX + size_t(l);
          const size_t r = e / TQC, c4 = e % TQC;
          const size_t gi = i0 + r, gk4 = k04 + c4;
          Vec4 v;
          v.v[0] = v.v[1] = v.v[2] = v.v[3] = static_cast<Scalar>(0.0f);
          if (gi < M && gk4 < Kv4) v = Xv[gi * sx4 + gk4];
          float* dst = &sX[r * SK + 4 * c4];
          dst[0]     = static_cast<float>(v.v[0]);
          dst[1]     = static_cast<float>(v.v[1]);
          dst[2]     = static_cast<float>(v.v[2]);
          dst[3]     = static_cast<float>(v.v[3]);
        }
        for (int l = 0; l < VLW; ++l) {
          const size_t e = size_t(t) * VLW + size_t(l);
          const size_t r = e / TQC, c4 = e % TQC;
          const size_t gj = j0 + r, gk4 = k04 + c4;
          Vec4 v;
          v.v[0] = v.v[1] = v.v[2] = v.v[3] = static_cast<Scalar>(0.0f);
          if (gj < N && gk4 < Kv4) v = Wv[gj * sw4 + gk4];
          float* dst = &sW[r * SK + 4 * c4];
          dst[0]     = static_cast<float>(v.v[0]);
          dst[1]     = static_cast<float>(v.v[1]);
          dst[2]     = static_cast<float>(v.v[2]);
          dst[3]     = static_cast<float>(v.v[3]);
        }
      } else {
        // Scalar path (odd K/strides/misaligned, all other scalars): XL X
        // elems and WL W elems per thread, converted to float in shared.
        for (int l = 0; l < XL; ++l) {
          const size_t e  = size_t(t) + size_t(l) * size_t(TeamSize);
          const size_t r  = e / TK, c = e % TK;
          const size_t gi = i0 + r, gk = k0 + c;
          float v = 0;
          if (gi < M && gk < K) v = static_cast<float>(Xd[gi * sx0 + gk]);
          sX[r * SK + c] = v;
        }
        for (int l = 0; l < WL; ++l) {
          const size_t e  = size_t(t) + size_t(l) * size_t(TeamSize);
          const size_t r  = e / TK, c = e % TK;
          const size_t gj = j0 + r, gk = k0 + c;
          float v = 0;
          if (gj < N && gk < K) v = static_cast<float>(Wd[gj * sw0 + gk]);
          sW[r * SK + c] = v;
        }
      }
      team.team_barrier();
      // -- register-blocked outer product over the slab (RT*CT FMA/thread/
      // K-step; sX row is warp-broadcast, sW column is bank-conflict-free
      // thanks to the SK stride). Both operands are explicitly staged in
      // registers (xr + wr) so each shared value is loaded once per kk and
      // reused RT/CT times from registers -- the sW address does not depend
      // on r, but without the explicit array the compiler may re-issue the
      // shared load inside the r-loop on some backends. UNROLL (per-geometry
      // template param, 4 for the small kernel, 2 elsewhere) sets the kk
      // partial-unroll: deeper independent LDS->FMA chains help when teams
      // are scarce; full unrolling explodes the live set into 170+ regs/thread
      // and craters occupancy to 1 block/SM.
      if constexpr (UNROLL == 4) {
KOKKOSDNN_PRAGMA_UNROLL_N(4)
        for (int kk = 0; kk < TK; ++kk) {
          float xr[RT];
KOKKOSDNN_PRAGMA_UNROLL()
          for (int r = 0; r < RT; ++r)
            xr[r] = sX[(tr * RT + r) * SK + kk];
          float wr[CT];
KOKKOSDNN_PRAGMA_UNROLL()
          for (int c = 0; c < CT; ++c)
            wr[c] = sW[(tc * CT + c) * SK + kk];
KOKKOSDNN_PRAGMA_UNROLL()
          for (int r = 0; r < RT; ++r) {
KOKKOSDNN_PRAGMA_UNROLL()
            for (int c = 0; c < CT; ++c)
              acc[r][c] += xr[r] * wr[c];
          }
        }
      } else {
KOKKOSDNN_PRAGMA_UNROLL_N(2)
        for (int kk = 0; kk < TK; ++kk) {
          float xr[RT];
KOKKOSDNN_PRAGMA_UNROLL()
          for (int r = 0; r < RT; ++r)
            xr[r] = sX[(tr * RT + r) * SK + kk];
          float wr[CT];
KOKKOSDNN_PRAGMA_UNROLL()
          for (int c = 0; c < CT; ++c)
            wr[c] = sW[(tc * CT + c) * SK + kk];
KOKKOSDNN_PRAGMA_UNROLL()
          for (int r = 0; r < RT; ++r) {
KOKKOSDNN_PRAGMA_UNROLL()
            for (int c = 0; c < CT; ++c)
              acc[r][c] += xr[r] * wr[c];
          }
        }
      }
      team.team_barrier();
    }

    // -- write-back with alpha/beta and optional fused bias, guarded at
    // M/N edges. Bias is added in FP32 before the single output cast
    // (matches torch addmm / cublasLt BIAS-epilogue order; the old split
    // gemm+bias path rounded to Scalar between the two ops, so bf16
    // results may differ from the split path by <= 1 ulp).
    for (int r = 0; r < RT; ++r) {
      for (int c = 0; c < CT; ++c) {
        const size_t i = i0 + size_t(tr * RT + r);
        const size_t j = j0 + size_t(tc * CT + c);
        if (i < M && j < N) {
          float yold =
              (beta_f != 0) ? beta_f * static_cast<float>(Yd[i * sy0 + j])
                            : 0.0f;
          const float bb =
              (bias != nullptr) ? static_cast<float>(bias[j]) : 0.0f;
          Yd[i * sy0 + j] =
              static_cast<Scalar>(alpha_f * acc[r][c] + yold + bb);
        }
      }
    }
  }
};

// Naive MDRange fallback for backends that cannot field a team
// (team_size_max < kernel TeamSize, e.g. Serial): one output element per
// thread, K-loop over global memory. Obviously correct, memory-bound --
// the tiled kernel below is what every team-capable backend runs.
// Optional fused bias (see TiledGemmFunctor::bias).
template <class ExecSpace, class Scalar>
void gemm_naive(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    float alpha_f, float beta_f,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label, const Scalar* bias = nullptr);

// Geometry-parameterized launch: shared-memory tiled kernel (design notes at
// Impl top). Portable -- TeamPolicy + team scratch only. Optional fused
// bias (raw device pointer, nullptr = plain GEMM) for gemm_bias so the
// native path is exactly one kernel launch.
template <class ExecSpace, class Scalar, int TM, int TN, int TK, int RT,
          int CT, int UNROLL>
void gemm_tiled_impl(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    float alpha_f, float beta_f,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label, const Scalar* bias = nullptr) {
  using Functor = TiledGemmFunctor<ExecSpace, Scalar, TM, TN, TK, RT, CT,
                                   false, UNROLL>;
  using FunctorV =
      TiledGemmFunctor<ExecSpace, Scalar, TM, TN, TK, RT, CT, true, UNROLL>;
  using Vec4 = GemmVec4<Scalar>;
  constexpr int SK = TK + 1;

  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
  if (M == 0 || N == 0 || K == 0) return;
  const size_t tilesM = (M + TM - 1) / TM;
  const size_t tilesN = (N + TN - 1) / TN;
  const size_t nSlabs = (K + TK - 1) / TK;
  const size_t shBytes =
      (size_t(TM) * SK + size_t(TN) * SK) * sizeof(float);

  // 4-wide cooperative loads need K divisible by 4 (quad cover), natural
  // alignment of both base pointers, and row strides that keep every row
  // start aligned (all Llama shapes qualify; odd test shapes take the
  // scalar path). Checked at runtime because views may in principle be
  // strided; all call sites pass contiguous views today.
  const auto ax = reinterpret_cast<uintptr_t>(X.data());
  const auto aw = reinterpret_cast<uintptr_t>(W.data());
  const bool use_vec =
      (K % 4 == 0) && (ax % sizeof(Vec4) == 0) &&
      (aw % sizeof(Vec4) == 0) &&
      ((X.stride(0) * sizeof(Scalar)) % sizeof(Vec4) == 0) &&
      ((W.stride(0) * sizeof(Scalar)) % sizeof(Vec4) == 0);

  // Team-capability probe FIRST: constructing the TeamPolicy below aborts
  // on backends that cannot field the team (e.g. Serial, max team size 1),
  // so probe max team size from a team_size-1 policy (valid everywhere) and
  // take the obviously-correct naive kernel when the team does not fit.
  // The query accounts for functor registers and the per-team scratch.
  {
    Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
    probe.set_scratch_size(0, Kokkos::PerTeam(shBytes));
    if (use_vec) {
      FunctorV fv{X, W, Y, alpha_f, beta_f, M, N, K,
                  tilesN, nSlabs, shBytes, bias};
      if (probe.team_size_max(fv, Kokkos::ParallelForTag{}) <
          FunctorV::TeamSize) {
        gemm_naive<ExecSpace, Scalar>(X, W, alpha_f, beta_f, Y, label, bias);
        return;
      }
    } else {
      Functor f{X, W, Y, alpha_f, beta_f, M, N, K,
                tilesN, nSlabs, shBytes, bias};
      if (probe.team_size_max(f, Kokkos::ParallelForTag{}) <
          Functor::TeamSize) {
        gemm_naive<ExecSpace, Scalar>(X, W, alpha_f, beta_f, Y, label, bias);
        return;
      }
    }
  }

  Kokkos::TeamPolicy<ExecSpace> policy(
      static_cast<int>(tilesM * tilesN), Functor::TeamSize);
  policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));

  if (use_vec) {
    FunctorV f{X, W, Y, alpha_f, beta_f, M, N, K,
               tilesN, nSlabs, shBytes, bias};
    Kokkos::parallel_for(label, policy, f);
  } else {
    Functor f{X, W, Y, alpha_f, beta_f, M, N, K,
              tilesN, nSlabs, shBytes, bias};
    Kokkos::parallel_for(label, policy, f);
  }
}

// Naive MDRange fallback for backends that cannot field a team
// (team_size_max < kernel TeamSize, e.g. Serial): one output element per
// thread, K-loop over global memory. Obviously correct, memory-bound --
// the tiled kernel above is what every team-capable backend runs.
template <class ExecSpace, class Scalar>
void gemm_naive(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    float alpha_f, float beta_f,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label, const Scalar* bias) {
  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
  if (M == 0 || N == 0 || K == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {M, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k)
          acc += static_cast<float>(X(i, k)) * static_cast<float>(W(j, k));
        const float bb =
            (bias != nullptr) ? static_cast<float>(bias[j]) : 0.0f;
        Y(i, j) = static_cast<Scalar>(alpha_f * acc +
                                      beta_f * static_cast<float>(Y(i, j)) +
                                      bb);
      });
}

template <class ExecSpace, class Scalar>
void gemm_tiled(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    float alpha_f, float beta_f,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label, const Scalar* bias = nullptr) {
  const size_t M = X.extent(0);
  // Small-M specialization ladder (parallelism comes from N; finer N-tiling
  // multiplies teams with the same K range in flight; no padded-row waste
  // in the M <= 16 kernel):
  //   M <= 16 : thin-fine TM=16/TN=32/2x2/TK=64, 128-thread teams
  //             (M=16/N=4096: 128 teams, ~12.5 KB shared/team).
  //   M <= 64 : mid  TM=32/TN=64/4x4/TK=32, 128-thread teams (M=64/N=4096:
  //             128 teams, ~12.7 KB shared/team).
  //   else    : base TM=64/TN=128/8x4/TK=32, 256-thread teams (M=4096/N=4096:
  //             64x32 = 2048 teams, ~25 KB shared/team).
  if (M <= size_t(GemmThinFineTM)) {
    gemm_tiled_impl<ExecSpace, Scalar, GemmThinFineTM, GemmThinFineTN,
                    GemmThinFineTK, GemmThinFineRT, GemmThinFineCT, 4>(
        X, W, alpha_f, beta_f, Y, label, bias);
  } else if (M <= size_t(GemmMidTM) * 2) {
    gemm_tiled_impl<ExecSpace, Scalar, GemmMidTM, GemmMidTN, GemmMidTK,
                    GemmMidRT, GemmMidCT, 2>(X, W, alpha_f, beta_f, Y, label,
                                             bias);
  } else {
    gemm_tiled_impl<ExecSpace, Scalar, GemmTM, GemmTN, GemmTK, GemmRT, GemmCT,
                    2>(X, W, alpha_f, beta_f, Y, label, bias);
  }
}

// ---------------------------------------------------------------------------
// Native GEMV: one team per N-row, vector-parallel K (design notes).
//
// Why the old kernel was slow: a RangePolicy over N with a scalar K-loop
// gives each warp 32 threads on 32 DIFFERENT rows at the same k -- row
// stride K elements apart, so every 4 B (float) / 2 B (bhalf_t) load pulls
// its own 32 B sector (~8x/16x DRAM amplification). That is the ~4x gap vs
// cuBLAS. Here consecutive threads of a team cooperate on ONE row, so
// consecutive threads read consecutive K -- full sectors, zero waste.
//
// Structure (portable: TeamPolicy + team scratch + parallel_reduce only):
//   * League = N teams (one row each); team size 256 (clamped at runtime
//     via team_size_max for teams that cannot field it, e.g. Serial -> 1).
//   * K is streamed in TK = 2048 slabs: the team cooperatively stages the x
//     slab in scratch once (coalesced, 4-wide when UseVec), then each
//     thread accumulates its strided share (quads in the vec path --
//     explicit 16 B / 8 B loads, so bhalf_t never goes through scalar 2 B
//     sector-wasting loads; scalar elems otherwise) from scratch + the
//     W row, which streams straight from DRAM with no shared pressure.
//   * One TeamThreadRange parallel_reduce over the per-thread partials at
//     the end (no per-row atomics, no per-slab collectives).
//   * UseVec needs K % 4 == 0, natural alignment of both base pointers and
//     quad-aligned W row stride (runtime check, same rule as gemm_tiled;
//     all Llama shapes qualify; odd test shapes take the scalar path,
//     which is still fully coalesced).
//   * alpha/beta write-back mirrors gemm so rank-2 M == 1 can reuse this
//     kernel (same shape, same working set); plain gemv passes 1/0.
// ---------------------------------------------------------------------------
constexpr int GemvTeamSize = 256;
constexpr int GemvTK       = 2048;  // K-slab staged through scratch

template <class ExecSpace, class Scalar, class XView, class YView, bool UseVec>
struct GemvFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using Vec4   = GemmVec4<Scalar>;
  using WView  = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  XView x;
  YView y;
  WView W;
  float alpha_f, beta_f;
  size_t N, K, nSlabs, shBytes, sw0;
  // Optional fused bias for gemv_bias (nullptr = plain GEMV); added in
  // FP32 before the single output cast, same order as the GEMM fusion.
  const Scalar* bias = nullptr;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    const size_t j = static_cast<size_t>(team.league_rank());
    const int Tk   = team.team_size();
    const int t    = team.team_rank();
    float* sh =
        static_cast<float*>(team.team_scratch(0).get_shmem(shBytes));
    float* sx   = sh;            // [GemvTK] staged x slab (FP32)
    float* psum = sh + GemvTK;   // [team_size] per-thread partials
    const Scalar* Xd = x.data();
    const Scalar* Wd = W.data();
    Scalar* Yd       = y.data();

    float acc = 0;
    if constexpr (UseVec) {
      // Vec path: K % 4 == 0 and GemvTK % 4 == 0, so every slab is an exact
      // number of quads -- no tail, no OOB quad reads.
      const Vec4* Xv = reinterpret_cast<const Vec4*>(Xd);
      const Vec4* Wv = reinterpret_cast<const Vec4*>(Wd);
      const size_t sw4 = sw0 / 4;
      for (size_t s = 0; s < nSlabs; ++s) {
        const size_t k0  = s * size_t(GemvTK);
        const size_t cur = (k0 + size_t(GemvTK) <= K) ? size_t(GemvTK) : K - k0;
        const size_t nq  = cur / 4;
        const size_t k04 = k0 / 4;
        // Stage x slab: strided quads keep consecutive threads on
        // consecutive quads (coalesced 16 B / 8 B segments).
        for (size_t q = size_t(t); q < nq; q += size_t(Tk)) {
          const Vec4 v = Xv[k04 + q];
          float* dst   = &sx[4 * q];
          dst[0]       = static_cast<float>(v.v[0]);
          dst[1]       = static_cast<float>(v.v[1]);
          dst[2]       = static_cast<float>(v.v[2]);
          dst[3]       = static_cast<float>(v.v[3]);
        }
        team.team_barrier();
        // Strided quad dots: consecutive threads, consecutive quads.
KOKKOSDNN_PRAGMA_UNROLL_N(4)
        for (size_t q = size_t(t); q < nq; q += size_t(Tk)) {
          const Vec4 w  = Wv[j * sw4 + k04 + q];
          const float* xs = &sx[4 * q];
          acc += xs[0] * static_cast<float>(w.v[0]) +
                 xs[1] * static_cast<float>(w.v[1]) +
                 xs[2] * static_cast<float>(w.v[2]) +
                 xs[3] * static_cast<float>(w.v[3]);
        }
        team.team_barrier();
      }
    } else {
      // Scalar path (odd K / misaligned): still team-cooperative, so
      // consecutive threads cover consecutive K (coalesced sectors).
      for (size_t s = 0; s < nSlabs; ++s) {
        const size_t k0  = s * size_t(GemvTK);
        const size_t cur = (k0 + size_t(GemvTK) <= K) ? size_t(GemvTK) : K - k0;
        for (size_t e = size_t(t); e < cur; e += size_t(Tk))
          sx[e] = static_cast<float>(Xd[k0 + e]);
        team.team_barrier();
        for (size_t e = size_t(t); e < cur; e += size_t(Tk))
          acc += sx[e] * static_cast<float>(Wd[j * sw0 + k0 + e]);
        team.team_barrier();
      }
    }

    psum[t] = acc;
    team.team_barrier();
    float team_sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, Tk),
        KOKKOS_LAMBDA(const int i, float& lsum) { lsum += psum[i]; },
        team_sum);
    if (t == 0) {
      float yold =
          (beta_f != 0) ? beta_f * static_cast<float>(Yd[j]) : 0.0f;
      const float bb =
          (bias != nullptr) ? static_cast<float>(bias[j]) : 0.0f;
      Yd[j] = static_cast<Scalar>(alpha_f * team_sum + yold + bb);
    }
  }
};

// Native GEMV launcher: runtime UseVec selection + team-size probe (same
// pattern as gemm_tiled_impl). FP32 accumulation, result cast to Scalar.
// XView/YView are rank-1 views (managed, or unmanaged row aliases for the
// rank-2 M == 1 reuse below); WView stays rank-2 LayoutRight. Optional
// fused bias (raw device pointer, nullptr = plain GEMV) for gemv_bias and
// the rank-2 M == 1 gemm_bias path.
template <class ExecSpace, class Scalar, class XView, class YView>
void gemv_native(
    const XView& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    float alpha_f, float beta_f, const YView& y,
    const std::string& label, const Scalar* bias = nullptr) {
  using Vec4 = GemmVec4<Scalar>;
  const size_t K = x.extent(0), N = W.extent(0);
  if (N == 0 || K == 0) return;
  const size_t nSlabs  = (K + size_t(GemvTK) - 1) / size_t(GemvTK);
  const size_t shBytes = (size_t(GemvTK) + size_t(GemvTeamSize)) * sizeof(float);
  const size_t sw0     = W.stride(0);

  // 4-wide loads need K divisible by 4 (exact quad cover), natural
  // alignment of both base pointers, and a quad-aligned W row stride.
  // x is a contiguous rank-1 view, so no stride check is needed for it.
  const auto ax = reinterpret_cast<uintptr_t>(x.data());
  const auto aw = reinterpret_cast<uintptr_t>(W.data());
  const bool use_vec =
      (K % 4 == 0) && (ax % sizeof(Vec4) == 0) &&
      (aw % sizeof(Vec4) == 0) &&
      ((sw0 * sizeof(Scalar)) % sizeof(Vec4) == 0);

  // Team-capability probe FIRST (same reason as gemm_tiled_impl): clamp to
  // what the backend can field (Serial -> 1). All loops below stride on
  // the runtime team size, so any size is correct.
  int team_size = GemvTeamSize;
  {
    Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
    probe.set_scratch_size(0, Kokkos::PerTeam(shBytes));
    int mx;
    if (use_vec) {
      GemvFunctor<ExecSpace, Scalar, XView, YView, true> f{
          x, y, W, alpha_f, beta_f, N, K, nSlabs, shBytes, sw0, bias};
      mx = probe.team_size_max(f, Kokkos::ParallelForTag{});
    } else {
      GemvFunctor<ExecSpace, Scalar, XView, YView, false> f{
          x, y, W, alpha_f, beta_f, N, K, nSlabs, shBytes, sw0, bias};
      mx = probe.team_size_max(f, Kokkos::ParallelForTag{});
    }
    if (mx < team_size) team_size = mx;
    if (team_size < 1) team_size = 1;
  }

  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(N), team_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));
  if (use_vec) {
    GemvFunctor<ExecSpace, Scalar, XView, YView, true> f{
        x, y, W, alpha_f, beta_f, N, K, nSlabs, shBytes, sw0, bias};
    Kokkos::parallel_for(label, policy, f);
  } else {
    GemvFunctor<ExecSpace, Scalar, XView, YView, false> f{
        x, y, W, alpha_f, beta_f, N, K, nSlabs, shBytes, sw0, bias};
    Kokkos::parallel_for(label, policy, f);
  }
}

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
// re-selected per call); on HIP the same ladder runs through hipBLAS
// (hipblasSgemm / hipblasGemmEx via the handle's cached hipBLAS handle);
// anything else (or a TPL failure) first tries the native CUDA tensor-core
// kernel (TF32 for float, BF16 tensor for bhalf_t -- SM80+, gated shapes,
// kill switch handle.use_tensor_cores; numerics in KokkosDNN_GemmTensorCuda.hpp)
// and then uses the
// portable shared-memory tiled kernel in Impl (design notes there), with
// thin-fine/mid small-M specializations (TM=16/TN=32/TK=64 for M <= 16,
// TM=32/TN=64/TK=32 for M <= 64), base TM=64/TN=128/8x4 for larger M --
// except rank-2 M == 1, which reuses the
// native team-per-row GEMV kernel (same shape/working set as gemv).
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
  // Backend named deliberately: cuBLAS dispatch genuinely requires CUDA.
  constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (is_cuda) {
    if (handle.use_cublas) {
      void* raw = handle.cublas_raw();
      if (raw != nullptr &&
          Impl::gemm_nt_cublas_layout_ok(M, N, K, X.stride(0), W.stride(0),
                                         Y.stride(0))) {
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
        if (ok) {
          Impl::tpl_note_gemm(true);
          return;
        }
        // else fall through to native (unsupported scalar or cuBLAS failure,
        // e.g. no BF16 tensor support on the arch)
      }
    }
  }
#else
  (void)handle;
#endif
#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
  // Backend named deliberately: hipBLAS dispatch genuinely requires HIP.
  // Mirror of the cuBLAS block above (hipblasSgemm / hipblasGemmEx on the
  // handle's cached hipBLAS handle, stream re-selected per call).
  constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
  if constexpr (is_hip) {
    if (handle.use_hipblas) {
      void* raw = handle.hipblas_raw();
      if (raw != nullptr &&
          Impl::gemm_nt_cublas_layout_ok(M, N, K, X.stride(0), W.stride(0),
                                         Y.stride(0))) {
        // No fence here: hipblas_*_with_handle() re-selects the Kokkos HIP
        // stream on the cached handle every call, so hipBLAS work is ordered
        // with surrounding Kokkos kernels on that stream.
        bool ok = false;
        auto h  = static_cast<hipblasHandle_t>(raw);
        if constexpr (std::is_same_v<Scalar, float>) {
          ok = Impl::hipblas_gemm_nt_with_handle(
              h, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), alpha, X.data(), W.data(), beta, Y.data());
        } else if constexpr (std::is_same_v<
                                 Scalar, Kokkos::Experimental::bhalf_t>) {
          // FP32 alpha/beta: hipblasGemmEx with HIPBLAS_COMPUTE_32F matches
          // the native kernel's FP32 accumulation.
          ok = Impl::hipblas_gemm_nt_bf16_with_handle(
              h, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), static_cast<float>(alpha), X.data(),
              W.data(), static_cast<float>(beta), Y.data());
        }
        if (ok) {
          Impl::tpl_note_gemm(true);
          return;
        }
        // else fall through to native (unsupported scalar or hipBLAS failure,
        // e.g. no BF16 support on the arch)
      }
    }
  }
#endif
#ifdef KOKKOSDNN_ENABLE_ACCELERATE
  // Host-memory spaces only (Serial/OpenMP/Threads): Accelerate's CPU
  // sgemm reads/writes the View buffers directly. float only; other
  // scalars (e.g. bhalf_t) fall through to the native kernel. Stateless,
  // so no handle caching -- fence once so prior Kokkos work on the
  // buffers is complete before the synchronous CPU call.
  constexpr bool is_host_mem_gemm =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace>;
  if constexpr (is_host_mem_gemm) {
    if (handle.use_accelerate) {
      if constexpr (std::is_same_v<Scalar, float>) {
        constexpr size_t kMax = static_cast<size_t>(
            (std::numeric_limits<int>::max)());
        if (M > 0 && N > 0 && K > 0 && M <= kMax && N <= kMax && K <= kMax) {
          Kokkos::fence();
          if (Impl::accelerate_gemm_nt_float(
                  static_cast<int>(M), static_cast<int>(N),
                  static_cast<int>(K), alpha, X.data(), X.stride(0), W.data(),
                  W.stride(0), beta, Y.data(), Y.stride(0))) {
            Impl::tpl_note_gemm(true);
            return;
          }
        }
      }
    }
  }
#endif
  // Native: rank-2 M == 1 is the same working set as GEMV (lm_head/vocab
  // row), so it reuses the team-per-row GEMV kernel with alpha/beta (one
  // team per N-row, vector-parallel K) instead of the tiled GEMM; M > 1
  // first tries the CUDA tensor-core kernel (TF32 for float, BF16 tensor
  // for bhalf_t; SM80+, gated shapes -- kill switch handle.use_tensor_cores)
  // and then uses the portable shared-memory tiled kernel (Impl notes).
  Impl::tpl_note_gemm(false);
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f  = static_cast<float>(beta);
  if (M == 1 && K > 0 && N > 0) {
    Kokkos::View<Scalar*, ExecSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        x1(X.data(), K), y1(Y.data(), N);
    Impl::gemv_native<ExecSpace, Scalar>(x1, W, alpha_f, beta_f, y1, label);
    return;
  }
  if (handle.use_tensor_cores &&
      Impl::try_gemm_tensor_cuda<ExecSpace, Scalar>(X, W, alpha_f, beta_f, Y,
                                                    nullptr))
    return;
  Impl::gemm_tiled<ExecSpace, Scalar>(X, W, alpha_f, beta_f, Y, label);
}

// Rank-1 (single token) variant used in decode: y[n] = sum_k x[k]*W[n,k].
// This is GEMM with M=1 at the pointer level, so it shares gemm's full
// dispatch ladder: cuBLAS on the handle's cached handle for float/bhalf_t
// on CUDA (sgemm / cublasGemmEx, FP32 accumulation), hipBLAS mirror on HIP,
// native kernel otherwise. The Llama lm_head (last-token hidden -> vocab logits) calls
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
  // Backend named deliberately: cuBLAS dispatch genuinely requires CUDA.
  constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (is_cuda) {
    if (handle.use_cublas && Impl::gemv_nt_cublas_layout_ok(N, K, W.stride(0))) {
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
        if (ok) {
          Impl::tpl_note_gemv(true);
          return;
        }
        // else fall through to native (unsupported scalar or cuBLAS failure)
      }
    }
  }
#else
  (void)handle;
#endif
#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
  // Backend named deliberately: hipBLAS dispatch genuinely requires HIP.
  // Mirror of the cuBLAS block above (M=1 path).
  constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
  if constexpr (is_hip) {
    if (handle.use_hipblas && Impl::gemv_nt_cublas_layout_ok(N, K, W.stride(0))) {
      void* raw = handle.hipblas_raw();
      if (raw != nullptr) {
        // No fence here: hipblas_*_with_handle() re-selects the Kokkos HIP
        // stream on the cached handle every call, so hipBLAS work is ordered
        // with surrounding Kokkos kernels on that stream.
        bool ok = false;
        auto h  = static_cast<hipblasHandle_t>(raw);
        if constexpr (std::is_same_v<Scalar, float>) {
          ok = Impl::hipblas_gemm_nt_with_handle(h, 1, static_cast<int>(N),
                                                static_cast<int>(K), 1.0f,
                                                x.data(), W.data(), 0.0f,
                                                y.data());
        } else if constexpr (std::is_same_v<
                                 Scalar, Kokkos::Experimental::bhalf_t>) {
          ok = Impl::hipblas_gemm_nt_bf16_with_handle(
              h, 1, static_cast<int>(N), static_cast<int>(K), 1.0f, x.data(),
              W.data(), 0.0f, y.data());
        }
        if (ok) {
          Impl::tpl_note_gemv(true);
          return;
        }
        // else fall through to native (unsupported scalar or hipBLAS failure)
      }
    }
  }
#endif
#ifdef KOKKOSDNN_ENABLE_ACCELERATE
  // Host-memory M==1 GEMM via Accelerate (same NT sgemm as gemm(); x is a
  // contiguous [K] row, y a contiguous [N] row). float only.
  constexpr bool is_host_mem_gemv =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace>;
  if constexpr (is_host_mem_gemv) {
    if (handle.use_accelerate && N > 0 && K > 0 && x.stride(0) == 1 &&
        y.stride(0) == 1) {
      if constexpr (std::is_same_v<Scalar, float>) {
        constexpr size_t kMaxV = static_cast<size_t>(
            (std::numeric_limits<int>::max)());
        if (N <= kMaxV && K <= kMaxV) {
          Kokkos::fence();
          if (Impl::accelerate_gemm_nt_float(
                  1, static_cast<int>(N), static_cast<int>(K), 1.0f, x.data(),
                  K, W.data(), W.stride(0), 0.0f, y.data(), N)) {
            Impl::tpl_note_gemv(true);
            return;
          }
        }
      }
    }
  }
#endif
  if (N == 0 || K == 0) return;
  // Native: team-per-row GEMV (one team per N-row, vector-parallel K over
  // scratch-staged x slabs; Impl notes). FP32 accumulation.
  Impl::tpl_note_gemv(false);
  Impl::gemv_native<ExecSpace, Scalar>(x, W, 1.0f, 0.0f, y, label);
}

}  // namespace KokkosDNN

#endif
