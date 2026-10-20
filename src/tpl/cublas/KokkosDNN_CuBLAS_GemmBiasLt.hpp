#ifndef KOKKOSDNN_CUBLAS_GEMMBIASLT_HPP
#define KOKKOSDNN_CUBLAS_GEMMBIASLT_HPP

// Fused row-major NT GEMM+bias via cublasLt (CUBLASLT_EPILOGUE_BIAS):
//   Y[M,N] = X[M,K] * W[N,K]^T + b[N]
// in a single kernel launch. FP32 accumulation (CUBLAS_COMPUTE_32F, float
// alpha/beta) matches the native kernel and the cublasGemmEx path, so bf16
// I/O keeps FP32 math end to end; the bias vector has the output (D) type
// and is added in FP32 before the single output cast (same order as torch
// addmm; the old split gemm+bias path rounded to Scalar between the two
// ops, so bf16 results may differ from it by <= 1 ulp).
//
// Layout: the same col-major identity as the sgemm/GemmEx paths --
// A=W (KxN col) transposed, B=X (KxM col) plain, D=Y (NxM col). The BIAS
// epilogue broadcasts a vector of length (# D rows) over all D columns,
// and # D rows is exactly N here, i.e. D_col[i,j] += b[i] <-> Y[j,i] +=
// b[i]: the row-major row-broadcast we need. COL is the default Lt order,
// so no order attribute is set (this also sidesteps any ROW-order bias
// orientation questions).
//
// Cost model: plan construction (descriptors + heuristic + one-time
// autotune of the top candidates) costs milliseconds once per (M, N, K,
// io-type), which would dominate the tiny M=1 time-path GEMMs that repeat
// identical (N, K) every layer -- hence plans are cached in a small state
// object owned by DNNHandle (created lazily, destroyed with the handle;
// single-threaded caller affinity, like the rest of the library). The cublasLt handle itself is cached there too; the Kokkos
// CUDA stream is passed per call, so no stream state goes stale.
//
// Failure contract: every entry returns false (caller falls back to the
// split cuBLAS GEMM + bias kernel, i.e. today's behavior) when Lt is
// unusable -- odd shapes the heuristic rejects, missing BF16 support,
// any API error. A plan whose matmul ever fails is evicted. Correctness
// never depends on this path succeeding.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Impl {

// Cached cublasLt state for the fused bias GEMM. Owned by DNNHandle as an
// opaque void* (see KokkosDNN_Handle.hpp); defined here so the create /
// destroy helpers see the complete type.
struct CublasLtBiasState {
  cublasLtHandle_t lt = nullptr;
  struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t A = nullptr, B = nullptr, D = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    bool has_algo = false;
    int M = 0, N = 0, K = 0;
    cudaDataType_t io = CUDA_R_32F;
  };
  std::vector<Plan> plans;

  CublasLtBiasState() = default;
  CublasLtBiasState(const CublasLtBiasState&) = delete;
  CublasLtBiasState& operator=(const CublasLtBiasState&) = delete;

  ~CublasLtBiasState() {
    for (auto& p : plans) {
      if (p.D != nullptr) cublasLtMatrixLayoutDestroy(p.D);
      if (p.B != nullptr) cublasLtMatrixLayoutDestroy(p.B);
      if (p.A != nullptr) cublasLtMatrixLayoutDestroy(p.A);
      if (p.op != nullptr) cublasLtMatmulDescDestroy(p.op);
    }
    plans.clear();
    if (lt != nullptr) {
      cublasLtDestroy(lt);
      lt = nullptr;
    }
  }
};

// Create fresh state (cublasLt handle + empty plan cache), or nullptr on
// failure. The caller caches the result (one per DNNHandle).
inline void* cublaslt_bias_state_create() {
  auto* st = new (std::nothrow) CublasLtBiasState();
  if (st == nullptr) return nullptr;
  if (cublasLtCreate(&st->lt) != CUBLAS_STATUS_SUCCESS) {
    delete st;
    return nullptr;
  }
  return static_cast<void*>(st);
}

// Destroy state created above; resets the slot to nullptr.
inline void cublaslt_bias_state_destroy(void*& slot) {
  if (slot != nullptr) {
    delete static_cast<CublasLtBiasState*>(slot);
    slot = nullptr;
  }
}

// Look up (or build) the plan for (M, N, K, io). Returns nullptr when Lt
// cannot serve this shape -- caller falls back, nothing is cached.
//
// Plan builds autotune once: the heuristic ranking is NOT performance
// order (measured: first pick 480 us vs third pick 359 us on a DiT bf16
// shape), so the top candidates are each timed on the caller's stream and
// the fastest is cached. The tuning writes Y, which the real call fully
// recomputes right afterwards (beta=0), so no caller state is clobbered.
// Cost is a few ms once per (M, N, K, io); every later call is a cache
// hit plus one matmul.
inline CublasLtBiasState::Plan* cublaslt_bias_plan_get(CublasLtBiasState* st,
                                                       cudaStream_t stream,
                                                       int M, int N, int K,
                                                       const void* X,
                                                       const void* W,
                                                       const void* bias,
                                                       void* Y,
                                                       cudaDataType_t io) {
  for (auto& p : st->plans) {
    if (p.has_algo && p.M == M && p.N == N && p.K == K && p.io == io) return &p;
  }
  // Miss: build descriptors. Col-major identity (see header comment):
  // A=W[K,N] transposed, B=X[K,M] plain, D=Y[N,M].
  CublasLtBiasState::Plan p;
  p.M = M;
  p.N = N;
  p.K = K;
  p.io = io;
  // Transposes mirror cublasSgemm(h, T, N, ...): the first operand (W,
  // layout A) is transposed, the second (X, layout B) is not.
  const cublasOperation_t transT = CUBLAS_OP_T, transN = CUBLAS_OP_N;
  const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
  const int32_t bias_type      = static_cast<int32_t>(io);
  bool ok =
      cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &transT, sizeof(transT)) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &transN, sizeof(transN)) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     &epi, sizeof(epi)) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulDescSetAttribute(p.op,
                                     CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                     &bias_type, sizeof(bias_type)) ==
          CUBLAS_STATUS_SUCCESS &&
      // A=W: rows=K cols=N ld=K; B=X: rows=K cols=M ld=K;
      // D=Y: rows=N cols=M ld=N.
      cublasLtMatrixLayoutCreate(&p.A, io, static_cast<uint64_t>(K),
                                 static_cast<uint64_t>(N),
                                 static_cast<int64_t>(K)) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&p.B, io, static_cast<uint64_t>(K),
                                 static_cast<uint64_t>(M),
                                 static_cast<int64_t>(K)) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&p.D, io, static_cast<uint64_t>(N),
                                 static_cast<uint64_t>(M),
                                 static_cast<int64_t>(N)) ==
          CUBLAS_STATUS_SUCCESS;
  if (!ok) {
    if (p.D != nullptr) cublasLtMatrixLayoutDestroy(p.D);
    if (p.B != nullptr) cublasLtMatrixLayoutDestroy(p.B);
    if (p.A != nullptr) cublasLtMatrixLayoutDestroy(p.A);
    if (p.op != nullptr) cublasLtMatmulDescDestroy(p.op);
    return nullptr;
  }
  // One zero-workspace heuristic query; DiT-size GEMMs all have kernels
  // that need no workspace, which keeps every call workspace-free. Ask
  // for a few candidates and time them (heuristic order is not
  // performance order); keep the fastest.
  constexpr int kNumCandidates = 4;
  cublasLtMatmulPreference_t pref = nullptr;
  size_t ws_bytes                  = 0;
  cublasLtMatmulHeuristicResult_t cand[kNumCandidates]{};
  int found = 0;
  bool tune_ok =
      cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                sizeof(ws_bytes)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulAlgoGetHeuristic(st->lt, p.op, p.A, p.B, p.D, p.D,
                                           pref, kNumCandidates, cand,
                                           &found) == CUBLAS_STATUS_SUCCESS &&
            found > 0;
  if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
  if (!tune_ok) {
    cublasLtMatrixLayoutDestroy(p.D);
    cublasLtMatrixLayoutDestroy(p.B);
    cublasLtMatrixLayoutDestroy(p.A);
    cublasLtMatmulDescDestroy(p.op);
    return nullptr;
  }
  // Autotune the candidates on the caller's stream. Y is fully
  // overwritten (beta=0 below), so tuning on the live buffers is safe;
  // the timed region covers exactly the candidate iters (single-threaded
  // caller, no other work can slip between the event markers).
  const float alpha_t = 1.0f, beta_t = 0.0f;
  int best = 0;
  {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    bool ev_ok = cudaEventCreate(&e0) == cudaSuccess &&
                 cudaEventCreate(&e1) == cudaSuccess;
    float best_ms = -1.0f;
    // Bias pointer must be set before any timed matmul (it is per-call
    // state on the op descriptor).
    if (ev_ok &&
        cublasLtMatmulDescSetAttribute(p.op,
                                       CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias, sizeof(bias)) ==
            CUBLAS_STATUS_SUCCESS) {
      for (int c = 0; c < found; ++c) {
        // One warmup iter, then the fastest of the timed iters wins.
        cublasLtMatmul(st->lt, p.op, &alpha_t, W, p.A, X, p.B, &beta_t, Y,
                       p.D, Y, p.D, &cand[c].algo, nullptr, 0, stream);
        cudaEventRecord(e0, stream);
        for (int r = 0; r < 3; ++r)
          cublasLtMatmul(st->lt, p.op, &alpha_t, W, p.A, X, p.B, &beta_t, Y,
                         p.D, Y, p.D, &cand[c].algo, nullptr, 0, stream);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= 3.0f;
        if (best_ms < 0.0f || ms < best_ms) {
          best_ms = ms;
          best    = c;
        }
      }
    }
    if (e0 != nullptr) cudaEventDestroy(e0);
    if (e1 != nullptr) cudaEventDestroy(e1);
    if (best_ms < 0.0f) {
      cublasLtMatrixLayoutDestroy(p.D);
      cublasLtMatrixLayoutDestroy(p.B);
      cublasLtMatrixLayoutDestroy(p.A);
      cublasLtMatmulDescDestroy(p.op);
      return nullptr;
    }
  }
  p.heur     = cand[best];
  p.has_algo = true;
  st->plans.push_back(std::move(p));
  return &st->plans.back();
}

// Untyped worker: Y[M,N] = alpha * X[M,K] * W[N,K]^T + b[N] via the cached
// plan. io selects CUDA_R_32F (float buffers) or CUDA_R_16BF (bhalf_t
// buffers, bit-identical to __nv_bfloat16; void* so this header does not
// depend on <cuda_bf16.h>).
inline bool cublaslt_gemm_bias_nt_impl(void* state, int M, int N, int K,
                                       float alpha, const void* X,
                                       const void* W, const void* bias,
                                       void* Y, cudaDataType_t io) {
  if (state == nullptr || M <= 0 || N <= 0 || K <= 0) return false;
  auto* st = static_cast<CublasLtBiasState*>(state);
  if (st->lt == nullptr) return false;
  // Stream comes from Kokkos every call: Lt work stays ordered with the
  // surrounding Kokkos kernels, no fence needed.
  const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
  CublasLtBiasState::Plan* plan =
      cublaslt_bias_plan_get(st, stream, M, N, K, X, W, bias, Y, io);
  if (plan == nullptr) return false;
  // Bias pointer is per-call state on the cached op descriptor.
  if (cublasLtMatmulDescSetAttribute(plan->op,
                                     CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                     &bias, sizeof(bias)) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  const float beta = 0.0f;
  // A=W (transposed), B=X, C=D=Y with beta=0: the old split path's
  // beta*Y term is absent by construction (gemm_bias always passes
  // beta=0), so C aliasing D is exact.
  cublasStatus_t s = cublasLtMatmul(st->lt, plan->op, &alpha, W, plan->A, X,
                                    plan->B, &beta, Y, plan->D, Y, plan->D,
                                    &plan->heur.algo, nullptr, 0, stream);
  if (s != CUBLAS_STATUS_SUCCESS) {
    // Evict the poisoned plan (bad algo for this shape/GPU); the caller
    // falls back to split cuBLAS + bias kernel and a later call rebuilds.
    cublasLtMatrixLayoutDestroy(plan->D);
    cublasLtMatrixLayoutDestroy(plan->B);
    cublasLtMatrixLayoutDestroy(plan->A);
    cublasLtMatmulDescDestroy(plan->op);
    for (auto it = st->plans.begin(); it != st->plans.end(); ++it) {
      if (&(*it) == plan) {
        st->plans.erase(it);
        break;
      }
    }
    return false;
  }
  return true;
}

// FP32 fused bias GEMM (mirrors cublas_gemm_nt_with_handle).
inline bool cublaslt_gemm_bias_nt_f32_with_state(void* state, int M, int N,
                                                 int K, float alpha,
                                                 const float* X,
                                                 const float* W,
                                                 const float* bias, float* Y) {
  return cublaslt_gemm_bias_nt_impl(state, M, N, K, alpha, X, W, bias, Y,
                                    CUDA_R_32F);
}

// BF16 fused bias GEMM (mirrors cublas_gemm_nt_bf16_with_handle):
// CUDA_R_16BF I/O with CUBLAS_COMPUTE_32F accumulation.
inline bool cublaslt_gemm_bias_nt_bf16_with_state(void* state, int M, int N,
                                                  int K, float alpha,
                                                  const void* X, const void* W,
                                                  const void* bias, void* Y) {
  using Kokkos::Experimental::bhalf_t;
  static_assert(sizeof(bhalf_t) == 2, "bhalf_t must be 2 bytes for CUDA_R_16BF");
  static_assert(alignof(bhalf_t) == 2, "bhalf_t must be 2-byte aligned");
  static_assert(std::is_standard_layout_v<bhalf_t>,
                "bhalf_t must wrap a single __nv_bfloat16 member");
  return cublaslt_gemm_bias_nt_impl(state, M, N, K, alpha, X, W, bias, Y,
                                    CUDA_R_16BF);
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
