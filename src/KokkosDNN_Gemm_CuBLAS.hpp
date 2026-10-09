#ifndef KOKKOSDNN_GEMM_CUBLAS_HPP
#define KOKKOSDNN_GEMM_CUBLAS_HPP

// cuBLAS dispatch for KokkosDNN::gemm/gemv. Included only when
// KOKKOSDNN_ENABLE_CUBLAS is defined AND the translation unit is
// compiled for CUDA. All other builds fall back to native Kokkos.
//
// All entry points take the caller's cached cublasHandle_t (owned by
// DNNHandle, created once per handle -- see KokkosDNN_Handle.hpp), never
// creating a handle per call. The stream is (re-)selected to the Kokkos
// CUDA stream on every call so the TPL work stays ordered with the
// surrounding Kokkos kernels.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <stdexcept>
#include <type_traits>

namespace KokkosDNN {
namespace Impl {

// Select the Kokkos CUDA stream on a cached handle. Must run before every
// cuBLAS call: the handle is shared across calls and the stream is part of
// its mutable state.
inline bool cublas_select_kokkos_stream(cublasHandle_t h) {
  if (h == nullptr) return false;
  const cudaStream_t s = Kokkos::Cuda().cuda_stream();
  return cublasSetStream(h, s) == CUBLAS_STATUS_SUCCESS;
}

// Y[M,N] = alpha * X[M,K] * W[N,K]^T + beta * Y[M,N], all row-major.
// Uses the col-major identity Yc = Wc^T * Xc where Yc/Wc/Xc share the
// same buffers interpreted column-major.
//
// FP32 path (cublasSgemm). See cublas_gemm_nt_bf16_with_handle below for
// the BF16 path; other scalars fall back to the native Kokkos kernel in
// KokkosDNN_Gemm.hpp.
inline bool cublas_gemm_nt_with_handle(cublasHandle_t h, int M, int N, int K,
                                       float alpha, const float* X,
                                       const float* W, float beta, float* Y) {
  if (!cublas_select_kokkos_stream(h)) return false;
  // A=W (KxN col) transposed, B=X (KxM col) plain, C=Y (NxM col).
  cublasStatus_t s =
      cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, X, K,
                  &beta, Y, N);
  return s == CUBLAS_STATUS_SUCCESS;
}

// Y[M,N] = alpha * X[M,K] * W[N,K]^T + beta * Y[M,N], all row-major BF16.
// Same col-major identity as above via cublasGemmEx: A/B/C in CUDA_R_16BF,
// accumulation in FP32 (CUBLAS_COMPUTE_32F, so alpha/beta stay float) to
// match the native kernel's FP32 accumulation. CUBLAS_GEMM_DEFAULT lets
// cuBLAS pick the algorithm (tensor cores on Ampere+); on arches without
// BF16 support the call fails and the caller falls through to native.
//
// Pointer contract: on CUDA, Kokkos::Experimental::bhalf_t is a wrapper
// holding exactly one __nv_bfloat16 (alignas, single member), so device
// buffers are bit-identical to CUDA_R_16BF arrays. Pointers cross here as
// void* so this header does not depend on <cuda_bf16.h>.
inline bool cublas_gemm_nt_bf16_with_handle(cublasHandle_t h, int M, int N,
                                            int K, float alpha, const void* X,
                                            const void* W, float beta,
                                            void* Y) {
  using Kokkos::Experimental::bhalf_t;
  static_assert(sizeof(bhalf_t) == 2, "bhalf_t must be 2 bytes for CUDA_R_16BF");
  static_assert(alignof(bhalf_t) == 2, "bhalf_t must be 2-byte aligned");
  static_assert(std::is_standard_layout_v<bhalf_t>,
                "bhalf_t must wrap a single __nv_bfloat16 member");
  if (!cublas_select_kokkos_stream(h)) return false;
  // A=W (KxN col) transposed, B=X (KxM col) plain, C=Y (NxM col).
  cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                                  W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta,
                                  Y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT);
  return s == CUBLAS_STATUS_SUCCESS;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif  // CUBLAS+CUDA
#endif
