// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CUBLAS_GEMMBF16_HPP
#define KOKKOSDNN_CUBLAS_GEMMBF16_HPP

// BF16 row-major NT GEMM via cublasGemmEx: Y[M,N] = alpha * X[M,K] *
// W[N,K]^T + beta * Y[M,N]. Same col-major identity as the FP32 path;
// A/B/C in CUDA_R_16BF, accumulation in FP32 (CUBLAS_COMPUTE_32F, so
// alpha/beta stay float) to match the native kernel's FP32 accumulation.
// CUBLAS_GEMM_DEFAULT lets cuBLAS pick the algorithm (tensor cores on
// Ampere+); on arches without BF16 support the call fails and the caller
// falls through to native.
//
// Pointer contract: on CUDA, Kokkos::Experimental::bhalf_t is a wrapper
// holding exactly one __nv_bfloat16 (alignas, single member), so device
// buffers are bit-identical to CUDA_R_16BF arrays. Pointers cross here as
// void* so this header does not depend on <cuda_bf16.h>.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>

#include <Kokkos_Half.hpp>
#include <type_traits>

#include "tpl/cublas/KokkosDNN_CuBLAS_Stream.hpp"

namespace KokkosDNN {
namespace Impl {

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

#endif
#endif
