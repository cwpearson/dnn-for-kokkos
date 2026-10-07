#ifndef KOKKOSDNN_GEMM_CUBLAS_HPP
#define KOKKOSDNN_GEMM_CUBLAS_HPP

// cuBLAS dispatch for KokkosDNN::gemm. Included only when
// KOKKOSDNN_ENABLE_CUBLAS is defined AND the translation unit is
// compiled for CUDA. All other builds fall back to native Kokkos.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Kokkos_Core.hpp>
#include <stdexcept>

namespace KokkosDNN {
namespace Impl {

// Y[M,N] = alpha * X[M,K] * W[N,K]^T + beta * Y[M,N], all row-major.
// Uses the col-major identity Yc = Wc^T * Xc where Yc/Wc/Xc share the
// same buffers interpreted column-major.
inline bool cublas_gemm_nt(int M, int N, int K, float alpha,
                            const float* X, const float* W, float beta,
                            float* Y) {
  cublasHandle_t h{};
  cublasCreate(&h);
  // A=W (KxN col) transposed, B=X (KxM col) plain, C=Y (NxM col).
  cublasStatus_t s =
      cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, X, K,
                  &beta, Y, N);
  cublasDestroy(h);
  return s == CUBLAS_STATUS_SUCCESS;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif  // CUBLAS+CUDA
#endif
