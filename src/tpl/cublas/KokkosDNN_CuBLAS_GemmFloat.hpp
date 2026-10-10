#ifndef KOKKOSDNN_CUBLAS_GEMMFLOAT_HPP
#define KOKKOSDNN_CUBLAS_GEMMFLOAT_HPP

// FP32 row-major NT GEMM via cuBLAS: Y[M,N] = alpha * X[M,K] * W[N,K]^T +
// beta * Y[M,N]. Uses the col-major identity Yc = Wc^T * Xc where Yc/Wc/Xc
// share the same buffers interpreted column-major.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>

#include "tpl/cublas/KokkosDNN_CuBLAS_Stream.hpp"

namespace KokkosDNN {
namespace Impl {

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

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
