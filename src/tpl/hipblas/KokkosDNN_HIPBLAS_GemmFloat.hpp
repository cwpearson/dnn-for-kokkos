#ifndef KOKKOSDNN_HIPBLAS_GEMMFLOAT_HPP
#define KOKKOSDNN_HIPBLAS_GEMMFLOAT_HPP

// FP32 row-major NT GEMM via hipBLAS: Y[M,N] = alpha * X[M,K] * W[N,K]^T +
// beta * Y[M,N]. Uses the col-major identity Yc = Wc^T * Xc where Yc/Wc/Xc
// share the same buffers interpreted column-major.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
#include <hipblas/hipblas.h>

#include "tpl/hipblas/KokkosDNN_HIPBLAS_Stream.hpp"

namespace KokkosDNN {
namespace Impl {

inline bool hipblas_gemm_nt_with_handle(hipblasHandle_t h, int M, int N, int K,
                                        float alpha, const float* X,
                                        const float* W, float beta, float* Y) {
  if (!hipblas_select_kokkos_stream(h)) return false;
  // A=W (KxN col) transposed, B=X (KxM col) plain, C=Y (NxM col).
  hipblasStatus_t s =
      hipblasSgemm(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, &alpha, W, K, X, K,
                   &beta, Y, N);
  return s == HIPBLAS_STATUS_SUCCESS;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
