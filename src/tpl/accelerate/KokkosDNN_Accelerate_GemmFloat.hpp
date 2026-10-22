#ifndef KOKKOSDNN_ACCELERATE_GEMMFLOAT_HPP
#define KOKKOSDNN_ACCELERATE_GEMMFLOAT_HPP

// FP32 row-major NT GEMM via Apple Accelerate (vecLib cblas_sgemm):
//   Y[M,N] = alpha * X[M,K] * W[N,K]^T + beta * Y[M,N]
// X/W/Y are LayoutRight device views on a host-memory execution space
// (Serial/OpenMP/Threads: memory_space is HostSpace), so the raw pointers
// are directly callable. Stateless: no handle, no stream selection -- the
// caller fences Kokkos work before invoking (synchronous CPU call).
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_ACCELERATE)
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#error "KokkosDNN_ENABLE_ACCELERATE requires Apple Accelerate (macOS/iOS)"
#endif

#include <cstddef>
#include <cstdint>
#include <limits>

namespace KokkosDNN {
namespace Impl {

// Contiguous-LayoutRight NT sgemm. Strides are row lengths; any strided
// subview returns false so the caller falls back to the native kernel.
// int-range is checked (cblas takes int M/N/K).
inline bool accelerate_gemm_nt_float(int M, int N, int K, float alpha,
                                     const float* X, size_t sx0,
                                     const float* W, size_t sw0, float beta,
                                     float* Y, size_t sy0) {
  if (M <= 0 || N <= 0 || K <= 0) return false;
  if (X == nullptr || W == nullptr || Y == nullptr) return false;
  if (sx0 != static_cast<size_t>(K) || sw0 != static_cast<size_t>(K) ||
      sy0 != static_cast<size_t>(N))
    return false;
  if (static_cast<int64_t>(M) * N > std::numeric_limits<int>::max())
    return false;
  // RowMajor: A=X MxK NoTrans (lda=K), B=W NxK Trans -> op KxN (ldb=K),
  // C=Y MxN (ldc=N).
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, X, K,
              W, K, beta, Y, N);
  return true;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
