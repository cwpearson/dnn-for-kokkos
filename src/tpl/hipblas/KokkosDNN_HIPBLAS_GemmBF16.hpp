#ifndef KOKKOSDNN_HIPBLAS_GEMMBF16_HPP
#define KOKKOSDNN_HIPBLAS_GEMMBF16_HPP

// BF16 row-major NT GEMM via hipblasGemmEx: Y[M,N] = alpha * X[M,K] *
// W[N,K]^T + beta * Y[M,N]. Same col-major identity as the FP32 path;
// A/B/C are BF16, accumulation is FP32 (so alpha/beta stay float) to match
// the native kernel's FP32 accumulation. HIPBLAS_GEMM_DEFAULT lets hipBLAS
// pick the algorithm; on arches without BF16 support the call fails and the
// caller falls through to native.
//
// hipBLAS changed the hipblasGemmEx signature: pre-V2 (default in e.g.
// ROCm 6.4 without -DHIPBLAS_V2) takes hipblasDatatype_t (HIPBLAS_R_*,
// compute HIPBLAS_R_32F), while the V2 API (opt-in since hipBLAS 2.0, and
// the only API in hipBLAS 3.x, which force-defines HIPBLAS_V2) takes
// hipDataType (HIP_R_*, compute hipblasComputeType_t HIPBLAS_COMPUTE_32F).
// Branch on HIPBLAS_V2 so both toolkits compile.
//
// Pointer contract: Kokkos::Experimental::bhalf_t is a 2-byte
// standard-layout wrapper around the backend bf16 type, so device buffers
// are bit-identical to BF16 arrays. Pointers cross here as void*
// so this header does not depend on the backend bf16 header.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
#include <hipblas/hipblas.h>

#include <Kokkos_Half.hpp>
#include <type_traits>

#include "tpl/hipblas/KokkosDNN_HIPBLAS_Stream.hpp"

namespace KokkosDNN {
namespace Impl {

inline bool hipblas_gemm_nt_bf16_with_handle(hipblasHandle_t h, int M, int N,
                                             int K, float alpha, const void* X,
                                             const void* W, float beta,
                                             void* Y) {
  using Kokkos::Experimental::bhalf_t;
  static_assert(sizeof(bhalf_t) == 2, "bhalf_t must be 2 bytes for HIP_R_16BF");
  static_assert(alignof(bhalf_t) == 2, "bhalf_t must be 2-byte aligned");
  static_assert(std::is_standard_layout_v<bhalf_t>,
                "bhalf_t must be a standard-layout bf16 wrapper");
  if (!hipblas_select_kokkos_stream(h)) return false;
  // A=W (KxN col) transposed, B=X (KxM col) plain, C=Y (NxM col).
#ifdef HIPBLAS_V2
  hipblasStatus_t s = hipblasGemmEx(
      h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, &alpha, W, HIP_R_16BF, K, X,
      HIP_R_16BF, K, &beta, Y, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT);
#else
  hipblasStatus_t s = hipblasGemmEx(
      h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, &alpha, W, HIPBLAS_R_16B, K, X,
      HIPBLAS_R_16B, K, &beta, Y, HIPBLAS_R_16B, N, HIPBLAS_R_32F,
      HIPBLAS_GEMM_DEFAULT);
#endif
  return s == HIPBLAS_STATUS_SUCCESS;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
