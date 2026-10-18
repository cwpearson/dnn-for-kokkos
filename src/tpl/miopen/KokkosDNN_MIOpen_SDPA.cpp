// MIOpen SDPA stub implementation for KokkosDNN attention.
//
// Single translation unit mirroring tpl/cudnn/KokkosDNN_CuDNN_SDPA.cpp.
// Unlike the cuDNN path there is no graph to build here: MIOpen
// (checked 3.5.2, miopen/miopen.h) exposes convolution, softmax,
// OpTensor, and Fusion but NO fused scaled-dot-product / FlashAttention
// API equivalent to cudnn-frontend's SDPA node — the header contains zero
// matches for "attention", "flash", "SDPA", or "scaled_dot". Every entry
// point below therefore returns false by construction, so callers fall
// back to the native Kokkos kernel (exact on every backend).
//
// Future real dispatch is a composition, not a fused kernel: hipBLAS
// batched GEMM for QK^T, MIOpen softmax forward for the row normalization,
// hipBLAS batched GEMM for AV. When that lands, only this TU (plus the
// TPL-off stubs at the bottom) changes; the raw-pointer/stride contract
// in KokkosDNN_MIOpen_Forward.hpp stays valid.
#include "KokkosDNN_config.h"

#include <Kokkos_Core.hpp>

#include "tpl/miopen/KokkosDNN_MIOpen_Forward.hpp"

#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)

#include <miopen/miopen.h>

namespace KokkosDNN {
namespace Impl {

bool miopen_sdpa_forward(void*, MiopenSdpaScalar, const void*, const void*,
                         const void*, void*, int64_t, int64_t, int64_t,
                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                         int64_t, int64_t, float) {
  // No fused SDPA in MIOpen (see file note); decline to native.
  return false;
}

bool miopen_sdpa_forward_noncausal(void*, MiopenSdpaScalar, const void*,
                                   const void*, const void*, void*, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, float) {
  // No fused SDPA in MIOpen (see file note); decline to native.
  return false;
}

bool miopen_sdpa_forward_sliding(void*, MiopenSdpaScalar, const void*,
                                 const void*, const void*, void*, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, float, int64_t) {
  // No fused SDPA in MIOpen (see file note); decline to native.
  return false;
}

}  // namespace Impl
}  // namespace KokkosDNN

#else

namespace KokkosDNN {
namespace Impl {

bool miopen_sdpa_forward(void*, MiopenSdpaScalar, const void*, const void*,
                         const void*, void*, int64_t, int64_t, int64_t,
                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                         int64_t, int64_t, float) {
  return false;
}

bool miopen_sdpa_forward_noncausal(void*, MiopenSdpaScalar, const void*,
                                   const void*, const void*, void*, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, float) {
  return false;
}

bool miopen_sdpa_forward_sliding(void*, MiopenSdpaScalar, const void*,
                                 const void*, const void*, void*, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, float, int64_t) {
  return false;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
