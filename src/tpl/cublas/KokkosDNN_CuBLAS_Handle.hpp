#ifndef KOKKOSDNN_CUBLAS_HANDLE_HPP
#define KOKKOSDNN_CUBLAS_HANDLE_HPP

// Cached cuBLAS handle lifecycle for DNNHandle (see KokkosDNN_Handle.hpp).
// Included into KokkosDNN_Handle.hpp when KOKKOSDNN_ENABLE_CUBLAS is set.
// Keeps <cublas_v2.h> out of the core header on builds without cuBLAS.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>

namespace KokkosDNN {
namespace Impl {

// Create a fresh cuBLAS handle, or nullptr on failure. The caller caches
// the result (one handle per DNNHandle); the stream is re-selected per
// use in KokkosDNN_CuBLAS_Stream.hpp.
inline void* cublas_handle_create() {
  cublasHandle_t h = nullptr;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return nullptr;
  return static_cast<void*>(h);
}

// Destroy a cached handle created above; resets the slot to nullptr.
inline void cublas_handle_destroy(void*& slot) {
  if (slot != nullptr) {
    cublasDestroy(static_cast<cublasHandle_t>(slot));
    slot = nullptr;
  }
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
