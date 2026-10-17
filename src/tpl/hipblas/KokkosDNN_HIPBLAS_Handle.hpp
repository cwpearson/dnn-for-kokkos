#ifndef KOKKOSDNN_HIPBLAS_HANDLE_HPP
#define KOKKOSDNN_HIPBLAS_HANDLE_HPP

// Cached hipBLAS handle lifecycle for DNNHandle (see KokkosDNN_Handle.hpp).
// Included into KokkosDNN_Handle.hpp when KOKKOSDNN_ENABLE_HIPBLAS is set.
// Keeps <hipblas/hipblas.h> out of the core header on builds without hipBLAS.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
#include <hipblas/hipblas.h>

namespace KokkosDNN {
namespace Impl {

// Create a fresh hipBLAS handle, or nullptr on failure. The caller caches
// the result (one handle per DNNHandle); the stream is re-selected per
// use in KokkosDNN_HIPBLAS_Stream.hpp.
inline void* hipblas_handle_create() {
  hipblasHandle_t h = nullptr;
  if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS) return nullptr;
  return static_cast<void*>(h);
}

// Destroy a cached handle created above; resets the slot to nullptr.
inline void hipblas_handle_destroy(void*& slot) {
  if (slot != nullptr) {
    hipblasDestroy(static_cast<hipblasHandle_t>(slot));
    slot = nullptr;
  }
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
