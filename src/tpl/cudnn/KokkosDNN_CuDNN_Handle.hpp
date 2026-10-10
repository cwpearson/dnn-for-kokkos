#ifndef KOKKOSDNN_CUDNN_HANDLE_HPP
#define KOKKOSDNN_CUDNN_HANDLE_HPP

// Cached cuDNN handle lifecycle for DNNHandle (see KokkosDNN_Handle.hpp).
// Included into KokkosDNN_Handle.hpp when KOKKOSDNN_ENABLE_CUDNN is set.
// Keeps <cudnn.h> out of the core header on builds without cuDNN.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
#include <cudnn.h>

namespace KokkosDNN {
namespace Impl {

// Create a fresh cuDNN handle, or nullptr on failure. The caller caches
// the result (one handle per DNNHandle); the stream is re-selected per
// use in KokkosDNN_CuDNN_SDPA.cpp.
inline void* cudnn_handle_create() {
  cudnnHandle_t h = nullptr;
  if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) return nullptr;
  return static_cast<void*>(h);
}

// Destroy a cached handle created above; resets the slot to nullptr.
inline void cudnn_handle_destroy(void*& slot) {
  if (slot != nullptr) {
    cudnnDestroy(static_cast<cudnnHandle_t>(slot));
    slot = nullptr;
  }
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
