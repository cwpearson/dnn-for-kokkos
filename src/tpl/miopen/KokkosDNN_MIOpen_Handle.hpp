// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MIOPEN_HANDLE_HPP
#define KOKKOSDNN_MIOPEN_HANDLE_HPP

// Cached MIOpen handle lifecycle for DNNHandle (see KokkosDNN_Handle.hpp).
// Included into KokkosDNN_Handle.hpp when KOKKOSDNN_ENABLE_MIOPEN is set.
// Keeps <miopen/miopen.h> out of the core header on builds without MIOpen.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
#include <miopen/miopen.h>

namespace KokkosDNN {
namespace Impl {

// Create a fresh MIOpen handle, or nullptr on failure. The caller caches
// the result (one handle per DNNHandle); the stream is re-selected per
// use at the dispatch site.
inline void* miopen_handle_create() {
  miopenHandle_t h = nullptr;
  if (miopenCreate(&h) != miopenStatusSuccess) return nullptr;
  return static_cast<void*>(h);
}

// Destroy a cached handle created above; resets the slot to nullptr.
inline void miopen_handle_destroy(void*& slot) {
  if (slot != nullptr) {
    miopenDestroy(static_cast<miopenHandle_t>(slot));
    slot = nullptr;
  }
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
