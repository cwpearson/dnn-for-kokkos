// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CUBLAS_STREAM_HPP
#define KOKKOSDNN_CUBLAS_STREAM_HPP

// Selects the Kokkos CUDA stream on a cached cuBLAS handle.
// Must run before every cuBLAS call: the handle is shared across calls
// and the stream is part of its mutable state.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Kokkos_Core.hpp>

namespace KokkosDNN {
namespace Impl {

inline bool cublas_select_kokkos_stream(cublasHandle_t h) {
  if (h == nullptr) return false;
  const cudaStream_t s = Kokkos::Cuda().cuda_stream();
  return cublasSetStream(h, s) == CUBLAS_STATUS_SUCCESS;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
