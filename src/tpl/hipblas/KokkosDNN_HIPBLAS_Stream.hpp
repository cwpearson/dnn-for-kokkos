// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_HIPBLAS_STREAM_HPP
#define KOKKOSDNN_HIPBLAS_STREAM_HPP

// Selects the Kokkos HIP stream on a cached hipBLAS handle.
// Must run before every hipBLAS call: the handle is shared across calls
// and the stream is part of its mutable state.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
#include <hipblas/hipblas.h>

#include <Kokkos_Core.hpp>

namespace KokkosDNN {
namespace Impl {

inline bool hipblas_select_kokkos_stream(hipblasHandle_t h) {
  if (h == nullptr) return false;
  const hipStream_t s = Kokkos::HIP().hip_stream();
  return hipblasSetStream(h, s) == HIPBLAS_STATUS_SUCCESS;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
