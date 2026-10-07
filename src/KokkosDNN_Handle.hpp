#ifndef KOKKOSDNN_HANDLE_HPP
#define KOKKOSDNN_HANDLE_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>
#include "KokkosDNN_config.h"

namespace KokkosDNN {

// Lightweight handle that records which third-party dispatches are active.
// Native Kokkos paths ignore it; call sites stay free of #ifdefs and CUDA
// handles (when used) are created per-call inside the TPL wrappers for
// PoC simplicity — a production version would cache them here.
template <class ExecSpace>
struct DNNHandle {
  bool use_cublas = false;
  bool use_cudnn  = false;

  DNNHandle() {
#if defined(KOKKOSDNN_ENABLE_CUBLAS) || defined(KOKKOSDNN_ENABLE_CUDNN)
    constexpr bool is_cuda =
        std::is_same_v<typename ExecSpace::memory_space,
                       Kokkos::CudaSpace> ||
        std::is_same_v<typename ExecSpace::memory_space,
                       Kokkos::CudaUVMSpace> ||
        std::is_same_v<typename ExecSpace::memory_space,
                       Kokkos::CudaHostPinnedSpace>;
    if constexpr (is_cuda) {
#ifdef KOKKOSDNN_ENABLE_CUBLAS
      use_cublas = true;
#endif
#ifdef KOKKOSDNN_ENABLE_CUDNN
      use_cudnn = true;
#endif
    }
#endif
  }
  DNNHandle(const DNNHandle&)            = delete;
  DNNHandle& operator=(const DNNHandle&) = delete;
};

}  // namespace KokkosDNN

#endif
