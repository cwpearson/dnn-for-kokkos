#ifndef KOKKOSDNN_HANDLE_HPP
#define KOKKOSDNN_HANDLE_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#endif
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
#include <cudnn.h>
#endif

namespace KokkosDNN {

// Lightweight handle that records which third-party dispatches are active.
// Native Kokkos paths ignore it; call sites stay free of #ifdefs.
//
// TPL handles are cached, not created per call: the cuBLAS/cuDNN handles
// are created lazily once per DNNHandle (first cublas_raw()/cudnn_raw()
// call that needs them) and destroyed with the handle. Previously every
// gemm/attention call built and tore down its own handle, i.e. ~7 GEMMs +
// 1 SDPA per layer x 32 layers ~= ~224 TPL setups per generated token.
// The per-call stream is (re-)selected from the Kokkos CUDA stream at each
// use, so one cached handle stays correctly ordered with surrounding
// Kokkos kernels. On non-CUDA builds (or with the TPL disabled) the
// accessors return nullptr and every op runs the native kernel
// (Native == Dispatched by construction).
//
// Ownership: non-copyable, movable (transfers the cached handles).
// Destroying the handle after Kokkos::finalize() is a programming error;
// TPL teardown may fail silently there, matching plain cuBLAS/cuDNN usage.
template <class ExecSpace>
struct DNNHandle {
  bool use_cublas = false;
  bool use_cudnn  = false;

  // Opaque cached TPL handles (cublasHandle_t / cudnnHandle_t when the
  // matching TPL is enabled on a CUDA build, nullptr otherwise). void*
  // keeps this header free of TPL types on builds that lack them.
  mutable void* cublas_ptr_ = nullptr;
  mutable void* cudnn_ptr_  = nullptr;

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

  ~DNNHandle() { destroy(); }

  DNNHandle(const DNNHandle&)            = delete;
  DNNHandle& operator=(const DNNHandle&) = delete;

  DNNHandle(DNNHandle&& other) noexcept
      : use_cublas(other.use_cublas),
        use_cudnn(other.use_cudnn),
        cublas_ptr_(other.cublas_ptr_),
        cudnn_ptr_(other.cudnn_ptr_) {
    other.use_cublas  = false;
    other.use_cudnn   = false;
    other.cublas_ptr_ = nullptr;
    other.cudnn_ptr_  = nullptr;
  }

  DNNHandle& operator=(DNNHandle&& other) noexcept {
    if (this != &other) {
      destroy();
      use_cublas  = other.use_cublas;
      use_cudnn   = other.use_cudnn;
      cublas_ptr_ = other.cublas_ptr_;
      cudnn_ptr_  = other.cudnn_ptr_;
      other.use_cublas  = false;
      other.use_cudnn   = false;
      other.cublas_ptr_ = nullptr;
      other.cudnn_ptr_  = nullptr;
    }
    return *this;
  }

  // Lazily created cached cuBLAS handle (nullptr when unusable: non-CUDA
  // space, TPL disabled, or creation failure -- callers fall back then).
  // Const so gemm/gemv/attention can take `const DNNHandle&`.
  void* cublas_raw() const {
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
    constexpr bool is_cuda =
        std::is_same_v<typename ExecSpace::memory_space,
                       Kokkos::CudaSpace> ||
        std::is_same_v<typename ExecSpace::memory_space,
                       Kokkos::CudaUVMSpace>;
    if constexpr (is_cuda) {
      if (use_cublas && cublas_ptr_ == nullptr) {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) cublas_ptr_ = h;
      }
      return cublas_ptr_;
    }
#endif
    return nullptr;
  }

  // Lazily created cached cuDNN handle (nullptr contract as above).
  void* cudnn_raw() const {
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
    constexpr bool is_cuda =
        std::is_same_v<typename ExecSpace::memory_space,
                       Kokkos::CudaSpace> ||
        std::is_same_v<typename ExecSpace::memory_space,
                       Kokkos::CudaUVMSpace>;
    if constexpr (is_cuda) {
      if (use_cudnn && cudnn_ptr_ == nullptr) {
        cudnnHandle_t h = nullptr;
        if (cudnnCreate(&h) == CUDNN_STATUS_SUCCESS) cudnn_ptr_ = h;
      }
      return cudnn_ptr_;
    }
#endif
    return nullptr;
  }

 private:
  void destroy() {
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
    if (cublas_ptr_ != nullptr) {
      cublasDestroy(static_cast<cublasHandle_t>(cublas_ptr_));
      cublas_ptr_ = nullptr;
    }
#else
    cublas_ptr_ = nullptr;
#endif
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
    if (cudnn_ptr_ != nullptr) {
      cudnnDestroy(static_cast<cudnnHandle_t>(cudnn_ptr_));
      cudnn_ptr_ = nullptr;
    }
#else
    cudnn_ptr_ = nullptr;
#endif
  }
};

}  // namespace KokkosDNN

#endif
