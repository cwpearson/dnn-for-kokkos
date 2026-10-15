#ifndef KOKKOSDNN_HANDLE_HPP
#define KOKKOSDNN_HANDLE_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>
#include "KokkosDNN_config.h"

// Third-party handle lifecycle lives in src/tpl/... and is included here
// only when the matching TPL is enabled. The core header itself stays
// free of TPL types (cached handles are opaque void*).
#ifdef KOKKOSDNN_ENABLE_CUBLAS
#include "tpl/cublas/KokkosDNN_CuBLAS_Handle.hpp"
#endif
#ifdef KOKKOSDNN_ENABLE_CUDNN
#include "tpl/cudnn/KokkosDNN_CuDNN_Handle.hpp"
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
#if (defined(KOKKOSDNN_ENABLE_CUBLAS) || defined(KOKKOSDNN_ENABLE_CUDNN)) && \
    defined(KOKKOS_ENABLE_CUDA)
    // Naming the backend here is deliberate: the TPL handles below are
    // CUDA-only, so dispatch must key off the CUDA execution space itself.
    constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
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
  // Lifecycle in tpl/cublas/KokkosDNN_CuBLAS_Handle.hpp.
  void* cublas_raw() const {
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
    // See ctor: backend named deliberately, this handle is CUDA-only.
    constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
    if constexpr (is_cuda) {
      if (use_cublas && cublas_ptr_ == nullptr) {
        cublas_ptr_ = Impl::cublas_handle_create();
      }
      return cublas_ptr_;
    }
#endif
    return nullptr;
  }

  // Lazily created cached cuDNN handle (nullptr contract as above).
  // Lifecycle in tpl/cudnn/KokkosDNN_CuDNN_Handle.hpp.
  void* cudnn_raw() const {
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
    // See ctor: backend named deliberately, this handle is CUDA-only.
    constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
    if constexpr (is_cuda) {
      if (use_cudnn && cudnn_ptr_ == nullptr) {
        cudnn_ptr_ = Impl::cudnn_handle_create();
      }
      return cudnn_ptr_;
    }
#endif
    return nullptr;
  }

 private:
  void destroy() {
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
    Impl::cublas_handle_destroy(cublas_ptr_);
#else
    cublas_ptr_ = nullptr;
#endif
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
    Impl::cudnn_handle_destroy(cudnn_ptr_);
#else
    cudnn_ptr_ = nullptr;
#endif
  }
};

}  // namespace KokkosDNN

#endif
