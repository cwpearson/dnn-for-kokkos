// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_HANDLE_HPP
#define KOKKOSDNN_HANDLE_HPP

#include <Kokkos_Core.hpp>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <utility>
#include "KokkosDNN_config.h"

// Third-party handle lifecycle lives in src/tpl/... and is included here
// only when the matching TPL is enabled. The core header itself stays
// free of TPL types (cached handles are opaque void*).
#ifdef KOKKOSDNN_ENABLE_CUBLAS
#include "tpl/cublas/KokkosDNN_CuBLAS_Handle.hpp"
#include "tpl/cublas/KokkosDNN_CuBLAS_GemmBiasLt.hpp"
#endif
#ifdef KOKKOSDNN_ENABLE_CUDNN
#include "tpl/cudnn/KokkosDNN_CuDNN_Handle.hpp"
#endif
#ifdef KOKKOSDNN_ENABLE_HIPBLAS
#include "tpl/hipblas/KokkosDNN_HIPBLAS_Handle.hpp"
#endif
#ifdef KOKKOSDNN_ENABLE_MIOPEN
#include "tpl/miopen/KokkosDNN_MIOpen_Handle.hpp"
#endif

namespace KokkosDNN {
namespace Impl {

// ---------------------------------------------------------------------------
// Env-gated TPL dispatch counters (observability for the dispatch audit).
//
// Every public GEMM/attention entry point records exactly one outcome per
// call (TPL-served vs native fallback) via tpl_note_gemm/gemv/bias/attn().
// Counting is active only when KOKKOSDNN_DEBUG_DISPATCH is set in the
// environment (read once, cached); otherwise each helper is a single
// never-taken branch. An atexit hook prints the table to stderr, so a short
// run (e.g. a few generated tokens) proves which ops reached cuBLAS/cuDNN:
//   KOKKOSDNN_DEBUG_DISPATCH=1 ./lfm2_5_generate ...
// Numerics-neutral: relaxed host-side atomics on the dispatch path only,
// never in device code. A gemm_bias split fallback (cuBLAS GEMM + bias
// kernel) counts once under bias and once under gemm (the inner gemm truly
// ran on the TPL); the LFM2.5 forward never calls gemm_bias, so its gemm
// row stays pure there.
// ---------------------------------------------------------------------------
struct TplDispatchCounters {
  std::atomic<uint64_t> tpl_gemm{0};
  std::atomic<uint64_t> native_gemm{0};
  std::atomic<uint64_t> tpl_gemv{0};
  std::atomic<uint64_t> native_gemv{0};
  std::atomic<uint64_t> tpl_bias{0};
  std::atomic<uint64_t> native_bias{0};
  std::atomic<uint64_t> tpl_attn{0};
  std::atomic<uint64_t> native_attn{0};
};

inline TplDispatchCounters& tpl_dispatch_counters() {
  static TplDispatchCounters c;
  return c;
}

inline bool tpl_dispatch_debug() {
  static const bool enabled =
      (std::getenv("KOKKOSDNN_DEBUG_DISPATCH") != nullptr);
  return enabled;
}

inline void tpl_dump_counters() {
  TplDispatchCounters& c = tpl_dispatch_counters();
  std::fprintf(stderr,
               "[KokkosDNN dispatch] gemm: tpl=%llu native=%llu | "
               "gemv: tpl=%llu native=%llu | gemm_bias: tpl=%llu native=%llu | "
               "attn: tpl=%llu native=%llu\n",
               (unsigned long long)c.tpl_gemm.load(std::memory_order_relaxed),
               (unsigned long long)c.native_gemm.load(std::memory_order_relaxed),
               (unsigned long long)c.tpl_gemv.load(std::memory_order_relaxed),
               (unsigned long long)c.native_gemv.load(std::memory_order_relaxed),
               (unsigned long long)c.tpl_bias.load(std::memory_order_relaxed),
               (unsigned long long)c.native_bias.load(std::memory_order_relaxed),
               (unsigned long long)c.tpl_attn.load(std::memory_order_relaxed),
               (unsigned long long)c.native_attn.load(
                   std::memory_order_relaxed));
}

inline void tpl_arm_dump_once() {
  static std::once_flag arm;
  std::call_once(arm, [] { std::atexit(tpl_dump_counters); });
}

inline void tpl_note(std::atomic<uint64_t>& took_tpl,
                     std::atomic<uint64_t>& took_native, bool tpl) {
  if (!tpl_dispatch_debug()) return;
  tpl_arm_dump_once();
  if (tpl)
    took_tpl.fetch_add(1, std::memory_order_relaxed);
  else
    took_native.fetch_add(1, std::memory_order_relaxed);
}

inline void tpl_note_gemm(bool took_tpl) {
  TplDispatchCounters& c = tpl_dispatch_counters();
  tpl_note(c.tpl_gemm, c.native_gemm, took_tpl);
}

inline void tpl_note_gemv(bool took_tpl) {
  TplDispatchCounters& c = tpl_dispatch_counters();
  tpl_note(c.tpl_gemv, c.native_gemv, took_tpl);
}

inline void tpl_note_bias(bool took_tpl) {
  TplDispatchCounters& c = tpl_dispatch_counters();
  tpl_note(c.tpl_bias, c.native_bias, took_tpl);
}

inline void tpl_note_attn(bool took_tpl) {
  TplDispatchCounters& c = tpl_dispatch_counters();
  tpl_note(c.tpl_attn, c.native_attn, took_tpl);
}

}  // namespace Impl

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
//
// The AMD contract mirrors the CUDA one: use_hipblas/use_miopen record
// whether hipBLAS/MIOpen dispatch is active (HIP execution space with the
// matching TPL enabled); hipblas_raw()/miopen_raw() lazily create one
// cached handle each via Impl::hipblas_handle_create() /
// Impl::miopen_handle_create() (see tpl/hipblas/KokkosDNN_HIPBLAS_Handle.hpp
// and tpl/miopen/KokkosDNN_MIOpen_Handle.hpp), re-selecting the Kokkos HIP
// stream per use. Opaque void* keeps this header free of TPL types.
template <class ExecSpace>
struct DNNHandle {
  bool use_cublas  = false;
  bool use_cudnn   = false;
  bool use_hipblas = false;
  bool use_miopen  = false;
  // Apple Accelerate (CPU BLAS) dispatch. Stateless -- no cached handle;
  // active on host-memory execution spaces (Serial/OpenMP/Threads) when
  // KokkosDNN_ENABLE_ACCELERATE is set. float only; other scalars fall back
  // to the native kernel at the call site.
  bool use_accelerate = false;
  // Native CUDA tensor-core GEMM (TF32 for float, BF16 tensor for bhalf_t;
  // see src/KokkosDNN_GemmTensorCuda.hpp). Active on CUDA SM80+ builds when
  // the shape passes the launcher gates; the portable SIMT kernel is always
  // the fallback. Default on (like the other performance dispatches);
  // set false for bit parity with the portable FP32 kernel.
  bool use_tensor_cores = false;

  // Opaque cached TPL handles (cublasHandle_t / cudnnHandle_t /
  // hipblasHandle_t / miopenHandle_t when the matching TPL is enabled on a
  // matching-backend build, nullptr otherwise). void* keeps this header
  // free of TPL types on builds that lack them.
  mutable void* cublas_ptr_  = nullptr;
  mutable void* cudnn_ptr_   = nullptr;
  mutable void* hipblas_ptr_ = nullptr;
  mutable void* miopen_ptr_  = nullptr;
  // Opaque cached cublasLt fused-bias state (Impl::CublasLtBiasState: one
  // cublasLt handle plus per-shape GEMM+bias plans). Same lazy/nullable
  // contract as the handles above; only used by gemm_bias/gemv_bias.
  mutable void* cublaslt_bias_ptr_ = nullptr;

  // Persistent workspace for the split-K single-token decode attention
  // partials (see KokkosDNN_Attention.hpp). Batch-1 decode calls attention
  // once per layer per token (~10 times per token for LFM2.5), so a per-call
  // device allocation would show up in the step time; this buffer is
  // allocated on first use and only grown afterwards. Mutable because the
  // attention entry points take the handle by const reference, and per
  // handle (not static) so it can never be shared across ExecSpace
  // instances; the existing usage is one handle per session/thread.
  mutable Kokkos::View<float*, ExecSpace> attn_decode_partials;

  // Returns a workspace of at least `floats` floats, growing (never
  // shrinking) the cached buffer. The fence on growth retires kernels that
  // may still be reading the old buffer before it is released.
  Kokkos::View<float*, ExecSpace> attn_decode_scratch(size_t floats) const {
    if (static_cast<size_t>(attn_decode_partials.extent(0)) < floats) {
      ExecSpace().fence();
      const size_t cap = floats + floats / 2 + 1024;
      attn_decode_partials =
          Kokkos::View<float*, ExecSpace>("KokkosDNN::attn_decode_partials",
                                          cap);
    }
    return attn_decode_partials;
  }

  DNNHandle() {
#if defined(KOKKOS_ENABLE_CUDA)
    // Naming the backend here is deliberate: the tensor-core kernel below
    // is CUDA-only (wmma), so the default keys off the CUDA execution
    // space itself. Independent of the TPL flags: tensor cores are native.
    constexpr bool is_cuda_tc = std::is_same_v<ExecSpace, Kokkos::Cuda>;
    if constexpr (is_cuda_tc) {
      use_tensor_cores = true;
    }
#endif
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
#if (defined(KOKKOSDNN_ENABLE_HIPBLAS) || defined(KOKKOSDNN_ENABLE_MIOPEN)) && \
    defined(KOKKOS_ENABLE_HIP)
    // Naming the backend here is deliberate: the TPL handles below are
    // HIP-only, so dispatch must key off the HIP execution space itself.
    constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
    if constexpr (is_hip) {
#ifdef KOKKOSDNN_ENABLE_HIPBLAS
      use_hipblas = true;
#endif
#ifdef KOKKOSDNN_ENABLE_MIOPEN
      use_miopen = true;
#endif
    }
#endif
#ifdef KOKKOSDNN_ENABLE_ACCELERATE
    // Host-memory spaces only (Serial/OpenMP/Threads/...), never a device
    // space: CUDA/HIP/SYCL device memory is not CPU-addressable, so
    // Accelerate must not claim those execution spaces. Keyed off the
    // memory space (a Kokkos property) rather than a backend name list.
    constexpr bool is_host_mem =
        std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace>;
    if constexpr (is_host_mem) {
      use_accelerate = true;
    }
#endif
  }

  ~DNNHandle() { destroy(); }

  DNNHandle(const DNNHandle&)            = delete;
  DNNHandle& operator=(const DNNHandle&) = delete;

  DNNHandle(DNNHandle&& other) noexcept
      : use_cublas(other.use_cublas),
        use_cudnn(other.use_cudnn),
        use_hipblas(other.use_hipblas),
        use_miopen(other.use_miopen),
        use_accelerate(other.use_accelerate),
        use_tensor_cores(other.use_tensor_cores),
        cublas_ptr_(other.cublas_ptr_),
        cudnn_ptr_(other.cudnn_ptr_),
        hipblas_ptr_(other.hipblas_ptr_),
        miopen_ptr_(other.miopen_ptr_),
        cublaslt_bias_ptr_(other.cublaslt_bias_ptr_),
        attn_decode_partials(std::move(other.attn_decode_partials)) {
    other.use_cublas   = false;
    other.use_cudnn    = false;
    other.use_hipblas  = false;
    other.use_miopen   = false;
    other.use_accelerate = false;
    other.use_tensor_cores = false;
    other.cublas_ptr_  = nullptr;
    other.cudnn_ptr_   = nullptr;
    other.hipblas_ptr_ = nullptr;
    other.miopen_ptr_  = nullptr;
    other.cublaslt_bias_ptr_ = nullptr;
  }

  DNNHandle& operator=(DNNHandle&& other) noexcept {
    if (this != &other) {
      destroy();
      use_cublas   = other.use_cublas;
      use_cudnn    = other.use_cudnn;
      use_hipblas  = other.use_hipblas;
      use_miopen   = other.use_miopen;
      use_accelerate = other.use_accelerate;
      use_tensor_cores = other.use_tensor_cores;
      cublas_ptr_  = other.cublas_ptr_;
      cudnn_ptr_   = other.cudnn_ptr_;
      hipblas_ptr_ = other.hipblas_ptr_;
      miopen_ptr_  = other.miopen_ptr_;
      cublaslt_bias_ptr_ = other.cublaslt_bias_ptr_;
      attn_decode_partials = std::move(other.attn_decode_partials);
      other.use_cublas   = false;
      other.use_cudnn    = false;
      other.use_hipblas  = false;
      other.use_miopen   = false;
      other.use_accelerate = false;
      other.use_tensor_cores = false;
      other.cublas_ptr_  = nullptr;
      other.cudnn_ptr_   = nullptr;
      other.hipblas_ptr_ = nullptr;
      other.miopen_ptr_  = nullptr;
      other.cublaslt_bias_ptr_ = nullptr;
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

  // Lazily created cached cublasLt fused-bias state (nullptr when
  // unusable: non-CUDA space, TPL disabled, or creation failure -- callers
  // fall back then). Const so gemm_bias/gemv_bias can take
  // `const DNNHandle&`. Lifecycle in
  // tpl/cublas/KokkosDNN_CuBLAS_GemmBiasLt.hpp.
  void* cublaslt_bias_state() const {
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
    // See ctor: backend named deliberately, this state is CUDA-only.
    constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
    if constexpr (is_cuda) {
      if (use_cublas && cublaslt_bias_ptr_ == nullptr) {
        cublaslt_bias_ptr_ = Impl::cublaslt_bias_state_create();
      }
      return cublaslt_bias_ptr_;
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

  // Lazily created cached hipBLAS handle (nullptr contract as above).
  // Lifecycle in tpl/hipblas/KokkosDNN_HIPBLAS_Handle.hpp.
  void* hipblas_raw() const {
#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
    // See ctor: backend named deliberately, this handle is HIP-only.
    constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
    if constexpr (is_hip) {
      if (use_hipblas && hipblas_ptr_ == nullptr) {
        hipblas_ptr_ = Impl::hipblas_handle_create();
      }
      return hipblas_ptr_;
    }
#endif
    return nullptr;
  }

  // Lazily created cached MIOpen handle (nullptr contract as above).
  // Lifecycle in tpl/miopen/KokkosDNN_MIOpen_Handle.hpp.
  void* miopen_raw() const {
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
    // See ctor: backend named deliberately, this handle is HIP-only.
    constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
    if constexpr (is_hip) {
      if (use_miopen && miopen_ptr_ == nullptr) {
        miopen_ptr_ = Impl::miopen_handle_create();
      }
      return miopen_ptr_;
    }
#endif
    return nullptr;
  }

  private:
   void destroy() {
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
    Impl::cublas_handle_destroy(cublas_ptr_);
    Impl::cublaslt_bias_state_destroy(cublaslt_bias_ptr_);
#else
    cublas_ptr_ = nullptr;
    cublaslt_bias_ptr_ = nullptr;
#endif
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
    Impl::cudnn_handle_destroy(cudnn_ptr_);
#else
    cudnn_ptr_ = nullptr;
#endif
#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
    Impl::hipblas_handle_destroy(hipblas_ptr_);
#else
    hipblas_ptr_ = nullptr;
#endif
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
    Impl::miopen_handle_destroy(miopen_ptr_);
#else
    miopen_ptr_ = nullptr;
#endif
  }
};

}  // namespace KokkosDNN

#endif
