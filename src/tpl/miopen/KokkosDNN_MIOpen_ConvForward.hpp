#ifndef KOKKOSDNN_MIOPEN_CONVFORWARD_HPP
#define KOKKOSDNN_MIOPEN_CONVFORWARD_HPP

// MIOpen convolution dispatch for the three native-only conv ops in
// src/KokkosDNN_Conv.hpp (dense 3x3 pad 1, depthwise odd-K pad K/2,
// grouped 1x1). This is new capability, not a mirror of the CUDA path:
// the NVIDIA build deliberately has NO cuDNN dispatch (cudnn_cnn
// sublibrary absent + frontend conv node limitations, see Conv.hpp file
// docs), while on AMD MIOpen convolution (miopenConvolutionForward /
// miopenConvolutionForwardBias + miopenSetConvolutionGroupCount) is fully
// available, including depthwise/grouped.
//
// File-ownership coordination: the attention effort owns
// src/tpl/miopen/* attention files plus the Handle/Types/umbrella
// headers. This file (and KokkosDNN_MIOpen_ConvForward.cpp) are the ONLY
// miopen files this conv work creates; the expected handle contract
// (option KokkosDNN_ENABLE_MIOPEN, macro KOKKOSDNN_ENABLE_MIOPEN, handle
// members use_miopen/miopen_ptr_ + void* miopen_raw() const,
// Impl::miopen_handle_create/destroy) is documented here, not redefined.
//
// Layout contract: the token-row activations [T=H*W, C] (LayoutRight,
// row-major, t = h*W + w) hold element (h, w, c) at offset (h*W+w)*C+c,
// i.e. N=1 NHWC memory order. The raw-pointer entry points below
// describe I/O with explicit NCHW strides (nStride=H*W*C, cStride=1,
// hStride=W*C, wStride=C) over that memory, so device pointers pass
// through zero-copy. Filters already match packed KCRS order:
//   dense     w [Cout,Cin,3,3]  -> (Cout, Cin, 3, 3)
//   depthwise w [C,K,K]         -> (C, 1, K, K) with groupCount = C
//   grouped   w [Cout,Cg]       -> (Cout, Cg, 1, 1) with groupCount = groups
// Bias b [Cout] (or [C] depthwise), or nullptr for the unbiased path, is
// applied with miopenConvolutionForwardBias over a (1, Cout, 1, 1) desc.
//
// Contract (mirrors the try_cudnn_* style): view-based try_miopen_*
// wrappers do the cheap checks (HIP space, use_miopen, float/bhalf_t,
// int-range dims, contiguous + non-null views) and forward raw
// pointers/strides plus the handle's cached MIOpen handle; the .cpp
// re-selects the Kokkos HIP stream on the cached handle per call.
// Any failure returns false and the caller falls back to native.
// Never throws.
#include "KokkosDNN_config.h"

namespace KokkosDNN {
namespace Impl {

// Scalar selector for the raw-pointer entry points (kept free of Kokkos
// scalar types; mirrors CudnnSdpaScalar). Declared unconditionally (like
// tpl/miopen/KokkosDNN_MIOpen_Forward.hpp) so the .cpp can always provide
// definitions: the real MIOpen path when the guard below holds, stubs
// returning false otherwise. This guarantees the symbols exist whenever a
// caller (guarded on the same macros) references them, regardless of
// include order for KOKKOS_ENABLE_HIP.
enum class MiopenConvScalar { Float, BFloat16 };

// Raw-pointer MIOpen forwards, implemented in
// tpl/miopen/KokkosDNN_MIOpen_ConvForward.cpp.
//
// cached_handle is the DNNHandle's cached miopenHandle_t (opaque void*;
// nullptr declines). Dims are element counts (must be positive, int
// range). b == nullptr selects the unbiased path. Returns true on
// success, false to fall back to native. Never throws.
bool miopen_conv3x3_forward(void* cached_handle, MiopenConvScalar dtype,
                            const void* x, const void* w, const void* b,
                            void* y, int Cin, int Cout, int H, int W);

bool miopen_depthwise_forward(void* cached_handle, MiopenConvScalar dtype,
                              const void* x, const void* w, const void* b,
                              void* y, int C, int K, int H, int W);

bool miopen_grouped1x1_forward(void* cached_handle, MiopenConvScalar dtype,
                               const void* x, const void* w, const void* b,
                               void* y, int T, int Cin, int Cout, int groups);

}  // namespace Impl
}  // namespace KokkosDNN

#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstddef>
#include <type_traits>

#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Impl {

// Dense 3x3, stride 1, pad 1: view-based dispatch over x [T,Cin],
// w [Cout,Cin,3,3], b [Cout] (or empty), y [T,Cout] with T = H*W.
template <class Scalar, class Exec, class XView, class WView, class BView,
          class YView>
inline bool try_miopen_conv3x3_forward(const DNNHandle<Exec>& handle,
                                       const XView& x, const WView& w,
                                       const BView& b, const YView& y, size_t H,
                                       size_t W) {
  constexpr bool is_hip = std::is_same_v<Exec, Kokkos::HIP>;
  if constexpr (!is_hip) {
    return false;
  } else {
    if (!handle.use_miopen) return false;
    MiopenConvScalar dtype;
    if constexpr (std::is_same_v<Scalar, float>) {
      dtype = MiopenConvScalar::Float;
    } else if constexpr (std::is_same_v<
                             Scalar, Kokkos::Experimental::bhalf_t>) {
      dtype = MiopenConvScalar::BFloat16;
    } else {
      return false;
    }
    // int-range dims (descriptors take int).
    const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
    if (T == 0 || Cin == 0 || Cout == 0 || H == 0 || W == 0) return false;
    if (T != H * W) return false;
    if (T > 2147483647u || Cin > 2147483647u || Cout > 2147483647u ||
        H > 2147483647u || W > 2147483647u)
      return false;
    // Zero-copy requires packed LayoutRight views.
    if (!x.span_is_contiguous() || !w.span_is_contiguous() ||
        !y.span_is_contiguous())
      return false;
    if (x.data() == nullptr || w.data() == nullptr || y.data() == nullptr)
      return false;
    const bool biased = (b.extent(0) != 0);
    if (biased) {
      if (!b.span_is_contiguous() || b.data() == nullptr) return false;
    }
    void* raw = handle.miopen_raw();
    if (raw == nullptr) return false;
    return miopen_conv3x3_forward(
        raw, dtype, static_cast<const void*>(x.data()),
        static_cast<const void*>(w.data()),
        biased ? static_cast<const void*>(b.data()) : nullptr,
        static_cast<void*>(y.data()), static_cast<int>(Cin),
        static_cast<int>(Cout), static_cast<int>(H), static_cast<int>(W));
  }
}

// Depthwise KxK, stride 1, pad K/2: view-based dispatch over x/y [T,C],
// w [C,K,K], b [C] (or empty) with T = H*W.
template <class Scalar, class Exec, class XView, class WView, class BView,
          class YView>
inline bool try_miopen_depthwise_forward(const DNNHandle<Exec>& handle,
                                         const XView& x, const WView& w,
                                         const BView& b, const YView& y,
                                         size_t K, size_t H, size_t W) {
  constexpr bool is_hip = std::is_same_v<Exec, Kokkos::HIP>;
  if constexpr (!is_hip) {
    return false;
  } else {
    if (!handle.use_miopen) return false;
    MiopenConvScalar dtype;
    if constexpr (std::is_same_v<Scalar, float>) {
      dtype = MiopenConvScalar::Float;
    } else if constexpr (std::is_same_v<
                             Scalar, Kokkos::Experimental::bhalf_t>) {
      dtype = MiopenConvScalar::BFloat16;
    } else {
      return false;
    }
    const size_t T = x.extent(0), C = x.extent(1);
    if (T == 0 || C == 0 || K == 0 || H == 0 || W == 0) return false;
    if (T != H * W) return false;
    if (T > 2147483647u || C > 2147483647u || K > 2147483647u ||
        H > 2147483647u || W > 2147483647u)
      return false;
    if (!x.span_is_contiguous() || !w.span_is_contiguous() ||
        !y.span_is_contiguous())
      return false;
    if (x.data() == nullptr || w.data() == nullptr || y.data() == nullptr)
      return false;
    const bool biased = (b.extent(0) != 0);
    if (biased) {
      if (!b.span_is_contiguous() || b.data() == nullptr) return false;
    }
    void* raw = handle.miopen_raw();
    if (raw == nullptr) return false;
    return miopen_depthwise_forward(
        raw, dtype, static_cast<const void*>(x.data()),
        static_cast<const void*>(w.data()),
        biased ? static_cast<const void*>(b.data()) : nullptr,
        static_cast<void*>(y.data()), static_cast<int>(C),
        static_cast<int>(K), static_cast<int>(H), static_cast<int>(W));
  }
}

// Grouped 1x1 (no padding): view-based dispatch over x [T,Cin],
// w [Cout,Cg], b [Cout] (or empty), y [T,Cout].
template <class Scalar, class Exec, class XView, class WView, class BView,
          class YView>
inline bool try_miopen_grouped1x1_forward(const DNNHandle<Exec>& handle,
                                          const XView& x, const WView& w,
                                          const BView& b, const YView& y,
                                          size_t groups) {
  constexpr bool is_hip = std::is_same_v<Exec, Kokkos::HIP>;
  if constexpr (!is_hip) {
    return false;
  } else {
    if (!handle.use_miopen) return false;
    MiopenConvScalar dtype;
    if constexpr (std::is_same_v<Scalar, float>) {
      dtype = MiopenConvScalar::Float;
    } else if constexpr (std::is_same_v<
                             Scalar, Kokkos::Experimental::bhalf_t>) {
      dtype = MiopenConvScalar::BFloat16;
    } else {
      return false;
    }
    const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
    if (T == 0 || Cin == 0 || Cout == 0 || groups == 0) return false;
    if (T > 2147483647u || Cin > 2147483647u || Cout > 2147483647u ||
        groups > 2147483647u)
      return false;
    if (!x.span_is_contiguous() || !w.span_is_contiguous() ||
        !y.span_is_contiguous())
      return false;
    if (x.data() == nullptr || w.data() == nullptr || y.data() == nullptr)
      return false;
    const bool biased = (b.extent(0) != 0);
    if (biased) {
      if (!b.span_is_contiguous() || b.data() == nullptr) return false;
    }
    void* raw = handle.miopen_raw();
    if (raw == nullptr) return false;
    return miopen_grouped1x1_forward(
        raw, dtype, static_cast<const void*>(x.data()),
        static_cast<const void*>(w.data()),
        biased ? static_cast<const void*>(b.data()) : nullptr,
        static_cast<void*>(y.data()), static_cast<int>(T),
        static_cast<int>(Cin), static_cast<int>(Cout),
        static_cast<int>(groups));
  }
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
