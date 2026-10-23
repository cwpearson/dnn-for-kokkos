// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// MIOpen convolution forwards declared in
// tpl/miopen/KokkosDNN_MIOpen_ConvForward.hpp.
//
// Single translation unit that includes <miopen/miopen.h> so the rest of
// the library stays free of MIOpen types (mirrors the
// tpl/cudnn/KokkosDNN_CuDNN_SDPA.cpp arrangement). Compiled only when the
// HIP-infra branch lands (option KokkosDNN_ENABLE_MIOPEN); until then this
// TU is excluded from the build by the surrounding guard.
//
// Method: NCHW descriptors with explicit strides over the token-row
// layout, one shared impl for all three ops:
//   * I/O: miopenSet4dTensorDescriptorEx (1, C, H, W) with
//     nStride=H*W*C, cStride=1, hStride=W*C, wStride=C (NHWC memory).
//   * Filters (packed KCRS, zero-copy): (Cout, Cin, 3, 3) dense,
//     (C, 1, K, K) depthwise with groupCount=C, (Cout, Cg, 1, 1)
//     grouped with groupCount=groups.
//   * Conv: miopenInitConvolutionDescriptor(miopenConvolution, pad, pad,
//     stride 1, dilation 1) + miopenSetConvolutionGroupCount when grouped.
//   * Algorithm: miopenConvolutionForwardGetWorkSpaceSize, device workspace
//     via hipMalloc, miopenFindConvolutionForwardAlgorithm (1 request,
//     non-exhaustive), miopenConvolutionForward with alpha=1/beta=0 (the
//     only 2D-supported values), optional miopenConvolutionForwardBias
//     over a (1, Cout, 1, 1) descriptor for the biased path.
// The cached handle's stream is re-selected from Kokkos::HIP().hip_stream
// per call so the one cached handle stays ordered with surrounding Kokkos
// kernels. Every miopen*/hip* failure (including bfloat16 configs the
// build does not support) returns false; never throws.
#include "KokkosDNN_config.h"

#include <Kokkos_Core.hpp>

#include "tpl/miopen/KokkosDNN_MIOpen_ConvForward.hpp"

#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)

#include <miopen/miopen.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace KokkosDNN {
namespace Impl {
namespace {

// RAII MIOpen tensor descriptor (non-copyable; null when creation fails).
struct TensorDesc {
  miopenTensorDescriptor_t d = nullptr;
  TensorDesc() {
    if (miopenCreateTensorDescriptor(&d) != miopenStatusSuccess) d = nullptr;
  }
  ~TensorDesc() {
    if (d != nullptr) miopenDestroyTensorDescriptor(d);
  }
  TensorDesc(const TensorDesc&)            = delete;
  TensorDesc& operator=(const TensorDesc&) = delete;
  bool valid() const { return d != nullptr; }
};

// RAII MIOpen convolution descriptor.
struct ConvDesc {
  miopenConvolutionDescriptor_t d = nullptr;
  ConvDesc() {
    if (miopenCreateConvolutionDescriptor(&d) != miopenStatusSuccess)
      d = nullptr;
  }
  ~ConvDesc() {
    if (d != nullptr) miopenDestroyConvolutionDescriptor(d);
  }
  ConvDesc(const ConvDesc&)            = delete;
  ConvDesc& operator=(const ConvDesc&) = delete;
  bool valid() const { return d != nullptr; }
};

// RAII device workspace (hipFree on scope exit; null when empty/failed).
struct Workspace {
  void* p = nullptr;
  explicit Workspace(size_t bytes) {
    if (bytes != 0 && hipMalloc(&p, bytes) != hipSuccess) p = nullptr;
  }
  ~Workspace() {
    if (p != nullptr) (void)hipFree(p);
  }
  Workspace(const Workspace&)            = delete;
  Workspace& operator=(const Workspace&) = delete;
};

// Shared forward-convolution impl. I/O are N=1 (H, W, C); the filter is
// packed KCRS (fC = per-group input channels); b_or_null == nullptr is
// the unbiased path. groups == 1 skips miopenSetConvolutionGroupCount.
bool miopen_conv_forward_impl(void* cached_handle, MiopenConvScalar dtype,
                              const void* x, const void* w,
                              const void* b_or_null, void* y, int C, int Cout,
                              int H, int W, int fC, int kH, int kW, int padH,
                              int padW, int groups) {
  if (cached_handle == nullptr || x == nullptr || w == nullptr ||
      y == nullptr)
    return false;
  if (C <= 0 || Cout <= 0 || H <= 0 || W <= 0 || fC <= 0 || kH <= 0 ||
      kW <= 0 || padH < 0 || padW < 0 || groups <= 0)
    return false;
  // Explicit-stride products in 64 bit before narrowing to int.
  const int64_t iH = H, iW = W, iC = C, iCo = Cout;
  const int64_t wcx = iW * iC, wco = iW * iCo;
  const int64_t nx = wcx * iH, ny = wco * iH;
  constexpr int64_t kIntMax = 2147483647;
  if (wcx > kIntMax || wco > kIntMax || nx > kIntMax || ny > kIntMax)
    return false;

  const miopenDataType_t dt = (dtype == MiopenConvScalar::BFloat16)
                                  ? miopenBFloat16
                                  : miopenFloat;
  miopenHandle_t h = static_cast<miopenHandle_t>(cached_handle);

  // Per-call stream re-selection on the cached handle (mirrors the cached
  // cuDNN handle pattern: one handle, ordered with Kokkos work).
  const hipStream_t kokkos_stream = Kokkos::HIP().hip_stream();
  if (miopenSetStream(h, kokkos_stream) != miopenStatusSuccess) return false;

  TensorDesc xDesc, wDesc, yDesc, bDesc;
  ConvDesc conv;
  if (!xDesc.valid() || !wDesc.valid() || !yDesc.valid() || !bDesc.valid() ||
      !conv.valid())
    return false;

  const int nStrideX = static_cast<int>(nx);
  const int nStrideY = static_cast<int>(ny);
  if (miopenSet4dTensorDescriptorEx(xDesc.d, dt, 1, C, H, W, nStrideX, 1,
                                    static_cast<int>(wcx), C) !=
      miopenStatusSuccess)
    return false;
  if (miopenSet4dTensorDescriptorEx(yDesc.d, dt, 1, Cout, H, W, nStrideY, 1,
                                    static_cast<int>(wco),
                                    Cout) != miopenStatusSuccess)
    return false;
  if (miopenSet4dTensorDescriptor(wDesc.d, dt, Cout, fC, kH, kW) !=
      miopenStatusSuccess)
    return false;
  if (miopenInitConvolutionDescriptor(conv.d, miopenConvolution, padH, padW,
                                      1, 1, 1, 1) != miopenStatusSuccess)
    return false;
  if (groups > 1 &&
      miopenSetConvolutionGroupCount(conv.d, groups) != miopenStatusSuccess)
    return false;
  if (b_or_null != nullptr &&
      miopenSet4dTensorDescriptor(bDesc.d, dt, 1, Cout, 1, 1) !=
          miopenStatusSuccess)
    return false;

  size_t wsSize = 0;
  // NOTE arg order: wDesc first, then xDesc.
  if (miopenConvolutionForwardGetWorkSpaceSize(h, wDesc.d, xDesc.d, conv.d,
                                               yDesc.d, &wsSize) !=
      miopenStatusSuccess)
    return false;
  Workspace ws(wsSize);
  if (wsSize != 0 && ws.p == nullptr) return false;

  miopenConvAlgoPerf_t perf{};
  int returned = 0;
  // NOTE arg order: xDesc/x first, then wDesc/w.
  if (miopenFindConvolutionForwardAlgorithm(h, xDesc.d, x, wDesc.d, w, conv.d,
                                            yDesc.d, y, 1, &returned, &perf,
                                            ws.p, wsSize,
                                            false) != miopenStatusSuccess)
    return false;
  if (returned < 1) return false;

  // 2D supports only alpha=1, beta=0; host float scalars per miopen.h.
  const float alpha = 1.0f, beta = 0.0f;
  if (miopenConvolutionForward(h, &alpha, xDesc.d, x, wDesc.d, w, conv.d,
                               perf.fwd_algo, &beta, yDesc.d, y, ws.p,
                               wsSize) != miopenStatusSuccess)
    return false;
  if (b_or_null != nullptr &&
      miopenConvolutionForwardBias(h, &alpha, bDesc.d, b_or_null, &beta,
                                   yDesc.d, y) != miopenStatusSuccess)
    return false;
  return true;
}

}  // namespace

bool miopen_conv3x3_forward(void* cached_handle, MiopenConvScalar dtype,
                            const void* x, const void* w, const void* b,
                            void* y, int Cin, int Cout, int H, int W) {
  try {
    return miopen_conv_forward_impl(cached_handle, dtype, x, w, b, y, Cin,
                                    Cout, H, W, Cin, 3, 3, 1, 1, 1);
  } catch (...) {
    return false;
  }
}

bool miopen_depthwise_forward(void* cached_handle, MiopenConvScalar dtype,
                              const void* x, const void* w, const void* b,
                              void* y, int C, int K, int H, int W) {
  try {
    return miopen_conv_forward_impl(cached_handle, dtype, x, w, b, y, C, C,
                                    H, W, 1, K, K, K / 2, K / 2, C);
  } catch (...) {
    return false;
  }
}

bool miopen_grouped1x1_forward(void* cached_handle, MiopenConvScalar dtype,
                               const void* x, const void* w, const void* b,
                               void* y, int T, int Cin, int Cout, int groups) {
  try {
    // Filter's per-group channel count derives from the packed [Cout,Cg]
    // weight: Cg = Cin / groups (validated by the caller). The 1x1 kernel
    // has no spatial extent, so the T token rows are viewed as H=T, W=1.
    if (T <= 0 || Cin <= 0 || groups <= 0 || Cin % groups != 0) return false;
    const int Cg = Cin / groups;
    return miopen_conv_forward_impl(cached_handle, dtype, x, w, b, y, Cin,
                                    Cout, T, 1, Cg, 1, 1, 0, 0, groups);
  } catch (...) {
    return false;
  }
}

}  // namespace Impl
}  // namespace KokkosDNN

#else

namespace KokkosDNN {
namespace Impl {

bool miopen_conv3x3_forward(void*, MiopenConvScalar, const void*, const void*,
                            const void*, void*, int, int, int, int) {
  return false;
}

bool miopen_depthwise_forward(void*, MiopenConvScalar, const void*,
                              const void*, const void*, void*, int, int, int,
                              int) {
  return false;
}

bool miopen_grouped1x1_forward(void*, MiopenConvScalar, const void*,
                               const void*, const void*, void*, int, int, int,
                               int) {
  return false;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
