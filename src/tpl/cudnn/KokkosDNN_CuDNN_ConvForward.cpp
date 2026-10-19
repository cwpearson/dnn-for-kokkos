// cuDNN convolution forwards declared in
// tpl/cudnn/KokkosDNN_CuDNN_ConvForward.hpp.
//
// Single translation unit that includes <cudnn.h> so the rest of the
// library stays free of cuDNN types (mirrors the
// tpl/cudnn/KokkosDNN_CuDNN_SDPA.cpp arrangement). Compiled only when the
// CUDA/cuDNN path is enabled; otherwise the stubs below keep the symbols
// defined.
//
// Method: NCHW descriptors with explicit strides over the token-row
// layout, one shared impl for all three ops:
//   * I/O: cudnnSetTensor4dDescriptorEx (1, C, H, W) with
//     nStride=H*W*C, cStride=1, hStride=W*C, wStride=C (NHWC memory).
//   * Filters (packed KCRS, zero-copy): (Cout, Cin, 3, 3) dense,
//     (C, 1, K, K) depthwise with groupCount=C, (Cout, Cg, 1, 1)
//     grouped with groupCount=groups.
//   * Conv: cudnnSetConvolution2dDescriptor(pad, pad, stride 1,
//     dilation 1, CUDNN_CROSS_CORRELATION, FLOAT compute) +
//     cudnnSetConvolutionGroupCount when grouped.  Math is
//     CUDNN_FMA_MATH for FP32 (full FP32, same numerics contract as the
//     im2col + sgemm lowering: no TF32 down-conversion) and
//     CUDNN_TENSOR_OP_MATH for BF16.
//   * Algorithm: the heuristic (no-execution) v7 query picks the algo
//     once per shape; the workspace is sized once and kept in a per-shape
//     plan cache with a per-key persistent device buffer (zero
//     steady-state allocations, same discipline as the SDPA plan cache),
//     then cudnnConvolutionForward with alpha=1/beta=0, optional
//     cudnnAddTensor over a (1, Cout, 1, 1) descriptor for the biased path.
// The cached handle's stream is re-selected from
// Kokkos::Cuda().cuda_stream() per call so the one cached handle stays
// ordered with surrounding Kokkos kernels. Every cudnn*/cuda* failure
// returns false; never throws.
#include "KokkosDNN_config.h"

#include <Kokkos_Core.hpp>

#include "tpl/cudnn/KokkosDNN_CuDNN_ConvForward.hpp"

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)

// Legacy cudnnConvolution* APIs are marked deprecated in cuDNN 9 headers;
// silence those warnings for this TU (they are the supported path here).
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <cudnn.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace KokkosDNN {
namespace Impl {
namespace {

// Persistent per-shape plan cache (mirrors the SDPA plan cache in
// KokkosDNN_CuDNN_SDPA.cpp): the heuristic v7 algo query runs once per
// key, and the device workspace is allocated once per key and reused, so
// steady-state decode performs no host-side selection and no allocation.
// FIFO-evicted past capacity; deterministic query failures are negatively
// cached in `bad`. The cached plan depends only on the key below, never
// on data pointers, so repeat calls re-execute with fresh pointers.
constexpr std::size_t kConvPlanCacheCapacity = 64;

struct ConvPlanKey {
  CudnnConvScalar dtype = CudnnConvScalar::Float;
  int C = 0, Cout = 0, H = 0, W = 0;
  int fC = 0, kH = 0, kW = 0, padH = 0, padW = 0, groups = 0;

  bool operator==(const ConvPlanKey& o) const {
    return dtype == o.dtype && C == o.C && Cout == o.Cout && H == o.H &&
           W == o.W && fC == o.fC && kH == o.kH && kW == o.kW &&
           padH == o.padH && padW == o.padW && groups == o.groups;
  }
};

struct ConvPlanKeyHash {
  static void mix(std::size_t& h, std::size_t v) {
    h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
  }
  std::size_t operator()(const ConvPlanKey& k) const noexcept {
    std::size_t h = std::hash<int>{}(static_cast<int>(k.dtype));
    mix(h, std::hash<int>{}(k.C));
    mix(h, std::hash<int>{}(k.Cout));
    mix(h, std::hash<int>{}(k.H));
    mix(h, std::hash<int>{}(k.W));
    mix(h, std::hash<int>{}(k.fC));
    mix(h, std::hash<int>{}(k.kH));
    mix(h, std::hash<int>{}(k.kW));
    mix(h, std::hash<int>{}(k.padH));
    mix(h, std::hash<int>{}(k.padW));
    mix(h, std::hash<int>{}(k.groups));
    return h;
  }
};

// Cached value: the selected algo, its workspace size, and a persistent
// device workspace buffer allocated once per key and reused.
struct ConvPlanEntry {
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  std::size_t wsSize = 0;
  void* ws = nullptr;

  ConvPlanEntry() = default;
  ConvPlanEntry(const ConvPlanEntry&) = delete;
  ConvPlanEntry& operator=(const ConvPlanEntry&) = delete;
  ~ConvPlanEntry() {
    if (ws != nullptr) (void)cudaFree(ws);
  }
};

struct ConvPlanCache {
  std::mutex mutex;
  std::unordered_map<ConvPlanKey, std::shared_ptr<ConvPlanEntry>,
                     ConvPlanKeyHash>
      plans;
  std::deque<ConvPlanKey> order;  // insertion order for FIFO eviction
  // Keys whose algo query deterministically failed. Query outcomes depend
  // only on the key (dims/strides/dtype), never on data pointers, so
  // repeat calls with a known-bad key decline immediately.
  std::unordered_set<ConvPlanKey, ConvPlanKeyHash> bad;
  std::deque<ConvPlanKey> bad_order;  // insertion order for FIFO eviction
};

ConvPlanCache& conv_plan_cache() {
  static ConvPlanCache cache;
  return cache;
}

void note_conv_bad(const ConvPlanKey& key) {
  ConvPlanCache& cache = conv_plan_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (cache.bad.find(key) == cache.bad.end()) {
    while (cache.bad.size() >= kConvPlanCacheCapacity &&
           !cache.bad_order.empty()) {
      cache.bad.erase(cache.bad_order.front());
      cache.bad_order.pop_front();
    }
    cache.bad.insert(key);
    cache.bad_order.push_back(key);
  }
}

// RAII cuDNN tensor descriptor (non-copyable; null when creation fails).
struct TensorDesc {
  cudnnTensorDescriptor_t d = nullptr;
  TensorDesc() {
    if (cudnnCreateTensorDescriptor(&d) != CUDNN_STATUS_SUCCESS) d = nullptr;
  }
  ~TensorDesc() {
    if (d != nullptr) (void)cudnnDestroyTensorDescriptor(d);
  }
  TensorDesc(const TensorDesc&) = delete;
  TensorDesc& operator=(const TensorDesc&) = delete;
  bool valid() const { return d != nullptr; }
};

// RAII cuDNN filter descriptor.
struct FilterDesc {
  cudnnFilterDescriptor_t d = nullptr;
  FilterDesc() {
    if (cudnnCreateFilterDescriptor(&d) != CUDNN_STATUS_SUCCESS) d = nullptr;
  }
  ~FilterDesc() {
    if (d != nullptr) (void)cudnnDestroyFilterDescriptor(d);
  }
  FilterDesc(const FilterDesc&) = delete;
  FilterDesc& operator=(const FilterDesc&) = delete;
  bool valid() const { return d != nullptr; }
};

// RAII cuDNN convolution descriptor.
struct ConvDesc {
  cudnnConvolutionDescriptor_t d = nullptr;
  ConvDesc() {
    if (cudnnCreateConvolutionDescriptor(&d) != CUDNN_STATUS_SUCCESS)
      d = nullptr;
  }
  ~ConvDesc() {
    if (d != nullptr) (void)cudnnDestroyConvolutionDescriptor(d);
  }
  ConvDesc(const ConvDesc&) = delete;
  ConvDesc& operator=(const ConvDesc&) = delete;
  bool valid() const { return d != nullptr; }
};

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

// Shared forward-convolution impl. I/O are N=1 (H, W, C); the filter is
// packed KCRS (fC = per-group input channels); b_or_null == nullptr is
// the unbiased path. groups == 1 skips cudnnSetConvolutionGroupCount.
bool cudnn_conv_forward_impl(void* cached_handle, CudnnConvScalar dtype,
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
  if (b_or_null != nullptr && iCo > kIntMax) return false;

  const cudnnDataType_t dt = (dtype == CudnnConvScalar::BFloat16)
                                 ? CUDNN_DATA_BFLOAT16
                                 : CUDNN_DATA_FLOAT;
  cudnnHandle_t h = static_cast<cudnnHandle_t>(cached_handle);

  // Per-call stream re-selection on the cached handle (mirrors the cached
  // SDPA handle pattern: one handle, ordered with Kokkos work).
  const cudaStream_t kokkos_stream = Kokkos::Cuda().cuda_stream();
  if (cudnnSetStream(h, kokkos_stream) != CUDNN_STATUS_SUCCESS) return false;

  TensorDesc xDesc, yDesc, bDesc;
  FilterDesc wDesc;
  ConvDesc conv;
  if (!xDesc.valid() || !yDesc.valid() || !bDesc.valid() || !wDesc.valid() ||
      !conv.valid())
    return false;

  if (cudnnSetTensor4dDescriptorEx(xDesc.d, dt, 1, C, H, W,
                                  static_cast<int>(nx), 1,
                                  static_cast<int>(wcx), C) !=
      CUDNN_STATUS_SUCCESS)
    return false;
  if (cudnnSetTensor4dDescriptorEx(yDesc.d, dt, 1, Cout, H, W,
                                  static_cast<int>(ny), 1,
                                  static_cast<int>(wco), Cout) !=
      CUDNN_STATUS_SUCCESS)
    return false;
  if (cudnnSetFilter4dDescriptor(wDesc.d, dt, CUDNN_TENSOR_NCHW, Cout, fC, kH,
                                kW) != CUDNN_STATUS_SUCCESS)
    return false;
  if (cudnnSetConvolution2dDescriptor(conv.d, padH, padW, 1, 1, 1, 1,
                                      CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS)
    return false;
  // FP32 I/O stays FP32 (same numerics contract as the im2col + sgemm
  // lowering and the old native kernel; no TF32 down-conversion). BF16
  // uses tensor-op math with FP32 accumulation.
  const cudnnMathType_t mathType = (dtype == CudnnConvScalar::BFloat16)
                                       ? CUDNN_TENSOR_OP_MATH
                                       : CUDNN_FMA_MATH;
  if (cudnnSetConvolutionMathType(conv.d, mathType) != CUDNN_STATUS_SUCCESS)
    return false;
  if (groups > 1 &&
      cudnnSetConvolutionGroupCount(conv.d, groups) != CUDNN_STATUS_SUCCESS)
    return false;
  if (b_or_null != nullptr &&
      cudnnSetTensor4dDescriptor(bDesc.d, CUDNN_TENSOR_NCHW, dt, 1, Cout, 1,
                                 1) != CUDNN_STATUS_SUCCESS)
    return false;

  // Plan-cache lookup: the algo + workspace size depend only on the key.
  ConvPlanKey key;
  key.dtype = dtype;
  key.C = C;
  key.Cout = Cout;
  key.H = H;
  key.W = W;
  key.fC = fC;
  key.kH = kH;
  key.kW = kW;
  key.padH = padH;
  key.padW = padW;
  key.groups = groups;
  std::shared_ptr<ConvPlanEntry> hit;
  {
    ConvPlanCache& cache = conv_plan_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.bad.find(key) != cache.bad.end()) return false;
    auto it = cache.plans.find(key);
    if (it != cache.plans.end()) hit = it->second;
  }

  if (hit == nullptr) {
    // Cache miss: heuristic algorithm pick (no benchmarking launches),
    // then workspace-size query and one persistent device allocation.
    // Any failure records a negative entry so repeat calls skip the
    // rebuild and fall through to native immediately.
    int maxCount = 0;
    if (cudnnGetConvolutionForwardAlgorithmMaxCount(h, &maxCount) !=
            CUDNN_STATUS_SUCCESS ||
        maxCount <= 0) {
      note_conv_bad(key);
      return false;
    }
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perfs(
        static_cast<std::size_t>(maxCount));
    int returned = 0;
    if (cudnnGetConvolutionForwardAlgorithm_v7(
            h, xDesc.d, wDesc.d, conv.d, yDesc.d, maxCount, &returned,
            perfs.data()) != CUDNN_STATUS_SUCCESS ||
        returned < 1) {
      note_conv_bad(key);
      return false;
    }
    cudnnConvolutionFwdAlgo_t algo = perfs[0].algo;
    bool have_algo = (perfs[0].status == CUDNN_STATUS_SUCCESS);
    for (int i = 0; have_algo == false && i < returned; ++i) {
      if (perfs[static_cast<std::size_t>(i)].status == CUDNN_STATUS_SUCCESS) {
        algo = perfs[static_cast<std::size_t>(i)].algo;
        have_algo = true;
      }
    }
    if (!have_algo) {
      note_conv_bad(key);
      return false;
    }
    std::size_t wsSize = 0;
    if (cudnnGetConvolutionForwardWorkspaceSize(h, xDesc.d, wDesc.d, conv.d,
                                                yDesc.d, algo, &wsSize) !=
        CUDNN_STATUS_SUCCESS) {
      note_conv_bad(key);
      return false;
    }
    void* ws = nullptr;
    if (wsSize != 0) {
      // Transient (OOM) rather than deterministic: decline without a
      // negative-cache entry so a later call can retry.
      if (cudaMalloc(&ws, wsSize) != cudaSuccess) return false;
    }
    try {
      auto entry = std::make_shared<ConvPlanEntry>();
      entry->algo = algo;
      entry->wsSize = wsSize;
      entry->ws = ws;
      ConvPlanCache& cache = conv_plan_cache();
      std::lock_guard<std::mutex> lock(cache.mutex);
      if (cache.bad.find(key) != cache.bad.end()) {
        // Lost the race with a concurrent failure: drop ours.
      } else if (cache.plans.find(key) == cache.plans.end()) {
        while (cache.plans.size() >= kConvPlanCacheCapacity &&
               !cache.order.empty()) {
          auto evict_it = cache.plans.find(cache.order.front());
          if (evict_it != cache.plans.end()) cache.plans.erase(evict_it);
          cache.order.pop_front();
        }
        cache.plans.emplace(key, entry);
        cache.order.push_back(key);
        ws = nullptr;  // owned by the entry now
        hit = entry;
      } else {
        hit = cache.plans.find(key)->second;
      }
    } catch (...) {
      if (ws != nullptr) (void)cudaFree(ws);
      return false;
    }
    if (ws != nullptr) (void)cudaFree(ws);
    if (hit == nullptr) {
      note_conv_bad(key);
      return false;
    }
  }

  // Execute the cached plan with the current call's pointers (the plan
  // bakes dims/strides, not data pointers). The shared_ptr copy keeps the
  // entry (and its workspace) alive past a concurrent eviction.
  const float alpha = 1.0f, beta = 0.0f;
  if (cudnnConvolutionForward(h, &alpha, xDesc.d, x, wDesc.d, w, conv.d,
                              hit->algo, hit->ws, hit->wsSize, &beta, yDesc.d,
                              y) != CUDNN_STATUS_SUCCESS)
    return false;
  if (b_or_null != nullptr) {
    const float one = 1.0f;
    if (cudnnAddTensor(h, &one, bDesc.d, b_or_null, &one, yDesc.d, y) !=
        CUDNN_STATUS_SUCCESS)
      return false;
  }
  return true;
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

}  // namespace

bool cudnn_conv3x3_forward(void* cached_handle, CudnnConvScalar dtype,
                           const void* x, const void* w, const void* b,
                           void* y, int Cin, int Cout, int H, int W) {
  try {
    return cudnn_conv_forward_impl(cached_handle, dtype, x, w, b, y, Cin,
                                   Cout, H, W, Cin, 3, 3, 1, 1, 1);
  } catch (...) {
    return false;
  }
}

bool cudnn_depthwise_forward(void* cached_handle, CudnnConvScalar dtype,
                             const void* x, const void* w, const void* b,
                             void* y, int C, int K, int H, int W) {
  try {
    return cudnn_conv_forward_impl(cached_handle, dtype, x, w, b, y, C, C, H,
                                   W, 1, K, K, K / 2, K / 2, C);
  } catch (...) {
    return false;
  }
}

bool cudnn_grouped1x1_forward(void* cached_handle, CudnnConvScalar dtype,
                              const void* x, const void* w, const void* b,
                              void* y, int T, int Cin, int Cout, int groups) {
  try {
    // Filter's per-group channel count derives from the packed [Cout,Cg]
    // weight: Cg = Cin / groups (validated by the caller). The 1x1 kernel
    // has no spatial extent, so the T token rows are viewed as H=T, W=1.
    if (T <= 0 || Cin <= 0 || groups <= 0 || Cin % groups != 0) return false;
    const int Cg = Cin / groups;
    return cudnn_conv_forward_impl(cached_handle, dtype, x, w, b, y, Cin,
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

bool cudnn_conv3x3_forward(void*, CudnnConvScalar, const void*, const void*,
                           const void*, void*, int, int, int, int) {
  return false;
}

bool cudnn_depthwise_forward(void*, CudnnConvScalar, const void*,
                             const void*, const void*, void*, int, int, int,
                             int) {
  return false;
}

bool cudnn_grouped1x1_forward(void*, CudnnConvScalar, const void*,
                              const void*, const void*, void*, int, int, int,
                              int) {
  return false;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
