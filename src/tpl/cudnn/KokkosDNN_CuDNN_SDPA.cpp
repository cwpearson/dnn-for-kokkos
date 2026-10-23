// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// cuDNN SDPA implementation for KokkosDNN grouped-query causal attention.
//
// Single translation unit that includes <cudnn_frontend.h> so the rest of
// the (header-only) library does not pay the frontend compile cost.
// The header declares cudnn_sdpa_forward() (causal),
// cudnn_sdpa_forward_noncausal() (dense), and
// cudnn_sdpa_forward_sliding() (sliding-window causal); here they are either stubs
// returning false (CUDNN disabled or non-CUDA build) or the real graph.
//
// Real path: one inference SDPA node (generate_stats=false) with three
// mask modes sharing one graph builder below. The causal variant bakes in
// bottom-right causal masking (diagonal_alignment=BOTTOM_RIGHT,
// right_bound=0), which matches the native kernel's mask when
// Ktot == q_off + T (checked by the caller); the sliding-window variant
// additionally sets diagonal_band_left_bound to the window size, masking
// columns at or before row_idx - window (visible band of `window` keys
// per query), which reproduces the native sliding mask under the same
// KV-prefix precondition; the non-causal variant omits the diagonal-mask
// attributes for dense attention. Q/K/V/O are described with
// the caller's Kokkos strides (B=1, dims [B,H,S,D]), so arbitrary
// Kokkos layouts and cache subviews work with no repack copy.
// Implementation selection is AUTO: unified flash for BF16, composite
// (matmul+softmax+matmul) for FP32. Any build/execute failure returns
// false so the caller falls back to the native Kokkos kernel.
#include "KokkosDNN_config.h"

#include <Kokkos_Core.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "KokkosDNN_Softmax.hpp"
#include "tpl/cudnn/KokkosDNN_CuDNN_Forward.hpp"

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)

#if defined(KOKKOSDNN_ENABLE_CUBLAS)
#include <cublas_v2.h>

#include "tpl/cublas/KokkosDNN_CuBLAS_GemmFloat.hpp"
#include "tpl/cublas/KokkosDNN_CuBLAS_Stream.hpp"
#endif

#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cuda_runtime.h>

#include <cstring>
#include <deque>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fe = cudnn_frontend;

namespace KokkosDNN {
namespace Impl {

namespace {
// Persistent execution-plan cache: Sana's DiT calls cross-attention 28x per
// forward with identical shapes/strides, so rebuilding the cudnn_frontend
// graph (validate + build with HeurMode A+FALLBACK, cuDNN JIT) and
// cudaMalloc/cudaFree-ing the workspace on every call is pure host-side
// overhead. The graph bakes dims/strides but NOT data pointers, so a built
// graph can be re-executed with the current call's pointers via the variant
// pack. Keyed on every parameter that affects the built plan; bounded with
// FIFO eviction (64 entries) so memory stays flat. 64 (not 16): MOSS-Image
// DiT keys one plan per joint length Sj = HW + Sv (plus one per text length
// Sv), and Sv varies per prompt while HW varies per resolution, so a long
// session easily exceeds 16 distinct shapes; each entry is just a host-side
// graph plus a small persistent workspace, so 64 stays cheap while covering
// realistic (resolution x prompt-length) diversity. A new-shape miss costs
// ~45 ms once, then all 42 layers hit.
constexpr std::size_t kSdpaPlanCacheCapacity = 64;

struct SdpaPlanKey {
  CudnnSdpaScalar dtype = CudnnSdpaScalar::Float;
  int64_t T = 0, Ktot = 0, Hq = 0, Hkv = 0, D = 0;
  // All 16 element strides: per-tensor (sT, sH, sD) plus the per-batch
  // footprint (B stride, unused for B=1 but baked into the graph).
  int64_t q_sT = 0, q_sH = 0, q_sD = 0, q_sB = 0;
  int64_t k_sT = 0, k_sH = 0, k_sD = 0, k_sB = 0;
  int64_t v_sT = 0, v_sH = 0, v_sD = 0, v_sB = 0;
  int64_t o_sT = 0, o_sH = 0, o_sD = 0, o_sB = 0;
  uint32_t scale_bits = 0;  // bit-exact float key, no epsilon issues
  bool causal = false;
  int64_t left_bound = -1;
  // NOTE: the frontend node name is intentionally NOT part of the key: the
  // (causal, left_bound) pair already distinguishes the three mask modes,
  // and hashing a std::string per call costs a heap round-trip (SSO holds
  // only 15 chars). The name is still passed to the graph builder.

  bool operator==(const SdpaPlanKey& o) const {
    return dtype == o.dtype && T == o.T && Ktot == o.Ktot && Hq == o.Hq &&
           Hkv == o.Hkv && D == o.D && q_sT == o.q_sT && q_sH == o.q_sH &&
           q_sD == o.q_sD && q_sB == o.q_sB && k_sT == o.k_sT &&
           k_sH == o.k_sH && k_sD == o.k_sD && k_sB == o.k_sB &&
           v_sT == o.v_sT && v_sH == o.v_sH && v_sD == o.v_sD &&
           v_sB == o.v_sB && o_sT == o.o_sT && o_sH == o.o_sH &&
           o_sD == o.o_sD && o_sB == o.o_sB &&
           scale_bits == o.scale_bits && causal == o.causal &&
           left_bound == o.left_bound;
  }
};

struct SdpaPlanKeyHash {
  static void mix(std::size_t& h, std::size_t v) {
    h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
  }
  std::size_t operator()(const SdpaPlanKey& k) const noexcept {
    std::size_t h = std::hash<int>{}(static_cast<int>(k.dtype));
    mix(h, std::hash<int64_t>{}(k.T));
    mix(h, std::hash<int64_t>{}(k.Ktot));
    mix(h, std::hash<int64_t>{}(k.Hq));
    mix(h, std::hash<int64_t>{}(k.Hkv));
    mix(h, std::hash<int64_t>{}(k.D));
    mix(h, std::hash<int64_t>{}(k.q_sT));
    mix(h, std::hash<int64_t>{}(k.q_sH));
    mix(h, std::hash<int64_t>{}(k.q_sD));
    mix(h, std::hash<int64_t>{}(k.q_sB));
    mix(h, std::hash<int64_t>{}(k.k_sT));
    mix(h, std::hash<int64_t>{}(k.k_sH));
    mix(h, std::hash<int64_t>{}(k.k_sD));
    mix(h, std::hash<int64_t>{}(k.k_sB));
    mix(h, std::hash<int64_t>{}(k.v_sT));
    mix(h, std::hash<int64_t>{}(k.v_sH));
    mix(h, std::hash<int64_t>{}(k.v_sD));
    mix(h, std::hash<int64_t>{}(k.v_sB));
    mix(h, std::hash<int64_t>{}(k.o_sT));
    mix(h, std::hash<int64_t>{}(k.o_sH));
    mix(h, std::hash<int64_t>{}(k.o_sD));
    mix(h, std::hash<int64_t>{}(k.o_sB));
    mix(h, std::hash<uint32_t>{}(k.scale_bits));
    mix(h, std::hash<bool>{}(k.causal));
    mix(h, std::hash<int64_t>{}(k.left_bound));
    return h;
  }
};

// Cached value: the built graph plus its tensor handles (needed to rebuild
// the variant pack with fresh pointers on each hit) and a persistent device
// workspace buffer allocated once per key (max-needed) and reused.
struct SdpaPlanEntry {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O;
  void* workspace = nullptr;

  SdpaPlanEntry() = default;
  SdpaPlanEntry(const SdpaPlanEntry&) = delete;
  SdpaPlanEntry& operator=(const SdpaPlanEntry&) = delete;
  ~SdpaPlanEntry() {
    if (workspace != nullptr) cudaFree(workspace);
  }
};

struct SdpaPlanCache {
  std::mutex mutex;
  std::unordered_map<SdpaPlanKey, std::shared_ptr<SdpaPlanEntry>,
                     SdpaPlanKeyHash>
      plans;
  std::deque<SdpaPlanKey> order;  // insertion order for FIFO eviction
  // Keys whose graph validate()/build() deterministically failed (e.g. no
  // engine for head_dim > 128). Validate/build outcomes depend only on the
  // key (dims/strides/dtype), never on the data pointers, so repeat calls
  // with a known-bad key skip the frontend rebuild entirely and fall
  // through immediately. Workspace-malloc and execute failures are
  // transient and are deliberately NOT cached. Same FIFO capacity as the
  // plan cache.
  std::unordered_set<SdpaPlanKey, SdpaPlanKeyHash> bad;
  std::deque<SdpaPlanKey> bad_order;  // insertion order for FIFO eviction
};

SdpaPlanCache& sdpa_plan_cache() {
  static SdpaPlanCache cache;
  return cache;
}

}  // namespace

// Cumulative per-process SDPA dispatch counters for the DiT attention-call
// overhead probe: every sdpa_forward_impl call lands in exactly one bucket
// (plan-cache hit, full rebuild miss, or known-bad early-out). Lock-free
// atomics; read via sdpa_plan_cache_stats(). Numerics-neutral. Defined
// outside the anonymous namespace above so the Forward.hpp declaration
// links (internal linkage there would give each TU its own counters).
struct SdpaPlanStats {
  std::atomic<uint64_t> hits{0};
  std::atomic<uint64_t> misses{0};
  std::atomic<uint64_t> bad_skips{0};
};

namespace {
SdpaPlanStats& sdpa_plan_stats() {
  static SdpaPlanStats s;
  return s;
}
}  // namespace

void sdpa_plan_cache_stats(uint64_t& hits, uint64_t& misses,
                           uint64_t& bad_skips) {
  SdpaPlanStats& s = sdpa_plan_stats();
  hits = s.hits.load(std::memory_order_relaxed);
  misses = s.misses.load(std::memory_order_relaxed);
  bad_skips = s.bad_skips.load(std::memory_order_relaxed);
}

namespace {

// Shared SDPA graph builder/executor. causal=false omits the diagonal-mask
// attributes entirely, which per the cuDNN frontend docs yields dense
// (non-causal) attention (left_bound is ignored). causal=true bakes in the
// bottom-right causal mask (right_bound=0); left_bound >= 0 additionally
// sets diagonal_band_left_bound, i.e. sliding-window causal attention where
// each query sees a band of `left_bound` keys back. All validation, stream
// handling, implementation selection, and failure semantics are shared so
// the three mask modes cannot diverge.
bool sdpa_forward_impl(void* cached_handle, CudnnSdpaScalar dtype,
                       const void* q, const void* k, const void* v, void* out,
                       int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
                       int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD,
                       int64_t k_sT, int64_t k_sH, int64_t k_sD, int64_t v_sT,
                       int64_t v_sH, int64_t v_sD, int64_t o_sT, int64_t o_sH,
                       int64_t o_sD, float scale, bool causal,
                       int64_t left_bound, const char* node_name) {
  if (cached_handle == nullptr) return false;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr)
    return false;
  if (T <= 0 || Ktot <= 0 || Hq <= 0 || Hkv <= 0 || D <= 0) return false;
  if (Hq % Hkv != 0) return false;
  // Element strides from the Kokkos views must be positive.
  if (q_sT <= 0 || q_sH <= 0 || q_sD <= 0) return false;
  if (k_sT <= 0 || k_sH <= 0 || k_sD <= 0) return false;
  if (v_sT <= 0 || v_sH <= 0 || v_sD <= 0) return false;
  if (o_sT <= 0 || o_sH <= 0 || o_sD <= 0) return false;

  fe::DataType_t io_type = fe::DataType_t::NOT_SET;
  if (dtype == CudnnSdpaScalar::Float) {
    io_type = fe::DataType_t::FLOAT;
  } else if (dtype == CudnnSdpaScalar::BFloat16) {
    io_type = fe::DataType_t::BFLOAT16;
  } else {
    return false;
  }

  // Narrow, version-gated exclusion (cuDNN <= 9.27.0, i.e. CUDNN_VERSION <=
  // 92700): the FP32 composite implementation silently computes only the
  // first query head per KV group at s_q == 1 with GQA (Hq != Hkv) while
  // reporting success (verified: non-leading group heads come back zero).
  // MHA s_q == 1 and GQA s_q >= 2 verified correct to ~1e-7, so only this
  // combination falls back to native. Newer cuDNN releases automatically
  // retry the TPL path here.
#if CUDNN_VERSION <= 92700
  if (dtype == CudnnSdpaScalar::Float && T == 1 && Hq != Hkv) return false;
#endif

  // No fence here: the cached handle re-selects the Kokkos CUDA stream
  // every call below, so cuDNN work is ordered with surrounding Kokkos
  // kernels on that stream (all KokkosDNN kernels launch on the default
  // instance). Same justification as the cuBLAS fence removal in
  // src/KokkosDNN_Gemm.hpp.
  // Cached handle owned by the DNNHandle (created once per handle, not per
  // call). Re-select the Kokkos CUDA stream every call so ordering with
  // surrounding Kokkos kernels is preserved without extra synchronization.
  cudnnHandle_t cudnn_handle = static_cast<cudnnHandle_t>(cached_handle);
  const cudaStream_t kokkos_stream = Kokkos::Cuda().cuda_stream();
  if (cudnnSetStream(cudnn_handle, kokkos_stream) != CUDNN_STATUS_SUCCESS)
    return false;

  // Build the cache key from every parameter that affects the built plan.
  // B=1; the per-batch footprint is baked into the graph as the B stride.
  SdpaPlanKey key;
  key.dtype = dtype;
  key.T = T;
  key.Ktot = Ktot;
  key.Hq = Hq;
  key.Hkv = Hkv;
  key.D = D;
  key.q_sT = q_sT;
  key.q_sH = q_sH;
  key.q_sD = q_sD;
  key.q_sB = Hq * T * D;
  key.k_sT = k_sT;
  key.k_sH = k_sH;
  key.k_sD = k_sD;
  key.k_sB = Hkv * Ktot * D;
  key.v_sT = v_sT;
  key.v_sH = v_sH;
  key.v_sD = v_sD;
  key.v_sB = Hkv * Ktot * D;
  key.o_sT = o_sT;
  key.o_sH = o_sH;
  key.o_sD = o_sD;
  key.o_sB = Hq * T * D;
  static_assert(sizeof(float) == sizeof(uint32_t),
                "float scale key needs 32 bits");
  std::memcpy(&key.scale_bits, &scale, sizeof(scale));
  key.causal = causal;
  key.left_bound = left_bound;

  // Cache hit: skip validate/build/workspace-query/malloc entirely; just
  // re-execute the built graph with the CURRENT call's pointers (the graph
  // bakes dims/strides, not data pointers). The shared_ptr copy keeps the
  // entry (and its workspace) alive past a concurrent eviction.
  // Known-bad key: a previous validate()/build() for this exact key
  // deterministically failed, so decline immediately (same fall-through
  // the caller would take after the rebuild fails).
  {
    std::shared_ptr<SdpaPlanEntry> hit;
    bool known_bad = false;
    {
      SdpaPlanCache& cache = sdpa_plan_cache();
      std::lock_guard<std::mutex> lock(cache.mutex);
      if (cache.bad.find(key) != cache.bad.end())
        known_bad = true;
      else {
        auto it = cache.plans.find(key);
        if (it != cache.plans.end()) hit = it->second;
      }
    }
    if (known_bad) {
      sdpa_plan_stats().bad_skips.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (hit != nullptr) {
      sdpa_plan_stats().hits.fetch_add(1, std::memory_order_relaxed);
      bool hit_ok = false;
      try {
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void*>
            variant_pack = {{hit->Q, const_cast<void*>(q)},
                            {hit->K, const_cast<void*>(k)},
                            {hit->V, const_cast<void*>(v)},
                            {hit->O, out}};
        hit_ok =
            hit->graph->execute(cudnn_handle, variant_pack, hit->workspace)
                .is_good();
      } catch (...) {
        hit_ok = false;
      }
      return hit_ok;
    }
    sdpa_plan_stats().misses.fetch_add(1, std::memory_order_relaxed);
  }

  // Cache miss: build exactly as before. Any failure returns false so the
  // caller falls back to the native Kokkos kernel; nothing is cached.
  bool ok = false;
  void* workspace = nullptr;
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O;

  try {
    // B=1; cuDNN dims are [B,H,S,D], strides are the caller's Kokkos
    // (token, head, dim) element strides mapped onto (S,H,D). The B
    // stride is unused for B=1; set it to the per-batch footprint.
    const int64_t b = 1;
    const std::vector<int64_t> q_dim = {b, Hq, T, D};
    const std::vector<int64_t> q_stride = {key.q_sB, q_sH, q_sT, q_sD};
    const std::vector<int64_t> k_dim = {b, Hkv, Ktot, D};
    const std::vector<int64_t> k_stride = {key.k_sB, k_sH, k_sT, k_sD};
    const std::vector<int64_t> v_dim = {b, Hkv, Ktot, D};
    const std::vector<int64_t> v_stride = {key.v_sB, v_sH, v_sT, v_sD};
    const std::vector<int64_t> o_dim = {b, Hq, T, D};
    const std::vector<int64_t> o_stride = {key.o_sB, o_sH, o_sT, o_sD};

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    Q = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Q")
                          .set_dim(q_dim)
                          .set_stride(q_stride));
    K = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("K")
                          .set_dim(k_dim)
                          .set_stride(k_stride));
    V = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("V")
                          .set_dim(v_dim)
                          .set_stride(v_stride));
    if (Q == nullptr || K == nullptr || V == nullptr) {
      return false;
    }

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name(node_name)
                            .set_attn_scale(scale)
                            .set_generate_stats(false);
    if (causal) {
      sdpa_options.set_diagonal_alignment(
          fe::DiagonalAlignment_t::BOTTOM_RIGHT);
      sdpa_options.set_diagonal_band_right_bound(0);
      if (left_bound >= 0) {
        // Sliding window: mask columns at or before row_idx - left_bound,
        // leaving a visible band of `left_bound` keys per query. With the
        // caller's KV prefix satisfying Ktot == q_off + T, bottom-right
        // alignment reproduces the native kernel's
        // [max(0, p + 1 - window), p] mask with left_bound == window.
        sdpa_options.set_diagonal_band_left_bound(left_bound);
      }
    }
    if (io_type == fe::DataType_t::FLOAT) {
      // FP32 I/O is only served by the composite (matmul + softmax +
      // matmul) implementation; AUTO selection does not route there on
      // all setups, so request it explicitly. BF16 stays on AUTO
      // (unified flash attention).
      sdpa_options.set_implementation(
          fe::AttentionImplementation_t::COMPOSITE);
    }

    auto [Out, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    if (Out == nullptr) {
      return false;
    }
    O = Out;
    O->set_output(true).set_dim(o_dim).set_stride(o_stride);

    auto note_bad = [&]() {
      // Record the deterministic validate/build failure so repeat calls
      // skip the frontend rebuild (FIFO-evicted past capacity, mirroring
      // the plan cache). Lock-protected; allocation-free on the hot path
      // after the first miss.
      SdpaPlanCache& cache = sdpa_plan_cache();
      std::lock_guard<std::mutex> lock(cache.mutex);
      if (cache.bad.find(key) == cache.bad.end()) {
        while (cache.bad.size() >= kSdpaPlanCacheCapacity &&
               !cache.bad_order.empty()) {
          cache.bad.erase(cache.bad_order.front());
          cache.bad_order.pop_front();
        }
        cache.bad.insert(key);
        cache.bad_order.push_back(key);
      }
    };
    if (!graph->validate().is_good()) {
      note_bad();
      return false;
    }
    if (!graph->build(cudnn_handle,
                      {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})
             .is_good()) {
      note_bad();
      return false;
    }

    int64_t workspace_size = 0;
    if (!graph->get_workspace_size(workspace_size).is_good()) {
      return false;
    }
    if (workspace_size > 0) {
      if (cudaMalloc(&workspace, static_cast<size_t>(workspace_size)) !=
          cudaSuccess) {
        return false;
      }
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{Q, const_cast<void*>(q)},
                        {K, const_cast<void*>(k)},
                        {V, const_cast<void*>(v)},
                        {O, out}};
    ok = graph->execute(cudnn_handle, variant_pack, workspace).is_good();
  } catch (...) {
    ok = false;
  }

  if (!ok) {
    if (workspace != nullptr) cudaFree(workspace);
    // NOTE: cudnn_handle is cached by the DNNHandle -- never destroyed here.
    return false;
  }

  // Full success only: publish the built plan + persistent workspace so
  // repeat calls skip the rebuild. A concurrent insert of the same key wins;
  // ours is then dropped (workspace freed by the entry destructor).
  try {
    auto entry = std::make_shared<SdpaPlanEntry>();
    entry->graph = graph;
    entry->Q = Q;
    entry->K = K;
    entry->V = V;
    entry->O = O;
    entry->workspace = workspace;

    SdpaPlanCache& cache = sdpa_plan_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.plans.find(key) == cache.plans.end()) {
      while (cache.plans.size() >= kSdpaPlanCacheCapacity &&
             !cache.order.empty()) {
        auto evict_it = cache.plans.find(cache.order.front());
        if (evict_it != cache.plans.end()) cache.plans.erase(evict_it);
        cache.order.pop_front();
      }
      cache.plans.emplace(key, entry);
      cache.order.push_back(key);
      workspace = nullptr;  // owned by the entry now
    }
  } catch (...) {
    // Cache bookkeeping failed; the compute itself succeeded, so report
    // success and let the local entry clean up the workspace.
  }
  if (workspace != nullptr) cudaFree(workspace);
  // NOTE: cudnn_handle is cached by the DNNHandle -- never destroyed here.
  return ok;
}

}  // namespace

bool cudnn_sdpa_forward(void* cached_handle, CudnnSdpaScalar dtype,
                        const void* q, const void* k, const void* v, void* out,
                        int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
                        int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD,
                        int64_t k_sT, int64_t k_sH, int64_t k_sD, int64_t v_sT,
                        int64_t v_sH, int64_t v_sD, int64_t o_sT, int64_t o_sH,
                        int64_t o_sD, float scale) {
  return sdpa_forward_impl(cached_handle, dtype, q, k, v, out, T, Ktot, Hq,
                           Hkv, D, q_sT, q_sH, q_sD, k_sT, k_sH, k_sD, v_sT,
                           v_sH, v_sD, o_sT, o_sH, o_sD, scale,
                           /*causal=*/true, -1, "kokkosdnn_sdpa");
}

bool cudnn_sdpa_forward_noncausal(
    void* cached_handle, CudnnSdpaScalar dtype, const void* q, const void* k,
    const void* v, void* out, int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
    int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD, int64_t k_sT,
    int64_t k_sH, int64_t k_sD, int64_t v_sT, int64_t v_sH, int64_t v_sD,
    int64_t o_sT, int64_t o_sH, int64_t o_sD, float scale) {
  return sdpa_forward_impl(cached_handle, dtype, q, k, v, out, T, Ktot, Hq,
                           Hkv, D, q_sT, q_sH, q_sD, k_sT, k_sH, k_sD, v_sT,
                           v_sH, v_sD, o_sT, o_sH, o_sD, scale,
                           /*causal=*/false, -1, "kokkosdnn_sdpa_noncausal");
}

// Single-head FP32 math fallback for shapes the fused node cannot serve
// (head_dim > 128): S = scale * Q * K^T via cuBLAS NT sgemm, stable
// in-place row softmax (core op, same stream), O = P * V via row-major
// cuBLAS sgemm. cuBLAS runs on the Kokkos CUDA stream (selected per call
// on the cached handle, same discipline as the core GEMM ladder), so no
// fence is needed between these steps. cuBLAS sgemm takes int dims; shapes
// past INT_MAX decline to native.
bool cudnn_sdpa_math_forward_ws(void* cublas_handle, const float* q,
                                const float* k, const float* v, float* out,
                                int64_t Sq, int64_t Sk, int64_t D,
                                int64_t q_sT, int64_t k_sT, int64_t v_sT,
                                int64_t o_sT, float scale, float* ws,
                                size_t ws_elems) {
#if !defined(KOKKOSDNN_ENABLE_CUBLAS)
  (void)cublas_handle;
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)Sq;
  (void)Sk;
  (void)D;
  (void)q_sT;
  (void)k_sT;
  (void)v_sT;
  (void)o_sT;
  (void)scale;
  return false;
#else
  try {
    if (cublas_handle == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || out == nullptr)
      return false;
    if (Sq <= 0 || Sk <= 0 || D <= 0) return false;
    // Packed single-head rows: each [S,D] head matrix is contiguous, so
    // both GEMMs map onto plain row-major sgemm calls with no copies.
    if (q_sT != D || k_sT != D || v_sT != D || o_sT != D) return false;
    if (Sq > INT_MAX || Sk > INT_MAX || D > INT_MAX) return false;
    if (ws == nullptr) return false;
    // Overflow-safe capacity check for the caller-provided scores plane.
    if (Sq > 0 &&
        static_cast<size_t>(Sk) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(Sq))
      return false;
    if (ws_elems < static_cast<size_t>(Sq) * static_cast<size_t>(Sk))
      return false;
    auto h = static_cast<cublasHandle_t>(cublas_handle);
    // S = scale * Q * K^T ([Sq,D] NT [Sk,D] -> [Sq,Sk]).
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::Cuda,
                 Kokkos::MemoryUnmanaged>
        sbuf(ws, static_cast<size_t>(Sq), static_cast<size_t>(Sk));
    if (!cublas_gemm_nt_with_handle(h, static_cast<int>(Sq),
                                    static_cast<int>(Sk), static_cast<int>(D),
                                    scale, q, k, 0.0f, sbuf.data()))
      return false;
    // Stable softmax over keys, in place (same stream, no fence).
    KokkosDNN::softmax_rows<Kokkos::Cuda, float>(sbuf, sbuf,
                                                "KokkosDNN::cudnn_math_sm");
    // O = P * V ([Sq,Sk] @ [Sk,D] -> [Sq,D]) via the row-major identity
    // C_col[D,Sq] = V_col[D,Sk] * P_col[Sk,Sq] (memory-identical to the
    // row-major product, so no transpose copy is needed).
    if (!cublas_select_kokkos_stream(h)) return false;
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t st = cublasSgemm(
        h, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(D),
        static_cast<int>(Sq), static_cast<int>(Sk), &one, v,
        static_cast<int>(D), sbuf.data(), static_cast<int>(Sk), &zero, out,
        static_cast<int>(D));
    return st == CUBLAS_STATUS_SUCCESS;
  } catch (...) {
    return false;
  }
#endif
}

bool cudnn_sdpa_math_forward(void* cublas_handle, const float* q,
                             const float* k, const float* v, float* out,
                             int64_t Sq, int64_t Sk, int64_t D, int64_t q_sT,
                             int64_t k_sT, int64_t v_sT, int64_t o_sT,
                             float scale) {
#if !defined(KOKKOSDNN_ENABLE_CUBLAS)
  (void)cublas_handle;
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)Sq;
  (void)Sk;
  (void)D;
  (void)q_sT;
  (void)k_sT;
  (void)v_sT;
  (void)o_sT;
  (void)scale;
  return false;
#else
  if (Sq <= 0 || Sk <= 0) return false;
  if (Sq > 0 &&
      static_cast<size_t>(Sk) >
          std::numeric_limits<size_t>::max() / static_cast<size_t>(Sq))
    return false;
  // Write-first scores plane (both GEMMs fully overwrite it: beta=0 on
  // the first, softmax rewrites every row in place), so no zero-fill.
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::Cuda> owned(
      Kokkos::ViewAllocateWithoutInitializing("KokkosDNN::cudnn_math_s"),
      static_cast<size_t>(Sq), static_cast<size_t>(Sk));
  return cudnn_sdpa_math_forward_ws(cublas_handle, q, k, v, out, Sq, Sk, D,
                                    q_sT, k_sT, v_sT, o_sT, scale,
                                    owned.data(),
                                    static_cast<size_t>(Sq) *
                                        static_cast<size_t>(Sk));
#endif
}

bool cudnn_sdpa_forward_sliding(
    void* cached_handle, CudnnSdpaScalar dtype, const void* q, const void* k,
    const void* v, void* out, int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
    int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD, int64_t k_sT,
    int64_t k_sH, int64_t k_sD, int64_t v_sT, int64_t v_sH, int64_t v_sD,
    int64_t o_sT, int64_t o_sH, int64_t o_sD, float scale, int64_t window) {
  if (window <= 0) return false;
  return sdpa_forward_impl(cached_handle, dtype, q, k, v, out, T, Ktot, Hq,
                           Hkv, D, q_sT, q_sH, q_sD, k_sT, k_sH, k_sD, v_sT,
                           v_sH, v_sD, o_sT, o_sH, o_sD, scale,
                           /*causal=*/true, window, "kokkosdnn_sdpa_sliding");
}

}  // namespace Impl
}  // namespace KokkosDNN

#else

namespace KokkosDNN {
namespace Impl {

bool cudnn_sdpa_forward(void*, CudnnSdpaScalar, const void*, const void*,
                        const void*, void*, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, float) {
  return false;
}

bool cudnn_sdpa_forward_noncausal(void*, CudnnSdpaScalar, const void*,
                                  const void*, const void*, void*, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, float) {
  return false;
}

bool cudnn_sdpa_forward_sliding(void*, CudnnSdpaScalar, const void*,
                                const void*, const void*, void*, int64_t,
                                int64_t, int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t, int64_t,
                                int64_t, float, int64_t) {
  return false;
}

bool cudnn_sdpa_math_forward(void*, const float*, const float*, const float*,
                             float*, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int64_t, int64_t, float) {
  return false;
}

bool cudnn_sdpa_math_forward_ws(void*, const float*, const float*,
                                const float*, float*, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t, int64_t,
                                float, float*, size_t) {
  return false;
}

void sdpa_plan_cache_stats(uint64_t& hits, uint64_t& misses,
                           uint64_t& bad_skips) {
  hits = 0;
  misses = 0;
  bad_skips = 0;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
