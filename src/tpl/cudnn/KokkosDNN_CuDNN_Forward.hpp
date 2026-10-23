// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CUDNN_FORWARD_HPP
#define KOKKOSDNN_CUDNN_FORWARD_HPP

// Raw-pointer cuDNN SDPA entry point, implemented in
// tpl/cudnn/KokkosDNN_CuDNN_SDPA.cpp.
//
// cached_handle is the DNNHandle's cached cudnnHandle_t (opaque void*;
// nullptr declines and the caller falls back to native). The callee
// re-selects the Kokkos CUDA stream on it per call, so the one cached
// handle stays ordered with surrounding Kokkos kernels instead of being
// created/destroyed per call.
// Strides are in elements for (token, head, dim) indexing, i.e. the
// Kokkos View strides: q_sT = q.stride(0), q_sH = q.stride(1), etc.
// Returns true on success, false to fall back to native. Never throws.
#include <cstddef>
#include <cstdint>

#include "tpl/cudnn/KokkosDNN_CuDNN_Types.hpp"

namespace KokkosDNN {
namespace Impl {

bool cudnn_sdpa_forward(void* cached_handle, CudnnSdpaScalar dtype,
                         const void* q, const void* k, const void* v, void* out,
                         int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
                         int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD,
                         int64_t k_sT, int64_t k_sH, int64_t k_sD, int64_t v_sT,
                         int64_t v_sH, int64_t v_sD, int64_t o_sT, int64_t o_sH,
                         int64_t o_sD, float scale);

// Non-causal (dense) SDPA variant for bidirectional attention: same
// signature/contract as cudnn_sdpa_forward, except the graph omits the
// diagonal-mask attributes so every query attends to all keys.
bool cudnn_sdpa_forward_noncausal(
    void* cached_handle, CudnnSdpaScalar dtype, const void* q, const void* k,
    const void* v, void* out, int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
    int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD, int64_t k_sT,
    int64_t k_sH, int64_t k_sD, int64_t v_sT, int64_t v_sH, int64_t v_sD,
    int64_t o_sT, int64_t o_sH, int64_t o_sD, float scale);

// Single-head FP32 math fallback for bidirectional attention, used when the
// fused cuDNN SDPA node above has no engine for the shape (cuDNN 9.x
// provides no SDPA engine for head_dim > 128 on pre-Hopper architectures;
// validate/build fail while D <= 128 succeeds). Computes the same FP32 op
// sequence as torch's SDPA math backend: S = scale * Q * K^T (cuBLAS),
// numerically-stable row softmax, O = P * V (cuBLAS).
//
// Requirements (checked inside; anything else returns false so the caller
// falls back to the native kernel): exactly one query head and one KV
// head, FP32, and packed rows (row stride == D, so each head's [S,D]
// matrix is a contiguous row-major GEMM operand). Q/K/V/O strides are in
// elements for (token, head, dim) indexing, i.e. the Kokkos View strides;
// S is a transient [Sq,Sk] scratch allocated per call. Returns true on
// success, false to fall back to native. Never throws.
bool cudnn_sdpa_math_forward(void* cublas_handle, const float* q,
                             const float* k, const float* v, float* out,
                             int64_t Sq, int64_t Sk, int64_t D, int64_t q_sT,
                             int64_t k_sT, int64_t v_sT, int64_t o_sT,
                             float scale);

// Caller-provided-workspace variant of the math fallback above: uses
// ws[0, ws_elems) (device memory, same stream) for the transient
// [Sq,Sk] scores instead of allocating per call. Requires
// ws != nullptr and ws_elems >= (size_t)Sq * (size_t)Sk; anything else
// returns false so the caller falls back (to the allocating entry or to
// native). Same GEMM/softmax sequence and numerics as above.
bool cudnn_sdpa_math_forward_ws(void* cublas_handle, const float* q,
                                const float* k, const float* v, float* out,
                                int64_t Sq, int64_t Sk, int64_t D,
                                int64_t q_sT, int64_t k_sT, int64_t v_sT,
                                int64_t o_sT, float scale, float* ws,
                                size_t ws_elems);

// Sliding-window causal variant: same contract as cudnn_sdpa_forward, plus
// a bottom-right-aligned diagonal band with left_bound == window, so each
// query attends to at most `window` keys back (mask columns at or before
// row_idx - window). window <= 0 declines (returns false); the caller
// routes window == 0 to cudnn_sdpa_forward instead.
bool cudnn_sdpa_forward_sliding(void* cached_handle, CudnnSdpaScalar dtype,
                                const void* q, const void* k, const void* v,
                                void* out, int64_t T, int64_t Ktot, int64_t Hq,
                                int64_t Hkv, int64_t D, int64_t q_sT,
                                int64_t q_sH, int64_t q_sD, int64_t k_sT,
                                int64_t k_sH, int64_t k_sD, int64_t v_sT,
                                int64_t v_sH, int64_t v_sD, int64_t o_sT,
                                int64_t o_sH, int64_t o_sD, float scale,
                                int64_t window);

// Cumulative per-process SDPA plan-cache counters (see
// KokkosDNN_CuDNN_SDPA.cpp): every sdpa_forward_impl call lands in exactly
// one bucket — cache hit, rebuild miss, or known-bad early-out. Probe-only,
// numerics-neutral; used by the DiT attention-overhead test to assert the
// cross-block plan reuse across cond-vs-neg shapes.
void sdpa_plan_cache_stats(uint64_t& hits, uint64_t& misses,
                           uint64_t& bad_skips);

}  // namespace Impl
}  // namespace KokkosDNN

#endif
