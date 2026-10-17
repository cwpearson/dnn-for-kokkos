#ifndef KOKKOSDNN_MIOPEN_FORWARD_HPP
#define KOKKOSDNN_MIOPEN_FORWARD_HPP

// Raw-pointer MIOpen SDPA entry points, implemented in
// tpl/miopen/KokkosDNN_MIOpen_SDPA.cpp.
//
// cached_handle is the DNNHandle's cached miopenHandle_t (opaque void*;
// nullptr declines and the caller falls back to native).
// Strides are in elements for (token, head, dim) indexing, i.e. the
// Kokkos View strides: q_sT = q.stride(0), q_sH = q.stride(1), etc.
// Returns true on success, false to fall back to native. Never throws.
//
// STUB STATUS: MIOpen (checked 3.5.2, miopen/miopen.h) exposes
// convolution, softmax, OpTensor, and Fusion but NO fused
// scaled-dot-product / FlashAttention graph API equivalent to
// cudnn-frontend's SDPA node — the header contains zero matches for
// "attention", "flash", "SDPA", or "scaled_dot". All three entry points
// below therefore return false by construction, so every attention op
// runs the portable native Kokkos kernel (exact, with no version-gated
// fallback to validate). Documented composition path for a future real
// TPL dispatch: hipBLAS batched GEMM for QK^T, MIOpen softmax forward for
// the row normalization, hipBLAS batched GEMM for AV. Do NOT attempt a
// fake fused kernel here.
#include <cstddef>
#include <cstdint>

#include "tpl/miopen/KokkosDNN_MIOpen_Types.hpp"

namespace KokkosDNN {
namespace Impl {

bool miopen_sdpa_forward(void* cached_handle, MiopenSdpaScalar dtype,
                         const void* q, const void* k, const void* v, void* out,
                         int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
                         int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD,
                         int64_t k_sT, int64_t k_sH, int64_t k_sD, int64_t v_sT,
                         int64_t v_sH, int64_t v_sD, int64_t o_sT, int64_t o_sH,
                         int64_t o_sD, float scale);

// Non-causal (dense) SDPA variant for bidirectional attention: same
// signature/contract as miopen_sdpa_forward, except every query attends
// to all keys. Currently a stub (returns false); see note above.
bool miopen_sdpa_forward_noncausal(
    void* cached_handle, MiopenSdpaScalar dtype, const void* q, const void* k,
    const void* v, void* out, int64_t T, int64_t Ktot, int64_t Hq, int64_t Hkv,
    int64_t D, int64_t q_sT, int64_t q_sH, int64_t q_sD, int64_t k_sT,
    int64_t k_sH, int64_t k_sD, int64_t v_sT, int64_t v_sH, int64_t v_sD,
    int64_t o_sT, int64_t o_sH, int64_t o_sD, float scale);

// Sliding-window causal variant: same contract as miopen_sdpa_forward,
// plus a bottom-right-aligned band so each query attends to at most
// `window` keys back. window <= 0 declines (returns false); the caller
// routes window == 0 to miopen_sdpa_forward instead. Currently a stub
// (returns false); see note above.
bool miopen_sdpa_forward_sliding(void* cached_handle, MiopenSdpaScalar dtype,
                                 const void* q, const void* k, const void* v,
                                 void* out, int64_t T, int64_t Ktot, int64_t Hq,
                                 int64_t Hkv, int64_t D, int64_t q_sT,
                                 int64_t q_sH, int64_t q_sD, int64_t k_sT,
                                 int64_t k_sH, int64_t k_sD, int64_t v_sT,
                                 int64_t v_sH, int64_t v_sD, int64_t o_sT,
                                 int64_t o_sH, int64_t o_sD, float scale,
                                 int64_t window);

}  // namespace Impl
}  // namespace KokkosDNN

#endif
