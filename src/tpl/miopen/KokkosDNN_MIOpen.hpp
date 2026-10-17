#ifndef KOKKOSDNN_MIOPEN_HPP
#define KOKKOSDNN_MIOPEN_HPP

// MIOpen scaled-dot-product attention dispatch point. Included into
// src/KokkosDNN_Attention.hpp only when KOKKOSDNN_ENABLE_MIOPEN is defined
// on a HIP build. The native Kokkos GQA kernel in KokkosDNN_Attention.hpp
// is always available; when a HIP ExecSpace is active, callers route here
// first.
//
// Contract (matches the native kernel in KokkosDNN_Attention.hpp):
//   * q     : [T, Hq, D]  queries (already RoPE'd)
//   * k, v  : [Ktot, Hkv, D] full KV cache prefix (RoPE'd keys)
//   * out   : [T, Hq, D]
//   * q_off : absolute position of q[0] inside the KV sequence; query t
//             attends to keys [0, q_off + t] (causal).
//   * scale : softmax scale (1/sqrt(D)).
// Views are layout-agnostic; Hq must be a multiple of Hkv (GQA).
// Causal masking uses bottom-right diagonal alignment, which is exactly
// this contract when the caller passes the valid KV prefix
// (Ktot == q_off + T). Calls with Ktot != q_off + T fall back to native.
//
// STUB STATUS: MIOpen (checked 3.5.2, miopen/miopen.h) exposes no fused
// SDPA/FlashAttention API, so every try_miopen_* entry point returns false
// and the native kernel runs (native-by-construction). Never throws; a
// future hipBLAS + MIOpen-softmax composition path drops into
// KokkosDNN_MIOpen_SDPA.cpp behind the same gate.
#include "tpl/miopen/KokkosDNN_MIOpen_Handle.hpp"
#include "tpl/miopen/KokkosDNN_MIOpen_Types.hpp"
#include "tpl/miopen/KokkosDNN_MIOpen_Forward.hpp"
#include "tpl/miopen/KokkosDNN_MIOpen_Try.hpp"

#endif
