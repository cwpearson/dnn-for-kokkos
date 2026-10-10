#ifndef KOKKOSDNN_CUDNN_HPP
#define KOKKOSDNN_CUDNN_HPP

// cuDNN scaled-dot-product attention dispatch point. Included into
// src/KokkosDNN_Attention.hpp only when KOKKOSDNN_ENABLE_CUDNN is defined.
// The native Kokkos GQA kernel in KokkosDNN_Attention.hpp is always
// available; when a CUDA ExecSpace is active, callers route here first.
//
// Contract (matches the native kernel in KokkosDNN_Attention.hpp):
//   * q     : [T, Hq, D]  queries (already RoPE'd)
//   * k, v  : [Ktot, Hkv, D] full KV cache prefix (RoPE'd keys)
//   * out   : [T, Hq, D]
//   * q_off : absolute position of q[0] inside the KV sequence; query t
//             attends to keys [0, q_off + t] (causal).
//   * scale : softmax scale (1/sqrt(D)).
// Views are layout-agnostic: D-contiguous inputs pass through zero-copy
// via their Kokkos strides; anything else is repacked into contiguous
// temporaries (cheap for the small Q/O, rare for the large K/V caches).
// Hq must be a multiple of Hkv (GQA).
//
// Causal masking uses bottom-right diagonal alignment, which is exactly
// this contract when the caller passes the valid KV prefix
// (Ktot == q_off + T, as the Llama model and all tests do). Calls with
// Ktot != q_off + T fall back to the native kernel.
//
// Returns true when cuDNN ran the attention, false to run the native
// kernel. Never throws: any cuDNN build/execute failure falls back.
#include "tpl/cudnn/KokkosDNN_CuDNN_Handle.hpp"
#include "tpl/cudnn/KokkosDNN_CuDNN_Types.hpp"
#include "tpl/cudnn/KokkosDNN_CuDNN_Forward.hpp"
#include "tpl/cudnn/KokkosDNN_CuDNN_Try.hpp"

#endif
