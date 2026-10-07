#ifndef KOKKOSDNN_ATTENTION_CUDNN_HPP
#define KOKKOSDNN_ATTENTION_CUDNN_HPP

// cuDNN scaled-dot-product attention dispatch point. The native Kokkos
// GQA kernel in KokkosDNN_Attention.hpp is always available; when
// KokkosDNN_ENABLE_CUDNN and a CUDA ExecSpace are active, callers route
// here first. This header documents the integration contract so the
// native path stays the single source of truth until cuDNN is wired in:
//
//   * inputs: q[o,h,d], k[n,h,d], v[n,h,d] device pointers, row-major,
//     plus per-request group/seq metadata (GQA expand + causal mask +
//     softmax scale). cuDNN graph API (cudnn_frontend, cudnn 9.x)
//     expresses exactly this as an SDPA operation.
//   * fallback: return false so the caller runs the native kernel.
//     This keeps builds green on machines without <cudnn.h>.
//
// To complete the TPL path: implement try_cudnn_sdpa_gqa() with
// cudnn_frontend::graph built from Q/K/V descriptors, call
// graph.build()/execute(), and return true on success.
// NOTE: the TPL path is only attempted for float; bhalf_t (and other
// scalars) always run the native kernel (cuDNN BF16 would need
// CUDNN_DATA_BFLOAT16 tensor descriptors).
#include "KokkosDNN_config.h"

namespace KokkosDNN {
namespace Impl {

// Always-false placeholder. Real implementation goes here (see above).
inline bool try_cudnn_sdpa_gqa_stub() { return false; }

}  // namespace Impl
}  // namespace KokkosDNN

#endif
