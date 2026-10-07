#ifndef KOKKOSDNN_ATTENTION_CUDNN_HPP
#define KOKKOSDNN_ATTENTION_CUDNN_HPP

// cuDNN scaled-dot-product attention dispatch point. The native Kokkos
// GQA kernel in KokkosDNN_Attention.hpp is always available; when
// KokkosDNN_ENABLE_CUDNN and a CUDA ExecSpace are active, callers route
// here first.
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
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include "KokkosDNN_config.h"

namespace KokkosDNN {
namespace Impl {

enum class CudnnSdpaScalar : int { Float = 0, BFloat16 = 1 };

// Raw-pointer entry point, implemented in KokkosDNN_Attention_CuDNN.cpp.
// Strides are in elements for (token, head, dim) indexing, i.e. the
// Kokkos View strides: q_sT = q.stride(0), q_sH = q.stride(1), etc.
// Returns true on success, false to fall back to native. Never throws.
bool cudnn_sdpa_forward(CudnnSdpaScalar dtype, const void* q, const void* k,
                        const void* v, void* out, int64_t T, int64_t Ktot,
                        int64_t Hq, int64_t Hkv, int64_t D, int64_t q_sT,
                        int64_t q_sH, int64_t q_sD, int64_t k_sT, int64_t k_sH,
                        int64_t k_sD, int64_t v_sT, int64_t v_sH,
                        int64_t v_sD, int64_t o_sT, int64_t o_sH,
                        int64_t o_sD, float scale);

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
// Layout-agnostic template over Kokkos views. Performs the cheap shape
// checks, maps the scalar type, and forwards raw pointers/strides.
//
// cuDNN needs the head-dim axis innermost (element stride 1). Q/O are
// small ([T, Hq, D]) and often LayoutLeft on CUDA, so they are repacked
// into contiguous LayoutRight temporaries when needed. K/V are the large
// KV-cache prefix and pass through zero-copy when already D-contiguous
// (the Llama model stores caches LayoutRight); otherwise they are
// repacked as well. Any cuDNN failure still falls back to native.
template <class Scalar, class QView, class KView, class VView, class OView>
inline bool try_cudnn_sdpa_gqa(const QView& q, const KView& k, const VView& v,
                               size_t q_off, const OView& out, float scale) {
  const int64_t T   = static_cast<int64_t>(q.extent(0));
  const int64_t Hq  = static_cast<int64_t>(q.extent(1));
  const int64_t D   = static_cast<int64_t>(q.extent(2));
  const int64_t Hkv = static_cast<int64_t>(k.extent(1));
  const int64_t Ktot = static_cast<int64_t>(k.extent(0));
  if (T <= 0 || Hq <= 0 || Hkv <= 0 || D <= 0 || Ktot <= 0) return false;
  if (static_cast<int64_t>(v.extent(0)) != Ktot) return false;
  if (static_cast<int64_t>(v.extent(1)) != Hkv) return false;
  if (static_cast<int64_t>(v.extent(2)) != D) return false;
  if (static_cast<int64_t>(out.extent(0)) != T) return false;
  if (static_cast<int64_t>(out.extent(1)) != Hq) return false;
  if (static_cast<int64_t>(out.extent(2)) != D) return false;
  // GQA requires whole-group mapping, matching the native kernel's
  // hkv = (h * Hkv) / Hq integer projection.
  if (Hq % Hkv != 0) return false;
  // Bottom-right causal matches the native mask iff the KV prefix is
  // exactly the queries' visible keys.
  if (Ktot != static_cast<int64_t>(q_off) + T) return false;
  if (q.data() == nullptr || k.data() == nullptr || v.data() == nullptr ||
      out.data() == nullptr)
    return false;

  CudnnSdpaScalar dtype;
  if constexpr (std::is_same_v<Scalar, float>) {
    dtype = CudnnSdpaScalar::Float;
    // No shape exclusion here: the only known-bad combination (FP32,
    // s_q == 1 with GQA on cuDNN <= 9.27.0) is guarded inside
    // cudnn_sdpa_forward, which returns false so we fall back to native.
    // MHA s_q == 1 and GQA s_q >= 2 verified correct to ~1e-7.
  } else if constexpr (std::is_same_v<Scalar,
                                      Kokkos::Experimental::bhalf_t>) {
    dtype = CudnnSdpaScalar::BFloat16;
  } else {
    return false;
  }

  using QExec = typename QView::execution_space;
  using KExec = typename KView::execution_space;
  using VExec = typename VView::execution_space;
  using OExec = typename OView::execution_space;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, QExec> q_tmp;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, KExec> k_tmp;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, VExec> v_tmp;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, OExec> o_tmp;

  const void* qp = static_cast<const void*>(q.data());
  const void* kp = static_cast<const void*>(k.data());
  const void* vp = static_cast<const void*>(v.data());
  void* op       = static_cast<void*>(out.data());
  int64_t q_sT = static_cast<int64_t>(q.stride(0)),
          q_sH = static_cast<int64_t>(q.stride(1)),
          q_sD = static_cast<int64_t>(q.stride(2));
  int64_t k_sT = static_cast<int64_t>(k.stride(0)),
          k_sH = static_cast<int64_t>(k.stride(1)),
          k_sD = static_cast<int64_t>(k.stride(2));
  int64_t v_sT = static_cast<int64_t>(v.stride(0)),
          v_sH = static_cast<int64_t>(v.stride(1)),
          v_sD = static_cast<int64_t>(v.stride(2));
  int64_t o_sT = static_cast<int64_t>(out.stride(0)),
          o_sH = static_cast<int64_t>(out.stride(1)),
          o_sD = static_cast<int64_t>(out.stride(2));

  if (q_sD != 1) {
    q_tmp = Kokkos::View<Scalar***, Kokkos::LayoutRight, QExec>(
        "KokkosDNN::cudnn_q", static_cast<size_t>(T),
        static_cast<size_t>(Hq), static_cast<size_t>(D));
    Kokkos::deep_copy(q_tmp, q);
    qp   = static_cast<const void*>(q_tmp.data());
    q_sT = static_cast<int64_t>(q_tmp.stride(0));
    q_sH = static_cast<int64_t>(q_tmp.stride(1));
    q_sD = static_cast<int64_t>(q_tmp.stride(2));
  }
  if (k_sD != 1) {
    k_tmp = Kokkos::View<Scalar***, Kokkos::LayoutRight, KExec>(
        "KokkosDNN::cudnn_k", static_cast<size_t>(Ktot),
        static_cast<size_t>(Hkv), static_cast<size_t>(D));
    Kokkos::deep_copy(k_tmp, k);
    kp   = static_cast<const void*>(k_tmp.data());
    k_sT = static_cast<int64_t>(k_tmp.stride(0));
    k_sH = static_cast<int64_t>(k_tmp.stride(1));
    k_sD = static_cast<int64_t>(k_tmp.stride(2));
  }
  if (v_sD != 1) {
    v_tmp = Kokkos::View<Scalar***, Kokkos::LayoutRight, VExec>(
        "KokkosDNN::cudnn_v", static_cast<size_t>(Ktot),
        static_cast<size_t>(Hkv), static_cast<size_t>(D));
    Kokkos::deep_copy(v_tmp, v);
    vp   = static_cast<const void*>(v_tmp.data());
    v_sT = static_cast<int64_t>(v_tmp.stride(0));
    v_sH = static_cast<int64_t>(v_tmp.stride(1));
    v_sD = static_cast<int64_t>(v_tmp.stride(2));
  }
  const bool repack_out = (o_sD != 1);
  if (repack_out) {
    o_tmp = Kokkos::View<Scalar***, Kokkos::LayoutRight, OExec>(
        "KokkosDNN::cudnn_o", static_cast<size_t>(T),
        static_cast<size_t>(Hq), static_cast<size_t>(D));
    op   = static_cast<void*>(o_tmp.data());
    o_sT = static_cast<int64_t>(o_tmp.stride(0));
    o_sH = static_cast<int64_t>(o_tmp.stride(1));
    o_sD = static_cast<int64_t>(o_tmp.stride(2));
  }

  const bool ok =
      cudnn_sdpa_forward(dtype, qp, kp, vp, op, T, Ktot, Hq, Hkv, D, q_sT,
                         q_sH, q_sD, k_sT, k_sH, k_sD, v_sT, v_sH, v_sD, o_sT,
                         o_sH, o_sD, scale);
  if (ok && repack_out) Kokkos::deep_copy(out, o_tmp);
  return ok;
}
#endif

}  // namespace Impl
}  // namespace KokkosDNN

#endif
