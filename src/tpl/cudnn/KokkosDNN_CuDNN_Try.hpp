// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CUDNN_TRY_HPP
#define KOKKOSDNN_CUDNN_TRY_HPP

// Layout-agnostic cuDNN SDPA dispatch over Kokkos views. Performs the
// cheap shape checks, maps the scalar type, and forwards raw
// pointers/strides along with the handle's cached cuDNN handle (created
// once per DNNHandle).
//
// cuDNN needs the head-dim axis innermost (element stride 1). Q/O are
// small ([T, Hq, D]) and often LayoutLeft on CUDA, so they are repacked
// into contiguous LayoutRight temporaries when needed. K/V are the large
// KV-cache prefix and pass through zero-copy when already D-contiguous
// (the Llama model stores caches LayoutRight); otherwise they are
// repacked as well. Any cuDNN failure still falls back to native.
//
// The fused SDPA node has no engine for head_dim > 128, so the
// bidirectional entry point additionally tries a single-head FP32 TPL math
// composition (cuBLAS S = scale*Q*K^T, stable row-softmax, cuBLAS O = P*V)
// before giving up to native; see cudnn_sdpa_math_forward.
#include "KokkosDNN_config.h"

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <type_traits>

#include "KokkosDNN_Handle.hpp"
#include "tpl/cudnn/KokkosDNN_CuDNN_Forward.hpp"

namespace KokkosDNN {
namespace Impl {

template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
inline bool try_cudnn_sdpa_gqa_impl(const DNNHandle<Exec>& handle,
                                    const QView& q, const KView& k,
                                    const VView& v, size_t q_off,
                                    const OView& out, float scale,
                                    int64_t window) {
  // window < 0 selects full causal attention; window > 0 selects
  // sliding-window causal attention with that band width. window == 0 is
  // rejected (the sliding entry point below never forwards it; the causal
  // entry point passes -1).
  if (window == 0) return false;
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

  const bool sliding = (window > 0);
  const bool ok =
      sliding ? cudnn_sdpa_forward_sliding(
                    handle.cudnn_raw(), dtype, qp, kp, vp, op, T, Ktot, Hq,
                    Hkv, D, q_sT, q_sH, q_sD, k_sT, k_sH, k_sD, v_sT, v_sH, v_sD,
                    o_sT, o_sH, o_sD, scale, window)
              : cudnn_sdpa_forward(handle.cudnn_raw(), dtype, qp, kp, vp, op,
                                   T, Ktot, Hq, Hkv, D, q_sT, q_sH, q_sD, k_sT,
                                   k_sH, k_sD, v_sT, v_sH, v_sD, o_sT, o_sH,
                                   o_sD, scale);
  if (ok && repack_out) Kokkos::deep_copy(out, o_tmp);
  return ok;
}

// Full causal dispatch: same mask as the native kernel when
// Ktot == q_off + T (checked inside).
template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
inline bool try_cudnn_sdpa_gqa(const DNNHandle<Exec>& handle, const QView& q,
                               const KView& k, const VView& v, size_t q_off,
                               const OView& out, float scale) {
  return try_cudnn_sdpa_gqa_impl<Scalar>(handle, q, k, v, q_off, out, scale,
                                         -1);
}

// Sliding-window causal dispatch: query t at absolute position p attends
// keys [max(0, p + 1 - sliding_window), p]. sliding_window == 0 declines
// (the caller routes that to try_cudnn_sdpa_gqa); otherwise the
// bottom-right-aligned cuDNN band reproduces the native sliding mask under
// the same Ktot == q_off + T precondition. Dtype mapping (float + bhalf_t
// only), GQA checks, Ktot check, and D-contiguity repack match the causal
// path. Returns true when cuDNN ran, false to run the native kernel.
template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
inline bool try_cudnn_sdpa_gqa_sliding(const DNNHandle<Exec>& handle,
                                       const QView& q, const KView& k,
                                       const VView& v, size_t q_off,
                                       const OView& out, float scale,
                                       size_t sliding_window) {
  if (sliding_window == 0) return false;
  return try_cudnn_sdpa_gqa_impl<Scalar>(handle, q, k, v, q_off, out, scale,
                                         static_cast<int64_t>(sliding_window));
}

// Non-causal (bidirectional) SDPA dispatch over Kokkos views. Mirrors
// try_cudnn_sdpa_gqa (dtype mapping, GQA Hq%Hkv check, D-contiguity
// repack of Q/K/V/O incl. repack_out deep_copy back) but without the
// causal Ktot == q_off + T requirement: bidirectional attention attends
// to all K keys, so there is no q_off parameter.
template <class Scalar, class Exec, class QView, class KView, class VView,
          class OView>
// Optional caller-provided workspace for the single-head FP32 math
// fallback below: attn_ws[0, attn_ws_elems) (device memory, same stream)
// replaces that fallback's per-call [T,Ktot] scores allocation. Requires
// attn_ws_elems >= (size_t)T * (size_t)Ktot; a null buffer keeps the
// previous allocating behavior, while an undersized one declines the math
// path (falls through to native below). Defaulted so all existing
// callers are unaffected.
inline bool try_cudnn_sdpa_bidir(const DNNHandle<Exec>& handle, const QView& q,
                                 const KView& k, const VView& v,
                                 const OView& out, float scale,
                                 float* attn_ws = nullptr,
                                 size_t attn_ws_elems = 0) {
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
  if (q.data() == nullptr || k.data() == nullptr || v.data() == nullptr ||
      out.data() == nullptr)
    return false;

  CudnnSdpaScalar dtype;
  if constexpr (std::is_same_v<Scalar, float>) {
    dtype = CudnnSdpaScalar::Float;
    // No shape exclusion here: the only known-bad combination (FP32,
    // s_q == 1 with GQA on cuDNN <= 9.27.0) is guarded inside
    // cudnn_sdpa_forward_noncausal, which returns false so we fall back
    // to native. MHA s_q == 1 and GQA s_q >= 2 verified correct to ~1e-7
    // on the causal path; the non-causal graph shares the same
    // implementation selection.
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

  const bool ok = cudnn_sdpa_forward_noncausal(
      handle.cudnn_raw(), dtype, qp, kp, vp, op, T, Ktot, Hq, Hkv, D, q_sT,
      q_sH, q_sD, k_sT, k_sH, k_sD, v_sT, v_sH, v_sD, o_sT, o_sH, o_sD, scale);
  if (ok) {
    if (repack_out) Kokkos::deep_copy(out, o_tmp);
    return true;
  }
#if defined(KOKKOSDNN_ENABLE_CUBLAS)
  // Fused fallback: the cuDNN SDPA node has no engine for head_dim > 128
  // (graph build fails; D <= 128 succeeds), which covers single-head
  // wide-dim shapes like the VAE mid-block (H == 1, D == 384). Those take
  // this TPL math composition (cuBLAS S = scale*Q*K^T, stable row-softmax,
  // cuBLAS O = P*V -- the same FP32 op sequence as torch's SDPA math
  // backend) instead of the slow native kernel. Multi-head or strided
  // shapes the math path declines still fall through to native below.
  if constexpr (std::is_same_v<Scalar, float>) {
    if (Hq == 1 && Hkv == 1 && q_sT == D && k_sT == D && v_sT == D &&
        o_sT == D) {
      void* ch = handle.cublas_raw();
      const auto* qf = static_cast<const float*>(qp);
      const auto* kf = static_cast<const float*>(kp);
      const auto* vf = static_cast<const float*>(vp);
      auto* of = static_cast<float*>(op);
      bool math_ok = false;
      if (ch != nullptr) {
        if (attn_ws != nullptr)
          math_ok = cudnn_sdpa_math_forward_ws(
              ch, qf, kf, vf, of, T, Ktot, D, q_sT, k_sT, v_sT, o_sT, scale,
              attn_ws, attn_ws_elems);
        else
          math_ok = cudnn_sdpa_math_forward(ch, qf, kf, vf, of, T, Ktot, D,
                                            q_sT, k_sT, v_sT, o_sT, scale);
      }
      if (math_ok) {
        if (repack_out) Kokkos::deep_copy(out, o_tmp);
        return true;
      }
    }
  }
#endif
  return false;
}

}  // namespace Impl
}  // namespace KokkosDNN

#endif
#endif
