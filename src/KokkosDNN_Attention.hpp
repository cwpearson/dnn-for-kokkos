#ifndef KOKKOSDNN_ATTENTION_HPP
#define KOKKOSDNN_ATTENTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#ifdef KOKKOSDNN_ENABLE_CUDNN
#include "tpl/cudnn/KokkosDNN_CuDNN.hpp"
#endif

namespace KokkosDNN {
namespace Impl {

// Streaming online-softmax GQA core shared by both public overloads below.
// See the public docs for contract; Scalar is the I/O dtype, accumulation
// is FP32, views are layout-agnostic (plain Views or subviews alike).
// sliding_window == 0 selects full causal attention; otherwise query t at
// absolute position p = q_off + t attends keys
// [max(0, p + 1 - sliding_window), p] (sliding-window causal mask).
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_gqa_streaming(const DNNHandle<ExecSpace>& handle, const QView& q,
                             const KView& k, const VView& v, size_t q_off,
                             const OView& out, float scale,
                             size_t sliding_window,
                             const std::string& label) {
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
    // The cuDNN SDPA graph bakes in a full bottom-right causal mask, so the
    // windowed path always runs the native kernel.
    if (sliding_window == 0 && handle.use_cudnn &&
        try_cudnn_sdpa_gqa<Scalar>(handle, q, k, v, q_off, out, scale))
      return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif
  const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1);
  const size_t Ktot = k.extent(0);
  if (T == 0 || Hq == 0 || D == 0) return;

  // One thread per (query token, query head); streams over keys with an
  // online softmax running state (m, l) and accumulates directly into out.
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        size_t khi = q_off + t + 1;
        if (khi > Ktot) khi = Ktot;
        size_t klo = 0;
        if (sliding_window > 0) {
          const size_t p = q_off + t;  // absolute position of query t
          const size_t start =
              (p + 1 >= sliding_window) ? (p + 1 - sliding_window) : 0;
          if (start > klo) klo = start;
        }
        const size_t hkv = (h * Hkv) / Hq;
        // Zero the output row (accumulator for the streaming update).
        for (size_t d = 0; d < D; ++d) out(t, h, d) = static_cast<Scalar>(0);
        if (khi <= klo) return;
        float m = -1e30f;  // running max; -inf equivalent (scores are >> this)
        float l = 0.0f;    // running sum of exp(s - m)
        for (size_t kk = klo; kk < khi; ++kk) {
          float s = 0;
          for (size_t d = 0; d < D; ++d)
            s += static_cast<float>(q(t, h, d)) *
                 static_cast<float>(k(kk, hkv, d));
          s *= scale;
          const float m_new = (m > s) ? m : s;
          // m == -1e30 on the first key => a = exp(-1e30) = 0, as desired.
          const float a = expf(m - m_new);
          const float g = expf(s - m_new);
          const float la = l * a;
          const float l_new = la + g;
          const float inv = 1.0f / l_new;
          const float c = la * inv;
          const float e = g * inv;
          for (size_t d = 0; d < D; ++d) {
            const float o_old = static_cast<float>(out(t, h, d));
            const float v_kd = static_cast<float>(v(kk, hkv, d));
            out(t, h, d) = static_cast<Scalar>(o_old * c + v_kd * e);
          }
          m = m_new;
          l = l_new;
        }
      });
}

}  // namespace Impl

// Grouped-query causal attention for one request (prefill or decode).
//   q     : [T, Hq, D]  queries (already RoPE'd)
//   k, v  : [Ktotal, Hkv, D] full KV cache prefix (RoPE'd keys)
//   out   : [T, Hq, D]
//   q_off : absolute position of q[0] inside the KV sequence (prefill
//           offset; 0 for full-prompt prefill, seq_len for decode step)
// Causal mask: query t attends to keys [0, q_off + t].
// GQA: head h uses kv head h * Hkv / Hq. Scale = 1/sqrt(D).
// Scalar-agnostic (float, bhalf_t); scores/softmax accumulate in FP32.
//
// Views are layout-agnostic (any Kokkos layout / subview / memory traits),
// so callers may pass a cache subview directly without repacking. Only
// extents and operator()(t/h/d) indexing are used.
//
// Memory: streaming online softmax (FlashAttention-style) over the key
// axis. No [T, Hq, K] score tensor is materialized; extra memory is O(1)
// per (t, h) thread beyond the outputs. This is what makes 128k contexts
// feasible: a full-context prefill (T=K=128k, Hq=32) would need ~2 PiB as
// a dense FP32 score tensor, while the streaming kernel needs only the
// [T, Hq, D] output.
//
// Numerics: per-key streaming rescaling
//   m_new = max(m, s), l_new = l*exp(m-m_new) + exp(s-m_new),
//   O_new = (O*l*exp(m-m_new) + V*exp(s-m_new)) / l_new,
// which is algebraically identical to softmax(QK^T*scale)*V in FP32.
// Element order matches a left-to-right softmax reference up to FP32
// rounding (verified in tests/Test_Attention.cpp against a host reference).
//
// When KokkosDNN_ENABLE_CUDNN and ExecSpace is CUDA, float and bhalf_t
// dispatch to cuDNN SDPA first (see tpl/cudnn/KokkosDNN_CuDNN.hpp for
// contract). One narrow, version-gated exception lives in
// cudnn_sdpa_forward: FP32 single-token GQA decode (T=1, Hq != Hkv) falls
// back to native on cuDNN <= 9.27.0, where the composite path computes
// only the first query head per KV group while reporting success.
// Other scalars run the native kernel. Otherwise native.
//
// Two overloads: explicit <ExecSpace, Scalar> (tests, Llama model) and a
// deduction overload taking Scalar from the query view (benchmarks). Both
// share the Impl core above so behavior cannot diverge.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_gqa_single(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa") {
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, 0, label);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_gqa_single(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, 0, label);
}

// Sliding-window causal GQA (e.g. Gemma4 sliding layers, window 512):
// same contract as attention_gqa_single, except query t at absolute
// position p = q_off + t attends keys [max(0, p + 1 - sliding_window), p]
// instead of the full prefix [0, p]. sliding_window == 0 degrades to full
// causal attention. GQA mapping, FP32 streaming numerics, and O(1) extra
// memory match the full-causal path; the windowed path always runs the
// native Kokkos kernel (no cuDNN dispatch: the cuDNN SDPA graph bakes in a
// full causal mask).
//
// Two overloads: explicit <ExecSpace, Scalar> and a deduction overload
// taking Scalar from the query view. Both share the Impl core above so
// behavior cannot diverge from attention_gqa_single.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_gqa_single_sliding(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    size_t sliding_window,
    const std::string& label = "KokkosDNN::attn_gqa_sliding") {
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, sliding_window, label);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_gqa_single_sliding(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    size_t sliding_window,
    const std::string& label = "KokkosDNN::attn_gqa_sliding") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, sliding_window, label);
}

namespace Impl {

// Streaming online-softmax bidirectional core shared by both public
// overloads below. Same structure as attention_gqa_streaming above, except
// every query attends to ALL keys (no causal truncation).
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_bidirectional_streaming(const DNNHandle<ExecSpace>& handle,
                                       const QView& q, const KView& k,
                                       const VView& v, const OView& out,
                                       float scale, const std::string& label) {
  const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1);
  const size_t Ktot = k.extent(0);
  if (T == 0 || Hq == 0 || D == 0) return;
  // scale <= 0 selects the ViT default 1/sqrt(head_dim); see public docs.
  const float eff =
      (scale > 0.0f) ? scale : 1.0f / std::sqrt(static_cast<float>(D));

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
    if (handle.use_cudnn &&
        try_cudnn_sdpa_bidir<Scalar>(handle, q, k, v, out, eff))
      return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif

  // One thread per (query token, query head); streams over the full key
  // axis with an online softmax running state (m, l) and accumulates
  // directly into out. No [T, Hq, K] score tensor is materialized.
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const size_t hkv = (h * Hkv) / Hq;
        for (size_t d = 0; d < D; ++d) out(t, h, d) = static_cast<Scalar>(0);
        if (Ktot == 0) return;
        float m = -1e30f;
        float l = 0.0f;
        for (size_t kk = 0; kk < Ktot; ++kk) {
          float s = 0;
          for (size_t d = 0; d < D; ++d)
            s += static_cast<float>(q(t, h, d)) *
                 static_cast<float>(k(kk, hkv, d));
          s *= eff;
          const float m_new = (m > s) ? m : s;
          const float a = expf(m - m_new);
          const float g = expf(s - m_new);
          const float la = l * a;
          const float l_new = la + g;
          const float inv = 1.0f / l_new;
          const float c = la * inv;
          const float e = g * inv;
          for (size_t d = 0; d < D; ++d) {
            const float o_old = static_cast<float>(out(t, h, d));
            const float v_kd = static_cast<float>(v(kk, hkv, d));
            out(t, h, d) = static_cast<Scalar>(o_old * c + v_kd * e);
          }
          m = m_new;
          l = l_new;
        }
      });
}

}  // namespace Impl

// Bidirectional (non-causal) multi-head self/cross attention for inference.
// ViT/DINOv3 encoder blocks attend over the full token sequence
// (1 CLS + 4 registers + Np patches, e.g. 201 @224px); there is no
// causality, no KV-cache, and no sampling, so this replaces the causal
// attention_gqa_single for that use case. The causal API is untouched.
//
//   q     : [T, Hq, Dh] queries (already projected, optionally RoPE'd)
//   k, v  : [K, Hkv, Dh] keys / values (K == T for ViT self-attention)
//   out   : [T, Hq, Dh]
// Bidirectional mask: every query t attends to ALL keys [0, K).
// GQA-compatible: head h uses kv head h * Hkv / Hq (== h for pure MHA such
// as ViT-B/16 with Hq == Hkv == 12, Dh == 64).
//
// Layout convention matches attention_gqa_single ([T, H, Dh] triples,
// layout-agnostic views/subviews; only extents and operator()(t/h/d) are
// used). Callers holding packed [T, D] activations with D == H*Dh may pass
// an unmanaged reinterpreted [T, H, Dh] view instead of repacking. There is
// no bias/mask parameter: ViT attention is dense with no additive bias.
// Scalar-agnostic (float, bhalf_t); scores/softmax accumulate in FP32 with
// the same per-key streaming rescaling numerics as the causal kernel.
//
// scale : softmax scale; pass <= 0 (the default) for 1/sqrt(Dh).
//
// TPL: float and bhalf_t dispatch to the non-causal cuDNN SDPA graph
// first (see tpl/cudnn/); any failure falls back to the native kernel.
// Other scalars run the native kernel. Otherwise native.
//
// Two overloads: explicit <ExecSpace, Scalar> and a deduction overload
// taking Scalar from the query view. Both share the Impl core above so
// behavior cannot diverge.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const OView& out, float scale = 0.0f,
    const std::string& label = "KokkosDNN::attn_bidir") {
  Impl::attention_bidirectional_streaming<ExecSpace, Scalar>(handle, q, k, v,
                                                            out, scale, label);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const OView& out, float scale = 0.0f,
    const std::string& label = "KokkosDNN::attn_bidir") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_bidirectional_streaming<ExecSpace, Scalar>(handle, q, k, v,
                                                            out, scale, label);
}

}  // namespace KokkosDNN

#endif
