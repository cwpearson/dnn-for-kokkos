#ifndef KOKKOSDNN_ATTENTION_HPP
#define KOKKOSDNN_ATTENTION_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_Attention_CuDNN.hpp"

namespace KokkosDNN {
namespace Impl {

// Streaming online-softmax GQA core shared by both public overloads below.
// See the public docs for contract; Scalar is the I/O dtype, accumulation
// is FP32, views are layout-agnostic (plain Views or subviews alike).
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void attention_gqa_streaming(const DNNHandle<ExecSpace>& handle, const QView& q,
                             const KView& k, const VView& v, size_t q_off,
                             const OView& out, float scale,
                             const std::string& label) {
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Scalar, float>) {
    if (handle.use_cudnn && try_cudnn_sdpa_gqa_stub()) return;
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
        size_t klen = q_off + t + 1;
        if (klen > Ktot) klen = Ktot;
        const size_t hkv = (h * Hkv) / Hq;
        // Zero the output row (accumulator for the streaming update).
        for (size_t d = 0; d < D; ++d) out(t, h, d) = static_cast<Scalar>(0);
        if (klen == 0) return;
        float m = -1e30f;  // running max; -inf equivalent (scores are >> this)
        float l = 0.0f;    // running sum of exp(s - m)
        for (size_t kk = 0; kk < klen; ++kk) {
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
// When KokkosDNN_ENABLE_CUDNN and ExecSpace is CUDA, float dispatches to
// cuDNN SDPA first (see Attention_CuDNN for contract); other scalars run
// the native kernel. Otherwise run native.
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
                                                  scale, label);
}

template <class ExecSpace, class QView, class KView, class VView, class OView>
void attention_gqa_single(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, size_t q_off, const OView& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa") {
  using Scalar = std::remove_const_t<typename QView::value_type>;
  Impl::attention_gqa_streaming<ExecSpace, Scalar>(handle, q, k, v, q_off, out,
                                                  scale, label);
}

}  // namespace KokkosDNN

#endif
