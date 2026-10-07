#ifndef KOKKOSDNN_ATTENTION_HPP
#define KOKKOSDNN_ATTENTION_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_Attention_CuDNN.hpp"

namespace KokkosDNN {

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
// When KokkosDNN_ENABLE_CUDNN and ExecSpace is CUDA, float dispatches to
// cuDNN SDPA first (see Attention_CuDNN for contract); other scalars run
// the native kernel. Otherwise run native.
template <class ExecSpace, class Scalar>
void attention_gqa_single(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, ExecSpace>& q,
    const Kokkos::View<Scalar***, ExecSpace>& k,
    const Kokkos::View<Scalar***, ExecSpace>& v, size_t q_off,
    const Kokkos::View<Scalar***, ExecSpace>& out, float scale,
    const std::string& label = "KokkosDNN::attn_gqa") {
#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Scalar, float>) {
    if (handle.use_cudnn && Impl::try_cudnn_sdpa_gqa_stub()) return;
  } else {
    (void)handle;
  }
#else
  (void)handle;
#endif
  const size_t T = q.extent(0), Hq = q.extent(1), D = q.extent(2);
  const size_t Hkv = k.extent(1);
  const size_t kv_groups = Hq / Hkv;

  // scores[t,h,kk] transient: T*Hq*Kmax floats. For PoC context lengths
  // (<= a few k tokens in tests/examples) this is acceptable; production
  // would tile over kk (FlashAttention-style) instead. FP32 regardless of
  // Scalar for softmax accuracy.
  const size_t Ktot = k.extent(0);
  Kokkos::View<float***, ExecSpace> scores("attn_scores", T, Hq, Ktot);

  Kokkos::parallel_for(
      label + "_scores",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                        {T, Hq, Ktot}),
      KOKKOS_LAMBDA(size_t t, size_t h, size_t kk) {
        size_t klen = q_off + t + 1;  // visible keys for query t
        if (kk >= klen || kk >= Ktot) {
          scores(t, h, kk) = -1e30f;
          return;
        }
        size_t hkv = (h * Hkv) / Hq;
        float acc  = 0;
        for (size_t d = 0; d < D; ++d)
          acc += static_cast<float>(q(t, h, d)) *
                 static_cast<float>(k(kk, hkv, d));
        scores(t, h, kk) = acc * scale;
      });

  // Softmax over kk with causal length klen, then weighted sum of v.
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        size_t klen = q_off + t + 1;
        if (klen > Ktot) klen = Ktot;
        float m = scores(t, h, 0);
        for (size_t kk = 1; kk < klen; ++kk)
          m = fmaxf(m, scores(t, h, kk));
        float s = 0;
        for (size_t kk = 0; kk < klen; ++kk) {
          float e = expf(scores(t, h, kk) - m);
          scores(t, h, kk) = e;
          s += e;
        }
        float inv = 1.0f / s;
        size_t hkv = (h * Hkv) / Hq;
        (void)kv_groups;
        for (size_t d = 0; d < D; ++d) {
          float acc = 0;
          for (size_t kk = 0; kk < klen; ++kk)
            acc += scores(t, h, kk) * static_cast<float>(v(kk, hkv, d));
          out(t, h, d) = static_cast<Scalar>(acc * inv);
        }
      });
}

}  // namespace KokkosDNN

#endif
