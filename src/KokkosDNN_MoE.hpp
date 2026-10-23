// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MOE_HPP
#define KOKKOSDNN_MOE_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

// Sparse Mixture-of-Experts routing (gating) ops.
//
// Two routing activations are supported because different model families
// gate differently:
//   * softmax (moe_route): Qwen3-style. Weights are softmaxed logits
//     (bias, when present, shifts the scores inside the softmax).
//   * sigmoid (moe_route_sigmoid): LFM2-MoE-style (HF transformers
//     Lfm2MoeTopKRouter). Weights are sigmoided logits; the expert bias,
//     when present, is added to the sigmoid outputs for top-k *selection*
//     only and is NOT part of the combination weights. With
//     norm_topk=true the gathered sigmoids are renormalized (over the K
//     selected experts, denominator carries a 1e-6 epsilon exactly as in
//     HF); otherwise the raw gathered sigmoids are the weights.
// The expert FFNs themselves reuse the existing gemm()/swiglu() ops, so no
// dedicated expert-GEMM kernel lives here.
//
// Canonical per-token usage with existing ops (E experts, top-k):
//   moe_route_sigmoid<Exec,Scalar>(logits, bias, K, norm_topk, scale, idx, w);
//   for (int k = 0; k < K; ++k) {
//     // gather tokens with idx(t,k) == e into xe  (token mask / compaction)
//     gemm(handle, 1, xe, Wgate[e], 0, gate);   // per-expert SwiGLU FFN
//     gemm(handle, 1, xe, Wup[e],   0, up);
//     swiglu<Exec,Scalar>(gate, up, act);
//     gemm(handle, 1, act, Wdown[e], 0, out_e);
//     // scatter-add w(t,k) * out_e into y (see moe_expert_weighted_sum
//     // below when the per-(token, expert) outputs are stacked instead)
//   }
// All kernels are Kokkos-portable (RangePolicy/MDRange only, FP32
// accumulation) and host-friendly: the T=1 decode path is just a
// single-iteration RangePolicy, so Serial/CPU execution spaces work.

namespace KokkosDNN {
namespace Impl {

// Device fragment: top-k selection + softmax over one token row.
//   logits[t,e] (+ bias[e] when have_bias) are the router scores.
// Greedy max selection writes descending-score expert ids into top_idx(t,:).
// Weight convention (FP32, then cast on write):
//   norm_topk=true  -> softmax over the K selected scores (sums to 1);
//   norm_topk=false -> softmax over all E scores, gathered at the selected
//                      ids (sums to <= 1; e.g. Qwen3 norm_topk_prob=false).
// Either way the weights are multiplied by routed_scale.
template <class ExecSpace, class Scalar>
KOKKOS_INLINE_FUNCTION void moe_topk_softmax(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits,
    const Kokkos::View<Scalar*, ExecSpace>& bias, bool have_bias, size_t t,
    size_t E, int K,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& top_idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w,
    bool norm_topk, float routed_scale) {
  // -- pass 1: greedy top-k (descending); ties keep the lower expert id --
  for (int k = 0; k < K; ++k) {
    float best = -3.4028235e38f;
    int be     = -1;
    for (size_t e = 0; e < E; ++e) {
      bool taken = false;
      for (int j = 0; j < k; ++j)
        if (top_idx(t, j) == static_cast<int>(e)) {
          taken = true;
          break;
        }
      if (taken) continue;
      float s = static_cast<float>(logits(t, e)) +
                (have_bias ? static_cast<float>(bias(e)) : 0.0f);
      if (s > best) {
        best = s;
        be   = static_cast<int>(e);
      }
    }
    top_idx(t, k) = be;
  }
  // -- pass 2: weights (selected scores are re-read; no temp storage) --
  if (norm_topk) {
    float mx = -3.4028235e38f;
    for (int k = 0; k < K; ++k) {
      float s = static_cast<float>(logits(t, static_cast<size_t>(top_idx(t, k)))) +
                (have_bias
                     ? static_cast<float>(bias(static_cast<size_t>(top_idx(t, k))))
                     : 0.0f);
      if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      float s = static_cast<float>(logits(t, static_cast<size_t>(top_idx(t, k)))) +
                (have_bias
                     ? static_cast<float>(bias(static_cast<size_t>(top_idx(t, k))))
                     : 0.0f);
      sum += expf(s - mx);
    }
    for (int k = 0; k < K; ++k) {
      float s = static_cast<float>(logits(t, static_cast<size_t>(top_idx(t, k)))) +
                (have_bias
                     ? static_cast<float>(bias(static_cast<size_t>(top_idx(t, k))))
                     : 0.0f);
      top_w(t, k) = expf(s - mx) / sum * routed_scale;
    }
  } else {
    float mx = -3.4028235e38f;
    for (size_t e = 0; e < E; ++e) {
      float s = static_cast<float>(logits(t, e)) +
                (have_bias ? static_cast<float>(bias(e)) : 0.0f);
      if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (size_t e = 0; e < E; ++e) {
      float s = static_cast<float>(logits(t, e)) +
                (have_bias ? static_cast<float>(bias(e)) : 0.0f);
      sum += expf(s - mx);
    }
    for (int k = 0; k < K; ++k) {
      float s = static_cast<float>(logits(t, static_cast<size_t>(top_idx(t, k)))) +
                (have_bias
                     ? static_cast<float>(bias(static_cast<size_t>(top_idx(t, k))))
                     : 0.0f);
      top_w(t, k) = expf(s - mx) / sum * routed_scale;
    }
  }
}

// Sigmoid helper for the LFM2-MoE router (host- and device-callable).
KOKKOS_INLINE_FUNCTION float moe_sigmoid(float s) {
  return 1.0f / (1.0f + expf(-s));
}

// Device fragment: top-k selection + sigmoid weights over one token row.
//   score[e] = sigmoid(logits[t,e]) (+ bias[e] when have_bias) drives the
//   greedy top-k selection (descending); ties keep the lower expert id.
// Weight convention (FP32, then cast on write):
//   the combination weights are the gathered *sigmoids* (bias excluded):
//   norm_topk=true  -> sigmoids renormalized over the K selected experts,
//                      w = sig / (sum + 1e-6), matching HF Lfm2MoeTopKRouter;
//   norm_topk=false -> raw gathered sigmoids (sums to <= K).
// Either way the weights are multiplied by routed_scale.
template <class ExecSpace, class Scalar>
KOKKOS_INLINE_FUNCTION void moe_topk_sigmoid(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits,
    const Kokkos::View<Scalar*, ExecSpace>& bias, bool have_bias, size_t t,
    size_t E, int K,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& top_idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w,
    bool norm_topk, float routed_scale) {
  // -- pass 1: greedy top-k (descending); ties keep the lower expert id --
  for (int k = 0; k < K; ++k) {
    float best = -3.4028235e38f;
    int be     = -1;
    for (size_t e = 0; e < E; ++e) {
      bool taken = false;
      for (int j = 0; j < k; ++j)
        if (top_idx(t, j) == static_cast<int>(e)) {
          taken = true;
          break;
        }
      if (taken) continue;
      const float s =
          moe_sigmoid(static_cast<float>(logits(t, e))) +
          (have_bias ? static_cast<float>(bias(e)) : 0.0f);
      if (s > best) {
        best = s;
        be   = static_cast<int>(e);
      }
    }
    top_idx(t, k) = be;
  }
  // -- pass 2: weights are gathered sigmoids (bias excluded) --
  if (norm_topk) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
      sum += moe_sigmoid(static_cast<float>(
          logits(t, static_cast<size_t>(top_idx(t, k)))));
    const float denom = sum + 1e-6f;
    for (int k = 0; k < K; ++k)
      top_w(t, k) = moe_sigmoid(static_cast<float>(
                        logits(t, static_cast<size_t>(top_idx(t, k))))) /
                    denom * routed_scale;
  } else {
    for (int k = 0; k < K; ++k)
      top_w(t, k) = moe_sigmoid(static_cast<float>(
                        logits(t, static_cast<size_t>(top_idx(t, k))))) *
                    routed_scale;
  }
}

}  // namespace Impl

// MoE routing: per-token top-k expert select + softmax weights.
//   logits      [T, E] router logits (any Scalar; scored in FP32)
//   expert_bias [E]    additive bias for load-balancing (e.g. DeepSeek-V3
//               auxiliary-loss-free bias); empty (extent 0) = no bias
//   top_k       number of experts per token (1 <= top_k <= E)
//   norm_topk   true: softmax over selected K (renormalized, sums to 1);
//               false: global softmax over E gathered at top-k (sums to <= 1)
//   routed_scale multiplier on the weights (e.g. Qwen3/Gemma3 scaling)
//   top_idx     [T, top_k] output expert ids (descending score)
//   top_w       [T, top_k] float output weights (softmax [+ scale])
// Scalar-agnostic; FP32 accumulation. Backend-portable RangePolicy kernel;
// T=1 decode is the same kernel with one iteration (host-friendly).
template <class ExecSpace, class Scalar>
void moe_route(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits,
    const Kokkos::View<Scalar*, ExecSpace>& expert_bias, int top_k,
    bool norm_topk, float routed_scale,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& top_idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w,
    const std::string& label = "KokkosDNN::moe_route") {
  const size_t T = logits.extent(0), E = logits.extent(1);
  const size_t K = top_k < 0 ? 0 : static_cast<size_t>(top_k);
  const bool have_bias = (expert_bias.extent(0) != 0);
  if (have_bias && expert_bias.extent(0) != E)
    throw std::invalid_argument("moe_route: expert_bias extent must be E");
  if (top_idx.extent(0) != T || top_idx.extent(1) != K || top_w.extent(0) != T ||
      top_w.extent(1) != K)
    throw std::invalid_argument("moe_route: output dimension mismatch");
  if (T == 0) return;
  if (top_k <= 0 || K > E)
    throw std::invalid_argument("moe_route: need 1 <= top_k <= E");
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        Impl::moe_topk_softmax<ExecSpace, Scalar>(
            logits, expert_bias, have_bias, t, E, top_k, top_idx, top_w,
            norm_topk, routed_scale);
      });
}

// MoE routing with sigmoid gating (LFM2-MoE-style, HF Lfm2MoeTopKRouter):
// per-token top-k expert select + sigmoid weights.
//   logits      [T, E] router logits (any Scalar; sigmoid computed in FP32)
//   expert_bias [E]    additive bias applied to the sigmoid outputs for
//               top-k *selection* only (auxiliary-loss-free routing); the
//               combination weights below exclude it. Empty = no bias.
//   top_k       number of experts per token (1 <= top_k <= E)
//   norm_topk   true: renormalize the K gathered sigmoids (sums to ~1,
//               denominator carries the HF 1e-6 epsilon, then scaled);
//               false: raw gathered sigmoids times routed_scale
//   routed_scale multiplier on the weights (LFM2.5: 1.0)
//   top_idx     [T, top_k] output expert ids (descending score)
//   top_w       [T, top_k] float output weights (sigmoid [+ renorm/scale])
// Scalar-agnostic; FP32 accumulation. Backend-portable RangePolicy kernel;
// T=1 decode is the same kernel with one iteration (host-friendly).
template <class ExecSpace, class Scalar>
void moe_route_sigmoid(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits,
    const Kokkos::View<Scalar*, ExecSpace>& expert_bias, int top_k,
    bool norm_topk, float routed_scale,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& top_idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w,
    const std::string& label = "KokkosDNN::moe_route_sigmoid") {
  const size_t T = logits.extent(0), E = logits.extent(1);
  const size_t K = top_k < 0 ? 0 : static_cast<size_t>(top_k);
  const bool have_bias = (expert_bias.extent(0) != 0);
  if (have_bias && expert_bias.extent(0) != E)
    throw std::invalid_argument(
        "moe_route_sigmoid: expert_bias extent must be E");
  if (top_idx.extent(0) != T || top_idx.extent(1) != K || top_w.extent(0) != T ||
      top_w.extent(1) != K)
    throw std::invalid_argument("moe_route_sigmoid: output dimension mismatch");
  if (T == 0) return;
  if (top_k <= 0 || K > E)
    throw std::invalid_argument("moe_route_sigmoid: need 1 <= top_k <= E");
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T), KOKKOS_LAMBDA(size_t t) {
        Impl::moe_topk_sigmoid<ExecSpace, Scalar>(
            logits, expert_bias, have_bias, t, E, top_k, top_idx, top_w,
            norm_topk, routed_scale);
      });
}

// Weighted sum of stacked per-(token, expert) outputs:
//   y[t,d] = sum_k top_w[t,k] * expert_out[t,k,d]
// expert_out [T, K, D] holds each selected expert's output for the token
// (gathered/scattered by the caller around per-expert gemm/swiglu calls),
// top_w [T, K] comes from moe_route / moe_route_sigmoid, y [T, D]. Scalar-agnostic (weights are
// float); FP32 accumulation; backend-portable MDRange kernel.
template <class ExecSpace, class Scalar>
void moe_expert_weighted_sum(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& expert_out,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& top_w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::moe_expert_weighted_sum") {
  const size_t T = y.extent(0), D = y.extent(1), K = top_w.extent(1);
  if (expert_out.extent(0) != T || expert_out.extent(1) != K ||
      expert_out.extent(2) != D || top_w.extent(0) != T)
    throw std::invalid_argument("moe_expert_weighted_sum: dimension mismatch");
  if (T == 0 || D == 0 || K == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k)
          acc += top_w(t, k) * static_cast<float>(expert_out(t, k, d));
        y(t, d) = static_cast<Scalar>(acc);
      });
}

// Gather rows for sparse MoE dispatch: dst[n,d] = src[rows[n],d].
// src [C,D], rows [N] with 0 <= rows[n] < C, dst [N,D]. Scalar-agnostic;
// backend-portable MDRange kernel. No-op when N or D is 0.
template <class ExecSpace, class Scalar>
void moe_gather_rows(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src,
    const Kokkos::View<int*, ExecSpace>& rows,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst,
    const std::string& label = "KokkosDNN::moe_gather") {
  const size_t D = src.extent(1), N = rows.extent(0);
  if (dst.extent(0) != N || (N > 0 && dst.extent(1) != D) ||
      (N > 0 && src.extent(1) != D))
    throw std::invalid_argument("moe_gather_rows: dimension mismatch");
  if (N == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, D}),
      KOKKOS_LAMBDA(size_t n, size_t d) { dst(n, d) = src(rows(n), d); });
}

// Weighted scatter-add for sparse MoE combine:
// dst[rows[n],d] += weights[n] * src[n,d] (FP32 accumulation).
// src [N,D], rows [N], weights [N], dst [C,D]. `rows` must hold unique
// entries (true for top-k dispatch: one token routes to distinct experts,
// so it appears at most once in any single expert's list); each (n,d)
// thread then owns its (t,d) output and no atomics are needed.
// Scalar-agnostic; backend-portable MDRange kernel. No-op when N or D is 0.
template <class ExecSpace, class Scalar>
void moe_scatter_add_weighted(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src,
    const Kokkos::View<int*, ExecSpace>& rows,
    const Kokkos::View<float*, ExecSpace>& weights,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst,
    const std::string& label = "KokkosDNN::moe_scatter") {
  const size_t N = rows.extent(0), D = dst.extent(1), C = dst.extent(0);
  if (src.extent(0) != N || weights.extent(0) != N ||
      (N > 0 && src.extent(1) != D))
    throw std::invalid_argument(
        "moe_scatter_add_weighted: dimension mismatch");
  if (N == 0 || D == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, D}),
      KOKKOS_LAMBDA(size_t n, size_t d) {
        const size_t t = static_cast<size_t>(rows(n));
        dst(t, d) = static_cast<Scalar>(static_cast<float>(dst(t, d)) +
                                        weights(n) *
                                            static_cast<float>(src(n, d)));
      });
}

}  // namespace KokkosDNN

#endif
