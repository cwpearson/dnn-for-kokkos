#ifndef KOKKOSDNN_MOE_HPP
#define KOKKOSDNN_MOE_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

// Sparse Mixture-of-Experts routing (gating) ops.
//
// The missing piece for MoE inference is the routing decision: per token,
// pick the top-k experts from the router logits and turn their scores into
// combination weights. The expert FFNs themselves reuse the existing
// gemm()/swiglu() ops, so no dedicated expert-GEMM kernel lives here.
//
// Canonical per-token usage with existing ops (E experts, top-k):
//   moe_route<Exec,Scalar>(logits, bias, K, norm_topk, scale, idx, w);
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

// Weighted sum of stacked per-(token, expert) outputs:
//   y[t,d] = sum_k top_w[t,k] * expert_out[t,k,d]
// expert_out [T, K, D] holds each selected expert's output for the token
// (gathered/scattered by the caller around per-expert gemm/swiglu calls),
// top_w [T, K] comes from moe_route, y [T, D]. Scalar-agnostic (weights are
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

}  // namespace KokkosDNN

#endif
