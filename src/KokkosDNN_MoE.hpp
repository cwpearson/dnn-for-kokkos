// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MOE_HPP
#define KOKKOSDNN_MOE_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Occupancy.hpp"

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
// accumulation) and host-friendly: the T=1 decode path is a single-iteration
// RangePolicy, so Serial/CPU execution spaces work.

namespace KokkosDNN {

namespace Impl {

// Team size the routers ask for: one thread per expert when the score range
// allows it, clamped to a power of two in [32, 256]. The launcher clamps it
// again to what the backend can field (moe_route_launch).
inline int moe_route_team_size(size_t E) {
  int ts = 32;
  while ((size_t)ts < E && ts < 256) ts *= 2;
  return ts;
}

// Per-team FP32 staging for the E expert scores.
template <class ExecSpace>
using MoeScoreScratch =
    Kokkos::View<float*, typename ExecSpace::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Sigmoid helper for the LFM2-MoE router (host- and device-callable).
KOKKOS_INLINE_FUNCTION float moe_sigmoid(float s) {
  return 1.0f / (1.0f + expf(-s));
}

// Gather the K selected experts of one token in descending score order, ties
// keeping the lower expert id (the greedy order every router here has always
// used), then write the combination weights. Runs on one thread over the
// staged team scratch, so the whole selection reads shared memory.
//
// Selected entries are marked -inf in the scratch instead of being re-checked
// by id on every pass: the id scan read `top_idx` from *global* memory inside
// the innermost loop (~400 dependent global loads per token), which is what
// made the old router cost ~50-60 us per layer at batch 1. Weights are
// recomputed from the router logits at the K selected ids (K reads), exactly
// as the original per-token formulation did, so values are unchanged.
//   sigmoid_weights=true : score[e] = sigmoid(logit[e]) + bias[e] (bias for
//       selection only); weights are the raw sigmoids of the selected ids
//       (renormalized over the K when norm_topk, else raw, times routed_scale).
//   sigmoid_weights=false: score[e] = logit[e] + bias[e]; weights are
//       softmax (over the K selected when norm_topk, else global).
template <class ExecSpace, class Scalar, class ScoreView, class BiasView,
          class IdxView, class WView>
KOKKOS_INLINE_FUNCTION void moe_select_from_scores(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits,
    const BiasView& bias, bool have_bias, ScoreView score, size_t E, int K,
    const IdxView& top_idx, const WView& top_w, size_t t, bool norm_topk,
    float routed_scale, bool sigmoid_weights) {
  constexpr float NEG = -3.4028235e38f;
  // Global softmax denominator (softmax, norm_topk=false) needs every score,
  // so compute it before the selection marks entries as taken.
  float gmx = NEG, gsum = 0.0f;
  if (!sigmoid_weights && !norm_topk) {
    for (size_t e = 0; e < E; ++e)
      if (score(e) > gmx) gmx = score(e);
    for (size_t e = 0; e < E; ++e) gsum += expf(score(e) - gmx);
  }
  // -- greedy top-k (descending; ties keep the lower expert id) --
  for (int k = 0; k < K; ++k) {
    float best = NEG;
    int be     = -1;
    for (size_t e = 0; e < E; ++e) {
      const float s = score(e);
      if (s > best) {
        best = s;
        be   = static_cast<int>(e);
      }
    }
    top_idx(t, k) = be;
    if (be >= 0) score(static_cast<size_t>(be)) = NEG;
  }
  // -- weights (from the router logits at the selected ids) --
  auto logit = [&](int k) {
    const size_t i  = static_cast<size_t>(top_idx(t, k));
    const float raw = static_cast<float>(logits(t, i));
    return sigmoid_weights
               ? moe_sigmoid(raw)
               : (raw + (have_bias ? static_cast<float>(bias(i)) : 0.0f));
  };
  if (sigmoid_weights) {
    if (norm_topk) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) sum += logit(k);
      const float denom = sum + 1e-6f;
      for (int k = 0; k < K; ++k)
        top_w(t, k) = logit(k) / denom * routed_scale;
    } else {
      for (int k = 0; k < K; ++k)
        top_w(t, k) = logit(k) * routed_scale;
    }
    return;
  }
  if (norm_topk) {
    float mx = NEG;
    for (int k = 0; k < K; ++k)
      if (logit(k) > mx) mx = logit(k);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += expf(logit(k) - mx);
    for (int k = 0; k < K; ++k)
      top_w(t, k) = expf(logit(k) - mx) / sum * routed_scale;
  } else {
    for (int k = 0; k < K; ++k)
      top_w(t, k) = expf(logit(k) - gmx) / gsum * routed_scale;
  }
}

// Router kernel body: stage the E scores for one token in team scratch (one
// thread per expert) and select on one thread. Two phases because the old
// formulation re-evaluated the activation inside the O(K*E) greedy scan, and
// the scan then walked global memory one dependent load at a time: on the RTX
// 3090 one LFM2.5 route kernel cost ~60 us and the staged version ~54 us,
// i.e. ~2.8 ms of a 14 ms decode token over 38 MoE layers. Reading the
// staged scores from team scratch (shared memory on GPUs) instead of global
// memory is what actually removes the latency chain.
//
//   softmax_weights=false: score[e] = sigmoid(logit[e]), bias for selection
//                          only, weights are the raw sigmoids
//   softmax_weights=true : score[e] = logit[e] + bias, weights are softmax
template <class ExecSpace, class Scalar, bool Sigmoid>
struct MoeRouteFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  // T=1 decode is one team; `score` is per-team scratch, not a member.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> logits;
  Kokkos::View<Scalar*, ExecSpace> bias;
  Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace> top_idx;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> top_w;
  size_t E;
  int K;
  bool have_bias, norm_topk;
  float routed_scale;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const size_t t  = static_cast<size_t>(team.league_rank());
    auto lg         = logits;
    auto sc         = MoeScoreScratch<ExecSpace>(team.team_scratch(0), E);
    const size_t E_ = E;
    const bool hb   = have_bias;
    auto bs         = bias;
    if constexpr (Sigmoid) {
      // Selection score = sigmoid + bias (the bias never enters the weights).
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, E_),
                           KOKKOS_LAMBDA(size_t e) {
                             sc(e) = moe_sigmoid(static_cast<float>(lg(t, e))) +
                                     (hb ? static_cast<float>(bs(e)) : 0.0f);
                           });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, E_),
                           KOKKOS_LAMBDA(size_t e) {
                             sc(e) = static_cast<float>(lg(t, e)) +
                                     (hb ? static_cast<float>(bs(e)) : 0.0f);
                           });
    }
    team.team_barrier();
    if (team.team_rank() != 0) return;
    moe_select_from_scores(logits, bias, have_bias, sc, E_, K, top_idx, top_w,
                           t, norm_topk, routed_scale, Sigmoid);
  }
};

// Launch a router: one team per token, sized to what the backend can field.
// Host backends allow team_size 1 only, and constructing a TeamPolicy with a
// larger team throws there, so probe `team_size_max` from a team_size-1 policy
// (valid everywhere) and clamp; every loop strides on the runtime team size,
// so a clamped team stays correct. The probe is cached (see
// KokkosDNN_Occupancy.hpp): decode runs one router per MoE layer, and the
// answer only depends on the exec-space instance, the functor type and the
// per-team scratch, which are all fixed for a given model/layer width.
template <class ExecSpace, class Functor>
void moe_route_launch(const std::string& label, size_t T, size_t E,
                      const Functor& f) {
  const int shbytes = static_cast<int>(E * sizeof(float));
  const int tmax = cached_team_size_max<ExecSpace>(
      f, static_cast<std::size_t>(shbytes));
  int ts         = Impl::moe_route_team_size(E);
  if (ts > tmax) ts = tmax;
  if (ts < 1) ts = 1;
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(T), ts);
  policy.set_scratch_size(0, Kokkos::PerTeam(shbytes));
  Kokkos::parallel_for(label, policy, f);
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
// Scalar-agnostic; FP32 accumulation. Backend-portable TeamPolicy kernel, one
// team per token with the E scores staged in team scratch; see the
// MoeRouteFunctor note for why. Selection order and weight formulas are
// unchanged.
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
  if (top_idx.extent(0) != T || top_idx.extent(1) != K ||
      top_w.extent(0) != T || top_w.extent(1) != K)
    throw std::invalid_argument("moe_route: output dimension mismatch");
  if (T == 0) return;
  if (top_k <= 0 || K > E)
    throw std::invalid_argument("moe_route: need 1 <= top_k <= E");
  Impl::MoeRouteFunctor<ExecSpace, Scalar, /*Sigmoid=*/false> f{
      logits, expert_bias, top_idx, top_w, E, top_k, have_bias, norm_topk,
      routed_scale};
  Impl::moe_route_launch<ExecSpace>(label, T, E, f);
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
// Scalar-agnostic; FP32 accumulation. Same kernel shape as moe_route (T=1
// decode is one team).
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
  if (top_idx.extent(0) != T || top_idx.extent(1) != K ||
      top_w.extent(0) != T || top_w.extent(1) != K)
    throw std::invalid_argument("moe_route_sigmoid: output dimension mismatch");
  if (T == 0) return;
  if (top_k <= 0 || K > E)
    throw std::invalid_argument("moe_route_sigmoid: need 1 <= top_k <= E");
  Impl::MoeRouteFunctor<ExecSpace, Scalar, /*Sigmoid=*/true> f{
      logits, expert_bias, top_idx, top_w, E, top_k, have_bias, norm_topk,
      routed_scale};
  Impl::moe_route_launch<ExecSpace>(label, T, E, f);
}

// Weighted sum of stacked per-(token, expert) outputs:
//   y[t,d] = sum_k top_w[t,k] * expert_out[t,k,d]
// expert_out [T, K, D] holds each selected expert's output for the token
// (gathered/scattered by the caller around per-expert gemm/swiglu calls),
// top_w [T, K] comes from moe_route / moe_route_sigmoid, y [T, D].
// Scalar-agnostic (weights are float); FP32 accumulation; backend-portable
// MDRange kernel.
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
