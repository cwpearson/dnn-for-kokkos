// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_MODEL_HPP
#define KOKKOSDNN_INKLING_MODEL_HPP

// Autoregressive Inkling 2 Mini text decoder (models/inkling).
//
// Forward math follows transformers modeling_inkling2_loop.py line by
// line (the AUTHORITATIVE reference):
//   * NormedEmbedding: h = weighted-RMSNorm(E[tok]) (Inkling2LoopRMSNorm
//     WITH weight, unlike Glimmer's scaleless embed norm).
//   * Per layer (4 kinds: sliding / sliding-loop / full / full-loop):
//     sliding: xin = BlockAttentionResidual(blocks, h) [1 query; skipped
//       on layer 0] -> input_layernorm -> q/k/v linears -> STATIC
//       depthwise short-conv on k/v (stateful across the chunk, residual +
//       /sqrt(K+1) norm) -> q ScaledL2Clip, k plain L2Clip, v unweighted
//       RMSNorm -> sliding-window causal GQA with hidden-conditioned
//       relative bias (ScaledL2Clip states, no log scaling) + per-head
//       sink -> gated output proj (sigmoid gate) -> static post-attention
//       conv -> attn residual (append-at-boundary or add) ->
//       mlp-input retrieval -> post_attn RMSNorm -> MoE -> static post-mlp
//       conv -> add.
//     full: 3-query retrieval (q/k/v streams; raw h on layer 0) ->
//       q/k/v input norms -> q/k/v linears -> DYNAMIC short-conv on k/v
//       (taps = static [K,C] + per-token weight_proj(normed stream),
//       /sqrt(2K+1)) -> q LogScaledL2Clip, k L2Clip, v unweighted RMSNorm
//       -> FULL causal GQA with log-scaled rel bias + sink -> same tail.
//     loop variants: same as their base kind, but every stream passes
//       through rank-64 MessageGates mixing in the previous pass's
//       messages (retrieved per loop layer from the previous pass's block
//       stack via the per-layer [4, D] msg_query), and KV + conv states
//       are per-pass slots (cache_offset in HF).
//   * Loop driver: pre-loop layers -> pass 0 over loop layers ->
//     passes 1..P-1 restart from the pre-loop states with messages from
//     the previous pass (message stack = previous pass's block rows after
//     stripping the first Bpre+1 rows + previous hidden) -> post-loop
//     layers -> final single-query retrieval -> output RMSNorm -> LM head
//     (first unpadded_vocab rows) -> final softcap 30.
//   * MoE: router gemm -> sigmoid scores + gate bias (selection only) ->
//     top-k, renorm + route_scale -> sparse dispatch (host grouping,
//     per-expert gathered fused gate_up SwiGLU with gate softcap on the
//     linear factor only + up softcap, down, weighted scatter-add in
//     FP32).
//
// Batch = 1 throughout. Chunked prefill: prompts longer than
// cfg.prefill_chunk_size are processed per chunk with KV write before
// read and stateful conv windows carried across chunks, which reproduces
// single-shot prefill token-for-token. Only the last chunk runs the head.
//
// Weight-seam contract (the sibling models/inkling Config/Weights agent
// owns the definitions; this header only forward-declares them and is
// templated on ConfigT/WeightsT so tests can substitute mocks):
//   ConfigT fields: n_layers, d_model, n_heads, n_kv_heads, head_dim,
//     d_rel, rel_extent, unpadded_vocab (+vocab_size), n_experts, top_k,
//     intermediate_size, sconv_kernel, swa_window, loop_start, loop_end,
//     loop_passes, gate_rank, attn_res_block, eps, qk_clip, rel_clip,
//     log_floor, log_alpha, gate_softcap, up_softcap, final_softcap,
//     route_scale, use_gate_bias, attn_res_normalize, local_layer_ids,
//     max_seq_len, prefill_chunk_size, eos_token; methods validate(),
//     kv_cache_bytes(scalar_bytes).
//   local_layer_ids convention: EITHER the sliding ("local") ids (20
//     entries for the real model) OR the 4 full-attention ids; full-ness
//     is resolved as contains() when the list is short (< n_layers/2),
//     else !contains().
//   WeightsT fields: tok_emb [V,D], tok_norm [D], out_w [U,D] (U ==
//     unpadded_vocab; padded GGUF rows are dropped at load, matching HF),
//     out_norm [D], out_res_q [1,D] F32, blocks[l] with: wq/wk/wv
//     [HqD or HkvD, D], wgate_attn [HqD,D], wo [D,HqD], wrel [D,H*R],
//     rel_proj [R,E] F32, wsink [H,D], q_scale [Hq,Hd] F32, rel_scale
//     [H,R] F32, k_conv/v_conv [Ckv,K] F32 (sliding), k_conv_static /
//     v_conv_static [K,Ckv] F32 + k_conv_proj/v_conv_proj [Hkv*K,D]
//     (full), post_attn_conv/post_mlp_conv [D,K] F32, attn_norm [D]
//     (sliding kinds), q_norm/k_norm/v_norm [D] (full kinds),
//     post_attn_norm [D], attn_res_q [nq,D] F32 (empty on layer-0
//     sliding; nq=3 on full kinds), mlp_res_q [1,D] F32, router [E,D],
//     router_bias [E], per-expert gate_up_exps[e] [2I,D] +
//     down_exps[e] [D,I] (dense vectors, one view per expert) or
//     gate_up_exps_q[e] / down_exps_q[e] as vector<QuantWeight> per
//     expert (packed), loop_q/k/v/mlp_down [2D,R] F32 +
//     loop_q/k/v/mlp_forget + loop_q/k/v/mlp_gain [R,D] F32 + msg_query
//     [4,D] F32 (loop layers).
//   All 2-D linears (wq/wk/wv/wgate_attn/wo/wrel/wsink/router/out_w and
//   per-expert packed) may be dense Views or Gguf::QuantWeight: the
//   QuantDispatch overloads select the implementation.

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "KokkosDNN_Dense.hpp"
#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_InklingConv.hpp"
#include "KokkosDNN_InklingGates.hpp"
#include "KokkosDNN_InklingNorm.hpp"
#include "KokkosDNN_InklingRelBias.hpp"
#include "KokkosDNN_InklingResidual.hpp"
#include "KokkosDNN_MoE.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_Generate.hpp"
#include "KokkosDNN_InklingQuantDispatch.hpp"

namespace KokkosDNN {
namespace Inkling {

// Forward declarations: the sibling models/inkling agent owns Config,
// Weights and Tokenizer. This header never includes them; Model/Session
// take them as template parameters defaulting to these names so the
// server's Session<Exec, Scalar[, StoreSpace]> instantiation resolves to
// the real types while tests substitute mocks.
struct Config;
template <class ExecSpace, class Scalar>
struct Weights;
class Tokenizer;
// Defined alongside the chat template (sibling); declared here for the
// Session default stop strings.
std::vector<std::string> default_stop_strings();

namespace Impl {

// True when layer l is a full-attention layer. local_ids holds EITHER the
// sliding ("local") ids (20 entries on the real model) or the 4
// full-attention ids {5,11,17,23}: a short list names the full layers
// (contains), a long list names the sliding layers (!contains).
template <class Ids>
bool is_full_layer(const Ids& local_ids, int n_layers, int l) {
  bool contains = false;
  for (size_t i = 0; i < local_ids.size(); ++i) {
    if (local_ids[i] == l) {
      contains = true;
      break;
    }
  }
  if (local_ids.size() * 2 < static_cast<size_t>(n_layers)) return contains;
  return !contains;
}

// Team-parallel causal GQA worker: one team per (token, head), threads split
// the head dim. Scores are computed once per (t, h, key) into team scratch
// and shared across the d-split weighted sum, instead of being recomputed
// per (t, h, d, key) as in the legacy MDRange kernel below. At decode
// (C == 1) the legacy form leaves one thread grinding ~200k FLOP through
// dependent scalar chains (~4 ms/launch on RTX 3090); the team form is
// ~1/30th the time. The max/sum reductions reassociate vs the legacy serial
// loop, so results agree to FP reassociation (~1e-6), not bit-for-bit
// (pinned by AttentionTeamMatchesLegacy at 1e-5).
template <class ExecSpace, class QView, class KView, class VView, class RView,
          class PView, class SView, class OView>
struct InklingAttentionTeam {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  QView q;
  KView k;
  VView v;
  size_t q_off;
  size_t window;
  RView rel;
  PView proj;
  SView sink;
  float scale;
  size_t rel_extent;
  OView out;
  size_t C, H, D, N, Hkv, R, E, grp;
  size_t Kmax;  // team-scratch floats per team (>= max keys per (t,h) row)

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    // Bind members to locals: device lambdas must not capture `this`
    // (deprecated in C++20, and the build treats warnings as errors).
    const QView q_ = q;
    const KView k_ = k;
    const VView v_ = v;
    const RView rel_ = rel;
    const PView proj_ = proj;
    const SView sink_ = sink;
    const OView out_ = out;
    const size_t H_ = H, D_ = D, N_ = N;
    const size_t R_ = R, E_ = E, grp_ = grp, Kmax_ = Kmax;
    const size_t q_off_ = q_off, window_ = window;
    const float scale_ = scale;
    const size_t th = static_cast<size_t>(team.league_rank());
    const size_t t = th / H_, h = th % H_;
    const size_t hkv = h / grp_;
    const size_t qt = q_off_ + t;
    size_t klo = 0;
    if (window_ > 0 && qt + 1 > window_) klo = qt + 1 - window_;
    size_t khi = qt;
    if (khi >= N_) khi = N_ - 1;
    if (khi < klo) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
            out_(t, h, d) = static_cast<typename OView::value_type>(0);
          });
      return;
    }
    const size_t nkeys = khi - klo + 1;
    const float sk = static_cast<float>(sink_(t, h));
    // Shared max (rank-0 value consumed by every thread): scratch-resident
    // so no thread divides by its own uninitialized copy. Layout:
    // scores[Kmax], max[1]. Scores are normalized in place in pass 2, so
    // pass 3 needs no extra reciprocal.
    float* sc = static_cast<float*>(
        team.team_scratch(1).get_shmem(sizeof(float) * (Kmax_ + 1)));
    float* sc_max = sc + Kmax_;
    // Pass 1: scores in parallel over keys (each key by exactly one
    // thread, same formula as the legacy kernel), then a max-reduction.
    // The legacy kernel ran this serially, which is fine for ~11 decode
    // keys but strands 127/128 threads on thousand-key prefill rows.
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nkeys), KOKKOS_LAMBDA(size_t i) {
          const size_t kk = klo + i;
          float dot = 0;
          for (size_t d = 0; d < D_; ++d)
            dot += static_cast<float>(q_(t, h, d)) *
                   static_cast<float>(k_(kk, hkv, d));
          float s = dot * scale_;
          const long dist = static_cast<long>(qt) - static_cast<long>(kk);
          if (dist >= 0 && static_cast<size_t>(dist) < E_) {
            float b = 0;
            for (size_t r = 0; r < R_; ++r)
              b += static_cast<float>(rel_(t, h, r)) *
                   proj_(r, static_cast<size_t>(dist));
            s += b;
          }
          sc[i] = s;
        });
    {
      float m = -1e30f;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, nkeys),
          KOKKOS_LAMBDA(size_t i, float& mloc) {
            if (sc[i] > mloc) mloc = sc[i];
          },
          Kokkos::Max<float>(m));
      Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
        sc_max[0] = (m > sk) ? m : sk;
        // Fold the sink logit into the max (legacy pass 1 seeds m = sk).
      });
    }
    team.team_barrier();
    const float m = sc_max[0];
    // Pass 2: exp/sum reduction, then normalize in place.
    {
      float denom = 0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, nkeys),
          KOKKOS_LAMBDA(size_t i, float& dloc) {
            const float e = Kokkos::expf(sc[i] - m);
            sc[i] = e;
            dloc += e;
          },
          denom);
      Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
        const float rden = 1.0f / (Kokkos::expf(sk - m) + denom);
        for (size_t i = 0; i < nkeys; ++i) sc[i] *= rden;
      });
    }
    team.team_barrier();
    // Pass 3: threads split d; each (t,h,d) accumulates keys in kk order
    // over the normalized staged scores (legacy pass-3 order, with the
    // 1/denom fold already applied above).
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, D_), KOKKOS_LAMBDA(size_t d) {
          float acc = 0;
          for (size_t kk = klo; kk <= khi; ++kk)
            acc += sc[kk - klo] * static_cast<float>(v_(kk, hkv, d));
          out_(t, h, d) = acc;
        });
  }
};

// Legacy single-thread-per-(t,h) form: kept as the fallback when the key
// count exceeds the team-scratch budget (long-context full attention).
// Matches eager_attention_forward with position_bias + sinks + sliding
// mask. FP32 accumulation, two passes over keys (max/sum, then weighted
// sum) so no [D]-sized stack array is needed; backend-portable MDRange
// kernel.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class RView, class PView, class SView, class OView>
void inkling_attention_forward_legacy(
    const QView& q, const KView& k, const VView& v, size_t q_off,
    size_t window, const RView& rel, const PView& proj, const SView& sink,
    float scale, size_t rel_extent, const OView& out,
    const std::string& label = "KokkosDNN::Inkling::attention") {
  const size_t C = q.extent(0), H = q.extent(1), D = q.extent(2);
  const size_t N = k.extent(0), Hkv = k.extent(1);
  const size_t R = rel.extent(2), E = proj.extent(1);
  if (v.extent(0) != N || v.extent(1) != Hkv || v.extent(2) != D ||
      k.extent(2) != D || rel.extent(0) != C || rel.extent(1) != H ||
      proj.extent(0) != R || sink.extent(0) != C || sink.extent(1) != H ||
      out.extent(0) != C || out.extent(1) != H || out.extent(2) != D ||
      Hkv == 0 || H % Hkv != 0 || rel_extent != E)
    throw std::invalid_argument("inkling_attention_forward: dim mismatch");
  if (C == 0 || H == 0 || D == 0 || N == 0) return;
  const size_t grp = H / Hkv;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const size_t hkv = h / grp;
        const size_t qt  = q_off + t;
        size_t klo       = 0;
        if (window > 0 && qt + 1 > window) klo = qt + 1 - window;
        size_t khi       = qt;
        if (khi >= N) khi = N - 1;
        if (khi < klo) {
          for (size_t d = 0; d < D; ++d) out(t, h, d) = static_cast<Scalar>(0);
          return;
        }
        const float sk = static_cast<float>(sink(t, h));
        // Pass 1: max and denominator over [keys..., sink].
        float m = sk;
        for (size_t kk = klo; kk <= khi; ++kk) {
          float dot = 0;
          for (size_t d = 0; d < D; ++d)
            dot += static_cast<float>(q(t, h, d)) *
                   static_cast<float>(k(kk, hkv, d));
          float s = dot * scale;
          const long dist = static_cast<long>(qt) - static_cast<long>(kk);
          if (dist >= 0 && static_cast<size_t>(dist) < E) {
            float b = 0;
            for (size_t r = 0; r < R; ++r)
              b += static_cast<float>(rel(t, h, r)) *
                   proj(r, static_cast<size_t>(dist));
            s += b;
          }
          if (s > m) m = s;
        }
        float denom = Kokkos::expf(sk - m);
        for (size_t kk = klo; kk <= khi; ++kk) {
          float dot = 0;
          for (size_t d = 0; d < D; ++d)
            dot += static_cast<float>(q(t, h, d)) *
                   static_cast<float>(k(kk, hkv, d));
          float s = dot * scale;
          const long dist = static_cast<long>(qt) - static_cast<long>(kk);
          if (dist >= 0 && static_cast<size_t>(dist) < E) {
            float b = 0;
            for (size_t r = 0; r < R; ++r)
              b += static_cast<float>(rel(t, h, r)) *
                   proj(r, static_cast<size_t>(dist));
            s += b;
          }
          denom += Kokkos::expf(s - m);
        }
        // Pass 2: weighted sum (sink contributes a zero value: dropped).
        const float rden = 1.0f / denom;
        for (size_t d = 0; d < D; ++d) {
          float acc = 0;
          for (size_t kk = klo; kk <= khi; ++kk) {
            float dot = 0;
            for (size_t dd = 0; dd < D; ++dd)
              dot += static_cast<float>(q(t, h, dd)) *
                     static_cast<float>(k(kk, hkv, dd));
            float s = dot * scale;
            const long dist = static_cast<long>(qt) - static_cast<long>(kk);
            if (dist >= 0 && static_cast<size_t>(dist) < E) {
              float b = 0;
              for (size_t r = 0; r < R; ++r)
                b += static_cast<float>(rel(t, h, r)) *
                     proj(r, static_cast<size_t>(dist));
              s += b;
            }
            acc += Kokkos::expf(s - m) * static_cast<float>(v(kk, hkv, d));
          }
          out(t, h, d) = static_cast<Scalar>(acc * rden);
        }
      });
}

// Team-parallel entry point (same contract as the legacy kernel above).
// Scores for the widest (t,h) row need Kmax team-scratch floats; beyond
// kTeamScratchKeys keys the launch would exceed portable scratch budgets,
// so those long-context full-attention rows keep the legacy kernel.
// Everything else (decode, sliding prefill, short-context full) takes the
// team path. Max/sum reductions reassociate vs the legacy serial loop
// (~1e-6 agreement; the legacy kernel is still used past the key budget).
inline constexpr size_t kInklingAttnTeamScratchKeys = 8192;

template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class RView, class PView, class SView, class OView>
void inkling_attention_forward(
    const QView& q, const KView& k, const VView& v, size_t q_off,
    size_t window, const RView& rel, const PView& proj, const SView& sink,
    float scale, size_t rel_extent, const OView& out,
    const std::string& label = "KokkosDNN::Inkling::attention") {
  const size_t C = q.extent(0), H = q.extent(1), D = q.extent(2);
  const size_t N = k.extent(0), Hkv = k.extent(1);
  const size_t R = rel.extent(2), E = proj.extent(1);
  if (v.extent(0) != N || v.extent(1) != Hkv || v.extent(2) != D ||
      k.extent(2) != D || rel.extent(0) != C || rel.extent(1) != H ||
      proj.extent(0) != R || sink.extent(0) != C || sink.extent(1) != H ||
      out.extent(0) != C || out.extent(1) != H || out.extent(2) != D ||
      Hkv == 0 || H % Hkv != 0 || rel_extent != E)
    throw std::invalid_argument("inkling_attention_forward: dim mismatch");
  if (C == 0 || H == 0 || D == 0 || N == 0) return;
  const size_t grp = H / Hkv;
  // Widest key row in this launch (closed form of the khi/klo logic in the
  // worker): full rows span [0, q_off+C), sliding rows the last `window`.
  const size_t nrows = q_off + C;
  const size_t Kmax =
      (window > 0) ? std::min(window, std::min(nrows, N)) : std::min(nrows, N);
  if (Kmax == 0) {
    Kokkos::deep_copy(out, typename OView::value_type());
    return;
  }
  if (Kmax > kInklingAttnTeamScratchKeys) {
    inkling_attention_forward_legacy<ExecSpace, Scalar>(
        q, k, v, q_off, window, rel, proj, sink, scale, rel_extent, out,
        label);
    return;
  }
  int team_size = 128;
  if ((int)D < team_size) {
    team_size = 1;
    while (team_size < (int)D) team_size *= 2;
  }
  using Policy = Kokkos::TeamPolicy<ExecSpace>;
  Policy policy((int)(C * H), team_size);
  // Scores plus the shared reciprocal denominator.
  policy.set_scratch_size(
      1, Kokkos::PerTeam(sizeof(float) * (Kmax + 1)));
  InklingAttentionTeam<ExecSpace, QView, KView, VView, RView, PView, SView,
                       OView>
      f{q, k, v, q_off,    window, rel, proj, sink, scale,
        rel_extent, out, C, H,     D,   N,    Hkv, R, E, grp, Kmax};
  Kokkos::parallel_for(label, policy, f);
}

// True when the layer type carries packed (QuantWeight) MoE experts
// (gate_up_exps_q/down_exps_q as vector<QuantWeight>); otherwise the
// dense rank-3 gate_up_exps/down_exps views are used. Lets the Model MoE
// path compile against either Weights layout.
template <class LayerT>
concept HasPackedExperts = requires(const LayerT& l) {
  l.gate_up_exps_q;
  l.down_exps_q;
};

// Per-expert SwiGLU application. wgu_e is one expert's fused gate_up
// ([2I, D] dense view or QuantWeight), wd_e its down ([D, I] or
// QuantWeight). gathered [n,D] -> gu = gathered @ wgu_e -> split halves
// (via exact copies into the packed gate/up carves) -> softcapped SwiGLU
// -> dn = act @ wd_e. The fused gate_up GEMM runs first for both weight
// kinds (decode: quant_gemv; prefill: dequantize-once + dense gemm); the
// split is two exact deep_copies, so no strided subview ever reaches a
// LayoutRight kernel parameter.
template <class ExecSpace, class Scalar, class WGU, class WD>
void moe_expert_linears(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gathered,
    const WGU& wgu_e, const WD& wd_e,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging_gu,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& staging_d,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gu,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& act,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dn,
    float gate_softcap, float up_softcap,
    const Kokkos::View<uint8_t*, ExecSpace>& act_scratch =
        Kokkos::View<uint8_t*, ExecSpace>()) {
  const size_t n = gathered.extent(0);
  const size_t E2I = gu.extent(1);
  if (E2I % 2 != 0 || gu.extent(0) != n || gate.extent(0) != n ||
      gate.extent(1) != E2I / 2 || up.extent(0) != n ||
      up.extent(1) != E2I / 2 || act.extent(0) != n ||
      act.extent(1) != E2I / 2 || dn.extent(0) != n)
    throw std::invalid_argument("moe_expert_linears: dimension mismatch");
  const size_t I = E2I / 2;
  linear_chunk<ExecSpace, Scalar>(handle, gathered, wgu_e, staging_gu, gu,
                                  "Inkling::moe_gate_up", act_scratch);
  auto gu_lo = Kokkos::subview(gu, Kokkos::ALL, std::make_pair(size_t(0), I));
  auto gu_hi =
      Kokkos::subview(gu, Kokkos::ALL, std::make_pair(I, size_t(2) * I));
  Kokkos::deep_copy(gate, gu_lo);
  Kokkos::deep_copy(up, gu_hi);
  inkling_softcapped_swiglu<ExecSpace, Scalar>(gate, up, gate_softcap,
                                               up_softcap, act);
  linear_chunk<ExecSpace, Scalar>(handle, act, wd_e, staging_d, dn,
                                  "Inkling::moe_down", act_scratch);
}

}  // namespace Impl

// Autoregressive Inkling 2 Mini text decoder. Owns dense KV caches plus
// stateful short-conv windows and chunk-sized scratch. Template on
// ExecSpace so the same code runs on Serial/OpenMP/CUDA, on Scalar
// (float or bhalf_t), and on the Config/Weights types (real sibling
// headers in server builds, duck-typed mocks in tests).
template <class ExecSpace, class Scalar = Kokkos::Experimental::bhalf_t,
          class ConfigT = Config,
          class WeightsT = Weights<ExecSpace, Scalar>>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using HostInt = typename ViewInt::host_mirror_type;

  ConfigT cfg;
  WeightsT weights;
  DNNHandle<ExecSpace> handle;
  // Dense KV caches: per layer, per loop pass (loop layers rent
  // loop_passes slots, one per pass; other layers rent 1). Each slot is
  // [max_seq_len, n_kv_heads, head_dim] K/V.
  std::vector<std::vector<View3>> k_cache, v_cache;
  // Stateful short-conv history: per layer, per pass slot, the last
  // (sconv_kernel-1) inputs of each of the 4 convs (k, v, post-attn,
  // post-mlp). k/v channels are n_kv_heads*head_dim; post convs d_model.
  struct ConvState {
    MatR k, v, pa, pm;
  };
  std::vector<std::vector<ConvState>> conv_;
  size_t seq_len = 0;

  // Persistent chunk-sized scratch (carved via subview; grows on demand,
  // never shrinks; steady decode performs zero View allocations).
  MatR h_buf_, pre_h_buf_, prev_h_buf_;
  MatR xin_buf_, sk_buf_, sv_buf_, hn_buf_;
  MatR q2_buf_, k2_buf_, v2_buf_, dyn_buf_, dynv_buf_;
  MatR rel2_buf_, sink2_buf_, gate2_buf_, o2_buf_;
  MatR moe_in_buf_, moe_logits_buf_;
  MatR gather_buf_, gu_buf_, act_buf_, down_buf_, moe_out_buf_;
  MatR moe_gate_buf_, moe_up_buf_;
  MatR msg_buf_, tmpc_buf_, tmpc_out_buf_, tmpdyn_buf_;
  MatR tmpk_buf_, tmpk_out_buf_;
  MatR blkpack_buf_, prevpack_buf_;
  // Message-gate scratch: pre-normed streams + hidden state (caller-owned
  // so the gate performs no per-call device allocation).
  MatR msg_rmsx_buf_, msg_rmsm_buf_;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> msg_h_buf_;
  View3 retr_buf_, retr1_buf_, msg_out_buf_;
  View3 blk_buf_, pre_blk_buf_;
  View3 attn_out_buf_;
  ViewInt pos_buf_;
  Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace> moe_idx_buf_;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> moe_w_buf_;
  Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> moe_idx_host_;
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> moe_w_host_;
  ViewInt moe_rows_buf_;
  HostInt moe_rows_host_;
  Kokkos::View<float*, ExecSpace> moe_wstage_buf_;
  typename Kokkos::View<float*, ExecSpace>::host_mirror_type moe_wstage_host_;
  // Exact-size dequant staging for the packed-quant path (see
  // KokkosDNN_InklingQuantDispatch.hpp): one buffer per distinct
  // projection shape. Exact sizing matters (cuBLAS row pitch == K).
  MatR deq_qkv_, deq_kv_, deq_o_, deq_rel_, deq_sink_, deq_router_, deq_dyn_;
  MatR deq_moe_gu_, deq_moe_d_;
  Kokkos::View<uint8_t*, ExecSpace> q8_act_buf_;
  View1 last_buf_, logits_buf_;
  size_t scratch_cap_ = 0;

  Model() = default;
  Model(const ConfigT& c, const WeightsT& w) : cfg(c), weights(w) {
    cfg.validate();
    kv_alloc_fresh();
    conv_alloc_fresh();
    last_buf_ = View1("last", (size_t)cfg.d_model);
    logits_buf_ = View1("logits", (size_t)cfg.unpadded_vocab);
  }

  size_t kv_cache_bytes() const { return cfg.kv_cache_bytes(sizeof(Scalar)); }

  // O(1)-ish reset: rewinds the position counter and zeroes the conv
  // history (stale conv history WOULD be observed: the first chunk's left
  // context reads history rows unconditionally). KV caches need no wipe:
  // forward() writes KV at [cpos, cpos+C) before reading prefix
  // [0, cpos+C), so stale tail data is never observed.
  void reset() {
    seq_len = 0;
    for (size_t l = 0; l < conv_.size(); ++l)
      for (size_t p = 0; p < conv_[l].size(); ++p) {
        auto& s = conv_[l][p];
        if (s.k.extent(0) != 0)
          Kokkos::deep_copy(s.k, static_cast<Scalar>(0));
        if (s.v.extent(0) != 0)
          Kokkos::deep_copy(s.v, static_cast<Scalar>(0));
        if (s.pa.extent(0) != 0)
          Kokkos::deep_copy(s.pa, static_cast<Scalar>(0));
        if (s.pm.extent(0) != 0)
          Kokkos::deep_copy(s.pm, static_cast<Scalar>(0));
      }
  }

 private:
  // NOTE (CUDA): the helpers below launch extended __host__ __device__
  // lambdas, which NVCC only accepts inside public member functions, so
  // they live in a public section. They are still implementation detail:
  // entry points are the constructor, reset(), forward(), and generate().
 public:
  size_t n_passes(size_t l) const {
    return is_loop_layer((int)l) ? (size_t)cfg.loop_passes : 1;
  }

  bool is_loop_layer(int l) const {
    return l >= cfg.loop_start && l < cfg.loop_end;
  }

  bool is_full_layer(int l) const {
    return Impl::is_full_layer(cfg.local_layer_ids, cfg.n_layers, l);
  }

  void kv_alloc_fresh() {
    const size_t MS = (size_t)cfg.max_seq_len;
    const size_t NKV = (size_t)cfg.n_kv_heads, HD = (size_t)cfg.head_dim;
    k_cache.clear();
    v_cache.clear();
    k_cache.resize((size_t)cfg.n_layers);
    v_cache.resize((size_t)cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      const size_t P = n_passes((size_t)l);
      k_cache[(size_t)l].resize(P);
      v_cache[(size_t)l].resize(P);
      for (size_t p = 0; p < P; ++p) {
        k_cache[(size_t)l][p] = View3("k_cache", MS, NKV, HD);
        v_cache[(size_t)l][p] = View3("v_cache", MS, NKV, HD);
      }
    }
  }

  void conv_alloc_fresh() {
    const size_t K = (size_t)cfg.sconv_kernel;
    const size_t Km1 = (K > 0) ? K - 1 : 0;
    const size_t Ckv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    const size_t D = (size_t)cfg.d_model;
    conv_.clear();
    conv_.resize((size_t)cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      const size_t P = n_passes((size_t)l);
      conv_[(size_t)l].resize(P);
      for (size_t p = 0; p < P; ++p) {
        auto& s = conv_[(size_t)l][p];
        s.k = MatR("conv_k", Ckv, Km1);
        s.v = MatR("conv_v", Ckv, Km1);
        s.pa = MatR("conv_pa", D, Km1);
        s.pm = MatR("conv_pm", D, Km1);
        // Zero-init: reset() also zeroes, but a fresh Model must start
        // from zero history too.
        Kokkos::deep_copy(s.k, static_cast<Scalar>(0));
        Kokkos::deep_copy(s.v, static_cast<Scalar>(0));
        Kokkos::deep_copy(s.pa, static_cast<Scalar>(0));
        Kokkos::deep_copy(s.pm, static_cast<Scalar>(0));
      }
    }
  }

  // Grow chunk-sized scratch to hold at least C rows. Exact-size growth;
  // the first, largest prefill chunk sizes everything, later T=1 decodes
  // reuse it.
  void ensure_scratch(size_t C) {
    const size_t D = (size_t)cfg.d_model;
    const size_t HqD = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
    const size_t HkvD = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    const size_t HR = (size_t)cfg.n_heads * (size_t)cfg.d_rel;
    const size_t NH = (size_t)cfg.n_heads, HD = (size_t)cfg.head_dim;
    const size_t NKV = (size_t)cfg.n_kv_heads;
    const size_t E = (size_t)cfg.n_experts, TK = (size_t)cfg.top_k;
    const size_t II = (size_t)cfg.intermediate_size;
    const size_t GR = (size_t)cfg.gate_rank;
    const size_t Bmax = (size_t)cfg.n_layers / (size_t)cfg.attn_res_block + 2;
    const size_t K = (size_t)cfg.sconv_kernel;
    if (C <= scratch_cap_ && h_buf_.extent(0) >= C &&
        h_buf_.extent(1) == D && gu_buf_.extent(1) == 2 * II &&
        moe_idx_buf_.extent(1) == TK && blk_buf_.extent(2) == D)
      return;
    h_buf_ = MatR("h", C, D);
    pre_h_buf_ = MatR("pre_h", C, D);
    prev_h_buf_ = MatR("prev_h", C, D);
    xin_buf_ = MatR("xin", C, D);
    sk_buf_ = MatR("sk", C, D);
    sv_buf_ = MatR("sv", C, D);
    hn_buf_ = MatR("hn", C, D);
    q2_buf_ = MatR("q2", C, HqD);
    k2_buf_ = MatR("k2", C, HkvD);
    v2_buf_ = MatR("v2", C, HkvD);
    dyn_buf_ = MatR("dyn", C, NKV * K);
    dynv_buf_ = MatR("dynv", C, NKV * K);
    rel2_buf_ = MatR("rel2", C, HR);
    sink2_buf_ = MatR("sink2", C, NH);
    gate2_buf_ = MatR("gate2", C, HqD);
    o2_buf_ = MatR("o2", C, D);
    moe_in_buf_ = MatR("moe_in", C, D);
    moe_logits_buf_ = MatR("moe_logits", C, E);
    gather_buf_ = MatR("gather", C, D);
    gu_buf_ = MatR("gu", C, 2 * II);
    act_buf_ = MatR("act", C, II);
    moe_gate_buf_ = MatR("moe_gate", C, II);
    moe_up_buf_ = MatR("moe_up", C, II);
    down_buf_ = MatR("down", C, D);
    moe_out_buf_ = MatR("moe_out", C, D);
    msg_buf_ = MatR("msg", C, D);
    msg_rmsx_buf_ = MatR("msg_rmsx", C, D);
    msg_rmsm_buf_ = MatR("msg_rmsm", C, D);
    msg_h_buf_ = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
        "msg_h", C, GR);
    tmpc_buf_ = MatR("tmpc", C + K, D);
    tmpc_out_buf_ = MatR("tmpc_out", C + K, D);
    tmpk_buf_ = MatR("tmpk", C + K, HkvD);
    tmpk_out_buf_ = MatR("tmpk_out", C + K, HkvD);
    tmpdyn_buf_ = MatR("tmpdyn", C + K, NKV * K);
    blkpack_buf_ = MatR("blkpack", C * Bmax, D);
    prevpack_buf_ = MatR("prevpack", C * Bmax, D);
    retr_buf_ = View3("retr", C, 3, D);
    retr1_buf_ = View3("retr1", C, 1, D);
    msg_out_buf_ = View3("msg_out", C, 4, D);
    blk_buf_ = View3("blk", C, Bmax, D);
    pre_blk_buf_ = View3("pre_blk", C, Bmax, D);
    attn_out_buf_ = View3("attn_out", C, NH, HD);
    pos_buf_ = ViewInt("pos", C);
    moe_idx_buf_ = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>(
        "moe_idx", C, TK);
    moe_w_buf_ = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
        "moe_w", C, TK);
    moe_idx_host_ = Kokkos::View<int**, Kokkos::LayoutRight,
                                 Kokkos::HostSpace>("moe_idx_host", C, TK);
    moe_w_host_ = Kokkos::View<float**, Kokkos::LayoutRight,
                               Kokkos::HostSpace>("moe_w_host", C, TK);
    moe_rows_buf_ = ViewInt("moe_rows", C);
    moe_rows_host_ = HostInt("moe_rows_host", C);
    moe_wstage_buf_ = Kokkos::View<float*, ExecSpace>("moe_wstage", C);
    moe_wstage_host_ = typename Kokkos::View<float*, ExecSpace>::
        host_mirror_type("moe_wstage_host", C);
    scratch_cap_ = C;
  }

  void ensure_deq_staging() {
    using WqT =
        decltype(std::declval<WeightsT>().blocks[std::declval<size_t>()].wq);
    if constexpr (!Impl::is_packed_weight_v<WqT>) {
      return;
    } else {
      const size_t D = (size_t)cfg.d_model;
      const size_t HqD = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
      const size_t HR = (size_t)cfg.n_heads * (size_t)cfg.d_rel;
      const size_t E = (size_t)cfg.n_experts;
      const size_t II = (size_t)cfg.intermediate_size;
      const size_t NKV = (size_t)cfg.n_kv_heads;
      const size_t K = (size_t)cfg.sconv_kernel;
      if (deq_qkv_.extent(0) != HqD || deq_qkv_.extent(1) != D)
        deq_qkv_ = MatR("deq_qkv", HqD, D);
      const size_t HKvD = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
      if (deq_kv_.extent(0) != HKvD || deq_kv_.extent(1) != D)
        deq_kv_ = MatR("deq_kv", HKvD, D);
      if (deq_o_.extent(0) != D || deq_o_.extent(1) != HqD)
        deq_o_ = MatR("deq_o", D, HqD);
      if (deq_rel_.extent(0) != HR || deq_rel_.extent(1) != D)
        deq_rel_ = MatR("deq_rel", HR, D);
      if (deq_sink_.extent(0) != (size_t)cfg.n_heads ||
          deq_sink_.extent(1) != D)
        deq_sink_ = MatR("deq_sink", (size_t)cfg.n_heads, D);
      if (deq_router_.extent(0) != E || deq_router_.extent(1) != D)
        deq_router_ = MatR("deq_router", E, D);
      if (deq_dyn_.extent(0) != NKV * K || deq_dyn_.extent(1) != D)
        deq_dyn_ = MatR("deq_dyn", NKV * K, D);
      if (deq_moe_gu_.extent(0) != 2 * II || deq_moe_gu_.extent(1) != D)
        deq_moe_gu_ = MatR("deq_moe_gu", 2 * II, D);
      if (deq_moe_d_.extent(0) != D || deq_moe_d_.extent(1) != II)
        deq_moe_d_ = MatR("deq_moe_d", D, II);
      const size_t need = Gguf::quant_gemv_int8_scratch_bytes(D);
      if (q8_act_buf_.extent(0) < need)
        q8_act_buf_ = Kokkos::View<uint8_t*, ExecSpace>("q8_act", need);
    }
  }

  // Static short-conv over a chunk with carried history:
  // tmp = [state | x] (state: [Ch, K-1] oldest-first), full-sequence conv,
  // out = last C rows, state := last K-1 rows of tmp.
  template <class WView>
  void static_conv_chunk(const MatR& x, const WView& w, MatR& state,
                         const MatR& tmp, const MatR& tmp_out,
                         const MatR& out) {
    const size_t C = x.extent(0), Ch = x.extent(1);
    const size_t K = w.extent(1);
    if (K == 0) throw std::invalid_argument("static_conv_chunk: K == 0");
    if (K == 1) {
      // No history: y = (w*x + x)/sqrt(2).
      inkling_short_conv_forward<ExecSpace, Scalar>(x, w, out);
      return;
    }
    const size_t Km1 = K - 1;
    auto tin = Kokkos::subview(tmp, std::make_pair(size_t(0), Km1 + C),
                               Kokkos::ALL);
    auto tout = Kokkos::subview(tmp_out, std::make_pair(size_t(0), Km1 + C),
                                Kokkos::ALL);
    // tin[0:Km1] = state^T (state is [Ch, Km1], tin is [rows, Ch]).
    Kokkos::parallel_for(
        "Inkling::conv_hist",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Km1, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { tin(r, c) = state(c, r); });
    Kokkos::parallel_for(
        "Inkling::conv_cur",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { tin(Km1 + r, c) = x(r, c); });
    inkling_short_conv_forward<ExecSpace, Scalar>(tin, w, tout);
    Kokkos::parallel_for(
        "Inkling::conv_take",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { out(r, c) = tout(Km1 + r, c); });
    // New history = last Km1 rows of tin.
    Kokkos::parallel_for(
        "Inkling::conv_save",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Km1, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { state(c, r) = tin(C + r, c); });
  }

  // Dynamic short-conv (full layers) over a chunk: dyn[t] is the
  // caller-computed weight_proj(raw[t]) row; tmp_dyn = [0 | dyn]; taps
  // use each token's own dyn row over the [state | x] window; out = last
  // C rows; x-state updated (dyn needs no state). Residual norm is
  // /sqrt(2K+1) (HF DynamicShortConvolution).
  template <class SView>
  void dynamic_conv_chunk(const MatR& x, const SView& static_w,
                          const MatR& dyn, size_t nheads, size_t head_dim,
                          MatR& state, const MatR& tmp, const MatR& tmp_out,
                          const MatR& tmp_dyn, const MatR& out) {
    const size_t C = x.extent(0), Ch = x.extent(1);
    const size_t K = static_w.extent(0);
    if (K == 0) throw std::invalid_argument("dynamic_conv_chunk: K == 0");
    if (K == 1) {
      inkling_dynamic_short_conv_forward<ExecSpace, Scalar>(
          x, static_w, dyn, nheads, head_dim, out);
      return;
    }
    const size_t Km1 = K - 1;
    auto tin = Kokkos::subview(tmp, std::make_pair(size_t(0), Km1 + C),
                               Kokkos::ALL);
    auto tout = Kokkos::subview(tmp_out, std::make_pair(size_t(0), Km1 + C),
                                Kokkos::ALL);
    auto tdyn = Kokkos::subview(tmp_dyn, std::make_pair(size_t(0), Km1 + C),
                                Kokkos::ALL);
    Kokkos::parallel_for(
        "Inkling::dconv_hist",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Km1, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { tin(r, c) = state(c, r); });
    Kokkos::parallel_for(
        "Inkling::dconv_cur",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { tin(Km1 + r, c) = x(r, c); });
    const size_t DK = tdyn.extent(1);
    Kokkos::parallel_for(
        "Inkling::dconv_dyn",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                         {Km1 + C, DK}),
        KOKKOS_LAMBDA(size_t r, size_t c) {
          tdyn(r, c) = (r < Km1) ? static_cast<Scalar>(0) : dyn(r - Km1, c);
        });
    inkling_dynamic_short_conv_forward<ExecSpace, Scalar>(
        tin, static_w, tdyn, nheads, head_dim, tout);
    Kokkos::parallel_for(
        "Inkling::dconv_take",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { out(r, c) = tout(Km1 + r, c); });
    Kokkos::parallel_for(
        "Inkling::dconv_save",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Km1, Ch}),
        KOKKOS_LAMBDA(size_t r, size_t c) { state(c, r) = tin(C + r, c); });
  }

  // BlockAttentionResidual retrieval with the attn_res_normalize switch.
  // blocks [C,Bmax,D] packed history, partial [C,D], query [Q,D] F32 ->
  // out [C,Q,D] (must match Q exactly). The normalized path packs the
  // first nblocks rows contiguously (the src op reads extent(1) as S, so
  // a middle-dim subview cannot be passed) and aliases the pack as an
  // exact [C,S,D].
  template <class BView, class PView, class QView, class OView>
  void attn_retrieve(const BView& blocks, size_t nblocks, const PView& partial,
                     const QView& query, bool normalize, float eps,
                     const OView& out) {
    // Hoisted by-value handles: device lambdas must not capture `this`
    // or reference params.
    BView blk_full = blocks;
    PView part = partial;
    QView qry = query;
    OView oo = out;
    const size_t C = part.extent(0), D = part.extent(1);
    const size_t Q = qry.extent(0), S = nblocks;
    if (oo.extent(0) != C || oo.extent(1) != Q || oo.extent(2) != D)
      throw std::invalid_argument("attn_retrieve: out must be [C,Q,D]");
    if (!normalize) {
      // Unnormalized variant (attn_res_normalize=false): single-head
      // softmax attention over the RAW stacked rows, scale 1/sqrt(D).
      if (S + 1 > 65)
        throw std::invalid_argument(
            "attn_retrieve: block depth exceeds worker capacity (64)");
      Kokkos::parallel_for(
          "Inkling::attn_retrieve_raw",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Q}),
          KOKKOS_LAMBDA(size_t t, size_t qq) {
            const float scale = 1.0f / Kokkos::sqrtf(static_cast<float>(D));
            float scr[65];
            float m = -1e30f;
            for (size_t s = 0; s <= S; ++s) {
              float a = 0;
              for (size_t d = 0; d < D; ++d) {
                const float v = (s < S) ? static_cast<float>(blk_full(t, s, d))
                                        : static_cast<float>(part(t, d));
                a += qry(qq, d) * v;
              }
              scr[s] = a * scale;
              if (scr[s] > m) m = scr[s];
            }
            float denom = 0;
            for (size_t s = 0; s <= S; ++s) {
              scr[s] = Kokkos::expf(scr[s] - m);
              denom += scr[s];
            }
            const float rden = 1.0f / denom;
            for (size_t d = 0; d < D; ++d) {
              float acc = 0;
              for (size_t s = 0; s <= S; ++s) {
                const float v = (s < S) ? static_cast<float>(blk_full(t, s, d))
                                        : static_cast<float>(part(t, d));
                acc += scr[s] * rden * v;
              }
              oo(t, qq, d) = static_cast<Scalar>(acc);
            }
          });
      return;
    }
    // Normalized path via the shared stateless op. Pack valid rows into
    // the persistent flat temp, then alias exactly.
    auto pack = Kokkos::subview(blkpack_buf_, std::make_pair(size_t(0), C * S),
                                Kokkos::ALL);
    if (S > 0) {
      Kokkos::parallel_for(
          "Inkling::blk_pack",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C * S, D}),
          KOKKOS_LAMBDA(size_t i, size_t d) {
            pack(i, d) = blk_full(i / S, i % S, d);
          });
    }
    if (S > 0) {
      UnmAlias3 bp(pack.data(), C, S, D);
      inkling_block_attention_residual<ExecSpace, Scalar>(bp, part, qry, eps,
                                                           oo);
    } else {
      // Degenerate empty history: the stack is [partial] only, so the
      // single-row softmax has weight 1 and the output is the partial
      // row broadcast over queries. (A zero-extent unmanaged alias is
      // not constructible on all backends, so the shared op — which
      // derives S from the view extent — cannot be used here.)
      Kokkos::parallel_for(
          "Inkling::blk_bcast",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, Q}),
          KOKKOS_LAMBDA(size_t t, size_t qq) {
            for (size_t d = 0; d < D; ++d) oo(t, qq, d) = part(t, d);
          });
    }
  }

  // MoE over [C, D] input mn -> out [C, D] (zeroed by the caller... no:
  // zeroed here). Router logits -> sigmoid moe_route_sigmoid with the
  // gate bias for selection only (or empty bias), top-k, renorm,
  // route_scale. Host grouping of (token, slot) under experts (LFM25
  // pattern), per routed expert a gathered fused gate_up SwiGLU plus a
  // weighted scatter-add in FP32.
  template <class MView>
  void moe_forward(int l, const MView& mn, const MatR& out) {
    const auto& b = weights.blocks[(size_t)l];
    const size_t C = mn.extent(0);
    const size_t E = (size_t)cfg.n_experts, TK = (size_t)cfg.top_k;
    auto logits = Kokkos::subview(moe_logits_buf_,
                                  std::make_pair(size_t(0), C), Kokkos::ALL);
    auto idx = Kokkos::subview(moe_idx_buf_, std::make_pair(size_t(0), C),
                               Kokkos::ALL);
    auto w = Kokkos::subview(moe_w_buf_, std::make_pair(size_t(0), C),
                             Kokkos::ALL);
    Impl::linear_chunk<ExecSpace, Scalar>(handle, mn, b.router, deq_router_,
                                          logits, "Inkling::router",
                                          q8_act_buf_);
    if ((bool)cfg.use_gate_bias) {
      moe_route_sigmoid<ExecSpace, Scalar>(logits, b.router_bias, (int)TK,
                                           true, cfg.route_scale, idx, w);
    } else {
      Kokkos::View<Scalar*, ExecSpace> empty("empty", 0);
      moe_route_sigmoid<ExecSpace, Scalar>(logits, empty, (int)TK, true,
                                           cfg.route_scale, idx, w);
    }
    auto idx_h = Kokkos::subview(moe_idx_host_, std::make_pair(size_t(0), C),
                                 Kokkos::ALL);
    auto w_h = Kokkos::subview(moe_w_host_, std::make_pair(size_t(0), C),
                               Kokkos::ALL);
    Kokkos::deep_copy(idx_h, idx);
    Kokkos::deep_copy(w_h, w);
    Kokkos::deep_copy(out, static_cast<Scalar>(0));
    auto gather = Kokkos::subview(gather_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
    auto gu = Kokkos::subview(gu_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto dn = Kokkos::subview(down_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto rows = Kokkos::subview(moe_rows_buf_, std::make_pair(size_t(0), C));
    auto rows_h =
        Kokkos::subview(moe_rows_host_, std::make_pair(size_t(0), C));
    auto wstage =
        Kokkos::subview(moe_wstage_buf_, std::make_pair(size_t(0), C));
    auto wstage_h =
        Kokkos::subview(moe_wstage_host_, std::make_pair(size_t(0), C));
    for (size_t e = 0; e < E; ++e) {
      size_t n = 0;
      for (size_t t = 0; t < C; ++t)
        for (size_t k = 0; k < TK; ++k)
          if ((size_t)idx_h(t, k) == e) {
            rows_h(n) = (int)t;
            wstage_h(n) = w_h(t, k);
            ++n;
          }
      if (n == 0) continue;
      auto rows_n = Kokkos::subview(rows, std::make_pair(size_t(0), n));
      auto wstage_n = Kokkos::subview(wstage, std::make_pair(size_t(0), n));
      {
        auto rh = Kokkos::subview(rows_h, std::make_pair(size_t(0), n));
        auto wh = Kokkos::subview(wstage_h, std::make_pair(size_t(0), n));
       Kokkos::deep_copy(rows_n, rh);
        Kokkos::deep_copy(wstage_n, wh);
      }
      auto g_n = Kokkos::subview(gather, std::make_pair(size_t(0), n),
                                 Kokkos::ALL);
      moe_gather_rows<ExecSpace, Scalar>(mn, rows_n, g_n);
      auto gu_e = Kokkos::subview(gu, std::make_pair(size_t(0), n),
                                  Kokkos::ALL);
      auto gate_e = Kokkos::subview(moe_gate_buf_,
                                    std::make_pair(size_t(0), n), Kokkos::ALL);
      auto up_e = Kokkos::subview(moe_up_buf_, std::make_pair(size_t(0), n),
                                  Kokkos::ALL);
      auto act_e = Kokkos::subview(act_buf_, std::make_pair(size_t(0), n),
                                   Kokkos::ALL);
      auto dn_e = Kokkos::subview(dn, std::make_pair(size_t(0), n),
                                  Kokkos::ALL);
      if constexpr (Impl::HasPackedExperts<
                        std::decay_t<decltype(b)>>) {
        Impl::moe_expert_linears<ExecSpace, Scalar>(
            handle, g_n, b.gate_up_exps_q[e], b.down_exps_q[e], deq_moe_gu_,
            deq_moe_d_, gu_e, gate_e, up_e, act_e, dn_e, cfg.gate_softcap,
            cfg.up_softcap, q8_act_buf_);
      } else {
        Impl::moe_expert_linears<ExecSpace, Scalar>(
            handle, g_n, b.gate_up_exps[e], b.down_exps[e], deq_moe_gu_,
            deq_moe_d_, gu_e, gate_e, up_e, act_e, dn_e, cfg.gate_softcap,
            cfg.up_softcap, q8_act_buf_);
      }
      moe_scatter_add_weighted<ExecSpace, Scalar>(dn_e, rows_n, wstage_n, out);
    }
  }

  // One decoder layer over chunk [C] at absolute cpos, KV/conv pass slot
  // p. h [C,D] in/out; blocks [C,Bmax,D] + nblocks in/out (appended at
  // block boundaries). msgs [C,4,D] (empty unless loop pass > 0).
  template <class HView, class BView, class MView>
  void layer_forward(int l, size_t p, const HView& h, const BView& blocks,
                     size_t& nblocks, size_t cpos, size_t C, const MView& msgs,
                     bool has_msgs) {
    // Hoisted by-value handles: device lambdas must not capture `this`
    // or reference params.
    HView hh = h;
    BView bb = blocks;
    MView mmg = msgs;
    const size_t nb0 = nblocks;
    const auto& b = weights.blocks[(size_t)l];
    const bool full = is_full_layer(l);
    const bool loop = is_loop_layer(l);
    const bool use_norm_res = (bool)cfg.attn_res_normalize;
    const float eps = cfg.eps;
    const size_t D = (size_t)cfg.d_model;
    const size_t NH = (size_t)cfg.n_heads, NKV = (size_t)cfg.n_kv_heads,
                 HD = (size_t)cfg.head_dim, DR = (size_t)cfg.d_rel;
    const size_t HqD = NH * HD;
    const size_t K = (size_t)cfg.sconv_kernel;
    const float attn_scale = 1.0f / Kokkos::sqrtf(static_cast<float>(HD));
    const size_t Erel = (size_t)cfg.rel_extent;

    auto qh = Kokkos::subview(xin_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto kh = Kokkos::subview(sk_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto vh = Kokkos::subview(sv_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto hn = Kokkos::subview(hn_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto q2 = Kokkos::subview(q2_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto k2 = Kokkos::subview(k2_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto v2 = Kokkos::subview(v2_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto rel2 = Kokkos::subview(rel2_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
    auto sink2 = Kokkos::subview(sink2_buf_, std::make_pair(size_t(0), C),
                                 Kokkos::ALL);
    auto gate2 = Kokkos::subview(gate2_buf_, std::make_pair(size_t(0), C),
                                 Kokkos::ALL);
    auto o2 = Kokkos::subview(o2_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL);
    auto pos = Kokkos::subview(pos_buf_, std::make_pair(size_t(0), C));
    // Message-gate scratch (shared across the sequential gate calls below;
    // gate outputs never alias these buffers).
    auto msg_rmsx = Kokkos::subview(msg_rmsx_buf_,
                                    std::make_pair(size_t(0), C), Kokkos::ALL);
    auto msg_rmsm = Kokkos::subview(msg_rmsm_buf_,
                                    std::make_pair(size_t(0), C), Kokkos::ALL);
    auto msg_h = Kokkos::subview(msg_h_buf_, std::make_pair(size_t(0), C),
                                 Kokkos::ALL);
    Kokkos::parallel_for(
        "Inkling::pos", Kokkos::RangePolicy<ExecSpace>(0, C),
        KOKKOS_LAMBDA(size_t t) { pos(t) = (int)(cpos + t); });
    // Head-major aliases (exact LayoutRight reinterpretations, no copies).
    using Unm3 =
        Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Unm2 =
        Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Unm3 q3(q2.data(), C, NH, HD), k3(k2.data(), C, NKV, HD),
        v3(v2.data(), C, NKV, HD), rel3(rel2.data(), C, NH, DR);
    auto a3 = Kokkos::subview(attn_out_buf_, std::make_pair(size_t(0), C),
                              Kokkos::ALL, Kokkos::ALL);

    // ---- 1. attention-input streams (retrieval + message gates) ----
    // retr1 carries Q=1 retrievals (sliding attention-input), retr3 the
    // Q=3 full retrieval; the src op requires out.extent(1) == Q exactly.
    {
      auto retr1 = Kokkos::subview(retr1_buf_, std::make_pair(size_t(0), C),
                                   Kokkos::ALL, Kokkos::ALL);
      auto retr3 = Kokkos::subview(retr_buf_, std::make_pair(size_t(0), C),
                                   Kokkos::ALL, Kokkos::ALL);
      auto moe_in = Kokkos::subview(moe_in_buf_,
                                    std::make_pair(size_t(0), C), Kokkos::ALL);
      auto msgt = Kokkos::subview(msg_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      const bool skip_attn_in =
          (!full && !loop && (l == 0 || b.attn_res_q.extent(0) == 0));
      if (skip_attn_in) {
        // Layer 0 sliding: no attention_input; streams copy h.
        Kokkos::parallel_for(
            "Inkling::xin_alias",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
            KOKKOS_LAMBDA(size_t t, size_t d) {
              qh(t, d) = hh(t, d);
              kh(t, d) = hh(t, d);
              vh(t, d) = hh(t, d);
            });
      } else if (!full) {
        if (b.attn_res_q.extent(0) != 1)
          throw std::invalid_argument(
              "Inkling::layer_forward: sliding layer needs 1 res query");
        attn_retrieve(bb, nb0, hh, b.attn_res_q, use_norm_res, eps, retr1);
        auto rq0 =
            Kokkos::subview(retr1, Kokkos::ALL, size_t(0), Kokkos::ALL);
        if (has_msgs) {
          // Pack rq + the 3 message rows (strided subviews) into packed
          // temps: message_gate needs packed rank-2 inputs and out must
          // not alias its inputs. o2/hn/msgt/moe_in are free at this
          // point (used later).
          auto mq =
              Kokkos::subview(mmg, Kokkos::ALL, size_t(0), Kokkos::ALL);
          auto mk =
              Kokkos::subview(mmg, Kokkos::ALL, size_t(1), Kokkos::ALL);
          auto mv =
              Kokkos::subview(mmg, Kokkos::ALL, size_t(2), Kokkos::ALL);
          Kokkos::deep_copy(o2, rq0);
          Kokkos::deep_copy(hn, mq);
          Kokkos::deep_copy(msgt, mk);
          Kokkos::deep_copy(moe_in, mv);
          inkling_message_gate<ExecSpace, Scalar>(
              o2, hn, b.loop_q_down, b.loop_q_forget, b.loop_q_gain, eps, qh,
              msg_rmsx, msg_rmsm, msg_h);
          inkling_message_gate<ExecSpace, Scalar>(
              o2, msgt, b.loop_k_down, b.loop_k_forget, b.loop_k_gain, eps,
              kh, msg_rmsx, msg_rmsm, msg_h);
          inkling_message_gate<ExecSpace, Scalar>(
              o2, moe_in, b.loop_v_down, b.loop_v_forget, b.loop_v_gain, eps,
              vh, msg_rmsx, msg_rmsm, msg_h);
        } else {
          Kokkos::deep_copy(qh, rq0);
          Kokkos::deep_copy(kh, qh);
          Kokkos::deep_copy(vh, qh);
        }
      } else {
        // Full kinds: 3-query retrieval (raw h on layer 0).
        if (l == 0) {
          Kokkos::parallel_for(
              "Inkling::qkv_alias",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
              KOKKOS_LAMBDA(size_t t, size_t d) {
                qh(t, d) = hh(t, d);
                kh(t, d) = hh(t, d);
                vh(t, d) = hh(t, d);
              });
        } else {
          if (b.attn_res_q.extent(0) != 3)
            throw std::invalid_argument(
                "Inkling::layer_forward: full layer needs 3 res queries");
          attn_retrieve(bb, nb0, hh, b.attn_res_q, use_norm_res, eps, retr3);
          auto rq0 =
              Kokkos::subview(retr3, Kokkos::ALL, size_t(0), Kokkos::ALL);
          auto rq1 =
              Kokkos::subview(retr3, Kokkos::ALL, size_t(1), Kokkos::ALL);
          auto rq2 =
              Kokkos::subview(retr3, Kokkos::ALL, size_t(2), Kokkos::ALL);
          Kokkos::deep_copy(qh, rq0);
          Kokkos::deep_copy(kh, rq1);
          Kokkos::deep_copy(vh, rq2);
        }
        if (has_msgs) {
          // Gate each packed stream; hn stages message rows, msgt the
          // gate outputs (out must not alias inputs).
          auto mq =
              Kokkos::subview(mmg, Kokkos::ALL, size_t(0), Kokkos::ALL);
          auto mk =
              Kokkos::subview(mmg, Kokkos::ALL, size_t(1), Kokkos::ALL);
          auto mv =
              Kokkos::subview(mmg, Kokkos::ALL, size_t(2), Kokkos::ALL);
          Kokkos::deep_copy(hn, mq);
          inkling_message_gate<ExecSpace, Scalar>(
              qh, hn, b.loop_q_down, b.loop_q_forget, b.loop_q_gain, eps,
              msgt, msg_rmsx, msg_rmsm, msg_h);
          Kokkos::deep_copy(qh, msgt);
          Kokkos::deep_copy(hn, mk);
          inkling_message_gate<ExecSpace, Scalar>(
              kh, hn, b.loop_k_down, b.loop_k_forget, b.loop_k_gain, eps,
              msgt, msg_rmsx, msg_rmsm, msg_h);
          Kokkos::deep_copy(kh, msgt);
          Kokkos::deep_copy(hn, mv);
          inkling_message_gate<ExecSpace, Scalar>(
              vh, hn, b.loop_v_down, b.loop_v_forget, b.loop_v_gain, eps,
              msgt, msg_rmsx, msg_rmsm, msg_h);
          Kokkos::deep_copy(vh, msgt);
        }
      }
    }

    // ---- 2. input norms (in place on the streams; query -> hn) ----
    if (!full) {
      rmsnorm<ExecSpace, Scalar>(qh, b.attn_norm, eps, hn);
      rmsnorm<ExecSpace, Scalar>(kh, b.attn_norm, eps, kh);
      rmsnorm<ExecSpace, Scalar>(vh, b.attn_norm, eps, vh);
    } else {
      rmsnorm<ExecSpace, Scalar>(qh, b.q_norm, eps, hn);
      rmsnorm<ExecSpace, Scalar>(kh, b.k_norm, eps, kh);
      rmsnorm<ExecSpace, Scalar>(vh, b.v_norm, eps, vh);
    }

    // ---- 3. dynamic weight_proj taps (full kinds, from the NORMED
    // streams: KeyValueProjection applies weight_proj to its normed
    // hidden_states input, not the raw residual stream). Must run after
    // the in-place norms above.
    auto dyn = Kokkos::subview(dyn_buf_, std::make_pair(size_t(0), C),
                               Kokkos::ALL);
    auto dynv = Kokkos::subview(dynv_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
    if (full) {
      Impl::linear_chunk<ExecSpace, Scalar>(handle, kh, b.k_conv_proj,
                                            deq_dyn_, dyn, "Inkling::kdyn",
                                            q8_act_buf_);
      Impl::linear_chunk<ExecSpace, Scalar>(handle, vh, b.v_conv_proj,
                                            deq_dyn_, dynv, "Inkling::vdyn",
                                            q8_act_buf_);
    }

    // ---- 4. q/k/v linears from the normed streams ----
    Impl::linear_chunk<ExecSpace, Scalar>(handle, hn, b.wq, deq_qkv_, q2,
                                          "Inkling::wq", q8_act_buf_);
    Impl::linear_chunk<ExecSpace, Scalar>(handle, kh, b.wk, deq_kv_, k2,
                                          "Inkling::wk", q8_act_buf_);
    Impl::linear_chunk<ExecSpace, Scalar>(handle, vh, b.wv, deq_kv_, v2,
                                          "Inkling::wv", q8_act_buf_);

    // ---- 5. k/v short-convs (stateful across the chunk) ----
    auto& cs = conv_[(size_t)l][p];
    auto tmpc = Kokkos::subview(tmpc_buf_, std::make_pair(size_t(0), C + K),
                                Kokkos::ALL);
    auto tmpc_out = Kokkos::subview(tmpc_out_buf_,
                                    std::make_pair(size_t(0), C + K),
                                    Kokkos::ALL);
    auto tmpdyn = Kokkos::subview(tmpdyn_buf_, std::make_pair(size_t(0), C + K),
                                  Kokkos::ALL);
    auto tmpk = Kokkos::subview(tmpk_buf_, std::make_pair(size_t(0), C + K),
                                Kokkos::ALL);
    auto tmpk_out = Kokkos::subview(tmpk_out_buf_,
                                    std::make_pair(size_t(0), C + K),
                                    Kokkos::ALL);
    if (!full) {
      static_conv_chunk(k2, b.k_conv, cs.k, tmpk, tmpk_out, k2);
      static_conv_chunk(v2, b.v_conv, cs.v, tmpk, tmpk_out, v2);
    } else {
      dynamic_conv_chunk(k2, b.k_conv_static, dyn, NKV, HD, cs.k, tmpk,
                         tmpk_out, tmpdyn, k2);
      dynamic_conv_chunk(v2, b.v_conv_static, dynv, NKV, HD, cs.v, tmpk,
                         tmpk_out, tmpdyn, v2);
    }

    // ---- 6. head clips (in place on the head aliases) ----
    if (!full) {
      inkling_scaled_l2_clip<ExecSpace, Scalar>(q3, b.q_scale, cfg.qk_clip,
                                                q3);
    } else {
      inkling_log_scaled_l2_clip<ExecSpace, Scalar>(
          q3, b.q_scale, cfg.qk_clip, pos, cfg.log_alpha,
          (float)cfg.log_floor, q3);
    }
    inkling_l2_clip<ExecSpace, Scalar>(k3, cfg.qk_clip, k3);
    inkling_unweighted_rmsnorm_3d<ExecSpace, Scalar>(v3, eps, v3);

    // ---- 7. rel-bias states + sink from the normed query stream ----
    Impl::linear_chunk<ExecSpace, Scalar>(handle, hn, b.wrel, deq_rel_, rel2,
                                          "Inkling::wrel", q8_act_buf_);
    if (!full) {
      inkling_scaled_l2_clip<ExecSpace, Scalar>(rel3, b.rel_scale,
                                                cfg.rel_clip, rel3);
    } else {
      inkling_log_scaled_l2_clip<ExecSpace, Scalar>(
          rel3, b.rel_scale, cfg.rel_clip, pos, cfg.log_alpha,
          (float)cfg.log_floor, rel3);
    }
    Impl::linear_chunk<ExecSpace, Scalar>(handle, hn, b.wsink, deq_sink_,
                                          sink2, "Inkling::wsink",
                                          q8_act_buf_);

    // ---- 8. KV write + attention ----
    {
      auto kc = k_cache[(size_t)l][p];
      auto vc = v_cache[(size_t)l][p];
      Kokkos::parallel_for(
          "Inkling::kv_write",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
              {0, 0, 0}, {C, NKV, HD}),
          KOKKOS_LAMBDA(size_t t, size_t kv, size_t d) {
            kc(cpos + t, kv, d) = k3(t, kv, d);
            vc(cpos + t, kv, d) = v3(t, kv, d);
          });
      auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                Kokkos::ALL, Kokkos::ALL);
      auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                Kokkos::ALL, Kokkos::ALL);
      const size_t window = full ? 0 : (size_t)cfg.swa_window;
      Impl::inkling_attention_forward<ExecSpace, Scalar>(
          q3, kp, vp, cpos, window, rel3, b.rel_proj, sink2, attn_scale,
          Erel, a3);
    }

    // ---- 9. gated output proj ----
    Unm2 flat(a3.data(), C, HqD);
    Impl::linear_chunk<ExecSpace, Scalar>(handle, hn, b.wgate_attn, deq_qkv_,
                                          gate2, "Inkling::wgate_attn",
                                          q8_act_buf_);
    inkling_gated_output_mul<ExecSpace, Scalar>(flat, gate2, flat);
    Impl::linear_chunk<ExecSpace, Scalar>(handle, flat, b.wo, deq_o_, o2,
                                          "Inkling::wo", q8_act_buf_);

    // ---- 10. post-attention conv (in place) ----
    static_conv_chunk(o2, b.post_attn_conv, cs.pa, tmpc, tmpc_out, o2);

    // ---- 11. attention residual / block append ----
    const bool boundary = (l % cfg.attn_res_block == 0);
    if (boundary) {
      Kokkos::parallel_for(
          "Inkling::blk_append",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
          KOKKOS_LAMBDA(size_t t, size_t d) { bb(t, nb0, d) = hh(t, d); });
      ++nblocks;
    } else {
      Kokkos::parallel_for(
          "Inkling::attn_add",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
          KOKKOS_LAMBDA(size_t t, size_t d) {
            o2(t, d) = static_cast<Scalar>(static_cast<float>(hh(t, d)) +
                                           static_cast<float>(o2(t, d)));
          });
    }

    // ---- 12. mlp-input retrieval (+ gate) -> norm -> MoE ----
    {
      auto retr = Kokkos::subview(retr1_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL, Kokkos::ALL);
      auto moe_in = Kokkos::subview(moe_in_buf_,
                                    std::make_pair(size_t(0), C), Kokkos::ALL);
      auto msgt = Kokkos::subview(msg_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      attn_retrieve(bb, nblocks, o2, b.mlp_res_q, use_norm_res, eps, retr);
      auto m = Kokkos::subview(retr, Kokkos::ALL, size_t(0), Kokkos::ALL);
      Kokkos::deep_copy(msgt, m);
      if (has_msgs) {
        auto mm = Kokkos::subview(mmg, Kokkos::ALL, size_t(3), Kokkos::ALL);
        Kokkos::deep_copy(moe_in, mm);
        inkling_message_gate<ExecSpace, Scalar>(
            msgt, moe_in, b.loop_mlp_down, b.loop_mlp_forget,
            b.loop_mlp_gain, eps, hn, msg_rmsx, msg_rmsm, msg_h);
        rmsnorm<ExecSpace, Scalar>(hn, b.post_attn_norm, eps, moe_in);
      } else {
        rmsnorm<ExecSpace, Scalar>(msgt, b.post_attn_norm, eps, moe_in);
      }
      auto moe_out = Kokkos::subview(moe_out_buf_,
                                     std::make_pair(size_t(0), C), Kokkos::ALL);
      moe_forward(l, moe_in, moe_out);
      // ---- 13. post-mlp conv + residual add ----
      static_conv_chunk(moe_out, b.post_mlp_conv, cs.pm, tmpc, tmpc_out,
                        moe_out);
      Kokkos::parallel_for(
          "Inkling::mlp_add",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
          KOKKOS_LAMBDA(size_t t, size_t d) {
            hh(t, d) = static_cast<Scalar>(static_cast<float>(o2(t, d)) +
                                          static_cast<float>(moe_out(t, d)));
          });
    }
  }

 public:
  // Forward T prompt tokens starting at absolute position seq_len.
  // Returns logits [unpadded_vocab] for the LAST token. Updates KV caches
  // and conv states, advances seq_len by T. Chunked prefill (see class
  // docs); the returned view aliases persistent scratch (valid until the
  // next forward() call).
  Kokkos::View<Scalar*, ExecSpace> forward(
      const Kokkos::View<int*, ExecSpace>& tokens) {
    const size_t T = tokens.extent(0);
    if (T == 0) throw std::invalid_argument("Model::forward: empty tokens");
    const size_t pos0 = seq_len;
    if (pos0 + T > (size_t)cfg.max_seq_len)
      throw std::runtime_error("Model::forward: sequence exceeds max_seq_len");
    const size_t D = (size_t)cfg.d_model;
    size_t Cmax = cfg.prefill_chunk_size <= 0
                      ? T
                      : (size_t)cfg.prefill_chunk_size;
    if (Cmax == 0) Cmax = T;
    if (Cmax > T) Cmax = T;
    ensure_scratch(Cmax);
    ensure_deq_staging();
    auto last = last_buf_;
    auto logits = logits_buf_;

    for (size_t off = 0; off < T; off += Cmax) {
      const size_t C = std::min(Cmax, T - off);
      const size_t cpos = pos0 + off;
      auto h = Kokkos::subview(h_buf_, std::make_pair(size_t(0), C),
                               Kokkos::ALL);
      auto toks = Kokkos::subview(tokens, std::make_pair(off, off + C));
      Impl::embed_forward<ExecSpace, Scalar>(weights.tok_emb, toks, h);
      rmsnorm<ExecSpace, Scalar>(h, weights.tok_norm, cfg.eps, h);

      auto blocks = Kokkos::subview(blk_buf_, std::make_pair(size_t(0), C),
                                    Kokkos::ALL, Kokkos::ALL);
      size_t nblocks = 0;
      // Pre-loop layers.
      for (int l = 0; l < cfg.loop_start; ++l) {
        auto nomsg = Kokkos::subview(msg_out_buf_,
                                     std::make_pair(size_t(0), size_t(0)),
                                     Kokkos::ALL, Kokkos::ALL);
        layer_forward(l, 0, h, blocks, nblocks, cpos, C, nomsg, false);
      }
      // Snapshot pre-loop states for later passes (pass 0 overwrites h
      // and appends to blocks).
      {
        auto pre_h = Kokkos::subview(pre_h_buf_, std::make_pair(size_t(0), C),
                                     Kokkos::ALL);
        Kokkos::deep_copy(pre_h, h);
        auto pre_blk = Kokkos::subview(pre_blk_buf_,
                                       std::make_pair(size_t(0), C),
                                       Kokkos::ALL, Kokkos::ALL);
        Kokkos::parallel_for(
            "Inkling::pre_blk_copy",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                {0, 0}, {C, nblocks * D}),
            KOKKOS_LAMBDA(size_t t, size_t i) {
              pre_blk(t, i / D, i % D) = blocks(t, i / D, i % D);
            });
      }
      const size_t npre = nblocks;
      // First pass over loop layers.
      for (int l = cfg.loop_start; l < cfg.loop_end; ++l) {
        auto nomsg = Kokkos::subview(msg_out_buf_,
                                     std::make_pair(size_t(0), size_t(0)),
                                     Kokkos::ALL, Kokkos::ALL);
        layer_forward(l, 0, h, blocks, nblocks, cpos, C, nomsg, false);
      }
      // Remaining passes restart from the pre-loop states with messages
      // from the previous pass.
      for (int pass = 1; pass < cfg.loop_passes; ++pass) {
        // Previous pass's final hidden + stripped block rows. The strip
        // count mirrors HF (num_pre_loop_blocks + 1), clamped at zero.
        const size_t Sall = nblocks;
        const size_t strip = std::min(npre + 1, Sall);
        const size_t Sstrip = Sall - strip;
        {
          auto prev_h = Kokkos::subview(prev_h_buf_,
                                        std::make_pair(size_t(0), C),
                                        Kokkos::ALL);
          Kokkos::deep_copy(prev_h, h);
          // Pack stripped prev block rows contiguously for the retrieval
          // (src op needs exact [C,S,D]).
          auto prevpack = Kokkos::subview(
              prevpack_buf_, std::make_pair(size_t(0), C * Sstrip),
              Kokkos::ALL);
          Kokkos::parallel_for(
              "Inkling::prev_pack",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                  {0, 0}, {C * Sstrip, D}),
              KOKKOS_LAMBDA(size_t i, size_t d) {
                const size_t t = (Sstrip == 0) ? 0 : i / Sstrip;
                const size_t s = (Sstrip == 0) ? 0 : i % Sstrip;
                prevpack(i, d) = (Sstrip == 0)
                                     ? static_cast<Scalar>(0)
                                     : blocks(t, strip + s, d);
              });
        }
        // Restore pre-loop hidden/blocks into the working buffers.
        {
          auto pre_h = Kokkos::subview(pre_h_buf_,
                                       std::make_pair(size_t(0), C),
                                       Kokkos::ALL);
          Kokkos::deep_copy(h, pre_h);
          auto pre_blk = Kokkos::subview(pre_blk_buf_,
                                         std::make_pair(size_t(0), C),
                                         Kokkos::ALL, Kokkos::ALL);
          Kokkos::parallel_for(
              "Inkling::blk_restore",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                  {0, 0}, {C, npre * D}),
              KOKKOS_LAMBDA(size_t t, size_t i) {
                blocks(t, i / D, i % D) = pre_blk(t, i / D, i % D);
              });
        }
        nblocks = npre;
        // Per loop layer: retrieve messages, forward with pass slot.
        using QTab = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        for (int l = cfg.loop_start; l < cfg.loop_end; ++l) {
          const auto& b = weights.blocks[(size_t)l];
          auto msgs = Kokkos::subview(msg_out_buf_,
                                      std::make_pair(size_t(0), C), Kokkos::ALL,
                                      Kokkos::ALL);
          {
            auto prevpack = Kokkos::subview(
                prevpack_buf_, std::make_pair(size_t(0), C * Sstrip),
                Kokkos::ALL);
            UnmAlias3 bp(prevpack.data(), C, Sstrip, D);
            auto prev_h = Kokkos::subview(prev_h_buf_,
                                          std::make_pair(size_t(0), C),
                                          Kokkos::ALL);
            QTab qtab(b.msg_query.data(), 1, 4, D);
            inkling_message_attention_residual<ExecSpace, Scalar>(
                bp, prev_h, qtab, 0, cfg.eps, msgs);
          }
          layer_forward(l, (size_t)pass, h, blocks, nblocks, cpos, C, msgs,
                        true);
        }
      }
      // Post-loop layers.
      for (int l = cfg.loop_end; l < cfg.n_layers; ++l) {
        auto nomsg = Kokkos::subview(msg_out_buf_,
                                     std::make_pair(size_t(0), size_t(0)),
                                     Kokkos::ALL, Kokkos::ALL);
        layer_forward(l, 0, h, blocks, nblocks, cpos, C, nomsg, false);
      }

      if (off + C < T) continue;
      // Final retrieval + norm + head on the last row only.
      {
        auto retr = Kokkos::subview(retr1_buf_, std::make_pair(size_t(0), C),
                                    Kokkos::ALL, Kokkos::ALL);
        attn_retrieve(blocks, nblocks, h, weights.out_res_q,
                       (bool)cfg.attn_res_normalize, cfg.eps, retr);
        auto rq_last =
            Kokkos::subview(retr, C - 1, size_t(0), Kokkos::ALL);
        Kokkos::deep_copy(last, rq_last);
        rmsnorm_row<ExecSpace, Scalar>(last, weights.out_norm, cfg.eps,
                                       last);
      }
      // lm_head over the unpadded vocab rows. Dense out_w must be
      // exactly [U, D] (padded rows are dropped at load, matching HF);
      // packed heads must also be exactly [U, D].
      {
        const size_t U = (size_t)cfg.unpadded_vocab;
        if constexpr (Impl::is_packed_weight_v<
                          decltype(std::declval<WeightsT>().out_w)>) {
          if (weights.out_w.rows != U)
            throw std::invalid_argument(
                "Model::forward: packed out_w rows must equal "
                "unpadded_vocab");
          Impl::lm_head_forward<ExecSpace, Scalar>(
              handle, last, weights.out_w, logits, "lm_head", q8_act_buf_);
        } else {
          if (weights.out_w.extent(0) != U || weights.out_w.extent(1) != D)
            throw std::invalid_argument(
                "Model::forward: dense out_w must be [unpadded_vocab, "
                "d_model]");
          Impl::lm_head_forward<ExecSpace, Scalar>(
              handle, last, weights.out_w, logits, "lm_head", q8_act_buf_);
        }
      }
      // Final softcap (rank-1 form of inkling_softcap, which only
      // exposes the rank-2 entry point).
      {
        const float cap = cfg.final_softcap;
        auto lg = logits;
        Kokkos::parallel_for(
            "Inkling::final_softcap",
            Kokkos::RangePolicy<ExecSpace>(0, lg.extent(0)),
            KOKKOS_LAMBDA(size_t i) {
              const float v = static_cast<float>(lg(i)) / cap;
              lg(i) = static_cast<Scalar>(Kokkos::tanhf(v) * cap);
            });
      }
    }
    seq_len += T;
    return logits;
  }

  // Host convenience: prefill + sampled decode loop. Shared driver
  // (Common::generate_impl); this wrapper only resolves the default
  // {cfg.eos_token} stop set. Returns all ids (prompt + generated).
  using TokenCallback = ::KokkosDNN::Common::TokenCallback;
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            const std::vector<int>& stop_ids,
                            const SampleOptions& opt, HostRNG* rng = nullptr,
                            TokenCallback on_token = nullptr) {
    const std::vector<int> stops =
        stop_ids.empty() ? std::vector<int>{cfg.eos_token} : stop_ids;
    return ::KokkosDNN::Common::generate_impl(*this, prompt, max_new, stops,
                                              opt, rng, std::move(on_token));
  }

  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            const std::vector<int>& stop_ids,
                            TokenCallback on_token) {
    return generate(prompt, max_new, stop_ids, SampleOptions{}, nullptr,
                    std::move(on_token));
  }

  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            const std::vector<int>& stop_ids) {
    return generate(prompt, max_new, stop_ids, SampleOptions{}, nullptr,
                    nullptr);
  }

  // Back-compatible single-eos overload: eos >= 0 stops on that id only,
  // eos < 0 (default) uses the config eos token.
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            int eos = -1) {
    if (eos < 0) return generate(prompt, max_new, std::vector<int>{});
    return generate(prompt, max_new, std::vector<int>{eos});
  }

 private:
  // Unmanaged block-stack alias (exact LayoutRight reinterpretation).
  using UnmAlias3 =
      Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
};  // class Model

}  // namespace Inkling
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_INKLING_MODEL_HPP
