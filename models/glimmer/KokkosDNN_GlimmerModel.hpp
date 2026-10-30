// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GLIMMER_MODEL_HPP
#define KOKKOSDNN_GLIMMER_MODEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "KokkosDNN_Activation.hpp"
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_KVCache.hpp"
#include "KokkosDNN_LayerScale.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_RoPE.hpp"
#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_SwiGLU.hpp"
#include "KokkosDNN_GlimmerConfig.hpp"
#include "KokkosDNN_GlimmerQuantDispatch.hpp"
#include "KokkosDNN_GlimmerWeights.hpp"

namespace KokkosDNN {
namespace Glimmer {
namespace Impl {

// Fused flatten + attention-output gating:
//   out[t,i] = attn[t,i/HD,i%HD] * sigmoid(g[t,i]).
// Merges the rank-3 -> rank-2 reshape pass with the sigmoid gate multiply:
// one launch and one write of [T, Hq*D] instead of two passes. Bit-identical
// to running them back-to-back (the reshape intermediate was already
// Scalar, so no rounding is skipped).
template <class ExecSpace, class Scalar>
void flatten_gate_sigmoid_mul(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& attn,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& g,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out) {
  const size_t T = attn.extent(0), H = attn.extent(1), D = attn.extent(2);
  if (g.extent(0) != T || g.extent(1) != H * D || out.extent(0) != T ||
      out.extent(1) != H * D)
    throw std::invalid_argument(
        "flatten_gate_sigmoid_mul: dimension mismatch");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      "Glimmer::flatten_gate_sigmoid_mul",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H * D}),
      KOKKOS_LAMBDA(size_t t, size_t i) {
        float gv = static_cast<float>(g(t, i));
        float s  = 1.0f / (1.0f + Kokkos::expf(-gv));
        out(t, i) = static_cast<Scalar>(
            static_cast<float>(attn(t, i / D, i % D)) * s);
      });
}

}  // namespace Impl

// Autoregressive Muse Glimmer 30B text model. Owns dense KV caches and
// scratch. Template on ExecSpace so the same code runs on
// Serial/OpenMP/CUDA, and on Scalar (float or bhalf_t; default bhalf_t
// matches the heretic checkpoint). All weights/activations/KV are Scalar;
// reductions/scales stay FP32; inv_freq stays FP32.
//
// Forward math (verified against transformers modeling_muse_glimmer.py; the
// scaffold DESIGN.md marks the same points as ASSUMPTION — confirmed here):
//   * embedding: h = scaleless-RMSNorm(E[tok]) (MuseGlimmerTextNormedEmbedding;
//     no output multiplier at the embed — the multiplier applies to logits);
//   * per layer: input CenteredRMSNorm (eps rms_norm_eps) ->
//     q = scaleless-QKNorm(Wq t) * qk_scale_factor (3.87; the scale is
//     fused into the q norm via the qk_qscale_ weight, one pass),
//     k = scaleless-QKNorm(Wk t), v = Wv t (v unnormed) ->
//     RoPE(theta = layer_rope_theta[l]) on q/k, SKIPPED on NoPE layers
//     (layer_rope_theta == 0, i.e. the full-attention layers: the HF model
//     passes position_embeddings=None there, so they rotate nothing) ->
//     sliding (window 2048) or full causal GQA with scale 1/sqrt(head_dim)
//     (the HF sdpa `scaling`; the 3.87 lives on q, not in the scale) ->
//     attn_out flat-reshaped and *= sigmoid(Wgate_attn t) in one fused
//     pass -> o_proj ->
//     post-attention CenteredRMSNorm (eps post_norm_eps) + residual ->
//     SwiGLU MLP (silu) under pre-FFN CenteredRMSNorm (eps rms) ->
//     post-FFN CenteredRMSNorm (eps post) + residual;
//   * head: final RMSNorm (scaled, eps rms) -> lm_head (untied) gemv ->
//     logits * output_multiplier -> final_logit_softcap * tanh(logits / cap).
//
// NOTE on Config::rope_theta_for_layer(): it maps a 0 entry to the global
// theta, but the Model deliberately does NOT use it: upstream a 0 entry
// means NoPE (no rotary at all on that layer), not "use the global theta".
// The Model keys off the raw layer_rope_theta entry instead.
//
// KV caches are dense FP by default: every layer owns its own
// [max_seq, n_kv, head_dim] K/V pair (no sharing, no paging). With
// Config::kv_int8 every layer instead owns symmetric-int8 pools plus
// [max_seq, n_kv] FP32 scale tables (no sharing, no paging in this step).
template <class ExecSpace, class Scalar = DefaultScalar>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using HostInt = typename ViewInt::host_mirror_type;
  // int8 KV pools (quantized mode): per-token per-head symmetric int8
  // values plus FP32 scales, one [max_seq, n_kv, head_dim] K/V pair and one
  // [max_seq, n_kv] scale pair per layer.
  using ViewI8 = Kokkos::View<int8_t***, Kokkos::LayoutRight, ExecSpace>;
  using ViewScale = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;
  DNNHandle<ExecSpace> handle;
  // Per-layer RoPE tables: inv_freq for layers with layer_rope_theta > 0,
  // empty view for NoPE (full-attention, theta 0) layers.
  std::vector<Kokkos::View<float*, ExecSpace>> rope_inv_;
  // Ones vector standing in for the absent QK-norm weight (with_scale=False
  // upstream): qk_rmsnorm(flat, ones, eps, out) == scaleless per-head norm.
  // qk_qscale_ is the q-norm counterpart with the q pre-scale
  // (qk_scale_factor) fused in: filling it with the scale makes the q
  // qk_rmsnorm compute norm and scale in a single pass (it multiplies by
  // w[d]). Refreshed in forward() if the config changed under us.
  View1 qk_ones_, qk_qscale_;
  float qk_qscale_used_ = -1.0f;
  // Dense KV caches, one [max_seq, n_kv, head_dim] K/V pair per layer.
  // Quantized mode (Config::kv_int8) rents the int8 pools + FP32 scale
  // tables below instead; the two modes never coexist (kv_alloc_fresh
  // clears the inactive pair).
  std::vector<Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>> k_cache,
      v_cache;
  std::vector<ViewI8> kq_cache, vq_cache;
  std::vector<ViewScale> ks_cache, vs_cache;
  size_t seq_len = 0;

  // Persistent scratch: all T-sized temporaries are carved (via subview,
  // which allocates nothing) out of these capacity buffers. Buffers grow on
  // demand in ensure_scratch() and are never shrunk, so the steady decode
  // state (T=1 after prefill) performs zero View allocations.
  // Fixed-size state (last/logits) is allocated once in the constructor.
  MatR h_buf_, tmp_buf_, tmp2_buf_, proj_buf_, down_buf_, hn_buf_;
  MatR q2_buf_, k2_buf_, v2_buf_, attn_flat_buf_, gate_attn_buf_;
  MatR gate_buf_, up_buf_, act_buf_;
  // Dequantized-weight staging for the packed-quant path (see
  // KokkosDNN_GlimmerQuantDispatch.hpp): one exact-size dense buffer per
  // distinct projection shape, reused across layers. Exact sizing matters:
  // the cuBLAS path requires weight row pitch == K, so carved subviews of
  // an oversized buffer would silently fall back to the native kernel.
  // Shapes: q/gate_attn [HqD, DIM], kv [HkvD, DIM], o [DIM, HqD],
  // ffn gate/up [FFN, DIM], down [DIM, FFN]. lm_head/embed need no staging
  // (single-row quant_gemv / packed-row gather). These stay empty while
  // Weights holds dense views (ensure_deq_staging keys off the block weight
  // type), so dense builds pay no extra memory.
  MatR deq_q_, deq_kv_, deq_o_, deq_ffn_, deq_down_;
  View3 q3_buf_, k3_buf_, v3_buf_, attn_out_buf_;
  ViewInt pos_buf_;
  View1 last_buf_, logits_buf_;
  size_t scratch_cap_ = 0;
  // Persistent input staging for generate(): prefill staging grows on
  // demand, the single-token decode slot is fixed at 1. The prompt copy is
  // a single host->device deep_copy per generate() (unavoidable staging of
  // host ids); decode steps never touch the host -- the next id is written
  // by a device kernel (see generate()).
  ViewInt tok_buf_;
  HostInt tok_host_;
  size_t tok_cap_ = 0;
  // Persistent single-token decode slot. Written on-device each step (no
  // host mirror + deep_copy per token).
  ViewInt one_buf_;

  Model() = default;
  Model(const Config& c, const Weights<ExecSpace, Scalar>& w)
      : cfg(c), weights(w) {
    // Validate before renting any KV cache: incompatible dims or an
    // out-of-range context fail here, not in a device kernel or a
    // multi-GiB allocation.
    cfg.validate();
    rope_inv_.resize((size_t)cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      const double theta = cfg.layer_rope_theta[(size_t)l];
      if (theta > 0.0)
        rope_inv_[(size_t)l] = rope_inv_freq<ExecSpace>(
            cfg.head_dim, (float)theta, "", 1.0, 1.0, 4.0, 0);
    }
    // Ones stand-in for the absent (scaleless) QK-norm weight, plus the
    // q-norm weight carrying the fused q pre-scale (see member docs).
    qk_ones_ = View1("qk_ones", (size_t)cfg.head_dim);
    qk_qscale_ = View1("qk_qscale", (size_t)cfg.head_dim);
    {
      auto h = Kokkos::create_mirror_view(qk_ones_);
      auto hs = Kokkos::create_mirror_view(qk_qscale_);
      const float qs = (float)cfg.qk_scale_factor;
      for (size_t i = 0; i < h.extent(0); ++i) {
        h(i) = static_cast<Scalar>(1.0f);
        hs(i) = static_cast<Scalar>(qs);
      }
      Kokkos::deep_copy(qk_ones_, h);
      Kokkos::deep_copy(qk_qscale_, hs);
      qk_qscale_used_ = qs;
    }
    kv_alloc_fresh();
    last_buf_ = View1("last", (size_t)cfg.dim);
    logits_buf_ = View1("logits", (size_t)cfg.vocab_size);
    one_buf_ = ViewInt("one", 1);
  }

  // Full-window KV bytes for the configured mode (all layers, K+V).
  size_t kv_cache_bytes() const {
    return cfg.kv_cache_bytes(sizeof(Scalar));
  }

  // O(1) reset: just rewinds the position counter. No cache wipe.
  // Correctness invariant: forward() writes KV at [cpos, cpos+C) before
  // reading prefix [0, cpos+C), with absolute positions and q_off=cpos,
  // so entries at/after seq_len are always overwritten before they are
  // read. Stale tail data from a previous sequence is never observed.
  void reset() { seq_len = 0; }

 private:
  // (Re)allocate KV caches for the configured mode: dense FP
  // [max_seq, n_kv, head_dim] K/V pairs, or symmetric-int8 pools plus
  // [max_seq, n_kv] FP32 scale tables when Config::kv_int8 is set.
  void kv_alloc_fresh() {
    const size_t MS  = (size_t)cfg.max_seq_len;
    const size_t NKV = (size_t)cfg.n_kv_heads, HD = (size_t)cfg.head_dim;
    k_cache.clear();
    v_cache.clear();
    kq_cache.clear();
    vq_cache.clear();
    ks_cache.clear();
    vs_cache.clear();
    if (cfg.use_quant_kv()) {
      for (int l = 0; l < cfg.n_layers; ++l) {
        kq_cache.emplace_back("kq_cache", MS, NKV, HD);
        vq_cache.emplace_back("vq_cache", MS, NKV, HD);
        ks_cache.emplace_back("ks_cache", MS, NKV);
        vs_cache.emplace_back("vs_cache", MS, NKV);
      }
      return;
    }
    for (int l = 0; l < cfg.n_layers; ++l) {
      k_cache.emplace_back("k_cache", MS, NKV, HD);
      v_cache.emplace_back("v_cache", MS, NKV, HD);
    }
  }

  // Grow chunk-sized scratch to hold at least C rows (exact-size growth;
  // the first, largest prefill chunk sizes everything, later T=1 decodes
  // reuse it). Also repairs buffers if cfg dims changed under us.
  // With chunked prefill C = min(prefill_chunk_size, T), so a long prompt
  // no longer materializes O(T·dim) activations.
  void ensure_scratch(size_t C) {
    const size_t DIM = (size_t)cfg.dim;
    const size_t HqD = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
    const size_t HkvD = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    const size_t FFN = (size_t)cfg.ffn_dim;
    const size_t NH = (size_t)cfg.n_heads, NKV = (size_t)cfg.n_kv_heads,
                 HD = (size_t)cfg.head_dim;
    if (C <= scratch_cap_ && h_buf_.extent(0) >= C &&
        h_buf_.extent(1) == DIM && tmp_buf_.extent(1) == DIM &&
        tmp2_buf_.extent(1) == DIM && proj_buf_.extent(1) == DIM &&
        down_buf_.extent(1) == DIM && hn_buf_.extent(1) == DIM &&
        q2_buf_.extent(1) == HqD && attn_flat_buf_.extent(1) == HqD &&
        gate_attn_buf_.extent(1) == HqD && k2_buf_.extent(1) == HkvD &&
        v2_buf_.extent(1) == HkvD && gate_buf_.extent(1) == FFN &&
        up_buf_.extent(1) == FFN && act_buf_.extent(1) == FFN &&
        q3_buf_.extent(1) == NH && q3_buf_.extent(2) == HD &&
        k3_buf_.extent(1) == NKV && k3_buf_.extent(2) == HD &&
        v3_buf_.extent(1) == NKV && v3_buf_.extent(2) == HD &&
        attn_out_buf_.extent(1) == NH && attn_out_buf_.extent(2) == HD &&
        pos_buf_.extent(0) >= C)
      return;
    h_buf_ = MatR("h", C, DIM);
    tmp_buf_ = MatR("tmp", C, DIM);
    tmp2_buf_ = MatR("tmp2", C, DIM);
    proj_buf_ = MatR("proj", C, DIM);
    down_buf_ = MatR("down", C, DIM);
    hn_buf_ = MatR("hn", C, DIM);
    q2_buf_ = MatR("q2", C, HqD);
    k2_buf_ = MatR("k2", C, HkvD);
    v2_buf_ = MatR("v2", C, HkvD);
    attn_flat_buf_ = MatR("attn_flat", C, HqD);
    gate_attn_buf_ = MatR("gate_attn", C, HqD);
    gate_buf_ = MatR("gate", C, FFN);
    up_buf_ = MatR("up", C, FFN);
    act_buf_ = MatR("act", C, FFN);
    q3_buf_ = View3("q", C, NH, HD);
    k3_buf_ = View3("k", C, NKV, HD);
    v3_buf_ = View3("v", C, NKV, HD);
    attn_out_buf_ = View3("attn_out", C, NH, HD);
    pos_buf_ = ViewInt("pos", C);
    scratch_cap_ = C;
  }

  void ensure_tok(size_t n) {
    if (n <= tok_cap_ && tok_buf_.extent(0) >= n && tok_host_.extent(0) >= n)
      return;
    tok_buf_ = ViewInt("toks", n);
    tok_host_ = HostInt("toks_host", n);
    tok_cap_ = n;
  }

  // Rent exact-size dequant staging for the packed-quant path (see member
  // docs above). Compiles away for dense Weights: the trait keys off the
  // block weight type, so dense builds skip every allocation here.
  // INTEGRATION POINT for the Weights stream: once BlockWeights::wq et al.
  // become Gguf::QuantWeight<ExecSpace>, this starts renting automatically;
  // no change needed here unless field names differ (then update the
  // decltype below plus the linear_chunk staging args in forward()).
  void ensure_deq_staging() {
    using WqT = decltype(std::declval<BlockWeights<ExecSpace, Scalar>>().wq);
    if constexpr (!Impl::is_packed_weight_v<WqT>) {
      return;
    } else {
      const size_t DIM = (size_t)cfg.dim;
      const size_t HqD = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
      const size_t HkvD = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
      const size_t FFN = (size_t)cfg.ffn_dim;
      // Exact-size (repair on dim change, mirroring ensure_scratch).
      if (deq_q_.extent(0) != HqD || deq_q_.extent(1) != DIM)
        deq_q_ = MatR("deq_q", HqD, DIM);
      if (deq_kv_.extent(0) != HkvD || deq_kv_.extent(1) != DIM)
        deq_kv_ = MatR("deq_kv", HkvD, DIM);
      if (deq_o_.extent(0) != DIM || deq_o_.extent(1) != HqD)
        deq_o_ = MatR("deq_o", DIM, HqD);
      if (deq_ffn_.extent(0) != FFN || deq_ffn_.extent(1) != DIM)
        deq_ffn_ = MatR("deq_ffn", FFN, DIM);
      if (deq_down_.extent(0) != DIM || deq_down_.extent(1) != FFN)
        deq_down_ = MatR("deq_down", DIM, FFN);
    }
  }

 public:
  // Forward T prompt tokens starting at absolute position seq_len.
  // Returns logits [vocab] for the LAST token. Updates KV caches and
  // advances seq_len by T.
  //
  // Chunked prefill: prompts longer than cfg.prefill_chunk_size (> 0) are
  // processed in chunks of that many tokens, so activations stay
  // O(chunk·dim) instead of O(T·dim). Chunking is exact with native
  // kernels: each chunk runs all layers for its tokens, reading previously
  // written KV (earlier chunks) plus its own prefix via the causal mask
  // (q_off = chunk start), which matches single-shot prefill token-for-
  // token. Only the last chunk runs the final norm + lm_head.
  // cfg.prefill_chunk_size <= 0 disables chunking (single-shot). Decode
  // (T=1) is always one chunk.
  //
  // The returned view aliases persistent scratch: it is valid until the
  // next forward() call, which overwrites it. Consume it (e.g. via
  // sample_greedy) before the next call. No View allocation happens here
  // once scratch has grown to the chunk size (steady decode: zero
  // allocations).
  Kokkos::View<Scalar*, ExecSpace> forward(
      const Kokkos::View<int*, ExecSpace>& tokens) {
    const size_t T = tokens.extent(0);
    if (T == 0) throw std::invalid_argument("Model::forward: empty tokens");
    const size_t pos0 = seq_len;
    if (pos0 + T > (size_t)cfg.max_seq_len)
      throw std::runtime_error("Model::forward: sequence exceeds max_seq_len");
    // Hoisted (device lambdas must not capture `this`/`cfg`).
    const int DIM = cfg.dim, HD = cfg.head_dim, NKV = cfg.n_kv_heads,
              VS = cfg.vocab_size, NL = cfg.n_layers;
    const float EPS = cfg.rms_norm_eps, POST_EPS = cfg.post_norm_eps;
    const float QK_SCALE = cfg.qk_scale_factor;
    const float ATTN_SCALE = 1.0f / std::sqrt((float)HD);
    const float OUT_MULT = (float)cfg.output_multiplier;
    const float SOFTCAP = cfg.final_logit_softcap;
    const size_t SWIN = (size_t)cfg.sliding_window;
    // Construction-time KV mode (hoisted: indexing the member cache vectors
    // inside a device lambda would capture `this`, illegal on CUDA; the
    // copied View handles below are device-safe).
    const bool kvq = cfg.use_quant_kv();
    // Resolve chunk size: <= 0 means single-shot.
    size_t Cmax = cfg.prefill_chunk_size <= 0
                      ? T
                      : (size_t)cfg.prefill_chunk_size;
    if (Cmax == 0) Cmax = T;
    if (Cmax > T) Cmax = T;
    ensure_scratch(Cmax);
    ensure_deq_staging();
    if (last_buf_.extent(0) != (size_t)DIM) last_buf_ = View1("last", DIM);
    if (logits_buf_.extent(0) != (size_t)VS)
      logits_buf_ = View1("logits", VS);

    // Hoisted member views: device lambdas below must capture these
    // locals, never `this`.
    auto last = last_buf_;
    auto logits = logits_buf_;
    auto ones = qk_ones_;
    // Refresh the fused q pre-scale weight if dims/config changed under us
    // (mirrors ensure_scratch's repair-on-dim-change above).
    if (qk_qscale_.extent(0) != (size_t)HD || qk_qscale_used_ != QK_SCALE) {
      qk_qscale_ = View1("qk_qscale", (size_t)HD);
      auto hqs = Kokkos::create_mirror_view(qk_qscale_);
      for (size_t i = 0; i < hqs.extent(0); ++i)
        hqs(i) = static_cast<Scalar>(QK_SCALE);
      Kokkos::deep_copy(qk_qscale_, hqs);
      qk_qscale_used_ = QK_SCALE;
    }
    auto qscale = qk_qscale_;

    const int HkvD = NKV * HD;

    for (size_t off = 0; off < T; off += Cmax) {
      const size_t C = std::min(Cmax, T - off);
      const size_t cpos = pos0 + off;
      // Carve C-row working views out of the capacity buffers. Subviews
      // share the allocation, so none of this allocates.
      auto h = Kokkos::subview(h_buf_, std::make_pair(size_t(0), C),
                               Kokkos::ALL);
      auto tmp = Kokkos::subview(tmp_buf_, std::make_pair(size_t(0), C),
                                 Kokkos::ALL);
      auto tmp2 = Kokkos::subview(tmp2_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto q2 = Kokkos::subview(q2_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
      auto k2 = Kokkos::subview(k2_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
      auto v2 = Kokkos::subview(v2_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
      auto gate = Kokkos::subview(gate_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto up = Kokkos::subview(up_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
      auto act = Kokkos::subview(act_buf_, std::make_pair(size_t(0), C),
                                 Kokkos::ALL);
      auto q3 = Kokkos::subview(q3_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL, Kokkos::ALL);
      auto k3 = Kokkos::subview(k3_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL, Kokkos::ALL);
      auto v3 = Kokkos::subview(v3_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL, Kokkos::ALL);
      auto pos = Kokkos::subview(pos_buf_, std::make_pair(size_t(0), C));
      auto attn_out = Kokkos::subview(attn_out_buf_,
                                      std::make_pair(size_t(0), C), Kokkos::ALL,
                                      Kokkos::ALL);
      auto attn_flat = Kokkos::subview(attn_flat_buf_,
                                       std::make_pair(size_t(0), C),
                                       Kokkos::ALL);
      auto gate_attn = Kokkos::subview(gate_attn_buf_,
                                       std::make_pair(size_t(0), C),
                                       Kokkos::ALL);
      auto proj = Kokkos::subview(proj_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto down = Kokkos::subview(down_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto hn = Kokkos::subview(hn_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
      auto toks =
          Kokkos::subview(tokens, std::make_pair(off, off + C));

      // Scaleless-normed embedding (no output multiplier here: upstream
      // applies it to logits, not to hidden states). Packed embed (if the
      // Weights stream packs it) goes through the packed-row gather.
      Impl::embed_forward<ExecSpace, Scalar>(weights.embed, toks, h);
      rmsnorm_noscale<ExecSpace, Scalar>(h, EPS, h);

      Kokkos::parallel_for(
          "pos_fill", Kokkos::RangePolicy<ExecSpace>(0, C),
          KOKKOS_LAMBDA(size_t t) { pos(t) = (int)(cpos + t); });

      for (int l = 0; l < NL; ++l) {
        auto& b = weights.blocks[(size_t)l];
        const bool full = cfg.is_full(l);
        const bool nope = cfg.layer_rope_theta[(size_t)l] <= 0.0;
        // 1. Attention block: input norm, QK projs + per-head norms (the q
        // pre-scale is fused into the q norm via the qk_qscale_ weight),
        // (optional) RoPE, KV write + GQA.
        rmsnorm_one_plus<ExecSpace, Scalar>(h, b.input_norm, EPS, tmp);
        // Linear dispatch (see KokkosDNN_GlimmerQuantDispatch.hpp): decode
        // chunks (C == 1) run quant_gemv on packed weights, prefill chunks
        // dequantize-once into exact staging + dense gemm. Dense Weights
        // take the plain gemm path in both cases (staging ignored).
        Impl::linear_chunk<ExecSpace, Scalar>(handle, tmp, b.wq, deq_q_, q2, "wq");
        Impl::linear_chunk<ExecSpace, Scalar>(handle, tmp, b.wk, deq_kv_, k2, "wk");
        Impl::linear_chunk<ExecSpace, Scalar>(handle, tmp, b.wv, deq_kv_, v2, "wv");
        qk_rmsnorm<ExecSpace, Scalar>(q2, qscale, EPS, q3);
        qk_rmsnorm<ExecSpace, Scalar>(k2, ones, EPS, k3);
        Kokkos::parallel_for(
            "reshape_v",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)HkvD}),
            KOKKOS_LAMBDA(size_t t, size_t i) {
              v3(t, i / HD, i % HD) = v2(t, i);
            });
        if (!nope) {
          auto inv = rope_inv_[(size_t)l];
          rope_neoX_inplace<ExecSpace, Scalar>(q3, k3, inv, HD, pos);
        }
        // KV write + attention, branched once per layer on the
        // construction-time mode (views hoisted to locals above: indexing
        // the member vectors inside a device lambda would capture `this`).
        //   dense FP: legacy contiguous write + prefix-subview attention.
        //   int8: quantizing write + dequant-on-the-fly native attention
        //     (full and sliding forms). The quantized entry points ignore
        //     the handle -- they always run the native streaming kernel
        //     (cuDNN SDPA has no int8 form) -- so cuDNN is bypassed in
        //     code, not just docs; TPL GEMMs elsewhere are unaffected.
        // Dense is bit-identical to the pre-quant path; int8 drifts
        // slightly (symmetric per-token per-head rounding).
        if (!kvq) {
          auto kc = k_cache[(size_t)l];
          auto vc = v_cache[(size_t)l];
          Kokkos::parallel_for(
              "kcache_write",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                  {0, 0, 0}, {C, (size_t)NKV, (size_t)HD}),
              KOKKOS_LAMBDA(size_t t, size_t hh, size_t d) {
                kc(cpos + t, hh, d) = k3(t, hh, d);
                vc(cpos + t, hh, d) = v3(t, hh, d);
              });
          // Pass the valid KV prefix as subviews: no repack/copy, and no extra
          // [K, Hkv, D] temporaries per layer. The attention kernel is
          // layout-agnostic and streams over keys, so long prefixes stay O(1)
          // in extra memory here too. q_off is the chunk start so causal
          // masking (query t attends to keys [0, cpos + t]) matches
          // single-shot prefill exactly.
          auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                    Kokkos::ALL, Kokkos::ALL);
          auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                    Kokkos::ALL, Kokkos::ALL);
          if (!full) {
            attention_gqa_single_sliding<ExecSpace, Scalar>(
                handle, q3, kp, vp, cpos, attn_out, ATTN_SCALE, SWIN);
          } else {
            attention_gqa_single<ExecSpace, Scalar>(handle, q3, kp, vp, cpos,
                                                    attn_out, ATTN_SCALE);
          }
        } else {
          auto kqc = kq_cache[(size_t)l];
          auto vqc = vq_cache[(size_t)l];
          auto ksc = ks_cache[(size_t)l];
          auto vsc = vs_cache[(size_t)l];
          kv_write_quantized_contiguous<ExecSpace, Scalar>(k3, v3, cpos, kqc,
                                                           ksc, vqc, vsc);
          // Full pools (not prefix subviews): the quantized loaders cap the
          // key range at q_off + t + 1 themselves, so the full pool plus the
          // chunk start is exact for both prefill chunks and decode steps.
          if (!full) {
            attention_gqa_quantized_sliding<ExecSpace, Scalar>(
                handle, q3, kqc, ksc, vqc, vsc, cpos, attn_out, ATTN_SCALE,
                SWIN);
          } else {
            attention_gqa_quantized<ExecSpace, Scalar>(handle, q3, kqc, ksc,
                                                       vqc, vsc, cpos,
                                                       attn_out, ATTN_SCALE);
          }
        }
        // Output gating (sigmoid on the normed-input gate proj) + o_proj.
        // The attn reshape and the gate multiply are fused into one pass
        // (runs after the gate proj, whose output it consumes).
        Impl::linear_chunk<ExecSpace, Scalar>(handle, tmp, b.wgate_attn, deq_q_, gate_attn,
                           "wgate_attn");
        Impl::flatten_gate_sigmoid_mul<ExecSpace, Scalar>(attn_out, gate_attn,
                                                          attn_flat);
        Impl::linear_chunk<ExecSpace, Scalar>(handle, attn_flat, b.wo, deq_o_, proj, "wo");
        rmsnorm_one_plus<ExecSpace, Scalar>(proj, b.post_attn_norm, POST_EPS,
                                            tmp2);
        Kokkos::parallel_for(
            "resid1",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += tmp2(t, d); });
        // 2. SwiGLU MLP under the pre-FFN norm + post-FFN norm.
        rmsnorm_one_plus<ExecSpace, Scalar>(h, b.pre_ffn_norm, EPS, tmp2);
        Impl::linear_chunk<ExecSpace, Scalar>(handle, tmp2, b.wgate, deq_ffn_, gate, "wgate");
        Impl::linear_chunk<ExecSpace, Scalar>(handle, tmp2, b.wup, deq_ffn_, up, "wup");
        swiglu<ExecSpace, Scalar>(gate, up, act);
        Impl::linear_chunk<ExecSpace, Scalar>(handle, act, b.wdown, deq_down_, down, "wdown");
        rmsnorm_one_plus<ExecSpace, Scalar>(down, b.post_ffn_norm, POST_EPS,
                                            tmp2);
        Kokkos::parallel_for(
            "resid2",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += tmp2(t, d); });
      }

      // Only the last chunk produces logits (earlier chunks only fill KV).
      if (off + C < T) continue;
      rmsnorm<ExecSpace, Scalar>(h, weights.final_norm, EPS, hn);
      Kokkos::parallel_for(
          "gather_last", Kokkos::RangePolicy<ExecSpace>(0, (size_t)DIM),
          KOKKOS_LAMBDA(size_t d) { last(d) = hn(C - 1, d); });
      // Untied lm_head through the shared head dispatch (dense gemv /
      // packed quant_gemv), then the output-multiplier + tanh softcap head:
      // logits = cap * tanh((lm_head(h) * mult) / cap).
      Impl::lm_head_forward<ExecSpace, Scalar>(handle, last, weights.lm_head, logits, "lm_head");
      scale_by_scalar<ExecSpace, Scalar>(logits, OUT_MULT, logits);
      logit_softcap_forward<ExecSpace, Scalar>(logits, SOFTCAP, logits);
    }
    seq_len += T;
    return logits;
  }

  // Host convenience: prefill + sampled decode loop.
  // Returns all ids (prompt + generated). Stops at any id in stop_ids
  // (the stop token itself is included) or max_new. An empty stop_ids
  // means just the config eos token.
  //
  // Sampling: `opt` selects greedy (temperature <= 0 or top_k == 1) vs
  // temperature / top-k / top-p / min-p sampling; `rng` supplies the draws
  // (a null rng with a non-greedy `opt` constructs HostRNG(opt.seed), so a
  // fixed seed reproduces the run). The repetition penalty looks at every
  // prior token (prompt + tokens generated so far, deduped). `on_token` is
  // the streaming hook: invoked once per generated token, in order,
  // immediately after it is sampled (before the next forward()).
  using TokenCallback = std::function<void(int)>;
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            const std::vector<int>& stop_ids,
                            const SampleOptions& opt, HostRNG* rng = nullptr,
                            TokenCallback on_token = nullptr) {
    reset();
    std::vector<int> out = prompt;
    std::vector<int> stops = stop_ids.empty()
                                  ? std::vector<int>{cfg.eos_token}
                                  : stop_ids;
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    auto is_stop = [&](int id) {
      return std::binary_search(stops.begin(), stops.end(), id);
    };
    const int max_seq = cfg.max_seq_len;
    HostRNG local_rng(opt.seed);
    HostRNG& r = rng ? *rng : local_rng;
    ensure_tok(prompt.size());
    auto toks = Kokkos::subview(tok_buf_, std::make_pair(size_t(0), prompt.size()));
    {
      auto h = Kokkos::subview(tok_host_, std::make_pair(size_t(0), prompt.size()));
      for (size_t i = 0; i < prompt.size(); ++i) h(i) = prompt[i];
      Kokkos::deep_copy(toks, h);
    }
    auto logits = forward(toks);
    int nxt = sample_token<ExecSpace, Scalar>(logits, opt, out, r);
    if ((int)out.size() < max_seq) out.push_back(nxt);
    if (on_token) on_token(nxt);
    if (one_buf_.extent(0) != 1) {
      one_buf_ = ViewInt("one", 1);
    }
    for (int s = 1; s < max_new; ++s) {
      if (is_stop(nxt)) break;
      if ((int)out.size() >= max_seq) break;
      // Device-side single-token staging: no host mirror + deep_copy per
      // token. Hoisted to a local: device lambdas must not capture `this`.
      // Logits likewise stay on device until sampling needs them (greedy
      // argmax runs on-device; only non-greedy sampling mirrors once).
      auto one = one_buf_;
      const int tok = nxt;
      Kokkos::parallel_for(
          "fill_one", Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(int i) {
            (void)i;
            one(0) = tok;
          });
      logits = forward(one_buf_);
      nxt = sample_token<ExecSpace, Scalar>(logits, opt, out, r);
      out.push_back(nxt);
      if (on_token) on_token(nxt);
    }
    return out;
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
};

}  // namespace Glimmer
}  // namespace KokkosDNN

#endif
