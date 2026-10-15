#ifndef KOKKOSDNN_LFM25_MODEL_HPP
#define KOKKOSDNN_LFM25_MODEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_MoE.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_RoPE.hpp"
#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_SwiGLU.hpp"
#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Weights.hpp"

namespace KokkosDNN {
namespace LFM25 {
namespace Impl {

// Short-conv split of the in_proj output: in_proj maps [dim] -> 3 stacked
// [B, C, X] projections (llama.cpp `shortconv.in_proj`, {n_embd, 3*n_embd}).
//   bx[t,d]   = B[t,d] * X[t,d]   (the depthwise-conv input)
//   gate[t,d] = C[t,d]            (the output gate, applied after the conv)
template <class ExecSpace, class Scalar>
void shortconv_split_bcx(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& bcx,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& bx,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const std::string& label = "KokkosDNN::LFM25::shortconv_split") {
  const size_t C = bcx.extent(0), D = bx.extent(1);
  if (bcx.extent(1) != 3 * D || gate.extent(0) != C || gate.extent(1) != D ||
      bx.extent(0) != C)
    throw std::invalid_argument("shortconv_split_bcx: dimension mismatch");
  if (C == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float b = static_cast<float>(bcx(t, d));
        float x = static_cast<float>(bcx(t, 2 * D + d));
        bx(t, d)   = static_cast<Scalar>(b * x);
        gate(t, d) = bcx(t, D + d);
      });
}

// Causal depthwise convolution over the b*x product with explicit leading
// history (the conv state at chunk start):
//   out[t,c] = sum_j w[c,j] * src[t-(K-1)+j, c],
//   src[s,c] = s < 0 ? hist[c, s+K-1] : bx[s,c].
// With a zero history this equals depthwise_conv1d_causal_forward(bx, w)
// (see KokkosDNN_Conv1D.hpp); the history form additionally covers chunks
// after the first, whose left context is the previous chunk's tail.
// Scalar-agnostic; FP32 accumulation; Kokkos-portable MDRange kernel.
template <class ExecSpace, class Scalar>
void shortconv_hist_conv(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& bx,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hist,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::LFM25::shortconv_conv") {
  const size_t C = bx.extent(0), D = bx.extent(1), K = w.extent(1);
  if (w.extent(0) != D || out.extent(0) != C || out.extent(1) != D ||
      hist.extent(0) != D || hist.extent(1) != (K > 0 ? K - 1 : 0))
    throw std::invalid_argument("shortconv_hist_conv: dimension mismatch");
  if (C == 0 || D == 0 || K == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float acc = 0.0f;
        for (size_t j = 0; j < K; ++j) {
          const long s = static_cast<long>(t) -
                         static_cast<long>(K - 1) + static_cast<long>(j);
          float v = (s < 0) ? static_cast<float>(hist(d, s + K - 1))
                            : static_cast<float>(bx((size_t)s, d));
          acc += static_cast<float>(w(d, j)) * v;
        }
        out(t, d) = static_cast<Scalar>(acc);
      });
}

// Roll the persistent conv state forward: new_hist = last (K-1) rows of
// (hist ++ bx), i.e. the (K-1) most recent b*x values. Called once per
// chunk per conv layer, after the conv above consumed the old history.
// No-op for K == 1 (stateless 1-tap conv).
template <class ExecSpace, class Scalar>
void shortconv_state_update(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& bx,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hist,
    const std::string& label = "KokkosDNN::LFM25::shortconv_state") {
  const size_t C = bx.extent(0), D = bx.extent(1);
  const size_t Km1 = hist.extent(1);
  if (hist.extent(0) != D)
    throw std::invalid_argument("shortconv_state_update: dimension mismatch");
  if (D == 0 || Km1 == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {D, Km1}),
      KOKKOS_LAMBDA(size_t d, size_t j) {
        const long s = static_cast<long>(C) - static_cast<long>(Km1) +
                       static_cast<long>(j);
        hist(d, j) = (s < 0) ? hist(d, (size_t)(s + Km1)) : bx((size_t)s, d);
      });
}

// Add one expert's contribution: y[t,d] += sum_{k: idx[t,k]==e} w[t,k]*oe[t,d].
// The expert outputs oe cover all tokens (no gather needed); rows the expert
// was not routed to contribute nothing. Scalar-agnostic; FP32 accumulation.
template <class ExecSpace, class Scalar>
void moe_masked_accumulate(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& oe,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& w,
    int expert,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::LFM25::moe_accum") {
  const size_t C = y.extent(0), D = y.extent(1), K = w.extent(1);
  if (oe.extent(0) != C || oe.extent(1) != D || idx.extent(0) != C ||
      idx.extent(1) != K || w.extent(0) != C)
    throw std::invalid_argument("moe_masked_accumulate: dimension mismatch");
  if (C == 0 || D == 0 || K == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {C, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float a = 0.0f;
        for (size_t k = 0; k < K; ++k)
          if (idx(t, k) == expert)
            a += w(t, k) * static_cast<float>(oe(t, d));
        y(t, d) = static_cast<Scalar>(static_cast<float>(y(t, d)) + a);
      });
}

}  // namespace Impl

// Autoregressive LFM2.5 hybrid model. Owns KV caches (attention layers
// only), short-conv states (conv layers), and scratch.
// Template on ExecSpace so the same code runs on Serial/OpenMP/CUDA, and
// on Scalar (float or bhalf_t; default bhalf_t matches the GGUF checkpoint).
// All weights/activations/KV are Scalar; reductions/scales stay FP32;
// inv_freq stays FP32.
//
// Per layer (llama.cpp lfm2/lfm2moe build graph):
//   h = rmsnorm(h, attn_norm); then
//   attention layer: qkv gemm -> per-head qk_rmsnorm -> NeoX RoPE ->
//     causal GQA attention (dense KV prefix) -> wo proj, or
//   conv layer: in_proj -> split B/C/X -> bx = B*X -> causal depthwise conv
//     over bx (chunk history = persistent conv state) -> gate by C ->
//     out_proj; then residual, ffn_norm, and
//   dense SwiGLU FFN, or MoE (router gemm -> moe_route with the optional
//     expert bias, top-k norm and routing scale -> per-expert SwiGLU via
//     quant_gemm -> bias-weighted accumulation); then residual.
// Final rmsnorm + (possibly tied) lm_head gemv.
//
// KV cache: dense FP per attention layer ([max_seq_len, n_kv, head_dim],
// rented upfront). Conv layers rent no KV cache, only their fixed
// [dim, K-1] conv state (K = conv_L_cache). Paged/int8 KV modes are NOT
// supported here (the constructor throws); the Llama example covers those.
// Chunked prefill, zero-allocation steady decode, and the generate() loop
// follow the Llama Model conventions.
template <class ExecSpace, class Scalar = DefaultScalar>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using HostInt = typename ViewInt::host_mirror_type;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;
  DNNHandle<ExecSpace> handle;
  Kokkos::View<float*, ExecSpace> inv_freq;
  // Dense FP KV caches, indexed by global layer (empty for conv layers).
  std::vector<Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>> k_cache,
      v_cache;
  // Persistent short-conv states [dim, K-1] (bx history, oldest-first),
  // indexed by global layer (empty for attention layers).
  std::vector<MatR> conv_state_;
  size_t seq_len = 0;

  // Persistent scratch: all T-sized temporaries are carved (via subview,
  // which allocates nothing) out of these capacity buffers. Buffers grow
  // on demand in ensure_scratch() and are never shrunk, so the steady
  // decode state (T=1 after prefill) performs zero View allocations.
  // Fixed-size state (last/logits) is allocated once in the constructor.
  MatR h_buf_, tmp_buf_, tmp2_buf_, proj_buf_, down_buf_, hn_buf_;
  MatR q2_buf_, k2_buf_, v2_buf_, attn_flat_buf_;
  MatR gate_buf_, up_buf_, act_buf_;
  MatR sc_bcx_buf_, sc_bx_buf_, sc_gate_buf_, sc_conv_buf_, sc_y_buf_;
  MatR moe_logits_buf_, moe_oe_buf_;
  Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace> moe_idx_buf_;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> moe_w_buf_;
  View3 q3_buf_, k3_buf_, v3_buf_, attn_out_buf_;
  ViewInt pos_buf_;
  View1 last_buf_, logits_buf_;
  size_t scratch_cap_ = 0;
  size_t scratch_ffn_ = 0;  // max(ffn_dim, moe_intermediate) at sizing time
  size_t scratch_E_   = 0;  // n_experts at sizing time
  size_t scratch_K_   = 0;  // top_k at sizing time
  // Persistent input staging for generate(): prefill staging grows on
  // demand, the single-token decode slot is fixed at 1.
  ViewInt tok_buf_;
  HostInt tok_host_;
  size_t tok_cap_ = 0;
  ViewInt one_buf_;

  Model() = default;
  Model(const Config& c, const Weights<ExecSpace, Scalar>& w)
      : cfg(c), weights(w) {
    // Validate before renting any cache: incompatible dims or an
    // out-of-range context fail here, not in a device kernel or a
    // multi-GiB allocation.
    cfg.validate();
    if (cfg.kv_page_size > 0 || cfg.kv_int8)
      throw std::invalid_argument(
          "LFM25::Model: paged/int8 KV caches are not supported in this "
          "example (dense FP only); use the Llama example for those modes");
    inv_freq = rope_inv_freq<ExecSpace>(cfg.head_dim, cfg.rope_theta);
    kv_alloc_fresh();
    conv_alloc_fresh();
    last_buf_   = View1("last", cfg.dim);
    logits_buf_ = View1("logits", cfg.vocab_size);
    one_buf_    = ViewInt("one", 1);
  }

  // Full-window KV bytes (attention layers, K+V).
  size_t kv_cache_bytes() const {
    return cfg.kv_cache_bytes(sizeof(Scalar));
  }
  size_t kv_allocated_bytes() const { return kv_cache_bytes(); }

  // O(1)-ish reset: rewinds the position counter and zeroes the conv
  // states (unlike KV entries, stale conv history WOULD be observed: the
  // first chunk's left context reads history rows unconditionally).
  // KV caches need no wipe: forward() writes KV at [cpos, cpos+C) before
  // reading prefix [0, cpos+C), so stale tail data is never observed.
  void reset() {
    seq_len = 0;
    for (int l = 0; l < cfg.n_layers; ++l) {
      if (!cfg.is_attention(l) && conv_state_[(size_t)l].extent(0) != 0)
        Kokkos::deep_copy(conv_state_[(size_t)l], static_cast<Scalar>(0));
    }
  }

 private:
  // Rent dense FP KV caches for attention layers only.
  void kv_alloc_fresh() {
    const size_t MS  = (size_t)cfg.max_seq_len;
    const size_t NKV = (size_t)cfg.n_kv_heads, HD = (size_t)cfg.head_dim;
    k_cache.clear();
    v_cache.clear();
    k_cache.resize((size_t)cfg.n_layers);
    v_cache.resize((size_t)cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      if (!cfg.is_attention(l)) continue;
      k_cache[(size_t)l] =
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>(
              "k_cache", MS, NKV, HD);
      v_cache[(size_t)l] =
          Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>(
              "v_cache", MS, NKV, HD);
    }
  }

  // Rent (zeroed) conv states for conv layers only.
  void conv_alloc_fresh() {
    const size_t Km1 = cfg.conv_L_cache > 0 ? (size_t)cfg.conv_L_cache - 1 : 0;
    conv_state_.clear();
    conv_state_.resize((size_t)cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      if (cfg.is_attention(l)) continue;
      conv_state_[(size_t)l] =
          MatR("conv_state", (size_t)cfg.dim, Km1);
      if (Km1 > 0)
        Kokkos::deep_copy(conv_state_[(size_t)l], static_cast<Scalar>(0));
    }
  }

  // Grow chunk-sized scratch to hold at least C rows. Also repairs buffers
  // if the config dims changed under us.
  void ensure_scratch(size_t T) {
    const size_t DIM  = (size_t)cfg.dim;
    const size_t HqD  = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
    const size_t HkvD = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    const size_t FFN  = (size_t)cfg.ffn_dim;
    const size_t MI   = (size_t)cfg.moe_intermediate;
    const size_t FFNm = FFN > MI ? FFN : MI;
    const size_t E    = (size_t)cfg.n_experts;
    const size_t TK   = (size_t)cfg.top_k;
    const size_t NH = (size_t)cfg.n_heads, NKV = (size_t)cfg.n_kv_heads,
                 HD = (size_t)cfg.head_dim;
    if (T <= scratch_cap_ && scratch_ffn_ == FFNm && scratch_E_ == E &&
        scratch_K_ == TK && h_buf_.extent(0) >= T && h_buf_.extent(1) == DIM &&
        q2_buf_.extent(1) == HqD && k2_buf_.extent(1) == HkvD &&
        gate_buf_.extent(1) == FFNm && sc_bcx_buf_.extent(1) == 3 * DIM &&
        q3_buf_.extent(1) == NH && q3_buf_.extent(2) == HD &&
        k3_buf_.extent(1) == NKV && k3_buf_.extent(2) == HD &&
        pos_buf_.extent(0) >= T &&
        (E == 0 || (moe_logits_buf_.extent(1) == E &&
                    moe_idx_buf_.extent(1) == TK)))
      return;
    h_buf_ = MatR("h", T, DIM);
    tmp_buf_ = MatR("tmp", T, DIM);
    tmp2_buf_ = MatR("tmp2", T, DIM);
    proj_buf_ = MatR("proj", T, DIM);
    down_buf_ = MatR("down", T, DIM);
    hn_buf_ = MatR("hn", T, DIM);
    q2_buf_ = MatR("q2", T, HqD);
    k2_buf_ = MatR("k2", T, HkvD);
    v2_buf_ = MatR("v2", T, HkvD);
    attn_flat_buf_ = MatR("attn_flat", T, HqD);
    gate_buf_ = MatR("gate", T, FFNm);
    up_buf_ = MatR("up", T, FFNm);
    act_buf_ = MatR("act", T, FFNm);
    sc_bcx_buf_ = MatR("sc_bcx", T, 3 * DIM);
    sc_bx_buf_ = MatR("sc_bx", T, DIM);
    sc_gate_buf_ = MatR("sc_gate", T, DIM);
    sc_conv_buf_ = MatR("sc_conv", T, DIM);
    sc_y_buf_ = MatR("sc_y", T, DIM);
    if (E > 0) {
      moe_logits_buf_ = MatR("moe_logits", T, E);
      moe_idx_buf_    = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>(
          "moe_idx", T, TK);
      moe_w_buf_ = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
          "moe_w", T, TK);
      moe_oe_buf_ = MatR("moe_oe", T, DIM);
    }
    q3_buf_ = View3("q", T, NH, HD);
    k3_buf_ = View3("k", T, NKV, HD);
    v3_buf_ = View3("v", T, NKV, HD);
    attn_out_buf_ = View3("attn_out", T, NH, HD);
    pos_buf_ = ViewInt("pos", T);
    scratch_cap_ = T;
    scratch_ffn_ = FFNm;
    scratch_E_   = E;
    scratch_K_   = TK;
  }

  void ensure_tok(size_t n) {
    if (n <= tok_cap_ && tok_buf_.extent(0) >= n && tok_host_.extent(0) >= n)
      return;
    tok_buf_ = ViewInt("toks", n);
    tok_host_ = HostInt("toks_host", n);
    tok_cap_ = n;
  }

 public:
  // Forward T prompt tokens starting at absolute position seq_len.
  // Returns logits [vocab] for the LAST token. Updates KV caches and conv
  // states, and advances seq_len by T.
  //
  // Chunked prefill: prompts longer than cfg.prefill_chunk_size (> 0) are
  // processed in chunks, so activations stay O(chunk·dim). Chunking is
  // exact: each chunk runs all layers for its tokens, reading previously
  // written KV plus its own prefix via the causal mask (q_off = chunk
  // start), and conv chunks chain through the persistent conv state, which
  // matches single-shot prefill token-for-token. Only the last chunk runs
  // the final norm + lm_head. cfg.prefill_chunk_size <= 0 disables
  // chunking (single-shot). Decode (T=1) is always one chunk.
  //
  // The returned view aliases persistent scratch: it is valid until the
  // next forward() call, which overwrites it. No View allocation happens
  // here once scratch has grown to the chunk size (steady decode: zero
  // allocations).
  Kokkos::View<Scalar*, ExecSpace> forward(
      const Kokkos::View<int*, ExecSpace>& tokens) {
    const size_t T = tokens.extent(0);
    if (T == 0) throw std::invalid_argument("Model::forward: empty tokens");
    const size_t pos0 = seq_len;
    if (pos0 + T > (size_t)cfg.max_seq_len)
      throw std::runtime_error("Model::forward: sequence exceeds max_seq_len");
    // Hoisted (device lambdas must not capture `this`/`cfg`).
    const int DIM = cfg.dim, HD = cfg.head_dim, NH = cfg.n_heads,
              NKV = cfg.n_kv_heads, VS = cfg.vocab_size, NL = cfg.n_layers;
    const float EPS = cfg.rms_norm_eps, SCALE = cfg.attn_scale;
    const bool ROT_NORM_TOPK = cfg.norm_topk_prob;
    const float ROUTED_SCALE = cfg.routed_scaling_factor;
    const int TOP_K = cfg.top_k;
    const size_t MI  = (size_t)cfg.moe_intermediate;
    const size_t FFN = (size_t)cfg.ffn_dim;
    // Resolve chunk size: <= 0 means single-shot.
    size_t Cmax =
        cfg.prefill_chunk_size <= 0 ? T : (size_t)cfg.prefill_chunk_size;
    if (Cmax == 0) Cmax = T;
    if (Cmax > T) Cmax = T;
    ensure_scratch(Cmax);
    if (last_buf_.extent(0) != (size_t)DIM) last_buf_ = View1("last", DIM);
    if (logits_buf_.extent(0) != (size_t)VS)
      logits_buf_ = View1("logits", VS);

    // Hoisted member views: device lambdas below must capture these
    // locals, never `this`.
    auto last   = last_buf_;
    auto logits = logits_buf_;

    const int HqD  = NH * HD;
    const int HkvD = NKV * HD;

    for (size_t off = 0; off < T; off += Cmax) {
      const size_t C    = std::min(Cmax, T - off);
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
      auto up =
          Kokkos::subview(up_buf_, std::make_pair(size_t(0), C), Kokkos::ALL);
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
      auto proj = Kokkos::subview(proj_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto down = Kokkos::subview(down_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto hn =
          Kokkos::subview(hn_buf_, std::make_pair(size_t(0), C), Kokkos::ALL);
      auto sc_bcx = Kokkos::subview(sc_bcx_buf_, std::make_pair(size_t(0), C),
                                    Kokkos::ALL);
      auto sc_bx = Kokkos::subview(sc_bx_buf_, std::make_pair(size_t(0), C),
                                   Kokkos::ALL);
      auto sc_gate = Kokkos::subview(sc_gate_buf_, std::make_pair(size_t(0), C),
                                     Kokkos::ALL);
      auto sc_conv = Kokkos::subview(sc_conv_buf_, std::make_pair(size_t(0), C),
                                     Kokkos::ALL);
      auto sc_y =
          Kokkos::subview(sc_y_buf_, std::make_pair(size_t(0), C), Kokkos::ALL);
      auto toks = Kokkos::subview(tokens, std::make_pair(off, off + C));

      embedding_gather<ExecSpace, Scalar>(weights.embed, toks, h);

      Kokkos::parallel_for(
          "pos_fill", Kokkos::RangePolicy<ExecSpace>(0, C),
          KOKKOS_LAMBDA(size_t t) { pos(t) = (int)(cpos + t); });

      for (int l = 0; l < NL; ++l) {
        auto& b = weights.blocks[l];
        rmsnorm<ExecSpace, Scalar>(h, b.attn_norm, EPS, tmp);
        if (b.is_attention) {
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, b.wq,
                                  static_cast<Scalar>(0), q2);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, b.wk,
                                  static_cast<Scalar>(0), k2);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, b.wv,
                                  static_cast<Scalar>(0), v2);
          // Fused reshape + per-head RMSNorm (llama.cpp q/k layernorm,
          // applied before RoPE).
          qk_rmsnorm<ExecSpace, Scalar>(q2, b.q_norm, EPS, q3);
          qk_rmsnorm<ExecSpace, Scalar>(k2, b.k_norm, EPS, k3);
          // v2 is still flat: reshape into v3.
          Kokkos::parallel_for(
              "reshape_v",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                  {0, 0}, {C, (size_t)HkvD}),
              KOKKOS_LAMBDA(size_t t, size_t i) {
                v3(t, i / HD, i % HD) = v2(t, i);
              });
          rope_neoX_inplace<ExecSpace, Scalar>(q3, k3, inv_freq, HD, pos);
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
          // Valid KV prefix as subviews: no repack/copy. q_off is the chunk
          // start so causal masking matches single-shot prefill exactly.
          auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                    Kokkos::ALL, Kokkos::ALL);
          auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                    Kokkos::ALL, Kokkos::ALL);
          attention_gqa_single<ExecSpace, Scalar>(handle, q3, kp, vp, cpos,
                                                  attn_out, SCALE);
          Kokkos::parallel_for(
              "flatten",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                                {C, (size_t)HqD}),
              KOKKOS_LAMBDA(size_t t, size_t i) {
                attn_flat(t, i) = attn_out(t, i / HD, i % HD);
              });
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), attn_flat,
                                  b.wo, static_cast<Scalar>(0), proj);
        } else {
          // Short-conv mixer: in_proj -> split B/C/X -> bx = B*X ->
          // causal depthwise conv over bx (persistent state = history) ->
          // gate by C -> out_proj.
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp,
                                  b.in_proj, static_cast<Scalar>(0), sc_bcx);
          Impl::shortconv_split_bcx<ExecSpace, Scalar>(sc_bcx, sc_bx, sc_gate);
          Impl::shortconv_hist_conv<ExecSpace, Scalar>(
              sc_bx, conv_state_[(size_t)l], b.conv_dw, sc_conv);
          Kokkos::parallel_for(
              "sc_gate",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                                {C, (size_t)DIM}),
              KOKKOS_LAMBDA(size_t t, size_t d) {
                sc_y(t, d) = static_cast<Scalar>(
                    static_cast<float>(sc_gate(t, d)) *
                    static_cast<float>(sc_conv(t, d)));
              });
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), sc_y,
                                  b.out_proj, static_cast<Scalar>(0), proj);
          // Advance the persistent state past this chunk (chains chunks).
          Impl::shortconv_state_update<ExecSpace, Scalar>(
              sc_bx, conv_state_[(size_t)l]);
        }
        Kokkos::parallel_for(
            "resid1",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += proj(t, d); });
        rmsnorm<ExecSpace, Scalar>(h, b.ffn_norm, EPS, tmp2);
        if (!b.is_moe) {
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp2, b.wgate,
                                  static_cast<Scalar>(0), gate);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp2, b.wup,
                                  static_cast<Scalar>(0), up);
          swiglu<ExecSpace, Scalar>(gate, up, act);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), act, b.wdown,
                                  static_cast<Scalar>(0), down);
        } else {
          // MoE: router -> top-k route (optional expert bias, top-k norm,
          // routing scale) -> per-expert SwiGLU via packed quant_gemm ->
          // weight-masked accumulation. O(n_experts) GEMMs per layer: the
          // PoC trades kernel count for zero host synchronization (no
          // gather/scatter index juggling).
          auto mlogits = Kokkos::subview(moe_logits_buf_,
                                         std::make_pair(size_t(0), C),
                                         Kokkos::ALL);
          auto midx = Kokkos::subview(moe_idx_buf_,
                                      std::make_pair(size_t(0), C),
                                      Kokkos::ALL);
          auto mw = Kokkos::subview(moe_w_buf_,
                                    std::make_pair(size_t(0), C), Kokkos::ALL);
          auto moe_oe = Kokkos::subview(moe_oe_buf_,
                                        std::make_pair(size_t(0), C),
                                        Kokkos::ALL);
          // Narrow expert-width working views out of the FFNm buffers.
          auto egate = Kokkos::subview(
              gate, Kokkos::ALL, std::make_pair(size_t(0), MI));
          auto eup = Kokkos::subview(up, Kokkos::ALL,
                                     std::make_pair(size_t(0), MI));
          auto eact = Kokkos::subview(act, Kokkos::ALL,
                                      std::make_pair(size_t(0), MI));
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp2,
                                  b.gate_inp, static_cast<Scalar>(0), mlogits);
          moe_route<ExecSpace, Scalar>(mlogits, b.exp_bias, TOP_K,
                                       ROT_NORM_TOPK, ROUTED_SCALE, midx, mw);
          Kokkos::deep_copy(down, static_cast<Scalar>(0));
          const size_t NE = b.gate_exps.size();
          for (size_t e = 0; e < NE; ++e) {
            Gguf::quant_gemm<ExecSpace, Scalar>(handle, tmp2, b.gate_exps[e],
                                                egate);
            Gguf::quant_gemm<ExecSpace, Scalar>(handle, tmp2, b.up_exps[e],
                                                eup);
            swiglu<ExecSpace, Scalar>(egate, eup, eact);
            Gguf::quant_gemm<ExecSpace, Scalar>(handle, eact, b.down_exps[e],
                                                moe_oe);
            Impl::moe_masked_accumulate<ExecSpace, Scalar>(
                moe_oe, midx, mw, (int)e, down);
          }
          (void)FFN;
        }
        Kokkos::parallel_for(
            "resid2",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += down(t, d); });
      }

      // Only the last chunk produces logits (earlier chunks only fill KV
      // and conv states).
      if (off + C < T) continue;
      rmsnorm<ExecSpace, Scalar>(h, weights.final_norm, EPS, hn);
      Kokkos::parallel_for(
          "gather_last", Kokkos::RangePolicy<ExecSpace>(0, (size_t)DIM),
          KOKKOS_LAMBDA(size_t d) { last(d) = hn(C - 1, d); });
      // lm_head through the shared gemv dispatch (cuBLAS M=1 on CUDA,
      // native kernel otherwise), routed via the cached TPL handle.
      gemv<ExecSpace, Scalar>(handle, last, weights.lm_head, logits,
                              "lm_head");
    }
    seq_len += T;
    return logits;
  }

  // Host convenience: prefill + sampled decode loop.
  // Returns all ids (prompt + generated). Stops at any id in stop_ids
  // (the stop token itself is included) or max_new. An empty stop_ids
  // means the default {eos}.
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
    std::vector<int> stops =
        stop_ids.empty() ? std::vector<int>{cfg.eos} : stop_ids;
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    auto is_stop = [&](int id) {
      return std::binary_search(stops.begin(), stops.end(), id);
    };
    const int max_seq = cfg.max_seq_len;
    HostRNG local_rng(opt.seed);
    HostRNG& r = rng ? *rng : local_rng;
    ensure_tok(prompt.size());
    auto toks =
        Kokkos::subview(tok_buf_, std::make_pair(size_t(0), prompt.size()));
    {
      auto h =
          Kokkos::subview(tok_host_, std::make_pair(size_t(0), prompt.size()));
      for (size_t i = 0; i < prompt.size(); ++i) h(i) = prompt[i];
      Kokkos::deep_copy(toks, h);
    }
    auto logits = forward(toks);
    int nxt     = sample_token<ExecSpace, Scalar>(logits, opt, out, r);
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
      auto one      = one_buf_;
      const int tok = nxt;
      Kokkos::parallel_for(
          "fill_one", Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(int i) {
            (void)i;
            one(0) = tok;
          });
      logits = forward(one_buf_);
      nxt    = sample_token<ExecSpace, Scalar>(logits, opt, out, r);
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
  // eos < 0 (default) uses the default stop set above.
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            int eos = -1) {
    if (eos < 0) return generate(prompt, max_new, std::vector<int>{});
    return generate(prompt, max_new, std::vector<int>{eos});
  }
};

}  // namespace LFM25
}  // namespace KokkosDNN

#endif
