#ifndef KOKKOSDNN_GEMMA4_MODEL_HPP
#define KOKKOSDNN_GEMMA4_MODEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <algorithm>
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
#include "KokkosDNN_LayerScale.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_RoPE.hpp"
#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_Gemma4Config.hpp"
#include "KokkosDNN_Gemma4Weights.hpp"

namespace KokkosDNN {
namespace Gemma4 {
namespace Impl {

// Fused reshape + per-head QK RMSNorm (example-local: the library rmsnorm
// op works on [tokens, dim] rows, while q/k norms run per attention head
// over [tokens, heads, head_dim]).
//   q3[t,hd,d] = q2[t, hd*D+d] / rms(q2[t, hd*D:(hd+1)*D]) * w[d]
// Computed in FP32, same formula as KokkosDNN::rmsnorm.
template <class ExecSpace, class Scalar>
void qk_reshape_norm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& flat,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, ExecSpace>& out,
    const std::string& label = "Gemma4::qk_reshape_norm") {
  const size_t T = out.extent(0), H = out.extent(1), D = out.extent(2);
  if (flat.extent(0) != T || flat.extent(1) != H * D || w.extent(0) != D)
    throw std::invalid_argument("qk_reshape_norm: dimension mismatch");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(flat(t, h * D + d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          out(t, h, d) = static_cast<Scalar>(
              static_cast<float>(flat(t, h * D + d)) * inv *
              static_cast<float>(w(d)));
      });
}

// Fused reshape + scaleless RMSNorm for v (example-local counterpart of
// rmsnorm_noscale over the single KV head):
//   out[t,0,d] = flat[t,d] / rms(flat[t,:]).
template <class ExecSpace, class Scalar>
void v_reshape_norm_noscale(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& flat,
    float eps, const Kokkos::View<Scalar***, ExecSpace>& out,
    const std::string& label = "Gemma4::v_reshape_norm") {
  const size_t T = out.extent(0), D = out.extent(2);
  if (flat.extent(0) != T || flat.extent(1) != D || out.extent(1) != 1)
    throw std::invalid_argument("v_reshape_norm_noscale: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T),
      KOKKOS_LAMBDA(size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(flat(t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d)
          out(t, 0, d) =
              static_cast<Scalar>(static_cast<float>(flat(t, d)) * inv);
      });
}

// PLE context projection norm + token/context mix (example-local).
// Per (token t, layer l) over ple_dim d:
//   ctx_n[d] = ctx[t,l*P+d]*ctx_scale / rms(ctx[t,l*P:(l+1)*P]*ctx_scale) * w[d]
//   mix[t,l*P+d] = (ctx_n[d] + tok[t,l*P+d]) * mix_scale
// tok already carries the sqrt(ple_dim) token scale (applied at gather).
template <class ExecSpace, class Scalar>
void ple_ctx_norm_mix(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& ctx,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& tok,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps, float ctx_scale,
    float mix_scale, int n_layers, int ple_dim,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& mix,
    const std::string& label = "Gemma4::ple_ctx_norm_mix") {
  const size_t T = ctx.extent(0), P = (size_t)ple_dim, NL = (size_t)n_layers;
  if (ctx.extent(1) != NL * P || tok.extent(0) != T ||
      tok.extent(1) != NL * P || w.extent(0) != P || mix.extent(0) != T ||
      mix.extent(1) != NL * P)
    throw std::invalid_argument("ple_ctx_norm_mix: dimension mismatch");
  if (T == 0 || NL == 0 || P == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, NL}),
      KOKKOS_LAMBDA(size_t t, size_t l) {
        float ss = 0;
        for (size_t d = 0; d < P; ++d) {
          float v = static_cast<float>(ctx(t, l * P + d)) * ctx_scale;
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / (float)P + eps);
        for (size_t d = 0; d < P; ++d) {
          float cn = static_cast<float>(ctx(t, l * P + d)) * ctx_scale * inv *
                     static_cast<float>(w(d));
          float tk = static_cast<float>(tok(t, l * P + d));
          mix(t, l * P + d) = static_cast<Scalar>((cn + tk) * mix_scale);
        }
      });
}

// PLE per-layer residual multiply (example-local): g[t,d] *= mix[t,l*P+d].
// The GeLU(tanh) on g runs beforehand via the public gelu_forward op.
template <class ExecSpace, class Scalar>
void ple_gate_mul(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& g,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& mix,
    int layer, int ple_dim,
    const std::string& label = "Gemma4::ple_gate_mul") {
  const size_t T = g.extent(0), P = (size_t)ple_dim;
  const size_t off = (size_t)layer * P;
  if (g.extent(1) != P || mix.extent(0) != T || mix.extent(1) <= off + P)
    throw std::invalid_argument("ple_gate_mul: dimension mismatch");
  if (T == 0 || P == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, P}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        g(t, d) = static_cast<Scalar>(static_cast<float>(g(t, d)) *
                                      static_cast<float>(mix(t, off + d)));
      });
}

}  // namespace Impl

// Autoregressive Gemma4 text model. Owns KV caches and scratch.
// Template on ExecSpace so the same code runs on Serial/OpenMP/CUDA, and
// on Scalar (float or bhalf_t; default bhalf_t matches the E2B-it
// checkpoint). All weights/activations/KV are Scalar; reductions/scales
// stay FP32; inv_freq stays FP32.
//
// Gemma4 structure (see examples/gemma4/DESIGN.md):
//   * scaled token embedding (sqrt(hidden)) + per-layer-embedding (PLE)
//     context/token mix computed per chunk;
//   * hybrid attention: sliding layers (window 512, default RoPE theta 10k)
//     vs full layers (proportional partial RoPE theta 1e6, factor 0.25),
//     both with QK RMSNorms and attention scale 1.0;
//   * KV sharing: layers >= kv_shared_start() reuse the last non-shared
//     cache of the same type (sliding <- L13, full <- L14 for E2B-it);
//   * gated-GeLU MLP, double-wide (12288) on KV-shared layers;
//   * per-layer PLE residual (gate -> mul -> proj -> norm -> resid) and a
//     learned layer_scalar multiply;
//   * final RMSNorm + tied lm_head + final_logit_softcapping (30).
//
// KV caches are dense FP only in this step: one [max_seq, 1, head_dim]
// cache per non-shared layer (12 sliding + 3 full for E2B-it). Shared
// layers hold no cache of their own.
//
// Memory note: BF16 E2B-it weights are ~10 GiB, so the real checkpoint
// needs a large-GPU or CPU host allocation; tiny_test() configs run
// anywhere. See Config::kv_cache_bytes().
template <class ExecSpace, class Scalar = DefaultScalar>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using HostInt = typename ViewInt::host_mirror_type;
  using Cache3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;
  DNNHandle<ExecSpace> handle;
  Kokkos::View<float*, ExecSpace> inv_freq_sliding;  // [128], theta 10k
  Kokkos::View<float*, ExecSpace> inv_freq_full;     // [256], p-RoPE 1e6/0.25
  // Dense KV caches for the non-shared layers, split by type.
  std::vector<Cache3> k_sliding, v_sliding;  // [MS,1,256] each
  std::vector<Cache3> k_full, v_full;        // [MS,1,512] each
  // Per-layer cache index (into the same-type vector above). Layers >=
  // kv_shared_start() point at the donor's cache and never write.
  std::vector<int> cache_idx;
  size_t seq_len = 0;

  // Persistent scratch: all chunk-sized temporaries are carved (via
  // row-subview, which allocates nothing) out of these capacity buffers.
  // Buffers grow on demand in ensure_scratch() and are never shrunk, so the
  // steady decode state (T=1 after prefill) performs zero View allocations.
  // Widths are exact per attention type / MLP width (no column slicing, so
  // every carved view keeps its LayoutRight layout); only rows are carved.
  // Fixed-size state (last/logits) is allocated once in the constructor.
  MatR h_buf_, tmp_buf_, tmp2_buf_, proj_buf_, down_buf_, hn_buf_;
  MatR q2s_buf_, q2f_buf_, k2s_buf_, k2f_buf_, v2s_buf_, v2f_buf_;
  MatR gate6_buf_, up6_buf_, act6_buf_;
  MatR gate12_buf_, up12_buf_, act12_buf_;
  MatR afs_buf_, aff_buf_;  // flattened attention out, per type
  MatR ple_tok_buf_, ple_ctx_buf_, ple_mix_buf_;  // [cap, NL*ple]
  MatR g256_buf_, r1536_buf_;
  View3 q3s_buf_, q3f_buf_, k3s_buf_, k3f_buf_, v3s_buf_, v3f_buf_;
  View3 aos_buf_, aof_buf_;
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
    inv_freq_sliding = rope_inv_freq<ExecSpace>(
        cfg.head_dim_sliding, cfg.rope_theta_sliding, "", 1.0, 1.0, 4.0, 0);
    inv_freq_full = rope_inv_freq_proportional<ExecSpace>(
        cfg.head_dim_full, cfg.rope_theta_full, cfg.partial_rotary_factor);
    kv_alloc_fresh();
    last_buf_ = View1("last", cfg.dim);
    logits_buf_ = View1("logits", cfg.vocab_size);
    one_buf_ = ViewInt("one", 1);
  }

  // Full-window KV bytes for the configured mode (all caches, K+V).
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
  // Donor layer for a KV-shared layer: the last non-shared layer of the
  // same attention type (sliding <- L13, full <- L14 for E2B-it).
  int kv_donor(int layer) const {
    for (int d = cfg.kv_shared_start() - 1; d >= 0; --d)
      if (cfg.is_full(d) == cfg.is_full(layer)) return d;
    throw std::logic_error("Model::kv_donor: no donor layer");
  }

  // (Re)allocate KV caches: one dense [max_seq, 1, head_dim] K/V pair per
  // non-shared layer, grouped by attention type. Shared layers reuse the
  // donor's cache (cache_idx points at it, nothing is rented for them).
  void kv_alloc_fresh() {
    const size_t MS = (size_t)cfg.max_seq_len;
    const int NL    = cfg.n_layers;
    k_sliding.clear(); v_sliding.clear();
    k_full.clear(); v_full.clear();
    cache_idx.assign((size_t)NL, -1);
    // First pass: non-shared layers rent their own cache.
    for (int l = 0; l < NL; ++l) {
      if (cfg.is_kv_shared(l)) continue;
      if (!cfg.is_full(l)) {
        cache_idx[(size_t)l] = (int)k_sliding.size();
        k_sliding.emplace_back("k_sliding", MS, (size_t)cfg.n_kv_heads,
                               (size_t)cfg.head_dim_sliding);
        v_sliding.emplace_back("v_sliding", MS, (size_t)cfg.n_kv_heads,
                               (size_t)cfg.head_dim_sliding);
      } else {
        cache_idx[(size_t)l] = (int)k_full.size();
        k_full.emplace_back("k_full", MS, (size_t)cfg.n_kv_heads,
                            (size_t)cfg.head_dim_full);
        v_full.emplace_back("v_full", MS, (size_t)cfg.n_kv_heads,
                            (size_t)cfg.head_dim_full);
      }
    }
    // Second pass: shared layers alias the donor's cache index.
    for (int l = 0; l < NL; ++l) {
      if (!cfg.is_kv_shared(l)) continue;
      cache_idx[(size_t)l] = cache_idx[(size_t)kv_donor(l)];
    }
  }

  // Grow chunk-sized scratch to hold at least C rows (exact-size growth;
  // the first, largest prefill chunk sizes everything, later T=1 decodes
  // reuse it). With chunked prefill C = min(prefill_chunk_size, T), so a
  // long prompt no longer materializes O(T·dim) activations.
  void ensure_scratch(size_t C) {
    const size_t DIM = (size_t)cfg.dim;
    const size_t NL  = (size_t)cfg.n_layers;
    const size_t PLE = NL * (size_t)cfg.ple_dim;
    const size_t NH  = (size_t)cfg.n_heads;
    const size_t DS  = (size_t)cfg.head_dim_sliding;
    const size_t DF  = (size_t)cfg.head_dim_full;
    const size_t F6  = (size_t)cfg.ffn_dim;
    const size_t F12 = (size_t)2 * (size_t)cfg.ffn_dim;
    if (C <= scratch_cap_ && h_buf_.extent(0) >= C &&
        h_buf_.extent(1) == DIM && ple_mix_buf_.extent(1) == PLE &&
        q2f_buf_.extent(1) == NH * DF && gate12_buf_.extent(1) == F12 &&
        q3f_buf_.extent(2) == DF)
      return;
    h_buf_ = MatR("h", C, DIM);
    tmp_buf_ = MatR("tmp", C, DIM);
    tmp2_buf_ = MatR("tmp2", C, DIM);
    proj_buf_ = MatR("proj", C, DIM);
    down_buf_ = MatR("down", C, DIM);
    hn_buf_ = MatR("hn", C, DIM);
    q2s_buf_ = MatR("q2s", C, NH * DS);
    q2f_buf_ = MatR("q2f", C, NH * DF);
    k2s_buf_ = MatR("k2s", C, DS);
    k2f_buf_ = MatR("k2f", C, DF);
    v2s_buf_ = MatR("v2s", C, DS);
    v2f_buf_ = MatR("v2f", C, DF);
    gate6_buf_ = MatR("gate6", C, F6);
    up6_buf_ = MatR("up6", C, F6);
    act6_buf_ = MatR("act6", C, F6);
    gate12_buf_ = MatR("gate12", C, F12);
    up12_buf_ = MatR("up12", C, F12);
    act12_buf_ = MatR("act12", C, F12);
    afs_buf_ = MatR("afs", C, NH * DS);
    aff_buf_ = MatR("aff", C, NH * DF);
    ple_tok_buf_ = MatR("ple_tok", C, PLE);
    ple_ctx_buf_ = MatR("ple_ctx", C, PLE);
    ple_mix_buf_ = MatR("ple_mix", C, PLE);
    g256_buf_ = MatR("gple", C, (size_t)cfg.ple_dim);
    r1536_buf_ = MatR("rple", C, DIM);
    q3s_buf_ = View3("q3s", C, NH, DS);
    q3f_buf_ = View3("q3f", C, NH, DF);
    k3s_buf_ = View3("k3s", C, 1, DS);
    k3f_buf_ = View3("k3f", C, 1, DF);
    v3s_buf_ = View3("v3s", C, 1, DS);
    v3f_buf_ = View3("v3f", C, 1, DF);
    aos_buf_ = View3("aos", C, NH, DS);
    aof_buf_ = View3("aof", C, NH, DF);
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
  // token. With cuBLAS dispatch the match is within GEMM tolerance
  // (~1e-7 FP32): cuBLAS picks kernels by M, so M=C chunks round
  // differently than an M=T single shot. Only the last
  // chunk runs the final norm + lm_head. cfg.prefill_chunk_size <= 0
  // disables chunking (single-shot). Decode (T=1) is always one chunk.
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
    const int DIM = cfg.dim, NH = cfg.n_heads, VS = cfg.vocab_size,
              NL = cfg.n_layers, PLED = cfg.ple_dim;
    const float EPS = cfg.rms_norm_eps;
    const float EMBED_SCALE = cfg.embed_scale();
    const float PLE_TOK_SCALE = cfg.ple_token_scale();
    const float PLE_CTX_SCALE = cfg.ple_ctx_scale();
    const float PLE_MIX = Config::ple_mix_scale();
    const float SOFTCAP = cfg.final_logit_softcap;
    const size_t PLEW = (size_t)NL * (size_t)PLED;
    // Resolve chunk size: <= 0 means single-shot.
    size_t Cmax = cfg.prefill_chunk_size <= 0
                      ? T
                      : (size_t)cfg.prefill_chunk_size;
    if (Cmax == 0) Cmax = T;
    if (Cmax > T) Cmax = T;
    ensure_scratch(Cmax);
    if (last_buf_.extent(0) != (size_t)DIM) last_buf_ = View1("last", DIM);
    if (logits_buf_.extent(0) != (size_t)VS)
      logits_buf_ = View1("logits", VS);

    // Hoisted member views: device lambdas below must capture these
    // locals, never `this`.
    auto last = last_buf_;
    auto logits = logits_buf_;
    auto inv_s = inv_freq_sliding;
    auto inv_f = inv_freq_full;

    for (size_t off = 0; off < T; off += Cmax) {
      const size_t C = std::min(Cmax, T - off);
      const size_t cpos = pos0 + off;
      auto rows = std::make_pair(size_t(0), C);
      // Carve C-row working views out of the capacity buffers. Row-range
      // subviews share the allocation, so none of this allocates.
      auto h = Kokkos::subview(h_buf_, rows, Kokkos::ALL);
      auto tmp = Kokkos::subview(tmp_buf_, rows, Kokkos::ALL);
      auto tmp2 = Kokkos::subview(tmp2_buf_, rows, Kokkos::ALL);
      auto proj = Kokkos::subview(proj_buf_, rows, Kokkos::ALL);
      auto down = Kokkos::subview(down_buf_, rows, Kokkos::ALL);
      auto hn = Kokkos::subview(hn_buf_, rows, Kokkos::ALL);
      auto q2s = Kokkos::subview(q2s_buf_, rows, Kokkos::ALL);
      auto q2f = Kokkos::subview(q2f_buf_, rows, Kokkos::ALL);
      auto k2s = Kokkos::subview(k2s_buf_, rows, Kokkos::ALL);
      auto k2f = Kokkos::subview(k2f_buf_, rows, Kokkos::ALL);
      auto v2s = Kokkos::subview(v2s_buf_, rows, Kokkos::ALL);
      auto v2f = Kokkos::subview(v2f_buf_, rows, Kokkos::ALL);
      auto gate6 = Kokkos::subview(gate6_buf_, rows, Kokkos::ALL);
      auto up6 = Kokkos::subview(up6_buf_, rows, Kokkos::ALL);
      auto act6 = Kokkos::subview(act6_buf_, rows, Kokkos::ALL);
      auto gate12 = Kokkos::subview(gate12_buf_, rows, Kokkos::ALL);
      auto up12 = Kokkos::subview(up12_buf_, rows, Kokkos::ALL);
      auto act12 = Kokkos::subview(act12_buf_, rows, Kokkos::ALL);
      auto afs = Kokkos::subview(afs_buf_, rows, Kokkos::ALL);
      auto aff = Kokkos::subview(aff_buf_, rows, Kokkos::ALL);
      auto ple_tok = Kokkos::subview(ple_tok_buf_, rows, Kokkos::ALL);
      auto ple_ctx = Kokkos::subview(ple_ctx_buf_, rows, Kokkos::ALL);
      auto ple_mix = Kokkos::subview(ple_mix_buf_, rows, Kokkos::ALL);
      auto gple = Kokkos::subview(g256_buf_, rows, Kokkos::ALL);
      auto rple = Kokkos::subview(r1536_buf_, rows, Kokkos::ALL);
      auto q3s = Kokkos::subview(q3s_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto q3f = Kokkos::subview(q3f_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto k3s = Kokkos::subview(k3s_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto k3f = Kokkos::subview(k3f_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto v3s = Kokkos::subview(v3s_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto v3f = Kokkos::subview(v3f_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto aos = Kokkos::subview(aos_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto aof = Kokkos::subview(aof_buf_, rows, Kokkos::ALL, Kokkos::ALL);
      auto pos = Kokkos::subview(pos_buf_, rows);
      auto toks = Kokkos::subview(tokens, std::make_pair(off, off + C));

      // Scaled token embedding: h = sqrt(dim) * E[tok].
      embedding_gather_scaled<ExecSpace, Scalar>(weights.embed, toks,
                                                EMBED_SCALE, h);
      // PLE, computed once per chunk: token-identity part (sqrt(ple_dim) *
      // E_ple[tok], [C, NL*ple]) and context part (RMSNorm over 256-slices
      // of x_main @ W_ctx / sqrt(dim)), mixed as (ctx + tok) / sqrt(2).
      embedding_gather_scaled<ExecSpace, Scalar>(weights.ple_embed, toks,
                                                PLE_TOK_SCALE, ple_tok);
      gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), h,
                              weights.ple_ctx, static_cast<Scalar>(0), ple_ctx);
      Impl::ple_ctx_norm_mix<ExecSpace, Scalar>(
          ple_ctx, ple_tok, weights.ple_ctx_norm, EPS, PLE_CTX_SCALE, PLE_MIX,
          NL, PLED, ple_mix);

      Kokkos::parallel_for(
          "pos_fill", Kokkos::RangePolicy<ExecSpace>(0, C),
          KOKKOS_LAMBDA(size_t t) { pos(t) = (int)(cpos + t); });

      for (int l = 0; l < NL; ++l) {
        auto& b = weights.blocks[(size_t)l];
        const bool full = cfg.is_full(l);
        const int D = cfg.head_dim(l);
        const int HqD = NH * D;
        const int FD = cfg.mlp_dim(l);
        const bool shared = cfg.is_kv_shared(l);
        const int ci = cache_idx[(size_t)l];
        // Hoisted per-layer views (device-safe locals).
        auto wq = b.wq, wk = b.wk, wv = b.wv, wo = b.wo;
        auto qn = b.q_norm, kn = b.k_norm;
        auto wg = b.wgate, wu = b.wup, wd = b.wdown;
        auto pg = b.ple_gate, pp = b.ple_proj;
        auto in_n = b.input_norm, po_n = b.post_attn_norm;
        auto pf_n = b.pre_ffn_norm, qf_n = b.post_ffn_norm;
        auto pl_n = b.ple_out_norm;
        const float ls = b.layer_scalar;

        // 1. Attention block with QK RMSNorms.
        rmsnorm<ExecSpace, Scalar>(h, in_n, EPS, tmp);
        if (!full) {
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wq,
                                  static_cast<Scalar>(0), q2s);
          Impl::qk_reshape_norm<ExecSpace, Scalar>(q2s, qn, EPS, q3s);
          if (!shared) {
            gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wk,
                                    static_cast<Scalar>(0), k2s);
            gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wv,
                                    static_cast<Scalar>(0), v2s);
            Impl::qk_reshape_norm<ExecSpace, Scalar>(k2s, kn, EPS, k3s);
            Impl::v_reshape_norm_noscale<ExecSpace, Scalar>(v2s, EPS, v3s);
            rope_neoX_inplace<ExecSpace, Scalar>(q3s, k3s, inv_s, D, pos);
            auto kc = k_sliding[(size_t)ci];
            auto vc = v_sliding[(size_t)ci];
            Kokkos::parallel_for(
                "kcache_write",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    {0, 0}, {C, (size_t)D}),
                KOKKOS_LAMBDA(size_t t, size_t d) {
                  kc(cpos + t, 0, d) = k3s(t, 0, d);
                  vc(cpos + t, 0, d) = v3s(t, 0, d);
                });
            auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            attention_gqa_single_sliding<ExecSpace, Scalar>(
                handle, q3s, kp, vp, cpos, aos, 1.0f,
                (size_t)cfg.sliding_window);
          } else {
            // KV-shared sliding layer: q is projected/normed/RoPE'd as
            // usual, but K/V come from the donor's (L13) cache prefix,
            // which keeps growing there (store_full_length semantics).
            View3 empty_k3;
            rope_neoX_inplace<ExecSpace, Scalar>(q3s, empty_k3, inv_s, D, pos);
            auto kc = k_sliding[(size_t)ci];
            auto vc = v_sliding[(size_t)ci];
            auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            attention_gqa_single_sliding<ExecSpace, Scalar>(
                handle, q3s, kp, vp, cpos, aos, 1.0f,
                (size_t)cfg.sliding_window);
          }
          Kokkos::parallel_for(
              "flatten_s",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                  {0, 0}, {C, (size_t)HqD}),
              KOKKOS_LAMBDA(size_t t, size_t i) {
                afs(t, i) = aos(t, i / (size_t)D, i % (size_t)D);
              });
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), afs, wo,
                                  static_cast<Scalar>(0), proj);
        } else {
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wq,
                                  static_cast<Scalar>(0), q2f);
          Impl::qk_reshape_norm<ExecSpace, Scalar>(q2f, qn, EPS, q3f);
          if (!shared) {
            gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wk,
                                    static_cast<Scalar>(0), k2f);
            gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wv,
                                    static_cast<Scalar>(0), v2f);
            Impl::qk_reshape_norm<ExecSpace, Scalar>(k2f, kn, EPS, k3f);
            Impl::v_reshape_norm_noscale<ExecSpace, Scalar>(v2f, EPS, v3f);
            rope_neoX_inplace<ExecSpace, Scalar>(q3f, k3f, inv_f, D, pos);
            auto kc = k_full[(size_t)ci];
            auto vc = v_full[(size_t)ci];
            Kokkos::parallel_for(
                "kcache_write",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    {0, 0}, {C, (size_t)D}),
                KOKKOS_LAMBDA(size_t t, size_t d) {
                  kc(cpos + t, 0, d) = k3f(t, 0, d);
                  vc(cpos + t, 0, d) = v3f(t, 0, d);
                });
            auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            attention_gqa_single<ExecSpace, Scalar>(handle, q3f, kp, vp, cpos,
                                                   aof, 1.0f);
          } else {
            // KV-shared full layer: same, reading the donor's (L14) full
            // prefix cache.
            View3 empty_k3;
            rope_neoX_inplace<ExecSpace, Scalar>(q3f, empty_k3, inv_f, D, pos);
            auto kc = k_full[(size_t)ci];
            auto vc = v_full[(size_t)ci];
            auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                      Kokkos::ALL, Kokkos::ALL);
            attention_gqa_single<ExecSpace, Scalar>(handle, q3f, kp, vp, cpos,
                                                   aof, 1.0f);
          }
          Kokkos::parallel_for(
              "flatten_f",
              Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                  {0, 0}, {C, (size_t)HqD}),
              KOKKOS_LAMBDA(size_t t, size_t i) {
                aff(t, i) = aof(t, i / (size_t)D, i % (size_t)D);
              });
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), aff, wo,
                                  static_cast<Scalar>(0), proj);
        }
        // Post-attention norm (applied to attn @ Wo, unlike Llama's
        // pre-norm placement) + residual.
        rmsnorm<ExecSpace, Scalar>(proj, po_n, EPS, tmp2);
        Kokkos::parallel_for(
            "resid1",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += tmp2(t, d); });

        // 2. Gated-GeLU MLP (double-wide on KV-shared layers).
        rmsnorm<ExecSpace, Scalar>(h, pf_n, EPS, tmp);
        if (FD == cfg.ffn_dim) {
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wg,
                                  static_cast<Scalar>(0), gate6);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wu,
                                  static_cast<Scalar>(0), up6);
          gated_gelu_multiply<ExecSpace, Scalar>(gate6, up6, act6,
                                                 GeluVariant::Tanh);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), act6, wd,
                                  static_cast<Scalar>(0), down);
        } else {
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wg,
                                  static_cast<Scalar>(0), gate12);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wu,
                                  static_cast<Scalar>(0), up12);
          gated_gelu_multiply<ExecSpace, Scalar>(gate12, up12, act12,
                                                 GeluVariant::Tanh);
          gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), act12, wd,
                                  static_cast<Scalar>(0), down);
        }
        rmsnorm<ExecSpace, Scalar>(down, qf_n, EPS, tmp2);
        Kokkos::parallel_for(
            "resid2",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += tmp2(t, d); });

        // 3. PLE residual: g = gelu_tanh(h @ Wgate); g *= p_l;
        //    r = RMSNorm((g @ Wproj)); h += r.
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), h, pg,
                                static_cast<Scalar>(0), gple);
        gelu_forward<ExecSpace, Scalar>(gple, gple, GeluVariant::Tanh);
        Impl::ple_gate_mul<ExecSpace, Scalar>(gple, ple_mix, l, PLED);
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), gple, pp,
                                static_cast<Scalar>(0), rple);
        rmsnorm<ExecSpace, Scalar>(rple, pl_n, EPS, tmp2);
        Kokkos::parallel_for(
            "resid3",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += tmp2(t, d); });

        // 4. Learned per-layer scalar.
        scale_by_scalar<ExecSpace, Scalar>(h, ls, h);
        (void)PLEW;
      }

      // Only the last chunk produces logits (earlier chunks only fill KV).
      if (off + C < T) continue;
      rmsnorm<ExecSpace, Scalar>(h, weights.final_norm, EPS, hn);
      Kokkos::parallel_for(
          "gather_last", Kokkos::RangePolicy<ExecSpace>(0, (size_t)DIM),
          KOKKOS_LAMBDA(size_t d) { last(d) = hn(C - 1, d); });
      // Tied lm_head through the shared gemv dispatch (cuBLAS M=1 on CUDA,
      // native kernel otherwise), then final logit softcapping.
      gemv<ExecSpace, Scalar>(handle, last, weights.lm_head, logits,
                              "lm_head");
      logit_softcap_forward<ExecSpace, Scalar>(logits, SOFTCAP, logits);
    }
    seq_len += T;
    return logits;
  }

  // Host convenience: prefill + sampled decode loop.
  // Returns all ids (prompt + generated). Stops at any id in stop_ids
  // (the stop token itself is included) or max_new. An empty stop_ids
  // means the generation_config defaults {eos 1, 106, 50}.
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
    std::vector<int> stops = stop_ids.empty() ? cfg.default_stop_ids
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
  // eos < 0 (default) uses the generation_config default stop set above.
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            int eos = -1) {
    if (eos < 0) return generate(prompt, max_new, std::vector<int>{});
    return generate(prompt, max_new, std::vector<int>{eos});
  }
};

}  // namespace Gemma4
}  // namespace KokkosDNN

#endif
