#ifndef KOKKOSDNN_LLAMA_MODEL_HPP
#define KOKKOSDNN_LLAMA_MODEL_HPP

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
#include "KokkosDNN_KVCache.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_RoPE.hpp"
#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_SwiGLU.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaWeights.hpp"

namespace KokkosDNN {
namespace Llama {

// Autoregressive Llama model. Owns KV caches and scratch.
// Template on ExecSpace so the same code runs on Serial/OpenMP/CUDA, and
// on Scalar (float or bhalf_t; default bhalf_t matches the 3.1 8B-Instruct
// checkpoint). All weights/activations/KV are Scalar; reductions/scales
// stay FP32; inv_freq stays FP32.
//
// KV-cache modes (Config::kv_page_size / kv_int8, fixed at construction):
//   * dense FP (defaults): per-layer [max_seq, n_kv, head_dim] rented
//     upfront, the legacy path.
//   * dense int8: same shape in int8 plus per-token per-head FP32 scales.
//   * paged FP / paged int8: per-layer pools of [pages, page, n_kv, head_dim]
//     addressed through an explicit identity block table, grown on demand so
//     a short prompt on a long-context config only rents the pages it
//     touches. Int8 halves KV bytes (plus scales) at the cost of small
//     numerical drift; paged dense is bit-exact vs dense.
//
// Memory note: BF16 8B weights are ~16 GiB (FP32 would be ~32 GiB), so a
// 24 GiB RTX 3090 can hold BF16 weights but not FP32. tiny_test() configs
// run anywhere; full 8B BF16 needs a large-GPU or CPU host allocation.
// KV cache per layer is [max_seq, n_kv, head_dim] (dense) or its paged /
// int8 equivalent; see Config::kv_cache_bytes().
template <class ExecSpace, class Scalar = DefaultScalar>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using HostInt = typename ViewInt::host_mirror_type;
  using KQ3 = Kokkos::View<int8_t***, Kokkos::LayoutRight, ExecSpace>;
  using KS2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using PagedPool = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using QPool = Kokkos::View<int8_t****, Kokkos::LayoutRight, ExecSpace>;
  using SPool = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;
  DNNHandle<ExecSpace> handle;
  Kokkos::View<float*, ExecSpace> inv_freq;
  // Dense FP caches (used iff !paged && !int8).
  std::vector<Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>> k_cache,
      v_cache;
  // Dense int8 caches + scales (used iff !paged && int8).
  std::vector<KQ3> kq_cache, vq_cache;
  std::vector<KS2> ks_cache, vs_cache;
  // Paged pools + identity block tables (used iff paged). FP pools for the
  // !int8 case, int8 + scale pools for the int8 case. Pools grow on demand
  // (kv_pages_alloc_ tracks resident pages per layer); tables are fixed at
  // the full logical page count.
  std::vector<PagedPool> kp_pool, vp_pool;
  std::vector<QPool> kqp_pool, vqp_pool;
  std::vector<SPool> ksp_pool, vsp_pool;
  std::vector<ViewInt> kv_table;
  std::vector<size_t> kv_pages_alloc_;
  size_t seq_len = 0;

  // Persistent scratch: all T-sized temporaries are carved (via subview,
  // which allocates nothing) out of these capacity buffers. Buffers grow
  // on demand in ensure_scratch() and are never shrunk, so the steady
  // decode state (T=1 after prefill) performs zero View allocations.
  // Fixed-size state (last/logits) is allocated once in the constructor.
  MatR h_buf_, tmp_buf_, tmp2_buf_, proj_buf_, down_buf_, hn_buf_;
  MatR q2_buf_, k2_buf_, v2_buf_, attn_flat_buf_;
  MatR gate_buf_, up_buf_, act_buf_;
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
    inv_freq = rope_inv_freq<ExecSpace>(
        cfg.head_dim, cfg.rope_theta, cfg.rope_scaling_type,
        cfg.rope_scaling_factor, cfg.rope_scaling_low_freq_factor,
        cfg.rope_scaling_high_freq_factor, cfg.rope_scaling_original_max);
    kv_alloc_fresh();
    last_buf_ = View1("last", cfg.dim);
    logits_buf_ = View1("logits", cfg.vocab_size);
    one_buf_ = ViewInt("one", 1);
  }

  // Full-window KV bytes for the configured mode (all layers, K+V).
  size_t kv_cache_bytes() const {
    return cfg.kv_cache_bytes(sizeof(Scalar));
  }
  // Currently resident KV bytes (paged pools only rent touched pages;
  // dense modes rent the full window upfront).
  size_t kv_allocated_bytes() const {
    const bool paged = cfg.kv_page_size > 0;
    const bool q = cfg.kv_int8;
    const size_t HD = (size_t)cfg.head_dim, NKV = (size_t)cfg.n_kv_heads;
    const size_t NL = (size_t)cfg.n_layers;
    if (!paged) return kv_cache_bytes();
    size_t bytes = 0;
    for (size_t l = 0; l < kv_pages_alloc_.size(); ++l) {
      const size_t toks = kv_pages_alloc_[l] * (size_t)cfg.kv_page_size;
      bytes += toks * NKV * HD * (q ? 1 : sizeof(Scalar)) * 2;
      if (q) bytes += toks * NKV * sizeof(float) * 2;
    }
    (void)NL;
    return bytes;
  }

  // O(1) reset: just rewinds the position counter. No cache wipe.
  // Correctness invariant: forward() writes KV at [cpos, cpos+C) before
  // reading prefix [0, cpos+C), with absolute positions and q_off=cpos,
  // so entries at/after seq_len are always overwritten before they are
  // read. Stale tail data from a previous sequence is never observed.
  // (Applies to every KV mode: paged pools keep their allocations.)
  void reset() { seq_len = 0; }

 private:
  // (Re)allocate KV caches for the construction-time cfg mode. Dense modes
  // rent the full window upfront; paged modes rent the identity block
  // tables upfront and grow the pools on demand in kv_ensure_capacity().
  void kv_alloc_fresh() {
    const bool paged = cfg.kv_page_size > 0;
    const bool q = cfg.kv_int8;
    const size_t MS = (size_t)cfg.max_seq_len;
    const size_t NKV = (size_t)cfg.n_kv_heads, HD = (size_t)cfg.head_dim;
    const int NL = cfg.n_layers;
    k_cache.clear(); v_cache.clear();
    kq_cache.clear(); vq_cache.clear(); ks_cache.clear(); vs_cache.clear();
    kp_pool.clear(); vp_pool.clear();
    kqp_pool.clear(); vqp_pool.clear(); ksp_pool.clear(); vsp_pool.clear();
    kv_table.clear(); kv_pages_alloc_.clear();
    if (!paged && !q) {
      k_cache.resize(NL); v_cache.resize(NL);
      for (int l = 0; l < NL; ++l) {
        k_cache[l] = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>(
            "k_cache", MS, NKV, HD);
        v_cache[l] = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>(
            "v_cache", MS, NKV, HD);
      }
    } else if (!paged && q) {
      kq_cache.resize(NL); vq_cache.resize(NL);
      ks_cache.resize(NL); vs_cache.resize(NL);
      for (int l = 0; l < NL; ++l) {
        kq_cache[l] = KQ3("kq_cache", MS, NKV, HD);
        vq_cache[l] = KQ3("vq_cache", MS, NKV, HD);
        ks_cache[l] = KS2("ks_cache", MS, NKV);
        vs_cache[l] = KS2("vs_cache", MS, NKV);
      }
    } else {
      const size_t page = (size_t)cfg.kv_page_size;
      const size_t npages = cfg.kv_num_pages();
      kv_table.resize(NL);
      kv_pages_alloc_.assign((size_t)NL, 0);
      for (int l = 0; l < NL; ++l) {
        kv_table[l] = ViewInt("kv_table", npages);
        KvDetail::fill_identity<ExecSpace>(kv_table[l]);
        if (!q) {
          kp_pool.emplace_back("kp_pool", size_t(0), page, NKV, HD);
          vp_pool.emplace_back("vp_pool", size_t(0), page, NKV, HD);
        } else {
          kqp_pool.emplace_back("kqp_pool", size_t(0), page, NKV, HD);
          vqp_pool.emplace_back("vqp_pool", size_t(0), page, NKV, HD);
          ksp_pool.emplace_back("ksp_pool", size_t(0), page, NKV);
          vsp_pool.emplace_back("vsp_pool", size_t(0), page, NKV);
        }
      }
      Kokkos::fence();
    }
  }

  // Grow paged pools so `tokens_needed` logical tokens are resident.
  // No-op for dense modes. Never shrinks; resident pages are reused across
  // reset() boundaries.
  void kv_ensure_capacity(size_t tokens_needed) {
    if (cfg.kv_page_size <= 0) return;
    const size_t page = (size_t)cfg.kv_page_size;
    const size_t npages = cfg.kv_num_pages();
    size_t need = (tokens_needed + page - 1) / page;
    if (need > npages) need = npages;
    if (need == 0) return;
    const bool q = cfg.kv_int8;
    const size_t NKV = (size_t)cfg.n_kv_heads, HD = (size_t)cfg.head_dim;
    for (int l = 0; l < cfg.n_layers; ++l) {
      if (kv_pages_alloc_[(size_t)l] >= need) continue;
      size_t have = kv_pages_alloc_[(size_t)l];
      size_t want = have == 0 ? need : have;
      while (want < need) {
        want *= 2;
        if (want > npages) { want = npages; break; }
      }
      if (!q) {
        PagedPool nk("kp_pool", want, page, NKV, HD);
        PagedPool nv("vp_pool", want, page, NKV, HD);
        if (have > 0) {
          KvDetail::copy_prefix_4<ExecSpace>(nk, kp_pool[(size_t)l]);
          KvDetail::copy_prefix_4<ExecSpace>(nv, vp_pool[(size_t)l]);
          Kokkos::fence();
        }
        kp_pool[(size_t)l] = nk;
        vp_pool[(size_t)l] = nv;
      } else {
        QPool nk("kqp_pool", want, page, NKV, HD);
        QPool nv("vqp_pool", want, page, NKV, HD);
        SPool sk("ksp_pool", want, page, NKV);
        SPool sv("vsp_pool", want, page, NKV);
        if (have > 0) {
          KvDetail::copy_prefix_4<ExecSpace>(nk, kqp_pool[(size_t)l]);
          KvDetail::copy_prefix_4<ExecSpace>(nv, vqp_pool[(size_t)l]);
          KvDetail::copy_prefix_3<ExecSpace>(sk, ksp_pool[(size_t)l]);
          KvDetail::copy_prefix_3<ExecSpace>(sv, vsp_pool[(size_t)l]);
          Kokkos::fence();
        }
        kqp_pool[(size_t)l] = nk;
        vqp_pool[(size_t)l] = nv;
        ksp_pool[(size_t)l] = sk;
        vsp_pool[(size_t)l] = sv;
      }
      kv_pages_alloc_[(size_t)l] = want;
    }
  }

  // Grow chunk-sized scratch to hold at least C rows (exact-size growth;
  // the first, largest prefill chunk sizes everything, later T=1 decodes
  // reuse it). Also repairs buffers if cfg dims changed under us.
  // With chunked prefill C = min(prefill_chunk_size, T), so a 128k prompt
  // no longer materializes O(T·dim) activations.
  void ensure_scratch(size_t T) {
    const size_t DIM = (size_t)cfg.dim;
    const size_t HqD = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
    const size_t HkvD = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    const size_t FFN = (size_t)cfg.ffn_dim;
    const size_t NH = (size_t)cfg.n_heads, NKV = (size_t)cfg.n_kv_heads,
                 HD = (size_t)cfg.head_dim;
    if (T <= scratch_cap_ && h_buf_.extent(0) >= T &&
        h_buf_.extent(1) == DIM && tmp_buf_.extent(1) == DIM &&
        tmp2_buf_.extent(1) == DIM && proj_buf_.extent(1) == DIM &&
        down_buf_.extent(1) == DIM && hn_buf_.extent(1) == DIM &&
        q2_buf_.extent(1) == HqD && attn_flat_buf_.extent(1) == HqD &&
        k2_buf_.extent(1) == HkvD && v2_buf_.extent(1) == HkvD &&
        gate_buf_.extent(1) == FFN && up_buf_.extent(1) == FFN &&
        act_buf_.extent(1) == FFN && q3_buf_.extent(1) == NH &&
        q3_buf_.extent(2) == HD && k3_buf_.extent(1) == NKV &&
        k3_buf_.extent(2) == HD && v3_buf_.extent(1) == NKV &&
        v3_buf_.extent(2) == HD && attn_out_buf_.extent(1) == NH &&
        attn_out_buf_.extent(2) == HD && pos_buf_.extent(0) >= T)
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
    gate_buf_ = MatR("gate", T, FFN);
    up_buf_ = MatR("up", T, FFN);
    act_buf_ = MatR("act", T, FFN);
    q3_buf_ = View3("q", T, NH, HD);
    k3_buf_ = View3("k", T, NKV, HD);
    v3_buf_ = View3("v", T, NKV, HD);
    attn_out_buf_ = View3("attn_out", T, NH, HD);
    pos_buf_ = ViewInt("pos", T);
    scratch_cap_ = T;
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
  // O(chunk·dim) instead of O(T·dim). Chunking is exact: each chunk runs
  // all layers for its tokens, reading previously written KV (earlier
  // chunks) plus its own prefix via the causal mask (q_off = chunk start),
  // which matches single-shot prefill token-for-token. Only the last
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
    const int DIM = cfg.dim, HD = cfg.head_dim, NH = cfg.n_heads,
              NKV = cfg.n_kv_heads, VS = cfg.vocab_size, NL = cfg.n_layers;
    const float EPS = cfg.rms_norm_eps, SCALE = cfg.attn_scale;
    const bool kv_paged = cfg.kv_page_size > 0;
    const bool kv_q = cfg.kv_int8;
    const size_t kv_page =
        kv_paged ? (size_t)cfg.kv_page_size : 0;
    // Resolve chunk size: <= 0 means single-shot.
    size_t Cmax = cfg.prefill_chunk_size <= 0
                      ? T
                      : (size_t)cfg.prefill_chunk_size;
    if (Cmax == 0) Cmax = T;
    if (Cmax > T) Cmax = T;
    ensure_scratch(Cmax);
    // Paged pools grow once per forward so the full [pos0, pos0+T) extent is
    // resident before the layer loop (dense modes: no-op). Never shrinks;
    // short prompts on a long-context config only rent touched pages.
    kv_ensure_capacity(pos0 + T);
    if (last_buf_.extent(0) != (size_t)DIM) last_buf_ = View1("last", DIM);
    if (logits_buf_.extent(0) != (size_t)VS)
      logits_buf_ = View1("logits", VS);

    // Hoisted member views: device lambdas below must capture these
    // locals, never `this`.
    auto last = last_buf_;
    auto logits = logits_buf_;

    const int HqD = NH * HD;
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
      auto proj = Kokkos::subview(proj_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto down = Kokkos::subview(down_buf_, std::make_pair(size_t(0), C),
                                  Kokkos::ALL);
      auto hn = Kokkos::subview(hn_buf_, std::make_pair(size_t(0), C),
                                Kokkos::ALL);
      auto toks =
          Kokkos::subview(tokens, std::make_pair(off, off + C));

      embedding_gather<ExecSpace, Scalar>(weights.embed, toks, h);

      Kokkos::parallel_for(
          "pos_fill", Kokkos::RangePolicy<ExecSpace>(0, C),
          KOKKOS_LAMBDA(size_t t) { pos(t) = (int)(cpos + t); });

      for (int l = 0; l < NL; ++l) {
        auto& b = weights.blocks[l];
        rmsnorm<ExecSpace, Scalar>(h, b.attn_norm, EPS, tmp);
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, b.wq,
                                static_cast<Scalar>(0), q2);
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, b.wk,
                                static_cast<Scalar>(0), k2);
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, b.wv,
                                static_cast<Scalar>(0), v2);
        Kokkos::parallel_for(
            "reshape_qkv",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)HqD}),
            KOKKOS_LAMBDA(size_t t, size_t i) {
              q3(t, i / HD, i % HD) = q2(t, i);
            });
        Kokkos::parallel_for(
            "reshape_kv",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                {0, 0}, {C, (size_t)HkvD}),
            KOKKOS_LAMBDA(size_t t, size_t i) {
              k3(t, i / HD, i % HD) = k2(t, i);
              v3(t, i / HD, i % HD) = v2(t, i);
            });
        rope_neoX_inplace<ExecSpace, Scalar>(q3, k3, inv_freq, HD, pos);
        // KV write + attention, branched once per layer on the
        // construction-time mode (hoisted to locals: indexing the member
        // vectors inside a device lambda would capture `this`, illegal on
        // CUDA; the copied View handles are device-safe).
        //   dense FP: legacy contiguous write + prefix-subview attention.
        //   dense int8: quantizing write + dequant-on-the-fly attention.
        //   paged FP / int8: block-table write + paged attention.
        // Paged dense is bit-exact vs dense; int8 drifts slightly.
        if (!kv_paged && !kv_q) {
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
          // layout-agnostic and streams over keys, so 128k prefixes stay O(1)
          // in extra memory here too. q_off is the chunk start so causal
          // masking (query t attends to keys [0, cpos + t]) matches
          // single-shot prefill exactly.
          auto kp = Kokkos::subview(kc, std::make_pair(size_t(0), cpos + C),
                                    Kokkos::ALL, Kokkos::ALL);
          auto vp = Kokkos::subview(vc, std::make_pair(size_t(0), cpos + C),
                                    Kokkos::ALL, Kokkos::ALL);
          attention_gqa_single<ExecSpace, Scalar>(handle, q3, kp, vp, cpos,
                                                  attn_out, SCALE);
        } else if (!kv_paged && kv_q) {
          auto kq = kq_cache[(size_t)l], vq = vq_cache[(size_t)l];
          auto ksc = ks_cache[(size_t)l], vsc = vs_cache[(size_t)l];
          kv_write_quantized_contiguous<ExecSpace, Scalar>(
              k3, v3, cpos, kq, ksc, vq, vsc);
          attention_gqa_quantized<ExecSpace, Scalar>(handle, q3, kq, ksc, vq,
                                                     vsc, cpos, attn_out,
                                                     SCALE);
        } else if (kv_paged && !kv_q) {
          auto kpp = kp_pool[(size_t)l], vpp = vp_pool[(size_t)l];
          auto tab = kv_table[(size_t)l];
          kv_write_paged_dense<ExecSpace, Scalar>(k3, v3, cpos, kpp, vpp, tab,
                                                  kv_page);
          attention_gqa_paged<ExecSpace, Scalar>(handle, q3, kpp, vpp, tab,
                                                 kv_page, cpos, attn_out,
                                                 SCALE);
        } else {
          auto kpp = kqp_pool[(size_t)l], vpp = vqp_pool[(size_t)l];
          auto ksp = ksp_pool[(size_t)l], vsp = vsp_pool[(size_t)l];
          auto tab = kv_table[(size_t)l];
          kv_write_paged_quantized<ExecSpace, Scalar>(
              k3, v3, cpos, kpp, ksp, vpp, vsp, tab, kv_page);
          attention_gqa_paged_quantized<ExecSpace, Scalar>(
              handle, q3, kpp, ksp, vpp, vsp, tab, kv_page, cpos, attn_out,
              SCALE);
        }
        Kokkos::parallel_for(
            "flatten",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)HqD}),
            KOKKOS_LAMBDA(size_t t, size_t i) {
              attn_flat(t, i) = attn_out(t, i / HD, i % HD);
            });
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), attn_flat,
                                b.wo, static_cast<Scalar>(0), proj);
        Kokkos::parallel_for(
            "resid1",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += proj(t, d); });
        rmsnorm<ExecSpace, Scalar>(h, b.ffn_norm, EPS, tmp2);
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp2, b.wgate,
                                static_cast<Scalar>(0), gate);
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp2, b.wup,
                                static_cast<Scalar>(0), up);
        swiglu<ExecSpace, Scalar>(gate, up, act);
        gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), act, b.wdown,
                                static_cast<Scalar>(0), down);
        Kokkos::parallel_for(
            "resid2",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {C, (size_t)DIM}),
            KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += down(t, d); });
      }

      // Only the last chunk produces logits (earlier chunks only fill KV).
      if (off + C < T) continue;
      rmsnorm<ExecSpace, Scalar>(h, weights.final_norm, EPS, hn);
      Kokkos::parallel_for(
          "gather_last", Kokkos::RangePolicy<ExecSpace>(0, (size_t)DIM),
          KOKKOS_LAMBDA(size_t d) { last(d) = hn(C - 1, d); });
      // lm_head through the shared gemv dispatch (cuBLAS M=1 on CUDA,
      // native kernel otherwise): same FP32 accumulation as the old naive
      // parallel_for(vocab), routed via the cached TPL handle.
      gemv<ExecSpace, Scalar>(handle, last, weights.lm_head, logits,
                              "lm_head");
    }
    seq_len += T;
    return logits;
  }

  // Host convenience: prefill + sampled decode loop.
  // Returns all ids (prompt + generated). Stops at any id in stop_ids
  // (the stop token itself is included) or max_new. An empty stop_ids
  // means the Instruct defaults {eos_token, eom_token, eot_token}.
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
                                 ? std::vector<int>{cfg.eos_token,
                                                    cfg.eom_token,
                                                    cfg.eot_token}
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
  // eos < 0 (default) uses the Instruct default stop set above.
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            int eos = -1) {
    if (eos < 0) return generate(prompt, max_new, std::vector<int>{});
    return generate(prompt, max_new, std::vector<int>{eos});
  }
};

}  // namespace Llama
}  // namespace KokkosDNN

#endif
