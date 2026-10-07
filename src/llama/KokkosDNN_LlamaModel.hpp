#ifndef KOKKOSDNN_LLAMA_MODEL_HPP
#define KOKKOSDNN_LLAMA_MODEL_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_RoPE.hpp"
#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_SwiGLU.hpp"
#include "llama/KokkosDNN_LlamaConfig.hpp"
#include "llama/KokkosDNN_LlamaWeights.hpp"

namespace KokkosDNN {
namespace Llama {

// Autoregressive Llama model (FP32 PoC). Owns KV caches and scratch.
// Template on ExecSpace so the same code runs on Serial/OpenMP/CUDA.
//
// Memory note: FP32 8B weights are ~32 GiB; a 24 GiB RTX 3090 cannot hold
// them. tiny_test() configs run anywhere; full 8B needs a large-GPU or
// CPU host allocation. KV cache per layer is [max_seq, n_kv, head_dim].
template <class ExecSpace>
class Model {
 public:
  Config cfg;
  Weights<ExecSpace> weights;
  DNNHandle<ExecSpace> handle;
  Kokkos::View<float*, ExecSpace> inv_freq;
  std::vector<Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>> k_cache,
      v_cache;
  size_t seq_len = 0;

  Model() = default;
  Model(const Config& c, const Weights<ExecSpace>& w) : cfg(c), weights(w) {
    inv_freq = rope_inv_freq<ExecSpace>(cfg.head_dim, cfg.rope_theta);
    k_cache.resize(cfg.n_layers);
    v_cache.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      k_cache[l] = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>(
          "k_cache", cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim);
      v_cache[l] = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>(
          "v_cache", cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim);
    }
  }

  void reset() {
    seq_len = 0;
    for (auto& v : k_cache) Kokkos::deep_copy(v, 0.0f);
    for (auto& v : v_cache) Kokkos::deep_copy(v, 0.0f);
  }

  // Forward T prompt tokens starting at absolute position seq_len.
  // Returns logits [vocab] for the LAST token. Updates KV caches and
  // advances seq_len by T.
  Kokkos::View<float*, ExecSpace> forward(
      const Kokkos::View<int*, ExecSpace>& tokens) {
    const size_t T = tokens.extent(0);
    const size_t pos0 = seq_len;
    // Hoisted (device lambdas must not capture `this`/`cfg`).
    const int DIM = cfg.dim, HD = cfg.head_dim, NH = cfg.n_heads,
              NKV = cfg.n_kv_heads, FFN = cfg.ffn_dim, VS = cfg.vocab_size,
              NL = cfg.n_layers;
    const float EPS = cfg.rms_norm_eps, SCALE = cfg.attn_scale;
    using MatR = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;

    MatR h("h", T, DIM);
    embedding_gather<ExecSpace>(weights.embed, tokens, h);

    Kokkos::View<int*, ExecSpace> pos("pos", T);
    {
      auto hm = Kokkos::create_mirror_view(pos);
      for (size_t t = 0; t < T; ++t) hm(t) = (int)(pos0 + t);
      Kokkos::deep_copy(pos, hm);
    }

    MatR tmp("tmp", T, DIM), tmp2("tmp2", T, DIM);
    const int HqD = NH * HD;
    const int HkvD = NKV * HD;

    Kokkos::View<float***, ExecSpace> q3("q", T, NH, HD);
    Kokkos::View<float***, ExecSpace> k3("k", T, NKV, HD);
    Kokkos::View<float***, ExecSpace> v3("v", T, NKV, HD);
    MatR q2("q2", T, HqD), k2("k2", T, HkvD), v2("v2", T, HkvD);
    MatR gate("gate", T, FFN), up("up", T, FFN), act("act", T, FFN);

    for (int l = 0; l < NL; ++l) {
      auto& b = weights.blocks[l];
      rmsnorm<ExecSpace>(h, b.attn_norm, EPS, tmp);
      gemm<ExecSpace>(handle, 1.0f, tmp, b.wq, 0.0f, q2);
      gemm<ExecSpace>(handle, 1.0f, tmp, b.wk, 0.0f, k2);
      gemm<ExecSpace>(handle, 1.0f, tmp, b.wv, 0.0f, v2);
      Kokkos::parallel_for(
          "reshape_qkv",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                            {T, (size_t)HqD}),
          KOKKOS_LAMBDA(size_t t, size_t i) {
            q3(t, i / HD, i % HD) = q2(t, i);
          });
      Kokkos::parallel_for(
          "reshape_kv",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
              {0, 0}, {T, (size_t)HkvD}),
          KOKKOS_LAMBDA(size_t t, size_t i) {
            k3(t, i / HD, i % HD) = k2(t, i);
            v3(t, i / HD, i % HD) = v2(t, i);
          });
      rope_neoX_inplace<ExecSpace>(q3, k3, inv_freq, HD, pos);
      Kokkos::parallel_for(
          "kcache_write",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
              {0, 0, 0}, {T, (size_t)NKV, (size_t)HD}),
          KOKKOS_LAMBDA(size_t t, size_t hh, size_t d) {
            k_cache[l](pos0 + t, hh, d) = k3(t, hh, d);
            v_cache[l](pos0 + t, hh, d) = v3(t, hh, d);
          });
      Kokkos::View<float***, ExecSpace> attn_out("attn_out", T, NH, HD);
      Kokkos::View<float***, ExecSpace> kp("kp", pos0 + T, NKV, HD);
      Kokkos::View<float***, ExecSpace> vp("vp", pos0 + T, NKV, HD);
      auto kcl = k_cache[l];
      auto vcl = v_cache[l];
      Kokkos::parallel_for(
          "kcache_read",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
              {0, 0, 0}, {pos0 + T, (size_t)NKV, (size_t)HD}),
          KOKKOS_LAMBDA(size_t s, size_t hh, size_t d) {
            kp(s, hh, d) = kcl(s, hh, d);
            vp(s, hh, d) = vcl(s, hh, d);
          });
      attention_gqa_single<ExecSpace>(handle, q3, kp, vp, pos0, attn_out,
                                      SCALE);
      MatR attn_flat("attn_flat", T, HqD);
      Kokkos::parallel_for(
          "flatten",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                            {T, (size_t)HqD}),
          KOKKOS_LAMBDA(size_t t, size_t i) {
            attn_flat(t, i) = attn_out(t, i / HD, i % HD);
          });
      MatR proj("proj", T, DIM);
      gemm<ExecSpace>(handle, 1.0f, attn_flat, b.wo, 0.0f, proj);
      Kokkos::parallel_for(
          "resid1",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                            {T, (size_t)DIM}),
          KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += proj(t, d); });
      rmsnorm<ExecSpace>(h, b.ffn_norm, EPS, tmp2);
      gemm<ExecSpace>(handle, 1.0f, tmp2, b.wgate, 0.0f, gate);
      gemm<ExecSpace>(handle, 1.0f, tmp2, b.wup, 0.0f, up);
      swiglu<ExecSpace>(gate, up, act);
      MatR down("down", T, DIM);
      gemm<ExecSpace>(handle, 1.0f, act, b.wdown, 0.0f, down);
      Kokkos::parallel_for(
          "resid2",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                            {T, (size_t)DIM}),
          KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += down(t, d); });
    }

    MatR hn("hn", T, DIM);
    rmsnorm<ExecSpace>(h, weights.final_norm, EPS, hn);
    Kokkos::View<float*, ExecSpace> last("last", DIM);
    Kokkos::parallel_for(
        "gather_last", Kokkos::RangePolicy<ExecSpace>(0, (size_t)DIM),
        KOKKOS_LAMBDA(size_t d) { last(d) = hn(T - 1, d); });
    Kokkos::View<float*, ExecSpace> logits("logits", VS);
    {
      auto W = weights.lm_head;
      Kokkos::parallel_for(
          "lm_head", Kokkos::RangePolicy<ExecSpace>(0, (size_t)VS),
          KOKKOS_LAMBDA(size_t v) {
            float acc = 0;
            for (int d = 0; d < DIM; ++d) acc += last(d) * W(v, d);
            logits(v) = acc;
          });
    }
    seq_len += T;
    return logits;
  }

  // Host convenience: prefill + greedy decode loop.
  // Returns all ids (prompt + generated). Stops at eos or max_new.
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            int eos = -1) {
    reset();
    std::vector<int> out = prompt;
    if (eos < 0) eos = cfg.eos_token;
    const int max_seq = cfg.max_seq_len;
    Kokkos::View<int*, ExecSpace> toks("toks", prompt.size());
    {
      auto h = Kokkos::create_mirror_view(toks);
      for (size_t i = 0; i < prompt.size(); ++i) h(i) = prompt[i];
      Kokkos::deep_copy(toks, h);
    }
    auto logits = forward(toks);
    int nxt = sample_greedy<ExecSpace>(logits);
    if ((int)out.size() < max_seq) out.push_back(nxt);
    for (int s = 1; s < max_new; ++s) {
      if (nxt == eos) break;
      if ((int)out.size() >= max_seq) break;
      Kokkos::View<int*, ExecSpace> one("one", 1);
      {
        auto h = Kokkos::create_mirror_view(one);
        h(0) = nxt;
        Kokkos::deep_copy(one, h);
      }
      logits = forward(one);
      nxt = sample_greedy<ExecSpace>(logits);
      out.push_back(nxt);
    }
    return out;
  }
};

}  // namespace Llama
}  // namespace KokkosDNN

#endif
