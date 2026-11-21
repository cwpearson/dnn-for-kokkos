// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Inkling 2 Mini text decoder: Model forward vs naive FP32 host reference,
// prefill-vs-incremental-decode consistency, sliding-window masking, and a
// Session smoke test. All weights are tiny synthetic random values; no
// checkpoint is needed.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <Kokkos_Random.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Mock Config/Weights/Tokenizer with EXACTLY the sibling contract names.
// These stand in for the sibling-owned models/inkling headers: they are
// defined BEFORE including the Model/Session headers, so the default
// template arguments (Config, Weights<Exec,Scalar>, Tokenizer) resolve to
// these mocks.
// ---------------------------------------------------------------------------
namespace KokkosDNN {
namespace Inkling {

struct Config {
  int n_layers = 2;
  int d_model = 32;
  int n_heads = 4;
  int n_kv_heads = 2;
  int head_dim = 8;
  int d_rel = 4;
  int rel_extent = 16;
  int vocab_size = 64;
  int unpadded_vocab = 48;
  int n_experts = 2;
  int top_k = 1;
  int intermediate_size = 16;
  int sconv_kernel = 3;
  int swa_window = 8;
  int loop_start = 1;
  int loop_end = 2;
  int loop_passes = 1;
  int gate_rank = 8;
  int attn_res_block = 6;
  float eps = 1e-6f;
  float qk_clip = 22.627f;
  float rel_clip = 8.0f;
  int log_floor = 8192;
  float log_alpha = 0.1f;
  float gate_softcap = 4.0f;
  float up_softcap = 25.0f;
  float final_softcap = 30.0f;
  float route_scale = 8.0f;
  bool use_gate_bias = true;
  bool attn_res_normalize = true;
  std::vector<int> local_layer_ids = {0};
  int max_seq_len = 32;
  int prefill_chunk_size = 0;
  int eos_token = 1;
  void validate() const {
    if (n_layers <= 0 || d_model <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || d_rel <= 0 || unpadded_vocab <= 0 || n_experts <= 0 ||
        top_k <= 0 || top_k > n_experts || intermediate_size <= 0 ||
        sconv_kernel <= 0 || max_seq_len <= 0 || loop_passes < 1)
      throw std::invalid_argument("mock Config::validate");
  }
  size_t kv_cache_bytes(size_t scalar_bytes) const {
    return (size_t)n_layers * (size_t)max_seq_len * (size_t)n_kv_heads *
           (size_t)head_dim * scalar_bytes * 2;
  }
};

template <class E, class S>
struct LayerWeights {
  using MatR = Kokkos::View<S**, Kokkos::LayoutRight, E>;
  using MatF = Kokkos::View<float**, Kokkos::LayoutRight, E>;
  using VecS = Kokkos::View<S*, E>;
  MatR wq, wk, wv, wgate_attn, wo, wrel, wsink, router;
  VecS router_bias;
  MatF rel_proj, q_scale, rel_scale;
  MatF k_conv, v_conv, k_conv_static, v_conv_static;
  MatF k_conv_proj, v_conv_proj, post_attn_conv, post_mlp_conv;
  VecS attn_norm, q_norm, k_norm, v_norm, post_attn_norm;
  MatF attn_res_q, mlp_res_q;
  std::vector<MatR> gate_up_exps, down_exps;
  MatF loop_q_down, loop_k_down, loop_v_down, loop_mlp_down;
  MatF loop_q_forget, loop_k_forget, loop_v_forget, loop_mlp_forget;
  MatF loop_q_gain, loop_k_gain, loop_v_gain, loop_mlp_gain;
  MatF msg_query;
};

template <class E, class S>
struct Weights {
  using MatR = Kokkos::View<S**, Kokkos::LayoutRight, E>;
  using VecS = Kokkos::View<S*, E>;
  using MatF = Kokkos::View<float**, Kokkos::LayoutRight, E>;
  MatR tok_emb, out_w;
  VecS tok_norm, out_norm;
  MatF out_res_q;
  std::vector<LayerWeights<E, S>> blocks;
  void allocate(const Config& cfg) {
    tok_emb = MatR("tok_emb", cfg.vocab_size, cfg.d_model);
    tok_norm = VecS("tok_norm", cfg.d_model);
    out_w = MatR("out_w", cfg.unpadded_vocab, cfg.d_model);
    out_norm = VecS("out_norm", cfg.d_model);
    out_res_q = MatF("out_res_q", 1, cfg.d_model);
    blocks.resize((size_t)cfg.n_layers);
  }
};

class Tokenizer {
 public:
  std::string decode(const std::vector<int>& ids, bool) const {
    std::string s;
    for (int id : ids) {
      s += "<" + std::to_string(id) + ">";
    }
    return s;
  }
};

inline std::vector<std::string> default_stop_strings() {
  return {"<|content_model_end_sampling|>"};
}

}  // namespace Inkling
}  // namespace KokkosDNN

#include "KokkosDNN_InklingModel.hpp"
#include "KokkosDNN_InklingMoeDecode.hpp"
#include "KokkosDNN_InklingSession.hpp"

namespace {

namespace Gguf = KokkosDNN::Gguf;

using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = float;
using Config = KokkosDNN::Inkling::Config;
template <class S>
using Weights = KokkosDNN::Inkling::Weights<Exec, S>;
template <class S>
using LayerW = KokkosDNN::Inkling::LayerWeights<Exec, S>;
using Model = KokkosDNN::Inkling::Model<Exec, Scalar, Config, Weights<Scalar>>;

// Fill a device view with uniform random values via its host mirror.
template <class View>
void fill_rand(View& v, uint64_t& seed, float lo = -1.0f, float hi = 1.0f) {
  auto h = Kokkos::create_mirror_view(v);
  uint64_t s = seed;
  for (size_t i = 0; i < h.size(); ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u =
        (double)(s >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
    h.data()[i] = static_cast<typename View::value_type>(
        lo + (hi - lo) * (float)u);
  }
  seed = s;
  Kokkos::deep_copy(v, h);
}

void random_weights(const Config& cfg, Weights<Scalar>& w, uint64_t seed = 7) {
  w.allocate(cfg);
  const size_t D = (size_t)cfg.d_model;
  const size_t HqD = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t HkvD = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  const size_t HR = (size_t)cfg.n_heads * (size_t)cfg.d_rel;
  const size_t E = (size_t)cfg.n_experts, I = (size_t)cfg.intermediate_size;
  const size_t K = (size_t)cfg.sconv_kernel;
  const size_t R = (size_t)cfg.gate_rank;
  fill_rand(w.tok_emb, seed);
  fill_rand(w.tok_norm, seed, 0.5f, 1.5f);
  fill_rand(w.out_w, seed, -0.1f, 0.1f);
  fill_rand(w.out_norm, seed, 0.5f, 1.5f);
  fill_rand(w.out_res_q, seed, -0.1f, 0.1f);
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b = w.blocks[(size_t)l];
    const bool full = KokkosDNN::Inkling::Impl::is_full_layer(
        cfg.local_layer_ids, cfg.n_layers, l);
    const bool loop =
        (l >= cfg.loop_start && l < cfg.loop_end);
    b.wq = typename LayerW<Scalar>::MatR("wq", HqD, D);
    b.wk = typename LayerW<Scalar>::MatR("wk", HkvD, D);
    b.wv = typename LayerW<Scalar>::MatR("wv", HkvD, D);
    b.wgate_attn = typename LayerW<Scalar>::MatR("wgate_attn", HqD, D);
    b.wo = typename LayerW<Scalar>::MatR("wo", D, HqD);
    b.wrel = typename LayerW<Scalar>::MatR("wrel", HR, D);
    b.wsink = typename LayerW<Scalar>::MatR("wsink", cfg.n_heads, D);
    b.router = typename LayerW<Scalar>::MatR("router", E, D);
    b.router_bias = typename LayerW<Scalar>::VecS("router_bias", E);
    b.rel_proj = typename LayerW<Scalar>::MatF("rel_proj", cfg.d_rel,
                                               cfg.rel_extent);
    b.q_scale = typename LayerW<Scalar>::MatF("q_scale", cfg.n_heads,
                                              cfg.head_dim);
    b.rel_scale = typename LayerW<Scalar>::MatF("rel_scale", cfg.n_heads,
                                                cfg.d_rel);
    fill_rand(b.wq, seed, -0.2f, 0.2f);
    fill_rand(b.wk, seed, -0.2f, 0.2f);
    fill_rand(b.wv, seed, -0.2f, 0.2f);
    fill_rand(b.wgate_attn, seed, -0.2f, 0.2f);
    fill_rand(b.wo, seed, -0.2f, 0.2f);
    fill_rand(b.wrel, seed, -0.2f, 0.2f);
    fill_rand(b.wsink, seed, -0.2f, 0.2f);
    fill_rand(b.router, seed, -0.5f, 0.5f);
    fill_rand(b.router_bias, seed, -0.2f, 0.2f);
    fill_rand(b.rel_proj, seed, -0.2f, 0.2f);
    fill_rand(b.q_scale, seed, 0.5f, 1.5f);
    fill_rand(b.rel_scale, seed, 0.5f, 1.5f);
    if (!full) {
      b.k_conv = typename LayerW<Scalar>::MatF("k_conv", HkvD, K);
      b.v_conv = typename LayerW<Scalar>::MatF("v_conv", HkvD, K);
      fill_rand(b.k_conv, seed, -0.2f, 0.2f);
      fill_rand(b.v_conv, seed, -0.2f, 0.2f);
      b.attn_norm = typename LayerW<Scalar>::VecS("attn_norm", D);
      fill_rand(b.attn_norm, seed, 0.5f, 1.5f);
    } else {
      b.k_conv_static =
          typename LayerW<Scalar>::MatF("k_conv_static", K, HkvD);
      b.v_conv_static =
          typename LayerW<Scalar>::MatF("v_conv_static", K, HkvD);
      b.k_conv_proj = typename LayerW<Scalar>::MatF(
          "k_conv_proj", (size_t)cfg.n_kv_heads * K, D);
      b.v_conv_proj = typename LayerW<Scalar>::MatF(
          "v_conv_proj", (size_t)cfg.n_kv_heads * K, D);
      fill_rand(b.k_conv_static, seed, -0.2f, 0.2f);
      fill_rand(b.v_conv_static, seed, -0.2f, 0.2f);
      fill_rand(b.k_conv_proj, seed, -0.2f, 0.2f);
      fill_rand(b.v_conv_proj, seed, -0.2f, 0.2f);
      b.q_norm = typename LayerW<Scalar>::VecS("q_norm", D);
      b.k_norm = typename LayerW<Scalar>::VecS("k_norm", D);
      b.v_norm = typename LayerW<Scalar>::VecS("v_norm", D);
      fill_rand(b.q_norm, seed, 0.5f, 1.5f);
      fill_rand(b.k_norm, seed, 0.5f, 1.5f);
      fill_rand(b.v_norm, seed, 0.5f, 1.5f);
    }
    b.post_attn_conv = typename LayerW<Scalar>::MatF("post_attn_conv", D,
                                                     K);
    b.post_mlp_conv = typename LayerW<Scalar>::MatF("post_mlp_conv", D, K);
    fill_rand(b.post_attn_conv, seed, -0.2f, 0.2f);
    fill_rand(b.post_mlp_conv, seed, -0.2f, 0.2f);
    b.post_attn_norm = typename LayerW<Scalar>::VecS("post_attn_norm", D);
    fill_rand(b.post_attn_norm, seed, 0.5f, 1.5f);
    const size_t nq = full ? 3 : 1;
    const bool need_attn_q =
        (full && l > 0) || (!full && !(!loop && l == 0));
    if (need_attn_q) {
      b.attn_res_q = typename LayerW<Scalar>::MatF("attn_res_q", nq, D);
      fill_rand(b.attn_res_q, seed, -0.1f, 0.1f);
    } else {
      b.attn_res_q = typename LayerW<Scalar>::MatF("attn_res_q", 0, D);
    }
    b.mlp_res_q = typename LayerW<Scalar>::MatF("mlp_res_q", 1, D);
    fill_rand(b.mlp_res_q, seed, -0.1f, 0.1f);
    b.gate_up_exps.resize(E);
    b.down_exps.resize(E);
    for (size_t e = 0; e < E; ++e) {
      b.gate_up_exps[e] = typename LayerW<Scalar>::MatR("gate_up_exp", 2 * I,
                                                        D);
      b.down_exps[e] = typename LayerW<Scalar>::MatR("down_exp", D, I);
      fill_rand(b.gate_up_exps[e], seed, -0.3f, 0.3f);
      fill_rand(b.down_exps[e], seed, -0.3f, 0.3f);
    }
    if (loop) {
      b.loop_q_down = typename LayerW<Scalar>::MatF("lq_down", 2 * D, R);
      b.loop_k_down = typename LayerW<Scalar>::MatF("lk_down", 2 * D, R);
      b.loop_v_down = typename LayerW<Scalar>::MatF("lv_down", 2 * D, R);
      b.loop_mlp_down = typename LayerW<Scalar>::MatF("lm_down", 2 * D,
                                                      R);
      b.loop_q_forget = typename LayerW<Scalar>::MatF("lq_f", R, D);
      b.loop_k_forget = typename LayerW<Scalar>::MatF("lk_f", R, D);
      b.loop_v_forget = typename LayerW<Scalar>::MatF("lv_f", R, D);
      b.loop_mlp_forget = typename LayerW<Scalar>::MatF("lm_f", R, D);
      b.loop_q_gain = typename LayerW<Scalar>::MatF("lq_g", R, D);
      b.loop_k_gain = typename LayerW<Scalar>::MatF("lk_g", R, D);
      b.loop_v_gain = typename LayerW<Scalar>::MatF("lv_g", R, D);
      b.loop_mlp_gain = typename LayerW<Scalar>::MatF("lm_g", R, D);
      b.msg_query = typename LayerW<Scalar>::MatF("msg_query", 4, D);
      fill_rand(b.loop_q_down, seed, -0.2f, 0.2f);
      fill_rand(b.loop_k_down, seed, -0.2f, 0.2f);
      fill_rand(b.loop_v_down, seed, -0.2f, 0.2f);
      fill_rand(b.loop_mlp_down, seed, -0.2f, 0.2f);
      fill_rand(b.loop_q_forget, seed, -0.2f, 0.2f);
      fill_rand(b.loop_k_forget, seed, -0.2f, 0.2f);
      fill_rand(b.loop_v_forget, seed, -0.2f, 0.2f);
      fill_rand(b.loop_mlp_forget, seed, -0.2f, 0.2f);
      fill_rand(b.loop_q_gain, seed, -0.2f, 0.2f);
      fill_rand(b.loop_k_gain, seed, -0.2f, 0.2f);
      fill_rand(b.loop_v_gain, seed, -0.2f, 0.2f);
      fill_rand(b.loop_mlp_gain, seed, -0.2f, 0.2f);
      fill_rand(b.msg_query, seed, -0.1f, 0.1f);
    }
  }
}

// ---------------------------------------------------------------------------
// Naive FP32 host reference: independent reimplementation of
// modeling_inkling2_loop.py math (no cache, full recompute, absolute
// positions via pos_off). Mirrors the Model's expected numerics exactly;
// any divergence from Model::forward is a Model bug (up to FP32
// reassociation noise).
// ---------------------------------------------------------------------------
struct R2 {
  size_t r = 0, c = 0;
  std::vector<float> v;
  float& operator()(size_t i, size_t j) { return v[i * c + j]; }
  const float& operator()(size_t i, size_t j) const { return v[i * c + j]; }
};
struct R3 {
  size_t a = 0, b = 0, c = 0;
  std::vector<float> v;
  float& operator()(size_t i, size_t j, size_t k) {
    return v[(i * b + j) * c + k];
  }
  const float& operator()(size_t i, size_t j, size_t k) const {
    return v[(i * b + j) * c + k];
  }
};

template <class View>
std::vector<float> to_host(const View& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<float>(h.data(), h.data() + h.size());
}

struct RLayer {
  R2 wq, wk, wv, wgate_attn, wo, wrel, wsink, router;
  std::vector<float> router_bias;
  R2 rel_proj, q_scale, rel_scale;
  R2 k_conv, v_conv, k_conv_static, v_conv_static;
  R2 k_conv_proj, v_conv_proj, post_attn_conv, post_mlp_conv;
  std::vector<float> attn_norm, q_norm, k_norm, v_norm, post_attn_norm;
  R2 attn_res_q, mlp_res_q;
  R3 gate_up_exps, down_exps;
  R2 loop_q_down, loop_k_down, loop_v_down, loop_mlp_down;
  R2 loop_q_forget, loop_k_forget, loop_v_forget, loop_mlp_forget;
  R2 loop_q_gain, loop_k_gain, loop_v_gain, loop_mlp_gain;
  R2 msg_query;
};

struct RWeights {
  R2 tok_emb, out_w;
  std::vector<float> tok_norm, out_norm;
  R2 out_res_q;
  std::vector<RLayer> layers;
};

R2 mk2(const std::vector<float>& v, size_t r, size_t c) {
  R2 o;
  o.r = r;
  o.c = c;
  o.v = v;
  return o;
}

RWeights ref_weights(const Config& cfg, const Weights<Scalar>& w) {
  RWeights rw;
  rw.tok_emb = mk2(to_host(w.tok_emb), w.tok_emb.extent(0),
                   w.tok_emb.extent(1));
  rw.out_w =
      mk2(to_host(w.out_w), w.out_w.extent(0), w.out_w.extent(1));
  rw.tok_norm = to_host(w.tok_norm);
  rw.out_norm = to_host(w.out_norm);
  rw.out_res_q =
      mk2(to_host(w.out_res_q), w.out_res_q.extent(0),
          w.out_res_q.extent(1));
  rw.layers.resize((size_t)cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    const auto& b = w.blocks[(size_t)l];
    auto& r = rw.layers[(size_t)l];
    r.wq = mk2(to_host(b.wq), b.wq.extent(0), b.wq.extent(1));
    r.wk = mk2(to_host(b.wk), b.wk.extent(0), b.wk.extent(1));
    r.wv = mk2(to_host(b.wv), b.wv.extent(0), b.wv.extent(1));
    r.wgate_attn = mk2(to_host(b.wgate_attn), b.wgate_attn.extent(0),
                       b.wgate_attn.extent(1));
    r.wo = mk2(to_host(b.wo), b.wo.extent(0), b.wo.extent(1));
    r.wrel =
        mk2(to_host(b.wrel), b.wrel.extent(0), b.wrel.extent(1));
    r.wsink =
        mk2(to_host(b.wsink), b.wsink.extent(0), b.wsink.extent(1));
    r.router =
        mk2(to_host(b.router), b.router.extent(0), b.router.extent(1));
    r.router_bias = to_host(b.router_bias);
    r.rel_proj = mk2(to_host(b.rel_proj), b.rel_proj.extent(0),
                     b.rel_proj.extent(1));
    r.q_scale = mk2(to_host(b.q_scale), b.q_scale.extent(0),
                    b.q_scale.extent(1));
    r.rel_scale = mk2(to_host(b.rel_scale), b.rel_scale.extent(0),
                      b.rel_scale.extent(1));
    if (b.k_conv.extent(0) != 0)
      r.k_conv = mk2(to_host(b.k_conv), b.k_conv.extent(0),
                     b.k_conv.extent(1));
    if (b.v_conv.extent(0) != 0)
      r.v_conv = mk2(to_host(b.v_conv), b.v_conv.extent(0),
                     b.v_conv.extent(1));
    if (b.k_conv_static.extent(0) != 0)
      r.k_conv_static = mk2(to_host(b.k_conv_static),
                            b.k_conv_static.extent(0),
                            b.k_conv_static.extent(1));
    if (b.v_conv_static.extent(0) != 0)
      r.v_conv_static = mk2(to_host(b.v_conv_static),
                            b.v_conv_static.extent(0),
                            b.v_conv_static.extent(1));
    if (b.k_conv_proj.extent(0) != 0)
      r.k_conv_proj = mk2(to_host(b.k_conv_proj), b.k_conv_proj.extent(0),
                          b.k_conv_proj.extent(1));
    if (b.v_conv_proj.extent(0) != 0)
      r.v_conv_proj = mk2(to_host(b.v_conv_proj), b.v_conv_proj.extent(0),
                          b.v_conv_proj.extent(1));
    r.post_attn_conv = mk2(to_host(b.post_attn_conv),
                           b.post_attn_conv.extent(0),
                           b.post_attn_conv.extent(1));
    r.post_mlp_conv = mk2(to_host(b.post_mlp_conv),
                          b.post_mlp_conv.extent(0),
                          b.post_mlp_conv.extent(1));
    if (b.attn_norm.extent(0) != 0) r.attn_norm = to_host(b.attn_norm);
    if (b.q_norm.extent(0) != 0) r.q_norm = to_host(b.q_norm);
    if (b.k_norm.extent(0) != 0) r.k_norm = to_host(b.k_norm);
    if (b.v_norm.extent(0) != 0) r.v_norm = to_host(b.v_norm);
    r.post_attn_norm = to_host(b.post_attn_norm);
    if (b.attn_res_q.extent(0) != 0)
      r.attn_res_q = mk2(to_host(b.attn_res_q), b.attn_res_q.extent(0),
                         b.attn_res_q.extent(1));
    r.mlp_res_q = mk2(to_host(b.mlp_res_q), b.mlp_res_q.extent(0),
                      b.mlp_res_q.extent(1));
    // Stack per-expert dense views into [E,2I,D] / [E,D,I] host tensors.
    {
      const size_t E = b.gate_up_exps.size();
      const size_t E2I =
          E > 0 ? b.gate_up_exps[0].extent(0) : 0;
      const size_t DD = E > 0 ? b.gate_up_exps[0].extent(1) : 0;
      const size_t II = E > 0 ? b.down_exps[0].extent(1) : 0;
      r.gate_up_exps.a = E;
      r.gate_up_exps.b = E2I;
      r.gate_up_exps.c = DD;
      r.gate_up_exps.v.assign(E * E2I * DD, 0);
      r.down_exps.a = E;
      r.down_exps.b = DD;
      r.down_exps.c = II;
      r.down_exps.v.assign(E * DD * II, 0);
      for (size_t e = 0; e < E; ++e) {
        auto gu = to_host(b.gate_up_exps[e]);
        auto dn = to_host(b.down_exps[e]);
        for (size_t j = 0; j < E2I; ++j)
          for (size_t d = 0; d < DD; ++d)
            r.gate_up_exps(e, j, d) = gu[j * DD + d];
        for (size_t d = 0; d < DD; ++d)
          for (size_t j = 0; j < II; ++j)
            r.down_exps(e, d, j) = dn[d * II + j];
      }
    }
    if (b.loop_q_down.extent(0) != 0) {
      r.loop_q_down = mk2(to_host(b.loop_q_down), b.loop_q_down.extent(0),
                          b.loop_q_down.extent(1));
      r.loop_k_down = mk2(to_host(b.loop_k_down), b.loop_k_down.extent(0),
                          b.loop_k_down.extent(1));
      r.loop_v_down = mk2(to_host(b.loop_v_down), b.loop_v_down.extent(0),
                          b.loop_v_down.extent(1));
      r.loop_mlp_down = mk2(to_host(b.loop_mlp_down),
                            b.loop_mlp_down.extent(0),
                            b.loop_mlp_down.extent(1));
      r.loop_q_forget = mk2(to_host(b.loop_q_forget),
                            b.loop_q_forget.extent(0),
                            b.loop_q_forget.extent(1));
      r.loop_k_forget = mk2(to_host(b.loop_k_forget),
                            b.loop_k_forget.extent(0),
                            b.loop_k_forget.extent(1));
      r.loop_v_forget = mk2(to_host(b.loop_v_forget),
                            b.loop_v_forget.extent(0),
                            b.loop_v_forget.extent(1));
      r.loop_mlp_forget = mk2(to_host(b.loop_mlp_forget),
                              b.loop_mlp_forget.extent(0),
                              b.loop_mlp_forget.extent(1));
      r.loop_q_gain = mk2(to_host(b.loop_q_gain), b.loop_q_gain.extent(0),
                          b.loop_q_gain.extent(1));
      r.loop_k_gain = mk2(to_host(b.loop_k_gain), b.loop_k_gain.extent(0),
                          b.loop_k_gain.extent(1));
      r.loop_v_gain = mk2(to_host(b.loop_v_gain), b.loop_v_gain.extent(0),
                          b.loop_v_gain.extent(1));
      r.loop_mlp_gain = mk2(to_host(b.loop_mlp_gain),
                            b.loop_mlp_gain.extent(0),
                            b.loop_mlp_gain.extent(1));
      r.msg_query = mk2(to_host(b.msg_query), b.msg_query.extent(0),
                        b.msg_query.extent(1));
    }
  }
  return rw;
}

R2 rmatmul(const R2& A, const R2& B) {
  // A [m,k], B [n,k] -> [m,n].
  R2 Y;
  Y.r = A.r;
  Y.c = B.r;
  Y.v.assign(A.r * B.r, 0);
  for (size_t i = 0; i < A.r; ++i)
    for (size_t j = 0; j < B.r; ++j) {
      double acc = 0;
      for (size_t k = 0; k < A.c; ++k) acc += (double)A(i, k) * B(j, k);
      Y(i, j) = (float)acc;
    }
  return Y;
}

void rrms_w(R2& x, const std::vector<float>& w, float eps) {
  for (size_t i = 0; i < x.r; ++i) {
    double ss = 0;
    for (size_t j = 0; j < x.c; ++j) ss += (double)x(i, j) * x(i, j);
    const float inv = (float)(1.0 / std::sqrt(ss / x.c + eps));
    for (size_t j = 0; j < x.c; ++j) x(i, j) *= inv * w[j];
  }
}

void rrms_now(R2& x, float eps) {
  for (size_t i = 0; i < x.r; ++i) {
    double ss = 0;
    for (size_t j = 0; j < x.c; ++j) ss += (double)x(i, j) * x(i, j);
    const float inv = (float)(1.0 / std::sqrt(ss / x.c + eps));
    for (size_t j = 0; j < x.c; ++j) x(i, j) *= inv;
  }
}

void rl2_rows(R2& x, size_t H, size_t Hd, float thr) {
  // x rows are [T*H, Hd].
  for (size_t i = 0; i < x.r; ++i) {
    double ss = 0;
    for (size_t j = 0; j < x.c; ++j) ss += (double)x(i, j) * x(i, j);
    const float s = std::min(1.0f, (float)(thr / (std::sqrt(ss) + 1e-6)));
    for (size_t j = 0; j < x.c; ++j) x(i, j) *= s;
  }
  (void)H;
  (void)Hd;
}

void rscaled_l2(R2& x, const R2& scale, float thr) {
  for (size_t i = 0; i < x.r; ++i)
    for (size_t j = 0; j < x.c; ++j) x(i, j) *= scale(i % scale.r, j);
  rl2_rows(x, scale.r, x.c, thr);
}

// LogScale applied per (token) row group: rows [t*H,(t+1)*H) share pos[t].
void rlogscale_rows(R2& x, size_t H, const std::vector<int>& pos, float alpha,
                    int fl) {
  const size_t T = pos.size();
  for (size_t t = 0; t < T; ++t) {
    const float ratio = ((float)pos[t] + 1.0f) / (float)fl;
    const float tau =
        1.0f + alpha * std::log(std::max(ratio, 1.0f));
    for (size_t h = 0; h < H; ++h)
      for (size_t d = 0; d < x.c; ++d) x(t * H + h, d) *= tau;
  }
}

R2 rstatic_conv(const R2& x, const R2& w) {
  // x [T,Ch], w [Ch,K] -> [T,Ch].
  const size_t T = x.r, Ch = x.c, K = w.c;
  R2 y;
  y.r = T;
  y.c = Ch;
  y.v.assign(T * Ch, 0);
  const float inv = 1.0f / std::sqrt((float)(K + 1));
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < Ch; ++c) {
      double acc = 0;
      for (size_t k = 0; k < K; ++k) {
        const long s = (long)t - (long)(K - 1) + (long)k;
        if (s < 0) continue;
        acc += (double)w(c, k) * x((size_t)s, c);
      }
      y(t, c) = (float)((acc + x(t, c)) * inv);
    }
  return y;
}

R2 rdynamic_conv(const R2& x, const R2& sw, const R2& dyn, size_t nheads,
                 size_t hd) {
  // x [T,Ch], sw [K,Ch], dyn [T,H*K] -> [T,Ch].
  const size_t T = x.r, Ch = x.c, K = sw.r;
  R2 y;
  y.r = T;
  y.c = Ch;
  y.v.assign(T * Ch, 0);
  const float inv = 1.0f / std::sqrt((float)(2 * K + 1));
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < Ch; ++c) {
      const size_t h = c / hd;
      double acc = 0;
      for (size_t k = 0; k < K; ++k) {
        const long s = (long)t - (long)(K - 1) + (long)k;
        if (s < 0) continue;
        const float tap = sw(k, c) + dyn(t, h * K + k);
        acc += (double)tap * x((size_t)s, c);
      }
      y(t, c) = (float)((acc + x(t, c)) * inv);
    }
  (void)nheads;
  return y;
}

// Retrieval over per-token stack rows: stack[t] = [S, D] list, partial[t].
R2 rretrieve(const std::vector<R2>& stack, const R2& partial, const R2& query,
             bool normalize, float eps) {
  const size_t T = partial.r, D = partial.c, Q = query.r;
  const size_t S = stack.empty() ? 0 : stack[0].r;
  R2 out;
  out.r = T;
  out.c = Q * D;
  out.v.assign(T * Q * D, 0);
  for (size_t t = 0; t < T; ++t) {
    // Normalized (or raw) rows.
    std::vector<std::vector<float>> nrm(S + 1, std::vector<float>(D));
    for (size_t s = 0; s <= S; ++s) {
      for (size_t d = 0; d < D; ++d)
        nrm[s][d] = (s < S) ? stack[t](s, d) : partial(t, d);
      if (normalize) {
        double ss = 0;
        for (size_t d = 0; d < D; ++d) ss += (double)nrm[s][d] * nrm[s][d];
        const float inv = (float)(1.0 / std::sqrt(ss / D + eps));
        for (size_t d = 0; d < D; ++d) nrm[s][d] *= inv;
      }
    }
    const float scale = 1.0f / std::sqrt((float)D);
    for (size_t q = 0; q < Q; ++q) {
      std::vector<float> scr(S + 1);
      float m = -1e30f;
      for (size_t s = 0; s <= S; ++s) {
        double a = 0;
        for (size_t d = 0; d < D; ++d) a += (double)query(q, d) * nrm[s][d];
        scr[s] = (float)(a * scale);
        m = std::max(m, scr[s]);
      }
      double denom = 0;
      for (size_t s = 0; s <= S; ++s) {
        scr[s] = std::exp(scr[s] - m);
        denom += scr[s];
      }
      for (size_t d = 0; d < D; ++d) {
        double acc = 0;
        for (size_t s = 0; s <= S; ++s) acc += scr[s] / denom * nrm[s][d];
        out(t, q * D + d) = (float)acc;
      }
    }
  }
  return out;
}

R2 rmsg_gate(const R2& x, const R2& msg, const R2& down, const R2& forget,
             const R2& gain, float eps) {
  const size_t T = x.r, D = x.c, R = down.c;
  R2 y;
  y.r = T;
  y.c = D;
  y.v.assign(T * D, 0);
  for (size_t t = 0; t < T; ++t) {
    double ssx = 0, ssm = 0;
    for (size_t d = 0; d < D; ++d) {
      ssx += (double)x(t, d) * x(t, d);
      ssm += (double)msg(t, d) * msg(t, d);
    }
    const float ix = (float)(1.0 / std::sqrt(ssx / D + eps));
    const float im = (float)(1.0 / std::sqrt(ssm / D + eps));
    std::vector<float> hg(R, 0);
    for (size_t r = 0; r < R; ++r) {
      double acc = 0;
      for (size_t d = 0; d < D; ++d)
        acc += (double)x(t, d) * ix * down(d, r) +
               (double)msg(t, d) * im * down(D + d, r);
      hg[r] = (float)acc;
    }
    for (size_t d = 0; d < D; ++d) {
      double f = 0, g = 0;
      for (size_t r = 0; r < R; ++r) {
        f += (double)forget(r, d) * hg[r];
        g += (double)gain(r, d) * hg[r];
      }
      const float ff = std::tanh((float)f), gg = std::tanh((float)g);
      y(t, d) = (1 - ff) * x(t, d) + gg * msg(t, d);
    }
  }
  return y;
}

float rsigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Attention over full key set with relbias + sink (reference, no cache).
R2 rattn(const R2& q, const R2& k, const R2& v, size_t NH, size_t Hkv,
         size_t Hd, const R2& rel, const R2& proj, const R2& sink, float scale,
         size_t rel_extent, size_t window, size_t q_off) {
  const size_t T = q.r / NH, N = k.r / Hkv;
  R2 out;
  out.r = T * NH;
  out.c = Hd;
  out.v.assign(T * NH * Hd, 0);
  const size_t grp = NH / Hkv;
  for (size_t t = 0; t < T; ++t)
    for (size_t h = 0; h < NH; ++h) {
      const size_t hkv = h / grp;
      const size_t qt = q_off + t;
      size_t klo = 0;
      if (window > 0 && qt + 1 > window) klo = qt + 1 - window;
      size_t khi = std::min(qt, N - 1);
      std::vector<float> sc;
      if (khi >= klo) sc.reserve(khi - klo + 1);
      float m = sink(t, h);
      for (size_t kk = klo; kk <= khi && khi >= klo; ++kk) {
        double dot = 0;
        for (size_t d = 0; d < Hd; ++d)
          dot += (double)q(t * NH + h, d) * k(kk * Hkv + hkv, d);
        float s = (float)(dot * scale);
        const long dd = (long)qt - (long)kk;
        if (dd >= 0 && (size_t)dd < rel_extent) {
          double b = 0;
          for (size_t r = 0; r < rel.c; ++r)
            b += (double)rel(t * NH + h, r) * proj(r, (size_t)dd);
          s += (float)b;
        }
        sc.push_back(s);
        m = std::max(m, s);
      }
      double denom = std::exp(sink(t, h) - m);
      for (float s : sc) denom += std::exp(s - m);
      for (size_t d = 0; d < Hd; ++d) {
        double acc = 0;
        for (size_t j = 0; j < sc.size(); ++j)
          acc += std::exp(sc[j] - m) *
                 (double)v((klo + j) * Hkv + hkv, d);
        out(t * NH + h, d) = (float)(acc / denom);
      }
    }
  return out;
}

// Full model reference. toks: token ids; W: host weights; pos_off: absolute
// position of toks[0] (for log-scale); window_ovr >= 0 overrides swa_window.
std::vector<float> ref_forward(const std::vector<int>& toks, const RWeights& W,
                               const Config& cfg, int pos_off = 0,
                               int window_ovr = -1) {
  const size_t T = toks.size(), D = (size_t)cfg.d_model;
  const size_t NH = (size_t)cfg.n_heads, Hkv = (size_t)cfg.n_kv_heads,
               Hd = (size_t)cfg.head_dim, DR = (size_t)cfg.d_rel;
  const size_t HqD = NH * Hd;
  const size_t E = (size_t)cfg.rel_extent;
  const size_t NE = (size_t)cfg.n_experts, TK = (size_t)cfg.top_k;
  const size_t II = (size_t)cfg.intermediate_size;
  const float scale = 1.0f / std::sqrt((float)Hd);
  const size_t window =
      window_ovr >= 0 ? (size_t)window_ovr : (size_t)cfg.swa_window;
  std::vector<int> apos(T);
  for (size_t t = 0; t < T; ++t) apos[t] = pos_off + (int)t;

  auto is_full = [&](int l) {
    return KokkosDNN::Inkling::Impl::is_full_layer(cfg.local_layer_ids,
                                                   cfg.n_layers, l);
  };

  // Embed + norm.
  R2 h;
  h.r = T;
  h.c = D;
  h.v.assign(T * D, 0);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d) h(t, d) = W.tok_emb((size_t)toks[t], d);
  rrms_w(h, W.tok_norm, cfg.eps);

  // Block stacks per token: vector of [S,D] rows.
  std::vector<R2> blocks(T);
  for (size_t t = 0; t < T; ++t) {
    blocks[t].r = 0;
    blocks[t].c = D;
  }
  auto blk_append = [&](const R2& hin) {
    for (size_t t = 0; t < T; ++t) {
      R2 nb;
      nb.r = blocks[t].r + 1;
      nb.c = D;
      nb.v.assign((blocks[t].r + 1) * D, 0);
      for (size_t s = 0; s < blocks[t].r; ++s)
        for (size_t d = 0; d < D; ++d) nb(s, d) = blocks[t](s, d);
      for (size_t d = 0; d < D; ++d) nb(blocks[t].r, d) = hin(t, d);
      blocks[t] = nb;
    }
  };

  std::function<R2(int, R2, std::vector<R2>&, const R2*)>
      layer_fwd = [&](int l, R2 hin, std::vector<R2>& blks,
                      const R2* msg) -> R2 {
    const RLayer& b = W.layers[(size_t)l];
    const bool full = is_full(l);
    // 1. streams
    R2 qh, kh, vh;
    qh.r = kh.r = vh.r = T;
    qh.c = kh.c = vh.c = D;
    qh.v.assign(T * D, 0);
    kh.v = qh.v;
    vh.v = qh.v;
    if (!full && (l == 0 || b.attn_res_q.r == 0)) {
      qh = kh = vh = hin;
    } else if (!full) {
      R2 r = rretrieve(blks, hin, b.attn_res_q, (bool)cfg.attn_res_normalize,
                       cfg.eps);
      for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
          qh(t, d) = r(t, d);
          kh(t, d) = r(t, d);
          vh(t, d) = r(t, d);
        }
    } else {
      if (l == 0) {
        qh = kh = vh = hin;
      } else {
        R2 r = rretrieve(blks, hin, b.attn_res_q,
                         (bool)cfg.attn_res_normalize, cfg.eps);
        for (size_t t = 0; t < T; ++t)
          for (size_t d = 0; d < D; ++d) {
            qh(t, d) = r(t, 0 * D + d);
            kh(t, d) = r(t, 1 * D + d);
            vh(t, d) = r(t, 2 * D + d);
          }
      }
    }
    if (msg != nullptr) {
      auto take = [&](const R2& m, size_t q) {
        R2 o;
        o.r = T;
        o.c = D;
        o.v.assign(T * D, 0);
        for (size_t t = 0; t < T; ++t)
          for (size_t d = 0; d < D; ++d) o(t, d) = m(t, q * D + d);
        return o;
      };
      const R2 mq = take(*msg, 0), mk = take(*msg, 1), mv = take(*msg, 2);
      if (!full) {
        qh = rmsg_gate(qh, mq, b.loop_q_down, b.loop_q_forget,
                       b.loop_q_gain, cfg.eps);
        kh = rmsg_gate(kh, mk, b.loop_k_down, b.loop_k_forget,
                       b.loop_k_gain, cfg.eps);
        vh = rmsg_gate(vh, mv, b.loop_v_down, b.loop_v_forget,
                       b.loop_v_gain, cfg.eps);
      } else {
        qh = rmsg_gate(qh, mq, b.loop_q_down, b.loop_q_forget,
                       b.loop_q_gain, cfg.eps);
        kh = rmsg_gate(kh, mk, b.loop_k_down, b.loop_k_forget,
                       b.loop_k_gain, cfg.eps);
        vh = rmsg_gate(vh, mv, b.loop_v_down, b.loop_v_forget,
                       b.loop_v_gain, cfg.eps);
      }
    }
    // 2-4. norms + linears
    R2 qhn = qh, khn = kh, vhn = vh;
    if (!full) {
      rrms_w(qhn, b.attn_norm, cfg.eps);
      rrms_w(khn, b.attn_norm, cfg.eps);
      rrms_w(vhn, b.attn_norm, cfg.eps);
    } else {
      rrms_w(qhn, b.q_norm, cfg.eps);
      rrms_w(khn, b.k_norm, cfg.eps);
      rrms_w(vhn, b.v_norm, cfg.eps);
    }
    R2 q2 = rmatmul(qhn, b.wq), k2 = rmatmul(khn, b.wk),
        v2 = rmatmul(vhn, b.wv);
    // 5. convs
    R2 kc, vc;
    if (!full) {
      kc = rstatic_conv(k2, b.k_conv);
      vc = rstatic_conv(v2, b.v_conv);
    } else {
      // Dynamic taps from the NORMED streams (HF KeyValueProjection gets
      // the normed hidden_states as its dynamic_input).
      R2 dk = rmatmul(khn, b.k_conv_proj), dv = rmatmul(vhn, b.v_conv_proj);
      kc = rdynamic_conv(k2, b.k_conv_static, dk, Hkv, Hd);
      vc = rdynamic_conv(v2, b.v_conv_static, dv, Hkv, Hd);
    }
    // 6. clips (rows are [T*H, Hd])
    // 6. clips over per-head rows: reinterpret the flat [T, H*D]
    // buffers as [T*H, D] row groups (same data, metadata only).
    R2 qh2 = q2;
    qh2.r = T * NH;
    qh2.c = Hd;
    R2 kh2 = kc;
    kh2.r = T * Hkv;
    kh2.c = Hd;
    if (!full) {
      rscaled_l2(qh2, b.q_scale, cfg.qk_clip);
    } else {
      rscaled_l2(qh2, b.q_scale, cfg.qk_clip);
      rlogscale_rows(qh2, NH, apos, cfg.log_alpha, cfg.log_floor);
    }
    rl2_rows(kh2, Hkv, Hd, cfg.qk_clip);
    {  // v: unweighted rms per head row
      R2 vh2 = vc;
      vh2.r = T * Hkv;
      vh2.c = Hd;
      rrms_now(vh2, cfg.eps);
      vc = vh2;
    }
    q2 = qh2;
    kc = kh2;
    // 7. rel + sink from qhn
    R2 rel = rmatmul(qhn, b.wrel);  // [T, H*DR]
    rel.r = T * NH;
    rel.c = DR;
    if (!full) {
      rscaled_l2(rel, b.rel_scale, cfg.rel_clip);
    } else {
      rscaled_l2(rel, b.rel_scale, cfg.rel_clip);
      rlogscale_rows(rel, NH, apos, cfg.log_alpha, cfg.log_floor);
    }
    R2 sink = rmatmul(qhn, b.wsink);  // [T, H]
    // 8. attention
    R2 kc3 = kc, vc3 = vc;
    kc3.r = T * Hkv;
    vc3.r = T * Hkv;
    R2 q23 = q2;
    q23.r = T * NH;
    R2 rel3 = rel;
    R2 att = rattn(q23, kc3, vc3, NH, Hkv, Hd, rel3, b.rel_proj, sink,
                   scale, E, full ? 0 : window, (size_t)pos_off);
    att.r = T;  // flatten [T*NH, Hd] back to [T, HqD] rows for the gate/out
    att.c = HqD;
    // 9. gate + out
    R2 g = rmatmul(qhn, b.wgate_attn);
    for (size_t i = 0; i < g.v.size(); ++i)
      att.v[i] *= rsigmoid(g.v[i]);
    R2 o = rmatmul(att, b.wo);
    // 10. post conv
    R2 oc = rstatic_conv(o, b.post_attn_conv);
    // 11. residual
    R2 res = oc;
    const bool boundary = (l % cfg.attn_res_block == 0);
    if (boundary) {
      blk_append(hin);
    } else {
      for (size_t i = 0; i < res.v.size(); ++i) res.v[i] += hin.v[i];
    }
    // 12. mlp
    R2 m = rretrieve(blks, res, b.mlp_res_q, (bool)cfg.attn_res_normalize,
                     cfg.eps);
    if (msg != nullptr) {
      R2 mm;
      mm.r = T;
      mm.c = D;
      mm.v.assign(T * D, 0);
      for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) mm(t, d) = (*msg)(t, 3 * D + d);
      m = rmsg_gate(m, mm, b.loop_mlp_down, b.loop_mlp_forget,
                    b.loop_mlp_gain, cfg.eps);
    }
    rrms_w(m, b.post_attn_norm, cfg.eps);
    // MoE
    R2 logits = rmatmul(m, b.router);
    // top-k sigmoid routing
    std::vector<std::vector<int>> sel(T);
    std::vector<std::vector<float>> sw(T);
    for (size_t t = 0; t < T; ++t) {
      std::vector<std::pair<float, int>> sc;
      for (size_t e = 0; e < NE; ++e) {
        const float s = rsigmoid(logits(t, e));
        const float bias =
            (bool)cfg.use_gate_bias ? b.router_bias[e] : 0.0f;
        sc.push_back({s + bias, (int)e});
      }
      std::sort(sc.begin(), sc.end(),
                [](const auto& x, const auto& y) { return x.first > y.first; });
      double sum = 0;
      for (size_t k = 0; k < TK; ++k) {
        const int e = sc[k].second;
        sel[t].push_back(e);
        const float s = rsigmoid(logits(t, (size_t)e));
        sw[t].push_back(s);
        sum += s;
      }
      for (size_t k = 0; k < TK; ++k)
        sw[t][k] = (float)(sw[t][k] / (sum + 1e-20) * cfg.route_scale);
    }
    R2 mo;
    mo.r = T;
    mo.c = D;
    mo.v.assign(T * D, 0);
    for (size_t e = 0; e < NE; ++e) {
      for (size_t t = 0; t < T; ++t)
        for (size_t k = 0; k < TK; ++k) {
          if (sel[t][k] != (int)e) continue;
          // expert e on token t
          std::vector<float> gu(2 * II, 0);
          for (size_t j = 0; j < 2 * II; ++j) {
            double acc = 0;
            for (size_t d = 0; d < D; ++d)
              acc += (double)m(t, d) * b.gate_up_exps(e, j, d);
            gu[j] = (float)acc;
          }
          std::vector<float> act(II);
          for (size_t j = 0; j < II; ++j) {
            const float gg = gu[j], uu = gu[II + j];
            act[j] = std::tanh(gg / cfg.gate_softcap) * cfg.gate_softcap *
                     rsigmoid(gg) * std::tanh(uu / cfg.up_softcap) *
                     cfg.up_softcap;
          }
          for (size_t d = 0; d < D; ++d) {
            double acc = 0;
            for (size_t j = 0; j < II; ++j)
              acc += (double)act[j] * b.down_exps(e, d, j);
            mo(t, d) += (float)acc * sw[t][k];
          }
        }
    }
    R2 mc = rstatic_conv(mo, b.post_mlp_conv);
    R2 hout = res;
    for (size_t i = 0; i < hout.v.size(); ++i) hout.v[i] += mc.v[i];
    return hout;
  };

  R2 hcur = h;
  // Pre-loop layers.
  for (int l = 0; l < cfg.loop_start; ++l)
    hcur = layer_fwd(l, hcur, blocks, nullptr);
  R2 pre_h = hcur;
  std::vector<R2> pre_blocks = blocks;
  const size_t npre = pre_blocks.empty() ? 0 : pre_blocks[0].r;
  // First pass over loop layers.
  for (int l = cfg.loop_start; l < cfg.loop_end; ++l)
    hcur = layer_fwd(l, hcur, blocks, nullptr);
  // Remaining passes restart from the pre-loop states with messages from
  // the previous pass.
  for (int pass = 1; pass < cfg.loop_passes; ++pass) {
    // stripped prev blocks + prev hidden
    std::vector<R2> prev_stripped(T);
    const size_t Sall = blocks.empty() ? 0 : blocks[0].r;
    size_t strip = std::min(npre + 1, Sall);
    const size_t Ss = Sall - strip;
    for (size_t t = 0; t < T; ++t) {
      prev_stripped[t].r = Ss;
      prev_stripped[t].c = D;
      prev_stripped[t].v.assign(Ss * D, 0);
      for (size_t s = 0; s < Ss; ++s)
        for (size_t d = 0; d < D; ++d)
          prev_stripped[t](s, d) = blocks[t](strip + s, d);
    }
    R2 prev_h = hcur;
    hcur = pre_h;
    blocks = pre_blocks;
    std::vector<R2> msgs(cfg.n_layers);
    for (int l = cfg.loop_start; l < cfg.loop_end; ++l) {
      msgs[l] = rretrieve(
          prev_stripped, prev_h,
          W.layers[(size_t)l].msg_query, true /*always normed*/, cfg.eps);
    }
    for (int l = cfg.loop_start; l < cfg.loop_end; ++l)
      hcur = layer_fwd(l, hcur, blocks, &msgs[(size_t)l]);
  }
  // post-loop
  for (int l = cfg.loop_end; l < cfg.n_layers; ++l)
    hcur = layer_fwd(l, hcur, blocks, nullptr);
  // final
  R2 fin = rretrieve(blocks, hcur, W.out_res_q,
                     (bool)cfg.attn_res_normalize, cfg.eps);
  R2 last1;
  last1.r = 1;
  last1.c = D;
  last1.v.assign(D, 0);
  for (size_t d = 0; d < D; ++d) last1(0, d) = fin(T - 1, d);
  rrms_w(last1, W.out_norm, cfg.eps);
  R2 lg = rmatmul(last1, W.out_w);
  for (float& x : lg.v) x = std::tanh(x / cfg.final_softcap) * cfg.final_softcap;
  return lg.v;
}

// ---------------------------------------------------------------------------
// Test helpers.
// ---------------------------------------------------------------------------
Kokkos::View<int*, Exec> dev_toks(const std::vector<int>& ids) {
  Kokkos::View<int*, Exec> v("toks", ids.size());
  auto h = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
  Kokkos::deep_copy(v, h);
  return v;
}

std::vector<float> model_logits(Model& m, const std::vector<int>& ids) {
  auto lg = m.forward(dev_toks(ids));
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lg);
  return std::vector<float>(h.data(), h.data() + h.size());
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

Config tiny_cfg() {
  Config c;
  c.n_layers = 2;
  c.loop_start = 1;
  c.loop_end = 2;
  return c;
}

// (a) full-forward vs naive reference for each of the 4 block kinds.
TEST(InklingModel, SlidingBlockKind) {
  Config cfg = tiny_cfg();
  cfg.loop_start = 2;
  cfg.loop_end = 2;  // no loop layers: both sliding
  cfg.loop_passes = 1;
  cfg.local_layer_ids = {0, 1};
  Weights<Scalar> w;
  random_weights(cfg, w, 11);
  Model m(cfg, w);
  const std::vector<int> ids = {3, 7, 11, 23, 31};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_LT(max_abs_diff(got, want), 1e-3f)
      << "maxdiff=" << max_abs_diff(got, want);
}

TEST(InklingModel, SlidingLoopBlockKind) {
  Config cfg = tiny_cfg();
  cfg.loop_passes = 2;  // layer 1 is a sliding loop layer with messages
  cfg.local_layer_ids = {0, 1};
  Weights<Scalar> w;
  random_weights(cfg, w, 23);
  Model m(cfg, w);
  const std::vector<int> ids = {5, 9, 13, 29, 37};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_LT(max_abs_diff(got, want), 1e-3f)
      << "maxdiff=" << max_abs_diff(got, want);
}

TEST(InklingModel, FullBlockKind) {
  Config cfg = tiny_cfg();
  cfg.loop_start = 2;
  cfg.loop_end = 2;
  cfg.loop_passes = 1;
  cfg.local_layer_ids = {0};  // layer 1 is full attention
  Weights<Scalar> w;
  random_weights(cfg, w, 37);
  Model m(cfg, w);
  const std::vector<int> ids = {2, 6, 12, 24, 33};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_LT(max_abs_diff(got, want), 1e-3f)
      << "maxdiff=" << max_abs_diff(got, want);
}

TEST(InklingModel, FullLoopBlockKind) {
  Config cfg = tiny_cfg();
  cfg.loop_passes = 2;  // layer 1 is a full loop layer with messages
  cfg.local_layer_ids = {0};
  Weights<Scalar> w;
  random_weights(cfg, w, 51);
  Model m(cfg, w);
  const std::vector<int> ids = {4, 8, 14, 26, 41};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_LT(max_abs_diff(got, want), 1e-3f)
      << "maxdiff=" << max_abs_diff(got, want);
}

// Mixed 4-layer reference check (sliding, sliding, sliding-loop,
// full-loop) with 3 loop passes to stress message passing.
TEST(InklingModel, MixedLoopPasses) {
  Config cfg;
  cfg.n_layers = 4;
  cfg.loop_start = 2;
  cfg.loop_end = 4;
  cfg.loop_passes = 3;
  cfg.max_seq_len = 32;
  cfg.local_layer_ids = {0, 1, 2};
  Weights<Scalar> w;
  random_weights(cfg, w, 67);
  Model m(cfg, w);
  const std::vector<int> ids = {3, 9, 15, 21, 27};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_LT(max_abs_diff(got, want), 1e-3f)
      << "maxdiff=" << max_abs_diff(got, want);
}

// Mixed 4-layer reference check with plain-full coverage
// (sliding, full-loop, full-loop, full), 3 passes.
TEST(InklingModel, MixedFullLoop) {
  Config cfg;
  cfg.n_layers = 4;
  cfg.loop_start = 1;
  cfg.loop_end = 3;
  cfg.loop_passes = 3;
  cfg.max_seq_len = 32;
  cfg.local_layer_ids = {0};
  Weights<Scalar> w;
  random_weights(cfg, w, 83);
  Model m(cfg, w);
  const std::vector<int> ids = {6, 12, 18, 24, 30};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_LT(max_abs_diff(got, want), 1e-3f)
      << "maxdiff=" << max_abs_diff(got, want);
}

// (b) prefill-vs-incremental-decode consistency: chunked prefill, single
// shot, and token-by-token decode must agree; decode continuation past a
// prefill must match a longer prefill.
TEST(InklingModel, PrefillDecodeConsistency) {
  Config cfg;
  cfg.n_layers = 4;
  cfg.loop_start = 2;
  cfg.loop_end = 4;
  cfg.loop_passes = 2;
  cfg.max_seq_len = 32;
  cfg.local_layer_ids = {0, 1, 2};
  Weights<Scalar> w;
  random_weights(cfg, w, 97);
  const std::vector<int> ids = {3, 7, 11, 23, 31, 5};

  Model m_chunk(cfg, w);
  m_chunk.cfg.prefill_chunk_size = 2;
  const std::vector<float> chunked = model_logits(m_chunk, ids);

  Model m_single(cfg, w);
  m_single.cfg.prefill_chunk_size = 0;
  const std::vector<float> single = model_logits(m_single, ids);
  ASSERT_EQ(chunked.size(), single.size());
  EXPECT_LT(max_abs_diff(chunked, single), 1e-4f);

  // Token-by-token decode from scratch.
  Model m_dec(cfg, w);
  m_dec.cfg.prefill_chunk_size = 0;
  std::vector<float> dec_last;
  for (size_t t = 0; t < ids.size(); ++t)
    dec_last = model_logits(m_dec, {ids[t]});
  EXPECT_LT(max_abs_diff(chunked, dec_last), 1e-4f);

  // Continuation: prefill 4, decode 2 more == prefill 6.
  Model m_cont(cfg, w);
  m_cont.cfg.prefill_chunk_size = 2;
  (void)model_logits(m_cont, {ids[0], ids[1], ids[2], ids[3]});
  (void)model_logits(m_cont, {ids[4]});
  const std::vector<float> cont = model_logits(m_cont, {ids[5]});
  EXPECT_LT(max_abs_diff(chunked, cont), 1e-4f);
}

// (c) sliding-window masking: windowed Model output must match a windowed
// reference and differ from a full-context reference.
TEST(InklingModel, SlidingWindowMasking) {
  Config cfg = tiny_cfg();
  cfg.loop_start = 2;
  cfg.loop_end = 2;
  cfg.loop_passes = 1;
  cfg.local_layer_ids = {0, 1};
  cfg.swa_window = 2;
  Weights<Scalar> w;
  random_weights(cfg, w, 113);
  Model m(cfg, w);
  const std::vector<int> ids = {3, 7, 11, 23, 31};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want_win = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want_win.size());
  EXPECT_LT(max_abs_diff(got, want_win), 1e-3f);
  const std::vector<float> want_full = ref_forward(ids, rw, cfg, 0, 10000);
  EXPECT_GT(max_abs_diff(got, want_full), 1e-3f);
}

// Session smoke: prefill/step/usage plus prefill_cached hit consistency.
TEST(InklingSession, SmokeAndPrefixCache) {
  using Sess = KokkosDNN::Inkling::Session<Exec, Scalar>;
  Config cfg = tiny_cfg();
  cfg.loop_start = 2;
  cfg.loop_end = 2;
  cfg.loop_passes = 1;
  cfg.local_layer_ids = {0, 1};
  Weights<Scalar> w;
  random_weights(cfg, w, 131);
  KokkosDNN::Inkling::Tokenizer tok;

  Sess s(cfg, w, tok);
  const std::vector<int> full = {3, 7, 11, 23};
  const std::vector<int> content = {3, 7, 11};
  s.prefill(full);
  auto o1 = s.step();
  EXPECT_FALSE(o1.finished);
  EXPECT_EQ(s.usage().prompt_tokens, 4u);
  EXPECT_EQ(s.usage().completion_tokens, 1u);
  EXPECT_EQ(s.generated(), 1u);

  // Uncached prefill_cached (miss) reproduces the same first token.
  Sess s2(cfg, w, tok);
  auto miss = s2.prefill_cached(full, content);
  EXPECT_FALSE(miss.hit);
  EXPECT_EQ(miss.reused, 0u);
  auto o2 = s2.step();
  EXPECT_EQ(o2.id, o1.id);

  // Second call hits the cache and reproduces the token again.
  auto hit = s2.prefill_cached(full, content);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.reused, content.size());
  auto o3 = s2.step();
  EXPECT_EQ(o3.id, o1.id);
  EXPECT_EQ(s2.prefix_cache_size(), 1u);
  s2.clear_prefix_cache();
  EXPECT_EQ(s2.prefix_cache_size(), 0u);
  (void)s2.flush();
  s2.reset();
  EXPECT_TRUE(s2.usage().completion_tokens == 0u);
}

// Unnormalized residual retrieval + disabled gate bias still match.
TEST(InklingModel, RawResidualNoGateBias) {
  Config cfg = tiny_cfg();
  cfg.loop_start = 2;
  cfg.loop_end = 2;
  cfg.loop_passes = 1;
  cfg.local_layer_ids = {0};
  cfg.attn_res_normalize = false;
  cfg.use_gate_bias = false;
  Weights<Scalar> w;
  random_weights(cfg, w, 149);
  Model m(cfg, w);
  const std::vector<int> ids = {3, 9, 15, 21, 27};
  const std::vector<float> got = model_logits(m, ids);
  const RWeights rw = ref_weights(cfg, w);
  const std::vector<float> want = ref_forward(ids, rw, cfg);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_LT(max_abs_diff(got, want), 1e-3f)
      << "maxdiff=" << max_abs_diff(got, want);
}

// Team-parallel attention must reproduce the legacy kernel to FP
// reassociation (the max/sum reductions reorder vs the legacy serial loop).
// Direct kernel-vs-kernel check across decode/prefill shapes, sliding +
// full windows, and an empty-row edge.
template <class Scalar>
void check_attention_team_matches_legacy(size_t C, size_t H, size_t N,
                                         size_t HD, size_t R, size_t E,
                                         size_t q_off, size_t window,
                                         unsigned seed) {
  using V3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec>;
  using VF = Kokkos::View<float**, Kokkos::LayoutRight, Exec>;
  Kokkos::Random_XorShift64_Pool<Exec> pool(seed);
  V3 q("q", C, H, HD), k("k", N, H > 0 ? 2 : 1, HD), v("v", N, 2, HD);
  // H must be a multiple of Hkv = 2 here.
  V3 rel("rel", C, H, R), out_t("out_t", C, H, HD),
      out_l("out_l", C, H, HD);
  V2 sink("sink", C, H);
  VF proj("proj", R, E);
  Kokkos::fill_random(q, pool, (Scalar)-1, (Scalar)1);
  Kokkos::fill_random(k, pool, (Scalar)-1, (Scalar)1);
  Kokkos::fill_random(v, pool, (Scalar)-1, (Scalar)1);
  Kokkos::fill_random(rel, pool, (Scalar)-1, (Scalar)1);
  Kokkos::fill_random(sink, pool, (Scalar)-1, (Scalar)1);
  Kokkos::fill_random(proj, pool, -1.0f, 1.0f);
  const float scale = 1.0f / std::sqrt((float)HD);
  KokkosDNN::Inkling::Impl::inkling_attention_forward<Exec, Scalar>(
      q, k, v, q_off, window, rel, proj, sink, scale, E, out_t);
  KokkosDNN::Inkling::Impl::inkling_attention_forward_legacy<Exec, Scalar>(
      q, k, v, q_off, window, rel, proj, sink, scale, E, out_l);
  Kokkos::fence();
  auto ht = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out_t);
  auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out_l);
  ASSERT_EQ(ht.size(), hl.size());
  for (size_t i = 0; i < ht.size(); ++i)
    EXPECT_NEAR(static_cast<float>(ht.data()[i]),
                static_cast<float>(hl.data()[i]), 1e-5f)
        << "team vs legacy mismatch at " << i << " (C=" << C << " H=" << H
        << " N=" << N << " window=" << window << " q_off=" << q_off << ")";
}

TEST(InklingModel, AttentionTeamMatchesLegacy) {
  // Decode-like: single row, few keys, sliding + full.
  check_attention_team_matches_legacy<float>(1, 4, 5, 8, 4, 32, 4, 1024, 7);
  check_attention_team_matches_legacy<float>(1, 4, 5, 8, 4, 32, 4, 0, 8);
  // Prefill-like: multiple rows, sliding window narrower than context.
  check_attention_team_matches_legacy<float>(3, 4, 9, 8, 4, 32, 6, 4, 9);
  check_attention_team_matches_legacy<float>(3, 4, 9, 8, 4, 32, 0, 0, 10);
  // Empty rows: window excludes every key (khi < klo zeroes the row).
  check_attention_team_matches_legacy<float>(1, 4, 2, 8, 4, 32, 9, 2, 11);
  // Real head width.
  check_attention_team_matches_legacy<float>(1, 8, 11, 128, 16, 128, 10, 1024,
                                            12);
  // bhalf (the deployed scalar): same tolerance bar.
  using bhalf_t = Kokkos::Experimental::bhalf_t;
  check_attention_team_matches_legacy<bhalf_t>(1, 4, 5, 8, 4, 32, 4, 1024, 13);
  check_attention_team_matches_legacy<bhalf_t>(1, 8, 11, 128, 16, 128, 10,
                                              1024, 14);
}

// Fused MoE decode vs a host reference computed from the dequantized packed
// bytes (exact weights, serial FP32 dots). Covers the real recipe (IQ4_NL
// gate_up + Q8_0 down) and all-Q8_0, with an unrouted expert present.
template <class Scalar>
void check_moe_decode_fused(Gguf::GgmlType gu_dt, Gguf::GgmlType down_dt,
                            unsigned seed) {
  using Gguf::GgmlType;
  const size_t E = 3, S = 2, K = 64, I = 32, D = 64;
  const float gs = 4.0f, us = 25.0f;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  // Random float weights, packed per dtype.
  std::vector<std::vector<float>> gu_f(E, std::vector<float>(2 * I * K));
  std::vector<std::vector<float>> down_f(E, std::vector<float>(D * I));
  for (size_t e = 0; e < E; ++e) {
    for (auto& v : gu_f[e]) v = uni(rng);
    for (auto& v : down_f[e]) v = uni(rng);
  }
  auto pack_q80 = [&](const std::vector<float>& f, size_t rows, size_t cols,
                      std::vector<uint8_t>& raw) {
    raw.assign(rows * (cols / 32) * 34, 0);
    for (size_t r = 0; r < rows; ++r)
      for (size_t b = 0; b < cols / 32; ++b) {
        float amax = 0;
        for (size_t i = 0; i < 32; ++i)
          amax = std::max(amax, std::fabs(f[r * cols + b * 32 + i]));
        const float d = amax > 0 ? amax / 127.0f : 1.0f;
        // Manual fp32->fp16 for the test packer.
        uint32_t bits;
        std::memcpy(&bits, &d, 4);
        const uint32_t sign = (bits >> 16) & 0x8000u;
        uint16_t dh;
        int exp = (int)((bits >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = (bits >> 13) & 0x3FFu;
        if (exp <= 0) {
          dh = (uint16_t)sign;
        } else if (exp >= 31) {
          dh = (uint16_t)(sign | 0x7BFFu);
        } else {
          dh = (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
        }
        Gguf::BlockQ80 blk{};
        blk.d = dh;
        for (size_t i = 0; i < 32; ++i) {
          const float v = f[r * cols + b * 32 + i];
          int q = (int)std::nearbyint(v / (amax > 0 ? d : 1.0f));
          blk.qs[i] = (int8_t)std::max(-127, std::min(127, q));
        }
        std::memcpy(raw.data() + (r * (cols / 32) + b) * 34, &blk, 34);
      }
  };
  auto pack_iq4nl = [&](const std::vector<float>& f, size_t rows, size_t cols,
                        std::vector<uint8_t>& raw) {
    raw.assign(rows * (cols / 32) * 18, 0);
    for (size_t r = 0; r < rows; ++r)
      for (size_t b = 0; b < cols / 32; ++b) {
        float amax = 0;
        for (size_t i = 0; i < 32; ++i)
          amax = std::max(amax, std::fabs(f[r * cols + b * 32 + i]));
        const float d = amax > 0 ? amax / 127.0f : 1.0f;
        uint32_t bits;
        std::memcpy(&bits, &d, 4);
        const uint32_t sign = (bits >> 16) & 0x8000u;
        int exp = (int)((bits >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = (bits >> 13) & 0x3FFu;
        uint16_t dh;
        if (exp <= 0) {
          dh = (uint16_t)sign;
        } else if (exp >= 31) {
          dh = (uint16_t)(sign | 0x7BFFu);
        } else {
          dh = (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
        }
        Gguf::BlockIQ4NL blk{};
        blk.d = dh;
        for (size_t i = 0; i < 16; ++i) {
          int best_lo = 0, best_hi = 0;
          float best_lo_d = 1e30f, best_hi_d = 1e30f;
          for (int c = 0; c < 16; ++c) {
            const float lv = (float)Gguf::iq4nl_value((uint8_t)c);
            const float dl = std::fabs(f[r * cols + b * 32 + i] / d - lv);
            const float dhh =
                std::fabs(f[r * cols + b * 32 + 16 + i] / d - lv);
            if (dl < best_lo_d) {
              best_lo_d = dl;
              best_lo = c;
            }
            if (dhh < best_hi_d) {
              best_hi_d = dhh;
              best_hi = c;
            }
          }
          blk.qs[i] = (uint8_t)(best_lo | (best_hi << 4));
        }
        std::memcpy(raw.data() + (r * (cols / 32) + b) * 18, &blk, 18);
      }
  };
  // Host reference from the dequantized packed bytes.
  auto dequant_row = [&](GgmlType dt, const uint8_t* raw, size_t row,
                         size_t cols, std::vector<float>& out) {
    out.assign(cols, 0);
    const size_t rb = (dt == GgmlType::Q8_0) ? cols / 32 * 34 : cols / 32 * 18;
    for (size_t b = 0; b < cols / 32; ++b) {
      float tmp[32];
      if (dt == GgmlType::Q8_0) {
        Gguf::dequant_q80_block(
            reinterpret_cast<const Gguf::BlockQ80*>(raw + row * rb + b * 34),
            tmp);
      } else {
        Gguf::dequant_iq4nl_block(
            reinterpret_cast<const Gguf::BlockIQ4NL*>(raw + row * rb + b * 18),
            tmp);
      }
      for (size_t i = 0; i < 32; ++i) out[b * 32 + i] = tmp[i];
    }
  };
  std::vector<float> x(K);
  for (auto& v : x) v = uni(rng);
  const std::vector<int> idx = {0, 2};  // expert 1 unrouted
  const std::vector<float> wts = {0.7f, 0.3f};
  // Host reference from the dequantized packed bytes, replaying the int8
  // stage-1 quantizer exactly (same formula as QuantizeActQ8Functor), so the
  // tolerance only covers kernel FP reassociation, not quantizer noise.
  auto quant_row = [&](const std::vector<float>& f, std::vector<float>& dq) {
    dq.assign(f.size(), 0);
    for (size_t b = 0; b < f.size() / 32; ++b) {
      float amax = 0;
      for (size_t i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(f[b * 32 + i]));
      const float d = amax / 127.0f;
      const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
      for (size_t i = 0; i < 32; ++i) {
        int q = (int)std::nearbyint(f[b * 32 + i] * inv);
        q = std::max(-127, std::min(127, q));
        dq[b * 32 + i] = d * (float)q;
      }
    }
  };
  std::vector<float> qx;
  quant_row(x, qx);
  std::vector<float> want(D, 0);
  {
    std::vector<float> act(S * I, 0);
    for (size_t s = 0; s < S; ++s) {
      const size_t e = (size_t)idx[s];
      std::vector<uint8_t> gu_raw;
      if (gu_dt == GgmlType::Q8_0) {
        pack_q80(gu_f[e], 2 * I, K, gu_raw);
      } else {
        pack_iq4nl(gu_f[e], 2 * I, K, gu_raw);
      }
      // Reference gate/up dots from the packed bytes and quantized input.
      std::vector<float> act_s(I, 0);
      for (size_t j = 0; j < I; ++j) {
        std::vector<float> gr, ur;
        dequant_row(gu_dt, gu_raw.data(), j, K, gr);
        dequant_row(gu_dt, gu_raw.data(), j + I, K, ur);
        double dg = 0, du = 0;
        for (size_t k = 0; k < K; ++k) {
          dg += (double)gr[k] * (double)qx[k];
          du += (double)ur[k] * (double)qx[k];
        }
        const float gcap = std::tanh((float)dg / gs) * gs;
        const float sig = 1.0f / (1.0f + std::exp(-(float)dg));
        const float ucap = std::tanh((float)du / us) * us;
        act_s[j] = gcap * sig * ucap;
        act[s * I + j] = act_s[j];
      }
    }
    // Reference down dots from the packed bytes and quantized slot rows.
    for (size_t s = 0; s < S; ++s) {
      const size_t e = (size_t)idx[s];
      std::vector<uint8_t> down_raw;
      if (down_dt == GgmlType::Q8_0) {
        pack_q80(down_f[e], D, I, down_raw);
      } else {
        pack_iq4nl(down_f[e], D, I, down_raw);
      }
      std::vector<float> slot(act.begin() + s * I, act.begin() + (s + 1) * I);
      std::vector<float> qslot;
      quant_row(slot, qslot);
      for (size_t d = 0; d < D; ++d) {
        std::vector<float> dr;
        dequant_row(down_dt, down_raw.data(), d, I, dr);
        double dd = 0;
        for (size_t i = 0; i < I; ++i) dd += (double)dr[i] * (double)qslot[i];
        want[d] += wts[s] * (float)dd;
      }
    }
  }
  // Device fused path.
  std::vector<Gguf::QuantWeight<Exec>> gu_ws(E), down_ws(E);
  for (size_t e = 0; e < E; ++e) {
    std::vector<uint8_t> gu_raw, down_raw;
    if (gu_dt == GgmlType::Q8_0) {
      pack_q80(gu_f[e], 2 * I, K, gu_raw);
    } else {
      pack_iq4nl(gu_f[e], 2 * I, K, gu_raw);
    }
    if (down_dt == GgmlType::Q8_0) {
      pack_q80(down_f[e], D, I, down_raw);
    } else {
      pack_iq4nl(down_f[e], D, I, down_raw);
    }
    gu_ws[e].dtype = gu_dt;
    gu_ws[e].rows = 2 * I;
    gu_ws[e].cols = K;
    gu_ws[e].data = Kokkos::View<uint8_t*, Exec>("gu", gu_raw.size());
    down_ws[e].dtype = down_dt;
    down_ws[e].rows = D;
    down_ws[e].cols = I;
    down_ws[e].data = Kokkos::View<uint8_t*, Exec>("down", down_raw.size());
    auto gu_h = Kokkos::create_mirror_view(gu_ws[e].data);
    auto down_h = Kokkos::create_mirror_view(down_ws[e].data);
    std::memcpy(gu_h.data(), gu_raw.data(), gu_raw.size());
    std::memcpy(down_h.data(), down_raw.data(), down_raw.size());
    Kokkos::deep_copy(gu_ws[e].data, gu_h);
    Kokkos::deep_copy(down_ws[e].data, down_h);
  }
  auto gu_tab = Gguf::make_quant_weight_table<Exec>(gu_ws, "t");
  auto down_tab = Gguf::make_quant_weight_table<Exec>(down_ws, "t");
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> xd("x", 1, K);
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> xidx("idx", 1, S);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> xwts("wts", 1, S);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> xact("act", S, I);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> xout("out", 1, D);
  {
    auto hx = Kokkos::create_mirror_view(xd);
    auto hi = Kokkos::create_mirror_view(xidx);
    auto hw = Kokkos::create_mirror_view(xwts);
    for (size_t k = 0; k < K; ++k) hx(0, k) = static_cast<Scalar>(x[k]);
    for (size_t s = 0; s < S; ++s) {
      hi(0, s) = idx[s];
      hw(0, s) = wts[s];
    }
    Kokkos::deep_copy(xd, hx);
    Kokkos::deep_copy(xidx, hi);
    Kokkos::deep_copy(xwts, hw);
  }
  ASSERT_TRUE(KokkosDNN::Inkling::Impl::moe_decode_fused_supported(
      gu_dt, K, down_dt, I));
  Kokkos::View<uint8_t*, Exec> qx_scratch(
      "qx", KokkosDNN::Gguf::quant_gemv_int8_scratch_bytes(K));
  Kokkos::View<uint8_t*, Exec> qa_scratch(
      "qa", KokkosDNN::Gguf::quant_gemv_int8_scratch_bytes(S * I));
  KokkosDNN::Inkling::Impl::inkling_moe_decode_fused<Exec, Scalar>(
      xd, gu_tab, down_tab, xidx, xwts, xact, qx_scratch, qa_scratch, xout,
      gs, us);
  Kokkos::fence();
  auto hout = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xout);
  ASSERT_EQ(hout.size(), want.size());
  // Quantizer noise is replayed in the reference, so the envelope only
  // covers kernel FP reassociation; nibble/LUT bugs miss by O(1).
  for (size_t d = 0; d < D; ++d)
    EXPECT_NEAR(static_cast<float>(hout(0, d)), want[d], 1e-3f)
        << "fused MoE decode mismatch at " << d;
}

TEST(InklingModel, MoeDecodeFusedMatchesReference) {
  check_moe_decode_fused<float>(Gguf::GgmlType::Q8_0, Gguf::GgmlType::Q8_0, 3);
  check_moe_decode_fused<float>(Gguf::GgmlType::IQ4_NL, Gguf::GgmlType::Q8_0,
                                4);
}

}  // namespace

