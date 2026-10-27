// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// End-to-end Glimmer text: presets, tiny forward/generate smoke (float +
// bhalf), determinism, chunked-prefill equivalence, and a gated real-weight
// check (skips without files, never downloads). No weight downloads:
// random weights + checked-in config fixture.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_GlimmerConfig.hpp"
#include "KokkosDNN_GlimmerModel.hpp"
#include "KokkosDNN_GlimmerTokenizer.hpp"
#include "KokkosDNN_GlimmerWeights.hpp"

namespace {

std::string glimmer_source_dir() {
  std::string f = __FILE__;
  size_t slash  = f.find_last_of('/');
  return (slash == std::string::npos) ? "." : f.substr(0, slash);
}

const char* getenv_str(const char* name) {
  const char* p = std::getenv(name);
  return (p && *p) ? p : nullptr;
}

std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',' ||
        c == ';' || c == ':') {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

bool file_readable(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

void check_glimmer_30b_dims(const KokkosDNN::Glimmer::Config& c) {
  EXPECT_EQ(c.dim, 6656);
  EXPECT_EQ(c.n_layers, 52);
  EXPECT_EQ(c.n_heads, 32);
  EXPECT_EQ(c.n_kv_heads, 2);
  // Explicit head_dim delta: 6656 / 32 = 208, but the arch uses 128.
  EXPECT_EQ(c.head_dim, 128);
  EXPECT_NE(c.dim / c.n_heads, c.head_dim);
  EXPECT_EQ(c.vocab_size, 202048);
  EXPECT_EQ(c.ffn_dim, 19968);
  EXPECT_FLOAT_EQ(c.rms_norm_eps, 1e-5f);
  EXPECT_FLOAT_EQ(c.post_norm_eps, 1e-8f);
  EXPECT_FLOAT_EQ(c.rope_theta, 500000.0f);
  EXPECT_EQ(c.sliding_window, 2048);
  EXPECT_FLOAT_EQ(c.final_logit_softcap, 20.0f);
  EXPECT_FLOAT_EQ(c.qk_scale_factor, 3.87f);
  EXPECT_DOUBLE_EQ(c.output_multiplier, 0.19611613513818404);
  EXPECT_EQ(c.bos_token, 200000);
  EXPECT_EQ(c.eos_token, 200001);
  EXPECT_EQ(c.pad_token, -1);  // HF pad_token_id is null
  EXPECT_FALSE(c.tie_word_embeddings);
  EXPECT_FALSE(c.attention_bias);
  EXPECT_EQ(c.hidden_activation, "silu");
  // 3x-sliding + 1x-full repeat: full layers at idx 3, 7, ..., 51.
  ASSERT_EQ(c.is_full_layer.size(), (size_t)52);
  ASSERT_EQ(c.layer_rope_theta.size(), (size_t)52);
  int n_full = 0;
  for (int l = 0; l < 52; ++l) {
    bool expect_full = (l % 4 == 3);
    EXPECT_EQ(c.is_full(l), expect_full) << "layer " << l;
    if (c.is_full(l)) ++n_full;
  }
  EXPECT_EQ(n_full, 13);
}

template <class Exec, class Scalar>
Kokkos::View<int*, Exec> make_tok_view(const std::vector<int>& ids) {
  Kokkos::View<int*, Exec> toks("toks", ids.size());
  auto h = Kokkos::create_mirror_view(toks);
  for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
  Kokkos::deep_copy(toks, h);
  return toks;
}

}  // namespace

TEST(GlimmerPreset, FactoryValues) {
  using KokkosDNN::Glimmer::Config;
  auto full = Config::glimmer_30b();
  check_glimmer_30b_dims(full);
  EXPECT_NO_THROW(full.validate());

  auto tiny = Config::tiny_test();
  EXPECT_EQ(tiny.n_layers, 4);
  // Tiny keeps the explicit-head_dim structure at small dims.
  EXPECT_NE(tiny.dim / tiny.n_heads, tiny.head_dim);
  EXPECT_NO_THROW(tiny.validate());
}

TEST(GlimmerPreset, FixtureParse) {
  using KokkosDNN::Glimmer::Config;
  const std::string dir = glimmer_source_dir();
  // Bare text_config fixture.
  Config from_fixture =
      Config::from_json_file(dir + "/config-glimmer-30b-text.json");
  check_glimmer_30b_dims(from_fixture);
  EXPECT_NO_THROW(from_fixture.validate());
}

template <class Scalar>
void check_glimmer_generate() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::Glimmer::Config::tiny_test();
  KokkosDNN::Glimmer::Weights<Exec, Scalar> w;
  KokkosDNN::Glimmer::random_init(cfg, w, 1234);
  KokkosDNN::Glimmer::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {2, 10, 20, 30};
  // Forward smoke: one logit row per vocab id.
  auto toks = make_tok_view<Exec, Scalar>(prompt);
  auto lg = model.forward(toks);
  ASSERT_EQ(lg.extent(0), (size_t)cfg.vocab_size);
  // Final logit softcapping bounds every logit by the cap (20).
  {
    auto h = Kokkos::create_mirror_view(lg);
    Kokkos::deep_copy(h, lg);
    for (size_t i = 0; i < h.extent(0); ++i) {
      float v = static_cast<float>(h(i));
      EXPECT_LE(std::abs(v), 20.0f + 1e-3f);
      EXPECT_TRUE(std::isfinite(v));
    }
  }
  // Generate smoke: tiny prompt, few new tokens, ids in range.
  auto out = model.generate(prompt, 2);
  ASSERT_EQ(out.size(), prompt.size() + 2);
  for (auto id : out) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, cfg.vocab_size);
  }
  // Determinism: rebuild and compare (greedy default).
  KokkosDNN::Glimmer::Model<Exec, Scalar> model2(cfg, w);
  EXPECT_EQ(model2.generate(prompt, 2), out);
}

TEST(GlimmerGenerate, Float) { check_glimmer_generate<float>(); }

TEST(GlimmerGenerate, BHalf) {
  check_glimmer_generate<Kokkos::Experimental::bhalf_t>();
}

// Sampled (non-greedy) decode with a fixed seed reproduces exactly.
TEST(GlimmerGenerate, SamplingDeterminism) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  auto cfg     = KokkosDNN::Glimmer::Config::tiny_test();
  KokkosDNN::Glimmer::Weights<Exec, Scalar> w;
  KokkosDNN::Glimmer::random_init(cfg, w, 7);
  KokkosDNN::SampleOptions opt;
  opt.temperature = 0.7f;
  opt.top_p       = 0.9f;
  opt.seed        = 12345;
  std::vector<int> prompt = {2, 10, 20, 30};
  KokkosDNN::Glimmer::Model<Exec, Scalar> m1(cfg, w);
  KokkosDNN::HostRNG r1(opt.seed);
  auto o1 = m1.generate(prompt, 6, {cfg.eos_token}, opt, &r1);
  KokkosDNN::Glimmer::Model<Exec, Scalar> m2(cfg, w);
  KokkosDNN::HostRNG r2(opt.seed);
  auto o2 = m2.generate(prompt, 6, {cfg.eos_token}, opt, &r2);
  EXPECT_EQ(o1, o2);
  EXPECT_EQ(o1.size(), prompt.size() + 6);
}

// Chunked prefill matches single-shot prefill token-for-token (native
// kernels; chunking only repartitions the same causal computation).
TEST(GlimmerGenerate, PrefillChunkEquivalence) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  auto cfg     = KokkosDNN::Glimmer::Config::tiny_test();
  cfg.max_seq_len = 32;
  KokkosDNN::Glimmer::Weights<Exec, Scalar> w;
  KokkosDNN::Glimmer::random_init(cfg, w, 99);
  std::vector<int> prompt = {2, 10, 20, 30, 40, 50, 60, 70};
  auto single = cfg;
  single.prefill_chunk_size = 0;
  KokkosDNN::Glimmer::Model<Exec, Scalar> m1(single, w);
  auto o1 = m1.generate(prompt, 3);
  auto chunked = cfg;
  chunked.prefill_chunk_size = 2;
  KokkosDNN::Glimmer::Model<Exec, Scalar> m2(chunked, w);
  auto o2 = m2.generate(prompt, 3);
  EXPECT_EQ(o1, o2);
}

// Both RoPE branches run: all-NoPE layers (full-attention thetas are 0
// upstream) and all-RoPE layers produce finite, in-range tokens.
TEST(GlimmerGenerate, NoPEAndRoPEBranches) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  auto cfg     = KokkosDNN::Glimmer::Config::tiny_test();
  KokkosDNN::Glimmer::Weights<Exec, Scalar> w;
  KokkosDNN::Glimmer::random_init(cfg, w, 5);
  std::vector<int> prompt = {2, 10, 20};
  auto nope = cfg;
  for (auto& th : nope.layer_rope_theta) th = 0.0;
  KokkosDNN::Glimmer::Model<Exec, Scalar> m1(nope, w);
  auto o1 = m1.generate(prompt, 2);
  EXPECT_EQ(o1.size(), prompt.size() + 2);
  auto ropes = cfg;
  for (auto& th : ropes.layer_rope_theta) th = 500000.0;
  KokkosDNN::Glimmer::Model<Exec, Scalar> m2(ropes, w);
  auto o2 = m2.generate(prompt, 2);
  EXPECT_EQ(o2.size(), prompt.size() + 2);
  for (auto id : o2) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, cfg.vocab_size);
  }
}

// Untied head: the loader must carry a separate lm_head (never aliased).
TEST(GlimmerGenerate, UntiedLmHead) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::Glimmer::Config::tiny_test();
  EXPECT_FALSE(cfg.tie_word_embeddings);
  KokkosDNN::Glimmer::Weights<Exec, float> w;
  KokkosDNN::Glimmer::random_init(cfg, w, 1234);
  EXPECT_NE(w.lm_head.data(), w.embed.data());
  KokkosDNN::Glimmer::Model<Exec, float> model(cfg, w);
  auto out = model.generate({2, 10, 20}, 2);
  EXPECT_EQ(out.size(), 5u);
}

// Real-weight check (gated; SKIP without files, never downloads).
// Needs RAM for the BF16 Glimmer 30B weights plus the tokenizer:
//   KOKKOSDNN_WEIGHTS        model.safetensors (sharded; list all files)
//   KOKKOSDNN_TOKENIZER_JSON tokenizer.json
TEST(GlimmerGenerate, RealWeightsBHalf) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = Kokkos::Experimental::bhalf_t;
  using namespace KokkosDNN::Glimmer;
  const char* wenv = getenv_str("KOKKOSDNN_WEIGHTS");
  const char* tenv = getenv_str("KOKKOSDNN_TOKENIZER_JSON");
  if (!wenv || !tenv) {
    GTEST_SKIP() << "real-weight check needs KOKKOSDNN_WEIGHTS + "
                 << "KOKKOSDNN_TOKENIZER_JSON; skipping (tests never download "
                 << "weights).";
  }
  auto wfiles = split_list(wenv);
  if (wfiles.empty())
    GTEST_SKIP() << "KOKKOSDNN_WEIGHTS parsed to zero files; skipping.";
  for (auto& f : wfiles)
    if (!file_readable(f))
      GTEST_SKIP() << "weights file not readable, skipping: " << f;
  if (!file_readable(tenv))
    GTEST_SKIP() << "tokenizer file not readable, skipping: " << tenv;

  Tokenizer tok(tenv);                   // throws -> FAIL
  Config cfg = Config::glimmer_30b();    // full 30B shapes
  cfg.max_seq_len = 128;                 // cap KV cache for the smoke prompt
  Weights<Exec, Scalar> w;
  load_safetensors(cfg, wfiles, w);  // throws -> FAIL (names the tensor)
  Model<Exec, Scalar> model(cfg, w);
  std::vector<int> ids = tok.encode("Hello", true);
  ASSERT_FALSE(ids.empty());
  auto toks = make_tok_view<Exec, Scalar>(ids);
  auto lg = model.forward(toks);
  ASSERT_EQ(lg.extent(0), (size_t)cfg.vocab_size);
  auto h = Kokkos::create_mirror_view(lg);
  Kokkos::deep_copy(h, lg);
  int best = 0;
  for (size_t i = 1; i < h.extent(0); ++i)
    if (static_cast<float>(h(i)) > static_cast<float>(h((size_t)best)))
      best = (int)i;
  std::printf("[glimmer:real] prompt ids=%zu argmax=%d top_logit=%g\n",
              ids.size(), best, (double)static_cast<float>(h((size_t)best)));
  EXPECT_GE(best, 0);
  EXPECT_LT(best, cfg.vocab_size);
  for (size_t i = 0; i < h.extent(0); ++i)
    EXPECT_LE(std::abs(static_cast<float>(h(i))), 20.0f + 1e-3f);
}
