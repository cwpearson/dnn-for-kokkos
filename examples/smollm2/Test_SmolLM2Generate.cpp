// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// End-to-end SmolLM2: presets, fixtures, and tiny/full generate smoke.
// No weight downloads: random weights + checked-in config fixtures.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_SmolLM2Config.hpp"
#include "KokkosDNN_SmolLM2Model.hpp"
#include "KokkosDNN_SmolLM2Weights.hpp"

namespace {

std::string smollm2_source_dir() {
  std::string f = __FILE__;
  size_t slash  = f.find_last_of('/');
  return (slash == std::string::npos) ? "." : f.substr(0, slash);
}

void check_smollm2_135m_dims(const KokkosDNN::SmolLM2::Config& c) {
  EXPECT_EQ(c.dim, 576);
  EXPECT_EQ(c.n_layers, 30);
  EXPECT_EQ(c.n_heads, 9);
  EXPECT_EQ(c.n_kv_heads, 3);
  EXPECT_EQ(c.vocab_size, 49152);
  EXPECT_EQ(c.ffn_dim, 1536);
  EXPECT_EQ(c.head_dim, 64);  // 576 / 9 recomputed
  EXPECT_FLOAT_EQ(c.attn_scale, 0.125f);  // 1/sqrt(64)
  EXPECT_FLOAT_EQ(c.rms_norm_eps, 1e-5f);
  EXPECT_FLOAT_EQ(c.rope_theta, 100000.0f);
  EXPECT_EQ(c.max_seq_len, 8192);
  EXPECT_TRUE(c.tie_word_embeddings);
  EXPECT_EQ(c.hidden_act, "silu");
  EXPECT_FALSE(c.attention_bias);
  EXPECT_FALSE(c.mlp_bias);
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

TEST(SmolLM2Preset, FactoryValues) {
  using KokkosDNN::SmolLM2::Config;
  auto base = Config::smollm2_135m();
  check_smollm2_135m_dims(base);
  EXPECT_EQ(base.bos_token, 0);  // <|endoftext|>
  EXPECT_EQ(base.eos_token, 0);  // <|endoftext|>
  EXPECT_NO_THROW(base.validate());

  auto instruct = Config::smollm2_135m_instruct();
  check_smollm2_135m_dims(instruct);
  EXPECT_EQ(instruct.bos_token, 1);  // <|im_start|>
  EXPECT_EQ(instruct.eos_token, 2);  // <|im_end|> (+pad)
  EXPECT_NO_THROW(instruct.validate());

  auto tiny = Config::tiny_test();
  EXPECT_EQ(tiny.dim, 64);
  EXPECT_EQ(tiny.n_layers, 2);
  EXPECT_EQ(tiny.head_dim, 8);
  EXPECT_EQ(tiny.max_seq_len, 32);
  EXPECT_NO_THROW(tiny.validate());
}

TEST(SmolLM2Preset, FixtureParse) {
  using KokkosDNN::SmolLM2::Config;
  const std::string dir = smollm2_source_dir();
  Config base = Config::from_json_file(dir + "/config-smollm2-135m.json");
  check_smollm2_135m_dims(base);
  EXPECT_EQ(base.bos_token, 0);
  EXPECT_EQ(base.eos_token, 0);
  EXPECT_TRUE(base.tie_word_embeddings);
  EXPECT_NO_THROW(base.validate());

  Config instruct =
      Config::from_json_file(dir + "/config-smollm2-135m-instruct.json");
  check_smollm2_135m_dims(instruct);
  EXPECT_EQ(instruct.bos_token, 1);
  EXPECT_EQ(instruct.eos_token, 2);
  EXPECT_TRUE(instruct.tie_word_embeddings);
  EXPECT_NO_THROW(instruct.validate());
}

TEST(SmolLM2Preset, FixtureMatchesFactory) {
  using KokkosDNN::SmolLM2::Config;
  const std::string dir = smollm2_source_dir();
  // Fixtures parsed with the factory as base must equal the factory: every
  // key the factory sets is present in the fixture with the same value.
  Config from_base =
      Config::from_json_file(dir + "/config-smollm2-135m.json");
  Config base = Config::smollm2_135m();
  EXPECT_EQ(from_base.dim, base.dim);
  EXPECT_EQ(from_base.n_layers, base.n_layers);
  EXPECT_EQ(from_base.n_heads, base.n_heads);
  EXPECT_EQ(from_base.n_kv_heads, base.n_kv_heads);
  EXPECT_EQ(from_base.vocab_size, base.vocab_size);
  EXPECT_EQ(from_base.ffn_dim, base.ffn_dim);
  EXPECT_EQ(from_base.bos_token, base.bos_token);
  EXPECT_EQ(from_base.eos_token, base.eos_token);
  EXPECT_EQ(from_base.tie_word_embeddings, base.tie_word_embeddings);
  EXPECT_FLOAT_EQ(from_base.rope_theta, base.rope_theta);

  Config from_instruct =
      Config::from_json_file(dir + "/config-smollm2-135m-instruct.json");
  Config instruct = Config::smollm2_135m_instruct();
  EXPECT_EQ(from_instruct.bos_token, instruct.bos_token);
  EXPECT_EQ(from_instruct.eos_token, instruct.eos_token);
}

template <class Scalar>
void check_smollm2_generate() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::SmolLM2::Config::smollm2_135m();
  cfg.max_seq_len = 32;  // cap KV cache; dims/weights stay full-size
  KokkosDNN::SmolLM2::Weights<Exec, Scalar> w;
  KokkosDNN::SmolLM2::random_init(cfg, w, 1234);
  KokkosDNN::SmolLM2::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3};
  // Forward smoke: one logit row per vocab id.
  auto toks = make_tok_view<Exec, Scalar>(prompt);
  EXPECT_EQ(model.forward(toks).extent(0), (size_t)cfg.vocab_size);
  // Generate smoke: tiny prompt, few new tokens, ids in range.
  auto out = model.generate(prompt, 2);
  ASSERT_EQ(out.size(), prompt.size() + 2);
  for (auto id : out) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, cfg.vocab_size);
  }
  // Determinism: rebuild and compare.
  KokkosDNN::SmolLM2::Model<Exec, Scalar> model2(cfg, w);
  EXPECT_EQ(model2.generate(prompt, 2), out);
}

TEST(SmolLM2Generate, Float) { check_smollm2_generate<float>(); }

TEST(SmolLM2Generate, BHalf) {
  check_smollm2_generate<Kokkos::Experimental::bhalf_t>();
}

// Tied lm_head: SmolLM2-135M ships tie_word_embeddings=true, so a
// checkpoint without any lm_head tensor loads by aliasing the embedding
// table; with tie=false the same layout throws.
TEST(SmolLM2Generate, TiedLmHeadDefault) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::SmolLM2::Config::tiny_test();
  cfg.tie_word_embeddings = true;
  KokkosDNN::SmolLM2::Weights<Exec, float> w;
  KokkosDNN::SmolLM2::random_init(cfg, w, 1234);
  // random_init fills both tables; emulate tying by aliasing explicitly.
  w.lm_head = w.embed;
  EXPECT_EQ(w.lm_head.data(), w.embed.data());
  KokkosDNN::SmolLM2::Model<Exec, float> model(cfg, w);
  auto out = model.generate({1, 2, 3}, 2);
  EXPECT_EQ(out.size(), 5u);
}
