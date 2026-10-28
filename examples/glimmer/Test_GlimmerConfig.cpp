// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Glimmer config scaffold tests: presets, the checked-in text_config
// fixture (bare and wrapped in a full config.json), and tiny preset.
// No weight downloads: checked-in config fixture only.
#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "KokkosDNN_GlimmerConfig.hpp"

namespace {

std::string glimmer_source_dir() {
#ifdef GLIMMER_MODEL_DIR
  return GLIMMER_MODEL_DIR;
#else
  std::string f = __FILE__;
  size_t slash  = f.find_last_of('/');
  return (slash == std::string::npos) ? "." : f.substr(0, slash);
#endif
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
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
    if (c.is_full(l)) {
      ++n_full;
      // Full layers are marked 0 and fall back to the global theta.
      EXPECT_DOUBLE_EQ(c.layer_rope_theta[(size_t)l], 0.0) << "layer " << l;
      EXPECT_DOUBLE_EQ(c.rope_theta_for_layer(l), 500000.0) << "layer " << l;
    } else {
      EXPECT_DOUBLE_EQ(c.rope_theta_for_layer(l), 500000.0) << "layer " << l;
    }
  }
  EXPECT_EQ(n_full, 13);
}

TEST(GlimmerConfig, Preset30B) {
  auto c = KokkosDNN::Glimmer::Config::glimmer_30b();
  EXPECT_NO_THROW(c.validate());
  check_glimmer_30b_dims(c);
}

TEST(GlimmerConfig, Tiny) {
  auto c = KokkosDNN::Glimmer::Config::tiny_test();
  EXPECT_NO_THROW(c.validate());
  // Tiny keeps the explicit-head_dim structure at small dims.
  EXPECT_NE(c.dim / c.n_heads, c.head_dim);
  EXPECT_EQ(c.n_layers, 4);
  ASSERT_EQ(c.layer_rope_theta.size(), (size_t)4);
  EXPECT_DOUBLE_EQ(c.rope_theta_for_layer(3), (double)c.rope_theta);
}

TEST(GlimmerConfig, FixtureBare) {
  auto c = KokkosDNN::Glimmer::Config::from_json_file(
      glimmer_source_dir() + "/config-glimmer-30b-text.json");
  check_glimmer_30b_dims(c);
}

TEST(GlimmerConfig, FixtureFullWrapper) {
  // The driver also accepts the full HF config.json: text fields nested
  // under "text_config" must win over nothing (base defaults agree here).
  std::string bare =
      read_file(glimmer_source_dir() + "/config-glimmer-30b-text.json");
  std::string full = "{\"model_type\": \"muse_glimmer\", \"text_config\": " +
                     bare + "}";
  auto c = KokkosDNN::Glimmer::Config::from_json_string(full);
  check_glimmer_30b_dims(c);
}

}  // namespace
