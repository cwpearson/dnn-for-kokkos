// LFM2.5 config tests (no weight downloads).
//
// Covers: the lfm2_5_24b_a2b() preset values, tiny_test() validation, and
// parsing/validation of the shipped config-lfm2_5-24b-a2b.json content via
// from_json_string, plus rejection of incompatible dims and unsupported
// model options.
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "KokkosDNN_LFM25Config.hpp"

using KokkosDNN::LFM25::Config;

TEST(LFM25Config, Preset24BA2B) {
  auto c = Config::lfm2_5_24b_a2b();
  EXPECT_EQ(c.dim, 2048);
  EXPECT_EQ(c.n_layers, 40);
  EXPECT_EQ(c.n_heads, 32);
  EXPECT_EQ(c.n_kv_heads, 8);
  EXPECT_EQ(c.head_dim, 64);  // 2048 / 32
  EXPECT_FLOAT_EQ(c.attn_scale, 0.125f);
  EXPECT_EQ(c.vocab_size, 128000);
  EXPECT_EQ(c.ffn_dim, 11776);
  EXPECT_EQ(c.moe_intermediate, 1536);
  EXPECT_EQ(c.n_experts, 64);
  EXPECT_EQ(c.top_k, 4);
  EXPECT_EQ(c.n_dense_layers, 2);
  EXPECT_FLOAT_EQ(c.rms_norm_eps, 1e-5f);
  EXPECT_FLOAT_EQ(c.rope_theta, 10000000.0f);
  EXPECT_TRUE(c.norm_topk_prob);
  EXPECT_TRUE(c.use_expert_bias);
  EXPECT_EQ(c.conv_L_cache, 3);
  EXPECT_EQ(c.num_attention_layers(), 10);
  EXPECT_EQ(c.num_conv_layers(), 30);
  EXPECT_TRUE(c.tie_word_embeddings);
  EXPECT_EQ(c.hidden_act, "silu");
  EXPECT_EQ(c.model_type, "lfm2_moe");
  EXPECT_EQ(c.bos, 124894);
  EXPECT_EQ(c.eos, 124900);
  EXPECT_EQ(c.pad, 124893);
  EXPECT_NO_THROW(c.validate());
}

TEST(LFM25Config, TinyTestValid) {
  auto c = Config::tiny_test();
  EXPECT_EQ(c.dim, 64);
  EXPECT_EQ(c.n_layers, 4);
  EXPECT_EQ(c.n_heads, 8);
  EXPECT_EQ(c.n_kv_heads, 4);
  EXPECT_EQ(c.head_dim, 8);
  EXPECT_EQ(c.vocab_size, 256);
  EXPECT_EQ(c.n_experts, 8);
  EXPECT_EQ(c.top_k, 2);
  EXPECT_EQ(c.n_dense_layers, 1);
  EXPECT_EQ(c.conv_L_cache, 3);
  // [conv, conv, attn, conv] rhythm (matches the i%4==2 full-model pattern).
  ASSERT_EQ((int)c.is_attention_layer.size(), 4);
  EXPECT_EQ(c.num_attention_layers(), 1);
  EXPECT_TRUE(c.is_attention(2));
  EXPECT_NO_THROW(c.validate());
}

// Exact content of examples/lfm2_5/config-lfm2_5-24b-a2b.json, exercised
// through from_json_string (no file I/O, no downloads).
TEST(LFM25Config, ShippedJsonContent) {
  const std::string js = R"({
  "model_type": "lfm2_moe",
  "hidden_size": 2048,
  "num_hidden_layers": 40,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 11776,
  "moe_intermediate_size": 1536,
  "num_experts": 64,
  "num_experts_per_tok": 4,
  "num_dense_layers": 2,
  "hidden_act": "silu",
  "norm_eps": 1e-05,
  "norm_topk_prob": true,
  "use_expert_bias": true,
  "routed_scaling_factor": 1.0,
  "rope_parameters": {
    "rope_theta": 10000000.0
  },
  "layer_types": [
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv"
  ],
  "max_position_embeddings": 131072,
  "vocab_size": 128000,
  "bos_token_id": 124894,
  "eos_token_id": 124900,
  "pad_token_id": 124893,
  "tie_word_embeddings": true,
  "max_seq_len": 8192,
  "prefill_chunk_size": 2048,
  "kv_page_size": 0,
  "kv_int8": false
})";
  Config got = Config::from_json_string(js);
  EXPECT_EQ(got.dim, 2048);
  EXPECT_EQ(got.n_layers, 40);
  EXPECT_EQ(got.n_heads, 32);
  EXPECT_EQ(got.n_kv_heads, 8);
  EXPECT_EQ(got.head_dim, 64);
  EXPECT_EQ(got.vocab_size, 128000);
  EXPECT_EQ(got.ffn_dim, 11776);
  EXPECT_EQ(got.moe_intermediate, 1536);
  EXPECT_EQ(got.n_experts, 64);
  EXPECT_EQ(got.top_k, 4);
  EXPECT_EQ(got.n_dense_layers, 2);
  EXPECT_FLOAT_EQ(got.rope_theta, 10000000.0f);
  EXPECT_EQ(got.num_attention_layers(), 10);
  EXPECT_TRUE(got.is_attention(2));
  EXPECT_TRUE(got.is_attention(38));
  EXPECT_FALSE(got.is_attention(0));
  // The non-HF "max_seq_len" alias wins over "max_position_embeddings"
  // (PoC runtime cap, not the 131072 arch limit).
  EXPECT_EQ(got.max_seq_len, 8192);
  EXPECT_EQ(got.bos, 124894);
  EXPECT_EQ(got.eos, 124900);
  EXPECT_EQ(got.pad, 124893);
  EXPECT_TRUE(got.tie_word_embeddings);
  EXPECT_NO_THROW(got.validate());
}

TEST(LFM25Config, HeadDimMismatchRejected) {
  // tiny_test wants head_dim 8 (64/8); an explicit mismatch throws.
  EXPECT_THROW(Config::from_json_string(R"({"head_dim": 16})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"head_dim": 7})",
                                        Config::tiny_test()),
               std::exception);
  // Matching explicit head_dim is accepted.
  Config ok =
      Config::from_json_string(R"({"head_dim": 8})", Config::tiny_test());
  EXPECT_EQ(ok.head_dim, 8);
  // hidden_size not divisible by num_attention_heads.
  EXPECT_THROW(Config::from_json_string(
                   R"({"hidden_size": 65, "num_attention_heads": 8})",
                   Config::tiny_test()),
               std::exception);
  // More kv heads than query heads / uneven GQA groups.
  EXPECT_THROW(Config::from_json_string(
                   R"({"num_attention_heads": 4, "num_key_value_heads": 8})",
                   Config::tiny_test()),
               std::exception);
}

TEST(LFM25Config, TopKExceedsExpertsRejected) {
  // top_k must satisfy 1 <= top_k <= num_experts (tiny: E=8).
  EXPECT_THROW(Config::from_json_string(R"({"num_experts_per_tok": 9})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"num_experts_per_tok": 0})",
                                        Config::tiny_test()),
               std::exception);
  Config c = Config::tiny_test();
  c.top_k  = c.n_experts + 1;
  EXPECT_THROW(c.validate(), std::exception);
}

TEST(LFM25Config, ConvLCacheRejected) {
  // The preset/shop config uses conv_L_cache == 3; non-positive values are
  // rejected (validate requires > 0, the parser rejects negatives outright).
  EXPECT_EQ(Config::lfm2_5_24b_a2b().conv_L_cache, 3);
  EXPECT_THROW(Config::from_json_string(R"({"conv_L_cache": 0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"conv_L_cache": -1})",
                                        Config::tiny_test()),
               std::exception);
  Config c      = Config::tiny_test();
  c.conv_L_cache = 0;
  EXPECT_THROW(c.validate(), std::exception);
}

TEST(LFM25Config, RopeRejected) {
  // Only positive finite rope_theta is accepted (top-level or nested).
  // Unknown rope option keys are ignored; the value checks below are what
  // fail loudly.
  EXPECT_THROW(Config::from_json_string(R"({"rope_theta": -1.0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"rope_theta": 0.0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"rope_parameters": null})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"rope_parameters": "yarn"})",
                                        Config::tiny_test()),
               std::exception);
}

TEST(LFM25Config, ModelTypeRejected) {
  // Only "lfm2_moe" (case-insensitive) is supported.
  EXPECT_THROW(Config::from_json_string(R"({"model_type": "llama"})",
                                        Config::tiny_test()),
               std::runtime_error);
  EXPECT_THROW(Config::from_json_string(R"({"model_type": "lfm2"})",
                                        Config::tiny_test()),
               std::runtime_error);
  EXPECT_NO_THROW(Config::from_json_string(R"({"model_type": "LFM2_MoE"})",
                                           Config::tiny_test()));
  Config c    = Config::tiny_test();
  c.model_type = "gpt2";
  EXPECT_THROW(c.validate(), std::exception);
  // hidden_act is likewise silu-only.
  EXPECT_THROW(Config::from_json_string(R"({"hidden_act": "gelu"})",
                                        Config::tiny_test()),
               std::runtime_error);
}

TEST(LFM25Config, MaxSeqLenRejected) {
  // Supported KV-cache window is [1, 131072].
  EXPECT_THROW(Config::from_json_string(
                   R"({"max_position_embeddings": 131073})"),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"max_seq_len": 1000000})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_THROW(Config::from_json_string(R"({"max_position_embeddings": 0})",
                                        Config::tiny_test()),
               std::exception);
  EXPECT_NO_THROW(Config::from_json_string(
      R"({"max_position_embeddings": 131072})", Config::tiny_test()));
  Config c     = Config::tiny_test();
  c.max_seq_len = 131072 + 1;
  EXPECT_THROW(c.validate(), std::exception);
}
