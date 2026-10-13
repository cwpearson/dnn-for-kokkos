// Tests for the native Gemma-2-2B-IT text encoder (examples/sana).
//
// Fast path (no downloads): Config defaults/validate/JSON parsing,
// tiny-config random-weight forward (finite + shape + determinism +
// padding-mask invariance), and the Sana select_index helper.
//
// Env-gated path (never downloads weights):
//   KOKKOSDNN_SANA_GEMMA_TOKENIZER  tokenizer.json for tokenizer sanity
//   KOKKOSDNN_SANA_GEMMA_WEIGHTS    .safetensors file or directory holding
//                                  the gemma-2-2b-it checkpoint (+ optional
//                                  sibling config.json / tokenizer.json)
//   KOKKOSDNN_SANA_GEMMA_CONFIG     config.json override (default: sibling
//                                  of the weights path)
//   KOKKOSDNN_SANA_GEMMA_GOLDEN_JSON reference JSON from
//                                  generate_gemma_golden.py (prompt + mean /
//                                  std / min / max / first8 of the [300,2304]
//                                  caption features)
//   KOKKOSDNN_SANA_GEMMA_TOL        abs tolerance override for mean/std
//                                  (default 5e-2; the torch reference runs in
//                                  BF16 while this encoder is FP32)
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_SanaTextEncoder.hpp"

namespace {

namespace SG = KokkosDNN::SanaGemma;
using Exec = Kokkos::DefaultExecutionSpace;

const char* getenv_str(const char* name) {
  const char* p = std::getenv(name);
  return (p && *p) ? p : nullptr;
}

bool file_readable(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

bool json_number(const std::string& text, const std::string& key,
                 double& out) {
  const std::string q = "\"" + key + "\"";
  size_t p = text.find(q);
  if (p == std::string::npos) return false;
  p = text.find(':', p + q.size());
  if (p == std::string::npos) return false;
  const char* begin = text.c_str() + p + 1;
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return false;
  out = v;
  return true;
}

bool json_string(const std::string& text, const std::string& key,
                 std::string& out) {
  const std::string q = "\"" + key + "\"";
  size_t p = text.find(q);
  if (p == std::string::npos) return false;
  p = text.find(':', p + q.size());
  if (p == std::string::npos) return false;
  size_t a = text.find('"', p + 1);
  if (a == std::string::npos) return false;
  size_t b = text.find('"', a + 1);
  if (b == std::string::npos) return false;
  out = text.substr(a + 1, b - a - 1);
  return true;
}

TEST(SanaTextEncoder, ConfigDefaults) {
  SG::Config c = SG::Config::gemma2_2b_it();
  EXPECT_EQ(c.hidden_size, 2304);
  EXPECT_EQ(c.num_hidden_layers, 26);
  EXPECT_EQ(c.num_attention_heads, 8);
  EXPECT_EQ(c.num_key_value_heads, 4);
  EXPECT_EQ(c.head_dim, 256);
  EXPECT_EQ(c.intermediate_size, 9216);
  EXPECT_EQ(c.vocab_size, 256000);
  EXPECT_EQ(c.sliding_window, 4096);
  EXPECT_FLOAT_EQ(c.query_pre_attn_scalar, 256.0f);
  EXPECT_FLOAT_EQ(c.attn_logit_softcapping, 50.0f);
  EXPECT_FLOAT_EQ(c.rope_theta, 10000.0f);
  EXPECT_FLOAT_EQ(c.rms_norm_eps, 1e-6f);
  // Alternation starts with sliding (layer 0 sliding, layer 1 full).
  ASSERT_EQ(c.is_full_layer.size(), 26u);
  EXPECT_EQ(c.is_full_layer[0], 0);
  EXPECT_EQ(c.is_full_layer[1], 1);
  EXPECT_EQ(c.is_full_layer[24], 0);
  EXPECT_EQ(c.is_full_layer[25], 1);
  EXPECT_NO_THROW(c.validate());
  // Attention scale is 1/sqrt(256) = 1/16 (differs from Gemma4's 1.0).
  EXPECT_FLOAT_EQ(c.attn_scale(), 0.0625f);
}

TEST(SanaTextEncoder, ConfigValidateRejectsBad) {
  SG::Config c = SG::Config::gemma2_2b_it();
  SG::Config bad = c;
  bad.attention_bias = true;
  EXPECT_THROW(bad.validate(), std::runtime_error);
  bad = c;
  bad.hidden_activation = "gelu";
  EXPECT_THROW(bad.validate(), std::runtime_error);
  bad = c;
  bad.num_attention_heads = 7;  // not a multiple of 4 kv heads
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = c;
  bad.is_full_layer.resize(3);
  EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(SanaTextEncoder, ConfigFromJson) {
  // New-style config with rope_parameters + explicit layer_types.
  const std::string js = R"({
    "hidden_size": 2304, "num_hidden_layers": 26,
    "num_attention_heads": 8, "num_key_value_heads": 4,
    "head_dim": 256, "intermediate_size": 9216, "vocab_size": 256000,
    "rms_norm_eps": 1e-06, "max_position_embeddings": 8192,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "sliding_window": 4096, "query_pre_attn_scalar": 256,
    "attn_logit_softcapping": 50.0, "hidden_activation": "gelu_pytorch_tanh",
    "model_type": "gemma2", "attention_bias": false,
    "tie_word_embeddings": true, "bos_token_id": 2, "eos_token_id": 1,
    "pad_token_id": 0
  })";
  // layer_types parsed separately (length must equal num_hidden_layers).
  std::string layers = "[";
  for (int i = 0; i < 26; ++i) {
    if (i) layers += ",";
    layers += (i % 2 == 0) ? "\"sliding_attention\"" : "\"full_attention\"";
  }
  layers += "]";
  const std::string js_full =
      js.substr(0, js.size() - 1) + ",\"layer_types\":" + layers + "}";
  SG::Config c = SG::Config::from_json_string(js_full);
  EXPECT_EQ(c.hidden_size, 2304);
  EXPECT_EQ(c.num_hidden_layers, 26);
  EXPECT_FLOAT_EQ(c.rope_theta, 10000.0f);
  ASSERT_EQ(c.is_full_layer.size(), 26u);
  EXPECT_EQ(c.is_full_layer[0], 0);
  EXPECT_EQ(c.is_full_layer[1], 1);
  // Legacy top-level rope_theta form.
  SG::Config c2 =
      SG::Config::from_json_string(R"({"rope_theta": 20000.0})");
  EXPECT_FLOAT_EQ(c2.rope_theta, 20000.0f);
  EXPECT_THROW(SG::Config::from_json_string(R"({"hidden_activation": "silu"})"),
               std::runtime_error);
  EXPECT_THROW(SG::Config::from_json_string(R"({"attention_bias": true})"),
               std::runtime_error);
}

TEST(SanaTextEncoder, SelectIndex) {
  // Identity when the tokenize already yields exactly max_length.
  auto sel = SG::sana_select_index(300, 300);
  ASSERT_EQ(sel.size(), 300u);
  for (int i = 0; i < 300; ++i) EXPECT_EQ(sel[(size_t)i], i);
  // Longer tokenize: BOS + last (max_length - 1).
  sel = SG::sana_select_index(310, 300);
  ASSERT_EQ(sel.size(), 300u);
  EXPECT_EQ(sel[0], 0);
  EXPECT_EQ(sel[1], 11);
  EXPECT_EQ(sel[299], 309);
  EXPECT_THROW(SG::sana_select_index(299, 300), std::invalid_argument);
}

TEST(SanaTextEncoder, TinySmokeForward) {
  SG::Config cfg = SG::Config::tiny_test();
  ASSERT_NO_THROW(cfg.validate());
  SG::Weights<Exec> w;
  SG::random_init(cfg, w, 1234);
  SG::Model<Exec> model(cfg, w);

  // Six ids: BOS + four content ids + one pad (mask 0).
  const std::vector<int> ids = {2, 10, 20, 30, 40, 0};
  const std::vector<char> mask = {1, 1, 1, 1, 1, 0};
  std::vector<float> h = model.forward(ids, mask);
  ASSERT_EQ(h.size(), 6u * 32u);
  for (float v : h) EXPECT_TRUE(std::isfinite(v)) << "non-finite hidden";

  // Determinism: a second forward agrees bit-for-bit.
  std::vector<float> h2 = model.forward(ids, mask);
  ASSERT_EQ(h2.size(), h.size());
  for (size_t i = 0; i < h.size(); ++i)
    EXPECT_EQ(h2[i], h[i]) << "nondeterministic at " << i;

  // Padding invariance: changing pad-region ids (and adding trailing pads)
  // must not change the real-prefix rows (causal + key-masked attention).
  const std::vector<int> ids_b = {2, 10, 20, 30, 40, 99};
  std::vector<float> hb = model.forward(ids_b, mask);
  ASSERT_EQ(hb.size(), h.size());
  for (size_t t = 0; t < 5; ++t)
    for (int d = 0; d < 32; ++d)
      EXPECT_EQ(hb[t * 32 + d], h[t * 32 + d])
          << "pad id leaked into row " << t;

  // Scratch reuse across sizes: grow to T=8, then shrink back to T=6 —
  // the T=6 rows must be unaffected by stale larger-scratch contents.
  const std::vector<int> ids8 = {2, 10, 20, 30, 40, 50, 60, 0};
  const std::vector<char> mask8 = {1, 1, 1, 1, 1, 1, 1, 0};
  std::vector<float> h8 = model.forward(ids8, mask8);
  ASSERT_EQ(h8.size(), 8u * 32u);
  for (float v : h8) EXPECT_TRUE(std::isfinite(v));
  std::vector<float> h_again = model.forward(ids, mask);
  ASSERT_EQ(h_again.size(), h.size());
  for (size_t i = 0; i < h.size(); ++i)
    EXPECT_EQ(h_again[i], h[i]) << "stale scratch leaked at " << i;
}

TEST(SanaTextEncoder, TinyEncodePrompt) {
  SG::Config cfg = SG::Config::tiny_test();
  SG::Weights<Exec> w;
  SG::random_init(cfg, w, 7);
  SG::Model<Exec> model(cfg, w);
  // encode_prompt needs a Tokenizer only for id mapping; exercise the
  // select_index + gather path via a stub is covered by SelectIndex, so
  // here just check the Model rejects empty input.
  EXPECT_THROW(model.forward({}, {}), std::invalid_argument);
  EXPECT_THROW(model.forward({1, 2}, {1}), std::invalid_argument);
}

// Tokenizer.json sanity against the real Gemma tokenizer (gated download
// done outside tests; path via env or the in-repo fetch dir).
TEST(SanaTextEncoder, TokenizerLoadsAndRoundtrips) {
  std::string tok_path;
  if (const char* e = getenv_str("KOKKOSDNN_SANA_GEMMA_TOKENIZER"))
    tok_path = e;
#ifdef SANA_WORKSPACE_DIR
  if (tok_path.empty() || !file_readable(tok_path))
    tok_path = std::string(SANA_WORKSPACE_DIR) + "/gemma-2-2b-it/tokenizer.json";
#endif
  if (tok_path.empty() || !file_readable(tok_path))
    GTEST_SKIP() << "set KOKKOSDNN_SANA_GEMMA_TOKENIZER=<tokenizer.json> "
                 << "(tests never download weights/tokenizers).";
  SG::Tokenizer tok(tok_path);
  EXPECT_TRUE(tok.loaded());
  // Real Gemma2 tokenizer: 256k vocab, BOS 2 / EOS 1 / PAD 0.
  EXPECT_EQ(tok.bos(), 2);
  EXPECT_EQ(tok.eos(), 1);
  EXPECT_EQ(tok.pad(), 0);
  std::vector<int> ids = tok.encode("a cyberpunk cat", true);
  ASSERT_GE(ids.size(), 2u);
  EXPECT_EQ(ids[0], 2);  // BOS first, matching add_special_tokens=True
  std::string back = tok.decode(ids, true);
  EXPECT_EQ(back, "a cyberpunk cat");
  // Empty prompt encodes to BOS only (the Sana null-caption path).
  EXPECT_EQ(tok.encode("", true), std::vector<int>({2}));
}

// Golden forward vs HF transformers Gemma2Model last-hidden states
// (generate_gemma_golden.py output). Loads the real checkpoint: slowish
// (~10 s load + ~1 min FP32 Serial forward for 300 tokens x 26 layers),
// hence env-gated.
TEST(SanaTextEncoder, GoldenVsTransformers) {
  const char* wenv = getenv_str("KOKKOSDNN_SANA_GEMMA_WEIGHTS");
  const char* genv = getenv_str("KOKKOSDNN_SANA_GEMMA_GOLDEN_JSON");
  if (!wenv || !genv)
    GTEST_SKIP() << "set KOKKOSDNN_SANA_GEMMA_WEIGHTS=<.safetensors|dir> and "
                 << "KOKKOSDNN_SANA_GEMMA_GOLDEN_JSON=<golden.json> "
                 << "(tests never download weights).";
  double tol = 5e-2;
  if (const char* t = getenv_str("KOKKOSDNN_SANA_GEMMA_TOL"))
    tol = std::stod(t);

  std::vector<std::string> files = SG::collect_safetensors(wenv);
  std::string cfg_path;
  if (const char* c = getenv_str("KOKKOSDNN_SANA_GEMMA_CONFIG"))
    cfg_path = c;
  else
    cfg_path = SG::sibling_file(wenv, "config.json");
  std::string tok_path;
  if (const char* c = getenv_str("KOKKOSDNN_SANA_GEMMA_TOKENIZER"))
    tok_path = c;
  else
    tok_path = SG::sibling_file(wenv, "tokenizer.json");
  if (tok_path.empty() || !file_readable(tok_path))
    GTEST_SKIP() << "tokenizer.json not found next to weights; set "
                 << "KOKKOSDNN_SANA_GEMMA_TOKENIZER.";
  SG::Config cfg = cfg_path.empty()
                       ? SG::Config::gemma2_2b_it()
                       : SG::Config::from_json_file(cfg_path);
  SG::Tokenizer tok(tok_path);
  SG::Weights<Exec> w;
  SG::load_safetensors(cfg, files, w);
  SG::Model<Exec> model(cfg, w);

  std::ifstream f(genv, std::ios::binary);
  if (!f) GTEST_SKIP() << "golden JSON not readable: " << genv;
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string js = ss.str();
  std::string prompt = "a cyberpunk cat";
  (void)json_string(js, "prompt", prompt);
  double rmean = 0, rstd = 0, rmin = 0, rmax = 0;
  if (!json_number(js, "mean", rmean) || !json_number(js, "std", rstd) ||
      !json_number(js, "min", rmin) || !json_number(js, "max", rmax))
    FAIL() << "golden JSON lacks mean/std/min/max: " << genv;

  std::vector<float> feats = SG::encode_prompt(cfg, tok, model, prompt, 300);
  ASSERT_EQ(feats.size(), 300u * 2304u);
  double mean = 0, mn = feats[0], mx = feats[0];
  for (float v : feats) {
    EXPECT_TRUE(std::isfinite(v));
    mean += v;
    mn = std::min(mn, (double)v);
    mx = std::max(mx, (double)v);
  }
  mean /= (double)feats.size();
  double var = 0;
  for (float v : feats) var += (v - mean) * (v - mean);
  const double std = std::sqrt(var / (double)feats.size());
  std::printf("[gemma golden] prompt=\"%s\" mean=%.6f std=%.6f min=%.4f "
              "max=%.4f (ref %.6f %.6f %.4f %.4f) tol=%.3g\n",
              prompt.c_str(), mean, std, mn, mx, rmean, rstd, rmin, rmax,
              tol);
  EXPECT_NEAR(mean, rmean, tol) << "caption mean vs transformers reference";
  EXPECT_NEAR(std, rstd, tol) << "caption std vs transformers reference";

  // Null caption encodes finite too (the unconditional branch).
  std::vector<float> null_feats = SG::encode_prompt(cfg, tok, model, "", 300);
  ASSERT_EQ(null_feats.size(), 300u * 2304u);
  for (float v : null_feats) EXPECT_TRUE(std::isfinite(v));
}

}  // namespace
