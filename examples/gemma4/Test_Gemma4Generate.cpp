// End-to-end Gemma4 text: presets, fixtures, tiny/full generate smoke,
// and a gated real-weight check (skips without files, never downloads).
// No weight downloads: random weights + checked-in config fixture.
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
#include "KokkosDNN_Gemma4Config.hpp"
#include "KokkosDNN_Gemma4Model.hpp"
#include "KokkosDNN_Gemma4Tokenizer.hpp"
#include "KokkosDNN_Gemma4Weights.hpp"

namespace {

std::string gemma4_source_dir() {
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

void check_gemma4_e2b_it_dims(const KokkosDNN::Gemma4::Config& c) {
  EXPECT_EQ(c.dim, 1536);
  EXPECT_EQ(c.n_layers, 35);
  EXPECT_EQ(c.n_heads, 8);
  EXPECT_EQ(c.n_kv_heads, 1);
  EXPECT_EQ(c.vocab_size, 262144);
  EXPECT_EQ(c.ffn_dim, 6144);
  EXPECT_EQ(c.head_dim_sliding, 256);
  EXPECT_EQ(c.head_dim_full, 512);
  EXPECT_FLOAT_EQ(c.attn_scale, 1.0f);  // QK-normed, NOT 1/sqrt(d)
  EXPECT_FLOAT_EQ(c.rms_norm_eps, 1e-6f);
  EXPECT_FLOAT_EQ(c.rope_theta_sliding, 10000.0f);
  EXPECT_FLOAT_EQ(c.rope_theta_full, 1000000.0f);
  EXPECT_FLOAT_EQ(c.partial_rotary_factor, 0.25f);
  EXPECT_EQ(c.sliding_window, 512);
  EXPECT_EQ(c.num_kv_shared_layers, 20);
  EXPECT_EQ(c.kv_shared_start(), 15);
  EXPECT_TRUE(c.use_double_wide_mlp);
  EXPECT_EQ(c.ple_dim, 256);
  EXPECT_EQ(c.ple_vocab, 262144);
  EXPECT_FLOAT_EQ(c.final_logit_softcap, 30.0f);
  EXPECT_TRUE(c.tie_word_embeddings);
  EXPECT_EQ(c.hidden_activation, "gelu_pytorch_tanh");
  EXPECT_FALSE(c.attention_bias);
  EXPECT_EQ(c.bos_token, 2);
  EXPECT_EQ(c.eos_token, 1);
  EXPECT_EQ(c.pad_token, 0);
  ASSERT_EQ(c.is_full_layer.size(), 35u);
  // 4x sliding + 1x full repeat, last forced full.
  for (int l = 0; l < 35; ++l)
    EXPECT_EQ((bool)c.is_full(l), (l % 5 == 4 || l == 34))
        << "layer " << l;
  // MLP widths: 6144 below the sharing threshold, 12288 above.
  for (int l = 0; l < 35; ++l)
    EXPECT_EQ(c.mlp_dim(l), l < 15 ? 6144 : 12288) << "layer " << l;
  // Head dims per type.
  for (int l = 0; l < 35; ++l)
    EXPECT_EQ(c.head_dim(l), c.is_full(l) ? 512 : 256) << "layer " << l;
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

TEST(Gemma4Preset, FactoryValues) {
  using KokkosDNN::Gemma4::Config;
  auto e2b = Config::gemma4_e2b_it();
  check_gemma4_e2b_it_dims(e2b);
  EXPECT_NO_THROW(e2b.validate());

  auto tiny = Config::tiny_test();
  EXPECT_EQ(tiny.dim, 32);
  EXPECT_EQ(tiny.n_layers, 4);
  EXPECT_EQ(tiny.kv_shared_start(), 2);
  EXPECT_NO_THROW(tiny.validate());
}

TEST(Gemma4Preset, FixtureParse) {
  using KokkosDNN::Gemma4::Config;
  const std::string dir = gemma4_source_dir();
  // Bare text_config fixture.
  Config from_fixture =
      Config::from_json_file(dir + "/config-gemma4-e2b-it.json");
  check_gemma4_e2b_it_dims(from_fixture);
  EXPECT_NO_THROW(from_fixture.validate());
}

TEST(Gemma4Preset, KVSharingStructure) {
  using KokkosDNN::Gemma4::Config;
  auto c = Config::gemma4_e2b_it();
  // 12 non-shared sliding caches (L0-3,5-8,10-13) + 3 full (L4,9,14).
  EXPECT_EQ(c.num_sliding_caches(), 12);
  EXPECT_EQ(c.num_full_caches(), 3);
  // Full-window BF16 KV bytes: (12*256 + 3*512) * max_seq * 2B * 2 (K+V).
  const size_t expect =
      (size_t)(12 * 256 + 3 * 512) * (size_t)c.max_seq_len * 2 * 2;
  EXPECT_EQ(c.kv_cache_bytes(2), expect);
  // tiny: L0 sliding + L1 full donors, L2/L3 shared.
  auto t = Config::tiny_test();
  EXPECT_EQ(t.num_sliding_caches(), 1);
  EXPECT_EQ(t.num_full_caches(), 1);
  EXPECT_TRUE(t.is_kv_shared(2));
  EXPECT_TRUE(t.is_kv_shared(3));
  EXPECT_FALSE(t.is_kv_shared(0));
  EXPECT_FALSE(t.is_kv_shared(1));
}

// Layer_scalar loads as a host float (BF16 [1] on disk); random_init
// defaults it to 1.0.
TEST(Gemma4Preset, LayerScalarDefault) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::Gemma4::Config::tiny_test();
  KokkosDNN::Gemma4::Weights<Exec, float> w;
  KokkosDNN::Gemma4::random_init(cfg, w, 1234);
  for (int l = 0; l < cfg.n_layers; ++l)
    EXPECT_FLOAT_EQ(w.blocks[(size_t)l].layer_scalar, 1.0f);
  // Shared layers carry no k/v projections (dead on-disk weight).
  for (int l = 0; l < cfg.n_layers; ++l) {
    if (cfg.is_kv_shared(l)) {
      EXPECT_EQ(w.blocks[(size_t)l].wk.size(), 0u);
      EXPECT_EQ(w.blocks[(size_t)l].wv.size(), 0u);
    } else {
      EXPECT_GT(w.blocks[(size_t)l].wk.size(), 0u);
      EXPECT_GT(w.blocks[(size_t)l].wv.size(), 0u);
    }
  }
}

template <class Scalar>
void check_gemma4_generate() {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::Gemma4::Config::tiny_test();
  KokkosDNN::Gemma4::Weights<Exec, Scalar> w;
  KokkosDNN::Gemma4::random_init(cfg, w, 1234);
  KokkosDNN::Gemma4::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {2, 10, 20, 30};
  // Forward smoke: one logit row per vocab id.
  auto toks = make_tok_view<Exec, Scalar>(prompt);
  auto lg = model.forward(toks);
  ASSERT_EQ(lg.extent(0), (size_t)cfg.vocab_size);
  // Final logit softcapping bounds every logit by the cap (30).
  {
    auto h = Kokkos::create_mirror_view(lg);
    Kokkos::deep_copy(h, lg);
    for (size_t i = 0; i < h.extent(0); ++i) {
      float v = static_cast<float>(h(i));
      EXPECT_LE(std::abs(v), 30.0f + 1e-3f);
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
  // Determinism: rebuild and compare.
  KokkosDNN::Gemma4::Model<Exec, Scalar> model2(cfg, w);
  EXPECT_EQ(model2.generate(prompt, 2), out);
}

TEST(Gemma4Generate, Float) { check_gemma4_generate<float>(); }

TEST(Gemma4Generate, BHalf) {
  check_gemma4_generate<Kokkos::Experimental::bhalf_t>();
}

// Tied lm_head: E2B-it ships tie_word_embeddings=true, so a checkpoint
// without any lm_head tensor loads by aliasing the embedding table.
TEST(Gemma4Generate, TiedLmHeadDefault) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = KokkosDNN::Gemma4::Config::tiny_test();
  EXPECT_TRUE(cfg.tie_word_embeddings);
  KokkosDNN::Gemma4::Weights<Exec, float> w;
  KokkosDNN::Gemma4::random_init(cfg, w, 1234);
  // random_init fills both tables; emulate tying by aliasing explicitly.
  w.lm_head = w.embed;
  EXPECT_EQ(w.lm_head.data(), w.embed.data());
  KokkosDNN::Gemma4::Model<Exec, float> model(cfg, w);
  auto out = model.generate({2, 10, 20}, 2);
  EXPECT_EQ(out.size(), 5u);
}

// Real-weight check (gated; SKIP without files, never downloads).
// Needs ~10 GiB RAM for BF16 E2B-it weights plus the tokenizer:
//   KOKKOSDNN_WEIGHTS        model.safetensors (single 10 GiB file)
//   KOKKOSDNN_TOKENIZER_JSON tokenizer.json
TEST(Gemma4Generate, RealWeightsBHalf) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = Kokkos::Experimental::bhalf_t;
  using namespace KokkosDNN::Gemma4;
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

  Tokenizer tok(tenv);                    // throws -> FAIL
  Config cfg = Config::gemma4_e2b_it();   // full E2B shapes
  cfg.max_seq_len = 512;                  // cap KV cache for the smoke prompt
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
  std::printf("[gemma4:real] prompt ids=%zu argmax=%d top_logit=%g\n",
              ids.size(), best, (double)static_cast<float>(h((size_t)best)));
  EXPECT_GE(best, 0);
  EXPECT_LT(best, cfg.vocab_size);
  for (size_t i = 0; i < h.extent(0); ++i)
    EXPECT_LE(std::abs(static_cast<float>(h(i))), 30.0f + 1e-3f);
}
