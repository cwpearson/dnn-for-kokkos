// Kokoro .pth inventory test: missing files throw; with
// KOKKOSDNN_KOKORO_WEIGHTS set, the real checkpoint parses to the expected
// tensor inventory (names/shapes/counts vs the released kokoro-v1_0.pth).
// With KOKKOSDNN_KOKORO_VOICE set, the voice pack loads as [510, 256].
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "KokkosDNN_KokoroPth.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

namespace {

const char* env(const char* k) { return std::getenv(k); }

TEST(KokoroPth, MissingFileThrows) {
  EXPECT_THROW(KokkosDNN::Kokoro::open_pth("/nonexistent/kokoro-v1_0.pth"),
               std::runtime_error);
}

TEST(KokoroPth, InventoryMatchesRelease) {
  const char* p = env("KOKKOSDNN_KOKORO_WEIGHTS");
  if (!p) GTEST_SKIP() << "set KOKKOSDNN_KOKORO_WEIGHTS to run";
  KokkosDNN::Kokoro::PthArchive a = KokkosDNN::Kokoro::open_pth(p);
  // 25 (bert) + 2 (bert_encoder) + 122 (predictor) + 375 (decoder) + 24
  // (text_encoder) = 548 tensors.
  EXPECT_EQ(a.tensors.size(), 548u);
  EXPECT_EQ(KokkosDNN::Kokoro::pth_total_params(a), 81763410);
  auto has = [&](const std::string& name,
                 const std::vector<int64_t>& shape) {
    for (auto& t : a.tensors) {
      if (t.name == name) {
        EXPECT_EQ(t.shape, shape) << name;
        EXPECT_EQ(t.dtype, "torch.FloatStorage") << name;
        return;
      }
    }
    FAIL() << "missing tensor '" << name << "'";
  };
  has("bert.module.embeddings.word_embeddings.weight", {178, 128});
  has("bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
      {768, 768});
  has("bert_encoder.module.weight", {512, 768});
  has("predictor.module.text_encoder.lstms.0.weight_ih_l0", {1024, 640});
  has("predictor.module.duration_proj.linear_layer.weight", {50, 512});
  has("predictor.module.F0.1.pool.weight_v", {512, 1, 3});
  has("predictor.module.F0_proj.weight", {1, 256, 1});
  has("text_encoder.module.cnn.1.0.weight_v", {512, 512, 5});
  has("text_encoder.module.lstm.weight_ih_l0_reverse", {1024, 512});
  has("decoder.module.encode.conv1.weight_v", {1024, 514, 3});
  has("decoder.module.decode.3.pool.weight_v", {1090, 1, 3});
  has("decoder.module.asr_res.0.weight_v", {64, 512, 1});
  has("decoder.module.generator.ups.0.weight_v", {512, 256, 20});
  has("decoder.module.generator.resblocks.5.convs2.2.weight_v", {128, 128, 11});
  has("decoder.module.generator.noise_convs.0.weight", {256, 22, 12});
  has("decoder.module.generator.conv_post.weight_v", {22, 128, 7});
  has("decoder.module.generator.m_source.l_linear.weight", {1, 9});
  // Spot-read one storage: word embeddings (first tensor, storage "1"?).
  // Just check the read path works and returns the right count.
  std::vector<float> e =
      KokkosDNN::Kokoro::read_pth_storage(a, a.tensors[0].storage_key);
  EXPECT_EQ((int64_t)e.size(), a.tensors[0].storage_numel);
}

TEST(KokoroPth, VoicePackLoads) {
  const char* p = env("KOKKOSDNN_KOKORO_VOICE");
  if (!p) GTEST_SKIP() << "set KOKKOSDNN_KOKORO_VOICE to run";
  KokkosDNN::Kokoro::PthArchive a = KokkosDNN::Kokoro::open_pth(p);
  ASSERT_EQ(a.tensors.size(), 1u);
  EXPECT_EQ(a.tensors[0].shape, (std::vector<int64_t>{510, 1, 256}));
  KokkosDNN::Kokoro::VoicePack vp = KokkosDNN::Kokoro::load_voice(p);
  EXPECT_EQ(vp.n_rows, 510);
  EXPECT_EQ(vp.rows.size(), 510u * 256u);
}

}  // namespace
