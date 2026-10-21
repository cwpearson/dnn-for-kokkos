#include <gtest/gtest.h>

#include <cstdlib>

#include <Kokkos_Core.hpp>

#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroModel.hpp"
#include "KokkosDNN_KokoroPth.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

// Real-weights golden test (env-gated, like the Llama tokenizer oracle):
//   KOKORO_PTH=/path/kokoro-v1_0.pth KOKORO_VOICE=/path/af_heart.pt
//   KOKORO_CONFIG=examples/kokoro/config-kokoro-v1_0.json
// Verifies the full pipeline runs end-to-end on the real checkpoint and
// produces finite 24kHz audio of the expected length (600 * sum(durations)).
TEST(KokoroGolden, RealWeightsEndToEnd) {
  const char* pth = std::getenv("KOKORO_PTH");
  const char* voice = std::getenv("KOKORO_VOICE");
  const char* cfgp = std::getenv("KOKORO_CONFIG");
  if (!pth || !voice) GTEST_SKIP() << "set KOKORO_PTH + KOKORO_VOICE";
  std::string cfg_path =
      cfgp ? cfgp : "examples/kokoro/config-kokoro-v1_0.json";
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> handle;
  KokkosDNN::Kokoro::Vocab vocab;
  KokkosDNN::Kokoro::Config cfg =
      KokkosDNN::Kokoro::load_config(cfg_path, vocab);
  auto ar = KokkosDNN::Kokoro::open_kokoro_pth(pth);
  EXPECT_EQ(ar.tensors.size(), 548u);
  EXPECT_GT(KokkosDNN::Kokoro::kokoro_total_params(ar), 80000000);
  KokkosDNN::Kokoro::Weights<Exec, float> w;
  KokkosDNN::Kokoro::load_kokoro_pth<Exec, float>(cfg, pth, w);
  const std::string phonemes = "h\xc9\x9b\xcb\x88lo\xca\x8a";
  int idx = (int)phonemes.size() - 1;
  Kokkos::View<float*, Exec> ref_s;
  KokkosDNN::Kokoro::load_voice_row<Exec, float>(voice, idx, ref_s);
  EXPECT_EQ(ref_s.extent(0), 256u);
  auto out = KokkosDNN::Kokoro::synthesize<Exec>(cfg, handle, w, vocab,
                                                  phonemes, ref_s, 1.0f);
  EXPECT_GT(out.audio.size(), 0u);
  size_t frames = 0;
  for (int d : out.durations) frames += (size_t)d;
  EXPECT_EQ(out.audio.size(), 600 * frames);
  for (float v : out.audio) EXPECT_TRUE(std::isfinite(v));
  double e = 0;
  for (float v : out.audio) e += (double)v * v;
  e /= out.audio.size();
  EXPECT_GT(e, 1e-8);
}
