#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroModel.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

TEST(KokoroSmoke, TinyForward) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> handle;
  KokkosDNN::Kokoro::Config cfg;
  cfg.hidden_dim = 16;
  cfg.style_dim = 8;
  cfg.plbert_hidden_size = 32;
  cfg.plbert_heads = 4;
  cfg.plbert_intermediate = 64;
  cfg.plbert_layers = 2;
  cfg.upsample_initial_channel = 32;
  cfg.max_tokens = 64;
  cfg.max_frames = 256;
  cfg.max_samples = 1 << 20;
  KokkosDNN::Kokoro::Vocab vocab;
  for (int c = 0; c < 26; ++c)
    vocab[std::string(1, (char)('a' + c))] = 1 + c;
  vocab[" "] = 27;
  KokkosDNN::Kokoro::Weights<Exec, float> w;
  KokkosDNN::Kokoro::random_init<Exec, float>(cfg, w, 42);
  Kokkos::View<float*, Exec> ref_s("ref", cfg.style_dim);
  Kokkos::deep_copy(ref_s, 0.1f);
  auto out = KokkosDNN::Kokoro::synthesize<Exec>(cfg, handle, w, vocab,
                                                  "hello", ref_s, 1.0f);
  EXPECT_GT(out.audio.size(), 0u);
  EXPECT_GT(out.durations.size(), 0u);
  for (float v : out.audio) EXPECT_TRUE(std::isfinite(v));
}
