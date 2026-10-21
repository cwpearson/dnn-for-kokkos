// Kokoro tiny smoke test: random weights, no files. Verifies the full
// synthesize() path runs, honors the length contract (audio == 300 * 2F
// samples), produces finite output, and is deterministic given the seed.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroModel.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = KokkosDNN::Kokoro::DefaultScalar;

struct Tiny {
  KokkosDNN::Kokoro::Config cfg;
  KokkosDNN::Kokoro::Weights<Exec, Scalar> w;
  std::vector<float> style;
  std::vector<int> ids;
  Tiny() {
    cfg = KokkosDNN::Kokoro::tiny_test_config();
    KokkosDNN::Kokoro::random_init(cfg, w, 1234);
    style.assign(2 * (size_t)cfg.style_dim, 0.0f);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : style) v = u(rng);
    ids = {1, 2, 3, 4, 5};
  }
};

TEST(KokoroSmoke, TinySynthesizeLengthAndFinite) {
  Tiny t;
  KokkosDNN::Kokoro::Model<Exec, Scalar> m(t.cfg, t.w);
  KokkosDNN::Kokoro::Intermediates dbg;
  std::vector<float> audio = m.synthesize(t.ids, t.style, 1.0f, 42, &dbg);
  ASSERT_GT(dbg.frames, 0);
  for (int d : dbg.pred_dur) EXPECT_GE(d, 1);
  EXPECT_EQ(audio.size(), (size_t)dbg.frames * 2 * 300);
  ASSERT_EQ(dbg.f0.size(), (size_t)dbg.frames * 2);
  ASSERT_EQ(dbg.ncurve.size(), (size_t)dbg.frames * 2);
  ASSERT_EQ(dbg.asr.size(),
            (size_t)dbg.frames * (size_t)t.cfg.hidden_dim);
  for (float v : audio) EXPECT_TRUE(std::isfinite(v)) << "non-finite audio";
  for (float v : dbg.f0) EXPECT_TRUE(std::isfinite(v));
  for (float v : dbg.asr) EXPECT_TRUE(std::isfinite(v));
  // Non-silent: durations come from random weights, audio must carry energy.
  double e = 0.0;
  for (float v : audio) e += (double)v * v;
  EXPECT_GT(e, 0.0);
}

TEST(KokoroSmoke, TinyDeterministicGivenSeed) {
  Tiny t;
  KokkosDNN::Kokoro::Model<Exec, Scalar> m(t.cfg, t.w);
  std::vector<float> a1 = m.synthesize(t.ids, t.style, 1.0f, 42, nullptr);
  std::vector<float> a2 = m.synthesize(t.ids, t.style, 1.0f, 42, nullptr);
  ASSERT_EQ(a1.size(), a2.size());
  for (size_t i = 0; i < a1.size(); ++i) EXPECT_EQ(a1[i], a2[i]) << i;
  std::vector<float> a3 = m.synthesize(t.ids, t.style, 1.0f, 43, nullptr);
  ASSERT_EQ(a1.size(), a3.size());
  double diff = 0.0;
  for (size_t i = 0; i < a1.size(); ++i) diff += std::fabs(a1[i] - a3[i]);
  EXPECT_GT(diff, 0.0);  // SineGen seed actually feeds the output
}

}  // namespace
