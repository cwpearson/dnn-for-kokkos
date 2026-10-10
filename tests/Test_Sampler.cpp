// Sampler unit tests: temperature / top-k / top-p / min-p, repetition
// penalty, seeded RNG determinism, greedy equivalence. No weight downloads;
// fixed logits + fixed seeds only (device tests use tiny views).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <vector>

#include "KokkosDNN_Sampler.hpp"

using KokkosDNN::HostRNG;
using KokkosDNN::SampleOptions;

namespace {

// Fixed logits fixture: argmax is 2.
std::vector<float> fixture_logits() {
  return {0.1f, 1.0f, 3.0f, 2.0f, -1.0f, 0.5f};
}

template <class Scalar = float>
Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> make_device_logits(
    const std::vector<float>& v) {
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> d("logits", v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < v.size(); ++i) h(i) = static_cast<Scalar>(v[i]);
  Kokkos::deep_copy(d, h);
  return d;
}

}  // namespace

TEST(Sampler, GreedyIsArgmax) {
  auto lg = fixture_logits();
  HostRNG rng(0);
  SampleOptions greedy;  // temperature=0
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, greedy, {}, rng), 2);
  // top_k == 1 forces greedy even at nonzero temperature.
  SampleOptions k1;
  k1.temperature = 1.0f;
  k1.top_k = 1;
  HostRNG rng2(7);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, k1, {}, rng2), 2);
  // Device greedy entry agrees (float + bhalf paths compile; float checked).
  auto d = make_device_logits(lg);
  EXPECT_EQ(KokkosDNN::sample_greedy(d), 2);
  HostRNG rng3(123);
  EXPECT_EQ(
      (KokkosDNN::sample_token<Kokkos::DefaultExecutionSpace, float>(
          d, greedy, {}, rng3)),
      2);
}

TEST(Sampler, TemperatureZeroEqualsGreedyDevice) {
  auto lg = fixture_logits();
  auto d = make_device_logits(lg);
  SampleOptions opt;
  opt.temperature = 0.0f;
  opt.top_p = 0.5f;  // ignored under greedy
  opt.min_p = 0.9f;  // ignored under greedy
  HostRNG rng(99);
  EXPECT_EQ(
      (KokkosDNN::sample_token<Kokkos::DefaultExecutionSpace, float>(d, opt, {},
                                                                     rng)),
      KokkosDNN::sample_greedy(d));
}

TEST(Sampler, FixedSeedReproducible) {
  auto lg = fixture_logits();
  SampleOptions opt;
  opt.temperature = 1.0f;
  opt.seed = 12345;
  HostRNG a(opt.seed), b(opt.seed);
  for (int i = 0; i < 16; ++i)
    EXPECT_EQ(KokkosDNN::sample_host_logits(lg, opt, {}, a),
              KokkosDNN::sample_host_logits(lg, opt, {}, b))
        << "iteration " << i;
  // Reseeding reproduces the first draw.
  HostRNG c(opt.seed);
  int first = KokkosDNN::sample_host_logits(lg, opt, {}, c);
  HostRNG d2(opt.seed);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, opt, {}, d2), first);
}

TEST(Sampler, TopKFiltering) {
  // Logits 0..9 ascending: top-3 keep must be {9, 8, 7} (desc order).
  std::vector<float> lg(10);
  for (int i = 0; i < 10; ++i) lg[i] = (float)i;
  auto kept = KokkosDNN::top_k_keep(lg, 3);
  EXPECT_EQ(kept, (std::vector<int>{9, 8, 7}));
  // k <= 0 / k >= V keep everything.
  EXPECT_EQ(KokkosDNN::top_k_keep(lg, 0).size(), 10u);
  EXPECT_EQ(KokkosDNN::top_k_keep(lg, 10).size(), 10u);
  // Sampling with top_k=1 always returns the argmax.
  SampleOptions opt;
  opt.temperature = 2.0f;
  opt.top_k = 1;
  HostRNG rng(5);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, opt, {}, rng), 9);
}

TEST(Sampler, TopPFiltering) {
  // One dominant logit: top_p=0.5 keeps only the argmax; top_p=1 keeps all.
  std::vector<float> lg = {10.0f, 0.0f, 0.0f, 0.0f};
  auto narrow = KokkosDNN::top_p_keep(lg, 0.5f);
  ASSERT_EQ(narrow.size(), 1u);
  EXPECT_EQ(narrow[0], 0);
  EXPECT_EQ(KokkosDNN::top_p_keep(lg, 1.0f).size(), 4u);
  EXPECT_EQ(KokkosDNN::top_p_keep(lg, 0.0f), (std::vector<int>{0}));
  // Two-way split: probs ~ [0.5, 0.5] after [1,1,...]: cumsum needs both.
  std::vector<float> even = {5.0f, 5.0f, -10.0f, -10.0f};
  auto kept = KokkosDNN::top_p_keep(even, 0.9f);
  EXPECT_EQ(kept.size(), 2u);
  EXPECT_TRUE(std::find(kept.begin(), kept.end(), 0) != kept.end());
  EXPECT_TRUE(std::find(kept.begin(), kept.end(), 1) != kept.end());
}

TEST(Sampler, MinPFiltering) {
  std::vector<float> lg = {4.0f, 2.0f, 0.0f, -2.0f};
  // min_p=0 keeps all; huge min_p keeps argmax only.
  EXPECT_EQ(KokkosDNN::min_p_keep(lg, 0.0f).size(), 4u);
  EXPECT_EQ(KokkosDNN::min_p_keep(lg, 1.0f), (std::vector<int>{0}));
  // min_p=0.5 with pmax≈0.9 keeps ids with p >= 0.45: only id 0 here.
  // (softmax([4,2,0,-2]) ≈ [0.865, 0.117, 0.016, 0.002])
  auto kept = KokkosDNN::min_p_keep(lg, 0.5f);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_EQ(kept[0], 0);
  // Lenient min_p keeps the top two.
  auto kept2 = KokkosDNN::min_p_keep(lg, 0.1f);
  ASSERT_GE(kept2.size(), 2u);
  EXPECT_EQ(kept2[0], 0);
  EXPECT_EQ(kept2[1], 1);
}

TEST(Sampler, RepetitionPenalty) {
  // Greedy would pick 0 (5.0); penalized by 2 -> 2.5 < 4.0, picks 1.
  std::vector<float> lg = {5.0f, 4.0f, 1.0f};
  SampleOptions greedy;
  HostRNG rng(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, greedy, {}, rng), 0);
  SampleOptions pen;
  pen.repetition_penalty = 2.0f;
  HostRNG rng2(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, pen, {0}, rng2), 1);
  // Negative-logit side: -1.0 * 2 = -2.0 (more suppressed).
  std::vector<float> work = {-1.0f, -0.5f};
  KokkosDNN::apply_repetition_penalty(work, {0}, 2.0f);
  EXPECT_FLOAT_EQ(work[0], -2.0f);
  EXPECT_FLOAT_EQ(work[1], -0.5f);
  // Penalty 1 is a no-op; dedupe (repeated prior id penalized once).
  std::vector<float> w2 = {5.0f, 4.0f};
  KokkosDNN::apply_repetition_penalty(w2, {0, 0, 0}, 1.0f);
  EXPECT_FLOAT_EQ(w2[0], 5.0f);
  std::vector<float> w3 = {5.0f, 4.0f};
  KokkosDNN::apply_repetition_penalty(w3, {0, 0, 0}, 2.0f);
  EXPECT_FLOAT_EQ(w3[0], 2.5f);
  // Out-of-range prior ids ignored, no throw.
  EXPECT_NO_THROW(
      KokkosDNN::apply_repetition_penalty(w3, {-3, 99999}, 2.0f));
}

TEST(Sampler, SampledIdInRangeAndUsesSeed) {
  std::vector<float> lg(32, 0.0f);
  for (size_t i = 0; i < lg.size(); ++i) lg[i] = (float)i * 0.1f;
  SampleOptions opt;
  opt.temperature = 0.8f;
  opt.top_k = 8;
  opt.top_p = 0.9f;
  opt.min_p = 0.05f;
  opt.seed = 2026;
  HostRNG rng(opt.seed);
  for (int i = 0; i < 8; ++i) {
    int id = KokkosDNN::sample_host_logits(lg, opt, {}, rng);
    EXPECT_GE(id, 0);
    EXPECT_LT(id, 32);
  }
  // Same seed replays the same trajectory through the device entry too.
  auto d = make_device_logits(lg);
  HostRNG r1(opt.seed), r2(opt.seed);
  for (int i = 0; i < 8; ++i) {
    int a = KokkosDNN::sample_token<Kokkos::DefaultExecutionSpace, float>(
        d, opt, {}, r1);
    int b = KokkosDNN::sample_token<Kokkos::DefaultExecutionSpace, float>(
        d, opt, {}, r2);
    EXPECT_EQ(a, b);
  }
}

TEST(Sampler, Llama318BInstructPreset) {
  // Meta's recommended Llama 3.1 Instruct sampling (upstream
  // generation_config.json: do_sample=true, temperature=0.6, top_p=0.9;
  // no top_k / repetition_penalty specified). The preset is sampling, not
  // greedy, and stays valid input for both host and device entries.
  SampleOptions o = SampleOptions::llama_3_1_8b_instruct();
  EXPECT_FLOAT_EQ(o.temperature, 0.6f);
  EXPECT_FLOAT_EQ(o.top_p, 0.9f);
  EXPECT_EQ(o.top_k, 0);
  EXPECT_FLOAT_EQ(o.min_p, 0.0f);
  EXPECT_FLOAT_EQ(o.repetition_penalty, 1.0f);
  EXPECT_FALSE(o.is_greedy());
  auto lg = fixture_logits();
  HostRNG rng(o.seed);
  int id = KokkosDNN::sample_host_logits(lg, o, {}, rng);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, (int)lg.size());
}
