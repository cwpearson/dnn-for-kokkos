// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Sampler unit tests: temperature / top-k / top-p / min-p, repetition
// penalty, seeded RNG determinism, greedy equivalence. No weight downloads;
// fixed logits + fixed seeds only (device tests use tiny views).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

#include "KokkosDNN_Sampler.hpp"

using KokkosDNN::HostRNG;
using KokkosDNN::SampleOptions;
using KokkosDNN::SampleResult;

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

TEST(Sampler, FrequencyPenaltySuppressesRepeats) {
  // Greedy baseline picks 0 (5.0). Seen 3x with frequency_penalty=1.0,
  // id 0 drops to 5-3=2 < 4, so greedy picks 1 instead.
  std::vector<float> lg = {5.0f, 4.0f, 1.0f};
  SampleOptions greedy;
  HostRNG r0(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, greedy, {0, 0, 0}, r0), 0);
  SampleOptions freq;
  freq.frequency_penalty = 1.0f;
  HostRNG r1(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, freq, {0, 0, 0}, r1), 1);
  // Helper math: scaled by occurrence count (OpenAI convention).
  std::vector<float> w = {5.0f, 4.0f};
  KokkosDNN::apply_frequency_presence(w, {0, 0, 0}, 0.5f, 0.0f);
  EXPECT_FLOAT_EQ(w[0], 5.0f - 3 * 0.5f);
  EXPECT_FLOAT_EQ(w[1], 4.0f);
  // Zero penalties are a no-op; out-of-range prior ids ignored, no throw.
  std::vector<float> w2 = {5.0f, 4.0f};
  EXPECT_NO_THROW(
      KokkosDNN::apply_frequency_presence(w2, {-3, 99999}, 1.0f, 1.0f));
  EXPECT_FLOAT_EQ(w2[0], 5.0f);
  EXPECT_FLOAT_EQ(w2[1], 4.0f);
}

TEST(Sampler, PresencePenaltyIsBinary) {
  // presence_penalty subtracts once per DISTINCT seen id, however often
  // it occurs: prior {0} and {0 x5} penalize identically.
  std::vector<float> once = {5.0f, 4.0f};
  KokkosDNN::apply_frequency_presence(once, {0}, 0.0f, 2.0f);
  std::vector<float> five = {5.0f, 4.0f};
  KokkosDNN::apply_frequency_presence(five, {0, 0, 0, 0, 0}, 0.0f, 2.0f);
  EXPECT_FLOAT_EQ(once[0], 3.0f);
  EXPECT_EQ(once, five);
  // Greedy flips from 0 to 1 under the binary penalty.
  SampleOptions pres;
  pres.presence_penalty = 2.0f;
  HostRNG rng(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits({5.0f, 4.0f, 1.0f}, pres, {0}, rng),
            1);
  // Unseen ids are untouched by either penalty.
  std::vector<float> w = {5.0f, 4.0f};
  KokkosDNN::apply_frequency_presence(w, {1}, 1.0f, 1.0f);
  EXPECT_FLOAT_EQ(w[0], 5.0f);
  EXPECT_FLOAT_EQ(w[1], 2.0f);  // 4 - 1*1 (freq) - 1 (presence)
}

TEST(Sampler, LogitBiasForcesChosenId) {
  auto lg = fixture_logits();  // argmax is 2
  SampleOptions greedy;
  greedy.logit_bias = {{5, 10.0f}};
  HostRNG rng(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, greedy, {}, rng), 5);
  // Helper math: additive per id.
  std::vector<float> w = {1.0f, 2.0f};
  KokkosDNN::apply_logit_bias(w, {{1, -5.0f}});
  EXPECT_FLOAT_EQ(w[0], 1.0f);
  EXPECT_FLOAT_EQ(w[1], -3.0f);
  // Empty bias is a no-op; out-of-range entries ignored, no throw.
  std::vector<float> w2 = {1.0f, 2.0f};
  EXPECT_NO_THROW(KokkosDNN::apply_logit_bias(w2, {{-1, 100.0f}, {7, 100.0f}}));
  EXPECT_EQ(w2, (std::vector<float>{1.0f, 2.0f}));
  SampleOptions biased_greedy;
  biased_greedy.logit_bias = {{99, 100.0f}};
  HostRNG rng2(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, biased_greedy, {}, rng2), 2);
  // Bias also steers the sampled path deterministically at the extreme:
  // +1e30 on id 4 overflows every other id out of the distribution.
  SampleOptions sampled;
  sampled.temperature = 1.0f;
  sampled.logit_bias = {{4, 1e30f}};
  HostRNG rng3(42);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, sampled, {}, rng3), 4);
}

TEST(Sampler, LogprobsTopNMatchesGreedy) {
  auto lg = fixture_logits();  // argmax is 2
  SampleOptions opt;
  opt.logprobs = 3;
  HostRNG rng(0);
  SampleResult r =
      KokkosDNN::sample_host_logits_result(lg, opt, {}, rng);
  ASSERT_EQ(r.top_ids.size(), 3u);
  ASSERT_EQ(r.logprobs.size(), 3u);
  // Top entry is the greedy argmax; entries in descending-logprob order.
  EXPECT_EQ(r.top_ids[0], 2);
  EXPECT_EQ(r.id, 2);  // greedy draw
  EXPECT_GT(r.logprobs[0], r.logprobs[1]);
  EXPECT_GT(r.logprobs[1], r.logprobs[2]);
  for (float lp : r.logprobs) EXPECT_LE(lp, 0.0f);  // log of a prob
  // Full-vocab distribution sums to ~1: top-3 mass in (0,1], and the
  // reported top-3 match an independent softmax reference.
  std::vector<double> ex(lg.size());
  double m = *std::max_element(lg.begin(), lg.end());
  double sum = 0.0;
  for (size_t i = 0; i < lg.size(); ++i) {
    ex[i] = std::exp((double)lg[i] - m);
    sum += ex[i];
  }
  double top_mass = 0.0;
  for (size_t k = 0; k < r.top_ids.size(); ++k) {
    int id = r.top_ids[k];
    EXPECT_NEAR(std::exp((double)r.logprobs[k]), ex[(size_t)id] / sum, 1e-5)
        << "top entry " << k;
    top_mass += ex[(size_t)id] / sum;
  }
  EXPECT_GT(top_mass, 0.0);
  EXPECT_LE(top_mass, 1.0);
  // N clamps to vocab size.
  SampleOptions big;
  big.logprobs = 1000;
  HostRNG rng2(0);
  SampleResult rb = KokkosDNN::sample_host_logits_result(lg, big, {}, rng2);
  EXPECT_EQ(rb.top_ids.size(), lg.size());
  // logprobs requested on the device entry too; greedy fast path reports
  // the same top id.
  auto d = make_device_logits(lg);
  HostRNG rng3(0);
  SampleResult rd =
      (KokkosDNN::sample_token_result<Kokkos::DefaultExecutionSpace, float>(
          d, opt, {}, rng3));
  EXPECT_EQ(rd.id, 2);
  EXPECT_EQ(rd.top_ids, r.top_ids);
}

TEST(Sampler, LogprobsOnSampledPath) {
  std::vector<float> lg = {2.0f, 1.0f, 0.5f, 0.0f};
  SampleOptions opt;
  opt.temperature = 1.0f;
  opt.logprobs = 2;
  HostRNG rng(7);
  SampleResult r =
      KokkosDNN::sample_host_logits_result(lg, opt, {}, rng);
  ASSERT_EQ(r.top_ids.size(), 2u);
  ASSERT_EQ(r.logprobs.size(), 2u);
  // Pre-truncation distribution: top-2 of the full vocab are {0, 1}.
  EXPECT_EQ(r.top_ids[0], 0);
  EXPECT_EQ(r.top_ids[1], 1);
  EXPECT_GT(r.logprobs[0], r.logprobs[1]);
  // Sampled id still goes through the (untruncated here) categorical path
  // and matches the legacy entry on the same RNG state.
  HostRNG a(7), b(7);
  int legacy = KokkosDNN::sample_host_logits(lg, opt, {}, a);
  SampleResult rr = KokkosDNN::sample_host_logits_result(lg, opt, {}, b);
  EXPECT_EQ(rr.id, legacy);
  EXPECT_EQ(a.next_u64(), b.next_u64());  // logprobs cost no RNG draws
}

TEST(Sampler, DefaultOptionsUnchanged) {
  // All new fields are neutral by default ...
  SampleOptions o;
  EXPECT_FLOAT_EQ(o.frequency_penalty, 0.0f);
  EXPECT_FLOAT_EQ(o.presence_penalty, 0.0f);
  EXPECT_TRUE(o.logit_bias.empty());
  EXPECT_EQ(o.logprobs, 0);
  // ... so default-constructed options sample identically to the legacy
  // paths: fixed logits + fixed seeds give the same draws through the
  // int-returning and result-returning entries, greedy or sampled, and
  // both entries consume identical RNG state.
  auto lg = fixture_logits();
  for (uint64_t seed : {0u, 1u, 12345u}) {
    for (float temp : {0.0f, 0.6f, 1.0f}) {
      SampleOptions opt;
      opt.temperature = temp;
      opt.top_p = 0.9f;
      HostRNG a(seed), b(seed);
      int via_int = KokkosDNN::sample_host_logits(lg, opt, {}, a);
      SampleResult via_res =
          KokkosDNN::sample_host_logits_result(lg, opt, {}, b);
      EXPECT_EQ(via_res.id, via_int)
          << "seed " << seed << " temp " << temp;
      EXPECT_TRUE(via_res.top_ids.empty());
      EXPECT_TRUE(via_res.logprobs.empty());  // logprobs=0 means off
      EXPECT_EQ(a.next_u64(), b.next_u64())
          << "seed " << seed << " temp " << temp;
    }
  }
  // Greedy default still argmaxes and never touches the RNG.
  SampleOptions greedy;
  HostRNG c(99), d(99);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, greedy, {}, c), 2);
  EXPECT_EQ(c.next_u64(), d.next_u64());
  // Device token entry agrees with the host entry under defaults.
  auto dev = make_device_logits(lg);
  SampleOptions sampled;
  sampled.temperature = 0.8f;
  HostRNG e(2026), f(2026);
  int dev_id =
      (KokkosDNN::sample_token<Kokkos::DefaultExecutionSpace, float>(dev,
                                                                     sampled,
                                                                     {}, e));
  int host_id = KokkosDNN::sample_host_logits(lg, sampled, {}, f);
  EXPECT_EQ(dev_id, host_id);
}
