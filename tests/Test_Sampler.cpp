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

// Deterministic pseudo-random logits (no <random> dependence): a reproducible
// spread with a clear argmax, used by the device-vs-host equivalence tests.
std::vector<float> synthetic_logits(size_t n, uint64_t seed) {
  std::vector<float> v(n);
  uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
  for (size_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    const float unit = static_cast<float>((s >> 40) & 0xFFFFFF) / 16777216.0f;
    v[i] = 4.0f * unit - 2.0f;
  }
  v[n / 3] = 6.0f;  // unambiguous argmax
  return v;
}

// Adjusted, temperature-scaled logits of the *host* pipeline, re-derived
// from the documented semantics so the device path is checked against the
// contract rather than against a second device implementation.
std::vector<float> reference_work(const std::vector<float>& logits,
                                  const SampleOptions& opt,
                                  const std::vector<int>& prior) {
  std::vector<float> work = logits;
  KokkosDNN::apply_repetition_penalty(work, prior, opt.repetition_penalty);
  KokkosDNN::apply_logit_bias(work, opt.logit_bias);
  KokkosDNN::apply_frequency_presence(work, prior, opt.frequency_penalty,
                                      opt.presence_penalty);
  if (opt.temperature > 0.0f && opt.temperature != 1.0f)
    for (auto& v : work) v /= opt.temperature;
  return work;
}

// Kept set of the host pipeline (sample_host_logits_result), in the host's
// descending-logit order.
std::vector<int> reference_kept(const std::vector<float>& work,
                                const SampleOptions& opt) {
  const size_t V = work.size();
  std::vector<int> order(V);
  for (size_t i = 0; i < V; ++i) order[i] = static_cast<int>(i);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return work[(size_t)a] > work[(size_t)b]; });
  if (opt.top_k > 1 && (size_t)opt.top_k < V) order.resize((size_t)opt.top_k);
  if (opt.top_p < 1.0f) {
    std::vector<float> vals(order.size());
    for (size_t i = 0; i < order.size(); ++i) vals[i] = work[(size_t)order[i]];
    const std::vector<float> probs = KokkosDNN::Impl::softmax(vals);
    double cum = 0.0;
    size_t keep = 0;
    for (size_t i = 0; i < order.size(); ++i) {
      cum += probs[i];
      keep = i + 1;
      if (cum >= (double)opt.top_p) break;
    }
    order.resize(keep);
  }
  if (opt.min_p > 0.0f && order.size() > 1) {
    std::vector<float> probs(order.size());
    const float mx = work[(size_t)order[0]];
    double sum = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
      probs[i] = std::exp(work[(size_t)order[i]] - mx);
      sum += probs[i];
    }
    const float thresh = opt.min_p * static_cast<float>(probs[0] / sum);
    size_t keep = 0;
    for (size_t i = 0; i < order.size(); ++i) {
      if (static_cast<float>(probs[i] / sum) >= thresh) keep = i + 1;
      else break;
    }
    order.resize(keep > 0 ? keep : 1);
  }
  return order;
}

// Exact probabilities of the reference kept set (renormalized over it).
std::vector<double> reference_probs(const std::vector<float>& work,
                                    const std::vector<int>& kept) {
  std::vector<double> p(kept.size());
  double sum = 0.0;
  for (size_t i = 0; i < kept.size(); ++i) {
    p[i] = std::exp((double)work[(size_t)kept[i]]);
    sum += p[i];
  }
  for (auto& v : p) v /= sum;
  return p;
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

TEST(Sampler, UntruncatedPathRespectsModifiers) {
  // No top-k / top-p / min-p filtering takes the index-order short-circuit;
  // the logit adjustments applied before it must still decide the draw.
  auto lg = fixture_logits();
  SampleOptions opt;
  opt.temperature = 0.7f;  // top_k 0, top_p 1, min_p 0 -> untruncated
  opt.logit_bias = {{5, 1e30f}};
  HostRNG rng(11);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, opt, {}, rng), 5);
  // Repetition penalty steers the same path away from the argmax when the
  // penalty is large enough to flip it (id 2 is the argmax of the fixture).
  SampleOptions rep;
  rep.temperature = 0.7f;
  rep.repetition_penalty = 100.0f;
  HostRNG rng2(3);
  EXPECT_NE(KokkosDNN::sample_host_logits(lg, rep, {2}, rng2), 2);
  // One uniform draw per token, exactly like the filtered path (the RNG
  // stream stays aligned whichever branch runs).
  HostRNG a(9), b(9);
  (void)KokkosDNN::sample_host_logits(lg, opt, {}, a);
  SampleOptions filtered = opt;
  filtered.top_p = 0.9f;
  (void)KokkosDNN::sample_host_logits(lg, filtered, {}, b);
  EXPECT_EQ(a.next_u64(), b.next_u64());
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

// ---------------------------------------------------------------------------
// Device sampling pipeline (KokkosDNN_Sampler.hpp Impl::DeviceSampler)
//
// The device path must reproduce the host pipeline's *distribution*: same
// penalties/bias, same temperature scaling, same truncation rules and the same
// logprob report. It resolves truncation on a 1/128 log-probability grid (see
// the header) instead of sorting, so the checks below are distribution-level
// (support + frequencies + exact logprob values), not bit-level id equality.
// ---------------------------------------------------------------------------

namespace {

using DevSampler = KokkosDNN::Impl::DeviceSampler<Kokkos::DefaultExecutionSpace>;

struct DeviceCase {
  const char* name;
  SampleOptions opt;
  std::vector<int> prior;
  bool check_freq;  // reference probs are exact only without penalties
};

std::vector<DeviceCase> device_cases() {
  std::vector<DeviceCase> cases;
  {
    DeviceCase c;
    c.name = "temperature only";
    c.opt.temperature = 0.8f;
    c.check_freq = true;
    cases.push_back(c);
  }
  {
    DeviceCase c;
    c.name = "top_k 5";
    c.opt.temperature = 1.0f;
    c.opt.top_k = 5;
    c.check_freq = true;
    cases.push_back(c);
  }
  {
    DeviceCase c;
    c.name = "top_p 0.9";
    c.opt.temperature = 1.0f;
    c.opt.top_p = 0.9f;
    c.check_freq = true;
    cases.push_back(c);
  }
  {
    DeviceCase c;
    c.name = "top_k 12 + top_p 0.85 + min_p 0.02";
    c.opt.temperature = 0.7f;
    c.opt.top_k = 12;
    c.opt.top_p = 0.85f;
    c.opt.min_p = 0.02f;
    c.check_freq = true;
    cases.push_back(c);
  }
  {
    DeviceCase c;
    c.name = "repetition + frequency + presence + logit_bias";
    c.opt.temperature = 0.9f;
    c.opt.repetition_penalty = 1.4f;
    c.opt.frequency_penalty = 0.5f;
    c.opt.presence_penalty = 0.3f;
    c.opt.logit_bias = {{7, 1.5f}, {11, -2.0f}};
    c.prior = {7, 7, 11, 3, 3, 3, 200};
    c.check_freq = false;
    cases.push_back(c);
  }
  return cases;
}

}  // namespace

TEST(SamplerDevice, TruncationMatchesHostKeptSet) {
  const size_t V = 257;
  const std::vector<float> logits = synthetic_logits(V, 7);
  auto dev = make_device_logits(logits);
  const int draws = 3000;
  for (const DeviceCase& c : device_cases()) {
    const std::vector<float> work = reference_work(logits, c.opt, c.prior);
    const std::vector<int> kept = reference_kept(work, c.opt);
    ASSERT_FALSE(kept.empty()) << c.name;
    std::vector<int> pos(V, -1);
    for (size_t i = 0; i < kept.size(); ++i) pos[(size_t)kept[i]] = (int)i;
    std::vector<int> counts(kept.size(), 0);
    DevSampler sampler(V);
    HostRNG rng(20261023);
    for (int i = 0; i < draws; ++i) {
      const int id = sampler.sample(dev, c.opt, c.prior, rng).id;
      ASSERT_GE(id, 0) << c.name;
      ASSERT_LT(id, (int)V) << c.name;
      ASSERT_GE(pos[(size_t)id], 0)
          << c.name << ": drew id " << id << " outside the host kept set";
      counts[(size_t)pos[(size_t)id]]++;
    }
    if (!c.check_freq) continue;
    // Empirical frequencies must follow the reference kept-set distribution.
    const std::vector<double> p = reference_probs(work, kept);
    for (size_t i = 0; i < kept.size(); ++i) {
      if (p[i] < 0.02) continue;  // too few expected counts to be meaningful
      const double freq = (double)counts[i] / (double)draws;
      EXPECT_NEAR(freq, p[i], 0.05) << c.name << " id " << kept[i];
    }
  }
}

TEST(SamplerDevice, GreedyHonoursPenaltiesAndBias) {
  // Host reference: same decisions, deterministic (no RNG draw).
  std::vector<float> lg = {5.0f, 4.0f, 1.0f};
  auto dev = make_device_logits(lg);
  SampleOptions greedy;
  HostRNG h0(0);
  EXPECT_EQ(KokkosDNN::sample_host_logits(lg, greedy, {}, h0), 0);

  SampleOptions pen;
  pen.repetition_penalty = 2.0f;
  HostRNG h1(0);
  const int host_pen = KokkosDNN::sample_host_logits(lg, pen, {0}, h1);
  HostRNG d1(0);
  EXPECT_EQ(DevSampler(3).sample(dev, pen, {0}, d1).id, host_pen);
  EXPECT_EQ(host_pen, 1);
  // Greedy never consumes the RNG (device and host agree on the stream).
  EXPECT_EQ(d1.next_u64(), h1.next_u64());

  SampleOptions freq;
  freq.frequency_penalty = 1.0f;
  HostRNG h2(0);
  const int host_freq = KokkosDNN::sample_host_logits(lg, freq, {0, 0, 0}, h2);
  HostRNG d2(0);
  EXPECT_EQ(DevSampler(3).sample(dev, freq, {0, 0, 0}, d2).id, host_freq);
  EXPECT_EQ(host_freq, 1);

  SampleOptions pres;
  pres.presence_penalty = 2.0f;
  HostRNG h3(0);
  const int host_pres = KokkosDNN::sample_host_logits(lg, pres, {0}, h3);
  HostRNG d3(0);
  EXPECT_EQ(DevSampler(3).sample(dev, pres, {0}, d3).id, host_pres);
  EXPECT_EQ(host_pres, 1);

  SampleOptions biased;
  biased.logit_bias = {{2, 10.0f}};
  HostRNG h4(0);
  const int host_bias = KokkosDNN::sample_host_logits(lg, biased, {}, h4);
  HostRNG d4(0);
  EXPECT_EQ(DevSampler(3).sample(dev, biased, {}, d4).id, host_bias);
  EXPECT_EQ(host_bias, 2);

  // A sampled draw with an extreme bias must land on the biased id and still
  // consume exactly one uniform double.
  SampleOptions steer;
  steer.temperature = 1.0f;
  steer.logit_bias = {{0, 1e30f}};
  const std::vector<float> wide = synthetic_logits(64, 3);
  auto wide_dev = make_device_logits(wide);
  HostRNG d5(11), ref5(11);
  DevSampler s5(64);
  EXPECT_EQ(s5.sample(wide_dev, steer, {}, d5).id, 0);
  (void)ref5.next_uniform();
  EXPECT_EQ(d5.next_u64(), ref5.next_u64());
}

TEST(SamplerDevice, LogprobsMatchHostReference) {
  const size_t V = 129;
  const std::vector<float> logits = synthetic_logits(V, 21);
  auto dev = make_device_logits(logits);
  // Greedy + logprobs: id is the argmax, logprobs come from the full
  // distribution.
  SampleOptions opt;
  opt.logprobs = 5;
  HostRNG rng(0);
  DevSampler sampler(V);
  const SampleResult r = sampler.sample(dev, opt, {}, rng);
  EXPECT_EQ(r.id, (int)(std::max_element(logits.begin(), logits.end()) -
                        logits.begin()));
  const std::vector<float> lp = KokkosDNN::Impl::log_softmax(logits);
  std::vector<int> expect_ids;
  std::vector<float> expect_lp;
  KokkosDNN::Impl::top_logprobs(lp, 5, expect_ids, expect_lp);
  ASSERT_EQ(r.top_ids.size(), expect_ids.size());
  for (size_t i = 0; i < expect_ids.size(); ++i) {
    EXPECT_EQ(r.top_ids[i], expect_ids[i]) << "rank " << i;
    EXPECT_NEAR(r.logprobs[i], expect_lp[i], 1e-4f) << "rank " << i;
  }

  // Sampled + logprobs: same report, and the id comes from the sampled path.
  SampleOptions sopt;
  sopt.temperature = 0.8f;
  sopt.top_p = 0.9f;
  sopt.logprobs = 4;
  const std::vector<float> scaled = [&] {
    std::vector<float> w = logits;
    for (auto& v : w) v /= 0.8f;
    return w;
  }();
  HostRNG rng2(5);
  DevSampler sampler2(V);
  const SampleResult r2 = sampler2.sample(dev, sopt, {}, rng2);
  const std::vector<float> lp2 = KokkosDNN::Impl::log_softmax(scaled);
  KokkosDNN::Impl::top_logprobs(lp2, 4, expect_ids, expect_lp);
  ASSERT_EQ(r2.top_ids.size(), expect_ids.size());
  for (size_t i = 0; i < expect_ids.size(); ++i) {
    EXPECT_EQ(r2.top_ids[i], expect_ids[i]) << "rank " << i;
    EXPECT_NEAR(r2.logprobs[i], expect_lp[i], 1e-4f) << "rank " << i;
  }
  // N clamps to the vocabulary on the device path too.
  SampleOptions big;
  big.temperature = 1.0f;
  big.logprobs = 1000;
  HostRNG rng3(9);
  const SampleResult r3 = DevSampler(32).sample(dev, big, {}, rng3);
  EXPECT_EQ(r3.top_ids.size(), V);
  EXPECT_EQ(r3.logprobs.size(), V);
}

TEST(SamplerDevice, StatisticalSanity) {
  // Untruncated temperature-1 sampling over a small vocabulary: empirical
  // counts must track the exact softmax (loose tolerance, fixed seed).
  const size_t V = 64;
  std::vector<float> logits(V);
  for (size_t i = 0; i < V; ++i)
    logits[i] = 0.05f * static_cast<float>(i) - 1.0f;
  logits[40] = 3.5f;  // dominant but not exclusive
  auto dev = make_device_logits(logits);
  SampleOptions opt;
  opt.temperature = 1.0f;
  std::vector<double> p(V);
  double sum = 0.0;
  for (size_t i = 0; i < V; ++i) {
    p[i] = std::exp((double)logits[i] - 3.5);
    sum += p[i];
  }
  for (auto& v : p) v /= sum;
  DevSampler sampler(V);
  HostRNG rng(4242);
  const int draws = 20000;
  std::vector<int> counts(V, 0);
  for (int i = 0; i < draws; ++i) {
    const int id = sampler.sample(dev, opt, {}, rng).id;
    ASSERT_GE(id, 0);
    ASSERT_LT(id, (int)V);
    counts[(size_t)id]++;
  }
  for (size_t i = 0; i < V; ++i) {
    if (p[i] < 0.01) continue;
    const double freq = (double)counts[i] / (double)draws;
    EXPECT_NEAR(freq, p[i], 0.03) << "id " << i;
  }
  EXPECT_GT(counts[40], 0);
}

TEST(SamplerDevice, ScratchReuseIsDeterministic) {
  // The same seed must replay the same trajectory whether the sampler is
  // fresh or already warm (buffers and host scratch reused across tokens,
  // including a vocabulary-size change in between).
  const size_t V = 97;
  const std::vector<float> logits = synthetic_logits(V, 99);
  auto dev = make_device_logits(logits);
  SampleOptions opt;
  opt.temperature = 0.75f;
  opt.top_k = 8;
  opt.repetition_penalty = 1.2f;
  std::vector<int> prior;
  DevSampler warm(V);
  HostRNG warm_rng(7);
  HostRNG fresh_rng(7);
  for (int i = 0; i < 24; ++i) {
    const int a = warm.sample(dev, opt, prior, warm_rng).id;
    DevSampler fresh(V);
    const int b = fresh.sample(dev, opt, prior, fresh_rng).id;
    EXPECT_EQ(a, b) << "token " << i;
    prior.push_back(a);
  }
  // A different vocabulary size in the middle of a run reallocates scratch
  // without changing behaviour.
  const std::vector<float> small = synthetic_logits(13, 5);
  auto small_dev = make_device_logits(small);
  HostRNG r1(3), r2(3);
  const int s1 = warm.sample(small_dev, opt, {}, r1).id;
  const int s2 = DevSampler(13).sample(small_dev, opt, {}, r2).id;
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(r1.next_u64(), r2.next_u64());
}

TEST(SamplerDevice, GreedyFastPathIsUntouched) {
  // Neutral greedy: argmax, no RNG draw, no scratch needed (the entry point
  // must not allocate a sampler for this case).
  auto lg = fixture_logits();
  auto d = make_device_logits(lg);
  SampleOptions opt;  // temperature 0
  HostRNG a(5), b(5);
  EXPECT_EQ((KokkosDNN::sample_token_result<Kokkos::DefaultExecutionSpace,
                                            float>(d, opt, {}, a))
                .id,
            2);
  EXPECT_EQ(a.next_u64(), b.next_u64());
  // top_k == 1 forces greedy at any temperature.
  SampleOptions k1;
  k1.temperature = 1.7f;
  k1.top_k = 1;
  HostRNG c(5), e(5);
  EXPECT_EQ(
      (KokkosDNN::sample_token_result<Kokkos::DefaultExecutionSpace, float>(
          d, k1, {}, c))
          .id,
      2);
  EXPECT_EQ(c.next_u64(), e.next_u64());
  // bf16 logits go through the same path.
  auto dbf = make_device_logits<Kokkos::Experimental::bhalf_t>(lg);
  SampleOptions sopt;
  sopt.temperature = 0.6f;
  sopt.top_p = 0.9f;
  HostRNG f(123);
  const int id =
      (KokkosDNN::sample_token_result<Kokkos::DefaultExecutionSpace,
                                      Kokkos::Experimental::bhalf_t>(dbf, sopt, {}, f)).id;
  EXPECT_EQ(id, 2);  // the fixture's argmax dominates a 0.9 nucleus
}

TEST(SamplerDevice, NucleusCoversHostKeptSet) {
  // A long, gently decaying distribution puts the top-p crossing hundreds of
  // ids deep, where a nucleus that is off by a few percent would silently
  // drop a whole tail of ids. Draw many tokens: nothing outside the host kept
  // set may appear *and* every id inside it must be reachable.
  const size_t V = 1000;
  std::vector<float> logits(V);
  for (size_t i = 0; i < V; ++i) logits[i] = -0.01f * static_cast<float>(i);
  logits[17] = 2.0f;
  SampleOptions opt;
  opt.temperature = 1.0f;
  opt.top_p = 0.95f;
  const std::vector<float> work = reference_work(logits, opt, {});
  const std::vector<int> kept = reference_kept(work, opt);
  ASSERT_GT(kept.size(), 200u) << "fixture should have a deep nucleus";
  ASSERT_LT(kept.size(), V);
  std::vector<char> in(V, 0);
  for (int id : kept) in[(size_t)id] = 1;

  auto dev = make_device_logits(logits);
  DevSampler sampler(V);
  HostRNG rng(31337);
  const int draws = 20000;
  std::vector<int> counts(V, 0);
  for (int i = 0; i < draws; ++i) {
    const int id = sampler.sample(dev, opt, {}, rng).id;
    ASSERT_GE(id, 0);
    ASSERT_LT(id, (int)V);
    ASSERT_EQ(in[(size_t)id], 1)
        << "drew id " << id << " outside the host nucleus (rank of "
        << kept.size() << " ids)";
    counts[(size_t)id]++;
  }
  // Every kept id must be reachable: the tail of the nucleus is drawn with
  // p ~ 1e-3, i.e. ~20 expected hits per id over 20k draws.
  int missed = 0;
  for (int id : kept)
    if (counts[(size_t)id] == 0) ++missed;
  EXPECT_EQ(missed, 0) << "nucleus tail is narrower than the host's";
}
