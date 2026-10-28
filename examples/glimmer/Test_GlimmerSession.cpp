// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Glimmer Session tests (no weight downloads): prefill+step termination,
// stop-id policy, reseed determinism, and prefill_cached equivalence.
// Random tiny model weights; the default-constructed (unloaded) tokenizer
// contributes empty pieces, so these tests exercise ids/usage/prefix-cache
// behavior rather than decoded text.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <memory>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_GlimmerModel.hpp"
#include "KokkosDNN_GlimmerSession.hpp"
#include "KokkosDNN_GlimmerTokenizer.hpp"
#include "KokkosDNN_GlimmerWeights.hpp"

using KokkosDNN::Glimmer::CachedPrefillOutcome;
using KokkosDNN::Glimmer::Config;
using KokkosDNN::Glimmer::ISession;
using KokkosDNN::Glimmer::Session;
using KokkosDNN::Glimmer::SessionOptions;
using KokkosDNN::Glimmer::StepOutcome;
using KokkosDNN::Glimmer::Tokenizer;
using KokkosDNN::Common::FinishReason;

using Exec = Kokkos::DefaultExecutionSpace;

namespace {

Config tiny_cfg() {
  Config c = Config::tiny_test();
  c.max_seq_len = 64;
  return c;
}

// Stop-id that can never be sampled: disables id-stop termination.
SessionOptions no_stop_opts(int max_new) {
  SessionOptions o;
  o.max_new = max_new;
  o.stop_ids = {999999};
  return o;
}

template <class Scalar>
Session<Exec, Scalar> make_session(const Config& cfg,
                                   SessionOptions opts = SessionOptions{}) {
  KokkosDNN::Glimmer::Weights<Exec, Scalar> w;
  KokkosDNN::Glimmer::random_init(cfg, w, 1234);
  return Session<Exec, Scalar>(cfg, w, Tokenizer{}, std::move(opts));
}

template <class Scalar>
std::vector<int> run_steps(Session<Exec, Scalar>& s, int n) {
  std::vector<int> ids;
  for (int i = 0; i < n && !s.finished(); ++i) ids.push_back(s.step().id);
  return ids;
}

}  // namespace

TEST(GlimmerSession, TerminateAndCountsFloat) {
  Config cfg = tiny_cfg();
  auto s = make_session<float>(cfg, no_stop_opts(3));
  s.prefill({1, 2, 3});
  EXPECT_EQ(s.usage().prompt_tokens, 3u);
  EXPECT_EQ(s.usage().completion_tokens, 0u);
  EXPECT_EQ(s.generated(), 0u);
  EXPECT_FALSE(s.finished());

  StepOutcome last;
  int n = 0;
  while (!s.finished() && n < 10) {
    last = s.step();
    ++n;
  }
  EXPECT_TRUE(s.finished());
  EXPECT_EQ(s.finish(), FinishReason::Length);  // max_new=3 exhausted
  EXPECT_TRUE(last.finished);
  EXPECT_EQ(last.finish, FinishReason::Length);
  EXPECT_EQ(s.generated(), 3u);
  EXPECT_EQ(s.usage().total(), 6u);

  // Sticky past the end: no more forwarding, same terminal outcome.
  StepOutcome again = s.step();
  EXPECT_TRUE(again.finished);
  EXPECT_EQ(again.finish, FinishReason::Length);
  EXPECT_EQ(s.generated(), 3u);

  // Multi-turn: prefill() again starts a new turn over the same weights.
  s.prefill({4, 5});
  EXPECT_FALSE(s.finished());
  EXPECT_EQ(s.usage().prompt_tokens, 2u);
  EXPECT_EQ(s.generated(), 0u);
  StepOutcome st = s.step();
  EXPECT_FALSE(st.finished);  // only 1 of max_new=3 so far
  EXPECT_EQ(s.generated(), 1u);
}

TEST(GlimmerSession, TerminateAndCountsBHalf) {
  using Scalar = Kokkos::Experimental::bhalf_t;
  Config cfg = tiny_cfg();
  auto s = make_session<Scalar>(cfg, no_stop_opts(2));
  s.prefill({1, 2, 3});
  auto ids = run_steps(s, 10);
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_TRUE(s.finished());
  EXPECT_EQ(s.finish(), FinishReason::Length);
}

TEST(GlimmerSession, StopId) {
  Config cfg = tiny_cfg();
  // Learn the first greedy id, then stop on exactly that id.
  auto probe = make_session<float>(cfg, no_stop_opts(8));
  probe.prefill({1, 2, 3}, 7, true);
  StepOutcome first = probe.step();

  SessionOptions o;
  o.stop_ids = {first.id};
  auto s = make_session<float>(cfg, o);
  s.prefill({1, 2, 3}, 7, true);
  StepOutcome st = s.step();
  EXPECT_EQ(st.id, first.id);
  EXPECT_TRUE(st.finished);
  EXPECT_TRUE(st.stop_hit);
  EXPECT_EQ(st.finish, FinishReason::Stop);
  EXPECT_TRUE(s.finished());
  // The stop token counts as a completion token (mirrors generate()).
  EXPECT_EQ(s.generated(), 1u);
}

TEST(GlimmerSession, ReseedDeterminism) {
  Config cfg = tiny_cfg();
  SessionOptions base = no_stop_opts(8);
  base.sampling.temperature = 0.6f;  // non-greedy: exercises RNG reseeding
  base.sampling.top_p = 0.9f;
  auto run = [&](uint64_t seed) {
    auto s = make_session<float>(cfg, base);
    s.prefill({1, 2, 3, 4}, seed, true);
    return run_steps(s, 4);
  };
  EXPECT_EQ(run(11), run(11));
  // Reseeding across turns on one session reproduces the same stream.
  auto s = make_session<float>(cfg, base);
  s.prefill({1, 2, 3, 4}, 11, true);
  const std::vector<int> a = run_steps(s, 4);
  s.prefill({1, 2, 3, 4}, 11, true);
  EXPECT_EQ(run_steps(s, 4), a);
}

TEST(GlimmerSession, CachedPrefillEquivalence) {
  Config cfg = tiny_cfg();
  const std::vector<int> content{1, 2, 3, 4, 5, 6};
  const std::vector<int> full{1, 2, 3, 4, 5, 6, 7, 8};

  // Reference: uncached prefill + greedy steps.
  auto ref = make_session<float>(cfg, no_stop_opts(4));
  ref.prefill(full, 5, true);
  const std::vector<int> ref_ids = run_steps(ref, 4);

  // Cached: first call misses and snapshots, second call hits.
  auto s = make_session<float>(cfg, no_stop_opts(4));
  CachedPrefillOutcome m = s.prefill_cached(full, content, 5, true);
  EXPECT_FALSE(m.hit);
  EXPECT_EQ(m.reused, 0u);
  EXPECT_EQ(s.prefix_cache_size(), 1u);
  CachedPrefillOutcome h = s.prefill_cached(full, content, 5, true);
  EXPECT_TRUE(h.hit);
  EXPECT_EQ(h.reused, content.size());
  EXPECT_EQ(run_steps(s, 4), ref_ids);

  // Non-prefix content falls back to an uncached prefill (still exact).
  auto t = make_session<float>(cfg, no_stop_opts(4));
  CachedPrefillOutcome f =
      t.prefill_cached(full, std::vector<int>{9, 9, 9}, 5, true);
  EXPECT_FALSE(f.hit);
  EXPECT_EQ(run_steps(t, 4), ref_ids);
}

TEST(GlimmerSession, PolymorphicFrontEnd) {
  // The server holds sessions behind ISession; exercise that path.
  Config cfg = tiny_cfg();
  KokkosDNN::Glimmer::Weights<Exec, float> w;
  KokkosDNN::Glimmer::random_init(cfg, w, 1234);
  std::unique_ptr<ISession> s =
      std::make_unique<Session<Exec, float>>(cfg, w, Tokenizer{},
                                             no_stop_opts(2));
  s->prefill({1, 2});
  EXPECT_FALSE(s->finished());
  StepOutcome a = s->step();
  EXPECT_FALSE(a.finished);
  EXPECT_EQ(s->generated(), 1u);
  StepOutcome b = s->step();
  EXPECT_TRUE(b.finished);
  EXPECT_EQ(s->finish(), FinishReason::Length);
  EXPECT_EQ(s->flush(), "");
  EXPECT_EQ(s->usage().total(), 4u);
}
