// Persistent Session tests: tiny random weights + a synthetic tokenizer
// fixture (no downloads). Covers: prefill+step loop termination, stop-id
// handling, max_new/max_seq_len Length caps, usage counts, reseed
// determinism, detokenizer-delta concatenation, and logprob passthrough.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Model.hpp"
#include "KokkosDNN_LFM25Session.hpp"
#include "KokkosDNN_LFM25Tokenizer.hpp"
#include "KokkosDNN_LFM25Weights.hpp"
#include "KokkosDNN_Sampler.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using KokkosDNN::LFM25::Session;
using KokkosDNN::LFM25::SessionOptions;
using KokkosDNN::LFM25::StepOutcome;
using KokkosDNN::Common::FinishReason;

// Minimal GPT-2-style fixture (same shape as Test_LFM25Tokenizer's: single
// lowercase chars + byte-mapped space + two merges; specials mirror LFM2.5
// with <|im_end|> = EOS/stop). Decodes are pure ASCII, so per-id decode
// pieces concatenate exactly to the whole-list decode.
std::string session_fixture_json() {
  return R"({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "d": 2, "e": 3, "h": 4, "i": 5, "l": 6, "m": 7,
    "n": 8, "o": 9, "r": 10, "s": 11, "t": 12, "u": 13, "w": 14, "y": 15,
    "H": 16,
    "1": 30, "2": 31,
    "hi": 20, "th": 21,
    "\u0120": 17, "\u010a": 18
  }, "merges": ["h i", "t h"]},
  "added_tokens": [
    {"id": 100, "content": "<|startoftext|>", "special": true},
    {"id": 101, "content": "<|im_start|>", "special": true},
    {"id": 102, "content": "<|im_end|>", "special": true},
    {"id": 103, "content": "<think>", "special": true},
    {"id": 104, "content": "</think>", "special": true},
    {"id": 105, "content": "<|pad|>", "special": true}
  ]
})";
}

std::string write_tmp(const std::string& body) {
  std::string p = "/tmp/kokkosdnn_lfm25_session_fixture_" +
                  std::to_string(::getpid()) + ".json";
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("cannot write " + p);
  f << body;
  f.close();
  return p;
}

struct Env {
  KokkosDNN::LFM25::Tokenizer tok;
  KokkosDNN::LFM25::Config cfg;
};

Env make_env() {
  Env e;
  e.tok = KokkosDNN::LFM25::Tokenizer(write_tmp(session_fixture_json()));
  e.cfg = KokkosDNN::LFM25::Config::tiny_test();
  e.cfg.vocab_size = e.tok.vocab_size();
  e.cfg.eos = e.tok.eos();
  e.cfg.bos = e.tok.bos();
  return e;
}

template <class Scalar>
Session<Exec, Scalar> make_session(const Env& e,
                                   SessionOptions opts = SessionOptions{}) {
  KokkosDNN::LFM25::Weights<Exec, Scalar> w;
  KokkosDNN::LFM25::random_init(e.cfg, w, 1234);
  return Session<Exec, Scalar>(e.cfg, w, e.tok, std::move(opts));
}

// Stop-id that can never be sampled: disables id-stop termination while
// keeping the configured (unreachable-in-fixture) stop strings.
SessionOptions no_stop_opts(int max_new) {
  SessionOptions o;
  o.max_new = max_new;
  o.stop_ids = {999999};
  return o;
}

template <class Scalar>
void check_terminate_and_counts() {
  using namespace KokkosDNN::LFM25;
  Env e = make_env();
  auto s = make_session<Scalar>(e, no_stop_opts(3));
  s.prefill({1, 2, 3});
  EXPECT_EQ(s.usage().prompt_tokens, 3u);
  EXPECT_EQ(s.usage().completion_tokens, 0u);
  EXPECT_EQ(s.generated(), 0u);
  EXPECT_FALSE(s.finished());

  int n = 0;
  StepOutcome last;
  while (!s.finished() && n < 10) {
    last = s.step();
    ++n;
  }
  EXPECT_TRUE(s.finished());
  EXPECT_EQ(s.finish(), FinishReason::Length);  // max_new=3 exhausted
  EXPECT_TRUE(last.finished);
  EXPECT_EQ(last.finish, FinishReason::Length);
  EXPECT_EQ(s.generated(), 3u);
  EXPECT_EQ(s.usage().completion_tokens, 3u);
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

TEST(LFM25Session, TerminateAndCountsFloat) { check_terminate_and_counts<float>(); }

TEST(LFM25Session, TerminateAndCountsBHalf) {
  check_terminate_and_counts<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_stop_id() {
  using namespace KokkosDNN::LFM25;
  Env e = make_env();
  // Learn the first greedy id, then stop on exactly that id.
  auto probe = make_session<Scalar>(e, no_stop_opts(8));
  probe.prefill({1, 2, 3}, 7, true);
  StepOutcome first = probe.step();

  SessionOptions o;
  o.stop_ids = {first.id};
  auto s = make_session<Scalar>(e, o);
  s.prefill({1, 2, 3}, 7, true);
  StepOutcome st = s.step();
  EXPECT_EQ(st.id, first.id);
  EXPECT_TRUE(st.finished);
  EXPECT_TRUE(st.stop_hit);
  EXPECT_EQ(st.finish, FinishReason::Stop);
  EXPECT_TRUE(s.finished());
  EXPECT_EQ(s.finish(), FinishReason::Stop);
  // The stop token counts as a completion token (mirrors generate()).
  EXPECT_EQ(s.generated(), 1u);
}

TEST(LFM25Session, StopIdFloat) { check_stop_id<float>(); }

TEST(LFM25Session, StopIdBHalf) {
  check_stop_id<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_reseed_determinism() {
  using namespace KokkosDNN::LFM25;
  Env e = make_env();
  SessionOptions base = no_stop_opts(8);
  base.sampling.temperature = 0.6f;  // non-greedy: exercises RNG reseeding
  base.sampling.top_p = 0.9f;
  auto run = [&](uint64_t seed) {
    auto s = make_session<Scalar>(e, base);
    s.prefill({1, 2, 3, 4}, seed, true);
    std::vector<int> ids;
    for (int i = 0; i < 4; ++i) {
      StepOutcome st = s.step();
      EXPECT_FALSE(st.finished);  // max_new=8, stops disabled
      ids.push_back(st.id);
    }
    return ids;
  };
  EXPECT_EQ(run(11), run(11));
  // prefill(ids) without an explicit seed follows the constructor seed.
  auto s = make_session<Scalar>(e, base);
  s.prefill({1, 2, 3, 4});
  std::vector<int> ids;
  for (int i = 0; i < 4; ++i) ids.push_back(s.step().id);
  EXPECT_EQ(ids, run(0));  // default SessionOptions::seed is 0
}

TEST(LFM25Session, ReseedDeterminismFloat) {
  check_reseed_determinism<float>();
}

TEST(LFM25Session, ReseedDeterminismBHalf) {
  check_reseed_determinism<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_detok_concat() {
  using namespace KokkosDNN::LFM25;
  Env e = make_env();
  auto s = make_session<Scalar>(e, no_stop_opts(4));
  s.prefill({1, 2, 3});
  std::string stream;
  std::vector<int> gen;
  for (int i = 0; i < 4; ++i) {
    StepOutcome st = s.step();
    stream += st.delta;
    gen.push_back(st.id);
  }
  stream += s.flush();
  // Per-id decode with the same invalid-id tolerance as Session::step:
  // the synthetic fixture vocab is sparse, so random tiny-model samples
  // can land on ids with no vocab entry.
  std::string expect;
  for (int id : gen) {
    try {
      expect += e.tok.decode({id});
    } catch (const std::exception&) {
    }
  }
  EXPECT_EQ(stream, expect);
}

TEST(LFM25Session, DetokConcatFloat) { check_detok_concat<float>(); }

TEST(LFM25Session, DetokConcatBHalf) {
  check_detok_concat<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_max_seq_len_cap() {
  using namespace KokkosDNN::LFM25;
  Env e = make_env();
  e.cfg.max_seq_len = 5;  // prompt of 3 + 2 decode forwards, then capped
  KokkosDNN::LFM25::Weights<Exec, Scalar> w;
  KokkosDNN::LFM25::random_init(e.cfg, w, 1234);
  Session<Exec, Scalar> s(e.cfg, w, e.tok, no_stop_opts(8));
  s.prefill({1, 2, 3});
  int n = 0;
  while (!s.finished() && n < 20) {
    (void)s.step();
    ++n;
  }
  EXPECT_TRUE(s.finished());
  EXPECT_EQ(s.finish(), FinishReason::Length);
  // 1 pending-logit sample + 2 forwards (seq 3->4->5), then the cap trips
  // without forwarding past it.
  EXPECT_EQ(s.generated(), 3u);
}

TEST(LFM25Session, MaxSeqLenCapFloat) { check_max_seq_len_cap<float>(); }

TEST(LFM25Session, MaxSeqLenCapBHalf) {
  check_max_seq_len_cap<Kokkos::Experimental::bhalf_t>();
}

TEST(LFM25Session, LogprobsPassthroughFloat) {
  using namespace KokkosDNN::LFM25;
  Env e = make_env();
  SessionOptions o = no_stop_opts(2);
  o.sampling.logprobs = 2;
  auto s = make_session<float>(e, o);
  s.prefill({1, 2, 3});
  StepOutcome st = s.step();
  ASSERT_EQ(st.top_ids.size(), 2u);
  ASSERT_EQ(st.logprobs.size(), 2u);
  for (float lp : st.logprobs) EXPECT_LE(lp, 0.0f);  // log-probabilities
}

TEST(LFM25Session, LogprobsEmptyWhenOffFloat) {
  using namespace KokkosDNN::LFM25;
  Env e = make_env();
  auto s = make_session<float>(e, no_stop_opts(2));
  s.prefill({1, 2, 3});
  StepOutcome st = s.step();
  EXPECT_TRUE(st.logprobs.empty());
  EXPECT_TRUE(st.top_ids.empty());
}

}  // namespace
