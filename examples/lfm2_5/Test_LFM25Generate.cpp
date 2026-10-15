// End-to-end tiny LFM2.5: random weights, forward + greedy decode.
// Exercises both bfloat16 (default) and fp32 scalar paths. No weight
// downloads; tiny_test() dims keep runtime to seconds.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Model.hpp"
#include "KokkosDNN_LFM25Weights.hpp"
#include "KokkosDNN_Sampler.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;

template <class Scalar>
Kokkos::View<int*, Exec> make_tok_view(const std::vector<int>& ids) {
  Kokkos::View<int*, Exec> toks("toks", ids.size());
  auto h = Kokkos::create_mirror_view(toks);
  for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
  Kokkos::deep_copy(toks, h);
  return toks;
}

template <class Scalar>
std::vector<float> logits_vec(const Kokkos::View<Scalar*, Exec>& lg) {
  auto h = Kokkos::create_mirror_view(lg);
  Kokkos::deep_copy(h, lg);
  std::vector<float> out(h.extent(0));
  for (size_t i = 0; i < h.extent(0); ++i) out[i] = static_cast<float>(h(i));
  return out;
}

template <class Scalar>
void expect_logits_close(const std::vector<float>& a,
                         const std::vector<float>& b, float tol) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_NEAR(a[i], b[i], tol) << "logit " << i;
}

template <class Scalar>
void expect_logits_exact(const std::vector<float>& a,
                         const std::vector<float>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) EXPECT_FLOAT_EQ(a[i], b[i]);
}

// Prefill + two greedy decode steps, returning per-step logits. Drives
// forward() directly so the KV cache and conv states (not generate()'s own
// reset) are exercised.
template <class Scalar>
std::vector<std::vector<float>> run_prefill_decode(
    KokkosDNN::LFM25::Model<Exec, Scalar>& m,
    const std::vector<int>& prompt) {
  std::vector<std::vector<float>> all;
  auto lg = m.forward(make_tok_view<Scalar>(prompt));
  all.push_back(logits_vec(lg));
  int nxt = KokkosDNN::sample_greedy<Exec, Scalar>(lg);
  for (int s = 0; s < 2; ++s) {
    auto lg2 = m.forward(make_tok_view<Scalar>(std::vector<int>{nxt}));
    all.push_back(logits_vec(lg2));
    nxt = KokkosDNN::sample_greedy<Exec, Scalar>(lg2);
  }
  return all;
}

}  // namespace

template <class Scalar>
void check_forward_shape() {
  auto cfg = KokkosDNN::LFM25::Config::tiny_test();
  KokkosDNN::LFM25::Weights<Exec, Scalar> w;
  KokkosDNN::LFM25::random_init(cfg, w, 1234);
  KokkosDNN::LFM25::Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};
  auto lg                 = model.forward(make_tok_view<Scalar>(prompt));
  EXPECT_EQ(lg.extent(0), (size_t)cfg.vocab_size);
  // Logits are finite (random tiny weights, no NaN/Inf plumbing issues).
  for (float v : logits_vec(lg)) EXPECT_TRUE(std::isfinite(v));
}

TEST(LFM25Generate, ForwardShapeFloat) { check_forward_shape<float>(); }

TEST(LFM25Generate, ForwardShapeBHalf) {
  check_forward_shape<Kokkos::Experimental::bhalf_t>();
}

// Chunked prefill (chunk 2) matches single-shot (chunk <= 0) logits:
// each chunk reads previously written KV plus its own prefix, and conv
// chunks chain through the persistent conv state. Native kernels are
// bit-exact across chunkings; TPL GEMMs are not, so compare with a small
// tolerance (1e-4 float, 0.5 bhalf).
template <class Scalar>
void check_chunked_prefill() {
  constexpr float tol = std::is_same_v<Scalar, float> ? 1e-4f : 0.5f;
  auto cfg            = KokkosDNN::LFM25::Config::tiny_test();
  KokkosDNN::LFM25::Weights<Exec, Scalar> w;
  KokkosDNN::LFM25::random_init(cfg, w, 1234);
  std::vector<int> prompt = {1, 2, 3, 4, 5, 6, 7, 8};

  auto forward_logits = [&](int chunk) {
    KokkosDNN::LFM25::Model<Exec, Scalar> m(cfg, w);
    m.cfg.prefill_chunk_size = chunk;
    return logits_vec(m.forward(make_tok_view<Scalar>(prompt)));
  };

  auto ref = forward_logits(0);  // single-shot reference
  ASSERT_EQ(ref.size(), (size_t)cfg.vocab_size);
  expect_logits_close<Scalar>(forward_logits(2), ref, tol);

  // End-to-end greedy generate() agrees across chunkings.
  KokkosDNN::SampleOptions gopt;  // temperature 0: greedy
  gopt.seed = 42;
  KokkosDNN::LFM25::Model<Exec, Scalar> m1(cfg, w), m2(cfg, w);
  m1.cfg.prefill_chunk_size = 0;
  m2.cfg.prefill_chunk_size = 2;
  KokkosDNN::HostRNG r1(gopt.seed), r2(gopt.seed);
  EXPECT_EQ(m2.generate(prompt, 4, std::vector<int>{}, gopt, &r2),
            m1.generate(prompt, 4, std::vector<int>{}, gopt, &r1));
}

TEST(LFM25ChunkedPrefill, Float) { check_chunked_prefill<float>(); }

TEST(LFM25ChunkedPrefill, BHalf) {
  check_chunked_prefill<Kokkos::Experimental::bhalf_t>();
}

// reset() rewinds the position counter and zeroes conv states (unlike KV
// entries, stale conv history WOULD be observed: the first chunk's left
// context reads history rows unconditionally), so a rerun reproduces a
// fresh model exactly.
template <class Scalar>
void check_reset_determinism() {
  using namespace KokkosDNN::LFM25;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  Model<Exec, Scalar> m(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};

  auto first = run_prefill_decode(m, prompt);
  EXPECT_EQ(m.seq_len, prompt.size() + 2);
  m.reset();
  EXPECT_EQ(m.seq_len, 0u);
  auto second = run_prefill_decode(m, prompt);
  ASSERT_EQ(first.size(), second.size());
  for (size_t i = 0; i < first.size(); ++i)
    expect_logits_exact<Scalar>(first[i], second[i]);

  // Fresh model agrees bit-for-bit (conv state included).
  Model<Exec, Scalar> fresh(cfg, w);
  auto ref = run_prefill_decode(fresh, prompt);
  ASSERT_EQ(ref.size(), second.size());
  for (size_t i = 0; i < ref.size(); ++i)
    expect_logits_exact<Scalar>(ref[i], second[i]);

  // Partial fill then reset: stale prefix must not leak into the rerun.
  (void)logits_vec(m.forward(make_tok_view<Scalar>(std::vector<int>{7, 8})));
  m.reset();
  auto rerun = run_prefill_decode(m, prompt);
  ASSERT_EQ(rerun.size(), ref.size());
  for (size_t i = 0; i < rerun.size(); ++i)
    expect_logits_exact<Scalar>(rerun[i], ref[i]);
}

TEST(LFM25Reset, DeterminismVsFreshFloat) {
  check_reset_determinism<float>();
}

TEST(LFM25Reset, DeterminismVsFreshBHalf) {
  check_reset_determinism<Kokkos::Experimental::bhalf_t>();
}

// Greedy generate (temperature 0, fixed seed) is deterministic: same seed
// + same prompt => identical output, across fresh models and across reuse
// of one model (generate() resets internally).
template <class Scalar>
void check_greedy_determinism() {
  using namespace KokkosDNN::LFM25;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  std::vector<int> prompt = {1, 2, 3, 4};

  KokkosDNN::SampleOptions gopt;
  gopt.temperature = 0.0f;  // greedy
  gopt.seed        = 2026;

  Model<Exec, Scalar> ma(cfg, w), mb(cfg, w);
  KokkosDNN::HostRNG ra(gopt.seed), rb(gopt.seed);
  auto oa = ma.generate(prompt, 4, std::vector<int>{}, gopt, &ra);
  auto ob = mb.generate(prompt, 4, std::vector<int>{}, gopt, &rb);
  ASSERT_EQ(oa.size(), prompt.size() + 4);
  EXPECT_EQ(oa, ob);
  for (int id : oa) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, cfg.vocab_size);
  }

  // Reusing one model reproduces the run (reset + conv-state rewind).
  KokkosDNN::HostRNG rc(gopt.seed);
  EXPECT_EQ(ma.generate(prompt, 4, std::vector<int>{}, gopt, &rc), oa);

  // Null RNG with a fixed seed is equally deterministic.
  Model<Exec, Scalar> mc(cfg, w), md(cfg, w);
  EXPECT_EQ(mc.generate(prompt, 4, std::vector<int>{}, gopt, nullptr),
            md.generate(prompt, 4, std::vector<int>{}, gopt, nullptr));
}

TEST(LFM25GreedyGenerate, DeterminismFloat) {
  check_greedy_determinism<float>();
}

TEST(LFM25GreedyGenerate, DeterminismBHalf) {
  check_greedy_determinism<Kokkos::Experimental::bhalf_t>();
}
