// Performance-TODO regression tests (always-on, tiny config, no downloads).
//
// Covers the "Performance" README item (minus weight quantization, out of
// scope):
//   (a) lm_head through the gemm/gemv dispatch -- gemv-vs-reference parity
//       (float + bhalf_t) and gemv-vs-gemm(M=1) agreement;
//   (b) cached/persistent TPL handles -- one handle drives many calls with
//       a stable accessor (no per-call setup observable);
//   (c) no per-token host mirrors + deep_copies -- decode reaches a steady
//       state (scratch/input capacities stop growing, single-token slot
//       stays 1) and token-by-token stepping is deterministic.
// Sampling streaming (device greedy argmax) is exercised implicitly: the
// generate() paths below run the default greedy options end to end.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"

namespace {

// Deterministic fill: small magnitudes, both signs.
template <class Scalar>
void fill_lm_head_mats(
    const Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace>& W) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto hx    = Kokkos::create_mirror_view(x);
  auto hw    = Kokkos::create_mirror_view(W);
  for (size_t k = 0; k < hx.extent(0); ++k)
    hx(k) = static_cast<Scalar>(0.02f * ((int)(k * 13) % 17) - 0.15f);
  for (size_t j = 0; j < hw.extent(0); ++j)
    for (size_t k = 0; k < hw.extent(1); ++k)
      hw(j, k) =
          static_cast<Scalar>(0.01f * ((int)(j * 7 + k * 3) % 19) - 0.08f);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(W, hw);
  (void)Exec{};
}

template <class Scalar>
void check_gemv_parity(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  // Odd shapes (scalar kernel edges) + a tiny vocab-like shape.
  for (auto [N, K] : {std::make_pair(257, 65), std::make_pair(256, 64)}) {
    Kokkos::View<Scalar*, Exec> x("x", K), y("y", N);
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> W("W", N, K);
    fill_lm_head_mats(x, W);
    KokkosDNN::gemv(h, x, W, y);
    Kokkos::fence();
    auto hx = Kokkos::create_mirror_view(x);
    auto hw = Kokkos::create_mirror_view(W);
    auto hy = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(hx, x);
    Kokkos::deep_copy(hw, W);
    Kokkos::deep_copy(hy, y);
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(hx(k)) * static_cast<float>(hw(j, k));
      EXPECT_NEAR(static_cast<float>(hy(j)), acc, tol)
          << "gemv parity at (" << N << "," << K << ") j=" << j;
    }
    // Same op as M=1 GEMM must agree (one dispatch ladder, no divergence).
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X2("X2", 1, K),
        Y2("Y2", 1, N);
    {
      auto hx2 = Kokkos::create_mirror_view(X2);
      for (int k = 0; k < K; ++k) hx2(0, k) = hx(k);
      Kokkos::deep_copy(X2, hx2);
    }
    KokkosDNN::gemm(h, static_cast<Scalar>(1), X2, W, static_cast<Scalar>(0),
                    Y2);
    Kokkos::fence();
    auto hy2 = Kokkos::create_mirror_view(Y2);
    Kokkos::deep_copy(hy2, Y2);
    for (int j = 0; j < N; ++j)
      EXPECT_FLOAT_EQ(static_cast<float>(hy(j)), static_cast<float>(hy2(0, j)))
          << "gemv-vs-gemm(M=1) at (" << N << "," << K << ") j=" << j;
  }
}

}  // namespace

TEST(LmHeadPerf, GemvParityFloat) { check_gemv_parity<float>(1e-4f); }

TEST(LmHeadPerf, GemvParityBHalf) {
  check_gemv_parity<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// One handle drives many gemm/gemv calls with a stable TPL accessor: the
// cached handle (or its nullptr "no TPL here" state on CPU builds) must not
// churn per call, and results stay correct throughout.
TEST(HandlePerf, ReuseAcrossManyCalls) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  void* cublas0 = h.cublas_raw();
  void* cudnn0  = h.cudnn_raw();
  EXPECT_EQ(h.cublas_raw(), cublas0);
  EXPECT_EQ(h.cudnn_raw(), cudnn0);

  const int M = 4, K = 8, N = 6;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  Kokkos::View<float*, Exec> x("x", K), y("y", N);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) hx(i, k) = 0.1f * (i * K + k + 1);
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k) hw(j, k) = 0.01f * (j * K - k + 3);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  {
    auto hx1 = Kokkos::create_mirror_view(x);
    for (int k = 0; k < K; ++k) hx1(k) = hx(0, k);
    Kokkos::deep_copy(x, hx1);
  }
  // ~2x the per-token GEMM count of a tiny 2-layer model, on one handle.
  for (int it = 0; it < 32; ++it) {
    KokkosDNN::gemm(h, 1.0f, X, W, 0.0f, Y);
    KokkosDNN::gemv(h, x, W, y);
  }
  Kokkos::fence();
  EXPECT_EQ(h.cublas_raw(), cublas0);  // still the same cached handle
  EXPECT_EQ(h.cudnn_raw(), cudnn0);
  auto hy = Kokkos::create_mirror_view(Y);
  auto hv = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(hy, Y);
  Kokkos::deep_copy(hv, y);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k) acc += hx(i, k) * hw(j, k);
      EXPECT_NEAR(hy(i, j), acc, 1e-4f);
    }
  for (int j = 0; j < N; ++j) {
    float acc = 0;
    for (int k = 0; k < K; ++k) acc += hx(0, k) * hw(j, k);
    EXPECT_NEAR(hv(j), acc, 1e-4f);
  }
}

template <class Scalar>
void check_decode_steady_state() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  Model<Exec, Scalar> model(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};

  auto first = model.generate(prompt, 4);
  ASSERT_EQ(first.size(), prompt.size() + 4);
  const size_t scratch_after_first = model.scratch_cap_;
  const size_t tok_after_first      = model.tok_cap_;
  EXPECT_EQ(model.one_buf_.extent(0), 1u);

  // Second generate on the same model: identical output with no capacity
  // growth -- decode performs no per-token View allocations or host
  // staging growth (O(1) reset + persistent scratch/slots).
  auto second = model.generate(prompt, 4);
  EXPECT_EQ(first, second);
  EXPECT_EQ(model.scratch_cap_, scratch_after_first);
  EXPECT_EQ(model.tok_cap_, tok_after_first);
  EXPECT_EQ(model.one_buf_.extent(0), 1u);

  // Longer prompt grows input staging once, then steadies again.
  std::vector<int> longer = {5, 6, 7, 8, 9, 10, 11, 12};
  Model<Exec, Scalar> fresh(cfg, w);
  EXPECT_EQ(model.generate(longer, 3), fresh.generate(longer, 3));
  const size_t scratch_grown = model.scratch_cap_;
  const size_t tok_grown      = model.tok_cap_;
  auto rep1                  = model.generate(longer, 3);
  auto rep2                  = model.generate(longer, 3);
  EXPECT_EQ(rep1, rep2);
  EXPECT_EQ(model.scratch_cap_, scratch_grown);
  EXPECT_EQ(model.tok_cap_, tok_grown);
}

TEST(DecodeSteadyState, Float) { check_decode_steady_state<float>(); }

TEST(DecodeSteadyState, BHalf) {
  check_decode_steady_state<Kokkos::Experimental::bhalf_t>();
}

// Token-by-token determinism: two identically built models stepped through
// forward() one token at a time (device single-token slots, greedy argmax
// on device) produce identical per-step logits and ids.
template <class Scalar>
void check_token_step_determinism() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(cfg, w, 1234);
  Model<Exec, Scalar> m1(cfg, w), m2(cfg, w);
  std::vector<int> prompt = {1, 2, 3, 4};

  auto to_device = [](const std::vector<int>& ids) {
    Kokkos::View<int*, Exec> v("toks", ids.size());
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
    Kokkos::deep_copy(v, h);
    return v;
  };
  auto to_host = [](const Kokkos::View<Scalar*, Exec>& lg) {
    auto h = Kokkos::create_mirror_view(lg);
    Kokkos::deep_copy(h, lg);
    std::vector<float> out(h.extent(0));
    for (size_t i = 0; i < h.extent(0); ++i) out[i] = static_cast<float>(h(i));
    return out;
  };

  auto l1 = m1.forward(to_device(prompt));
  auto l2 = m2.forward(to_device(prompt));
  EXPECT_EQ(to_host(l1), to_host(l2));
  int n1 = KokkosDNN::sample_greedy<Exec, Scalar>(l1);
  int n2 = KokkosDNN::sample_greedy<Exec, Scalar>(l2);
  EXPECT_EQ(n1, n2);
  for (int s = 0; s < 3; ++s) {
    Kokkos::View<int*, Exec> o1("one", 1), o2("one", 1);
    {
      auto h1 = Kokkos::create_mirror_view(o1);
      auto h2 = Kokkos::create_mirror_view(o2);
      h1(0) = n1;
      h2(0) = n2;
      Kokkos::deep_copy(o1, h1);
      Kokkos::deep_copy(o2, h2);
    }
    auto g1 = m1.forward(o1);
    auto g2 = m2.forward(o2);
    EXPECT_EQ(to_host(g1), to_host(g2)) << "logits diverged at decode step " << s;
    n1 = KokkosDNN::sample_greedy<Exec, Scalar>(g1);
    n2 = KokkosDNN::sample_greedy<Exec, Scalar>(g2);
    EXPECT_EQ(n1, n2) << "sampled id diverged at decode step " << s;
  }
}

TEST(TokenStepDeterminism, Float) { check_token_step_determinism<float>(); }

TEST(TokenStepDeterminism, BHalf) {
  check_token_step_determinism<Kokkos::Experimental::bhalf_t>();
}
