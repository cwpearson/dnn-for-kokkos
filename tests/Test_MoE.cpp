// MoE routing (top-k select + softmax/sigmoid weights), expert weighted-sum
// helper, and fused per-head QK RMSNorm vs host references.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_MoE.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;

// Independent host reference: sort-based top-k (score desc, ties -> lower
// id), softmax over selected (norm_topk) or global softmax gathered at the
// selected ids (!norm_topk), times routed_scale.
void host_moe_ref(const std::vector<std::vector<float>>& logits,
                  const std::vector<float>& bias, int K, bool norm_topk,
                  float scale, std::vector<std::vector<int>>& idx,
                  std::vector<std::vector<float>>& w) {
  const size_t T = logits.size(), E = logits[0].size();
  idx.assign(T, std::vector<int>(K, -1));
  w.assign(T, std::vector<float>(K, 0));
  for (size_t t = 0; t < T; ++t) {
    std::vector<int> order(E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      float sa = logits[t][a] + (bias.empty() ? 0.0f : bias[a]);
      float sb = logits[t][b] + (bias.empty() ? 0.0f : bias[b]);
      return sa > sb;
    });
    float denom = 0.0f, mx = -3.4028235e38f;
    if (norm_topk) {
      for (int k = 0; k < K; ++k) {
        float s = logits[t][order[k]] +
                  (bias.empty() ? 0.0f : bias[order[k]]);
        if (s > mx) mx = s;
      }
      for (int k = 0; k < K; ++k) {
        float s = logits[t][order[k]] +
                  (bias.empty() ? 0.0f : bias[order[k]]);
        denom += std::exp(s - mx);
      }
    } else {
      for (size_t e = 0; e < E; ++e) {
        float s = logits[t][e] + (bias.empty() ? 0.0f : bias[e]);
        if (s > mx) mx = s;
      }
      for (size_t e = 0; e < E; ++e) {
        float s = logits[t][e] + (bias.empty() ? 0.0f : bias[e]);
        denom += std::exp(s - mx);
      }
    }
    for (int k = 0; k < K; ++k) {
      idx[t][k] = order[k];
      float s =
          logits[t][order[k]] + (bias.empty() ? 0.0f : bias[order[k]]);
      w[t][k] = std::exp(s - mx) / denom * scale;
    }
  }
}

std::vector<std::vector<float>> fixed_logits() {
  return {{0.1f, 2.0f, -1.0f, 0.5f, 1.2f, -0.3f},
          {1.5f, 1.5f, 1.5f, -2.0f, 0.0f, 3.0f},   // 3-way tie -> ids 0,1,2
          {-0.5f, -0.2f, -0.8f, -0.1f, -1.5f, -0.4f}};
}

void run_route(const std::vector<std::vector<float>>& logits,
               const std::vector<float>& bias, int K, bool norm_topk,
               float scale, Kokkos::View<int**, Kokkos::LayoutRight, Exec>& idx,
               Kokkos::View<float**, Kokkos::LayoutRight, Exec>& w) {
  const size_t T = logits.size(), E = logits[0].size();
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> lg("lg", T, E);
  auto hlg = Kokkos::create_mirror_view(lg);
  for (size_t t = 0; t < T; ++t)
    for (size_t e = 0; e < E; ++e) hlg(t, e) = logits[t][e];
  Kokkos::deep_copy(lg, hlg);
  Kokkos::View<float*, Exec> b("b", bias.size());
  if (!bias.empty()) {
    auto hb = Kokkos::create_mirror_view(b);
    for (size_t e = 0; e < bias.size(); ++e) hb(e) = bias[e];
    Kokkos::deep_copy(b, hb);
  }
  KokkosDNN::moe_route<Exec, float>(lg, b, K, norm_topk, scale, idx, w);
  Kokkos::fence();
}

}  // namespace

TEST(MoE, TopKSoftmaxMatchesHost) {
  const auto logits = fixed_logits();
  const size_t T = logits.size();
  const int K = 2;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", T, K);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", T, K);
  run_route(logits, {}, K, true, 1.0f, idx, w);
  std::vector<std::vector<int>> ridx;
  std::vector<std::vector<float>> rw;
  host_moe_ref(logits, {}, K, true, 1.0f, ridx, rw);
  auto gidx = Kokkos::create_mirror_view(idx);
  auto gw = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(gidx, idx);
  Kokkos::deep_copy(gw, w);
  for (size_t t = 0; t < T; ++t) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      EXPECT_EQ(gidx(t, k), ridx[t][k]) << " at (" << t << "," << k << ")";
      EXPECT_NEAR(gw(t, k), rw[t][k], 1e-5f) << " at (" << t << "," << k << ")";
      sum += gw(t, k);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << " weights must sum to 1 at t=" << t;
  }
}

TEST(MoE, ExpertBiasShiftsSelection) {
  const auto logits = fixed_logits();
  const size_t T = logits.size(), E = logits[0].size();
  const int K = 2;
  // Boost expert 3 far above the rest on every token.
  std::vector<float> bias(E, 0.0f);
  bias[3] = 10.0f;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", T, K);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", T, K);
  run_route(logits, bias, K, true, 1.0f, idx, w);
  std::vector<std::vector<int>> ridx;
  std::vector<std::vector<float>> rw;
  host_moe_ref(logits, bias, K, true, 1.0f, ridx, rw);
  auto gidx = Kokkos::create_mirror_view(idx);
  auto gw = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(gidx, idx);
  Kokkos::deep_copy(gw, w);
  for (size_t t = 0; t < T; ++t) {
    EXPECT_EQ(gidx(t, 0), 3) << " biased expert must win at t=" << t;
    EXPECT_EQ(gidx(t, 0), ridx[t][0]);
    for (int k = 0; k < K; ++k)
      EXPECT_NEAR(gw(t, k), rw[t][k], 1e-5f) << " at (" << t << "," << k << ")";
  }
}

TEST(MoE, NoRenormMatchesGlobalSoftmax) {
  const auto logits = fixed_logits();
  const size_t T = logits.size();
  const int K = 2;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", T, K);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", T, K);
  run_route(logits, {}, K, false, 1.0f, idx, w);
  std::vector<std::vector<int>> ridx;
  std::vector<std::vector<float>> rw;
  host_moe_ref(logits, {}, K, false, 1.0f, ridx, rw);
  auto gidx = Kokkos::create_mirror_view(idx);
  auto gw = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(gidx, idx);
  Kokkos::deep_copy(gw, w);
  for (size_t t = 0; t < T; ++t) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      EXPECT_EQ(gidx(t, k), ridx[t][k]);
      EXPECT_NEAR(gw(t, k), rw[t][k], 1e-5f) << " at (" << t << "," << k << ")";
      sum += gw(t, k);
    }
    EXPECT_LT(sum, 1.0f) << " unnormalized top-k weights sum below 1";
    EXPECT_GT(sum, 0.0f);
  }
}

TEST(MoE, RoutedScaleAndSingleTokenDecode) {
  // routed_scale multiplies the (renormalized) weights; T=1 decode path.
  const std::vector<std::vector<float>> logits = {{0.3f, -0.7f, 1.1f, 0.0f}};
  const size_t E = 4;
  const int K = 3;
  const float scale = 2.5f;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", 1, K);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", 1, K);
  run_route(logits, {}, K, true, scale, idx, w);
  std::vector<std::vector<int>> ridx;
  std::vector<std::vector<float>> rw;
  host_moe_ref(logits, {}, K, true, scale, ridx, rw);
  auto gidx = Kokkos::create_mirror_view(idx);
  auto gw = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(gidx, idx);
  Kokkos::deep_copy(gw, w);
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    EXPECT_EQ(gidx(0, k), ridx[0][k]);
    EXPECT_NEAR(gw(0, k), rw[0][k], 1e-5f);
    sum += gw(0, k);
  }
  EXPECT_NEAR(sum, scale, 1e-5f);
  (void)E;
}

TEST(MoE, WeightedSumMatchesHost) {
  const size_t T = 3, K = 2, D = 5;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> eo("eo", T, K, D);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", T, K);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> y("y", T, D);
  auto heo = Kokkos::create_mirror_view(eo);
  auto hw = Kokkos::create_mirror_view(w);
  for (size_t t = 0; t < T; ++t)
    for (size_t k = 0; k < K; ++k) {
      hw(t, k) = 0.2f * static_cast<float>(t + 1) + 0.3f * (float)k + 0.1f;
      for (size_t d = 0; d < D; ++d)
        heo(t, k, d) =
            0.5f * (float)(t * 7 + k * 3 + d) - 1.0f;
    }
  Kokkos::deep_copy(eo, heo);
  Kokkos::deep_copy(w, hw);
  KokkosDNN::moe_expert_weighted_sum<Exec, float>(eo, w, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d) {
      double want = 0.0;
      for (size_t k = 0; k < K; ++k) want += (double)hw(t, k) * (double)heo(t, k, d);
      EXPECT_NEAR(got(t, d), want, 1e-5f) << " at (" << t << "," << d << ")";
    }
}

TEST(MoE, RejectsBadArgs) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> lg("lg", 2, 4);
  Kokkos::View<float*, Exec> b("b", 4), bad_b("bad_b", 3), empty_b;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", 2, 2),
      bad_idx("bad_idx", 2, 3);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", 2, 2);
  EXPECT_NO_THROW(
      (KokkosDNN::moe_route<Exec, float>(lg, empty_b, 2, true, 1.0f, idx, w)));
  EXPECT_NO_THROW(
      (KokkosDNN::moe_route<Exec, float>(lg, b, 2, true, 1.0f, idx, w)));
  EXPECT_THROW(
      (KokkosDNN::moe_route<Exec, float>(lg, bad_b, 2, true, 1.0f, idx, w)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::moe_route<Exec, float>(lg, empty_b, 0, true, 1.0f, idx, w)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::moe_route<Exec, float>(lg, empty_b, 5, true, 1.0f, idx, w)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::moe_route<Exec, float>(lg, empty_b, 2, true, 1.0f, bad_idx,
                                        w)),
      std::invalid_argument);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> eo("eo", 2, 2, 3);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> y("y", 2, 3),
      bad_y("bad_y", 2, 4);
  EXPECT_THROW(
      (KokkosDNN::moe_expert_weighted_sum<Exec, float>(eo, w, bad_y)),
      std::invalid_argument);
}

TEST(MoE, GatherRowsMatchesHost) {
  // dst[n,d] = src[rows[n],d], incl. repeated and permuted rows.
  const size_t C = 4, D = 3, N = 5;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> src("src", C, D);
  Kokkos::View<int*, Exec> rows("rows", N);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> dst("dst", N, D);
  auto hsrc = Kokkos::create_mirror_view(src);
  auto hrows = Kokkos::create_mirror_view(rows);
  for (size_t t = 0; t < C; ++t)
    for (size_t d = 0; d < D; ++d)
      hsrc(t, d) = 10.0f * (float)t + (float)d;
  const int order[5] = {3, 0, 3, 1, 2};
  for (size_t n = 0; n < N; ++n) hrows(n) = order[n];
  Kokkos::deep_copy(src, hsrc);
  Kokkos::deep_copy(rows, hrows);
  KokkosDNN::moe_gather_rows<Exec, float>(src, rows, dst);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(dst);
  Kokkos::deep_copy(got, dst);
  for (size_t n = 0; n < N; ++n)
    for (size_t d = 0; d < D; ++d)
      EXPECT_FLOAT_EQ(got(n, d), hsrc(order[n], d)) << " at (" << n << "," << d
                                                    << ")";
  // Empty gather is a no-op (dst untouched, no throw).
  Kokkos::View<int*, Exec> norows("norows", 0);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> nodst("nodst", 0, D);
  EXPECT_NO_THROW(
      (KokkosDNN::moe_gather_rows<Exec, float>(src, norows, nodst)));
  // Mismatched dst rows throw.
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad("bad", N + 1, D);
  EXPECT_THROW((KokkosDNN::moe_gather_rows<Exec, float>(src, rows, bad)),
               std::invalid_argument);
}

TEST(MoE, ScatterAddWeightedMatchesHost) {
  // dst[t,d] += weights[n] * src[n,d] over unique rows.
  const size_t C = 4, D = 3, N = 3;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> src("ssrc", N, D);
  Kokkos::View<int*, Exec> rows("srows", N);
  Kokkos::View<float*, Exec> weights("sw", N);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> dst("sdst", C, D);
  auto hsrc = Kokkos::create_mirror_view(src);
  auto hrows = Kokkos::create_mirror_view(rows);
  auto hw = Kokkos::create_mirror_view(weights);
  auto hdst = Kokkos::create_mirror_view(dst);
  const int order[3] = {2, 0, 3};
  const float wv[3] = {0.5f, -1.0f, 2.0f};
  for (size_t n = 0; n < N; ++n) {
    hrows(n) = order[n];
    hw(n) = wv[n];
    for (size_t d = 0; d < D; ++d)
      hsrc(n, d) = (float)(n * 10 + d) - 3.0f;
  }
  for (size_t t = 0; t < C; ++t)
    for (size_t d = 0; d < D; ++d) hdst(t, d) = 0.25f * (float)(t + d);
  Kokkos::deep_copy(src, hsrc);
  Kokkos::deep_copy(rows, hrows);
  Kokkos::deep_copy(weights, hw);
  Kokkos::deep_copy(dst, hdst);
  KokkosDNN::moe_scatter_add_weighted<Exec, float>(src, rows, weights, dst);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(dst);
  Kokkos::deep_copy(got, dst);
  for (size_t n = 0; n < N; ++n)
    for (size_t d = 0; d < D; ++d)
      hdst(order[n], d) += wv[n] * hsrc(n, d);
  for (size_t t = 0; t < C; ++t)
    for (size_t d = 0; d < D; ++d)
      EXPECT_FLOAT_EQ(got(t, d), hdst(t, d)) << " at (" << t << "," << d << ")";
  // Empty scatter is a no-op.
  Kokkos::View<int*, Exec> norows("snorows", 0);
  Kokkos::View<float*, Exec> now("snow", 0);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> nosrc("snosrc", 0, D);
  EXPECT_NO_THROW(
      (KokkosDNN::moe_scatter_add_weighted<Exec, float>(nosrc, norows, now,
                                                        dst)));
  // Mismatched weight count throws.
  Kokkos::View<float*, Exec> badw("sbadw", N + 1);
  EXPECT_THROW(
      (KokkosDNN::moe_scatter_add_weighted<Exec, float>(src, rows, badw, dst)),
      std::invalid_argument);
}

// Independent host reference for sigmoid routing (HF Lfm2MoeTopKRouter):
// selection over sigmoid(logit) + bias, weights are gathered sigmoids
// (bias excluded), renormalized over K with the HF 1e-6 epsilon when
// norm_topk, times scale.
void host_moe_sigmoid_ref(const std::vector<std::vector<float>>& logits,
                          const std::vector<float>& bias, int K, bool norm_topk,
                          float scale, std::vector<std::vector<int>>& idx,
                          std::vector<std::vector<float>>& w) {
  const size_t T = logits.size(), E = logits[0].size();
  auto sig = [](float s) { return 1.0f / (1.0f + std::exp(-s)); };
  idx.assign(T, std::vector<int>(K, -1));
  w.assign(T, std::vector<float>(K, 0));
  for (size_t t = 0; t < T; ++t) {
    std::vector<int> order(E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      float sa = sig(logits[t][a]) + (bias.empty() ? 0.0f : bias[a]);
      float sb = sig(logits[t][b]) + (bias.empty() ? 0.0f : bias[b]);
      return sa > sb;
    });
    float denom = 1e-6f;
    if (norm_topk) {
      for (int k = 0; k < K; ++k) denom += sig(logits[t][order[k]]);
    }
    for (int k = 0; k < K; ++k) {
      idx[t][k] = order[k];
      const float s = sig(logits[t][order[k]]);
      w[t][k]     = norm_topk ? s / denom * scale : s * scale;
    }
  }
}

void run_route_sigmoid(const std::vector<std::vector<float>>& logits,
                       const std::vector<float>& bias, int K, bool norm_topk,
                       float scale,
                       Kokkos::View<int**, Kokkos::LayoutRight, Exec>& idx,
                       Kokkos::View<float**, Kokkos::LayoutRight, Exec>& w) {
  const size_t T = logits.size(), E = logits[0].size();
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> lg("lg", T, E);
  auto hlg = Kokkos::create_mirror_view(lg);
  for (size_t t = 0; t < T; ++t)
    for (size_t e = 0; e < E; ++e) hlg(t, e) = logits[t][e];
  Kokkos::deep_copy(lg, hlg);
  Kokkos::View<float*, Exec> b("b", bias.size());
  if (!bias.empty()) {
    auto hb = Kokkos::create_mirror_view(b);
    for (size_t e = 0; e < bias.size(); ++e) hb(e) = bias[e];
    Kokkos::deep_copy(b, hb);
  }
  KokkosDNN::moe_route_sigmoid<Exec, float>(lg, b, K, norm_topk, scale, idx, w);
  Kokkos::fence();
}

TEST(MoE, SigmoidMatchesHost) {
  const auto logits = fixed_logits();
  const size_t T = logits.size(), E = logits[0].size();
  const int K = 2;
  for (bool norm_topk : {true, false}) {
    Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", T, K);
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", T, K);
    run_route_sigmoid(logits, {}, K, norm_topk, 1.0f, idx, w);
    std::vector<std::vector<int>> ridx;
    std::vector<std::vector<float>> rw;
    host_moe_sigmoid_ref(logits, {}, K, norm_topk, 1.0f, ridx, rw);
    auto gidx = Kokkos::create_mirror_view(idx);
    auto gw   = Kokkos::create_mirror_view(w);
    Kokkos::deep_copy(gidx, idx);
    Kokkos::deep_copy(gw, w);
    for (size_t t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        EXPECT_EQ(gidx(t, k), ridx[t][k])
            << " at (" << t << "," << k << ") norm_topk=" << norm_topk;
        EXPECT_NEAR(gw(t, k), rw[t][k], 1e-5f)
            << " at (" << t << "," << k << ") norm_topk=" << norm_topk;
      }
      if (norm_topk) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += gw(t, k);
        // HF renormalizes by sum + 1e-6, so weights sum to just under 1.
        EXPECT_NEAR(sum, 1.0f, 2e-5f) << " at t=" << t;
      }
    }
  }
  (void)E;
}

TEST(MoE, SigmoidBiasSelectsButDoesNotWeight) {
  // Bias shifts selection (expert 3 wins) but the weights are pure
  // renormalized sigmoids: zeroing the bias of a non-selected expert must
  // not change the weights, and the biased run must match a host reference
  // that excludes the bias from the weights.
  const auto logits = fixed_logits();
  const size_t T = logits.size(), E = logits[0].size();
  const int K = 2;
  std::vector<float> bias(E, 0.0f);
  bias[3] = 10.0f;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", T, K);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", T, K);
  run_route_sigmoid(logits, bias, K, true, 1.0f, idx, w);
  std::vector<std::vector<int>> ridx;
  std::vector<std::vector<float>> rw;
  host_moe_sigmoid_ref(logits, bias, K, true, 1.0f, ridx, rw);
  auto gidx = Kokkos::create_mirror_view(idx);
  auto gw   = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(gidx, idx);
  Kokkos::deep_copy(gw, w);
  for (size_t t = 0; t < T; ++t) {
    EXPECT_EQ(gidx(t, 0), 3) << " biased expert must win at t=" << t;
    for (int k = 0; k < K; ++k) {
      EXPECT_EQ(gidx(t, k), ridx[t][k]);
      EXPECT_NEAR(gw(t, k), rw[t][k], 1e-5f) << " at (" << t << "," << k << ")";
    }
    // Weights are bias-free sigmoids: sigmoid(l)/(sigmoid(l0)+sigmoid(l1)).
    auto sig = [](float s) { return 1.0f / (1.0f + std::exp(-s)); };
    const float s0 = sig(logits[t][static_cast<size_t>(gidx(t, 0))]);
    const float s1 = sig(logits[t][static_cast<size_t>(gidx(t, 1))]);
    EXPECT_NEAR(gw(t, 0), s0 / (s0 + s1 + 1e-6f), 1e-5f)
        << " weight must exclude bias at t=" << t;
  }
}

TEST(MoE, SigmoidMatchesSoftmaxOnTiesOnlyInSelection) {
  // Sigmoid routing and softmax routing select the same experts on unbiased
  // ties (both monotonic in the logits) but produce different weights;
  // this guards against regressing the LFM path back to softmax weights.
  const std::vector<std::vector<float>> logits = {{0.3f, -0.7f, 1.1f, 0.0f}};
  const int K = 3;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx_sig("is", 1, K),
      idx_sm("im", 1, K);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w_sig("ws", 1, K),
      w_sm("wm", 1, K);
  run_route_sigmoid(logits, {}, K, true, 1.0f, idx_sig, w_sig);
  run_route(logits, {}, K, true, 1.0f, idx_sm, w_sm);
  auto gs = Kokkos::create_mirror_view(w_sig);
  auto gm = Kokkos::create_mirror_view(w_sm);
  Kokkos::deep_copy(gs, w_sig);
  Kokkos::deep_copy(gm, w_sm);
  auto is = Kokkos::create_mirror_view(idx_sig);
  auto im = Kokkos::create_mirror_view(idx_sm);
  Kokkos::deep_copy(is, idx_sig);
  Kokkos::deep_copy(im, idx_sm);
  for (int k = 0; k < K; ++k) EXPECT_EQ(is(0, k), im(0, k));
  bool any_diff = false;
  for (int k = 0; k < K; ++k)
    if (std::abs(gs(0, k) - gm(0, k)) > 1e-3f) any_diff = true;
  EXPECT_TRUE(any_diff) << "sigmoid and softmax weights must differ";
}

TEST(MoE, SigmoidRejectsBadArgs) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> lg("lg", 2, 4);
  Kokkos::View<float*, Exec> b("b", 4), bad_b("bad_b", 3), empty_b;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", 2, 2),
      bad_idx("bad_idx", 2, 3);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", 2, 2);
  EXPECT_NO_THROW(
      (KokkosDNN::moe_route_sigmoid<Exec, float>(lg, empty_b, 2, true, 1.0f,
                                                 idx, w)));
  EXPECT_NO_THROW((KokkosDNN::moe_route_sigmoid<Exec, float>(lg, b, 2, true,
                                                             1.0f, idx, w)));
  EXPECT_THROW(
      (KokkosDNN::moe_route_sigmoid<Exec, float>(lg, bad_b, 2, true, 1.0f, idx,
                                                 w)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::moe_route_sigmoid<Exec, float>(lg, empty_b, 0, true, 1.0f,
                                                 idx, w)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::moe_route_sigmoid<Exec, float>(lg, empty_b, 5, true, 1.0f,
                                                 idx, w)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::moe_route_sigmoid<Exec, float>(lg, empty_b, 2, true, 1.0f,
                                                 bad_idx, w)),
      std::invalid_argument);
}

// Fused reshape + per-head QK RMSNorm vs per-head host reference.
template <class Scalar>
void check_qk_rmsnorm(float tol) {
  const size_t T = 3, H = 2, D = 6;
  const float eps = 1e-5f;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> flat("flat", T, H * D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", T, H, D),
      out3("out3", T, H, D), q3("q3", T, H, D);
  Kokkos::View<Scalar*, Exec> w("w", D);
  auto hflat = Kokkos::create_mirror_view(flat);
  auto hw = Kokkos::create_mirror_view(w);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < H * D; ++i)
      hflat(t, i) =
          static_cast<Scalar>(0.4f * (float)(t + 1) - 0.15f * (float)i);
  for (size_t d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(1.0f + 0.1f * d);
  Kokkos::deep_copy(flat, hflat);
  Kokkos::deep_copy(w, hw);
  auto hq3 = Kokkos::create_mirror_view(q3);
  for (size_t t = 0; t < T; ++t)
    for (size_t h = 0; h < H; ++h)
      for (size_t d = 0; d < D; ++d) hq3(t, h, d) = hflat(t, h * D + d);
  Kokkos::deep_copy(q3, hq3);
  KokkosDNN::qk_rmsnorm<Exec, Scalar>(flat, w, eps, out);
  KokkosDNN::qk_rmsnorm3d<Exec, Scalar>(q3, w, eps, out3);
  // In-place 3d call must also work.
  KokkosDNN::qk_rmsnorm3d<Exec, Scalar>(q3, w, eps, q3);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  auto got3 = Kokkos::create_mirror_view(out3);
  Kokkos::deep_copy(got3, out3);
  auto gotip = Kokkos::create_mirror_view(q3);
  Kokkos::deep_copy(gotip, q3);
  for (size_t t = 0; t < T; ++t)
    for (size_t h = 0; h < H; ++h) {
      double ss = 0.0;
      for (size_t d = 0; d < D; ++d) {
        double v = static_cast<double>(hflat(t, h * D + d));
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / (double)D + eps);
      for (size_t d = 0; d < D; ++d) {
        double want = static_cast<double>(hflat(t, h * D + d)) * inv *
                      static_cast<double>(hw(d));
        EXPECT_NEAR(static_cast<double>(got(t, h, d)), want, tol)
            << "qk_rmsnorm mismatch at (" << t << "," << h << "," << d << ")";
        EXPECT_NEAR(static_cast<double>(got3(t, h, d)), want, tol)
            << "qk_rmsnorm3d mismatch at (" << t << "," << h << "," << d
            << ")";
        EXPECT_NEAR(static_cast<double>(gotip(t, h, d)), want, tol)
            << "qk_rmsnorm3d in-place mismatch at (" << t << "," << h << ","
            << d << ")";
      }
    }
  // Dimension mismatch throws.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad_flat("bf", T, H * D - 1);
  EXPECT_THROW((KokkosDNN::qk_rmsnorm<Exec, Scalar>(bad_flat, w, eps, out)),
               std::invalid_argument);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad3("b3", T, H, D - 1);
  EXPECT_THROW((KokkosDNN::qk_rmsnorm3d<Exec, Scalar>(q3, w, eps, bad3)),
               std::invalid_argument);
}

TEST(QKRMSNorm, Float) { check_qk_rmsnorm<float>(1e-5f); }

TEST(QKRMSNorm, BHalf) {
  check_qk_rmsnorm<Kokkos::Experimental::bhalf_t>(5e-2f);
}
