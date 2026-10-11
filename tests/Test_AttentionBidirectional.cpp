// Bidirectional (non-causal) MHA: host-reference agreement (float +
// bfloat16, MHA + GQA mappings), default-scale == 1/sqrt(Dh), and the
// non-causal property (early outputs depend on future tokens, unlike the
// causal kernel).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include "KokkosDNN.hpp"

// Host reference: dense softmax over ALL keys, FP32 accumulation.
template <class Scalar, class HQ, class HK, class HV>
void host_bidir_ref(const HQ& hq, const HK& hk, const HV& hv, float scale,
                    int T, int K, int Hq, int Hkv, int D, float* ref) {
  auto at = [=](float* b, int t, int h, int d) -> float& {
    return b[(size_t)(t * Hq + h) * D + d];
  };
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < Hq; ++h) {
      int hkv = (h * Hkv) / Hq;
      float m = -1e30f;
      for (int kk = 0; kk < K; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, h, d)) *
               static_cast<float>(hk(kk, hkv, d));
        m = std::max(m, s * scale);
      }
      float denom = 0;
      for (int kk = 0; kk < K; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, h, d)) *
               static_cast<float>(hk(kk, hkv, d));
        denom += std::exp(s * scale - m);
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (int kk = 0; kk < K; ++kk) {
          float s = 0;
          for (int dd = 0; dd < D; ++dd)
            s += static_cast<float>(hq(t, h, dd)) *
                 static_cast<float>(hk(kk, hkv, dd));
          acc += std::exp(s * scale - m) * static_cast<float>(hv(kk, hkv, d));
        }
        at(ref, t, h, d) = acc / denom;
      }
    }
}

template <class Scalar>
void check_bidir_reference(int T, int K, int Hq, int Hkv, int D, float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  Kokkos::View<Scalar***, Exec> q("q", T, Hq, D), k("k", K, Hkv, D),
      v("v", K, Hkv, D), out("out", T, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) =
            static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f);
  for (int n = 0; n < K; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f);
        hv(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f);
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);

  // Deduction overload (Scalar from query view).
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, out, scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);

  std::vector<float> ref((size_t)T * Hq * D);
  host_bidir_ref<Scalar>(hq, hk, hv, scale, T, K, Hq, Hkv, D, ref.data());
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(static_cast<float>(got(t, hh, d)),
                    ref[(size_t)(t * Hq + hh) * D + d], tol)
            << "bidir reference mismatch at (" << t << "," << hh << "," << d
            << ")";
}

// ViT-B/16 MHA shape (12 heads, head_dim 64), self-attention K == T.
TEST(AttentionBidirectional, FloatMHAReference) {
  check_bidir_reference<float>(/*T=*/6, /*K=*/6, /*Hq=*/4, /*Hkv=*/4,
                               /*D=*/8, 1e-4f);
}

TEST(AttentionBidirectional, FloatGQAReference) {
  check_bidir_reference<float>(/*T=*/5, /*K=*/7, /*Hq=*/4, /*Hkv=*/2,
                               /*D=*/8, 1e-4f);
}

TEST(AttentionBidirectional, BHalfMHAReference) {
  check_bidir_reference<Kokkos::Experimental::bhalf_t>(
      /*T=*/5, /*K=*/5, /*Hq=*/2, /*Hkv=*/2, /*D=*/8, 5e-2f);
}

// ViT-B/16 head_dim default scale: omitted scale (<= 0 => 1/sqrt(Dh))
// must match an explicit 1/sqrt(Dh) call, using Dh=64 like the model.
TEST(AttentionBidirectional, DefaultScaleMatchesExplicit) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 4, H = 2, D = 64;
  const float explicit_scale = 1.0f / std::sqrt(static_cast<float>(D));
  Kokkos::View<float***, Exec> q("q", T, H, D), k("k", T, H, D),
      v("v", T, H, D), o_def("o_def", T, H, D), o_exp("o_exp", T, H, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < H; ++hh)
      for (int d = 0; d < D; ++d) {
        hq(t, hh, d) = 0.02f * ((t * 11 + hh * 7 + d) % 23) - 0.2f;
        hk(t, hh, d) = 0.02f * ((t * 5 + hh * 13 + d * 2) % 29) - 0.2f;
        hv(t, hh, d) = 0.02f * ((t * 3 + hh * 17 + d) % 31) - 0.3f;
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  // Explicit <Exec, Scalar> overload with defaulted scale vs explicit scale.
  KokkosDNN::attention_mha_bidirectional<Exec, float>(h, q, k, v, o_def);
  KokkosDNN::attention_mha_bidirectional<Exec, float>(h, q, k, v, o_exp,
                                                     explicit_scale);
  Kokkos::fence();
  auto gd = Kokkos::create_mirror_view(o_def);
  auto ge = Kokkos::create_mirror_view(o_exp);
  Kokkos::deep_copy(gd, o_def);
  Kokkos::deep_copy(ge, o_exp);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < H; ++hh)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(gd(t, hh, d), ge(t, hh, d), 1e-6f)
            << "default-scale mismatch at (" << t << "," << hh << "," << d
            << ")";
}

// Non-causal property: perturbing a FUTURE key/value (last token) must
// change the FIRST query's output under the bidirectional kernel, while
// the causal kernel's first output (q_off=0, attending only to key 0)
// must be bit-identical.
TEST(AttentionBidirectional, FutureTokensAffectEarlyOutputs) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 4, Hq = 2, Hkv = 2, D = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  Kokkos::View<float***, Exec> q("q", T, Hq, D), k("k", T, Hkv, D),
      v("v", T, Hkv, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) = 0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f;
  for (int n = 0; n < T; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) = 0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f;
        hv(n, hh, d) = 0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f;
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);

  Kokkos::View<float***, Exec> ob1("ob1", T, Hq, D), oc1("oc1", T, Hq, D);
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, ob1, scale);
  KokkosDNN::attention_gqa_single<Exec>(h, q, k, v, /*q_off=*/0, oc1, scale);
  Kokkos::fence();

  // Perturb only the LAST token's K/V (strictly future for query 0).
  {
    auto hk2 = Kokkos::create_mirror_view(k);
    auto hv2 = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(hk2, k);
    Kokkos::deep_copy(hv2, v);
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk2(T - 1, hh, d) += 2.5f;
        hv2(T - 1, hh, d) += 2.5f;
      }
    Kokkos::deep_copy(k, hk2);
    Kokkos::deep_copy(v, hv2);
  }

  Kokkos::View<float***, Exec> ob2("ob2", T, Hq, D), oc2("oc2", T, Hq, D);
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, ob2, scale);
  KokkosDNN::attention_gqa_single<Exec>(h, q, k, v, /*q_off=*/0, oc2, scale);
  Kokkos::fence();
  auto gb1 = Kokkos::create_mirror_view(ob1);
  auto gb2 = Kokkos::create_mirror_view(ob2);
  auto gc1 = Kokkos::create_mirror_view(oc1);
  auto gc2 = Kokkos::create_mirror_view(oc2);
  Kokkos::deep_copy(gb1, ob1);
  Kokkos::deep_copy(gb2, ob2);
  Kokkos::deep_copy(gc1, oc1);
  Kokkos::deep_copy(gc2, oc2);

  float bidir_delta = 0, causal_delta = 0;
  for (int hh = 0; hh < Hq; ++hh)
    for (int d = 0; d < D; ++d) {
      bidir_delta =
          std::max(bidir_delta, std::abs(gb2(0, hh, d) - gb1(0, hh, d)));
      causal_delta =
          std::max(causal_delta, std::abs(gc2(0, hh, d) - gc1(0, hh, d)));
    }
  EXPECT_GT(bidir_delta, 1e-3f)
      << "bidirectional out(0) ignores future tokens (looks causal)";
  EXPECT_EQ(causal_delta, 0.0f)
      << "causal out(0) changed under future-token perturbation";
}
