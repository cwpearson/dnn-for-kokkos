// Attention: single query attending to a single key must return its value
// (float + bfloat16). Also covers embedding_gather on both scalars.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>

#include "KokkosDNN.hpp"

template <class Scalar>
void check_attention(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int Hq = 4, Hkv = 2, D = 8;
  Kokkos::View<Scalar***, Exec> q("q", 1, Hq, D), k("k", 1, Hkv, D),
      v("v", 1, Hkv, D), out("out", 1, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int hh = 0; hh < Hq; ++hh)
    for (int d = 0; d < D; ++d)
      hq(0, hh, d) = static_cast<Scalar>(0.1f * (hh + d + 1));
  for (int hh = 0; hh < Hkv; ++hh)
    for (int d = 0; d < D; ++d) {
      hk(0, hh, d) = static_cast<Scalar>(0.2f * (hh - d));
      hv(0, hh, d) = static_cast<Scalar>(1.0f * hh + 0.01f * d + 5.0f);
    }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  KokkosDNN::attention_gqa_single<Exec, Scalar>(h, q, k, v, 0, out,
                                               1.0f / 8.0f);
  // embedding_gather smoke cover on the same scalar.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> tab("tab", 4, D),
      emb("emb", 2, D);
  Kokkos::View<int*, Exec> toks("toks", 2);
  {
    auto ht = Kokkos::create_mirror_view(tab);
    for (int i = 0; i < 4; ++i)
      for (int d = 0; d < D; ++d)
        ht(i, d) = static_cast<Scalar>(0.1f * i + 0.01f * d);
    Kokkos::deep_copy(tab, ht);
    auto hx = Kokkos::create_mirror_view(toks);
    hx(0) = 3;
    hx(1) = 1;
    Kokkos::deep_copy(toks, hx);
  }
  KokkosDNN::embedding_gather<Exec, Scalar>(tab, toks, emb);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int hh = 0; hh < Hq; ++hh) {
    int hkv = (hh * Hkv) / Hq;
    for (int d = 0; d < D; ++d)
      EXPECT_NEAR(static_cast<float>(got(0, hh, d)),
                  static_cast<float>(hv(0, hkv, d)), tol)
          << "attention single-key mismatch at (" << hh << "," << d << ")";
  }
  auto ge = Kokkos::create_mirror_view(emb);
  Kokkos::deep_copy(ge, emb);
  auto gt = Kokkos::create_mirror_view(tab);
  Kokkos::deep_copy(gt, tab);
  for (int d = 0; d < D; ++d) {
    EXPECT_EQ(static_cast<float>(ge(0, d)), static_cast<float>(gt(3, d)))
        << "embedding_gather mismatch at (" << d << ")";
    EXPECT_EQ(static_cast<float>(ge(1, d)), static_cast<float>(gt(1, d)))
        << "embedding_gather mismatch at (" << d << ")";
  }
}

TEST(Attention, Float) { check_attention<float>(1e-4f); }

TEST(Attention, BHalf) { check_attention<Kokkos::Experimental::bhalf_t>(5e-2f); }
