// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// RMSNorm vs host reference (float + bfloat16).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "KokkosDNN.hpp"

template <class Scalar>
void check_rmsnorm(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 3, D = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  Kokkos::View<Scalar*, Exec> w("w", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.5f * (t + 1) - 0.25f * d);
  for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(1.0f + 0.1f * d);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  const float eps = 1e-5f;
  KokkosDNN::rmsnorm<Exec, Scalar>(x, w, eps, y);
  // Row variant too.
  Kokkos::View<Scalar*, Exec> xr("xr", D), yr("yr", D);
  {
    auto h = Kokkos::create_mirror_view(xr);
    for (int d = 0; d < D; ++d) h(d) = hx(0, d);
    Kokkos::deep_copy(xr, h);
  }
  KokkosDNN::rmsnorm_row<Exec, Scalar>(xr, w, eps, yr);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto gotr = Kokkos::create_mirror_view(yr);
  Kokkos::deep_copy(gotr, yr);
  for (int t = 0; t < T; ++t) {
    float ss = 0;
    for (int d = 0; d < D; ++d) {
      float v = static_cast<float>(hx(t, d));
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / D + eps);
    for (int d = 0; d < D; ++d) {
      float want =
          static_cast<float>(hx(t, d)) * inv * static_cast<float>(hw(d));
      EXPECT_NEAR(static_cast<float>(got(t, d)), want, tol)
          << "rmsnorm mismatch at (" << t << "," << d << ")";
    }
  }
  {
    float ss = 0;
    for (int d = 0; d < D; ++d) {
      float v = static_cast<float>(hx(0, d));
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / D + eps);
    for (int d = 0; d < D; ++d) {
      float want =
          static_cast<float>(hx(0, d)) * inv * static_cast<float>(hw(d));
      EXPECT_NEAR(static_cast<float>(gotr(d)), want, tol)
          << "rmsnorm_row mismatch at (" << d << ")";
    }
  }
}

TEST(RMSNorm, Float) { check_rmsnorm<float>(1e-5f); }

TEST(RMSNorm, BHalf) { check_rmsnorm<Kokkos::Experimental::bhalf_t>(5e-2f); }

template <class Scalar>
void check_rmsnorm_batched(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const float eps = 1e-5f;
  const int B = 2, T = 4, D = 8;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D),
      y("y", B, T, D);
  Kokkos::View<Scalar*, Exec> w("w", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d)
        hx(b, t, d) =
            static_cast<Scalar>(0.4f * (b + 1) - 0.2f * t + 0.1f * d);
  for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(1.0f + 0.1f * d);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  KokkosDNN::rmsnorm<Exec, Scalar>(x, w, eps, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double v = static_cast<double>(hx(b, t, d));
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d) {
        double want = static_cast<double>(hx(b, t, d)) * inv *
                      static_cast<double>(hw(d));
        EXPECT_NEAR(static_cast<double>(got(b, t, d)), want, tol)
            << "rmsnorm rank-3 mismatch at (" << b << "," << t << "," << d
            << ")";
      }
    }
}

TEST(RMSNormBatched, Float) { check_rmsnorm_batched<float>(1e-5f); }
TEST(RMSNormBatched, BHalf) {
  check_rmsnorm_batched<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// Gemma2 (1 + w) RMSNorm variant vs host reference (float + bfloat16).
// Zero weights must reproduce the scaleless variant (family consistency).
template <class Scalar>
void check_rmsnorm_one_plus(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 3, D = 8;
  const float eps = 1e-6f;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D),
      y_ns("y_ns", T, D);
  Kokkos::View<Scalar*, Exec> w("w", D), w0("w0", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hw0 = Kokkos::create_mirror_view(w0);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.5f * (t + 1) - 0.25f * d);
  for (int d = 0; d < D; ++d) {
    hw(d) = static_cast<Scalar>(0.05f * d - 0.1f);  // small, zero-centered
    hw0(d) = static_cast<Scalar>(0);
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(w0, hw0);
  KokkosDNN::rmsnorm_one_plus<Exec, Scalar>(x, w, eps, y);
  KokkosDNN::rmsnorm_one_plus<Exec, Scalar>(x, w0, eps, y_ns);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto got_ns = Kokkos::create_mirror_view(y_ns);
  Kokkos::deep_copy(got_ns, y_ns);
  for (int t = 0; t < T; ++t) {
    float ss = 0;
    for (int d = 0; d < D; ++d) {
      float v = static_cast<float>(hx(t, d));
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / D + eps);
    for (int d = 0; d < D; ++d) {
      float want = static_cast<float>(hx(t, d)) * inv *
                   (1.0f + static_cast<float>(hw(d)));
      EXPECT_NEAR(static_cast<float>(got(t, d)), want, tol)
          << "rmsnorm_one_plus mismatch at (" << t << "," << d << ")";
      float want_ns = static_cast<float>(hx(t, d)) * inv;
      EXPECT_NEAR(static_cast<float>(got_ns(t, d)), want_ns, tol)
          << "rmsnorm_one_plus zero-weight mismatch at (" << t << "," << d
          << ")";
    }
  }
  // Dimension mismatch throws.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad("bad", T, D - 1);
  EXPECT_THROW(
      (KokkosDNN::rmsnorm_one_plus<Exec, Scalar>(x, w, eps, bad)),
      std::invalid_argument);
}

TEST(RMSNormOnePlus, Float) { check_rmsnorm_one_plus<float>(1e-5f); }

TEST(RMSNormOnePlus, BHalf) {
  check_rmsnorm_one_plus<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// Fused (1+w)-RMSNorm + residual add: matches the host serial reference
// within existing rmsnorm tolerances, and the device two-pass
// rmsnorm_one_plus + h += tmp form within ~1 ulp (float; the fused add may
// FMA-contract, a single rounding instead of two) or exactly (bhalf, where
// the explicit narrow cast pins both roundings).
template <class Scalar>
void check_rmsnorm_one_plus_resid(float tol_host, float tol_twopass) {
  using Exec = Kokkos::DefaultExecutionSpace;
  for (int T : {1, 3}) {
    const int D = 8;
    const float eps = 1e-6f;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D),
        hf("hfused", T, D), tmp("tmp", T, D), hs("hseq", T, D);
    Kokkos::View<Scalar*, Exec> w("w", D);
    auto hx = Kokkos::create_mirror_view(x);
    auto hh = Kokkos::create_mirror_view(hf);
    auto hw = Kokkos::create_mirror_view(w);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d) {
        hx(t, d) = static_cast<Scalar>(0.5f * (t + 1) - 0.25f * d);
        hh(t, d) = static_cast<Scalar>(0.1f * d - 0.3f * t);
      }
    for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(0.05f * d - 0.1f);
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(w, hw);
    Kokkos::deep_copy(hf, hh);
    Kokkos::deep_copy(hs, hh);
    // Snapshot init values for the host reference: on host backends
    // (Serial) mirrors may alias the device views, which the fused kernel
    // below overwrites.
    std::vector<float> xref((size_t)T * (size_t)D),
        href((size_t)T * (size_t)D), wref((size_t)D);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d) {
        xref[(size_t)t * (size_t)D + (size_t)d] =
            static_cast<float>(hx(t, d));
        href[(size_t)t * (size_t)D + (size_t)d] =
            static_cast<float>(hh(t, d));
      }
    for (int d = 0; d < D; ++d) wref[(size_t)d] = static_cast<float>(hw(d));
    // Device two-pass reference.
    KokkosDNN::rmsnorm_one_plus<Exec, Scalar>(x, w, eps, tmp);
    Kokkos::parallel_for(
        "seq_add", Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>({0, 0},
                                                                {(size_t)T, (size_t)D}),
        KOKKOS_LAMBDA(int t, int d) { hs(t, d) += tmp(t, d); });
    // Fused.
    KokkosDNN::rmsnorm_one_plus_resid<Exec, Scalar>(x, w, eps, hf);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(hf);
    Kokkos::deep_copy(got, hf);
    auto want = Kokkos::create_mirror_view(hs);
    Kokkos::deep_copy(want, hs);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(static_cast<double>(got(t, d)),
                    static_cast<double>(want(t, d)), tol_twopass)
            << "rmsnorm_one_plus_resid mismatch at T=" << T << " (" << t
            << "," << d << ")";
    // Host serial reference (functionality, same tolerances as the
    // unfused rmsnorm_one_plus test). Uses the init snapshots (see above).
    for (int t = 0; t < T; ++t) {
      float ss = 0;
      for (int d = 0; d < D; ++d) {
        float v = xref[(size_t)t * (size_t)D + (size_t)d];
        ss += v * v;
      }
      float inv = 1.0f / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d) {
        float n =
            xref[(size_t)t * (size_t)D + (size_t)d] * inv * (1.0f + wref[d]);
        float want_host = href[(size_t)t * (size_t)D + (size_t)d] + n;
        EXPECT_NEAR(static_cast<double>(got(t, d)), want_host, tol_host)
            << "rmsnorm_one_plus_resid host mismatch at T=" << T << " (" << t
            << "," << d << ")";
      }
    }
  }
  // Dimension mismatch throws.
  const int D = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", 2, D),
      bad("bad", 2, D - 1);
  Kokkos::View<Scalar*, Exec> w("w", D);
  EXPECT_THROW(
      (KokkosDNN::rmsnorm_one_plus_resid<Exec, Scalar>(x, w, 1e-6f, bad)),
      std::invalid_argument);
}

TEST(RMSNormOnePlusResid, Float) {
  check_rmsnorm_one_plus_resid<float>(1e-5f, 1e-6f);
}

TEST(RMSNormOnePlusResid, BHalf) {
  check_rmsnorm_one_plus_resid<Kokkos::Experimental::bhalf_t>(5e-2f, 5e-2f);
}

// Fused Q/K per-head norms + V reshape: bit-identical to separate
// qk_rmsnorm calls plus a reshape copy.
template <class Scalar>
void check_qk_rmsnorm_qkv_fused() {
  using Exec = Kokkos::DefaultExecutionSpace;
  for (int T : {1, 3}) {
    const int NH = 4, NKV = 2, HD = 8;
    const float eps = 1e-6f;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> q2("q2", T, NH * HD),
        k2("k2", T, NKV * HD), v2("v2", T, NKV * HD);
    Kokkos::View<Scalar*, Exec> wq("wq", HD), wk("wk", HD);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> q3s("q3s", T, NH, HD),
        k3s("k3s", T, NKV, HD), v3s("v3s", T, NKV, HD), q3f("q3f", T, NH, HD),
        k3f("k3f", T, NKV, HD), v3f("v3f", T, NKV, HD);
    auto hq = Kokkos::create_mirror_view(q2);
    auto hk = Kokkos::create_mirror_view(k2);
    auto hv = Kokkos::create_mirror_view(v2);
    auto hwq = Kokkos::create_mirror_view(wq);
    auto hwk = Kokkos::create_mirror_view(wk);
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < NH * HD; ++i)
        hq(t, i) = static_cast<Scalar>(0.25f * ((t + 1) * (i % 7 + 1)) - 3.0f);
      for (int i = 0; i < NKV * HD; ++i) {
        hk(t, i) = static_cast<Scalar>(0.2f * ((t + 2) * (i % 5 + 1)) - 2.0f);
        hv(t, i) = static_cast<Scalar>(0.3f * (i - t) + 0.5f);
      }
    }
    for (int d = 0; d < HD; ++d) {
      hwq(d) = static_cast<Scalar>(1.0f + 0.05f * d);
      hwk(d) = static_cast<Scalar>(1.0f);
    }
    Kokkos::deep_copy(q2, hq);
    Kokkos::deep_copy(k2, hk);
    Kokkos::deep_copy(v2, hv);
    Kokkos::deep_copy(wq, hwq);
    Kokkos::deep_copy(wk, hwk);
    // Separate reference.
    KokkosDNN::qk_rmsnorm<Exec, Scalar>(q2, wq, eps, q3s);
    KokkosDNN::qk_rmsnorm<Exec, Scalar>(k2, wk, eps, k3s);
    Kokkos::parallel_for(
        "ref_reshape_v",
        Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>({0, 0}, {(size_t)T,
                                                              (size_t)NKV * HD}),
        KOKKOS_LAMBDA(int t, int i) { v3s(t, i / HD, i % HD) = v2(t, i); });
    // Fused.
    KokkosDNN::qk_rmsnorm_qkv_fused<Exec, Scalar>(q2, wq, q3f, k2, wk, k3f,
                                                 v2, v3f, eps);
    Kokkos::fence();
    auto gq = Kokkos::create_mirror_view(q3f);
    Kokkos::deep_copy(gq, q3f);
    auto wq3 = Kokkos::create_mirror_view(q3s);
    Kokkos::deep_copy(wq3, q3s);
    auto gk = Kokkos::create_mirror_view(k3f);
    Kokkos::deep_copy(gk, k3f);
    auto wk3 = Kokkos::create_mirror_view(k3s);
    Kokkos::deep_copy(wk3, k3s);
    auto gv = Kokkos::create_mirror_view(v3f);
    Kokkos::deep_copy(gv, v3f);
    auto wv3 = Kokkos::create_mirror_view(v3s);
    Kokkos::deep_copy(wv3, v3s);
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < NH; ++h)
        for (int d = 0; d < HD; ++d)
          EXPECT_EQ(gq(t, h, d), wq3(t, h, d))
              << "qkv_fused q mismatch at T=" << T << " (" << t << "," << h
              << "," << d << ")";
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < NKV; ++h)
        for (int d = 0; d < HD; ++d) {
          EXPECT_EQ(gk(t, h, d), wk3(t, h, d))
              << "qkv_fused k mismatch at T=" << T << " (" << t << "," << h
              << "," << d << ")";
          EXPECT_EQ(gv(t, h, d), wv3(t, h, d))
              << "qkv_fused v mismatch at T=" << T << " (" << t << "," << h
              << "," << d << ")";
        }
  }
  // Dimension mismatch throws (wrong head count in qout).
  const int HD = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> q2("q2", 2, 4 * HD),
      k2("k2", 2, 2 * HD), v2("v2", 2, 2 * HD);
  Kokkos::View<Scalar*, Exec> wq("wq", HD), wk("wk", HD);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> q3bad("q3bad", 2, 3, HD),
      q3("q3", 2, 4, HD), k3("k3", 2, 2, HD), v3("v3", 2, 2, HD);
  EXPECT_THROW((KokkosDNN::qk_rmsnorm_qkv_fused<Exec, Scalar>(
                   q2, wq, q3bad, k2, wk, k3, v2, v3, 1e-6f)),
               std::invalid_argument);
}

TEST(QkRmsNormQkvFused, Float) { check_qk_rmsnorm_qkv_fused<float>(); }

TEST(QkRmsNormQkvFused, BHalf) {
  check_qk_rmsnorm_qkv_fused<Kokkos::Experimental::bhalf_t>();
}

