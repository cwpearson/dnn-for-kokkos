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

#include "KokkosDNN_RMSNorm.hpp"

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
// rmsnorm_one_plus + h += tmp form within a few ulp (float; the team-parallel
// sum-of-squares reassociates and the fused add may FMA-contract) or within
// narrow-Scalar rounding (bhalf, where both forms round to Scalar).
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

// Fused resid-add + following norm: matches the sequential
// rmsnorm_one_plus_resid + rmsnorm_one_plus pair and the host serial
// reference. Distinct eps1/eps2 mirror the Glimmer use (tight post-norm eps
// into the regular eps).
template <class Scalar>
void check_rmsnorm_one_plus_resid_then_one_plus(float tol_host,
                                                float tol_twopass) {
  using Exec = Kokkos::DefaultExecutionSpace;
  for (int T : {1, 3}) {
    const int D = 8;
    const float eps1 = 1e-8f, eps2 = 1e-5f;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D),
        hf("hfused", T, D), yf("yfused", T, D), hs("hseq", T, D),
        ys("yseq", T, D);
    Kokkos::View<Scalar*, Exec> w1("w1", D), w2("w2", D);
    auto hx = Kokkos::create_mirror_view(x);
    auto hh = Kokkos::create_mirror_view(hf);
    auto hw1 = Kokkos::create_mirror_view(w1);
    auto hw2 = Kokkos::create_mirror_view(w2);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d) {
        hx(t, d) = static_cast<Scalar>(0.5f * (t + 1) - 0.25f * d);
        hh(t, d) = static_cast<Scalar>(0.1f * d - 0.3f * t);
      }
    for (int d = 0; d < D; ++d) {
      hw1(d) = static_cast<Scalar>(0.05f * d - 0.1f);
      hw2(d) = static_cast<Scalar>(-0.03f * d + 0.07f);
    }
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(w1, hw1);
    Kokkos::deep_copy(w2, hw2);
    Kokkos::deep_copy(hf, hh);
    Kokkos::deep_copy(hs, hh);
    // Snapshots for the host reference: on host backends (Serial) mirrors
    // may alias the device views, which the kernels below overwrite.
    std::vector<float> xref((size_t)T * (size_t)D),
        href((size_t)T * (size_t)D), w1ref((size_t)D), w2ref((size_t)D);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d) {
        xref[(size_t)t * (size_t)D + (size_t)d] =
            static_cast<float>(hx(t, d));
        href[(size_t)t * (size_t)D + (size_t)d] =
            static_cast<float>(hh(t, d));
      }
    for (int d = 0; d < D; ++d) {
      w1ref[(size_t)d] = static_cast<float>(hw1(d));
      w2ref[(size_t)d] = static_cast<float>(hw2(d));
    }
    // Sequential device reference.
    KokkosDNN::rmsnorm_one_plus_resid<Exec, Scalar>(x, w1, eps1, hs);
    KokkosDNN::rmsnorm_one_plus<Exec, Scalar>(hs, w2, eps2, ys);
    // Fused (distinct y).
    KokkosDNN::rmsnorm_one_plus_resid_then_one_plus<Exec, Scalar>(x, w1, eps1,
                                                                 hf, w2, eps2,
                                                                 yf);
    Kokkos::fence();
    auto got_h = Kokkos::create_mirror_view(hf);
    Kokkos::deep_copy(got_h, hf);
    auto want_h = Kokkos::create_mirror_view(hs);
    Kokkos::deep_copy(want_h, hs);
    auto got_y = Kokkos::create_mirror_view(yf);
    Kokkos::deep_copy(got_y, yf);
    auto want_y = Kokkos::create_mirror_view(ys);
    Kokkos::deep_copy(want_y, ys);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d) {
        EXPECT_NEAR(static_cast<double>(got_h(t, d)),
                    static_cast<double>(want_h(t, d)), tol_twopass)
            << "resid_then_norm h mismatch at T=" << T << " (" << t << ","
            << d << ")";
        EXPECT_NEAR(static_cast<double>(got_y(t, d)),
                    static_cast<double>(want_y(t, d)), tol_twopass)
            << "resid_then_norm y mismatch at T=" << T << " (" << t << ","
            << d << ")";
      }
    // Host serial reference (functionality).
    for (int t = 0; t < T; ++t) {
      float ss1 = 0;
      for (int d = 0; d < D; ++d) {
        float v = xref[(size_t)t * (size_t)D + (size_t)d];
        ss1 += v * v;
      }
      float inv1 = 1.0f / std::sqrt(ss1 / D + eps1);
      std::vector<float> hnew((size_t)D);
      for (int d = 0; d < D; ++d)
        hnew[(size_t)d] = href[(size_t)t * (size_t)D + (size_t)d] +
                          xref[(size_t)t * (size_t)D + (size_t)d] * inv1 *
                              (1.0f + w1ref[(size_t)d]);
      float ss2 = 0;
      for (int d = 0; d < D; ++d) ss2 += hnew[(size_t)d] * hnew[(size_t)d];
      float inv2 = 1.0f / std::sqrt(ss2 / D + eps2);
      for (int d = 0; d < D; ++d) {
        float want_h_host = hnew[(size_t)d];
        float want_y_host = hnew[(size_t)d] * inv2 * (1.0f + w2ref[(size_t)d]);
        EXPECT_NEAR(static_cast<double>(got_h(t, d)), want_h_host, tol_host)
            << "host h mismatch at T=" << T << " (" << t << "," << d << ")";
        EXPECT_NEAR(static_cast<double>(got_y(t, d)), want_y_host, tol_host)
            << "host y mismatch at T=" << T << " (" << t << "," << d << ")";
      }
    }
    // In-place y == h is supported (second-phase reduce completes before
    // any write): rerun fused aliasing y onto h and compare.
    Kokkos::deep_copy(hf, hh);
    KokkosDNN::rmsnorm_one_plus_resid_then_one_plus<Exec, Scalar>(x, w1, eps1,
                                                                 hf, w2, eps2,
                                                                 hf);
    Kokkos::fence();
    auto got_ip = Kokkos::create_mirror_view(hf);
    Kokkos::deep_copy(got_ip, hf);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(static_cast<double>(got_ip(t, d)),
                    static_cast<double>(want_y(t, d)), tol_twopass)
            << "in-place mismatch at T=" << T << " (" << t << "," << d << ")";
  }
  // Dimension mismatch throws.
  const int D = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", 2, D), h("h", 2, D),
      bad("bad", 2, D - 1);
  Kokkos::View<Scalar*, Exec> w("w", D);
  EXPECT_THROW((KokkosDNN::rmsnorm_one_plus_resid_then_one_plus<Exec, Scalar>(
                   x, w, 1e-8f, h, w, 1e-5f, bad)),
               std::invalid_argument);
}

TEST(RMSNormOnePlusResidThenOnePlus, Float) {
  check_rmsnorm_one_plus_resid_then_one_plus<float>(1e-5f, 1e-6f);
}

TEST(RMSNormOnePlusResidThenOnePlus, BHalf) {
  check_rmsnorm_one_plus_resid_then_one_plus<Kokkos::Experimental::bhalf_t>(
      5e-2f, 5e-2f);
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


// Fused RMSNorm + AdaLN-zero modulation (rank-2 and rank-3) vs host
// reference (float + bfloat16). Rank-2 conditions on [D], rank-3 on
// per-sample [B, D] rows broadcast over tokens.
template <class Scalar>
void check_rmsnorm_modulate(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const float eps = 1e-6f;
  // Rank-2.
  {
    const int T = 3, D = 8;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D),
        y("y", T, D);
    Kokkos::View<Scalar*, Exec> w("w", D), shift("sh", D), scale("sc", D);
    auto hx = Kokkos::create_mirror_view(x);
    auto hw = Kokkos::create_mirror_view(w);
    auto hsh = Kokkos::create_mirror_view(shift);
    auto hsc = Kokkos::create_mirror_view(scale);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d)
        hx(t, d) = static_cast<Scalar>(0.4f * (t + 1) - 0.2f * d);
    for (int d = 0; d < D; ++d) {
      hw(d) = static_cast<Scalar>(1.0f + 0.05f * d);
      hsh(d) = static_cast<Scalar>(0.1f * d);
      hsc(d) = static_cast<Scalar>(-0.02f * d);
    }
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(w, hw);
    Kokkos::deep_copy(shift, hsh);
    Kokkos::deep_copy(scale, hsc);
    KokkosDNN::rmsnorm_modulate<Exec, Scalar>(x, w, eps, shift, scale, y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(got, y);
    for (int t = 0; t < T; ++t) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double v = static_cast<double>(hx(t, d));
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d) {
        double n =
            static_cast<double>(hx(t, d)) * inv * static_cast<double>(hw(d));
        double want = n * (1.0 + static_cast<double>(hsc(d))) +
                      static_cast<double>(hsh(d));
        EXPECT_NEAR(static_cast<double>(got(t, d)), want, tol)
            << "rank-2 mismatch at (" << t << "," << d << ")";
      }
    }
  }
  // Rank-3 batched.
  {
    const int B = 2, T = 3, D = 8;
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D),
        y("y", B, T, D);
    Kokkos::View<Scalar*, Exec> w("w", D);
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> shift("sh", B, D),
        scale("sc", B, D);
    auto hx = Kokkos::create_mirror_view(x);
    auto hw = Kokkos::create_mirror_view(w);
    auto hsh = Kokkos::create_mirror_view(shift);
    auto hsc = Kokkos::create_mirror_view(scale);
    for (int b = 0; b < B; ++b)
      for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
          hx(b, t, d) =
              static_cast<Scalar>(0.2f * b - 0.15f * t + 0.1f * d - 0.3f);
    for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(1.0f + 0.05f * d);
    for (int b = 0; b < B; ++b)
      for (int d = 0; d < D; ++d) {
        hsh(b, d) = static_cast<Scalar>(0.05f * b + 0.1f * d);
        hsc(b, d) = static_cast<Scalar>(-0.03f * b + 0.02f * d);
      }
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(w, hw);
    Kokkos::deep_copy(shift, hsh);
    Kokkos::deep_copy(scale, hsc);
    // Snapshots: on host-space backends (Serial) mirrors may alias the
    // device views, which the in-place call below overwrites.
    std::vector<float> mxref((size_t)B * T * D), mwref((size_t)D),
        mshref((size_t)B * D), mscref((size_t)B * D);
    for (int b = 0; b < B; ++b)
      for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
          mxref[((size_t)b * T + t) * D + d] =
              static_cast<float>(hx(b, t, d));
    for (int d = 0; d < D; ++d) mwref[(size_t)d] = static_cast<float>(hw(d));
    for (int b = 0; b < B; ++b)
      for (int d = 0; d < D; ++d) {
        mshref[(size_t)b * D + d] = static_cast<float>(hsh(b, d));
        mscref[(size_t)b * D + d] = static_cast<float>(hsc(b, d));
      }
    KokkosDNN::rmsnorm_modulate<Exec, Scalar>(x, w, eps, shift, scale, y);
    // In-place safe: x may alias y.
    KokkosDNN::rmsnorm_modulate<Exec, Scalar>(x, w, eps, shift, scale, x);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(got, y);
    auto got_ip = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(got_ip, x);
    for (int b = 0; b < B; ++b)
      for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int d = 0; d < D; ++d) {
          double v = mxref[((size_t)b * T + t) * D + d];
          ss += v * v;
        }
        double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int d = 0; d < D; ++d) {
          double n = mxref[((size_t)b * T + t) * D + d] * inv *
                     mwref[(size_t)d];
          double want = n * (1.0 + mscref[(size_t)b * D + d]) +
                        mshref[(size_t)b * D + d];
          EXPECT_NEAR(static_cast<double>(got(b, t, d)), want, tol)
              << "rank-3 mismatch at (" << b << "," << t << "," << d << ")";
          EXPECT_NEAR(static_cast<double>(got_ip(b, t, d)), want, tol)
              << "rank-3 in-place mismatch at (" << b << "," << t << ","
              << d << ")";
        }
      }
  }
  // Dimension mismatch throws (rank-2 and rank-3).
  {
    const int D = 8;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", 2, D),
        bad("bad", 2, D - 1);
    Kokkos::View<Scalar*, Exec> w("w", D), c("c", D);
    EXPECT_THROW(
        (KokkosDNN::rmsnorm_modulate<Exec, Scalar>(x, w, eps, c, c, bad)),
        std::invalid_argument);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x3("x3", 1, 2, D),
        bad3("bad3", 1, 2, D - 1);
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> c2("c2", 1, D);
    EXPECT_THROW(
        (KokkosDNN::rmsnorm_modulate<Exec, Scalar>(x3, w, eps, c2, c2, bad3)),
        std::invalid_argument);
  }
}

TEST(RMSNormModulate, Float) { check_rmsnorm_modulate<float>(1e-5f); }

TEST(RMSNormModulate, BHalf) {
  check_rmsnorm_modulate<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// Batched (rank-4) per-head QK RMSNorm vs host reference.
template <class Scalar>
void check_qk_rmsnorm4d(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int B = 2, S = 3, H = 2, D = 8;
  const float eps = 1e-6f;
  Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> q("q", B, S, H, D),
      out("o", B, S, H, D);
  Kokkos::View<Scalar*, Exec> w("w", D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hw = Kokkos::create_mirror_view(w);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d)
          hq(b, s, h, d) = static_cast<Scalar>(0.13f * (b + 1) -
                                               0.07f * (s + 1) +
                                               0.11f * (h + 1) + 0.05f * d);
  for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(0.9f + 0.03f * d);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(w, hw);
  // Snapshot: on host-space backends mirrors may alias the device views,
  // which the in-place call below overwrites.
  std::vector<float> q4ref((size_t)B * S * H * D), w4ref((size_t)D);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d)
          q4ref[(((size_t)b * S + s) * H + h) * D + d] =
              static_cast<float>(hq(b, s, h, d));
  for (int d = 0; d < D; ++d) w4ref[(size_t)d] = static_cast<float>(hw(d));
  KokkosDNN::qk_rmsnorm4d<Exec, Scalar>(q, w, eps, out);
  // In-place safe.
  KokkosDNN::qk_rmsnorm4d<Exec, Scalar>(q, w, eps, q);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  auto got_ip = Kokkos::create_mirror_view(q);
  Kokkos::deep_copy(got_ip, q);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h) {
        double ss = 0;
        for (int d = 0; d < D; ++d) {
          double v = q4ref[(((size_t)b * S + s) * H + h) * D + d];
          ss += v * v;
        }
        double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int d = 0; d < D; ++d) {
          double want = q4ref[(((size_t)b * S + s) * H + h) * D + d] * inv *
                        w4ref[(size_t)d];
          EXPECT_NEAR(static_cast<double>(got(b, s, h, d)), want, tol)
              << "mismatch at (" << b << "," << s << "," << h << "," << d
              << ")";
          EXPECT_NEAR(static_cast<double>(got_ip(b, s, h, d)), want, tol)
              << "in-place mismatch at (" << b << "," << s << "," << h
              << "," << d << ")";
        }
      }
  // Unmanaged zero-copy reshapes (the DiT alias pattern) bind directly.
  {
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> flat("flat", S, H * D);
    auto hflat = Kokkos::create_mirror_view(flat);
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d)
          hflat(s, h * D + d) =
              static_cast<Scalar>(q4ref[((size_t)s * H + h) * D + d]);
    Kokkos::deep_copy(flat, hflat);
    Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        alias(flat.data(), 1, S, H, D);
    KokkosDNN::qk_rmsnorm4d<Exec, Scalar>(alias, w, eps, alias);
    Kokkos::fence();
    auto gflat = Kokkos::create_mirror_view(flat);
    Kokkos::deep_copy(gflat, flat);
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d)
          EXPECT_NEAR(static_cast<double>(gflat(s, h * D + d)),
                      static_cast<double>(got(0, s, h, d)), tol)
              << "unmanaged mismatch at (" << s << "," << h << "," << d
              << ")";
  }
  // Dimension mismatch throws.
  Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> bad("bad", B, S, H,
                                                          D - 1);
  EXPECT_THROW((KokkosDNN::qk_rmsnorm4d<Exec, Scalar>(q, w, eps, bad)),
               std::invalid_argument);
}

TEST(QkRmsNorm4d, Float) { check_qk_rmsnorm4d<float>(1e-5f); }

TEST(QkRmsNorm4d, BHalf) {
  check_qk_rmsnorm4d<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// Per-head QK RMSNorm with the (1+w) convention vs host reference.
template <class Scalar>
void check_qk_rmsnorm_oneplus_3d(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 3, H = 2, D = 4;
  const float eps = 1e-6f;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", T, H, D),
      y("y", T, H, D);
  Kokkos::View<Scalar*, Exec> w("w", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d)
        hx(t, h, d) = static_cast<Scalar>(0.3f * (t + 1) - 0.2f * (h + 1) +
                                          0.1f * (d + 1));
  for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(0.05f * (d - 1));
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  // Snapshot: on host-space backends mirrors may alias the device views,
  // which the in-place call below overwrites.
  std::vector<float> opref((size_t)T * H * D), owref((size_t)D);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d)
        opref[((size_t)t * H + h) * D + d] =
            static_cast<float>(hx(t, h, d));
  for (int d = 0; d < D; ++d) owref[(size_t)d] = static_cast<float>(hw(d));
  KokkosDNN::qk_rmsnorm_oneplus_3d<Exec, Scalar>(x, w, eps, y);
  // In-place safe.
  KokkosDNN::qk_rmsnorm_oneplus_3d<Exec, Scalar>(x, w, eps, x);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto got_ip = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(got_ip, x);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double v = opref[((size_t)t * H + h) * D + d];
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d) {
        double want = opref[((size_t)t * H + h) * D + d] * inv *
                      (1.0 + owref[(size_t)d]);
        EXPECT_NEAR(static_cast<double>(got(t, h, d)), want, tol)
            << t << h << d;
        EXPECT_NEAR(static_cast<double>(got_ip(t, h, d)), want, tol)
            << "in-place " << t << h << d;
      }
    }
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad("bad", T, H, D - 1);
  EXPECT_THROW(
      (KokkosDNN::qk_rmsnorm_oneplus_3d<Exec, Scalar>(x, w, eps, bad)),
      std::invalid_argument);
}

TEST(QkRmsNormOnePlus3d, Float) { check_qk_rmsnorm_oneplus_3d<float>(1e-5f); }

TEST(QkRmsNormOnePlus3d, BHalf) {
  check_qk_rmsnorm_oneplus_3d<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// Fused gated RMSNorm (silu gate) vs host reference.
template <class Scalar>
void check_rmsnorm_gated_3d(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 3, H = 2, D = 4;
  const float eps = 1e-6f;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> c("c", T, H, D),
      z("z", T, H, D), o("o", T, H, D);
  Kokkos::View<Scalar*, Exec> w("w", D);
  auto hc = Kokkos::create_mirror_view(c);
  auto hz = Kokkos::create_mirror_view(z);
  auto hw = Kokkos::create_mirror_view(w);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d) {
        hc(t, h, d) = static_cast<Scalar>(0.2f * (t + 1) - 0.1f * (d + h));
        hz(t, h, d) = static_cast<Scalar>(0.15f * (d + 1) + 0.05f * (t - h));
      }
  for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(1.0f + 0.02f * d);
  Kokkos::deep_copy(c, hc);
  Kokkos::deep_copy(z, hz);
  Kokkos::deep_copy(w, hw);
  KokkosDNN::rmsnorm_gated_3d<Exec, Scalar>(c, z, w, eps, o);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(o);
  Kokkos::deep_copy(got, o);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double x = static_cast<double>(hc(t, h, d));
        ss += x * x;
      }
      double inv = 1.0 / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d) {
        double zv = static_cast<double>(hz(t, h, d));
        double silu = zv / (1.0 + std::exp(-zv));
        double want = static_cast<double>(hc(t, h, d)) * inv *
                      static_cast<double>(hw(d)) * silu;
        EXPECT_NEAR(static_cast<double>(got(t, h, d)), want, tol)
            << t << h << d;
      }
    }
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad("bad", T, H, D - 1);
  EXPECT_THROW(
      (KokkosDNN::rmsnorm_gated_3d<Exec, Scalar>(c, z, w, eps, bad)),
      std::invalid_argument);
}

TEST(RMSNormGated3d, Float) { check_rmsnorm_gated_3d<float>(1e-5f); }

TEST(RMSNormGated3d, BHalf) {
  check_rmsnorm_gated_3d<Kokkos::Experimental::bhalf_t>(5e-2f);
}
