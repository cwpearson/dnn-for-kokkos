// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// LayerNorm vs host reference (float + bfloat16).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <stdexcept>

#include "KokkosDNN.hpp"

template <class Scalar>
void check_layernorm(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 3, D = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  Kokkos::View<Scalar*, Exec> w("w", D), b("b", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hb = Kokkos::create_mirror_view(b);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.5f * (t + 1) - 0.25f * d);
  for (int d = 0; d < D; ++d) {
    hw(d) = static_cast<Scalar>(1.0f + 0.1f * d);
    hb(d) = static_cast<Scalar>(-0.5f + 0.05f * d);
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  const float eps = 1e-5f;
  KokkosDNN::layernorm_forward<Exec, Scalar>(x, w, b, eps, y);
  // Row variant too.
  Kokkos::View<Scalar*, Exec> xr("xr", D), yr("yr", D);
  {
    auto h = Kokkos::create_mirror_view(xr);
    for (int d = 0; d < D; ++d) h(d) = hx(0, d);
    Kokkos::deep_copy(xr, h);
  }
  KokkosDNN::layernorm_row<Exec, Scalar>(xr, w, b, eps, yr);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto gotr = Kokkos::create_mirror_view(yr);
  Kokkos::deep_copy(gotr, yr);
  auto ref = [&](int t, int d, float& want) {
    float mean = 0;
    for (int k = 0; k < D; ++k) mean += static_cast<float>(hx(t, k));
    mean /= D;
    float var = 0;
    for (int k = 0; k < D; ++k) {
      float c = static_cast<float>(hx(t, k)) - mean;
      var += c * c;
    }
    var /= D;
    float inv = 1.0f / std::sqrt(var + eps);
    want = (static_cast<float>(hx(t, d)) - mean) * inv *
               static_cast<float>(hw(d)) +
           static_cast<float>(hb(d));
  };
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) {
      float want = 0;
      ref(t, d, want);
      EXPECT_NEAR(static_cast<float>(got(t, d)), want, tol)
          << "layernorm mismatch at (" << t << "," << d << ")";
    }
  for (int d = 0; d < D; ++d) {
    float want = 0;
    ref(0, d, want);
    EXPECT_NEAR(static_cast<float>(gotr(d)), want, tol)
        << "layernorm_row mismatch at (" << d << ")";
  }
}

TEST(LayerNorm, Float) { check_layernorm<float>(1e-5f); }

TEST(LayerNorm, BHalf) { check_layernorm<Kokkos::Experimental::bhalf_t>(5e-2f); }

// Zero-variance row: normalized term vanishes, output must equal bias.
TEST(LayerNorm, ConstantRowEqualsBias) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 2, D = 8;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  Kokkos::View<float*, Exec> w("w", D), b("b", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hb = Kokkos::create_mirror_view(b);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) hx(t, d) = 3.25f;
  for (int d = 0; d < D; ++d) {
    hw(d) = 1.0f + 0.1f * d;
    hb(d) = -0.5f + 0.05f * d;
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::layernorm_forward<Exec, float>(x, w, b, 1e-5f, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      EXPECT_NEAR(got(t, d), static_cast<float>(hb(d)), 1e-5f)
          << "constant-row mismatch at (" << t << "," << d << ")";
}

TEST(LayerNorm, DimensionMismatchThrows) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", 2, 8), y("y", 2, 8);
  Kokkos::View<float*, Exec> w("w", 8), b("b", 8), bad("bad", 7);
  Kokkos::View<float*, Exec> xr("xr", 8), yr("yr", 8);
  EXPECT_THROW((KokkosDNN::layernorm_forward<Exec, float>(x, bad, b, 1e-5f, y)),
               std::invalid_argument);
  EXPECT_THROW((KokkosDNN::layernorm_forward<Exec, float>(x, w, bad, 1e-5f, y)),
               std::invalid_argument);
  EXPECT_THROW((KokkosDNN::layernorm_row<Exec, float>(xr, bad, b, 1e-5f, yr)),
               std::invalid_argument);
}

template <class Scalar>
void check_layernorm_noaffine(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const float eps = 1e-5f;
  const int T = 3, D = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  auto hx = Kokkos::create_mirror_view(x);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.5f * (t + 1) - 0.25f * d);
  Kokkos::deep_copy(x, hx);
  KokkosDNN::layernorm_noaffine_forward<Exec, Scalar>(x, eps, y);
  const int B = 2, T3 = 4;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x3("x3", B, T3, D),
      y3("y3", B, T3, D);
  auto hx3 = Kokkos::create_mirror_view(x3);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T3; ++t)
      for (int d = 0; d < D; ++d)
        hx3(b, t, d) =
            static_cast<Scalar>(0.4f * (b + 1) - 0.2f * t + 0.1f * d);
  Kokkos::deep_copy(x3, hx3);
  KokkosDNN::layernorm_noaffine_forward<Exec, Scalar>(x3, eps, y3);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto got3 = Kokkos::create_mirror_view(y3);
  Kokkos::deep_copy(got3, y3);
  for (int t = 0; t < T; ++t) {
    double mean = 0;
    for (int d = 0; d < D; ++d) mean += static_cast<double>(hx(t, d));
    mean /= D;
    double var = 0;
    for (int d = 0; d < D; ++d) {
      double c = static_cast<double>(hx(t, d)) - mean;
      var += c * c;
    }
    var /= D;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int d = 0; d < D; ++d) {
      double want = (static_cast<double>(hx(t, d)) - mean) * inv;
      EXPECT_NEAR(static_cast<double>(got(t, d)), want, tol)
          << "layernorm_noaffine rank-2 mismatch at (" << t << "," << d
          << ")";
    }
  }
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T3; ++t) {
      double mean = 0;
      for (int d = 0; d < D; ++d) mean += static_cast<double>(hx3(b, t, d));
      mean /= D;
      double var = 0;
      for (int d = 0; d < D; ++d) {
        double c = static_cast<double>(hx3(b, t, d)) - mean;
        var += c * c;
      }
      var /= D;
      double inv = 1.0 / std::sqrt(var + eps);
      for (int d = 0; d < D; ++d) {
        double want = (static_cast<double>(hx3(b, t, d)) - mean) * inv;
        EXPECT_NEAR(static_cast<double>(got3(b, t, d)), want, tol)
            << "layernorm_noaffine rank-3 mismatch at (" << b << "," << t
            << "," << d << ")";
      }
    }
}

TEST(LayerNormNoAffine, Float) { check_layernorm_noaffine<float>(1e-5f); }
TEST(LayerNormNoAffine, BHalf) {
  check_layernorm_noaffine<Kokkos::Experimental::bhalf_t>(5e-2f);
}
