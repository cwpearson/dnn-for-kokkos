// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// SiLU forward vs double-precision host reference (float + bfloat16).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>

#include "KokkosDNN.hpp"

namespace {

template <class Scalar>
void check_silu(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 3, D = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  auto hx = Kokkos::create_mirror_view(x);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.5f * (t - 1) - 0.3f * d);
  Kokkos::deep_copy(x, hx);
  KokkosDNN::silu_forward<Exec, Scalar>(x, y);
  // Rank-1 path.
  Kokkos::View<Scalar*, Exec> xr("xr", D), yr("yr", D);
  {
    auto h = Kokkos::create_mirror_view(xr);
    for (int d = 0; d < D; ++d) h(d) = hx(0, d);
    Kokkos::deep_copy(xr, h);
  }
  KokkosDNN::silu_forward<Exec, Scalar>(xr, yr);
  // In-place safety.
  KokkosDNN::silu_forward<Exec, Scalar>(y, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto gotr = Kokkos::create_mirror_view(yr);
  Kokkos::deep_copy(gotr, yr);
  auto silu_ref = [](double v) { return v / (1.0 + std::exp(-v)); };
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) {
      double want = silu_ref(silu_ref(static_cast<double>(hx(t, d))));
      EXPECT_NEAR(static_cast<double>(got(t, d)), want, tol)
          << "silu mismatch at (" << t << "," << d << ")";
    }
  for (int d = 0; d < D; ++d) {
    double want = silu_ref(static_cast<double>(hx(0, d)));
    EXPECT_NEAR(static_cast<double>(gotr(d)), want, tol)
        << "silu rank-1 mismatch at (" << d << ")";
  }
}

}  // namespace

TEST(Silu, Float) { check_silu<float>(1e-5f); }
TEST(Silu, BHalf) { check_silu<Kokkos::Experimental::bhalf_t>(5e-2f); }
