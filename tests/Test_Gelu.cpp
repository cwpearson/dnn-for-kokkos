// GELU forward vs host reference (exact erf + tanh approximation).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "KokkosDNN.hpp"

namespace {

// Double-precision host references (HF ACT2FN definitions).
double gelu_exact_ref(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}
double gelu_tanh_ref(double x) {
  static const double c = 0.7978845608028654;  // sqrt(2/pi)
  return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
}

const std::vector<float>& probe_values() {
  static const std::vector<float> v = {0.0f,  1.0f,  -1.0f, 0.5f, -0.5f,
                                       0.1f,  -0.1f, 2.0f,  -2.0f, 3.0f,
                                       -3.0f, 5.0f,  -5.0f};
  return v;
}

}  // namespace

// Spot values computed independently (Python `math.erf`/`math.tanh`):
//   exact(+1)=0.8413447461  tanh(+1)=0.8411919906
//   exact(-1)=-0.1586552539 tanh(-1)=-0.1588080094
TEST(Gelu, KnownScalars) {
  EXPECT_NEAR(gelu_exact_ref(0.0), 0.0, 1e-12);
  EXPECT_NEAR(gelu_tanh_ref(0.0), 0.0, 1e-12);
  EXPECT_NEAR(gelu_exact_ref(1.0), 0.8413447461, 1e-9);
  EXPECT_NEAR(gelu_tanh_ref(1.0), 0.8411919906, 1e-9);
  EXPECT_NEAR(gelu_exact_ref(-1.0), -0.1586552539, 1e-9);
  EXPECT_NEAR(gelu_tanh_ref(-1.0), -0.1588080094, 1e-9);
  // The two variants must be distinguishable here: guards against
  // accidentally swapping the erf/tanh implementations.
  EXPECT_GT(std::abs(gelu_exact_ref(1.0) - gelu_tanh_ref(1.0)), 1e-4);
  EXPECT_GT(std::abs(gelu_exact_ref(-3.0) - gelu_tanh_ref(-3.0)), 1e-4);
}

template <class Scalar>
void check_gelu_2d(KokkosDNN::GeluVariant variant, double tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto& vals = probe_values();
  const size_t R = 2, C = vals.size();
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", R, C), y("y", R, C);
  auto hx = Kokkos::create_mirror_view(x);
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      hx(i, j) = static_cast<Scalar>(vals[j] * (i == 0 ? 1.0f : -1.0f));
  Kokkos::deep_copy(x, hx);
  KokkosDNN::gelu_forward<Exec, Scalar>(x, y, variant);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j) {
      double xin = static_cast<double>(vals[j] * (i == 0 ? 1.0f : -1.0f));
      double want = variant == KokkosDNN::GeluVariant::Tanh
                        ? gelu_tanh_ref(xin)
                        : gelu_exact_ref(xin);
      EXPECT_NEAR(static_cast<double>(got(i, j)), want, tol)
          << "gelu_2d mismatch at (" << i << "," << j << ") x=" << xin;
    }
}

TEST(Gelu, ExactErfFloat2D) {
  check_gelu_2d<float>(KokkosDNN::GeluVariant::ExactErf, 1e-6);
}

TEST(Gelu, TanhFloat2D) {
  check_gelu_2d<float>(KokkosDNN::GeluVariant::Tanh, 1e-6);
}

// Default variant must be the exact erf form (HF ACT2FN["gelu"]).
TEST(Gelu, DefaultIsExactErf) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto& vals = probe_values();
  const size_t R = 1, C = vals.size();
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", R, C), y("y", R, C);
  auto hx = Kokkos::create_mirror_view(x);
  for (size_t j = 0; j < C; ++j) hx(0, j) = vals[j];
  Kokkos::deep_copy(x, hx);
  KokkosDNN::gelu_forward<Exec, float>(x, y);  // no variant -> ExactErf
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (size_t j = 0; j < C; ++j)
    EXPECT_NEAR(got(0, j), gelu_exact_ref(vals[j]), 1e-6)
        << "default-variant mismatch at x=" << vals[j];
}

TEST(Gelu, InPlaceAndRank1) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto& vals = probe_values();
  const size_t N = vals.size();
  // Rank-1 out-of-place, tanh variant.
  Kokkos::View<float*, Exec> x("x", N), y("y", N);
  {
    auto h = Kokkos::create_mirror_view(x);
    for (size_t i = 0; i < N; ++i) h(i) = vals[i];
    Kokkos::deep_copy(x, h);
  }
  KokkosDNN::gelu_forward<Exec, float>(x, y, KokkosDNN::GeluVariant::Tanh);
  // Rank-2 in-place, exact variant.
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> z("z", 2, N);
  {
    auto h = Kokkos::create_mirror_view(z);
    for (size_t j = 0; j < N; ++j) {
      h(0, j) = vals[j];
      h(1, j) = -vals[j];
    }
    Kokkos::deep_copy(z, h);
  }
  KokkosDNN::gelu_forward<Exec, float>(z, z);  // in-place, default erf
  Kokkos::fence();
  auto got1 = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got1, y);
  auto got2 = Kokkos::create_mirror_view(z);
  Kokkos::deep_copy(got2, z);
  for (size_t i = 0; i < N; ++i)
    EXPECT_NEAR(got1(i), gelu_tanh_ref(vals[i]), 1e-6)
        << "rank-1 tanh mismatch at x=" << vals[i];
  for (size_t j = 0; j < N; ++j) {
    EXPECT_NEAR(got2(0, j), gelu_exact_ref(vals[j]), 1e-6)
        << "in-place mismatch at x=" << vals[j];
    EXPECT_NEAR(got2(1, j), gelu_exact_ref(-vals[j]), 1e-6)
        << "in-place mismatch at x=" << -vals[j];
  }
}

TEST(Gelu, BHalf) {
  using Scalar = Kokkos::Experimental::bhalf_t;
  check_gelu_2d<Scalar>(KokkosDNN::GeluVariant::ExactErf, 5e-2);
  check_gelu_2d<Scalar>(KokkosDNN::GeluVariant::Tanh, 5e-2);
}

TEST(Gelu, DimensionMismatchThrows) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", 2, 8), y("y", 2, 8),
      bad("bad", 2, 7);
  Kokkos::View<float*, Exec> xr("xr", 8), yr("yr", 8), badr("badr", 7);
  EXPECT_THROW(
      (KokkosDNN::gelu_forward<Exec, float>(x, bad)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::gelu_forward<Exec, float>(xr, badr)),
      std::invalid_argument);
  EXPECT_NO_THROW((KokkosDNN::gelu_forward<Exec, float>(x, y)));
  EXPECT_NO_THROW((KokkosDNN::gelu_forward<Exec, float>(xr, yr)));
}
