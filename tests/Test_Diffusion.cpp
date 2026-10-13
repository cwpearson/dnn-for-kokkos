// Generic DiT diffusion ops vs host reference: AdaLN modulation
// (rank-2 + rank-3) and sinusoidal timestep embeddings.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>

#include "KokkosDNN.hpp"

namespace {

template <class Scalar>
void check_adaln_modulate(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  // Rank-2: x[T,D], shift[D], scale[D].
  const int T = 3, D = 8;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  Kokkos::View<Scalar*, Exec> shift("shift", D), scale("scale", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hsh = Kokkos::create_mirror_view(shift);
  auto hsc = Kokkos::create_mirror_view(scale);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.3f * t - 0.2f * d);
  for (int d = 0; d < D; ++d) {
    hsh(d) = static_cast<Scalar>(0.1f * d);
    hsc(d) = static_cast<Scalar>(-0.05f * d);
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(shift, hsh);
  Kokkos::deep_copy(scale, hsc);
  KokkosDNN::adaln_modulate_forward<Exec, Scalar>(x, shift, scale, y);
  // Rank-3: x[B,T,D], shift/scale[B,D].
  const int B = 2, T3 = 4;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x3("x3", B, T3, D),
      y3("y3", B, T3, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> sh3("sh3", B, D),
      sc3("sc3", B, D);
  auto hx3 = Kokkos::create_mirror_view(x3);
  auto hsh3 = Kokkos::create_mirror_view(sh3);
  auto hsc3 = Kokkos::create_mirror_view(sc3);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T3; ++t)
      for (int d = 0; d < D; ++d)
        hx3(b, t, d) =
            static_cast<Scalar>(0.2f * b - 0.15f * t + 0.1f * d);
  for (int b = 0; b < B; ++b)
    for (int d = 0; d < D; ++d) {
      hsh3(b, d) = static_cast<Scalar>(0.05f * b + 0.1f * d);
      hsc3(b, d) = static_cast<Scalar>(-0.03f * b + 0.02f * d);
    }
  Kokkos::deep_copy(x3, hx3);
  Kokkos::deep_copy(sh3, hsh3);
  Kokkos::deep_copy(sc3, hsc3);
  KokkosDNN::adaln_modulate_forward<Exec, Scalar>(x3, sh3, sc3, y3);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto got3 = Kokkos::create_mirror_view(y3);
  Kokkos::deep_copy(got3, y3);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) {
      double want = static_cast<double>(hx(t, d)) *
                        (1.0 + static_cast<double>(hsc(d))) +
                    static_cast<double>(hsh(d));
      EXPECT_NEAR(static_cast<double>(got(t, d)), want, tol)
          << "adaln rank-2 mismatch at (" << t << "," << d << ")";
    }
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T3; ++t)
      for (int d = 0; d < D; ++d) {
        double want = static_cast<double>(hx3(b, t, d)) *
                          (1.0 + static_cast<double>(hsc3(b, d))) +
                      static_cast<double>(hsh3(b, d));
        EXPECT_NEAR(static_cast<double>(got3(b, t, d)), want, tol)
            << "adaln rank-3 mismatch at (" << b << "," << t << "," << d
            << ")";
      }
}

void check_timestep_sinusoidal() {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int N = 4, F = 256;
  Kokkos::View<float*, Exec> t("t", N);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> out("out", N, F);
  auto ht = Kokkos::create_mirror_view(t);
  ht(0) = 0.0f;
  ht(1) = 1.0f;
  ht(2) = 10.0f;
  ht(3) = 999.0f;
  Kokkos::deep_copy(t, ht);
  KokkosDNN::timestep_sinusoidal<Exec>(t, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  const int H = F / 2;
  // Reference is double-precision while the kernel is FP32 by design; for
  // large t the float rounding of t*freq alone exceeds 1e-5, so use 5e-5.
  const double tol = 5e-5;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < H; ++j) {
      double freq =
          std::exp(-std::log(10000.0) * static_cast<double>(j) / H);
      double arg = static_cast<double>(ht(i)) * freq;
      EXPECT_NEAR(static_cast<double>(got(i, j)), std::cos(arg), tol)
          << "timestep cos mismatch at (" << i << "," << j << ")";
      EXPECT_NEAR(static_cast<double>(got(i, j + H)), std::sin(arg), tol)
          << "timestep sin mismatch at (" << i << "," << j << ")";
    }
}
}  // namespace

TEST(DiffusionAdaLNModulate, Float) { check_adaln_modulate<float>(1e-5f); }
TEST(DiffusionAdaLNModulate, BHalf) {
  check_adaln_modulate<Kokkos::Experimental::bhalf_t>(5e-2f);
}
TEST(DiffusionTimestep, Float) { check_timestep_sinusoidal(); }
