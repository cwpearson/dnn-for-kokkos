// RMSNorm vs host reference (float + bfloat16).
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

template <class Scalar>
int run_rmsnorm(float tol) {
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
      if (std::fabs(static_cast<float>(got(t, d)) - want) > tol) {
        std::printf("FAIL rmsnorm\n");
        return 1;
      }
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
      if (std::fabs(static_cast<float>(gotr(d)) - want) > tol) {
        std::printf("FAIL rmsnorm_row\n");
        return 1;
      }
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (run_rmsnorm<float>(1e-5f)) return 1;
    if (run_rmsnorm<Kokkos::Experimental::bhalf_t>(5e-2f)) return 1;
    std::printf("PASS test_rmsnorm\n");
  }
  Kokkos::finalize();
  return 0;
}
