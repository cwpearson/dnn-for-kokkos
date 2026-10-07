// RMSNorm vs host reference.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    const int T = 3, D = 8;
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
    Kokkos::View<float*, Exec> w("w", D);
    auto hx = Kokkos::create_mirror_view(x);
    auto hw = Kokkos::create_mirror_view(w);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d) hx(t, d) = 0.5f * (t + 1) - 0.25f * d;
    for (int d = 0; d < D; ++d) hw(d) = 1.0f + 0.1f * d;
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(w, hw);
    const float eps = 1e-5f;
    KokkosDNN::rmsnorm<Exec>(x, w, eps, y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(got, y);
    for (int t = 0; t < T; ++t) {
      float ss = 0;
      for (int d = 0; d < D; ++d) ss += hx(t, d) * hx(t, d);
      float inv = 1.0f / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d) {
        float want = hx(t, d) * inv * hw(d);
        if (std::fabs(got(t, d) - want) > 1e-5f) {
          std::printf("FAIL rmsnorm\n");
          return 1;
        }
      }
    }
    std::printf("PASS test_rmsnorm\n");
  }
  Kokkos::finalize();
  return 0;
}
