// SwiGLU vs host reference.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    const int T = 2, F = 16;
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> g("g", T, F), u("u", T, F),
        o("o", T, F);
    auto hg = Kokkos::create_mirror_view(g);
    auto hu = Kokkos::create_mirror_view(u);
    for (int t = 0; t < T; ++t)
      for (int f = 0; f < F; ++f) {
        hg(t, f) = 0.2f * (t - f);
        hu(t, f) = 0.1f * (t + f + 1);
      }
    Kokkos::deep_copy(g, hg);
    Kokkos::deep_copy(u, hu);
    KokkosDNN::swiglu<Exec>(g, u, o);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(o);
    Kokkos::deep_copy(got, o);
    for (int t = 0; t < T; ++t)
      for (int f = 0; f < F; ++f) {
        float gg = hg(t, f);
        float want = (gg / (1 + expf(-gg))) * hu(t, f);
        if (std::fabs(got(t, f) - want) > 1e-5f) {
          std::printf("FAIL swiglu\n");
          return 1;
        }
      }
    std::printf("PASS test_swiglu\n");
  }
  Kokkos::finalize();
  return 0;
}
