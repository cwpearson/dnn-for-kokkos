// Softmax rows sum to 1 and match host reference.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    const int R = 4, C = 6;
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", R, C), y("y", R, C);
    auto hx = Kokkos::create_mirror_view(x);
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) hx(r, c) = 0.3f * (r - c) + 0.1f * c * c;
    Kokkos::deep_copy(x, hx);
    KokkosDNN::softmax_rows<Exec>(x, y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(got, y);
    for (int r = 0; r < R; ++r) {
      float m = hx(r, 0);
      for (int c = 1; c < C; ++c) m = fmaxf(m, hx(r, c));
      float s = 0;
      for (int c = 0; c < C; ++c) s += expf(hx(r, c) - m);
      float sum = 0;
      for (int c = 0; c < C; ++c) {
        float want = expf(hx(r, c) - m) / s;
        sum += got(r, c);
        if (std::fabs(got(r, c) - want) > 1e-5f) {
          std::printf("FAIL softmax\n");
          return 1;
        }
      }
      if (std::fabs(sum - 1.0f) > 1e-5f) {
        std::printf("FAIL softmax sum\n");
        return 1;
      }
    }
    std::printf("PASS test_softmax\n");
  }
  Kokkos::finalize();
  return 0;
}
