// Softmax rows sum to 1 and match host reference (float + bfloat16).
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

template <class Scalar>
int run_softmax(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int R = 4, C = 6;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", R, C), y("y", R, C);
  auto hx = Kokkos::create_mirror_view(x);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      hx(r, c) = static_cast<Scalar>(0.3f * (r - c) + 0.1f * c * c);
  Kokkos::deep_copy(x, hx);
  KokkosDNN::softmax_rows<Exec, Scalar>(x, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (int r = 0; r < R; ++r) {
    float m = static_cast<float>(hx(r, 0));
    for (int c = 1; c < C; ++c)
      m = fmaxf(m, static_cast<float>(hx(r, c)));
    float s = 0;
    for (int c = 0; c < C; ++c)
      s += expf(static_cast<float>(hx(r, c)) - m);
    float sum = 0;
    for (int c = 0; c < C; ++c) {
      float want = expf(static_cast<float>(hx(r, c)) - m) / s;
      sum += static_cast<float>(got(r, c));
      if (std::fabs(static_cast<float>(got(r, c)) - want) > tol) {
        std::printf("FAIL softmax\n");
        return 1;
      }
    }
    if (std::fabs(sum - 1.0f) > tol) {
      std::printf("FAIL softmax sum\n");
      return 1;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (run_softmax<float>(1e-5f)) return 1;
    if (run_softmax<Kokkos::Experimental::bhalf_t>(5e-2f)) return 1;
    std::printf("PASS test_softmax\n");
  }
  Kokkos::finalize();
  return 0;
}
