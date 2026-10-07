// SwiGLU vs host reference (float + bfloat16).
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

template <class Scalar>
int run_swiglu(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 2, F = 16;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> g("g", T, F), u("u", T, F),
      o("o", T, F);
  auto hg = Kokkos::create_mirror_view(g);
  auto hu = Kokkos::create_mirror_view(u);
  for (int t = 0; t < T; ++t)
    for (int f = 0; f < F; ++f) {
      hg(t, f) = static_cast<Scalar>(0.2f * (t - f));
      hu(t, f) = static_cast<Scalar>(0.1f * (t + f + 1));
    }
  Kokkos::deep_copy(g, hg);
  Kokkos::deep_copy(u, hu);
  KokkosDNN::swiglu<Exec, Scalar>(g, u, o);
  // Greedy sampler smoke cover on the same scalar.
  Kokkos::View<Scalar*, Exec> logits("logits", 8);
  {
    auto hl = Kokkos::create_mirror_view(logits);
    for (int i = 0; i < 8; ++i) hl(i) = static_cast<Scalar>(0.1f * i);
    hl(3) = static_cast<Scalar>(5.0f);
    Kokkos::deep_copy(logits, hl);
  }
  if (KokkosDNN::sample_greedy<Exec, Scalar>(logits) != 3) {
    std::printf("FAIL sampler\n");
    return 1;
  }
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(o);
  Kokkos::deep_copy(got, o);
  for (int t = 0; t < T; ++t)
    for (int f = 0; f < F; ++f) {
      float gg = static_cast<float>(hg(t, f));
      float want = (gg / (1 + expf(-gg))) * static_cast<float>(hu(t, f));
      if (std::fabs(static_cast<float>(got(t, f)) - want) > tol) {
        std::printf("FAIL swiglu\n");
        return 1;
      }
    }
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (run_swiglu<float>(1e-5f)) return 1;
    if (run_swiglu<Kokkos::Experimental::bhalf_t>(5e-2f)) return 1;
    std::printf("PASS test_swiglu\n");
  }
  Kokkos::finalize();
  return 0;
}
