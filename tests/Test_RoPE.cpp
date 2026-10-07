// RoPE preserves per-head norm (rotation is orthogonal).
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    const int T = 2, H = 2, D = 8;
    Kokkos::View<float***, Exec> q("q", T, H, D), k("k", T, H, D);
    auto hq = Kokkos::create_mirror_view(q);
    auto hk = Kokkos::create_mirror_view(k);
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) {
          hq(t, h, d) = 0.1f * (t * 100 + h * 10 + d + 1);
          hk(t, h, d) = -0.05f * (t * 100 + h * 10 + d + 2);
        }
    Kokkos::deep_copy(q, hq);
    Kokkos::deep_copy(k, hk);
    auto inv = KokkosDNN::rope_inv_freq<Exec>(D, 500000.0f);
    Kokkos::View<int*, Exec> pos("pos", T);
    {
      auto hp = Kokkos::create_mirror_view(pos);
      hp(0) = 0;
      hp(1) = 17;
      Kokkos::deep_copy(pos, hp);
    }
    KokkosDNN::rope_neoX_inplace<Exec>(q, k, inv, D, pos);
    Kokkos::fence();
    auto gq = Kokkos::create_mirror_view(q);
    auto gk = Kokkos::create_mirror_view(k);
    Kokkos::deep_copy(gq, q);
    Kokkos::deep_copy(gk, k);
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < H; ++h) {
        float n0 = 0, n1 = 0, m0 = 0, m1 = 0;
        for (int d = 0; d < D; ++d) {
          n0 += hq(t, h, d) * hq(t, h, d);
          n1 += gq(t, h, d) * gq(t, h, d);
          m0 += hk(t, h, d) * hk(t, h, d);
          m1 += gk(t, h, d) * gk(t, h, d);
        }
        if (std::fabs(n0 - n1) > 1e-3f || std::fabs(m0 - m1) > 1e-3f) {
          std::printf("FAIL rope norm %f %f\n", n0, n1);
          return 1;
        }
      }
    std::printf("PASS test_rope\n");
  }
  Kokkos::finalize();
  return 0;
}
