// Attention: single query attending to a single key must return its value.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include "KokkosDNN.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    KokkosDNN::DNNHandle<Exec> h;
    const int Hq = 4, Hkv = 2, D = 8;
    Kokkos::View<float***, Exec> q("q", 1, Hq, D), k("k", 1, Hkv, D),
        v("v", 1, Hkv, D), out("out", 1, Hq, D);
    auto hq = Kokkos::create_mirror_view(q);
    auto hk = Kokkos::create_mirror_view(k);
    auto hv = Kokkos::create_mirror_view(v);
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d) hq(0, hh, d) = 0.1f * (hh + d + 1);
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(0, hh, d) = 0.2f * (hh - d);
        hv(0, hh, d) = 1.0f * hh + 0.01f * d + 5.0f;
      }
    Kokkos::deep_copy(q, hq);
    Kokkos::deep_copy(k, hk);
    Kokkos::deep_copy(v, hv);
    KokkosDNN::attention_gqa_single<Exec>(h, q, k, v, 0, out, 1.0f / 8.0f);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(got, out);
    for (int hh = 0; hh < Hq; ++hh) {
      int hkv = (hh * Hkv) / Hq;
      for (int d = 0; d < D; ++d)
        if (std::fabs(got(0, hh, d) - hv(0, hkv, d)) > 1e-4f) {
          std::printf("FAIL attention single-key\n");
          return 1;
        }
    }
    std::printf("PASS test_attention\n");
  }
  Kokkos::finalize();
  return 0;
}
