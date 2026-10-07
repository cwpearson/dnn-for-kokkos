// RoPE preserves per-head norm (rotation is orthogonal, float + bfloat16).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>

#include "KokkosDNN.hpp"

template <class Scalar>
void check_rope(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 2, H = 2, D = 8;
  Kokkos::View<Scalar***, Exec> q("q", T, H, D), k("k", T, H, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d) {
        hq(t, h, d) = static_cast<Scalar>(0.1f * (t * 100 + h * 10 + d + 1));
        hk(t, h, d) =
            static_cast<Scalar>(-0.05f * (t * 100 + h * 10 + d + 2));
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
  KokkosDNN::rope_neoX_inplace<Exec, Scalar>(q, k, inv, D, pos);
  Kokkos::fence();
  auto gq = Kokkos::create_mirror_view(q);
  auto gk = Kokkos::create_mirror_view(k);
  Kokkos::deep_copy(gq, q);
  Kokkos::deep_copy(gk, k);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      float n0 = 0, n1 = 0, m0 = 0, m1 = 0;
      for (int d = 0; d < D; ++d) {
        float a = static_cast<float>(hq(t, h, d));
        float b = static_cast<float>(gq(t, h, d));
        float c = static_cast<float>(hk(t, h, d));
        float e = static_cast<float>(gk(t, h, d));
        n0 += a * a;
        n1 += b * b;
        m0 += c * c;
        m1 += e * e;
      }
      EXPECT_NEAR(n1, n0, tol) << "rope q norm changed at (" << t << "," << h
                               << ")";
      EXPECT_NEAR(m1, m0, tol) << "rope k norm changed at (" << t << "," << h
                               << ")";
    }
}

TEST(RoPE, Float) { check_rope<float>(1e-3f); }

TEST(RoPE, BHalf) { check_rope<Kokkos::Experimental::bhalf_t>(5e-1f); }
