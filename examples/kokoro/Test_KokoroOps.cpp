#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosDNN.hpp"
TEST(KokoroOps, Conv1d) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", 1, 2, 6), w("w", 3, 2, 3), y("y", 1, 3, 6);
  Kokkos::View<float*, Exec> b("b", 3);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hb = Kokkos::create_mirror_view(b);
  for (size_t i = 0; i < 6; ++i)
    for (size_t c = 0; c < 2; ++c) hx(0, c, i) = (float)(i + c);
  for (size_t o = 0; o < 3; ++o)
    for (size_t i = 0; i < 2; ++i)
      for (size_t k = 0; k < 3; ++k) hw(o, i, k) = 0.1f * (o + i + k);
  for (size_t o = 0; o < 3; ++o) hb(o) = (float)o;
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::conv1d_forward<Exec, float>(x, w, b, y, 1, 1, 1, 1, "t");
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  for (size_t o = 0; o < 3; ++o)
    for (size_t t = 0; t < 6; ++t) {
      float acc = (float)o;
      for (size_t k = 0; k < 3; ++k) {
        long ti = (long)t + (long)k - 1;
        if (ti < 0 || ti >= 6) continue;
        for (size_t i = 0; i < 2; ++i) acc += 0.1f * (o + i + k) * (float)(ti + i);
      }
      EXPECT_NEAR(hy(0, o, t), acc, 1e-4f);
    }
}
TEST(KokoroOps, ConvTranspose1d) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", 1, 1, 4), w("w", 1, 1, 3), y("y", 1, 1, 8);
  Kokkos::View<float*, Exec> b;
  auto hx = Kokkos::create_mirror_view(x);
  for (size_t i = 0; i < 4; ++i) hx(0, 0, i) = (float)(i + 1);
  Kokkos::deep_copy(x, hx);
  auto hw = Kokkos::create_mirror_view(w);
  hw(0, 0, 0) = 0;
  hw(0, 0, 1) = 1;
  hw(0, 0, 2) = 0;
  Kokkos::deep_copy(w, hw);
  KokkosDNN::convtranspose1d_forward<Exec, float>(x, w, b, y, 2, 1, 1, 1, 1, "t");
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  for (size_t t = 0; t < 8; ++t) EXPECT_TRUE(std::isfinite(hy(0, 0, t)));
}
TEST(KokoroOps, AdaINAndSnake) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", 1, 4, 8), y("y", 1, 4, 8), a("a", 1, 4, 1);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> fc("fc", 8, 2), s("s", 1, 2);
  Kokkos::deep_copy(x, 0.5f);
  Kokkos::deep_copy(a, 1.0f);
  Kokkos::deep_copy(fc, 0.1f);
  Kokkos::deep_copy(s, 0.2f);
  KokkosDNN::snake_forward<Exec, float>(x, a, y, "sn");
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  EXPECT_NEAR(hy(0, 0, 0), 0.5f + std::sin(0.5f) * std::sin(0.5f), 1e-4f);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> z("z", 1, 4, 8);
  Kokkos::View<float*, Exec> fb("fb", 8);
  Kokkos::deep_copy(fb, 0.0f);
  KokkosDNN::adain_forward<Exec, float>(h, x, fc, fb, s, z, 1e-5f, "ad");
  auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, z);
  for (size_t c = 0; c < 4; ++c)
    for (size_t t = 0; t < 8; ++t) EXPECT_TRUE(std::isfinite(hz(0, c, t)));
}
TEST(KokoroOps, UpsampleNearest) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", 1, 1, 3), y("y", 1, 1, 6);
  auto hx = Kokkos::create_mirror_view(x);
  hx(0, 0, 0) = 1;
  hx(0, 0, 1) = 2;
  hx(0, 0, 2) = 3;
  Kokkos::deep_copy(x, hx);
  KokkosDNN::nearest_upsample1d<Exec, float>(x, y, 2, "u");
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  EXPECT_EQ(hy(0, 0, 0), 1);
  EXPECT_EQ(hy(0, 0, 1), 1);
  EXPECT_EQ(hy(0, 0, 2), 2);
  EXPECT_EQ(hy(0, 0, 5), 3);
}
TEST(KokoroOps, StftRoundtripTmp) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 300, F = 11, Fr = 1 + T / 5;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", 1, T);
  auto xh = Kokkos::create_mirror_view(x);
  for (int t = 0; t < T; ++t) xh(0, t) = sinf(t * 0.1f);
  Kokkos::deep_copy(x, xh);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> mag("m", 1, F, Fr), pha("p", 1, F, Fr);
  KokkosDNN::stft_forward<Exec, float>(x, 20, 5, mag, pha, "s");
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> y("y", 1, T);
  KokkosDNN::istft_inverse<Exec, float>(mag, pha, 20, 5, y, "i");
  auto yh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  double err = 0, mx = 0;
  for (int t = 10; t < T - 10; ++t) {
    double d = yh(0, t) - (double)xh(0, t);
    err = std::max(err, fabs(d));
    mx = std::max(mx, fabs((double)xh(0, t)));
  }
  EXPECT_LT(err, 1e-3);
  auto mh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mag);
  EXPECT_GT((double)mh(0, 0, 0), 0.0);
}

TEST(KokoroOps, LSTMForward) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", 1, 4, 3), y("y", 1, 4, 6);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> wih("wih", 12, 3), whh("whh", 12, 3);
  Kokkos::View<float*, Exec> bi("bi", 12), bh("bh", 12);
  Kokkos::deep_copy(x, 0.1f);
  Kokkos::deep_copy(wih, 0.1f);
  Kokkos::deep_copy(whh, 0.1f);
  Kokkos::deep_copy(bi, 0.0f);
  Kokkos::deep_copy(bh, 0.0f);
  Kokkos::View<int*, Exec> no_lengths;
  KokkosDNN::lstm_forward_bidir<Exec, float>(h, x, wih, whh, bi, bh, wih, whh, bi, bh, y, no_lengths, "l");
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  for (size_t t = 0; t < 4; ++t)
    for (size_t c = 0; c < 6; ++c) EXPECT_TRUE(std::isfinite(hy(0, t, c)));
}
