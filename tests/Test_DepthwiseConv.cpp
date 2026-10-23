// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Depthwise 3x3 convolution (stride 1, pad 1, groups=Cin) vs naive host
// reference: rank-2 biased/unbiased, rank-3 batched, argument validation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>

#include "KokkosDNN.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::HostSpace;

float probe(size_t a, size_t b, size_t c = 0) {
  return 0.06f * static_cast<float>((a * 31 + b * 17 + c * 13) % 19) - 0.5f;
}

// Naive host depthwise 3x3 (zero padding, stride 1), single spatial map.
void host_depthwise_ref(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& x,
    const Kokkos::View<float***, Kokkos::LayoutRight, Host>& w,
    const Kokkos::View<float*, Host>& b, bool biased,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& y, size_t H,
    size_t W) {
  const size_t C = x.extent(1);
  for (size_t h = 0; h < H; ++h)
    for (size_t ww = 0; ww < W; ++ww)
      for (size_t c = 0; c < C; ++c) {
        double acc = biased ? b(c) : 0.0;
        for (int kh = 0; kh < 3; ++kh)
          for (int kw = 0; kw < 3; ++kw) {
            const int ih = (int)h + kh - 1, iw = (int)ww + kw - 1;
            if (ih < 0 || (size_t)ih >= H || iw < 0 || (size_t)iw >= W) continue;
            acc += (double)x((size_t)ih * W + (size_t)iw, c) *
                   (double)w(c, (size_t)kh, (size_t)kw);
          }
        y(h * W + ww, c) = static_cast<float>(acc);
      }
}

void expect_near_2d(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& got,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& want, float tol) {
  ASSERT_EQ(got.extent(0), want.extent(0));
  ASSERT_EQ(got.extent(1), want.extent(1));
  for (size_t i = 0; i < got.extent(0); ++i)
    for (size_t j = 0; j < got.extent(1); ++j)
      EXPECT_NEAR(got(i, j), want(i, j), tol) << " at (" << i << "," << j << ")";
}

}  // namespace

TEST(DepthwiseConv, Rank2BiasedMatchesHost) {
  const size_t H = 3, W = 3, T = H * W, C = 2;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hw("hw", C, 3, 3);
  Kokkos::View<float*, Host> hb("hb", C);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c) hx(t, c) = probe(t, c);
  for (size_t c = 0; c < C; ++c) {
    for (size_t kh = 0; kh < 3; ++kh)
      for (size_t kw = 0; kw < 3; ++kw) hw(c, kh, kw) = probe(c, kh, kw);
    hb(c) = 0.25f * (float)c - 0.1f;
  }
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C), y("y", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", C, 3, 3);
  Kokkos::View<float*, Exec> b("b", C);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::depthwise_conv3x3_forward(x, w, b, y, H, W);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, C),
      ref("ref", T, C);
  Kokkos::deep_copy(got, y);
  host_depthwise_ref(hx, hw, hb, true, ref, H, W);
  expect_near_2d(got, ref, 1e-5f);
}

TEST(DepthwiseConv, Rank2UnbiasedMatchesHost) {
  const size_t H = 3, W = 3, T = H * W, C = 2;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hw("hw", C, 3, 3);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c) hx(t, c) = probe(t, c, 9);
  for (size_t c = 0; c < C; ++c)
    for (size_t kh = 0; kh < 3; ++kh)
      for (size_t kw = 0; kw < 3; ++kw) hw(c, kh, kw) = probe(kh, kw, c);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C), y("y", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", C, 3, 3);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::View<float*, Exec> empty_b;
  KokkosDNN::depthwise_conv3x3_forward(x, w, empty_b, y, H, W);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, C),
      ref("ref", T, C);
  Kokkos::deep_copy(got, y);
  Kokkos::View<float*, Host> hb;
  host_depthwise_ref(hx, hw, hb, false, ref, H, W);
  expect_near_2d(got, ref, 1e-5f);
}

TEST(DepthwiseConv, Rank3BatchedMatchesHost) {
  const size_t B = 2, H = 3, W = 3, T = H * W, C = 2;
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hx("hx", B, T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hw("hw", C, 3, 3);
  Kokkos::View<float*, Host> hb("hb", C);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c) hx(b, t, c) = probe(b * T + t, c);
  for (size_t c = 0; c < C; ++c) {
    for (size_t kh = 0; kh < 3; ++kh)
      for (size_t kw = 0; kw < 3; ++kw) hw(c, kh, kw) = probe(c, kh, kw);
    hb(c) = -0.2f + 0.3f * (float)c;
  }
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", B, T, C),
      y("y", B, T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", C, 3, 3);
  Kokkos::View<float*, Exec> b("b", C);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::depthwise_conv3x3_forward(x, w, b, y, H, W);
  Kokkos::fence();
  Kokkos::View<float***, Kokkos::LayoutRight, Host> got("got", B, T, C);
  Kokkos::deep_copy(got, y);
  for (size_t bb = 0; bb < B; ++bb) {
    Kokkos::View<float**, Kokkos::LayoutRight, Host> hxb(
        hx.data() + bb * T * C, T, C);
    Kokkos::View<float**, Kokkos::LayoutRight, Host> ref("ref", T, C);
    host_depthwise_ref(hxb, hw, hb, true, ref, H, W);
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c)
        EXPECT_NEAR(got(bb, t, c), ref(t, c), 1e-5f)
            << " at (" << bb << "," << t << "," << c << ")";
  }
}

TEST(DepthwiseConv, RejectsBadArgs) {
  const size_t H = 3, W = 3, T = H * W, C = 2;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C),
      y("y", T, C), bad_y("bad_y", T + 1, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", C, 3, 3),
      bad_w("bad_w", C + 1, 3, 3);
  Kokkos::View<float*, Exec> b("b", C), bad_b("bad_b", C + 1), empty_b;
  EXPECT_NO_THROW(KokkosDNN::depthwise_conv3x3_forward(x, w, b, y, H, W));
  EXPECT_NO_THROW(KokkosDNN::depthwise_conv3x3_forward(x, w, empty_b, y, H, W));
  EXPECT_THROW(KokkosDNN::depthwise_conv3x3_forward(x, w, b, y, H, W + 1),
               std::invalid_argument);  // T != H*W
  EXPECT_THROW(KokkosDNN::depthwise_conv3x3_forward(x, w, b, y, 0, W),
               std::invalid_argument);  // zero H
  EXPECT_THROW(KokkosDNN::depthwise_conv3x3_forward(x, bad_w, b, y, H, W),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::depthwise_conv3x3_forward(x, w, bad_b, y, H, W),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::depthwise_conv3x3_forward(x, w, b, bad_y, H, W),
               std::invalid_argument);
}
