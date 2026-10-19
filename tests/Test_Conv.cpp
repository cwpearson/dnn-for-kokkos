// Promoted spatial convolutions: dense 3x3, depthwise KxK, grouped 1x1.
// Native correctness vs host references (biased/unbiased, argument
// validation). The dense 3x3 runs the im2col + gemm_bias lowering (so on
// TPL builds this also exercises the dispatched GEMM path); depthwise and
// grouped are native-only (see the TPL-status note in
// src/KokkosDNN_Conv.hpp). Dispatched-vs-native agreement is covered by
// the VAE decoder tests, which run the same operators through the decoder
// on every backend.
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

// Naive host dense 3x3 (double accumulation, zero padding, stride 1).
void host_conv3x3_ref(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& x,
    const Kokkos::View<float****, Kokkos::LayoutRight, Host>& w,
    const Kokkos::View<float*, Host>& b, bool biased,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& y, size_t H,
    size_t W) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  for (size_t t = 0; t < T; ++t)
    for (size_t o = 0; o < Cout; ++o) {
      double acc = biased ? b(o) : 0.0;
      const size_t h = t / W, ww = t % W;
      for (size_t i = 0; i < Cin; ++i)
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = (int)h + kh - 1;
          if (ih < 0 || (size_t)ih >= H) continue;
          for (int kw = 0; kw < 3; ++kw) {
            const int iw = (int)ww + kw - 1;
            if (iw < 0 || (size_t)iw >= W) continue;
            acc += (double)x((size_t)ih * W + (size_t)iw, i) *
                   (double)w(o, i, (size_t)kh, (size_t)kw);
          }
        }
      y(t, o) = static_cast<float>(acc);
    }
}

// Naive host depthwise KxK (double accumulation, zero padding, stride 1).
void host_depthwise_ref(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& x,
    const Kokkos::View<float***, Kokkos::LayoutRight, Host>& w,
    const Kokkos::View<float*, Host>& b, bool biased,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& y, size_t K,
    size_t H, size_t W) {
  const size_t T = x.extent(0), C = x.extent(1);
  const int P = (int)(K / 2);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c) {
      double acc = biased ? b(c) : 0.0;
      const size_t h = t / W, ww = t % W;
      for (size_t kh = 0; kh < K; ++kh) {
        const int ih = (int)h + (int)kh - P;
        if (ih < 0 || (size_t)ih >= H) continue;
        for (size_t kw = 0; kw < K; ++kw) {
          const int iw = (int)ww + (int)kw - P;
          if (iw < 0 || (size_t)iw >= W) continue;
          acc += (double)x((size_t)ih * W + (size_t)iw, c) *
                 (double)w(c, kh, kw);
        }
      }
      y(t, c) = static_cast<float>(acc);
    }
}

// Naive host grouped 1x1 (double accumulation).
void host_grouped1x1_ref(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& w,
    const Kokkos::View<float*, Host>& b, bool biased,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& y,
    size_t groups) {
  const size_t T = x.extent(0), Cout = w.extent(0), Cg = w.extent(1);
  const size_t Co = Cout / groups;
  for (size_t t = 0; t < T; ++t)
    for (size_t o = 0; o < Cout; ++o) {
      double acc = biased ? b(o) : 0.0;
      const size_t g = o / Co;
      for (size_t k = 0; k < Cg; ++k)
        acc += (double)x(t, g * Cg + k) * (double)w(o, k);
      y(t, o) = static_cast<float>(acc);
    }
}

void expect_near_2d(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& got,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& want, float tol) {
  ASSERT_EQ(got.extent(0), want.extent(0));
  ASSERT_EQ(got.extent(1), want.extent(1));
  for (size_t i = 0; i < got.extent(0); ++i)
    for (size_t j = 0; j < got.extent(1); ++j)
      EXPECT_NEAR(got(i, j), want(i, j), tol) << " at (" << i << "," << j
                                              << ")";
}

}  // namespace

TEST(Conv3x3, MatchesHostReferenceBiased) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 5, W = 4, T = H * W, Cin = 3, Cout = 4;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      href("href", T, Cout), hgot("hgot", T, Cout);
  Kokkos::View<float****, Kokkos::LayoutRight, Host> hw("hw", Cout, Cin, 3, 3);
  Kokkos::View<float*, Host> hb("hb", Cout);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i);
  for (size_t o = 0; o < Cout; ++o) {
    for (size_t i = 0; i < Cin; ++i)
      for (size_t kh = 0; kh < 3; ++kh)
        for (size_t kw = 0; kw < 3; ++kw)
          hw(o, i, kh, kw) = 0.5f * probe(o, i, kh * 3 + kw);
    hb(o) = 0.1f * (float)o - 0.2f;
  }
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      y("y", T, Cout);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> w("w", Cout, Cin, 3, 3);
  Kokkos::View<float*, Exec> b("b", Cout);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::conv3x3_forward(h, x, w, b, y, H, W);
  Kokkos::fence();
  Kokkos::deep_copy(hgot, y);
  host_conv3x3_ref(hx, hw, hb, true, href, H, W);
  expect_near_2d(hgot, href, 1e-4f);
}

TEST(Conv3x3, MatchesHostReferenceUnbiased) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 3, W = 6, T = H * W, Cin = 5, Cout = 2;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      href("href", T, Cout), hgot("hgot", T, Cout);
  Kokkos::View<float****, Kokkos::LayoutRight, Host> hw("hw", Cout, Cin, 3, 3);
  Kokkos::View<float*, Host> hb;
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i, 7);
  for (size_t o = 0; o < Cout; ++o)
    for (size_t i = 0; i < Cin; ++i)
      for (size_t kh = 0; kh < 3; ++kh)
        for (size_t kw = 0; kw < 3; ++kw)
          hw(o, i, kh, kw) = 0.5f * probe(o, kh, kw + i);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      y("y", T, Cout);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> w("w", Cout, Cin, 3, 3);
  Kokkos::View<float*, Exec> no_bias;
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  KokkosDNN::conv3x3_forward(h, x, w, no_bias, y, H, W);
  Kokkos::fence();
  Kokkos::deep_copy(hgot, y);
  host_conv3x3_ref(hx, hw, hb, false, href, H, W);
  expect_near_2d(hgot, href, 1e-4f);
}

TEST(Conv3x3, LoweredPathMatchesHostReference) {
  // Wide-output shape (Cout > ConvSkinnyMaxCout) exercising the im2col +
  // gemm_bias lowering rather than the skinny team kernel.
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 8, W = 8, T = H * W, Cin = 8, Cout = 40;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      href("href", T, Cout), hgot("hgot", T, Cout);
  Kokkos::View<float****, Kokkos::LayoutRight, Host> hw("hw", Cout, Cin, 3, 3);
  Kokkos::View<float*, Host> hb("hb", Cout);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i, 11);
  for (size_t o = 0; o < Cout; ++o) {
    for (size_t i = 0; i < Cin; ++i)
      for (size_t kh = 0; kh < 3; ++kh)
        for (size_t kw = 0; kw < 3; ++kw)
          hw(o, i, kh, kw) = 0.5f * probe(o, i, kh * 3 + kw);
    hb(o) = 0.1f * (float)(o % 5) - 0.2f;
  }
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      y("y", T, Cout);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> w("w", Cout, Cin, 3, 3);
  Kokkos::View<float*, Exec> b("b", Cout), no_bias;
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::conv3x3_forward(h, x, w, b, y, H, W);
  Kokkos::fence();
  Kokkos::deep_copy(hgot, y);
  host_conv3x3_ref(hx, hw, hb, true, href, H, W);
  expect_near_2d(hgot, href, 1e-4f);
  KokkosDNN::conv3x3_forward(h, x, w, no_bias, y, H, W);
  Kokkos::fence();
  Kokkos::deep_copy(hgot, y);
  host_conv3x3_ref(hx, hw, hb, false, href, H, W);
  expect_near_2d(hgot, href, 1e-4f);
}

TEST(Conv3x3, RejectsBadArgs) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 4, W = 4, T = H * W, Cin = 3, Cout = 5;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      y("y", T, Cout);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> w("w", Cout, Cin, 3, 3);
  Kokkos::View<float*, Exec> b("b", Cout), empty_b;
  EXPECT_NO_THROW(KokkosDNN::conv3x3_forward(h, x, w, b, y, H, W));
  EXPECT_NO_THROW(KokkosDNN::conv3x3_forward(h, x, w, empty_b, y, H, W));
  EXPECT_THROW(KokkosDNN::conv3x3_forward(h, x, w, b, y, H, W + 1),
               std::invalid_argument);  // T != H*W
  EXPECT_THROW(KokkosDNN::conv3x3_forward(h, x, w, b, y, 0, W),
               std::invalid_argument);  // zero H
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> bad_w("bad_w", Cout,
                                                           Cin + 1, 3, 3);
  EXPECT_THROW(KokkosDNN::conv3x3_forward(h, x, bad_w, empty_b, y, H, W),
               std::invalid_argument);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_y("bad_y", T, Cout + 1);
  EXPECT_THROW(KokkosDNN::conv3x3_forward(h, x, w, empty_b, bad_y, H, W),
               std::invalid_argument);
  Kokkos::View<float*, Exec> bad_b("bad_b", Cout + 1);
  EXPECT_THROW(KokkosDNN::conv3x3_forward(h, x, w, bad_b, y, H, W),
               std::invalid_argument);
}

TEST(DepthwiseConv, MatchesHostReferenceK3AndK5) {
  KokkosDNN::DNNHandle<Exec> h;
  for (const size_t K : {3, 5}) {
    const size_t H = 4, W = 5, T = H * W, C = 6;
    Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, C),
        href("href", T, C), hgot("hgot", T, C);
    Kokkos::View<float***, Kokkos::LayoutRight, Host> hw("hw", C, K, K);
    Kokkos::View<float*, Host> hb("hb", C);
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c) hx(t, c) = probe(t, c, K);
    for (size_t c = 0; c < C; ++c) {
      for (size_t kh = 0; kh < K; ++kh)
        for (size_t kw = 0; kw < K; ++kw)
          hw(c, kh, kw) = 0.4f * probe(c, kh, kw);
      hb(c) = 0.05f * (float)c - 0.1f;
    }
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C),
        y("y", T, C);
    Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", C, K, K);
    Kokkos::View<float*, Exec> b("b", C), empty_b;
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(w, hw);
    Kokkos::deep_copy(b, hb);
    // Biased path.
    KokkosDNN::depthwise_conv_forward(h, x, w, b, y, K, H, W);
    Kokkos::fence();
    Kokkos::deep_copy(hgot, y);
    host_depthwise_ref(hx, hw, hb, true, href, K, H, W);
    expect_near_2d(hgot, href, 1e-4f);
    // Unbiased path.
    KokkosDNN::depthwise_conv_forward(h, x, w, empty_b, y, K, H, W);
    Kokkos::fence();
    Kokkos::deep_copy(hgot, y);
    host_depthwise_ref(hx, hw, hb, false, href, K, H, W);
    expect_near_2d(hgot, href, 1e-4f);
  }
}

TEST(DepthwiseConv, RejectsBadArgs) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 4, W = 4, T = H * W, C = 3, K = 3;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C),
      y("y", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", C, K, K);
  Kokkos::View<float*, Exec> b("b", C), empty_b;
  EXPECT_NO_THROW(KokkosDNN::depthwise_conv_forward(h, x, w, b, y, K, H, W));
  EXPECT_THROW(KokkosDNN::depthwise_conv_forward(h, x, w, b, y, 4, H, W),
               std::invalid_argument);  // even K
  EXPECT_THROW(KokkosDNN::depthwise_conv_forward(h, x, w, b, y, 0, H, W),
               std::invalid_argument);  // zero K
  EXPECT_THROW(KokkosDNN::depthwise_conv_forward(h, x, w, b, y, K, H, W + 1),
               std::invalid_argument);  // T != H*W
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_w("bad_w", C + 1, K,
                                                          K);
  EXPECT_THROW(KokkosDNN::depthwise_conv_forward(h, x, bad_w, empty_b, y, K, H,
                                                 W),
               std::invalid_argument);
}

TEST(GroupedConv1x1, MatchesHostReference) {
  KokkosDNN::DNNHandle<Exec> h;
  // VAE-like grouped shape: Cin=3C, Cout=3C, groups=3*heads, Cg=32.
  const size_t T = 12, heads = 2, C = 64, Cin = 3 * C, Cout = 3 * C;
  const size_t groups = 3 * heads, Cg = 32;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      href("href", T, Cout), hgot("hgot", T, Cout), hw("hw", Cout, Cg);
  Kokkos::View<float*, Host> hb("hb", Cout);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i, 3);
  for (size_t o = 0; o < Cout; ++o) {
    for (size_t k = 0; k < Cg; ++k) hw(o, k) = 0.25f * probe(o, k, 5);
    hb(o) = 0.05f * (float)(o % 7) - 0.1f;
  }
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      w("w", Cout, Cg), y("y", T, Cout);
  Kokkos::View<float*, Exec> b("b", Cout), empty_b;
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::grouped_conv1x1_forward(h, x, w, b, y, groups);
  Kokkos::fence();
  Kokkos::deep_copy(hgot, y);
  host_grouped1x1_ref(hx, hw, hb, true, href, groups);
  expect_near_2d(hgot, href, 1e-4f);
  KokkosDNN::grouped_conv1x1_forward(h, x, w, empty_b, y, groups);
  Kokkos::fence();
  Kokkos::deep_copy(hgot, y);
  host_grouped1x1_ref(hx, hw, hb, false, href, groups);
  expect_near_2d(hgot, href, 1e-4f);
}

TEST(GroupedConv1x1, RejectsBadArgs) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t T = 8, Cin = 12, Cout = 18, Cg = 4, groups = 3;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      w("w", Cout, Cg), y("y", T, Cout);
  Kokkos::View<float*, Exec> b("b", Cout), empty_b;
  EXPECT_NO_THROW(KokkosDNN::grouped_conv1x1_forward(h, x, w, b, y, groups));
  EXPECT_THROW(KokkosDNN::grouped_conv1x1_forward(h, x, w, b, y, 0),
               std::invalid_argument);  // zero groups
  EXPECT_THROW(KokkosDNN::grouped_conv1x1_forward(h, x, w, b, y, 4),
               std::invalid_argument);  // Cout % groups != 0
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_w("bad_w", Cout, Cg + 1);
  EXPECT_THROW(KokkosDNN::grouped_conv1x1_forward(h, x, bad_w, empty_b, y,
                                                 groups),
               std::invalid_argument);  // Cin != groups*Cg
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_y("bad_y", T, Cout + 1);
  EXPECT_THROW(KokkosDNN::grouped_conv1x1_forward(h, x, w, empty_b, bad_y,
                                                 groups),
               std::invalid_argument);
}
