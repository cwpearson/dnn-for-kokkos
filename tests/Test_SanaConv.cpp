// Sana GLUMBConv MixFFN ops vs naive host references: conv1x1 (biased +
// unbiased), depthwise 3x3 (rank-2 biased/unbiased, rank-3 batched),
// glumbconv vs a composed host pipeline (rank-2, unbiased-depthwise variant,
// and a batched Sana-600M-ratio config), patch_embed_1x1 vs a host 1x1 conv,
// unpatchify vs a host transpose (+ patch/unpatch round-trip), and argument
// validation.
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

float host_silu(float x) { return x / (1.0f + std::exp(-x)); }

// Naive host 1x1 conv: y[t,o] = b[o] + sum_i x[t,i]*w[o,i].
void host_conv1x1_ref(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& w,
    const Kokkos::View<float*, Host>& b, bool biased,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& y) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  for (size_t t = 0; t < T; ++t)
    for (size_t o = 0; o < Cout; ++o) {
      double acc = biased ? b(o) : 0.0;
      for (size_t i = 0; i < Cin; ++i) acc += (double)x(t, i) * (double)w(o, i);
      y(t, o) = static_cast<float>(acc);
    }
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

// Composed host GLUMBConv pipeline (double accumulation, float SiLU to match
// the device's FP32 math): u=SiLU(x@inv+b), dw=depthwise(u),
// h=a*SiLU(g), y=h@pt.
void host_glumb_ref(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& inv_w,
    const Kokkos::View<float*, Host>& inv_b, bool inv_biased,
    const Kokkos::View<float***, Kokkos::LayoutRight, Host>& dw_w,
    const Kokkos::View<float*, Host>& dw_b, bool dw_biased,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& pt_w,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& y, size_t H,
    size_t W) {
  const size_t T = x.extent(0), Cin = x.extent(1);
  const size_t inv_out = inv_w.extent(0), Hid = inv_out / 2;
  const size_t Cout = pt_w.extent(0);
  Kokkos::View<float**, Kokkos::LayoutRight, Host> u("u", T, inv_out);
  Kokkos::View<float**, Kokkos::LayoutRight, Host> dw("dw", T, inv_out);
  for (size_t t = 0; t < T; ++t)
    for (size_t o = 0; o < inv_out; ++o) {
      double acc = inv_biased ? inv_b(o) : 0.0;
      for (size_t i = 0; i < Cin; ++i) acc += (double)x(t, i) * (double)inv_w(o, i);
      u(t, o) = host_silu(static_cast<float>(acc));
    }
  host_depthwise_ref(u, dw_w, dw_b, dw_biased, dw, H, W);
  for (size_t t = 0; t < T; ++t)
    for (size_t o = 0; o < Cout; ++o) {
      double acc = 0.0;
      for (size_t j = 0; j < Hid; ++j) {
        const float a = dw(t, j);
        const float g = host_silu(dw(t, j + Hid));
        acc += (double)(a * g) * (double)pt_w(o, j);
      }
      y(t, o) = static_cast<float>(acc);
    }
}

// Naive host 1x1 patch embed: tok[b, h*W+w, d] = bias[d] +
// sum_c img[b,c,h,w]*wt[d,c].
void host_patch1x1_ref(
    const Kokkos::View<float****, Kokkos::LayoutRight, Host>& img,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& wt,
    const Kokkos::View<float*, Host>& bs, bool biased,
    const Kokkos::View<float***, Kokkos::LayoutRight, Host>& tok) {
  const size_t B = img.extent(0), C = img.extent(1);
  const size_t H = img.extent(2), W = img.extent(3), D = wt.extent(0);
  for (size_t b = 0; b < B; ++b)
    for (size_t h = 0; h < H; ++h)
      for (size_t w = 0; w < W; ++w)
        for (size_t d = 0; d < D; ++d) {
          double acc = biased ? bs(d) : 0.0;
          for (size_t c = 0; c < C; ++c)
            acc += (double)img(b, c, h, w) * (double)wt(d, c);
          tok(b, h * W + w, d) = static_cast<float>(acc);
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

TEST(SanaConv1x1, BiasedMatchesHost) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t T = 6, Cin = 4, Cout = 5;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      hw("hw", Cout, Cin);
  Kokkos::View<float*, Host> hb("hb", Cout);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i);
  for (size_t o = 0; o < Cout; ++o) {
    for (size_t i = 0; i < Cin; ++i) hw(o, i) = probe(o, i, 3);
    hb(o) = probe(o, 7) * 0.5f;
  }
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      w("w", Cout, Cin), y("y", T, Cout);
  Kokkos::View<float*, Exec> b("b", Cout);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::conv1x1_forward(h, x, w, b, y);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, Cout),
      ref("ref", T, Cout);
  Kokkos::deep_copy(got, y);
  host_conv1x1_ref(hx, hw, hb, true, ref);
  expect_near_2d(got, ref, 1e-4f);
}

TEST(SanaConv1x1, UnbiasedEmptyBias) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t T = 5, Cin = 3, Cout = 4;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      hw("hw", Cout, Cin);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i, 1);
  for (size_t o = 0; o < Cout; ++o)
    for (size_t i = 0; i < Cin; ++i) hw(o, i) = probe(o, i, 5);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      w("w", Cout, Cin), y("y", T, Cout);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::View<float*, Exec> empty_b;
  KokkosDNN::conv1x1_forward(h, x, w, empty_b, y);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, Cout),
      ref("ref", T, Cout);
  Kokkos::deep_copy(got, y);
  Kokkos::View<float*, Host> hb;
  host_conv1x1_ref(hx, hw, hb, false, ref);
  expect_near_2d(got, ref, 1e-5f);
  // Dimension mismatch still throws.
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_y("bad_y", T, Cout + 1);
  EXPECT_THROW(KokkosDNN::conv1x1_forward(h, x, w, empty_b, bad_y),
               std::invalid_argument);
  Kokkos::View<float*, Exec> bad_b("bad_b", Cout + 1);
  EXPECT_THROW(KokkosDNN::conv1x1_forward(h, x, w, bad_b, y),
               std::invalid_argument);
}

TEST(SanaConvDepthwise, Rank2BiasedMatchesHost) {
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

TEST(SanaConvDepthwise, Rank2UnbiasedMatchesHost) {
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

TEST(SanaConvDepthwise, Rank3BatchedMatchesHost) {
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

TEST(SanaConvDepthwise, RejectsBadArgs) {
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

TEST(SanaConvGLUMB, Rank2MatchesComposed) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 3, W = 3, T = H * W, Cin = 4, Hid = 3, Cout = 5;
  const size_t inv_out = 2 * Hid;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      hiw("hiw", inv_out, Cin), hpw("hpw", Cout, Hid);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hdw("hdw", inv_out, 3, 3);
  Kokkos::View<float*, Host> hib("hib", inv_out), hdb("hdb", inv_out);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i);
  for (size_t o = 0; o < inv_out; ++o) {
    for (size_t i = 0; i < Cin; ++i) hiw(o, i) = 0.5f * probe(o, i, 2);
    for (size_t kh = 0; kh < 3; ++kh)
      for (size_t kw = 0; kw < 3; ++kw) hdw(o, kh, kw) = 0.5f * probe(o, kh, kw);
    hib(o) = 0.1f * (float)o - 0.2f;
    hdb(o) = 0.05f * (float)o - 0.1f;
  }
  for (size_t o = 0; o < Cout; ++o)
    for (size_t j = 0; j < Hid; ++j) hpw(o, j) = 0.5f * probe(o, j, 4);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      inv_w("inv_w", inv_out, Cin), pt_w("pt_w", Cout, Hid), y("y", T, Cout);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> dw_w("dw_w", inv_out, 3, 3);
  Kokkos::View<float*, Exec> inv_b("inv_b", inv_out), dw_b("dw_b", inv_out);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(inv_w, hiw);
  Kokkos::deep_copy(inv_b, hib);
  Kokkos::deep_copy(dw_w, hdw);
  Kokkos::deep_copy(dw_b, hdb);
  Kokkos::deep_copy(pt_w, hpw);
  KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H, W, y);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, Cout),
      ref("ref", T, Cout);
  Kokkos::deep_copy(got, y);
  host_glumb_ref(hx, hiw, hib, true, hdw, hdb, true, hpw, ref, H, W);
  expect_near_2d(got, ref, 1e-4f);
}

TEST(SanaConvGLUMB, Rank2UnbiasedDepthwiseMatchesComposed) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 3, W = 2, T = H * W, Cin = 3, Hid = 2, Cout = 4;
  const size_t inv_out = 2 * Hid;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx("hx", T, Cin),
      hiw("hiw", inv_out, Cin), hpw("hpw", Cout, Hid);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hdw("hdw", inv_out, 3, 3);
  Kokkos::View<float*, Host> hib("hib", inv_out);
  for (size_t t = 0; t < T; ++t)
    for (size_t i = 0; i < Cin; ++i) hx(t, i) = probe(t, i, 6);
  for (size_t o = 0; o < inv_out; ++o) {
    for (size_t i = 0; i < Cin; ++i) hiw(o, i) = 0.5f * probe(o, i, 8);
    for (size_t kh = 0; kh < 3; ++kh)
      for (size_t kw = 0; kw < 3; ++kw) hdw(o, kh, kw) = 0.5f * probe(kh, kw, o);
    hib(o) = 0.1f * (float)o - 0.15f;
  }
  for (size_t o = 0; o < Cout; ++o)
    for (size_t j = 0; j < Hid; ++j) hpw(o, j) = 0.5f * probe(o, j, 11);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      inv_w("inv_w", inv_out, Cin), pt_w("pt_w", Cout, Hid), y("y", T, Cout);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> dw_w("dw_w", inv_out, 3, 3);
  Kokkos::View<float*, Exec> inv_b("inv_b", inv_out), empty_b;
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(inv_w, hiw);
  Kokkos::deep_copy(inv_b, hib);
  Kokkos::deep_copy(dw_w, hdw);
  Kokkos::deep_copy(pt_w, hpw);
  KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, empty_b, pt_w, H, W,
                               y);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, Cout),
      ref("ref", T, Cout);
  Kokkos::deep_copy(got, y);
  Kokkos::View<float*, Host> hdb;
  host_glumb_ref(hx, hiw, hib, true, hdw, hdb, false, hpw, ref, H, W);
  expect_near_2d(got, ref, 1e-4f);
}

// Batched path at Sana-600M proportions (hidden = 2.5 * in), tiny scale:
// in=8, hidden=20, out=8 over 4x4 maps, B=2.
TEST(SanaConvGLUMB, Rank3SanaRatioMatchesComposed) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t B = 2, H = 4, W = 4, T = H * W, Cin = 8, Hid = 20, Cout = 8;
  const size_t inv_out = 2 * Hid;
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hx("hx", B, T, Cin);
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hiw("hiw", inv_out, Cin),
      hpw("hpw", Cout, Hid);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> hdw("hdw", inv_out, 3, 3);
  Kokkos::View<float*, Host> hib("hib", inv_out), hdb("hdb", inv_out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t i = 0; i < Cin; ++i) hx(b, t, i) = probe(b * T + t, i);
  for (size_t o = 0; o < inv_out; ++o) {
    for (size_t i = 0; i < Cin; ++i) hiw(o, i) = 0.25f * probe(o, i, 2);
    for (size_t kh = 0; kh < 3; ++kh)
      for (size_t kw = 0; kw < 3; ++kw) hdw(o, kh, kw) = 0.25f * probe(o, kh, kw);
    hib(o) = 0.05f * (float)(o % 7) - 0.1f;
    hdb(o) = 0.03f * (float)(o % 5) - 0.05f;
  }
  for (size_t o = 0; o < Cout; ++o)
    for (size_t j = 0; j < Hid; ++j) hpw(o, j) = 0.25f * probe(o, j, 4);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", B, T, Cin),
      y("y", B, T, Cout);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> inv_w("inv_w", inv_out, Cin),
      pt_w("pt_w", Cout, Hid);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> dw_w("dw_w", inv_out, 3, 3);
  Kokkos::View<float*, Exec> inv_b("inv_b", inv_out), dw_b("dw_b", inv_out);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(inv_w, hiw);
  Kokkos::deep_copy(inv_b, hib);
  Kokkos::deep_copy(dw_w, hdw);
  Kokkos::deep_copy(dw_b, hdb);
  Kokkos::deep_copy(pt_w, hpw);
  KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H, W, y);
  Kokkos::fence();
  Kokkos::View<float***, Kokkos::LayoutRight, Host> got("got", B, T, Cout);
  Kokkos::deep_copy(got, y);
  for (size_t b = 0; b < B; ++b) {
    Kokkos::View<float**, Kokkos::LayoutRight, Host> hxb(hx.data() + b * T * Cin,
                                                         T, Cin);
    Kokkos::View<float**, Kokkos::LayoutRight, Host> ref("ref", T, Cout);
    host_glumb_ref(hxb, hiw, hib, true, hdw, hdb, true, hpw, ref, H, W);
    for (size_t t = 0; t < T; ++t)
      for (size_t o = 0; o < Cout; ++o)
        EXPECT_NEAR(got(b, t, o), ref(t, o), 1e-4f)
            << " at (" << b << "," << t << "," << o << ")";
  }
}

TEST(SanaConvGLUMB, RejectsBadArgs) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 3, W = 3, T = H * W, Cin = 4, Hid = 3, Cout = 5;
  const size_t inv_out = 2 * Hid;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      inv_w("inv_w", inv_out, Cin), pt_w("pt_w", Cout, Hid), y("y", T, Cout);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> dw_w("dw_w", inv_out, 3, 3);
  Kokkos::View<float*, Exec> inv_b("inv_b", inv_out), dw_b("dw_b", inv_out),
      empty_b;
  EXPECT_NO_THROW(
      KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H, W, y));
  EXPECT_NO_THROW(KokkosDNN::glumbconv_forward(h, x, inv_w, empty_b, dw_w,
                                               empty_b, pt_w, H, W, y));
  EXPECT_THROW(
      KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H,
                                   W + 1, y),
      std::invalid_argument);  // T != H*W
  EXPECT_THROW(
      KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, 0, W, y),
      std::invalid_argument);  // zero H
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> odd_inv("odd_inv", inv_out + 1,
                                                           Cin);
  EXPECT_THROW(KokkosDNN::glumbconv_forward(h, x, odd_inv, empty_b, dw_w,
                                            empty_b, pt_w, H, W, y),
               std::invalid_argument);  // odd inverted width
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_dw("bad_dw", inv_out, 3,
                                                           3);
  // bad_dw has the right shape here; make a wrong-channel one:
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_dw_c("bad_dw_c",
                                                             inv_out + 2, 3, 3);
  EXPECT_THROW(KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, bad_dw_c, dw_b,
                                            pt_w, H, W, y),
               std::invalid_argument);
  (void)bad_dw;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_pt("bad_pt", Cout,
                                                          Hid + 1);
  EXPECT_THROW(
      KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, bad_pt, H, W, y),
      std::invalid_argument);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_y("bad_y", T, Cout + 1);
  EXPECT_THROW(KokkosDNN::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b,
                                            pt_w, H, W, bad_y),
               std::invalid_argument);
  Kokkos::View<float*, Exec> bad_ib("bad_ib", inv_out + 1);
  EXPECT_THROW(KokkosDNN::glumbconv_forward(h, x, inv_w, bad_ib, dw_w, dw_b,
                                            pt_w, H, W, y),
               std::invalid_argument);
}

TEST(SanaConvPatch, Embed1x1MatchesHost) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t B = 2, C = 3, H = 4, W = 4, T = H * W, D = 5;
  Kokkos::View<float****, Kokkos::LayoutRight, Host> himg("himg", B, C, H, W);
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hwt("hwt", D, C);
  Kokkos::View<float*, Host> hbs("hbs", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < C; ++c)
      for (size_t hh = 0; hh < H; ++hh)
        for (size_t w = 0; w < W; ++w)
          himg(b, c, hh, w) = probe(b * C * H * W + c * H * W + hh * W + w, c);
  for (size_t d = 0; d < D; ++d) {
    for (size_t c = 0; c < C; ++c) hwt(d, c) = probe(d, c, 3);
    hbs(d) = 0.2f * (float)d - 0.3f;
  }
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> images("images", B, C, H,
                                                            W);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> weight("weight", D, C);
  Kokkos::View<float*, Exec> bias("bias", D), empty_b;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> tok("tok", B, T, D),
      tok_u("tok_u", B, T, D);
  Kokkos::deep_copy(images, himg);
  Kokkos::deep_copy(weight, hwt);
  Kokkos::deep_copy(bias, hbs);
  KokkosDNN::patch_embed_1x1_forward(h, images, weight, bias, tok);
  KokkosDNN::patch_embed_1x1_forward(h, images, weight, empty_b, tok_u);
  Kokkos::fence();
  Kokkos::View<float***, Kokkos::LayoutRight, Host> got("got", B, T, D),
      got_u("got_u", B, T, D), ref("ref", B, T, D), ref_u("ref_u", B, T, D);
  Kokkos::deep_copy(got, tok);
  Kokkos::deep_copy(got_u, tok_u);
  host_patch1x1_ref(himg, hwt, hbs, true, ref);
  Kokkos::View<float*, Host> hb;
  host_patch1x1_ref(himg, hwt, hb, false, ref_u);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        EXPECT_NEAR(got(b, t, d), ref(b, t, d), 1e-4f)
            << " biased at (" << b << "," << t << "," << d << ")";
        EXPECT_NEAR(got_u(b, t, d), ref_u(b, t, d), 1e-4f)
            << " unbiased at (" << b << "," << t << "," << d << ")";
      }
}

TEST(SanaConvPatch, UnpatchifyMatchesHost) {
  const size_t B = 2, H = 3, W = 3, T = H * W, C = 2;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> htok("htok", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> htok3("htok3", B, T, C);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c) {
        const float v = probe(b * T + t, c);
        htok3(b, t, c) = v;
        if (b == 0) htok(t, c) = v;
      }
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> tok("tok", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> img("img", C, H, W),
      tok3("tok3", B, T, C);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> imgs("imgs", B, C, H, W);
  Kokkos::deep_copy(tok, htok);
  Kokkos::deep_copy(tok3, htok3);
  KokkosDNN::unpatchify_forward(tok, img, H, W);
  KokkosDNN::unpatchify_forward(tok3, imgs, H, W);
  Kokkos::fence();
  Kokkos::View<float***, Kokkos::LayoutRight, Host> got("got", C, H, W);
  Kokkos::View<float****, Kokkos::LayoutRight, Host> got3("got3", B, C, H, W);
  Kokkos::deep_copy(got, img);
  Kokkos::deep_copy(got3, imgs);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c)
      EXPECT_NEAR(got(c, t / W, t % W), htok(t, c), 1e-6f)
          << " rank-2 at (" << t << "," << c << ")";
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c)
        EXPECT_NEAR(got3(b, c, t / W, t % W), htok3(b, t, c), 1e-6f)
            << " rank-3 at (" << b << "," << t << "," << c << ")";
  // Validation.
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_img("bad_img", C, H,
                                                            W + 1);
  EXPECT_THROW(KokkosDNN::unpatchify_forward(tok, bad_img, H, W),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::unpatchify_forward(tok, img, 0, W),
               std::invalid_argument);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_tok("bad_tok", T + 1, C);
  EXPECT_THROW(KokkosDNN::unpatchify_forward(bad_tok, img, H, W),
               std::invalid_argument);
}

// End-to-end: identity 1x1 patch embed followed by unpatchify recovers the
// VAE-latent images exactly (Sana patch_size=1 is a pure reshape under an
// identity projection).
TEST(SanaConvPatch, PatchUnpatchRoundTrip) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t B = 2, C = 4, H = 4, W = 4, T = H * W;
  Kokkos::View<float****, Kokkos::LayoutRight, Host> himg("himg", B, C, H, W);
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hwt("hwt", C, C);
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < C; ++c)
      for (size_t hh = 0; hh < H; ++hh)
        for (size_t w = 0; w < W; ++w)
          himg(b, c, hh, w) = probe(b * 97 + c * 13 + hh * W + w, 7);
  for (size_t i = 0; i < C; ++i)
    for (size_t j = 0; j < C; ++j) hwt(i, j) = (i == j) ? 1.0f : 0.0f;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> images("images", B, C, H,
                                                            W),
      back("back", B, C, H, W);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> weight("weight", C, C);
  Kokkos::View<float*, Exec> empty_b;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> tok("tok", B, T, C);
  Kokkos::deep_copy(images, himg);
  Kokkos::deep_copy(weight, hwt);
  KokkosDNN::patch_embed_1x1_forward(h, images, weight, empty_b, tok);
  KokkosDNN::unpatchify_forward(tok, back, H, W);
  Kokkos::fence();
  Kokkos::View<float****, Kokkos::LayoutRight, Host> got("got", B, C, H, W);
  Kokkos::deep_copy(got, back);
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < C; ++c)
      for (size_t hh = 0; hh < H; ++hh)
        for (size_t w = 0; w < W; ++w)
          EXPECT_NEAR(got(b, c, hh, w), himg(b, c, hh, w), 1e-5f)
              << " at (" << b << "," << c << "," << hh << "," << w << ")";
}
