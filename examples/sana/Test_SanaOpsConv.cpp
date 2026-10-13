// Sana fused GLUMBConv MixFFN vs composed host pipeline (rank-2,
// unbiased-depthwise variant, batched ratio config, argument validation).
// Model-specific stitching; lives with the Sana example (see
// KokkosDNN_SanaOps.hpp), not the core library.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>

#include "KokkosDNN.hpp"
#include "KokkosDNN_SanaOps.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::HostSpace;

float probe(size_t a, size_t b, size_t c = 0) {
  return 0.06f * static_cast<float>((a * 31 + b * 17 + c * 13) % 19) - 0.5f;
}

float host_silu(float x) { return x / (1.0f + std::exp(-x)); }

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

TEST(SanaOpsGLUMB, Rank2MatchesComposed) {
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
  KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H, W, y);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, Cout),
      ref("ref", T, Cout);
  Kokkos::deep_copy(got, y);
  host_glumb_ref(hx, hiw, hib, true, hdw, hdb, true, hpw, ref, H, W);
  expect_near_2d(got, ref, 1e-4f);
}

TEST(SanaOpsGLUMB, Rank2UnbiasedDepthwiseMatchesComposed) {
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
  KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, empty_b, pt_w, H, W,
                               y);
  Kokkos::fence();
  Kokkos::View<float**, Kokkos::LayoutRight, Host> got("got", T, Cout),
      ref("ref", T, Cout);
  Kokkos::deep_copy(got, y);
  Kokkos::View<float*, Host> hdb;
  host_glumb_ref(hx, hiw, hib, true, hdw, hdb, false, hpw, ref, H, W);
  expect_near_2d(got, ref, 1e-4f);
}

TEST(SanaOpsGLUMB, Rank3SanaRatioMatchesComposed) {
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
  KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H, W, y);
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

TEST(SanaOpsGLUMB, RejectsBadArgs) {
  KokkosDNN::DNNHandle<Exec> h;
  const size_t H = 3, W = 3, T = H * W, Cin = 4, Hid = 3, Cout = 5;
  const size_t inv_out = 2 * Hid;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, Cin),
      inv_w("inv_w", inv_out, Cin), pt_w("pt_w", Cout, Hid), y("y", T, Cout);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> dw_w("dw_w", inv_out, 3, 3);
  Kokkos::View<float*, Exec> inv_b("inv_b", inv_out), dw_b("dw_b", inv_out),
      empty_b;
  EXPECT_NO_THROW(
      KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H, W, y));
  EXPECT_NO_THROW(KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, empty_b, dw_w,
                                               empty_b, pt_w, H, W, y));
  EXPECT_THROW(
      KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, H,
                                   W + 1, y),
      std::invalid_argument);  // T != H*W
  EXPECT_THROW(
      KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, pt_w, 0, W, y),
      std::invalid_argument);  // zero H
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> odd_inv("odd_inv", inv_out + 1,
                                                           Cin);
  EXPECT_THROW(KokkosDNN::Sana::glumbconv_forward(h, x, odd_inv, empty_b, dw_w,
                                            empty_b, pt_w, H, W, y),
               std::invalid_argument);  // odd inverted width
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_dw("bad_dw", inv_out, 3,
                                                           3);
  // bad_dw has the right shape here; make a wrong-channel one:
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_dw_c("bad_dw_c",
                                                             inv_out + 2, 3, 3);
  EXPECT_THROW(KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, bad_dw_c, dw_b,
                                            pt_w, H, W, y),
               std::invalid_argument);
  (void)bad_dw;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_pt("bad_pt", Cout,
                                                          Hid + 1);
  EXPECT_THROW(
      KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b, bad_pt, H, W, y),
      std::invalid_argument);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_y("bad_y", T, Cout + 1);
  EXPECT_THROW(KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, inv_b, dw_w, dw_b,
                                            pt_w, H, W, bad_y),
               std::invalid_argument);
  Kokkos::View<float*, Exec> bad_ib("bad_ib", inv_out + 1);
  EXPECT_THROW(KokkosDNN::Sana::glumbconv_forward(h, x, inv_w, bad_ib, dw_w, dw_b,
                                            pt_w, H, W, y),
               std::invalid_argument);
}
