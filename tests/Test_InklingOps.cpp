// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Inkling2 loop stateless ops vs naive FP32 host references (float + bhalf).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>
#include <vector>

#include "KokkosDNN_InklingConv.hpp"
#include "KokkosDNN_InklingGates.hpp"
#include "KokkosDNN_InklingNorm.hpp"
#include "KokkosDNN_InklingRelBias.hpp"
#include "KokkosDNN_InklingResidual.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;

template <class Scalar>
Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> make2(size_t a, size_t b,
                                                        unsigned seed) {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> v("v", a, b);
  Kokkos::Random_XorShift64_Pool<Exec> pool(seed);
  Kokkos::fill_random(v, pool, static_cast<Scalar>(-1), static_cast<Scalar>(1));
  return v;
}

template <class Scalar>
std::vector<std::vector<float>> to_host2(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec>& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  std::vector<std::vector<float>> o(h.extent(0),
                                    std::vector<float>(h.extent(1)));
  for (size_t i = 0; i < h.extent(0); ++i)
    for (size_t j = 0; j < h.extent(1); ++j)
      o[i][j] = static_cast<float>(h(i, j));
  return o;
}

template <class Scalar>
std::vector<float> to_host1(const Kokkos::View<Scalar*, Exec>& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  std::vector<float> o(h.extent(0));
  for (size_t i = 0; i < h.extent(0); ++i) o[i] = static_cast<float>(h(i));
  return o;
}

template <class Scalar>
Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> host_to_device2(
    const std::vector<std::vector<float>>& h) {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> v("v", h.size(),
                                                      h[0].size());
  auto m = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < h.size(); ++i)
    for (size_t j = 0; j < h[0].size(); ++j)
      m(i, j) = static_cast<Scalar>(h[i][j]);
  Kokkos::deep_copy(v, m);
  return v;
}

float rms_inv(const std::vector<float>& row, float eps) {
  double ss = 0;
  for (float v : row) ss += (double)v * v;
  return (float)(1.0 / std::sqrt(ss / row.size() + eps));
}

}  // namespace

template <class Scalar>
void check_inkling_norm(float tol) {
  const size_t T = 3, H = 2, D = 8;
  auto x2 = make2<Scalar>(T, D, 11);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> y2("y2", T, D);
  KokkosDNN::inkling_unweighted_rmsnorm<Exec, Scalar>(x2, 1e-6f, y2);
  Kokkos::fence();
  auto hx = to_host2(x2);
  auto hy = to_host2(y2);
  for (size_t t = 0; t < T; ++t) {
    const float inv = rms_inv(hx[t], 1e-6f);
    for (size_t d = 0; d < D; ++d)
      EXPECT_NEAR(hy[t][d], hx[t][d] * inv, tol) << t << " " << d;
  }

  // Rank-3 path.
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x3("x3", T, H, D),
      y3("y3", T, H, D);
  {
    auto m = Kokkos::create_mirror_view(x3);
    for (size_t t = 0; t < T; ++t)
      for (size_t h = 0; h < H; ++h)
        for (size_t d = 0; d < D; ++d)
          m(t, h, d) = static_cast<Scalar>(0.25f * ((t + h + d) % 5) - 0.5f);
    Kokkos::deep_copy(x3, m);
  }
  KokkosDNN::inkling_unweighted_rmsnorm_3d<Exec, Scalar>(x3, 1e-6f, y3);
  Kokkos::fence();
  auto hx3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x3);
  auto hy3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y3);
  for (size_t t = 0; t < T; ++t)
    for (size_t h = 0; h < H; ++h) {
      std::vector<float> row(D);
      for (size_t d = 0; d < D; ++d) row[d] = (float)hx3(t, h, d);
      const float inv = rms_inv(row, 1e-6f);
      for (size_t d = 0; d < D; ++d)
        EXPECT_NEAR((float)hy3(t, h, d), row[d] * inv, tol);
    }

  // L2Clip with rows above and below the threshold.
  {
    std::vector<std::vector<std::vector<float>>> xr(
        T, std::vector<std::vector<float>>(H, std::vector<float>(D)));
    for (size_t t = 0; t < T; ++t)
      for (size_t h = 0; h < H; ++h)
        for (size_t d = 0; d < D; ++d)
          xr[t][h][d] = (h == 0) ? 0.05f * (d + 1) : 3.0f * (d % 2 ? 1 : -1);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> xc("xc", T, H, D),
        yc("yc", T, H, D);
    auto m = Kokkos::create_mirror_view(xc);
    for (size_t t = 0; t < T; ++t)
      for (size_t h = 0; h < H; ++h)
        for (size_t d = 0; d < D; ++d) m(t, h, d) = (Scalar)xr[t][h][d];
    Kokkos::deep_copy(xc, m);
    const float thr = 2.0f;
    KokkosDNN::inkling_l2_clip<Exec, Scalar>(xc, thr, yc);
    // Scaled variant with unit scale must match.
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> sc("sc", H, D);
    {
      auto ms = Kokkos::create_mirror_view(sc);
      for (size_t h = 0; h < H; ++h)
        for (size_t d = 0; d < D; ++d) ms(h, d) = 1.0f + 0.1f * (h + d);
      Kokkos::deep_copy(sc, ms);
    }
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> ys("ys", T, H, D);
    KokkosDNN::inkling_scaled_l2_clip<Exec, Scalar>(xc, sc, thr, ys);
    // Log-scaled variant.
    Kokkos::View<int*, Exec> pos("pos", T);
    {
      auto mp = Kokkos::create_mirror_view(pos);
      mp(0) = 0;
      mp(1) = 100000;
      mp(2) = 7;
      Kokkos::deep_copy(pos, mp);
    }
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> yl("yl", T, H, D);
    KokkosDNN::inkling_log_scaled_l2_clip<Exec, Scalar>(xc, sc, thr, pos, 0.1f,
                                                       8192.0f, yl);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> yls("yls", T, H, D);
    KokkosDNN::inkling_log_scale<Exec, Scalar>(ys, pos, 0.1f, 8192.0f, yls);
    Kokkos::fence();
    auto hyc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), yc);
    auto hys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ys);
    auto hyl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), yl);
    auto hyls = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), yls);
    auto hsc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sc);
    auto hpos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos);
    for (size_t t = 0; t < T; ++t)
      for (size_t h = 0; h < H; ++h) {
        double n = 0;
        for (size_t d = 0; d < D; ++d) n += (double)xr[t][h][d] * xr[t][h][d];
        n = std::sqrt(n);
        const float c = std::fmin(1.0f, (float)(thr / (n + 1e-6)));
        double ns = 0;
        for (size_t d = 0; d < D; ++d) {
          const double v = xr[t][h][d] * hsc(h, d);
          ns += v * v;
        }
        ns = std::sqrt(ns);
        const float cs = std::fmin(1.0f, (float)(thr / (ns + 1e-6)));
        const float tau =
            1.0f + 0.1f * std::log(std::fmax(((float)hpos(t) + 1) / 8192.0f,
                                             1.0f));
        for (size_t d = 0; d < D; ++d) {
          EXPECT_NEAR((float)hyc(t, h, d), (float)(xr[t][h][d] * c), tol);
          EXPECT_NEAR((float)hys(t, h, d),
                      (float)(xr[t][h][d] * hsc(h, d) * cs), tol);
          EXPECT_NEAR((float)hyl(t, h, d),
                      (float)(xr[t][h][d] * hsc(h, d) * cs * tau), tol);
          // log_scale(scaled_l2_clip) must equal the fused op (bhalf
          // rounding across two different kernel reduction trees: 5e-3).
          EXPECT_NEAR((float)hyls(t, h, d), (float)hyl(t, h, d), 5e-3f);
        }
      }
  }
}

TEST(InklingNorm, Float) { check_inkling_norm<float>(1e-5f); }
// Note on BHalf tolerances below: the reference is computed in host order
// in FP64 while the kernels use backend-specific team-parallel reductions,
// so bhalf results (eps ~ 4e-3) legitimately differ by a few ulps across
// backends. 5e-3 confirms correct functionality on O(1) values everywhere;
// it is not freezing a rounding artifact of any single backend.
TEST(InklingNorm, BHalf) {
  // 5e-3: cross-backend reduction-order rounding on bhalf (see note above).
  check_inkling_norm<Kokkos::Experimental::bhalf_t>(5e-3f);
}

template <class Scalar>
void check_inkling_conv(float tol) {
  const size_t T = 6, D = 8, K = 4;
  auto x = make2<Scalar>(T, D, 21);
  auto w = make2<Scalar>(D, K, 22);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> y("y", T, D);
  KokkosDNN::inkling_short_conv_forward<Exec, Scalar>(x, w, y);
  Kokkos::fence();
  auto hx = to_host2(x);
  auto hw = to_host2(w);
  auto hy = to_host2(y);
  const float inv = 1.0f / std::sqrt((float)(K + 1));
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d) {
      double acc = 0;
      for (size_t k = 0; k < K; ++k) {
        const int s = (int)t - (int)(K - 1) + (int)k;
        if (s < 0) continue;
        acc += (double)hw[d][k] * hx[s][d];
      }
      EXPECT_NEAR(hy[t][d], (float)((acc + hx[t][d]) * inv), tol)
          << t << " " << d;
    }

  // Stateful decode: seed the state with the first T-1 tokens, then step
  // the final token and compare against the full-sequence output row.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> state("state", D, K - 1);
  {
    auto ms = Kokkos::create_mirror_view(state);
    for (size_t d = 0; d < D; ++d)
      for (size_t k = 0; k < K - 1; ++k) {
        const int s = (int)(T - 1) - (int)(K - 1) + (int)k;
        ms(d, k) = (s < 0) ? (Scalar)0 : (Scalar)hx[s][d];
      }
    Kokkos::deep_copy(state, ms);
  }
  Kokkos::View<Scalar*, Exec> xn("xn", D), yd("yd", D);
  {
    auto mx = Kokkos::create_mirror_view(xn);
    auto hxlast = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    for (size_t d = 0; d < D; ++d) mx(d) = (Scalar)hxlast(T - 1, d);
    Kokkos::deep_copy(xn, mx);
  }
  KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xn, state, w, yd);
  Kokkos::fence();
  auto hyd = to_host1(yd);
  for (size_t d = 0; d < D; ++d)
    EXPECT_NEAR(hyd[d], hy[T - 1][d], tol) << d;
  // In-place decode (x aliases y, as the Model calls it): the state append
  // must observe the input, not the just-written output. Re-run from the
  // same pre-state with aliased views and compare against the out-of-place
  // result above.
  {
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> state_pre("state_pre",
                                                                D, K - 1);
    {
      auto ms = Kokkos::create_mirror_view(state_pre);
      for (size_t d = 0; d < D; ++d)
        for (size_t k = 0; k < K - 1; ++k) {
          const int s = (int)(T - 1) - (int)(K - 1) + (int)k;
          ms(d, k) = (s < 0) ? (Scalar)0 : (Scalar)hx[s][d];
        }
      Kokkos::deep_copy(state_pre, ms);
    }
    Kokkos::View<Scalar*, Exec> xio("xio", D);
    Kokkos::deep_copy(xio, xn);
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> state_io("state_io", D,
                                                               K - 1);
    Kokkos::deep_copy(state_io, state_pre);
    KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xio, state_io,
                                                             w, xio);
    Kokkos::fence();
    auto hio = to_host1(xio);
    auto ms = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  state_io);
    auto ms_ref =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);
    for (size_t d = 0; d < D; ++d) {
      EXPECT_NEAR(hio[d], hyd[d], tol) << "in-place " << d;
      for (size_t k = 0; k < K - 1; ++k)
        EXPECT_FLOAT_EQ(static_cast<float>(ms(d, k)),
                        static_cast<float>(ms_ref(d, k)))
            << "in-place state " << d << " " << k;
    }
  }
  // State now holds the last K-1 inputs.
  {
    auto ms = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);
    for (size_t d = 0; d < D; ++d)
      for (size_t k = 0; k < K - 1; ++k) {
        const size_t s = (T - 1) - (K - 1) + k + 1;
        EXPECT_NEAR((float)ms(d, k), hx[s][d], 1e-6f) << d << " " << k;
      }
  }

  // Dynamic variant.
  const size_t H = 2, Dh = D / H;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> sw("sw", K, D);
  auto dyn = make2<Scalar>(T, H * K, 23);
  {
    auto m = Kokkos::create_mirror_view(sw);
    for (size_t k = 0; k < K; ++k)
      for (size_t d = 0; d < D; ++d)
        m(k, d) = 0.1f * ((k + d) % 3) - 0.1f;
    Kokkos::deep_copy(sw, m);
  }
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> ydyn("ydyn", T, D);
  KokkosDNN::inkling_dynamic_short_conv_forward<Exec, Scalar>(x, sw, dyn, H, Dh,
                                                             ydyn);
  Kokkos::fence();
  auto hsw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sw);
  auto hdyn = to_host2(dyn);
  auto hydyn = to_host2(ydyn);
  const float invd = 1.0f / std::sqrt((float)(2 * K + 1));
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < D; ++c) {
      const size_t h = c / Dh;
      double acc = 0;
      for (size_t k = 0; k < K; ++k) {
        const int s = (int)t - (int)(K - 1) + (int)k;
        if (s < 0) continue;
        acc += ((double)hsw(k, c) + hdyn[t][h * K + k]) * hx[s][c];
      }
      EXPECT_NEAR(hydyn[t][c], (float)((acc + hx[t][c]) * invd), tol)
          << t << " " << c;
    }
  // Stateful dynamic decode of the last token.
  Kokkos::View<Scalar*, Exec> dyn_row("dyn_row", H * K), ydd("ydd", D);
  {
    auto m = Kokkos::create_mirror_view(dyn_row);
    for (size_t i = 0; i < H * K; ++i) m(i) = (Scalar)hdyn[T - 1][i];
    Kokkos::deep_copy(dyn_row, m);
  }
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> state2("state2", D, K - 1);
  {
    auto ms = Kokkos::create_mirror_view(state2);
    for (size_t d = 0; d < D; ++d)
      for (size_t k = 0; k < K - 1; ++k) {
        const int s = (int)(T - 1) - (int)(K - 1) + (int)k;
        ms(d, k) = (s < 0) ? (Scalar)0 : (Scalar)hx[s][d];
      }
    Kokkos::deep_copy(state2, ms);
  }
  KokkosDNN::inkling_dynamic_short_conv_update_state<Exec, Scalar>(
      xn, state2, sw, dyn_row, H, Dh, ydd);
  Kokkos::fence();
  auto hydd = to_host1(ydd);
  for (size_t d = 0; d < D; ++d)
    EXPECT_NEAR(hydd[d], hydyn[T - 1][d], tol) << d;
  // In-place dynamic decode (x aliases y): same transition must result.
  {
    Kokkos::View<Scalar*, Exec> xio("xio", D);
    Kokkos::deep_copy(xio, xn);
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> state_io("state_io", D,
                                                               K - 1);
    {
      auto ms = Kokkos::create_mirror_view(state_io);
      for (size_t d = 0; d < D; ++d)
        for (size_t k = 0; k < K - 1; ++k) {
          const int s = (int)(T - 1) - (int)(K - 1) + (int)k;
          ms(d, k) = (s < 0) ? (Scalar)0 : (Scalar)hx[s][d];
        }
      Kokkos::deep_copy(state_io, ms);
    }
    KokkosDNN::inkling_dynamic_short_conv_update_state<Exec, Scalar>(
        xio, state_io, sw, dyn_row, H, Dh, xio);
    Kokkos::fence();
    auto hio = to_host1(xio);
    auto ms = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  state_io);
    auto ms_ref =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state2);
    for (size_t d = 0; d < D; ++d) {
      EXPECT_NEAR(hio[d], hydd[d], tol) << "in-place dyn " << d;
      for (size_t k = 0; k < K - 1; ++k)
        EXPECT_FLOAT_EQ(static_cast<float>(ms(d, k)),
                        static_cast<float>(ms_ref(d, k)))
            << "in-place dyn state " << d << " " << k;
    }
  }
}

TEST(InklingConv, Float) { check_inkling_conv<float>(1e-5f); }
TEST(InklingConv, BHalf) {
  // 5e-3: cross-backend reduction-order rounding on bhalf (see note above).
  check_inkling_conv<Kokkos::Experimental::bhalf_t>(5e-3f);
}

// Fused k+v single-token conv pairs vs the two single-side update_state
// calls: bit-identical outputs and states (same FP32 order per side), both
// out-of-place and in-place (x aliases y, as the Model calls them).
template <class Scalar>
void check_inkling_conv_pair() {
  const size_t D = 8, K = 4;
  auto xk = make2<Scalar>(1, D, 51);
  auto xv = make2<Scalar>(1, D, 52);
  auto wk = make2<Scalar>(D, K, 53);
  auto wv = make2<Scalar>(D, K, 54);
  Kokkos::View<Scalar*, Exec> xk1("xk1", D), xv1("xv1", D);
  {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xk);
    auto m = Kokkos::create_mirror_view(xk1);
    for (size_t d = 0; d < D; ++d) m(d) = h(0, d);
    Kokkos::deep_copy(xk1, m);
  }
  {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xv);
    auto m = Kokkos::create_mirror_view(xv1);
    for (size_t d = 0; d < D; ++d) m(d) = h(0, d);
    Kokkos::deep_copy(xv1, m);
  }
  auto seed_state = [&](unsigned s) {
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> st("st", D, K - 1);
    Kokkos::Random_XorShift64_Pool<Exec> pool(s);
    Kokkos::fill_random(st, pool, static_cast<Scalar>(-1),
                        static_cast<Scalar>(1));
    return st;
  };
  auto sk_ref = seed_state(55);
  auto sv_ref = seed_state(56);
  auto sk_fused = seed_state(55);
  auto sv_fused = seed_state(56);
  Kokkos::View<Scalar*, Exec> yk_ref("yk_ref", D), yv_ref("yv_ref", D),
      yk_f("yk_f", D), yv_f("yv_f", D);
  KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xk1, sk_ref, wk,
                                                           yk_ref);
  KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xv1, sv_ref, wv,
                                                           yv_ref);
  KokkosDNN::inkling_static_conv_update_state_pair<Exec, Scalar>(
      xk1, sk_fused, wk, yk_f, xv1, sv_fused, wv, yv_f);
  Kokkos::fence();
  auto hk_ref = to_host1(yk_ref);
  auto hv_ref = to_host1(yv_ref);
  auto hk_f = to_host1(yk_f);
  auto hv_f = to_host1(yv_f);
  auto hsk =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sk_fused);
  auto hsv =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sv_fused);
  auto hsk_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sk_ref);
  auto hsv_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sv_ref);
  for (size_t d = 0; d < D; ++d) {
    EXPECT_FLOAT_EQ(hk_f[d], hk_ref[d]) << d;
    EXPECT_FLOAT_EQ(hv_f[d], hv_ref[d]) << d;
    for (size_t k = 0; k < K - 1; ++k) {
      EXPECT_FLOAT_EQ(static_cast<float>(hsk(d, k)),
                      static_cast<float>(hsk_ref(d, k)))
          << d << " " << k;
      EXPECT_FLOAT_EQ(static_cast<float>(hsv(d, k)),
                      static_cast<float>(hsv_ref(d, k)))
          << d << " " << k;
    }
  }
  // In-place (x aliases y on each side): pair must match aliased singles.
  {
    Kokkos::View<Scalar*, Exec> xio_k("xio_k", D), xio_v("xio_v", D);
    Kokkos::deep_copy(xio_k, xk1);
    Kokkos::deep_copy(xio_v, xv1);
    auto sk2 = seed_state(55);
    auto sv2 = seed_state(56);
    KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xio_k, sk2, wk,
                                                             xio_k);
    KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xio_v, sv2, wv,
                                                             xio_v);
    Kokkos::View<Scalar*, Exec> xio_kf("xio_kf", D), xio_vf("xio_vf", D);
    Kokkos::deep_copy(xio_kf, xk1);
    Kokkos::deep_copy(xio_vf, xv1);
    auto sk2f = seed_state(55);
    auto sv2f = seed_state(56);
    KokkosDNN::inkling_static_conv_update_state_pair<Exec, Scalar>(
        xio_kf, sk2f, wk, xio_kf, xio_vf, sv2f, wv, xio_vf);
    Kokkos::fence();
    auto hk_io = to_host1(xio_k);
    auto hv_io = to_host1(xio_v);
    auto hk_iof = to_host1(xio_kf);
    auto hv_iof = to_host1(xio_vf);
    for (size_t d = 0; d < D; ++d) {
      EXPECT_FLOAT_EQ(hk_iof[d], hk_io[d]) << d;
      EXPECT_FLOAT_EQ(hv_iof[d], hv_io[d]) << d;
    }
  }
  // K == 1 (stateless): pair matches the two singles.
  {
    const size_t K1 = 1;
    auto w1k = make2<Scalar>(D, K1, 57);
    auto w1v = make2<Scalar>(D, K1, 58);
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> s1k("s1k", D, 0),
        s1v("s1v", D, 0), s1kf("s1kf", D, 0), s1vf("s1vf", D, 0);
    Kokkos::View<Scalar*, Exec> a("a", D), b("b", D), af("af", D),
        bf("bf", D);
    KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xk1, s1k, w1k,
                                                             a);
    KokkosDNN::inkling_short_conv_update_state<Exec, Scalar>(xv1, s1v, w1v,
                                                             b);
    KokkosDNN::inkling_static_conv_update_state_pair<Exec, Scalar>(
        xk1, s1kf, w1k, af, xv1, s1vf, w1v, bf);
    Kokkos::fence();
    auto ha = to_host1(a);
    auto hb = to_host1(b);
    auto haf = to_host1(af);
    auto hbf = to_host1(bf);
    for (size_t d = 0; d < D; ++d) {
      EXPECT_FLOAT_EQ(haf[d], ha[d]) << d;
      EXPECT_FLOAT_EQ(hbf[d], hb[d]) << d;
    }
  }
  // Dynamic pair vs the two dynamic singles.
  {
    const size_t H = 2, Dh = D / H;
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> swk("swk", K, D),
        swv("swv", K, D);
    {
      auto m = Kokkos::create_mirror_view(swk);
      auto n = Kokkos::create_mirror_view(swv);
      for (size_t k = 0; k < K; ++k)
        for (size_t d = 0; d < D; ++d) {
          m(k, d) = 0.1f * ((k + d) % 3) - 0.1f;
          n(k, d) = 0.1f * ((k * d) % 3) - 0.15f;
        }
      Kokkos::deep_copy(swk, m);
      Kokkos::deep_copy(swv, n);
    }
    Kokkos::View<Scalar*, Exec> dk("dk", H * K), dv("dv", H * K);
    {
      auto m = Kokkos::create_mirror_view(dk);
      auto n = Kokkos::create_mirror_view(dv);
      for (size_t i = 0; i < H * K; ++i) {
        m(i) = static_cast<Scalar>(0.2f * (i % 4) - 0.3f);
        n(i) = static_cast<Scalar>(0.15f * (i % 5) - 0.2f);
      }
      Kokkos::deep_copy(dk, m);
      Kokkos::deep_copy(dv, n);
    }
    auto sk3 = seed_state(59);
    auto sv3 = seed_state(60);
    auto sk3f = seed_state(59);
    auto sv3f = seed_state(60);
    Kokkos::View<Scalar*, Exec> ydk("ydk", D), ydv("ydv", D), ydkf("ydkf", D),
        ydvf("ydvf", D);
    KokkosDNN::inkling_dynamic_short_conv_update_state<Exec, Scalar>(
        xk1, sk3, swk, dk, H, Dh, ydk);
    KokkosDNN::inkling_dynamic_short_conv_update_state<Exec, Scalar>(
        xv1, sv3, swv, dv, H, Dh, ydv);
    KokkosDNN::inkling_dynamic_short_conv_update_state_pair<Exec, Scalar>(
        xk1, sk3f, swk, dk, ydkf, xv1, sv3f, swv, dv, ydvf, H, Dh);
    Kokkos::fence();
    auto hk3 = to_host1(ydk);
    auto hv3 = to_host1(ydv);
    auto hk3f = to_host1(ydkf);
    auto hv3f = to_host1(ydvf);
    auto hsk3 =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sk3f);
    auto hsk3r =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sk3);
    for (size_t d = 0; d < D; ++d) {
      EXPECT_FLOAT_EQ(hk3f[d], hk3[d]) << d;
      EXPECT_FLOAT_EQ(hv3f[d], hv3[d]) << d;
      for (size_t k = 0; k < K - 1; ++k)
        EXPECT_FLOAT_EQ(static_cast<float>(hsk3(d, k)),
                        static_cast<float>(hsk3r(d, k)))
            << d << " " << k;
    }
  }
}

TEST(InklingConvPair, Float) { check_inkling_conv_pair<float>(); }
TEST(InklingConvPair, BHalf) {
  check_inkling_conv_pair<Kokkos::Experimental::bhalf_t>();
}

template <class Scalar>
void check_inkling_relbias(float tol) {
  const size_t Tq = 3, Tk = 4, H = 2, R = 4, E = 6;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> rs("rs", Tq, H, R);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> proj("proj", R, E);
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> dist("dist", Tq, Tk);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bias("bias", Tq, Tk, H);
  {
    auto m = Kokkos::create_mirror_view(rs);
    for (size_t q = 0; q < Tq; ++q)
      for (size_t h = 0; h < H; ++h)
        for (size_t r = 0; r < R; ++r)
          m(q, h, r) = (Scalar)(0.2f * ((q + h + r) % 4) - 0.3f);
    Kokkos::deep_copy(rs, m);
  }
  {
    auto m = Kokkos::create_mirror_view(proj);
    for (size_t r = 0; r < R; ++r)
      for (size_t e = 0; e < E; ++e) m(r, e) = 0.1f * ((r + e) % 5) - 0.2f;
    Kokkos::deep_copy(proj, m);
  }
  // distances: q - k with kv offset 1 -> some negative, some >= E.
  {
    auto m = Kokkos::create_mirror_view(dist);
    for (size_t q = 0; q < Tq; ++q)
      for (size_t k = 0; k < Tk; ++k) m(q, k) = (int)(q + 2) - (int)k - 1;
    Kokkos::deep_copy(dist, m);
  }
  KokkosDNN::inkling_relative_position_bias<Exec, Scalar>(rs, proj, dist, E,
                                                         bias);
  Kokkos::fence();
  auto hrs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rs);
  auto hproj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), proj);
  auto hdist = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dist);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bias);
  for (size_t q = 0; q < Tq; ++q)
    for (size_t k = 0; k < Tk; ++k)
      for (size_t h = 0; h < H; ++h) {
        const int d = hdist(q, k);
        float want = 0;
        if (d >= 0 && d < (int)E) {
          double acc = 0;
          for (size_t r = 0; r < R; ++r)
            acc += (double)(float)hrs(q, h, r) * hproj(r, (size_t)d);
          want = (float)acc;
        }
        EXPECT_NEAR((float)hb(q, k, h), want, tol) << q << k << h;
      }
}

TEST(InklingRelBias, Float) { check_inkling_relbias<float>(1e-5f); }
TEST(InklingRelBias, BHalf) {
  check_inkling_relbias<Kokkos::Experimental::bhalf_t>(1e-3f);
}

template <class Scalar>
void check_inkling_gates(float tol) {
  const size_t T = 3, F = 8, D = 8, R = 4;
  auto gate = make2<Scalar>(T, F, 31);
  auto up = make2<Scalar>(T, F, 32);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> capped("capped", T, F),
      ssw("ssw", T, F);
  KokkosDNN::inkling_softcap<Exec, Scalar>(gate, 4.0f, capped);
  KokkosDNN::inkling_softcapped_swiglu<Exec, Scalar>(gate, up, 4.0f, 25.0f,
                                                    ssw);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> attn("attn", T, D),
      gg("gg", T, D), gout("gout", T, D);
  {
    auto m = Kokkos::create_mirror_view(attn);
    auto n = Kokkos::create_mirror_view(gg);
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        m(t, d) = (Scalar)(0.3f * ((t + d) % 4) - 0.4f);
        n(t, d) = (Scalar)(0.5f * ((t * d) % 3) - 0.5f);
      }
    Kokkos::deep_copy(attn, m);
    Kokkos::deep_copy(gg, n);
  }
  KokkosDNN::inkling_gated_output_mul<Exec, Scalar>(attn, gg, gout);
  Kokkos::fence();
  auto hgate = to_host2(gate);
  auto hup = to_host2(up);
  auto hcapped = to_host2(capped);
  auto hssw = to_host2(ssw);
  for (size_t t = 0; t < T; ++t)
    for (size_t f = 0; f < F; ++f) {
      const float g = hgate[t][f], u = hup[t][f];
      EXPECT_NEAR(hcapped[t][f], std::tanh(g / 4.0f) * 4.0f, tol);
      const float want = std::tanh(g / 4.0f) * 4.0f *
                         (1.0f / (1.0f + std::exp(-g))) *
                         (std::tanh(u / 25.0f) * 25.0f);
      EXPECT_NEAR(hssw[t][f], want, tol * 8) << t << " " << f;
    }
  auto hattn = to_host2(attn);
  auto hgg = to_host2(gg);
  auto hgout = to_host2(gout);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d) {
      const float sig = 1.0f / (1.0f + std::exp(-hgg[t][d]));
      EXPECT_NEAR(hgout[t][d], hattn[t][d] * sig, tol);
    }

  // MessageGate vs host reference.
  auto x = make2<Scalar>(T, D, 33);
  auto msg = make2<Scalar>(T, D, 34);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> down("down", 2 * D, R),
      forget("forget", R, D), gain("gain", R, D);
  {
    auto a = Kokkos::create_mirror_view(down);
    auto b = Kokkos::create_mirror_view(forget);
    auto c = Kokkos::create_mirror_view(gain);
    for (size_t i = 0; i < 2 * D; ++i)
      for (size_t r = 0; r < R; ++r) a(i, r) = 0.1f * ((i + r) % 5) - 0.2f;
    for (size_t r = 0; r < R; ++r)
      for (size_t d = 0; d < D; ++d) {
        b(r, d) = 0.15f * ((r + d) % 4) - 0.2f;
        c(r, d) = 0.12f * ((r * d) % 5) - 0.25f;
      }
    Kokkos::deep_copy(down, a);
    Kokkos::deep_copy(forget, b);
    Kokkos::deep_copy(gain, c);
  }
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> mout("mout", T, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> rms_x("rms_x", T, D),
      rms_m("rms_m", T, D);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> mh("mh", T, R);
  KokkosDNN::inkling_message_gate<Exec, Scalar>(x, msg, down, forget, gain,
                                               1e-6f, mout, rms_x, rms_m,
                                               mh);
  Kokkos::fence();
  auto hx = to_host2(x);
  auto hmsg = to_host2(msg);
  auto hdown = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), down);
  auto hforget =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forget);
  auto hgain = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gain);
  auto hmout = to_host2(mout);
  for (size_t t = 0; t < T; ++t) {
    const float invx = rms_inv(hx[t], 1e-6f), invm = rms_inv(hmsg[t], 1e-6f);
    for (size_t d = 0; d < D; ++d) {
      double h[4] = {0, 0, 0, 0};
      for (size_t r = 0; r < R; ++r)
        for (size_t dd = 0; dd < D; ++dd)
          h[r] += hx[t][dd] * invx * hdown(dd, r) +
                  hmsg[t][dd] * invm * hdown(D + dd, r);
      double f = 0, g = 0;
      for (size_t r = 0; r < R; ++r) {
        f += hforget(r, d) * h[r];
        g += hgain(r, d) * h[r];
      }
      const float want = (float)((1.0 - std::tanh(f)) * hx[t][d] +
                                 std::tanh(g) * hmsg[t][d]);
      EXPECT_NEAR(hmout[t][d], want, tol * 4) << t << " " << d;
    }
  }
}

TEST(InklingGates, Float) { check_inkling_gates<float>(1e-5f); }
TEST(InklingGates, BHalf) {
  // 5e-3: cross-backend reduction-order rounding on bhalf (see note above).
  check_inkling_gates<Kokkos::Experimental::bhalf_t>(5e-3f);
}

template <class Scalar>
void check_inkling_residual(float tol, size_t Q) {
  const size_t T = 2, S = 3, D = 8;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> blocks("blocks", T, S, D);
  auto partial = make2<Scalar>(T, D, 41);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> query("query", Q, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", T, Q, D);
  {
    auto m = Kokkos::create_mirror_view(blocks);
    for (size_t t = 0; t < T; ++t)
      for (size_t s = 0; s < S; ++s)
        for (size_t d = 0; d < D; ++d)
          m(t, s, d) = (Scalar)(0.2f * ((t + s + d) % 5) - 0.4f);
    Kokkos::deep_copy(blocks, m);
    auto q = Kokkos::create_mirror_view(query);
    for (size_t qq = 0; qq < Q; ++qq)
      for (size_t d = 0; d < D; ++d) q(qq, d) = 0.1f * ((qq + d) % 4) - 0.15f;
    Kokkos::deep_copy(query, q);
  }
  KokkosDNN::inkling_block_attention_residual<Exec, Scalar>(blocks, partial,
                                                           query, 1e-6f, out);
  Kokkos::fence();
  auto hblocks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     blocks);
  auto hpart = to_host2(partial);
  auto hq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), query);
  auto hout = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  const float scale = 1.0f / std::sqrt((float)D);
  for (size_t t = 0; t < T; ++t) {
    // Stacked + normalized rows.
    std::vector<std::vector<float>> nrm(
        S + 1, std::vector<float>(D));
    for (size_t s = 0; s <= S; ++s) {
      std::vector<float> row(D);
      for (size_t d = 0; d < D; ++d)
        row[d] = (s < S) ? (float)hblocks(t, s, d) : hpart[t][d];
      const float inv = rms_inv(row, 1e-6f);
      for (size_t d = 0; d < D; ++d) nrm[s][d] = row[d] * inv;
    }
    for (size_t q = 0; q < Q; ++q) {
      std::vector<float> scr(S + 1);
      float m = -1e30f;
      for (size_t s = 0; s <= S; ++s) {
        double a = 0;
        for (size_t d = 0; d < D; ++d) a += hq(q, d) * nrm[s][d];
        scr[s] = (float)(a * scale);
        m = std::max(m, scr[s]);
      }
      double denom = 0;
      for (size_t s = 0; s <= S; ++s) {
        scr[s] = std::exp(scr[s] - m);
        denom += scr[s];
      }
      for (size_t d = 0; d < D; ++d) {
        double acc = 0;
        for (size_t s = 0; s <= S; ++s) acc += scr[s] / denom * nrm[s][d];
        EXPECT_NEAR((float)hout(t, q, d), (float)acc, tol) << t << q << d;
      }
    }
  }

  // Message variant: table [L, 4, D], row loop_layer_idx.
  const size_t L = 3, idx = 1;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> table("table", L, 4, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> mout("mout", T, 4, D);
  {
    auto m = Kokkos::create_mirror_view(table);
    for (size_t l = 0; l < L; ++l)
      for (size_t q = 0; q < 4; ++q)
        for (size_t d = 0; d < D; ++d)
          m(l, q, d) = 0.1f * ((l + q + d) % 4) - 0.15f;
    Kokkos::deep_copy(table, m);
  }
  KokkosDNN::inkling_message_attention_residual<Exec, Scalar>(
      blocks, partial, table, idx, 1e-6f, mout);
  Kokkos::fence();
  auto htable = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), table);
  auto hmout = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mout);
  for (size_t t = 0; t < T; ++t) {
    std::vector<std::vector<float>> nrm(
        S + 1, std::vector<float>(D));
    for (size_t s = 0; s <= S; ++s) {
      std::vector<float> row(D);
      for (size_t d = 0; d < D; ++d)
        row[d] = (s < S) ? (float)hblocks(t, s, d) : hpart[t][d];
      const float inv = rms_inv(row, 1e-6f);
      for (size_t d = 0; d < D; ++d) nrm[s][d] = row[d] * inv;
    }
    for (size_t q = 0; q < 4; ++q) {
      std::vector<float> scr(S + 1);
      float m = -1e30f;
      for (size_t s = 0; s <= S; ++s) {
        double a = 0;
        for (size_t d = 0; d < D; ++d) a += htable(idx, q, d) * nrm[s][d];
        scr[s] = (float)(a * scale);
        m = std::max(m, scr[s]);
      }
      double denom = 0;
      for (size_t s = 0; s <= S; ++s) {
        scr[s] = std::exp(scr[s] - m);
        denom += scr[s];
      }
      for (size_t d = 0; d < D; ++d) {
        double acc = 0;
        for (size_t s = 0; s <= S; ++s) acc += scr[s] / denom * nrm[s][d];
        EXPECT_NEAR((float)hmout(t, q, d), (float)acc, tol) << t << q << d;
      }
    }
  }
}

TEST(InklingResidual, FloatQ1) { check_inkling_residual<float>(1e-5f, 1); }
TEST(InklingResidual, FloatQ3) { check_inkling_residual<float>(1e-5f, 3); }
TEST(InklingResidual, BHalfQ1) {
  // 5e-3: cross-backend reduction-order rounding on bhalf (see note above).
  check_inkling_residual<Kokkos::Experimental::bhalf_t>(5e-3f, 1);
}
TEST(InklingResidual, BHalfQ3) {
  // 5e-3: cross-backend reduction-order rounding on bhalf (see note above).
  check_inkling_residual<Kokkos::Experimental::bhalf_t>(5e-3f, 3);
}
