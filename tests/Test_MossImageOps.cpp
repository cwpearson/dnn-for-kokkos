// MOSS-Image ops vs host reference: shift-aware timestep embedding,
// fused RMSNorm+AdaLN-zero modulation, gated residuals, batched QK RMSNorm,
// 4-coordinate RoPE build/apply, joint attention wrapper, CFG/SLG
// combiners, VAE denormalize/depth-to-space, SiTU activation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <vector>

#include "KokkosDNN.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using bhalf = Kokkos::Experimental::bhalf_t;

TEST(MossTimestepShifted, Float) {
  const int N = 4, F = 8;
  Kokkos::View<float*, Exec> t("t", N);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> out("out", N, F);
  auto ht = Kokkos::create_mirror_view(t);
  ht(0) = 0.0f;
  ht(1) = 0.25f;
  ht(2) = 0.5f;
  ht(3) = 1.0f;
  Kokkos::deep_copy(t, ht);
  const float shift = 5.0f;
  KokkosDNN::MossImage::moss_timestep_shifted<Exec>(t, shift, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  const int H = F / 2;
  for (int i = 0; i < N; ++i) {
    double ts = shift * ht(i) / (1.0 + (shift - 1.0) * ht(i));
    for (int j = 0; j < H; ++j) {
      double freq = std::exp(-std::log(10000.0) * j / H);
      double arg = ts * freq;
      EXPECT_NEAR(got(i, j), std::cos(arg), 1e-5)
          << "cos mismatch at (" << i << "," << j << ")";
      EXPECT_NEAR(got(i, j + H), std::sin(arg), 1e-5)
          << "sin mismatch at (" << i << "," << j << ")";
    }
  }
  // t=0 maps to 0 -> cos row all ones, sin row all zeros.
  for (int j = 0; j < H; ++j) {
    EXPECT_NEAR(got(0, j), 1.0, 1e-6);
    EXPECT_NEAR(got(0, j + H), 0.0, 1e-6);
  }
  EXPECT_THROW(
      KokkosDNN::MossImage::moss_timestep_shifted<Exec>(t, 0.0f, out),
      std::invalid_argument);
}

template <class Scalar>
void check_rmsnorm_modulate(float tol) {
  const int B = 2, T = 3, D = 8;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D),
      y("y", B, T, D);
  Kokkos::View<Scalar*, Exec> w("w", D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> shift("sh", B, D),
      scale("sc", B, D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hsh = Kokkos::create_mirror_view(shift);
  auto hsc = Kokkos::create_mirror_view(scale);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d)
        hx(b, t, d) =
            static_cast<Scalar>(0.2f * b - 0.15f * t + 0.1f * d - 0.3f);
  for (int d = 0; d < D; ++d) hw(d) = static_cast<Scalar>(1.0f + 0.05f * d);
  for (int b = 0; b < B; ++b)
    for (int d = 0; d < D; ++d) {
      hsh(b, d) = static_cast<Scalar>(0.05f * b + 0.1f * d);
      hsc(b, d) = static_cast<Scalar>(-0.03f * b + 0.02f * d);
    }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(shift, hsh);
  Kokkos::deep_copy(scale, hsc);
  const float eps = 1e-6f;
  KokkosDNN::MossImage::moss_rmsnorm_modulate_forward<Exec, Scalar>(
      x, w, eps, shift, scale, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double v = static_cast<double>(hx(b, t, d));
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d) {
        double n = static_cast<double>(hx(b, t, d)) * inv *
                   static_cast<double>(hw(d));
        double want = n * (1.0 + static_cast<double>(hsc(b, d))) +
                      static_cast<double>(hsh(b, d));
        EXPECT_NEAR(static_cast<double>(got(b, t, d)), want, tol)
            << "mismatch at (" << b << "," << t << "," << d << ")";
      }
    }
}

TEST(MossRmsnormModulate, Float) { check_rmsnorm_modulate<float>(1e-5f); }
TEST(MossRmsnormModulate, BHalf) {
  check_rmsnorm_modulate<bhalf>(5e-2f);
}

TEST(MossGateResidual, Float) {
  const int B = 2, T = 3, D = 8;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", B, T, D),
      br("br", B, T, D), out("out", B, T, D);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> gate("g", B, D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hbr = Kokkos::create_mirror_view(br);
  auto hg = Kokkos::create_mirror_view(gate);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d) {
        hx(b, t, d) = 0.1f * b - 0.2f * t + 0.3f * d;
        hbr(b, t, d) = -0.4f * b + 0.1f * t - 0.05f * d;
      }
  for (int b = 0; b < B; ++b)
    for (int d = 0; d < D; ++d) hg(b, d) = 0.5f * b + 0.01f * d;
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(br, hbr);
  Kokkos::deep_copy(gate, hg);
  KokkosDNN::MossImage::moss_gate_residual_forward<Exec, float>(x, br, gate,
                                                                out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(got(b, t, d),
                    hx(b, t, d) + hg(b, d) * hbr(b, t, d), 1e-6f)
            << "mismatch at (" << b << "," << t << "," << d << ")";
}

template <class Scalar>
void check_qk_rmsnorm(float tol) {
  const int B = 2, S = 5, H = 2, Dh = 8;
  Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> q("q", B, S, H, Dh),
      out("o", B, S, H, Dh);
  Kokkos::View<Scalar*, Exec> w("w", Dh);
  auto hq = Kokkos::create_mirror_view(q);
  auto hw = Kokkos::create_mirror_view(w);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < Dh; ++d)
          hq(b, s, h, d) = static_cast<Scalar>(0.13f * (b + 1) -
                                               0.07f * (s + 1) +
                                               0.11f * (h + 1) + 0.05f * d);
  for (int d = 0; d < Dh; ++d) hw(d) = static_cast<Scalar>(0.9f + 0.03f * d);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(w, hw);
  const float eps = 1e-6f;
  KokkosDNN::MossImage::moss_qk_rmsnorm_forward<Exec, Scalar>(q, w, eps, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h) {
        double ss = 0;
        for (int d = 0; d < Dh; ++d) {
          double v = static_cast<double>(hq(b, s, h, d));
          ss += v * v;
        }
        double inv = 1.0 / std::sqrt(ss / Dh + eps);
        for (int d = 0; d < Dh; ++d) {
          double want = static_cast<double>(hq(b, s, h, d)) * inv *
                        static_cast<double>(hw(d));
          EXPECT_NEAR(static_cast<double>(got(b, s, h, d)), want, tol)
              << "mismatch at (" << b << "," << s << "," << h << "," << d
              << ")";
        }
      }
}

TEST(MossQkRmsnorm, Float) { check_qk_rmsnorm<float>(1e-5f); }
TEST(MossQkRmsnorm, BHalf) { check_qk_rmsnorm<bhalf>(5e-2f); }

// Fused QK-norm+RoPE must match the unfused norm->rope sequence to
// tolerance (the write-back keeps the same Scalar rounding of the
// normalized values and FP32 op order, but the team-parallel reductions
// may reorder summation at ~ulp level). In-place fused (as the blocks call
// it) vs out-of-place unfused.
template <class Scalar>
void check_qk_rmsnorm_rope_fused(float tol) {
  const int B = 2, S = 5, H = 2, Dh = 8;
  Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> q("q", B, S, H, Dh),
      k("k", B, S, H, Dh), qf("qf", B, S, H, Dh), kf("kf", B, S, H, Dh),
      qn("qn", B, S, H, Dh), kn("kn", B, S, H, Dh), rq("rq", B, S, H, Dh),
      rk("rk", B, S, H, Dh);
  Kokkos::View<Scalar*, Exec> qw("qw", Dh), kw("kw", Dh);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> rope("rope", 1, S, Dh);
  auto hq = Kokkos::create_mirror_view(q);
  auto hqw = Kokkos::create_mirror_view(qw);
  auto hkw = Kokkos::create_mirror_view(kw);
  auto hk = Kokkos::create_mirror_view(k);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < Dh; ++d) {
          hq(b, s, h, d) = static_cast<Scalar>(0.31f * (b + 1) -
                                               0.17f * (s + 1) +
                                               0.11f * (h + 1) + 0.05f * d);
          hk(b, s, h, d) = static_cast<Scalar>(-0.23f * (b + 1) +
                                               0.19f * (s + 1) -
                                               0.07f * (h + 1) - 0.03f * d);
        }
  for (int d = 0; d < Dh; ++d) {
    hqw(d) = static_cast<Scalar>(0.9f + 0.03f * d);
    hkw(d) = static_cast<Scalar>(1.1f - 0.02f * d);
  }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(qf, hq);
  Kokkos::deep_copy(kf, hk);
  Kokkos::deep_copy(qw, hqw);
  Kokkos::deep_copy(kw, hkw);
  KokkosDNN::MossImage::moss_rope_build_forward<Exec>(1, S, 0, 10000.0f, rope);
  const float eps = 1e-6f;
  // Unfused reference sequence, out of place.
  KokkosDNN::MossImage::moss_qk_rmsnorm_forward<Exec, Scalar>(q, qw, eps, qn);
  KokkosDNN::MossImage::moss_qk_rmsnorm_forward<Exec, Scalar>(k, kw, eps, kn);
  KokkosDNN::MossImage::moss_rope_apply_forward<Exec, Scalar>(qn, rope, rq);
  KokkosDNN::MossImage::moss_rope_apply_forward<Exec, Scalar>(kn, rope, rk);
  // Fused, in place (the DiT call pattern).
  KokkosDNN::MossImage::moss_qk_rmsnorm_rope_forward<Exec, Scalar>(
      qf, qw, kf, kw, eps, rope, qf, kf);
  Kokkos::fence();
  auto grq = Kokkos::create_mirror_view(rq);
  auto grk = Kokkos::create_mirror_view(rk);
  auto gqf = Kokkos::create_mirror_view(qf);
  auto gkf = Kokkos::create_mirror_view(kf);
  Kokkos::deep_copy(grq, rq);
  Kokkos::deep_copy(grk, rk);
  Kokkos::deep_copy(gqf, qf);
  Kokkos::deep_copy(gkf, kf);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < Dh; ++d) {
          EXPECT_NEAR(static_cast<double>(gqf(b, s, h, d)),
                      static_cast<double>(grq(b, s, h, d)), tol)
              << "q mismatch at (" << b << "," << s << "," << h << "," << d
              << ")";
          EXPECT_NEAR(static_cast<double>(gkf(b, s, h, d)),
                      static_cast<double>(grk(b, s, h, d)), tol)
              << "k mismatch at (" << b << "," << s << "," << h << "," << d
              << ")";
        }
}

TEST(MossQkRmsnormRope, FusedMatchesUnfusedFloat) {
  check_qk_rmsnorm_rope_fused<float>(1e-5f);
}
TEST(MossQkRmsnormRope, FusedMatchesUnfusedBHalf) {
  check_qk_rmsnorm_rope_fused<bhalf>(5e-2f);
}

TEST(MossRope, BuildApply) {
  const int Hh = 2, Ww = 3, TL = 4, Dh = 16;  // nb = 2
  const size_t S = Hh * Ww + TL;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> rope("rope", 1, S, Dh);
  const float base = 10000.0f;
  KokkosDNN::MossImage::moss_rope_build_forward<Exec>(Hh, Ww, TL, base, rope);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(rope);
  Kokkos::deep_copy(got, rope);
  const size_t nb = Dh / 8;
  const double tol = 1e-5;
  for (size_t s = 0; s < S; ++s) {
    double coord[4];
    if (s < (size_t)(Hh * Ww)) {
      size_t r = s / Ww, c = s % Ww;
      coord[0] = 0.0;
      coord[1] = TL;
      coord[2] = (double)r - (Hh - 1) / 2.0;
      coord[3] = (double)c - (Ww - 1) / 2.0;
    } else {
      coord[0] = 0.0;
      coord[1] = (double)(s - Hh * Ww);
      coord[2] = 0.0;
      coord[3] = 0.0;
    }
    for (size_t k = 0; k < nb; ++k) {
      double band = 1.0 / std::pow(10000.0, (double)k / nb);
      for (int m = 0; m < 4; ++m) {
        double ph = coord[m] * band;
        EXPECT_NEAR(got(0, s, (size_t)m * nb + k), std::sin(ph), tol)
            << "sin mismatch at (" << s << "," << m << "," << k << ")";
        EXPECT_NEAR(got(0, s, 4 * nb + (size_t)m * nb + k), std::cos(ph),
                    tol)
            << "cos mismatch at (" << s << "," << m << "," << k << ")";
      }
    }
  }
  // Apply: rotate q, compare against the same half-rotation on host.
  const int B = 1, H = 2;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> x("x", B, S, H, Dh),
      out("o", B, S, H, Dh);
  auto hx = Kokkos::create_mirror_view(x);
  for (int s = 0; s < (int)S; ++s)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d)
        hx(0, s, h, d) = 0.21f * s - 0.13f * h + 0.07f * d - 0.5f;
  Kokkos::deep_copy(x, hx);
  KokkosDNN::MossImage::moss_rope_apply_forward<Exec, float>(x, rope, out);
  Kokkos::fence();
  auto gout = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(gout, out);
  for (int s = 0; s < (int)S; ++s)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh / 2; ++d) {
        double sn = got(0, s, d), cs = got(0, s, d + Dh / 2);
        double a = hx(0, s, h, d), bv = hx(0, s, h, d + Dh / 2);
        EXPECT_NEAR(gout(0, s, h, d), a * cs - bv * sn, 1e-5)
            << "rot0 mismatch at (" << s << "," << h << "," << d << ")";
        EXPECT_NEAR(gout(0, s, h, d + Dh / 2), bv * cs + a * sn, 1e-5)
            << "rot1 mismatch at (" << s << "," << h << "," << d << ")";
      }
}

TEST(MossJointAttention, FloatVsHostSoftmax) {
  const int T = 6, H = 2, Dh = 8, K = 6;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> q("q", T, H, Dh),
      k("k", K, H, Dh), v("v", K, H, Dh), out("o", T, H, Dh);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        hq(t, h, d) = 0.2f * t - 0.1f * h + 0.05f * d - 0.3f;
        hk(t, h, d) = -0.15f * t + 0.12f * h - 0.04f * d + 0.2f;
        hv(t, h, d) = 0.1f * t + 0.07f * h - 0.09f * d + 0.1f;
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  KokkosDNN::DNNHandle<Exec> handle;
  const float scale = 1.0f / std::sqrt((float)Dh);
  KokkosDNN::MossImage::moss_joint_attention_forward<Exec, float>(
      handle, q, k, v, out, scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      double smax = -1e30, denom = 0;
      double s[6];
      for (int kk = 0; kk < K; ++kk) {
        double dot = 0;
        for (int d = 0; d < Dh; ++d)
          dot += (double)hq(t, h, d) * (double)hk(kk, h, d);
        s[kk] = dot * scale;
        if (s[kk] > smax) smax = s[kk];
      }
      for (int kk = 0; kk < K; ++kk) denom += std::exp(s[kk] - smax);
      for (int d = 0; d < Dh; ++d) {
        double want = 0;
        for (int kk = 0; kk < K; ++kk)
          want += std::exp(s[kk] - smax) / denom * (double)hv(kk, h, d);
        EXPECT_NEAR((double)got(t, h, d), want, 2e-5)
            << "mismatch at (" << t << "," << h << "," << d << ")";
      }
    }
}

TEST(MossCfgCombine, Rank2AndRank4) {
  // Rank-2.
  const int N = 4, D = 6;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> c2("c", N, D), u2("u", N, D),
      o2("o", N, D);
  auto hc = Kokkos::create_mirror_view(c2);
  auto hu = Kokkos::create_mirror_view(u2);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < D; ++j) {
      hc(i, j) = 0.3f * i - 0.2f * j;
      hu(i, j) = -0.1f * i + 0.4f * j + 0.5f;
    }
  Kokkos::deep_copy(c2, hc);
  Kokkos::deep_copy(u2, hu);
  const float w = 3.0f;
  KokkosDNN::MossImage::moss_cfg_combine_forward<Exec, float>(c2, u2, w, o2);
  Kokkos::fence();
  auto go2 = Kokkos::create_mirror_view(o2);
  Kokkos::deep_copy(go2, o2);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < D; ++j)
      EXPECT_NEAR(go2(i, j), hu(i, j) + w * (hc(i, j) - hu(i, j)), 1e-6f);
  // Rank-4 latent grid.
  const int B = 1, C = 4, H = 2, W = 2;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> c4("c", B, C, H, W),
      u4("u", B, C, H, W), o4("o", B, C, H, W);
  auto hc4 = Kokkos::create_mirror_view(c4);
  auto hu4 = Kokkos::create_mirror_view(u4);
  for (int b = 0; b < B; ++b)
    for (int cc = 0; cc < C; ++cc)
      for (int h = 0; h < H; ++h)
        for (int ww = 0; ww < W; ++ww) {
          hc4(b, cc, h, ww) = 0.1f * cc + 0.2f * h - 0.3f * ww;
          hu4(b, cc, h, ww) = -0.2f * cc + 0.1f * h + 0.4f * ww + 0.1f;
        }
  Kokkos::deep_copy(c4, hc4);
  Kokkos::deep_copy(u4, hu4);
  KokkosDNN::MossImage::moss_cfg_combine_forward<Exec, float>(c4, u4, w, o4);
  Kokkos::fence();
  auto go4 = Kokkos::create_mirror_view(o4);
  Kokkos::deep_copy(go4, o4);
  for (int cc = 0; cc < C; ++cc)
    for (int h = 0; h < H; ++h)
      for (int ww = 0; ww < W; ++ww)
        EXPECT_NEAR(go4(0, cc, h, ww),
                    hu4(0, cc, h, ww) +
                        w * (hc4(0, cc, h, ww) - hu4(0, cc, h, ww)),
                    1e-6f);
  EXPECT_THROW((KokkosDNN::MossImage::moss_cfg_combine_forward<Exec, float>(
                   c2, u2, -1.0f, o2)),
               std::invalid_argument);
}

TEST(MossSlgCorrect, Rank2AndRank4) {
  const int N = 4, D = 6;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> base("b", N, D),
      c2("c", N, D), cs2("cs", N, D), o2("o", N, D);
  auto hb = Kokkos::create_mirror_view(base);
  auto hc = Kokkos::create_mirror_view(c2);
  auto hs = Kokkos::create_mirror_view(cs2);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < D; ++j) {
      hb(i, j) = 0.5f * i + 0.1f * j;
      hc(i, j) = 0.3f * i - 0.2f * j;
      hs(i, j) = -0.1f * i + 0.4f * j;
    }
  Kokkos::deep_copy(base, hb);
  Kokkos::deep_copy(c2, hc);
  Kokkos::deep_copy(cs2, hs);
  const float s = 0.5f;
  KokkosDNN::MossImage::moss_slg_correct_forward<Exec, float>(base, c2, cs2, s,
                                                              o2);
  Kokkos::fence();
  auto go2 = Kokkos::create_mirror_view(o2);
  Kokkos::deep_copy(go2, o2);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < D; ++j)
      EXPECT_NEAR(go2(i, j), hb(i, j) + s * (hc(i, j) - hs(i, j)), 1e-6f);
  // Rank-4: base == cond_skip with guidance 1 recovers cond_skip + s*(c-cs).
  const int B = 1, C = 4, H = 2, W = 2;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> b4("b", B, C, H, W),
      c4("c", B, C, H, W), s4("s", B, C, H, W), o4("o", B, C, H, W);
  auto hb4 = Kokkos::create_mirror_view(b4);
  auto hc4 = Kokkos::create_mirror_view(c4);
  auto hs4 = Kokkos::create_mirror_view(s4);
  for (int cc = 0; cc < C; ++cc)
    for (int h = 0; h < H; ++h)
      for (int ww = 0; ww < W; ++ww) {
        hb4(0, cc, h, ww) = 0.2f * cc - 0.1f * h + 0.3f * ww;
        hc4(0, cc, h, ww) = 0.1f * cc + 0.2f * h - 0.3f * ww;
        hs4(0, cc, h, ww) = -0.2f * cc + 0.1f * h + 0.4f * ww;
      }
  Kokkos::deep_copy(b4, hb4);
  Kokkos::deep_copy(c4, hc4);
  Kokkos::deep_copy(s4, hs4);
  KokkosDNN::MossImage::moss_slg_correct_forward<Exec, float>(b4, c4, s4, s,
                                                              o4);
  Kokkos::fence();
  auto go4 = Kokkos::create_mirror_view(o4);
  Kokkos::deep_copy(go4, o4);
  for (int cc = 0; cc < C; ++cc)
    for (int h = 0; h < H; ++h)
      for (int ww = 0; ww < W; ++ww)
        EXPECT_NEAR(go4(0, cc, h, ww),
                    hb4(0, cc, h, ww) +
                        s * (hc4(0, cc, h, ww) - hs4(0, cc, h, ww)),
                    1e-6f);
}

TEST(MossVae, DenormAndDepthToSpace) {
  const int B = 1, C = 8, H = 2, W = 3;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> z("z", B, C, H, W),
      dn("dn", B, C, H, W);
  Kokkos::View<float*, Exec> mean("m", C), var("v", C);
  auto hz = Kokkos::create_mirror_view(z);
  auto hm = Kokkos::create_mirror_view(mean);
  auto hv = Kokkos::create_mirror_view(var);
  for (int cc = 0; cc < C; ++cc)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        hz(0, cc, h, w) = 0.25f * cc - 0.1f * h + 0.05f * w - 0.4f;
  for (int cc = 0; cc < C; ++cc) {
    hm(cc) = 0.02f * cc - 0.05f;
    hv(cc) = 0.8f + 0.05f * cc;
  }
  Kokkos::deep_copy(z, hz);
  Kokkos::deep_copy(mean, hm);
  Kokkos::deep_copy(var, hv);
  const float eps = 1e-4f;
  KokkosDNN::MossImage::moss_vae_denormalize_forward<Exec, float>(z, mean, var,
                                                                  eps, dn);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(dn);
  Kokkos::deep_copy(got, dn);
  for (int cc = 0; cc < C; ++cc) {
    double rstd = std::sqrt((double)hv(cc) + eps);
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        EXPECT_NEAR(got(0, cc, h, w),
                    (double)hz(0, cc, h, w) * rstd + (double)hm(cc), 1e-6)
            << "denorm mismatch at (" << cc << "," << h << "," << w << ")";
  }
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> up("up", B, C / 4, 2 * H,
                                                        2 * W);
  KokkosDNN::MossImage::moss_vae_depth_to_space_2x<Exec, float>(dn, up);
  Kokkos::fence();
  auto gup = Kokkos::create_mirror_view(up);
  Kokkos::deep_copy(gup, up);
  for (int co = 0; co < C / 4; ++co)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j)
            EXPECT_EQ(gup(0, co, 2 * h + i, 2 * w + j),
                      got(0, co * 4 + i * 2 + j, h, w));
}

template <class Scalar>
void check_situ(float tol) {
  const int N = 4, F = 16;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> h("h", N, F), g("g", N, F),
      o("o", N, F);
  auto hh = Kokkos::create_mirror_view(h);
  auto hg = Kokkos::create_mirror_view(g);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < F; ++j) {
      hh(i, j) = static_cast<Scalar>(0.5f * i - 0.3f * j + 0.1f);
      hg(i, j) = static_cast<Scalar>(-0.2f * i + 0.4f * j - 0.2f);
    }
  Kokkos::deep_copy(h, hh);
  Kokkos::deep_copy(g, hg);
  KokkosDNN::MossImage::moss_situ_forward<Exec, Scalar>(h, g, o);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(o);
  Kokkos::deep_copy(got, o);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < F; ++j) {
      double hd = static_cast<double>(hh(i, j));
      double gd = static_cast<double>(hg(i, j));
      double want = (4.0 * std::tanh(gd / 4.0) / (1.0 + std::exp(-gd))) *
                    (25.0 * std::tanh(hd / 25.0));
      EXPECT_NEAR(static_cast<double>(got(i, j)), want, tol)
          << "mismatch at (" << i << "," << j << ")";
    }
  // Rank-3 overload smoke: same math over [B, T, F].
  const int B = 2, T = 3;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> h3("h3", B, T, F),
      g3("g3", B, T, F), o3("o3", B, T, F);
  auto hh3 = Kokkos::create_mirror_view(h3);
  auto hg3 = Kokkos::create_mirror_view(g3);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
      for (int f = 0; f < F; ++f) {
        hh3(b, t, f) = static_cast<Scalar>(0.1f * b - 0.2f * t + 0.05f * f);
        hg3(b, t, f) = static_cast<Scalar>(0.3f * b + 0.1f * t - 0.04f * f);
      }
  Kokkos::deep_copy(h3, hh3);
  Kokkos::deep_copy(g3, hg3);
  KokkosDNN::MossImage::moss_situ_forward<Exec, Scalar>(h3, g3, o3);
  Kokkos::fence();
  auto got3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(got3, o3);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
      for (int f = 0; f < F; ++f) {
        double hd = static_cast<double>(hh3(b, t, f));
        double gd = static_cast<double>(hg3(b, t, f));
        double want = (4.0 * std::tanh(gd / 4.0) / (1.0 + std::exp(-gd))) *
                      (25.0 * std::tanh(hd / 25.0));
        EXPECT_NEAR(static_cast<double>(got3(b, t, f)), want, tol)
            << "rank-3 mismatch at (" << b << "," << t << "," << f << ")";
      }
}

TEST(MossSitu, Float) { check_situ<float>(1e-5f); }
TEST(MossSitu, BHalf) { check_situ<bhalf>(5e-2f); }

TEST(MossSkipLayers, Validate) {
  EXPECT_EQ(KokkosDNN::MossImage::moss_checked_skip_layers({22}, 42),
            std::vector<int>({22}));
  EXPECT_TRUE(
      KokkosDNN::MossImage::moss_checked_skip_layers({}, 42).empty());
  EXPECT_THROW(KokkosDNN::MossImage::moss_checked_skip_layers({42}, 42),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::MossImage::moss_checked_skip_layers({-1}, 42),
               std::invalid_argument);
}

TEST(MossConfig, GroundedDefaults) {
  auto c = KokkosDNN::MossImage::Config::moss_image_sft();
  EXPECT_EQ(c.dim, 3072);
  EXPECT_EQ(c.ffn_dim, 8192);
  EXPECT_EQ(c.num_heads, 24);
  EXPECT_EQ(c.head_dim(), 128);
  EXPECT_EQ(c.num_layers, 42);
  EXPECT_EQ(c.text_dim, 4096);
  EXPECT_EQ(c.freq_dim, 256);
  EXPECT_EQ(c.latent_channels, 128);
  EXPECT_DOUBLE_EQ(c.shift, 5.0);
  EXPECT_DOUBLE_EQ(c.guidance_scale, 3.0);
  EXPECT_NO_THROW(c.validate());
  auto t = KokkosDNN::MossImage::Config::tiny_test();
  EXPECT_NO_THROW(t.validate());
}

}  // namespace
