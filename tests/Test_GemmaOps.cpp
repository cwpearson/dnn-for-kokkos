// Gemma4 core ops vs host references (float + bfloat16):
// scaled embedding gather, gated-GeLU multiply, scaleless RMSNorm,
// proportional partial-RoPE table (+ tail passthrough), sliding-window
// causal attention, uniform scalar scale, logit softcapping.
//
// NOTE: reference inputs live in explicitly-allocated HostSpace views and
// results are read back with create_mirror_view_and_copy, so the tests are
// correct on backends where create_mirror_view would alias device memory
// (e.g. Serial) even across the in-place checks below.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "KokkosDNN.hpp"

namespace {

using bhalf = Kokkos::Experimental::bhalf_t;
using HostSpace = Kokkos::HostSpace;

template <class Scalar>
using HostMat = Kokkos::View<Scalar**, Kokkos::LayoutRight, HostSpace>;

// Double-precision host references.
double gelu_tanh_ref(double x) {
  static const double c = 0.7978845608028654;  // sqrt(2/pi)
  return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
}

template <class Scalar>
void check_embed_scaled(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int V = 6, D = 8, T = 3;
  const float scale = std::sqrt(1536.0f);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> tab("tab", V, D),
      out("out", T, D);
  Kokkos::View<int*, Exec> toks("toks", T);
  HostMat<Scalar> ht("ht", V, D);
  for (int i = 0; i < V; ++i)
    for (int d = 0; d < D; ++d)
      ht(i, d) = static_cast<Scalar>(0.1f * i - 0.03f * d);
  Kokkos::deep_copy(tab, ht);
  {
    Kokkos::View<int*, HostSpace> hx("hx", T);
    hx(0) = 5;
    hx(1) = 0;
    hx(2) = 2;
    Kokkos::deep_copy(toks, hx);
  }
  KokkosDNN::embedding_gather_scaled<Exec, Scalar>(tab, toks, scale, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  const int id[3] = {5, 0, 2};
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) {
      float want = scale * static_cast<float>(ht(id[t], d));
      EXPECT_NEAR(static_cast<float>(got(t, d)), want, tol)
          << "embed_scaled mismatch at (" << t << "," << d << ")";
    }
  // Dimension mismatch throws.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad("bad", T, D - 1);
  EXPECT_THROW(
      (KokkosDNN::embedding_gather_scaled<Exec, Scalar>(tab, toks, scale, bad)),
      std::invalid_argument);
}

template <class Scalar>
void check_gated_gelu(double tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const size_t T = 2, F = 16;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> gate("gate", T, F),
      up("up", T, F), out("out", T, F);
  HostMat<Scalar> hg("hg", T, F), hu("hu", T, F);
  for (size_t t = 0; t < T; ++t)
    for (size_t f = 0; f < F; ++f) {
      float g = 0.25f * (float)((t * F + f) % 9) - 1.0f;
      float u = 0.1f * (float)((t * 7 + f * 3) % 11) - 0.5f;
      hg(t, f) = static_cast<Scalar>(g);
      hu(t, f) = static_cast<Scalar>(u);
    }
  Kokkos::deep_copy(gate, hg);
  Kokkos::deep_copy(up, hu);
  KokkosDNN::gated_gelu_multiply<Exec, Scalar>(gate, up, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  for (size_t t = 0; t < T; ++t)
    for (size_t f = 0; f < F; ++f) {
      double want = gelu_tanh_ref(static_cast<double>(hg(t, f))) *
                    static_cast<double>(hu(t, f));
      EXPECT_NEAR(static_cast<double>(got(t, f)), want, tol)
          << "gated_gelu mismatch at (" << t << "," << f << ")";
    }
  // In-place over gate gives the same result.
  KokkosDNN::gated_gelu_multiply<Exec, Scalar>(gate, up, gate);
  Kokkos::fence();
  auto got2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, gate);
  for (size_t t = 0; t < T; ++t)
    for (size_t f = 0; f < F; ++f)
      EXPECT_NEAR(static_cast<double>(got2(t, f)),
                  static_cast<double>(got(t, f)), tol)
          << "gated_gelu in-place mismatch at (" << t << "," << f << ")";
}

template <class Scalar>
void check_rmsnorm_noscale(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 3, D = 8;
  const float eps = 1e-6f;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  HostMat<Scalar> hx("hx", T, D);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.5f * (t + 1) - 0.25f * d);
  Kokkos::deep_copy(x, hx);
  KokkosDNN::rmsnorm_noscale<Exec, Scalar>(x, eps, y);
  Kokkos::View<Scalar*, Exec> xr("xr", D), yr("yr", D);
  {
    Kokkos::View<Scalar*, HostSpace> h("h", D);
    for (int d = 0; d < D; ++d) h(d) = hx(1, d);
    Kokkos::deep_copy(xr, h);
  }
  KokkosDNN::rmsnorm_row_noscale<Exec, Scalar>(xr, eps, yr);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, y);
  auto gotr = Kokkos::create_mirror_view_and_copy(HostSpace{}, yr);
  for (int t = 0; t < T; ++t) {
    float ss = 0;
    for (int d = 0; d < D; ++d) {
      float v = static_cast<float>(hx(t, d));
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / D + eps);
    for (int d = 0; d < D; ++d) {
      float want = static_cast<float>(hx(t, d)) * inv;
      EXPECT_NEAR(static_cast<float>(got(t, d)), want, tol)
          << "rmsnorm_noscale mismatch at (" << t << "," << d << ")";
    }
  }
  {
    float ss = 0;
    for (int d = 0; d < D; ++d) {
      float v = static_cast<float>(hx(1, d));
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / D + eps);
    for (int d = 0; d < D; ++d) {
      float want = static_cast<float>(hx(1, d)) * inv;
      EXPECT_NEAR(static_cast<float>(gotr(d)), want, tol)
          << "rmsnorm_row_noscale mismatch at (" << d << ")";
    }
  }
}

template <class Scalar>
void check_scale_by_scalar(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = 2, D = 6;
  const float s = 0.0178f;  // Gemma4-style learned layer scalar
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D), y("y", T, D);
  HostMat<Scalar> hx("hx", T, D);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(0.3f * (t + 1) - 0.11f * d);
  Kokkos::deep_copy(x, hx);
  KokkosDNN::scale_by_scalar<Exec, Scalar>(x, s, y);
  Kokkos::View<Scalar*, Exec> xr("xr", D), yr("yr", D);
  {
    Kokkos::View<Scalar*, HostSpace> h("h", D);
    for (int d = 0; d < D; ++d) h(d) = hx(0, d);
    Kokkos::deep_copy(xr, h);
  }
  KokkosDNN::scale_by_scalar<Exec, Scalar>(xr, s, yr);
  // In-place rank-2.
  KokkosDNN::scale_by_scalar<Exec, Scalar>(x, s, x);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, y);
  auto gotr = Kokkos::create_mirror_view_and_copy(HostSpace{}, yr);
  auto gotip = Kokkos::create_mirror_view_and_copy(HostSpace{}, x);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) {
      float want = s * static_cast<float>(hx(t, d));
      EXPECT_NEAR(static_cast<float>(got(t, d)), want, tol)
          << "scale_by_scalar mismatch at (" << t << "," << d << ")";
      EXPECT_NEAR(static_cast<float>(gotip(t, d)), want, tol)
          << "scale_by_scalar in-place mismatch at (" << t << "," << d << ")";
    }
  for (int d = 0; d < D; ++d) {
    float want = s * static_cast<float>(hx(0, d));
    EXPECT_NEAR(static_cast<float>(gotr(d)), want, tol)
        << "scale_by_scalar rank-1 mismatch at (" << d << ")";
  }
}

template <class Scalar>
void check_logit_softcap(double tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const float cap = 30.0f;
  const size_t R = 2, C = 9;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", R, C), y("y", R, C);
  HostMat<Scalar> hx("hx", R, C);
  const float vals[9] = {0.0f,  1.0f,  -1.0f,  15.0f, -45.0f,
                         300.0f, -300.0f, 7.5f, -0.25f};
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      hx(i, j) = static_cast<Scalar>(vals[j] * (i == 0 ? 1.0f : -1.0f));
  Kokkos::deep_copy(x, hx);
  KokkosDNN::logit_softcap_forward<Exec, Scalar>(x, cap, y);
  Kokkos::View<Scalar*, Exec> xr("xr", C), yr("yr", C);
  {
    Kokkos::View<Scalar*, HostSpace> h("h", C);
    for (size_t j = 0; j < C; ++j) h(j) = hx(0, j);
    Kokkos::deep_copy(xr, h);
  }
  KokkosDNN::logit_softcap_forward<Exec, Scalar>(xr, cap, yr);
  // In-place rank-1.
  KokkosDNN::logit_softcap_forward<Exec, Scalar>(xr, cap, xr);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, y);
  auto gotr = Kokkos::create_mirror_view_and_copy(HostSpace{}, yr);
  auto gotip = Kokkos::create_mirror_view_and_copy(HostSpace{}, xr);
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j) {
      double xin = (double)vals[j] * (i == 0 ? 1.0 : -1.0);
      double want = (double)cap * std::tanh(xin / (double)cap);
      EXPECT_NEAR(static_cast<double>(got(i, j)), want, tol)
          << "softcap mismatch at (" << i << "," << j << ") x=" << xin;
    }
  for (size_t j = 0; j < C; ++j) {
    double want = (double)cap * std::tanh((double)vals[j] / (double)cap);
    EXPECT_NEAR(static_cast<double>(gotr(j)), want, tol)
        << "softcap rank-1 mismatch at x=" << vals[j];
    EXPECT_NEAR(static_cast<double>(gotip(j)), want, tol)
        << "softcap in-place mismatch at x=" << vals[j];
  }
  // Saturation: |300| >> 30 collapses to +/-cap.
  EXPECT_NEAR(static_cast<double>(got(0, 5)), (double)cap, tol);
  EXPECT_NEAR(static_cast<double>(got(0, 6)), (double)-cap, tol);
  // Non-positive cap throws.
  EXPECT_THROW((KokkosDNN::logit_softcap_forward<Exec, Scalar>(x, 0.0f, y)),
               std::invalid_argument);
  EXPECT_THROW((KokkosDNN::logit_softcap_forward<Exec, Scalar>(xr, -1.0f, yr)),
               std::invalid_argument);
}

}  // namespace

TEST(GemmaOps, EmbedScaledFloat) { check_embed_scaled<float>(1e-4f); }

TEST(GemmaOps, EmbedScaledBHalf) { check_embed_scaled<bhalf>(5e-1f); }

TEST(GemmaOps, GatedGeluFloat) { check_gated_gelu<float>(1e-6); }

TEST(GemmaOps, GatedGeluBHalf) { check_gated_gelu<bhalf>(5e-2); }

TEST(GemmaOps, RmsNormNoscaleFloat) { check_rmsnorm_noscale<float>(1e-5f); }

TEST(GemmaOps, RmsNormNoscaleBHalf) { check_rmsnorm_noscale<bhalf>(5e-2f); }

TEST(GemmaOps, ScaleByScalarFloat) { check_scale_by_scalar<float>(1e-6f); }

TEST(GemmaOps, ScaleByScalarBHalf) { check_scale_by_scalar<bhalf>(5e-2f); }

TEST(GemmaOps, LogitSoftcapFloat) { check_logit_softcap<float>(1e-5); }

TEST(GemmaOps, LogitSoftcapBHalf) { check_logit_softcap<bhalf>(5e-2); }

// Proportional partial-RoPE table: values + zero tail + arg validation.
TEST(GemmaOps, ProportionalRopeTable) {
  const int D = 512;
  const float theta = 1e6f, partial = 0.25f;
  auto hv = KokkosDNN::rope_inv_freq_proportional_host(D, theta, partial);
  ASSERT_EQ(hv.size(), (size_t)(D / 2));  // 256 entries
  // n_rot = 0.25 * 512 / 2 = 64 rotary pairs.
  for (int j = 0; j < 64; ++j) {
    double want = 1.0 / std::pow(1e6, 2.0 * j / 512.0);
    EXPECT_NEAR(hv[(size_t)j], (float)want, 1e-6f) << "rotary entry " << j;
    EXPECT_GT(hv[(size_t)j], 0.0f);
  }
  for (int j = 64; j < 256; ++j)
    EXPECT_EQ(hv[(size_t)j], 0.0f) << "expected zero tail at " << j;
  EXPECT_THROW(
      KokkosDNN::rope_inv_freq_proportional_host(511, theta, partial),
      std::invalid_argument);
  EXPECT_THROW(KokkosDNN::rope_inv_freq_proportional_host(D, theta, 0.0f),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::rope_inv_freq_proportional_host(D, theta, 1.5f),
               std::invalid_argument);
}

// Zero-angle tail passes through rope_neoX_inplace unchanged (float+bhalf).
template <class Scalar>
void check_proportional_rope_passthrough(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int D = 512, H = 1, T = 2;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> q("q", T, H, D),
      k("k", T, H, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, HostSpace> hq("hq", T, H, D),
      hk("hk", T, H, D);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) {
      hq(t, 0, d) = static_cast<Scalar>(0.05f * ((t * 131 + d * 7) % 17) - 0.2f);
      hk(t, 0, d) = static_cast<Scalar>(0.05f * ((t * 29 + d * 3) % 19) - 0.1f);
    }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  auto inv = KokkosDNN::rope_inv_freq_proportional<Exec>(D, 1e6f, 0.25f);
  Kokkos::View<int*, Exec> pos("pos", T);
  {
    Kokkos::View<int*, HostSpace> hp("hp", T);
    hp(0) = 0;
    hp(1) = 17;
    Kokkos::deep_copy(pos, hp);
  }
  KokkosDNN::rope_neoX_inplace<Exec, Scalar>(q, k, inv, D, pos);
  Kokkos::fence();
  auto gq = Kokkos::create_mirror_view_and_copy(HostSpace{}, q);
  auto gk = Kokkos::create_mirror_view_and_copy(HostSpace{}, k);
  // Trailing 384 dims (d in [64,256) U [320,512)) are untouched.
  for (int t = 0; t < T; ++t)
    for (int d = 64; d < 256; ++d) {
      EXPECT_NEAR(static_cast<float>(gq(t, 0, d)),
                  static_cast<float>(hq(t, 0, d)), tol)
          << "rope tail changed at (" << t << "," << d << ")";
      EXPECT_NEAR(static_cast<float>(gq(t, 0, d + 256)),
                  static_cast<float>(hq(t, 0, d + 256)), tol)
          << "rope tail changed at (" << t << "," << d + 256 << ")";
      EXPECT_NEAR(static_cast<float>(gk(t, 0, d)),
                  static_cast<float>(hk(t, 0, d)), tol)
          << "rope k tail changed at (" << t << "," << d << ")";
      EXPECT_NEAR(static_cast<float>(gk(t, 0, d + 256)),
                  static_cast<float>(hk(t, 0, d + 256)), tol)
          << "rope k tail changed at (" << t << "," << d + 256 << ")";
    }
  // Leading rotary part still rotates (pos 17 moves q measurably).
  float moved = 0;
  for (int d = 0; d < 64; ++d)
    moved += std::abs(static_cast<float>(gq(1, 0, d)) -
                      static_cast<float>(hq(1, 0, d)));
  EXPECT_GT(moved, 1e-3f);
}

TEST(GemmaOps, ProportionalRopePassthroughFloat) {
  check_proportional_rope_passthrough<float>(1e-6f);
}

TEST(GemmaOps, ProportionalRopePassthroughBHalf) {
  check_proportional_rope_passthrough<bhalf>(5e-2f);
}

// Sliding-window causal attention vs host reference. Gemma4 geometry:
// 8 Q heads / 1 KV head, scale 1.0. A tiny window (3) exercises the lower
// clamp; window 512 / 0 must match full causal on a short sequence.
template <class Scalar>
void check_sliding_reference(size_t window, float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 5, Hq = 8, Hkv = 1, D = 8, Ktot = 5;
  const size_t q_off = 0;
  const float scale = 1.0f;

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> q("q", T, Hq, D),
      out("out", T, Hq, D), out_full("out_full", T, Hq, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, HostSpace> hq("hq", T, Hq, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, HostSpace> hk("hk", Ktot, Hkv,
                                                             D),
      hv("hv", Ktot, Hkv, D);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) =
            static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f);
  for (int n = 0; n < Ktot; ++n)
    for (int d = 0; d < D; ++d) {
      hk(n, 0, d) = static_cast<Scalar>(0.05f * ((n * 5 + d) % 19) - 0.2f);
      hv(n, 0, d) = static_cast<Scalar>(0.05f * ((n * 3 + d * 2) % 23) - 0.3f);
    }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);

  KokkosDNN::attention_gqa_single_sliding<Exec, Scalar>(h, q, k, v, q_off, out,
                                                       scale, window);
  KokkosDNN::attention_gqa_single<Exec, Scalar>(h, q, k, v, q_off, out_full,
                                               scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  auto got_full = Kokkos::create_mirror_view_and_copy(HostSpace{}, out_full);

  // Host reference with the sliding lower clamp.
  for (int t = 0; t < T; ++t) {
    size_t p = q_off + (size_t)t;
    size_t khi = p + 1;
    if (khi > (size_t)Ktot) khi = Ktot;
    size_t klo = (window > 0 && p + 1 >= window) ? (p + 1 - window) : 0;
    for (int hh = 0; hh < Hq; ++hh) {
      float m = -1e30f;
      for (size_t kk = klo; kk < khi; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, hh, d)) *
               static_cast<float>(hk(kk, 0, d));
        m = std::max(m, s * scale);
      }
      float denom = 0;
      for (size_t kk = klo; kk < khi; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, hh, d)) *
               static_cast<float>(hk(kk, 0, d));
        denom += std::exp(s * scale - m);
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (size_t kk = klo; kk < khi; ++kk) {
          float s = 0;
          for (int dd = 0; dd < D; ++dd)
            s += static_cast<float>(hq(t, hh, dd)) *
                 static_cast<float>(hk(kk, 0, dd));
          acc += std::exp(s * scale - m) * static_cast<float>(hv(kk, 0, d));
        }
        EXPECT_NEAR(static_cast<float>(got(t, hh, d)), acc / denom, tol)
            << "sliding(w=" << window << ") mismatch at (" << t << "," << hh
            << "," << d << ")";
      }
      // Window >= sequence length (or 0) must equal full causal.
      if (window == 0 || window >= (size_t)T) {
        for (int d = 0; d < D; ++d)
          EXPECT_NEAR(static_cast<float>(got(t, hh, d)),
                      static_cast<float>(got_full(t, hh, d)), tol)
              << "sliding(w=" << window << ") != full at (" << t << "," << hh
              << "," << d << ")";
      }
    }
  }
}

TEST(GemmaOps, SlidingWindow3Float) { check_sliding_reference<float>(3, 1e-4f); }

TEST(GemmaOps, SlidingWindow3BHalf) {
  check_sliding_reference<bhalf>(3, 5e-2f);
}

TEST(GemmaOps, SlidingWindow512MatchesFullFloat) {
  check_sliding_reference<float>(512, 1e-4f);
}

TEST(GemmaOps, SlidingWindow0MatchesFullFloat) {
  check_sliding_reference<float>(0, 1e-4f);
}

// Sliding window with a nonzero prefill offset (decode-style): query at
// absolute pos p attends [p+1-window, p].
TEST(GemmaOps, SlidingWindowOffsetFloat) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 2, Hq = 8, Hkv = 1, D = 4, Ktot = 10;
  const size_t q_off = 8, window = 4;  // p = 8, 9 -> klo = 5, 6
  const float scale = 1.0f;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> q("q", T, Hq, D),
      out("out", T, Hq, D);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D);
  Kokkos::View<float***, Kokkos::LayoutRight, HostSpace> hq("hq", T, Hq, D);
  Kokkos::View<float***, Kokkos::LayoutRight, HostSpace> hk("hk", Ktot, Hkv, D),
      hv("hv", Ktot, Hkv, D);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) = 0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f;
  for (int n = 0; n < Ktot; ++n)
    for (int d = 0; d < D; ++d) {
      hk(n, 0, d) = 0.05f * ((n * 5 + d) % 19) - 0.2f;
      hv(n, 0, d) = 0.05f * ((n * 3 + d * 2) % 23) - 0.3f;
    }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  KokkosDNN::attention_gqa_single_sliding<Exec, float>(h, q, k, v, q_off, out,
                                                      scale, window);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);
  for (int t = 0; t < T; ++t) {
    size_t p = q_off + (size_t)t;
    size_t klo = (p + 1 >= window) ? (p + 1 - window) : 0;
    size_t khi = p + 1;
    for (int hh = 0; hh < Hq; ++hh) {
      float m = -1e30f;
      for (size_t kk = klo; kk < khi; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d) s += hq(t, hh, d) * hk(kk, 0, d);
        m = std::max(m, s * scale);
      }
      float denom = 0;
      for (size_t kk = klo; kk < khi; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d) s += hq(t, hh, d) * hk(kk, 0, d);
        denom += std::exp(s * scale - m);
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (size_t kk = klo; kk < khi; ++kk) {
          float s = 0;
          for (int dd = 0; dd < D; ++dd) s += hq(t, hh, dd) * hk(kk, 0, dd);
          acc += std::exp(s * scale - m) * hv(kk, 0, d);
        }
        EXPECT_NEAR(got(t, hh, d), acc / denom, 1e-4f)
            << "offset sliding mismatch at (" << t << "," << hh << "," << d
            << ")";
      }
    }
  }
}
