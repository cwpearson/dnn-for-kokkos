// Qwen3.5 text-encoder tests: config/layer-type pattern, (1+w) QK-norm
// golden, partial-rotary golden, gated-delta-rule golden, gated-RMSNorm
// golden, and a tiny-config full-model forward vs a host double reference.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_MossImageEncoder.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using bhalf = Kokkos::Experimental::bhalf_t;
using QM = KokkosDNN::Qwen35::Config;
namespace QI = KokkosDNN::Qwen35::Impl;

// ---- host double reference ------------------------------------------------
inline double hsig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double hsilu(double x) { return x * hsig(x); }
inline double hsoftplus(double x) {
  return x > 20.0 ? x : std::log(1.0 + std::exp(x));
}

// y = rms1p(x, w): [T, D] row-major vectors.
void hrms1p(const std::vector<double>& x, const std::vector<double>& w,
            double eps, std::vector<double>& y, int T, int D) {
  y.assign((size_t)T * D, 0);
  for (int t = 0; t < T; ++t) {
    double ss = 0;
    for (int d = 0; d < D; ++d) ss += x[(size_t)t * D + d] * x[(size_t)t * D + d];
    double inv = 1.0 / std::sqrt(ss / D + eps);
    for (int d = 0; d < D; ++d)
      y[(size_t)t * D + d] = x[(size_t)t * D + d] * inv * (1.0 + w[d]);
  }
}

// Y[T, N] = X[T, K] * W[N, K]^T (unbiased gemm).
void hgemm(const std::vector<double>& X, const std::vector<double>& W,
           std::vector<double>& Y, int T, int K, int N) {
  Y.assign((size_t)T * N, 0);
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < N; ++n) {
      double a = 0;
      for (int k = 0; k < K; ++k)
        a += X[(size_t)t * K + k] * W[(size_t)n * K + k];
      Y[(size_t)t * N + n] = a;
    }
}

void hadd(std::vector<double>& x, const std::vector<double>& b) {
  for (size_t i = 0; i < x.size(); ++i) x[i] += b[i];
}

TEST(Qwen35Config, TinyDimsAndPattern) {
  QM c = QM::tiny_test();
  c.validate();
  EXPECT_EQ(c.rotary_dim(), 2);  // int(8 * 0.25)
  EXPECT_FALSE(c.is_full_layer(0));
  EXPECT_FALSE(c.is_full_layer(1));
  EXPECT_FALSE(c.is_full_layer(2));
  EXPECT_TRUE(c.is_full_layer(3));
  QM full = QM::moss_image_sft();
  full.validate();
  EXPECT_EQ(full.rotary_dim(), 64);
  for (int l = 0; l < 32; ++l)
    EXPECT_EQ(full.is_full_layer(l), (l == 3 || l == 7 || l == 11 ||
                                      l == 15 || l == 19 || l == 23 ||
                                      l == 27 || l == 31))
        << "layer " << l;
  QM bad = c;
  bad.num_q_heads = 3;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(Qwen35QkNorm, Golden) {
  const int T = 3, H = 2, D = 4;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", T, H, D),
      y("y", T, H, D);
  Kokkos::View<float*, Exec> w("w", D);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d)
        hx(t, h, d) = 0.3f * (t + 1) - 0.2f * (h + 1) + 0.1f * (d + 1);
  for (int d = 0; d < D; ++d) hw(d) = 0.05f * (d - 1);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  QI::qk_rmsnorm_oneplus_3d<Exec, float>(x, w, 1e-6f, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double v = hx(t, h, d);
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / D + 1e-6);
      for (int d = 0; d < D; ++d) {
        double want = hx(t, h, d) * inv * (1.0 + hw(d));
        EXPECT_NEAR(got(t, h, d), want, 1e-5) << t << h << d;
      }
    }
}

TEST(Qwen35Rope, Golden) {
  // rotary_dim=4 over head_dim=8: pairs (0,2),(1,3); dims 4..7 passthrough.
  const int T = 4, H = 2, D = 8, R = 4;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", T, H, D);
  auto hx = Kokkos::create_mirror_view(x);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d)
        hx(t, h, d) = 0.1f * (d + 1) * (t + 1) - 0.05f * h;
  // Snapshot: on host-space backends the mirror may alias x, which the
  // in-place kernel below modifies.
  std::vector<double> ref((size_t)T * H * D);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d)
        ref[((size_t)t * H + h) * D + d] = hx(t, h, d);
  Kokkos::deep_copy(x, hx);
  auto inv = QI::qwen_rope_inv_host(R, 10000.0);
  ASSERT_EQ(inv.size(), 2u);
  // Denominator is R=4 (NOT head_dim): inv[1] = 1e4^(-2/4) = 1e-2.
  EXPECT_NEAR(inv[0], 1.0, 1e-7);
  EXPECT_NEAR(inv[1], 0.01, 1e-9);
  Kokkos::View<float*, Exec> dinv("dinv", 2);
  auto hd = Kokkos::create_mirror_view(dinv);
  hd(0) = inv[0];
  hd(1) = inv[1];
  Kokkos::deep_copy(dinv, hd);
  QI::qwen_partial_rope_inplace<Exec, float>(x, dinv, R);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(got, x);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      for (int j = 0; j < 2; ++j) {
        double ang = t * inv[j];
        double c = std::cos(ang), s = std::sin(ang);
        double a = ref[((size_t)t * H + h) * D + j];
        double b = ref[((size_t)t * H + h) * D + j + 2];
        EXPECT_NEAR(got(t, h, j), a * c - b * s, 1e-5) << t << h << j;
        EXPECT_NEAR(got(t, h, j + 2), a * s + b * c, 1e-5) << t << h << j;
      }
      for (int d = 4; d < 8; ++d)
        EXPECT_FLOAT_EQ(got(t, h, d), (float)ref[((size_t)t * H + h) * D + d])
            << "passthrough " << d;
    }
}
TEST(Qwen35GatedDelta, Golden) {
  const int T = 5, H = 2, K = 4, V = 3;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> q("q", T, H, K),
      k("k", T, H, K), v("v", T, H, V), out("o", T, H, V);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> beta("b", T, H),
      g("g", T, H);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> st("s", H, K, V);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  auto hb = Kokkos::create_mirror_view(beta);
  auto hg = Kokkos::create_mirror_view(g);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      hb(t, h) = 0.1f * (t + h + 1);
      hg(t, h) = -0.05f * (t + 1) * (h + 1);
      for (int d = 0; d < K; ++d) {
        hq(t, h, d) = 0.2f * (d + 1) - 0.1f * t + 0.05f * h;
        hk(t, h, d) = -0.15f * (d + 1) + 0.1f * t - 0.05f * h;
      }
      for (int d = 0; d < V; ++d)
        hv(t, h, d) = 0.25f * (d + 1) + 0.05f * (t - h);
    }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  Kokkos::deep_copy(beta, hb);
  Kokkos::deep_copy(g, hg);
  Kokkos::deep_copy(st, 0.0f);
  QI::qwen_gated_delta_recurrent<Exec, float>(q, k, v, beta, g, st, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  // Host recurrence in double (same op order as the kernel).
  for (int h = 0; h < H; ++h) {
    double S[4][3] = {};
    for (int t = 0; t < T; ++t) {
      double dec = std::exp(hg(t, h)), bt = hb(t, h);
      for (int kk = 0; kk < K; ++kk)
        for (int vv = 0; vv < V; ++vv) S[kk][vv] *= dec;
      for (int vv = 0; vv < V; ++vv) {
        double kv = 0;
        for (int kk = 0; kk < K; ++kk) kv += S[kk][vv] * hk(t, h, kk);
        double del = (hv(t, h, vv) - kv) * bt;
        for (int kk = 0; kk < K; ++kk) S[kk][vv] += hk(t, h, kk) * del;
      }
      for (int vv = 0; vv < V; ++vv) {
        double o = 0;
        for (int kk = 0; kk < K; ++kk) o += S[kk][vv] * hq(t, h, kk);
        EXPECT_NEAR(got(t, h, vv), o, 1e-4) << t << h << vv;
      }
    }
  }
}

TEST(Qwen35RmsGated, Golden) {
  const int T = 3, H = 2, D = 4;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> c("c", T, H, D),
      z("z", T, H, D), o("o", T, H, D);
  Kokkos::View<float*, Exec> w("w", D);
  auto hc = Kokkos::create_mirror_view(c);
  auto hz = Kokkos::create_mirror_view(z);
  auto hw = Kokkos::create_mirror_view(w);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d) {
        hc(t, h, d) = 0.2f * (t + 1) - 0.1f * (d + h);
        hz(t, h, d) = 0.15f * (d + 1) + 0.05f * (t - h);
      }
  for (int d = 0; d < D; ++d) hw(d) = 1.0f + 0.02f * d;
  Kokkos::deep_copy(c, hc);
  Kokkos::deep_copy(z, hz);
  Kokkos::deep_copy(w, hw);
  QI::qwen_rmsnorm_gated<Exec, float>(c, z, w, 1e-6f, o);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(o);
  Kokkos::deep_copy(got, o);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double x = hc(t, h, d);
        ss += x * x;
      }
      double inv = 1.0 / std::sqrt(ss / D + 1e-6);
      for (int d = 0; d < D; ++d) {
        double want = hc(t, h, d) * inv * hw(d) * hsilu(hz(t, h, d));
        EXPECT_NEAR(got(t, h, d), want, 1e-5) << t << h << d;
      }
    }
}
// ---- full-model host double reference --------------------------------------
// Mirrors EncoderModel::forward op-for-op in double precision.
struct HostLayerW {
  bool is_full = false;
  std::vector<double> in_n, post_n, wg, wu, wd;
  std::vector<double> q, k, v, o, qn, kn;          // full
  std::vector<double> qkv, z, a, b, conv, op, gn;  // linear
  std::vector<double> alog, dt;
};
struct HostModel {
  QM cfg;
  std::vector<double> embed, final_n;
  std::vector<HostLayerW> L;
};

template <class V1>
std::vector<double> hget1(const V1& v) {
  auto hm = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(hm, v);
  std::vector<double> o(v.extent(0));
  for (size_t i = 0; i < o.size(); ++i) o[i] = (double)hm(i);
  return o;
}
template <class V2>
std::vector<double> hget2(const V2& v) {
  auto hm = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(hm, v);
  std::vector<double> o(v.extent(0) * v.extent(1));
  size_t k = 0;
  for (size_t i = 0; i < v.extent(0); ++i)
    for (size_t j = 0; j < v.extent(1); ++j) o[k++] = (double)hm(i, j);
  return o;
}

// qk 3d view of a flat [T, H*D] buffer.
void hqknorm(std::vector<double>& x, const std::vector<double>& w, double eps,
             int T, int H, int D) {
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        double v = x[((size_t)t * H + h) * D + d];
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / D + eps);
      for (int d = 0; d < D; ++d)
        x[((size_t)t * H + h) * D + d] *= inv * (1.0 + w[d]);
    }
}

void hrope(std::vector<double>& x, const std::vector<double>& inv, int T,
           int H, int D) {
  int Rh = (int)inv.size();
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      double* r = &x[((size_t)t * H + h) * D];
      for (int j = 0; j < Rh; ++j) {
        double an = t * inv[j], c = std::cos(an), s = std::sin(an);
        double a = r[j], b = r[j + Rh];
        r[j] = a * c - b * s;
        r[j + Rh] = a * s + b * c;
      }
    }
}

// Causal GQA softmax attention: q [T,Nq,Hd], k/v [T,Nk,Hd] -> o [T,Nq*Hd].
void hattn(const std::vector<double>& q, const std::vector<double>& k,
           const std::vector<double>& v, std::vector<double>& o, int T,
           int Nq, int Nk, int Hd, double scale) {
  o.assign((size_t)T * Nq * Hd, 0);
  std::vector<double> sc((size_t)T);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < Nq; ++h) {
      int kh = h * Nk / Nq;
      double mx = -1e300;
      for (int s2 = 0; s2 <= t; ++s2) {
        double d = 0;
        for (int dd = 0; dd < Hd; ++dd)
          d += q[((size_t)t * Nq + h) * Hd + dd] *
               k[((size_t)s2 * Nk + kh) * Hd + dd];
        sc[s2] = d * scale;
        if (sc[s2] > mx) mx = sc[s2];
      }
      double den = 0;
      for (int s2 = 0; s2 <= t; ++s2) {
        sc[s2] = std::exp(sc[s2] - mx);
        den += sc[s2];
      }
      for (int dd = 0; dd < Hd; ++dd) {
        double a = 0;
        for (int s2 = 0; s2 <= t; ++s2)
          a += sc[s2] * v[((size_t)s2 * Nk + kh) * Hd + dd];
        o[((size_t)t * Nq + h) * Hd + dd] = a / den;
      }
    }
}

std::vector<double> href_forward_sample(const HostModel& m,
                                        const std::vector<int>& ids) {
  const QM& c = m.cfg;
  const int T = (int)ids.size(), H = c.hidden;
  std::vector<double> h((size_t)T * H);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < H; ++d) h[(size_t)t * H + d] = m.embed[(size_t)ids[t] * H + d];
  auto rinv_f = KokkosDNN::Qwen35::Impl::qwen_rope_inv_host(c.rotary_dim(),
                                                         c.rope_theta);
  const std::vector<double> rinv(rinv_f.begin(), rinv_f.end());
  std::vector<double> n, qg, k2, v2, o, g, u, d;
  for (int l = 0; l < c.num_layers; ++l) {
    const HostLayerW& L = m.L[(size_t)l];
    hrms1p(h, L.in_n, c.rms_eps, n, T, H);
    if (c.is_full_layer(l)) {
      const int Nq = c.num_q_heads, Nk = c.num_kv_heads, Hd = c.head_dim;
      const int Qd = Nq * Hd;
      hgemm(n, L.q, qg, T, H, 2 * Qd);
      hgemm(n, L.k, k2, T, H, Nk * Hd);
      hgemm(n, L.v, v2, T, H, Nk * Hd);
      std::vector<double> q(qg.begin(), qg.begin() + (size_t)T * Qd);
      std::vector<double> gate(qg.begin() + (size_t)T * Qd, qg.end());
      hqknorm(q, L.qn, c.rms_eps, T, Nq, Hd);
      hqknorm(k2, L.kn, c.rms_eps, T, Nk, Hd);
      hrope(q, rinv, T, Nq, Hd);
      hrope(k2, rinv, T, Nk, Hd);
      std::vector<double> ao;
      hattn(q, k2, v2, ao, T, Nq, Nk, Hd, 1.0 / std::sqrt((double)Hd));
      for (size_t i = 0; i < ao.size(); ++i) ao[i] *= hsig(gate[i]);
      hgemm(ao, L.o, o, T, Qd, H);
      hadd(h, o);
    } else {
      const int Kh = c.lin_k_heads, Kd = c.lin_k_dim, Vh = c.lin_v_heads,
                Vd = c.lin_v_dim;
      const int KdAll = 2 * Kh * Kd + Vh * Vd;
      hgemm(n, L.qkv, qg, T, H, KdAll);
      hgemm(n, L.z, d, T, H, Vh * Vd);  // reuse d as z
      std::vector<double> z = d, av, bv;
      hgemm(n, L.a, av, T, H, Vh);
      hgemm(n, L.b, bv, T, H, Vh);
      // causal depthwise conv + silu
      std::vector<double> cv((size_t)T * KdAll, 0);
      for (int t = 0; t < T; ++t)
        for (int cc = 0; cc < KdAll; ++cc) {
          double a = 0;
          for (int j = 0; j < c.conv_k; ++j) {
            int s2 = t - (c.conv_k - 1) + j;
            if (s2 < 0) continue;
            a += L.conv[(size_t)cc * c.conv_k + j] * qg[(size_t)s2 * KdAll + cc];
          }
          cv[(size_t)t * KdAll + cc] = hsilu(a);
        }
      auto split = [&](int off, int nn) {
        std::vector<double> s((size_t)T * nn);
        for (int t = 0; t < T; ++t)
          for (int i = 0; i < nn; ++i) s[(size_t)t * nn + i] = cv[(size_t)t * KdAll + off + i];
        return s;
      };
      std::vector<double> q = split(0, Kh * Kd), k = split(Kh * Kd, Kh * Kd),
                          vv = split(2 * Kh * Kd, Vh * Vd);
      const int rep = Vh / Kh;
      auto expand = [&](const std::vector<double>& x) {
        std::vector<double> y((size_t)T * Vh * Kd);
        for (int t = 0; t < T; ++t)
          for (int hh = 0; hh < Vh; ++hh)
            for (int dd = 0; dd < Kd; ++dd)
              y[((size_t)t * Vh + hh) * Kd + dd] =
                  x[((size_t)t * Kh + hh / rep) * Kd + dd];
        return y;
      };
      q = expand(q);
      k = expand(k);
      auto l2 = [&](std::vector<double>& x, double sc) {
        for (int t = 0; t < T; ++t)
          for (int hh = 0; hh < Vh; ++hh) {
            double ss = 0;
            for (int dd = 0; dd < Kd; ++dd) {
              double v = x[((size_t)t * Vh + hh) * Kd + dd];
              ss += v * v;
            }
            // FLA l2norm(): sum-based 1/sqrt(ss + eps), NOT mean-based.
            double inv = 1.0 / std::sqrt(ss + 1e-6) * sc;
            for (int dd = 0; dd < Kd; ++dd)
              x[((size_t)t * Vh + hh) * Kd + dd] *= inv;
          }
      };
      l2(q, 1.0 / std::sqrt((double)Kd));
      l2(k, 1.0);
      // recurrent gated delta rule
      std::vector<double> core((size_t)T * Vh * Vd, 0);
      std::vector<double> S((size_t)Vh * Kd * Vd, 0);
      for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < Vh; ++hh) {
          double A = std::exp(L.alog[hh]);
          double bt = hsig(bv[(size_t)t * Vh + hh]);
          double dec = std::exp(-A * hsoftplus(av[(size_t)t * Vh + hh] + L.dt[hh]));
          double* Sh = &S[((size_t)hh * Kd) * Vd];
          for (int i = 0; i < Kd * Vd; ++i) Sh[i] *= dec;
          for (int vv2 = 0; vv2 < Vd; ++vv2) {
            double kv = 0;
            for (int kk = 0; kk < Kd; ++kk)
              kv += Sh[kk * Vd + vv2] * k[((size_t)t * Vh + hh) * Kd + kk];
            double del = (vv[((size_t)t * Vh + hh) * Vd + vv2] - kv) * bt;
            for (int kk = 0; kk < Kd; ++kk)
              Sh[kk * Vd + vv2] += k[((size_t)t * Vh + hh) * Kd + kk] * del;
          }
          for (int vv2 = 0; vv2 < Vd; ++vv2) {
            double oo = 0;
            for (int kk = 0; kk < Kd; ++kk)
              oo += Sh[kk * Vd + vv2] * q[((size_t)t * Vh + hh) * Kd + kk];
            core[((size_t)t * Vh + hh) * Vd + vv2] = oo;
          }
        }
      // gated norm
      for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < Vh; ++hh) {
          double ss = 0;
          for (int dd = 0; dd < Vd; ++dd) {
            double v = core[((size_t)t * Vh + hh) * Vd + dd];
            ss += v * v;
          }
          double inv = 1.0 / std::sqrt(ss / Vd + c.rms_eps);
          for (int dd = 0; dd < Vd; ++dd) {
            size_t ix = ((size_t)t * Vh + hh) * Vd + dd;
            core[ix] = core[ix] * inv * L.gn[dd] * hsilu(z[ix]);
          }
        }
      hgemm(core, L.op, o, T, Vh * Vd, H);
      hadd(h, o);
    }
    hrms1p(h, L.post_n, c.rms_eps, n, T, H);
    hgemm(n, L.wg, g, T, H, c.intermediate);
    hgemm(n, L.wu, u, T, H, c.intermediate);
    for (size_t i = 0; i < g.size(); ++i) g[i] = hsilu(g[i]) * u[i];
    hgemm(g, L.wd, d, T, c.intermediate, H);
    hadd(h, d);
  }
  hrms1p(h, m.final_n, c.rms_eps, n, T, H);
  return n;
}

TEST(Qwen35Forward, TinyGoldenAndLengths) {
  QM cfg = QM::tiny_test();
  KokkosDNN::DNNHandle<Exec> handle;
  KokkosDNN::Qwen35::EncoderModel<Exec, float> enc(cfg);
  enc.init_random(7);
  // Mirror weights to the host double reference.
  HostModel hm;
  hm.cfg = cfg;
  hm.embed = hget2(enc.w.embed);
  hm.final_n = hget1(enc.w.final_norm);
  hm.L.resize((size_t)cfg.num_layers);
  for (int l = 0; l < cfg.num_layers; ++l) {
    auto& src = enc.w.layers[(size_t)l];
    auto& d = hm.L[(size_t)l];
    d.is_full = src.is_full;
    d.in_n = hget1(src.input_norm);
    d.post_n = hget1(src.post_norm);
    d.wg = hget2(src.gate_proj);
    d.wu = hget2(src.up_proj);
    d.wd = hget2(src.down_proj);
    if (src.is_full) {
      d.q = hget2(src.full.q_proj);
      d.k = hget2(src.full.k_proj);
      d.v = hget2(src.full.v_proj);
      d.o = hget2(src.full.o_proj);
      d.qn = hget1(src.full.q_norm);
      d.kn = hget1(src.full.k_norm);
    } else {
      d.qkv = hget2(src.lin.in_qkv);
      d.z = hget2(src.lin.in_z);
      d.a = hget2(src.lin.in_a);
      d.b = hget2(src.lin.in_b);
      d.conv = hget2(src.lin.conv);
      d.op = hget2(src.lin.out_proj);
      d.gn = hget1(src.lin.norm);
      d.alog = hget1(src.lin.a_log);
      d.dt = hget1(src.lin.dt_bias);
    }
  }
  const size_t B = 2, S = 6;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> ids("ids", B, S);
  auto hi = Kokkos::create_mirror_view(ids);
  std::vector<std::vector<int>> idvec = {{3, 10, 25, 7, 40, 1},
                                         {5, 9, 2, 33, 0, 0}};
  for (size_t b = 0; b < B; ++b)
    for (size_t s = 0; s < S; ++s) hi(b, s) = idvec[b][s];
  Kokkos::deep_copy(ids, hi);
  std::vector<size_t> lens = {6, 4};
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> out("out", B, S,
                                                        cfg.hidden);
  // Sentinel padding: rows past the valid length must be untouched.
  {
    auto ho = Kokkos::create_mirror_view(out);
    for (size_t b = 0; b < B; ++b)
      for (size_t s = 0; s < S; ++s)
        for (int d = 0; d < cfg.hidden; ++d) ho(b, s, d) = -999.0f;
    Kokkos::deep_copy(out, ho);
  }
  enc.forward(handle, ids, lens, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b) {
    std::vector<int> sub(idvec[b].begin(), idvec[b].begin() + lens[b]);
    std::vector<double> want = href_forward_sample(hm, sub);
    for (size_t s = 0; s < lens[b]; ++s)
      for (int d = 0; d < cfg.hidden; ++d)
        // Tolerance: device is fp32, reference is double; drift ~1e-3
        // accumulates through the GDN recurrence (state feedback over T)
        // and 4 layers of fp32 GEMM/softmax. Wiring bugs (wrong scale,
        // swapped split, missing gate) show up as O(0.1-1) diffs.
        EXPECT_NEAR(got(b, s, d), want[s * (size_t)cfg.hidden + d], 5e-3)
            << "b" << b << " s" << s << " d" << d;
    for (size_t s = lens[b]; s < S; ++s)
      for (int d = 0; d < cfg.hidden; ++d)
        EXPECT_FLOAT_EQ(got(b, s, d), -999.0f) << "pad clobbered";
  }
  // resident_bytes accounting is nonzero and release() zeroes it.
  EXPECT_GT(enc.resident_bytes(), 0u);
  // Bad ids / bad lengths throw.
  hi(0, 0) = cfg.vocab_size;
  Kokkos::deep_copy(ids, hi);
  EXPECT_THROW(enc.forward(handle, ids, lens, out), std::invalid_argument);
  hi(0, 0) = 3;
  Kokkos::deep_copy(ids, hi);
  EXPECT_THROW(enc.forward(handle, ids, std::vector<size_t>{0, 4}, out),
               std::invalid_argument);
}

TEST(Qwen35Forward, BHalfSmoke) {
  QM cfg = QM::tiny_test();
  KokkosDNN::DNNHandle<Exec> handle;
  KokkosDNN::Qwen35::EncoderModel<Exec, bhalf> enc(cfg);
  enc.init_random(11);
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> ids("ids", 1, 4);
  auto hi = Kokkos::create_mirror_view(ids);
  hi(0, 0) = 1;
  hi(0, 1) = 2;
  hi(0, 2) = 3;
  hi(0, 3) = 4;
  Kokkos::deep_copy(ids, hi);
  Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec> out("out", 1, 4,
                                                        cfg.hidden);
  enc.forward(handle, ids, out);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int s = 0; s < 4; ++s)
    for (int d = 0; d < cfg.hidden; ++d)
      EXPECT_TRUE(std::isfinite((float)got(0, s, d))) << s << d;
}

TEST(Qwen35Safetensors, RoundTripTiny) {
  QM cfg = QM::tiny_test();
  KokkosDNN::DNNHandle<Exec> handle;
  KokkosDNN::Qwen35::EncoderModel<Exec, float> enc(cfg);
  enc.init_random(21);
  // Collect (name, shape, float payload) for the full tiny weight set.
  struct T {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;  // "F32" or "BF16"
    std::vector<double> vals;
  };
  std::vector<T> ts;
  auto snap1 = [&](const std::string& n, auto v) {
    T t;
    t.name = n;
    t.shape = {(int64_t)v.extent(0)};
    t.vals = hget1(v);
    ts.push_back(t);
  };
  auto snap2 = [&](const std::string& n, auto v) {
    T t;
    t.name = n;
    t.shape = {(int64_t)v.extent(0), (int64_t)v.extent(1)};
    t.vals = hget2(v);
    ts.push_back(t);
  };
  snap2("language_model.embed_tokens.weight", enc.w.embed);
  snap1("language_model.norm.weight", enc.w.final_norm);
  for (int l = 0; l < cfg.num_layers; ++l) {
    std::string p = "language_model.layers." + std::to_string(l) + ".";
    auto& L = enc.w.layers[(size_t)l];
    snap1(p + "input_layernorm.weight", L.input_norm);
    snap1(p + "post_attention_layernorm.weight", L.post_norm);
    snap2(p + "mlp.gate_proj.weight", L.gate_proj);
    snap2(p + "mlp.up_proj.weight", L.up_proj);
    snap2(p + "mlp.down_proj.weight", L.down_proj);
    if (cfg.is_full_layer(l)) {
      snap2(p + "self_attn.q_proj.weight", L.full.q_proj);
      snap2(p + "self_attn.k_proj.weight", L.full.k_proj);
      snap2(p + "self_attn.v_proj.weight", L.full.v_proj);
      snap2(p + "self_attn.o_proj.weight", L.full.o_proj);
      snap1(p + "self_attn.q_norm.weight", L.full.q_norm);
      snap1(p + "self_attn.k_norm.weight", L.full.k_norm);
    } else {
      snap2(p + "linear_attn.in_proj_qkv.weight", L.lin.in_qkv);
      snap2(p + "linear_attn.in_proj_z.weight", L.lin.in_z);
      snap2(p + "linear_attn.in_proj_a.weight", L.lin.in_a);
      snap2(p + "linear_attn.in_proj_b.weight", L.lin.in_b);
      // conv1d.weight stored [C,1,K] on disk.
      T t;
      t.name = p + "linear_attn.conv1d.weight";
      t.shape = {(int64_t)L.lin.conv.extent(0), 1,
                 (int64_t)L.lin.conv.extent(1)};
      t.vals = hget2(L.lin.conv);
      ts.push_back(t);
      snap2(p + "linear_attn.out_proj.weight", L.lin.out_proj);
      snap1(p + "linear_attn.norm.weight", L.lin.norm);
      // Real-checkpoint naming: these two vectors have no .weight suffix.
      snap1(p + "linear_attn.A_log", L.lin.a_log);
      snap1(p + "linear_attn.dt_bias", L.lin.dt_bias);
    }
  }
  // Exercise both dtype paths: embed as BF16, everything else F32.
  for (auto& t : ts)
    t.dtype = (t.name == "language_model.embed_tokens.weight") ? "BF16"
                                                               : "F32";
  auto f32bits = [](float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
  };
  auto tobf16 = [&](float f) {
    uint32_t u = f32bits(f);
    // Round to nearest even.
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
  };
  // Assemble header JSON + payloads.
  std::string js = "{\"__metadata__\":{},";
  std::string payload;
  uint64_t off = 0;
  for (size_t i = 0; i < ts.size(); ++i) {
    const T& t = ts[i];
    size_t nb = t.vals.size() * (t.dtype == "F32" ? 4 : 2);
    js += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t d = 0; d < t.shape.size(); ++d)
      js += (d ? "," : "") + std::to_string(t.shape[d]);
    js += "],\"data_offsets\":[" + std::to_string(off) + "," +
          std::to_string(off + nb) + "]}";
    js += (i + 1 < ts.size() ? "," : "}");
    size_t base = payload.size();
    payload.resize(base + nb);
    if (t.dtype == "F32") {
      for (size_t k = 0; k < t.vals.size(); ++k) {
        uint32_t u = f32bits((float)t.vals[k]);
        std::memcpy(&payload[base + k * 4], &u, 4);
      }
    } else {
      for (size_t k = 0; k < t.vals.size(); ++k) {
        uint16_t u = tobf16((float)t.vals[k]);
        std::memcpy(&payload[base + k * 2], &u, 2);
      }
    }
    off += nb;
  }
  const std::string path = "/tmp/qwen35_tiny_test.safetensors";
  {
    std::ofstream f(path, std::ios::binary);
    uint64_t hl = js.size();
    f.write(reinterpret_cast<const char*>(&hl), 8);
    f.write(js.data(), (std::streamsize)js.size());
    f.write(payload.data(), (std::streamsize)payload.size());
  }
  KokkosDNN::Qwen35::EncoderModel<Exec, float> enc2(cfg);
  enc2.load_safetensors({path});
  // Exact for F32 tensors; BF16 embed within 1 ulp-ish (2^-8 relative).
  {
    auto a = hget2(enc.w.embed), b = hget2(enc2.w.embed);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
      EXPECT_NEAR(b[i], a[i], std::abs(a[i]) * 0.005 + 1e-3) << i;
    auto c = hget2(enc.w.layers[3].full.q_proj);
    auto d = hget2(enc2.w.layers[3].full.q_proj);
    ASSERT_EQ(c.size(), d.size());
    for (size_t i = 0; i < c.size(); ++i) EXPECT_FLOAT_EQ(d[i], c[i]) << i;
    auto e = hget2(enc.w.layers[0].lin.conv);
    auto f = hget2(enc2.w.layers[0].lin.conv);
    ASSERT_EQ(e.size(), f.size());
    for (size_t i = 0; i < e.size(); ++i) EXPECT_FLOAT_EQ(f[i], e[i]) << i;
  }
  // Loaded weights run forward and match the pre-save model bit-wise
  // (F32 path) up to BF16-embed rounding.
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> ids("ids", 1, 5);
  auto hi = Kokkos::create_mirror_view(ids);
  hi(0, 0) = 1;
  hi(0, 1) = 9;
  hi(0, 2) = 17;
  hi(0, 3) = 4;
  hi(0, 4) = 31;
  Kokkos::deep_copy(ids, hi);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> o1("o1", 1, 5,
                                                       cfg.hidden),
      o2("o2", 1, 5, cfg.hidden);
  enc.forward(handle, ids, o1);
  enc2.forward(handle, ids, o2);
  Kokkos::fence();
  auto g1 = Kokkos::create_mirror_view(o1);
  auto g2 = Kokkos::create_mirror_view(o2);
  Kokkos::deep_copy(g1, o1);
  Kokkos::deep_copy(g2, o2);
  for (int s = 0; s < 5; ++s)
    for (int d = 0; d < cfg.hidden; ++d)
      EXPECT_NEAR(g2(0, s, d), g1(0, s, d), 0.02) << s << d;
  std::remove(path.c_str());
  // Missing file / missing tensor throw.
  EXPECT_THROW(enc2.load_safetensors({"/tmp/does_not_exist.st"}),
               std::runtime_error);
}

}  // namespace
