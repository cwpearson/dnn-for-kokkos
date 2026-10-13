// Sana fused attention ops vs naive triple-loop host references (float + bhalf):
// Model-specific stitching; lives with the Sana example (see
// KokkosDNN_SanaOps.hpp), not the core library.
// LiteLA linear self-attention (with RMSNorm weights and the Identity
// empty-norm path) and bidirectional multi-head cross attention, each in
// rank-2 and rank-3 (batched) form.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_SanaOps.hpp"

namespace {

using bhalf = Kokkos::Experimental::bhalf_t;
using HostSpace = Kokkos::HostSpace;
using Exec = Kokkos::DefaultExecutionSpace;

template <class Scalar>
using HostMat = Kokkos::View<Scalar**, Kokkos::LayoutRight, HostSpace>;
template <class Scalar>
using HostVec = Kokkos::View<Scalar*, HostSpace>;
template <class Scalar>
using DevMat = Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec>;
template <class Scalar>
using DevVec = Kokkos::View<Scalar*, Exec>;
template <class Scalar>
using DevMat3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>;

constexpr float kNormEps = 1e-6f;
constexpr float kLiteEps = 1e-15f;

// Deterministic pseudo-data in a small range (keeps bhalf outputs O(1)).
float det_val(int a, int b, int mod, float s, float o) {
  return s * static_cast<float>((a * 31 + b * 17) % mod) + o;
}

// ---- LiteLA host reference: one [T, D] sequence -> [T, D] output ----
template <class Scalar>
void litela_host_ref(const HostMat<Scalar>& hx, const HostMat<Scalar>& hw,
                     const HostVec<Scalar>& hqn, const HostVec<Scalar>& hkn,
                     bool with_norm, const HostMat<Scalar>& hpw,
                     const HostVec<Scalar>& hpb, size_t H, float* ref) {
  const size_t T = hx.extent(0), D = hx.extent(1), Dh = D / H;
  const size_t W = 3 * D;
  std::vector<float> qkv(T * W), q(T * D), k(T * D), v(T * D);
  for (size_t t = 0; t < T; ++t)
    for (size_t j = 0; j < W; ++j) {
      double acc = 0;
      for (size_t kk = 0; kk < D; ++kk)
        acc += static_cast<double>(hx(t, kk)) * static_cast<double>(hw(j, kk));
      qkv[t * W + j] = static_cast<float>(acc);
    }
  for (size_t t = 0; t < T; ++t) {
    float qss = 0, kss = 0;
    for (size_t d = 0; d < D; ++d) {
      qss += qkv[t * W + d] * qkv[t * W + d];
      kss += qkv[t * W + D + d] * qkv[t * W + D + d];
    }
    const float qinv =
        with_norm ? 1.0f / std::sqrt(qss / static_cast<float>(D) + kNormEps) : 1.0f;
    const float kinv =
        with_norm ? 1.0f / std::sqrt(kss / static_cast<float>(D) + kNormEps) : 1.0f;
    for (size_t d = 0; d < D; ++d) {
      float qv = qkv[t * W + d];
      float kv = qkv[t * W + D + d];
      if (with_norm) {
        qv = qv * qinv * static_cast<float>(hqn(d));
        kv = kv * kinv * static_cast<float>(hkn(d));
      }
      q[t * D + d] = qv > 0.0f ? qv : 0.0f;
      k[t * D + d] = kv > 0.0f ? kv : 0.0f;
      v[t * D + d] = qkv[t * W + 2 * D + d];
    }
  }
  auto at = [&](const std::vector<float>& m, size_t t, size_t h, size_t e) {
    return m[(t * H + h) * Dh + e];
  };
  std::vector<float> Ksum(H * Dh, 0.0f), VK(H * Dh * Dh, 0.0f);
  for (size_t h = 0; h < H; ++h)
    for (size_t e = 0; e < Dh; ++e) {
      float s = 0;
      for (size_t n = 0; n < T; ++n) s += at(k, n, h, e);
      Ksum[h * Dh + e] = s;
      for (size_t d = 0; d < Dh; ++d) {
        float a = 0;
        for (size_t n = 0; n < T; ++n) a += at(v, n, h, d) * at(k, n, h, e);
        VK[(h * Dh + d) * Dh + e] = a;
      }
    }
  std::vector<float> o(T * D);
  for (size_t n = 0; n < T; ++n)
    for (size_t h = 0; h < H; ++h)
      for (size_t d = 0; d < Dh; ++d) {
        float num = 0, den = kLiteEps;
        for (size_t e = 0; e < Dh; ++e) {
          num += VK[(h * Dh + d) * Dh + e] * at(q, n, h, e);
          den += Ksum[h * Dh + e] * at(q, n, h, e);
        }
        o[(n * H + h) * Dh + d] = num / den;
      }
  for (size_t t = 0; t < T; ++t)
    for (size_t j = 0; j < D; ++j) {
      double acc = 0;
      for (size_t kk = 0; kk < D; ++kk)
        acc += static_cast<double>(o[t * D + kk]) * static_cast<double>(hpw(j, kk));
      ref[t * D + j] = static_cast<float>(acc) + static_cast<float>(hpb(j));
    }
}

template <class Scalar>
void check_litela_2d(bool with_norm, float tol) {
  const size_t T = 5, D = 8, H = 2, W = 3 * D;
  KokkosDNN::DNNHandle<Exec> h;
  DevMat<Scalar> x("x", T, D), qkvw("w", W, D), pw("pw", D, D), out("out", T, D);
  DevVec<Scalar> qnw("qnw", with_norm ? D : 0), knw("knw", with_norm ? D : 0),
      pb("pb", D);
  HostMat<Scalar> hx("hx", T, D), hw("hw", W, D), hpw("hpw", D, D);
  HostVec<Scalar> hqn("hqn", with_norm ? D : 0), hkn("hkn", with_norm ? D : 0),
      hpb("hpb", D);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(det_val((int)t, (int)d, 11, 0.15f, -0.7f));
  for (size_t j = 0; j < W; ++j)
    for (size_t kk = 0; kk < D; ++kk)
      hw(j, kk) = static_cast<Scalar>(det_val((int)j, (int)kk, 13, 0.12f, -0.6f));
  for (size_t j = 0; j < D; ++j)
    for (size_t kk = 0; kk < D; ++kk)
      hpw(j, kk) = static_cast<Scalar>(det_val((int)kk, (int)j, 7, 0.2f, -0.5f));
  for (size_t d = 0; d < D; ++d) {
    if (with_norm) {
      hqn(d) = static_cast<Scalar>(0.8f + 0.05f * (d % 5));
      hkn(d) = static_cast<Scalar>(0.9f + 0.04f * ((d + 2) % 5));
    }
    hpb(d) = static_cast<Scalar>(det_val((int)d, 3, 5, 0.1f, -0.2f));
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(qkvw, hw);
  Kokkos::deep_copy(pw, hpw);
  if (with_norm) {
    Kokkos::deep_copy(qnw, hqn);
    Kokkos::deep_copy(knw, hkn);
  }
  Kokkos::deep_copy(pb, hpb);

  KokkosDNN::Sana::linear_attention_lite_la<Exec, Scalar>(h, x, qkvw, qnw, knw, pw, pb,
                                                   out, H);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);

  std::vector<float> ref(T * D);
  litela_host_ref(hx, hw, hqn, hkn, with_norm, hpw, hpb, H, ref.data());
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d)
      EXPECT_NEAR(static_cast<float>(got(t, d)), ref[t * D + d], tol)
          << (with_norm ? "litela norm" : "litela identity") << " mismatch at ("
          << t << "," << d << ")";
}

template <class Scalar>
void check_litela_3d(float tol) {
  const size_t B = 2, T = 3, D = 8, H = 2, W = 3 * D;
  KokkosDNN::DNNHandle<Exec> h;
  DevMat3<Scalar> x("x", B, T, D), out("out", B, T, D);
  DevMat<Scalar> qkvw("w", W, D), pw("pw", D, D);
  DevVec<Scalar> qnw("qnw", 0), knw("knw", 0), pb("pb", D);  // Identity norms
  Kokkos::View<Scalar***, Kokkos::LayoutRight, HostSpace> hx("hx", B, T, D);
  HostMat<Scalar> hw("hw", W, D), hpw("hpw", D, D);
  HostVec<Scalar> hqn("hqn", 0), hkn("hkn", 0), hpb("hpb", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d)
        hx(b, t, d) = static_cast<Scalar>(
            det_val((int)(b * T + t), (int)d, 11, 0.15f, -0.7f));
  for (size_t j = 0; j < W; ++j)
    for (size_t kk = 0; kk < D; ++kk)
      hw(j, kk) = static_cast<Scalar>(det_val((int)j, (int)kk, 13, 0.12f, -0.6f));
  for (size_t j = 0; j < D; ++j)
    for (size_t kk = 0; kk < D; ++kk)
      hpw(j, kk) = static_cast<Scalar>(det_val((int)kk, (int)j, 7, 0.2f, -0.5f));
  for (size_t d = 0; d < D; ++d)
    hpb(d) = static_cast<Scalar>(det_val((int)d, 1, 5, 0.1f, -0.2f));
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(qkvw, hw);
  Kokkos::deep_copy(pw, hpw);
  Kokkos::deep_copy(pb, hpb);

  KokkosDNN::Sana::linear_attention_lite_la<Exec, Scalar>(h, x, qkvw, qnw, knw, pw, pb,
                                                   out, H);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);

  for (size_t b = 0; b < B; ++b) {
    HostMat<Scalar> hxb("hxb", T, D);
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) hxb(t, d) = hx(b, t, d);
    std::vector<float> ref(T * D);
    litela_host_ref<Scalar>(hxb, hw, hqn, hkn, false, hpw, hpb, H, ref.data());
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d)
        EXPECT_NEAR(static_cast<float>(got(b, t, d)), ref[t * D + d], tol)
            << "litela batched mismatch at (" << b << "," << t << "," << d << ")";
  }
}

// ---- Cross-attention host reference: [T, D] x [L, D] -> [T, D] ----
template <class Scalar>
void cross_host_ref(const HostMat<Scalar>& hx, const HostMat<Scalar>& hc,
                    const HostMat<Scalar>& hqw, const HostVec<Scalar>& hqb,
                    const HostMat<Scalar>& hkv, const HostVec<Scalar>& hkvb,
                    const HostMat<Scalar>& hpw, const HostVec<Scalar>& hpb,
                    size_t H, float* ref) {
  const size_t T = hx.extent(0), D = hx.extent(1), L = hc.extent(0);
  const size_t Dh = D / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  std::vector<float> q(T * D), k(L * D), v(L * D);
  for (size_t t = 0; t < T; ++t)
    for (size_t j = 0; j < D; ++j) {
      double acc = 0;
      for (size_t kk = 0; kk < D; ++kk)
        acc += static_cast<double>(hx(t, kk)) * static_cast<double>(hqw(j, kk));
      q[t * D + j] = static_cast<float>(acc) + static_cast<float>(hqb(j));
    }
  for (size_t l = 0; l < L; ++l)
    for (size_t j = 0; j < 2 * D; ++j) {
      double acc = 0;
      for (size_t kk = 0; kk < D; ++kk)
        acc += static_cast<double>(hc(l, kk)) * static_cast<double>(hkv(j, kk));
      const float kv = static_cast<float>(acc) + static_cast<float>(hkvb(j));
      if (j < D)
        k[l * D + j] = kv;
      else
        v[l * D + (j - D)] = kv;
    }
  std::vector<float> o(T * D, 0.0f);
  for (size_t t = 0; t < T; ++t)
    for (size_t h = 0; h < H; ++h) {
      std::vector<float> s(L);
      float m = -1e30f;
      for (size_t l = 0; l < L; ++l) {
        float dot = 0;
        for (size_t e = 0; e < Dh; ++e)
          dot += q[(t * H + h) * Dh + e] * k[(l * H + h) * Dh + e];
        s[l] = dot * scale;
        m = std::max(m, s[l]);
      }
      float denom = 0;
      for (size_t l = 0; l < L; ++l) denom += std::exp(s[l] - m);
      for (size_t d = 0; d < Dh; ++d) {
        float acc = 0;
        for (size_t l = 0; l < L; ++l)
          acc += std::exp(s[l] - m) * v[(l * H + h) * Dh + d];
        o[(t * H + h) * Dh + d] = acc / denom;
      }
    }
  for (size_t t = 0; t < T; ++t)
    for (size_t j = 0; j < D; ++j) {
      double acc = 0;
      for (size_t kk = 0; kk < D; ++kk)
        acc += static_cast<double>(o[t * D + kk]) * static_cast<double>(hpw(j, kk));
      ref[t * D + j] = static_cast<float>(acc) + static_cast<float>(hpb(j));
    }
}

template <class Scalar>
void check_cross_2d(float tol) {
  const size_t T = 4, L = 5, D = 8, H = 2;
  KokkosDNN::DNNHandle<Exec> h;
  DevMat<Scalar> x("x", T, D), c("c", L, D), qw("qw", D, D), kvw("kvw", 2 * D, D),
      pw("pw", D, D), out("out", T, D);
  DevVec<Scalar> qb("qb", D), kvb("kvb", 2 * D), pb("pb", D);
  HostMat<Scalar> hx("hx", T, D), hc("hc", L, D), hqw("hqw", D, D),
      hkv("hkv", 2 * D, D), hpw("hpw", D, D);
  HostVec<Scalar> hqb("hqb", D), hkvb("hkvb", 2 * D), hpb("hpb", D);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d)
      hx(t, d) = static_cast<Scalar>(det_val((int)t, (int)d, 9, 0.2f, -0.6f));
  for (size_t l = 0; l < L; ++l)
    for (size_t d = 0; d < D; ++d)
      hc(l, d) = static_cast<Scalar>(det_val((int)l + 3, (int)d, 7, 0.25f, -0.5f));
  for (size_t j = 0; j < D; ++j) {
    for (size_t kk = 0; kk < D; ++kk) {
      hqw(j, kk) = static_cast<Scalar>(det_val((int)j, (int)kk, 11, 0.15f, -0.5f));
      hpw(j, kk) = static_cast<Scalar>(det_val((int)kk + 1, (int)j, 7, 0.2f, -0.5f));
    }
    hqb(j) = static_cast<Scalar>(det_val((int)j, 0, 5, 0.1f, -0.2f));
    hpb(j) = static_cast<Scalar>(det_val((int)j, 2, 5, 0.1f, -0.2f));
  }
  for (size_t j = 0; j < 2 * D; ++j) {
    for (size_t kk = 0; kk < D; ++kk)
      hkv(j, kk) = static_cast<Scalar>(det_val((int)j + 5, (int)kk, 11, 0.15f, -0.5f));
    hkvb(j) = static_cast<Scalar>(det_val((int)j, 4, 5, 0.1f, -0.2f));
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(c, hc);
  Kokkos::deep_copy(qw, hqw);
  Kokkos::deep_copy(qb, hqb);
  Kokkos::deep_copy(kvw, hkv);
  Kokkos::deep_copy(kvb, hkvb);
  Kokkos::deep_copy(pw, hpw);
  Kokkos::deep_copy(pb, hpb);

  KokkosDNN::Sana::cross_attention_mha_bidirectional<Exec, Scalar>(h, x, c, qw, qb, kvw,
                                                             kvb, pw, pb, out, H);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);

  std::vector<float> ref(T * D);
  cross_host_ref(hx, hc, hqw, hqb, hkv, hkvb, hpw, hpb, H, ref.data());
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d)
      EXPECT_NEAR(static_cast<float>(got(t, d)), ref[t * D + d], tol)
          << "cross mismatch at (" << t << "," << d << ")";
}

template <class Scalar>
void check_cross_3d(float tol) {
  const size_t B = 2, T = 3, L = 4, D = 8, H = 2;
  KokkosDNN::DNNHandle<Exec> h;
  DevMat3<Scalar> x("x", B, T, D), c("c", B, L, D), out("out", B, T, D);
  DevMat<Scalar> qw("qw", D, D), kvw("kvw", 2 * D, D), pw("pw", D, D);
  DevVec<Scalar> qb("qb", D), kvb("kvb", 2 * D), pb("pb", D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, HostSpace> hx("hx", B, T, D),
      hc("hc", B, L, D);
  HostMat<Scalar> hqw("hqw", D, D), hkv("hkv", 2 * D, D), hpw("hpw", D, D);
  HostVec<Scalar> hqb("hqb", D), hkvb("hkvb", 2 * D), hpb("hpb", D);
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d)
        hx(b, t, d) = static_cast<Scalar>(
            det_val((int)(b * T + t), (int)d, 9, 0.2f, -0.6f));
    for (size_t l = 0; l < L; ++l)
      for (size_t d = 0; d < D; ++d)
        hc(b, l, d) = static_cast<Scalar>(
            det_val((int)(b * L + l + 3), (int)d, 7, 0.25f, -0.5f));
  }
  for (size_t j = 0; j < D; ++j) {
    for (size_t kk = 0; kk < D; ++kk) {
      hqw(j, kk) = static_cast<Scalar>(det_val((int)j, (int)kk, 11, 0.15f, -0.5f));
      hpw(j, kk) = static_cast<Scalar>(det_val((int)kk + 1, (int)j, 7, 0.2f, -0.5f));
    }
    hqb(j) = static_cast<Scalar>(det_val((int)j, 0, 5, 0.1f, -0.2f));
    hpb(j) = static_cast<Scalar>(det_val((int)j, 2, 5, 0.1f, -0.2f));
  }
  for (size_t j = 0; j < 2 * D; ++j) {
    for (size_t kk = 0; kk < D; ++kk)
      hkv(j, kk) = static_cast<Scalar>(det_val((int)j + 5, (int)kk, 11, 0.15f, -0.5f));
    hkvb(j) = static_cast<Scalar>(det_val((int)j, 4, 5, 0.1f, -0.2f));
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(c, hc);
  Kokkos::deep_copy(qw, hqw);
  Kokkos::deep_copy(qb, hqb);
  Kokkos::deep_copy(kvw, hkv);
  Kokkos::deep_copy(kvb, hkvb);
  Kokkos::deep_copy(pw, hpw);
  Kokkos::deep_copy(pb, hpb);

  KokkosDNN::Sana::cross_attention_mha_bidirectional<Exec, Scalar>(h, x, c, qw, qb, kvw,
                                                             kvb, pw, pb, out, H);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(HostSpace{}, out);

  for (size_t b = 0; b < B; ++b) {
    HostMat<Scalar> hxb("hxb", T, D), hcb("hcb", L, D);
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) hxb(t, d) = hx(b, t, d);
    for (size_t l = 0; l < L; ++l)
      for (size_t d = 0; d < D; ++d) hcb(l, d) = hc(b, l, d);
    std::vector<float> ref(T * D);
    cross_host_ref<Scalar>(hxb, hcb, hqw, hqb, hkv, hkvb, hpw, hpb, H, ref.data());
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d)
        EXPECT_NEAR(static_cast<float>(got(b, t, d)), ref[t * D + d], tol)
            << "cross batched mismatch at (" << b << "," << t << "," << d << ")";
  }
}

}  // namespace

TEST(SanaOpsAttention, LiteLaFloat) { check_litela_2d<float>(true, 1e-4f); }

TEST(SanaOpsAttention, LiteLaIdentityFloat) { check_litela_2d<float>(false, 1e-4f); }

TEST(SanaOpsAttention, LiteLaBatchedFloat) { check_litela_3d<float>(1e-4f); }

TEST(SanaOpsAttention, LiteLaBHalf) { check_litela_2d<bhalf>(true, 2e-2f); }

TEST(SanaOpsAttention, LiteLaIdentityBHalf) { check_litela_2d<bhalf>(false, 2e-2f); }

TEST(SanaOpsAttentionCross, MhaFloat) { check_cross_2d<float>(1e-4f); }

TEST(SanaOpsAttentionCross, MhaBatchedFloat) { check_cross_3d<float>(1e-4f); }

TEST(SanaOpsAttentionCross, MhaBHalf) { check_cross_2d<bhalf>(2e-2f); }
