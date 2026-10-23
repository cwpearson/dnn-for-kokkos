// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// 2D axial RoPE (DINOv3 ViT): host-reference match, prefix skip,
// identity at zero coord, dynamic H/W, out-of-place parity.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <vector>

#include "KokkosDNN.hpp"

namespace {

// NOTE: host snapshots below use explicit HostSpace views (not
// create_mirror_view): on a host backend such as Serial the mirror may
// alias device memory, which would let the in-place kernel mutate the
// "original" values out from under the reference checks.
// Explicit LayoutRight on both host and device keeps deep_copy legal on
// CUDA (default device layout is LayoutLeft, default host is LayoutRight).
template <class Scalar>
using HostView3 = Kokkos::View<Scalar***, Kokkos::LayoutRight,
                               Kokkos::HostSpace>;
template <class Scalar, class Exec>
using DevView3 =
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>;

constexpr float kTwoPi = 6.283185307179586f;

// Normalized center coordinate along one axis (mirrors Impl::rope2d_coord).
float axis_coord(int i, int n) { return 2.0f * ((float)i + 0.5f) / (float)n - 1.0f; }

// Host reference for one patch token: quarter layout [h, w, h, w] with
// rotate_half pairs (d, d+Dh/2), angles 2*pi*coord*inv_freq.
void host_rope2d_token(const std::vector<float>& in, std::vector<float>& out,
                       int Dh, float ch, float cw,
                       const std::vector<float>& inv) {
  const int Qh = Dh / 4;
  for (int d = 0; d < Qh; ++d) {
    const float ah  = kTwoPi * ch * inv[(size_t)d];
    const float chh = std::cos(ah), shh = std::sin(ah);
    const float h1 = in[(size_t)d], h2 = in[(size_t)(d + 2 * Qh)];
    out[(size_t)d]            = h1 * chh - h2 * shh;
    out[(size_t)(d + 2 * Qh)] = h1 * shh + h2 * chh;
    const float aw  = kTwoPi * cw * inv[(size_t)d];
    const float cww = std::cos(aw), sww = std::sin(aw);
    const float w1 = in[(size_t)(d + Qh)], w2 = in[(size_t)(d + 3 * Qh)];
    out[(size_t)(d + Qh)]     = w1 * cww - w2 * sww;
    out[(size_t)(d + 3 * Qh)] = w1 * sww + w2 * cww;
  }
}

float fill_val(int t, int h, int d) {
  return 0.01f * (float)(t * 131 + h * 17 + d + 1);
}

}  // namespace

TEST(RoPE2D, InvFreq) {
  // head_dim=64, theta=100 -> 16 entries, inv[i] = 100^(-4i/64).
  auto hv = KokkosDNN::rope2d_inv_freq_host(64, 100.0f);
  ASSERT_EQ(hv.size(), 16u);
  EXPECT_FLOAT_EQ(hv[0], 1.0f);
  for (int i = 0; i < 16; ++i)
    EXPECT_NEAR(hv[(size_t)i],
                (float)(1.0 / std::pow(100.0, 4.0 * i / 64.0)), 1e-6f)
        << "i=" << i;
  // Device builder mirrors the host reference.
  using Exec = Kokkos::DefaultExecutionSpace;
  auto dev   = KokkosDNN::rope2d_inv_freq<Exec>(64, 100.0f);
  ASSERT_EQ(dev.extent(0), 16u);
  Kokkos::View<float*, Kokkos::HostSpace> hm("hm", 16);
  Kokkos::deep_copy(hm, dev);
  for (int i = 0; i < 16; ++i) EXPECT_FLOAT_EQ(hm(i), hv[(size_t)i]);
  // head_dim must be divisible by 4.
  EXPECT_THROW(KokkosDNN::rope2d_inv_freq_host(62, 100.0f),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::rope2d_inv_freq_host(0, 100.0f),
               std::invalid_argument);
}

TEST(RoPE2D, MatchesHostReferenceAndSkipsPrefix) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const int Dh = 64, H = 2, Hp = 2, Wp = 2;
  const size_t prefix = 5, Np = 4, T = prefix + Np;
  HostView3<Scalar> hq("hq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> hk("hk", T, (size_t)H, (size_t)Dh);
  for (size_t t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        hq(t, h, d) = fill_val((int)t, h, d);
        hk(t, h, d) = -0.5f * fill_val((int)t, h, d);
      }
  DevView3<Scalar, Exec> q("q", T, H, Dh), k("k", T, H, Dh);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  auto inv = KokkosDNN::rope2d_inv_freq<Exec>(Dh, 100.0f);
  KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(q, k, inv, Hp, Wp, prefix);
  Kokkos::fence();
  HostView3<Scalar> gq("gq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> gk("gk", T, (size_t)H, (size_t)Dh);
  Kokkos::deep_copy(gq, q);
  Kokkos::deep_copy(gk, k);

  auto inv_h = KokkosDNN::rope2d_inv_freq_host(Dh, 100.0f);
  // Prefix rows (CLS + registers) are bit-exact untouched.
  for (size_t t = 0; t < prefix; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        EXPECT_FLOAT_EQ(gq(t, h, d), hq(t, h, d))
            << "q prefix changed at (" << t << "," << h << "," << d << ")";
        EXPECT_FLOAT_EQ(gk(t, h, d), hk(t, h, d))
            << "k prefix changed at (" << t << "," << h << "," << d << ")";
      }
  // Patch rows match the host reference (q and k).
  for (size_t p = 0; p < Np; ++p) {
    const float ch = axis_coord((int)(p / (size_t)Wp), Hp);
    const float cw = axis_coord((int)(p % (size_t)Wp), Wp);
    for (int h = 0; h < H; ++h) {
      std::vector<float> in((size_t)Dh), ref((size_t)Dh);
      for (int d = 0; d < Dh; ++d) in[(size_t)d] = hq(prefix + p, h, d);
      host_rope2d_token(in, ref, Dh, ch, cw, inv_h);
      for (int d = 0; d < Dh; ++d)
        EXPECT_NEAR(gq(prefix + p, h, d), ref[(size_t)d], 1e-5f)
            << "q mismatch at patch " << p << " head " << h << " lane " << d;
      for (int d = 0; d < Dh; ++d) in[(size_t)d] = hk(prefix + p, h, d);
      host_rope2d_token(in, ref, Dh, ch, cw, inv_h);
      for (int d = 0; d < Dh; ++d)
        EXPECT_NEAR(gk(prefix + p, h, d), ref[(size_t)d], 1e-5f)
            << "k mismatch at patch " << p << " head " << h << " lane " << d;
    }
  }
}

TEST(RoPE2D, IdentityAtZeroCoord) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  // Odd 3x3 grid: center patch (1,1) has coord (0,0) -> cos=1, sin=0.
  const int Dh = 64, H = 1, Hp = 3, Wp = 3;
  const size_t prefix = 5, Np = 9, T = prefix + Np;
  HostView3<Scalar> hq("hq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> hk("hk", T, (size_t)H, (size_t)Dh);
  for (size_t t = 0; t < T; ++t)
    for (int d = 0; d < Dh; ++d) {
      hq(t, 0, d) = fill_val((int)t, 0, d);
      hk(t, 0, d) = fill_val((int)t, 3, d);
    }
  DevView3<Scalar, Exec> q("q", T, H, Dh), k("k", T, H, Dh);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  auto inv = KokkosDNN::rope2d_inv_freq<Exec>(Dh, 100.0f);
  KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(q, k, inv, Hp, Wp, prefix);
  Kokkos::fence();
  HostView3<Scalar> gq("gq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> gk("gk", T, (size_t)H, (size_t)Dh);
  Kokkos::deep_copy(gq, q);
  Kokkos::deep_copy(gk, k);
  // Center patch p = 1*3+1 = 4 is bit-exact identity.
  const size_t pc = prefix + 4;
  for (int d = 0; d < Dh; ++d) {
    EXPECT_FLOAT_EQ(gq(pc, 0, d), hq(pc, 0, d)) << "q center lane " << d;
    EXPECT_FLOAT_EQ(gk(pc, 0, d), hk(pc, 0, d)) << "k center lane " << d;
  }
  // A corner patch is actually rotated (sanity: not identity everywhere).
  const size_t p0 = prefix + 0;
  float maxdiff   = 0;
  for (int d = 0; d < Dh; ++d)
    maxdiff = std::max(maxdiff, std::abs(gq(p0, 0, d) - hq(p0, 0, d)));
  EXPECT_GT(maxdiff, 1e-3f);
}

TEST(RoPE2D, DynamicGridShape) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  // Same Np=4, T=9 with different grids must give different outputs.
  const int Dh = 64, H = 2;
  const size_t prefix = 5, T = 9;
  auto run = [&](int Hp, int Wp) {
    HostView3<Scalar> hq("hq", T, (size_t)H, (size_t)Dh);
    for (size_t t = 0; t < T; ++t)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < Dh; ++d) hq(t, h, d) = fill_val((int)t, h, d);
    DevView3<Scalar, Exec> q("q", T, H, Dh), k("k", T, H, Dh);
    Kokkos::deep_copy(q, hq);
    Kokkos::deep_copy(k, hq);
    auto inv = KokkosDNN::rope2d_inv_freq<Exec>(Dh, 100.0f);
    KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(q, k, inv, Hp, Wp, prefix);
    Kokkos::fence();
    HostView3<Scalar> gq("gq", T, (size_t)H, (size_t)Dh);
    Kokkos::deep_copy(gq, q);
    return gq;
  };
  auto r22 = run(2, 2);
  auto r41 = run(4, 1);
  auto r14 = run(1, 4);
  float d41 = 0, d14 = 0;
  for (size_t t = prefix; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        d41 = std::max(d41, std::abs(r22(t, h, d) - r41(t, h, d)));
        d14 = std::max(d14, std::abs(r22(t, h, d) - r14(t, h, d)));
      }
  EXPECT_GT(d41, 1e-3f);
  EXPECT_GT(d14, 1e-3f);
}

TEST(RoPE2D, OutOfPlaceMatchesInplace) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const int Dh = 64, H = 2, Hp = 2, Wp = 3;
  const size_t prefix = 5, Np = 6, T = prefix + Np;
  HostView3<Scalar> hq("hq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> hk("hk", T, (size_t)H, (size_t)Dh);
  for (size_t t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        hq(t, h, d) = fill_val((int)t, h, d);
        hk(t, h, d) = -0.5f * fill_val((int)t, h, d);
      }
  DevView3<Scalar, Exec> q("q", T, H, Dh), k("k", T, H, Dh);
  DevView3<Scalar, Exec> qo("qo", T, H, Dh), ko("ko", T, H, Dh);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  auto inv = KokkosDNN::rope2d_inv_freq<Exec>(Dh, 100.0f);
  KokkosDNN::rope2d_axial_forward<Exec, Scalar>(q, k, qo, ko, inv, Hp, Wp,
                                               prefix);
  Kokkos::fence();
  // Inputs unchanged by the out-of-place path.
  HostView3<Scalar> cq("cq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> ck("ck", T, (size_t)H, (size_t)Dh);
  Kokkos::deep_copy(cq, q);
  Kokkos::deep_copy(ck, k);
  for (size_t t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        EXPECT_FLOAT_EQ(cq(t, h, d), hq(t, h, d));
        EXPECT_FLOAT_EQ(ck(t, h, d), hk(t, h, d));
      }
  // In-place on copies must agree bit-for-bit (same kernel, same order).
  DevView3<Scalar, Exec> qi("qi", T, H, Dh), ki("ki", T, H, Dh);
  Kokkos::deep_copy(qi, q);
  Kokkos::deep_copy(ki, k);
  KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(qi, ki, inv, Hp, Wp, prefix);
  Kokkos::fence();
  HostView3<Scalar> gqo("gqo", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> gko("gko", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> gqi("gqi", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> gki("gki", T, (size_t)H, (size_t)Dh);
  Kokkos::deep_copy(gqo, qo);
  Kokkos::deep_copy(gko, ko);
  Kokkos::deep_copy(gqi, qi);
  Kokkos::deep_copy(gki, ki);
  for (size_t t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        EXPECT_FLOAT_EQ(gqo(t, h, d), gqi(t, h, d))
            << "q out-of-place vs in-place at (" << t << "," << h << "," << d
            << ")";
        EXPECT_FLOAT_EQ(gko(t, h, d), gki(t, h, d))
            << "k out-of-place vs in-place at (" << t << "," << h << "," << d
            << ")";
      }
}

TEST(RoPE2D, RejectsBadArgs) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const int Dh = 64, H = 1;
  const size_t prefix = 5;
  DevView3<Scalar, Exec> q("q", 9, H, Dh), k("k", 9, H, Dh);
  auto inv   = KokkosDNN::rope2d_inv_freq<Exec>(Dh, 100.0f);
  auto inv16 = KokkosDNN::rope2d_inv_freq<Exec>(16, 100.0f);  // wrong length
  // T=9 != prefix + Hp*Wp.
  EXPECT_THROW(
      (KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(q, k, inv, 3, 3, prefix)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(q, k, inv16, 2, 2,
                                                    prefix)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(q, k, inv, 0, 4, prefix)),
      std::invalid_argument);
  DevView3<Scalar, Exec> qo("qo", 9, H, Dh), ko("ko", 9, H, Dh);
  EXPECT_THROW((KokkosDNN::rope2d_axial_forward<Exec, Scalar>(q, k, qo, ko,
                                                             inv, 3, 3,
                                                             prefix)),
               std::invalid_argument);
}

TEST(RoPE2D, BHalfNormPreserved) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = Kokkos::Experimental::bhalf_t;
  const int Dh = 64, H = 2, Hp = 2, Wp = 2;
  const size_t prefix = 5, Np = 4, T = prefix + Np;
  HostView3<Scalar> hq("hq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> hk("hk", T, (size_t)H, (size_t)Dh);
  for (size_t t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < Dh; ++d) {
        hq(t, h, d) = static_cast<Scalar>(fill_val((int)t, h, d));
        hk(t, h, d) = static_cast<Scalar>(-0.5f * fill_val((int)t, h, d));
      }
  DevView3<Scalar, Exec> q("q", T, H, Dh), k("k", T, H, Dh);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  auto inv = KokkosDNN::rope2d_inv_freq<Exec>(Dh, 100.0f);
  KokkosDNN::rope2d_axial_inplace<Exec, Scalar>(q, k, inv, Hp, Wp, prefix);
  Kokkos::fence();
  HostView3<Scalar> gq("gq", T, (size_t)H, (size_t)Dh);
  HostView3<Scalar> gk("gk", T, (size_t)H, (size_t)Dh);
  Kokkos::deep_copy(gq, q);
  Kokkos::deep_copy(gk, k);
  for (size_t t = prefix; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      float n0 = 0, n1 = 0, m0 = 0, m1 = 0;
      for (int d = 0; d < Dh; ++d) {
        float a = static_cast<float>(hq(t, h, d));
        float b = static_cast<float>(gq(t, h, d));
        float c = static_cast<float>(hk(t, h, d));
        float e = static_cast<float>(gk(t, h, d));
        n0 += a * a;
        n1 += b * b;
        m0 += c * c;
        m1 += e * e;
      }
      EXPECT_NEAR(n1, n0, 1e-2f * std::abs(n0)) << "q norm at (" << t << ","
                                               << h << ")";
      EXPECT_NEAR(m1, m0, 1e-2f * std::abs(m0)) << "k norm at (" << t << ","
                                               << h << ")";
    }
}
