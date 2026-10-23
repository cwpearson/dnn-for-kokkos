// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Dense conv1d / transposed conv1d: device vs host reference (stride,
// padding, dilation, groups incl. depthwise), argument validation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

#include "KokkosDNN_Conv1DGeneral.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;

void host_conv1d(const std::vector<float>& x, const std::vector<float>& w,
                 const std::vector<float>& b, bool biased, size_t Lin,
                 size_t Cin, size_t Cout, size_t Cg, size_t K, size_t stride,
                 size_t pad, size_t dil, size_t groups,
                 std::vector<float>& y, size_t& Lout) {
  Lout = (Lin + 2 * pad - dil * (K - 1) - 1) / stride + 1;
  const size_t Co = Cout / groups;
  y.assign(Lout * Cout, 0.0f);
  for (size_t t = 0; t < Lout; ++t)
    for (size_t o = 0; o < Cout; ++o) {
      float acc = biased ? b[o] : 0.0f;
      const size_t g = o / Co;
      for (size_t i = 0; i < Cg; ++i)
        for (size_t k = 0; k < K; ++k) {
          const long s = (long)(t * stride + k * dil) - (long)pad;
          if (s < 0 || (size_t)s >= Lin) continue;
          acc += w[(o * Cg + i) * K + k] * x[(size_t)s * Cin + g * Cg + i];
        }
      y[t * Cout + o] = acc;
    }
}

void host_convt(const std::vector<float>& x, const std::vector<float>& w,
                const std::vector<float>& b, bool biased, size_t Lin,
                size_t Cin, size_t Co, size_t K, size_t stride, size_t pad,
                size_t op, size_t groups, std::vector<float>& y, size_t& Lout) {
  const size_t Cout = Co * groups, Cg = Cin / groups;
  Lout = (Lin - 1) * stride - 2 * pad + K + op;
  y.assign(Lout * Cout, 0.0f);
  for (size_t t = 0; t < Lout; ++t)
    for (size_t o = 0; o < Cout; ++o) {
      float acc = biased ? b[o] : 0.0f;
      const size_t g = o / Co, oo = o % Co;
      for (size_t i = 0; i < Cg; ++i)
        for (size_t k = 0; k < K; ++k) {
          const long num = (long)t + (long)pad - (long)k;
          if (num < 0 || num % (long)stride != 0) continue;
          const size_t s = (size_t)num / stride;
          if (s >= Lin) continue;
          acc += w[((g * Cg + i) * Co + oo) * K + k] * x[s * Cin + g * Cg + i];
        }
      y[t * Cout + o] = acc;
    }
}

float probe(size_t i) {
  return 0.037f * (float)((i * 53 + 7) % 31) - 0.5f;
}

TEST(Conv1DGeneral, Dense) {
  const size_t Lin = 9, Cin = 4, Cout = 6, K = 5;
  std::vector<float> hx(Lin * Cin), hw(Cout * Cin * K), hb(Cout);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = probe(i);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.2f * probe(i + 1);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = probe(i + 2);
  std::vector<float> ref;
  size_t Lout = 0;
  host_conv1d(hx, hw, hb, true, Lin, Cin, Cout, Cin, K, 2, 2, 1, 1, ref,
              Lout);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", Lin, Cin),
      y("y", Lout, Cout);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", Cout, Cin, K);
  Kokkos::View<float*, Exec> b("b", Cout);
  auto f2 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t c = 0; c < h.extent(1); ++c) h(a, c) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  auto f3 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t c = 0; c < h.extent(1); ++c)
        for (size_t d = 0; d < h.extent(2); ++d) h(a, c, d) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  auto f1 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t a = 0; a < h.extent(0); ++a) h(a) = s[a];
    Kokkos::deep_copy(v, h);
  };
  f2(x, hx);
  f3(w, hw);
  f1(b, hb);
  KokkosDNN::conv1d_forward<Exec, float>(x, w, b, y, 2, 2, 1, 1);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (size_t t = 0; t < Lout; ++t)
    for (size_t o = 0; o < Cout; ++o)
      EXPECT_NEAR(got(t, o), ref[t * Cout + o], 1e-5f)
          << "at (" << t << "," << o << ")";
}

TEST(Conv1DGeneral, Transpose) {
  const size_t Lin = 5, Cin = 4, Co = 3, K = 4;  // Cout = 6, groups 2
  const size_t Cout = 6, groups = 2;
  std::vector<float> hx(Lin * Cin), hw(Cin * Co * K), hb(Cout);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = probe(i + 9);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.2f * probe(i + 11);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = probe(i + 13);
  std::vector<float> ref;
  size_t Lout = 0;
  host_convt(hx, hw, hb, true, Lin, Cin, Co, K, 3, 1, 0, groups, ref, Lout);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", Lin, Cin),
      y("y", Lout, Cout);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", Cin, Co, K);
  Kokkos::View<float*, Exec> b("b", Cout);
  auto f2 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t c = 0; c < h.extent(1); ++c) h(a, c) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  auto f3 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t c = 0; c < h.extent(1); ++c)
        for (size_t d = 0; d < h.extent(2); ++d) h(a, c, d) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  auto f1 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t a = 0; a < h.extent(0); ++a) h(a) = s[a];
    Kokkos::deep_copy(v, h);
  };
  f2(x, hx);
  f3(w, hw);
  f1(b, hb);
  KokkosDNN::conv_transpose1d_forward<Exec, float>(x, w, b, y, 3, 1, 0, 2);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (size_t t = 0; t < Lout; ++t)
    for (size_t o = 0; o < Cout; ++o)
      EXPECT_NEAR(got(t, o), ref[t * Cout + o], 1e-5f)
          << "at (" << t << "," << o << ")";
}

TEST(Conv1DGeneral, DepthwiseTransposeMatchesGrouped) {
  // groups=Cin depthwise transpose equals grouped path with Co=1.
  const size_t Lin = 6, C = 4, K = 3;
  std::vector<float> hx(Lin * C), hw(C * 1 * K), hb(C);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = probe(i + 21);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.3f * probe(i + 23);
  std::vector<float> ref;
  size_t Lout = 0;
  host_convt(hx, hw, hb, false, Lin, C, 1, K, 2, 1, 1, C, ref, Lout);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", Lin, C),
      y("y", Lout, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", C, 1, K);
  Kokkos::View<float*, Exec> b("b", 0);
  auto f2 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t c = 0; c < h.extent(1); ++c) h(a, c) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  auto f3 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t c = 0; c < h.extent(1); ++c)
        for (size_t d = 0; d < h.extent(2); ++d) h(a, c, d) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  f2(x, hx);
  f3(w, hw);
  KokkosDNN::conv_transpose1d_forward<Exec, float>(x, w, b, y, 2, 1, 1, C);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (size_t t = 0; t < Lout; ++t)
    for (size_t o = 0; o < C; ++o)
      EXPECT_NEAR(got(t, o), ref[t * C + o], 1e-5f)
          << "at (" << t << "," << o << ")";
}

TEST(Conv1DGeneral, BadArgsThrow) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", 4, 3),
      y("y", 4, 2);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> w("w", 2, 3, 3);
  Kokkos::View<float*, Exec> b("b", 2);
  auto bad1 = [&]() {
    KokkosDNN::conv1d_forward<Exec, float>(x, w, b, y, 0, 1, 1, 1);
  };
  EXPECT_THROW(bad1(), std::invalid_argument);
  auto bad2 = [&]() {
    KokkosDNN::conv_transpose1d_forward<Exec, float>(x, w, b, y, 1, 1, 0, 3);
  };
  EXPECT_THROW(bad2(), std::invalid_argument);
}

}  // namespace
