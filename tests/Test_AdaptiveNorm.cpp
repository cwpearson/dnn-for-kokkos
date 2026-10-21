// Adaptive norms (instance / AdaIN / AdaLN): device vs host reference,
// argument validation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "KokkosDNN_AdaptiveNorm.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;

float probe(size_t i) {
  return 0.043f * (float)((i * 41 + 5) % 23) - 0.45f;
}

void host_instance(const std::vector<float>& x, size_t T, size_t C, float eps,
                   std::vector<float>& y) {
  y.assign(T * C, 0.0f);
  for (size_t c = 0; c < C; ++c) {
    double sum = 0.0;
    for (size_t t = 0; t < T; ++t) sum += x[t * C + c];
    const double mu = sum / (double)T;
    double sq = 0.0;
    for (size_t t = 0; t < T; ++t) {
      const double d = x[t * C + c] - mu;
      sq += d * d;
    }
    const double inv = 1.0 / std::sqrt(sq / (double)T + eps);
    for (size_t t = 0; t < T; ++t)
      y[t * C + c] = (float)((x[t * C + c] - mu) * inv);
  }
}

void host_adax(const std::vector<float>& n, const std::vector<float>& h,
               size_t T, size_t C, std::vector<float>& y) {
  y.assign(T * C, 0.0f);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c)
      y[t * C + c] = (1.0f + h[c]) * n[t * C + c] + h[C + c];
}

auto f2fill = [](auto& v, const std::vector<float>& s) {
  auto h = Kokkos::create_mirror_view(v);
  size_t k = 0;
  for (size_t a = 0; a < h.extent(0); ++a)
    for (size_t c = 0; c < h.extent(1); ++c) h(a, c) = s[k++];
  Kokkos::deep_copy(v, h);
};

TEST(AdaptiveNorm, Instance) {
  const size_t T = 9, C = 6;
  std::vector<float> hx(T * C);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = probe(i);
  std::vector<float> ref;
  host_instance(hx, T, C, 1e-5f, ref);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C), y("y", T, C);
  f2fill(x, hx);
  KokkosDNN::instance_norm1d<Exec, float>(x, 1e-5f, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c)
      EXPECT_NEAR(got(t, c), ref[t * C + c], 1e-5f);
}

TEST(AdaptiveNorm, AdaIN) {
  const size_t T = 7, C = 5, S = 4;
  std::vector<float> hx(T * C), hs(S), hw(2 * C * S), hb(2 * C);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = probe(i + 1);
  for (size_t i = 0; i < hs.size(); ++i) hs[i] = 0.3f * probe(i + 2);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.2f * probe(i + 3);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = probe(i + 4);
  std::vector<float> nn;
  host_instance(hx, T, C, 1e-5f, nn);
  // h = hw @ hs + hb.
  std::vector<float> hh(2 * C, 0.0f);
  for (size_t r = 0; r < 2 * C; ++r) {
    hh[r] = hb[r];
    for (size_t j = 0; j < S; ++j) hh[r] += hw[r * S + j] * hs[j];
  }
  std::vector<float> ref;
  host_adax(nn, hh, T, C, ref);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C), y("y", T, C);
  Kokkos::View<float*, Exec> s("s", S);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", 2 * C, S);
  Kokkos::View<float*, Exec> b("b", 2 * C);
  f2fill(x, hx);
  {
    auto h = Kokkos::create_mirror_view(s);
    for (size_t i = 0; i < S; ++i) h(i) = hs[i];
    Kokkos::deep_copy(s, h);
  }
  f2fill(w, hw);
  {
    auto h = Kokkos::create_mirror_view(b);
    for (size_t i = 0; i < 2 * C; ++i) h(i) = hb[i];
    Kokkos::deep_copy(b, h);
  }
  KokkosDNN::adain1d_forward<Exec, float>(x, s, w, b, 1e-5f, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c)
      EXPECT_NEAR(got(t, c), ref[t * C + c], 1e-5f);
}

TEST(AdaptiveNorm, AdaLN) {
  const size_t T = 6, C = 8, S = 3;
  std::vector<float> hx(T * C), hs(S), hw(2 * C * S), hb(2 * C);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = probe(i + 11);
  for (size_t i = 0; i < hs.size(); ++i) hs[i] = 0.3f * probe(i + 12);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.2f * probe(i + 13);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = probe(i + 14);
  // Layer-norm reference over channels per row.
  std::vector<float> nn(T * C);
  for (size_t t = 0; t < T; ++t) {
    double sum = 0.0;
    for (size_t c = 0; c < C; ++c) sum += hx[t * C + c];
    const double mu = sum / (double)C;
    double sq = 0.0;
    for (size_t c = 0; c < C; ++c) {
      const double d = hx[t * C + c] - mu;
      sq += d * d;
    }
    const double inv = 1.0 / std::sqrt(sq / (double)C + 1e-5);
    for (size_t c = 0; c < C; ++c)
      nn[t * C + c] = (float)((hx[t * C + c] - mu) * inv);
  }
  std::vector<float> hh(2 * C, 0.0f);
  for (size_t r = 0; r < 2 * C; ++r) {
    hh[r] = hb[r];
    for (size_t j = 0; j < S; ++j) hh[r] += hw[r * S + j] * hs[j];
  }
  std::vector<float> ref;
  host_adax(nn, hh, T, C, ref);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", T, C), y("y", T, C);
  Kokkos::View<float*, Exec> s("s", S);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", 2 * C, S);
  Kokkos::View<float*, Exec> b("b", 2 * C);
  f2fill(x, hx);
  {
    auto h = Kokkos::create_mirror_view(s);
    for (size_t i = 0; i < S; ++i) h(i) = hs[i];
    Kokkos::deep_copy(s, h);
  }
  f2fill(w, hw);
  {
    auto h = Kokkos::create_mirror_view(b);
    for (size_t i = 0; i < 2 * C; ++i) h(i) = hb[i];
    Kokkos::deep_copy(b, h);
  }
  KokkosDNN::adaln_forward<Exec, float>(x, s, w, b, 1e-5f, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c)
      EXPECT_NEAR(got(t, c), ref[t * C + c], 1e-5f);
}

TEST(AdaptiveNorm, BadArgsThrow) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", 4, 3),
      y("y", 4, 3);
  Kokkos::View<float*, Exec> s("s", 2);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w("w", 6, 2);
  Kokkos::View<float*, Exec> b("b", 6);
  auto bad1 = [&]() { KokkosDNN::instance_norm1d<Exec, float>(x, -1.0f, y); };
  EXPECT_THROW(bad1(), std::invalid_argument);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> wbad("w", 4, 2);
  auto bad2 = [&]() {
    KokkosDNN::adain1d_forward<Exec, float>(x, s, wbad, b, 1e-5f, y);
  };
  EXPECT_THROW(bad2(), std::invalid_argument);
}

}  // namespace
