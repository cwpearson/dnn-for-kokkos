// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Causal depthwise 1D conv: prefill vs host reference, stepwise decode
// equivalence with the streaming cache, argument validation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "KokkosDNN_Conv1D.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::HostSpace;

// Host reference: y[t,c] = b[c] + sum_j w[c,j] * x_pad[t-(K-1)+j, c]
// (zero padding for negative source indices).
void host_conv1d_ref(
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& w,
    const Kokkos::View<float*, Host>& b, bool biased,
    const Kokkos::View<float**, Kokkos::LayoutRight, Host>& y) {
  const size_t T = x.extent(0), C = x.extent(1), K = w.extent(1);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c) {
      double acc = biased ? (double)b(c) : 0.0;
      for (size_t j = 0; j < K; ++j) {
        long s = (long)t - (long)(K - 1) + (long)j;
        if (s < 0) continue;
        acc += (double)w(c, j) * (double)x((size_t)s, c);
      }
      y(t, c) = static_cast<float>(acc);
    }
}

float probe(size_t a, size_t b) {
  return 0.07f * static_cast<float>((a * 37 + b * 23) % 23) - 0.6f;
}

struct Fixture {
  size_t T, C, K;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hx, hw, ref;
  Kokkos::View<float*, Host> hb;
  bool biased;
  Fixture(size_t T_, size_t C_, size_t K_, bool biased_)
      : T(T_), C(C_), K(K_), biased(biased_) {
    hx = Kokkos::View<float**, Kokkos::LayoutRight, Host>("hx", T, C);
    hw = Kokkos::View<float**, Kokkos::LayoutRight, Host>("hw", C, K);
    ref = Kokkos::View<float**, Kokkos::LayoutRight, Host>("ref", T, C);
    hb = Kokkos::View<float*, Host>("hb", biased ? C : 0);
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c) hx(t, c) = probe(t, c + T);
    for (size_t c = 0; c < C; ++c) {
      for (size_t k = 0; k < K; ++k) hw(c, k) = probe(c, k * 5 + 1);
      if (biased) hb(c) = 0.3f * (float)c - 0.2f;
    }
    host_conv1d_ref(hx, hw, hb, biased, ref);
  }
};

void check_forward(const Fixture& f, float tol) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", f.T, f.C),
      w("w", f.C, f.K), y("y", f.T, f.C);
  Kokkos::View<float*, Exec> b("b", f.biased ? f.C : 0);
  Kokkos::deep_copy(x, f.hx);
  Kokkos::deep_copy(w, f.hw);
  if (f.biased) Kokkos::deep_copy(b, f.hb);
  KokkosDNN::depthwise_conv1d_causal_forward<Exec, float>(x, w, b, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  for (size_t t = 0; t < f.T; ++t)
    for (size_t c = 0; c < f.C; ++c)
      EXPECT_NEAR(got(t, c), f.ref(t, c), tol) << " at (" << t << "," << c
                                              << ")";
}

// Gather row t of x into xt. Free function (not inside TEST body): NVCC
// rejects extended __host__ __device__ lambdas whose enclosing parent
// function (gtest's TestBody) has private access.
template <typename ExecSpace, typename Scalar>
void gather_row(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, size_t t,
    const Kokkos::View<Scalar*, ExecSpace>& xt) {
  Kokkos::parallel_for(
      "gather", Kokkos::RangePolicy<ExecSpace>(0, xt.extent(0)),
      KOKKOS_LAMBDA(size_t c) { xt(c) = x(t, c); });
}

// Step through all T tokens with a zero-initialized cache; every step output
// must equal the corresponding prefill row.
void check_stepwise(const Fixture& f, float tol) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", f.T, f.C),
      w("w", f.C, f.K);
  Kokkos::View<float*, Exec> b("b", f.biased ? f.C : 0);
  Kokkos::deep_copy(x, f.hx);
  Kokkos::deep_copy(w, f.hw);
  if (f.biased) Kokkos::deep_copy(b, f.hb);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> cache("cache", f.C,
                                                         f.K - 1);
  Kokkos::deep_copy(cache, 0.0f);
  Kokkos::View<float*, Exec> xt("xt", f.C), yt("yt", f.C);
  for (size_t t = 0; t < f.T; ++t) {
    gather_row<Exec, float>(x, t, xt);
    KokkosDNN::depthwise_conv1d_causal_step<Exec, float>(xt, w, b, cache, yt);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(yt);
    Kokkos::deep_copy(got, yt);
    for (size_t c = 0; c < f.C; ++c)
      EXPECT_NEAR(got(c), f.ref(t, c), tol) << " step " << t << " ch " << c;
  }
}

}  // namespace

TEST(Conv1D, ForwardK3BiasedMatchesHost) {
  check_forward(Fixture(8, 4, 3, true), 1e-5f);
}

TEST(Conv1D, ForwardArbitraryKMatchesHost) {
  check_forward(Fixture(9, 3, 1, true), 1e-5f);
  check_forward(Fixture(9, 3, 5, true), 1e-5f);
  check_forward(Fixture(6, 5, 4, false), 1e-5f);
  check_forward(Fixture(1, 2, 3, true), 1e-5f);  // single-token prefill
}

TEST(Conv1D, StepwiseDecodeEqualsPrefill) {
  check_stepwise(Fixture(8, 4, 3, true), 1e-5f);
  check_stepwise(Fixture(7, 3, 5, true), 1e-5f);
  check_stepwise(Fixture(6, 5, 4, false), 1e-5f);
  check_stepwise(Fixture(5, 2, 1, true), 1e-5f);  // K=1: empty cache
}

TEST(Conv1D, StepwiseBHalfConsistent) {
  // bhalf_t: prefill vs stepwise consistency (no host reference; avoids
  // double-rounding tolerance questions).
  using Scalar = Kokkos::Experimental::bhalf_t;
  const size_t T = 6, C = 4, K = 3;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, C), w("w", C, K),
      y("y", T, C);
  Kokkos::View<Scalar*, Exec> b("b", C);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hb = Kokkos::create_mirror_view(b);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c) hx(t, c) = static_cast<Scalar>(probe(t, c));
  for (size_t c = 0; c < C; ++c) {
    for (size_t k = 0; k < K; ++k) hw(c, k) = static_cast<Scalar>(probe(c, k));
    hb(c) = static_cast<Scalar>(0.1f * (float)c);
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::depthwise_conv1d_causal_forward<Exec, Scalar>(x, w, b, y);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> cache("cache", C, K - 1);
  Kokkos::deep_copy(cache, Scalar(0));
  Kokkos::View<Scalar*, Exec> xt("xt", C), yt("yt", C);
  for (size_t t = 0; t < T; ++t) {
    gather_row<Exec, Scalar>(x, t, xt);
    KokkosDNN::depthwise_conv1d_causal_step<Exec, Scalar>(xt, w, b, cache, yt);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(yt);
    Kokkos::deep_copy(got, yt);
    auto grow = Kokkos::create_mirror_view(
        Kokkos::subview(y, t, Kokkos::ALL()));
    Kokkos::deep_copy(grow, Kokkos::subview(y, t, Kokkos::ALL()));
    for (size_t c = 0; c < C; ++c)
      EXPECT_NEAR(static_cast<float>(got(c)), static_cast<float>(grow(c)),
                  5e-2f)
          << " step " << t << " ch " << c;
  }
}

TEST(Conv1D, RejectsBadArgs) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> x("x", 4, 3), y("y", 4, 3),
      w("w", 3, 3), bad_w("bad_w", 4, 3), k0_w("k0_w", 3, 0),
      bad_y("bad_y", 5, 3);
  Kokkos::View<float*, Exec> b("b", 3), bad_b("bad_b", 2), empty_b;
  EXPECT_NO_THROW(
      (KokkosDNN::depthwise_conv1d_causal_forward<Exec, float>(x, w, b, y)));
  EXPECT_NO_THROW((KokkosDNN::depthwise_conv1d_causal_forward<Exec, float>(
      x, w, empty_b, y)));
  EXPECT_THROW((KokkosDNN::depthwise_conv1d_causal_forward<Exec, float>(
                    x, bad_w, b, y)),
               std::invalid_argument);
  EXPECT_THROW((KokkosDNN::depthwise_conv1d_causal_forward<Exec, float>(
                    x, w, bad_b, y)),
               std::invalid_argument);
  EXPECT_THROW((KokkosDNN::depthwise_conv1d_causal_forward<Exec, float>(
                    x, w, b, bad_y)),
               std::invalid_argument);
  EXPECT_THROW((KokkosDNN::depthwise_conv1d_causal_forward<Exec, float>(
                    x, k0_w, b, y)),
               std::invalid_argument);
  // Step: cache must be [C, K-1].
  Kokkos::View<float*, Exec> xt("xt", 3), yt("yt", 3);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> cache("cache", 3, 2),
      bad_cache("bad_cache", 3, 1), k1_cache("k1_cache", 3, 0);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w1("w1", 3, 1);
  EXPECT_NO_THROW(
      (KokkosDNN::depthwise_conv1d_causal_step<Exec, float>(xt, w, b, cache,
                                                           yt)));
  EXPECT_THROW(
      (KokkosDNN::depthwise_conv1d_causal_step<Exec, float>(xt, w, b, bad_cache,
                                                           yt)),
      std::invalid_argument);
  EXPECT_NO_THROW(
      (KokkosDNN::depthwise_conv1d_causal_step<Exec, float>(xt, w1, empty_b,
                                                           k1_cache, yt)));
}
