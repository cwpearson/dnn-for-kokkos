// GEMM: Y = X * W^T vs host reference (float + bfloat16).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>

#include "KokkosDNN.hpp"

template <class Scalar>
void check_gemm(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int M = 5, K = 7, N = 4;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hy = Kokkos::create_mirror_view(Y);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      hx(i, k) = static_cast<Scalar>(0.1f * (i * K + k + 1));
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      hw(j, k) = static_cast<Scalar>(0.01f * (j * K - k + 3));
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) hy(i, j) = static_cast<Scalar>(1.0f);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(Y, hy);
  KokkosDNN::gemm(h, static_cast<Scalar>(2), X, W, static_cast<Scalar>(1), Y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(got, Y);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(hx(i, k)) * static_cast<float>(hw(j, k));
      float want = 2 * acc + 1;
      EXPECT_NEAR(static_cast<float>(got(i, j)), want, tol)
          << "gemm mismatch at (" << i << "," << j << ")";
    }
  // gemv path as well (decode shape).
  {
    Kokkos::View<Scalar*, Exec> x("x", K), y("y", N);
    auto hxv = Kokkos::create_mirror_view(x);
    for (int k = 0; k < K; ++k) hxv(k) = hx(0, k);
    Kokkos::deep_copy(x, hxv);
    KokkosDNN::gemv(h, x, W, y);
    Kokkos::fence();
    auto gotv = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(gotv, y);
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(hxv(k)) * static_cast<float>(hw(j, k));
      EXPECT_NEAR(static_cast<float>(gotv(j)), acc, tol)
          << "gemv mismatch at (" << j << ")";
    }
  }
}

TEST(Gemm, Float) { check_gemm<float>(1e-4f); }

TEST(Gemm, BHalf) { check_gemm<Kokkos::Experimental::bhalf_t>(5e-2f); }
