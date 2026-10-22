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

// Tile-edge coverage for the tiled native kernel: exercises M/N/K tile
// remainders, the vec/scalar load paths (K%4==0 vs odd K), and non-trivial
// alpha/beta against a host reference. M=17 routes to the mid geometry
// (TM=32/TN=64/TK=32: M edge + second/third N tiles, K edge over 3 slabs),
// M=13 to thin-fine (TM=16/TN=32/TK=64: N edge + K edge).
template <class Scalar>
void check_gemm_edges(int M, int N, int K, float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  // Force the native kernel even when cuBLAS dispatch is available.
  KokkosDNN::DNNHandle<Exec> h;
  h.use_cublas = false;
  h.use_cudnn  = false;
  h.use_accelerate = false;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hy = Kokkos::create_mirror_view(Y);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      hx(i, k) = static_cast<Scalar>(0.05f * ((i * 37 + k * 11) % 41) - 0.5f);
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      hw(j, k) = static_cast<Scalar>(0.05f * ((j * 23 + k * 5) % 37) - 0.4f);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      hy(i, j) = static_cast<Scalar>(0.1f * ((i + j) % 7) - 0.3f);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(Y, hy);
  // Snapshot the beta input: on host execution spaces the mirror hy may
  // alias Y, so gemm would overwrite the reference values (same pattern as
  // PortableHost below; host mirrors of device views on team-capable
  // backends are separate allocations, where this aliasing does not happen).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace> hy0("hy0", M,
                                                                    N);
  Kokkos::deep_copy(hy0, hy);
  const float alpha = 2.0f, beta = 0.5f;
  KokkosDNN::gemm(h, static_cast<Scalar>(alpha), X, W,
                  static_cast<Scalar>(beta), Y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(got, Y);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(hx(i, k)) * static_cast<float>(hw(j, k));
      float want = alpha * acc + beta * static_cast<float>(hy0(i, j));
      EXPECT_NEAR(static_cast<float>(got(i, j)), want, tol)
          << "gemm edge mismatch at (" << i << "," << j << ") M=" << M
          << " N=" << N << " K=" << K;
    }
}

TEST(GemmEdges, FloatBase) { check_gemm_edges<float>(17, 130, 68, 2e-3f); }
TEST(GemmEdges, FloatSkinny) { check_gemm_edges<float>(13, 70, 67, 2e-3f); }
TEST(GemmEdges, BHalfBase) {
  check_gemm_edges<Kokkos::Experimental::bhalf_t>(17, 130, 68, 0.5f);
}
TEST(GemmEdges, BHalfSkinny) {
  check_gemm_edges<Kokkos::Experimental::bhalf_t>(13, 70, 67, 0.5f);
}

// Portability: the host execution space cannot always field a team
// (DefaultHostExecutionSpace is Serial with max team size 1 on CUDA and
// Serial-only builds), so gemm must take the naive fallback there -- same
// result, verified against the host reference (team-capable backends run
// the tiled kernel through the same gemm entry point).
TEST(GemmEdges, PortableHost) {
  using Exec = Kokkos::DefaultHostExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int M = 13, N = 70, K = 67;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hy = Kokkos::create_mirror_view(Y);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      hx(i, k) = 0.05f * ((i * 37 + k * 11) % 41) - 0.5f;
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      hw(j, k) = 0.05f * ((j * 23 + k * 5) % 37) - 0.4f;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) hy(i, j) = 0.1f * ((i + j) % 7) - 0.3f;
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(Y, hy);
  // Snapshot the beta input: on host execution spaces the mirror hy may
  // alias Y, so gemm would overwrite the reference values (host mirrors of
  // device views are separate allocations, where this aliasing does not
  // happen).
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> hy0("hy0", M,
                                                                    N);
  Kokkos::deep_copy(hy0, hy);
  KokkosDNN::gemm(h, 2.0f, X, W, 0.5f, Y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(got, Y);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k) acc += hx(i, k) * hw(j, k);
      float want = 2.0f * acc + 0.5f * hy0(i, j);
      EXPECT_NEAR(got(i, j), want, 2e-3f)
          << "host gemm mismatch at (" << i << "," << j << ")";
    }
}
