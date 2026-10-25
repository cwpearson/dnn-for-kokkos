// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// gemm_bias / gemv_bias vs host reference (float + half + bfloat16):
// Y = X * W^T + b with row-broadcast bias and an empty-bias unbiased path.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <stdexcept>

#include "KokkosDNN.hpp"

template <class Scalar>
void check_gemm_bias(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  // Non-square (M != N != K, K != N) so a missing-transpose (X*W) mixup
  // cannot accidentally pass on shapes alone.
  const int M = 4, K = 6, N = 5;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  Kokkos::View<Scalar*, Exec> b("b", N);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hb = Kokkos::create_mirror_view(b);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      hx(i, k) = static_cast<Scalar>(0.1f * (i * K + k + 1) - 0.3f);
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      hw(j, k) = static_cast<Scalar>(0.05f * (j * K - k + 2));
  for (int j = 0; j < N; ++j)
    hb(j) = static_cast<Scalar>(-0.5f + 0.25f * j);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::gemm_bias(h, X, W, b, Y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(got, Y);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(hx(i, k)) * static_cast<float>(hw(j, k));
      float want = acc + static_cast<float>(hb(j));
      EXPECT_NEAR(static_cast<float>(got(i, j)), want, tol)
          << "gemm_bias mismatch at (" << i << "," << j << ")";
    }
}

TEST(GemmBias, Float) { check_gemm_bias<float>(1e-4f); }

TEST(GemmBias, BHalf) { check_gemm_bias<Kokkos::Experimental::bhalf_t>(5e-2f); }

TEST(GemmBias, Half) { check_gemm_bias<Kokkos::Experimental::half_t>(5e-2f); }

// Skinny M=1 path (time-modulation GEMMs are B=1 rows): fused result vs
// host reference, on both the dispatched handle (cublasLt BIAS epilogue
// on CUDA) and a forced-native handle (bias folded into the GEMV
// write-back). Non-square K != N so a transpose mixup cannot hide.
template <class Scalar>
void check_gemm_bias_m1(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  for (int force_native = 0; force_native < 2; ++force_native) {
    KokkosDNN::DNNHandle<Exec> h;
    if (force_native) {
      h.use_cublas  = false;
      h.use_cudnn   = false;
      h.use_hipblas = false;
      h.use_miopen  = false;
      h.use_accelerate = false;
    }
    const int M = 1, K = 32, N = 48;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K),
        W("W", N, K), Y("Y", M, N);
    Kokkos::View<Scalar*, Exec> b("b", N);
    auto hx = Kokkos::create_mirror_view(X);
    auto hw = Kokkos::create_mirror_view(W);
    auto hb = Kokkos::create_mirror_view(b);
    // Bounded modular fills: outputs stay O(1) so absolute tolerances are
    // meaningful (unbounded fills reach O(1000), where summation-order
    // noise alone exceeds a tight abs tol).
    for (int k = 0; k < K; ++k)
      hx(0, k) = static_cast<Scalar>(0.02f * ((k * 7) % 17) - 0.1f);
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        hw(j, k) = static_cast<Scalar>(0.02f * ((j + k * 3) % 19) - 0.1f);
    for (int j = 0; j < N; ++j)
      hb(j) = static_cast<Scalar>(0.05f * (j % 11) - 0.2f);
    Kokkos::deep_copy(X, hx);
    Kokkos::deep_copy(W, hw);
    Kokkos::deep_copy(b, hb);
    KokkosDNN::gemm_bias(h, X, W, b, Y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(Y);
    Kokkos::deep_copy(got, Y);
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(hx(0, k)) * static_cast<float>(hw(j, k));
      float want = acc + static_cast<float>(hb(j));
      EXPECT_NEAR(static_cast<float>(got(0, j)), want, tol)
          << "gemm_bias M=1 mismatch at (0," << j << ") native=" << force_native;
    }
  }
}

TEST(GemmBias, FloatM1Skinny) { check_gemm_bias_m1<float>(1e-4f); }

TEST(GemmBias, BHalfM1Skinny) {
  check_gemm_bias_m1<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// Fused (single-pass) vs split (gemm + broadcast add) agreement: catches
// orientation, broadcast-axis, or epilogue-shape bugs in the fusion while
// tolerating legitimate kernel differences (cublasLt vs sgemm/GemmEx
// summation order, single vs double output rounding for bf16). Covers the
// M=1 row path and a multi-row tile path on both handles.
template <class Scalar>
void check_gemm_bias_matches_split(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int Ms[2] = {1, 5};
  const int K = 8, N = 7;
  for (int force_native = 0; force_native < 2; ++force_native) {
    for (int mi = 0; mi < 2; ++mi) {
      const int M = Ms[mi];
      KokkosDNN::DNNHandle<Exec> h;
      if (force_native) {
        h.use_cublas  = false;
        h.use_cudnn   = false;
        h.use_hipblas = false;
        h.use_miopen  = false;
      }
      Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K),
          W("W", N, K), Yf("Yf", M, N), Ys("Ys", M, N);
      Kokkos::View<Scalar*, Exec> b("b", N);
      auto hx = Kokkos::create_mirror_view(X);
      auto hw = Kokkos::create_mirror_view(W);
      auto hb = Kokkos::create_mirror_view(b);
      for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
          hx(i, k) = static_cast<Scalar>(0.1f * (i * K + k + 1) - 0.3f);
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
          hw(j, k) = static_cast<Scalar>(0.05f * (j * K - k + 2));
      for (int j = 0; j < N; ++j)
        hb(j) = static_cast<Scalar>(-0.5f + 0.25f * j);
      Kokkos::deep_copy(X, hx);
      Kokkos::deep_copy(W, hw);
      Kokkos::deep_copy(b, hb);
      KokkosDNN::gemm_bias(h, X, W, b, Yf);
      KokkosDNN::gemm(h, static_cast<Scalar>(1), X, W, static_cast<Scalar>(0),
                      Ys);
      KokkosDNN::Impl::add_bias_2d<Exec, Scalar>(Ys, b, "split_bias");
      Kokkos::fence();
      auto gotf = Kokkos::create_mirror_view(Yf);
      Kokkos::deep_copy(gotf, Yf);
      auto gots = Kokkos::create_mirror_view(Ys);
      Kokkos::deep_copy(gots, Ys);
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
          EXPECT_NEAR(static_cast<float>(gotf(i, j)),
                      static_cast<float>(gots(i, j)), tol)
              << "fused-vs-split mismatch at (" << i << "," << j << ") M=" << M
              << " native=" << force_native;
    }
  }
}

TEST(GemmBias, FloatMatchesSplit) { check_gemm_bias_matches_split<float>(1e-3f); }

TEST(GemmBias, BHalfMatchesSplit) {
  check_gemm_bias_matches_split<Kokkos::Experimental::bhalf_t>(1e-1f);
}

// Hand-computed transpose pin: identity rows of W select columns of X,
// so Y = X * W^T + b is unambiguous (no-transpose X*W is not even
// shape-valid here: [2,3] * [4,3]).
TEST(GemmBias, TransposeSemantics) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", 2, 3),
      W("W", 4, 3), Y("Y", 2, 4);
  Kokkos::View<float*, Exec> b("b", 4);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hb = Kokkos::create_mirror_view(b);
  hx(0, 0) = 1;
  hx(0, 1) = 2;
  hx(0, 2) = 3;
  hx(1, 0) = 4;
  hx(1, 1) = 5;
  hx(1, 2) = 6;
  for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 3; ++k) hw(j, k) = (j < 3 && j == k) ? 1.0f : (j == 3 ? 1.0f : 0.0f);
  hb(0) = 10;
  hb(1) = 20;
  hb(2) = 30;
  hb(3) = 40;
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::gemm_bias(h, X, W, b, Y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(got, Y);
  // Row 0: [1, 2, 3, 6] + b; row 1: [4, 5, 6, 15] + b.
  const float want[2][4] = {{11, 22, 33, 46}, {14, 25, 36, 55}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(got(i, j), want[i][j], 1e-5f)
          << "transpose pin mismatch at (" << i << "," << j << ")";
}

// Empty (default-constructed) bias = unbiased path (DINOv3 k projection):
// must match both the host reference without bias and plain gemm output.
TEST(GemmBias, UnbiasedEmptyBiasMatchesGemm) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int M = 3, K = 5, N = 4;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Yb("Yb", M, N), Yg("Yg", M, N);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) hx(i, k) = 0.2f * (i * K + k + 1) - 0.5f;
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k) hw(j, k) = 0.07f * (j * K - k + 4);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::View<float*, Exec> empty_b;  // extent 0 -> unbiased
  ASSERT_EQ(empty_b.extent(0), 0u);
  KokkosDNN::gemm_bias(h, X, W, empty_b, Yb);
  KokkosDNN::gemm(h, 1.0f, X, W, 0.0f, Yg);
  Kokkos::fence();
  auto gotb = Kokkos::create_mirror_view(Yb);
  Kokkos::deep_copy(gotb, Yb);
  auto gotg = Kokkos::create_mirror_view(Yg);
  Kokkos::deep_copy(gotg, Yg);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k) acc += hx(i, k) * hw(j, k);
      EXPECT_NEAR(gotb(i, j), acc, 1e-5f)
          << "unbiased ref mismatch at (" << i << "," << j << ")";
      EXPECT_NEAR(gotb(i, j), gotg(i, j), 1e-6f)
          << "unbiased-vs-gemm mismatch at (" << i << "," << j << ")";
    }
}

TEST(GemmBias, Rank1BiasedAndUnbiased) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int K = 6, N = 5;
  Kokkos::View<float*, Exec> x("x", K), b("b", N), y("y", N), yu("yu", N);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> W("W", N, K);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(W);
  auto hb = Kokkos::create_mirror_view(b);
  for (int k = 0; k < K; ++k) hx(k) = 0.3f * (k + 1) - 0.4f;
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k) hw(j, k) = 0.11f * (j + 1) - 0.02f * k;
  for (int j = 0; j < N; ++j) hb(j) = -1.0f + 0.5f * j;
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::gemv_bias(h, x, W, b, y);
  Kokkos::View<float*, Exec> empty_b;
  KokkosDNN::gemv_bias(h, x, W, empty_b, yu);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  auto gotu = Kokkos::create_mirror_view(yu);
  Kokkos::deep_copy(gotu, yu);
  for (int j = 0; j < N; ++j) {
    float acc = 0;
    for (int k = 0; k < K; ++k) acc += hx(k) * hw(j, k);
    EXPECT_NEAR(got(j), acc + hb(j), 1e-5f) << "gemv_bias mismatch at (" << j << ")";
    EXPECT_NEAR(gotu(j), acc, 1e-5f) << "gemv_bias unbiased mismatch at (" << j << ")";
  }
}

TEST(GemmBias, DimensionMismatchThrows) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", 3, 4), W("W", 5, 4),
      Y("Y", 3, 5);
  Kokkos::View<float*, Exec> b("b", 5), bad_b("bad_b", 4);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> badW("badW", 5, 3),
      badY("badY", 3, 4);
  Kokkos::View<float*, Exec> x("x", 4), y("y", 5), bad_y("bad_y", 4);
  EXPECT_THROW((KokkosDNN::gemm_bias(h, X, badW, b, Y)), std::invalid_argument);
  EXPECT_THROW((KokkosDNN::gemm_bias(h, X, W, b, badY)), std::invalid_argument);
  EXPECT_THROW((KokkosDNN::gemm_bias(h, X, W, bad_b, Y)), std::invalid_argument);
  EXPECT_NO_THROW((KokkosDNN::gemm_bias(h, X, W, b, Y)));
  Kokkos::View<float*, Exec> empty_b;
  EXPECT_NO_THROW((KokkosDNN::gemm_bias(h, X, W, empty_b, Y)));
  EXPECT_THROW((KokkosDNN::gemv_bias(h, x, W, bad_b, y)), std::invalid_argument);
  EXPECT_THROW((KokkosDNN::gemv_bias(h, x, W, b, bad_y)), std::invalid_argument);
  EXPECT_NO_THROW((KokkosDNN::gemv_bias(h, x, W, b, y)));
  EXPECT_NO_THROW((KokkosDNN::gemv_bias(h, x, W, empty_b, y)));
}

// CUDA tensor-core path with fused bias (src/KokkosDNN_GemmTensorCuda.hpp):
// K/N-deep aligned shapes take the single-pass wmma fused-bias kernel on
// SM80+ CUDA and the portable fused kernel everywhere else; one numerics
// contract covers both (TF32-aware tolerances, same rationale as
// GemmTensorCores in Test_Gemm.cpp). Covers full tiles and M/N edge
// masking with a K-tail slab.
template <class Scalar>
void check_gemm_bias_large_aligned(int M, int N, int K, float tol_host,
                                   float tol_portable) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  h.use_cublas       = false;
  h.use_cudnn        = false;
  h.use_accelerate   = false;
  h.use_tensor_cores = true;  // native path; portable fallback elsewhere
  KokkosDNN::DNNHandle<Exec> hp;
  hp.use_cublas       = false;
  hp.use_cudnn        = false;
  hp.use_accelerate   = false;
  hp.use_tensor_cores = false;  // forced portable reference
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N), Yp("Yp", M, N);
  Kokkos::View<Scalar*, Exec> b("b", N);
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hb = Kokkos::create_mirror_view(b);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      hx(i, k) = static_cast<Scalar>(0.05f * ((i * 37 + k * 11) % 41) - 0.5f);
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      hw(j, k) = static_cast<Scalar>(0.05f * ((j * 23 + k * 5) % 37) - 0.4f);
  for (int j = 0; j < N; ++j)
    hb(j) = static_cast<Scalar>(2.0f * (j % 9) - 8.0f);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(b, hb);
  KokkosDNN::gemm_bias(h, X, W, b, Y);
  KokkosDNN::gemm_bias(hp, X, W, b, Yp);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(got, Y);
  auto gotp = Kokkos::create_mirror_view(Yp);
  Kokkos::deep_copy(gotp, Yp);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(hx(i, k)) * static_cast<float>(hw(j, k));
      float want = acc + static_cast<float>(hb(j));
      EXPECT_NEAR(static_cast<float>(got(i, j)), want, tol_host)
          << "tensor gemm_bias mismatch at (" << i << "," << j << ") M=" << M
          << " N=" << N << " K=" << K;
      EXPECT_NEAR(static_cast<float>(got(i, j)),
                  static_cast<float>(gotp(i, j)), tol_portable)
          << "tensor-vs-portable bias mismatch at (" << i << "," << j
          << ") M=" << M << " N=" << N << " K=" << K;
    }
}

TEST(GemmBiasTensorCores, FloatFullTiles) {
  check_gemm_bias_large_aligned<float>(128, 256, 256, 0.5f, 0.3f);
}

TEST(GemmBiasTensorCores, FloatEdgesAndKTail) {
  check_gemm_bias_large_aligned<float>(100, 200, 168, 0.5f, 0.3f);
}

TEST(GemmBiasTensorCores, BHalfFullTiles) {
  check_gemm_bias_large_aligned<Kokkos::Experimental::bhalf_t>(128, 256, 256,
                                                               1.0f, 0.25f);
}

TEST(GemmBiasTensorCores, BHalfEdgesAndKTail) {
  check_gemm_bias_large_aligned<Kokkos::Experimental::bhalf_t>(100, 200, 168,
                                                               1.0f, 0.25f);
}
