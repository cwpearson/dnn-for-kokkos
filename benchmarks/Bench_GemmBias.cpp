// Biased-linear benchmarks: Native vs Dispatched on Llama 3.1 8B shapes.
//
//   Y[M,N] = X[M,K] * W[N,K]^T + b[N]   (rank-2: gemm_bias / dense_forward)
//   y[N]   = W[N,K] * x[K] + b[N]        (rank-1: gemv_bias)
//
// GEMM reuse: gemm_bias/gemv_bias and dense_forward route the GEMM part
// through fused single-pass paths: cublasLt with the BIAS epilogue on
// CUDA (float via R_32F, bhalf_t via R_16BF with FP32 accumulation, using
// the handle's cached Lt state), bias folded into the native kernel
// write-back elsewhere. When Lt cannot serve a shape, the CUDA path falls
// back to cuBLAS sgemm/cublasGemmEx plus a separate broadcast-bias
// kernel. Two variants per case:
//   Native     — handle with use_cublas=false, always the Kokkos kernel.
//   Dispatched — default handle, routes to cuBLAS when built with
//                KokkosDNN_ENABLE_CUBLAS on a CUDA space; otherwise native.
//
// Run e.g.:
//   ./kokkosdnn_bench --benchmark_filter='GemmBias.*|GemvBias.*|DenseForward.*'
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include "KokkosDNN.hpp"

namespace {

template <class Scalar>
void fill_gemm_bias(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>& W,
    const Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>& Y) {
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hb = Kokkos::create_mirror_view(b);
  auto hy = Kokkos::create_mirror_view(Y);
  for (size_t i = 0; i < hx.extent(0); ++i)
    for (size_t k = 0; k < hx.extent(1); ++k)
      hx(i, k) = static_cast<Scalar>(0.001f * ((i * 31 + k * 17) % 97) - 0.05f);
  for (size_t j = 0; j < hw.extent(0); ++j)
    for (size_t k = 0; k < hw.extent(1); ++k)
      hw(j, k) = static_cast<Scalar>(0.001f * ((j * 13 + k * 7) % 89) - 0.04f);
  for (size_t j = 0; j < hb.extent(0); ++j)
    hb(j) = static_cast<Scalar>(0.001f * ((j * 11) % 83) - 0.04f);
  for (size_t i = 0; i < hy.extent(0); ++i)
    for (size_t j = 0; j < hy.extent(1); ++j) hy(i, j) = static_cast<Scalar>(0);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(b, hb);
  Kokkos::deep_copy(Y, hy);
}

template <class Scalar>
void fill_gemv_bias(
    const Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>& W,
    const Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace>& b) {
  auto hxv = Kokkos::create_mirror_view(x);
  auto hw  = Kokkos::create_mirror_view(W);
  auto hb  = Kokkos::create_mirror_view(b);
  for (size_t k = 0; k < hxv.extent(0); ++k)
    hxv(k) = static_cast<Scalar>(0.001f * ((k * 17) % 97) - 0.05f);
  for (size_t j = 0; j < hw.extent(0); ++j)
    for (size_t k = 0; k < hw.extent(1); ++k)
      hw(j, k) = static_cast<Scalar>(0.001f * ((j * 13 + k * 7) % 89) - 0.04f);
  for (size_t j = 0; j < hb.extent(0); ++j)
    hb(j) = static_cast<Scalar>(0.001f * ((j * 11) % 83) - 0.04f);
  Kokkos::deep_copy(x, hxv);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(b, hb);
}

template <class Scalar>
void run_gemm_bias(benchmark::State& state, bool force_native) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int M = static_cast<int>(state.range(0));
  const int N = static_cast<int>(state.range(1));
  const int K = static_cast<int>(state.range(2));

  KokkosDNN::DNNHandle<Exec> h_native;
  h_native.use_cublas = false;
  h_native.use_cudnn  = false;
  h_native.use_accelerate = false;
  KokkosDNN::DNNHandle<Exec> h_tpl;  // configured default (TPL when enabled)
  const KokkosDNN::DNNHandle<Exec>* hp = force_native ? &h_native : &h_tpl;

  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  Kokkos::View<Scalar*, Exec> b("b", N);
  fill_gemm_bias(X, W, b, Y);
  Kokkos::fence();

  // GFLOPS counts the GEMM (2*M*N*K); the M*N bias add is negligible.
  const double flops = 2.0 * M * N * static_cast<double>(K);
  state.counters["GFLOPS"] = benchmark::Counter(
      1e-9 * flops, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::gemm_bias(*hp, X, W, b, Y);
    Kokkos::fence();
  }
}

template <class Scalar>
void run_gemv_bias(benchmark::State& state, bool force_native) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int N = static_cast<int>(state.range(0));
  const int K = static_cast<int>(state.range(1));

  KokkosDNN::DNNHandle<Exec> h_native;
  h_native.use_cublas = false;
  h_native.use_cudnn  = false;
  h_native.use_accelerate = false;
  KokkosDNN::DNNHandle<Exec> h_tpl;
  const KokkosDNN::DNNHandle<Exec>* hp = force_native ? &h_native : &h_tpl;

  Kokkos::View<Scalar*, Exec> x("x", K), b("b", N), y("y", N);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> W("W", N, K);
  fill_gemv_bias(x, W, b);
  Kokkos::fence();

  // GFLOPS counts the GEMV (2*N*K); the N-element bias add is negligible.
  const double flops = 2.0 * N * static_cast<double>(K);
  state.counters["GFLOPS"] = benchmark::Counter(
      1e-9 * flops, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::gemv_bias(*hp, x, W, b, y);
    Kokkos::fence();
  }
}

template <class Scalar>
void run_dense_forward(benchmark::State& state, bool force_native) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int M = static_cast<int>(state.range(0));
  const int N = static_cast<int>(state.range(1));
  const int K = static_cast<int>(state.range(2));

  KokkosDNN::DNNHandle<Exec> h_native;
  h_native.use_cublas = false;
  h_native.use_cudnn  = false;
  h_native.use_accelerate = false;
  KokkosDNN::DNNHandle<Exec> h_tpl;
  const KokkosDNN::DNNHandle<Exec>* hp = force_native ? &h_native : &h_tpl;

  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  Kokkos::View<Scalar*, Exec> b("b", N);
  fill_gemm_bias(X, W, b, Y);
  Kokkos::fence();

  // GFLOPS counts the GEMM (2*M*N*K); the M*N bias add is negligible.
  const double flops = 2.0 * M * N * static_cast<double>(K);
  state.counters["GFLOPS"] = benchmark::Counter(
      1e-9 * flops, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::dense_forward(*hp, X, W, b, Y);
    Kokkos::fence();
  }
}

using bhalf_t = Kokkos::Experimental::bhalf_t;

static void BM_GemmBias_Float_Native(benchmark::State& state) {
  run_gemm_bias<float>(state, true);
}
static void BM_GemmBias_Float_Dispatched(benchmark::State& state) {
  run_gemm_bias<float>(state, false);
}
static void BM_GemmBias_BHalf_Native(benchmark::State& state) {
  run_gemm_bias<bhalf_t>(state, true);
}
static void BM_GemmBias_BHalf_Dispatched(benchmark::State& state) {
  run_gemm_bias<bhalf_t>(state, false);
}
static void BM_GemvBias_Float_Native(benchmark::State& state) {
  run_gemv_bias<float>(state, true);
}
static void BM_GemvBias_Float_Dispatched(benchmark::State& state) {
  run_gemv_bias<float>(state, false);
}
static void BM_GemvBias_BHalf_Native(benchmark::State& state) {
  run_gemv_bias<bhalf_t>(state, true);
}
static void BM_GemvBias_BHalf_Dispatched(benchmark::State& state) {
  run_gemv_bias<bhalf_t>(state, false);
}
static void BM_DenseForward_Float_Native(benchmark::State& state) {
  run_dense_forward<float>(state, true);
}
static void BM_DenseForward_Float_Dispatched(benchmark::State& state) {
  run_dense_forward<float>(state, false);
}
static void BM_DenseForward_BHalf_Native(benchmark::State& state) {
  run_dense_forward<bhalf_t>(state, true);
}
static void BM_DenseForward_BHalf_Dispatched(benchmark::State& state) {
  run_dense_forward<bhalf_t>(state, false);
}

// Llama 3.1 8B-Instruct shapes, trimmed for CPU CI (no M=64, no 128k vocab):
// attn proj 4096x4096, FFN up/gate 14336x4096, FFN down 4096x14336.
#define KOKKOSDNN_GEMMBIAS_ARGS(BM)                                        \
  BM->Args({1, 4096, 4096})->Args({16, 4096, 4096})                        \
      ->Args({16, 14336, 4096})->Args({16, 4096, 14336})

KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_GemmBias_Float_Native)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_GemmBias_Float_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_GemmBias_BHalf_Native)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_GemmBias_BHalf_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_DenseForward_Float_Native)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_DenseForward_Float_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_DenseForward_BHalf_Native)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMMBIAS_ARGS(BENCHMARK(BM_DenseForward_BHalf_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));

// Decode shapes: y[N] = W[N,K] * x[K] + b[N].
#define KOKKOSDNN_GEMVBIAS_ARGS(BM) \
  BM->Args({4096, 4096})->Args({14336, 4096})->Args({4096, 14336})

KOKKOSDNN_GEMVBIAS_ARGS(BENCHMARK(BM_GemvBias_Float_Native)->Unit(benchmark::kMillisecond)->ArgNames({"N", "K"}));
KOKKOSDNN_GEMVBIAS_ARGS(BENCHMARK(BM_GemvBias_Float_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"N", "K"}));
KOKKOSDNN_GEMVBIAS_ARGS(BENCHMARK(BM_GemvBias_BHalf_Native)->Unit(benchmark::kMillisecond)->ArgNames({"N", "K"}));
KOKKOSDNN_GEMVBIAS_ARGS(BENCHMARK(BM_GemvBias_BHalf_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"N", "K"}));

}  // namespace
