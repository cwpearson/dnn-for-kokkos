// GEMM/GEMV benchmarks: Native vs Dispatched (cuBLAS) on Llama 3.3 8B shapes.
//
//   Y[M,N] = X[M,K] * W[N,K]^T  (W row-major [out_features, in_features])
//
// Two variants per float case:
//   Native     — handle with use_cublas=false, always the Kokkos kernel.
//   Dispatched — default handle, routes float GEMM to cuBLAS when built with
//                KokkosDNN_ENABLE_CUBLAS on a CUDA space; otherwise native.
// bhalf_t has no TPL path (falls back to native), so only Native is registered.
//
// Run e.g.:
//   ./kokkosdnn_bench --benchmark_filter='Gemm.*'
//   ./kokkosdnn_bench --benchmark_filter='Gemm/Float.*4096/4096' --benchmark_counters_tabular=true
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include "KokkosDNN.hpp"

namespace {

template <class Scalar>
void fill_gemm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>& Y) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto hx = Kokkos::create_mirror_view(X);
  auto hw = Kokkos::create_mirror_view(W);
  auto hy = Kokkos::create_mirror_view(Y);
  for (size_t i = 0; i < hx.extent(0); ++i)
    for (size_t k = 0; k < hx.extent(1); ++k)
      hx(i, k) = static_cast<Scalar>(0.001f * ((i * 31 + k * 17) % 97) - 0.05f);
  for (size_t j = 0; j < hw.extent(0); ++j)
    for (size_t k = 0; k < hw.extent(1); ++k)
      hw(j, k) = static_cast<Scalar>(0.001f * ((j * 13 + k * 7) % 89) - 0.04f);
  for (size_t i = 0; i < hy.extent(0); ++i)
    for (size_t j = 0; j < hy.extent(1); ++j) hy(i, j) = static_cast<Scalar>(0);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(W, hw);
  Kokkos::deep_copy(Y, hy);
  (void)Exec{};
}

template <class Scalar>
void run_gemm(benchmark::State& state, bool force_native) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int M = static_cast<int>(state.range(0));
  const int N = static_cast<int>(state.range(1));
  const int K = static_cast<int>(state.range(2));

  KokkosDNN::DNNHandle<Exec> h_native;
  h_native.use_cublas = false;
  h_native.use_cudnn  = false;
  KokkosDNN::DNNHandle<Exec> h_tpl;  // configured default (TPL when enabled)
  const KokkosDNN::DNNHandle<Exec>* hp = force_native ? &h_native : &h_tpl;

  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), W("W", N, K),
      Y("Y", M, N);
  fill_gemm(X, W, Y);
  Kokkos::fence();

  const double flops = 2.0 * M * N * static_cast<double>(K);
  state.counters["GFLOPS"] = benchmark::Counter(
      1e-9 * flops, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::gemm(*hp, static_cast<Scalar>(1), X, W, static_cast<Scalar>(0),
                    Y);
    Kokkos::fence();
  }
}

template <class Scalar>
void run_gemv(benchmark::State& state, bool force_native) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int N = static_cast<int>(state.range(0));
  const int K = static_cast<int>(state.range(1));

  KokkosDNN::DNNHandle<Exec> h_native;
  h_native.use_cublas = false;
  h_native.use_cudnn  = false;
  KokkosDNN::DNNHandle<Exec> h_tpl;
  const KokkosDNN::DNNHandle<Exec>* hp = force_native ? &h_native : &h_tpl;

  Kokkos::View<Scalar*, Exec> x("x", K), y("y", N);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> W("W", N, K);
  {
    auto hxv = Kokkos::create_mirror_view(x);
    auto hw  = Kokkos::create_mirror_view(W);
    for (int k = 0; k < K; ++k)
      hxv(k) = static_cast<Scalar>(0.001f * ((k * 17) % 97) - 0.05f);
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        hw(j, k) = static_cast<Scalar>(0.001f * ((j * 13 + k * 7) % 89) - 0.04f);
    Kokkos::deep_copy(x, hxv);
    Kokkos::deep_copy(W, hw);
  }
  Kokkos::fence();

  const double flops = 2.0 * N * static_cast<double>(K);
  state.counters["GFLOPS"] = benchmark::Counter(
      1e-9 * flops, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::gemv(*hp, x, W, y);
    Kokkos::fence();
  }
}

using bhalf_t = Kokkos::Experimental::bhalf_t;

static void BM_Gemm_Float_Native(benchmark::State& state) {
  run_gemm<float>(state, true);
}
static void BM_Gemm_Float_Dispatched(benchmark::State& state) {
  run_gemm<float>(state, false);
}
static void BM_Gemm_BHalf_Native(benchmark::State& state) {
  run_gemm<bhalf_t>(state, true);
}
static void BM_Gemv_Float_Native(benchmark::State& state) {
  run_gemv<float>(state, true);
}
static void BM_Gemv_Float_Dispatched(benchmark::State& state) {
  run_gemv<float>(state, false);
}
static void BM_Gemv_BHalf_Native(benchmark::State& state) {
  run_gemv<bhalf_t>(state, true);
}

// Llama 3.3 8B GEMM shapes (M tokens, N out_features, K in_features):
// attn proj 4096x4096, FFN up/gate 14336x4096, FFN down 4096x14336,
// lm_head 128256x4096 (decode M=1 only — full-vocab prefill is OOM-prone).
// M=64 prefill rows are GPU-oriented; on Serial they take seconds per iter,
// so pass --benchmark_filter to subset, or --benchmark_min_time=0.01.
#define KOKKOSDNN_GEMM_ARGS(BM)                                               \
  BM->Args({1, 4096, 4096})->Args({16, 4096, 4096})->Args({64, 4096, 4096})    \
      ->Args({16, 14336, 4096})->Args({16, 4096, 14336})->Args({1, 14336, 4096})

KOKKOSDNN_GEMM_ARGS(BENCHMARK(BM_Gemm_Float_Native)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMM_ARGS(BENCHMARK(BM_Gemm_Float_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));
KOKKOSDNN_GEMM_ARGS(BENCHMARK(BM_Gemm_BHalf_Native)->Unit(benchmark::kMillisecond)->ArgNames({"M", "N", "K"}));

BENCHMARK(BM_Gemm_Float_Native)
    ->Args({1, 128256, 4096})
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"M", "N", "K"});
BENCHMARK(BM_Gemm_Float_Dispatched)
    ->Args({1, 128256, 4096})
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"M", "N", "K"});
BENCHMARK(BM_Gemm_BHalf_Native)
    ->Args({1, 128256, 4096})
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"M", "N", "K"});

// Decode GEMV shapes: y[N] = x[K] * W[N,K]^T.
#define KOKKOSDNN_GEMV_ARGS(BM) \
  BM->Args({4096, 4096})->Args({14336, 4096})->Args({4096, 14336})->Args({128256, 4096})

KOKKOSDNN_GEMV_ARGS(BENCHMARK(BM_Gemv_Float_Native)->Unit(benchmark::kMillisecond)->ArgNames({"N", "K"}));
KOKKOSDNN_GEMV_ARGS(BENCHMARK(BM_Gemv_Float_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"N", "K"}));
KOKKOSDNN_GEMV_ARGS(BENCHMARK(BM_Gemv_BHalf_Native)->Unit(benchmark::kMillisecond)->ArgNames({"N", "K"}));

}  // namespace
