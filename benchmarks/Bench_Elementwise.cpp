// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Memory-bound elementwise benchmarks (native Kokkos only, no TPL path):
// RMSNorm, row softmax, SwiGLU, RoPE at Llama 3.1 8B-Instruct dims (dim=4096,
// ffn_dim=14336, head_dim=128, n_heads=32/n_kv=8). Throughput is reported as
// GB/s over the logical bytes touched per iteration.
//
//   ./kokkosdnn_bench --benchmark_filter='RMSNorm.*|Softmax.*|SwiGLU.*|RoPE.*'
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include "KokkosDNN.hpp"

namespace {

using bhalf_t = Kokkos::Experimental::bhalf_t;

template <class Scalar>
void run_rmsnorm(benchmark::State& state) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T   = static_cast<int>(state.range(0));
  const int Dim = 4096;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, Dim),
      y("y", T, Dim);
  Kokkos::View<Scalar*, Exec> w("w", Dim);
  {
    auto hx = Kokkos::create_mirror_view(x);
    auto hw = Kokkos::create_mirror_view(w);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < Dim; ++d)
        hx(t, d) = static_cast<Scalar>(0.01f * ((t + d) % 17) - 0.08f);
    for (int d = 0; d < Dim; ++d) hw(d) = static_cast<Scalar>(1);
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(w, hw);
  }
  Kokkos::fence();
  const double bytes =
      (2.0 * T * Dim + Dim) * static_cast<double>(sizeof(Scalar));
  state.counters["GBps"] = benchmark::Counter(
      1e-9 * bytes, benchmark::Counter::kIsIterationInvariantRate);
  for (auto _ : state) {
    KokkosDNN::rmsnorm<Exec, Scalar>(x, w, 1e-5f, y);
    Kokkos::fence();
  }
}

template <class Scalar>
void run_softmax(benchmark::State& state) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int R = static_cast<int>(state.range(0));
  const int C = static_cast<int>(state.range(1));
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", R, C),
      y("y", R, C);
  {
    auto hx = Kokkos::create_mirror_view(x);
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c)
        hx(r, c) = static_cast<Scalar>(0.01f * ((r * 3 + c) % 19) - 0.09f);
    Kokkos::deep_copy(x, hx);
  }
  Kokkos::fence();
  const double bytes =
      2.0 * R * C * static_cast<double>(sizeof(Scalar));
  state.counters["GBps"] = benchmark::Counter(
      1e-9 * bytes, benchmark::Counter::kIsIterationInvariantRate);
  for (auto _ : state) {
    KokkosDNN::softmax_rows<Exec, Scalar>(x, y);
    Kokkos::fence();
  }
}

template <class Scalar>
void run_swiglu(benchmark::State& state) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = static_cast<int>(state.range(0));
  const int F = 14336;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> g("g", T, F), u("u", T, F),
      o("o", T, F);
  {
    auto hg = Kokkos::create_mirror_view(g);
    auto hu = Kokkos::create_mirror_view(u);
    for (int t = 0; t < T; ++t)
      for (int f = 0; f < F; ++f) {
        hg(t, f) = static_cast<Scalar>(0.01f * ((t + f) % 13) - 0.06f);
        hu(t, f) = static_cast<Scalar>(0.01f * ((t * 2 + f) % 11) - 0.05f);
      }
    Kokkos::deep_copy(g, hg);
    Kokkos::deep_copy(u, hu);
  }
  Kokkos::fence();
  const double bytes =
      3.0 * T * F * static_cast<double>(sizeof(Scalar));
  state.counters["GBps"] = benchmark::Counter(
      1e-9 * bytes, benchmark::Counter::kIsIterationInvariantRate);
  for (auto _ : state) {
    KokkosDNN::swiglu<Exec, Scalar>(g, u, o);
    Kokkos::fence();
  }
}

template <class Scalar>
void run_rope(benchmark::State& state) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T   = static_cast<int>(state.range(0));
  const int Hq  = 32, Hkv = 8, Dh = 128;
  Kokkos::View<Scalar***, Exec> q("q", T, Hq, Dh), k("k", T, Hkv, Dh);
  auto inv = KokkosDNN::rope_inv_freq<Exec>(Dh, 500000.0f);
  Kokkos::View<int*, Exec> pos("pos", T);
  {
    auto hq = Kokkos::create_mirror_view(q);
    auto hk = Kokkos::create_mirror_view(k);
    auto hp = Kokkos::create_mirror_view(pos);
    for (int t = 0; t < T; ++t) {
      hp(t) = t;
      for (int h = 0; h < Hq; ++h)
        for (int d = 0; d < Dh; ++d)
          hq(t, h, d) = static_cast<Scalar>(0.01f * ((t + h + d) % 7) - 0.03f);
      for (int h = 0; h < Hkv; ++h)
        for (int d = 0; d < Dh; ++d)
          hk(t, h, d) = static_cast<Scalar>(0.01f * ((t + h * 2 + d) % 9) - 0.04f);
    }
    Kokkos::deep_copy(q, hq);
    Kokkos::deep_copy(k, hk);
    Kokkos::deep_copy(pos, hp);
  }
  Kokkos::fence();
  const double bytes = 2.0 * T * (Hq + Hkv) * Dh *
                       static_cast<double>(sizeof(Scalar));
  state.counters["GBps"] = benchmark::Counter(
      1e-9 * bytes, benchmark::Counter::kIsIterationInvariantRate);
  for (auto _ : state) {
    KokkosDNN::rope_neoX_inplace<Exec, Scalar>(q, k, inv, Dh, pos);
    Kokkos::fence();
  }
}

static void BM_RMSNorm_Float(benchmark::State& s) { run_rmsnorm<float>(s); }
static void BM_RMSNorm_BHalf(benchmark::State& s) { run_rmsnorm<bhalf_t>(s); }
static void BM_Softmax_Float(benchmark::State& s) { run_softmax<float>(s); }
static void BM_Softmax_BHalf(benchmark::State& s) { run_softmax<bhalf_t>(s); }
static void BM_SwiGLU_Float(benchmark::State& s) { run_swiglu<float>(s); }
static void BM_SwiGLU_BHalf(benchmark::State& s) { run_swiglu<bhalf_t>(s); }
static void BM_RoPE_Float(benchmark::State& s) { run_rope<float>(s); }
static void BM_RoPE_BHalf(benchmark::State& s) { run_rope<bhalf_t>(s); }

BENCHMARK(BM_RMSNorm_Float)->Args({16})->Args({128})->Unit(benchmark::kMillisecond)->ArgNames({"T"});
BENCHMARK(BM_RMSNorm_BHalf)->Args({16})->Args({128})->Unit(benchmark::kMillisecond)->ArgNames({"T"});
// Rows = T*Hq attention-score rows; cols = cached keys.
BENCHMARK(BM_Softmax_Float)->Args({512, 128})->Args({2048, 512})->Unit(benchmark::kMillisecond)->ArgNames({"R", "C"});
BENCHMARK(BM_Softmax_BHalf)->Args({512, 128})->Args({2048, 512})->Unit(benchmark::kMillisecond)->ArgNames({"R", "C"});
BENCHMARK(BM_SwiGLU_Float)->Args({16})->Args({128})->Unit(benchmark::kMillisecond)->ArgNames({"T"});
BENCHMARK(BM_SwiGLU_BHalf)->Args({16})->Args({128})->Unit(benchmark::kMillisecond)->ArgNames({"T"});
BENCHMARK(BM_RoPE_Float)->Args({16})->Args({128})->Unit(benchmark::kMillisecond)->ArgNames({"T"});
BENCHMARK(BM_RoPE_BHalf)->Args({16})->Args({128})->Unit(benchmark::kMillisecond)->ArgNames({"T"});

}  // namespace
