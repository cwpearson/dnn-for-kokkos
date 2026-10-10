// Grouped-query attention benchmarks: Native vs Dispatched (cuDNN SDPA).
//
// Shapes mirror Llama 3.1 8B-Instruct: Hq=32 query heads, Hkv=8 KV heads, D=128.
// Args are (T query tokens, Ktot cached keys). q_off = Ktot - T (causal fill
// at the end of the cache). FLOPs ≈ 4*T*Hq*Klen*D (scores + weighted sum).
//
// Native is always the Kokkos kernel. Dispatched uses the default handle and
// routes float attention to cuDNN SDPA when built with
// KokkosDNN_ENABLE_CUDNN on a CUDA space; otherwise (including all bhalf_t
// and non-CUDA builds) it falls back to native, so Native == Dispatched.
// bhalf_t therefore registers Native only.
//
//   ./kokkosdnn_bench --benchmark_filter='Attention.*'
//   ./kokkosdnn_bench --benchmark_filter='Attention/Float.*' --benchmark_counters_tabular=true
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include "KokkosDNN.hpp"

namespace {

constexpr int kHq  = 32;
constexpr int kHkv = 8;
constexpr int kD   = 128;

template <class Scalar>
void run_attention(benchmark::State& state, bool force_native) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T    = static_cast<int>(state.range(0));
  const int Ktot = static_cast<int>(state.range(1));
  const size_t q_off = static_cast<size_t>(Ktot - T);

  KokkosDNN::DNNHandle<Exec> h_native;
  h_native.use_cublas = false;
  h_native.use_cudnn  = false;
  KokkosDNN::DNNHandle<Exec> h_tpl;
  const KokkosDNN::DNNHandle<Exec>* hp = force_native ? &h_native : &h_tpl;

  Kokkos::View<Scalar***, Exec> q("q", T, kHq, kD), k("k", Ktot, kHkv, kD),
      v("v", Ktot, kHkv, kD), out("out", T, kHq, kD);
  {
    auto hq = Kokkos::create_mirror_view(q);
    auto hk = Kokkos::create_mirror_view(k);
    auto hv = Kokkos::create_mirror_view(v);
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < kHq; ++h)
        for (int d = 0; d < kD; ++d)
          hq(t, h, d) =
              static_cast<Scalar>(0.01f * ((t * 7 + h * 13 + d) % 23) - 0.1f);
    for (int n = 0; n < Ktot; ++n)
      for (int h = 0; h < kHkv; ++h)
        for (int d = 0; d < kD; ++d) {
          hk(n, h, d) =
              static_cast<Scalar>(0.01f * ((n * 5 + h * 11 + d) % 29) - 0.1f);
          hv(n, h, d) =
              static_cast<Scalar>(0.01f * ((n * 3 + h * 17 + d) % 31) - 0.1f);
        }
    Kokkos::deep_copy(q, hq);
    Kokkos::deep_copy(k, hk);
    Kokkos::deep_copy(v, hv);
  }
  Kokkos::fence();

  const float scale = 1.0f / 11.3137085f;  // 1/sqrt(128)
  // Visible keys for the last query ≈ Ktot; use it for the FLOP estimate.
  const double flops =
      4.0 * T * kHq * static_cast<double>(Ktot) * static_cast<double>(kD);
  state.counters["GFLOPS"] = benchmark::Counter(
      1e-9 * flops, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::attention_gqa_single(*hp, q, k, v, q_off, out, scale);
    Kokkos::fence();
  }
}

using bhalf_t = Kokkos::Experimental::bhalf_t;

static void BM_Attention_Float_Native(benchmark::State& state) {
  run_attention<float>(state, true);
}
static void BM_Attention_Float_Dispatched(benchmark::State& state) {
  run_attention<float>(state, false);
}
static void BM_Attention_BHalf_Native(benchmark::State& state) {
  run_attention<bhalf_t>(state, true);
}

// (T, Ktot): decode T=1 plus short prefills. T=128/Ktot=512 is
// GPU-oriented on the naive kernel (4*T*Hq*Ktot*D FLOPs with one thread per
// (t, h)); the kernel itself holds no score tensor (streaming softmax).
#define KOKKOSDNN_ATTN_ARGS(BM) \
  BM->Args({1, 128})->Args({1, 512})->Args({16, 128})->Args({64, 256})->Args({128, 512})

KOKKOSDNN_ATTN_ARGS(BENCHMARK(BM_Attention_Float_Native)->Unit(benchmark::kMillisecond)->ArgNames({"T", "Ktot"}));
KOKKOSDNN_ATTN_ARGS(BENCHMARK(BM_Attention_Float_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"T", "Ktot"}));
KOKKOSDNN_ATTN_ARGS(BENCHMARK(BM_Attention_BHalf_Native)->Unit(benchmark::kMillisecond)->ArgNames({"T", "Ktot"}));

}  // namespace
