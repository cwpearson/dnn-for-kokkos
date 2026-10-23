// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Bidirectional (non-causal) MHA benchmarks: Native vs Dispatched (cuDNN SDPA).
//
// Single config Hq=12, Hkv=12, D=64 for all token counts, matching DINOv3
// ViT-B/16 MHA dims (pure MHA: Hq == Hkv == 12, Dh == 64). Args are
// (T query tokens, K keys); K == T for ViT self-attention.
//   {65, 65}   small sanity shape
//   {201, 201} 224px: 1 CLS + 4 regs + 196 patches
//   {577, 577} 384px: 1 CLS + 4 regs + 576 patches
// FLOPs ≈ 4*T*Hq*K*D (scores + weighted sum).
//
// Native is always the Kokkos kernel. Dispatched uses the default handle and
// routes float/bhalf_t attention to cuDNN non-causal SDPA when built with
// KokkosDNN_ENABLE_CUDNN on a CUDA space; otherwise it falls back to native,
// so Native == Dispatched.
//
//   ./kokkosdnn_bench --benchmark_filter='AttentionBidir.*'
//   ./kokkosdnn_bench --benchmark_filter='AttentionBidir/Float.*' --benchmark_counters_tabular=true
#include <benchmark/benchmark.h>

#include <cmath>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include "KokkosDNN.hpp"

namespace {

constexpr int kHq  = 12;
constexpr int kHkv = 12;
constexpr int kD   = 64;

template <class Scalar>
void run_attention_bidir(benchmark::State& state, bool force_native) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int T = static_cast<int>(state.range(0));
  const int K = static_cast<int>(state.range(1));

  KokkosDNN::DNNHandle<Exec> h_native;
  h_native.use_cublas = false;
  h_native.use_cudnn  = false;
  KokkosDNN::DNNHandle<Exec> h_tpl;
  const KokkosDNN::DNNHandle<Exec>* hp = force_native ? &h_native : &h_tpl;

  Kokkos::View<Scalar***, Exec> q("q", T, kHq, kD), k("k", K, kHkv, kD),
      v("v", K, kHkv, kD), out("out", T, kHq, kD);
  {
    auto hq = Kokkos::create_mirror_view(q);
    auto hk = Kokkos::create_mirror_view(k);
    auto hv = Kokkos::create_mirror_view(v);
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < kHq; ++h)
        for (int d = 0; d < kD; ++d)
          hq(t, h, d) =
              static_cast<Scalar>(0.01f * ((t * 7 + h * 13 + d) % 23) - 0.1f);
    for (int n = 0; n < K; ++n)
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

  const float scale =
      1.0f / std::sqrt(static_cast<float>(kD));  // 1/sqrt(64)
  const double flops = 4.0 * T * kHq * static_cast<double>(K) *
                       static_cast<double>(kD);
  state.counters["GFLOPS"] = benchmark::Counter(
      1e-9 * flops, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::attention_mha_bidirectional(*hp, q, k, v, out, scale);
    Kokkos::fence();
  }
}

using bhalf_t = Kokkos::Experimental::bhalf_t;

static void BM_AttentionBidir_Float_Native(benchmark::State& state) {
  run_attention_bidir<float>(state, true);
}
static void BM_AttentionBidir_Float_Dispatched(benchmark::State& state) {
  run_attention_bidir<float>(state, false);
}
static void BM_AttentionBidir_BHalf_Native(benchmark::State& state) {
  run_attention_bidir<bhalf_t>(state, true);
}
static void BM_AttentionBidir_BHalf_Dispatched(benchmark::State& state) {
  run_attention_bidir<bhalf_t>(state, false);
}

// (T, K): ViT-B/16 self-attention token counts (K == T).
#define KOKKOSDNN_ATTN_BIDIR_ARGS(BM) \
  BM->Args({65, 65})->Args({201, 201})->Args({577, 577})

KOKKOSDNN_ATTN_BIDIR_ARGS(BENCHMARK(BM_AttentionBidir_Float_Native)->Unit(benchmark::kMillisecond)->ArgNames({"T", "K"}));
KOKKOSDNN_ATTN_BIDIR_ARGS(BENCHMARK(BM_AttentionBidir_Float_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"T", "K"}));
KOKKOSDNN_ATTN_BIDIR_ARGS(BENCHMARK(BM_AttentionBidir_BHalf_Native)->Unit(benchmark::kMillisecond)->ArgNames({"T", "K"}));
KOKKOSDNN_ATTN_BIDIR_ARGS(BENCHMARK(BM_AttentionBidir_BHalf_Dispatched)->Unit(benchmark::kMillisecond)->ArgNames({"T", "K"}));

}  // namespace
