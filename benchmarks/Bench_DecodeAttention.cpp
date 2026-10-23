// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Single-token (T == 1) decode attention microbenchmark.
//
// Decode attention is a memory-bound GEMV per query head: it streams
// 2 * Ktot * D * sizeof(Scalar) bytes per head per layer, once per decoded
// token, so at 10-15k context it dominates batch-1 decode. This benchmark
// isolates that kernel (attention_gqa_single with T == 1) across the KV
// lengths a long-context decode actually walks through.
//
// Reported counters:
//   GB/s        bytes as if every query head streamed K/V itself, i.e.
//               2 * Hq * Ktot * D * sizeof(Scalar) -- the traffic the pre
//               split-K kernel actually issued, so numbers stay comparable
//               with earlier measurements.
//   GB/s_kv     the bytes the split-K kernel really reads from DRAM,
//               2 * Hkv * Ktot * D * sizeof(Scalar): one block covers every
//               query head of a GQA group, so a KV head is streamed once.
//   GB/s        == GB/s_kv for MQA/GQA shapes with Hq == Hkv (no sharing).
//
// Shapes:
//   LFM25-BHalf/Float : Hq=32, Hkv=8, D=64 (LFM2.5-24B-A2B attention layers)
//   MHA-BHalf         : Hq=8,  Hkv=8, D=64 (no GQA sharing)
//   MQA-BHalf         : Hq=8,  Hkv=1, D=64 (one KV head, group of 8)
//
//   ./kokkosdnn_bench --benchmark_filter='DecodeAttention'
//   ./kokkosdnn_bench --benchmark_filter='DecodeAttention_LFM25_BHalf'
//       --benchmark_counters_tabular=true
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>

#include "KokkosDNN.hpp"

namespace {

using bhalf_t = Kokkos::Experimental::bhalf_t;

template <class Scalar>
void run_decode_attention(benchmark::State& state, int hq, int hkv, int d) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const size_t Ktot = static_cast<size_t>(state.range(0));
  // Single-token decode at the end of a full cache: the query attends every
  // cached key (q_off = Ktot - 1 with T == 1).
  const size_t q_off = Ktot - 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  KokkosDNN::DNNHandle<Exec> h;
  // Native kernel only: no TPL dispatch is reachable at T == 1 by design
  // (a cuDNN SDPA graph bakes the KV length in, so every decode token would
  // rebuild a plan), but zero the flags so the measurement cannot drift if
  // that ever changes.
  h.use_cublas = false;
  h.use_cudnn = false;

  Kokkos::View<Scalar***, Exec> q("q", 1, hq, d), k("k", Ktot, hkv, d),
      v("v", Ktot, hkv, d), out("out", 1, hq, d);
  {
    auto hq_m = Kokkos::create_mirror_view(q);
    auto hk_m = Kokkos::create_mirror_view(k);
    auto hv_m = Kokkos::create_mirror_view(v);
    for (int hh = 0; hh < hq; ++hh)
      for (int dd = 0; dd < d; ++dd)
        hq_m(0, hh, dd) =
            static_cast<Scalar>(0.01f * ((hh * 13 + dd) % 23) - 0.1f);
    for (size_t n = 0; n < Ktot; ++n)
      for (int hh = 0; hh < hkv; ++hh)
        for (int dd = 0; dd < d; ++dd) {
          hk_m(n, hh, dd) =
              static_cast<Scalar>(0.01f * ((n * 5 + hh * 11 + dd) % 29) - 0.1f);
          hv_m(n, hh, dd) =
              static_cast<Scalar>(0.01f * ((n * 3 + hh * 17 + dd) % 31) - 0.1f);
        }
    Kokkos::deep_copy(q, hq_m);
    Kokkos::deep_copy(k, hk_m);
    Kokkos::deep_copy(v, hv_m);
  }
  Kokkos::fence();

  // Warm-up also allocates the cached partial workspace, so the timed loop
  // measures the two kernel launches only.
  KokkosDNN::attention_gqa_single<Exec, Scalar>(h, q, k, v, q_off, out, scale);
  Kokkos::fence();

  const double bytes_row = 2.0 * hq * static_cast<double>(Ktot) * d *
                           static_cast<double>(sizeof(Scalar));
  const double bytes_kv = 2.0 * hkv * static_cast<double>(Ktot) * d *
                          static_cast<double>(sizeof(Scalar));
  state.counters["GB/s"] = benchmark::Counter(
      bytes_row, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["GB/s_kv"] = benchmark::Counter(
      bytes_kv, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["ms/call"] =
      benchmark::Counter(1e-3, benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    KokkosDNN::attention_gqa_single<Exec, Scalar>(h, q, k, v, q_off, out,
                                                 scale);
    Kokkos::fence();
  }
}

void BM_DecodeAttention_LFM25_BHalf(benchmark::State& state) {
  run_decode_attention<bhalf_t>(state, 32, 8, 64);
}

void BM_DecodeAttention_LFM25_Float(benchmark::State& state) {
  run_decode_attention<float>(state, 32, 8, 64);
}

void BM_DecodeAttention_MHA_BHalf(benchmark::State& state) {
  run_decode_attention<bhalf_t>(state, 8, 8, 64);
}

void BM_DecodeAttention_MQA_BHalf(benchmark::State& state) {
  run_decode_attention<bhalf_t>(state, 8, 1, 64);
}

#define KOKKOSDNN_DECODE_ARGS(BM) \
  BM->Args({1024})->Args({4096})->Args({8192})->Args({16384})

KOKKOSDNN_DECODE_ARGS(
    BENCHMARK(BM_DecodeAttention_LFM25_BHalf)
        ->Unit(benchmark::kMillisecond)
        ->ArgNames({"Ktot"}));
KOKKOSDNN_DECODE_ARGS(
    BENCHMARK(BM_DecodeAttention_LFM25_Float)
        ->Unit(benchmark::kMillisecond)
        ->ArgNames({"Ktot"}));
KOKKOSDNN_DECODE_ARGS(
    BENCHMARK(BM_DecodeAttention_MHA_BHalf)
        ->Unit(benchmark::kMillisecond)
        ->ArgNames({"Ktot"}));
KOKKOSDNN_DECODE_ARGS(
    BENCHMARK(BM_DecodeAttention_MQA_BHalf)
        ->Unit(benchmark::kMillisecond)
        ->ArgNames({"Ktot"}));

}  // namespace
