// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Batch-1 decode sampling: device pipeline (KokkosDNN_Sampler.hpp) against
// the host reference it replaced.
//
// The host reference is what a sampled decode step used to cost: mirror the
// whole logits row (V * 4 B), convert it to float element by element, then run
// penalties / temperature / top-k / top-p / min-p / categorical on the host.
// The device pipeline keeps all of that on device and returns one int.
//
//   ./kokkosdnn_bench --benchmark_filter='Sampler.*'
//
// V = 128000 matches LFM2.5-24B-A2B (and Llama-3-class) vocabularies.
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <vector>

#include "KokkosDNN_Sampler.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using View = Kokkos::View<float*, Exec>;

// Deterministic logits with a clear argmax and a realistic spread.
std::vector<float> make_logits(std::size_t vocab, std::uint64_t seed) {
  std::vector<float> v(vocab);
  std::uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
  for (std::size_t i = 0; i < vocab; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    const float unit = static_cast<float>((s >> 40) & 0xFFFFFF) / 16777216.0f;
    v[i] = 8.0f * unit - 4.0f;
  }
  v[vocab / 3] = 9.0f;
  return v;
}

View device_logits(const std::vector<float>& v) {
  View d("logits", v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

struct Case {
  const char* name;
  float temperature;
  int top_k;
  float top_p;
  float repetition;
};

const Case kCases[] = {
    {"temperature_only", 0.8f, 0, 1.0f, 1.0f},
    {"top_k_40", 0.8f, 40, 1.0f, 1.0f},
    {"top_p_0_95", 0.8f, 0, 0.95f, 1.0f},
    {"top_k_40_top_p_0_95", 0.8f, 40, 0.95f, 1.0f},
    {"penalties", 0.8f, 0, 0.95f, 1.3f},
};

KokkosDNN::SampleOptions options(const Case& c) {
  KokkosDNN::SampleOptions o;
  o.temperature = c.temperature;
  o.top_k = c.top_k;
  o.top_p = c.top_p;
  o.repetition_penalty = c.repetition;
  return o;
}

// Device pipeline: one uniform draw, all O(V) work on device, 4-byte D2H.
void run_device(benchmark::State& state, const Case& c) {
  const std::size_t V = static_cast<std::size_t>(state.range(0));
  const View d = device_logits(make_logits(V, 12345));
  std::vector<int> prior(64, 7);  // 64-token history for the penalties
  const KokkosDNN::SampleOptions opt = options(c);
  KokkosDNN::Impl::DeviceSampler<Exec> sampler(V);
  KokkosDNN::HostRNG rng(42);
  for (int i = 0; i < 16; ++i)  // warm up
    benchmark::DoNotOptimize(sampler.sample(d, opt, prior, rng).id);
  for (auto _ : state) {
    int id = sampler.sample(d, opt, prior, rng).id;
    benchmark::DoNotOptimize(id);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["V"] = static_cast<double>(V);
}

// Host reference: full-row D2H + float conversion + host pipeline, i.e. the
// per-token cost of the path the device pipeline replaced.
void run_host(benchmark::State& state, const Case& c) {
  const std::size_t V = static_cast<std::size_t>(state.range(0));
  const View d = device_logits(make_logits(V, 12345));
  std::vector<int> prior(64, 7);
  const KokkosDNN::SampleOptions opt = options(c);
  auto mirror = Kokkos::create_mirror_view(d);
  std::vector<float> f(V);
  KokkosDNN::HostRNG rng(42);
  for (auto _ : state) {
    Kokkos::deep_copy(mirror, d);
    for (std::size_t i = 0; i < V; ++i) f[i] = static_cast<float>(mirror(i));
    int id =
        KokkosDNN::sample_host_logits_result(f.data(), V, opt, prior, rng).id;
    benchmark::DoNotOptimize(id);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["V"] = static_cast<double>(V);
}

void run_greedy_device(benchmark::State& state) {
  const std::size_t V = static_cast<std::size_t>(state.range(0));
  const View d = device_logits(make_logits(V, 12345));
  for (auto _ : state) {
    int id = KokkosDNN::device_argmax<Exec, float>(d);
    benchmark::DoNotOptimize(id);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["V"] = static_cast<double>(V);
}

void bm_device(benchmark::State& state, const Case& c) { run_device(state, c); }
void bm_host(benchmark::State& state, const Case& c) { run_host(state, c); }

}  // namespace

// One pair per option set: device pipeline vs the host reference it replaced.
// V = 128000 (LFM2.5 / Llama-3 vocab), plus one small-vocabulary case.
#define KOKKOSDNN_IMPL_BENCH_PAIR(NAME, CASE, VOCAB)              \
  BENCHMARK_CAPTURE(bm_device, NAME, CASE)                        \
      ->Name("Sampler/Device/" #NAME)                             \
      ->Arg(VOCAB)                                                \
      ->Unit(benchmark::kMillisecond);                            \
  BENCHMARK_CAPTURE(bm_host, NAME, CASE)                          \
      ->Name("Sampler/Host/" #NAME)                               \
      ->Arg(VOCAB)                                                \
      ->Unit(benchmark::kMillisecond);

KOKKOSDNN_IMPL_BENCH_PAIR(temperature_only, kCases[0], 128000)
KOKKOSDNN_IMPL_BENCH_PAIR(top_k_40, kCases[1], 128000)
KOKKOSDNN_IMPL_BENCH_PAIR(top_p_0_95, kCases[2], 128000)
KOKKOSDNN_IMPL_BENCH_PAIR(top_k_40_top_p_0_95, kCases[3], 128000)
KOKKOSDNN_IMPL_BENCH_PAIR(penalties, kCases[4], 128000)
KOKKOSDNN_IMPL_BENCH_PAIR(top_p_0_95_small_vocab, kCases[2], 256)

BENCHMARK(run_greedy_device)
    ->Name("Sampler/Device/greedy_argmax")
    ->Arg(128000)
    ->Unit(benchmark::kMillisecond);
