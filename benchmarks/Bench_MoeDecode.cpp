// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Sparse-MoE decode (batch-1) microbenchmark: the two fused expert GEMV
// stages of Gguf::quant_moe_decode, timed separately.
//
// Decode MoE is weight-bandwidth bound: one token reads top_k * 2 * MI * K
// of packed gate/up weights and top_k * K * MI of packed down weights per
// layer, and nothing else (the activation is a few KB). The useful ceiling is
// therefore GB/s per stage:
//
//   gate_up: top_k * 2 * MI * row_bytes(K)     read per call
//   down:    top_k * K  * row_bytes(MI)        read per call
//
// Shape: LFM2.5-24B-A2B (K=2048, MI=1536, top_k=4, 64 experts, bf16
// activations) with the file's real Q4_K_M recipe -- gate/up Q4_K, down Q6_K
// -- which is the shape where nsys showed the expert kernels as the two
// largest per-token costs. The expert payloads are synthesized (random packed
// bytes with plausible fp16 scales): kernel cost depends on the packed layout
// and dtype, not on the weight values.
//
// The routed slots rotate over 8 disjoint expert sets, so every call streams
// a different ~25 MB of weights from DRAM instead of re-reading a handful of
// experts that fit in L2 (which would flatter the numbers by ~2x).
//
// moe_read_probe is a pure 16-byte-read kernel over the same byte count: on
// this RTX 3090 it is what "100% of achievable read bandwidth" looks like for
// one MoE layer's payload, and the MoE stages are instruction-bound well below
// it (see the numbers in the commit message). It also fixes the measurement
// regime: the card is power-capped, so run with --benchmark_min_time=1s to get
// sustained (throttled) numbers that match what a decode loop actually sees.
//
//   ./kokkosdnn_bench --benchmark_filter='MoeDecode'
//       --benchmark_counters_tabular=true --benchmark_min_time=1s
#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cstring>
#include <utility>
#include <vector>

#include "KokkosDNN.hpp"

namespace {

using Exec    = Kokkos::DefaultExecutionSpace;
using bhalf_t = Kokkos::Experimental::bhalf_t;
using KokkosDNN::Gguf::GgmlType;

// Deterministic packed payload: plausible fp16 super-block scale/min and
// random sub-block scales / quants.
Kokkos::View<uint8_t*, Exec> make_packed(GgmlType dtype, size_t rows,
                                         size_t cols, uint32_t seed) {
  const size_t rb        = KokkosDNN::Gguf::ggml_row_bytes(dtype, cols);
  const size_t bytes     = rows * rb;
  const bool q4k         = dtype == GgmlType::Q4_K;
  const size_t blk_bytes = q4k ? 144u : 210u;
  Kokkos::View<uint8_t*, Exec> d("packed", bytes);
  auto h = Kokkos::create_mirror_view(d);
  uint32_t s = seed;
  auto next  = [&s]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  for (size_t blk = 0; blk < bytes / blk_bytes; ++blk) {
    uint8_t* p = h.data() + blk * blk_bytes;
    for (size_t i = 0; i < blk_bytes; ++i) p[i] = (uint8_t)(next() >> 11);
    // Small positive fp16 magnitudes so the dequant arithmetic sees normal
    // values instead of inf/denormal noise.
    const uint16_t scale = 0x1C00u;
    if (q4k) {
      std::memcpy(p + 0, &scale, 2);  // d
      std::memcpy(p + 2, &scale, 2);  // dmin
    } else {
      std::memcpy(p + 208, &scale, 2);  // d
    }
  }
  Kokkos::deep_copy(d, h);
  return d;
}

// Rotation sets for the routed slots: one disjoint group of experts per step.
constexpr size_t kRot = 8;

struct MoeBench {
  size_t K = 2048, MI = 1536, S = 4, E = 64;
  std::vector<KokkosDNN::Gguf::QuantWeight<Exec>> gate, up, down;
  KokkosDNN::Gguf::QuantWeightTable<Exec> tg, tu, td;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx_all;  // [kRot, S]
  std::vector<Kokkos::View<int**, Kokkos::LayoutRight, Exec>> idx;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> wts;
  Kokkos::View<bhalf_t**, Kokkos::LayoutRight, Exec> X, act, Y;

  void init() {
    using namespace KokkosDNN::Gguf;
    for (size_t e = 0; e < E; ++e) {
      QuantWeight<Exec> g, u, dn;
      // Real Q4_K_M recipe: gate/up Q4_K, down Q6_K.
      g.dtype = u.dtype = GgmlType::Q4_K;
      dn.dtype          = GgmlType::Q6_K;
      g.rows = u.rows = MI;
      g.cols = u.cols = K;
      dn.rows         = K;
      dn.cols         = MI;
      g.data          = make_packed(GgmlType::Q4_K, MI, K, 0x1234u + e);
      u.data          = make_packed(GgmlType::Q4_K, MI, K, 0x5678u + e);
      dn.data         = make_packed(GgmlType::Q6_K, K, MI, 0x9abcu + e);
      gate.push_back(g);
      up.push_back(u);
      down.push_back(dn);
    }
    tg = make_quant_weight_table(gate, "bench::gate");
    tu = make_quant_weight_table(up, "bench::up");
    td = make_quant_weight_table(down, "bench::down");

    idx_all = Kokkos::View<int**, Kokkos::LayoutRight, Exec>("idx", kRot, S);
    auto hi = Kokkos::create_mirror_view(idx_all);
    for (size_t r = 0; r < kRot; ++r)
      for (size_t s = 0; s < S; ++s) hi(r, s) = (int)((r * 8 + s) % E);
    Kokkos::deep_copy(idx_all, hi);
    for (size_t r = 0; r < kRot; ++r)
      idx.push_back(
          Kokkos::subview(idx_all, std::make_pair(r, r + 1), Kokkos::ALL));

    wts = Kokkos::View<float**, Kokkos::LayoutRight, Exec>("wts", 1, S);
    auto hw = Kokkos::create_mirror_view(wts);
    for (size_t s = 0; s < S; ++s) hw(0, s) = 0.25f;
    Kokkos::deep_copy(wts, hw);

    X   = Kokkos::View<bhalf_t**, Kokkos::LayoutRight, Exec>("X", 1, K);
    act = Kokkos::View<bhalf_t**, Kokkos::LayoutRight, Exec>("act", S, MI);
    Y   = Kokkos::View<bhalf_t**, Kokkos::LayoutRight, Exec>("Y", 1, K);
    auto hx = Kokkos::create_mirror_view(X);
    for (size_t k = 0; k < K; ++k) hx(0, k) = bhalf_t(0.01f * (int)(k % 29));
    Kokkos::deep_copy(X, hx);
    Kokkos::fence();
  }

  double gate_up_bytes() const {
    return (double)S * 2.0 * (double)MI *
           (double)KokkosDNN::Gguf::ggml_row_bytes(GgmlType::Q4_K, K);
  }
  double down_bytes() const {
    return (double)S * (double)K *
           (double)KokkosDNN::Gguf::ggml_row_bytes(GgmlType::Q6_K, MI);
  }
};

MoeBench& bench_data() {
  static MoeBench b;
  static bool init = false;
  if (!init) {
    b.init();
    init = true;
  }
  return b;
}

void BM_MoeDecode_GateUp(benchmark::State& state) {
  MoeBench& b = bench_data();
  KokkosDNN::Gguf::Impl::quant_moe_decode_gate_up<Exec, bhalf_t>(
      b.X, b.tg, b.tu, b.idx[0], b.wts, b.act, "bench");
  Kokkos::fence();
  state.counters["GB/s"] = benchmark::Counter(
      b.gate_up_bytes(), benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  size_t i = 0;
  for (auto _ : state) {
    KokkosDNN::Gguf::Impl::quant_moe_decode_gate_up<Exec, bhalf_t>(
        b.X, b.tg, b.tu, b.idx[i++ % kRot], b.wts, b.act, "bench");
    Kokkos::fence();
  }
}

void BM_MoeDecode_Down(benchmark::State& state) {
  MoeBench& b = bench_data();
  KokkosDNN::Gguf::Impl::quant_moe_decode_down<Exec, bhalf_t>(
      b.act, b.td, b.idx[0], b.wts, b.Y, "bench");
  Kokkos::fence();
  state.counters["GB/s"] = benchmark::Counter(
      b.down_bytes(), benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  size_t i = 0;
  for (auto _ : state) {
    KokkosDNN::Gguf::Impl::quant_moe_decode_down<Exec, bhalf_t>(
        b.act, b.td, b.idx[i++ % kRot], b.wts, b.Y, "bench");
    Kokkos::fence();
  }
}

void BM_MoeDecode_Fused(benchmark::State& state) {
  MoeBench& b = bench_data();
  KokkosDNN::Gguf::Impl::quant_moe_decode_gate_up<Exec, bhalf_t>(
      b.X, b.tg, b.tu, b.idx[0], b.wts, b.act, "bench");
  KokkosDNN::Gguf::Impl::quant_moe_decode_down<Exec, bhalf_t>(
      b.act, b.td, b.idx[0], b.wts, b.Y, "bench");
  Kokkos::fence();
  state.counters["GB/s"] = benchmark::Counter(
      b.gate_up_bytes() + b.down_bytes(),
      benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  size_t i = 0;
  for (auto _ : state) {
    auto id = b.idx[i++ % kRot];
    KokkosDNN::Gguf::Impl::quant_moe_decode_gate_up<Exec, bhalf_t>(
        b.X, b.tg, b.tu, id, b.wts, b.act, "bench");
    KokkosDNN::Gguf::Impl::quant_moe_decode_down<Exec, bhalf_t>(
        b.act, b.td, id, b.wts, b.Y, "bench");
    Kokkos::fence();
  }
}

// Pure 16-byte-read kernel over one gate_up payload: the read-bandwidth
// ceiling for this access size on this GPU, i.e. what a perfect MoE kernel
// could reach.
struct ReadProbe {
  const uint4* p;
  size_t n;  // 16-byte units
  Kokkos::View<int*, Exec> out;
  KOKKOS_FUNCTION void operator()(const size_t i) const {
    const size_t base = i * 4;
    if (base + 3 >= n) return;
    const uint4 a = p[base], b = p[base + 1], c = p[base + 2], d = p[base + 3];
    const uint32_t s = a.x ^ b.x ^ c.x ^ d.x ^ a.y ^ b.y ^ c.y ^ d.y ^ a.z ^
                       b.z ^ c.z ^ d.z ^ a.w ^ b.w ^ c.w ^ d.w;
    if (s == 0xdeadbeefu) out(0) = 1;
  }
};

void BM_MoeDecode_ReadProbe(benchmark::State& state) {
  MoeBench& b = bench_data();
  const size_t bytes = (size_t)(b.gate_up_bytes());
  Kokkos::View<uint8_t*, Exec> buf("probe", bytes);
  Kokkos::deep_copy(buf, uint8_t(3));
  Kokkos::View<int*, Exec> out("out", 1);
  const size_t n16 = bytes / 16;
  ReadProbe f{reinterpret_cast<const uint4*>(buf.data()), n16, out};
  Kokkos::parallel_for("probe_warm", (n16 + 3) / 4, f);
  Kokkos::fence();
  state.counters["GB/s"] = benchmark::Counter(
      (double)bytes, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  for (auto _ : state) {
    Kokkos::parallel_for("probe", (n16 + 3) / 4, f);
    Kokkos::fence();
  }
}

BENCHMARK(BM_MoeDecode_ReadProbe);
BENCHMARK(BM_MoeDecode_GateUp)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MoeDecode_Down)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MoeDecode_Fused)->Unit(benchmark::kMicrosecond);

}  // namespace
