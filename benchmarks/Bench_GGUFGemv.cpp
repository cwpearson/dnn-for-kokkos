// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// GEMV bandwidth microbenchmark against a real GGUF checkpoint: loads the
// packed matrices a Glimmer-30B decode step reads and times
// KokkosDNN::Gguf::quant_gemv on each shape.
//
// usage: kokkosdnn_bench_gguf_gemv <gguf> [iters] [rows] [tpr] [groups] [shape]
//   rows != 0 bypasses quant_gemv and drives the row-blocked kernel directly
//   with that layout (rows per thread group, threads per row, row groups per
//   team). rows == -99 lists tensors of block 0 and exits.
//
// How to read the output: shapes are timed round-robin, one launch each per
// round, so every shape sees the same GPU clock state. `min` is the best case
// (fresh clocks) and `median` is what a sustained back-to-back sequence of
// these kernels gets -- this GPU is power-capped, and a dense GEMV loop settles
// ~2x slower than its first ~100 ms while a pure-read kernel keeps its
// bandwidth, so both numbers matter depending on the duty cycle being modelled.
// Weight bytes are counted from the packed payload, so GB/s is DRAM read
// bandwidth (the activation row is small enough to stay in cache).
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_Types.hpp"

using Exec = Kokkos::DefaultExecutionSpace;
// The Glimmer decode path runs bf16 activations (KokkosDNN::DefaultScalar), and
// the activation side of the dot is what the packed kernels re-read, so the
// benchmark uses the same scalar type.
using Scalar = KokkosDNN::bhalf_t;
using namespace KokkosDNN;

namespace {

const char* kShapes[] = {
    "blk.0.attn_q.weight",      "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",      "blk.0.attn_gate.weight",
    "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",      "blk.0.ffn_down.weight",
    "output.weight",
};

double now_s() {
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
             .count() *
         1e-9;
}

// Read-only streaming ceiling: 16-byte loads over a buffer well past L2.
double read_ceiling_gbs(size_t bytes, int reps) {
  Kokkos::View<uint8_t*, Exec> buf("bwprobe", bytes);
  Kokkos::View<float, Exec> out("bwout");
  auto f = KOKKOS_LAMBDA(size_t i, float& s) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(buf.data()) + i * 4;
    s += (float)(p[0] + p[1] + p[2] + p[3]);
  };
  const size_t n = bytes / 16;
  Kokkos::parallel_reduce("bwprobe", n, f, out);
  Kokkos::fence();
  const double t0 = now_s();
  for (int r = 0; r < reps; ++r)
    Kokkos::parallel_reduce("bwprobe", n, f, out);
  Kokkos::fence();
  return (double)bytes * (double)reps / (now_s() - t0) / 1e9;
}

struct Case {
  std::string name;
  Gguf::QuantWeight<Exec> w;
  Kokkos::View<Scalar*, Exec> x, y;
  int nlayers = 1;  // multiplicity in one decode step
};

struct Stat {
  double mn = 0, med = 0, mx = 0, gbs = 0;
};

// Variant launch: drives the row-blocked kernel directly with an explicit
// (rows per thread group, threads per row, row groups per team) layout.
template <Gguf::GgmlType Dt, int ROWS>
void launch_rows(const Case& c, int tpr, int groups) {
  Gguf::Impl::quant_gemv_unit_blocked<Exec, Scalar, Dt, ROWS>(
      c.x, c.w, c.y, tpr, groups, "bench_variant");
}

void launch_variant(const Case& c, int rows, int tpr, int groups) {
  using G = Gguf::GgmlType;
#define KOKKOSDNN_BENCH_ROWS(DT)                                        \
  case DT:                                                              \
    if (rows >= 4)                                                      \
      launch_rows<DT, 4>(c, tpr, groups);                               \
    else if (rows == 2)                                                 \
      launch_rows<DT, 2>(c, tpr, groups);                               \
    else                                                                \
      launch_rows<DT, 1>(c, tpr, groups);                               \
    return
  switch (c.w.dtype) {
    KOKKOSDNN_BENCH_ROWS(G::Q4_K);
    KOKKOSDNN_BENCH_ROWS(G::Q5_K);
    KOKKOSDNN_BENCH_ROWS(G::Q6_K);
    KOKKOSDNN_BENCH_ROWS(G::Q8_0);
    default: return;
  }
#undef KOKKOSDNN_BENCH_ROWS
}

// One timed launch (rows > 0 drives the variant path).
void launch(const Case& c, int rows, int tpr, int groups) {
  if (rows > 0) {
    launch_variant(c, rows, tpr, groups);
    return;
  }
  Gguf::quant_gemv(Exec(), c.x, c.w, c.y);
}

// Checksum so a variant that changes the math cannot look fast.
double checksum(const Case& c) {
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c.y);
  double s = 0;
  for (size_t i = 0; i < hy.extent(0); i += 97)
    s += (double)static_cast<float>(hy(i));
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <gguf> [iters] [rows] [tpr] [grp] [shape]\n",
                 argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const int iters        = argc > 2 ? std::atoi(argv[2]) : 60;
  const int rows         = argc > 3 ? std::atoi(argv[3]) : 0;
  const int tpr          = argc > 4 ? std::atoi(argv[4]) : 32;
  const int groups       = argc > 5 ? std::atoi(argv[5]) : 4;
  const std::string only = argc > 6 ? argv[6] : "";

  Gguf::GgufFile f(path);
  if (rows == -99) {
    for (const auto& t : f.tensors()) {
      if (t.name.find("blk.") != std::string::npos &&
          t.name.find("blk.0.") == std::string::npos)
        continue;
      std::printf("%-34s %-5s %6zu x %6zu  %8.2f MB\n", t.name.c_str(),
                  Gguf::ggml_type_name(t.dtype).c_str(), t.rows(), t.cols(),
                  (double)t.nbytes / 1e6);
    }
    return 0;
  }

  int nlayers = 0;
  for (const auto& t : f.tensors())
    if (t.name.rfind("blk.", 0) == 0)
      nlayers = std::max(nlayers, std::atoi(t.name.c_str() + 4) + 1);

  std::vector<Case> cases;
  for (const char* name : kShapes) {
    if (!f.contains(name)) continue;
    if (!only.empty() && only != name) continue;
    Case c;
    c.name = name;
    c.w    = Gguf::load_quant_weight<Exec>(f, f.find(name));
    c.x    = Kokkos::View<Scalar*, Exec>("x", c.w.cols);
    c.y    = Kokkos::View<Scalar*, Exec>("y", c.w.rows);
    Kokkos::deep_copy(c.x, Scalar(0.01f));
    c.nlayers = std::string(name).rfind("blk.0.", 0) == 0 ? nlayers : 1;
    cases.push_back(std::move(c));
  }
  if (cases.empty()) {
    std::fprintf(stderr, "no benchmark cases selected\n");
    return 1;
  }

  // Reference output of the dispatch path for the variant comparison.
  std::vector<double> gold;
  for (const auto& c : cases) {
    launch(c, 0, 0, 0);
    Kokkos::fence();
    gold.push_back(checksum(c));
  }

  // Shapes are measured round-robin, one launch each per round, so every shape
  // sees the same GPU clock state: the min is the best case (fresh clocks) and
  // the median is what a sustained back-to-back sequence of these kernels gets.
  // Running one shape for many iterations in a row would instead bake the
  // clock drift into whichever shape happens to be measured last.
  std::vector<std::vector<double>> t(cases.size());
  for (size_t i = 0; i < cases.size(); ++i) t[i].reserve((size_t)iters);
  for (int r = 0; r < iters; ++r) {
    for (size_t i = 0; i < cases.size(); ++i) {
      Kokkos::fence();
      const double t0 = now_s();
      launch(cases[i], rows, tpr, groups);
      Kokkos::fence();
      t[i].push_back(now_s() - t0);
    }
  }
  for (size_t i = 0; i < cases.size(); ++i) {
    if (rows > 0) {
      const double got = checksum(cases[i]);
      if (std::fabs(got - gold[i]) > 1e-3 * (1.0 + std::fabs(gold[i])))
        std::printf("  WARNING %s: checksum %.6g vs %.6g\n",
                    cases[i].name.c_str(), got, gold[i]);
    }
    std::sort(t[i].begin(), t[i].end());
  }
  std::printf("%-26s %9s %9s %9s %9s %9s\n", "shape", "MB", "min ms",
              "med ms", "max ms", "GB/s(min)");
  double tot_bytes = 0, tot_ms = 0, tot_ms_med = 0;
  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const double mn = t[i].front() * 1e3;
    const double md = t[i][t[i].size() / 2] * 1e3;
    const double mx = t[i].back() * 1e3;
    std::printf("%-26s %9.2f %9.4f %9.4f %9.4f %9.1f\n", c.name.c_str(),
                (double)c.w.bytes() / 1e6, mn, md, mx,
                (double)c.w.bytes() / (mn * 1e-3) / 1e9);
    tot_bytes += (double)c.w.bytes() * (double)c.nlayers;
    tot_ms += mn * (double)c.nlayers;
    tot_ms_med += md * (double)c.nlayers;
  }
  std::printf("per-token weights (min):  %.2f GB in %.2f ms => %.1f GB/s\n",
              tot_bytes / 1e9, tot_ms, tot_bytes / (tot_ms * 1e-3) / 1e9);
  std::printf("per-token weights (median): %.2f GB in %.2f ms => %.1f GB/s\n",
              tot_bytes / 1e9, tot_ms_med,
              tot_bytes / (tot_ms_med * 1e-3) / 1e9);
  std::printf("read-only streaming ceiling: %.0f GB/s\n",
              read_ceiling_gbs((size_t)1 << 30, 4));
  return 0;
}
