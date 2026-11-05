// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Micro-spike: Kokkos 5.2.1 graphs for decode-step launch-overhead reduction.
//
// Workload mimics one decode step's op mix at small scale:
//   elementwise (embed/norm-ish) + parallel_reduce (norm denominator) +
//   TeamPolicy GEMV (quant-gemv proxy) + elementwise (swiglu-ish),
// chained NCHAIN times (~200 kernel nodes, decode is ~900-1100 launches).
//
// Compares: (a) eager loop, (b) explicit Kokkos graph (then_parallel_for
// chain), (c) cuda_capture child-graph wrapping the eager region.
// Also demonstrates the cpos-staleness problem (host scalar baked into a
// captured functor) and the device-counter fix.
#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
using Exec = Kokkos::Cuda;
#else
using Exec = Kokkos::Serial;
#endif

using ViewF = Kokkos::View<float*, Exec>;

namespace {

// One "layer": gather-scale (parallel_for) + sumsq (parallel_reduce into a
// device scalar view) + normalize (parallel_for) + GEMV proxy (TeamPolicy)
// + activation (parallel_for). All ops take an explicit exec so the same
// body runs eagerly and inside cuda_capture.
struct LayerBufs {
  ViewF x, tmp, mv_out;
  Kokkos::View<float, Exec> sumsq;
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> w;
};

KOKKOS_FUNCTION float swiglu_fn(float g, float u) {
  return (g / (1.0f + expf(-g))) * u;
}

void run_layer(const Exec& ec, const LayerBufs& b, int n, int rows, int cols,
               float cpos_dev_sentinel_ignored = 0.0f) {
  (void)cpos_dev_sentinel_ignored;
  auto x = b.x;
  auto tmp = b.tmp;
  auto sumsq = b.sumsq;
  auto mv = b.mv_out;
  auto w = b.w;
  Kokkos::parallel_for(
      "spike:gather", Kokkos::RangePolicy<Exec>(ec, 0, n),
      KOKKOS_LAMBDA(int i) { tmp(i) = x(i) * 1.001f + 0.5f; });
  Kokkos::parallel_reduce(
      "spike:sumsq", Kokkos::RangePolicy<Exec>(ec, 0, n),
      KOKKOS_LAMBDA(int i, float& s) { s += tmp(i) * tmp(i); },
      Kokkos::Sum<float, Exec>(sumsq));
  Kokkos::parallel_for(
      "spike:norm", Kokkos::RangePolicy<Exec>(ec, 0, n),
      KOKKOS_LAMBDA(int i) {
        float d = sqrtf(sumsq() / (float)n + 1e-5f);
        x(i) = tmp(i) / d;
      });
  Kokkos::parallel_for(
      "spike:gemv",
      Kokkos::TeamPolicy<Exec>(ec, rows, Kokkos::AUTO, 32),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<Exec>::member_type& team) {
        int r = team.league_rank();
        float acc = 0.0f;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(team, cols),
            [&](int c, float& s) { s += w(r, c) * x(c % n); }, acc);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { mv(r) = acc; });
      });
  Kokkos::parallel_for(
      "spike:act", Kokkos::RangePolicy<Exec>(ec, 0, n),
      KOKKOS_LAMBDA(int i) {
        float g = x(i);
        float u = mv(i % mv.extent(0));
        x(i) = swiglu_fn(g, u);
      });
}

struct Stats {
  double mean_us, std_us;
};

template <class F>
Stats bench(int reps, F&& f) {
  std::vector<double> ts;
  ts.reserve((size_t)reps);
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  // Drop first 10 (warmup) for the stats.
  const size_t skip = 10;
  double m = 0;
  for (size_t i = skip; i < ts.size(); ++i) m += ts[i];
  m /= (double)(ts.size() - skip);
  double v = 0;
  for (size_t i = skip; i < ts.size(); ++i) v += (ts[i] - m) * (ts[i] - m);
  v /= (double)(ts.size() - skip);
  return {m, std::sqrt(v)};
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    // argv: N ROWS COLS NCHAIN REPS. Tiny-kernel default models decode
    // (T=1 rows, ~1-2 us kernels, launch-overhead dominated).
    int N = argc > 1 ? atoi(argv[1]) : 7168;
    int ROWS = argc > 2 ? atoi(argv[2]) : 2048;
    int COLS = argc > 3 ? atoi(argv[3]) : 2048;
    int NCHAIN = argc > 4 ? atoi(argv[4]) : 200;
    const int REPS = argc > 5 ? atoi(argv[5]) : 200;
    printf("config: N=%d ROWS=%d COLS=%d NCHAIN=%d (~%d launches/step)\n", N,
           ROWS, COLS, NCHAIN, 5 * NCHAIN);

    LayerBufs b{ViewF("x", N), ViewF("tmp", N), ViewF("mv", ROWS),
                Kokkos::View<float, Exec>("sumsq"),
                Kokkos::View<float**, Kokkos::LayoutRight, Exec>("w", ROWS,
                                                                 COLS)};
    // Init (outside timing).
    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(int i) { b.x(i) = (float)(i % 1024) * 0.001f; });
    Kokkos::parallel_for(
        "initw", Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>({0, 0},
                                                              {ROWS, COLS}),
        KOKKOS_LAMBDA(int r, int c) {
          b.w(r, c) = (float)((r * 7 + c * 13) % 256) * 0.0001f;
        });
    Kokkos::fence();

    auto host_x = Kokkos::create_mirror_view(b.x);
    auto snapshot = [&] {
      Kokkos::deep_copy(host_x, b.x);
      return host_x;
    };

    // (a) Eager baseline.
    Exec ec;
    Stats se = bench(REPS, [&] {
      for (int l = 0; l < NCHAIN; ++l) run_layer(ec, b, N, ROWS, COLS);
    });
    Kokkos::deep_copy(host_x, b.x);
    std::vector<float> expect(host_x.data(), host_x.data() + N);
    printf("eager:            %8.1f us +- %5.1f\n", se.mean_us, se.std_us);

#if defined(KOKKOS_ENABLE_CUDA)
    {
      auto empty = Kokkos::Experimental::create_graph<Exec>(
          [&](Kokkos::Experimental::Graph<Exec>::root_t root) { (void)root; });
      empty.instantiate();
      Stats s0 = bench(REPS, [&] { empty.submit(); });
      printf("null graph submit: %7.2f us +- %4.2f\n", s0.mean_us, s0.std_us);
    }
    // (b) Explicit graph: rebuild run_layer as a then_* chain. Reductions
    // write into the device sumsq view (host scalars are illegal in graphs).
    auto graph = Kokkos::Experimental::create_graph<Exec>(
        [&](Kokkos::Experimental::Graph<Exec>::root_t root) {
      auto x = b.x;
      auto tmp = b.tmp;
      auto sumsq = b.sumsq;
      auto mv = b.mv_out;
      auto w = b.w;
      // Chain type: each then_* returns a distinct GraphNodeRef; hold the
      // fully type-erased GraphNodeRef<Exec> (implicit conversion) so one
      // variable can sequence the whole chain in a loop.
      using AnyNode = Kokkos::Experimental::GraphNodeRef<Exec>;
      AnyNode node = root.then_parallel_for(
          "spike:gather", Kokkos::RangePolicy<Exec>(0, N),
          KOKKOS_LAMBDA(int i) { tmp(i) = x(i) * 1.001f + 0.5f; });
      for (int l = 0; l < NCHAIN; ++l) {
        if (l > 0)
          node = node.then_parallel_for(
              "spike:gather", Kokkos::RangePolicy<Exec>(0, N),
              KOKKOS_LAMBDA(int i) { tmp(i) = x(i) * 1.001f + 0.5f; });
        node = node.then_parallel_reduce(
            "spike:sumsq", Kokkos::RangePolicy<Exec>(0, N),
            KOKKOS_LAMBDA(int i, float& s) { s += tmp(i) * tmp(i); },
            Kokkos::Sum<float, Exec>(sumsq));
        node = node.then_parallel_for(
            "spike:norm", Kokkos::RangePolicy<Exec>(0, N),
            KOKKOS_LAMBDA(int i) {
              float d = sqrtf(sumsq() / (float)N + 1e-5f);
              x(i) = tmp(i) / d;
            });
        node = node.then_parallel_for(
            Kokkos::TeamPolicy<Exec>(ROWS, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<Exec>::member_type& team) {
              int r = team.league_rank();
              float acc = 0.0f;
              Kokkos::parallel_reduce(
                  Kokkos::TeamVectorRange(team, COLS),
                  [&](int c, float& s) { s += w(r, c) * x(c % N); }, acc);
              Kokkos::single(Kokkos::PerTeam(team),
                             [&]() { mv(r) = acc; });
            });
        node = node.then_parallel_for(
            "spike:act", Kokkos::RangePolicy<Exec>(0, N),
            KOKKOS_LAMBDA(int i) {
              float g = x(i);
              float u = mv(i % mv.extent(0));
              x(i) = swiglu_fn(g, u);
            });
      }
      (void)node;
    });
    graph.instantiate();
    // Reset state, replay.
    Kokkos::parallel_for(
        "reinit", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(int i) { b.x(i) = (float)(i % 1024) * 0.001f; });
    Kokkos::fence();
    Stats sg = bench(REPS, [&] { graph.submit(); });
    auto got = snapshot();
    double maxdiff = 0;
    for (int i = 0; i < N; ++i)
      maxdiff = std::max(maxdiff, (double)std::abs(got(i) - expect[i]));
    printf("explicit graph:   %8.1f us +- %5.1f   maxdiff=%g\n", sg.mean_us,
           sg.std_us, maxdiff);
    printf("gain explicit: %.2fx\n", se.mean_us / sg.mean_us);

    // (c) cuda_capture child graph over the unmodified eager region.
    auto cap_graph = Kokkos::Experimental::create_graph<Exec>([&](Kokkos::Experimental::Graph<Exec>::root_t root) {
      auto cap = root.cuda_capture(
          ec, [&](const Exec& cec) {
            for (int l = 0; l < NCHAIN; ++l)
              run_layer(cec, b, N, ROWS, COLS);
          });
      (void)cap;
    });
    cap_graph.instantiate();
    Kokkos::parallel_for(
        "reinit2", Kokkos::RangePolicy<Exec>(0, N),
        KOKKOS_LAMBDA(int i) { b.x(i) = (float)(i % 1024) * 0.001f; });
    Kokkos::fence();
    Stats sc = bench(REPS, [&] { cap_graph.submit(); });
    auto got2 = snapshot();
    double maxdiff2 = 0;
    for (int i = 0; i < N; ++i)
      maxdiff2 = std::max(maxdiff2, (double)std::abs(got2(i) - expect[i]));
    printf("capture graph:    %8.1f us +- %5.1f   maxdiff=%g\n", sc.mean_us,
           sc.std_us, maxdiff2);
    printf("gain capture:  %.2fx\n", se.mean_us / sc.mean_us);

    // (d) cpos staleness: host scalar baked into functor at capture time.
    {
      ViewF y("y", 1);
      int cpos = 5;
      auto stale = Kokkos::Experimental::create_graph<Exec>([&](Kokkos::Experimental::Graph<Exec>::root_t root) {
        (void)root.cuda_capture(ec, [&](const Exec& cec) {
          Kokkos::parallel_for(
              "stale", Kokkos::RangePolicy<Exec>(cec, 0, 1),
              KOKKOS_LAMBDA(int i) {
                (void)i;
                y(0) = (float)cpos;
              });
        });
      });
      stale.instantiate();
      cpos = 6;  // host advances; captured graph keeps the old value
      stale.submit();
      Kokkos::fence();
      auto hy = Kokkos::create_mirror_view(y);
      Kokkos::deep_copy(hy, y);
      printf("staleness demo: cpos host=%d device-saw=%.0f (stale=%s)\n", cpos,
             (double)hy(0), hy(0) == 5.0f ? "YES" : "no");

      // Fix: device-resident counter updated between replays.
      Kokkos::View<int*, Exec> dpos("dpos", 1);
      Kokkos::deep_copy(dpos, 5);
      auto fixed = Kokkos::Experimental::create_graph<Exec>([&](Kokkos::Experimental::Graph<Exec>::root_t root) {
        (void)root.cuda_capture(ec, [&](const Exec& cec) {
          Kokkos::parallel_for(
              "fixed", Kokkos::RangePolicy<Exec>(cec, 0, 1),
              KOKKOS_LAMBDA(int i) {
                (void)i;
                y(0) = (float)dpos(0);
              });
        });
      });
      fixed.instantiate();
      Kokkos::deep_copy(dpos, 6);
      fixed.submit();
      Kokkos::fence();
      Kokkos::deep_copy(hy, y);
      printf("device-counter: dpos host=6 device-saw=%.0f (fixed=%s)\n",
             (double)hy(0), hy(0) == 6.0f ? "YES" : "no");
    }
#else
    printf("CUDA disabled: graph sections skipped (Serial fallback = eager)\n");
#endif
  }
  Kokkos::finalize();
  return 0;
}
