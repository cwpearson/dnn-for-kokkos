# Graph-decode spike (prototype, not wired into the build)

Standalone micro-spike for the Kokkos-graphs decode investigation. It is
deliberately NOT referenced by the top-level `CMakeLists.txt`, so the Serial
(and every other) build is unaffected.

## Build (needs a CUDA Kokkos install, e.g. the perf-deps one)

```sh
cmake -S spikes/graph_decode -B /tmp/graph-spike-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/graph-spike-build -j
flock /tmp/opencode/gpu.lock -c "/tmp/graph-spike-build/graph_spike [N ROWS COLS NCHAIN REPS]"
```

The source is backend-conditional (`#if defined(KOKKOS_ENABLE_CUDA)`; Serial
falls back to the eager loop), which is the pattern a production insertion
should follow (see report: capture object owned by `Session`, eager fallback
when CUDA/graphs are unavailable).

## What it shows

- `eager` vs `explicit graph` (`create_graph` + `then_*` chain) vs `capture
  graph` (`cuda_capture` child graph over the unmodified eager region):
  bit-identical replay (`maxdiff=0`) with launch overhead collapsed to one
  `cudaGraphLaunch` (~1 us null-submit).
- `staleness demo`: a host scalar (`cpos` analogue) baked into a captured
  functor goes stale on replay; the `device-counter` variant (position in
  device memory, updated between replays) picks up the new value. This is
  the core production blocker/fix pair.
