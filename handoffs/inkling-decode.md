# HANDOFF — Inkling 2 Mini server vs llama.cpp (RTX 3090)

## 0. Status (2026-11-21, `main` @ `945dc6c`, RTX 3090, greedy, IQ4_NL GGUF)

| workload | kokkosdnn_server (TPL build) | llama-server | gap |
|---|---|---|---|
| decode, 128 gen (short prompt) | **24.1 tok/s** (was 0.5) | 47.7 tok/s | **2.0x** |
| decode, 64 gen (1457-tok prompt) | 19.9 tok/s (was 3.0) | 44.4 tok/s | 2.2x |
| prefill 1457 tok | 3.99 s = 365 tok/s | 3.19 s = 456 tok/s | 1.25x |
| prefill 27 tok | 0.42 s | 0.18 s (14 tok) | ~2.3x |

Prefill is close enough to declare victory (~1.25x, same cuBLAS+dequant
structure as llama). Decode went 0.5 -> 24 tok/s in 7 commits and is now
overhead-bound on both sides (Inkling MoE streams only ~2.5 GB/token after
top-8 routing; neither side is near HBM bandwidth).

Methodology: ` /tmp/opencode/inkling-bench/run.sh {llama,kdnn} {short,med} N`
runs ONE server at a time (single contended GPU), warms up, times one greedy
request; ground truth is each server's own per-turn timings (kdnn stderr
`decode_tok_s`, llama `timings`). llama.cpp built with `-DGGML_CUDA=ON`,
`-DCMAKE_CUDA_ARCHITECTURES=86`. kdnn TPL build: `KokkosDNN_ENABLE_CUBLAS=ON`
`KokkosDNN_ENABLE_CUDNN=ON` (the old `build-inkling-cuda` has both OFF and
is NOT a fair baseline). nsys profiles: `/tmp/opencode/inkling-bench/*.sqlite`.

## 1. What was done (all on main, tests green)

1. `7386df9` IQ4_NL int8-dot GEMV (`Wd::IQ4NL`, mirrors Q80) + MoE scratch
   plumbing + `--arch inkling` config-sniff fix (HF config wraps a
   `text_config` object that matched the glimmer sniff first).
2. `7430f53` Team-parallel attention / residual / message-gate (the 4 ms,
   0.7 ms, 0.2 ms single-thread-per-row kernels). Found live: only rank 0
   computed `denom` while all threads divided by it (-> inf) — fixed by
   sharing via scratch; added in-place conv aliasing regression coverage.
3. `17e704c` Attention passes 1-2 parallel over keys (prefill 9.6 -> 4.1 s).
4. `22442f8` Residual score reduction parallel over d (26 -> 3 ms/tok).
5. `141322b` Fused MoE decode, 2 launches/visit, zero host fences.
6. `608d3a8` Fused MoE dots moved to the int8 path (19 -> 8.5 ms/tok).
7. `945dc6c` Shortconv single-token update path at C==1 + aliasing fix
   (the update kernels re-read `x_new` after overwriting `y`; all decode
   call sites alias x and out).

Suites: test_gguf 26/26, test_inklingmodel 12/12 (incl. team-vs-legacy
attention at 1e-5, MoE-fused vs host reference at 1e-3 with quantizer-aware
reference, in-place conv checks), test_inklingops 12/12, server parsing pass.

## 2. Where decode time goes now (nsys, short prompt, per token)

| cost | note |
|---|---|
| 12.7 ms non-MoE GEMV int8 (~400 launches) | 7 linears + router + lm_head x48 visits; launch-dominated |
| 8.5 ms fused MoE (66 launches) | int8 dots, ~200 GB/s effective; near dp4a arithmetic roof |
| 2.5 ms activation quantize (~750 launches) | one per GEMV; most inputs shared (hn feeds 5) |
| 3.2 ms residual + 3.3 ms gate + 2 ms norms | small grids, serial single() sections |
| ~9 ms gaps | ~5000 launches x ~2 us Kokkos overhead |

## 3. Ranked follow-ups

1. **CUDA-graph capture of the decode step (~6-10 ms).** Shapes are fixed
   at C == 1 and steady state allocates nothing, so the whole token is
   capturable. Kills the launch-overhead floor both sides share. Needs
   care: no Kokkos fence / host-visible deep_copy inside the region
   (true today), same-stream capture, replay with persistent views.
2. **MoE prefill grouped-GEMM.** Prefill still dequantizes every expert
   per chunk (~98 GB/chunk). Device-side grouped GEMM over packed experts
   (no dense traffic) — design notes exist from the prefill workstream;
   Inkling's fused decode tables (`InklingMoeTableCache`) are reusable.
3. **Activation-quantize sharing.** Hoist one quantize per unique input
   row per token through a prequantized GEMV entry point (~2 ms).
   Touches shared `GgufMvq` API used by Glimmer/LFM2.5 — coordinate.
4. **GEMV small-matrix efficiency.** dp4a unpack overhead caps IQ4_NL at
   ~200-700 GB/s depending on shape; tensor-core MMQ is the real fix
   (large project, shared with Glimmer).

## 4. Known non-issues (do not chase)

- `build-inkling-cuda` (CUBLAS/CUDNN OFF) numbers are not comparable.
- Long-context decode no longer collapses (was 3.0, now 19.9 tok/s).
- Weight load is mmap-fast (~1 s H2D); TTFT is compute, not I/O.
- Pushing to origin/main needs explicit permission; main is ahead.
