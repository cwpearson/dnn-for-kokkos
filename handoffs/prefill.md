# HANDOFF — LFM2.5 prefill gap (the remaining ~100×)

## 0. Re-measured 2026-10-23 on `main` @ `aa3d917` (RTX 3090, greedy)

The decode work (split-KV attention, device sampling, cached probes) landed;
prefill is now the sole large gap and it is **dominated by a fixed per-chunk
cost, not by token count**:

| prompt | chunks | prefill | tok/s |
|---|---|---|---|
| 15 (server, warm suffix) | 1 | **944 ms** | 16 |
| 6002 | 3 | 2254 ms | 2662 |
| 14006 | 7 | ~5000 ms | 2828 |

So ~750-950 ms **per 2048-token chunk regardless of how full it is**. That is
why a prefix-cache hit with a 265-token tail still costs ~1 s, and why a cold
10.6k prompt costs ~5.7 s (6 chunks). `ceil(tokens/prefill_chunk_size)` is the
real cost model.

### nsys breakdown of a 2.25 s / 3-chunk prefill window

| stage | ms | launches |
|---|---|---|
| cuBLAS/cutlass GEMM (per-expert) | 750 | 6087 |
| `dequant_to_dense` (experts) | 359 | 6525 |
| `moe_scatter_add_weighted` | 157 | 2175 |
| model elementwise lambdas | 133 | 424 |
| `moe_gather_rows` | 92 | 2175 |
| `swiglu` | 90 | 2181 |
| shortconv | 60 | 270 |
| attention (cuDNN SDPA) | 47 | 30 |
| rmsnorm / rope / gemv / combine | 48 | 898 |

Effective bandwidth: GEMM ~61 GB/s of B-matrix streaming, dequant ~150 GB/s,
gather 25 GB/s, scatter 15 GB/s, swiglu 23 GB/s — all far below the 3090's
~700 GB/s usable read bandwidth, so this is not a hardware limit.

### Root cause

`moe_forward_sparse` (models/lfm2_5/KokkosDNN_LFM25Model.hpp) materializes every
routed expert's three matrices as bf16 once per layer per chunk
(`dequant_to_dense` 3 x 1536x2048 x 2 B = 18.9 MB per expert), runs cuBLAS on
the gathered rows, then scatters. With 64 experts x 38 MoE layers that is
**~92 GB of bf16 write+read traffic per chunk, independent of C** — at the
measured ~85 GB/s effective that alone is ~1 s. The GEMM then runs at M =
C*4/64 rows per expert (M=16 at C=265, M=128 at a full chunk), i.e. a
memory-bound shape that cuBLAS feeds at ~61 GB/s, 10x off the limit.

Byte accounting per weight element: dequantize-once costs ~4 B (write bf16 +
read it back), packed on-the-fly (`Gguf::quant_gemm`) costs N_rows x 0.56 B, so
the crossover is N_rows ~ 7, i.e. C ~ 112 tokens. Below that the packed path is
cheaper on bytes; above it, dequantize-once wins — **but both are 5-10x off
their bounds**, so the crossover is not where the win is.

### What to do (ranked)

1. **Packed quantized expert GEMM with per-weight reuse across M rows** (MMQ
   style: dequantize each weight element once into registers/shared and
   multiply the M gathered activation rows there). Removes the 92 GB entirely
   and turns the GEMM memory-bound-optimal. Biggest lever, largest effort.
2. **Fuse the epilogue chain into the expert GEMM**: gather -> gate/up ->
   SiLU -> down -> weighted scatter is currently 5 kernels per expert
   (gather 92 ms + gemm 750 ms + swiglu 90 ms + scatter 157 ms per 3 chunks).
3. **Interim vectorization** of `dequant_to_dense`, `moe_gather_rows`,
   `moe_scatter_add_weighted`, `swiglu`: each is a pure elementwise/
   permutation kernel running at 15-150 GB/s, so a 128-bit-load + coalesced
   version is a 2-3x win on ~700 ms of the 2.25 s without touching numerics.
4. **Adaptive path by rows per expert** (packed below N~7, dequantize-once
   above) — cheap, helps short tool-call suffixes, not the 265-token case.
5. Host grouping per layer (`deep_copy` of <= C ints/floats, twice per layer)
   is no longer the problem: it was fixed for decode and is O(76) fences per
   chunk, not 4864.

Reproduce with:
```
nsys profile -o prof -t cuda ./build/examples/lfm2_5/lfm2_5_generate \
  --weights ...gguf --tokenizer ...json --prompt "$(python3 -c \
  "print('hello world. '*2000, end='')")" --max-new 1 --max-seq-len 8192 --timings
```
then aggregate `CUPTI_ACTIVITY_KIND_KERNEL` over the final `prefill_ms` window
(the load phase emits the same `dequant_to_dense` kernels, so a time window is
required).


**Status: open problem. Decode is now within ~2× of llama.cpp (82.9 vs ~164 tok/s
on RTX 3090 after `56b4963`); prefill is the dominant remaining gap. No worktree
exists yet — create one from current `main` when starting.**

## 1. Measured baseline (RTX 3090, LFM2.5-24B-A2B Q4_K_M, greedy)

From `/tmp/opencode/kdnn-bench/bench_results/SUMMARY.md` (§4–5), **measured before
the native decode fixes**:

| phase | KokkosDNN-tpl | llama.cpp | gap |
|---|---|---|---|
| Prefill (902 tok) | **13.5 [5.3] tok/s** | **~2400 tok/s** | **~180× [~450×]** |
| SM during prefill | 68–77% steady, 100% spikes | — | busy-but-inefficient |

**Re-measure first.** The grouped-quant GEMM in `56b4963` also accelerates the
chunk/prefill MoE path (`moe_forward_sparse` calls the same `quant_gemm`), and
team-parallel norms help prefill too — so current-main prefill is already faster
than 13.5 tok/s by an unknown amount. Re-run the bench matrix (§5) on current
`main` before designing fixes, or you will optimize a stale number.

## 2. Why prefill is slow (evidence-backed suspects, largest first)

1. **~144k narrow per-expert GEMMs per 902-token prefill.** 38 MoE layers × top-4
   routing × per-chunk expert fan-out: each expert runs its own gathered
   `quant_gemm` gate/up/down triple on a thin row-slice. No grouped/fused MoE
   GEMM exists. SMs are 68–77% busy while delivering ~13 tok/s → small-matrix /
   launch-overhead inefficiency, not idleness. (Bench §6.)
2. **Host-side expert grouping per layer per chunk** (`moe_forward_sparse`:
   2 + 2×experts blocking `deep_copy`s per layer). The C==1 decode path fixed
   this for decode (`56b4963`); the C>1 chunk path still pays it, scaled by
   chunk count.
3. **Unfused elementwise chains** around the MoE/GEMM core: `moe_route_sigmoid`
   (~65 µs single-threaded × 38 layers ≈ 2 ms/tok — nsys-verified on decode,
   scales with chunk rows), resid adds, flatten/reshape, gather-last. ~1 ms/tok
   of pure launch overhead at decode; worse at prefill widths.
4. **Weight-load TTFT (~36 s vs llama ~3 s).** Separate from steady prefill rate
   but dominates end-to-end latency: kdnn re-reads + host-dequantizes 14.5 GB
   and H2Ds ~20 GB per process; llama mmaps + offloads quants directly
   (~14 GB resident). Also +40% VRAM for the same checkpoint (19.6 vs 14 GB),
   which shrinks batching headroom.
5. **Attention over growing KV** is *not* the prime suspect (10/40 layers are
   attention, GQA 32/8, D=64, cuDNN SDPA already dispatches) — verify with nsys
   before touching it.

## 3. Candidate directions (roughly ranked by expected payoff)

- **A. Grouped/batched expert GEMM.** Fuse each layer's per-expert
  gate/up/down triples into one grouped kernel (CUTLASS grouped GEMM, or a
  batched dequantize-then-GEMM over experts). This is the single biggest lever:
  it converts ~144k narrow launches into ~10² wide ones. Must handle packed
  Q4_K (dequant-on-the-fly or a TPL dequantize pass first). Device-side expert
  grouping (or persistent staging) to kill the host round-trips in the same pass.
- **B. Team-parallel `moe_route_sigmoid` + fused epilogues.** Same treatment
  the norms got in `56b4963` (one team per row, `TeamThreadRange` over the
  expert axis). Cheap, portable, no numerics risk beyond FP reassociation.
- **C. Fuse the elementwise chains.** resid-add, flatten/reshape, `sc_gate`
  multiply, gather-last into neighboring kernels or single fusions. Reduces
  launch count ~2–3×/layer at prefill widths.
- **D. Weight-load fast path.** mmap + direct quant offload (llama-style) or at
  minimum cache the dequantized device image across processes. Cuts TTFT ~10×
  and the +40% VRAM overhead; orthogonal to tok/s but dominates user-visible
  latency.
- **E. Prefill-chunk tuning.** `prefill_chunk_size` trades activation memory
  for kernel width; re-sweep it on current main (default 2048 may be too narrow
  for MoE efficiency on H100, too wide for 3090 VRAM).

Out of scope: punishing attention (already TPL'd), short-conv depthwise (K=3,
30 layers but cheap), server batching (separate workstream; helps throughput
only after per-request prefill is fixed).

## 4. Methodology (from bench §3/§8 — follow it or numbers won't compare)

- Same weights file, same prompts, greedy decode both sides
  (`--temperature 0 --repetition-penalty 1.0` ≈ pure argmax).
- kdnn driver pays full weight load per process: derive rates by subtraction —
  `prefill = 895/(L1−S1)`, `decode(short) = 63/(S64−S1)`; do NOT quote raw wall.
- 1 Hz `nvidia-smi dmon` per run for SM%/VRAM; steady-state windowing
  (VRAM ≥98% of run max) to exclude the load phase.
- **Isolated GPU.** Shared-box contention cost 3–6× in the original matrix
  (load avg ~23). All prefill claims require an otherwise-idle GPU; re-run on
  H100 for the number that matters (§9 extrapolation: decode is memory-bound so
  H100 lifts both ~3× relatively evenly — the gap is structural, not clock-speed).
- Scripts: `/tmp/opencode/kdnn-bench/bench_results/{run_bench.sh,analyze.py}`.
- Profile before cutting code: `nsys profile` of one prefill (902-tok LONG
  prompt) on current main; rank kernels before choosing A vs B vs C.

## 5. Success criteria / milestones

- **M1 (close-out):** prefill gap <10× vs llama.cpp on the same isolated GPU
  (i.e. ≥~240 tok/s on 3090, proportionate on H100), with no `native_*`
  regression in the TPL dispatch table (see `kdnn-tpl/HANDOFF.md` §4).
- **M2 (stretch):** TTFT (excl. weight load) within 3×; weight-load fast path (D)
  tracked separately — it needs I/O design, not kernels.
- **Numerics bar** (from `56b4963` experience): grouped/norm-style changes shift
  first-token logits (RMS ~1.6, routing flips on near-ties). Require: existing
  tests green, chunk-1 ≡ chunk-N token-stream equivalence on tiny + real
  weights, and an explicit logit-envelope note in the commit message. Evals use
  tolerance/perplexity, never token-exactness.

## 6. Starting point

```bash
git -C /workspace worktree add --detach /tmp/opencode/kdnn-prefill origin/main
# rebuild native example in-worktree, re-run bench matrix (§4), nsys the prefill
```

Reference paths: weights `/workspace/weights/lfm2_5/LFM2.5-24B-A2B-Q4_K_M.gguf`,
tokenizer `/workspace/weights/lfm2_5/tokenizer.json`, Kokkos
`/workspace/install/kokkos-cuda`, `nsys`/`ncu` in `/usr/local/cuda/bin`.
Commit trailer (repo policy): `Co-authored-by: Muse Spark 1.3 <noreply@carlpearson.net>`.
