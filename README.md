# DNN for Kokkos

> **WARNING — unofficial, experimental project.**
> This is **not** part of, endorsed by, or affiliated with the Kokkos ecosystem.
> It is an independent experiment for the author to experiment with agentic coding.
> Do not treat it as a supported, reviewed, or production-ready library.

## What this is

DNN for Kokkos is a small, Kokkos-compatible DNN library built as a proof-of-concept around **Llama 3.1 8B-Instruct inference**.
It provides:

- Native Kokkos implementations of every operator the model needs (GEMM/GEMV, RMSNorm, softmax, RoPE, SwiGLU, embedding gather, grouped-query causal attention, greedy sampler).
- Optional dispatch to CUDA libraries when enabled at configure time: GEMM to **cuBLAS** (FP32 via `sgemm`, BF16 via `cublasGemmEx` with FP32 accumulation), attention to **cuDNN SDPA** (with automatic fallback to the native kernel where cuDNN has no suitable path).
- Unit tests and a tiny-random-weights end-to-end path so the full pipeline runs without downloading a 16 GB checkpoint.

Scope is deliberately narrow: **bfloat16 storage/compute by default** (`--fp32` opts in), no weight quantization. A small set of training ops (dense forward/backward, sigmoid/tanh/ReLU, BCE-with-logits, SGD/Adam) complements the inference path. All ops are templated on the scalar type (`float`, `Kokkos::Experimental::bhalf_t`) with FP32 accumulation; TPL paths cover GEMM for both scalars and fall back to native otherwise.

## Requirements

- CMake ≥ 3.22, C++20 compiler
- An installed Kokkos (Serial is enough for tests; CUDA backend for GPU)
- Optional: CUDA Toolkit (for `KokkosDNN_ENABLE_CUBLAS`), cuDNN (for `KokkosDNN_ENABLE_CUDNN`)

## Build

```bash
cmake -S . -B build \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure
```

Options:

| Option | Default | Effect |
| --- | --- | --- |
| `KokkosDNN_ENABLE_TESTS` | ON | build `tests/` |
| `KokkosDNN_ENABLE_EXAMPLES` | ON | build `examples/` (`llama_generate`, `xor_train`, + example tests) |
| `KokkosDNN_ENABLE_BENCHMARKS` | OFF | build `benchmarks/kokkosdnn_bench` (Google Benchmark, fetched if needed) |
| `KokkosDNN_ENABLE_CUBLAS` | OFF | GEMM via cuBLAS on CUDA spaces |
| `KokkosDNN_ENABLE_CUDNN` | OFF | attention via cuDNN SDPA (with automatic native fallback; needs `<cudnn.h>` + `cudnn-frontend`, fetched automatically) |

## Examples

- [llama](examples/llama/) — Llama 3.1 8B-Instruct (plus 3.2 1B) inference with KV-cache autoregressive generation via the `llama_generate` driver.
- [xor](examples/xor/) — 2-layer MLP that learns the XOR truth table with full-batch Adam via the `xor_train` driver.

See each directory's README for flags, run commands, and design notes.

## Benchmarks

`benchmarks/kokkosdnn_bench` (Google Benchmark, opt-in via
`-DKokkosDNN_ENABLE_BENCHMARKS=ON`) times the core ops at Llama
3.1 8B-Instruct shapes and reports `GFLOPS` (GEMM/GEMV/attention) or `GBps`
(elementwise) counters. GEMM registers `Native` vs `Dispatched`
variants for both `float` and `bhalf_t`: `Native` forces the Kokkos kernel
(`use_cublas/use_cudnn = false`), `Dispatched` uses the default handle, so
on a CUDA build with `KokkosDNN_ENABLE_CUBLAS` it exercises the cuBLAS
path. GEMV (decode/lm_head shape) shares the same cuBLAS dispatch, so it
also registers both variants.

```bash
./build/benchmarks/kokkosdnn_bench --benchmark_list_tests
./build/benchmarks/kokkosdnn_bench --benchmark_filter='Gemm.*' --benchmark_counters_tabular=true
./build/benchmarks/kokkosdnn_bench --benchmark_filter='Attention.*' --benchmark_counters_tabular=true
./build/benchmarks/kokkosdnn_bench --benchmark_filter='BM_Gemm_Float_(Native|Dispatched)/M:16.*' \
  --benchmark_min_time=0.5s --benchmark_format=json > gemm.json
```

Without the TPL options (e.g. CPU-only CI) `Native == Dispatched` by
construction. Large prefill rows (`M=64`, `T=128`, lm-head vocab) are
GPU-oriented: the naive native kernel takes seconds per iteration on Serial,
so subset with `--benchmark_filter` or lower `--benchmark_min_time` there.

## Limitations

- Attention uses a streaming online-softmax kernel (FlashAttention-style):
  one thread per `(query token, query head)` streams over keys with O(1)
  extra memory, so no `[T, Hq, K]` score tensor is ever materialized and
  128k contexts are feasible. See `examples/llama/` for the long-context
  KV-cache and chunked-prefill notes.
- Sampling supports greedy, `temperature` / `top-k` / `top-p` / `min-p`,
  repetition penalty, and seeded RNG; no batching.

## TODO — toward real Llama 3.1 8B-Instruct inference

Minimum checklist for a faithful, usable 8B run (ordered by priority):

- Performance (future): CPU TPL path (no `OpenBLAS`/`MKL` dispatch);
  weight quantization out of scope per above.
- Small-model examples: `SmolLM2-135M` (same `llama` + `silu` + no-bias
  family; needs preset + tokenizer/vocab work; fast CPU/CI download).
- Small-model examples: `Qwen3-0.6B` / `1.7B` (needs QK-Norm support plus
  tokenizer/preset work; verify bias and RoPE deltas against their
  `config.json`).
