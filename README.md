# DNN for Kokkos

> **WARNING — unofficial, experimental project.**
> This is **not** part of, endorsed by, or affiliated with the Kokkos ecosystem.
> It is an independent experiment for the author to experiment with agentic coding.
> Do not treat it as a supported or production-ready library.

## What this is

DNN for Kokkos is a small, Kokkos-compatible deep neural network library.
It provides:

- [Native Kokkos implementations of common DNN operators](https://cwpearson.github.io/dnn-for-kokkos/api.html) (GEMM/GEMV with optional bias, dense forward/backward, embedding gather, RMSNorm/LayerNorm, softmax, causal grouped-query and bidirectional attention, 1D and 2D-axial RoPE, SwiGLU/GELU and other activations, patch embedding, token assembly, LayerScale, losses, SGD/Adam, sampling, KV-cache helpers).
- Optional dispatch to CUDA libraries when enabled at configure time: GEMM to **cuBLAS** (FP32 via `sgemm`, BF16 via `cublasGemmEx` with FP32 accumulation), attention to **cuDNN SDPA** (with automatic fallback to the native kernel where cuDNN has no suitable path).
- Unit tests plus end-to-end paths in the examples so the full pipelines run without downloading large checkpoints.

Scope is deliberately narrow: **bfloat16 storage/compute by default** (`--fp32` opts in), no weight quantization. All ops are templated on the scalar type (`float`, `Kokkos::Experimental::bhalf_t`) with FP32 accumulation; TPL paths cover GEMM for both scalars and fall back to native otherwise.

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
| `KokkosDNN_ENABLE_EXAMPLES` | ON | build `examples/` (`llama_generate`, `smollm2_generate`, `xor_train`, `dinov3_encode`, + example tests) |
| `KokkosDNN_ENABLE_BENCHMARKS` | OFF | build `benchmarks/kokkosdnn_bench` (Google Benchmark, fetched if needed) |
| `KokkosDNN_ENABLE_CUBLAS` | OFF | GEMM via cuBLAS on CUDA spaces |
| `KokkosDNN_ENABLE_CUDNN` | OFF | attention via cuDNN SDPA (with automatic native fallback; needs `<cudnn.h>` + `cudnn-frontend`, fetched automatically) |
| `KokkosDNN_ENABLE_IMAGE` | OFF | optional image component (`KokkosDNN::Image`, stb-backed I/O + preprocessing) |

## Examples

- [llama](examples/llama/) — Llama 3.1 8B-Instruct (plus 3.2 1B) text inference with KV-cache autoregressive generation.
- [smollm2](examples/smollm2/) — SmolLM2-135M (base + Instruct) text inference with KV-cache autoregressive generation.
- [xor](examples/xor/) — 2-layer MLP that learns the XOR truth table with full-batch Adam.
- [dinov3](examples/dinov3/) — DINOv3 ViT-B/16 image-encoder inference.

See each directory's README for flags, run commands, and design notes.

## Benchmarks

`benchmarks/kokkosdnn_bench` (Google Benchmark, opt-in via
`-DKokkosDNN_ENABLE_BENCHMARKS=ON`) times the core ops at representative
transformer shapes and reports `GFLOPS` (GEMM/GEMV/attention) or `GBps`
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
