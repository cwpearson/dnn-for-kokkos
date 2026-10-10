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
- A Llama example (`examples/llama/`): 3.1 8B-Instruct config defaults, a safetensors checkpoint loader (on-disk F32/F16/BF16 are converted to the model scalar), a faithful Hugging Face `tokenizer.json` loader (special-token split, tiktoken Split pre-tokenizer, GPT-2 byte mapping, per-piece BPE; validated ID-for-ID against `tokenizers`), a KV-cache autoregressive model, and a `llama_generate` driver.
- Unit tests and a tiny-random-weights end-to-end path so the full pipeline runs without downloading a 16 GB checkpoint.

Scope is deliberately narrow: **bfloat16 storage/compute by default** (`--fp32` opts in), **inference only** (prefill + greedy decode), no training, no weight quantization (the KV cache has an optional int8 mode, see below). All ops are templated on the scalar type (`float`, `Kokkos::Experimental::bhalf_t`) with FP32 accumulation; TPL paths cover GEMM for both scalars and fall back to native otherwise.

## Layout

```text
CMakeLists.txt            standalone build, find_package(Kokkos)
cmake/                    config header + package config templates
src/
  KokkosDNN.hpp           umbrella header
  KokkosDNN_Types.hpp     FP32 View aliases
  KokkosDNN_Handle.hpp    TPL dispatch handle (keeps call sites ifdef-free)
  KokkosDNN_Gemm.hpp      Y = X·Wᵀ (+ cuBLAS path)
  KokkosDNN_RMSNorm.hpp
  KokkosDNN_Softmax.hpp
  KokkosDNN_RoPE.hpp      NeoX-style rotary, rope_theta = 500000
  KokkosDNN_SwiGLU.hpp
  KokkosDNN_Embedding.hpp
  KokkosDNN_Attention.hpp (+ _CuDNN: working SDPA dispatch, _CuDNN.cpp TU)
  KokkosDNN_KVCache.hpp   paged / int8 KV (block-table attention,
                          per-token per-head quant scales)
  KokkosDNN_Sampler.hpp   greedy argmax
  KokkosDNN_PrintConfiguration.cpp
examples/llama/           Llama 3 example (not part of the library)
  KokkosDNN_LlamaConfig.hpp    3.1 8B-Instruct defaults + tiny_test()
  KokkosDNN_LlamaWeights.{hpp,cpp}
  KokkosDNN_Tokenizer.{hpp,cpp}
  KokkosDNN_LlamaModel.hpp     KV-cache Model::forward/generate (dense /
                          paged x FP / int8 modes)
  llama_generate.cpp      driver
  Test_LlamaGenerate.cpp / Test_Safetensors.cpp / Test_LlamaKVCache.cpp (example-level tests)
tests/                    one executable per core op
benchmarks/               Google Benchmark micros (native vs TPL dispatch)
```

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
| `KokkosDNN_ENABLE_EXAMPLES` | ON | build `examples/llama/llama_generate` (+ example Llama tests) |
| `KokkosDNN_ENABLE_BENCHMARKS` | OFF | build `benchmarks/kokkosdnn_bench` (Google Benchmark, fetched if needed) |
| `KokkosDNN_ENABLE_CUBLAS` | OFF | GEMM via cuBLAS on CUDA spaces |
| `KokkosDNN_ENABLE_CUDNN` | OFF | attention via cuDNN SDPA (with automatic native fallback; needs `<cudnn.h>` + `cudnn-frontend`, fetched automatically) |

## Run the PoC

Tiny random model (no downloads, runs anywhere):

```bash
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 8
```

Real checkpoint (needs weights + tokenizer from `meta-llama/Llama-3.1-8B-Instruct` or equivalent):

```bash
./build/examples/llama/llama_generate \
  --weights model-00001-of-00004.safetensors model-00002-of-00004.safetensors ... \
  --tokenizer tokenizer.json \
  --prompt "Explain Kokkos in one sentence." \
  --max-new 64
```

Expected tensor names are the Hugging Face ones (`model.embed_tokens.weight`, `model.layers.N.self_attn.q_proj.weight`, …, `lm_head.weight`).

Stopping defaults to the Instruct set `{eos 128001, eom 128008, eot 128009}`
(from `tokenizer.json` when `--tokenizer` is given, else `Config`); append
extra ids with `--stop <id> ...`.

Context defaults to `max_seq_len = 8192` (PoC cap); the arch max `131072`
is supported, including `llama3`-type rope scaling from the
Llama 3.1 8B-Instruct `config.json`.

KV-cache modes (`--kv-page-size N`, `--kv-int8` / `--kv-fp`; config.json
keys `kv_page_size`, `kv_int8` / `kv_quantized`, `kv_quant: "int8"|"none"`):
dense FP by default; `--kv-page-size 16` pages the cache into blocks
addressed through an explicit block table and grown on demand, so a short
prompt on a long-context config only rents the pages it touches;
`--kv-int8` stores symmetric int8 KV with per-token per-head FP32 scales
(~2x fewer KV bytes plus a small scale table). The two combine (paged
int8). Paged dense is bit-exact vs dense; int8 trades small numerical
drift for the smaller footprint (see `Test_LlamaKVCache.cpp`).

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

## Memory note

BF16 8B weights are ~16 GiB, so a full 8B BF16 run fits on a 24 GiB card
such as an RTX 3090. FP32 (`llama_generate --fp32`) needs ~32 GiB and does
**not** fit there. Use the tiny config for local smoke tests and a
large-GPU or CPU-Kokkos build for the real checkpoint.

## Limitations

- Attention uses a streaming online-softmax kernel (FlashAttention-style):
  one thread per `(query token, query head)` streams over keys with O(1)
  extra memory, so no `[T, Hq, K]` score tensor is ever materialized and
  128k contexts are feasible. The model additionally passes KV-cache
  prefixes as subviews (no per-layer repack copy).
  Remaining 128k considerations: the BF16 KV cache itself is ~256 MiB per
  layer (~8 GiB for 32 layers at 128k) in dense FP; `--kv-int8` roughly
  halves that (plus one FP32 scale per token per KV head), and paged KV
  (`--kv-page-size 16`) rents only touched pages instead of the full
  window upfront (see `Config::kv_cache_bytes()`; the driver prints the
  full-window figure). Prefill activations are bounded by
  chunked prefill (`Config::prefill_chunk_size`, default 2048):
  `forward()` processes long prompts in chunks, so scratch stays
  O(chunk·dim) regardless of context length (exact match to single-shot).
- The tokenizer implements the full HF `tokenizers` pipeline for the Llama 3
  `tokenizer.json` (special-token longest-match split, the tiktoken Split
  regex with Unicode `\p{L}`/`\p{N}`, GPT-2 byte mapping, per-piece BPE,
  byte-level decode that skips specials by default). It is validated ID-for-ID
  against Python `tokenizers` on the real file: `Test_Tokenizer.RealFileOracle`
  (runs when `KOKKOSDNN_TOKENIZER_JSON` points at a Llama 3 `tokenizer.json`;
  92 targeted + 2000 fuzz differential cases pass). Re-validate a new file with
  `KOKKOSDNN_TOKENIZER_JSON=tokenizer.json ./build/examples/llama/test_tokenizer`.
- Sampling supports greedy, `temperature` / `top-k` / `top-p` / `min-p`,
  repetition penalty, and seeded RNG; no batching.

## TODO — toward real Llama 3.1 8B-Instruct inference

Minimum checklist for a faithful, usable 8B run (ordered by priority):

- Performance (future): CPU TPL path (no `OpenBLAS`/`MKL` dispatch);
  weight quantization out of scope per above.
