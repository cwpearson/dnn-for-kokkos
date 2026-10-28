# DNN for Kokkos

> **WARNING — unofficial, experimental project.**
> This is **not** part of, endorsed by, or affiliated with the Kokkos ecosystem.
> It is an independent experiment for the author to experiment with agentic coding.
> Do not treat it as a supported or production-ready library.

## What this is

DNN for Kokkos is a small, Kokkos-compatible deep neural network library.
It provides:

- [Native Kokkos implementations of common DNN operators](https://cwpearson.github.io/dnn-for-kokkos/api.html) (GEMM/GEMV with optional fused bias, dense forward/backward, embedding gather, RMSNorm/LayerNorm/GroupNorm, softmax, causal grouped-query and bidirectional attention, 1D and 2D-axial RoPE, SwiGLU/GELU/SiLU and other activations, convolutions (dense/depthwise/grouped, Conv1D/general), patch embedding, token assembly, LayerScale, LSTM, adaptive norms, audio DSP (STFT/upsample), MoE top-k routing with packed-quant expert GEMM, GGUF checkpoint loading, losses, SGD/Adam, sampling, KV-cache helpers).
- Diffusion-transformer operators for Sana image generation: SiLU, scaled RMSNorm, affine-free LayerNorm, AdaLN modulation (`x*(1+scale)+shift`), sinusoidal timestep embeddings, LiteLA linear attention (ReLU kernel), bidirectional cross-attention, GLUMBConv MixFFN (1x1 + depthwise 3x3 + SiLU gating), 1x1 patch embedding / unpatchify, and a FlowEuler sampler with classifier-free guidance.
- Optional dispatch to vendor libraries when enabled at configure time: GEMM to **cuBLAS** on CUDA (FP32 via `sgemm`, BF16 via `cublasGemmEx` with FP32 accumulation, fused GEMM+bias via cublasLt epilogue) or **hipBLAS** on HIP, GEMM/GEMV (FP32) to **Apple Accelerate** on host spaces, attention to **cuDNN SDPA**, convolutions (dense/depthwise/grouped) to **cuDNN / MIOpen** — each with automatic fallback to the native kernel where the TPL has no suitable path (MIOpen has no fused SDPA, so its attention path is plumbing with native fallback). TPL dispatch requires contiguous (`LayoutRight`, row-pitch == extent) views — padded/strided inputs decline the TPL and run the native kernel. To observe which path each op takes, set `KOKKOSDNN_DEBUG_DISPATCH=1`: every `gemm`/`gemv`/`gemm_bias`/attention call is counted as TPL-served vs native fallback and the table is printed to stderr at exit (no-op overhead when unset). Note the MoE-expert `quant_gemm` has no counter — it is native-only by design (packed Q4_K has no TPL mapping), so a zero `native_gemm` row does not mean every GEMM reached a TPL.
- Unit tests plus end-to-end paths in the examples so the full pipelines run without downloading large checkpoints.

Scope is deliberately narrow: **bfloat16 storage/compute by default** (`--fp32` opts in). All ops are templated on the scalar type (`float`, `Kokkos::Experimental::bhalf_t`) with FP32 accumulation; quantized GGUF checkpoints are supported via host-side dequantization (MoE experts stay packed for `quant_gemm`); TPL paths cover GEMM for both scalars (cuBLAS/hipBLAS) plus convolutions/attention (cuDNN/MIOpen) and fall back to native otherwise.

## Requirements

- CMake ≥ 3.22, C++20 compiler
- An installed Kokkos >= 5.2.1 (Serial is enough for tests; CUDA backend for GPU)
- Optional: CUDA Toolkit (for `KokkosDNN_ENABLE_CUBLAS`), cuDNN (for `KokkosDNN_ENABLE_CUDNN`)
- Optional: ROCm (for `KokkosDNN_ENABLE_HIPBLAS` / `KokkosDNN_ENABLE_MIOPEN`)
- Optional: Apple Accelerate (for `KokkosDNN_ENABLE_ACCELERATE`, macOS only, no install needed)
- Supported platforms: Linux and macOS (tested in CI on `ubuntu-latest` / `ubuntu-24.04` and `macos-latest`). Windows is untested and there are no shared-library export macros, so `BUILD_SHARED_LIBS` targets Linux/macOS.

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
| `KokkosDNN_ENABLE_EXAMPLES` | ON | build `examples/` (`llama_generate`, `smollm2_generate`, `gemma4_generate`, `glimmer_generate`, `lfm2_5_generate`, `xor_train`, `dinov3_encode`, `sana_generate`, `moss_image_generate`, `kokoro_tts`, + example tests) |
| `KokkosDNN_ENABLE_BENCHMARKS` | OFF | build `benchmarks/kokkosdnn_bench` (Google Benchmark, fetched if needed) |
| `KokkosDNN_ENABLE_CUBLAS` | OFF | GEMM via cuBLAS on CUDA spaces (incl. fused GEMM+bias via cublasLt) |
| `KokkosDNN_ENABLE_CUDNN` | OFF | attention + convolutions via cuDNN (with automatic native fallback; needs `<cudnn.h>` + `cudnn-frontend`, fetched automatically) |
| `KokkosDNN_ENABLE_HIPBLAS` | OFF | GEMM via hipBLAS on HIP spaces |
| `KokkosDNN_ENABLE_MIOPEN` | OFF | convolutions via MIOpen on HIP spaces (attention path falls back to native; no fused SDPA in MIOpen) |
| `KokkosDNN_ENABLE_ACCELERATE` | OFF | GEMM/GEMV via Apple Accelerate `cblas_sgemm` on host spaces (float only; native fallback otherwise) |
| `KokkosDNN_ENABLE_TEST_DISCOVERY` | ON | register gtest tests via `gtest_discover_tests` (set OFF for GPU-less build-only CI, where test binaries cannot run at build time) |
| `KokkosDNN_ENABLE_IMAGE` | OFF | optional image component (`KokkosDNN::image`, stb-backed I/O + preprocessing) |
| `KokkosDNN_ENABLE_MODEL_LFM25` | ON | build LFM2.5 model component (`KokkosDNN::lfm25`) |
| `KokkosDNN_ENABLE_MODEL_GLIMMER` | ON | build Muse Glimmer 30B text model component (`KokkosDNN::glimmer`) |
| `KokkosDNN_ENABLE_MODEL_MOSS` | ON | build MOSS-Image model component (`KokkosDNN::moss`) |
| `KokkosDNN_ENABLE_SERVER` | OFF | build OpenAI-compatible HTTP server (`kokkosdnn_server`; requires `KokkosDNN_ENABLE_MODEL_LFM25=ON`) |
| `KokkosDNN_ENABLE_COMPILER_WARNINGS` | OFF | enable compiler warnings as errors (mirroring `Kokkos_ENABLE_COMPILER_WARNINGS`) |
| `BUILD_SHARED_LIBS` | OFF | build shared libraries (Linux/macOS only; Windows is untested) |

## Examples

- [llama](examples/llama/) — Llama 3.1 8B-Instruct (plus 3.2 1B) text inference with KV-cache autoregressive generation.
- [smollm2](examples/smollm2/) — SmolLM2-135M (base + Instruct) text inference with KV-cache autoregressive generation.
- [gemma4](examples/gemma4/) — Gemma 3 4B-E2B-IT text inference with KV-cache autoregressive generation.
- [lfm2_5](examples/lfm2_5/) — LFM2.5-24B-A2B hybrid-MoE text inference (40 blocks: 10 full-attention + 30 short-conv, 64-expert top-4 MoE, GGUF checkpoint + byte-level BPE `tokenizer.json` loaders, `--tiny` random model for weight-free runs).
- [xor](examples/xor/) — 2-layer MLP that learns the XOR truth table with full-batch Adam.
- [dinov3](examples/dinov3/) — DINOv3 ViT-B/16 image-encoder inference.
- [sana](examples/sana/) — Sana-0.6B (SanaMS_600M_P1_D28) text-to-image with FlowEuler + CFG (DiT weights load directly from the `.pth` checkpoint; native Gemma-2-2B-IT text encoder via `--text-weights` and native DC-AE f32c32 decoder via `--vae-safetensors`; without them conditioning falls back to the checkpoint null embedding unless `--prompt-embeds` is given, and output is raw latents plus a channel-mean grayscale preview).
- [moss_image](examples/moss_image/) — MOSS-Image-SFT text-to-image, fully in-process: in-repo BPE tokenizer + on-device Qwen3.5 encoder, FlowEuler + CFG DiT (`--staged` keeps the MiB embeds resident on-device while the 16.8 GB text encoder and 10.2 GB DiT are never resident together on a 24 GB GPU; `--prompt-embeds-in` loads precomputed file embeds, e.g. from `scripts/moss_encode.py`).
- [kokoro](examples/kokoro/) — Kokoro-82M text-to-speech (phoneme input): native StyleTTS2 + ISTFTNet inference with `--voice` packs, 24 kHz WAV out (weights load directly from the `kokoro-v1_0.pth` checkpoint via a dependency-free pickle reader).

See each directory's README for flags, run commands, and design notes.

## Server

`server/kokkosdnn_server` (opt-in via `-DKokkosDNN_ENABLE_SERVER=ON`) is an
OpenAI-compatible HTTP front-end for LFM2.5 and Glimmer
(`GET /v1/models`, `POST /v1/chat/completions` with SSE streaming,
`POST /v1/completions` alias; `--arch` selects the model family).
See [server](server/README.md) for flags, routes, and request params. The
`server/Dockerfile` CUDA image serves either arch with weights
volume-mounted at run time.

## Benchmarks

`benchmarks/kokkosdnn_bench` (Google Benchmark, opt-in via
`-DKokkosDNN_ENABLE_BENCHMARKS=ON`) times the core ops at representative
transformer shapes and reports `GFLOPS` (GEMM/GEMV/attention) or `GBps`
(elementwise) counters. GEMM registers `Native` vs `Dispatched`
variants for both `float` and `bhalf_t`: `Native` forces the Kokkos kernel
(`use_cublas/use_cudnn = false`), `Dispatched` uses the default handle, so
on a CUDA/HIP build with `KokkosDNN_ENABLE_CUBLAS`/`KokkosDNN_ENABLE_HIPBLAS` it exercises the TPL
path. GEMV (decode/lm_head shape) shares the same TPL GEMM dispatch, so it
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

- Sana scope: 28-block Linear DiT (FP32) + native Gemma-2-2B text
  encoder (`--text-weights`, gated HF download) + native DC-AE VAE
  decoder (`--vae-safetensors`, diffusers layout): `--prompt` alone
  still falls back to the checkpoint null embedding, and decode without
  the VAE weights is a channel-mean preview (plus raw `.f32` latents).
  Geometry is fixed at 1024px (32x32x32 latents; `--tiny` uses 4x4). A
  full-resolution run is hours on a Serial CPU build; use a CUDA/cuBLAS
  build for anything beyond smoke tests.
- LFM2.5 scope: hybrid MoE (40 blocks, 64-expert top-4, sigmoid gating) with
  GGUF loading (F32/F16/BF16, Q8_0, Q4_K, Q6_K; other dtypes throw), MoE
  experts dequantized once per routed expert per layer into persistent
  staging for the dense `gemm()` dispatch (chunk path; single-row decode
  keeps on-the-fly `quant_gemm`), dense-FP KV cache only, and no
  full-weights golden test (needs a real checkpoint + tokenizer).
- MOSS-Image scope: ~27 GB checkpoint (10.2 GB DiT + 16.8 GB Qwen3.5 text
  encoder + VAE); `--staged` keeps encoder and DiT resident one at a time
  for 24 GB GPUs. Serial-CPU full runs are slow and the text encoder needs
  ~32 GB host RAM there; without `--vae-weights --vae-decode` output is a
  channel-mean preview plus raw `.f32` latents. See
  [moss_image](examples/moss_image/) for the full pipeline, fit table, and
  reference-parity notes.
- Kokoro scope: FP32 only (`--fp32` accepted no-op); `--phonemes` is the
  primary input (KModel is language-blind, no bundled G2P; `--text` is a
  naive vocab fallback). SineGen uses its own host RNG stream, so audio
  differs from the torch reference by RNG (deterministic given `--seed`);
  durations, F0/N, and ASR features match the reference and are golden
  tested. Long paragraphs scale linearly (audio = 300·F₂ samples).

## License

Apache-2.0 WITH LLVM-exception (see [LICENSE](LICENSE)).
Every source file carries an `SPDX-License-Identifier` /
`SPDX-FileCopyrightText` header. The generated Unicode tables in
`models/moss_image/KokkosDNN_QwenTokenizerTables.inc` additionally fall
under the Unicode License v3 (attribution in `LICENSE`).
