# Glimmer example

Muse Glimmer 30B text-path inference built on the core library: 30B text config defaults, a safetensors checkpoint loader, a BPE `tokenizer.json` loader, a hybrid sliding/full KV-cache autoregressive model, and a `glimmer_generate` driver.

Scope: **text only** (`text_config`); vision tower and multimodal glue are out of scope. **Inference only** with **bfloat16 storage/compute by default** (`--fp32` opts in). There is no weight quantization; KV caches are dense FP. The driver encodes raw prompts directly.

> Status: **driver implemented**. `glimmer_generate` runs tiny random models (`--tiny`), full-size shapes with random weights (`--model glimmer-30b`), and real checkpoints (`--weights`, `--tokenizer`); `KokkosDNN_Glimmer{Config,Weights,Model,Tokenizer}.hpp` are functional and covered by `test_glimmer_generate` + `test_glimmer_tokenizer`. Tensor names and the chat template carry ASSUMPTION markers (see `DESIGN.md` and the headers); confirm them against `modeling_muse_glimmer.py` and the upstream chat template when published.

## Quickstart (0 to working)

```bash
# From the repo root. Needs CMake >= 3.22, C++20, installed Kokkos.

# 1. Build and install Kokkos once (CUDA, AMPERE86)
git clone --branch 5.2.1 --recursive https://github.com/kokkos/kokkos.git kokkos-src
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-install \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build kokkos-build --parallel 8
cmake --install kokkos-build

# 2. Configure and build the example targets (CUDA, cuBLAS + cuDNN)
cmake -S . -B build -DCMAKE_CXX_COMPILER=/path/to/kokkos-src/bin/nvcc_wrapper -DKokkos_ROOT=/path/to/kokkos-install -DCMAKE_BUILD_TYPE=Release -DKokkosDNN_ENABLE_CUBLAS=ON -DKokkosDNN_ENABLE_CUDNN=ON
cmake --build build --parallel 8 --target glimmer_generate test_glimmer_generate test_glimmer_tokenizer

# 3. Download the checkpoint with the `hf` CLI.
pip install -U "huggingface_hub[cli]"

hf download darkc0de/Muse-Glimmer-30B-heretic model.safetensors* tokenizer.json \
  tokenizer_config.json config.json generation_config.json \
  --local-dir weights/muse-glimmer-30b-text

# Real checkpoint run (~60 GiB weights).
./build/examples/glimmer/glimmer_generate \
  --model glimmer-30b --config examples/glimmer/config-glimmer-30b-text.json \
  --weights weights/muse-glimmer-30b-text/model.safetensors \
  --tokenizer weights/muse-glimmer-30b-text/tokenizer.json \
  --prompt "Hello" --max-new 64
```

`config-glimmer-30b-text.json` is the HF `text_config` object verbatim; the driver also accepts the full HF `config.json` (present keys win).

## Example commands

Tiny random model (no downloads, runs anywhere):

```bash
./build/examples/glimmer/glimmer_generate --tiny --prompt "Hello" --max-new 8
```

Full-size shape with random weights (no downloads; needs ~60 GiB RAM):

```bash
./build/examples/glimmer/glimmer_generate --model glimmer-30b --prompt "Hello" --max-new 8
```

Inspect what a (possibly still downloading) checkpoint file contains, without loading the model:

```bash
./build/examples/glimmer/glimmer_generate \
  --weights weights/muse-glimmer-30b-text/model.safetensors --list-tensors
```

FP32 comparison run:

```bash
./build/examples/glimmer/glimmer_generate \
  --model glimmer-30b --config examples/glimmer/config-glimmer-30b-text.json \
  --weights weights/muse-glimmer-30b-text/model.safetensors \
  --tokenizer weights/muse-glimmer-30b-text/tokenizer.json \
  --prompt "Hello" --max-new 64 --fp32
```

Useful flags: `--temperature 0` (greedy; the generation_config default is greedy), `--top-k/--top-p/--min-p/--repetition-penalty/--seed`, `--stop <id> ...`, `--stop-string`, `--add-bos/--no-add-bos`, `--prefill-chunk N` (`0` = single-shot), `--max-seq-len N` (validated against `[1, 131072]`), `--chat/--system/--role/--messages` (Muse-style template, assumption — see header), `--list-tensors`. `--weights` accepts glob patterns for sharded checkpoints.

## Notes

- Separate family from [llama](../llama/), [smollm2](../smollm2/), and [gemma4](../gemma4/): config, weights, model, and tokenizer will be split-form duplicates in namespace `KokkosDNN::Glimmer` (only the core `src/` ops are reused).
- Weights: single `model.safetensors`-style layout (sharded at 30B scale, ~60 GiB BF16); only the `model.language_model.*` tensors are loaded. With `tie_word_embeddings=false` the loader requires a separate `lm_head` tensor and must NOT alias the embedding table. See `DESIGN.md` for the authoritative math.
- Dtype: native bfloat16 (`Kokkos::Experimental::bhalf_t`, the checkpoint dtype) for weights/activations/KV; reductions, scales, and `inv_freq` stay FP32 — same convention as the Llama/SmolLM2/Gemma4 examples.
- Attention: QK RMSNorms plus the `qk_scale_factor` (3.87) score multiplier (NOT `1/sqrt(d)`); sliding layers use the sliding GQA kernel (window 2048, RoPE theta 500000); full layers use full GQA with NoPE (no rotary — `layer_rope_theta` 0).
- Head: `output_multiplier`-scaled hidden states, final RMSNorm, separate `lm_head` (`gemv` over it), `20·tanh(logits/20)` softcapping. Only library ops are reused; no new `src/` ops are expected for the text path except where the gap table says so. See `DESIGN.md` for the authoritative math.
- Memory: BF16 weights are ~60 GiB, so a full run needs a large-GPU or CPU-Kokkos build. The dense KV cache at the default `max_seq_len = 8192` is ~218 MiB BF16 per the `Config::kv_cache_bytes()` formula (52 layers × 8192 tokens × 2 K/V × 2 heads × 128 dim × 2 B); cap it with `--max-seq-len`.
- TPL dispatch: GEMM via cuBLAS and full attention via cuDNN SDPA on CUDA builds with `KokkosDNN_ENABLE_CUBLAS`/`KokkosDNN_ENABLE_CUDNN`, native kernels otherwise (the sliding-window path always runs the native kernel: the cuDNN SDPA graph bakes in a full causal mask).

## Verification

```bash
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure -R 'Glimmer'
./build/examples/glimmer/glimmer_generate --tiny --prompt "Hello" --max-new 8
```

Unit tests: `Test_GlimmerConfig.cpp` (presets, fixture, full-config wrapper, tiny preset), `Test_GlimmerGenerate.cpp` (tiny forward/generate smoke, determinism, chunked-prefill equivalence, NoPE/RoPE branches), and `Test_GlimmerTokenizer.cpp`.

## Performance

No dedicated benchmark; GEMM/attention shapes route through the shared library dispatch, so measure with the repo microbenchmarks (`benchmarks/kokkosdnn_bench`, opt-in via `-DKokkosDNN_ENABLE_BENCHMARKS=ON`). Prefill cost is bounded by `--prefill-chunk N` (O(N·dim) activations); decode is `gemv`-bound on the 202048-wide head.
