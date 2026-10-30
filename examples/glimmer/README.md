# Glimmer example

Muse Glimmer 30B text-path inference built on the core library: 30B text config defaults, a safetensors checkpoint loader, a BPE `tokenizer.json` loader, a hybrid sliding/full KV-cache autoregressive model, and a `glimmer_generate` driver.

Scope: **text only** (`text_config`); vision tower and multimodal glue are out of scope. **Inference only** with **bfloat16 storage/compute by default** (`--fp32` opts in). Safetensors weights load at full precision; a `.gguf` checkpoint keeps its packing on device (F32/F16/BF16 exact, Q8_0/Q4_K/Q5_K/Q6_K stay packed — the Q4_K_M recipe is ~17GB resident — with on-the-fly `quant_gemv` decode and dequant-once + dense GEMM prefill). KV caches are dense FP by default (`--kv-int8` selects the symmetric-int8 cache). The driver encodes raw prompts directly.

> Status: **driver implemented**. `glimmer_generate` runs tiny random models (`--tiny`), full-size shapes with random weights (`--model glimmer-30b`), and real checkpoints (`--weights`, `--tokenizer`); the model library (`KokkosDNN_Glimmer{Config,Weights,Model,Tokenizer,Session,PrefixCache}.hpp`, `KokkosDNN::glimmer` in `models/glimmer/`) is functional and covered by `test_glimmer_generate` + `test_glimmer_session` + `test_glimmer_tokenizer`. Tensor names carry ASSUMPTION markers (see `DESIGN.md` and the headers); confirm them against `modeling_muse_glimmer.py`. The chat template follows the upstream `chat_template.jinja` (ATEM tool protocol with `to=self` reasoning and `to=<tool>` tool channels); the `kokkosdnn_server --arch glimmer` front-end serves it with full OpenAI tools support.

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
# The full-precision weights live in the official Meta repo (gated:
# accept the model terms at https://huggingface.co/meta-models/Muse-Glimmer-30B
# then authenticate). Sharded at 30B scale: 2 shards, ~60 GiB BF16.
pip install -U "huggingface_hub[cli]"
hf auth login

hf download meta-models/Muse-Glimmer-30B \
  --include "model-*.safetensors" \
  --include "model.safetensors.index.json" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json" \
  --include "config.json" \
  --include "generation_config.json" \
  --local-dir weights/muse-glimmer-30b-text

# Q4_K_M GGUF (~16.8 GB, ~17GB resident) for the 24GB recipe below.
# The GGUF repo holds only .gguf files (no tokenizer.json); reuse the
# tokenizer.json from the safetensors download above.
hf download meta-models/Muse-Glimmer-30B-GGUF \
  --include "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf" \
  --local-dir weights/muse-glimmer-30b-text

# Real checkpoint run (~60 GiB weights).
./build/examples/glimmer/glimmer_generate \
  --model glimmer-30b --config models/glimmer/config-glimmer-30b-text.json \
  --weights "weights/muse-glimmer-30b-text/model-*.safetensors" \
  --tokenizer weights/muse-glimmer-30b-text/tokenizer.json \
  --prompt "Hello" --max-new 64
```

`config-glimmer-30b-text.json` is the `text_config` object of the official
`meta-models/Muse-Glimmer-30B` `config.json` verbatim; the driver also accepts the full HF `config.json` (present keys win).

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
  --weights "weights/muse-glimmer-30b-text/model-*.safetensors" --list-tensors
```

FP32 comparison run:

```bash
./build/examples/glimmer/glimmer_generate \
  --model glimmer-30b --config models/glimmer/config-glimmer-30b-text.json \
  --weights "weights/muse-glimmer-30b-text/model-*.safetensors" \
  --tokenizer weights/muse-glimmer-30b-text/tokenizer.json \
  --prompt "Hello" --max-new 64 --fp32
```

GGUF checkpoint run (llama.cpp naming, packed quant stays on device):

```bash
./build/examples/glimmer/glimmer_generate \
  --model glimmer-30b --config models/glimmer/config-glimmer-30b-text.json \
  --weights weights/muse-glimmer-30b-text/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --tokenizer weights/muse-glimmer-30b-text/tokenizer.json \
  --prompt "Hello" --max-new 64
```

A single `--weights` entry ending in `.gguf` (case-insensitive) selects the GGUF path (`KokkosDNN::Glimmer::load_gguf`); F32/F16/BF16 decode exactly, Q8_0/Q4_K/Q5_K/Q6_K (i.e. the Q4_K_M recipe, which stores those per-tensor dtypes) stay packed on device with on-the-fly dequant (decode) / dequant-once + dense GEMM (prefill chunks), anything else throws. Mixing `.gguf` with safetensors shards is an error. `--list-tensors` works on `.gguf` files too.

24GB recipe (packed GGUF + int8 KV + chunk cap): `--kv-int8` halves the KV cache (214.5 MiB at the 8192 default, 3432 MiB at the 131072 arch max vs 416 MiB / ~6.5 GiB dense) and `--prefill-chunk N` caps prefill scratch (~440 MiB at the 2048 default, ~110 MiB at 512). ~17GB Q4_K_M weights + 3432 MiB int8 KV + ~614 MiB dequant staging + scratch ≈ 21.4 GiB, fitting 24GB:

```bash
./build/examples/glimmer/glimmer_generate \
  --model glimmer-30b --config models/glimmer/config-glimmer-30b-text.json \
  --weights weights/muse-glimmer-30b-text/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --tokenizer weights/muse-glimmer-30b-text/tokenizer.json \
  --prompt "Hello" --max-new 64 --kv-int8 --max-seq-len 131072 --prefill-chunk 512
```

The same flags exist on `kokkosdnn_server --arch glimmer` (see `server/README.md`); the one-entry prefix cache works in both dense and int8 modes for agentic loops.

GGUF conversion-trap handling (verified against llama.cpp `conversion/muse_glimmer.py` + `src/models/muse-glimmer.cpp`):

- Norm shift: conversion adds `+1` to the four layer norms (`norm_shift`, since GGUF `rms_norm` is `x/rms*w` while the HF CenteredRMSNorm is `x/rms*(1+w)`); the final norm is untouched (shift 0). The loader subtracts the shift back so the Model's `one_plus` path sees HF-centered weights.
- Q/K RoPE layout: conversion unpermutes q/k into ggml interleaved (NORM) layout; the loader re-permutes rows into the HF permuted (rotate_half) layout consumed by `rope_neoX_inplace` (tested: loader output equals the pre-conversion layout exactly).
- Synthesized QK norms: GGUF `attn_q_norm` (full `qk_scale_factor` 3.87) / `attn_k_norm` (ones) absorb the q pre-scale into a *scaled* RMSNorm, while the Model runs a *scaleless* QK norm plus the separate 3.87 multiply — mathematically identical (`rms(x)*3.87`, `rms(x)*1`, same eps), so the loader ignores both tensors (files with or without them load identically).
- Metadata: `muse-glimmer.{logit_scale, final_logit_softcapping, attention.sliding_window, attention.sliding_window_pattern}` are read and validated against `Config` (throw on missing/mismatch), never silently overriding it.

Useful flags: `--temperature 0` (greedy; the driver defaults to greedy while the official `generation_config.json` defaults to sampling: `do_sample=true`, temperature 1.0, `top_p` 0.95, `top_k` 64), `--top-k/--top-p/--min-p/--repetition-penalty/--seed`, `--stop <id> ...`, `--stop-string`, `--add-bos/--no-add-bos`, `--prefill-chunk N` (`0` = single-shot), `--max-seq-len N` (validated against `[1, 131072]`), `--kv-int8 | --kv-fp | --kv-quant int8|none` (KV precision; config-file keys `kv_int8`/`kv_quantized`/`kv_quant` agree, last explicit CLI source wins), `--chat/--system/--role/--messages` (Muse-style template, assumption — see header), `--list-tensors`. `--weights` accepts glob patterns for sharded checkpoints, or a single `.gguf` file for a llama.cpp GGUF checkpoint (see above).

## Notes

- Separate family from [llama](../llama/), [smollm2](../smollm2/), and [gemma4](../gemma4/): the model library (config, weights, model, tokenizer, session) lives in `models/glimmer/` as `KokkosDNN::glimmer` in namespace `KokkosDNN::Glimmer` (only the core `src/` ops are shared).
- Weights: `model-*-of-*.safetensors` sharded layout (2 shards at 30B scale, ~60 GiB BF16); only the `model.language_model.*` tensors are loaded. With `tie_word_embeddings=false` the loader requires a separate `lm_head` tensor and must NOT alias the embedding table. See `DESIGN.md` for the authoritative math.
- Dtype: native bfloat16 (`Kokkos::Experimental::bhalf_t`, the checkpoint dtype) for weights/activations/KV; reductions, scales, and `inv_freq` stay FP32 — same convention as the Llama/SmolLM2/Gemma4 examples.
- Attention: QK RMSNorms plus the `qk_scale_factor` (3.87) score multiplier (NOT `1/sqrt(d)`); sliding layers use the sliding GQA kernel (window 2048, RoPE theta 500000); full layers use full GQA with NoPE (no rotary — `layer_rope_theta` 0).
- Head: `output_multiplier`-scaled hidden states, final RMSNorm, separate `lm_head` (`gemv` over it), `20·tanh(logits/20)` softcapping. Only library ops are reused; no new `src/` ops are expected for the text path except where the gap table says so. See `DESIGN.md` for the authoritative math.
- Memory: BF16 weights are ~60 GiB, so a full run needs a large-GPU or CPU-Kokkos build. The dense KV cache at the default `max_seq_len = 8192` is 416 MiB BF16 per the `Config::kv_cache_bytes()` formula (52 layers × 8192 tokens × 2 K/V × 2 heads × 128 dim × 2 B); cap it with `--max-seq-len`, or halve it with `--kv-int8` (214.5 MiB at 8192; 3432 MiB at the 131072 arch max vs ~6.5 GiB dense). A Q4_K_M `.gguf` checkpoint is ~17GB resident, so packed GGUF + `--kv-int8` + `--prefill-chunk` fits 24GB GPUs.
- TPL dispatch: GEMM via cuBLAS and full attention via cuDNN SDPA on CUDA builds with `KokkosDNN_ENABLE_CUBLAS`/`KokkosDNN_ENABLE_CUDNN`, native kernels otherwise (the sliding-window path always runs the native kernel: the cuDNN SDPA graph bakes in a full causal mask).

## Verification

```bash
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure -R 'Glimmer'
./build/examples/glimmer/glimmer_generate --tiny --prompt "Hello" --max-new 8
```

Unit tests: `Test_GlimmerConfig.cpp` (presets, fixture, full-config wrapper, tiny preset), `Test_GlimmerGenerate.cpp` (tiny forward/generate smoke, determinism, chunked-prefill equivalence, NoPE/RoPE branches), `Test_GlimmerSession.cpp` (prefill/step termination, stop-id policy, reseed determinism, prefix-cache equivalence, `ISession` front-end), `Test_GlimmerWeightsGGUF.cpp` (synthetic-GGUF loader roundtrip: norm shift, Q/K re-permute, synthesized-norm handling, mixed dtypes, metadata validation), and `Test_GlimmerTokenizer.cpp` (fixture BPE/pre-tokenizer, ATEM chat template render/parse, channel stream parser).

## Performance

No dedicated benchmark; GEMM/attention shapes route through the shared library dispatch, so measure with the repo microbenchmarks (`benchmarks/kokkosdnn_bench`, opt-in via `-DKokkosDNN_ENABLE_BENCHMARKS=ON`). Prefill cost is bounded by `--prefill-chunk N` (O(N·dim) activations); decode is `gemv`-bound on the 202048-wide head.
