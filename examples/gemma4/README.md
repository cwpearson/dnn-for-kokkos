# Gemma4 example

Gemma4 E2B-it text-path inference built on the core library: E2B-it config defaults, a safetensors checkpoint loader, a SentencePiece-▁ BPE `tokenizer.json` loader, a hybrid sliding/full KV-cache autoregressive model with KV sharing and per-layer-embedding (PLE), and a `gemma4_generate` driver.

Scope: **text only** (`text_config`); vision/audio towers and MoE blocks are out of scope. **Inference only** with **bfloat16 storage/compute by default** (`--fp32` opts in). There is no weight quantization; KV caches are dense FP. No `--chat` flag: the driver encodes raw prompts directly.

## Quickstart (0 to working)

```bash
# From repo root (/workspace). Needs CMake >= 3.22, C++20, installed Kokkos.

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

# 2. Configure and build the example target (CUDA, cuBLAS + cuDNN)
cmake -S . -B build -DCMAKE_CXX_COMPILER=/path/to/kokkos-src/bin/nvcc_wrapper -DKokkos_ROOT=/path/to/kokkos-install -DCMAKE_BUILD_TYPE=Release -DKokkosDNN_ENABLE_CUBLAS=ON -DKokkosDNN_ENABLE_CUDNN=ON
cmake --build build --parallel 8 --target gemma4_generate

# 3. Download the checkpoint with the `hf` CLI.
# Gated repo: request access at https://huggingface.co/google/gemma-4-E2B-it,
# then authenticate (needs read access to the repo).
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=<your-token>   # or: hf auth login

hf download google/gemma-4-E2B-it model.safetensors tokenizer.json \
  tokenizer_config.json config.json generation_config.json \
  --local-dir weights/gemma-4-E2B-it \
  --token "$HF_TOKEN"

# Real checkpoint run (~10 GiB weights).
./build/examples/gemma4/gemma4_generate \
  --model gemma4-e2b-it --config examples/gemma4/config-gemma4-e2b-it.json \
  --weights weights/gemma-4-E2B-it/model.safetensors \
  --tokenizer weights/gemma-4-E2B-it/tokenizer.json \
  --prompt "Hello" --max-new 64
```

`config-gemma4-e2b-it.json` is the HF `text_config` object verbatim; the driver also accepts the full HF `config.json` (present keys win).

## Example commands

Tiny random model (no downloads, runs anywhere):

```bash
./build/examples/gemma4/gemma4_generate --tiny --prompt "Hello" --max-new 8
```

Full-size shape with random weights (no downloads; needs ~10 GiB RAM):

```bash
./build/examples/gemma4/gemma4_generate --model gemma4-e2b-it --prompt "Hello" --max-new 8
```

Inspect what a (possibly still downloading) checkpoint file contains, without loading the model:

```bash
./build/examples/gemma4/gemma4_generate \
  --weights weights/gemma-4-E2B-it/model.safetensors --list-tensors
```

FP32 comparison run:

```bash
./build/examples/gemma4/gemma4_generate \
  --model gemma4-e2b-it --config examples/gemma4/config-gemma4-e2b-it.json \
  --weights weights/gemma-4-E2B-it/model.safetensors \
  --tokenizer weights/gemma-4-E2B-it/tokenizer.json \
  --prompt "Hello" --max-new 64 --fp32
```

Useful flags: `--temperature 0` (greedy; default is upstream `generation_config.json`: `temperature=1.0`, `top_k=64`, `top_p=0.95`), `--top-k/--top-p/--min-p/--repetition-penalty/--seed` (seed defaults to 0), `--stop <id> ...` (default stop set `{1 (<eos>), 106, 50}`), `--stop-string`, `--add-bos/--no-add-bos`, `--prefill-chunk N` (`0` = single-shot), `--max-seq-len N` (validated against `[1, 131072]`).

## Notes

- Separate family from [llama](../llama/) and [smollm2](../smollm2/): config, weights, model, and tokenizer are split-form duplicates in namespace `KokkosDNN::Gemma4` (only the core `src/` ops are reused).
- Weights: single-file layout keeps all towers in one `model.safetensors` (~10 GiB BF16); only the `model.language_model.*` tensors are loaded (600 of 2011; audio/vision tensors are never requested). Expected names are the HF ones (`model.language_model.embed_tokens.weight`, `model.language_model.layers.N.self_attn.q_proj.weight`, …). With `tie_word_embeddings=true` the loader aliases the embedding table since no `lm_head` tensor exists. If shards are incomplete, `load_safetensors` throws naming the first missing required tensor.
- Dtype: native bfloat16 (`Kokkos::Experimental::bhalf_t`, the checkpoint dtype) for weights/activations/KV; reductions, scales, `inv_freq`, and `layer_scalar` stay FP32 — same convention as Llama/SmolLM2.
- KV sharing: layers 15–34 (the trailing `num_kv_shared_layers = 20`) hold no KV of their own: sliding layers reuse the L13 cache, full layers the L14 cache. Donor caches keep the full prefix; the sliding-window mask (512) applies at attention time. Dead k/v/k_norm tensors for shared layers are skipped by the loader.
- PLE (per-layer embedding): token part `sqrt(256)·E_ple[tok]` plus context part `RMSNorm(x_main·W_ctx/√1536)`, mixed as `(ctx + tok)/√2` once per chunk, then per layer `RMSNorm(gelu_tanh(x·Wgate)·p_l·Wproj)` added to the residual.
- Double-wide MLP: layers ≥ 15 use 12288-wide gated-GeLU (`GeluVariant::Tanh`); earlier layers 6144.
- Attention: scale 1.0 (QK RMSNorms replace `1/√d`); sliding layers use the sliding GQA kernel (window 512, RoPE theta 10k); full layers use full GQA with the proportional partial-RoPE table (theta 1e6, factor 0.25) via the unchanged NeoX kernel.
- Head: `sqrt(1536)`-scaled embedding, final RMSNorm, tied `lm_head` (`gemv` over the embedding table), `30·tanh(logits/30)` softcapping. Only library ops from the Gemma4 step plus pre-existing ops are used; no new `src/` ops. See `DESIGN.md` for the authoritative math.
- Memory: BF16 weights are ~10 GiB, so a full run needs a large-GPU or CPU-Kokkos build (a 24 GiB card fits weights but leaves little headroom with the full-window KV cache; cap it with `--max-seq-len`). FP32 needs ~20 GiB. The dense KV cache at the default `max_seq_len = 8192` is ~75 MiB BF16; see `Config::kv_cache_bytes()` (the driver prints the full-window figure).
- TPL dispatch: GEMM via cuBLAS and full attention via cuDNN SDPA on CUDA builds with `KokkosDNN_ENABLE_CUBLAS`/`KokkosDNN_ENABLE_CUDNN`, native kernels otherwise (the sliding-window path always runs the native kernel: the cuDNN SDPA graph bakes in a full causal mask). No new TPL kernels were needed.

## Verification

```bash
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure -R 'Gemma4'
./build/examples/gemma4/gemma4_generate --tiny --prompt "Hello" --max-new 8
```

The tiny run is also the unit-test end-to-end path (`Test_Gemma4Generate.cpp`: presets, fixture, tiny forward/generate smoke, gated real-weight check).

## Performance

No dedicated benchmark; GEMM/attention shapes route through the shared library dispatch, so measure with the repo microbenchmarks (`benchmarks/kokkosdnn_bench`, opt-in via `-DKokkosDNN_ENABLE_BENCHMARKS=ON`). Prefill cost is bounded by `--prefill-chunk N` (O(N·dim) activations); decode is `gemv`-bound on the 262144-wide tied head.
