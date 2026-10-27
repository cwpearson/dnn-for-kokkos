# SmolLM2 example

SmolLM2-135M (base + Instruct) text inference built on the core library: 135M config defaults, a safetensors checkpoint loader, a GPT-2-style BPE `tokenizer.json` loader, a KV-cache autoregressive model, and a `smollm2_generate` driver.

## Quickstart (0 to working)

From the repo root (`/workspace`); needs CMake ≥ 3.22, a C++20 compiler, and an installed Kokkos. The checkpoint is public (~269 MB), no access request needed.

```bash
# 1. Build and install Kokkos once (Serial shown; for GPU add
#    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON and set
#    CMAKE_CXX_COMPILER to kokkos-src/bin/nvcc_wrapper)
git clone --branch 5.2.1 --recursive https://github.com/kokkos/kokkos.git kokkos-src
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-install \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build kokkos-build --parallel 8
cmake --install kokkos-build

# 2. Configure and build the example target
cmake -S . -B build \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8 --target smollm2_generate

# 3. Download the checkpoint with the `hf` CLI
pip install -U "huggingface_hub[cli]"
hf download HuggingFaceTB/SmolLM2-135M model.safetensors tokenizer.json \
  --local-dir weights/smollm2-135m
./build/examples/smollm2/smollm2_generate \
  --model smollm2-135m --config examples/smollm2/config-smollm2-135m.json \
  --weights weights/smollm2-135m/model.safetensors \
  --tokenizer weights/smollm2-135m/tokenizer.json \
  --prompt "Hello" --max-new 64
```

For the Instruct twin, replace the repo ID with `HuggingFaceTB/SmolLM2-135M-Instruct` (local dir `weights/smollm2-135m-instruct`, config `examples/smollm2/config-smollm2-135m-instruct.json`, `--model smollm2-135m-instruct --chat`).

## Example commands

Random-weights smoke (no downloads, runs anywhere; also the unit-test end-to-end path):

```bash
./build/examples/smollm2/smollm2_generate --model smollm2-135m --prompt "Hello" --max-new 8
./build/examples/smollm2/smollm2_generate --model smollm2-135m-instruct --chat --prompt "Hello" --max-new 8
```

Real base checkpoint (files from the Quickstart download):

```bash
./build/examples/smollm2/smollm2_generate \
  --model smollm2-135m --config examples/smollm2/config-smollm2-135m.json \
  --weights weights/smollm2-135m/model.safetensors \
  --tokenizer weights/smollm2-135m/tokenizer.json \
  --prompt "Hello" --max-new 64
```

Real Instruct checkpoint with chat template:

```bash
hf download HuggingFaceTB/SmolLM2-135M-Instruct model.safetensors tokenizer.json \
  --local-dir weights/smollm2-135m-instruct
./build/examples/smollm2/smollm2_generate \
  --model smollm2-135m-instruct --config examples/smollm2/config-smollm2-135m-instruct.json \
  --weights weights/smollm2-135m-instruct/model.safetensors \
  --tokenizer weights/smollm2-135m-instruct/tokenizer.json \
  --chat --prompt "Hello" --max-new 64
```

Deterministic greedy decode (e.g. for CI):

```bash
./build/examples/smollm2/smollm2_generate \
  --model smollm2-135m --config examples/smollm2/config-smollm2-135m.json \
  --weights weights/smollm2-135m/model.safetensors \
  --tokenizer weights/smollm2-135m/tokenizer.json \
  --prompt "Hello" --max-new 32 --temperature 0
```

Other flags: `--fp32`, `--system "..."` (implies chat), `--role`, `--messages`, `--add-bos` / `--no-add-bos`, `--stop <id> ...`, `--stop-string "..."`, `--prefill-chunk N`, `--max-seq-len N`, `--kv-page-size N`, `--kv-int8` / `--kv-fp`, `--temperature` / `--top-k` / `--top-p` / `--min-p` / `--repetition-penalty` / `--seed`.

## Notes

- Scope: **inference only** with **bfloat16 storage/compute by default** (`--fp32` opts in). There is no weight quantization; the KV cache has an optional int8 mode (see below). On-disk F32/F16/BF16 weights are converted to the model scalar.
- Layout: `KokkosDNN_SmolLM2Config.hpp` (model `Config` + HF `config.json` overlay + `kv_cache_bytes()`), `KokkosDNN_SmolLM2Model.hpp` (`forward()` / `generate()`, chunked prefill), `KokkosDNN_SmolLM2Weights.{hpp,cpp}` (safetensors loader), `KokkosDNN_SmolLM2Tokenizer.{hpp,cpp}` (`tokenizer.json` loader/encoder/decoder + ChatML helpers), `smollm2_generate.cpp` (driver), `config-smollm2-135m{,-instruct}.json` (faithful fixtures, copies of the HF `config.json` values). Separate family from [llama](../llama/): split-form duplicates in namespace `KokkosDNN::SmolLM2`, only the core `src/` ops are reused.
- Arch (both variants): dim 576, 30 layers, 9 heads, 3 KV heads, 1536 FFN, 49152 vocab, `rms_norm_eps=1e-5`, `rope_theta=100000`, `max_position_embeddings=8192`, `tie_word_embeddings=true`, `hidden_act=silu`, no attention/MLP bias, plain NeoX RoPE (checkpoints ship `rope_scaling=null`).
- `--model` selects `smollm2-135m` (default; aliases `smollm2`, `135m`) or `smollm2-135m-instruct` (alias `smollm2-instruct`; case-insensitive, optional `huggingfacetb-` prefix). Passing `--model` explicitly opts out of the tiny default (full-size shape, random weights, no downloads); `--tiny` still forces `tiny_test` dims. Llama model names are not accepted here (use the llama driver).
- Expected tensor names are the Hugging Face ones (`model.embed_tokens.weight`, `model.layers.N.self_attn.q_proj.weight`, …, `lm_head.weight`); with `tie_word_embeddings=true` the loader aliases the embedding table when no separate `lm_head.weight` is present.
- Tokenizer: GPT-2-style BPE over `tokenizer.json` (Digits + ByteLevel pre-tokenizer, GPT-2 byte mapping, per-piece BPE) plus ChatML template helpers.
- Stopping defaults to the single GPT-2 EOS id (base `0`, Instruct `2`; from `tokenizer.json` when `--tokenizer` is given, else `Config`); append extra ids with `--stop <id> ...`, decoded-text stops with repeatable `--stop-string`.
- BOS: `--add-bos` by default ensures exactly one leading BOS id (tokenizer BOS id, else Config BOS: `<|endoftext|>`/0 base, `<|im_start|>`/1 Instruct); `--no-add-bos` strips it; the chat path never double-counts the template's literal BOS.
- Chat: `--chat` wraps `--prompt` as one turn via the SmolLM2-Instruct ChatML template (`<|im_start|>`/`<|im_end|>` blocks + open assistant header, default-system injection when no system turn is present); `--system` prepends a system turn (implies chat).
- Sampling defaults to `temperature=0.6`, `top_p=0.9`; `--temperature 0` restores greedy decode. Every sampling flag overrides the preset; seed defaults to 0 so runs repeat unless `--seed` is given.
- `--config` overlays an HF `config.json` onto the selected base (present keys win); `--max-seq-len` caps the KV-cache context (validated against `[1, 8192]`, the arch max).
- KV-cache modes (`--kv-page-size N`, `--kv-int8` / `--kv-fp`): dense FP by default; `--kv-page-size 16` pages the cache into blocks addressed through a block table and grown on demand; `--kv-int8` stores symmetric int8 KV with per-token per-head FP32 scales. Paged dense is bit-exact vs dense; int8 trades small numerical drift for ~2x fewer KV bytes.
- `--prefill-chunk N` caps prefill activations to O(N·dim) (`0` = single-shot); default is `Config::prefill_chunk_size` (2048).

## Verification

```bash
ctest --test-dir build -R SmolLM2 --output-on-failure
```

- Serial CI (`.github/workflows/smollm2.yaml`) downloads `model.safetensors` + `tokenizer.json` from `HuggingFaceTB/SmolLM2-135M` (~269 MB, cached) and runs the driver with the checked-in config fixture, greedy (`--temperature 0`), `--max-new 32`.
- Example tests: `Test_SmolLM2Generate.cpp`, `Test_SmolLM2Tokenizer.cpp` (built with `KokkosDNN_ENABLE_TESTS=ON`).

## Performance

- BF16 135M weights are ~0.27 GiB, so the full preset runs anywhere (CPU Serial included); greedy `--max-new 32` on single-core Serial takes tens of seconds.
- SmolLM2 shapes route through the same dispatch as the rest of the library — GEMM via cuBLAS (`sgemm`/`cublasGemmEx`) and attention via cuDNN SDPA on CUDA builds with `KokkosDNN_ENABLE_CUBLAS`/`KokkosDNN_ENABLE_CUDNN`, native kernels otherwise. No new TPL kernels were needed (shapes/dtypes already covered); verified by code inspection plus the Serial build, not GPU execution.
