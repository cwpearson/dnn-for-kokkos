# Llama example

Llama 3.1 8B-Instruct (plus 3.2 1B) inference built on the core library: safetensors checkpoint loader, faithful Hugging Face `tokenizer.json` loader, KV-cache autoregressive model, and a `llama_generate` driver.

Scope: **inference only** (prefill + sampling decode) with **bfloat16 storage/compute by default** (`--fp32` opts in). There is no weight quantization; the KV cache has an optional int8 mode (see Notes).

## Quickstart (0 to working)

From repo root `/workspace`: build, fetch the gated 8B-Instruct checkpoint, run it.

```bash
# 1. Build (Serial is enough; CUDA backend for GPU)
cmake -S . -B build \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8

# 2. Get weights + tokenizer (gated repo: request access, then log in)
pip install -U huggingface_hub
export HF_TOKEN=<token>  # or: hf auth login
hf download meta-llama/Llama-3.1-8B-Instruct \
  --include "tokenizer.json" \
  --include "*.safetensors" \
  --local-dir llama-3.1-8b \
  --token "$HF_TOKEN"

# 3. Run the real checkpoint
./build/examples/llama/llama_generate \
  --weights llama-3.1-8b/*.safetensors \
  --tokenizer llama-3.1-8b/tokenizer.json \
  --prompt "Explain Kokkos in one sentence." \
  --max-new 64
```

Expected tensor names are the Hugging Face ones (`model.embed_tokens.weight`, `model.layers.N.self_attn.q_proj.weight`, …, `lm_head.weight`).

## Example commands

Tiny random model (no downloads; also the unit-test end-to-end path):

```bash
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 8
```

Real 8B-Instruct checkpoint:

```bash
./build/examples/llama/llama_generate \
  --weights llama-3.1-8b/*.safetensors \
  --tokenizer llama-3.1-8b/tokenizer.json \
  --prompt "Explain Kokkos in one sentence." \
  --max-new 64
```

1B variant (`--model 3.2-1b`; aliases `llama-3.2-1b`, `1b`, case-insensitive; default stays `3.1-8b`). Passing `--model` opts out of the tiny default, so this runs the 1B-shaped model with random weights and no downloads (`--tiny` still forces `tiny_test` dims, so don't combine them):

```bash
./build/examples/llama/llama_generate --model 3.2-1b --prompt "Hello" --max-new 8
```

Real 1B checkpoint (gated repo, same login as above) with the faithful config fixture (`config-llama-3-2-1b.json`, a copy of the HF `config.json` values):

```bash
hf download meta-llama/Llama-3.2-1B \
  --include "tokenizer.json" \
  --include "*.safetensors" \
  --local-dir llama-3.2-1b \
  --token "$HF_TOKEN"
./build/examples/llama/llama_generate \
  --model 3.2-1b --config examples/llama/config-llama-3-2-1b.json \
  --weights llama-3.2-1b/model.safetensors \
  --tokenizer llama-3.2-1b/tokenizer.json \
  --prompt "Hello" --max-new 64
```

Sampling and chat flags (sampling defaults to Meta's recommended Llama 3.1 Instruct preset `temperature=0.6`, `top_p=0.9`; `--temperature 0` restores greedy; every sampling flag overrides the preset; seed defaults to `0` so runs repeat unless `--seed` is given):

```bash
# Greedy decode
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 8 --temperature 0
# Custom sampling
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 32 \
  --temperature 0.6 --top-p 0.9 --top-k 40 --min-p 0.05 --repetition-penalty 1.1 --seed 42
# Chat template + system prompt
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 16 --chat --system "Be concise."
# BOS / stop / prefill / context overrides
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 8 \
  --no-add-bos --stop-string "<|eot_id|>" --prefill-chunk 2048 --max-seq-len 8192
# KV-cache modes
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 8 --kv-page-size 16
./build/examples/llama/llama_generate --tiny --prompt "Hello" --max-new 8 --kv-int8
```

Full flag list lives at the top of `llama_generate.cpp` (`--tiny`, `--model`, `--weights`, `--tokenizer`, `--config`, `--prompt`, `--max-new`, `--stop`, `--fp32`, `--prefill-chunk`, `--max-seq-len`, `--kv-page-size`, `--kv-int8` / `--kv-fp`, `--chat`, `--system`, `--role`, `--messages`, `--add-bos` / `--no-add-bos`, `--stop-string`, `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty`, `--seed`).

## Notes

Layout: `KokkosDNN_LlamaConfig.hpp` (model `Config` + HF `config.json` overlay + `kv_cache_bytes()`), `KokkosDNN_LlamaModel.hpp` (`forward()` / `generate()` / chunked prefill), `KokkosDNN_LlamaWeights.{hpp,cpp}` (safetensors loader), `KokkosDNN_Tokenizer.{hpp,cpp}` (tokenizer + chat template helpers), `llama_generate.cpp` (driver), `config-llama-3-2-1b.json` (1B fixture), `Test_Llama*.cpp` / `Test_Safetensors.cpp` / `Test_Tokenizer.cpp` (tests), `generate_golden.py` (golden-data helper).

Stopping defaults to the Instruct set `{eos 128001, eom 128008, eot 128009}` (from `tokenizer.json` when `--tokenizer` is given, else `Config`); append extra ids with `--stop <id> ...`.

Context defaults to `max_seq_len = 8192` (PoC cap); the arch max `131072` is supported, including `llama3`-type rope scaling from the Llama 3.1 8B-Instruct `config.json`. `--config` overlays an HF `config.json` onto the selected base (present keys win); `--max-seq-len N` caps the KV-cache context (validated against `[1, 131072]`).

KV-cache modes (`--kv-page-size N`, `--kv-int8` / `--kv-fp`; config.json keys `kv_page_size`, `kv_int8` / `kv_quantized`, `kv_quant: "int8"|"none"`): dense FP by default; `--kv-page-size 16` pages the cache into blocks addressed through an explicit block table and grown on demand, so a short prompt on a long-context config only rents the pages it touches; `--kv-int8` stores symmetric int8 KV with per-token per-head FP32 scales (~2x fewer KV bytes plus a small scale table). The two combine (paged int8). Paged dense is bit-exact vs dense; int8 trades small numerical drift for the smaller footprint (see `Test_LlamaKVCache.cpp`).

1B notes: `tie_word_embeddings=true` means the checkpoint has no separate `lm_head.weight` — the loader aliases the embedding table instead; rope scaling is `llama3`-type with factor 32 (vs 8 for 3.1 8B); BF16 weights are ~2.5 GiB plus the KV cache (see `Config::kv_cache_bytes()`; the driver prints the full-window figure).

Memory: BF16 8B weights are ~16 GiB, so a full 8B BF16 run fits on a 24 GiB card such as an RTX 3090. FP32 (`--fp32`) needs ~32 GiB and does **not** fit there. Use the tiny config for local smoke tests and a large-GPU or CPU-Kokkos build for the real checkpoint. BF16 on-device conversion: on-disk F32/F16/BF16 safetensors values are converted to the model scalar on load.

Long-context design: the model passes KV-cache prefixes as subviews (no per-layer repack copy). The BF16 KV cache itself is ~256 MiB per layer (~8 GiB for 32 layers at 128k) in dense FP; `--kv-int8` roughly halves that (plus one FP32 scale per token per KV head), and paged KV (`--kv-page-size 16`) rents only touched pages instead of the full window upfront (see `Config::kv_cache_bytes()`; the driver prints the full-window figure).

Tokenizer: the full HF `tokenizers` pipeline for the Llama 3 `tokenizer.json` (special-token longest-match split, the tiktoken Split regex with Unicode `\p{L}`/`\p{N}`, GPT-2 byte mapping, per-piece BPE, byte-level decode that skips specials by default). Chat: `--chat` wraps `--prompt` as one `role` (default `"user"`) turn via the Llama 3.1 8B-Instruct template; `--system` prepends a system turn (implies chat); `--messages` wraps its value as the user turn (implies chat). BOS: `--add-bos` (default) ensures exactly one leading `<|begin_of_text|>`; `--no-add-bos` strips it; the chat path never double-counts the template's literal BOS. `--stop-string` truncates decoded generated text before the earliest match (needs `--tokenizer`). Prefill: `--prefill-chunk N` caps activations at O(N·dim) (`0` = single-shot); default is `Config::prefill_chunk_size` (2048).

## Verification

```bash
ctest --test-dir build -R Llama --output-on-failure
```

Tokenizer is validated ID-for-ID against Python `tokenizers` on the real file: `Test_Tokenizer.RealFileOracle` (runs when `KOKKOSDNN_TOKENIZER_JSON` points at a Llama 3 `tokenizer.json`; 92 targeted + 2000 fuzz differential cases pass). Re-validate a new file with:

```bash
KOKKOSDNN_TOKENIZER_JSON=tokenizer.json ./build/examples/llama/test_tokenizer
```

## Performance

Chunked prefill (`Config::prefill_chunk_size`, default 2048) keeps scratch at O(chunk·dim) regardless of context length: bit-exact vs single-shot with native kernels, within ~1e-7 FP32 with cuBLAS (whose kernels vary with M). Future work: CPU TPL path (no `OpenBLAS`/`MKL` dispatch); weight quantization is out of scope.
