# Llama example

Llama 3.1 8B-Instruct inference built on the core library: 3.1 8B-Instruct config defaults, a safetensors checkpoint loader (on-disk F32/F16/BF16 are converted to the model scalar), a faithful Hugging Face `tokenizer.json` loader (special-token split, tiktoken Split pre-tokenizer, GPT-2 byte mapping, per-piece BPE; validated ID-for-ID against `tokenizers`), a KV-cache autoregressive model, and a `llama_generate` driver.

Scope: the Llama path is **inference only** (prefill + greedy decode by default, sampling opt-in, see below) with **bfloat16 storage/compute by default** (`--fp32` opts in). There is no weight quantization; the KV cache has an optional int8 mode (see below).

Layout:

- `KokkosDNN_LlamaConfig.hpp` — model `Config` (dims, RoPE, context, KV-cache layout, `tie_word_embeddings`, `hidden_act`/bias checks) plus HF `config.json` overlay and `kv_cache_bytes()`.
- `KokkosDNN_LlamaModel.hpp` — KV-cache autoregressive model (`forward()`, `generate()`, chunked prefill).
- `KokkosDNN_LlamaWeights.{hpp,cpp}` — safetensors loader.
- `KokkosDNN_Tokenizer.{hpp,cpp}` — `tokenizer.json` loader/encoder/decoder plus chat template helpers.
- `llama_generate.cpp` — `llama_generate` driver.
- `config-llama-3-2-1b.json` — faithful 1B config fixture (copy of the HF `config.json` values).
- `Test_Llama*.cpp`, `Test_Safetensors.cpp`, `Test_Tokenizer.cpp` — example tests.
- `generate_golden.py` — golden-data helper.

## Run

Tiny random model (no downloads, runs anywhere; also the unit-test end-to-end path so the full pipeline runs without downloading a 16 GB checkpoint):

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

Sampling defaults to Meta's recommended Llama 3.1 Instruct settings
(`temperature=0.6`, `top_p=0.9`); `--temperature 0` restores greedy decode.
Every sampling flag (`--temperature`, `--top-k`, `--top-p`, `--min-p`,
`--repetition-penalty`, `--seed`) overrides the preset; seed defaults to 0
so runs repeat unless `--seed` is given. Other flags: `--chat` / `--system`
/ `--role` / `--messages` (chat template), `--add-bos` / `--no-add-bos`
(BOS handling), `--stop-string` (decoded-text stop), `--prefill-chunk N`
(O(N·dim) prefill activations, `0` = single-shot), `--config` (HF
`config.json` overlay, present keys win), `--max-seq-len N` (KV-cache
context cap, validated against `[1, 131072]`).

### Llama 3.2 1B

The driver also supports the 1B base via `--model 3.2-1b` (aliases
`llama-3.2-1b`, `1b`; case-insensitive; the default stays `3.1-8b`).
Passing `--model` opts out of the tiny default, so this runs the 1B-shaped
model with random weights and no downloads:

```bash
./build/examples/llama/llama_generate --model 3.2-1b --prompt "Hello" --max-new 8
```

(`--tiny` still forces `tiny_test` dims, so don't combine it with
`--model` when you want the 1B shape.)

With a real `meta-llama/Llama-3.2-1B` checkpoint, add the weights,
tokenizer, and the faithful config fixture
(`config-llama-3-2-1b.json`, a copy of the HF
`config.json` values that also documents the 1B hyperparameters):

```bash
./build/examples/llama/llama_generate \
  --model 3.2-1b --config examples/llama/config-llama-3-2-1b.json \
  --weights model.safetensors --tokenizer tokenizer.json \
  --prompt "Hello" --max-new 64
```

Notes: `tie_word_embeddings=true` means the checkpoint has no separate
`lm_head.weight` — the loader aliases the embedding table instead; rope
scaling is `llama3`-type with factor 32 (vs 8 for 3.1 8B); BF16 weights
are ~2.5 GiB plus the KV cache (see `Config::kv_cache_bytes()`; the
driver prints the full-window figure).

## Memory note

BF16 8B weights are ~16 GiB, so a full 8B BF16 run fits on a 24 GiB card
such as an RTX 3090. FP32 (`llama_generate --fp32`) needs ~32 GiB and does
**not** fit there. Use the tiny config for local smoke tests and a
large-GPU or CPU-Kokkos build for the real checkpoint.

## Long-context design

The model passes KV-cache prefixes as subviews (no per-layer repack copy).
Remaining 128k considerations: the BF16 KV cache itself is ~256 MiB per
layer (~8 GiB for 32 layers at 128k) in dense FP; `--kv-int8` roughly
halves that (plus one FP32 scale per token per KV head), and paged KV
(`--kv-page-size 16`) rents only touched pages instead of the full
window upfront (see `Config::kv_cache_bytes()`; the driver prints the
full-window figure). Prefill activations are bounded by
chunked prefill (`Config::prefill_chunk_size`, default 2048):
`forward()` processes long prompts in chunks, so scratch stays
O(chunk·dim) regardless of context length (bit-exact vs single-shot
with native kernels; within ~1e-7 FP32 with cuBLAS, whose kernels vary
with M).

## Tokenizer

The tokenizer implements the full HF `tokenizers` pipeline for the Llama 3
`tokenizer.json` (special-token longest-match split, the tiktoken Split
regex with Unicode `\p{L}`/`\p{N}`, GPT-2 byte mapping, per-piece BPE,
byte-level decode that skips specials by default). It is validated ID-for-ID
against Python `tokenizers` on the real file: `Test_Tokenizer.RealFileOracle`
(runs when `KOKKOSDNN_TOKENIZER_JSON` points at a Llama 3 `tokenizer.json`;
92 targeted + 2000 fuzz differential cases pass). Re-validate a new file with
`KOKKOSDNN_TOKENIZER_JSON=tokenizer.json ./build/examples/llama/test_tokenizer`.
