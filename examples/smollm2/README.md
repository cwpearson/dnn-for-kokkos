# SmolLM2 example

SmolLM2-135M (base + Instruct) inference built on the core library: 135M config defaults, a safetensors checkpoint loader (on-disk F32/F16/BF16 are converted to the model scalar), a GPT-2-style BPE `tokenizer.json` loader (Digits + ByteLevel pre-tokenizer, GPT-2 byte mapping, per-piece BPE), a KV-cache autoregressive model, and a `smollm2_generate` driver.

Separate family from [llama](../llama/): config, weights, model, and tokenizer are split-form duplicates in namespace `KokkosDNN::SmolLM2` (no shared headers with `examples/llama/` — only the core `src/` ops are reused).

Scope: the SmolLM2 path is **inference only** (prefill + greedy decode by default, sampling opt-in, see below) with **bfloat16 storage/compute by default** (`--fp32` opts in). There is no weight quantization; the KV cache has an optional int8 mode (see below).

Layout:

- `KokkosDNN_SmolLM2Config.hpp` — model `Config` (dims, RoPE, context, KV-cache layout, `tie_word_embeddings`, `hidden_act`/bias checks) plus HF `config.json` overlay and `kv_cache_bytes()`.
- `KokkosDNN_SmolLM2Model.hpp` — KV-cache autoregressive model (`forward()`, `generate()`, chunked prefill).
- `KokkosDNN_SmolLM2Weights.{hpp,cpp}` — safetensors loader.
- `KokkosDNN_SmolLM2Tokenizer.{hpp,cpp}` — GPT-2-style `tokenizer.json` loader/encoder/decoder plus ChatML template helpers.
- `smollm2_generate.cpp` — `smollm2_generate` driver.
- `config-smollm2-135m.json`, `config-smollm2-135m-instruct.json` — faithful config fixtures (copies of the HF `config.json` values).
- `Test_SmolLM2Generate.cpp`, `Test_SmolLM2Tokenizer.cpp` — example tests.

## Run

Random-weights smoke (no downloads, runs anywhere; also the unit-test end-to-end path):

```bash
./build/examples/smollm2/smollm2_generate --model smollm2-135m --prompt "Hello" --max-new 8
./build/examples/smollm2/smollm2_generate --model smollm2-135m-instruct --chat --prompt "Hello" --max-new 8
```

Real checkpoint (needs weights + tokenizer from `HuggingFaceTB/SmolLM2-135M` or the Instruct twin):

```bash
./build/examples/smollm2/smollm2_generate \
  --model smollm2-135m --config examples/smollm2/config-smollm2-135m.json \
  --weights model.safetensors --tokenizer tokenizer.json \
  --prompt "Hello" --max-new 64
```

Expected tensor names are the Hugging Face ones (`model.embed_tokens.weight`, `model.layers.N.self_attn.q_proj.weight`, …, `lm_head.weight`; with `tie_word_embeddings=true` the loader aliases the embedding table when no separate `lm_head.weight` is present).

`--model` selects `smollm2-135m` (default; aliases `smollm2`, `135m`) or `smollm2-135m-instruct` (alias `smollm2-instruct`; case-insensitive, optional `huggingfacetb-` prefix). Llama model names are not accepted here (use the llama driver).

Stopping defaults to the single GPT-2 EOS id (base `0`, Instruct `2`; from `tokenizer.json` when `--tokenizer` is given, else `Config`); append extra ids with `--stop <id> ...`.

Context defaults to `max_seq_len = 8192` (the arch max); plain NeoX RoPE with `rope_theta = 100000`.

KV-cache modes (`--kv-page-size N`, `--kv-int8` / `--kv-fp`): dense FP by default; `--kv-page-size 16` pages the cache into blocks addressed through an explicit block table and grown on demand; `--kv-int8` stores symmetric int8 KV with per-token per-head FP32 scales. Paged dense is bit-exact vs dense; int8 trades small numerical drift for the smaller footprint.

Chat: `--chat` wraps `--prompt` as one turn via the SmolLM2-Instruct ChatML template (`<|im_start|>`/`<|im_end|>` blocks + open assistant header, with default-system injection when no system turn is present); `--system` prepends a system turn (implies chat).

Sampling defaults to `temperature=0.6`, `top_p=0.9`; `--temperature 0` restores greedy decode. Every sampling flag (`--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty`, `--seed`) overrides the preset; seed defaults to 0 so runs repeat unless `--seed` is given.

## TPL dispatch

SmolLM2 shapes route through the same dispatch as the rest of the library —
GEMM via cuBLAS (`sgemm`/`cublasGemmEx`) and attention via cuDNN SDPA on CUDA
builds with `KokkosDNN_ENABLE_CUBLAS`/`KokkosDNN_ENABLE_CUDNN`, native
kernels otherwise. No new TPL kernels were needed (shapes/dtypes already
covered); verified by code inspection plus the Serial build, not GPU
execution.
