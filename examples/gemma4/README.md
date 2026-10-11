# Gemma4 example (text path)

Gemma4 E2B-it **text-path** inference built on the core library: E2B-it text
config defaults, a safetensors checkpoint loader (on-disk F32/F16/BF16 are
converted to the model scalar), a SentencePiece-▁ BPE `tokenizer.json`
loader (space→▁ normalizer, per-piece BPE, `<0xXX>` byte fallback), a hybrid
sliding/full KV-cache autoregressive model with KV sharing and
per-layer-embedding (PLE), and a `gemma4_generate` driver.

Separate family from [llama](../llama/) and [smollm2](../smollm2/): config,
weights, model, and tokenizer are split-form duplicates in namespace
`KokkosDNN::Gemma4` (no shared headers with those examples — only the core
`src/` ops are reused).

Scope: the Gemma4 path is **text only** (`text_config`); vision/audio towers
and MoE blocks are out of scope. **Inference only** (prefill + decode with
opt-in sampling, see below) with **bfloat16 storage/compute by default**
(`--fp32` opts in). There is no weight quantization; KV caches are dense FP.
The canonical Gemma4 chat template (`chat_template.jinja`, tool-calling /
channel markers) is out of scope for this step: the driver encodes raw
prompts directly (no `--chat` flag).

Layout:

- `KokkosDNN_Gemma4Config.hpp` — model `Config` (dims, per-type head dims,
  `layer_types`, sliding window, per-type RoPE params, KV-sharing threshold,
  double-wide MLP, PLE dims, `final_logit_softcapping`, `tie_word_embeddings`,
  `hidden_activation`/bias checks) plus HF `config.json` overlay (full file
  with `text_config`, or a bare `text_config` object) and `kv_cache_bytes()`.
- `KokkosDNN_Gemma4Model.hpp` — KV-cache autoregressive model (`forward()`,
  `generate()`, chunked prefill). Example-local fused kernels
  (`KokkosDNN::Gemma4::Impl`): per-head QK reshape+norm, v reshape+noscale
  norm, PLE context norm+mix, PLE gate multiply.
- `KokkosDNN_Gemma4Weights.{hpp,cpp}` — safetensors loader (mmap-backed
  streaming, same style as the Llama loader).
- `KokkosDNN_Gemma4Tokenizer.{hpp,cpp}` — SentencePiece-▁ `tokenizer.json`
  loader/encoder/decoder.
- `gemma4_generate.cpp` — `gemma4_generate` driver.
- `config-gemma4-e2b-it.json` — faithful config fixture (the HF
  `text_config` object verbatim).
- `Test_Gemma4Generate.cpp` — example tests (presets, fixture, tiny
  forward/generate smoke, gated real-weight check).
- `DESIGN.md` — text-path architecture gap analysis (authoritative math).

## Key design decisions

- **Dtype:** native bfloat16 (`Kokkos::Experimental::bhalf_t`, the
  checkpoint dtype) for all weights/activations/KV; reductions, scales,
  `inv_freq`, and `layer_scalar` stay FP32 — same convention as the
  Llama/SmolLM2 examples. `--fp32` opts into FP32 for comparison.
- **KV sharing:** layers 15–34 (the trailing `num_kv_shared_layers = 20`)
  hold no KV of their own: sliding layers reuse the L13 cache, full layers
  the L14 cache (the last non-shared layer of each type). Non-shared layers
  rent one dense `[max_seq, 1, head_dim]` K/V pair each (12 sliding d256 +
  3 full d512). Donor caches keep the full prefix (`store_full_length`);
  the sliding window mask (512) applies at attention time. Dead k/v/k_norm
  tensors stored on disk for shared layers are skipped by the loader.
- **PLE (per-layer embedding):** token part `sqrt(256)·E_ple[tok]` reshaped
  `[T, 35, 256]` plus context part
  `RMSNorm(x_main·W_ctx/√1536)`, mixed as `(ctx + tok)/√2` once per chunk,
  then per layer `RMSNorm(gelu_tanh(x·Wgate)·p_l·Wproj)` added to the
  residual.
- **Double-wide MLP:** layers ≥ 15 use 12288-wide gated-GeLU
  (`gated_gelu_multiply` with `GeluVariant::Tanh`); earlier layers 6144.
- **Attention:** scale 1.0 (QK RMSNorms replace `1/√d`); sliding layers use
  `attention_gqa_single_sliding` (window 512, default RoPE theta 10k);
  full layers use `attention_gqa_single` with the proportional partial RoPE
  table (theta 1e6, factor 0.25 → 64 rotating pairs of 256, trailing 384
  dims passthrough) via the unchanged NeoX kernel.
- **Head:** `sqrt(1536)`-scaled embedding, final RMSNorm, tied `lm_head`
  (`gemv` over the embedding table — the checkpoint stores no `lm_head`),
  `30·tanh(logits/30)` softcapping (`logit_softcap_forward`).

Only library ops from the Gemma4 step (`embedding_gather_scaled`,
`gated_gelu_multiply`, `rmsnorm_noscale`, `rope_inv_freq_proportional`,
`attention_gqa_single_sliding`, `scale_by_scalar`, `logit_softcap_forward`)
plus pre-existing ops (`gemm`, `gemv`, `rmsnorm`, `gelu_forward`,
`rope_neoX_inplace`, sampler) are used; no new `src/` ops.

## Weights

The text checkpoint is `google/gemma-4-E2B-it` (gated — request access on
Hugging Face first). The single-file layout keeps all towers in one
`model.safetensors` (~10 GiB, BF16); this example only loads the
`model.language_model.*` tensors (600 of 2011; audio/vision tensors are
never requested).

With `huggingface_hub` (or `hf download`):

```bash
hf download google/gemma-4-E2B-it model.safetensors tokenizer.json \
  tokenizer_config.json config.json generation_config.json --local-dir weights/gemma-4-E2B-it
```

Expected tensor names are the Hugging Face ones
(`model.language_model.embed_tokens.weight`,
`model.language_model.layers.N.self_attn.q_proj.weight`, …). With
`tie_word_embeddings=true` the loader aliases the embedding table since no
`lm_head` tensor exists. To inspect what a (possibly still downloading)
checkpoint file actually contains, without loading the model:

```bash
./build/examples/gemma4/gemma4_generate \
  --weights weights/gemma-4-E2B-it/model.safetensors --list-tensors
```

If shards are incomplete, `load_safetensors` throws naming the first
missing required tensor.

## Run

Tiny random model (no downloads, runs anywhere; also the unit-test
end-to-end path):

```bash
./build/examples/gemma4/gemma4_generate --tiny --prompt "Hello" --max-new 8
```

Full-size shape with random weights (no downloads; needs ~10 GiB RAM):

```bash
./build/examples/gemma4/gemma4_generate --model gemma4-e2b-it --prompt "Hello" --max-new 8
```

Real checkpoint (needs the files from above):

```bash
./build/examples/gemma4/gemma4_generate \
  --model gemma4-e2b-it --config examples/gemma4/config-gemma4-e2b-it.json \
  --weights weights/gemma-4-E2B-it/model.safetensors \
  --tokenizer weights/gemma-4-E2B-it/tokenizer.json \
  --prompt "Hello" --max-new 64
```

Stopping defaults to the `generation_config.json` set `{1 (<eos>), 106,
50}`; append extra ids with `--stop <id> ...`. Sampling defaults to the
upstream `generation_config.json` (`temperature=1.0`, `top_k=64`,
`top_p=0.95`); `--temperature 0` restores greedy decode. Every sampling
flag (`--temperature`, `--top-k`, `--top-p`, `--min-p`,
`--repetition-penalty`, `--seed`) overrides the preset; seed defaults to 0
so runs repeat unless `--seed` is given. Other flags: `--add-bos` /
`--no-add-bos` (BOS handling), `--stop-string` (decoded-text stop),
`--prefill-chunk N` (O(N·dim) prefill activations, `0` = single-shot),
`--config` (HF `config.json` overlay — full file or bare `text_config`,
present keys win), `--max-seq-len N` (KV-cache context cap, validated
against `[1, 131072]`).

## Memory note

BF16 E2B-it weights are ~10 GiB, so a full BF16 run needs a large-GPU or
CPU-Kokkos build (a 24 GiB card fits weights but leaves little headroom
with the full-window KV cache; cap it with `--max-seq-len`). FP32
(`gemma4_generate --fp32`) needs ~20 GiB and does **not** fit there. Use
the tiny config for local smoke tests. The dense KV cache at the default
`max_seq_len = 8192` is ~75 MiB BF16 (12×256 + 3×512 per token × 2 for
K+V); see `Config::kv_cache_bytes()` (the driver prints the full-window
figure).

## TPL dispatch

Gemma4 shapes route through the same dispatch as the rest of the library —
GEMM via cuBLAS (`sgemm`/`cublasGemmEx`) and full attention via cuDNN SDPA
on CUDA builds with `KokkosDNN_ENABLE_CUBLAS`/`KokkosDNN_ENABLE_CUDNN`,
native kernels otherwise (the sliding-window path always runs the native
kernel: the cuDNN SDPA graph bakes in a full causal mask). No new TPL
kernels were needed (shapes/dtypes already covered); verified by code
inspection, not GPU execution.
