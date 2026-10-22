# LFM2.5 example

LFM2.5-24B-A2B hybrid-MoE text inference built on the core library: 40 blocks (10 full-attention + 30 short-conv), SwiGLU dense FFNs in the first 2 layers then 64-expert top-4 MoE, GGUF checkpoint loader, byte-level BPE `tokenizer.json` loader, and the `lfm2_5_generate` driver (bfloat16 by default, inference only).

> Note: the model library (`KokkosDNN_LFM25*.hpp/.cpp`, `config-lfm2_5-24b-a2b.json`)
> now lives in `models/lfm2_5/` as the installable `KokkosDNN::lfm25` target;
> this directory keeps only the driver, tests, and docs.

## Quickstart (0 to working)

From repo root `/workspace` — build, fetch checkpoint files, run:

```bash
cmake -S . -B build -DKokkos_ROOT=/path/to/kokkos-install -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8 --target lfm2_5_generate

# HF retrieval: official GGUF repo + base repo for the tokenizer.
# GGUF repo `LiquidAI/LFM2.5-24B-A2B-GGUF` holds only .gguf files
# (no tokenizer.json); fetch `tokenizer.json` from the base repo
# `LiquidAI/LFM2.5-24B-A2B`. Both repos are public (no auth needed).
# Q4_K_M file is ~14.5 GB, tokenizer.json is ~17 MB.
pip install -U huggingface_hub  # provides the `hf` CLI
hf download LiquidAI/LFM2.5-24B-A2B-GGUF \
  --include "LFM2.5-24B-A2B-Q4_K_M.gguf" \
  --local-dir ./weights/lfm2_5
hf download LiquidAI/LFM2.5-24B-A2B \
  --include "tokenizer.json" \
  --local-dir ./weights/lfm2_5
ls ./weights/lfm2_5  # expect LFM2.5-24B-A2B-Q4_K_M.gguf + tokenizer.json
# (fallback mirror: ggml-org/LFM2.5-24B-A2B-GGUF, same filenames)

./build/examples/lfm2_5/lfm2_5_generate \
  --weights ./weights/lfm2_5/LFM2.5-24B-A2B-Q4_K_M.gguf \
  --tokenizer ./weights/lfm2_5/tokenizer.json \
  --config models/lfm2_5/config-lfm2_5-24b-a2b.json \
  --prompt "Explain attention in one sentence." --max-new 64
```

## Example commands

```bash
# Tiny random model, no downloads (also the unit-test end-to-end path):
./build/examples/lfm2_5/lfm2_5_generate --tiny --prompt "Hello" --max-new 8

# FP32 + small prefill chunks:
./build/examples/lfm2_5/lfm2_5_generate --tiny --fp32 --prefill-chunk 2 \
  --prompt "Hello" --max-new 8

# Chat (Instruct ChatML + optional system turn):
./build/examples/lfm2_5/lfm2_5_generate \
  --weights ./weights/lfm2_5/LFM2.5-24B-A2B-Q4_K_M.gguf \
  --tokenizer ./weights/lfm2_5/tokenizer.json \
  --config models/lfm2_5/config-lfm2_5-24b-a2b.json \
  --chat --system "You are a concise assistant." \
  --prompt "What is 2+2?" --max-new 128

# Sampling / stopping / context overrides:
./build/examples/lfm2_5/lfm2_5_generate --tiny --prompt "Hello" --max-new 16 \
  --temperature 0 --seed 0 --stop 124900 --stop-string "<|im_end|>" \
  --max-seq-len 64 --prefill-chunk 0
```

## Notes

- Flags (see `lfm2_5_generate.cpp` header): `--tiny` (default on; `--weights` opts out), `--weights model.gguf`, `--tokenizer tok.json`, `--config config.json` (HF overlay, present keys win), `--prompt`, `--max-new`, `--max-seq-len`, `--prefill-chunk N` (`0` = single-shot), `--chat` / `--system` / `--role`, `--add-bos` (default) / `--no-add-bos`, `--stop <id> ...`, `--stop-string ...`, `--fp32` (default bfloat16), `--temperature` / `--top-k` / `--top-p` / `--min-p` / `--repetition-penalty` / `--seed`.
- Sampling defaults: `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05` (`top_p=1.0`, `min_p=0.0`); `--temperature 0` = greedy; seed defaults to `0`.
- Tokenizer/chat: byte-level BPE `tokenizer.json` (128k); `bos=<|startoftext|>` (124894), stop `eos=<|im_end|>` (124900); `--chat` renders one `role` turn via LFM2.5-Instruct ChatML with open assistant header; `--system` implies chat; exactly-one-BOS invariant; decoded-text stop strings default to `{"<|im_end|>"}`.
- GGUF tensors (llama.cpp `lfm2moe` names, `.weight`/`.bias` suffix or bare stem): `token_embd`, `token_embd_norm` (final norm; legacy alias `output_norm` also accepted), `output` (absent in the official Q4_K_M file — covered by the tie fallback below), `blk.{l}.attn_norm` / `blk.{l}.ffn_norm`, attention `blk.{l}.attn_q/k/v/output` + `blk.{l}.attn_q_norm/k_norm`, conv `blk.{l}.shortconv.in_proj/conv/out_proj`, dense `blk.{l}.ffn_gate/up/down`, MoE `blk.{l}.ffn_gate_inp` + `blk.{l}.ffn_gate_exps/up_exps/down_exps` (stacked 3-D, ggml order `{cols,rows,experts}`) + `blk.{l}.exp_probs_b` (legacy alias `blk.{l}.ffn_exp_probs_b` also accepted). Tie note: `tie_word_embeddings=true` aliases the embedding table as `lm_head` when no `output` tensor exists; with `false` that checkpoint errors.
- GGUF dtype matrix (per-tensor; dense dequantized on host, MoE experts stay packed via `quant_gemm`):

  | dtype | dense | MoE expert |
  |---|---|---|
  | F32 / F16 / BF16 | yes | yes |
  | Q8_0 | yes | yes |
  | Q4_K | yes | yes |
  | Q6_K | yes | yes |
  | Q4_0 / Q5_0 / Q5_1 / Q5_K / others | no (throws) | no (throws) |

  `Q4_K_M` is a file-level recipe, not a storage type (mostly Q4_K, some Q6_K, norms F32) — covered by the rows above.
- Config: `config-lfm2_5-24b-a2b.json` is the faithful HF `config.json` copy plus PoC knobs (`max_seq_len`, `prefill_chunk_size`, `kv_*`); KV cache is dense FP only (paged/int8 rejected).

## Verification

```bash
ctest --test-dir build -R LFM25 --output-on-failure
./build/examples/lfm2_5/lfm2_5_generate --tiny --prompt "Hello" --max-new 8
```

Unit tests (`Test_LFM25Config`, `Test_LFM25Generate`, `Test_LFM25Tokenizer`) are tiny-synthetic only, no downloads. No full-weights golden test (needs a real checkpoint + tokenizer).

## Performance

- KV cache: only the 10 attention layers rent KV (`10 × seq × 8 kv heads × 64 head_dim × 2 (K+V) × elem`): 160 MiB BF16 at the PoC default `max_seq_len=8192` (320 MiB FP32); the full 131072 arch window rents ~2.5 GiB BF16 — raise via `--max-seq-len` / `--config` explicitly. The 30 conv layers rent only fixed `[dim, conv_L_cache-1]` state each (~240 KiB total at dim 2048).
- `--prefill-chunk N` caps prefill activations at `O(N·dim)`; chunking is exact (KV prefix + conv-state chaining).
- MoE path runs one narrow GEMM per routed expert per layer (decode: at most top_k); unrouted experts are skipped. No TPL dispatch on the gather-GEMM path yet.
