# LFM2.5 example

LFM2.5-24B-A2B (Liquid AI) hybrid MoE inference built on the core library:
40 blocks mixing short-convolution layers with full-attention layers (every
4th layer starting at index 2, i.e. 10 attention + 30 conv), SwiGLU dense
FFNs in the first 2 layers and 64-expert top-4 MoE FFNs after that, a GGUF
checkpoint loader, a byte-level BPE `tokenizer.json` loader, a KV-cache /
conv-state autoregressive model, and a `lfm2_5_generate` driver.

Separate family from [llama](../llama/): config, weights, model, and
tokenizer are split-form duplicates in namespace `KokkosDNN::LFM25` (no
shared headers with `examples/llama/` — only the core `src/` ops are
reused).

Scope: the LFM2.5 path is **inference only** (prefill + decode, sampling
opt-in, see below) with **bfloat16 storage/compute by default** (`--fp32`
opts in). No weight download is needed for `--tiny` (random weights +
byte-fallback prompt); real weights/tokenizer come from a GGUF checkpoint
plus the HF tokenizer files. The KV cache is **dense FP only for now**
(paged/int8 modes are rejected by the `Model` constructor; see TODOs).

Layout:

- `KokkosDNN_LFM25Config.hpp` — model `Config` (dims, MoE, conv state,
  RoPE, context, `tie_word_embeddings`, `hidden_act`/`model_type` checks)
  plus HF `config.json` overlay and `kv_cache_bytes()`.
- `KokkosDNN_LFM25Model.hpp` — hybrid autoregressive model (`forward()`,
  `generate()`, chunked prefill, conv-state handling).
- `KokkosDNN_LFM25Weights.{hpp,cpp}` — GGUF loader + `random_init()`.
- `KokkosDNN_LFM25Tokenizer.{hpp,cpp}` — byte-level BPE `tokenizer.json`
  loader/encoder/decoder plus ChatML-with-thinking template helpers.
- `lfm2_5_generate.cpp` — `lfm2_5_generate` driver.
- `config-lfm2_5-24b-a2b.json` — faithful config fixture (copy of the HF
  `config.json` values, plus the PoC `max_seq_len`/`prefill_chunk_size`/
  `kv_*` runtime knobs).
- `Test_LFM25Config.cpp`, `Test_LFM25Generate.cpp`,
  `Test_LFM25Tokenizer.cpp` — example tests (tiny synthetic only, no
  downloads).

## Run

Random-weights smoke (no downloads, runs anywhere; also the unit-test
end-to-end path):

```bash
./build/examples/lfm2_5/lfm2_5_generate --tiny --prompt "Hello" --max-new 8
./build/examples/lfm2_5/lfm2_5_generate --tiny --fp32 --prefill-chunk 2 --prompt "Hello" --max-new 8
```

Real checkpoint (GGUF weights + HF tokenizer + config overlay):

```bash
./build/examples/lfm2_5/lfm2_5_generate \
  --weights lfm2_5-24b-a2b-q4_k_m.gguf --tokenizer tokenizer.json \
  --config examples/lfm2_5/config-lfm2_5-24b-a2b.json \
  --prompt "Explain attention in one sentence." --max-new 64
```

Chat (Instruct ChatML + thinking blocks):

```bash
./build/examples/lfm2_5/lfm2_5_generate \
  --weights lfm2_5-24b-a2b-q4_k_m.gguf --tokenizer tokenizer.json \
  --config examples/lfm2_5/config-lfm2_5-24b-a2b.json \
  --chat --system "You are a concise assistant." \
  --prompt "What is 2+2?" --max-new 128
```

`--chat` wraps `--prompt` as one turn via the LFM2.5-Instruct ChatML
renderer (open assistant header appended); `--system` prepends a system
turn (implies chat). Assistant turns that carry a thinking trace render as
`<think>{trace}</think>` inside the turn; the stop default truncates
generation at `<|im_end|>`.

Expected GGUF tensor names are the llama.cpp ones (`token_embd`,
`output_norm`, `output`, `blk.{l}.attn_norm` / `blk.{l}.ffn_norm`,
attention layers `blk.{l}.attn_q/k/v/output` + `blk.{l}.attn_q_norm/k_norm`,
conv layers `blk.{l}.shortconv.in_proj/conv/out_proj`, dense layers
`blk.{l}.ffn_gate/up/down`, MoE layers `blk.{l}.ffn_gate_inp` +
`blk.{l}.ffn_gate_exps/up_exps/down_exps` (stacked 3-D) +
`blk.{l}.ffn_exp_probs_b`; each with a `.weight`/`.bias` suffix, bare
stems also accepted). Tie note: LFM2.5 ships `tie_word_embeddings=true`,
so when the checkpoint has no `output` tensor the loader aliases the
embedding table as `lm_head` instead of throwing; with
`tie_word_embeddings=false` the same checkpoint is an error.

GGUF dtype support matrix (per-tensor dispatch; dense tensors are
dequantized on the host, MoE experts stay packed and run through
`quant_gemm`):

| dtype | dense | MoE expert |
|---|---|---|
| F32 / F16 / BF16 | yes | yes |
| Q8_0 | yes | yes |
| Q4_K | yes | yes |
| Q6_K | yes | yes |
| Q4_0 / Q5_0 / Q5_1 / Q5_K / others | no (throws) | no (throws) |

"Q4_K_M" is a file-level quantization *recipe*, not a storage type: those
checkpoints store most tensors as Q4_K (a few as Q6_K, norms as F32), so
the Q4_K + Q6_K (+ Q8_0/F32/F16/BF16) coverage above is Q4_K_M inference.

Context / KV math: only the 10 full-attention layers rent KV cache, sized
`10 attn layers x seq x 8 kv heads x 64 head_dim x 2 (K+V) x elem`.
At the PoC default `max_seq_len = 8192` that is 160 MiB in BF16 (320 MiB
in FP32); the full 131072 arch window would rent 2.5 GiB in BF16, so the
default stays at 8192 and `--max-seq-len` / `--config` raise it
explicitly. The 30 conv layers rent no KV cache, only a fixed
`[dim, conv_L_cache - 1]` state each (~240 KiB total at dim 2048). Use
`--prefill-chunk N` to cap prefill activations at `O(N·dim)` (`0` =
single-shot); chunking is exact (KV prefix + conv-state chaining).

Sampling defaults to `temperature=0.1`, `top_k=50`,
`repetition_penalty=1.05` (`top_p=1.0`, `min_p=0.0`); `--temperature 0`
restores greedy decode. Every sampling flag (`--temperature`, `--top-k`,
`--top-p`, `--min-p`, `--repetition-penalty`, `--seed`) overrides the
preset; seed defaults to 0 so runs repeat unless `--seed` is given.

## Tokenizer / chat

The tokenizer loads a HF `tokenizer.json` (byte-level BPE, 128k):
literal special/added tokens split first (longest match, never broken up
by BPE), then Digits + GPT-2 ByteLevel pre-tokenization, GPT-2 byte
mapping, per-piece BPE with byte fallback. Conventions:
`bos=<|startoftext|>` (124894), `eos`/stop=`<|im_end|>` (124900);
`encode`/`decode` default to `add_bos=true` / `skip_special_tokens=true`,
and `ensure_bos`/`encode_chatml` keep the exactly-one-BOS invariant.
The Instruct template is ChatML with thinking support:

```text
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
[<think>{thinking}</think>
]{content}<|im_end|>
<|im_start|>assistant
```

A thinking block is emitted inside an assistant turn only when that turn
carries non-empty `thinking` and `preserve_thinking` is true. Decoded-text
stop strings (default `{"<|im_end|>"}`) truncate the generation before the
earliest match.

## TODOs

- Paged / int8 KV caches (config keys exist; the `Model` constructor
  currently throws on them — dense FP only, unlike the Llama example).
- Perf / TPL dispatch for the MoE gather-GEMM path (today one GEMM per
  expert per layer with zero host sync; kernel count scales with expert
  count).
- Full-weights golden test as future work (needs a real checkpoint +
  tokenizer; unit tests stay tiny-synthetic by design).
