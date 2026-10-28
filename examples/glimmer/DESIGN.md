# Muse Glimmer 30B Text Path — Config Scaffold and Math Deltas

Scope: TEXT path only (`text_config` in the HF
`darkc0de/Muse-Glimmer-30B-heretic` `config.json`, verified verbatim into
`config-glimmer-30b-text.json`). Vision tower and multimodal glue are out of
scope. This stream owns `KokkosDNN_GlimmerConfig.hpp` + fixture + CMake;
Weights/Model/Tokenizer/generate land in follow-up streams.

Upstream references:

- Model weights/config: <https://huggingface.co/darkc0de/Muse-Glimmer-30B-heretic>
  (`config.json`, `text_config` object; upstream family card: Meta Muse).
- Transformers implementation: `modeling_muse_glimmer.py` /
  `configuration_muse_glimmer.py` (`MuseGlimmerTextConfig`). Exact op
  placement (score scaling, output multiplier, post-norm positions) must be
  confirmed against that file by the Model stream; assumptions below are
  marked ASSUMPTION.

## 1. text_config summary (from config.json)

| Key | Value |
| --- | --- |
| `num_hidden_layers` | 52 |
| `hidden_size` | 6656 |
| `intermediate_size` | 19968 (SwiGLU) |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 2 (all layers) |
| `head_dim` | **128, explicit** (6656 / 32 = 208 — NOT derived) |
| `layer_types` | 3× sliding + 1× full repeating → full layers at idx 3,7,...,51 (13 full, 39 sliding) |
| `layer_rope_theta` | 500000.0 on sliding layers, **0 on full layers** (= NoPE: no rotary on full layers) |
| `sliding_window` | 2048 (sliding layers only) |
| `rope_parameters` | `{rope_type: default, rope_theta: 500000}` |
| `hidden_activation` | `silu` (SwiGLU) |
| `qk_scale_factor` | 3.87 |
| `output_multiplier` | 0.19611613513818404 |
| `final_logit_softcapping` | 20.0 |
| `rms_norm_eps` | 1e-5 |
| `post_norm_eps` | 1e-8 |
| `tie_word_embeddings` | false (separate `lm_head` required) |
| `vocab_size` | 202048 |
| `attention_bias` | false; `attention_dropout` 0.0 |
| `max_position_embeddings` | 131072 |
| `bos/eos/pad` | 200000 / 200001 / null (absent) |
| `dtype` | bfloat16 |

## 2. Math deltas vs Gemma4 / Llama

1. **Explicit `head_dim` (128 ≠ hidden/heads).** Llama derives
   `head_dim = dim / n_heads` and validates equality; Gemma4 has two
   per-type head dims. Glimmer fixes a single explicit 128 while
   `6656 / 32 = 208`, so projection shapes are
   `q: [32·128, 6656]`, `k/v: [2·128, 6656]`, `o: [6656, 32·128]`.
   `Config::recompute_derived()` is therefore a documented no-op and
   `validate()` requires only positivity/evenness — never divisibility.
2. **Hybrid sliding(2048)/full.** Same 3+1 motif as Gemma4's 4+1 but with a
   2048 window (vs 512) and no KV sharing, no per-type head dims, no
   partial-RoPE: both layer kinds rotate the full head dim with the same
   theta (see 3).
3. **Per-layer RoPE theta (NoPE on full layers).** `layer_rope_theta` is
   500000 on all sliding layers and 0 on all full layers. Confirmed
   against `modeling_muse_glimmer.py`: 0 means NoPE — the HF model passes
   `position_embeddings=None` on those layers, so they rotate nothing and
   only the attention window differs. The Model keys off the raw
   `layer_rope_theta` entry (`Config::rope_theta_for_layer()` still falls
   back to the global theta for callers that need a numeric theta); the
   fixture test pins the 0-marking.
4. **QK scale factor (3.87).** After QK RMSNorm, attention scores are
   multiplied by `qk_scale_factor = 3.87` — NOT by `1/sqrt(128) ≈ 0.0884`.
   (Gemma4 instead uses scale 1.0 with QK norms; Llama uses `1/sqrt(d)` with
   no QK norms.) ASSUMPTION on exact placement (pre- vs post-mask add,
   interaction with sliding masking): confirm against
   `modeling_muse_glimmer.py` in the Model stream.
5. **`output_multiplier` (~0.19611).** A scalar applied to the final hidden
   states (before the final norm/head). ASSUMPTION on exact placement;
   confirm against the modeling file. Stored as double to keep the full
   `0.19611613513818404` precision.
6. **Logit softcapping 20.** `logits = 20·tanh(logits/20)` (vs 30 on
   Gemma4; Llama has none).
7. **`post_norm_eps` (1e-8) vs `rms_norm_eps` (1e-5).** Two epsilons:
   the tighter one is for the post-attention / post-FFN RMSNorms.
   ASSUMPTION on which norms take which epsilon; confirm against the
   modeling file. Both are validated positive/finite.
8. **SwiGLU MLP (`silu`), single width 19968.** Llama-like gated-SiLU MLP;
   no Gemma4-style gated-GeLU-tanh, no double-wide layers, no PLE, no
   layer scalars.
9. **Untied head.** `tie_word_embeddings = false`, so the loader must read
   a separate `lm_head [202048, 6656]` tensor and must NOT alias the
   embedding table (inverse of the Gemma4/Llama-3.2-1B tied case).
10. **No KV sharing, no MoE, no attention bias.** Every layer owns its
    `[max_seq, 2, 128]` K/V cache: `kv_cache_bytes = layers·T·kv·d·B·2`
    (≈ 218 MiB BF16 at the PoC default T = 8192).

## 3. Native dtype decision

Config `dtype: bfloat16` → Kokkos native type
`Kokkos::Experimental::bhalf_t`. All weights/activations/KV are `bhalf_t`;
reductions, scales, `inv_freq`, norms stay FP32 — same convention as the
Llama/SmolLM2/Gemma4 examples.

## 4. Implementation plan (next steps, other streams)

1. **Weights:** safetensors loader for the `model.language_model.*` prefix;
   require `lm_head`, never alias the embedding table; single head dim 128
   for all layers.
2. **Model:** Llama-model skeleton (chunked prefill, dense KV per layer);
   per-layer RoPE table from the raw `layer_rope_theta` entries (empty
   table = NoPE on 0-marked full layers); sliding-window mask
   2048 on sliding layers; QK RMSNorm + 3.87 score scale; post-norms with
   `post_norm_eps`; `output_multiplier` before the head; tied-... (untied)
   `lm_head` gemv + softcap-20 head.
3. **Tokenizer:** BPE `tokenizer.json` loader (vocab 202048 + added tokens;
   bos 200000 / eos 200001, no pad token).
4. **Validation:** golden tests vs HF `transformers` (eager, bf16→fp32
   logits) on a tiny prompt; per-op tests vs the closed forms above;
   prefill-vs-decode KV consistency incl. window eviction.
