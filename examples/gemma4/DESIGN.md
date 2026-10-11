# Gemma4 E2B-it Text Path — Architecture Gap Analysis

Scope: TEXT path only (`text_config` in `/workspace/gemma-4-E2B-it/config.json`,
authoritative). Vision/audio towers, MoE blocks (`enable_moe_block: false`),
and multimodal glue are out of scope for this step.

Upstream references:

- Model card: <https://huggingface.co/google/gemma-4-E2B-it>
- Architecture doc (hybrid attention, unified global KV, p-RoPE, PLE): model card
  "Models Overview" section.
- Transformers implementation:
  `src/transformers/models/gemma4/modeling_gemma4.py` (generated from
  `modular_gemma4.py`), `configuration_gemma4.py` (`Gemma4TextConfig`).
- RoPE init: `transformers/modeling_rope_utils.py::_compute_proportional_rope_parameters`.
- Local checkpoint: `/workspace/gemma-4-E2B-it/model.safetensors` (2011 tensors,
  verified below with `safetensors.torch`).

## 1. text_config summary (from config.json)

| Key | Value |
| --- | --- |
| `num_hidden_layers` | 35 |
| `hidden_size` | 1536 |
| `intermediate_size` | 6144 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 1 (all layers; `num_global_key_value_heads: null`, `attention_k_eq_v: false`) |
| `head_dim` (sliding layers) | 256 |
| `global_head_dim` (full layers, via `per_layer_config` override) | 512 |
| `layer_types` | 4× sliding + 1× full repeating (`sliding_window_pattern = 6`), last forced full → full layers at idx 4,9,14,19,24,29,34 (7 full, 28 sliding) |
| `sliding_window` | 512 (sliding layers only) |
| `rope_parameters.sliding_attention` | `{rope_type: default, rope_theta: 10000}` |
| `rope_parameters.full_attention` | `{rope_type: proportional, partial_rotary_factor: 0.25, rope_theta: 1e6}` (no `factor` → 1.0) |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `use_double_wide_mlp` | true (applies to KV-shared layers only, idx ≥ 15) |
| `num_kv_shared_layers` | 20 → first shared idx = 35 − 20 = 15 |
| `hidden_size_per_layer_input` | 256 (PLE on); `vocab_size_per_layer_input` 262144 |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | true (no `lm_head` tensor in checkpoint — verified) |
| `vocab_size` | 262144 |
| `rms_norm_eps` | 1e-6 |
| `attention_bias` | false; `attention_dropout` 0.0; text attention softcap None |
| `max_position_embeddings` | 131072 |
| `bos/eos/pad` | 2 / 1 / 0; generation stops `[1, 106, 50]` (generation_config.json) |
| sampling defaults | temperature 1.0, top_k 64, top_p 0.95 |
| `dtype` | bfloat16 |

Checkpoint verification (`model.safetensors`, torch.bfloat16):

- `language_model.embed_tokens.weight` [262144, 1536]; NO `lm_head` (tied).
- `language_model.embed_tokens_per_layer.weight` [262144, 35·256 = 8960].
- `language_model.per_layer_model_projection.weight` [8960, 1536];
  `per_layer_projection_norm.weight` [256]; `language_model.norm.weight` [1536].
- Per sliding layer (e.g. L0): q [2048=8·256, 1536], k/v [256=1·256, 1536],
  o [1536, 2048]; gate/up [6144, 1536], down [1536, 6144].
- Per full layer (e.g. L4): q [4096=8·512, 1536], k/v [512, 1536],
  o [1536, 4096].
- Layers 15–34 MLP is double-wide: gate/up [12288, 1536], down [1536, 12288].
- Every layer (incl. 15–34) stores `k_proj`/`v_proj`/`k_norm` weights on disk,
  but `Gemma4TextModel._keys_to_ignore_on_load_unexpected` drops them for
  KV-shared layers — they are dead weight, not used at runtime.
- No `v_norm.weight` anywhere (`v_norm` is `with_scale=False`).
- Per-layer tensors: `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm`,
  `per_layer_input_gate` [256, 1536], `per_layer_projection` [1536, 256],
  `post_per_layer_input_norm`, `layer_scalar` [1] (learned, ≠ 1, e.g.
  L0 ≈ 0.0178, L4 ≈ 0.498, L34 ≈ 0.167).
- Tokenizer: BPE (`tokenizer.json`), vocab 262144 + 24 added tokens.

## 2. Native dtype decision

Config `dtype: bfloat16` → Kokkos native type `Kokkos::Experimental::bhalf_t`
(`src/KokkosDNN_Types.hpp` already aliases `bhalf_t`; gemm/gemm_bias/attention
already cover it). All weights/activations/KV are `bhalf_t`; reductions,
scales, `inv_freq`, norms stay FP32 — same convention as the Llama/SmolLM2
examples.

## 3. Exact forward math (text path)

Per token row `x` (dim 1536), `e = embed_scale · E[id]`, `embed_scale = √1536`.

PLE (computed once per forward, then sliced per layer `p_i`, dim 256):

- `tok_i = reshape(E_ple[id] · √256, [35, 256])` (`E_ple`: [262144, 8960]).
- `ctx = RMSNorm_{g_ple}(x_main · W_ctx / √1536)`, `W_ctx`: [8960, 1536].
- `p = (ctx_reshaped + tok) / √2`, shapes [T, 35, 256].

Decoder layer `i` (type sliding/full, head dim `d` = 256/512, `H = 8`):

1. `h1 = RMSNorm_in(x)`; `q = RMSNorm_q(x·Wq)` reshaped [T, 8, d].
2. If `i < 15`: `k = RMSNorm_k(x·Wk)`, `v = RMSNorm_v_noscale(x·Wv)`
   ([T, 1, d]); RoPE both; write KV cache. If `i ≥ 15`: reuse shared
   `(K, V)` of the last non-shared layer of the same type
   (sliding ← layer 13, full ← layer 14; includes its cache growth).
3. `attn = softmax(q·Kᵀ · 1.0 + mask)·V`, `scaling = 1.0` (QK-normed, NOT
   `1/√d`); mask = causal (full) or sliding-window-512 causal (sliding).
   `h_attn = RMSNorm_post_attn(attn·Wo)`; `x += h_attn`.
4. `h2 = RMSNorm_pre_ffn(x)`; `m = down(gelu_tanh(gate) · up)`,
   gate/up width 6144 (`i < 15`) or 12288 (`i ≥ 15`).
   `h_mlp = RMSNorm_post_ffn(m)`; `x += h_mlp`.
5. PLE residual: `g = gelu_tanh(x·Wgate_i)` ([T, 256]); `g *= p_i`;
   `r = RMSNorm_post_ple(g·Wproj_i)`; `x += r`.
6. `x *= layer_scalar[i]` (learned scalar).

Head: `h = RMSNorm_final(x_last)`; `logits = h·Eᵀ` (tied);
`logits = 30·tanh(logits/30)`.

RoPE details:

- Sliding: `inv_freq[j] = 10000^(−2j/256)`, `j ∈ [0, 128)`; full NeoX rotation.
- Full (proportional, partial 0.25): `rope_angles = int(0.25·512//2) = 64`;
  `inv_freq[j] = (1e6)^(−2j/512)` for `j ∈ [0, 64)`, `0` for `j ∈ [64, 256)`.
  Zero entries give angle 0 (cos = 1, sin = 0) → trailing 384 dims pass through.
  Same NeoX kernel works with the zero-padded table.

GeLU tanh (matches `KokkosDNN::Impl::gelu_tanh_approx`):
`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.

## 4. Library inventory (`src/`)

| File | Provides |
| --- | --- |
| `KokkosDNN_Gemm.hpp` | `gemm` (Y = α·X·Wᵀ + β·Y, FP32 accum, cuBLAS), `gemv` |
| `KokkosDNN_GemmBias.hpp` / `KokkosDNN_Dense.hpp` | biased linear (attention_bias false → not needed) |
| `KokkosDNN_RMSNorm.hpp` | `rmsnorm` [T,D] + `rmsnorm_row`; formula matches Gemma4 (`y = x/rms·w`, no bias). Scaleless variant (v_norm) NOT supported — `w` is mandatory |
| `KokkosDNN_Activation.hpp` | `gelu_forward` incl. `GeluVariant::Tanh` — exact formula needed |
| `KokkosDNN_SwiGLU.hpp` | SwiGLU — NOT used by Gemma4 (plain gated GeLU instead) |
| `KokkosDNN_RoPE.hpp` | `rope_inv_freq` (default/llama3) + `rope_neoX_inplace`; no proportional/partial variant |
| `KokkosDNN_Attention.hpp` | `attention_gqa_single`: full-causal GQA, caller `scale`, FP32 streaming; NO sliding-window mask, NO scale-1.0-by-default issue (caller passes scale — OK) |
| `KokkosDNN_KVCache.hpp` | paged/int8 KV primitives (optional; dense path suffices for step 1) |
| `KokkosDNN_Embedding.hpp` | `embedding_gather` (no scale factor) |
| `KokkosDNN_Softmax.hpp` | row softmax (attention kernel is fused; not needed directly) |
| `KokkosDNN_Sampler.hpp` | greedy + temp/top-k/top-p/min-p + repetition penalty — covers temp 1.0/top_k 64/top_p 0.95 |
| `KokkosDNN_LayerNorm.hpp` | biased LayerNorm — not needed (all norms are RMSNorm) |
| `KokkosDNN_LayerScale.hpp` | check: separately verify — layer_scalar needs scalar-mul + PLE ops live in example, not src |

Reference examples: `examples/llama/` (`KokkosDNN_LlamaModel.hpp`,
`KokkosDNN_LlamaConfig.hpp`, weights loader with tied-embedding support,
chunked prefill, dense/paged/int8 KV) and `examples/smollm2/` (same skeleton,
smaller). `examples/CMakeLists.txt` registers `llama`, `dinov3`, `smollm2`,
`xor` — `gemma4` to be added.

## 5. Per-op gap table

| # | Required op | Status | Exact formula / notes |
| --- | --- | --- | --- |
| 1 | Scaled embedding gather | MISSING (trivial) | `out[t,:] = √1536 · E[tok[t],:]`. `embedding_gather` exists but has no scale; add scalar-mul kernel or `embedding_gather_scaled` in src |
| 2 | GeLU tanh MLP activation | EXISTS | `gelu_forward(..., GeluVariant::Tanh)`: `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| 3 | Gated-MLP multiply | MISSING (trivial) | `out = gelu(gate) · up` elementwise [T, F] (F = 6144 / 12288). NOT SwiGLU — new tiny kernel (or reuse pattern from `swiglu` with gelu) |
| 4 | RMSNorm (scaled) | EXISTS | `rmsnorm`: `y = x/√(mean(x²)+eps)·w`, eps 1e-6 |
| 5 | RMSNorm scaleless (v_norm) | MISSING (trivial) | `y = x/√(mean(x²)+eps)` (no weight). Extend `rmsnorm` with optional/empty-weight path or new `rmsnorm_noscale` |
| 6 | QK RMSNorm over head dim | EXISTS (via 4/5) | `q_norm`/`k_norm`: RMSNorm dim d=256/512 per head; reshaped views can reuse `rmsnorm` on flattened [T·H, d] |
| 7 | Default RoPE (sliding) | EXISTS | `rope_inv_freq(256, 10000)` + `rope_neoX_inplace`; formula `inv[j] = θ^(−2j/d)` matches |
| 8 | Proportional partial RoPE (full) | MISSING | `inv[j] = (1e6)^(−2j/512)`, j<64; `0` for j≥64. New `rope_inv_freq_proportional(head_dim=512, theta=1e6, partial=0.25)` host helper; kernel unchanged (zero angle = passthrough) |
| 9 | GQA causal attention, scale 1.0 | EXISTS | `attention_gqa_single` with `scale = 1.0f`; GQA map `h·Hkv/Hq`, Hkv = 1 |
| 10 | Sliding-window causal attention (512) | MISSING | Same streaming kernel + key-range clamp: query at absolute pos `p` attends keys `[max(0,p−511), p]`. New flag/param (e.g. `sliding_window` size_t, 0 = full). Needed for 28/35 layers |
| 11 | KV-cache sharing (15–34 reuse 13/14) | MISSING (example-level) | No new src op: model holds 2 KV pools (sliding-d256 from L13, full-d512 from L14); shared layers skip k/v proj+write, attend over the shared prefix. `store_full_length_kv` semantics: L13/L14 must keep full prefix even past window |
| 12 | Double-wide MLP shapes | EXISTS (via gemm) | Just config dims: layers ≥15 use F = 12288 in existing `gemm` calls |
| 13 | PLE: token-identity gather + reshape | MISSING (example-level) | `E_ple`: [262144, 8960]; gather + `·√256` + reshape [T,35,256]. Uses op from #1 |
| 14 | PLE: context projection + norm + mix | MISSING (example-level) | `ctx = RMSNorm(gemm(x, W_ctx)·(1/√1536))`, reshape, `p = (ctx+tok)/√2`. Uses existing `gemm`+`rmsnorm` |
| 15 | PLE: per-layer gate→mul→proj→norm→resid | MISSING (example-level) | `r = RMSNorm_post(gemm(gelu_tanh(gemm(x,Wg))·p_i, Wp))`; `x += r`. Uses #2/#3/#4 + `gemm` |
| 16 | Post-attention / post-FFN norms | EXISTS | `rmsnorm` (4/5 cover all 4 per-layer norms + final) |
| 17 | Layer scalar | MISSING (trivial) | `x *= s[i]` elementwise [T, 1536]. Tiny kernel (cf. `KokkosDNN_LayerScale.hpp` — verify/reuse) |
| 18 | Final logit softcapping | MISSING (trivial) | `logits = 30·tanh(logits/30)` elementwise [V=262144]. New tiny kernel (rank-1), applied after tied `gemv` head |
| 19 | Tied lm_head | EXISTS (pattern) | Llama weights loader already shares embed table when tied; reuse pattern |
| 20 | Attention logit softcapping | NOT NEEDED | Text path has no attention softcap (`softcap=None` in `eager_attention_forward`; only audio tower uses one) |
| 21 | MoE / `attention_k_eq_v` / biases | NOT NEEDED | `enable_moe_block: false`, `attention_k_eq_v: false`, `attention_bias: false` |
| 22 | Tokenizer (BPE, 262144+24) | MISSING (example-level) | `tokenizer.json` present; example needs BPE loader like Llama/SmolLM2 `*Tokenizer` (check their impl for reuse) |

New src kernels needed: #1 (or fuse), #3, #5, #8-helper, #10, #17, #18 —
all small elementwise/table helpers except #10 (attention mask clamp).
Model-level (example dir): #11, #13–#15, #19-pattern, #22.

## 6. Implementation plan (next steps)

1. **src ops (one header each or batched, with unit tests in `tests/`):**
   a. `embedding_gather_scaled` (or scale param): `out = s·table[tok]`.
   b. Gated-GeLU multiply kernel (gate·up with tanh-GeLU on gate).
   c. `rmsnorm` scaleless path (empty `w` → skip weight multiply).
   d. `rope_inv_freq_proportional(d, θ, partial)`: zero-padded table.
   e. Sliding-window support in `attention_gqa_single` (window param, key
      lower-bound clamp; keep O(1) memory, cuDNN bypass for windowed path).
   f. `logit_softcapping(x, c)` rank-1 + rank-2 (`c·tanh(x/c)`).
   g. Scalar-mul (`x *= s`) — check `KokkosDNN_LayerScale.hpp` first for reuse.
2. **examples/gemma4/:** `KokkosDNN_Gemma4Config.hpp` (text_config fields:
   per-type head_dim, layer_types, window, rope params per type, PLE dims,
   double-wide threshold idx 15, softcap 30, eps, tied), `KokkosDNN_Gemma4Weights.*`
   (safetensors loader: `language_model.*` prefix, dead k/v on shared layers
   ignored, tied head), `KokkosDNN_Gemma4Model.hpp` (Llama-model skeleton:
   chunked prefill, 2 shared KV pools + per-type caches, PLE precompute,
   per-layer path §3), tokenizer (BPE reuse), `CMakeLists.txt` + register in
   `examples/CMakeLists.txt`.
3. **Validation:** golden tests vs HF `transformers` (eager, bf16→fp32 logits)
   on a tiny prompt; per-op tests vs closed-form formulas above; prefill vs
   decode KV-consistency incl. window eviction and shared-KV reuse.
