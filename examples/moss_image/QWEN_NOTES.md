# Qwen3.5 Text-Encoder Ground Truth Notes (MOSS-Image-SFT)

Date (UTC): 2026-10-18. Worktree: `/tmp/wt/moss-enc`.
All files below saved under `/tmp/wt/moss-enc/reference/`.

## Fetched artifacts (verified, do not re-guess)

| File | Source | Bytes |
|---|---|---|
| `config.json` | `https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/raw/main/text_encoder/config.json` | 2813 |
| `tokenizer_config.json` | `.../tokenizer/tokenizer_config.json` | 1123 |
| `chat_template.jinja` | `.../tokenizer/chat_template.jinja` | 7756 |
| `tokenizer.json` | `.../resolve/main/tokenizer/tokenizer.json` (raw path is a 133 B git-lfs pointer) | 19989325 |
| `transformers-5.12.1-py3-none-any.whl` | `pip download transformers==5.12.1 --no-deps` | 11150587 |
| `transformers_src/transformers/models/qwen3_5/modeling_qwen3_5.py` | extracted from wheel, 2117 lines | — |
| `transformers_src/transformers/models/qwen3_5/configuration_qwen3_5.py` | extracted from wheel, 192 lines | — |
| `text_encoder_index.json` | `.../text_encoder/model.safetensors.index.json` | — |
| `pipeline_moss_image.py` | `.../pipeline_moss_image.py` (440 lines) | — |

Config matches `transformers_version 5.12.1`, `model_type qwen3_5`,
`architectures ["Qwen3_5Model"]`, top-level `dtype bfloat16`.

## Verified dims (`text_encoder/config.json` → `text_config`)

- vocab 248320, hidden 4096, 32 layers, `num_attention_heads` 16 Q /
  `num_key_value_heads` 4 KV, `head_dim` 256, `intermediate_size` 12288,
  `hidden_act silu`, `attention_bias false`, **`attn_output_gate true`**,
  `rms_norm_eps` 1e-6, `max_position_embeddings` 262144,
  `tie_word_embeddings` false, `pad_token_id`/`bos_token_id` null,
  `eos_token_id` 248044.
- Hybrid `layer_types`: 3× `linear_attention` + 1× `full_attention`
  repeating; full at layers **3,7,11,15,19,23,27,31** (0-based) — confirmed
  both from `config.json` and from the safetensors index tensor names
  (`layers.{3,7,...,31}.self_attn.*` vs `layers.{0,1,2,...}.linear_attn.*`).
- Linear-attn params: `linear_num_key_heads` 16 × `linear_key_head_dim` 128
  (key_dim 2048), `linear_num_value_heads` 32 × `linear_value_head_dim` 128
  (value_dim 4096), `linear_conv_kernel_dim` 4, `mamba_ssm_dtype float32`.
- RoPE: `partial_rotary_factor` 0.25, `rope_theta` 10M,
  `mrope_section` [11,11,10], `mrope_interleaved` true, `rope_type` default.
- Weight names (759 tensors total: 426 `language_model.*`, 333 `visual.*`,
  no `lm_head`): text path `language_model.{embed_tokens,norm,layers.0..31.*}`;
  full layers `self_attn.{q,k,v,o,q_norm,k_norm}_proj`; linear layers
  `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,conv1d,A_log,dt_bias,norm}`;
  per-layer `input_layernorm`, `post_attention_layernorm`,
  `mlp.{gate,up,down}_proj`. All text linear weights unbiased.

## Verified math (`modeling_qwen3_5.py`, transformers 5.12.1)

- **Full attention** (`Qwen3_5Attention`, src lines 645–717):
  `qg = q_proj(h)` [.., 2·16·256] (no bias), chunk → query [..,16,256],
  gate [..,4096]; `q_norm`/`k_norm` = `Qwen3_5RMSNorm(256)` on heads;
  RoPE via `apply_rotary_pos_emb` with `rotary_dim = cos.shape[-1]`;
  scale = `head_dim**-0.5` = 1/16; causal mask; softmax fp32;
  `out = reshape(attn) * sigmoid(gate)` (the `attn_output_gate`);
  then `o_proj` (no bias).
- **Qwen3_5RMSNorm** (lines 736–750): `(1 + w)` variant, zero-init
  (`output = norm_fp32 * (1 + w)`); eps 1e-6. Applies to input/post/final
  norms and to q_norm/k_norm (per head dim 256).
- **Partial RoPE** (`Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`,
  lines 115–145): `dim = int(head_dim * partial_rotary_factor)` = 64;
  `inv[j] = theta^(-2j/64)`, j = 0..31 (denominator is 64, NOT head_dim —
  differs from `rope_inv_freq_proportional_host`, so a local kernel is used).
  ` Rotary layout (lines 570–605): first `rotary_dim` = 64 dims rotated as
  pairs `(d, d+32)`, remaining 192 dims passthrough.
- **MROPE interleave** (lines 169–184): with section [11,11,10] the 32
  frequency slots split 11 T + 11 H + 10 W. For text-only encoder forwards
  all three position rows are identical `arange(S)`, so the interleave is
  an exact no-op; positions are 0..S-1 per prompt.
- **Linear attention = Gated DeltaNet** (`Qwen3_5GatedDeltaNet`, lines
  371–559), NOT a scalar SSM. Exact torch fallbacks verified (lines
  222–368), ported 1:1 into `KokkosDNN_MossImageEncoder.hpp`:
  1. `mixed_qkv = in_proj_qkv(h)` [B,S,8192]; depthwise causal conv1d k=4,
     groups=8192, no bias, left zero-pad, then SiLU
     (`F.silu(conv1d(x)[:, :, :S])`, line 497).
  2. `z = in_proj_z(h)` → [B,S,32,128]; `b = in_proj_b(h)`,
     `a = in_proj_a(h)` → [B,S,32].
  3. Split q/k [B,S,16,128], v [B,S,32,128]; `beta = sigmoid(b)`;
     `g = -exp(A_log) * softplus(a + dt_bias)` (fp32; line 518).
  4. repeat_interleave q,k 16 → 32 heads (ratio 2, line 519–521).
  5. QK L2-norm eps 1e-6, then `q *= 1/sqrt(128)` (lines 258–276).
  6. Recurrent gated delta rule, fp32, per (batch, head) state
     S [128,128] = 0; per token t (lines 352–363):
     `S *= exp(g_t); kv = (S·k_t).sum_k; delta = (v_t − kv)·beta_t;`
     `S += k_t ⊗ delta; out_t = (S·q_t).sum_k`.
     (The `chunk_gated_delta_rule`, lines 246–324, is a mathematically
     equivalent tiled form with chunk 64; the recurrent form is the
     reference math and is what the header implements.)
  7. `Qwen3_5RMSNormGated(128)` (lines 187–202): fp32 RMS norm × `weight`
     (ones-init, plain multiply) × `silu(z)`; then `out_proj` (no bias).
- **MLP** (lines 720–733): `down(silu(gate(x)) * up(x))`, no biases.
- **Decoder layer** (lines 756–809): `h += mixer(input_norm(h))`;
  `h += mlp(post_norm(h))`; final `norm(h)`.
- **Text-only forward** (`Qwen3_5TextModel.forward`, lines 1155–1218):
  embed → causal mask → 32 layers → final norm → `last_hidden_state`.
  With all-ones attention mask the linear-attn mask is `None`
  (`_update_linear_attn_mask`, lines 1220–1232); padding rows are
  additionally zeroed by `apply_mask_to_padding_states` (lines 205–214).
- **encode_prompt** (`pipeline_moss_image.py`, lines 313–338): per-prompt
  unpadded forwards (`tokenizer.apply_chat_template([{role:user, content}],
  tokenize=False, add_generation_prompt=True, enable_thinking=True)` →
  `tokenizer(fmt, return_tensors='pt', truncation=True,
  max_length=max_sequence_length)` → `text_encoder(**inputs, use_cache=False)`
  → `last_hidden_state[0]`); zero-padded `[B,L,4096]` batch + bool mask.
  Range enforced in `__call__`: 1 ≤ max_sequence_length ≤ 4096.
- **Chat template** (`chat_template.jinja` lines 147–154): generation prompt
  appends `<|im_start|>assistant\n<think>\n` when `enable_thinking` is true
  (default; `enable_thinking=false` emits `<think>\n\n</think>\n\n` instead).
- **Tokenizer** (`tokenizer.json`): BPE byte-level, 248044 merge vocab +
  33 added specials (max id 248076) = 248077 < 248320 (243 ids reserved);
  `Qwen2Tokenizer`, `model_max_length` 262144, pad `<|endoftext|>`,
  eos `<|im_end|>` (248044), no truncation/padding defaults in JSON
  (truncation is passed per-call by the pipeline).

## UNKNOWN / explicitly out of scope

- Per-tensor safetensors dtypes/shapes (headers not fetched; only names from
  the index). The loader accepts F32/BF16 and validates shapes against the
  config-derived table.
- FLA fused-kernel (`fla`, `causal_conv1d`) numerics: ported math matches the
  torch fallbacks exactly; agreement with the fused kernels is expected
  within fp32 rounding only (not verified — no GPU/torch run here).
- Vision tower (`visual.*`, 333 tensors): present in checkpoint, never
  executed by `encode_prompt`; not implemented.
- `mtp_*` (multi-token prediction) keys: ignored at load per
  `_keys_to_ignore_on_load_unexpected`; not implemented.
- Whether the MOSS pipeline runs the encoder in bf16 vs fp32 at runtime
  (`from_components_pretrained` defaults `torch_dtype=bf16`); the header
  supports `float` and `bhalf_t` I/O with FP32 accumulation either way.
- Tokenizer `tokenizer.json` pre-tokenizer regex differs slightly from
  `tokenizer_config.json` (`\p{L}+` vs `[\p{L}\p{M}]+`); tokenization stays
  host-side in transformers, so this does not affect the C++ port.
