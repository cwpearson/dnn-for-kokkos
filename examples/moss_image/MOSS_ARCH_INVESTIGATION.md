# MOSS-Image-SFT Architecture Investigation (read-only)

Date (UTC): 2026-10-17. Worktree: `/tmp/wt/moss-arch` (detached HEAD of KokkosDNN library).
Upstream: HF repo `OpenMOSS-Team/MOSS-Image-SFT` @ `fc5de0d56fc910cbca6f6625a70f5f582b26deb1`,
GitHub `OpenMOSS/MOSS-Image`. No code edited. Fetched JSON configs verified via
`curl` + `cat` (copies in `/tmp/moss_cfg/`); safetensors binaries NOT downloaded
(per-tensor shapes/dtypes below are DERIVED or UNKNOWN as marked).

## 1. Model identity / size

- Family: single-stream MMDiT text-to-image (diffusers `MOSSImagePipeline`, custom
  `pipeline_moss_image.py`, `trust_remote_code=True`, diffusers `0.38.0`).
- HF label: "Model size 5B params, Tensor type BF16·F32". The pipeline docstring's
  reference component paths are `Qwen/Qwen3.5-9B` (text encoder) and
  `black-forest-labs/FLUX.2-klein-4B` (VAE source); the "5B" figure is consistent
  with the transformer shard index alone (see below), NOT the whole repo.
- Dtypes (verified): `text_encoder/config.json` `dtype: bfloat16`; model card
  "transformer and text encoder use bfloat16; VAE and initial noise use float32";
  pipeline docstring: bf16 DiT/text-encoder weights, fp32 RoPE frequencies and
  phase/sin/cos then cast to video activation dtype, fp32 init noise, fp32 CFG +
  Euler updates, fp32 VAE decode. Per-tensor dtype: UNKNOWN (headers not fetched).
- `model_index.json` (verified): scheduler `pipeline_moss_image/MOSSImageScheduler`,
  text_encoder `transformers/Qwen3_5Model`, tokenizer `transformers/Qwen2Tokenizer`,
  transformer `pipeline_moss_image/MOSSImageTransformer`, vae `diffusers/AutoencoderKLFlux2`.

## 2. File list (from `https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Image-SFT`)

- `.gitattributes`, `.gitignore`, `LICENSE`, `README.md`, assets (`teaser.webp`,
  `comparison/`, `showcase/`, badges), `model_index.json`,
  `pipeline_moss_image.py` (440 lines, saved to `/tmp/moss_pipeline.py`),
  `scheduler/pipeline_moss_image.py`, `scheduler/scheduler_config.json`,
  `tokenizer/{chat_template.jinja, tokenizer.json, tokenizer_config.json}`,
  `transformer/{config.json, diffusion_pytorch_model.safetensors.index.json,
  diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors, pipeline_moss_image.py}`,
  `text_encoder/{config.json, model.safetensors.index.json,
  model-0000{1..4}-of-00004.safetensors}`,
  `vae/{config.json, diffusion_pytorch_model.safetensors}`.
- Weight-file sizes (verified): transformer index `total_size: 10164617280`
  (~10.2 GB ≈ 5.08 B bf16 params), 463 tensors; shard-00001 `x-linked-size:
  4991978016`. Text-encoder index `total_size: 16785390048` (~16.8 GB),
  759 tensors; shard-00001 `x-linked-size: 4973910744`. VAE single file
  `x-linked-size: 249521372` (~250 MB). Shard-00002/00003 and 00002..00004
  sizes: UNKNOWN (only shard-00001 HEAD-checked each).
- Weight names: transformer uses `video_embedder`, `text_embedder.{proj,blocks.0..1.*}`,
  `time_embedding.{0,2}`, `shared_block_modulation`, `video_head_modulation`,
  `layers.0..41.{norm1,norm2,q_norm,k_norm,proj.{q,k,v,o,ffn_up_gate,ffn_down}}`,
  `final_norm_video`, `final_linear_video`. Text encoder uses
  `language_model.{embed_tokens,norm,layers.0..31.*,}` plus `visual.*`
  (patch_embed/pos_embed/blocks.0..26/merger). VAE tensor names: UNKNOWN
  (no index file; single safetensors header not fetched).

## 3. Transformer (MOSSImageTransformer) — key dims (verified `transformer/config.json`)

`dim 3072`, `ffn_dim 8192`, `freq_dim 256`, `latent_channels 128`,
`num_heads 24` (head_dim = 128), `num_layers 42`, `rope_base 10000.0`,
`text_dim 4096`. Head-dim/8 = 16 RoPE bands; constructor requires dim % heads == 0,
head_dim % 8 == 0, even freq_dim.

Derived weight shapes (DERIVED from config + `pipeline_moss_image.py`, not tensor headers):
video_embedder `[3072,128]`+bias; text proj `[3072,4096]`+bias; per `_TextBlock`
qkv `[9216,3072]`+bias, proj `[3072,3072]`+bias, ffn_gate/up `[8192,3072]`+bias,
ffn_down `[3072,8192]` (+bias UNKNOWN — `nn.Linear` default; header not fetched);
time MLP `[3072,256]`+bias → SiLU → `[3072,3072]`+bias; shared_mod `[18432,3072]`+bias;
head_mod `[6144,3072]`+bias; per `_Block` q/k/v/o `[3072,3072]` no bias,
ffn_up_gate `[16384,3072]` no bias, ffn_down `[3072,8192]` no bias; norms are
vectors `[3072]` (block norm1/norm2, final), per-head q/k norms `[128]`;
final_linear `[128,3072]` no bias.

Block diagram (verified from pipeline source):
`[B,128,H,W]` latents + `[B,S,4096]` text + `[B,S]` binary mask →
video path: flatten→transpose→`video_embedder` (→ dim 3072);
text path: `_TextProjection` (Linear 4096→3072, then 2× `_TextBlock`: RMSNorm →
fused qkv → per-head QK-RMSNorm → joint SDPA with RoPE → proj → residual →
RMSNorm → SiTU FFN → residual);
time path: float64 sinusoid (`freq = 10000^(-arange(128)/128)`, cos|sin concat →
float32) → MLP → `shared_block_modulation` → `[B,6,3072]`,
`video_head_modulation` → `[B,2,3072]`;
per-sample (batch looped): `x = concat(video_tokens[H*W], text_tokens[S_valid])`
(padding excluded per-sample via mask; each sample needs ≥1 valid token);
42× `_Block`: adaLN `(1+scale)+shift` → q/k/v (weight dtype) → QK-RMSNorm →
joint self-attention over image+text (no cross-attention) with 4D RoPE →
o-proj → gated residual; adaLN → `ffn_up_gate` chunk(hidden,gate) →
SiTU → `ffn_down` → gated residual; `skip_layers` skips whole blocks;
head: video tokens only → final RMSNorm → adaLN (head mod) → final Linear →
`[B,128,H,W]`. RoPE `_RoPE`: fp32 CPU `bands`, coords `(r,f,h,w)`,
image grid centered (`h,w -= (H-1)/2,(W-1)/2`, `f = text_length`), text rows
`(0, idx, 0, 0)`; `sin|cos` flattened, `unsqueeze(0)`; recomputed per forward
from `(H, W, S_valid)`. Activation `_situ(h,g) =
(4·tanh(g/4)·sigmoid(g)) · (25·tanh(h/25))` — NEW (not SiLU/GELU/SwiGLU).

## 4. Text encoder (Qwen3_5Model; verified `text_encoder/config.json`)

- `transformers 5.12.1`, `Qwen2Tokenizer`, vocab `248320`, text path:
  hidden `4096`, 32 layers, 16 Q / 4 KV heads, head_dim `256`,
  intermediate `12288`, `hidden_act silu`, no attention bias,
  `attn_output_gate: true`, hybrid `layer_types`: 3× `linear_attention` +
  1× `full_attention` repeating (full at layers 3,7,11,15,19,23,27,31),
  linear-attn params: key heads 16×dim 128, value heads 32×dim 128,
  conv kernel 4, `mamba_ssm_dtype float32`; per-layer weight names include
  `self_attn.{q,k,v,o,q_norm,k_norm}_proj` (full layers) and
  `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,conv1d,
  A_log,dt_bias,norm}` (linear layers); partial rotary 0.25,
  rope_theta 10M, mrope interleaved section `[11,11,10]`; rms_eps 1e-6;
  `embed_tokens` + final `norm`; no lm_head use at inference.
  Internal SSM/recurrence math: UNKNOWN (transformers 5.x source not fetched).
- Vision tower present in weights/config (`depth 27`, hidden 1152, 16 heads,
  patch 16, merge 2, gelu_tanh, out 4096, `visual.*` tensors) but
  `encode_prompt` uses the text path (`AutoModel(**tok) → last_hidden_state[0]`);
  whether vision weights are ever executed in T2I: UNKNOWN.
- Encoding (verified): per-prompt unpadded forwards (comment: avoids padding
  effects in "Qwen GDN"), chat template `apply_chat_template(...,
  enable_thinking=True)`, truncation `max_length=max_sequence_length`
  (pipeline range 1..4096; tokenizer `model_max_length 262144`), zero-padded
  batch + boolean mask; negative prompt encoded the same way (default `''`).

## 5. VAE (verified `vae/config.json` + model card + pipeline `__call__`)

`AutoencoderKLFlux2`, in/out 3, latent 32, encoder `block_out [128,256,512,512]`,
decoder `[96,192,384,384]`, 2 layers/block, GroupNorm 32 groups, SiLU,
mid-block attention, quant/post-quant conv, patch `[2,2]`, sample 1024,
`force_upcast true`, BN eps 1e-4. Model card: "bundled VAE uses the FLUX.2 small
decoder". Decode-only at inference: BN-stat rescale
(`z·sqrt(running_var+eps)+running_mean`), reshape
`[B,C,H,W]→[B,C//4,2H,2W]` de-interleave, `vae.decode`, clamp, VAE-scale
factor 16 (`VaeImageProcessor(vae_scale_factor=16)`), so 1024px ↔ 64×64×32
latents (channels UNKNOWN: 32 vs 32/4=8 after reshape — code reshapes with
`c//4`; exact latent-channel convention: UNKNOWN, needs diffusers source).
Encoder weights ship but are unused at inference; KL/scaling_factor: UNKNOWN.

## 6. Sampler / scheduler (verified `scheduler_config.json` + pipeline source)

- `MOSSImageScheduler(FlowMatchEulerDiscreteScheduler)`: `num_train_timesteps
  1000`, `shift 5.0`, `training_shift 1.0`, `time_shift_type exponential`,
  `base_image_seq_len 256`, `max_image_seq_len 4096`, `base_shift 0.5`,
  `max_shift 1.15`; all of karras/beta/dynamic/exponential-sigmas false,
  `stochastic_sampling false`, `invert_sigmas false`.
- `set_timesteps(n=50)`: `grid(c,s)=s·linspace(1, s/(1000+s-1), c)/(1+(s-1)·linspace)`;
  inference grid (shift 5.0) snapped to nearest training-grid (shift 1.0) points
  (`unique_consecutive`), `_shift=1.0` to avoid double-shifting.
- `__call__` defaults: 1024×1024, 50 steps, `guidance_scale 3.0`,
  `uncond_skip_layers (22,)` (zero-based DiT index; negative branch only),
  SLG off (`slg_layers None, slg_scale 0.0, window [0,1)`), `negative_prompt ''`,
  dims must be positive multiples of 16. Forwards/step: 2 default (cond + skipped
  uncond); +1 when SLG active; `guidance_scale==1` → cond-only (+optional SLG
  pair). Combine (fp32): `uncond + g·(cond−uncond) [+ slg_scale·(cond−cond_skip)]`.
  Note `do_cfg = (guidance != 1.0)`, so 0 ≤ g < 1 also takes the CFG branch
  (verified in source). USERGUIDE adds: ≥50 steps, CFG range 3–6 (short prompts
  higher), Base-model negative tags `nsfw, worst quality, bad anatomy, watermark,
  jpeg artifact`, SLG variants (standard scale 3 / two-pass uncond-skip 22, also
  24/26 help), resolutions 1M/1.5M/2M px, extrapolate 0.25M–4M.

## 7. Prompt / layout conditioning (verified model card + USERGUIDE.md)

Inputs: natural language (English recommended; Chinese supported), tag sequences
(order Quality→Score→Meta→Year→Character→Series→Artist→General), or a JSON layout
string (`json.dumps(layout, ensure_ascii=False)` → `pipe`). Layout schema:
`summary.{meta[], canvas, caption}`, `items[]` with `{label: text|object, caption,
text? (render string), bbox: [y_min,x_min,y_max,x_max]}`,
`bbox_format "yxyx"`, `bbox_coord_space "abs_centralized"` (pixel coords, origin
at canvas center; 1024² → ±512). Text breaks: insert `\n` literally. Prompt
enhancer: external skill (`skills/moss-image-prompt-enhancer/SKILL.md`;
LLM-side, no device op). Caption format for post-training also documented
(long/medium/short + watermark boxes + borders) — training-only.

## 8. What KokkosDNN (worktree) already has

- `src/`: GEMM (`gemm`, incl. TPL ladder), `gemm_bias`, dense, embedding
  (incl. scaled gather), GQA/MHA bidirectional attention (+softcap/sliding/
  streaming/paged/KVCache), RoPE-2D axial, RMSNorm (+QK variant, one-plus),
  LayerNorm, activations (SiLU/GELU-tanh-exact/sigmoid/tanh/relu/softcap),
  SwiGLU, softmax, diffusion helpers (`adaln_modulate_forward`,
  `timestep_sinusoidal`), patch-embed/unpatchify, conv3x3/dense/grouped-1x1,
  depthwise 3x3 (+3D), Conv1D causal depthwise, MoE, sampler (LLM-side),
  tokens/utils. TPL policy (per headers): GEMM/attention = cuBLAS/cuDNN
  dispatch candidates; conv/norm/elementwise = native Kokkos.
- `examples/sana/`: Sana linear-DiT (28×1152: LiteLA linear self-attn 36×32,
  16-head cross-attn, GLUMBConv FFN ratio 2.5, adaLN-single, freq-256
  timestep MLP, caption proj 2304→1152 + scaled RMSNorm 0.01, patch 1,
  in/out 32ch), Gemma2-2B-IT encoder (26×2304, 8Q/4KV, hd 256, GQA + softcap 50
  + sliding/full + tanh-GELU MLP, FP32 Serial), DC-AE f32c32 VAE decoder
  (32ch, 6 stages ResBlock/EfficientViT, 32× compression, scale 0.41407),
  FlowEuler sampler (linear-flow shift schedule `shift 4.0`, steps 18–20,
  CFG 5.0, B=2 batched CFG, bit-portable Gaussian init).

## 9. Gap analysis — existing op → reuse vs NEW (TPL relevance)

| MOSS-Image need | Existing KokkosDNN op | Verdict | TPL relevance |
|---|---|---|---|
| All Linear/GEMM (qkv, proj, FFN, embedders, time/mod MLPs, text proj) | `gemm` / `gemm_bias` (+Dense) | REUSE | GEMM yes (cuBLAS) |
| Joint image+text self-attention (42×24, hd128, per-sample mask, SDPA) | `attention_mha_bidirectional` (+GQA/softcap infra) | REUSE/ADAPT (mask + concat layout; scale UNKNOWN — 1/√128 assumed, verify) | attention yes |
| Per-head QK RMSNorm (DiT + 2 text-refiner blocks) | `qk_rmsnorm` / `qk_rmsnorm3d`, `rmsnorm` | REUSE | norm no |
| Plain RMSNorms (block, final, text) | `rmsnorm` | REUSE | no |
| adaLN modulate + gated residuals (6-lane cond, 2-lane head) | `adaln_modulate_forward` (+Sana fused kernels as pattern) | REUSE | no |
| Timestep float64 sinusoid (freq 256) + SiLU MLP | `timestep_sinusoidal` | ADAPT (f64 freq path is new; verify existing precision) | no |
| 4D centered image+text RoPE (16 bands, fp32, per-(H,W,S) build) | `rope2d_axial_*` | NEW (different geometry/math) | no |
| SiTU activation `4·tanh(g/4)·σ(g)·25·tanh(h/25)` | SiLU/sigmoid/tanh pieces in Activation | NEW fused op (compose from REUSE pieces) | no |
| Fused qkv (text blocks) vs split q/k/v/o (main blocks) | Sana weight-ori/transpose shims | REUSE pattern | GEMM yes |
| Patch embed (latent 128ch) / unpatchify | `patchembed_*`, `unpatchify_forward` | REUSE/ADAPT (patch-2 VAE path TBD) | no |
| Qwen3.5 text encoder full-attn layers (16Q/4KV, hd256, partial RoPE, gated output?) | GQA + RoPE + RMSNorm + SwiGLU infra | ADAPT (partial-rotary + `attn_output_gate` are new) | attention yes / rest no |
| Qwen3.5 linear-attention layers (in_proj_qkv/z/a/b, conv1d k4, A_log/dt_bias, norm) | `depthwise_conv1d_causal_*` (closest) | NEW (recurrence/SSM math UNKNOWN — needs transformers-5.x spec) | no (verify; likely native) |
| Token embedding (vocab 248320×4096) + chat-template/mask host logic | `embedding_gather_scaled`, `Tokens` | REUSE (device) + NEW host glue | no |
| FLUX.2 VAE decoder (GroupNorm-32, SiLU, mid-attn, quant/post-quant conv, 2×2 patch) | Conv family + Sana DC-AE VAE as pattern | NEW decoder (GroupNorm op NOT found in `src/*.hpp` inventory; BN-stat rescale + de-interleave reshape are new glue) | conv/norm no |
| FlowEuler→nearest-grid scheduler + skip-layer CFG/SLG logic | Sana `FlowEuler` (shift-4 linear schedule) | ADAPT host-side (new grid snap + `skip_layers` plumbing) | n/a (host) |
| Layout-JSON / prompt-enhancer / tokenizer | LLM `Sampler`/`Tokens` (host patterns) | NEW host-only glue (no device op) | n/a |
| bf16 DiT/text-encoder + fp32 accum/norms/RoPE/noise | Sana path is FP32-first | GAP: verify bf16 coverage of GEMM/attention/norms before porting | GEMM/attention yes |

Largest NEW device work: (1) 4D RoPE, (2) SiTU FFN, (3) Qwen3.5 linear-attn
layer, (4) FLUX.2 VAE decoder (incl. GroupNorm), (5) joint masked
attention batching. Largest REUSE: GEMM ladder, bidirectional attention core, RMSNorm/QK-norm,
adaLN, embeddings, conv family, FlowEuler CFG skeleton.

## 10. Open unknowns (do NOT assume)

Per-tensor shapes/dtypes (headers not fetched); shard sizes except first of each;
VAE latent/BN/scale-factor conventions; attention scale factor; text FFN bias;
vision-tower execution; SSM math; prefill-vs-paged strategy for the 9B encoder;
tokenizer chat-template exact string; ComfyUI/Gradio parity details; training
(captioning/LoRA/DMD) — out of scope for inference port.
