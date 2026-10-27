# Qwen-Image-2.1-Turbo DiT: native vs diffusers reference parity analysis

READ-ONLY analysis (no source edits, no GPU/torch run). Authoritative reference:
`diffusers 0.42.0.dev0` at
`/home/appuser/.local/lib/python3.13/site-packages/diffusers`.

Files read end to end:

| Role | Path |
| --- | --- |
| torch transformer | `.../diffusers/models/transformers/transformer_qwenimage21.py` (1034 lines) |
| torch pipeline | `.../diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` (861 lines) |
| torch scheduler | `.../diffusers/schedulers/scheduling_flow_match_euler_discrete.py` |
| torch emb/norm prims | `.../diffusers/models/embeddings.py:1027`, `.../models/normalization.py:510` |
| native DiT | `models/qwen_image_21/KokkosDNN_QwenImage21Dit.hpp` (1170 lines) |
| native config | `models/qwen_image_21/KokkosDNN_QwenImage21Config.hpp` |
| driver / sampler adapter | `examples/qwen_image_21/qwen_image_21_generate.cpp` |
| core ops | `src/KokkosDNN_{Gemm,Attention,LayerNorm,Activation}.hpp` |
| golden | `scripts/qwen_image_21_golden.py`, `scripts/qwen_image_21_golden_checksums.txt` |
| checkpoint config | `weights/Qwen-Image-2.1-Turbo/transformer/config.json` |
| golden artifacts | `weights/Qwen-Image-2.1-Turbo-golden/` (+ `-golden-run.log`) |

Geometry for the golden run: B=1, latent 64x64x64ch -> N=4096 tokens, C=64,
D=4096, 32 heads x 128, S(text)=19, T=S+N=4115, 32 blocks, mlp 12288.

Line-number stability: `models/qwen_image_21/KokkosDNN_QwenImage21Dit.hpp` is
untouched in the working tree (mtime 12:59), so its numbers are exact.
`examples/qwen_image_21/qwen_image_21_generate.cpp` and
`scripts/qwen_image_21_golden.py` are being edited concurrently (parity dumps +
the caption row-count fix landed *during* this analysis); their line numbers are
for the state read here, the quoted code and flags are not.

---

## 1. Reference call graph (T2I, `use_kv_cache=True`)

```
pipeline_qwenimage21.__call__
  encode_prompt -> _get_qwen_prompt_embeds              L240-337
    prompt_template_t2i (system block dropped, _drop_idx)  L219-235
    text encoder hidden_states[-1] with final RMSNorm neutralized (hook) L300-315
    prompt_embeds_mask all-ones -> encode_prompt sets it to None  L390
  prepare_latents: randn_tensor(dtype=bf16) -> _pack_latents [B,N,C]  L416-491
  img_shapes = [[(1,64,64)]], image_pad_mask = [F*19, T*1024] -> append_target_slots
                                                          L721-754
  retrieve_timesteps(sigmas=sample_sigmas, mu=...)        L743
    sigma = f32(sample_sigmas); timesteps = sigma*1000 (f32)  scheduler L349,376
  for i,t in enumerate(timesteps):                        L769
    kv_mode = "extract" (i==0) else "cached"              L775
    timestep = t.expand(B).to(bf16)                       L782
    transformer(hidden_states, timestep/1000, context, img_shapes, img_mask,
                kv_cache, kv_cache_mode)                  L784-795
    noise_pred = noise_pred[:, -N:]                       L796
    latents = scheduler.step(...)  # f32 add, round to bf16  L818

QwenImage21Transformer2DModel.forward                     L869-1034
  hidden_states   = img_in(x)                             L922   (Linear 64->4096, no patch)
  context         = txt_in(ctx)                           L923
                    = out_layer(GELU_tanh(in_layer(ZeroCenterRMSNorm(ctx))))   L188-200
  repeats         = where(img_mask,4,1) == 1 (T2I)        L927
  joint           = cat([context(S), zeros(N/4)]) -> repeat_interleave -> scatter img rows
                    == [text(S) | target(N)]              L931-939
  rotary          = pos_embed(img_shapes, image_pad_mask) L941   (3-axis, [T,64] complex)
  timestep        = cat([t_bf16, zeros(1)])               L948   (B+1 rows: sampled, t=0)
  modulation_mask = target_token_mask                     L949
  temb            = time_text_embed(timestep)             L952   (1000*t, sin/cos 256, MLP)
  modulation      = Sequential(SiLU, Linear(4D))          L953   ([2, 4D])
  prefix_len      = T - N = 19                            L967
  segments        = [(0,19,is_text=True)]  (AttnProcessor path)  L376-395 / L508
  32 x QwenImage21TransformerBlock                        L587-652
  norm_out        = LN_noaffine(x) * (1 + linear(silu(temb_row)))  L1028 / L215-235
  output          = proj_out(all rows)                    L1029
```

Attention, exact prefill segmentation (default `QwenImage21AttnProcessor`, not flex):

```
text segment  : q=rows[0:19], k/v=rows[0:19], mask = tril(19x19)   L514-533
target block  : q=rows[19:4115], k/v=ALL rows, mask = None         L535-543
q/k/v         : Linear (no bias) -> unflatten(-1, 32,128)          L336-345
              -> RMSNorm(128) -> to(bf16)                          L347-348
              -> RoPE complex, fp32 inside, back to bf16           L116-131, L351-353
softmax scale : SDPA default 1/sqrt(128)
```

---

## 2. Side-by-side op table

`torch` = `transformer_qwenimage21.py` unless noted; `native` =
`models/qwen_image_21/KokkosDNN_QwenImage21Dit.hpp`.

| # | op | torch | native | verdict |
| --- | --- | --- | --- | --- |
| 1 | patch embedding | `img_in` `nn.Linear(64,4096)` at L799/L922; `patch_size=1` -> **no patching**, x is already `[B,N,C]` (packed by pipeline L416) | `gemm(img_b[N,64], img_in_W[4096,64])` L915-918 | **match** |
| 2 | text projection | `ZeroCenterRMSNorm(4096,1e-6)` = `x*rsqrt(mean(x^2)+eps)*(w+1)`, **no mean removal** (L170-186); `Linear->GELU(tanh)->Linear` (L197-200) | `qwen21dit_zero_center_norm` L146-169, `gemm->gelu_tanh->gemm` L899-913 | **match** (same formula, same eps) |
| 3 | joint layout | `[text(S) | target(N)]`; scatter at `image_pad_mask` (L931-939) | `[text | target]` concat L964-971 | **match** |
| 4 | timestep sinusoid | `time_factor=1000` * `t` then `[cos(128) | sin(128)]`, 256-d (L136-167); freqs `exp(-ln(1e4)*i/128)` | `1000*timestep*inv`, `[cos | sin]`, F=256 (L877-893) | match **except bf16 quantization of `t` (risk R3)** |
| 5 | timestep MLP | `TimestepEmbedding(256->4096)`: Linear -> SiLU -> Linear, `sample_proj_bias=False` (embeddings.py:1027-1072); input cast to bf16 | `gemm(time1)->silu->gemm(time2)`; sinusoid stored as `Scalar` L920-935 | **match** |
| 6 | t=0 / sampled split | `cat([t,0])`, `_select_modulation_rows`: rows `[0,B)` real, `[-1]` t=0, text rows take t=0 (L238-257, L948-949) | 2-row `s3`/`mod4`; `a_z..g2_z` = row 1 (L939-962); per-region split at `S` (L989, L1028, L1042) | **match** |
| 7 | shared modulation | `Sequential(SiLU, Linear(4096->16384))`, lane order `[mod1.scale, mod1.gate, mod2.scale, mod2.gate]` (L802, L631, L613) | `silu(t1)->gemm(mod_W)`, `a=0:D, g1=D:2D, b=2D:3D, g2=3D:4D` L940-962 | **match** |
| 8 | block input norm | `img_norm1/2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)` (L602, L642) | `layernorm_noaffine_forward` (LayerNorm.hpp:219, no weight/bias, biased var, eps inside sqrt) L987, L1031 | **match** |
| 9 | block modulation | `x*(1+scale)`, `gate` applied as `tanh(gate)*upd` (L613-616, L644, L647) | `apply_scale_split` L989; `gated_residual_split` = `x+tanh(g)*u` L1028, L1042 | **match** (rounding lanes differ, R4) |
| 10 | attention q/k/v | `to_q/k/v` no bias, `unflatten(-1,(H,hd))` (L336-345) | 3 GEMMs [T,D] + `[T,H,128]` reinterpret L993-998 | **match** |
| 11 | qk norm | `RMSNorm(128, eps=1e-6)`, fp32 variance, `*.weight` (normalization.py:541-560), `.to(value.dtype)` | `qwen21dit_qk_rmsnorm` fp32 sum-sq, `*w` L216-239 | **match** (rounding lanes differ, R4) |
| 12 | RoPE | 3 axes `[16,56,56]`, theta 1e4, `pos_index=arange(8192)`, `neg_index=-1..-1024` (L655-707); text: p0=p1=p2=token index; target: p0=19, p1=h-(H-H/2), p2=w-(W-W/2); complex multiply in fp32 (L116-131) | `qwen21dit_rope_3axis` L242-291: identical positions, identical freq law, real-pair rotation | **match** for square/even grids (verified offsets -32..31 for 64x64) |
| 13 | attention | text: causal 19x19; target: full over 4115 keys; scale 1/sqrt(128) (L508-543) | `attention_gqa_single(q_txt,k_pre,v_pre,q_off=0)` (causal) + `attention_mha_bidirectional(q_tgt,k_full,v_full)` L1019-1025 | **match** structurally; different kernels (R5) |
| 14 | attn out proj | `to_out[0]` Linear, Dropout(0) (L461) | `gemm(o2f, to_o)` L1027 | **match** |
| 15 | MLP | SwiGLU: `out(silu(gate_layer(x)) * proj(x))`, hidden 12288 (L203-212) | `gemm(proj)->p2`, `gemm(gate)->g2`, `swiglu(g2,p2)` = `silu(g)*p` L1036-1041 | **match** (chunk order correct) |
| 16 | fp16 clip | `if dtype==float16: clip(-65504,65504)` (L649-650) | skipped | **match** for bf16 (clip is a no-op at bf16) |
| 17 | final AdaLN | scale-only: `LN_noaffine(x) * (1+linear(silu(temb)))`, **uses `temb`, not `modulation`**; row-selected t=0/sampled (L233-235, L1028) | `norm_out_W(silu(s3))` -> `ns2` rows, `apply_scale_split(ns_z, ns_s)` L1046-1065 | **match** (easy to get wrong; native is right) |
| 18 | output proj | `proj_out` on **all** rows, pipeline slices `[-N:]` (L1029, pipeline L796) | `gemm(hn_tgt[N,D], proj_out_W[64,D])` — target rows only L1066-1070 | **match** in value; breaks the golden checksum comparison (R1) |
| 19 | precision | bf16 params/activations, fp32 accumulate in Linear/SDPA/norm, bf16 rounding after every op | `DScalar=bhalf_t`, cuBLAS `CUBLAS_COMPUTE_32F` (Gemm.hpp:33,751), fp32 elementwise | **match** in class; per-op rounding lanes differ (R4) |
| 20 | pad-key mask | `encoder_hidden_states_mask` -> `joint_key_valid` (L956-965); **None** for this golden (`prompt_embeds_mask.all()` -> None, pipeline L390) | none implemented | **match for this golden only**; unexercised path |
| 21 | Euler step | `f32(sample) + (sigma_next-sigma)*model_output`, round to bf16 (scheduler L493-533); sigmas include terminal 0 | `x = bf16(float(x) + ds*float(v))`, `ds = f32(sig[i+1]-sig[i])` driver L1418-1440 | **match** |
| 22 | sampler inputs | `noise_pred = out[:, -N:]`, `latents` bf16, sigma[0]=1.0 | driver slices/packs identically L1486-1520 | **match** |

Verified claims from the commit messages (**all confirmed against the source**):
`[text|target]` layout; affine-free block/final LayerNorm; `time_factor` 1000
applied once (raw-sigma contract, no double scale); t=0 vs sampled modulation
split incl. the `[-1]` row; 3-axis RoPE position and frequency tables; block-causal
prefill as two segments; 297 checkpoint tensors / no biases (index.json matches
`qwen21_dit_expected_tensors` exactly).

Non-issues worth not "fixing": no patch embedding (patch_size=1); `img_in` has no
patching; fp16 clip absent is correct for bf16; `norm_out` consuming `temb`
(not `modulation`) is correct; the final AdaLN has **no shift**.

---

## 3. Ranked divergence risks

Impact numbers marked *measured* were computed on CPU with numpy against the real
checkpoint weights (bf16 rounded exactly, FP32 accumulation), no GPU/torch forward.

### R1 — P0 (measurement, not numerics): the step-0 golden checksum covers 4115 rows, the native emits 4096

* torch: `assert` step 0 `noise_pred` is `(1, 4115, 64)` — the **raw transformer
  output before the pipeline slice** (hook fires at `transformer_qwenimage21.py:1029`,
  the slice happens later at `pipeline_qwenimage21.py:796`).
  `scripts/qwen_image_21_golden_checksums.txt`: `step 0 ... f64sum=-21028.175584077835`.
* native: `out` is `[B,N,C]`, i.e. 4096 rows (`Dit.hpp:1066-1070`), dumped as
  `step00_noise_pred.f32` (4096x64).
* Consequence: comparing the two f64 sums differs by the 19 text rows
  (19*64 = 1216 of 263,360 elements = 0.46 % of the sample count). With the
  observed overall element mean (-21028/263360 = -0.0798) the text rows alone
  contribute ~0.4-3 % depending on their mean. **This is almost certainly the
  reported "within 0.6 %"** — an artifact, not parity.
* Fix: golden must export the target slice (`npred[0, -4096:]`) *and* the text rows
  separately; native should also emit the text-row output (see E6) so the step-0
  tensor can be compared 1:1.

### R2 — P0 (control of inputs): the two sides are not fed identical inputs yet

Three independent offenders, any of which swamps the DiT analysis:

1. **Context.** Golden = bf16 text-encoder output encoded on **CPU**
   (`prompt_embeds[0,0] f64sum=-66.24072265625`). The manifest itself records a
   previous **GPU** encode of the same prompt at `-167.2802734375` — a 2.5x change in
   the row sum. The native path runs its own bf16 encoder on GPU. Until the golden
   `prompt_embeds` is fed to the native DiT, step-0 differences cannot be attributed.
2. **Initial latents.** Golden = `torch.randn` on a CPU `Generator(42)`, cast to
   **bf16** (`pipeline_qwenimage21.py:485-487` with `dtype=prompt_embeds.dtype`).
   The native default is its own `mt19937` gaussian (`driver:2143`) — a different
   noise field, i.e. everything differs. `--init-latents-in` (driver:2131-2141) fixes
   this, but the golden must first export `latents` in `[C,H,W]` C-major.
3. **Row count of `--prompt-embeds-in` (was a blocker, now fixed).** At
   analysis time `EncoderStage::run` forced `st.L = be_max_seq(cfg)` = **512**
   (`driver:549`, old `1294`), so a 19-row golden file was rejected; padding to 512
   rows is *not* a workaround because the 493 zero rows become real text tokens in
   the joint sequence (T becomes 4608) and the target block attends to them.
   The working tree now carries the fix (`load_caption_raw(path, 0, Cc)`: infer
   `L = nbytes/4/Cc`, clamp against `be_max_seq`) — keep that clamp, and never feed
   a zero-padded context.

### R3 — P1 (real DiT divergence, steps >= 1 only): reference double-rounds the timestep through bf16

* torch: `timestep = t.expand(B).to(latents.dtype)` -> `bf16` (`pipeline:782`), then
  `timestep/1000` -> `bf16` again (`pipeline:786`), then
  `self.time_factor * timestep.float()` (`transformer:149`). So the effective
  argument is `1000 * bf16(bf16(1000*sigma)/1000)`, not `1000*sigma`.
* native: `1000.0 * timestep * inv` in **double** from an exact `sigma`
  (`Dit.hpp:886`), where `sigma = float(1000*sigma_f32)/1000` (driver:1514).
* *Measured* (numpy, exact bf16 rounding):

  | sigma | t=1000s (f32) | bf16(t) | ref eff | native eff | delta | rel |
  | --- | --- | --- | --- | --- | --- | --- |
  | 1.000000 | 1000.00 | 1000.0 | 1000.00000 | 1000.00000 | 0.000 | 0 % |
  | 0.978453 | 978.453 | 980.0 | 980.46875 | 978.45300 | +2.016 | 0.21 % |
  | 0.954180 | 954.180 | 956.0 | 957.03125 | 954.17999 | +2.851 | 0.30 % |
  | 0.926626 | 926.626 | 928.0 | 929.68750 | 926.62604 | +3.062 | 0.33 % |
  | 0.895080 | 895.080 | 896.0 | 894.53125 | 895.07996 | -0.549 | -0.06 % |
  | 0.845148 | 845.148 | 844.0 | 843.75000 | 845.14801 | -1.398 | -0.17 % |
  | 0.704534 | 704.534 | 704.0 | 703.12500 | 704.53400 | -1.409 | -0.20 % |
  | 0.414568 | 414.568 | 414.0 | 414.06250 | 414.56799 | -0.506 | -0.12 % |

  Channel 0 of the sinusoid has frequency 1.0, so these are **phase errors of up to
  3.06 rad** — that channel is effectively decorrelated. Downstream *measured*
  effect with the real `linear_1/linear_2/modulation` weights:
  `||d temb||/||temb|| = 2.1-2.5 %`, `||d modulation||/||modulation|| = 0.4-0.6 %`,
  `rms(d tanh(gate)) = 1.2e-3` against `rms(tanh(gate)) = 0.475` (~0.26 %),
  `rms(d attn_scale) = 0.008` against `rms(scale) = 0.60` (~1.3 % of the scale).
  That is a systematic, every-token, every-block perturbation of the conditioned
  image rows: expect ~0.3-1 % output drift **per step** for steps 1..7.
* **Step 0 is unaffected** (sigma = 1.0 is bf16-exact both ways), which is exactly
  why this hides in a step-0-only comparison and then shows up in the trajectory.
* Fix: replicate the quantization in `forward()` before the sinusoid:
  ```cpp
  // reference lane: t -> bf16, /1000 -> bf16, then *time_factor
  float t32  = (float)(1000.0 * timestep);          // t as f32
  Scalar tq  = static_cast<Scalar>(t32);            // bf16(t)
  Scalar q   = static_cast<Scalar>((float)tq / 1000.0f);  // bf16(t/1000)
  double eff = 1000.0 * (double)(float)q;          // time_factor * sigma
  // and for bit-exactness the angle multiply is f32 in the reference:
  //   arg = (float)(eff) * (float)inv;   (1e-4 rad at arg=1000, below 1 bf16 ulp)
  ```
  Gate it (e.g. `Config::timestep_bf16_quantize`, default true for bf16 builds) so
  fp32 builds keep the exact contract. Put it in **exactly one place**: if the
  driver pre-quantizes sigma to bf16 (`DitForward`, driver:1514) and the model then
  multiplies by 1000, you get a *third* value (0.978453 -> driver 0.9785156 -> model
  978.5156) which matches neither the reference (980.46875) nor the exact 978.453.
  Do the same in a scalar-only test so it is pinned without a checkpoint.
  Rationale for matching this artifact rather than being "more accurate": the
  golden is defined by this lane, and no torch run with exact timesteps exists to
  compare against, so leaving it out converts a defined target into a permanent
  0.3-1 %/step systematic difference on steps 1..7.

### R4 — P2: per-op bf16 rounding lanes (sets the achievable floor, ~1e-3)

The native performs each elementwise combination in one fp32 expression and rounds
once; torch rounds at every op boundary. Concretely:

| op | torch rounding | native | delta |
| --- | --- | --- | --- |
| `x + tanh(g)*u` (L644, L647) | `tanh` -> bf16, product -> bf16, add -> bf16 (3 roundings) | 1 rounding (`Dit.hpp:346-397` gated residual) | ~1 ulp of the update per element, per block |
| `x*(1+scale)` (L616, L235) | `1+scale` -> bf16, product -> bf16 | 1 rounding (`Dit.hpp:303-344` apply_scale) | scale ~0.6 -> multiplier quantized to 0.2 % |
| qk RMSNorm (L347, normalization.py:541) | `x*rrms` -> bf16, then `*weight`(bf16) -> bf16 | 1 rounding (`Dit.hpp:217-240`) | ~1 ulp on q/k |
| `silu`/`gelu`/`tanh` | torch bf16 ops use fp32 opmath, round to bf16 | same (fp32 compute, `Scalar` store) | match |
| LayerNorm | fp32 stats (Welford), bf16 out | fp32 two-pass stats, bf16 out | match (~1e-7) |

Expected aggregate: ~1 ulp per element per block -> `rel_fro ~ 1e-3` at the output
with random-walk accumulation over 32 blocks. Cheap to tighten (add an explicit
`Scalar` round after `tanh(g)` and after `tanh(g)*u`), but not the cause of an O(1 %)
mismatch.

### R5 — P2: attention kernel differences (unavoidable, quantify it)

text segment: torch SDPA with a bool `[1,1,19,19]` mask (mem-efficient/math path)
vs native cuDNN SDPA with a **bottom-right causal** graph
(`Attention.hpp:1409-1430`, `T=19 > 1` so it takes the TPL path, and
`q_off=0, Ktot==T` makes bottom-right == top-left). The streaming fp32
online-softmax kernel (`bidir_expf`, degree-5 exp ~2.5e-6 rel err) is only the
fallback when cuDNN is off/declines. Target block: torch flash/efficient SDPA over
4096x4115 vs native `try_cudnn_sdpa_bidir` (`Attention.hpp:2386-2402`). Both
fp32-accumulate, so the difference is ~1e-3 relative on the attention output —
same order as R4.

### R6 — P2: the text path has no isolated probe (and is only 0.5 % of the keys)

Because text rows attend only to text keys and use the t=0 modulation rows, **the
text rows' hidden states are independent of the sampled sigma and of the image
latents at every layer**. Their `proj_out` is therefore an invariant, and it is
already inside the golden's step-0 tensor (rows `[0,19)`). Emitting it from the
native gives a free, isolated test of `txt_in` + 32 blocks of the t=0 lane + the
head, with no interaction with the image path. Cost: one 19-row GEMM (only when
dumping).

### R7 — P2: unexercised `/ unimplemented` reference paths (do not matter now, will later)

* `encoder_hidden_states_mask` / `joint_key_valid` (L956-965): Native has no key
  mask. Fine for this golden (all-ones -> `None`), but a padded-prompt golden will
  diverge, and **padding the context to 512 rows as a workaround for R2.3 is wrong**.
* KV-cache decode (`kv_cache_mode="cached"`, L970-985) is out of scope by design;
  step 1..7 comparisons must therefore be prefilled independently (no cache in the
  native), which changes the *sequence length* the reference sees at steps 1..7
  (reference: 4096 queries x 4115 keys from cache; native: 4115 queries). Values
  agree (target rows attend to everything either way), but see E7.
* The flex-attention path (`QwenImage21FlexAttnProcessor`) is *not* the default
  (`_default_processor_cls = QwenImage21AttnProcessor`); the `create_block_mask`
  128-token quantization is irrelevant unless someone switches processors.

---

## 4. Experiment plan (localization ladder)

Conventions. The golden script already writes the driver's own contract
(`git diff scripts/qwen_image_21_golden.py`, section "(f) parity artifacts", on by
default): `prompt_embeds.f32 [L,Cc]`, `initial_latents.f32 [C,H,W]` C-major
`lat[(c*H+h)*W+w]`, `stepNN_noise_pred.f32 [C,H,W]` (image tokens only),
`stepNN_latents.f32 [C,H,W]`, and `step00_noise_pred_full.f32 [T,C]` for the
un-sliced prefill. **Both sides are therefore byte-comparable without transposes**
and `initial_latents.f32` can be fed straight to `--init-latents-in`. The only
transform left is slicing/splitting `step00_noise_pred_full.f32`
(`[0:S)` = text-row canary, `[S:T)` = what the pipeline consumes).

### E0 — plumbing gate (must pass bit-exactly before anything else)

Prerequisite `--prompt-embeds-in` row-count inference is in the working tree
(R2.3); verify the driver logs `[caption ...] 19x4096 features (L inferred from
file)`.

```sh
# golden (GPU, workstream B)
python3 scripts/qwen_image_21_golden.py \
  --weights /workspace/weights/Qwen-Image-2.1-Turbo \
  --out /tmp/q21gold                      # (f) artifacts default ON

# native
build/qwen-image/examples/qwen_image_21/qwen_image_21_generate \
  --weights /workspace/weights/Qwen-Image-2.1-Turbo/transformer \
  --prompt-embeds-in /tmp/q21gold/prompt_embeds.f32 \
  --init-latents-in  /tmp/q21gold/initial_latents.f32 \
  --height 64 --width 64 --sigmas turbo --cfg 1.0 --staged \
  --dump-steps /tmp/q21nat
```
Pass criteria (exact, no tolerance):
* `/tmp/q21nat/prompt_embeds.f32` == `/tmp/q21gold/prompt_embeds.f32`
  (whole-tensor f64sum; the golden manifest records it as
  `prompt_embeds_f64sum`)
* `/tmp/q21nat/initial_latents.f32` == golden `= -62.91889151930809` (manifest
  `initial_latents_f64sum`)
* device-side cross-check of the golden artifacts themselves: the *full* step-0
  tensor must still sum to `-21028.175584077835` (manifest
  `step00_noise_pred_full_f64sum`) while the new image-only
  `step00_noise_pred` sum is a different number — that delta **is** R1 quantified
  for this prompt.
If either file is not bit-identical, the handoff (row count / layout / bf16 cast)
is wrong — stop; nothing after this is interpretable.

### E1 — step 0, target slice

```sh
python3 <path>/qwen21_cmp.py /tmp/q21gold /tmp/q21nat   # full source in §7
```
The tool prints, for each pair: `cos`, `rel_fro`, `maxabs`, native rms, and both
f64 sums, plus the golden step-0 text/target/full sum breakdown.
Calibration (justified in §3/R4): `rel_fro <= 2e-3` and `cos >= 0.99999` = at the
bf16 rounding floor (done); `rel_fro 2e-3..1e-2` = one systematic lane difference,
go to E2-E5; `rel_fro >= 2e-2` = structural bug.
Do **not** gate on the f64 sum alone: it is cancellation-dominated
(`|sum| = 21028` over 263k elements whose mean is `-0.0798`), so 1-ulp noise moves
it by O(1) while a systematic 0.05 % bias moves it by O(10).

Measured on the golden artifact (`weights/...-golden/..._final_latents.f32`,
CPU/numpy): `rms = 0.981`, `maxabs = 4.4375`, and **every element is exactly
bf16-representable** — i.e. the whole trajectory lives in the bf16 lane, and the
native (also bf16) can in principle match it element-for-element. With rms ~1 and
sum|v| ~ 2e5 against a signed sum of 2.1e4 (10x cancellation), pure rounding noise
moves an f64 sum by ~sqrt(263360)*1e-3 ~ 0.4; anything at the 10-100 level is
systematic. That is why R1's row-set artifact (~100) looks exactly like a "0.5 %"
numeric result.

### E2 — conditioning/projection (one tensor isolates three ops)

Dump `joint_in = [text(S) | target(N)]` right after `txt_in`/`img_in`/concat
(native `Dit.hpp:964-971`), compare rows `[0,S)` (isolates `txt_in` + norm + GELU)
and `[S,T)` (isolates `img_in` + layout). Torch: forward hooks on
`transformer.txt_in` and `transformer.img_in` (`[1,19,4096]`, `[1,4096,4096]`) and
concatenate, or write `joint_in.f32` directly with a pre-hook on
`transformer_blocks[0]`. Expect exact bf16 agreement for the target rows (pure
GEMM, only accumulation order differs: `rel_fro < 1e-6`).

### E3 — timestep/modulation (step 0 must be near-exact; the R3 canary)

Dump `temb` (`[2,4096]`) and `modulation` (`[2,16384]`) in the native at
`Dit.hpp:936` / `950`; hook `transformer.time_text_embed` and
`transformer.modulation`. At step 0: expect `rel_fro < 1e-3`, row 1 (t=0) should
be **bit-identical** (sinusoid is a constant there). Running the same comparison at
step 1 should show the predicted ~2 %/0.5 % gap — that validates R3 empirically.

### E4 — locate the first diverging block

Dump `blockNN_joint_out` for `NN in {0,1,15,31}` (native: host copy of `joint` after
the block's second gated residual, `Dit.hpp:1042`); torch: forward hooks on
`transformer_blocks[NN]` (`[1,4115,4096]`). Report target rows and text rows
separately. First block whose target rows exceed `rel_fro 2e-3` while its input
matched => bug inside that block; if block 0 already exceeds while `joint_in` and
`modulation` match => attention/RoPE/MLP (then dump `q`/`k` post-RoPE and
`attn_out` for block 0 only).

### E5 — head

Dump `norm_out_y` (`[T,4096]`, after the scale, `Dit.hpp:1065`) and
`proj_out_all` (`[T,64]`, all rows — see E6). If blocks match but the output does
not, the error is in the `temb`-row selection for `norm_out` or in the target-row
slicing.

### E6 — free text-row canary

`proj_out_all[:19]` must equal the golden's
`step00_noise_pred_full.f32[:19]` **and** the text rows of every later step's
pre-slice output (sigma-invariant, because text rows attend only to text keys and
read the t=0 modulation rows). Perfect isolation of the text/t=0 lane: a mismatch
here with tight image rows points at `txt_in`/t=0 lanes only.

### E7 — trajectory (after E1 is clean)

`stepNN_latents.f32` is written by both sides in the same `[C,H,W]` layout, so it
compares directly; `step07_latents` should also match
`weights/Qwen-Image-2.1-Turbo-golden/qwen_image_21_turbo_final_latents.f32`
(`f64sum=12098.474563598633`, rms 0.981, all values bf16-exact). Caveat: from
step 1 on the reference decodes from the KV cache (4096 queries x 4115 keys) while
the native re-prefills (4115 queries); the target-row *values* are equivalent but
the kernels differ, so expect slightly larger `rel_fro` there. A drift that grows
monotonically per step while every step-0 tensor matches is the R3 signature.

---

## 5. Proposed dump surface (minimal, actionable)

Already landed (no work): the golden's `(f) parity artifacts` block, the driver's
`--dump-steps` (`prompt_embeds`, `initial_latents`, `stepNN_noise_pred`,
`stepNN_latents`, `manifest.json`) — same layout on both sides — and the
`--prompt-embeds-in` row-count inference (was R2.3 / the E0 blocker).

Still missing, in priority order:

| flag | effect | blocks |
| --- | --- | --- |
| `--dump-blocks 0,1,15,31` | `blockNN_joint_out.f32` `[T,D]` (+ `blockNN_attn_out` / `blockNN_mlp_out`) | E4 |
| `--dump-text-rows` | `proj_out_all.f32` `[T,C]`, plus `step00_text_rows.f32` `[S,C]` | E5, E6 |
| default with `--dump-steps` | `joint_in.f32` `[T,D]`, `temb.f32` `[2,D]`, `modulation.f32` `[2,4D]`, `norm_out_y.f32` `[T,D]` | E2, E3 |
| `final_latents.f32` in the dump dir | copy `--out <stem>.f32` (currently only next to the PNG) | E7 |

Naming: reuse `blockNN_*` (block index, not step) and keep the existing
`stepNN_*` prefix for per-step tensors so `qwen21_cmp.py` finds everything.

Model API: a `const DitDumpHook*` last argument on
`QwenImage21DitModel::forward` (`Dit.hpp:839`) with
`std::function<void(const char*, const Scalar*, size_t rows, size_t cols)>`; copy
to host only when set (bf16 -> f32), so the default path stays allocation-free.
Dump names must match the torch file names byte for byte.

Torch side (no diffusers patch — forward hooks only; keep step 0 only, since
`txt_in`/`modulation`/blocks re-run every step):
```python
STEP = [0]           # bumped by the existing transformer-output hook
def hook(name, kind='rows'):
    def f(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        t = t.detach()[0] if t.dim() == 3 else t.detach()
        if STEP[0] != 0:
            return
        t.to(torch.float32).cpu().numpy().tofile(f"{OUT}/{name}.f32")
        print(f"[dump] {name} {tuple(t.shape)} f64sum={f64sum(t):.17g} "
              f"rms={t.float().pow(2).mean().sqrt().item():.6g}")
    return f
pipe.transformer.time_text_embed.register_forward_hook(hook("temb"))
pipe.transformer.modulation.register_forward_hook(hook("modulation"))
for i in (0, 1, 15, 31):
    pipe.transformer.transformer_blocks[i].register_forward_hook(hook(f"block{i:02d}_joint_out"))
pipe.transformer.norm_out.register_forward_hook(hook("norm_out_y"))
pipe.transformer.proj_out.register_forward_hook(hook("proj_out_all"))
# joint_in: pre-hook on transformer_blocks[0] -> args[0] is the joint tensor
pipe.transformer.transformer_blocks[0].register_forward_pre_hook(
    lambda m, a, kw: (STEP[0] == 0 and
        a[0][0].detach().to(torch.float32).cpu().numpy().tofile(f"{OUT}/joint_in.f32")))
```
Recommended additions to every printed checksum (both sides): `f64sum` (already),
plus `rms` and `maxabs`; for step 0 keep **two** sums (target slice, text rows) —
the full 4115-row sum alone is what produced the misleading "0.6 %".

The comparison tool (full source in §7; scratch copy at
`/tmp/opencode/qwen21_cmp.py`, self-tested on synthetic data with an exact
round-trip) compares the same-layout files pair by pair and prints `cos`,
`rel_fro`, `maxabs`, rms and both f64 sums, plus the golden step-0
full/text/target sum breakdown.

---

## 6. If the implementer only changes three things

1. Stop comparing against the 4115-row f64 sum. The image-only `step00` artifact
   (already landing in the golden script) removes R1; the full-tensor sum stays only
   as a cross-check that the new dump reproduces `-21028.175584077835`. Re-measure
   before believing the existing "0.6 %".
2. Pin the inputs (row-count inference has landed): run E0 with the golden
   `prompt_embeds.f32` + `initial_latents.f32` — it either passes bit-exactly or
   names the plumbing bug. Every later number is meaningless if E0 is not exact.
3. Replicate the reference's bf16 timestep double rounding (R3) — it is the only
   *known* systematic DiT divergence, it is invisible at step 0, and it accounts
   for ~0.3-1 % per step from step 1 on.

---

## 7. Appendix: comparison tool (`qwen21_cmp.py`)

Both sides already share one on-disk layout, so this compares file-to-file. It also
prints the golden step-0 breakdown (full / text / target) that quantifies R1.
Self-tested: feeding it identical tensors on both sides reports
`cos=1.000000 rel_fro=0.000e+00` and identical f64 sums.

```python
#!/usr/bin/env python3
"""Compare KokkosDNN --dump-steps tensors with the torch golden artifacts.

Both sides use the SAME on-disk layout (scripts/qwen_image_21_golden.py
"(f) parity artifacts" and examples/qwen_image_21/README.md "--dump-steps"):

  <dir>/prompt_embeds.f32        [L, Cc]    row-major
  <dir>/initial_latents.f32      [C, H, W]  lat[(c*H+h)*W+w]
  <dir>/stepNN_noise_pred.f32    [C, H, W]  image tokens only
  <dir>/stepNN_latents.f32       [C, H, W]  post-Euler latents
  <dir>/stepNN_joint_out.f32     [T, C]     (proposed per-block dump)
  <dir>/stepNN_norm_out_y.f32    [T, C]     (proposed head dump)
  golden only: step00_noise_pred_full.f32 [T, C], T = text_len + N

so no transposes are needed; only the full step-0 prefill tensor needs a row
slice (target rows) or a row split (text rows = the sigma-invariant canary).

Usage:
  qwen21_cmp.py <golden_dir> <native_dump_dir> [--C 64 --H 64 --W 64]
"""
import argparse
import os
import numpy as np


def rd(path, n=None):
    v = np.fromfile(path, dtype='<f4')
    if n is not None and v.size != n:
        raise SystemExit(f"{path}: {v.size} floats, expected {n}")
    return v


def report(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        print(f"{name:38s} SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    d = a - b
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    print(f"{name:38s} cos={float((a*b).sum()/(na*nb+1e-30)):.6f} "
          f"rel_fro={np.linalg.norm(d)/(nb+1e-30):.3e} "
          f"maxabs={np.abs(d).max():.3e} rms_n={np.sqrt((a*a).mean()):.4e} "
          f"f64_nat={a.sum():.17g} f64_gold={b.sum():.17g}")


def stats(name, v):
    a = np.asarray(v, np.float64)
    print(f"{name:38s} rms={np.sqrt((a*a).mean()):.6e} maxabs={np.abs(a).max():.4e} "
          f"f64sum={a.sum():.17g} n={a.size}")


def pair(g, n, name, note=''):
    gp, np_ = os.path.join(g, name), os.path.join(n, name)
    have_g, have_n = os.path.exists(gp), os.path.exists(np_)
    if not (have_g and have_n):
        if have_g or have_n:
            print(f"{name:38s} missing on the "
                  f"{'native' if have_g else 'golden'} side")
        return False
    report(name + note, rd(np_), rd(gp))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('golden')
    ap.add_argument('native')
    ap.add_argument('--C', type=int, default=64)
    ap.add_argument('--H', type=int, default=64)
    ap.add_argument('--W', type=int, default=64)
    ap.add_argument('--text-len', type=int, default=19)
    args = ap.parse_args()
    C, H, W, S = args.C, args.H, args.W, args.text_len
    N = H * W
    g, n = args.golden, args.native

    print('== E0: plumbing (both lines must be exact, no tolerance) ==')
    pair(g, n, 'prompt_embeds.f32')
    pair(g, n, 'initial_latents.f32')

    print('== step 0 prefill: golden full [T,C] vs native image-only dump ==')
    full = os.path.join(g, 'step00_noise_pred_full.f32')
    if os.path.exists(full):
        v = rd(full)
        if v.size == (S + N) * C:
            v = v.reshape(S + N, C)
            stats('golden step00 full (old checksum row set)', v)
            stats('golden step00 text rows [0:S]   (canary)', v[:S])
            stats('golden step00 target rows [S:T]', v[S:])
            nat0 = os.path.join(n, 'step00_noise_pred.f32')
            if os.path.exists(nat0):
                img = rd(nat0).reshape(C, H, W).transpose(1, 2, 0).reshape(N, C)
                report('step00 target: native vs golden', img, v[S:])
            txt = os.path.join(n, 'step00_text_rows.f32')
            if os.path.exists(txt):
                report('step00 text:   native vs golden',
                       rd(txt, S * C).reshape(S, C), v[:S])
        else:
            print(f"  step00_noise_pred_full.f32: {v.size} floats, "
                  f"expected {(S+N)*C}")

    print('== per-step velocity / latents (image tokens) ==')
    for step in range(16):
        names = (f'step{step:02d}_noise_pred.f32', f'step{step:02d}_latents.f32')
        if not any(os.path.exists(os.path.join(d, nm))
                   for nm in names for d in (g, n)):
            if step == 0:
                continue
            break
        for nm in names:
            pair(g, n, nm)

    print('== per-block / head dumps (if present) ==')
    for blk in range(64):
        if not pair(g, n, f'block{blk:02d}_joint_out.f32', '  [target]'):
            break
    for name in ('norm_out_y.f32', 'proj_out_all.f32', 'joint_in.f32',
                 'temb.f32', 'modulation.f32'):
        pair(g, n, name)

    print('== final latents ==')
    for name in ('final_latents.f32', 'qwen_image_21_turbo_final_latents.f32'):
        if pair(g, n, name):
            break
    else:
        print('final_latents.f32                      '
              'copy the driver --out <stem>.f32 into the dump dir')


if __name__ == '__main__':
    main()
```

Notes:
* the native driver writes the final sampler output next to `--out`
  (`<stem>.f32`, `[C,H,W]`) rather than into the dump dir; copy or symlink it as
  `final_latents.f32` to have E7 checked automatically;
* the golden's `step00_noise_pred_full.f32` must stay un-sliced so the tool can
  split text rows `[0,19)` from target rows `[19,4115)`;
* `--C/--H/--W/--text-len` are only for non-1024px smoke runs.
