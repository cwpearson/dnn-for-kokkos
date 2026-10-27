# Qwen3-VL text-encoder parity: analysis + verification plan

Status: analysis only. No source edits, no commits, no GPU/torch runs.
Scope: `models/qwen_image_21/KokkosDNN_QwenImage21Encoder.hpp` (native Qwen3-VL-8B
text tower) vs the torch golden produced by `scripts/qwen_image_21_golden.py`
(`diffusers==0.42.0.dev0`, `transformers==5.20.2`).

> Note on the driver: `examples/qwen_image_21/qwen_image_21_generate.cpp` is
> being edited by another workstream while this was written; its line numbers
> moved twice during the analysis. All driver line refs below are approximate
> (`~`) and should be re-checked before use.

---

## 0. TL;DR

1. **The 19 conditioning rows are explained exactly** by the reference recipe:
   raw T2I template, `drop_idx = 14`, `hidden_states[-1]` = last *decoder-layer*
   output (pre-final-norm), bf16, no output projection. I re-derived the
   reference token count from the checkpoint's own `tokenizer.json`
   (`33 tokens = 14 system + 3 "<|im_start|>user\n" + 11 prompt + 5 tail`,
   `33 - 14 = 19`), which independently confirms the template, the tokenizer and
   the drop index without running torch.
2. **The encoder math matches `modeling_qwen3_vl.py` v5.20.2 step for step** in
   all structural respects (see §2). The only divergences are bf16 rounding
   sites (RoPE applies cos/sin in bf16 in torch vs fp32-then-round natively;
   attention is fp32-accumulate natively vs bf16 softmax/PV in the torch math
   path; GEMM tiling order differs). No mask/token/scale bug was found.
3. **`hidden_states[-1]` really is pre-norm** in the golden env
   (`transformers` `@capture_outputs` + `_can_record_outputs`), so the encoder's
   "neutralized final norm" is correct. The header comment that "transformers>=5.0
   ties `hidden_states[-1]` to the post-norm value" is wrong for v5.20.2 and
   should not be used as a reason to change the encoder.
4. **The documented CPU-vs-GPU prompt-embed difference is expected bf16
   reduction-order noise, amplified by massive activations and by the missing
   final norm**, not evidence of a token/mask bug. Row-sum checksums are a
   pathological metric for these tensors (see §3) and must not be used as the
   parity criterion.
5. **The decisive measurement is one DiT-only run** with golden
   `prompt_embeds.f32` + golden `initial_latents.f32`. Today the driver cannot
   consume a 19-row embed file (`load_caption_raw` demands exactly
   `max_seq_len * context_dim = 512*4096` floats), so Option A needs the small
   loader change proposed in §5. Option B (encoder-only) needs **no** driver
   change: `--dump-steps DIR` already writes `prompt_embeds.f32`.

---

## 1. How the 19 rows are produced, on both sides

### 1.1 Reference (torch)

The installed `diffusers 0.42.0.dev0` tree could not be read from this sandbox
(`/` outside `/workspace` is permission-denied), and upstream
`huggingface/diffusers@main` does not contain `QwenImage21Pipeline` at all
(`src/diffusers/pipelines/qwenimage/__init__.py` lists only the Qwen-Image /
Edit / Layered pipelines). What *is* verified:

* `model_index.json` pins `QwenImage21Pipeline`, `Qwen3VLForConditionalGeneration`,
  `Qwen3VLProcessor`, `FlowMatchEulerDiscreteScheduler`.
* `processor/tokenizer_config.json`: `Qwen2Tokenizer`, `pad_token=<|endoftext|>`,
  `eos=<|im_end|>`, `bos=null`, `split_special_tokens=false`.
* `processor/chat_template.jinja` renders, for
  `[{system: S}, {user: P}]` + `add_generation_prompt=True`:
  `<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{P}<|im_end|>\n<|im_start|>assistant\n`
  (jinja lines 22–34, 39–58, 118–120), i.e. *exactly*
  `qwen21_format_prompt_t2i()` (`Encoder.hpp:112-125`). The "raw template, not
  `apply_chat_template`" note is a red herring: for this fixed system message the
  two are byte-identical.
* The public sibling pipeline
  (`diffusers@pipeline_qwenimage.py:174-226`) shows the inherited recipe:
  ```python
  self.prompt_template_encode = "<|im_start|>system\n{...}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
  self.prompt_template_encode_start_idx = 34          # :176
  txt_tokens = self.tokenizer(txt, max_length=self.tokenizer_max_length + drop_idx,
                              padding=True, truncation=True, return_tensors="pt")   # :201
  out = self.text_encoder(input_ids=..., attention_mask=..., output_hidden_states=True)
  hidden_states = out.hidden_states[-1]               # :209
  split = [e[drop_idx:] for e in split_masked(hidden_states, mask)]               # :211
  prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype)                 # :221
  ```
  with `tokenizer_max_length=1024` (no padding for a single prompt because
  `padding=True` pads to the longest). For Qwen-Image-2.1 the system message is
  `"Comprehend and analyze the provided prompt."` and `drop_idx = 14`; the
  golden's `(1, 19, 4096)` shape is consistent with that and with nothing else
  (see §1.2).
* Dtype: text encoder loaded `dtype=torch.bfloat16`; embeds stay bf16
  (`pe.dtype == torch.bfloat16` in the golden log). The DiT context dim equals
  the encoder hidden dim (4096), so **no projection/cast to the DiT dim** — the
  only cast is bf16 → the DiT's own `txt_in` (ZeroCenterRMSNorm + Linear, inside
  the DiT, `Dit.hpp:899-913`).

### 1.2 Token arithmetic (independent confirmation, no torch needed)

Using only `weights/Qwen-Image-2.1-Turbo/processor/tokenizer.json` (GPT-2
byte-level vocab, space = `Ġ`), for
`"A capybara wearing a wizard hat, oil painting"`:

| segment | pieces | ids |
|---|---|---|
| system block | `<|im_start|>` `system` `\n` … `<|im_end|>` `\n` | 151644 8948 198 … 151645 198 |
| sys message | `Comp` `reh` `end` `Ġand` `Ġanalyze` `Ġthe` `Ġprovided` `Ġprompt` `.` | 13552 11063 408 323 23643 279 3897 9934 13 |
| user prefix | `<|im_start|>` `user` `\n` | 151644 872 198 |
| prompt | `A` `Ġcap` `yb` `ara` `Ġwearing` `Ġa` `Ġwizard` `Ġhat` `,` `Ġoil` `Ġpainting` | 32 2062 84307 5059 12233 264 33968 8896 11 5590 18824 |
| tail | `<|im_end|>` `\n` `<|im_start|>` `assistant` `\n` | 151645 198 151644 77091 198 |

* system block = **14** ids (`1+1+1+9+1+1`)
* user prefix = **3**, prompt = **11**, tail = **5** → total **33**
* `33 - 14 = 19` = the golden's sequence length, and golden step-0 forward is
  `4115 = 4096 + 19` rows. Two independent goldens agree.

The two multi-piece words are pinned by vocab membership: `Comprehend` has no
single token (`rehend`/`hend` absent, `Comp`/`reh`/`end` present), and `capybara`
has no 1- or 2-piece spelling (`Ġcapyb`/`capyb`/`capy`/`Ġcapybara` all absent,
`Ġcap`+`yb`+`ara` present). Any other split changes the total and would break
`19`. → **template, system prompt, tokenizer, and drop_idx=14 are correct.**

Surviving rows 0..18 = token indices 14..32, i.e. row 0 is the user-turn
`<|im_start|>` and row 18 is the trailing `\n` — matching the driver comment.

### 1.3 Native

`examples/.../qwen_image_21_generate.cpp` `EncoderStage::run_qwen` (~1193-1280):

1. loads `tokenizer.json` into raw tables (`load_raw_qwen_tables`), tokenizes the
   **whole** `qwen21_format_prompt_t2i(prompt)` string with
   `encode_qwen_raw(tabs, text, max_len=512)` (~1224): added-special longest
   match over raw bytes, then NFC → Qwen2 Split regex → rank-merge BPE;
2. re-tokenizes the system block alone and `<|im_start|>user\n` alone and asserts
   `head_ids == sys_ids ++ prefix_ids` (~1211-1222) — a good self-check;
3. `drop = sys_ids.size()` (= 14 here), `S = ids.size()` (= 33);
4. runs `QwenImage21EncoderModel<bhalf_t>::forward(handle, toks[1,S], hid[1,S,D], drop)`
   on the **full 33 tokens**, positions 0..32;
5. copies rows `drop..S-1` to `st.embeds` as f32 (exact bf16→f32), `st.L = 19`;
6. `stage_device()` → f32 views, later cast to `bhalf_t` for the DiT (lossless).

This matches the reference's "encode the full 33, then slice `[drop_idx:]`".

---

## 2. Encoder forward, side by side

Reference: `transformers==5.20.2`, `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
(line numbers from the v5.20.2 tag). Native:
`models/qwen_image_21/KokkosDNN_QwenImage21Encoder.hpp`.

| step | reference (torch) | native | verdict |
|---|---|---|---|
| embedding | `embed_tokens(input_ids)`, no scale (`:815`) | `embedding_gather` (`:722`) | ✅ |
| M-RoPE positions | text-only: `arange(S)` broadcast ×3 (`:796`, `:807`) | `pos = t` (`:502`) | ✅ (unpadded only) |
| RoPE inv_freq | `1/base**(arange(0,head_dim,2)/head_dim)`, theta 5e6 (`:389`) | `theta^(-2j/Hd)`, j<Hd/2 (`:472-480`) | ✅ |
| M-RoPE recomposition | interleaved copy from rows H/W, no-op for text-only (`:410-415`) | plain RoPE (documented no-op, `:482-487`) | ✅ |
| RoPE pairing | `rotate_half` = half-split `(d, d+Hd/2)`, then `cat(f,f)` (`:196`) | `(a,b)=(x[j],x[j+Rh])`, same rotation (`:508-509`) | ✅ |
| RoPE precision | `cos/sin` cast to **bf16** (`:404`), then bf16 mul/add (`:196`) | fp32 angle/rotate, **one** bf16 rounding (`:503-510`) | ⚠️ rounding (native is more accurate) |
| RMSNorm (input/post) | `fp32 → mean(x²) → rsqrt → .to(bf16) → * weight` (`:431-433`) | identical two-rounding form (`:540-544`) | ✅ (sum order differs) |
| RMSNorm eps | 1e-6 (`config`) | `norm_eps=1e-6` (`:61`) | ✅ |
| q/k/v proj | `nn.Linear(bias=False)` fp32-accum bf16 GEMM (`:492-506`) | `gemm`, fp32 accumulate (cuBLAS 32F / native fp32) | ✅ (tiling order) |
| q/k norm | `Qwen3VLTextRMSNorm(head_dim=128)` on the **last** dim, before RoPE (`:508-509`) | `qwen21_qk_rmsnorm` over `Hd`, before rope (`:734-737`) | ✅ |
| scaling | `head_dim**-0.5` = 1/√128 (`:249`) | `1/sqrt(Hd)` (`:699`) | ✅ |
| mask | causal 4-D mask from `create_causal_mask`; all-ones padding mask (`:807`) | causal by construction, `q_off=0` (`:740`) | ✅ (no pads) |
| GQA | `repeat_kv` = repeat_interleave, q head h → kv head h//4 | h·Hkv/Hq (`Attention.hpp:1484`) | ✅ |
| softmax | `softmax(..., dtype=float32).to(query.dtype)` in eager (`:232`); SDPA internally | FP32 online softmax, FP32 PV accum (`Attention.hpp:1485-1502`) | ⚠️ rounding (native more accurate) |
| o_proj + residual | `o_proj`, `residual + hidden` (`:534`, `:587`) | `gemm` + fp32 add→bf16 (`:742-744`) | ✅ |
| MLP | `down(silu(gate(x)) * up(x))`, no bias (`:550`) | `swiglu(g,u)` = `silu(g)*u`, then `down` (`:747-753`) | ✅ gating order |
| MLP precision | bf16 silu, bf16 product | `Scalar(silu(g))`, then `Scalar(sg*u)` (`:596-598`) | ✅ |
| final norm | `last_hidden_state = self.norm(h)` (`:840`) **but** `hidden_states[-1]` = last decoder layer output (pre-norm) because `@capture_outputs` records each `Qwen3VLTextDecoderLayer` and prepends embeds (`_can_record_outputs`, `:622`; `output_capturing.py`) | final norm skipped (`:756-761`) | ✅ **pre-norm is right** |
| lm_head | `logits = self.lm_head(post_norm_hidden)` (`:1383`) — computed but unused for conditioning | `lm_head.weight` skipped at load (`:353-356`) | ✅ (no effect on embeds) |
| context projection | none (4096 = 4096) | none | ✅ |
| output dtype | bf16 | bf16 (`bhalf_t`), dumped as f32 (exact) | ✅ |

### 2.1 Divergences that matter (all numerics, none structural)

| # | divergence | where | expected effect |
|---|---|---|---|
| D1 | RoPE cos/sin applied in bf16 in torch vs fp32 rotate + single bf16 round natively | `Encoder.hpp:503-510` vs `modeling_qwen3_vl.py:404,196` | ~1 bf16 ulp on q/k for a subset of elements; feeds D3 |
| D2 | attention: torch bf16 softmax weights + bf16 PV per matmul (math/SDPA path) vs native FP32 online softmax | `Attention.hpp:1485-1502` | native is *more* accurate; reference-side error is part of the "noise floor" |
| D3 | pre-norm output means massive activations (≈1e4 on sink channels, bf16 ulp 64) go straight into the conditioning, so last-bit differences are visible | `Encoder.hpp:756` | the reason row sums move by ~100 (§3) |
| D4 | GEMM tiling/split-k and RMSNorm reduction order differ (fp32 both sides) | `Gemm.hpp`, `Encoder.hpp:536-539` | ~1e-6 relative per op, bf16-rounds to ±1 ulp sometimes |
| D5 | on CUDA builds with `KOKKOSDNN_ENABLE_CUDNN`, bf16 prefill may go to cuDNN SDPA instead of the native kernel (`Attention.hpp:1505-1516`) | build config | another kernel family; makes native results build-dependent. **Record it in the manifest.** |

Non-parity observations (not bugs for this prompt):

* **N1 (edge case, real):** reference truncation is
  `tokenizer(max_length=tokenizer_max_length + drop_idx)` *then* drop
  (`pipeline_qwenimage.py:201-211`), i.e. up to 512 content rows for Qwen-Image.
  The driver truncates the full string to `max_len = be_max_seq(cfg) = 512`
  **before** dropping (`generate.cpp ~1224`), yielding at most **498** content
  rows. Divergence only for prompts ≳498 tokens; fix by tokenizing to
  `512 + drop` (or truncating after the drop).
* **N2 (perf, not parity):** `qwen21_rmsnorm` recomputes the full row sum for
  every `(t,d)` (`:532-545`) → O(T·D²) instead of O(T·D); ~40 GFLOP/forward of
  redundant work. Same structure as some other headers in the tree; not a
  correctness issue.
* **N3 (documented):** left-padded batches are not masked (`:663-669`). Fine for
  the single unpadded golden; would silently corrupt a batch. The reference *does*
  pad (`padding=True`) but with one prompt there are no pads.
* **N4 (comment wrong):** `Encoder.hpp:635-637` claims transformers ≥5.0 makes
  `hidden_states[-1]` post-norm. In v5.20.2 it is pre-norm (§2 table). Fix the
  comment so nobody "corrects" the encoder.

---

## 3. Why the CPU-encoded and GPU-encoded prompt embeds differ

`scripts/qwen_image_21_weights_manifest.txt:64-67` records
`[0,0] -66.24072265625 → -167.2802734375` and
`[0,-1] 487.0107421875 → 473.9169921875` for the same prompt, and calls it
"bf16 CPU vs CUDA kernel differences". That is the right conclusion, and here is
the quantitative reason plus one thing the note does not say.

**The metric is the problem.** Both numbers are `float64` sums over 4096 bf16
values. A change of +101 in a row sum is a change of `101/4096 = 0.025` in the
*mean* element. If the row has RMS `s`, the relative perturbation is
`0.025/s` — a fraction of a percent for any plausible pre-norm RMS (5–50). The
[0,0] row sum is only 66 in magnitude, i.e. the sum almost cancels, so it looks
like a 150 % change while the vector barely moved. `[0,-1]` moved 2.7 % in the
same run. **Row sums say nothing about parity here.**

**Where the absolute magnitude comes from.** `hidden_states[-1]` is pre-final-norm,
so Qwen3's "massive activation" channels (the attention-sink channels, ~1e4 in the
early layers per the header's own measurement note at `Encoder.hpp:518-521`) are
present unnormalized in the output. At 8192–16384, one bf16 ulp is 64–128. The
underlying fp32 difference produced by D1/D2/D4 is ~1e-3 relative, which is
*below* one ulp for those channels most of the time and *flips* one when the value
sits near a rounding boundary. One or two flips per row = ±64..128 in the row
sum, with a per-row relative L2 of well under 1 %. That is exactly the observed
pattern (`[0,0]` +101, `[0,-1]` −13).

**Why row 0 is hit hardest.** Row 0 of the conditioning is token 14, which attends
to only 15 keys — tokens 0..14, including the sequence's sink positions — whereas
row 18 spreads attention over 19 keys and 14 more layers of mixing. The row
closest to the sink inherits the most sink-channel magnitude, hence the largest
absolute-sum sensitivity. This is *also* why a `drop_idx` off by one would be
catastrophic (row 0 would be a different token ***and*** see a different prefix):
it is not off — §1.2 proves 19 = 33 − 14.

**Verdict:** expected bf16 reduction-order chaos, not a bug. But it is not *only*
"bf16 kernel differences": it is bf16 kernels **×** unnormalized massive
activations. Two consequences:

1. The canonical reference must be the **CPU-encoded** embeds (the ones that
   produced the versioned latents in `qwen_image_21_golden_checksums.txt`).
2. The tolerance must be set from a **measured reference noise floor**
   (`rel-L2(torch CPU embeds, torch CUDA embeds)`), not from intuition.

---

## 4. Verification design

### 4.1 Tensors to compare (all already-defined formats)

| id | native side | reference side | layout |
|---|---|---|---|
| **T1** prompt embeds | `--dump-steps DIR` → `DIR/prompt_embeds.f32` (dumped from `enc.device`, f32 of the bf16 values, `generate.cpp ~2191-2200`) | `prompt_embeds.f32` written by `qwen_image_21_golden.py` (`:366-368`) | `[L,Cc]` row-major f32 LE, L=19, Cc=4096 |
| **T2** initial latents | `--init-latents-in golden/initial_latents.f32`; dumped back as `DIR/initial_latents.f32` | `initial_latents.f32` (`:373-374`) | `[C,H,W]` C-major, 64×64×64 |
| **T3** step-0 velocity | `DIR/step00_noise_pred.f32` | `step00_noise_pred.f32` (`:396`) | `[C,H,W]` C-major |
| **T4** final latents | `DIR/step07_latents.f32` / the always-written `<stem>.f32` | `step07_latents.f32` / `qwen_image_21_turbo_final_latents.f32` | `[C,H,W]` C-major |

`DIR/prompt_embeds.f32` is only produced when the in-process encoder actually ran;
the manifest records `"prompt_source": "qwen"` (`enc.source`). ⚠️ The current
golden directory (`/workspace/weights/Qwen-Image-2.1-Turbo-golden/`) contains
only the final latents `.f32`/`.npy` and the PNG: the run predates
`--dump-artifacts` (the log has no `[f]` lines). **T1/T3 must be regenerated**
before anything else.

### 4.2 Metrics (compute with numpy, f32 inputs, f64 accumulation)

For each pair `(A native, G golden)`, same shape:

```
rel_l2      = ||A-G||_F / ||G||_F
row_rel_l2  = per-row ||A_i-G_i|| / ||G_i||   (report max)
row_cos     = <A_i,G_i> / (||A_i|| ||G_i||)   (report min)   # scale-invariant
row_norm_r  = ||A_i|| / ||G_i||               (report max |log ratio|)
max_abs     = max |A-G|
f64_sum     = sum(A) vs sum(G)                # log only, never a pass criterion
```

`row_cos` / `row_norm_r` are the *decision* metrics for the encoder because the
DiT normalizes each context row (`Dit.hpp:148-175`, ZeroCenterRMSNorm, per-row
RMS, weight+1): the DiT only sees row **direction**. A uniform per-row scale
error in the encoder is invisible downstream; an angular error is not.

### 4.3 Thresholds

Calibrate once, then freeze:

* **F** = `rel_l2(torch CPU embeds, torch CUDA embeds)` for the same prompt —
  the reference noise floor. (Two runs of the golden script with
  `--placement manual` vs `--placement full`, or the small embeds-only script in
  §5.1. Estimate from the published sums: F ≈ 0.5–2 %; **measure it.**)
* **Encoder parity (T1), PASS:** `rel_l2 ≤ max(2 %, 2·F)`, `min row_cos ≥ 0.999`,
  `max row_rel_l2 ≤ 5 %`, `max |log row_norm_r| ≤ 0.10`.
* **Encoder parity, STRONG:** `rel_l2 ≤ 1 %`.
* **Treat as a bug, not noise:** `min row_cos < 0.99`, any row-norm ratio outside
  ±20 %, or `rel_l2 > 10 %` — that is a template / drop / tokenizer / post-norm
  mistake, not rounding.
* **DiT-only (T3), PASS:** `rel_l2(step00_noise_pred) ≤ 1 %` (bf16 GEMM noise
  should be ~0.2–0.5 %); **T4** (final latents) `≤ 2 %`.
* Reference self-check before using any golden file: T1's `[0,0]` and `[0,-1]`
  f64 sums must equal `-66.24072265625` and `487.0107421875` to the last digit;
  T2's f64 sum must be `-62.91889151930809`; T4's `12098.474563598633`.

### 4.4 The attribution experiment (why it answers the question)

Run, all with `--sigmas turbo --cfg 1.0 --steps 8 --height 64 --width 64`:

| run | conditioning | init latents | measures |
|---|---|---|---|
| **R1** | in-process encoder (`--tokenizer` + `--text-encoder`) | golden T2 | encoder + DiT + sampler (≈ today's 1.88 %) |
| **R2** | golden T1 (`--prompt-embeds-in`) | golden T2 | **DiT + sampler only** |
| **R3** | in-process encoder | golden T2 | duplicate of R1, for a same-build stability check |

Then:

* `E_enc  = rel_l2(R1 T1, golden T1)` — the encoder, in isolation.
* `E_dit  = rel_l2(R2 T3, golden T3)` — a **single** DiT forward, no error
  accumulation: the cleanest DiT number.
* `E_e2e  = rel_l2(R2 T4, golden T4)` — DiT + 8 Euler steps.
* `E_full = rel_l2(R1 T4, golden T4)` — today's 1.88 %.

Attribution rule (errors independent, first order): `E_full² ≈ E_enc² + E_e2e²`.
With `E_full = 1.88 %`:

* if `E_dit ≈ 0.5 %` then `E_enc ≈ 1.8 %` → **encoder-dominated**;
* if `E_e2e ≈ 1.8 %` then `E_enc ≈ 0.5 %` → **DiT-dominated** (and the encoder is
  already at the noise floor).

So `E_dit` is the single decisive number. Nothing else needs to be measured first.

**Prerequisite:** R1/R2/R3 must use the golden T2 file. If the current 1.88 %
baseline used the driver's own `mt19937` noise instead of `--init-latents-in`,
the number is meaningless (a different initial noise changes the image
completely, not by 1.9 %). Re-run with the golden T2 and re-baseline.

### 4.5 Sanity checks that cost nothing

* `rel_l2(R1 T2, golden T2)` must be **0** (bitwise). If not, the sampler's
  x_init packing is wrong and every other number is poisoned.
* Token ids: compare the driver's ids with §1.2. One line in the manifest would
  make this permanent.
* Attention path: record whether `KOKKOSDNN_ENABLE_CUDNN` was on for the run
  (D5); a cuDNN-on build and a cuDNN-off build are two different references.

---

## 5. Proposed driver/script changes (not applied — driver is owned elsewhere)

Priority order, smallest diff first.

1. **`--prompt-embeds-in` must accept `L < max_seq` (blocker for Option A).**
   `Qwen21::load_caption_raw` (`generate.cpp ~467-485`) returns only when
   `n == expect_L*expect_Cc` (512·4096); a 19-row file throws. Minimal change:
   ```cpp
   if (out.size() % (size_t)expect_Cc == 0 &&
       out.size() / (size_t)expect_Cc <= (size_t)expect_L)
     return out;                        // [L,Cc] with L <= max_seq
   ```
   and in `EncoderStage::run` set `st.L = out.size() / st.Cc` instead of keeping
   `be_max_seq`. The README already documents the contract as `[L,Cc]`, and
   `run_sample_loop_from_device` already takes `L = cond_d.extent(0)` and only
   bounds-checks `L <= be_max_seq` (`~1510-1515`), so nothing else moves.
2. **`--encode-only` (or `--dump-embeds FILE`).** Run tokenize + encoder, dump
   `[L,Cc]` f32 + a sidecar JSON (`tokens`, `drop_idx`, `S`, `L`, template id),
   `release()`, exit before the DiT load. Value: encoder parity in ~1 min
   without 14 GB of DiT, and it runs while the GPU is otherwise busy.
3. **Manifest additions** (`--dump-steps`): `"tokens": [...]`, `"drop_idx": N`,
   `"seq_len": S`, `"cond_len": L`, `"attention_path": "native"|"cudnn"`,
   `"backend"`. Makes every parity run self-describing and turns §1.2 into an
   automated check.
4. **Truncation fix (N1):** tokenize to `be_max_seq(cfg) + drop` and truncate
   after the drop. No effect on the golden prompt.
5. **Optional `--decode-latents-in FILE`**: feed golden final latents straight to
   the VAE decode to separate VAE error from sampler error (only worth it if
   R2's T4 is tight but the PNG PSNR is not).
6. **New script `scripts/qwen_image_21_embeds.py`** (CPU-only, text encoder only,
   ~18 GB RAM, no DiT/VAE): tokenizer + the exact `encode_prompt` recipe
   (`hidden_states[-1]`, `[drop_idx:]`, dtype bf16), writes `prompt_embeds.f32`,
   `prompt_ids.json`, and asserts the published `[0,0]`/`[0,-1]` sums. With
   `--device cuda` it also produces the GPU variant → **F** in §4.3, in one
   process, without the 30 GB pipeline.
7. **New script `scripts/qwen_image_21_compare.py`**: the §4.2 metric table for
   any two `.f32` dumps, so the numbers are reproducible and reviewable.
8. **New test `tests/Test_QwenImage21Encoder.cpp`**: tiny-config forward
   determinism + the `qwen21_*` helpers (rope table vs `theta^(-2j/Hd)`,
   double-rounding RMSNorm, SwiGLU order). Cheap regression insurance; there is
   currently **no** test for this encoder (only `Test_MossImageEncoder`).

Do **not** touch: the template strings, `drop_idx` derivation, or the pre-norm
final-norm skip — all three are confirmed correct.

---

## 6. Top risks, ranked

1. **No reference embed dump exists** (T1/T3). Everything else is blocked on
   regenerating them; the published f64 sums are the only ground truth today.
2. **Option A is blocked** by `load_caption_raw`'s exact-512-rows check. Without
   the fix, the DiT can never be tested independently, and the 1.88 % stays
   unattributed.
3. **The 1.88 % baseline may not be using the golden initial latents.** If it is
   not, it is not a parity number at all. Verify `initial_latents.f32` first.
4. **Reference noise floor unknown.** A 2 % tolerance may be simultaneously too
   loose to catch a real encoder bug and too tight for a legitimate GPU build.
   Measure F (§4.3), and decide the pass criterion from it.
5. **`KOKKOSDNN_ENABLE_CUDNN` changes bf16 attention numerics** (D5). Two builds
   can disagree by the same order as the reference. Freeze the build config for
   parity runs and record it.
6. **DiT-side: `rope_theta = 10000` is an explicit assumption**
   (`QwenImage21Config.hpp:66`, "not published ... golden validation TODO"). If
   `E_dit` comes back large, this is the first thing to check, not the encoder.
7. **Comment drift**: `Encoder.hpp:635-637` mis-states the transformers
   `hidden_states[-1]` semantics; a future reader could "fix" the correct
   pre-norm skip into a bug.

## 7. What could not be verified here

* The installed `diffusers 0.42.0.dev0` source: the sandbox denies reads outside
  `/workspace`, and upstream `main` has no `QwenImage21Pipeline`. The
  `encode_prompt` behaviour is therefore inferred from the public
  `QwenImagePipeline` (§1.1) and pinned by the golden's `19` rows (§1.2) — not
  read line-by-line. If the pipeline is available elsewhere, confirm exactly
  three things: the template string, `prompt_template_encode_start_idx == 14`,
  and `tokenizer_max_length` (512 vs 1024).
* Any measured number (no torch/GPU runs allowed while the build is in use).
* Whether the current baseline enables cuDNN SDPA.
