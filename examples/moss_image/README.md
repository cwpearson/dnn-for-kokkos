# MOSS-Image-SFT text-to-image example

C++20 Kokkos port of a MOSS-Image-SFT-style flow-matching DiT stack (joint
self+cross attention transformer + FlowEuler sampler with CFG and
skip-layer guidance) + `moss_image_generate` driver.

> Scope: the public MOSS-Image-SFT checkpoint layout is not pinned here, so
> this example defines and documents its own canonical key layout (below).
> Full-weight runs need a checkpoint converted to that layout; `--tiny`
> (random weights) always works and is what CI/smoke uses.

## Quickstart (0 to working)

Run from repo root (`/workspace`):

```bash
cmake -S . -B build \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkosDNN_ENABLE_IMAGE=ON
cmake --build build --target moss_image_generate --parallel 8

# tiny smoke: random weights, 4x4 latents, 2 Euler steps (no downloads)
./build/examples/moss_image/moss_image_generate --tiny --steps 2
# help
./build/examples/moss_image/moss_image_generate --help
# full geometry, null conditioning (prompt logged only; SLOW on serial CPU)
./build/examples/moss_image/moss_image_generate \
  --weights moss_image_sft.safetensors --height 64 --width 64 \
  --steps 50 --cfg 3.0 --shift 5.0 --skip-layers 22 --seed 42 \
  --out moss_out.png
# + precomputed caption features (highest precedence)
./build/examples/moss_image/moss_image_generate \
  --weights moss_image_sft.safetensors --text-weights feats.f32 \
  --prompt "a red panda reading a newspaper" --steps 50 \
  --out moss_out.png
```

Staged 24GB runs (encoder and DiT are never resident at once; the MiB-scale
embeds stay resident on the device while the GiB encoder weights are
released, so there is no disk round-trip and no second command):

```bash
./build/examples/moss_image/moss_image_generate \
  --weights moss_image_sft.safetensors --prompt "a red panda" \
  --staged --steps 50 --out moss_out.png
# sample from precomputed file embeds instead (alias of --text-weights):
./build/examples/moss_image/moss_image_generate \
  --weights moss_image_sft.safetensors --prompt-embeds-in embeds.f32 \
  --steps 50 --out moss_out.png
# tiny staged smoke (no downloads; exercises the same plumbing):
./build/examples/moss_image/moss_image_generate \
  --tiny --steps 2 --staged
```

## Complete pipeline (real weights)

End-to-end with the public `OpenMOSS-Team/MOSS-Image-SFT` checkpoint
(Apache-2.0). The driver builds against both the checked-in placeholder
geometry and the true-pipeline headers
(`src/KokkosDNN_MossImageDit.hpp`, `src/KokkosDNN_MossImageEncoder.hpp`,
`src/KokkosDNN_MossImageVae.hpp`, landing from parallel worktrees): each
phase is probed with `__has_include` and falls back with a logged reason,
so the `[backend dit=... encoder=... vae=...]` banner tells you exactly
which path a run took.

### (a) Retrieve the pieces

~27 GB total. Neither the build nor the tests download these (only
`--help` / `--tiny` runs are weight-free):

| piece | files | payload |
|---|---|---|
| transformer (DiT, ~5.08B bf16 params, 463 tensors) | `transformer/*` | ~10.2 GB (`total_size: 10164617280`) |
| text encoder (Qwen3.5, 759 tensors, bf16) | `text_encoder/*` | ~16.8 GB (`total_size: 16785390048`) |
| tokenizer (Qwen2Tokenizer) | `tokenizer/*` | KB |
| VAE (`AutoencoderKLFlux2`, fp32) | `vae/*` | ~0.25 GB (`249521372` B) |
| reference pipeline (read-only reference, 440 lines) | `pipeline_moss_image.py` | KB |
| **total on disk** | | **~27.2 GB** |

```bash
# huggingface_hub CLI (`pip install huggingface_hub`); dry-run strings only —
# do NOT run these on small disks, and never commit weights to the repo.
hf download OpenMOSS-Team/MOSS-Image-SFT --include "transformer/*" \
  --local-dir moss-sft
hf download OpenMOSS-Team/MOSS-Image-SFT --include "text_encoder/*" \
  --local-dir moss-sft
hf download OpenMOSS-Team/MOSS-Image-SFT --include "tokenizer/*" \
  --local-dir moss-sft
hf download OpenMOSS-Team/MOSS-Image-SFT --include "vae/*" \
  --local-dir moss-sft
hf download OpenMOSS-Team/MOSS-Image-SFT \
  --include "pipeline_moss_image.py" --local-dir moss-sft
```

Only the MiB-scale `.f32` handoff files (`[L,Cc]` embeds, `[C,H,W]`
latents) ever need to move between machines — never the 27 GB weights.

### (b) Build

```bash
cmake -S . -B build \
  -DKokkos_ROOT=/tmp/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkosDNN_ENABLE_IMAGE=ON
cmake --build build --target moss_image_generate --parallel 8
```

Configure logs which true-pipeline headers are visible
(`moss_image: KokkosDNN_MossImageDit.hpp absent/visible`, same for
Encoder/Vae) — status only, never a hard dependency. FP32-first
throughout; on CUDA builds with `KokkosDNN_ENABLE_CUBLAS=ON` /
`KokkosDNN_ENABLE_CUDNN=ON` the GEMMs/attention dispatch to cuBLAS/cuDNN
via the core handles (no backend-gated code in the driver).

### (c) Staged 24GB run

Driver `--height`/`--width` are **latent** dims (x16 = pixels, so the
default 64 means the reference 1024px; true geometry writes
`[128,64,64]` latents):

```bash
GEN=build/examples/moss_image/moss_image_generate
# Single process, one component resident at a time: encode (Qwen forward
# when the encoder header + weights are present, else --prompt-embeds-in
# file, else null) -> release encoder weights, keeping the MiB embeds
# resident on device -> fence -> sample (DiT only) -> release + fence ->
# VAE decode.
$GEN --tokenizer moss-sft/tokenizer --text-encoder moss-sft/text_encoder \
  --prompt "a red panda reading a newspaper" --negative-prompt "" \
  --weights moss-sft/transformer \
  --staged --height 64 --width 64 --steps 50 --cfg 3.0 --shift 5.0 \
  --skip-layers 22 --seed 42 \
  --vae-weights moss-sft/vae --vae-decode --out moss_out.png
# ... with positive-branch SLG (reference: slg_layers=[24], slg_scale=3.0),
# sampling from the script-produced file embeds:
$GEN --weights moss-sft/transformer --prompt-embeds-in /tmp/tool_embeds.f32 \
  --steps 50 --slg-layers 24 --slg-scale 3.0 --out moss_slg.png
# tiny smoke (no downloads; exercises encode/sample/decode plumbing):
$GEN --tiny --steps 2
# offline file-embed fallback (no torch/diffusers/weights/GPU): the --null
# output reproduces the null-embedding path bit-for-bit.
python3 scripts/moss_encode.py --null \
  --prompt "smoke" --out /tmp/tool_embeds.f32 --L 512 --Cc 2048
$GEN --weights moss_image_sft.safetensors \
  --prompt-embeds-in /tmp/tool_embeds.f32 --steps 50 --out moss_out.png
```

### (d) New-flag deltas

| flag | default | meaning |
|---|---|---|
| `--tokenizer` | — | HF tokenizer dir (Qwen2Tokenizer files; needs `--text-encoder` + encoder header for a real forward) |
| `--text-encoder` | — | Qwen text-encoder weights (`.safetensors` or dir) |
| `--negative-prompt` | `""` | negative caption (Qwen-encoded when the encoder is available, else zeros) |
| `--slg-layers` | — | comma-separated block ids for a positive-branch skipped forward (e.g. `24`; empty disables) |
| `--slg-scale` | 0.0 | SLG correction scale (`>0` + non-empty `--slg-layers` enables the third forward) |
| `--slg-start`/`--slg-stop` | 0.0/1.0 | window gating uncond skipping AND SLG, as a fraction of steps in denoising order |
| `--vae-decode` | off | run the real VAE decode when the VAE header + `--vae-weights` are present (else grayscale preview with a logged reason) |

Reference-parity semantics (verified against
`pipeline_moss_image.py::__call__` / `MOSSImageScheduler::set_timesteps`):
`do_cfg = (guidance != 1.0)` (so `0 <= g < 1` takes the CFG branch, as in
the reference); nearest-grid sigma schedule (inference grid snapped to the
training-grid points, terminal 0, model timesteps `1000*sig`, Euler update
`x += dt*v`); uncond skipping and SLG share the
`[slg_start, slg_stop)` window (plain CFG outside it); with
`guidance == 1` the SLG-only form `cond_skip + slg_scale*(cond-cond_skip)`
applies. One placeholder-only caveat: `Moss::Model::forward` applies a
single skip per forward, so a multi-entry `--slg-layers` uses the first
entry with a warning (the true `MossImageDit::Model` takes the whole set).

### (e) Fit table + what still falls back

Staged peaks (from `STAGED_MEMORY_BUDGET.md` §3, 24 GiB budget):

| strategy | peak resident | fits? |
|---|---|---|
| simultaneous (encoder + DiT + VAE + act) | ~28–31 GiB | **no** |
| staged encode (Qwen, true header) | ~17–19 GiB | yes |
| staged sample (DiT + CFG 2x) | ~12–14 GiB | yes |
| staged decode (VAE) | ~2–4 GiB | yes |

Landed in this tree: all three true-pipeline headers are present, so runs
print `[backend dit=moss-image-dit encoder=qwen vae=moss-image-vae]` —
sampling runs the true 42x3072 MMDiT (per-sample text mask, 4D RoPE, SiTU),
decode runs the FLUX.2-small VAE when `--vae-weights --vae-decode` are
given (else grayscale preview). Encode: the on-device Qwen35::EncoderModel
(full + Gated-DeltaNet linear attention, no stubs) is in
`src/KokkosDNN_MossImageEncoder.hpp`; the driver keeps the file/null
handoff — produce ids/embeds with `scripts/moss_encode_qwen.py`
(`--ids-only` / full forward) and pass `--prompt-embeds-in`, or run the
EncoderModel from your own TU. `--negative-prompt` without precomputed
neg embeds still yields zeros for the uncond branch.

## 24GB fit table (staged text-encoder/DiT split)

Real-checkpoint resident budgets (bf16):

| phase | resident | budget | fits 24GB? |
|---|---|---|---|
| encode (Qwen text encoder) | ~16.8 GB | 16.8 GB | yes (alone) |
| sample (MMDiT DiT) | ~10.2 GB | 10.2 GB | yes (alone) |
| VAE decode (joins sample phase) | ~0.25 GB | 0.25 GB | yes |
| staged peak (max phase + VAE + activations) | ~17.1 GB + activations | 24 GB | yes |
| unstaged (encoder + DiT co-resident) | ~27+ GB | 24 GB | **no** |

The driver prints a per-phase resident estimate
(`[memory phase=...] weights + activations`) and issues a
`Kokkos::fence()` between phases; `Weights::release()` /
`BlockWeights::release()` drop every device View (assign `{}`) before the
next phase loads. The current example-geometry estimates print smaller
numbers than the table above (the example DiT is ~0.8B params, not the full
5B checkpoint); the table governs real-checkpoint runs.

Checkpoint → flag mapping: `--weights` = DiT transformer checkpoint
(single `.safetensors` shard file; the HF `transformer/` dir ships 3 shards
— pass them as one concatenated download or point at each shard in turn,
see `MossImageDitWeights::load_safetensors` which also accepts a shard
list); `--text-weights` = raw float32 LE caption features `[L,Cc]`
(`[1,L,Cc]`/`[1,1,L,Cc]` flattened also accepted, last block wins; produce
with `scripts/moss_encode_qwen.py` or `scripts/moss_encode.py`);
`--tokenizer`/`--text-encoder` document the Qwen tokenizer/encoder source
for the script path (the driver stages file embeds on-device);
`--vae-weights` = FLUX.2 VAE `.safetensors` (single 250MB file), decoded
only with `--vae-decode`.

## Driver flags (`moss_image_generate`)

| flag | default | meaning |
|---|---|---|
| `--tiny` | off | random-weights smoke model, 4x4 latents; ignores `--text-weights`/`--vae-weights`/geometry |
| `--weights` | `moss_image_sft.safetensors` | DiT checkpoint (`.safetensors`, or torch.save `.pth`) |
| `--prompt` | `a red panda reading a newspaper` | Qwen-encoded with `--tokenizer` + `--text-encoder` (true header), else needs `--text-weights`, else null |
| `--negative-prompt` | `""` | negative caption (encoded when the Qwen encoder is available, else zeros) |
| `--tokenizer` | — | HF tokenizer dir (Qwen2Tokenizer files) |
| `--text-encoder` | — | Qwen text-encoder weights (`.safetensors` or dir); with `--tokenizer` runs the real Qwen forward |
| `--text-weights` | — | raw F32 caption features `[L,Cc]` |
| `--prompt-embeds-in` | — | alias of `--text-weights` (wins if both differ) |
| `--staged` | off | encode → release encoder weights (MiB embeds stay resident on device) → fence → load DiT → sample (encoder/DiT never co-resident, no disk round-trip) |
| `--height`/`--width` | 64 | latent H/W (x16 = pixels, so 64 means 1024px; placeholder path needs multiples of 16, must tile `patch_size`) |
| `--steps` | 50 | Euler steps (nearest-grid schedule; count may shrink via `unique_consecutive`) |
| `--cfg` (`--guidance`) | 3.0 | CFG scale (`!=1` takes the CFG branch, matching the reference) |
| `--shift` (`--flow-shift`) | 5.0 | sigma shift (`> 0`) |
| `--skip-layers` (`--skip-layer`) | 22 | uncond-branch skip layer (`-1` disables); window-gated by `--slg-start`/`--slg-stop` |
| `--slg-layers` | — | comma-separated block ids for a positive-branch skipped forward (empty disables) |
| `--slg-scale` | 0.0 | SLG correction scale (`>0` + `--slg-layers` enables the third forward) |
| `--slg-start`/`--slg-stop` | 0.0/1.0 | window gating uncond skipping and SLG, as a fraction of steps (`0 <= start <= stop <= 1`) |
| `--seed` | 42 | latent init seed |
| `--vae-weights` | — | VAE checkpoint (validated whenever given) |
| `--vae-decode` | off | real VAE decode when the VAE header + weights are present (else grayscale preview with a logged reason) |
| `--out` | `moss_out.png` | PNG with `KokkosDNN_ENABLE_IMAGE=ON`, else `.ppm`; raw F32 latents always written to `<stem>.f32` (`[C,H,W]`) |

## Notes

- Architecture (true pipeline): DiT dim3072/42 blocks/24 heads/hd128,
  text refiner (Linear 4096→3072 + 2 TextBlocks), float64 timestep sinusoid
  + SiLU MLP + shared/head modulation, joint image+text self-attention with
  4D RoPE + per-head QK RMSNorm, SiTU FFN, gated residuals, skip-layer
  identity; Qwen3.5-9B encoder (32 hybrid full + Gated-DeltaNet linear
  layers, 4096 hidden) via `Qwen35::EncoderModel`; FLUX.2-small VAE decoder
  (GroupNorm-32, SiLU, mid-attention, 2x2 patch, scale 16) via
  `MossImage::Vae::DecoderModel`. Sampler: nearest-grid FlowEuler
  (shift 5.0 snapped to the training grid, `unique_consecutive`),
  `v = uncond + g·(cond−uncond)[+ slg·(cond−cond_skip)]`, mt19937
  Box-Muller init; default uncond skip {22}, window-gated with SLG.
- Limitations: FP32-first latents/scheduler (DiT/encoder weights load
  BF16+F32, compute accumulates FP32); on-device Qwen forward takes token
  ids (BPE/chat-template stays in `scripts/moss_encode_qwen.py`);
  serial-CPU full runs are slow — CUDA parallelizes the same kernels
  (GEMMs→cuBLAS, attentions→cuDNN, norms/convs elementwise-native).
- Staged plumbing status: safetensors/`.pth` loaders stream through
  HostSpace chunks with one `deep_copy` per tensor (no 2x residency, float +
  bhalf_t); `Weights`/`BlockWeights` expose `release()`; the driver runs an
  `EncoderStage` (file/null embeds today) with `--staged` /
  `--prompt-embeds-in` and per-phase `[memory phase=...]` logging. The
  embeds stay resident on the device across the encoder-release / DiT-load
  boundary (device-to-device staging into the sample loop, no disk, no
  second process).
  What the future Qwen port must fill in: tokenize the prompt, stream
  `text_encoder/*.safetensors` into device views with the same HostSpace
  pattern, run the Qwen forward + text refiner on-device into the
  `EncoderStage` embeds, and keep the existing `release()` + `fence()`
  before the DiT load (the ~16.8GB encoder budget in the fit table then
  becomes live weights instead of a projection). The driver side is done:
  `EncoderStage::run` already prefers the Qwen forward (header +
  `--tokenizer` + `--text-encoder`), the sample loop already runs the
  reference CFG/SLG/nearest-grid semantics through a backend-generic
  forward, and decode already releases the DiT before the VAE phase (see
  "Complete pipeline" above).
- Targets (`CMakeLists.txt`): `kokkosdnn_moss` lib + `moss_image_generate`.
  Sources: `KokkosDNN_Moss{Config,Weights,Model,Sampler}` + driver.
  Wired into `examples/CMakeLists.txt` (`add_subdirectory(moss_image)`,
  same unconditional style as the other examples).

## Design notes (dispatch)

- `Model::forward` uses only core-library entry points: `gemm` /
  `gemm_bias` for every projection (patch embed, timestep MLP, caption
  projection, QKV/cap-KV, attention out-projection, FFN, final linear) and
  `attention_mha_bidirectional` for the joint self+cross attention;
  norms/activations use `layernorm_forward`, `rmsnorm`, `gelu_forward`
  (tanh). On CUDA builds with `KokkosDNN_ENABLE_CUBLAS=ON`, the GEMMs
  dispatch to cuBLAS (sgemm) via the handle's cached handle; with
  `KokkosDNN_ENABLE_CUDNN=ON`, the joint attention dispatches to the cuDNN
  SDPA graph first (falls back to the native kernel on failure). Everything
  else runs the portable Kokkos kernels on every backend (no
  backend-gated code in the example).
- Per-step cost: two B=1 forwards when CFG is active (branches differ by
  `skip_layer`, so they cannot share one B=2 forward), one B=1 forward
  otherwise. Scratch is grow-only (steady state performs no View
  allocations).

## Canonical checkpoint key layout

`.safetensors` keys are suffix-matched (a `model.`-style prefix is fine):

```text
patch_embed.proj.{weight,bias}
time_embed.mlp.0.{weight,bias}, time_embed.mlp.2.{weight,bias}
time_embed.block.{weight,bias}            # Linear(D -> 6D)
cap_embed.proj.{weight,bias}, cap_embed.norm.weight
blocks.{i}.scale_shift_table              # [6, D]
blocks.{i}.attn.qkv.weight                # [3D, D]
blocks.{i}.attn.cap_kv.weight             # [2D, D]
blocks.{i}.attn.proj.{weight,bias}        # [D, D]
blocks.{i}.mlp.fc1.{weight,bias}          # [H, D]
blocks.{i}.mlp.fc2.{weight,bias}          # [D, H]
final.scale_shift_table                    # [2, D]
final.linear.{weight,bias}                 # [out_ch*P*P, D] (== [out_ch, D] at P=1)
```

The `.pth` loader maps the same list positionally (Nth key ↔ Nth
`data/{N}` float32 storage, key order validated as suffix matches). All
tensors are float32. `patch_embed.proj.weight` is `[D, in_ch*P*P]`
row-major (torch `[D, in_ch, P, P]` flattened with `col=(c*P+kh)*P+kw`,
matching the model's patchify order).

## Verification

```bash
./build/examples/moss_image/moss_image_generate --tiny --steps 2
./build/examples/moss_image/moss_image_generate --help
# staged plumbing on the smoke model (in-memory device handoff):
./build/examples/moss_image/moss_image_generate \
  --tiny --steps 2 --staged
# unit tests for the merged MossImage ops (tests-enabled build):
ctest -R Moss
```

exercises config → random init → model forward → FlowEuler + CFG/SLG loop
→ preview + `.f32` dump with no downloads, plus the 15-test
`Test_MossImageOps` suite (`ctest -R Moss`: 15/15 pass). Real-weight
verification is pending the landed Dit/Encoder/Vae headers plus a
checkpoint: load the transformer with `--weights`, the encoder with
`--tokenizer`/`--text-encoder` (or file embeds from
`scripts/moss_encode.py`), decode with `--vae-weights --vae-decode`,
compare against the reference sampler (tolerances TBD once dtypes/layouts
are pinned), and add golden tests mirroring
`examples/sana/Test_SanaGolden.cpp`.
