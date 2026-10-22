# MOSS-Image-SFT text-to-image example

C++20 Kokkos port of a MOSS-Image-SFT-style flow-matching DiT stack (joint
self+cross attention transformer + FlowEuler sampler with CFG and
skip-layer guidance) + `moss_image_generate` driver.

Loads the published `OpenMOSS-Team/MOSS-Image-SFT` checkpoint layout
directly (BF16, sharded `.safetensors`); `--tiny` (random weights) always
works and is what CI/smoke uses.

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
# complete in-process run (needs the checkpoint from §(a) below):
# tokenize + Qwen encode -> release encoder -> DiT sample -> VAE decode,
# one component resident at a time (SLOW on serial CPU; see §(c))
./build/examples/moss_image/moss_image_generate \
  --tokenizer moss-sft/tokenizer --text-encoder moss-sft/text_encoder \
  --prompt "a red panda reading a newspaper" \
  --weights moss-sft/transformer \
  --vae-weights moss-sft/vae/diffusion_pytorch_model.safetensors \
  --vae-decode --staged --steps 50 --seed 42 --out moss_out.png
# + precomputed caption features instead of the in-process encode
# (alias of --text-weights; cross-machine / offline path)
./build/examples/moss_image/moss_image_generate \
  --weights moss-sft/transformer --prompt-embeds-in feats.f32 \
  --steps 50 --out moss_out.png
```

Staged 24GB runs (encoder and DiT are never resident at once; the MiB-scale
embeds stay resident on the device while the GiB encoder weights are
released, so there is no disk round-trip and no second command):

```bash
./build/examples/moss_image/moss_image_generate \
  --weights moss-sft/transformer --prompt "a red panda" \
  --staged --steps 50 --out moss_out.png
# sample from precomputed file embeds instead (alias of --text-weights):
./build/examples/moss_image/moss_image_generate \
  --weights moss-sft/transformer --prompt-embeds-in embeds.f32 \
  --steps 50 --out moss_out.png
# tiny staged smoke (no downloads; exercises the same plumbing):
./build/examples/moss_image/moss_image_generate \
  --tiny --steps 2 --staged
```

## Complete pipeline (real weights)

End-to-end with the public `OpenMOSS-Team/MOSS-Image-SFT` checkpoint
(Apache-2.0). The driver builds against both the checked-in placeholder
geometry and the true-pipeline headers
(`models/moss_image/KokkosDNN_MossImageDit.hpp`, `models/moss_image/KokkosDNN_MossImageEncoder.hpp`,
`models/moss_image/KokkosDNN_MossImageVae.hpp`): each
phase is probed with `__has_include` and falls back with a logged reason,
so the `[backend dit=... encoder=... vae=...]` banner tells you exactly
which path a run took. With a normal build all three true-pipeline headers
are present, so the published `transformer/` / `text_encoder/` / `vae/`
shards load with no conversion.

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
Encoder/Vae) — status only, never a hard dependency. bf16 weights
throughout (fp32 compute accumulation); on CUDA builds with
`KokkosDNN_ENABLE_CUBLAS=ON` / `KokkosDNN_ENABLE_CUDNN=ON` the GEMMs/attention
dispatch to cuBLAS/cuDNN via the core handles (no backend-gated code in the
driver). Serial-CPU full runs are slow; the 9B text encoder additionally
needs ~32 GB host RAM in fp32 (Kokkos `bhalf_t` is `float` without a GPU
backend), so prefer a CUDA/HIP build for real-weights runs.

### (c) Staged 24GB run

Driver `--height`/`--width` are **latent** dims (x16 = pixels, so the
default 64 means the reference 1024px; true geometry writes
`[128,64,64]` latents):

```bash
GEN=build/examples/moss_image/moss_image_generate
# Complete in-process run, one component resident at a time: tokenize +
# Qwen forward (bhalf weights) -> release encoder, keeping the MiB embeds
# resident on device -> fence -> sample (DiT only, bhalf) -> release +
# fence -> VAE decode (fp32). --tokenizer is the HF `tokenizer/` dir
# (needs tokenizer.json); --text-encoder is the `text_encoder/` dir
# (or a single .safetensors); --weights is the `transformer/` dir
# (or a single .safetensors).
$GEN --tokenizer moss-sft/tokenizer --text-encoder moss-sft/text_encoder \
  --prompt "a red panda reading a newspaper" --negative-prompt "" \
  --weights moss-sft/transformer \
  --staged --height 64 --width 64 --steps 50 --cfg 3.0 --shift 5.0 \
  --skip-layers 22 --seed 42 \
  --vae-weights moss-sft/vae/diffusion_pytorch_model.safetensors \
  --vae-decode --out moss_out.png
# ... with positive-branch SLG (reference: slg_layers=[24], slg_scale=3.0):
$GEN --tokenizer moss-sft/tokenizer --text-encoder moss-sft/text_encoder \
  --prompt "a red panda reading a newspaper" \
  --weights moss-sft/transformer \
  --staged --steps 50 --slg-layers 24 --slg-scale 3.0 --out moss_slg.png
# tiny smoke (no downloads; exercises encode/sample/decode plumbing):
$GEN --tiny --steps 2
# file-embed path instead of the in-process encode (cross-machine /
# offline; embeds produced by scripts/moss_encode_qwen.py or the --null
# fallback below, [L,Cc] = [512,4096] for the true geometry):
$GEN --weights moss-sft/transformer --prompt-embeds-in /tmp/tool_embeds.f32 \
  --steps 50 --out moss_out.png
# offline file-embed fallback (no torch/diffusers/weights/GPU): the --null
# output reproduces the null-embedding path bit-for-bit.
python3 scripts/moss_encode.py --null \
  --prompt "smoke" --out /tmp/tool_embeds.f32 --L 512 --Cc 4096
```

### (d) New-flag deltas

| flag | default | meaning |
|---|---|---|
| `--tokenizer` | — | dir holding `tokenizer.json` (Qwen2Tokenizer files); with `--text-encoder` runs the in-process Qwen forward (byte-level BPE + chat template, verified against transformers 5.12.1) |
| `--text-encoder` | — | Qwen text-encoder weights: `text_encoder/` shard dir or single `.safetensors` |
| `--negative-prompt` | `""` | negative caption, encoded in-process alongside the prompt (else zeros on the file/null paths) |
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
given (else grayscale preview). Encode runs fully in-process: the repo's
byte-level BPE (`models/moss_image/KokkosDNN_QwenTokenizer.hpp`, NFC + pre-tokenizer +
merges verified id-for-id against transformers 5.12.1) tokenizes the
prompt, and the on-device Qwen35::EncoderModel (full + Gated-DeltaNet
linear attention, no stubs, `models/moss_image/KokkosDNN_MossImageEncoder.hpp`) encodes
the prompt and `--negative-prompt` in one batched forward; its weights are
released before the DiT load. `--prompt-embeds-in` remains as the
file-embed fallback (cross-machine / offline; produce with
`scripts/moss_encode_qwen.py` or `scripts/moss_encode.py`).

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
next phase loads.

Checkpoint → flag mapping: `--weights` = DiT transformer checkpoint
(`transformer/` shard dir or single `.safetensors`); `--tokenizer` = dir
holding `tokenizer.json` (the HF `tokenizer/` dir works as-is);
`--text-encoder` = `text_encoder/` shard dir or single `.safetensors`;
`--text-weights` / `--prompt-embeds-in` = raw float32 LE caption features
`[L,Cc]` (produce with `scripts/moss_encode_qwen.py` or
`scripts/moss_encode.py`; `[1,L,Cc]`/`[1,1,L,Cc]` flattened also accepted,
last block wins); `--vae-weights` = FLUX.2 VAE `.safetensors` (single
250MB file), decoded only with `--vae-decode`.

## Driver flags (`moss_image_generate`)

| flag | default | meaning |
|---|---|---|
| `--tiny` | off | random-weights smoke model, 4x4 latents; ignores `--text-weights`/`--vae-weights`/geometry |
| `--weights` | `moss_image_sft.safetensors` | DiT transformer checkpoint: `transformer/` shard dir or single `.safetensors` (torch.save `.pth` on the placeholder path) |
| `--prompt` | `a red panda reading a newspaper` | Qwen-encoded in-process with `--tokenizer` + `--text-encoder`, else needs `--text-weights`, else null |
| `--negative-prompt` | `""` | negative caption (encoded in-process alongside the prompt, else zeros) |
| `--tokenizer` | — | dir holding `tokenizer.json` (Qwen2Tokenizer files); with `--text-encoder` runs the in-process Qwen forward |
| `--text-encoder` | — | Qwen text-encoder weights: `text_encoder/` shard dir or single `.safetensors` |
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
- Limitations: bf16 weights (encoder + DiT; fp32 latents/scheduler,
  fp32 compute accumulation; fp32 VAE); serial-CPU full runs are slow AND
  the 9B text encoder needs ~32 GB host RAM there (`bhalf_t` is `float`
  without a GPU backend) — prefer a CUDA/HIP build for real-weights runs.
  Chat template covers the reference single-user-turn path only.
- Staged plumbing: safetensors loaders stream per tensor with one
  `deep_copy` each (no 2x residency) and accept F32/BF16; `Weights`
  exposes `release()`; the driver runs an `EncoderStage` (in-process Qwen
  forward, else `--prompt-embeds-in` file, else null) with `--staged` and
  per-phase `[memory phase=...]` logging. The embeds stay resident on the
  device across the encoder-release / DiT-load boundary (device-to-device
  staging into the sample loop, no disk, no second process).
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

## Checkpoint key layout (MOSS-Image-SFT `transformer/`)

The loader accepts the published `transformer/` shards as-is (BF16 or F32;
pass the shard dir or a single `.safetensors`). Keys are matched by exact
name with an optional `model.`-style dotted prefix tolerated;
`rope.bands` is skipped (rebuilt as an fp32 constant). 463 tensors total
(`diffusion_pytorch_model.safetensors.index.json`), 462 loaded:

```text
video_embedder.{weight,bias}              # latent patch embed
text_embedder.proj.{weight,bias}          # caption Linear 4096->3072
text_embedder.blocks.{0,1}.qkv.{weight,bias} ...
text_embedder.blocks.{0,1}.proj.{weight,bias}
text_embedder.blocks.{0,1}.{q_norm,k_norm,norm1,norm2}.weight
text_embedder.blocks.{0,1}.ffn_{gate,up,down}.{weight,bias}
time_embedding.{0,2}.{weight,bias}        # timestep MLP (float64 sinusoid in)
shared_block_modulation.{weight,bias}     # shared modulation
video_head_modulation.{weight,bias}
layers.{i}.proj.{q,k,v,o}.weight          # joint self+cross attention (no bias)
layers.{i}.{q_norm,k_norm}.weight          # per-head QK RMSNorm
layers.{i}.{norm1,norm2}.weight
layers.{i}.proj.ffn_{up_gate,down}.weight  # SiTU FFN (no bias)
final_norm_video.weight
final_linear_video.weight
```

The placeholder-geometry path (built only when the true-pipeline headers
are absent) instead uses the
legacy `blocks.{i}.attn.qkv.weight`-style layout with `.pth` positional
mapping; see "Design notes" in the git history. It requires converting the
published checkpoint to that layout and only accepts F32.

## Verification

```bash
./build/examples/moss_image/moss_image_generate --tiny --steps 2
./build/examples/moss_image/moss_image_generate --help
# staged plumbing on the smoke model (in-memory device handoff):
./build/examples/moss_image/moss_image_generate \
  --tiny --steps 2 --staged
# unit tests (tests-enabled build): tokenizer goldens need the real
# tokenizer.json (skipped without it), encoder/DiT/VAE suites run anywhere
KOKKOSDNN_QWEN_TOKENIZER_JSON=/path/to/tokenizer.json ctest -R "wen|Moss"
```

exercises config → random init → model forward → FlowEuler + CFG/SLG loop
→ preview + `.f32` dump with no downloads, plus `Test_MossImageOps`,
`Test_MossImageDit`, `Test_MossImageEncoder` (incl. tiny forward goldens
and the safetensors round-trip) and `Test_QwenTokenizer` (10 golden
id-sequences vs transformers 5.12.1, NFC/pre-tokenizer edge cases).
Real-weights verification: tokenize parity is covered by the golden tests;
the encoder loader is checked tensor-by-tensor against the published
`text_encoder` index (names/shapes/dtypes); end-to-end, run §(c) on a
CUDA/HIP build and compare latents against the reference
`pipeline_moss_image.py` (tolerances TBD),
mirroring `examples/sana/Test_SanaGolden.cpp`.
