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

Staged 24GB runs (encoder and DiT are never resident at once):

```bash
# 1. encode only: resolves caption embeds, writes them, exits before any
#    DiT weight is loaded
./build/examples/moss_image/moss_image_generate \
  --weights moss_image_sft.safetensors --prompt "a red panda" \
  --encode-only --prompt-embeds-out embeds.f32
# 2. sample from the staged embeds (alias of --text-weights)
./build/examples/moss_image/moss_image_generate \
  --weights moss_image_sft.safetensors --prompt-embeds-in embeds.f32 \
  --steps 50 --out moss_out.png
# ... or both phases in one process (encode -> write -> release -> fence
#     -> load DiT -> sample):
./build/examples/moss_image/moss_image_generate \
  --weights moss_image_sft.safetensors --prompt "a red panda" \
  --staged --prompt-embeds-out embeds.f32 --steps 50 --out moss_out.png
# tiny staged smoke (no downloads; exercises the same plumbing):
./build/examples/moss_image/moss_image_generate \
  --tiny --steps 2 --encode-only --prompt-embeds-out /tmp/staged_embeds.f32
./build/examples/moss_image/moss_image_generate \
  --tiny --steps 2 --prompt-embeds-in /tmp/staged_embeds.f32
```

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

Checkpoint → flag mapping: `--weights` = DiT checkpoint (`.safetensors`
preferred; torch.save `.pth`/`.pt`/`.bin` also accepted via the ordered key
list below); `--text-weights` = raw float32 LE caption features
`[L,Cc]` (`[1,L,Cc]`/`[1,1,L,Cc]` flattened also accepted, last block wins);
`--vae-weights` is reserved/validated only (the MOSS VAE decoder is not
ported; a grayscale preview is written regardless).

## Driver flags (`moss_image_generate`)

| flag | default | meaning |
|---|---|---|
| `--tiny` | off | random-weights smoke model, 4x4 latents; ignores `--text-weights`/`--vae-weights`/geometry |
| `--weights` | `moss_image_sft.safetensors` | DiT checkpoint (`.safetensors`, or torch.save `.pth`) |
| `--prompt` | `a red panda reading a newspaper` | logged; conditions the model only via `--text-weights` |
| `--text-weights` | — | raw F32 caption features `[L,Cc]` |
| `--prompt-embeds-in` | — | alias of `--text-weights` (wins if both differ) |
| `--prompt-embeds-out` | — | write the encode-stage embeds to a raw F32 file |
| `--staged` | off | encode → write embeds → release encoder → fence → load DiT → sample (encoder/DiT never co-resident) |
| `--encode-only` | off | run the encode/embed stage, optionally write embeds, then exit WITHOUT loading DiT weights |
| `--height`/`--width` | 64 | latent H/W, each a positive multiple of 16 (must tile `patch_size=2`) |
| `--steps` | 50 | Euler steps |
| `--cfg` (`--guidance`) | 3.0 | CFG scale (`<=1` disables guidance) |
| `--shift` (`--flow-shift`) | 5.0 | sigma shift |
| `--skip-layers` (`--skip-layer`) | 22 | SLG layer skipped on the uncond branch (`-1` disables) |
| `--seed` | 42 | latent init seed |
| `--vae-weights` | — | reserved (validated; preview written regardless) |
| `--out` | `moss_out.png` | PNG with `KokkosDNN_ENABLE_IMAGE=ON`, else `.ppm`; raw F32 latents always written to `<stem>.f32` (`[C,H,W]`) |

## Notes

- Architecture: patch-2 embed of 16ch latents → `(H/2)x(W/2)` tokens;
  timestep cos/sin MLP + shared `t_block` Linear(D→6D); 24 blocks
  (adaLN-single, joint self+cross attention: queries from modulated tokens,
  keys/values from tokens ++ RMSNorm'd `[L,2048]` caption, 24 heads x 64 dim,
  then gated residual; FFN fc1→GELU-tanh→fc2 → gated residual); final
  adaLN-modulate → linear → unpatchify. FlowEuler sampler
  (`sig = shift·u/(1+(shift−1)·u)`, shift 5.0, `v = uncond +
  g·(cond−uncond)`, mt19937 Box-Muller init). SLG: the uncond branch runs
  with block 22 as identity; the cond branch is full depth.
- Limitations: no MOSS text encoder (pass precomputed features or null);
  no MOSS VAE decoder (preview only); FP32 only; serial-CPU full runs are
  slow — CUDA parallelizes the same kernels.
- Staged plumbing status: safetensors/`.pth` loaders stream through
  HostSpace chunks with one `deep_copy` per tensor (no 2x residency, float +
  bhalf_t); `Weights`/`BlockWeights` expose `release()`; the driver runs an
  `EncoderStage` (file/null embeds today) with `--encode-only` / `--staged`
  / `--prompt-embeds-{in,out}` and per-phase `[memory phase=...]` logging.
  What the future Qwen port must fill in: tokenize the prompt, stream
  `text_encoder/*.safetensors` into device views with the same HostSpace
  pattern, run the Qwen forward + text refiner on-device into the
  `EncoderStage` embeds, and keep the existing `release()` + `fence()`
  before the DiT load (the ~16.8GB encoder budget in the fit table then
  becomes live weights instead of a projection).
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
```

exercises config → random init → model forward → FlowEuler + CFG/SLG loop
→ preview + `.f32` dump with no downloads. Real-weight verification is
pending a checkpoint in the layout above (see "what still needs real
weights" in the task summary): load it with `--weights`, compare against
the reference sampler (tolerances TBD once dtypes/layouts are pinned),
and add golden tests mirroring `examples/sana/Test_SanaGolden.cpp`.
