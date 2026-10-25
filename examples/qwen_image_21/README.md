# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Qwen-Image-2.1 text-to-image driver (TEXT-TO-IMAGE ONLY)

`qwen_image_21_generate` runs staged text-to-image inference:

```
encode -> fence -> release encoder -> load DiT -> sample from
device-resident embeds -> release DiT -> fence -> VAE decode (or preview)
```

so the text encoder, the DiT, and the VAE are never resident at once.
No disk round-trip and no second process: the MiB-scale embeds are separate
device allocations from the GiB-scale encoder weights and stay resident
across the DiT load. Raw fp32 latents (`<stem>.f32`, `[C,H,W]` row-major)
are always written; `--out` holds a VAE RGB image or a grayscale preview.

Editing is excluded: Qwen-Image-Edit (reference-image conditioning) is out
of scope for this driver; there is no image-conditioning flag by design.

## Checkpoint layout (Hugging Face, ~32.5 GB total)

```
<ckpt>/
  transformer/                        DiT, 2 safetensors shards (~14.2 GB)
    diffusion_pytorch_model-00001-of-00002.safetensors
    diffusion_pytorch_model-00002-of-00002.safetensors
    *.safetensors.index.json
  text_encoder/
    model.safetensors                 Qwen text encoder (~17.5 GB)
  processor/
    tokenizer.json                    chat template + tokenizer
  vae/
    diffusion_pytorch_model.safetensors   VAE decoder (~0.676 GB)
```

`--weights` takes the `transformer/` dir (or a single `.safetensors`);
`--text-encoder` takes `text_encoder/` (or `model.safetensors`);
`--tokenizer` takes the `processor/` dir; `--vae-weights` takes the VAE
file (or dir). The `models/qwen_image_21` headers are present, so `--tiny`
runs the real DiT geometry on random weights and full-geometry runs use
the checkpoint files above. In-process Qwen encoding (`--tokenizer` +
`--text-encoder`: raw-text BPE of the T2I template + `EncoderModel`
forward, uncond zeros) and real VAE decode (runtime `vae/config.json`
affine denormalize + `VaeModel::decode` to RGBA) are wired in; when their
flags/headers are absent those phases fall back to file/null embeds +
grayscale preview with a logged reason instead of failing.

## Flags

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--tiny` | off | random-weights smoke model, 4x4 latents, 2 steps, no downloads |
| `--weights` | `transformer` | DiT checkpoint: `transformer/` shard dir or single `.safetensors` |
| `--prompt` | `a red panda reading a newspaper` | caption text |
| `--tokenizer` | — | dir holding `tokenizer.json`; with `--text-encoder` runs the in-process Qwen forward |
| `--text-encoder` | — | Qwen text-encoder weights: dir or single `.safetensors` |
| `--text-weights` | — | raw float32 LE caption features `[L,Cc]` |
| `--prompt-embeds-in` | — | alias of `--text-weights` (wins if both differ) |
| `--height`/`--width` | 64 | latent H/W; x16 = pixels, so 64 means 1024px (placeholder path needs multiples of 16) |
| `--steps` | 8 (Turbo) | Euler steps (`--tiny` forces 2 unless `--steps` is given) |
| `--cfg`/`--guidance` | 1.0 | CFG scale (!=1 takes the CFG branch) |
| `--shift`/`--flow-shift` | 3.0 | flow-matching sigma shift (ignored under `--sigmas turbo`) |
| `--sigmas` | `driver` | `driver`: nearest-grid schedule from `--steps`/`--shift` (default, unchanged); `turbo`: checkpoint `sample_sigmas` verbatim (mu-shifted already, 8 steps, shift=1.0/no-shift) |
| `--seed` | 42 | latent init seed (ignored when `--init-latents-in` is given) |
| `--init-latents-in` | — | raw float32 LE initial latents `[C,H,W]` row-major, C-major `lat[(c*H+h)*W+w]` (PRE-denormalization noise, no mean/std); replaces the mt19937 sample |
| `--vae-weights` | — | VAE checkpoint (validated whenever given) |
| `--vae-decode` | off | real VAE decode when the VAE header + weights are present, else grayscale preview |
| `--out` | `qwen_image_21_out.png` | PNG with `KokkosDNN_ENABLE_IMAGE=ON`, else PPM; raw `.f32` latents always written |
| `--staged` | off | fenced staged release between phases (see below) |
| `--dump-steps` | — | `DIR`: write golden-parity tensors (raw float32 LE) + `manifest.json`, then exit normally |

Conditioning precedence: in-process Qwen (`--tokenizer` + `--text-encoder`)
-> `--prompt-embeds-in` file -> null zeros (prompt logged).

## Parity dumps (`--dump-steps DIR`)

Golden-parity work needs per-step tensors, so `--dump-steps DIR` writes them
(raw little-endian float32, driver layout: latents `[C,H,W]` row-major
C-major `lat[(c*H+h)*W+w]`, prompt embeds `[L,Cc]` row-major):

| File | Content |
| ---- | ------- |
| `prompt_embeds.f32` | cond conditioning exactly as staged for the DiT |
| `prompt_embeds_neg.f32` | uncond view (only when `--cfg != 1`) |
| `initial_latents.f32` | the sampler's `x_init`, no extra scaling |
| `stepNN_noise_pred.f32` | model output `v` used for step NN (CFG-combined when CFG runs) |
| `stepNN_noise_pred_cond.f32`, `_uncond.f32` | per-branch outputs (only under CFG) |
| `stepNN_latents.f32` | post-Euler-update latents |
| `manifest.json` | `format` (`kokkosdnn-qwen21-dump-v1`), prompt/source, H/W, channels, steps, exact `sigmas`, seed, cfg, `use_cfg`, `staged`, `--sigmas` mode, model label, per-tensor f64 sums |

Each tensor is also logged as `[dump] <name> <numel> floats f64sum=<%.17g>`,
followed by `[dump] wrote <n> tensors to DIR`. The extra device-to-host
copies only happen when the flag is set, so the default sample path is
unchanged. Works under `--tiny` (`--tiny --steps 2 --dump-steps /tmp/q21`
dumps 512x16 zero embeds, 4x4 latents, and two steps).

## 24 GB fit table (bf16 projections)

| Phase | Resident | Fits 24 GB? |
| ----- | -------- | ----------- |
| encode | 17.5 GB encoder | FITS (released before DiT load) |
| sample | 14.2 GB DiT + GiB-scale activations | FITS |
| vae decode | 0.7 GB | FITS |
| staged peak | ~18 GB + activations | FITS |
| unstaged (all resident) | ~32 GB + activations | NO |

## Staged recipe

```sh
# Smoke test (no weights, CPU or GPU):
qwen_image_21_generate --tiny --steps 2 --staged

# Staged full run (once the models/qwen_image_21 headers land):
qwen_image_21_generate --staged \
  --weights <ckpt>/transformer \
  --text-encoder <ckpt>/text_encoder --tokenizer <ckpt>/processor \
  --prompt "a red panda reading a newspaper" \
  --height 64 --width 64 --steps 8 --cfg 1.0 \
  --vae-weights <ckpt>/vae/diffusion_pytorch_model.safetensors --vae-decode \
  --out qwen_out.png
```

`--staged` fences after encode, logs `encode-released`, keeps the
`~14 MiB` of embeds resident on-device across the DiT load
(`[staged] keeping X MiB ... no disk round-trip`), samples straight from
them (device-to-device, never through host or disk), then releases the
DiT, fences, and decodes.
