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

## Interactive loop (default) vs one-off

Without `--prompt` on a TTY (and without the file/parity handoff flags
`--prompt-embeds-in`/`--text-weights`, `--init-latents-in`, `--dump-steps`,
`--decode-latents-in`), the driver runs as a REPL: it prints the current
latent `HxW`, asks for a prompt, asks for an output filename, generates,
and repeats indefinitely until EOF or `quit`/`exit`. The previous prompt is
offered for editing each time (GNU readline when available), and `/w N` /
`/h N` set the latent width/height going forward. An empty prompt re-asks;
an empty filename auto-numbers `qwen_image_21_out_<n>.png`; the per-iteration
seed is `--seed` + iteration, so repeats of the same prompt still vary.

```sh
qwen_image_21_generate --staged \
  --weights <ckpt>/transformer \
  --text-encoder <ckpt>/text_encoder --tokenizer <ckpt>/processor \
  --vae-weights <ckpt>/vae/diffusion_pytorch_model.safetensors --vae-decode
# [size] latent 64x64 (x16 pixels = 1024x1024)
# prompt> a red panda reading a newspaper
# output> red_panda.png
# prompt> quit
```

The encoder and transformer weights are loaded ONCE into host memory before
the first prompt; every iteration copies the needed component host->device,
runs the phase, and releases the device copy, so `--staged` keeps the same
one-component-resident property per iteration without re-reading the
~32 GB checkpoint from disk.

Giving `--prompt` (or any file/parity flag, or piping stdin) runs ONE
generation and exits, which is what the golden/CI harnesses use.

## Acquiring the checkpoint (Hugging Face, ~32.5 GB)

Public repo `Qwen/Qwen-Image-2.1-Turbo` (~30.3 GiB on disk; ~32.5 GB
decimal). Neither the build nor the tests download it (only `--help` /
`--tiny` runs are weight-free). Never commit weights to the repo.

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen-Image-2.1-Turbo --local-dir <ckpt>
```

Sharded/full download on a small disk: fetch per piece instead (repeat
`--include`; each command merges into the same `<ckpt>` dir):

```bash
hf download Qwen/Qwen-Image-2.1-Turbo --include "transformer/*" \
  --local-dir <ckpt>     # DiT, 2 shards, ~14.2 GB
hf download Qwen/Qwen-Image-2.1-Turbo --include "text_encoder/*" \
  --local-dir <ckpt>     # Qwen text encoder, 1 shard, ~17.5 GB
hf download Qwen/Qwen-Image-2.1-Turbo --include "processor/*" \
  --local-dir <ckpt>     # tokenizer.json + configs, KB
hf download Qwen/Qwen-Image-2.1-Turbo --include "vae/*" \
  --local-dir <ckpt>     # VAE decoder, ~0.676 GB
hf download Qwen/Qwen-Image-2.1-Turbo \
  --include "scheduler/*" --include "model_index.json" \
  --local-dir <ckpt>     # Turbo 8-step sigmas + component map, KB
```

Subset guide: VAE-only parity (`--decode-latents-in`) needs just `vae/`;
encoder-only (`--encoder` gate in `scripts/qwen_image_21_verify.sh`) needs
`text_encoder/` + `processor/`; the strict end-to-end gate needs
`transformer/` plus the golden artifacts (see below).

Flag mapping: `--weights` = `<ckpt>/transformer` (shard dir or single
`.safetensors`); `--text-encoder` = `<ckpt>/text_encoder` (single-shard
`model.safetensors`); `--tokenizer` = `<ckpt>/processor` (dir holding
`tokenizer.json`); `--vae-weights` =
`<ckpt>/vae/diffusion_pytorch_model.safetensors` (validated whenever
given, decoded only with `--vae-decode`). Only the MiB-scale `.f32`
handoff files (`[L,Cc]` embeds, `[C,H,W]` latents) ever need to move
between machines — never the 32 GB weights.

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
| `--prompt` | none (loops) | caption text; giving it runs one generation and exits, omitting it (TTY, no file/parity flags) enters the interactive prompt/filename loop |
| `--tokenizer` | — | dir holding `tokenizer.json`; with `--text-encoder` runs the in-process Qwen forward |
| `--text-encoder` | — | Qwen text-encoder weights: dir or single `.safetensors` |
| `--text-weights` | — | raw float32 LE caption features `[L,Cc]`; `L` is inferred from the file size (and must be <= the backend max sequence length) |
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
| `--decode-latents-in` | — | raw fp32 LE latents to decode directly (skips encode/DiT/sampling; needs `--vae-weights`); H/W are latent dims |
| `--decode-latents-layout` | `chw` | `chw` = driver layout `lat[(c*H+h)*W+w]`; `token` = torch packed `[H*W,C]` (golden final latents) |
| `--decode-out-f32` | — | also write the post-clamp RGBA decode as raw f32 `[H,W,4]` (VAE-only parity) |
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

## VAE-only parity

`--decode-latents-in FILE --decode-latents-layout token --vae-weights
<ckpt>/vae/diffusion_pytorch_model.safetensors --height 64 --width 64`
feeds the golden final latents straight through the runtime
`vae/config.json` affine and the native decoder, without the encoder, the
DiT, or the sampler (seconds instead of minutes). Use `--decode-out-f32`
to also dump the post-clamp RGBA in `[-1,1]` for float-space diffs:

```sh
qwen_image_21_generate \
  --decode-latents-in <golden>/qwen_image_21_turbo_final_latents.f32 \
  --decode-latents-layout token \
  --vae-weights <ckpt>/vae/diffusion_pytorch_model.safetensors \
  --height 64 --width 64 --decode-out-f32 /tmp/q21_vae_rgba.f32 \
  --out /tmp/q21_vae.png
```

The golden PNG is itself a bf16 decode, so this path tops out at ~47 dB
PSNR against it (an independent fp32 decoder reaches the same); the native
decoder's semantic parity is ~61 dB against a torch fp32 decode of the same
latents.

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
# Smoke test (no weights, CPU or GPU; --prompt keeps it one-off):
qwen_image_21_generate --tiny --steps 2 --staged --prompt "smoke test"

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

## Golden verification

The example is checked against a diffusers/torch reference for the fixed
prompt `"A capybara wearing a wizard hat, oil painting"`, 1024x1024, seed 42,
and the checkpoint's 8-step Turbo schedule. Fetch the checkpoint as above,
then generate the reference once (torch + CUDA) and run the harness:

```sh
python3 scripts/qwen_image_21_golden.py \
  --weights /workspace/weights/Qwen-Image-2.1-Turbo \
  --out /workspace/weights/Qwen-Image-2.1-Turbo-golden

scripts/qwen_image_21_verify.sh \
  --golden /workspace/weights/Qwen-Image-2.1-Turbo-golden
```

`qwen_image_21_golden.py` needs `torch` (CUDA build recommended),
`diffusers` with `QwenImage21Pipeline` (install from source:
`pip install git+https://github.com/huggingface/diffusers.git`),
`transformers>=5.17.0`, `accelerate`, `pillow`, `safetensors`, and `numpy`
(manifest run: torch 2.14.1+cu130, diffusers 0.42.0.dev0, transformers
5.20.2). It writes prompt embeddings, initial latents, every
per-step `noise_pred` and post-Euler latents, the final latents, the PNG, and
a `manifest.json`. Regenerating them reproduces the versioned scalar
checksums in `scripts/qwen_image_21_golden_checksums.txt` bit-for-bit.
`qwen_image_21_compare.py` reports per-tensor max/mean/RMS/relative-L2/
cosine/sums and image PSNR.

The harness runs three gates:

1. **Strict end-to-end (identical inputs).** Feeds the torch prompt
   embeddings and torch seed-42 latents to the driver, samples 8 Turbo
   steps, dumps every step, and compares all tensors + the decoded PNG.
   Gate: relative L2 <= 2e-2 per tensor, image PSNR >= 35 dB.
2. **VAE-only.** Decodes the golden final latents through
   `--decode-latents-in` (no encoder, no DiT, no sampler). Gate: PSNR
   >= 45 dB, the ceiling for a bf16-decoded golden PNG.
3. **Encoder-only (`--encoder`).** Runs the in-process GPU Qwen forward and
   one sigma=1.0 forward on the golden latents; gates `prompt_embeds` at
   relative L2 <= 3e-2 and per-row cosine >= 0.999 (bf16 device noise, see
   below).

Measured on an RTX 3090 (2026-10-24; native Kokkos GEMM/attention, bf16):

| Check | Result |
| ----- | ------ |
| prompt_embeds / initial_latents (strict) | exact, relative L2 0 |
| per-step `noise_pred` | relative L2 0.78% - 1.34% |
| final latents | relative L2 **1.04%**, cosine 0.99995 |
| decoded image | **41.54 dB** PSNR vs the golden PNG |
| VAE-only decode of golden latents | **48.24 dB** PSNR (>= 45 dB gate) |
| in-process encoder `prompt_embeds` | relative L2 2.34%, min row cosine 0.99917 |
| step-0 `noise_pred` with in-process embeds | relative L2 0.88% (same as strict) |
| end-to-end with in-process GPU encode | relative L2 ~1.9% (encoder noise dominates) |

The strict gate deliberately uses the golden prompt embeddings: the
reference encodes on CPU and the example encodes on GPU, and bf16
reduction-order differences on the Qwen3-VL tower's large activations move
individual token rows enough to dominate the end-to-end error (the row sums
in `scripts/qwen_image_21_golden_checksums.txt` are device-sensitive for
this reason). Feeding the golden embeddings isolates DiT + VAE numerics, so
a regression in either is not masked by encoder noise.
