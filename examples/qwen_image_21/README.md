# Qwen-Image-2.1 example

Text-to-image (`qwen_image_21_generate`) plus one-shot reference-image editing (`qwen_image_21_edit`) for `Qwen/Qwen-Image-2.1-Turbo`, built on the core library with staged execution (encode → sample → decode) so the 17.5 GB text encoder, 14.2 GB DiT, and VAE are never resident at once.

Scope: **inference only** (Qwen3-VL joint text+vision encode, DiT FlowEuler sampling with CFG, VAE encode/decode). `generate` is text-only by design; all image conditioning lives in `edit`. Both drivers share `qwen_image_21_common.hpp`, so sampler, dumps, and decode cannot drift apart.

## Quickstart

From repo root `/workspace`:

```bash
# 1. Build and install Kokkos once (CUDA, AMPERE86)
git clone --branch 5.2.1 --recursive https://github.com/kokkos/kokkos.git kokkos-src
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-install \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build kokkos-build --parallel 8
cmake --install kokkos-build

# 2. Configure and build the example targets (CUDA, cuBLAS + cuDNN)
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkosDNN_ENABLE_CUBLAS=ON \
  -DKokkosDNN_ENABLE_CUDNN=ON \
  -DKokkosDNN_ENABLE_IMAGE=ON
cmake --build build --parallel 8 \
  --target qwen_image_21_generate qwen_image_21_edit

# 3. Get the checkpoint (public repo, ~32.5 GB / ~30.3 GiB on disk)
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen-Image-2.1-Turbo --local-dir <ckpt>

# 4. Run the real checkpoint
./build/examples/qwen_image_21/qwen_image_21_generate --ckpt <ckpt> \
  --prompt "a red panda reading a newspaper" --out qwen_out.png
```

`--ckpt` is the canonical checkpoint root; every component is derived from it (`transformer/`, `text_encoder/model.safetensors`, `processor/tokenizer.json`, `vae/`, plus `scheduler/` + `model_index.json` for the Turbo schedule). On a small disk fetch pieces with repeated `--include "transformer/*"` / `"text_encoder/*"` / `"processor/*"` / `"vae/*"` / `"scheduler/*" "model_index.json"` into the same `<ckpt>` dir. VAE-only parity needs just `vae/`; the strict end-to-end gate needs `transformer/` plus the golden artifacts (see Verification).

## Example commands

Tiny smoke runs (random weights, 4x4 latents, no downloads; `--tiny` forces 2 steps unless `--steps` is given):

```bash
./build/examples/qwen_image_21/qwen_image_21_generate --tiny --steps 2 --prompt "smoke test"
./build/examples/qwen_image_21/qwen_image_21_edit --tiny --prompt "smoke edit"
```

`generate` without `--prompt` on a TTY (and without any file/parity flag) runs as a REPL: it prints the current pixel `HxW` (and latent `HxW`), asks for a prompt, asks for a filename, generates, and repeats until EOF or `quit`/`exit`. The previous prompt is editable in place (UP/DN history, never written to disk), `/w N` / `/h N` resize going forward, ESC at the filename step goes back to the prompt step, empty filename auto-numbers `qwen_image_21_out_<n>.png`, and the per-iteration seed is `--seed + iteration`. Giving `--prompt` (or any of `--prompt-embeds-in` / `--text-weights`, `--init-latents-in`, `--dump-steps`, `--decode-latents-in`, or piping stdin) runs ONE generation and exits:

```bash
./build/examples/qwen_image_21/qwen_image_21_generate --ckpt <ckpt>
# [size] 1024x1024 px (latent 64x64)
# prompt> a red panda reading a newspaper
# output> red_panda.png
# prompt> quit
```

Full staged runs (staged is the default; pass `--no-staged` for single-phase):

```bash
./build/examples/qwen_image_21/qwen_image_21_generate --ckpt <ckpt> \
  --prompt "a red panda reading a newspaper" \
  --height 1024 --width 1024 --steps 8 --cfg 1.0 \
  --out qwen_out.png
./build/examples/qwen_image_21/qwen_image_21_edit --ckpt <ckpt> \
  --image sketch.png --prompt "make it photorealistic" \
  --steps 8 --cfg 1.0 --out edit.png
```

Sampler / parity variants (same flags on both drivers unless noted):

```bash
# Turbo schedule verbatim (mu-shifted 8-step sigmas) instead of the default driver grid
./build/examples/qwen_image_21/qwen_image_21_generate --ckpt <ckpt> \
  --prompt "..." --steps 8 --sigmas turbo --shift 3.0 --seed 42 --out qwen_out.png
# File/parity handoffs (raw float32 LE; embeds [L,Cc], latents [C,H,W] C-major)
./build/examples/qwen_image_21/qwen_image_21_generate --ckpt <ckpt> \
  --prompt-embeds-in embeds.f32 --init-latents-in latents.f32 \
  --dump-steps /tmp/q21 --out qwen_out.png
# VAE-only decode of golden final latents (no encoder/DiT/sampler)
./build/examples/qwen_image_21/qwen_image_21_generate --ckpt <ckpt> \
  --decode-latents-in <golden>/qwen_image_21_turbo_final_latents.f32 \
  --decode-latents-layout token \
  --height 1024 --width 1024 --decode-out-f32 /tmp/q21_vae_rgba.f32 \
  --out /tmp/q21_vae.png
```

`edit` is one-shot only (no REPL; `--image` is repeatable, up to 10, in `--image` order like the reference `image=[...]` list). `--height`/`--width` override only the output size; otherwise geometry comes from the reference `calculate_dimensions` at `--output-resolution` with `/32` rounding (each image from its own aspect; the default output size derives from the LAST image's aspect). `--ref-latents-in` (one file per `--image` in order, or none) overrides the default in-process VAE encode of each resized condition image (else zeros); `--vis-rows-in` carries raw int32 LE vision-slot rows when no joint build is available. Full flag list lives in `qwen_image_21_generate.cpp` / `qwen_image_21_edit.cpp` (`--tiny`, `--ckpt`, `--prompt`, `--text-weights` / `--prompt-embeds-in`, `--height`, `--width`, `--steps`, `--cfg` / `--guidance`, `--shift` / `--flow-shift`, `--sigmas driver|turbo`, `--seed`, `--init-latents-in`, `--no-vae-decode`, `--decode-latents-in` / `--decode-latents-layout` / `--decode-out-f32`, `--out`, `--no-staged`, `--dump-steps`, plus edit-only `--image`, `--output-resolution`, `--ref-latents-in` / `--ref-latents-layout`, `--vis-rows-in`).

## Notes

Layout: `qwen_image_21_generate.cpp` (T2I driver + REPL), `qwen_image_21_edit.cpp` (one-shot edit driver), `qwen_image_21_common.hpp` (shared encode / sample / decode / dumps / writer; NOT installed). The true-pipeline model headers live in `models/qwen_image_21/` and are probed via `__has_include`, never required — without them (or without `--ckpt`) each phase falls back to placeholder / file / null + grayscale preview with a logged reason, so configure and `--tiny` always work.

Pipeline: `encode -> fence -> release encoder -> load DiT -> sample from device-resident embeds -> release DiT -> fence -> VAE decode`. The MiB-scale embeds are separate device allocations from the GiB-scale encoder weights and stay resident across the DiT load — no disk round-trip, no second process. In the REPL the encoder and transformer are loaded once into host memory and copied host→device per iteration, keeping the one-component-resident property without re-reading ~32 GB per prompt. Each phase logs its resident estimate.

| Phase | Resident | Fits 24 GB? |
| ----- | -------- | ----------- |
| encode | 17.5 GB encoder | FITS (released before DiT load) |
| sample | 14.2 GB DiT + activations | FITS |
| vae decode | 0.7 GB | FITS |
| staged peak | ~18 GB + activations | FITS |
| unstaged (all resident) | ~32 GB + activations | NO |

Conditioning precedence: in-process Qwen (`--ckpt`: raw-text BPE of the T2I template, or the edit template with one `<imageN>` vision slot per `--image` through the Qwen3-VL vision tower + `forward_joint` with true multi-image M-RoPE and DeepStack, uncond zeros) → `--prompt-embeds-in` file (alias `--text-weights`) → null zeros, with the prompt source logged. `--tiny` skips the Qwen forward. VAE decode runs by default from `<ckpt>/vae` (runtime `vae/config.json` affine + native decode to RGBA); `--no-vae-decode` writes the channel-mean grayscale preview instead. `edit` stages VAE-encoded reference latents as packed cond tokens (`[cond|target]` + `img_shapes`/`img_mask` via `Dit::forward_cond`); without vision-slot rows (joint build minus `drop_idx`, or `--vis-rows-in`) it falls back to target-only sampling with a logged reason.

Geometry: `--height`/`--width` are output pixels, each rounded to the nearest 16·patch-px multiple (ties up, minimum one stride) so the latent grid tiles the DiT patch. Dumps (`--dump-steps DIR`) write raw little-endian float32 in driver layout (latents `[C,H,W]` C-major `lat[(c*H+h)*W+w]`) — `initial_latents.f32`, per-step `stepNN_noise_pred.f32` (+ `_cond`/`_uncond` under CFG) and `stepNN_latents.f32`, plus `ref_latents.f32` (single image) or `ref_latents_<i>.f32` (multi-image) and `img_mask.f32` on the edit cond path — with a `manifest.json` (`format`, prompt/source, H/W, channels, steps, sigmas, seed, cfg, `use_cfg`, `staged`, model label, per-tensor f64 sums). Each tensor is also logged as `[dump] <name> <numel> floats f64sum=...`; the extra device→host copies only happen when the flag is set.

## Verification

No ctest coverage (torch references never run from ctest). Golden check for the fixed prompt `"A capybara wearing a wizard hat, oil painting"`, 1024x1024, seed 42, 8-step Turbo schedule:

```bash
python3 scripts/qwen_image_21_golden.py \
  --weights /workspace/weights/Qwen-Image-2.1-Turbo \
  --out /workspace/weights/Qwen-Image-2.1-Turbo-golden
scripts/qwen_image_21_verify.sh \
  --golden /workspace/weights/Qwen-Image-2.1-Turbo-golden
```

`qwen_image_21_golden.py` needs torch (CUDA build recommended), `diffusers` with `QwenImage21Pipeline` (install from source), `transformers>=5.17.0`, `accelerate`, `pillow`, `safetensors`, `numpy`. It writes prompt embeddings, initial latents, every per-step `noise_pred` and post-Euler latents, final latents, PNG, and `manifest.json`; regenerating reproduces `scripts/qwen_image_21_golden_checksums.txt` bit-for-bit. `qwen_image_21_compare.py` reports per-tensor max/mean/RMS/relative-L2/cosine/sums and image PSNR.

Two gates: **strict end-to-end** (torch embeds + torch seed-42 latents in, 8 Turbo steps, dump compare + decoded PNG; relative L2 ≤ 2e-2 per tensor, PSNR ≥ 35 dB) and **VAE-only** (golden final latents through `--decode-latents-in`; PSNR ≥ 45 dB, the ceiling against a bf16-decoded golden PNG). Measured on an RTX 3090 (2026-10-24, native Kokkos GEMM/attention, bf16): initial latents exact (rel L2 0); per-step `noise_pred` rel L2 0.78%–1.34%; final latents rel L2 **1.04%**, cosine 0.99995; decoded image **41.54 dB**; VAE-only **48.24 dB**. The strict gate deliberately feeds golden embeds: the reference encodes on CPU while the example encodes on GPU, and bf16 reduction-order differences on the tower's large activations dominate end-to-end error — feeding golden embeds isolates DiT + VAE numerics. `verify.sh --in-process` runs the full in-process pipeline as an informational non-strict run; `--edit` adds the strict identical-inputs edit gate plus the in-process VAE-encode gate. The native decoder reaches ~61 dB against a torch fp32 decode of the same latents.

## Performance

BF16 DiT/encoder with FP32 sampler math; no weight quantization. Staged peak is ~18 GB + activations (fits 24 GB); unstaged needs ~32 GB + activations (does not fit). Serial-CPU full-res runs are hours (same kernels, no parallelism) — use CUDA for production; `--tiny` (4x4 latents, 2 steps) is the fast path for smoke tests and stays under a second on either backend. VAE-only decode is seconds instead of minutes since it skips encoder, DiT, and sampler. Do not benchmark the grayscale preview path (`--no-vae-decode`) as a proxy for decode throughput.
