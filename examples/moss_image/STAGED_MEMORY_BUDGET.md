# MOSS-Image-SFT staged memory budget (24 GB target)

Reference GPU: NVIDIA GeForce RTX 3090, 24576 MiB = **24 GiB**
(25.77 GB decimal). Weight sizes from `/tmp/opencode/moss-verify-report.md`
(safetensors payloads; dtypes verified via header peek, full 27.2 GB weights
**not** downloaded).

## 1. Weight footprints

| Component | Params | Dtype | Payload (bytes) | Payload (GiB) |
|---|---|---|---|---|
| Text encoder (Qwen3.5, `text_encoder/`, 759 tensors) | 8 392 695 024 (~8.4 B) | BF16 | 16 785 390 048 | ~15.63 |
| Transformer DiT (`transformer/`, 463 tensors; `rope.bands` FP32, rest BF16) | ~5.08 B ("5B" DiT) | BF16 + 1 FP32 tensor | 10 164 617 280 | ~9.47 |
| VAE (`vae/`, `AutoencoderKLFlux2`, 251 tensors) | — | FP32 | 249 521 372 | ~0.23 |
| **All weights (on-disk incl. headers)** | | | **27 199 675 780 (~27.20 GB / ~25.33 GiB)** | **~25.33** |

The weights alone (~25.33 GiB) already exceed the 24 GiB budget before any
activations, CUDA context (~0.5–1 GiB), or fragmentation margin.

## 2. Activation estimate, 1024 px, CFG 2x

Canonical run: 1024x1024, 50 steps, `guidance_scale=3.0` (2 DiT forwards per
step: cond + uncond-with-skip-22), shift 5.0, fp32 noise/scheduler updates,
bf16 DiT / bf16 text encoder / fp32 VAE (see `moss_golden.py` canon).

| Tensor / stage | Shape | Dtype | Size |
|---|---|---|---|
| Initial/final latents | [1, 128, 64, 64] (1024/16) | FP32 | 2.0 MiB |
| Prompt embeds (example geometry) | [512, 2048] | FP32 | 4.0 MiB |
| Prompt embeds (real SFT, e.g. S=512) | [512, 4096] | FP32 | 8.0 MiB |
| DiT activations per forward, 1024 px (42 layers, dim 3072, ~1024 img + up to 4096 text tokens; QKVO + attention scores + FFN 8192) | — | BF16/FP32 mix | ~1–2 GiB transient |
| CFG 2x (cond + uncond alive per step) | 2x above | — | ~2–4 GiB transient peak |
| Text-encoder forward transient (8.4 B model, short prompt) | — | BF16 | ~1–2 GiB transient |
| VAE decode transient (32 ch spatial, 1024 px) | — | FP32 | ~1–2 GiB transient |

## 3. Fit table: simultaneous vs staged (24 GiB budget)

| Strategy | Peak resident | Fits 24 GiB? |
|---|---|---|
| Simultaneous: encoder 15.63 + DiT 9.47 + VAE 0.23 + DiT act ~2–4 (CFG 2x) + ctx ~0.5–1 | **~28–31 GiB** | **No** (OOM before step 1) |
| Staged phase A (encoder only): 15.63 + enc transient ~1–2 + ctx | **~17–19 GiB** | **Yes** (~5 GiB margin) |
| Staged phase B (DiT only): 9.47 + act ~2–4 (CFG 2x) + ctx | **~12–14 GiB** | **Yes** (~10 GiB margin) |
| Staged phase C (VAE only): 0.23 + decode ~1–2 + ctx | **~2–4 GiB** | **Yes** |
| Embed handoff on host: [L, Cc] FP32 (e.g. 512x2048 = 4 MiB) | negligible | — |

Staging converts one impossible ~30 GiB peak into three comfortable peaks
(~18 / ~14 / ~4 GiB), each with headroom for allocator fragmentation.

## 4. Two-process commands (encode once, reuse embeds)

```bash
# Process 1 -- encode (GPU box with weights, or --null offline):
python3 scripts/moss_encode.py \
    --prompt "a red panda reading a newspaper" \
    --out prompt_embeds.f32 --L 512 --Cc 2048
# offline fallback (no torch/diffusers/weights/GPU):
python3 scripts/moss_encode.py --null \
    --prompt "smoke" --out /tmp/tool_embeds.f32 --L 512 --Cc 2048

# Process 2 -- generate on the 24 GB box (DiT + VAE only):
moss_image_generate --prompt-embeds-in prompt_embeds.f32 \
    --weights moss_image_sft.safetensors \
    --height 1024 --width 1024 --steps 50 --cfg 3.0 --shift 5.0 \
    --skip-layers 22 --seed 42 --out moss_out.png
```

The `.f32` file is raw float32 LE `[L, Cc]` row-major (see
`scripts/moss_encode.py` header and `Moss::load_caption_raw`); the
`<out>.json` sidecar records `L`, `Cc`, `model`, `prompt`, `seq_len`.
Copy the tiny embed file (MiB) to the GPU box instead of the 16.8 GB encoder.

## 5. In-process `--staged` phase diagram

Single `moss_image_generate --staged` process, one component resident at a
time; the MiB-scale `[L, Cc]` cond/uncond embeds stay resident on the device
across the encoder-release / DiT-load boundary (no host spill, no disk):

```
--staged
  |
  v
[1] load text encoder (15.6 GiB) + tokenizer
  |
  v
[2] encode prompt -> device [L, Cc] cond + [L, Cc] uncond (stay resident)
  |
  v
[3] release encoder views/weights + Kokkos fence -> free ~15.6 GiB
  |
  v
[4] load DiT (9.5 GiB); embeds already resident on device
  |
  v
[5] flow-Euler sample, CFG 2x (cond + skip-22 uncond), 50 steps, fp32 updates
  |
  v
[6] release DiT (optional) -> load VAE (0.23 GiB) -> decode -> image
```

Phase peaks: [1–2] ~18 GiB, [4–5] ~14 GiB, [6] ~4 GiB -- all under 24 GiB.

## 6. C++ flag contract (impl worktree)

The staged implementation of `moss_image_generate` MUST accept:

| Flag | Meaning |
|---|---|
| `--prompt-embeds-in <file>` | load raw float32 LE `[L, Cc]` caption features (same format as today's `--text-weights`); keep `--text-weights` as a deprecated alias |
| `--staged` | in-process staging per the diagram above (encode -> release encoder weights, embeds stay resident on device -> fence -> load DiT -> sample from device embeds -> VAE) |

Compatibility: `scripts/moss_encode.py --out X.f32` output must load via
`--prompt-embeds-in X.f32`; `--null` output (zeros) must reproduce today's
null-embedding smoke path bit-for-bit.
```

