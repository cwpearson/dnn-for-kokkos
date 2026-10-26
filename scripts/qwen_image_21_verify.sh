#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#
# End-to-end golden verification for the Qwen-Image-2.1-Turbo example.
#
# Runs the native driver on the fixed torch-golden inputs (prompt embeddings
# + initial latents), dumps every stage, and compares against the diffusers
# reference artifacts produced by scripts/qwen_image_21_golden.py.
#
#   scripts/qwen_image_21_verify.sh [options]
#
# Options:
#   --binary PATH     qwen_image_21_generate binary to test
#                     (default: build/qwen-image/examples/qwen_image_21/...)
#   --weights DIR     checkpoint root (default: /workspace/weights/Qwen-Image-2.1-Turbo)
#   --golden DIR      golden artifact dir (default: <weights>-golden)
#   --work DIR        scratch dir for driver output (default: /tmp/qwen21-verify)
#   --regen-golden    regenerate the golden artifacts (needs torch + CUDA)
#   --in-process      also run the full in-process pipeline (GPU encoder,
#                     driver latent init) as an informational, non-strict run
#   --encoder         also run the in-process GPU encoder (one sigma=1 step)
#                     and gate prompt_embeds against the golden
#   --no-vae-only     skip the VAE-only decode of the golden final latents
#   --no-decode       skip the VAE image comparison (latents only)
#
# Exit status: 0 only when the strict latent/step comparison passes (and, if
# images are compared, PSNR >= the comparator's threshold).
#
# The golden artifacts (prompt_embeds.f32, initial_latents.f32, per-step
# tensors, final latents, PNG) are large and live next to the weights; they
# are not versioned. Regenerate them with --regen-golden or directly:
#
#   python3 scripts/qwen_image_21_golden.py \
#       --weights /workspace/weights/Qwen-Image-2.1-Turbo \
#       --out /workspace/weights/Qwen-Image-2.1-Turbo-golden
#
# The strict run is an identical-inputs test: it deliberately feeds the
# torch prompt embeddings and torch seed-42 latents so DiT + VAE numerics are
# compared in isolation from encoder-device and RNG differences.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
weights="/workspace/weights/Qwen-Image-2.1-Turbo"
golden=""
work="/tmp/qwen21-verify"
binary="$repo_root/build/qwen-image/examples/qwen_image_21/qwen_image_21_generate"
regen=0
in_process=0
encoder=0
vae_only=1
decode=1
prompt="A capybara wearing a wizard hat, oil painting"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) binary="$2"; shift 2 ;;
    --weights) weights="$2"; shift 2 ;;
    --golden) golden="$2"; shift 2 ;;
    --work) work="$2"; shift 2 ;;
    --regen-golden) regen=1; shift ;;
    --in-process) in_process=1; shift ;;
    --encoder) encoder=1; shift ;;
    --no-vae-only) vae_only=0; shift ;;
    --no-decode) decode=0; shift ;;
    -h|--help) sed -n '2,52p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$golden" ]] || golden="${weights}-golden"
[[ -x "$binary" ]] || { echo "error: no executable at $binary (build with -DKokkosDNN_ENABLE_IMAGE=ON?)" >&2; exit 2; }
[[ -d "$weights" ]] || { echo "error: no checkpoint at $weights" >&2; exit 2; }

golden_png="$golden/qwen_image_21_turbo_capybara.png"
if [[ "$regen" == 1 || ! -f "$golden/prompt_embeds.f32" || ! -f "$golden/initial_latents.f32" ]]; then
  echo "==> generating torch golden artifacts in $golden"
  rm -rf "$golden"
  python3 "$repo_root/scripts/qwen_image_21_golden.py" --weights "$weights" --out "$golden"
fi

for f in prompt_embeds.f32 initial_latents.f32 \
         qwen_image_21_turbo_final_latents.f32; do
  [[ -f "$golden/$f" ]] || {
    echo "error: golden artifact $golden/$f is missing;" \
         "rerun with --regen-golden" >&2
    exit 2
  }
done

rm -rf "$work"
mkdir -p "$work/driver"

echo "==> strict identical-inputs run: torch embeds + torch latents"
"$binary" --staged \
  --weights "$weights/transformer" \
  --prompt "$prompt" \
  --prompt-embeds-in "$golden/prompt_embeds.f32" \
  --init-latents-in "$golden/initial_latents.f32" \
  --height 1024 --width 1024 --steps 8 --sigmas turbo --cfg 1.0 \
  --vae-weights "$weights/vae/diffusion_pytorch_model.safetensors" --vae-decode \
  --dump-steps "$work/driver" \
  --out "$work/out.png" 2>&1 | tee "$work/driver.log"

compare_args=(--golden "$golden" --actual "$work/driver" --strict
              --final-golden "$golden/qwen_image_21_turbo_final_latents.f32"
              --final-golden-layout tokens
              --final-actual "$work/out.f32")
if [[ "$decode" == 1 && -f "$golden_png" && -f "$work/out.png" ]]; then
  compare_args+=(--images "$golden_png" "$work/out.png" --psnr-min 35)
fi

echo "==> comparing dumped tensors to the torch golden"
python3 "$repo_root/scripts/qwen_image_21_compare.py" "${compare_args[@]}" \
  | tee "$work/compare.log"

if [[ "$vae_only" == 1 && -f "$weights/vae/diffusion_pytorch_model.safetensors" ]]; then
  # VAE-only gate: golden final latents -> runtime affine -> native decode,
  # no encoder/DiT/sampler. The golden PNG is itself a bf16 decode, so ~48 dB
  # (not 60+) is the achievable ceiling for this comparison.
  echo "==> VAE-only decode of the golden final latents"
  "$binary" \
    --decode-latents-in "$golden/qwen_image_21_turbo_final_latents.f32" \
    --decode-latents-layout token \
    --vae-weights "$weights/vae/diffusion_pytorch_model.safetensors" \
    --height 1024 --width 1024 \
    --out "$work/vae_only.png" 2>&1 | tee "$work/vae_only.log"
  if [[ -f "$work/vae_only.png" ]]; then
    python3 "$repo_root/scripts/qwen_image_21_compare.py" \
      --images "$golden_png" "$work/vae_only.png" \
      --psnr-min 45 --strict | tee "$work/vae_compare.log"
  fi
fi

if [[ "$encoder" == 1 ]]; then
  # Encoder-only gate: in-process GPU Qwen encode + ONE sigma=1.0 forward on
  # the golden initial latents. The step tensors inherit the encoder's bf16
  # reduction-order noise, so only prompt_embeds is gated here.
  echo "==> encoder check (in-process Qwen forward, one step)"
  mkdir -p "$work/encoder"
  "$binary" --staged \
    --weights "$weights/transformer" \
    --text-encoder "$weights/text_encoder" --tokenizer "$weights/processor" \
    --prompt "$prompt" \
    --height 1024 --width 1024 --steps 1 --sigmas driver --cfg 1.0 \
    --init-latents-in "$golden/initial_latents.f32" \
    --dump-steps "$work/encoder/driver" \
    --out "$work/encoder/out.png" 2>&1 | tee "$work/encoder.log"
  echo "==> encoder prompt_embeds vs the torch golden (tolerance: bf16 device noise)"
  python3 "$repo_root/scripts/qwen_image_21_compare.py" \
    --golden "$golden" --actual "$work/encoder/driver" \
    --only prompt_embeds.f32 --rtol 3e-2 --row-cos-min 0.999 --strict \
    | tee "$work/encoder_compare.log"
fi

if [[ "$in_process" == 1 ]]; then
  echo "==> informational full in-process run (GPU encoder, driver RNG)"
  mkdir -p "$work/inproc"
  "$binary" --staged \
    --weights "$weights/transformer" \
    --text-encoder "$weights/text_encoder" --tokenizer "$weights/processor" \
    --prompt "$prompt" \
    --height 1024 --width 1024 --steps 8 --sigmas turbo --cfg 1.0 --seed 42 \
    --vae-weights "$weights/vae/diffusion_pytorch_model.safetensors" --vae-decode \
    --dump-steps "$work/inproc/driver" \
    --out "$work/inproc/out.png" 2>&1 | tee "$work/inproc.log"
  echo "==> informational (non-strict; different RNG and encode device)"
  python3 "$repo_root/scripts/qwen_image_21_compare.py" \
    --golden "$golden" --actual "$work/inproc/driver" \
    --final-golden "$golden/qwen_image_21_turbo_final_latents.f32" \
    --final-golden-layout tokens \
    --final-actual "$work/inproc/out.f32" || true
fi

echo "PASS: Qwen-Image-2.1-Turbo matches the torch golden"
