#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
"""Torch reference for the Sana DC-AE VAE decoder golden test.

Loads the diffusers DC-AE checkpoint, decodes small fixed-seed latents, and
dumps input latents + output stats to JSON for Test_SanaVAE.GoldenDecodeVsTorch.

    python3 examples/sana/generate_vae_golden.py \
        --repo mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
        --out vae_golden.json [--latent-hw 4] [--seed 7]

Needs torch + diffusers (CPU fp32 is fine). Never run from ctest; the C++
test only reads the JSON via KOKKOSDNN_SANA_VAE_GOLDEN_JSON.
"""
import argparse
import json

import torch
from diffusers import AutoencoderDC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    ap.add_argument("--out", default="vae_golden.json")
    ap.add_argument("--latent-hw", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ae = AutoencoderDC.from_pretrained(args.repo, torch_dtype=torch.float32)
    ae.eval()
    print("scaling_factor:", ae.config.scaling_factor)
    assert abs(ae.config.scaling_factor - 0.41407) < 1e-6

    g = torch.Generator().manual_seed(args.seed)
    lat = torch.randn(1, 32, args.latent_hw, args.latent_hw, generator=g)
    with torch.no_grad():
        rgb = ae.decode(lat / ae.config.scaling_factor, return_dict=False)[0]
    rgb = rgb.float()
    print("latents:", tuple(lat.shape), "rgb:", tuple(rgb.shape))

    flat = rgb.reshape(-1)
    payload = {
        "repo": args.repo,
        "seed": args.seed,
        "latent_channels": 32,
        "latent_hw": args.latent_hw,
        "latents": lat.reshape(-1).tolist(),
        "out_hw": rgb.shape[2],
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "first8": flat[:8].tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print("wrote", args.out, {k: payload[k] for k in ("mean", "std", "min", "max")})


if __name__ == "__main__":
    main()
