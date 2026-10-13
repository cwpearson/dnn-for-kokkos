#!/usr/bin/env python3
"""Torch reference for the KokkosDNN Sana example (Test_SanaGolden).

Runs a single SanaMS_600M_P1_D28 forward on CPU with fixed-seed latents and
the checkpoint null caption, and dumps output statistics to JSON for
Test_SanaGolden.RealWeightsForwardFp32 (KOKKOSDNN_SANA_GOLDEN_JSON).

Requires: torch + an NVlabs/Sana checkout (imports diffusion.model.builder).

    python3 examples/sana/generate_golden.py \
        --repo /path/to/Sana \
        --weights Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth \
        --out /tmp/sana_golden.json [--seed 1234] [--timestep 1000]

The C++ side reproduces this with: fixed latents = sample_gaussian(32*32*32,
seed), caption = y_embedding rows (null), t = <timestep> (SanaMS truncates
to long internally; the C++ Model does the same), fp32 CPU.
"""

import argparse
import json
import sys

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="NVlabs/Sana checkout dir")
    ap.add_argument("--weights", required=True, help=".pth checkpoint")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timestep", type=int, default=1000)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from diffusion.model.builder import build_model

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.weights, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    model = build_model(
        "SanaMS_600M_P1_D28",
        input_size=32,
        patch_size=1,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=2.5,
        caption_channels=2304,
        model_max_length=300,
        use_pe=False,
        y_norm=True,
        y_norm_scale_factor=0.01,
        attn_type="linear",
        cross_attn_type="flash",
        ffn_type="glumbconv",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        pred_sigma=False,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    model.eval().float()

    # Null caption straight from the checkpoint buffer, shaped as the
    # pipeline's caption_embs [B,1,L,C].
    y_null = sd["y_embedder.y_embedding"].float()  # [300,2304]
    y = y_null.unsqueeze(0).unsqueeze(0)  # [1,1,300,2304]

    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, 32, 32, 32, generator=gen)
    t = torch.tensor([args.timestep], dtype=torch.long)
    with torch.no_grad():
        out = model(x.float(), t, y.float())
    out = out.float()
    flat = out.reshape(-1)
    golden = {
        "model": "SanaMS_600M_P1_D28",
        "weights": args.weights,
        "seed": args.seed,
        "timestep": args.timestep,
        "latent_shape": [1, 32, 32, 32],
        "caption": "y_embedding null [1,1,300,2304]",
        "dtype": "float32",
        "mean": float(out.mean()),
        "std": float(out.std(unbiased=False)),
        "min": float(out.min()),
        "max": float(out.max()),
        "first8": [float(v) for v in flat[:8]],
    }
    with open(args.out, "w") as f:
        json.dump(golden, f, indent=2)
    print("mean=%.6f std=%.6f min=%.4f max=%.4f -> %s"
          % (golden["mean"], golden["std"], golden["min"], golden["max"],
             args.out))


if __name__ == "__main__":
    main()
