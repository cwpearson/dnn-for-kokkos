#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
"""Stage-A prompt encoder for the MOSS-Image staged split.

Produces the caption-embedding file consumed by the C++ driver
(`moss_image_generate --prompt-embeds-in <file>`):

  Embed file format (must match `Moss::load_caption_raw` in
  `examples_moss_baseline/KokkosDNN_MossWeights.hpp`):
    - raw float32, little-endian, row-major `[L, Cc]` (no header)
    - exactly `L * Cc` floats (`L <= model_max_length`, `Cc == caption_channels`)
    - flattened `[1, L, Cc]` / `[1, 1, L, Cc]` layouts are also accepted
      by the loader (last `[L, Cc]` block wins)

Modes:
  (a) real Qwen encode (default): loads the MOSS-Image pipeline with
      `DiffusionPipeline.from_pretrained(..., custom_pipeline="pipeline_moss_image",
      trust_remote_code=True)`, calls `pipe.encode_prompt(prompt, L)`, pads the
      resulting `[1, S, Cc]` (S <= L) with zero rows to `[L, Cc]`, and saves raw
      float32 LE plus a `<out>.json` sidecar (`L`, `Cc`, `model`, `prompt`,
      `seq_len`). Requires torch + diffusers and the model weights
      (~27 GB for MOSS-Image-SFT -- never fetched by this tooling step;
      pass a small local config/model for smoke tests).
  (b) offline null (`--null`): writes `L * Cc` zeros + sidecar, no torch /
      diffusers / weights / GPU needed, so the C++ staged smoke test runs
      offline.

Examples:
  # offline null embeds matching the example's default geometry:
  python3 scripts/moss_encode.py --null \\
      --out /tmp/tool_embeds.f32 --L 512 --Cc 2048
  # real encode once weights are staged locally (needs GPU for 8.4B encoder):
  python3 scripts/moss_encode.py --prompt "a red panda reading a newspaper" \\
      --out prompt_embeds.f32 --L 512 --Cc 2048 --model /path/to/MOSS-Image-SFT
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Image-SFT"
DEFAULT_PROMPT = "a red panda reading a newspaper"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Encode a prompt to a raw float32 LE [L, Cc] embed file.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="caption text to encode (default: %(default)r)")
    ap.add_argument("--out", default="prompt_embeds.f32",
                    help="output raw embed path (default: %(default)s)")
    ap.add_argument("--L", type=int, default=512,
                    help="caption token capacity / rows (default: %(default)s; "
                         "must equal the C++ model_max_length)")
    ap.add_argument("--Cc", type=int, default=2048,
                    help="caption embedding dim / cols (default: %(default)s; "
                         "must equal the C++ caption_channels; "
                         "MOSS-Image-SFT text_dim is 4096)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF repo or local dir for the real-encode path "
                         "(default: %(default)s)")
    ap.add_argument("--null", action="store_true",
                    help="offline mode: write zeros instead of running "
                         "the Qwen text encoder")
    return ap.parse_args(argv)


def write_sidecar(out_path, *, mode, model, prompt, L, Cc, seq_len):
    sidecar = {
        "mode": mode,
        "model": model,
        "prompt": prompt,
        "L": L,
        "Cc": Cc,
        "seq_len": seq_len,
        "dtype": "float32",
        "endianness": "little",
        "layout": "[L, Cc] row-major",
        "num_floats": L * Cc,
        "num_bytes": L * Cc * 4,
    }
    sidecar_path = str(out_path) + ".json"
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar_path


def run_null(args):
    if args.L <= 0 or args.Cc <= 0:
        raise ValueError("--L and --Cc must be positive")
    out = Path(args.out)
    n = args.L * args.Cc
    chunk = struct.pack("<1024f", *([0.0] * 1024))
    with open(out, "wb") as f:
        remaining = n
        while remaining >= 1024:
            f.write(chunk)
            remaining -= 1024
        if remaining:
            f.write(struct.pack("<%df" % remaining, *([0.0] * remaining)))
    sidecar = write_sidecar(out, mode="null", model=None,
                            prompt=args.prompt, L=args.L, Cc=args.Cc,
                            seq_len=0)
    print(f"wrote {n} zero floats ({n * 4} bytes) to {out}")
    print(f"wrote sidecar to {sidecar}")


def run_real(args):
    if args.L <= 0 or args.Cc <= 0:
        raise ValueError("--L and --Cc must be positive")
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "error: torch is not installed; use --null for the offline path "
            "or install torch/diffusers first.")
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        raise SystemExit(
            "error: diffusers is not installed; use --null for the offline "
            "path or install diffusers first.")

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    print(f"[encode] model={args.model} device={device} "
          f"max_len={args.L} expect_Cc={args.Cc}")

    with torch.no_grad():
        # encode_prompt(prompt, max_sequence_length) -> ([1, S, Cc], [1, S]).
        prompt_embeds, _mask = pipe.encode_prompt(args.prompt, args.L)

    if prompt_embeds.dim() != 3 or prompt_embeds.shape[0] != 1:
        raise RuntimeError(
            f"unexpected encode_prompt output shape {tuple(prompt_embeds.shape)}"
            " (expected [1, S, Cc])")
    seq_len = prompt_embeds.shape[1]
    got_cc = prompt_embeds.shape[2]
    if got_cc != args.Cc:
        raise RuntimeError(
            f"text encoder dim {got_cc} != --Cc {args.Cc} "
            f"(MOSS-Image-SFT text_dim is 4096; pass --Cc {got_cc})")
    if seq_len > args.L:
        raise RuntimeError(
            f"encoded length {seq_len} exceeds --L {args.L}; "
            "increase --L (and the C++ model_max_length)")

    # float32 CPU, pad zero rows to [L, Cc], save raw LE.
    import numpy as np

    arr = prompt_embeds[0].detach().to("cpu", torch.float32).numpy()
    padded = np.zeros((args.L, args.Cc), dtype=np.float32)
    padded[:seq_len] = arr
    out = Path(args.out)
    with open(out, "wb") as f:
        f.write(padded.astype("<f4", copy=False).tobytes(order="C"))
    sidecar = write_sidecar(out, mode="qwen", model=args.model,
                            prompt=args.prompt, L=args.L, Cc=args.Cc,
                            seq_len=seq_len)
    print(f"wrote [{args.L}, {args.Cc}] embeds "
          f"(seq_len={seq_len}) to {out}")
    print(f"wrote sidecar to {sidecar}")


def main(argv=None):
    args = parse_args(argv)
    if args.null:
        run_null(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main(sys.argv[1:])
