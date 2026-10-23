#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
"""Generate a golden token file for KokkosDNN's DINOv3 real-weight validation.

Compares against Hugging Face `transformers` on the ViT-B/16 encoder
`facebook/dinov3-vitb16-pretrain-lvd1689m`. The C++ test
`Dinov3Golden.RealWeightsFp32` (examples/dinov3/Test_Dinov3Golden.cpp)
loads this file plus the same safetensors checkpoint and checks the CLS
vector, the first 4 patch rows and the patch mean with a max-abs-diff
tolerance.

No Kokkos dependency — only torch + transformers (+ Pillow for --image):

    pip install torch transformers safetensors Pillow

Usage (read-only on the weights dir; golden JSON goes to --out):

    # deterministic synthetic input (default; no image file needed):
    python examples/dinov3/generate_golden.py \
        --weights-dir /workspace/dinov3-vitb16-pretrain-lvd1689m \
        --synthetic --out /tmp/golden_dinov3.json

    # a real image instead (preprocessed with the HF image processor):
    python examples/dinov3/generate_golden.py \
        --weights-dir /path/to/dinov3-vitb16-pretrain-lvd1689m \
        --image /path/to/img.jpg --out /tmp/golden_dinov3_img.json

Run the C++ check with:

    KOKKOSDNN_DINOV3_WEIGHTS=/path/to/model.safetensors \\
    KOKKOSDNN_DINOV3_GOLDEN_JSON=/tmp/golden_dinov3.json \\
        ./build/examples/dinov3/test_dinov3_golden
    # optional: KOKKOSDNN_DINOV3_TOL=2e-3 to override the default 1e-3.

Notes / contract with the C++ test:
  * Synthetic pixels use the canonical formula shared with
    dinov3_encode.cpp / Test_Dinov3Smoke.cpp / Test_Dinov3Golden.cpp
    (CHW float32, C=3, SxS with S=--image-size):
        v(c,y,x) = ((y*S + x) % 256) / 255 + 0.01 * c
    They are fed DIRECTLY as pixel_values (no HF processor rescale /
    normalize) on both sides, so the inputs are bit-identical by
    construction; the script additionally stores the pixel tensor so the
    C++ side can cross-check its recomputation.
  * --image inputs are preprocessed with the checkpoint's HF image
    processor; the resulting pixel_values tensor is stored verbatim in the
    golden file, so the C++ side compares on exactly the same floats.
  * Reference forward runs in float32 (the checkpoint torch_dtype); golden
    vectors are stored as float32 JSON numbers. Tolerance is documented in
    Test_Dinov3Golden.cpp (default max-abs-diff <= 1e-3, override with
    KOKKOSDNN_DINOV3_TOL).
  * Golden content is the full CLS vector [768], the first 4 patch-token
    rows and the patch-token mean — full patch tokens (196x768) are
    omitted to keep the file small (< ~5MB).
"""

import argparse
import json
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights-dir",
                   default="/workspace/dinov3-vitb16-pretrain-lvd1689m",
                   help="local HF snapshot dir (config.json + model.safetensors)")
    p.add_argument("--out", required=True, help="output golden JSON path")
    p.add_argument("--image", default=None,
                   help="input image path (default: deterministic synthetic)")
    p.add_argument("--synthetic", action="store_true", default=True,
                   help="use the deterministic synthetic gradient image "
                        "(default; ignored when --image is given)")
    p.add_argument("--image-size", type=int, default=224,
                   help="square image size (must match the checkpoint config)")
    p.add_argument("--device", default="cpu",
                   help="torch device for the reference forward (cpu/cuda)")
    return p.parse_args()


def synthetic_pixels(S, C=3):
    """Canonical synthetic CHW image (see module docstring)."""
    import torch
    ys = torch.arange(S, dtype=torch.float32).view(1, S, 1).expand(C, S, S)
    xs = torch.arange(S, dtype=torch.float32).view(1, 1, S).expand(C, S, S)
    ch = torch.arange(C, dtype=torch.float32).view(C, 1, 1).expand(C, S, S)
    return ((ys * S + xs) % 256) / 255.0 + 0.01 * ch


def main():
    args = parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModel

    print(f"[golden] loading {args.weights_dir} ...", flush=True)
    model = AutoModel.from_pretrained(args.weights_dir)
    model.to(args.device)
    model.eval()
    cfg = model.config
    R = int(getattr(cfg, "num_register_tokens", 4))
    D = int(cfg.hidden_size)
    S = int(args.image_size)
    assert S == int(cfg.image_size), (
        f"--image-size {S} != checkpoint image_size {cfg.image_size}")

    if args.image:
        proc = AutoImageProcessor.from_pretrained(args.weights_dir)
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        pv = proc(images=img, return_tensors="pt")["pixel_values"].to(args.device)
        input_tag = args.image
        print(f"[golden] image input {args.image} -> pixel_values {list(pv.shape)}",
              flush=True)
    else:
        pv = synthetic_pixels(S).unsqueeze(0).to(args.device)
        input_tag = "synthetic"
        print(f"[golden] synthetic input CHW [3,{S},{S}] "
              f"(v=((y*S+x)%256)/255+0.01*c)", flush=True)

    with torch.inference_mode():
        out = model(pixel_values=pv.float())
        # last_hidden_state: [1, 1 + R + Np, D], row 0 = CLS,
        # rows [1, 1+R) = registers, rest = row-major patch tokens.
        toks = out.last_hidden_state[0].float()
        T = toks.shape[0]
        Np = T - 1 - R
        assert Np == (S // int(cfg.patch_size)) ** 2, (
            f"token count {T} inconsistent with patch grid (R={R})")
        cls = toks[0]
        first4 = toks[1 + R:1 + R + 4]
        pmean = toks[1 + R:].mean(dim=0)
        cls_n = cls.norm().item()
        print(f"[golden] tokens [T={T}, D={D}] (1 CLS + {R} regs + {Np} "
              f"patches), CLS L2={cls_n:.6f}", flush=True)

    golden = {
        "model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dtype": "float32",
        "image_size": S,
        "hidden_size": D,
        "num_register_tokens": R,
        "num_patch_tokens": Np,
        "input": input_tag,
        "synthetic_formula": "v(c,y,x) = ((y*S + x) % 256) / 255 + 0.01 * c",
        "pixels": {
            "channels": pv.shape[1], "height": pv.shape[2],
            "width": pv.shape[3],
            "values": pv[0].flatten().tolist(),
        },
        "cls": cls.tolist(),
        "patch_first4": first4.tolist(),
        "patch_mean": pmean.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(golden, f)
        f.write("\n")
    print(f"[golden] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
