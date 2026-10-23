#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
"""Reference Gemma-2-2B-IT caption features for the Sana example golden test.

Loads google/gemma-2-2b-it (or a local snapshot) with HF transformers on CPU,
encodes --prompt exactly like NVlabs/Sana app/sana_pipeline.py (tokenize
with padding/truncation to --max-length, Gemma2Model last-hidden states,
select_index = [0] + [-(max_length-1)..-1] gather), and dumps aggregate
stats + leading values to JSON for Test_SanaTextEncoder.GoldenVsTransformers.

The torch reference runs in BF16 (checkpoint dtype); the C++ encoder is
FP32, so the golden test compares mean/std with a tolerant default (5e-2).

Needs: torch, transformers. Never run from ctest (tests never download).

Example:
  python3 examples/sana/generate_gemma_golden.py \
      --model google/gemma-2-2b-it \
      --prompt "a cyberpunk cat" \
      --out gemma_golden.json
"""
import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it",
                    help="HF model id or local snapshot dir")
    ap.add_argument("--prompt", default="a cyberpunk cat")
    ap.add_argument("--max-length", type=int, default=300)
    ap.add_argument("--chi-prompt", default="",
                    help="optional complex-human-instruction prefix")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--out", default="gemma_golden.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()

    max_length_all = args.max_length
    prompts_all = [args.prompt]
    if args.chi_prompt:
        chi = args.chi_prompt
        prompts_all = [chi + args.prompt]
        num_chi = len(tok.encode(chi))
        max_length_all = num_chi + args.max_length - 2  # Sana magic number

    cap = tok(prompts_all, max_length=max_length_all, padding="max_length",
              truncation=True, return_tensors="pt")
    select_index = [0] + list(range(-args.max_length + 1, 0))
    with torch.no_grad():
        hidden = model(cap.input_ids,
                       attention_mask=cap.attention_mask)[0].float()
    feats = hidden[:, select_index, :]  # [1, max_length, 2304]
    flat = feats.reshape(-1)

    golden = {
        "model": args.model,
        "prompt": args.prompt,
        "chi_prompt": args.chi_prompt,
        "max_length": args.max_length,
        "shape": list(feats.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "first8": [float(v) for v in flat[:8]],
    }
    with open(args.out, "w") as f:
        json.dump(golden, f, indent=2)
    print(json.dumps({k: v for k, v in golden.items() if k != "first8"},
                     indent=2))
    print("first8:", golden["first8"])
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
