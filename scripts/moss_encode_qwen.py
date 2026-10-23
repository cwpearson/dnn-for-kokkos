#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
"""Qwen3.5 prompt-frontend reference for the MOSS-Image staged split.

Reference implementation of the host-side frontend (tokenizer/BPE +
chat template via transformers AutoTokenizer) whose exact outputs the
in-process C++ tokenizer (`models/moss_image/KokkosDNN_QwenTokenizer.hpp`, used by
`moss_image_generate --tokenizer/--text-encoder`) reproduces id-for-id
(see tests/Test_QwenTokenizer.cpp goldens). It also produces the token
ids consumed by the C++ Qwen3.5 encoder
(`KokkosDNN::Qwen35::EncoderModel::forward`, ids [B, S] + valid lengths),
plus an optional raw .f32 embed handoff compatible with
`Moss::load_caption_raw` ([L, Cc] + sidecar json) for parity checks and
for the `--prompt-embeds-in` cross-machine/offline path.

Pipeline fidelity (verified against pipeline_moss_image.py encode_prompt):
  - per-prompt: apply_chat_template([{role:user, content}], tokenize=False,
    add_generation_prompt=True, enable_thinking=True)
  - tokenizer(fmt, return_tensors='pt', truncation=True,
    max_length=max_sequence_length), range 1..4096
  - C++ side runs one unpadded forward per prompt; padding rows of the
    [L, Cc] handoff are zeros with a bool mask (mask[:seq_len] = True).

Outputs (all paths write sidecars):
  --ids-out PATH : raw little-endian int32 ids [seq_len] + <PATH>.json
    sidecar (mode/ids, model, prompt, L, seq_len, max ids, chat flags).
  --out PATH     : raw float32 LE [L, Cc] embeds + <PATH>.json sidecar
    matching scripts/moss_encode.py (mode, model, prompt, L, Cc, seq_len,
    dtype float32, layout "[L, Cc] row-major").

Modes:
  (a) --null: zeros for --out, deterministic pseudo-ids for --ids-out
      (no transformers/weights needed; C++ staged smoke test runs offline).
  (b) real (default): tokenizes with AutoTokenizer (needs transformers +
      the tokenizer files only — no model weights); --out needs --model
      weights unless --ids-only is given.

Examples:
  python3 scripts/moss_encode_qwen.py --null \\
      --ids-out /tmp/p.ids.i32 --out /tmp/p.f32 --L 512 --Cc 4096
  python3 scripts/moss_encode_qwen.py --prompt "a red panda" \\
      --tokenizer OpenMOSS-Team/MOSS-Image-SFT --subfolder tokenizer \\
      --ids-out prompt.ids.i32 --ids-only
  python3 scripts/moss_encode_qwen.py --prompt "a red panda" \\
      --tokenizer /path/to/MOSS-Image-SFT --model /path/to/MOSS-Image-SFT \\
      --ids-out prompt.ids.i32 --out prompt.f32 --L 512 --Cc 4096
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
        description="Qwen3.5 chat-template tokenizer + .f32 handoff.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tokenizer", default=DEFAULT_MODEL,
                    help="HF repo or local dir holding tokenizer files")
    ap.add_argument("--tokenizer-subfolder", default="tokenizer",
                    help="subfolder under --tokenizer (default: %(default)s; "
                         "empty string for a bare tokenizer dir)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF repo or local dir for the reference-encode path")
    ap.add_argument("--model-subfolder", default="text_encoder",
                    help="subfolder holding the text encoder (default: "
                         "%(default)s; empty string for a bare model dir)")
    ap.add_argument("--L", type=int, default=512,
                    help="caption token capacity / rows (1..4096)")
    ap.add_argument("--Cc", type=int, default=4096,
                    help="caption embedding dim (MOSS-Image-SFT text_dim)")
    ap.add_argument("--ids-out", default="prompt.ids.i32",
                    help="output raw int32 LE ids path")
    ap.add_argument("--out", default="prompt_embeds.f32",
                    help="output raw float32 LE [L, Cc] path")
    ap.add_argument("--ids-only", action="store_true",
                    help="only write --ids-out (no model weights needed)")
    ap.add_argument("--null", action="store_true",
                    help="offline mode: zeros + pseudo-ids, no transformers")
    ap.add_argument("--no-thinking", action="store_true",
                    help="pass enable_thinking=False to the chat template "
                         "(default is True, matching the pipeline)")
    return ap.parse_args(argv)


def write_ids(path, ids, *, model, prompt, L, thinking):
    out = Path(path)
    with open(out, "wb") as f:
        f.write(struct.pack("<%di" % len(ids), *ids))
    sidecar = {
        "mode": "ids",
        "model": model,
        "prompt": prompt,
        "L": L,
        "seq_len": len(ids),
        "max_id": max(ids) if ids else 0,
        "dtype": "int32",
        "endianness": "little",
        "layout": "[seq_len] token ids for Qwen35::EncoderModel::forward",
        "chat_template": {"add_generation_prompt": True,
                          "enable_thinking": thinking},
        "truncation": {"enabled": True, "max_length": L},
    }
    sp = str(out) + ".json"
    Path(sp).write_text(json.dumps(sidecar, indent=2) + "\n")
    return sp


def write_embeds(path, arr2d, *, mode, model, prompt, L, Cc, seq_len):
    import numpy as np

    want = np.zeros((L, Cc), dtype=np.float32)
    want[:seq_len] = np.asarray(arr2d, dtype=np.float32)[:seq_len]
    out = Path(path)
    with open(out, "wb") as f:
        f.write(want.astype("<f4", copy=False).tobytes(order="C"))
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
    sp = str(out) + ".json"
    Path(sp).write_text(json.dumps(sidecar, indent=2) + "\n")
    return sp


def run_null(args):
    if not 1 <= args.L <= 4096 or args.Cc <= 0:
        raise ValueError("--L must be in 1..4096 and --Cc positive")
    pseudo = [(3 * i + 7) % 248320 for i in range(min(args.L, 16))]
    sp = write_ids(args.ids_out, pseudo, model=None, prompt=args.prompt,
                   L=args.L, thinking=not args.no_thinking)
    print(f"wrote {len(pseudo)} pseudo-ids to {args.ids_out} + {sp}")
    if not args.ids_only:
        n = args.L * args.Cc
        chunk = struct.pack("<1024f", *([0.0] * 1024))
        with open(args.out, "wb") as f:
            remaining = n
            while remaining >= 1024:
                f.write(chunk)
                remaining -= 1024
            if remaining:
                f.write(struct.pack("<%df" % remaining, *([0.0] * remaining)))
        sidecar = {
            "mode": "null",
            "model": None,
            "prompt": args.prompt,
            "L": args.L,
            "Cc": args.Cc,
            "seq_len": 0,
            "dtype": "float32",
            "endianness": "little",
            "layout": "[L, Cc] row-major",
            "num_floats": n,
            "num_bytes": n * 4,
        }
        sp2 = str(args.out) + ".json"
        Path(sp2).write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"wrote {n} zero floats to {args.out} + {sp2}")


def load_tokenizer(args):
    from transformers import AutoTokenizer

    kw = {} if not args.tokenizer_subfolder else {
        "subfolder": args.tokenizer_subfolder}
    try:
        return AutoTokenizer.from_pretrained(args.tokenizer, **kw)
    except Exception:
        # Fall back to the repo root (some mirrors stage the tokenizer
        # files without the subfolder).
        return AutoTokenizer.from_pretrained(args.tokenizer)


def tokenize(tok, prompt, L, thinking):
    msgs = [{"role": "user", "content": prompt}]
    fmt = tok.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=True,
                                  enable_thinking=thinking)
    # Plain python lists: the ids path must not require torch.
    enc = tok(fmt, truncation=True, max_length=L)
    return fmt, list(enc["input_ids"])


def run_real(args):
    if not 1 <= args.L <= 4096 or args.Cc <= 0:
        raise ValueError("--L must be in 1..4096 and --Cc positive")
    thinking = not args.no_thinking
    tok = load_tokenizer(args)
    fmt, ids = tokenize(tok, args.prompt, args.L, thinking)
    print(f"[tokenize] seq_len={len(ids)} max_id={max(ids) if ids else 0}")
    print(f"[tokenize] formatted head: {fmt[:120]!r}")
    sp = write_ids(args.ids_out, ids, model=args.tokenizer,
                   prompt=args.prompt, L=args.L, thinking=thinking)
    print(f"wrote {len(ids)} ids to {args.ids_out} + {sp}")
    if args.ids_only:
        return
    import torch
    from transformers import AutoModel

    kw = {} if not args.model_subfolder else {
        "subfolder": args.model_subfolder}
    try:
        te = AutoModel.from_pretrained(args.model, dtype=torch.float32,
                                       **kw).eval()
    except Exception:
        te = AutoModel.from_pretrained(args.model,
                                       torch_dtype=torch.float32,
                                       **kw).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te.to(dev)
    ii = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        hs = te(input_ids=ii, use_cache=False,
                return_dict=True).last_hidden_state[0]
    got_cc = hs.shape[-1]
    if got_cc != args.Cc:
        raise RuntimeError(f"text dim {got_cc} != --Cc {args.Cc}")
    arr = hs.detach().to("cpu", torch.float32).numpy()
    sp2 = write_embeds(args.out, arr, mode="qwen", model=args.model,
                       prompt=args.prompt, L=args.L, Cc=args.Cc,
                       seq_len=len(ids))
    print(f"wrote [{args.L}, {args.Cc}] embeds "
          f"(seq_len={len(ids)}) to {args.out} + {sp2}")


def main(argv=None):
    args = parse_args(argv)
    if args.null:
        run_null(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main(sys.argv[1:])
