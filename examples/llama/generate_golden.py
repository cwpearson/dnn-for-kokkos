#!/usr/bin/env python3
"""Generate a golden logits file for KokkosDNN's real-weight validation.

Compares against Hugging Face `transformers` on 1-2 prompts. The C++ test
`LlamaGolden.RealWeights*` (examples/llama/Test_LlamaGolden.cpp) loads this
file plus the same checkpoint/tokenizer and checks prefill argmax, top-k
max-abs-diff and NLL/perplexity, plus one T=1 greedy decode step.

No Kokkos dependency — only torch + transformers (install e.g.):
    pip install torch transformers safetensors

Usage:
    # gated repo: log in first (huggingface-cli login), then e.g.
    python examples/llama/generate_golden.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --out /path/to/golden_llama31_8b.json

    # local checkpoint dir + CUDA (BF16 8B needs ~16 GiB CPU RAM otherwise):
    python examples/llama/generate_golden.py \
        --model /path/to/Llama-3.1-8B-Instruct --device cuda --out golden.json

    # custom prompts (default: the two below; the C++ test uses the first 2):
    python examples/llama/generate_golden.py --model ... --out golden.json \
        --prompt "Hello, I am" --prompt "Explain Kokkos in one sentence."

Run the C++ check with:
    KOKKOSDNN_WEIGHTS="model-00001-of-00004.safetensors:..." \\
    KOKKOSDNN_TOKENIZER_JSON=tokenizer.json \\
    KOKKOSDNN_GOLDEN_JSON=golden.json \\
        ./build/examples/llama/test_llama_golden
    # plus KOKKOSDNN_GOLDEN_FP32=1 for the FP32 path (~32 GiB RAM).

Notes / contract with the C++ test:
  * Tokenization uses add_special_tokens=True (BOS 128000 prepended). This
    matches the C++ Tokenizer::encode default (add_bos=true); the test
    asserts encode(prompt) == golden input_ids, so a mismatch fails loudly
    instead of comparing shifted logits.
  * Logits are cast to float32 before top-k/NLL so the golden values are
    not quantized to BF16; tolerances are documented in Test_LlamaGolden.cpp
    (BF16 max-abs-diff <= 0.5, FP32 <= 0.05, argmax exact; override with
    KOKKOSDNN_GOLDEN_TOL).
  * NLL of the greedy next token under the full-vocab softmax is stored;
    the C++ test recomputes it from its own logits and reports corpus
    perplexity = exp(mean NLL) over the prefill + one decode step.
  * The second greedy token (next_id/next_nll/next_top_*) covers the T=1
    GQA decode path at full 8B geometry (the FP32-cuDNN-fallback shape
    class noted in src/KokkosDNN_Attention.hpp).
"""

import argparse
import json
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HF model id or local checkpoint dir")
    p.add_argument("--out", required=True, help="output golden JSON path")
    p.add_argument("--prompt", action="append", default=[],
                   help="prompt text (repeatable; default: two fixed prompts)")
    p.add_argument("--top-k", type=int, default=64,
                   help="number of top logits/ids stored per position")
    p.add_argument("--device", default="cpu",
                   help="torch device for the reference forward (cpu/cuda)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                   help="reference compute dtype (bfloat16 matches the ckpt)")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = args.prompt or ["Hello, I am", "Explain Kokkos in one sentence."]
    prompts = prompts[:2]  # the C++ test compares the first 2 entries

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"[golden] loading {args.model} dtype={args.dtype} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    model.eval()

    entries = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt",
                      add_special_tokens=True)["input_ids"].to(args.device)
            ids_list = ids[0].tolist()
            assert ids_list[0] == 128000, (
                f"BOS (128000) missing from tokenized prompt {ids_list!r}; "
                "the C++ Tokenizer::encode default prepends BOS, so fix "
                "tokenization before comparing logits.")
            logits = model(ids).logits[0, -1, :].float()  # [vocab] fp32
            lse = torch.logsumexp(logits, dim=-1).item()
            top = torch.topk(logits, args.top_k)
            top_ids = top.indices.tolist()
            top_logits = top.values.tolist()
            argmax = int(torch.argmax(logits).item())
            nll = lse - logits[argmax].item()
            print(f"[golden] prompt={prompt!r} ids={ids_list} "
                  f"argmax={argmax} nll={nll:.4f}", flush=True)

            # One greedy decode step (T=1 path in the C++ model).
            ids2 = torch.cat([ids, torch.tensor([[argmax]],
                                                device=args.device)], dim=1)
            logits2 = model(ids2).logits[0, -1, :].float()
            lse2 = torch.logsumexp(logits2, dim=-1).item()
            top2 = torch.topk(logits2, args.top_k)
            next_id = int(torch.argmax(logits2).item())
            next_nll = lse2 - logits2[next_id].item()
            print(f"[golden]   next_id={next_id} next_nll={next_nll:.4f}",
                  flush=True)

            entries.append({
                "prompt": prompt,
                "input_ids": ids_list,
                "top_ids": top_ids,
                "top_logits": top_logits,
                "argmax": argmax,
                "nll": nll,
                "next_id": next_id,
                "next_top_ids": top2.indices.tolist(),
                "next_top_logits": top2.values.tolist(),
                "next_nll": next_nll,
            })

    golden = {"model": args.model, "dtype": args.dtype,
              "add_bos": True, "top_k": args.top_k, "entries": entries}
    with open(args.out, "w") as f:
        json.dump(golden, f, indent=1)
    print(f"[golden] wrote {args.out} ({len(entries)} entries)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
