#!/usr/bin/env python3
"""Torch golden reference for Qwen/Qwen-Image-2.1-Turbo (text-to-image only).

Loads the diffusers ``QwenImage21Pipeline`` from local weights, runs a fixed
1024x1024 text-to-image prompt (seed 42, checkpoint's built-in 8-step Turbo
schedule, ``use_kv_cache=True``), and dumps:

  (a) prompt_embeds shape + first/last row checksums (float64 sum)
  (b) initial latents checksum
  (c) per-step noise_pred checksums (one transformer forward call per step)
  (d) final latents checksum + saved raw float32 file
  (e) decoded PNG

All checksums are printed to stdout.

Requirements (CUDA build of torch recommended; CPU-only works but is slow):
    pip install git+https://github.com/huggingface/diffusers.git
    pip install "transformers>=5.17.0" accelerate pillow torch safetensors

Usage:
    python3 scripts/qwen_image_21_golden.py \
        --weights /workspace/weights/Qwen-Image-2.1-Turbo \
        --out goldens/qwen_image_21_turbo
"""

from __future__ import annotations

import argparse
import sys


def die(msg: str) -> "sys.NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


try:
    import torch
except ImportError:
    die(
        "torch is required but not installed. "
        "Install with e.g. `pip install torch --index-url "
        "https://download.pytorch.org/whl/cu128` (pick a CUDA build "
        "matching your driver) or `pip install torch` for CPU-only."
    )

try:
    from diffusers import QwenImage21Pipeline
except ImportError:
    die(
        "diffusers with QwenImage21Pipeline is required but not available. "
        "Install the latest source with "
        "`pip install git+https://github.com/huggingface/diffusers.git` "
        "plus `pip install \"transformers>=5.17.0\" accelerate pillow`."
    )

try:
    import numpy as np
except ImportError:
    die("numpy is required but not installed (`pip install numpy`).")

PROMPT = "A capybara wearing a wizard hat, oil painting"
WIDTH, HEIGHT = 1024, 1024
SEED = 42


def f64sum(t: "torch.Tensor") -> float:
    """float64 checksum of a tensor (computed on CPU for determinism)."""
    return float(t.detach().to(device="cpu", dtype=torch.float64).sum().item())


def as_tensor(x):
    """Unwrap a transformer forward output to its predicted-noise tensor."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "sample"):  # e.g. Transformer2DModelOutput
        return x.sample
    if isinstance(x, (tuple, list)):
        return as_tensor(x[0])
    if isinstance(x, dict) and "sample" in x:
        return as_tensor(x["sample"])
    raise TypeError(f"cannot unwrap transformer output of type {type(x)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        default="/workspace/weights/Qwen-Image-2.1-Turbo",
        help="local dir with the Qwen-Image-2.1-Turbo checkpoint",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output dir (default: <repo>/goldens/qwen_image_21_turbo)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--placement",
        choices=["manual", "model-offload", "full"],
        default="manual",
        help="manual: text encoder stays on CPU, transformer+VAE on CUDA "
        "(default; fits 24 GB VRAM). model-offload: diffusers CPU offload "
        "hooks. full: everything on --device (needs a very large GPU).",
    )
    args = ap.parse_args()

    import os

    outdir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "goldens",
        "qwen_image_21_turbo",
    )
    os.makedirs(outdir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        die("no CUDA device found; rerun with --device cpu (slow) or install CUDA torch.")

    print(f"weights : {args.weights}")
    print(f"outdir  : {outdir}")
    print(f"device  : {args.device}")
    print(f"prompt  : {PROMPT!r}")
    print(f"size    : {WIDTH}x{HEIGHT}, seed {args.seed}")
    print(f"torch   : {torch.__version__}")
    import diffusers

    print(f"diffusers: {diffusers.__version__}")

    pipe = QwenImage21Pipeline.from_pretrained(args.weights, dtype=torch.bfloat16)
    # NOTE: the full checkpoint (~32 GB of weights: 17.5 GB text encoder +
    # 14.2 GB transformer + VAE) does not fit in 24 GB of VRAM. "manual"
    # placement encodes the prompt on CPU (text encoder never touches the
    # GPU: a GPU encode leaves ~14 GB pinned after .to("cpu")), then moves
    # only transformer+VAE plus the tiny prompt tensors to CUDA for
    # sampling. Whole-model and per-layer CPU-offload hooks both fail on a
    # 24 GB card with this checkpoint (OOM / tied-embedding meta tensors).
    if args.placement == "model-offload":
        pipe.enable_model_cpu_offload()
    elif args.placement == "full":
        pipe.to(args.device)
    else:
        # Manual: everything stays on CPU for the prompt-encoding stage;
        # transformer+VAE move to CUDA afterwards (see below).
        pass
    pipe.set_progress_bar_config(disable=True)

    # ---- (a) prompt embeds -------------------------------------------------
    # Prefer the pipeline's own encoder helper; fall back to a forward hook
    # on the text encoder for forwards-compatibility across diffusers revs.
    embeds_captured: dict = {}
    if hasattr(pipe, "encode_prompt"):
        import inspect

        sig = inspect.signature(pipe.encode_prompt)  # device/dtype-agnostic call
        kwargs: dict = {"prompt": PROMPT}
        if "device" in sig.parameters:
            kwargs["device"] = pipe.device
        result = pipe.encode_prompt(**kwargs)
        # Arity varies across diffusers revs (2- or 3-tuple); we need the
        # first two entries: (prompt_embeds, prompt_embeds_mask).
        if isinstance(result, (tuple, list)):
            prompt_embeds = result[0]
            prompt_embeds_mask = result[1] if len(result) > 1 else None
        else:
            prompt_embeds, prompt_embeds_mask = result, None
    else:  # hook-based fallback
        handles = [
            pipe.text_encoder.register_forward_hook(
                lambda m, i, o: embeds_captured.setdefault(
                    "hidden", as_tensor(o).detach().cpu()
                )
            )
        ]
        try:
            pipe(prompt=PROMPT, width=64, height=64, use_kv_cache=False)
        finally:
            for h in handles:
                h.remove()
        prompt_embeds = embeds_captured.get("hidden")
        prompt_embeds_mask = None
        if prompt_embeds is None:
            die("could not capture prompt embeds (no encode_prompt, hook found nothing)")

    pe = prompt_embeds.detach().cpu()
    print(f"[a] prompt_embeds shape={tuple(pe.shape)} dtype={pe.dtype}")
    # First/last token rows of the (single) batch entry.
    print(f"[a] prompt_embeds[0,0]  f64sum={f64sum(pe[0][0]):.17g}")
    print(f"[a] prompt_embeds[0,-1] f64sum={f64sum(pe[0][-1]):.17g}")
    if prompt_embeds_mask is not None:
        print(f"[a] prompt_embeds_mask shape={tuple(prompt_embeds_mask.shape)}")

    if args.placement == "manual":
        # Sampling needs transformer+VAE on CUDA; the text encoder stays
        # on CPU (it never touches the GPU: a GPU encode leaves ~14 GB
        # pinned even after .to("cpu")). Move the tiny prompt tensors
        # over explicitly since the pipeline will not do it for us.
        dev = torch.device(args.device)
        prompt_embeds = prompt_embeds.to(dev)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(dev)
        pipe.transformer.to(args.device)
        pipe.vae.to(args.device)
        torch.cuda.empty_cache()

    # ---- hooks for latents trajectory + per-step noise_pred ---------------
    latents_traj: list[torch.Tensor] = []  # post-step latents, one entry per step
    noise_preds: list[torch.Tensor] = []  # transformer outputs, one per forward call

    def on_step_end(pipe_obj, step_index, timestep, callback_kwargs):
        lat = callback_kwargs.get("latents", None)
        if lat is not None:
            latents_traj.append(lat.detach().cpu())
        return callback_kwargs

    # Forward hook (not a forward replacement) so accelerate CPU-offload
    # hooks installed by enable_model_cpu_offload() keep working.
    def on_transformer_out(module, inputs, output):
        noise_preds.append(as_tensor(output).detach().cpu())

    pipe.transformer.register_forward_hook(on_transformer_out)

    # Initial latents: wrap prepare_latents when the pipeline exposes it.
    initial_latents: list[torch.Tensor] = []
    if hasattr(pipe, "prepare_latents"):
        orig_prep = pipe.prepare_latents

        def recording_prepare(*p_args, **p_kwargs):
            out = orig_prep(*p_args, **p_kwargs)
            lat0 = out[0] if isinstance(out, tuple) else out
            initial_latents.append(lat0.detach().cpu())
            return out

        pipe.prepare_latents = recording_prepare  # type: ignore[method-assign]

    # ---- full text-to-image run --------------------------------------------
    # NOTE: no num_inference_steps / sigmas override: the Turbo checkpoint
    # ships its recommended 8-step schedule, loaded automatically.
    # The tensor-input allowlist kwarg is only passed when supported.
    generator = torch.Generator("cpu").manual_seed(args.seed)
    import inspect as _inspect

    call_params = _inspect.signature(pipe.__call__).parameters
    # Tensor-input allowlist kwarg was renamed across diffusers revs.
    tensor_inputs_kw = next(
        (n for n in ("callback_on_step_end_inputs", "callback_on_step_end_tensor_inputs")
         if n in call_params),
        None,
    )
    call_kwargs = dict(
        width=WIDTH,
        height=HEIGHT,
        use_kv_cache=True,
        generator=generator,
        callback_on_step_end=on_step_end,
    )
    if "prompt_embeds" in call_params:
        # Bypass the text encoder (it stays on CPU in manual placement).
        call_kwargs["prompt_embeds"] = prompt_embeds
        if prompt_embeds_mask is not None and "prompt_embeds_mask" in call_params:
            call_kwargs["prompt_embeds_mask"] = prompt_embeds_mask
    else:
        call_kwargs["prompt"] = PROMPT
    if tensor_inputs_kw is not None:
        call_kwargs[tensor_inputs_kw] = ["latents"]
    result = pipe(**call_kwargs)
    image = result.images[0]

    # ---- (b) initial latents ----------------------------------------------
    if initial_latents:
        lat0 = initial_latents[0]
        print(f"[b] initial latents shape={tuple(lat0.shape)}")
        print(f"[b] initial latents f64sum={f64sum(lat0):.17g}")
    else:
        print("[b] initial latents: NOT captured (pipeline has no prepare_latents)")

    # ---- (c) per-step noise_pred -------------------------------------------
    print(f"[c] transformer forward calls: {len(noise_preds)} (expect 8 for Turbo)")
    for i, npred in enumerate(noise_preds):
        print(f"[c] step {i}: noise_pred shape={tuple(npred.shape)} "
              f"f64sum={f64sum(npred):.17g}")

    # ---- (d) final latents --------------------------------------------------
    if not latents_traj:
        die("no per-step latents captured via callback_on_step_end")
    final_lat = latents_traj[-1]
    print(f"[d] steps recorded: {len(latents_traj)}")
    print(f"[d] final latents shape={tuple(final_lat.shape)}")
    print(f"[d] final latents f64sum={f64sum(final_lat):.17g}")
    lat_f32 = final_lat.to(torch.float32).numpy()
    f32_path = os.path.join(outdir, "qwen_image_21_turbo_final_latents.f32")
    lat_f32.tofile(f32_path)
    np.save(os.path.join(outdir, "qwen_image_21_turbo_final_latents.npy"), lat_f32)
    print(f"[d] wrote {f32_path} ({lat_f32.nbytes} bytes, "
          f"raw float32, shape {lat_f32.shape})")

    # ---- (e) decoded PNG ----------------------------------------------------
    png_path = os.path.join(outdir, "qwen_image_21_turbo_capybara.png")
    image.save(png_path)
    print(f"[e] wrote {png_path} (size {image.size})")
    print("DONE")


if __name__ == "__main__":
    main()
