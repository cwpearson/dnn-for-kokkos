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
  (f) parity artifacts for the C++ driver (--dump-artifacts, default ON)

All checksums are printed to stdout.

Driver dump contract (``examples/qwen_image_21/qwen_image_21_generate.cpp
--dump-steps DIR``), every file little-endian raw float32:

  DIR/prompt_embeds.f32        [L, Cc] row-major, exact DiT conditioning
  DIR/initial_latents.f32      [C, H, W] channel-major: lat[(c*H+h)*W+w]
  DIR/step{ii}_noise_pred.f32  [C, H, W] image tokens only (cfg=1 cond branch)
  DIR/step{ii}_latents.f32     [C, H, W] post-Euler-step latents
  DIR/manifest.json            prompt/size/steps/sigmas/seed/cfg/sums/versions

Torch packs latents as [1, H*W, C] token rows (row-major over (h, w)), so a
token tensor [N, C] becomes the driver's [C, H, W] via
reshape(H, W, C).permute(2, 0, 1). With use_kv_cache=True the step-0 forward
output is [1, L+N, C] over the joint sequence [text | target] (text first,
per models/qwen_image_21/KokkosDNN_QwenImage21Dit.hpp), so only the last N
rows are the image prediction the pipeline consumes; the golden therefore
dumps that slice as step00_noise_pred.f32 and the whole [L+N, C] tensor as
step00_noise_pred_full.f32 for reference.

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
import json
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
# The VAE compresses 16x, so the DiT sees one token per 16x16 pixel tile.
LATENT_SCALE = 16


def f64sum(t: "torch.Tensor") -> float:
    """float64 checksum of a tensor (computed on CPU for determinism)."""
    return float(t.detach().to(device="cpu", dtype=torch.float64).sum().item())


def to_f32(x: "torch.Tensor") -> "np.ndarray":
    """Detach to contiguous CPU float32 (bf16 -> f32 is exact)."""
    return x.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def tokens_to_chw(tok: "torch.Tensor", h: int, w: int) -> "np.ndarray":
    """[N, C] token rows (row-major over h,w) -> [C, h, w] driver layout."""
    n, c = int(tok.shape[0]), int(tok.shape[1])
    if n != h * w:
        die(f"token count {n} != latent grid {h}x{w} = {h * w}")
    return np.ascontiguousarray(to_f32(tok).reshape(h, w, c).transpose(2, 0, 1))


def write_f32(path: str, arr: "np.ndarray") -> None:
    """Write little-endian raw float32 and log the byte count."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    a.tofile(path)
    print(f"[f] wrote {path} ({a.nbytes} bytes, raw float32, shape {a.shape})")


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
    ap.add_argument(
        "--dump-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write the C++ driver parity dump (prompt_embeds.f32, "
        "initial_latents.f32, step{ii}_noise_pred.f32, step{ii}_latents.f32, "
        "step00_noise_pred_full.f32, manifest.json) next to the PNG "
        "(default: on; --no-dump-artifacts keeps only the final latents/PNG)",
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
    pipeline_arg = "prompt_embeds" if "prompt_embeds" in call_kwargs else "prompt"
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

    # ---- (f) parity artifacts for the C++ driver ---------------------------
    # Layout contract: see the module docstring. All files are little-endian
    # raw float32 in the driver's channel-major [C, H, W] latent layout.
    if args.dump_artifacts:
        steps = len(latents_traj)
        channels = int(final_lat.shape[-1])
        lat_h = HEIGHT // LATENT_SCALE
        lat_w = WIDTH // LATENT_SCALE
        ntok = lat_h * lat_w
        if int(final_lat.shape[1]) != ntok:
            die(
                f"latent token count {int(final_lat.shape[1])} != "
                f"{lat_h}x{lat_w} = {ntok}"
            )

        # Conditioning actually passed to the DiT: [1, L, Cc] -> [L, Cc].
        pe_f32 = to_f32(pe[0])
        prompt_len = int(pe_f32.shape[0])
        write_f32(os.path.join(outdir, "prompt_embeds.f32"), pe_f32)

        # Initial latents were captured packed as torch rows [1, N, C].
        if not initial_latents:
            die("cannot dump initial_latents.f32: prepare_latents was not captured")
        x0_chw = tokens_to_chw(initial_latents[0][0], lat_h, lat_w)
        write_f32(os.path.join(outdir, "initial_latents.f32"), x0_chw)

        per_step: list[dict] = [
            {"step": i} for i in range(max(len(noise_preds), len(latents_traj)))
        ]

        # Per-step velocity prediction. Step 0 (KV-cache prefill) covers the
        # joint [text | target] sequence; the pipeline itself keeps only the
        # last N rows (noise_pred[:, -latents.size(1):]), the image tokens.
        for i, npred in enumerate(noise_preds):
            full = npred[0]  # [rows, C]
            rows = int(full.shape[0])
            if rows == ntok + prompt_len:
                img = full[prompt_len:]
            elif rows == ntok:
                img = full
            else:
                die(
                    f"step {i}: noise_pred has {rows} rows, expected {ntok} "
                    f"(image only) or {ntok + prompt_len} (text|target)"
                )
            img_chw = tokens_to_chw(img, lat_h, lat_w)
            write_f32(os.path.join(outdir, f"step{i:02d}_noise_pred.f32"), img_chw)
            per_step[i]["noise_pred_shape"] = [
                channels,
                lat_h,
                lat_w,
            ]
            per_step[i]["noise_pred_f64sum"] = float(
                np.sum(img_chw, dtype=np.float64)
            )
            if i == 0:
                full_f32 = to_f32(full)
                write_f32(
                    os.path.join(outdir, "step00_noise_pred_full.f32"), full_f32
                )
                per_step[i]["noise_pred_full_shape"] = list(full_f32.shape)
                per_step[i]["noise_pred_full_f64sum"] = float(
                    np.sum(full_f32, dtype=np.float64)
                )

        # Post-Euler-step latents (latents_traj[i] is the state after step i).
        for i, lat in enumerate(latents_traj):
            lat_chw = tokens_to_chw(lat[0], lat_h, lat_w)
            write_f32(os.path.join(outdir, f"step{i:02d}_latents.f32"), lat_chw)
            per_step[i]["latents_shape"] = [channels, lat_h, lat_w]
            per_step[i]["latents_f64sum"] = float(
                np.sum(lat_chw, dtype=np.float64)
            )

        # Schedule actually used. The pipeline feeds the checkpoint's
        # sample_sigmas (8 entries) through the scheduler, which appends the
        # terminal 0 -> 9 values for 8 steps; that is exactly the driver's
        # `--sigmas turbo` array (sample_sigmas + 0.0).
        sigmas = None
        sigma_source = "unavailable"
        sched = getattr(pipe, "scheduler", None)
        sched_sigmas = (
            getattr(sched, "sigmas", None) if sched is not None else None
        )
        if sched_sigmas is not None:
            if hasattr(sched_sigmas, "detach"):  # torch tensor
                sched_sigmas = sched_sigmas.detach().reshape(-1).tolist()
            sigmas = [float(s) for s in sched_sigmas]
            sigma_source = "pipe.scheduler.sigmas after the run"
        else:
            pcfg = getattr(pipe, "config", None)
            cfg_sigmas = None
            if pcfg is not None:
                if hasattr(pcfg, "get"):
                    cfg_sigmas = pcfg.get("sample_sigmas")
                else:
                    cfg_sigmas = getattr(pcfg, "sample_sigmas", None)
            if cfg_sigmas:
                sigmas = [float(s) for s in cfg_sigmas] + [0.0]
                sigma_source = "checkpoint sample_sigmas + terminal 0"
        if sigmas is not None and len(sigmas) != steps + 1:
            print(
                f"[f] WARNING: {len(sigmas)} sigmas for {steps} steps "
                f"(expected {steps + 1})",
                file=sys.stderr,
            )

        try:
            import transformers

            transformers_version = transformers.__version__
        except Exception:  # pragma: no cover - informational only
            transformers_version = "unavailable"

        manifest = {
            "tool": "scripts/qwen_image_21_golden.py",
            "reference": "diffusers.QwenImage21Pipeline",
            "prompt": PROMPT,
            "prompt_source": {
                "method": "pipeline.encode_prompt() with the pipeline's raw "
                "T2I template (no chat template)",
                "pipeline_arg": pipeline_arg,
                "prompt_embeds_dtype": str(pe.dtype).replace("torch.", ""),
            },
            "width": WIDTH,
            "height": HEIGHT,
            "latent_shape": [channels, lat_h, lat_w],
            "latent_tokens": ntok,
            "latent_channels": channels,
            "latent_layout": "[C,H,W] channel-major: lat[(c*H+h)*W+w]",
            "prompt_embeds_shape": [prompt_len, int(pe_f32.shape[1])],
            "prompt_embeds_dtype": str(pe.dtype).replace("torch.", ""),
            "prompt_embeds_f64sum": float(np.sum(pe_f32, dtype=np.float64)),
            "initial_latents_f64sum": float(np.sum(x0_chw, dtype=np.float64)),
            "final_latents_f64sum": float(np.sum(lat_f32, dtype=np.float64)),
            "steps": steps,
            "num_transformer_calls": len(noise_preds),
            "sigmas": sigmas,
            "sigma_source": sigma_source,
            "seed": args.seed,
            "generator": f"torch.Generator('cpu').manual_seed({args.seed})",
            "cfg": 1.0,
            "cfg_note": "pipeline true_cfg_scale default = 1.0 (guidance off, "
            "single cond branch)",
            "use_kv_cache": True,
            "placement": args.placement,
            "device": args.device,
            "artifact_dtype": "float32 little-endian raw",
            "sums_note": "all f64 sums are over the float32 artifacts written "
            "here (not over the model's bf16 tensors)",
            "versions": {
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "transformers": transformers_version,
                "numpy": np.__version__,
            },
            "per_step": per_step,
        }
        manifest_path = os.path.join(outdir, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print(f"[f] wrote {manifest_path} ({steps} steps, "
              f"{len(noise_preds)} transformer calls, sigmas={sigmas})")
        print(f"[f] parity artifacts in {outdir}")

    print("DONE")


if __name__ == "__main__":
    main()
