#!/usr/bin/env python3
"""Compare Qwen-Image-2.1-Turbo parity dumps (numpy only, no torch import).

Given a golden dir (from ``scripts/qwen_image_21_golden.py``) and an actual
dir (from ``examples/qwen_image_21/qwen_image_21_generate.cpp --dump-steps``),
compare every tensor both sides have:

    initial_latents.f32      [C, H, W]    initial noise (driver layout)
    ref_latents.f32          [C, Hc, Wc]  edit-mode VAE condition latents
    img_mask.f32             [V] 0/1      edit-mode image mask row
    step{ii}_noise_pred.f32  [C, H, W]    image tokens only (cfg=1 cond branch)
    step{ii}_latents.f32     [C, H, W]    post-Euler-step latents
    final latents            optional --final-golden/--final-actual

Files are little-endian raw float32 (a ``.npy`` is accepted too). The dumps'
``manifest.json`` (when present) supplies display shapes; the comparison only
requires matching element counts. Final latents are compared after
flattening, because the two sides may pack them differently: torch writes
[L*W, C] token rows while the driver writes [C, H, W]. Those two packings are
transposes of each other, so pass
``--final-golden-layout tokens`` (or ``--final-actual-layout tokens``) for the
torch-packed side and the tool reorders it into the [C, H, W] comparison
order first. With the default ``flat`` layout the files are compared
element-for-element as stored.

Thresholds are *numerics* thresholds -- how close two fp32/bf16 pipelines are
expected to land -- not frozen implementation details, so retune them as the
backends and precisions evolve. Default: relative L2 <= 2e-2 for every
tensor. ``--rtol`` overrides that relative-L2 threshold; ``--atol``
optionally adds a max|diff| gate. Without ``--strict`` the tool always exits
0 (baseline report mode); with ``--strict`` it exits 1 if any compared tensor
fails (or an image PSNR is below ``--psnr-min``, default 25 dB).

Steps present on only one side (e.g. a driver dump without the step-0
prefill output) are reported as skipped, with a note, and never crash the
tool or change the exit code.

Usage:
    python3 scripts/qwen_image_21_compare.py --golden DIR --actual DIR
    python3 scripts/qwen_image_21_compare.py --golden DIR --actual DIR \\
        --final-golden torch_final.f32 --final-golden-layout tokens \\
        --final-actual driver_final.f32 --strict
    python3 scripts/qwen_image_21_compare.py --images golden.png actual.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import NoReturn

import numpy as np

REL_L2_DEFAULT = 2e-2
PSNR_MIN_DEFAULT = 25.0
F32_DTYPE = np.dtype("<f4")
# Qwen-Image-2.1 fixed geometry, used only to label shapes when a manifest
# does not carry them (the transforms themselves never depend on these).
QWEN21_CONTEXT_DIM = 4096
QWEN21_LATENT_CHANNELS = 64
# Exact step tensor names: step00_noise_pred.f32, step01_latents.f32, ...
STEP_RE = re.compile(r"^step(\d+)_(noise_pred|latents)\.f32$")
CORE_FILES = ("initial_latents.f32",)
# Edit-mode reference tensors: present on both sides only for edit dumps
# (the T2I golden/driver never write them); compared when shared, skipped
# with a note otherwise, never fatal.
EDIT_REF_FILES = ("ref_latents.f32", "img_mask.f32")
PROMPT_EMBEDS = "prompt_embeds.f32"


def parse_row_spec(spec: str, nrows: int) -> list[int]:
    """Parse a --rows spec (e.g. "8:1034" or "0:8,1034:1059", end-exclusive)."""
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"^(\d+):(\d+)$", part) if part else None
        if m is None:
            die(f"--rows: bad range {part!r}, want START:END[,START:END...]")
        start, end = int(m.group(1)), int(m.group(2))
        if not 0 <= start < end <= nrows:
            die(f"--rows range {part!r} out of range for {nrows} rows")
        indices.extend(range(start, end))
    if not indices:
        die("--rows selected no rows")
    return indices


def die(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def read_array(path: str) -> tuple[np.ndarray, tuple | None]:
    """Read raw little-endian float32 (or .npy) as float64 + known shape."""
    if not os.path.isfile(path):
        die(f"no such file: {path}")
    if path.endswith(".npy"):
        try:
            arr = np.load(path)
        except Exception as exc:  # noqa: BLE001 - report and bail out
            die(f"cannot load {path}: {exc}")
        shape = tuple(int(d) for d in arr.shape)
    else:
        arr = np.fromfile(path, dtype=F32_DTYPE)
        shape = None
    if arr.size == 0:
        die(f"empty file: {path}")
    return np.asarray(arr, dtype=np.float64).reshape(-1), shape


def load_manifest(directory: str) -> dict | None:
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - informational only
        warn(f"cannot parse {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


class Geometry:
    """Shapes recovered from one or both dump manifests, for display only."""

    def __init__(self, manifests: list[dict | None]) -> None:
        self.channels = None
        self.height = None
        self.width = None
        self.prompt_shape = None
        for m in manifests:
            if not m:
                continue
            latent = m.get("latent_shape")
            if (
                self.channels is None
                and isinstance(latent, (list, tuple))
                and len(latent) == 3
            ):
                self.channels, self.height, self.width = (int(v) for v in latent)
            if self.channels is None:
                # Accept the flat driver-manifest spelling too. The driver's
                # height/width are output pixels (with latent_h/latent_w and
                # latent_shape alongside); the golden manifest's are pixels
                # as well, but that one always ships latent_shape (above).
                c = m.get("C", m.get("latent_channels"))
                height = m.get("H", m.get("latent_h", m.get("height")))
                width = m.get("W", m.get("latent_w", m.get("width")))
                if c and height and width:
                    self.channels, self.height, self.width = (
                        int(c),
                        int(height),
                        int(width),
                    )
            prompt = m.get("prompt_embeds_shape")
            if (
                self.prompt_shape is None
                and isinstance(prompt, (list, tuple))
                and len(prompt) == 2
            ):
                self.prompt_shape = (int(prompt[0]), int(prompt[1]))

    def latent(self) -> tuple[int, int, int] | None:
        if self.channels and self.height and self.width:
            return (self.channels, self.height, self.width)
        return None

    def shape_for(self, name: str, count: int) -> tuple | None:
        latent = self.latent()
        if name == "prompt_embeds.f32":
            if (
                self.prompt_shape
                and self.prompt_shape[0] * self.prompt_shape[1] == count
            ):
                return self.prompt_shape
            if count % QWEN21_CONTEXT_DIM == 0:
                return (count // QWEN21_CONTEXT_DIM, QWEN21_CONTEXT_DIM)
            return None
        if name.endswith("_noise_pred_full.f32"):
            channels = self.channels or QWEN21_LATENT_CHANNELS
            if count % channels == 0:
                return (count // channels, channels)
            return None
        if latent and latent[0] * latent[1] * latent[2] == count:
            return latent
        channels = self.channels or QWEN21_LATENT_CHANNELS
        if count % channels == 0:
            side = int(round(math.sqrt(count // channels)))
            if side > 0 and channels * side * side == count:
                return (channels, side, side)
        return None


def shape_str(shape: tuple | None, count: int) -> str:
    if shape is None:
        return f"flat[{count}]"
    return "[" + ",".join(str(d) for d in shape) + "]"


def fmt_exp(x: float, width: int = 11, prec: int = 4) -> str:
    if x is None:
        return f"{'n/a':>{width}}"
    if isinstance(x, float) and math.isnan(x):
        return f"{'nan':>{width}}"
    if isinstance(x, float) and math.isinf(x):
        return f"{'inf' if x > 0 else '-inf':>{width}}"
    return f"{x:>{width}.{prec}e}"


def fmt_fixed(x: float, width: int = 10, prec: int = 6) -> str:
    if x is None:
        return f"{'n/a':>{width}}"
    if isinstance(x, float) and math.isnan(x):
        return f"{'nan':>{width}}"
    if isinstance(x, float) and math.isinf(x):
        return f"{'inf' if x > 0 else '-inf':>{width}}"
    return f"{x:>{width}.{prec}f}"


def metrics(golden: np.ndarray, actual: np.ndarray) -> dict:
    """Difference metrics for two equal-length float64 vectors."""
    diff = actual - golden
    with np.errstate(invalid="ignore", over="ignore"):
        abs_diff = np.abs(diff)
        d_norm = float(np.linalg.norm(diff))
        g_norm = float(np.linalg.norm(golden))
        a_norm = float(np.linalg.norm(actual))
        if g_norm > 0.0:
            rel_l2 = d_norm / g_norm
        else:
            rel_l2 = 0.0 if d_norm == 0.0 else float("inf")
        if g_norm > 0.0 and a_norm > 0.0:
            cosine = float(np.dot(golden, actual) / (g_norm * a_norm))
        else:
            cosine = 1.0 if np.array_equal(golden, actual) else 0.0
        return {
            "max_abs": float(abs_diff.max()),
            "mean_abs": float(abs_diff.mean()),
            "rms": float(np.sqrt(np.mean(diff * diff))),
            "rel_l2": float(rel_l2),
            "cosine": float(cosine),
            "sum_golden": float(np.sum(golden, dtype=np.float64)),
            "sum_actual": float(np.sum(actual, dtype=np.float64)),
            "nonfinite_golden": int((~np.isfinite(golden)).sum()),
            "nonfinite_actual": int((~np.isfinite(actual)).sum()),
        }


def cross_check_manifests(golden: dict | None, actual: dict | None) -> None:
    """Warn when the two dumps describe different runs (never a failure)."""
    if not golden or not actual:
        return
    for key in ("prompt", "seed", "cfg", "steps"):
        if key in golden and key in actual and golden[key] != actual[key]:
            warn(
                f"manifest {key} differs: golden={golden[key]!r} "
                f"actual={actual[key]!r}"
            )
    gsig = golden.get("sigmas")
    asig = actual.get("sigmas")
    if isinstance(gsig, list) and isinstance(asig, list):
        if len(gsig) != len(asig):
            warn(f"manifest sigmas length differs: {len(gsig)} vs {len(asig)}")
        else:
            worst = max(
                (abs(float(a) - float(b)) for a, b in zip(gsig, asig)),
                default=0.0,
            )
            if worst > 1e-6:
                warn(f"manifest sigmas differ (max |delta| {worst:.3e})")


def list_steps(directory: str) -> tuple[dict[int, str], dict[int, str]]:
    noise: dict[int, str] = {}
    latents: dict[int, str] = {}
    if not os.path.isdir(directory):
        return noise, latents
    for fname in os.listdir(directory):
        m = STEP_RE.match(fname)
        if not m:
            continue
        target = noise if m.group(2) == "noise_pred" else latents
        target[int(m.group(1))] = fname
    return noise, latents


def collect_pairs(
    golden_dir: str, actual_dir: str
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Return (pairs, skipped) for the fixed tensors and every shared step."""
    pairs: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []

    for name in CORE_FILES:
        gpath = os.path.join(golden_dir, name)
        apath = os.path.join(actual_dir, name)
        ge, ae = os.path.isfile(gpath), os.path.isfile(apath)
        if ge and ae:
            pairs.append((name, gpath, apath))
        elif ge:
            skipped.append((name, "missing from the actual dir"))
        elif ae:
            skipped.append((name, "missing from the golden dir"))

    for name in EDIT_REF_FILES:
        gpath = os.path.join(golden_dir, name)
        apath = os.path.join(actual_dir, name)
        ge, ae = os.path.isfile(gpath), os.path.isfile(apath)
        if ge and ae:
            pairs.append((name, gpath, apath))
        elif ge or ae:
            skipped.append(
                (name, "only on one side (edit dumps only; skipped)"))

    gnoise, glat = list_steps(golden_dir)
    anoise, alat = list_steps(actual_dir)
    for kind, gmap, amap in (
        ("noise_pred", gnoise, anoise),
        ("latents", glat, alat),
    ):
        for idx in sorted(set(gmap) | set(amap)):
            name = f"step{idx:02d}_{kind}.f32"
            if idx in gmap and idx in amap:
                pairs.append(
                    (name, os.path.join(golden_dir, gmap[idx]),
                     os.path.join(actual_dir, amap[idx]))
                )
            elif idx in gmap:
                # e.g. step-0 prefill noise_pred: golden has it, driver does not.
                skipped.append((name, "missing from the actual dir"))
            else:
                skipped.append((name, "missing from the golden dir"))
    return pairs, skipped


def normalize_tokens(
    flat: np.ndarray, path: str, geometry: Geometry
) -> tuple[np.ndarray, tuple]:
    """Reorder torch [N, C] token rows (row-major over h,w) to [C, H, W]."""
    latent = geometry.latent()
    if latent is None:
        die(
            f"{path}: token layout needs latent C,H,W; no manifest.json with "
            "latent_shape (or C/H/W) was found in either dump"
        )
    channels, height, width = latent
    if flat.size != channels * height * width:
        die(
            f"{path}: {flat.size} elements do not fit the manifest latent "
            f"shape [{channels},{height},{width}]"
        )
    reordered = flat.reshape(height * width, channels).transpose(1, 0)
    return reordered.reshape(-1).copy(), latent


def compare_pair(
    name: str,
    gpath: str,
    apath: str,
    geometry: Geometry,
    rtol: float,
    atol: float | None,
    glayout: str = "flat",
    alayout: str = "flat",
    row_cos_min: float | None = None,
    row_cos_mean_min: float | None = None,
    rows: str | None = None,
) -> dict:
    golden_flat, golden_shape = read_array(gpath)
    actual_flat, actual_shape = read_array(apath)
    count = golden_flat.size
    if actual_flat.size != count:
        die(
            f"{name}: element count mismatch, golden {count} vs actual "
            f"{actual_flat.size}"
        )

    notes: list[str] = []
    layout_shape = None
    if glayout == "tokens":
        golden_flat, layout_shape = normalize_tokens(
            golden_flat, gpath, geometry
        )
        notes.append("golden transposed from [N,C] token packing")
    if alayout == "tokens":
        actual_flat, actual_layout_shape = normalize_tokens(
            actual_flat, apath, geometry
        )
        notes.append("actual transposed from [N,C] token packing")
        layout_shape = layout_shape or actual_layout_shape

    if layout_shape is not None:
        gshape = ashape = layout_shape
    else:
        gshape = golden_shape or geometry.shape_for(name, count)
        ashape = actual_shape or geometry.shape_for(name, count)
    if gshape is not None and ashape is not None and gshape != ashape:
        warn(f"{name}: shapes differ ({gshape} vs {ashape}); comparing flattened")
        gshape = ashape = None
    shown_shape = gshape or ashape

    if rows is not None and name == PROMPT_EMBEDS:
        if shown_shape is None or len(shown_shape) != 2:
            die(f"{name}: --rows needs a 2-D prompt shape, got {shown_shape}")
        nrows = shown_shape[0]
        keep = parse_row_spec(rows, nrows)
        gg = golden_flat.reshape(shown_shape)[keep, :]
        aa = actual_flat.reshape(shown_shape)[keep, :]
        shown_shape = (len(keep), shown_shape[1])
        gshape = ashape = shown_shape
        golden_flat = gg.reshape(-1).copy()
        actual_flat = aa.reshape(-1).copy()
        count = golden_flat.size
        notes.append(f"rows {rows} selected ({len(keep)}/{nrows})")

    m = metrics(golden_flat, actual_flat)

    reasons: list[str] = []
    if m["nonfinite_golden"] or m["nonfinite_actual"]:
        reasons.append(
            f"non-finite values (golden {m['nonfinite_golden']}, "
            f"actual {m['nonfinite_actual']})"
        )
    if m["rel_l2"] > rtol:
        reasons.append(f"rel-L2 {m['rel_l2']:.3e} > {rtol:.1e}")
    if atol is not None and m["max_abs"] > atol:
        reasons.append(f"max|diff| {m['max_abs']:.3e} > atol {atol:.1e}")

    row_cos = None
    row_cos_mean = None
    if (
        (row_cos_min is not None or row_cos_mean_min is not None)
        and gshape is not None
        and len(gshape) == 2
    ):
        # Row-wise cosine is the meaningful gate for prompt embeddings: bf16
        # device noise moves token rows almost rigidly, so a global rel-L2
        # built from large row sums overstates the disagreement. Both the
        # minimum and the mean over rows are reported; the minimum gates
        # localized corruption while the mean gates implementation-level
        # drift (scattered low rows from input noise fail min but pass mean).
        gg = golden_flat.reshape(gshape)
        aa = actual_flat.reshape(gshape)
        num = np.sum(gg * aa, axis=1)
        den = np.linalg.norm(gg, axis=1) * np.linalg.norm(aa, axis=1)
        cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        if cos.size:
            row_cos = float(np.min(cos))
            row_cos_mean = float(np.mean(cos))
        if row_cos_min is not None and row_cos is not None and row_cos < row_cos_min:
            reasons.append(f"min row cosine {row_cos:.6f} < {row_cos_min:g}")
        if (
            row_cos_mean_min is not None
            and row_cos_mean is not None
            and row_cos_mean < row_cos_mean_min
        ):
            reasons.append(
                f"mean row cosine {row_cos_mean:.6f} < {row_cos_mean_min:g}"
            )

    return {
        "name": name,
        "shape": shown_shape,
        "count": count,
        "notes": notes,
        "row_cos": row_cos,
        "row_cos_mean": row_cos_mean,
        **m,
        "ok": not reasons,
        "reasons": reasons,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'tensor':<26} {'shape':<13} {'max|d|':>11} {'mean|d|':>11} "
        f"{'rms(d)':>11} {'rel-L2':>10} {'cos':>9} "
        f"{'sum(golden)':>14} {'sum(actual)':>14}  result"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<26} {shape_str(row['shape'], row['count']):<13} "
            f"{fmt_exp(row['max_abs'])} {fmt_exp(row['mean_abs'])} "
            f"{fmt_exp(row['rms'])} {fmt_exp(row['rel_l2'], 10)} "
            f"{fmt_fixed(row['cosine'], 9)} "
            f"{fmt_exp(row['sum_golden'], 14, 6)} "
            f"{fmt_exp(row['sum_actual'], 14, 6)}  "
            f"{'PASS' if row['ok'] else 'FAIL'}"
        )
    for row in rows:
        if row.get("row_cos") is not None:
            print(f"  {row['name']}: min row cosine {row['row_cos']:.6f}")
        if row.get("row_cos_mean") is not None:
            print(f"  {row['name']}: mean row cosine {row['row_cos_mean']:.6f}")


def compare_images(gpath: str, apath: str, psnr_min: float) -> dict:
    try:
        from PIL import Image
    except ImportError:
        die(
            "Pillow is required for --images "
            "(`pip install pillow`); tensor comparison does not need it"
        )

    for path in (gpath, apath):
        if not os.path.isfile(path):
            die(f"no such image: {path}")
    with Image.open(gpath) as im:
        golden = im.convert("RGB")
    with Image.open(apath) as im:
        actual = im.convert("RGB")
    if golden.size != actual.size:
        die(
            f"image size mismatch: golden {golden.size[0]}x{golden.size[1]}, "
            f"actual {actual.size[0]}x{actual.size[1]}"
        )

    g = np.asarray(golden, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    diff = a - g
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    if mse > 0.0:
        psnr = 10.0 * math.log10(255.0 * 255.0 / mse)
    else:
        psnr = float("inf")
    per_channel = [float(abs_diff[..., c].max()) for c in range(g.shape[2])]

    print("== images ==")
    print(f"golden : {gpath} ({golden.size[0]}x{golden.size[1]}, RGB)")
    print(f"actual : {apath} ({actual.size[0]}x{actual.size[1]}, RGB)")
    print(f"PSNR            : {fmt_fixed(psnr, 0, 2)} dB "
          f"(min {psnr_min:g} dB)")
    print(f"max channel diff: {fmt_fixed(max(per_channel), 0, 0)} "
          f"(mean channel max {fmt_fixed(float(np.mean(per_channel)), 0, 2)})")
    print(f"mean abs diff   : {fmt_fixed(float(abs_diff.mean()), 0, 2)}")
    ok = psnr >= psnr_min
    print(f"result          : {'PASS' if ok else 'FAIL'}")
    return {
        "name": "image PSNR",
        "ok": ok,
        "reasons": [] if ok else [f"PSNR {psnr:.2f} dB < {psnr_min:g} dB"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--golden", metavar="DIR", help="golden dump directory")
    ap.add_argument("--actual", metavar="DIR", help="actual dump directory")
    ap.add_argument(
        "--final-golden",
        metavar="FILE",
        help="optional final latents file (raw float32 or .npy) from the "
        "golden side; may be packed differently than --final-actual",
    )
    ap.add_argument(
        "--final-actual",
        metavar="FILE",
        help="optional final latents file from the actual side",
    )
    ap.add_argument(
        "--final-golden-layout",
        choices=("flat", "tokens"),
        default="flat",
        help="packing of --final-golden: 'flat' compares elements as stored, "
        "'tokens' treats it as torch [N,C] token rows and reorders it to "
        "[C,H,W] first (default: flat)",
    )
    ap.add_argument(
        "--final-actual-layout",
        choices=("flat", "tokens"),
        default="flat",
        help="packing of --final-actual (default: flat)",
    )
    ap.add_argument(
        "--images",
        nargs=2,
        metavar=("GOLDEN_PNG", "ACTUAL_PNG"),
        help="also compare decoded PNGs (Pillow required)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any compared tensor or image fails",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=REL_L2_DEFAULT,
        help=f"relative L2 threshold (default {REL_L2_DEFAULT:g})",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=None,
        help="optional max|diff| threshold, in addition to --rtol (default: "
        "no max|diff| gate)",
    )
    ap.add_argument(
        "--psnr-min",
        type=float,
        default=PSNR_MIN_DEFAULT,
        help=f"minimum image PSNR in dB for --images (default "
        f"{PSNR_MIN_DEFAULT:g})",
    )
    ap.add_argument(
        "--only",
        metavar="NAMES",
        default=None,
        help="comma-separated tensor names to compare (e.g. "
        "prompt_embeds.f32); default: every shared tensor",
    )
    ap.add_argument(
        "--row-cos-min",
        type=float,
        default=None,
        metavar="C",
        help="optional minimum per-row cosine for 2-D tensors (e.g. 0.999 "
        "for prompt_embeds); the row length comes from the manifest shape",
    )
    ap.add_argument(
        "--row-cos-mean-min",
        type=float,
        default=None,
        metavar="C",
        help="optional minimum mean per-row cosine for 2-D tensors; gates "
        "implementation drift instead of scattered input-noise rows "
        "(e.g. 0.985 for vision rows)",
    )
    ap.add_argument(
        "--rows",
        metavar="RANGES",
        default=None,
        help="optional prompt_embeds row subset (e.g. 8:1034 or "
        "0:8,1034:1059, end-exclusive); gates whole-tensor metrics and "
        "per-row cosine on the subset only; default: all rows",
    )
    args = ap.parse_args(argv)

    if args.rtol <= 0.0:
        die("--rtol must be > 0")
    if args.atol is not None and args.atol < 0.0:
        die("--atol must be >= 0")
    if bool(args.golden) != bool(args.actual):
        die("--golden and --actual must be given together")
    if bool(args.final_golden) != bool(args.final_actual):
        die("--final-golden and --final-actual must be given together")
    if args.final_golden and not args.golden:
        die(
            "--final-golden/--final-actual need --golden/--actual: the latent "
            "geometry used to reorder token-packed files comes from a manifest"
        )
    if not args.golden and not args.images:
        die("nothing to do: pass --golden/--actual and/or --images")

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    if args.golden:
        for directory in (args.golden, args.actual):
            if not os.path.isdir(directory):
                die(f"not a directory: {directory}")
        manifests = [load_manifest(args.golden), load_manifest(args.actual)]
        geometry = Geometry(manifests)
        if not any(manifests):
            warn(
                "no manifest.json in either dump; shapes fall back to flat "
                "vectors (metrics are unaffected)"
            )
        cross_check_manifests(manifests[0], manifests[1])

        print("== tensors ==")
        print(f"golden : {args.golden}")
        print(f"actual : {args.actual}")
        print(f"threshold: rel-L2 <= {args.rtol:g}"
              + (f", max|diff| <= {args.atol:g}" if args.atol is not None else ""))

        pairs, skipped = collect_pairs(args.golden, args.actual)
        if args.final_golden:
            pairs.append(("final_latents", args.final_golden, args.final_actual))
        if args.only:
            wanted = {s.strip() for s in args.only.split(",") if s.strip()}
            pairs = [p for p in pairs if p[0] in wanted]
            skipped = [s for s in skipped if s[0] in wanted]
            if not pairs:
                die(f"--only {args.only!r} matched no shared tensor")
        for name, gpath, apath in pairs:
            is_final = name == "final_latents"
            rows.append(
                compare_pair(
                    name,
                    gpath,
                    apath,
                    geometry,
                    args.rtol,
                    args.atol,
                    glayout=args.final_golden_layout if is_final else "flat",
                    alayout=args.final_actual_layout if is_final else "flat",
                    row_cos_min=args.row_cos_min,
                    row_cos_mean_min=args.row_cos_mean_min,
                    rows=args.rows,
                )
            )

        if not rows:
            die(
                "no comparable tensors found (no initial_latents.f32 / "
                "step{ii}_*.f32 in both dirs)"
            )
        print_table(rows)
        for row in rows:
            for note in row["notes"]:
                print(f"NOTE: {row['name']}: {note}")
        for name, detail in sorted(skipped):
            print(f"NOTE: {name} {detail} (skipped)")

    image_result = None
    if args.images:
        if rows:
            print()
        image_result = compare_images(args.images[0], args.images[1],
                                      args.psnr_min)

    tensor_failures = [r for r in rows if not r["ok"]]
    failures = list(tensor_failures)
    if image_result is not None and not image_result["ok"]:
        failures.append(image_result)

    if rows or image_result is not None:
        print()
        print("== summary ==")
        if rows:
            print(f"tensors compared : {len(rows)}")
            print(f"tensors passed   : {len(rows) - len(tensor_failures)}")
            print(f"tensors failed   : {len(tensor_failures)}")
            print(f"tensors skipped  : {len(skipped)}")
        if image_result is not None:
            print(f"image            : "
                  f"{'PASS' if image_result['ok'] else 'FAIL'}")
        for row in failures:
            detail = "; ".join(row["reasons"]) or "threshold violation"
            print(f"  FAIL {row['name']}: {detail}")
        print(f"RESULT: {'PASS' if not failures else 'FAIL'}")

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
