#!/usr/bin/env python3
"""Generate Kokoro golden reference JSON via the torch KModel.

Runs the reference pipeline (kokoro pip model.py, v0.7.x) on fixed phoneme
strings with a fixed torch seed and saves deterministic intermediates plus
audio properties. The C++ test (Test_KokoroGolden.cpp) compares against this
file; audio is stochastic (SineGen RNG), so only length/RMS/peak are saved
for it, while pred_dur/F0/N/asr are bit-tolerant deterministic goldens.

Usage:
    python3 examples/kokoro/generate_kokoro_golden.py \
        --out examples/kokoro/golden_short.json [--phonemes "hɛloʊ wɜːld"]
"""

import argparse
import json
import sys

try:
    import torch
    from kokoro.model import KModel
except ImportError as e:
    raise SystemExit(
        "need the `kokoro` pip package for the torch reference "
        "(pip install 'kokoro==0.7.*', plus torch/transformers): "
        f"{e}"
    ) from e


def run_case(model, pack, phonemes, speed, seed):
    ids = [i for i in map(model.vocab.get, phonemes) if i is not None]
    ref_s = pack[len(phonemes) - 1]
    torch.manual_seed(seed)
    with torch.no_grad():
        # Replicate KModel.forward with intermediates exposed.
        input_ids = torch.LongTensor([[0, *ids, 0]])
        input_lengths = torch.LongTensor([input_ids.shape[-1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1
        ).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1]), pred_dur
        )
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = model(phonemes, ref_s, speed)
    return {
        "phonemes": phonemes,
        "speed": speed,
        "seed": seed,
        "ids": ids,
        "n_codepoints": len(phonemes),
        "pred_dur": pred_dur.tolist(),
        "frames": int(pred_dur.sum()),
        "f0": F0_pred.squeeze(0).tolist(),
        "ncurve": N_pred.squeeze(0).tolist(),
        "asr": asr.squeeze(0).tolist(),  # [512, F]
        "audio_len": int(audio.numel()),
        "audio_rms": float((audio**2).mean().sqrt()),
        "audio_peak": float(audio.abs().max()),
        "audio_mean": float(audio.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phonemes", default="hɛloʊ wɜːld")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import logging

    logging.disable(50)
    from huggingface_hub import hf_hub_download

    model = KModel()
    model.eval()
    vp = hf_hub_download(repo_id=KModel.REPO_ID,
                         filename=f"voices/{args.voice}.pt")
    pack = torch.load(vp, map_location="cpu", weights_only=True)
    case = run_case(model, pack, args.phonemes, args.speed, args.seed)
    case["voice"] = args.voice
    with open(args.out, "w") as f:
        json.dump(case, f)
    print(f"wrote {args.out}: frames={case['frames']} "
          f"audio_len={case['audio_len']} rms={case['audio_rms']:.6f}")


if __name__ == "__main__":
    main()
