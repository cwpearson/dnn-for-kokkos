# Kokoro example

Kokoro-82M text-to-speech inference (StyleTTS2 + ISTFTNet) built on the
core library: dependency-free `kokoro-v1_0.pth` checkpoint loader
(weight_norm fused on host), phoneme tokenization from `config.json`,
PLBERT encoder, duration/pitch/energy prosody predictor, text encoder,
length regulation, and the ISTFTNet neural vocoder (harmonic source +
STFT/iSTFT, 24 kHz output).

Scope: **inference only**, batch-1, **float32**.

## Quickstart (0 to working)

From the worktree root (`/workspace/kokoro-tts`) — build, fetch checkpoint
files, run:

```bash
cmake -S . -B build -DKokkos_ROOT=/path/to/kokkos-install -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8 --target kokoro_tts

pip install -U huggingface_hub  # provides the `hf` CLI
export PATH=$HOME/.local/bin:$PATH
hf download hexgrad/Kokoro-82M \
  --include "config.json" \
  --include "kokoro-v1_0.pth" \
  --include "voices/af_heart.pt" \
  --local-dir ./kokoro-weights

./build/examples/kokoro/kokoro_tts \
  --weights ./kokoro-weights/kokoro-v1_0.pth \
  --voice ./kokoro-weights/voices/af_heart.pt \
  --config examples/kokoro/config-kokoro-v1_0.json \
  --phonemes "hɛˈloʊ" --output hello.wav
```

Expected tensor inventory: 548 tensors (~82M params); voices are
`[510,1,256]` float32 packs. The voice vector is row
`clamp(len(phonemes)-1)` of the pack, matching `KPipeline.infer`
(`pack[len(ps)-1]`).

## Example commands

```bash
# Tiny random model, no downloads (also the unit-test end-to-end path):
./build/examples/kokoro/kokoro_tts --tiny --phonemes "hello" --output tiny.wav

# Real checkpoint, different speed / voice row:
./build/examples/kokoro/kokoro_tts \
  --weights ./kokoro-weights/kokoro-v1_0.pth \
  --voice ./kokoro-weights/voices/af_heart.pt \
  --config examples/kokoro/config-kokoro-v1_0.json \
  --phonemes "hɛˈloʊ wɜːld" --speed 1.1 --voice-index 42 --output out.wav
```

## Notes

- Flags (see `kokoro_tts.cpp` header): `--tiny`, `--weights FILE`,
  `--voice FILE`, `--config FILE` (default
  `examples/kokoro/config-kokoro-v1_0.json`), `--phonemes STR` / `--text STR`
  (alias; plain text is mapped letter-by-letter, no G2P), `--output FILE`,
  `--speed F`, `--voice-index I` (default: `len(phonemes)-1`, clamped).
- Input is an IPA phoneme string: each UTF-8 character is looked up in the
  `config.json` vocab (unknown chars dropped), wrapped with boundary id `0`
  on both sides. Proper G2P (misaki/espeak-ng) is out of scope.
- Deterministic: harmonic phase accumulation starts at zero and the
  SineGen/noise draws use a fixed seed, so output is reproducible but differs
  stochastically from torch (which draws random phases/noise).
- Output length is exactly `600 * sum(pred_dur)` samples at 24 kHz
  (upsampling 2x decoder + 10x/6x generator + 5x iSTFT hop, times the
  doubled F0 frame rate).
- Layout: `KokkosDNN_KokoroConfig.{hpp,cpp}` (config + vocab JSON),
  `KokkosDNN_KokoroPth.{hpp,cpp}` (ZIP/pickle reader for the model file
  and single-tensor voice files), `KokkosDNN_KokoroWeights.{hpp,cpp}`
  (device views + weight_norm fusion + voice row select),
  `KokkosDNN_KokoroModel.{hpp,cpp}` (`synthesize()` pipeline),
  `kokoro_tts.cpp` (driver + WAV writer),
  `config-kokoro-v1_0.json` (HF config copy), `Test_*` (tests).

## Verification

```bash
ctest --test-dir build -R Kokoro --output-on-failure
KOKORO_PTH=./kokoro-weights/kokoro-v1_0.pth \
KOKORO_VOICE=./kokoro-weights/voices/af_heart.pt \
KOKORO_CONFIG=examples/kokoro/config-kokoro-v1_0.json \
  ctest --test-dir build -R KokoroGolden --output-on-failure
```

Unit tests (`Test_KokoroOps`, `Test_KokoroSmoke`) are tiny-synthetic only,
no downloads. `Test_KokoroGolden` needs the real checkpoint + voice (skips
otherwise) and checks the 548-tensor inventory, output length, finiteness,
and non-silence.
