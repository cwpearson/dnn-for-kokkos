# Kokoro-82M text-to-speech example

C++20 Kokkos port of the `hexgrad/Kokoro-82M` StyleTTS2 + ISTFTNet inference
stack (82M params, FP32, 24 kHz output). Reference: the `kokoro` pip package
(`kokoro/model.py`, `modules.py`, `istftnet.py`, v0.7.x):
`KModel.forward(phonemes, ref_s, speed)` with `af_heart`-style voice packs.

## Quickstart (0 to working)

Run from the repo root.

```bash
# 1. Build and install Kokkos once (CUDA, AMPERE86)
git clone --branch 5.2.1 --recursive https://github.com/kokkos/kokkos.git kokkos-src
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<path-to>/kokkos-install \
  -DCMAKE_CXX_COMPILER=<path-to>/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build kokkos-build --parallel 16
cmake --install kokkos-build

# 2. Configure and build the example target (CUDA, cuBLAS + cuDNN)
cmake -S . -B build-cuda \
  -DCMAKE_CXX_COMPILER=<path-to>/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ROOT=<path-to>/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkosDNN_ENABLE_CUBLAS=ON \
  -DKokkosDNN_ENABLE_CUDNN=ON
cmake --build build-cuda --parallel 16 \
  --target kokoro_tts test_kokoro_smoke test_kokoro_pth test_kokoro_golden

# 3. Download the checkpoint files with the `hf` CLI
pip install -U "huggingface_hub[cli]"
# working run: no downloads needed (random weights, tiny channels)
./build-cuda/examples/kokoro/kokoro_tts --tiny --out tiny.wav

# full stack (downloads ~330 MB model + ~0.5 MB voice)
hf download hexgrad/Kokoro-82M --local-dir ./kokoro-weights \
  --include 'kokoro-v1_0.pth' 'voices/af_heart.pt'
./build-cuda/examples/kokoro/kokoro_tts \
  --weights ./kokoro-weights/kokoro-v1_0.pth \
  --voice ./kokoro-weights/voices/af_heart.pt \
  --phonemes "hɛloʊ wɜːld" --out hello.wav
```

`--phonemes` is the primary input: KModel is language-blind and consumes
phoneme symbols, not graphemes (produce them with the `kokoro` KPipeline in
quiet mode, or `espeak-ng -v en --ipa`). `--text` maps characters present in
the vocab directly (no G2P; unmapped characters are skipped) — demo quality
only. WAV is peak-normalized PCM16; raw float32 always lands at `<stem>.f32`.

Driver flags (`kokoro_tts`):

| flag | default | meaning |
|---|---|---|
| `--weights` | `kokoro-v1_0.pth` | torch.save Kokoro-82M checkpoint |
| `--voice` | `af_heart.pt` | voice pack (`voices/*.pt`) |
| `--phonemes` | `hɛloʊ wɜːld` | raw IPA string (primary input) |
| `--text` | — | naive fallback: vocab chars mapped directly, no G2P |
| `--speed` | `1.0` | duration divisor |
| `--seed` | `42` | SineGen RNG seed (see Notes) |
| `--out` | `kokoro_out.wav` | WAV path (PCM16 peak-normalized; raw F32 to `<stem>.f32`) |
| `--tiny` | off | random-weights smoke model (no files) |

## Notes

- Architecture: phoneme ids (+BOS/EOS pads) → PL-BERT Albert (128-dim
  embed, 768 hidden, 12 heads, one shared layer applied 12x, GELU-tanh) →
  Linear 768→512 → DurationEncoder (3x BiLSTM-640/AdaLN) → duration_proj
  (50-way sigmoid sum → `pred_dur`, alignment) → shared BiLSTM → F0/N Adain
  stacks (second block upsamples x2) → F0/N curves; ids → TextEncoder
  (embed 512, 3x Conv1d k5 + BiLSTM-512) → aligned ASR features; Decoder
  (F0/N Conv1d k3/s2 downsample, AdainResBlk encode + 4x decode with last
  block x2) → ISTFTNet Generator (nearest-f0 x300 SineGen excitation, STFT
  harmonics, ups x10/x6, AdaINResBlocks with Snake activation, conv_post →
  magnitude/phase) → inverse STFT (n_fft 20, hop 5, Hann) → 300·F₂ samples
  at 24 kHz. FP32 throughout; batch size is always 1; no padding (masks are
  all-valid by construction).
- WeightNorm is fused on the host at load time (`w = g·v/‖v‖`), so every
  device kernel sees plain dense weights. The `.pth`/`.pt` files are parsed
  by a dependency-free ZIP + protocol-2 pickle reader
  (`KokkosDNN_KokoroPth.*`, nested-dict capable, float32 only).
- SineGen randomness: torch's CPU RNG stream is not reproduced (its
  uniform/normal draws don't match `std::mt19937` + `<random>`); Kokoro
  audio therefore differs from the torch reference by RNG stream but is
  deterministic given `--seed`. Voice row selection follows the pipeline
  (`pack[len(phonemes)-1]`, length in Unicode code points). Duration,
  alignment, F0/N, and ASR stages are exactly reproducible and covered by
  golden tests.
- Targets (`CMakeLists.txt`): `kokkosdnn_kokoro` lib + `kokoro_tts` +
  `test_kokoro_{smoke,pth,golden}`. Sources:
  `KokkosDNN_Kokoro{Config,Weights,Model,Pth}` plus driver `kokoro_tts.cpp`.
  Core ops added for this example live in `src/`: `KokkosDNN_LSTM` (bidir
  LSTM), `KokkosDNN_Conv1DGeneral` (dense/transposed conv1d),
  `KokkosDNN_AdaptiveNorm` (InstanceNorm/AdaIN/AdaLN),
  `KokkosDNN_AudioDSP` (Hann/STFT/ISTFT/resample), leaky-ReLU + Snake in
  `KokkosDNN_Activation`.

## Verification

Golden tests (torch reference never runs from ctest):

1. `python3 examples/kokoro/generate_kokoro_golden.py --out golden.json`
   (needs the `kokoro` pip package + checkpoint), then
   `KOKKOSDNN_KOKORO_WEIGHTS=.../kokoro-v1_0.pth KOKKOSDNN_KOKORO_VOICE=.../af_heart.pt KOKKOSDNN_KOKORO_GOLDEN=golden.json ./build-cuda/examples/kokoro/test_kokoro_golden`
   (tols `KOKKOSDNN_KOKORO_TOL`, default 1e-3; audio RMS tol
   `KOKKOSDNN_KOKORO_AUDIO_TOL`, default 0.02).
2. `.pth` inventory: `KOKKOSDNN_KOKORO_WEIGHTS=... test_kokoro_pth
   --gtest_filter=KokoroPth.InventoryMatchesRelease` (548 tensors,
   81,763,410 params, spot shapes).
3. Weight-free: `test_kokoro_smoke` (tiny random model: length contract,
   finiteness, seed determinism) and the core-op unit tests
   (`test_lstm`, `test_conv1dgeneral`, `test_adaptivenorm`, `test_audiodsp` —
   the STFT test carries embedded torch spectra).

`ctest -R 'Kokoro|lstm|lstm|conv1dgeneral|adaptivenorm|audiodsp'` stays fast:
real-weight forwards are env-gated.

## Performance

Tiny smoke forward is milliseconds on either backend. Full 82M forward on
the short demo (~50 frames) is seconds on Serial, well under a second of
kernel time on CUDA (RTX 3090); the generator convolutions at audio rate
dominate. Long paragraphs scale linearly in frames (audio = 300·F₂).
