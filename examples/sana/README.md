# Sana image-generation example

C++20 Kokkos port of the NVlabs/Sana `SanaMS_600M_P1_D28` text-to-image stack (600M DiT, 1024px, MultiLing) + FlowEuler sampler + `sana_generate` driver. Reference: `diffusion/model/nets/sana*.py`, `app/sana_pipeline.py`, `diffusion/scheduler/flow_euler_sampler.py`, config `Sana_600M_img1024.yaml`.

## Quickstart (0 to working)

Run from repo root (`/workspace`). Gemma is gated: accept the Terms of Use at <https://huggingface.co/google/gemma-2-2b-it> once, then export a granted token as `HF_TOKEN`.

```bash
# 1. Build and install Kokkos once (CUDA, AMPERE86)
git clone --branch 5.2.1 --recursive https://github.com/kokkos/kokkos.git kokkos-src
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-install \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build kokkos-build --parallel 8
cmake --install kokkos-build

# 2. Configure and build the example target (CUDA, cuBLAS + cuDNN)
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkosDNN_ENABLE_CUBLAS=ON \
  -DKokkosDNN_ENABLE_CUDNN=ON \
  -DKokkosDNN_ENABLE_IMAGE=ON
cmake --build build --parallel 8 --target sana_generate

# 3. Download the checkpoint files with the `hf` CLI
pip install -U "huggingface_hub[cli]"

# DiT backbone (.pth)
hf download Efficient-Large-Model/Sana_600M_1024px \
  --local-dir ./Sana_600M_1024px
# text encoder (gated)
hf download google/gemma-2-2b-it \
  --local-dir ./gemma-2-2b-it --token "$HF_TOKEN"
# VAE decoder (diffusers layout)
hf download mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
  --local-dir ./dc-ae-f32c32-sana-1.0-diffusers

# working run: no downloads needed
./build/examples/sana/sana_generate --tiny --steps 2
# full stack
./build/examples/sana/sana_generate \
  --weights Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth \
  --text-weights ./gemma-2-2b-it --prompt "a cyberpunk cat" --steps 18 \
  --vae-safetensors ./dc-ae-f32c32-sana-1.0-diffusers/diffusion_pytorch_model.safetensors \
  --out sana_out.png
```

Checkpoint → flag mapping: `--weights` = DiT `.pth`; `--text-weights` (alias `--gemma-weights`, file or shard dir) = Gemma-2-2B-IT snapshot (needs sibling `tokenizer.json` + `config.json` unless `--tokenizer`/`--text-config` override); `--vae-safetensors` = **diffusers** `diffusion_pytorch_model.safetensors` (native efficientvit `model.safetensors` has different key names and is rejected). `--prompt-embeds` (raw F32 `[1,1,300,2304]`; `[1,300,2304]`/`[300,2304]` accepted) overrides `--text-weights`.

## Example commands

```bash
# tiny smoke: random weights, 4x4 latents, 2 Euler steps (no downloads)
./build/examples/sana/sana_generate --tiny --steps 2
# real DiT, null conditioning (prompt logged only)
./build/examples/sana/sana_generate \
  --weights Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth \
  --steps 18 --guidance 5.0 --seed 42 --out sana_out.png
# + native Gemma prompt encoding
./build/examples/sana/sana_generate --weights .../Sana_600M_1024px_MultiLing.pth \
  --text-weights ./gemma-2-2b-it --prompt "a cyberpunk cat" --steps 18
# + precomputed features (highest precedence) and/or true VAE decode
./build/examples/sana/sana_generate --weights .../Sana_600M_1024px_MultiLing.pth \
  --prompt-embeds feats.f32 --steps 18 \
  --vae-safetensors ./dc-ae-f32c32-sana-1.0-diffusers/diffusion_pytorch_model.safetensors
# inventory checks (fast, no weight load)
./build/examples/sana/test_sana_golden
./build/examples/sana/test_sana_golden --gtest_filter=SanaGolden.PthInventoryMatchesRelease
KOKKOSDNN_SANA_VAE_SAFETENSORS=... ./build/examples/sana/test_sana_vae \
  --gtest_filter=SanaVAE.HeaderInventoryMatchesDiffusers
```

Driver flags (`sana_generate`; verified against `sana_generate.cpp`):

| flag | default | meaning |
|---|---|---|
| `--tiny` | off | random-weights smoke model, 4x4 latents; ignores `--prompt-embeds`/`--text-weights`/`--vae-safetensors` |
| `--weights` | `Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth` | torch.save DiT checkpoint |
| `--prompt` | `a cyberpunk cat` | natively encoded only with `--text-weights`, else logged and null embedding used |
| `--prompt-embeds` | — | raw F32 Gemma features; overrides `--text-weights` |
| `--text-weights` (`--gemma-weights`) | — | Gemma `.safetensors` file or shard dir |
| `--tokenizer` (`--gemma-tokenizer`) / `--text-config` | — | `tokenizer.json` / `config.json` overrides |
| `--chi-prompt` | — | complex-human-instruction prefix |
| `--steps` / `--guidance` / `--flow-shift` / `--seed` | 18 / 5.0 / 4.0 / 42 | Euler steps / CFG scale (`<=1` disables) / sigma shift / latent seed |
| `--out` | `sana_out.png` | PNG (`.jpg`/`.jpeg` writes JPEG) with `KokkosDNN_ENABLE_IMAGE=ON`, else `.ppm` |
| `--vae-safetensors` | — | DC-AE decoder weights; without it a channel-mean grayscale preview (32x nearest) is written |
| `--vae-decode` | on (`--no-vae-decode` for preview) | real VAE decode when `--vae-safetensors` is given, else grayscale preview |
| `--fp32` | — | accepted no-op (float is the checkpoint dtype) |

## Notes

- Architecture: patch-1 embed of 32ch/32x32 DC-AE latents → 1024x1152 tokens, no pos-embed (`use_pe=false`); timestep cos/sin MLP + shared `t_block`; 28 blocks (adaLN, LiteLA ReLU linear self-attn 36x32, softmax cross-attn 16x72 over RMSNorm'd `[300,2304]` caption, GLUMBConv MLP); FlowEuler sampler (`sig = shift·u/(1+(shift−1)·u)`, shift 4.0, `v = uncond + g·(cond−uncond)`, mt19937 Box-Muller init). Gemma-2-2B-IT encoder: hidden 2304, 26 layers, 8Q/4KV GQA, MLP 9216 GeLU-tanh, vocab 256000, theta 10000, `(1+w)` RMSNorms, softcap 50; `encode_prompt` = Sana template/truncate/pad-300 + `select_index` gather. VAE: DC-AE f32c32 (`latent/0.41407` → decode → `x*0.5+0.5` → u8; 6 stages 128→1024, ResBlock x3 + EfficientViT x3, 32x upsample).
- Limitations: `--prompt` alone never conditions (null `y_embedding` fallback; with `--text-weights` the uncond branch encodes `""`); no PAG, no caption-mask trimming (all 300 tokens attended); fixed 1024 geometry; FP32 only; serial-CPU full-res is hours — CUDA parallelizes the same kernels.
- Targets (`CMakeLists.txt`): `kokkosdnn_sana` lib + `sana_generate` + `test_sana_{ops,smoke,sampler,vae,text_encoder,golden}`. Sources: `KokkosDNN_Sana{Config,Weights,Model,Sampler,TextEncoder,VAE}` + dependency-free `.pth` ZIP/pickle readers (`load_torch_pth` for the driver, `open_pth`/`read_pth_storage` for inventory; bit-identical).

## Verification

Golden tests (torch references never run from ctest):

1. DiT: `python3 examples/sana/generate_golden.py --repo <Sana> --weights <pth> --out golden.json`, then `KOKKOSDNN_SANA_WEIGHTS=<pth> KOKKOSDNN_SANA_GOLDEN_JSON=golden.json ./build/examples/sana/test_sana_golden --gtest_filter=SanaGolden.RealWeightsForwardFp32` (tol `KOKKOSDNN_SANA_TOL`, default 2e-3). Loader cross-check: both readers agree bit-exactly; `y_embedding` matches an independent Python parse (mean 0.099986, std 2.773899).
2. Text encoder: `python3 examples/sana/generate_gemma_golden.py --model google/gemma-2-2b-it --prompt "a cyberpunk cat" --out gemma_golden.json`, then `KOKKOSDNN_SANA_GEMMA_WEIGHTS=gemma-2-2b-it KOKKOSDNN_SANA_GEMMA_GOLDEN_JSON=gemma_golden.json ./build/examples/sana/test_sana_text_encoder --gtest_filter=SanaTextEncoder.GoldenVsTransformers` (tol `KOKKOSDNN_SANA_GEMMA_TOL`, default 5e-2; torch ref is BF16, encoder is FP32). Weight-free: C++ matches NumPy Gemma2 to 9.5e-7 max-abs on tiny weights.
3. VAE: `python3 examples/sana/generate_vae_golden.py --repo mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers --out vae_golden.json`, then `KOKKOSDNN_SANA_VAE_SAFETENSORS=... KOKKOSDNN_SANA_VAE_GOLDEN_JSON=vae_golden.json ./build/examples/sana/test_sana_vae --gtest_filter=SanaVAE.GoldenDecodeVsTorch` (tol `KOKKOSDNN_SANA_VAE_TOL`, default 5e-3). Header inventory covers all 196 expected `decoder.*` tensors.

`ctest --test-dir build -R Sana --output-on-failure` stays fast: real-weight forwards are env-gated.

## Performance

Serial CPU build (16 cores idle): tiny forward (4x4, depth-2) ~2 ms; tiny 2-step E2E < 0.1 s; `.pth` inventory parse ~1.7 s; full `.pth` load ~25 s + 600M forward (4x4) ~23 s; 600M forward at 32x32 ~25 min each, x2 for CFG; Gemma-2B FP32 encode = minutes (~5 GB BF16 → ~10 GB FP32 views, ~1.5 TFLOP forward). A full 18-step 1024px run is therefore hours on serial — use accelerated builds for production.
