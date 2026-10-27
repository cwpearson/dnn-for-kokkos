# DINOv3 example

ViT-B/16 (`vitb16_lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`) image-encoder inference built on the core library. The `dinov3_encode` driver encodes one image (or deterministic synthetic pixels) into `[T, D]` tokens where row 0 is CLS.

## Quickstart (0 to working)

```bash
# From the repo root (/workspace). Real-image input needs KokkosDNN_ENABLE_IMAGE=ON;
# without it --image is ignored (with a warning) and synthetic pixels are used.
# Accept the gated-repo conditions once at
# https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m, then:

# 1. Build and install Kokkos once (Serial shown; for GPU add
#    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON and set
#    CMAKE_CXX_COMPILER to kokkos-src/bin/nvcc_wrapper)
git clone --branch 5.2.1 --recursive https://github.com/kokkos/kokkos.git kokkos-src
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-install \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build kokkos-build --parallel 8
cmake --install kokkos-build

# 2. Configure and build the example target
cmake -S . -B build \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkosDNN_ENABLE_IMAGE=ON
cmake --build build --parallel 8 --target dinov3_encode

# 3. Download the checkpoint with the `hf` CLI
pip install -U "huggingface_hub[cli]"  # plus torch transformers safetensors Pillow for the golden test

hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir ./weights/dinov3-vitb16-pretrain-lvd1689m \
  --token "$HF_TOKEN"
ls ./weights/dinov3-vitb16-pretrain-lvd1689m
# expect model.safetensors (~343 MB) + config.json + preprocessor_config.json

# Smoke path, no download needed:
./build/examples/dinov3/dinov3_encode --tiny

# Real weights on synthetic pixels:
./build/examples/dinov3/dinov3_encode \
  --weights ./weights/dinov3-vitb16-pretrain-lvd1689m/model.safetensors \
  --config ./weights/dinov3-vitb16-pretrain-lvd1689m/config.json --fp32

# Real weights on a real image (needs the IMAGE=ON build above):
./build/examples/dinov3/dinov3_encode \
  --weights ./weights/dinov3-vitb16-pretrain-lvd1689m/model.safetensors \
  --config ./weights/dinov3-vitb16-pretrain-lvd1689m/config.json \
  --image /path/to/img.jpg --fp32
```

## Example commands

```bash
# Tiny random model (default when no --weights): tiny_test() config + random weights.
./build/examples/dinov3/dinov3_encode --tiny

# Real checkpoint, synthetic gradient pixels (no image file needed).
./build/examples/dinov3/dinov3_encode \
  --weights ./weights/dinov3-vitb16-pretrain-lvd1689m/model.safetensors \
  --config ./weights/dinov3-vitb16-pretrain-lvd1689m/config.json

# Real checkpoint + real image (needs -DKokkosDNN_ENABLE_IMAGE=ON).
./build/examples/dinov3/dinov3_encode \
  --weights ./weights/dinov3-vitb16-pretrain-lvd1689m/model.safetensors \
  --config ./weights/dinov3-vitb16-pretrain-lvd1689m/config.json \
  --image /path/to/img.jpg --print-tokens 8

# Flags: --tiny | --config config.json (overlays HF config.json onto the
# tiny_test()/vitb16() base) | --weights model.safetensors [...] (multi-shard
# OK) | --image img.jpg | --fp32 (DefaultScalar is already float, so currently
# a no-op) | --print-tokens N (prints first N CLS values, default 8).
```

## Notes

Target arch (per HF `dinov3_vit` modeling/config): patch 16, hidden 768, 12 layers, 12 heads (`head_dim=64`, MHA), MLP 3072 with exact-erf `gelu`, 1 CLS + 4 register tokens, biased `q/v/o/up/down` linears (`k` unbiased), LayerNorm `eps=1e-5`, 2D axial RoPE (`theta=100`, patch-tokens-only, 5-token prefix skip), per-block LayerScale, final LayerNorm. This example layers thin support code over existing core ops (`layernorm_forward`, `gelu_forward`, `gemm_bias`/`gemv_bias`, bidirectional `attention_mha_bidirectional`, `rope2d_axial_forward`, `patchembed_forward` via `im2col+gemm_bias`, token assembly, `layerscale_residual_forward`): `KokkosDNN_Dinov3Config.hpp` (`Config::vitb16()`/`tiny_test()`/`validate()`/`from_json_file`), `KokkosDNN_Dinov3Weights.{hpp,cpp}` (safetensors loader with F32/F16/BF16 conversion), `KokkosDNN_Dinov3Model.hpp` (`forward(pixels [C,H,W]) -> tokens [T,D]`, row 0 = CLS), and the `dinov3_encode.cpp` driver. Image I/O + preprocessing lives in the opt-in `src/images/` component (`-DKokkosDNN_ENABLE_IMAGE=ON`, OFF by default); without it the driver falls back to deterministic synthetic pixels `v(c,y,x) = ((y*S + x) % 256) / 255 + 0.01 * c`, shared bit-identically with the smoke and golden tests.

## Verification

No-download smoke test (shape `[1+R+Np, D]`, all-finite outputs, bit-for-bit determinism across two forwards, plus a Config JSON round-trip; runs without the image component):

```bash
ctest --test-dir build -R Dinov3 --output-on-failure
./build/examples/dinov3/test_dinov3_smoke
```

Golden test against HF `transformers` on `facebook/dinov3-vitb16-pretrain-lvd1689m` (FP32 CLS vector + first 4 patch rows + patch mean; the C++ side never downloads and SKIPs gracefully without the files):

```bash
python examples/dinov3/generate_golden.py \
  --weights-dir ./weights/dinov3-vitb16-pretrain-lvd1689m \
  --synthetic --out /tmp/golden_dinov3.json
# or: --image /path/to/img.jpg  (HF-processed pixels stored verbatim)

KOKKOSDNN_DINOV3_WEIGHTS=./weights/dinov3-vitb16-pretrain-lvd1689m/model.safetensors \
KOKKOSDNN_DINOV3_GOLDEN_JSON=/tmp/golden_dinov3.json \
  ./build/examples/dinov3/test_dinov3_golden
# optional overrides:
#   KOKKOSDNN_DINOV3_TOL=2e-3     (default 1e-3 FP32 max-abs-diff)
#   KOKKOSDNN_DINOV3_CONFIG=.../config.json  (overlay onto vitb16())
```

Synthetic inputs are fed directly as `pixel_values` with no HF processor normalization on either side, so inputs match by construction (the C++ side additionally cross-checks its recomputation against the stored pixel tensor to `<= 1e-6`); `--image` goldens store the HF-preprocessed tensor verbatim so both sides compare on exactly the same floats.

## Performance

One forward pass covers all `1+4+Np` tokens (201 @224px: 1 CLS + 4 registers + 196 patches); there is no KV-cache, batching, or sampling. Compute defaults to float (`DefaultScalar`), and the golden file keeps only the full CLS `[768]`, the first 4 patch rows, and the patch mean instead of all 196x768 patch tokens to stay small (< ~5 MB).
