# DINOv3 example

ViT-B/16 (`vitb16_lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`)
image-encoder inference built on the core library.

Target arch (per HF `dinov3_vit` modeling/config + convert script):
patch 16, hidden 768, 12 layers, 12 heads (`head_dim=64`, MHA),
MLP 3072 with `gelu` (no SwiGLU/gating), 1 CLS + 4 register tokens,
biased `q/v/o/up/down` linears (`k` unbiased), LayerNorm `eps=1e-5`,
2D axial RoPE (`theta=100`), per-block LayerScale, final LayerNorm.
Single forward over `1+4+Np` tokens (201 @224px); no KV-cache, no sampling.

This directory holds the Config/Weights/Model/Driver plus smoke and
golden tests, wired into `examples/CMakeLists.txt` via
`examples/dinov3/CMakeLists.txt`.

## Weights

`facebook/dinov3-vitb16-pretrain-lvd1689m` is gated — accept the
conditions at <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
once, then download with the `hf` CLI (`pip install -U huggingface_hub`):

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir ./weights/dinov3-vitb16-pretrain-lvd1689m \
  --token "$HF_TOKEN"
ls ./weights/dinov3-vitb16-pretrain-lvd1689m
# expect model.safetensors (~343 MB) + config.json + preprocessor_config.json
```

Keep the full snapshot: `config.json` defines the arch and
`preprocessor_config.json` defines the image normalization, both needed
by the future driver / golden test.

## TODO

Core `src/` ops (all missing today):

- [x] `layernorm_forward` (weight+bias, FP32 accum) — `RMSNorm` is not a substitute. Done: `src/KokkosDNN_LayerNorm.hpp` (`KokkosDNN::layernorm_forward`/`layernorm_row`).
- [x] `gelu_forward` — pin HF `ACT2FN["gelu"]` erf-vs-tanh variant first. Done: `src/KokkosDNN_Activation.hpp` (`KokkosDNN::gelu_forward`, `GeluVariant::ExactErf` default = HF `gelu`, `Tanh` = `gelu_new`).
- [x] Biased linear for inference (`gemm_bias`: `Y=X·Wᵀ+b`, reuse cuBLAS path). Done: `src/KokkosDNN_GemmBias.hpp` (`KokkosDNN::gemm_bias`/`gemv_bias`, empty bias = unbiased).
- [x] Bidirectional full MHA (non-causal; current `attention_gqa_single` is causal-only; cuDNN graph is causal-only too). Done: `src/KokkosDNN_Attention.hpp` (`KokkosDNN::attention_mha_bidirectional`, native Kokkos, FP32 streaming softmax).
- [x] 2D axial RoPE (`head_dim/4` inv_freq, dynamic H/W coords, `rotate_half`, patch-tokens-only, prefix skip). Done: `src/KokkosDNN_RoPE.hpp` (`KokkosDNN::rope2d_axial_forward`/`_inplace`, theta=100, prefix_skip=5).
- [x] PatchEmbed Conv2d (`3→768`, `k=16/s=16`) or `im2col+gemm` helper. Done: `src/KokkosDNN_PatchEmbed.hpp` (`KokkosDNN::patchembed_forward`, `im2col+gemm_bias`).
- [x] Token assembly + batching (`cat[cls,registers,patches]`, `[B,T,D]` policy). Done: `src/KokkosDNN_Tokens.hpp` (`KokkosDNN::tokens_assemble_forward`, broadcast + single-seq).
- [x] `layerscale` broadcast multiply + residual-add helper (`DropPath` is inference identity). Done: `src/KokkosDNN_LayerScale.hpp` (`KokkosDNN::layerscale_residual_forward`/`layerscale_forward`).

Example support (`examples/dinov3/`):

- [x] `Config` for `dinov3_vit` fields (patch/hidden/layers/heads/registers/bias flags/layerscale/rope/image size). Done: `KokkosDNN_Dinov3Config.hpp` (`KokkosDNN::Dinov3::Config`: `vitb16()`/`tiny_test()`/`validate()`/`from_json_file`/`from_json_string`/`num_tokens()`/`num_patch_tokens()`, dependency-free JSON overlay).
- [x] Safetensors loader map (`patch_embed/cls_token/register_tokens/norms/qkv-o/mlp/layerscale/final-norm` HF names; Llama loader names do not apply). Done: `KokkosDNN_Dinov3Weights.{hpp,cpp}` (`Weights<Exec,Scalar>::allocate`, `random_init`, `load_safetensors` with F32/F16/BF16 conversion, mmap-backed reads; k projection unbiased, no `bk` field).
- [x] Image preprocessing (resize/crop, normalize, `HWC→CHW`, batching; no image I/O in repo today). Done: `src/images/` (`KokkosDNN::Image`, stb-backed I/O, opt-in `-DKokkosDNN_ENABLE_IMAGE=ON`, OFF by default).
- [x] Encoder driver (`dinov3_forward(pixel) → cls/patch tokens`) + tiny-random smoke path (no download). Done: `KokkosDNN_Dinov3Model.hpp` (`Model<Exec,Scalar>::forward(pixels [C,H,W]) → tokens [T,D]`, row 0 = CLS, `cls_token()`/`patch_tokens()` aliases) + `dinov3_encode.cpp` driver (`--tiny` default random path, `--weights/--config/--image/--fp32/--print-tokens`) + `Test_Dinov3Smoke.cpp` (shape/finite/determinism + Config JSON round-trip, no image component needed).
- [x] Golden test vs HF `facebook/dinov3-vitb16-pretrain-lvd1689m` (CLS + patch tokens, tolerance per scalar). Done: `generate_golden.py` (HF reference: full CLS [768] + first 4 patch rows + patch mean + reference pixel tensor) + `Test_Dinov3Golden.cpp` (FP32 max-abs-diff ≤ 1e-3 default, `KOKKOSDNN_DINOV3_TOL` override; SKIPs without files, never downloads). See "Golden test" below.
- [x] Wire into `examples/CMakeLists.txt` + driver flags (`--weights/--image/--fp32`) once the above land. Done: `examples/dinov3/CMakeLists.txt` (static lib `kokkosdnn_dinov3`, alias `KokkosDNN::dinov3`; exe `dinov3_encode`; `test_dinov3_smoke` + `test_dinov3_golden` via `gtest_discover_tests`) + `add_subdirectory(dinov3)` in `examples/CMakeLists.txt`. Driver flags: `--tiny`, `--config config.json`, `--weights model.safetensors [...]`, `--image img.jpg` (needs `-DKokkosDNN_ENABLE_IMAGE=ON`, else synthetic with a warning), `--fp32`, `--print-tokens N`.

## Golden test

Generate the reference (read-only on the weights snapshot; needs
`torch` + `transformers`):

```bash
python examples/dinov3/generate_golden.py \
  --weights-dir ./weights/dinov3-vitb16-pretrain-lvd1689m \
  --synthetic --out /tmp/golden_dinov3.json
# or: --image /path/to/img.jpg  (HF-processed pixels stored verbatim)
```

Run the C++ check (never downloads; SKIPs gracefully without the files):

```bash
KOKKOSDNN_DINOV3_WEIGHTS=./weights/dinov3-vitb16-pretrain-lvd1689m/model.safetensors \
KOKKOSDNN_DINOV3_GOLDEN_JSON=/tmp/golden_dinov3.json \
  ./build/examples/dinov3/test_dinov3_golden
# optional overrides:
#   KOKKOSDNN_DINOV3_TOL=2e-3     (default 1e-3 FP32 max-abs-diff)
#   KOKKOSDNN_DINOV3_CONFIG=.../config.json  (overlay onto vitb16())
```

Synthetic inputs use the canonical formula
`v(c,y,x) = ((y*S + x) % 256) / 255 + 0.01 * c`, shared by
`dinov3_encode.cpp`, `Test_Dinov3Smoke.cpp`, `Test_Dinov3Golden.cpp` and
`generate_golden.py --synthetic`, fed directly as `pixel_values` (no HF
processor normalization) on both sides.
