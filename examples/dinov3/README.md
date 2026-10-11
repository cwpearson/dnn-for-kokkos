# DINOv3 example (placeholder — no code yet)

Planned: ViT-B/16 (`vitb16_lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`)
image-encoder inference built on the core library.

Target arch (per HF `dinov3_vit` modeling/config + convert script):
patch 16, hidden 768, 12 layers, 12 heads (`head_dim=64`, MHA),
MLP 3072 with `gelu` (no SwiGLU/gating), 1 CLS + 4 register tokens,
biased `q/v/o/up/down` linears (`k` unbiased), LayerNorm `eps=1e-5`,
2D axial RoPE (`theta=100`), per-block LayerScale, final LayerNorm.
Single forward over `1+4+Np` tokens (201 @224px); no KV-cache, no sampling.

This directory currently holds only this TODO list. Not wired into
`examples/CMakeLists.txt`, so the build is unaffected.

## TODO

Core `src/` ops (all missing today):

- [x] `layernorm_forward` (weight+bias, FP32 accum) — `RMSNorm` is not a substitute. Done: `src/KokkosDNN_LayerNorm.hpp` (`KokkosDNN::layernorm_forward`/`layernorm_row`).
- [x] `gelu_forward` — pin HF `ACT2FN["gelu"]` erf-vs-tanh variant first. Done: `src/KokkosDNN_Activation.hpp` (`KokkosDNN::gelu_forward`, `GeluVariant::ExactErf` default = HF `gelu`, `Tanh` = `gelu_new`).
- [x] Biased linear for inference (`gemm_bias`: `Y=X·Wᵀ+b`, reuse cuBLAS path). Done: `src/KokkosDNN_GemmBias.hpp` (`KokkosDNN::gemm_bias`/`gemv_bias`, empty bias = unbiased).
- [x] Bidirectional full MHA (non-causal; current `attention_gqa_single` is causal-only; cuDNN graph is causal-only too). Done: `src/KokkosDNN_Attention.hpp` (`KokkosDNN::attention_mha_bidirectional`, native Kokkos, FP32 streaming softmax).
- [ ] 2D axial RoPE (`head_dim/4` inv_freq, dynamic H/W coords, `rotate_half`, patch-tokens-only, prefix skip).
- [ ] PatchEmbed Conv2d (`3→768`, `k=16/s=16`) or `im2col+gemm` helper.
- [ ] Token assembly + batching (`cat[cls,registers,patches]`, `[B,T,D]` policy).
- [ ] `layerscale` broadcast multiply + residual-add helper (`DropPath` is inference identity).

Example support (`examples/dinov3/`, all missing):

- [ ] `Config` for `dinov3_vit` fields (patch/hidden/layers/heads/registers/bias flags/layerscale/rope/image size).
- [ ] Safetensors loader map (`patch_embed/cls_token/register_tokens/norms/qkv-o/mlp/layerscale/final-norm` HF names; Llama loader names do not apply).
- [x] Image preprocessing (resize/crop, normalize, `HWC→CHW`, batching; no image I/O in repo today). Done: `src/images/` (`KokkosDNN::Image`, stb-backed I/O, opt-in `-DKokkosDNN_ENABLE_IMAGE=ON`, OFF by default).
- [ ] Encoder driver (`dinov3_forward(pixel) → cls/patch tokens`) + tiny-random smoke path (no download).
- [ ] Golden test vs HF `facebook/dinov3-vitb16-pretrain-lvd1689m` (CLS + patch tokens, tolerance per scalar).
- [ ] Wire into `examples/CMakeLists.txt` + driver flags (`--weights/--image/--fp32`) once the above land.
