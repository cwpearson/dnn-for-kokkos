# Sana image-generation example

C++20 Kokkos port of the NVlabs/Sana `SanaMS_600M_P1_D28` text-to-image stack
(Apache-2.0 reference: `diffusion/model/nets/sana*.py`, `sana_blocks.py`,
`basic_modules.py`, `app/sana_pipeline.py`, `diffusion/scheduler/
flow_euler_sampler.py`, config `configs/sana_config/1024ms/
Sana_600M_img1024.yaml`): Linear-Diffusion-Transformer DiT backbone (600M,
1024px, MultiLing) + FlowEuler sampler + `sana_generate` driver.

## Files

- `KokkosDNN_SanaConfig.hpp` — hyperparameters + `validate()` +
  `sana600m_1024()` / `tiny_test()` factories.
- `KokkosDNN_SanaWeights.{hpp,cpp}` — weight views, `random_init()`, and
  `load_torch_pth()` (dependency-free `.pth` reader: minimal ZIP parser +
  pickle key scanner + raw float32 storage mapping, no libtorch).
- `KokkosDNN_SanaPth.hpp` / `SanaWeightsPth.cpp` — full pickle inventory
  reader (`open_pth`: ZIP central directory + protocol-2 pickle VM over the
  closed opcode set torch.save emits; `read_pth_storage` streams one fp32
  record). Used by the golden test for metadata-only validation; the driver
  loads weights via `load_torch_pth` (both agree bit-exactly, see below).
- `KokkosDNN_SanaModel.hpp` — `Model::forward()` (latents + timesteps +
  caption embeddings → velocity prediction), persistent scratch, zero-alloc
  steady state.
- `KokkosDNN_SanaSampler.hpp` — FlowEuler sampler: `build_sigmas()` /
  `build_model_timesteps()`, `guided_predict()` (CFG), `euler_step()`,
  seeded `sample_gaussian()`, and the `run_flow_euler()` device loop.
- `sana_generate.cpp` — sampling driver (flags below; `--help` prints an
  abbreviated usage, this README is the full reference).
- `CMakeLists.txt` — `kokkosdnn_sana` lib + `sana_generate` + smoke/sampler/vae/text-encoder/golden tests.
- `Test_SanaSmoke.cpp` — tiny-config random-weight forward smoke test.
- `Test_SanaSampler.cpp` — scheduler math + tiny 2-step Euler loop.
- `Test_SanaGolden.cpp` — checkpoint inventory + opt-in real-weight forward.
- `generate_golden.py` — torch reference script (needs torch + Sana repo).
- `KokkosDNN_SanaTextEncoder.{hpp,cpp}` — native Gemma-2-2B-IT text
  encoder (FP32, example-scoped, no KV cache): `Config`
  (`gemma2_2b_it()` / `tiny_test()`, `validate()`, HF `config.json`
  parsing incl. `rope_parameters` + `layer_types`), streaming
  F32/F16/BF16→float safetensors loader, SentencePiece-▁ BPE
  `Tokenizer` (`tokenizer.json` loader, same pipeline as the Gemma4
  example), and `Model::forward()` (token ids + padding mask →
  last-hidden states) plus `encode_prompt()` (Sana template/truncation/
  padding + `select_index` gather → `[300,2304]`). Core `gemm`,
  `embedding_gather_scaled`, `gated_gelu_multiply` (GeLU-tanh) and
  `rope_neoX_inplace` are reused; the only new kernels are
  example-local `rmsnorm_one_plus` (Gemma2's `(1+w)` norm, no core
  equivalent) and `attention_causal_softcap` (tanh-50 softcapped causal
  GQA with padding mask — sliding-4096 vs full is exact-identical at
  L=300, so one kernel covers all 26 layers).
- `Test_SanaTextEncoder.cpp` — config/JSON/select-index unit tests,
  tiny random-weight forward smoke (finite + deterministic + pad-mask
  invariance), env-gated real-tokenizer sanity + env-gated transformers
  golden forward.
- `generate_gemma_golden.py` — transformers reference script
  (Gemma2Model last-hidden + select_index → stats JSON; needs torch +
  transformers, never run from ctest).
- `KokkosDNN_SanaVAE.{hpp,cpp}` — native DC-AE f32c32 VAE decoder
  (FP32, example-scoped): `Config` (`dcae_f32c32_sana()` /
  `tiny_test()`, `validate()`), strict F32 safetensors loader
  (diffusers key layout, see below), and `Decoder::decode()`
  (latents `[32,H,W]` → RGB `[3,32H,32W]`). All kernels are
  example-local except 1x1 convs/linears, which reuse
  `KokkosDNN::gemm_bias` (no new core ops).
- `Test_SanaVAE.cpp` — config validation, tiny random-weight decode
  smoke, loader error paths, env-gated diffusers header inventory +
  env-gated torch-reference golden decode.
- `generate_vae_golden.py` — torch reference script (needs torch +
  diffusers): fixed-seed latents → `AutoencoderDC.decode` stats JSON.

## Quick start

```sh
cmake --build build -j8
# smoke: tiny random model, 2 Euler steps, 4x4 latents (no downloads)
./build/examples/sana/sana_generate --tiny --steps 2
# tests (fast; the real-weight forward is env-gated, see below)
ctest -R Sana
# validate the .pth inventory without loading 2.4 GB (needs the checkpoint)
./build/examples/sana/test_sana_golden
```

## Driver flags (`sana_generate`)

| flag | default | meaning |
|---|---|---|
| `--tiny` | off | random-weights smoke model, 4x4 latents |
| `--weights` | `Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth` | torch.save checkpoint |
| `--prompt` | `a cyberpunk cat` | encoded natively with `--text-weights`, else logged only (null fallback) |
| `--prompt-embeds` | — | raw float32 LE Gemma features `[1,1,300,2304]` (`[1,300,2304]`/`[300,2304]` accepted); overrides `--text-weights` |
| `--text-weights` | — | gemma-2-2b-it `.safetensors` file (or directory of shards) for native `--prompt` encoding (`--gemma-weights` alias); needs `tokenizer.json` + `config.json` beside it unless overridden |
| `--tokenizer` | — | `tokenizer.json` override (`--gemma-tokenizer` alias) |
| `--text-config` | — | Gemma `config.json` override |
| `--chi-prompt` | — | complex-human-instruction prefix (Sana `chi_prompt`) |
| `--steps` | 18 | FlowEuler steps |
| `--guidance` | 5.0 | CFG scale (`<= 1` disables guidance) |
| `--flow-shift` | 4.0 | scheduler time shift (yaml value) |
| `--seed` | 42 | latent init seed |
| `--out` | `sana_out.png` | preview path (PNG with `KokkosDNN_ENABLE_IMAGE=ON`, else `.ppm`) |
| `--fp32` | — | accepted; float is the only (checkpoint-native) dtype |
| `--vae-safetensors` | — | DC-AE decoder weights (diffusers layout, see below); without it a grayscale preview is written |

Outputs: raw fp32 latents `<stem>.f32` (`[C,H,W]` row-major) always, plus
either true RGB via the DC-AE decoder (with `--vae-safetensors`:
1024x1024 PNG with `KokkosDNN_ENABLE_IMAGE=ON`, else `.ppm`) or a
channel-mean → grayscale preview, nearest-neighbor 32x (1024x1024 at full
resolution). With `--text-weights`, `--prompt` is encoded by the native
Gemma-2-2B-IT encoder (and the unconditional branch encodes `""` the same
way, matching `sana_pipeline.py`); without it, conditioning falls back to
the checkpoint null block (`y_embedding`, also the unconditional branch),
so the prompt text does not affect the image — pass `--text-weights` or
precomputed `--prompt-embeds` features for real conditioning.

## VAE decoder (DC-AE f32c32)

Checkpoint (exact file the driver accepts):

- HF repo: `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`
- file: `diffusion_pytorch_model.safetensors` (F32)
- config: `in_channels=3`, `latent_channels=32`,
  `decoder_block_out_channels=[128,256,512,512,1024,1024]`,
  `decoder_layers_per_block=[3,3,3,3,3,3]`,
  `decoder_block_types=[ResBlock x3, EfficientViTBlock x3]`,
  `decoder_qkv_multiscales=[(),(),(),(5,),(5,),(5,)]`,
  `upsample_block_type="interpolate"`, `decoder_norm_types="rms_norm"`,
  `decoder_act_fns="silu"`, `decoder_in_shortcut=True`,
  `decoder_conv_act_fn="relu"`, `scaling_factor=0.41407`.

Findings (verified against the live checkpoint header + diffusers
`autoencoder_dc.py` @ main, i.e. PR #9708, and NVlabs/Sana
`diffusion/model/builder.py::vae_decode` + `app/sana_pipeline.py`):

- The Sana training recipe uses the efficientvit DC-AE
  (`mit-han-lab/dc-ae-f32c32-sana-1.0`, arch
  `dc_ae_f32c32("dc-ae-f32c32-sana-1.0")`: 6 stages, widths
  `[128,256,512,512,1024,1024]`, depths `[3,3,3,3,3,3]`,
  `[ResBlock x3, EViTS5_GLU x3]`, `InterpolateConv` upsampling, `trms2d`
  norm, SiLU, `scaling_factor=0.41407`); the diffusers port keeps the
  math and renames keys (`decoder.project_in` →
  `decoder.conv_in`, `decoder.stages.i` → `decoder.up_blocks.i`,
  `decoder.project_out` → `decoder.norm_out` + `decoder.conv_out`).
  The raw efficientvit `model.safetensors` is NOT accepted (different
  key names); use the `-diffusers` file above.
- Decode path: `latent / 0.41407` → `AutoencoderDC.decode` (no tiling
  needed at 1024px) → sample in `[-1,1]` → `x*0.5+0.5` clamped to u8.
  This driver implements exactly that (`to_u8_rgb`).
- Decoder dataflow: `conv_in` 3x3 (32→1024) + channel-repeat shortcut →
  3 blocks/stage from 1024ch@32x32 down to 128ch@1024x1024 (stages 5..0,
  one nearest-2x + conv3x3 + repeat/pixel-shuffle shortcut upsample per
  boundary) → RMSNorm(128, eps 1e-5, weight+bias) → ReLU →
  conv3x3 (128→3). ResBlock = conv3x3+bias → SiLU → conv3x3 (no bias) →
  RMSNorm(w+b) → +residual. EfficientViTBlock = Sana multiscale linear
  attention (ReLU kernel, fp32, eps 1e-15; heads=C/32, head_dim 32,
  one 5x5 multiscale branch concatenated as 2x heads; quadratic form
  when H*W ≤ 32) + GLUMBConv (expand 4, SiLU, depthwise 3x3, SiLU gate,
  point 1x1, RMSNorm) with a residual around each.
- The driver E2E contract matches `sana_pipeline.py`: FlowEuler latents
  (this example) → `vae_decode("dc-ae", ...)` (this decoder) → u8 RGB.
- Note the 1.1 refresh (`dc-ae-f32c32-sana-1.1`, improved decoder) is a
  different checkpoint with the same layout; the loader validates exact
  shapes, so it loads whenever shapes match, but goldens target 1.0.

```sh
# decode real latents to 1024x1024 RGB (Serial CPU: slow, ~1h, correctness first)
./build/examples/sana/sana_generate --weights .../Sana_600M_1024px_MultiLing.pth \
    --prompt-embeds feats.f32 --steps 18 --out sana_out.png \
    --vae-safetensors /path/to/diffusion_pytorch_model.safetensors
```

## Verifying the VAE decoder against the torch reference

1. `python3 examples/sana/generate_vae_golden.py
   --repo mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers --out vae_golden.json`
   — fixed-seed 32ch 4x4 latents → fp32 CPU `AutoencoderDC.decode`
   (divides by 0.41407 first), dumps input latents + mean/std/min/max/first8.
2. `KOKKOSDNN_SANA_VAE_SAFETENSORS=/path/to/diffusion_pytorch_model.safetensors
   KOKKOSDNN_SANA_VAE_GOLDEN_JSON=vae_golden.json ./build/examples/sana/test_sana_vae
   --gtest_filter=SanaVAE.GoldenDecodeVsTorch` — loads the same file,
   decodes the same latents (4x4 → 128x128 RGB), compares stats
   (tolerance `KOKKOSDNN_SANA_VAE_TOL`, default 5e-3).
3. Header-only inventory (fast, no 0.5 GB weight load):
   `KOKKOSDNN_SANA_VAE_SAFETENSORS=... ./build/examples/sana/test_sana_vae
   --gtest_filter=SanaVAE.HeaderInventoryMatchesDiffusers` — every one of
   the 196 expected `decoder.*` tensors must be present with exact shapes.

## Weights

Three checkpoints make up the full Sana-0.6B stack. The DiT backbone is
already in-tree; the text encoder and VAE live in repo-root snapshot dirs
(same convention as `dinov3-vitb16-pretrain-lvd1689m/`,
`gemma-4-E2B-it/` — full snapshots, never committed):

| checkpoint | HF repo | local dir | status |
|---|---|---|---|
| Sana DiT 600M 1024px MultiLing (`.pth`, 2374659028 B) | `Efficient-Large-Model/Sana_600M_1024px` | `Sana_600M_1024px/` | in-tree |
| Gemma-2-2B-IT text encoder (BF16, 2 shards) | `google/gemma-2-2b-it` (gated) | `gemma-2-2b-it/` | NOT fetched — needs `$HF_TOKEN` (see below) |
| DC-AE f32c32 VAE (`model.safetensors`, 1249046372 B, F32) | `mit-han-lab/dc-ae-f32c32-sana-1.0` | `dc-ae-f32c32-sana-1.0/` | fetched |
| DC-AE f32c32 diffusers repackaging (`diffusion_pytorch_model.safetensors`, 1249044836 B, F32) | `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers` | `dc-ae-f32c32-sana-1.0-diffusers/` | fetched (accepted by `--vae-safetensors`) |

### Gemma-2-2B-IT text encoder (gated)

`google/gemma-2-2b-it` is gated — accept the Gemma Terms of Use at
<https://huggingface.co/google/gemma-2-2b-it> once (logged-in account),
mint a token with access to the repo, export it as `HF_TOKEN`, then
download with the `hf` CLI (`pip install -U huggingface_hub`):

```bash
hf download google/gemma-2-2b-it \
  --local-dir ./gemma-2-2b-it \
  --token "$HF_TOKEN"
ls ./gemma-2-2b-it
# expect model-00001-of-00002.safetensors + model-00002-of-00002.safetensors
# (~5 GB total, BF16) + model.safetensors.index.json + config.json +
# tokenizer.json + tokenizer.model + tokenizer_config.json
```

Without a granted token the Hub returns `401 GatedRepoError` (verified —
no `HF_TOKEN` was available in this environment, so the dir above does
not exist yet; the sibling file list is public, file contents are not).
Shard byte sizes are therefore unverified — confirm with `ls` after
downloading. Keep the full snapshot: `config.json` defines the arch and
`tokenizer.json` / `tokenizer.model` define the SentencePiece vocabulary,
both needed by the encoder loader / golden test.

Upstream arch (implemented in `KokkosDNN_SanaTextEncoder`; re-verify
`config.json` values against these after download — the loader parses and
validates them at runtime): `model_type` gemma2, hidden 2304 (= Sana
`caption_channels`), 26 layers, 8 Q heads / 4 KV heads (GQA, `head_dim`
256), MLP 9216 (`gelu_pytorch_tanh`), vocab 256000, sliding window 4096
with alternating sliding/full layers (layer 0 sliding),
`rms_norm_eps` 1e-6, `query_pre_attn_scalar` 256, logit softcapping 50
(attention) / 30 (final, unused — the encoder emits hidden states, not
logits), `tie_word_embeddings` true, RoPE theta 10000 (NeoX-style, all
layers), unbiased projections, `(1+w)` RMSNorms (4/layer + final), scaled
embedding (×48). Sana conditions on the encoder's last-hidden-state
features (`[300,2304]`, dumped to the `--prompt-embeds` raw-f32 format
`[1,1,300,2304]`). Differences vs Gemma4-E2B (`examples/gemma4`): no PLE,
no KV sharing, no double-wide MLP, no QK norms, no partial RoPE,
attention scale 1/√256 (not 1.0), single RoPE theta.

### DC-AE f32c32 VAE decoder (public)

The Sana VAE is the MIT HAN Lab DC-AE f32c32 checkpoint trained for Sana
([project page](https://hanlab.mit.edu/projects/dc-ae)): 32x spatial
compression (1024px → 32x32 latents, 32 channels — matching the DiT's
`in_channels=32` patch grid), `scaling_factor` 0.41407, 312.25M params,
all F32. Two HF repos carry the same weights under different tensor
naming — the native one (loaded by NVlabs/Sana via `DCAE_HF`) and the
diffusers port (diffusers PR #9708, `AutoencoderDC`):

```bash
hf download mit-han-lab/dc-ae-f32c32-sana-1.0 \
  --local-dir ./dc-ae-f32c32-sana-1.0
hf download mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
  --local-dir ./dc-ae-f32c32-sana-1.0-diffusers
ls ./dc-ae-f32c32-sana-1.0 ./dc-ae-f32c32-sana-1.0-diffusers
# expect model.safetensors (1249046372 B) + config.json in the former,
# diffusion_pytorch_model.safetensors (1249044836 B) + config.json
# (AutoencoderDC arch fields) in the latter; doc/assets files optional
```

Header-only inventory (no weights loaded; both files, same 312.25M
params, F32 throughout):

- native `model.safetensors`: 339 tensors — `decoder.*` ×178
  (efficientvit naming: `decoder.project_in`,
  `decoder.stages.{0..5}...`, `decoder.project_out`), `encoder.*` ×161.
- diffusers `diffusion_pytorch_model.safetensors`: 375 tensors —
  `decoder.*` ×196 (diffusers naming: `decoder.conv_in/conv_out/norm_out`,
  `decoder.up_blocks.{0..5}...`), `encoder.*` ×179. Same values,
  repackaged (split norms/blocks), so tensor counts differ.

The diffusers `config.json` is the arch reference (`latent_channels`
32, 6 decoder stages 128→1024, `ResBlock` ×3 then `EfficientViTBlock`
×3, `rms_norm`, SiLU); the native `config.json` is just
`{"model_name": "dc-ae-f32c32-sana-1.0"}`.

### Driver flag mapping

How the three checkpoints map to driver flags (all wired):

- `--weights` → DiT `.pth` (wired today;
  `Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth`).
- `--prompt-embeds` → precomputed Gemma `[1,1,300,2304]` raw f32 (wired;
  still the highest-precedence caption source).
- `--text-weights` (`--gemma-weights` alias) → gemma-2-2b-it checkpoint
  file or shard directory, wired (native encoder in-driver; needs
  `tokenizer.json` + `config.json` beside it unless `--tokenizer` /
  `--text-config` override). `--chi-prompt` carries an optional Sana
  `chi_prompt` prefix.
- `--vae-safetensors` → DC-AE decoder weights, wired (see "VAE decoder
  (DC-AE f32c32)" above): point it at the **diffusers** file
  (`...-diffusers/diffusion_pytorch_model.safetensors`). The native
  `dc-ae-f32c32-sana-1.0/model.safetensors` carries the same values under
  efficientvit key names and is NOT accepted by the loader.

## Architecture notes (verified against the reference)

- Patch-1 Conv1x1 latent embed: 32ch DC-AE latents on a 32x32 grid
  (1024px / downsample 32) → 1024 tokens of dim 1152. No pos_embed is added
  (`use_pe=false`; the stored `pos_embed` record is all zeros).
- Timestep: integer truncation (`SanaMS` `.long()`), `[cos,sin]` freq embed
  (dim 256, period 10000) → MLP(256→1152→1152, SiLU) → global
  `t_block` SiLU+Linear(1152→6912) shared by all blocks.
- Per block: `(scale_shift_table[6,D] + t0).chunk(6)` →
  shift/scale/gate × (attn, mlp); LayerNorm eps 1e-6, affine off.
- Self-attention: LiteLA linear attention (ReLU kernel, fp32, eps 1e-8,
  36 heads × dim 32, fused unbiased QKV, biased proj). No qk-norm/RoPE.
- Cross-attention: softmax SDPA over the RMSNorm'd caption (16 heads × 72,
  all projections biased; K/V split into separate GEMMs from the fused
  `[2D,D]` weight — row ranges are contiguous, so the head views stay
  dense).
- MLP: GLUMBConv (inverted 1x1 D→5760 +SiLU, depthwise 3x3 groups=5760,
  SiLU-gated GLU halve to 2880, point 1x1 →1152 unbiased).
- Caption: `[300,2304]` → fc1→GELU-tanh→fc2 → RMSNorm (eps 1e-5, learned
  weight ~0.01 scale).
- Final: AdaLN (`[2,1152]` table + t) → Linear(1152→32) → unpatchify.
- Sampler (`KokkosDNN_SanaSampler.hpp`): shifted sigma schedule
  `sig = shift·u/(1+(shift−1)·u)` over `linspace(1000, 1000·σ_min, steps)`
  with terminal 0 (diffusers `FlowMatchEulerDiscreteScheduler`, shift 4.0);
  `v = uncond + g·(cond − uncond)`; `x += (σ_{i+1} − σ_i)·v`; `x_T ∼ N(0,1)`
  via portable mt19937 Box-Muller.

## Limitations

- Native Gemma conditioning needs `--text-weights` (gated HF download,
  see Weights); `--prompt` alone still uses the null-embedding fallback.
  `--prompt-embeds` (dump Gemma-2-2B features with the Sana repo,
  `[:, None][:, :, select_index]` → `[1,1,300,2304]`) overrides the
  native encoder when both are given.
- No DC-AE bundled: true VAE decode needs `--vae-safetensors` pointing at
  the diffusers-layout file (see above); otherwise a channel-mean preview
  only. Serial-CPU full-res decode is slow (correctness first); CUDA builds
  parallelize the same kernels.
- No PAG (perturbed-attention guidance): Sana's FlowEuler path is plain
  CFG; PAG exists only on the DPM path.
- No caption padding-mask trimming (the xformers `masked_select` path):
  all 300 caption tokens are attended.
- Fixed 1024 geometry (32x32 latents, patch 1); `--tiny` uses 4x4 latents.
- Float32 only (checkpoint dtype); no bf16 path.

## Verifying the DiT forward against the torch reference

1. `python3 examples/sana/generate_golden.py --repo <Sana> --weights <pth>
   --out golden.json` — single fp32 CPU forward (seed 1234, t=1000, null
   caption), dumps mean/std/min/max/first8.
2. `KOKKOSDNN_SANA_WEIGHTS=<pth> KOKKOSDNN_SANA_GOLDEN_JSON=golden.json
   ./build/examples/sana/test_sana_golden --gtest_filter=
   SanaGolden.RealWeightsForwardFp32` — loads the same `.pth`, runs the same
   forward, compares mean/std (tolerance `KOKKOSDNN_SANA_TOL`, default
   2e-3; without the JSON it checks finite + prints).
3. Loader cross-check (no torch needed): both bundled readers agree —
   `load_torch_pth` and `open_pth`/`read_pth_storage` produce bit-identical
   tensors, and `y_embedding` stats match an independent pure-Python parse
   (mean 0.099986, std 2.773899 over 691200 floats).

## Verifying the text encoder against the torch reference

1. `python3 examples/sana/generate_gemma_golden.py --model
   google/gemma-2-2b-it --prompt "a cyberpunk cat" --out gemma_golden.json`
   — BF16 CPU Gemma2Model last-hidden + `select_index` gather, dumps
   shape/mean/std/min/max/first8 of the `[1,300,2304]` features.
2. `KOKKOSDNN_SANA_GEMMA_WEIGHTS=gemma-2-2b-it
   KOKKOSDNN_SANA_GEMMA_GOLDEN_JSON=gemma_golden.json
   ./build/examples/sana/test_sana_text_encoder --gtest_filter=
   SanaTextEncoder.GoldenVsTransformers` — loads the checkpoint
   (file or shard dir; `config.json`/`tokenizer.json` siblings unless
   `KOKKOSDNN_SANA_GEMMA_CONFIG` / `KOKKOSDNN_SANA_GEMMA_TOKENIZER`
   override), encodes the same prompt, compares mean/std (tolerance
   `KOKKOSDNN_SANA_GEMMA_TOL`, default 5e-2 — the torch reference runs
   in BF16 while this encoder is FP32).
3. Weight-free cross-checks (no torch needed): the C++ forward matches
   an independent pure-NumPy Gemma2 implementation bit-nearly (max abs
   diff 9.5e-7 over `[12,32]` on random tiny weights), and the
   Sana tokenizer is encode/decode-identical to the Gemma4 example
   tokenizer on the same `tokenizer.json`.

## Example commands

```sh
# tiny smoke + 2-step Euler preview (fast, exercises sampler + I/O)
./build/examples/sana/sana_generate --tiny --steps 2
# tiny with explicit seed/guidance (guidance is a no-op on null == null)
./build/examples/sana/sana_generate --tiny --steps 4 --seed 7 --guidance 1.0
# real weights, null conditioning (SLOW on serial CPU, see timing)
./build/examples/sana/sana_generate --weights /path/to/Sana_600M_1024px_MultiLing.pth \
    --steps 18 --guidance 5.0 --seed 42 --out sana_out.png
# real weights + native Gemma-2-2B-IT prompt encoding (SLOW, see timing)
./build/examples/sana/sana_generate --weights /path/to/...pth \
    --text-weights /path/to/gemma-2-2b-it --prompt "a cyberpunk cat" --steps 18
# real weights + precomputed Gemma features (overrides --text-weights)
./build/examples/sana/sana_generate --weights /path/to/...pth \
    --prompt-embeds feats.f32 --prompt "a cyberpunk cat" --steps 18
# inventory check against the real archive (fast, ~2 s)
./build/examples/sana/test_sana_golden --gtest_filter=SanaGolden.PthInventoryMatchesRelease
```

## Measured timing (serial CPU build, 16 cores idle)

| workload | time |
|---|---|
| tiny forward (4x4, depth-2 random) | ~2 ms |
| tiny 2-step Euler end-to-end | < 0.1 s |
| `.pth` inventory parse (91 KiB pickle) | ~1.7 s |
| full `.pth` load (2.4 GB stream) + 600M forward, 4x4 latents | ~25 s load, ~23 s forward |
| 600M forward, 32x32 latents (extrapolated, linears ∝ tokens) | ~25 min each; ×2 for CFG |
| Gemma-2B text encode, 300 tokens FP32 serial (weight load + forward) | minutes (streams the ~5 GB BF16 checkpoint into ~10 GB FP32 views; forward ≈ 1.5 TFLOP) — env-gated, correctness first |

A full 18-step 1024px run is therefore hours on a serial build — the
real-weight paths are functional but intended for accelerated (CUDA/cuBLAS)
builds; `ctest -R Sana` stays fast by env-gating the real-weight forward.
