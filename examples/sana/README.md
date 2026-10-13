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
- `CMakeLists.txt` — `kokkosdnn_sana` lib + `sana_generate` + smoke/sampler/golden tests.
- `Test_SanaSmoke.cpp` — tiny-config random-weight forward smoke test.
- `Test_SanaSampler.cpp` — scheduler math + tiny 2-step Euler loop.
- `Test_SanaGolden.cpp` — checkpoint inventory + opt-in real-weight forward.
- `generate_golden.py` — torch reference script (needs torch + Sana repo).

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
| `--prompt` | `a cyberpunk cat` | logged only (no Gemma bundled; see below) |
| `--prompt-embeds` | — | raw float32 LE Gemma features `[1,1,300,2304]` (`[1,300,2304]`/`[300,2304]` accepted) |
| `--steps` | 18 | FlowEuler steps |
| `--guidance` | 5.0 | CFG scale (`<= 1` disables guidance) |
| `--flow-shift` | 4.0 | scheduler time shift (yaml value) |
| `--seed` | 42 | latent init seed |
| `--out` | `sana_out.png` | preview path (PNG with `KokkosDNN_ENABLE_IMAGE=ON`, else `.ppm`) |
| `--fp32` | — | accepted; float is the only (checkpoint-native) dtype |
| `--vae-safetensors` | — | unsupported stub: errors out (no DC-AE bundled) |

Outputs: raw fp32 latents `<stem>.f32` (`[C,H,W]` row-major) plus a
channel-mean → grayscale preview, nearest-neighbor 32x (1024x1024 at full
resolution). Without `--prompt-embeds`, conditioning falls back to the
checkpoint null block (`y_embedding`, also the unconditional branch), so the
prompt text does not affect the image — pass precomputed Gemma features for
real conditioning.

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

- No Gemma bundled: `--prompt` alone uses the null-embedding fallback;
  real conditioning needs `--prompt-embeds` (dump Gemma-2-2B features with
  the Sana repo, `[:, None][:, :, select_index]` → `[1,1,300,2304]`).
- No DC-AE bundled: no true VAE decode; preview only. `--vae-safetensors`
  is an intentional error stub.
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

## Example commands

```sh
# tiny smoke + 2-step Euler preview (fast, exercises sampler + I/O)
./build/examples/sana/sana_generate --tiny --steps 2
# tiny with explicit seed/guidance (guidance is a no-op on null == null)
./build/examples/sana/sana_generate --tiny --steps 4 --seed 7 --guidance 1.0
# real weights, null conditioning (SLOW on serial CPU, see timing)
./build/examples/sana/sana_generate --weights /path/to/Sana_600M_1024px_MultiLing.pth \
    --steps 18 --guidance 5.0 --seed 42 --out sana_out.png
# real weights + precomputed Gemma features
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

A full 18-step 1024px run is therefore hours on a serial build — the
real-weight paths are functional but intended for accelerated (CUDA/cuBLAS)
builds; `ctest -R Sana` stays fast by env-gating the real-weight forward.
