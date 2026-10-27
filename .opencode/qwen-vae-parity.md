# Qwen-Image-2.1-Turbo — VAE decode vs torch/diffusers reference

Status: **analysis + verification plan** (read-only: no source edits, no commits, no GPU jobs run).
Date: 2026-10-24. Line numbers refer to the tree state at analysis time; the worktree is being
edited concurrently by another workstream, so anchors may drift by a few lines — they are greppable
by the quoted symbol names.

Sides compared:

| tag | file |
| --- | --- |
| **native** | `models/qwen_image_21/KokkosDNN_QwenImage21Vae.hpp` (this repo) + decode call in `examples/qwen_image_21/qwen_image_21_generate.cpp:1773-1848` |
| **torch VAE** | `/home/appuser/.local/lib/python3.13/site-packages/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py` (= `autoencoder_kl_qwenimage21.py` below) |
| **torch pipeline** | `.../diffusers/modular_pipelines/qwenimage21/decoders.py` (unpack + affine + decode + postprocess), `.../diffusers/image_processor.py` |
| **golden** | `scripts/qwen_image_21_golden.py` → `/workspace/weights/Qwen-Image-2.1-Turbo-golden/{...final_latents.f32, ...capybara.png}` |

---

## 0. TL;DR

1. **The native VAE decode is semantically correct.** Decoding the *golden* latents with the native
   fp32 path and comparing to an independent **torch fp32** decode of the same latents gives
   **61.6 dB PSNR (RGB), max 9/255, mean 0.045/255, 0.01 % of pixels differ by >2 levels**.
   In float space the disagreement is 9.2e-4 relative RMS (60.7 dB) — and that residual is *mostly
   on the torch side*: the torch "fp32" reference used cuDNN **TF32** convs (torch 2.14 default,
   verified `torch.backends.cudnn.conv.fp32_precision == "tf32"`).
2. **The 39.4 dB end-to-end number is dominated by DiT latent error, not the VAE.** VAE-only parity
   against the *shipped golden PNG* is **47.8 dB**, i.e. the VAE accounts for only
   `10^-(4.78-3.94) ≈ 14 %` of the e2e error power (≈86 % is DiT/tokenizer). torch's own fp32 decode
   reaches **47.75 dB** against that same PNG. The PNG is a **bf16** decode (pipeline ran
   `dtype=torch.bfloat16`), so ~47.8 dB is the *ceiling* for any fp32 decoder; a second independent
   bf16 decode lands at 46.5 dB. Nothing better than ≈47 dB is achievable against this PNG without
   reproducing torch's exact bf16 rounding.
3. **All risky layout/normalisation items check out** (conv weight layout, NCHW↔token rows, RMSNorm
   eps/gamma shapes, SiLU, nearest-2x, pad-1 zero padding, DupUp channel map + `first_chunk` slice,
   RGBA channel order, alpha = channel 3, `x*0.5+0.5` → u8). Details + line numbers in §2/§3.
4. **One controllable bf16 divergence was found** (and, while this analysis ran, was **already fixed**
   in the uncommitted working tree): the latent affine used to apply constants in fp32 and round
   once, while torch casts `latents_mean/std` to bf16 and rounds **twice** (`mul` → bf16,
   `add` → bf16). 43 % of the 4.2 M affine inputs differ by 1 bf16 ulp. Measured cost of that 1-ulp
   input difference on an otherwise identical bf16 VAE: **46.50 dB** vs the golden PNG — *larger*
   than the 48.9 dB cost of running the whole VAE in bf16 instead of fp32. See §4.4 / gate C.
5. **Strongest test = VAE-only**: new `--decode-latents-in FILE` path that bypasses encode + DiT +
   sampler, feeds the golden latents through the *runtime* affine + `QwenImage21VaeModel::decode`,
   and writes the image. Proposed patch, exact element conversion, metrics and thresholds in §4–§7.

---

## 1. What the reference actually does for these latents

`QwenImage21DecodeStep` (`qwenimage21/decoders.py:157-186`):

```
latents [B, H*W, 64]                        # packed, token-major (ndim==3 -> no unsqueeze)
  -> .to(vae.dtype)                         # decoders.py:170  -> bf16 in the golden run
  -> unpack: transpose(1,2).reshape(B,C,1,H,W)   # decoders.py:29-35 via :89-91
  -> latents * latents_std + latents_mean    # decoders.py:181  bf16 consts, bf16 mul, bf16 add
vae.decode(...)                             # decoders.py:182
  _decode: post_quant_conv -> decoder(x[:,:,i:i+1], feat_cache, first_chunk=True)  (1305-1324)
           torch.clamp(out, -1, 1)                                                  (1328)
  -> [:, :, 0]                              # drop the temporal dim -> [B,4,H,W]
image_processor.postprocess                 # decoders.py:183
  denormalize: (x*0.5+0.5).clamp(0,1)       # image_processor.py:223-235
  pt_to_numpy: permute(0,2,3,1).float()     # image_processor.py:192-206  -> RGBA HWC
  numpy_to_pil: (x*255).round().uint8       # image_processor.py:129-150 -> PIL mode "RGBA"
```

Unpack is a pure reshape: `pack_latents` = `view(B,C,H*W).transpose(1,2)`
(`qwenimage21/inputs.py:26-30`), `unpack_latents` = `transpose(1,2).reshape(B,C,1,H,W)`
(`qwenimage21/decoders.py:29-35`). So token row `t = h*W + w`, channel inner.

Because `AutoencoderKLQwenImage21` is built with **`is_residual=True`** (`autoencoder_kl_qwenimage21.py:1135`)
and `temperal_downsample=[F,T,T,T]`, the decoder is 5 × `ResidualUpBlock`
(`autoencoder_kl_qwenimage21.py:649-735`), not `UpBlock`: each block adds a `DupUp3D` `avg_shortcut`
and the checkpoint confirms 5 `decoder.up_blocks.*` groups (verified by reading the safetensors key
table). Geometry: widths `1152,1152,1152,576,288,144`, 3 resnets per block, upsampler on blocks 0–3,
`factor_t = 2,2,2,1` (`temperal_upsample = reversed(temperal_downsample)` at `:1146`,
`up_flag = i != 4` at `:859`, `tmp` shortcut at `:676`). The `if i > 0 and not is_residual: in_dim //= 2`
branch at `:855-856` is **dead** for this checkpoint — the native header correctly does not model it.

`feat_cache` is *not* `None` in `_decode`, but with a single frame every `CausalConv3d` receives
`cache_x=None` (`:172-176` raises only for a non-`None` cache), and `Resample` takes the
`feat_cache[idx] == None → "Rep"` branch (`:288-292`), i.e. **`time_conv` is never executed** and
`DupUp3D(first_chunk=True)` slices `f1 = factor_t - 1` (`:132-133`). Exactly what the native header
asserts (`Vae.hpp:29-34`, `:293-296`, `:1171-1173`).

---

## 2. Side-by-side op table

| # | stage | torch (recipe @ line) | native (`Vae.hpp:line`) | check | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | latent affine | `*std + mean`, bf16 consts, per-channel `[1,64,1,1,1]` (`decoders.py:171-181`) | `denormalize_latents` `:844-869`, `qwen21vae_denormalize_inplace` `:778-811`: fp32 single-round for `float`; **two-op bf16 lane (`kTwoOpBf16`) added in the working tree during this analysis** | sign/order `x*std+mean` identical, constants from runtime `vae/config.json` (`:215-231`) | was **divergent** (R2); now matches for bf16 |
| 2 | `post_quant_conv` 1×1 | `CausalConv3d(64,64,1)` (`:1163`), no pad (`:167-183`) | `gemm_bias` with `[64,64]` alias of `[64,64,1,1]` (`:1097-1105`, `bind_mat` `:477-492`) | W used as `[out,in]` ⇒ `y = x·Wᵀ+b`, matches `nn.Conv2d` | OK |
| 3 | `conv_in` 3×3 | `CausalConv3d(64,1152,3,pad 1)` (`:846`) | `conv3x3_forward` on `[T,64]`, `w[1152,64,3,3]` as-is (`:1106-1113`) | zero-pad 1, cross-correlation (no kernel flip), `kh/kw` order `i*9+kh*3+kw` | OK |
| 4 | mid ResNet ×2 | `norm1(RMS)→SiLU→conv3→norm2→SiLU→conv3` + shortcut (`:358-400`) | `resnet_stage` `:1002-1049` (same order; shortcut precomputed into `tS`) | 1×1 shortcut conv iff `cin!=cout` (+bias), else identity | OK |
| 5 | RMSNorm | `F.normalize(x, dim=1)` on fp32 (`:211-214`), `eps=1e-12`, then `*sqrt(C)*gamma`; gamma `[C,1,1,1]` when `images=False`, `[C,1,1]` for attention (`:199-207`) | `qwen21vae_rmsnorm_inplace` `:612-636`: `x / max(‖x‖,1e-12) * sqrt(C) * gamma` | eps floor, channel axis (token layout `[T,C]`), gamma shapes `{C,1,1,1}` (`:258`) vs `{C,1,1}` (`:270`) | OK (rounding path differs, R4) |
| 6 | activation | `get_activation("silu")` (`:346`, `:844`) | `qwen21vae_silu_inplace` `:564-575` `x*sigmoid(x)` | SiLU, no GELU anywhere | OK |
| 7 | mid attention | norm → `to_qkv` 1×1 → `[bT,1,3C,T]`→`chunk(3,-1)` → `F.scaled_dot_product_attention` (1 head, scale `1/sqrt(C)`) → `proj` 1×1 → `+x` (`:421-447`, `:435`) | `attn_stage` `:1051-1092`: RMS → `to_qkv` → `split_qkv` `:692-708` (`[0:C],[C:2C],[2C:3C]`) → scores `1/sqrt(C)` `:711-728` → row softmax `:730-754` → AV `:756-772` → `proj` → `+x` | split order, scale, single head, residual on pre-norm x | OK (score/prob precision R3) |
| 8 | up blocks ×5 | `ResidualUpBlock` (`:649-735`): 3 resnets (`current_dim→out_dim` for j=0, `out_dim→out_dim` after) + `upsample3d/2d` + `DupUp3D` (`:676-681`, `:732`) | `decode_sample` `:1150-1198`; widths from `Config::dim_mult_at` `:147-156`, `resnets_per_up_block = num_res_blocks+1` `:146` | 5 blocks, 3 resnets, `conv_shortcut` only on j=0 of blocks 2,3,4 | OK |
| 9 | DupUp `avg_shortcut` | `repeat_interleave(out*ft*4/in)` → view `(B,out,ft,2,2,T,H,W)` → interleave → `first_chunk` slice `f1=ft-1` (`:112-135`) | `qwen21vae_dupup_add` `:663-694`: `c = (((co*ft+f1)*2+f2)*2+f3) / repeats`, `f1=ft-1`, `repeats=uout*ft*4/uin` `:1188-1191` | channel map + `h2/2,w2/2` gather identical; `ft=2,2,2,1` from `temporal_up` `:164-170` (reversed `temporal_downsample`) | OK |
| 10 | upsampler | `QwenImage21Upsample(2.0,"nearest-exact")` (fp32 interp, cast back, `:221-233`) → `Conv2d(dim,dim,3,pad 1)` (`:261-269`) | `qwen21vae_nearest_upsample2x` `:639-657` (`h2/2,w2/2`) → `conv3x3_forward` (`:1179-1186`) | nearest-exact **is** nearest for integer scale 2; pad 1 symmetric | OK |
| 11 | `time_conv` | exists (`:270`), only in the `feat_cache` multi-frame branch (`:288-303`) — **never on this path** | required present, never executed (`require_decoder_tensors` `:956-999`) | 3 of 5 up blocks have it in the checkpoint (`up_blocks.3` has none) | OK / by design |
| 12 | head | `norm_out` RMS(`images=False`) → SiLU → `conv_out` 3×3 → 4ch (`:887-888`, `:892-930`) | `:1199-1222` | 144→4, gamma `[144,1,1,1]` | OK |
| 13 | clamp | `torch.clamp(out,-1,1)` (`:1328`) | `qwen21vae_clamp_inplace` `:1219-1220` | −1..1 | OK |
| 14 | output mapping | `[:, :, 0]` → `permute(0,2,3,1)` → `(x*0.5+0.5).clamp(0,1)*255` `.round()` (`decoders.py:182`, `image_processor.py:129-150,192-235`) | driver `:1832-1841`: clamp(−1,1) → `*0.5+0.5` → clamp(0,1) → `lround(255v)`; PPM drops ch 3 (`:2391-2408`) | RGBA order, alpha = decoded ch 3 (not a mask), `[−1,1]→[0,1]` applied once | OK (round-half mode R6) |

Additional verified facts:

* Channel order: cross-channel PSNR matrix (native vs golden) is strongly diagonal
  (R↔R 47.0, G↔G 46.7, B↔B 50.6, A↔A 54.9 dB; best off-diagonal 20 dB) ⇒ output really is RGBA,
  alpha is the decoded 4th channel, no BGR/RGB swap.
* Checkpoint/expected-tensor agreement: every key & shape in
  `qwen21vae_expected_decoder_tensors` (`:245-305`) matches
  `weights/Qwen-Image-2.1-Turbo/vae/diffusion_pytorch_model.safetensors` exactly (238 tensors,
  3 `time_conv` pairs on up_blocks 0–2 only, `[1152,1152,3,3]` … `[4,144,3,3]`, gamma
  `[C,1,1,1]` for resnets/`norm_out` and `[1152,1,1]` for the attention norm).
* Golden latent values are **exactly bf16-representable** (100 % round-trip) ⇒ the pipeline ran the
  latents in bf16 and nothing is lost when the driver casts them to `bhalf_t`.

---

## 3. Ranked risks / divergences

Impact numbers are *measured* from real artifacts (see Appendix A), RGB u8 metrics against the
golden PNG unless stated otherwise.

| # | risk | where | severity | measured / expected impact | action |
| --- | --- | --- | --- | --- | --- |
| **R1** | **Harness conflates DiT and VAE error**: a full-run image diff cannot attribute error. The reported e2e 39.4 dB is *DiT-latent* dominated (VAE-only ceiling against the same PNG is 47.8 dB, i.e. ~14 % of the e2e MSE). | driver sample path (`:2279-2311`) + `:2349-2360` | **high (blocking the question)** | e2e 39.4 dB vs VAE-only 47.8 dB | add the VAE-only path (§4) |
| **R2** | **bf16 affine ordering/constant dtype**: torch casts `latents_mean/std` to bf16 and rounds `mul` then `add` separately (`decoders.py:171-181`); the pre-change native kept fp32 constants and rounded the whole `x*std+mean` once. | `Vae.hpp:778-811`, `:844-869` vs `decoders.py:170-181` | medium → **fixed in the working tree** (uncommitted, `kTwoOpBf16` lane) | 43 % of the 4.2 M affine inputs differ by 1 bf16 ulp (mean 2.2e-3 relative); on an otherwise identical bf16 VAE that 1-ulp input difference alone yields **46.50 dB** vs the golden PNG, vs 48.9 dB for the whole bf16-vs-fp32 precision loss | done; **verify via gate C** (§6) |
| **R3** | **Attention scores/probs stored in bf16**: native writes `s = Scalar(scale·qk)` (`:726-727`) and softmax output (`:746-752`) to bf16 scratch; torch SDPA keeps scores/softmax in fp32 internally (`:435`). At 1024 px this is the largest single op (T=4096, C=1152 ⇒ ~77 GFLOP). | `:711-772` vs `:421-447` | low-medium | bundled into R5's ≤1-level residual; not separately measurable without a GPU run | optional: fp32 score/prob scratch (33 MB → 67 MB) |
| **R4** | RMSNorm rounding path: torch rounds `normalize` to bf16, then `*scale`, then `*gamma` (3 bf16 roundings, `:213-216`); native does one fp32 chain and one cast (`:612-636`). | `:612-636` vs `:209-219` | low | sub-ulp, absorbed in the 61.6 dB fp32 / 47 dB bf16 residuals | none |
| **R5** | Conv/GEMM accumulation order: cuDNN (torch) vs cuDNN-dispatch or im2col+cuBLAS/native tiled GEMM (`src/KokkosDNN_Conv.hpp:394-434`, `src/KokkosDNN_GemmBias.hpp:105-118`) — all fp32 accumulate, different summation trees. | — | low | fp32 vs fp32: 61.6 dB (max 9/255, mean 0.045); 99.99 % of pixels within ±2 | none |
| **R6** | u8 rounding mode: native `lround` = round-half-away, numpy `round` = round-half-even (`image_processor.py:147`). | driver `:1840` | low | flips only when `255·v` is exactly `.5` ⇒ ±1 level on isolated pixels | none (document) |
| **R7** | The reference "fp32" decode used **TF32** convs (torch 2.14 default `cudnn.conv.fp32_precision="tf32"`), so the fp32 gate below compares native fp32 against a *slightly inaccurate* reference. | reference scripts | info | bounds the achievable fp32 agreement from above; measured 61.6 dB is therefore a **floor** | disable TF32 in the fp32 reference (§7) |
| **R8** | `time_conv` / `feat_cache` semantics | `:29-34`, `:956-999` vs `:266-303`, `:1317-1324` | info | verified never executed with `first_chunk=True`, T=1 (checkpoint has no `time_conv` on up_blocks.3, matching `temporal_up(3)==0`) | keep presence check |
| **R9** | Golden image is **square** (latent 64×64 ⇒ 1024×1024), so an h/w-index transposition in the token↔NCHW mapping would be **invisible** in this test. | `:530-562`, `:639-657` | low | 0 impact on this test; a real bug class left uncovered | follow-up: non-square (e.g. 1024×768) synthetic torch reference |
| **R10** | Driver casts the host latents to bf16 *before* the affine (`:1812-1822`, `DScalar = bhalf_t` at `:521`). | driver `:521`, `:1819` | info | lossless for the golden file (values are bf16-exact), but would silently round a future fp32 reference | add `--vae-fp32` (§4.3) |

Failure-signature guide (useful when a gate fails): a layout/transpose/eps bug shows up as
PSNR < 25 dB with *structured* (edge/block) error; a channel swap shows PSNR < 20 dB with
colour inversion; a norm-eps or activation bug shows a smooth global bias/darkening; the residual
we actually see (47–62 dB, no spatial structure, `corr(|Δ|, image gradient) = 0.34`, border mean
0.39 vs interior 0.58) is pure numeric noise.

---

## 4. Proposed patch: `--decode-latents-in` (VAE-only decode)

Goal: golden latents → runtime affine → native decode → PNG, **without** the 17.5 GB encoder or the
14.2 GB DiT. Runtime ≈ VAE weight load + one decode (~3 s on the 3090-class box instead of ~40 min),
so the test is cheap enough to run on every VAE change.

### 4.1 New flags

```
--decode-latents-in FILE   raw fp32 LE latents; implies --vae-decode and skips
                           encode + DiT load + sampler entirely. Needs
                           --vae-weights (and real weights; --tiny is rejected).
--decode-latents-layout L  chw (default, driver layout lat[(c*H+h)*W+w]) |
                           token   ([H*W, C] token-major t[(h*W+w)*C+c], i.e.
                           the golden qwen_image_21_turbo_final_latents.f32)
--decode-out-f32 FILE      optional: also write the post-clamp RGBA decode as raw
                           f32 [4,H,W] (channel-major) for float-space diffs
--vae-fp32                 optional: run the decoder in float instead of bf16
```

`--height/--width` keep their meaning: **latent** H,W (64,64 ⇒ 1024×1024). Validate
`file_size == 4*C*H*W` and `C == vae/config.json z_dim` (64).

### 4.2 Code (minimal, ~110 added lines, no change to decode numerics)

```cpp
// after load_init_latents_raw() (~line 588)
// Raw fp32 LE latents for --decode-latents-in. Both layouts are validated
// against C*H*W; the token-major form is exactly
// scripts/qwen_image_21_golden.py's `final_lat.to(torch.float32).numpy().tofile()`
// ([1,H*W,C], token row t = h*W + w, channel inner).
inline std::vector<float> load_decode_latents(const std::string& path,
                                              size_t C, size_t H, size_t W,
                                              bool token_major) {
  std::vector<float> raw = load_init_latents_raw(path, C, H, W);  // size-checked
  if (!token_major) return raw;
  std::vector<float> out(raw.size());
  for (size_t h = 0; h < H; ++h)
    for (size_t w = 0; w < W; ++w)
      for (size_t c = 0; c < C; ++c)
        out[(c * H + h) * W + w] = raw[(h * W + w) * C + c];
  return out;
}

// VAE-only entry point: define right after try_vae_decode_rgba() (~line 1848)
// plus write_output_image(); called from main() before Phase 1 (no encoder,
// no DiT load, no sampler).
inline void run_vae_only(const std::string& lat_path, bool token_major,
                         size_t C, size_t H, size_t W,
                         const std::string& vae_weights, const std::string& out,
                         const std::string& f32_out) {
#ifdef QWEN21_DRIVER_HAVE_VAE
  std::vector<float> lat = load_decode_latents(lat_path, C, H, W, token_major);
  std::printf("[decode-latents %s] %zux%zux%zu floats (%s layout)\n",
              lat_path.c_str(), C, H, W, token_major ? "token" : "chw");
  std::vector<float> rgba_f32;
  size_t oH = 0, oW = 0;
  int oCh = 1;
  std::vector<unsigned char> pix;
  if (!try_vae_decode_rgba(lat, C, H, W, vae_weights, pix, oH, oW, oCh,
                           f32_out.empty() ? nullptr : &rgba_f32))
    throw std::runtime_error("VAE-only decode failed");
  if (!f32_out.empty()) save_raw_f32(f32_out, rgba_f32);
  write_output_image(out, pix, oH, oW, oCh, /*decoded=*/true);
#else
  throw std::runtime_error(
      "--decode-latents-in needs KokkosDNN_QwenImage21Vae.hpp in this build");
#endif
}
```

Three supporting edits:

1. `try_vae_decode_rgba` (`:1773-1848`): add a trailing
   `std::vector<float>* rgba_f32 = nullptr` parameter and fill it inside the existing
   `h,w,c` loop (values `rh(0,c,h,w)` post-clamp) — no extra passes.
2. Extract the existing `#ifdef KOKKOSDNN_ENABLE_IMAGE … #else … PPM … #endif` block
   (`:2375-2414`) verbatim into `write_output_image(out, pix, oH, oW, oCh, decoded)` and call it
   from both the normal tail and `run_vae_only` (keeps one writer, no duplication).
3. Early branch inside the existing `try`, **after** `H/W/C` are known (`:2033-2053`) and before
   Phase 1 (`:2068`):

```cpp
    // ---- VAE-only parity path (--decode-latents-in) ----
    if (!decode_latents_in.empty()) {
      if (vae_weights.empty())
        throw std::runtime_error(
            "--decode-latents-in needs --vae-weights (implies --vae-decode)");
      if (tiny)
        throw std::runtime_error(
            "--decode-latents-in needs the real VAE (drop --tiny)");
      run_vae_only(decode_latents_in, decode_layout == "token", C, (size_t)H,
                   (size_t)W, vae_weights, out, decode_out_f32);
      Kokkos::finalize();  // nothing else runs on this path
      return 0;
    }
```

`--vae-fp32` (optional, recommended): make `try_vae_decode_rgba` a template on the scalar and
dispatch `vae_fp32 ? try_vae_decode_rgba<float>(…) : try_vae_decode_rgba<DScalar>(…)`; the decode
path is already `Scalar`-generic, so this is 4 lines + a template parameter. This is what lets the
*driver* run the 61.6 dB semantic gate without the scratch harness.

Also update `usage()` (`:1850-1911`) and the flag table in
`examples/qwen_image_21/README.md` (rows after `--vae-decode`), plus a 4-line "VAE-only parity"
subsection pointing at this document.

### 4.3 Why this is the right scope

* No new numerics: same `QwenImage21VaeModel<Exec,DScalar>`, same runtime `vae/config.json` affine
  (`qwen21_load_vae_config_json`), same `x*0.5+0.5` → u8 mapping — so the test measures the code
  the pipeline actually uses (not a parallel implementation).
* Single flag, no checkpoint with 32 GB of unused weights, no sampling, deterministic.
* `--vae-weights` validation (`:2142-2162`) already exists and is reused.

### 4.4 R2 fix (bf16-exact latent affine) — **already implemented in the working tree**

Torch does `z.to(bf16) * std.to(bf16)` → bf16, then `+ mean.to(bf16)` → bf16
(`decoders.py:170-181`). During this analysis the working tree gained exactly that lane in
`Impl::qwen21vae_denormalize_inplace` (`Vae.hpp:778-811`, `kTwoOpBf16` selected on
`VScalar == bhalf_t`, fp32 path untouched):

```cpp
const VScalar s = static_cast<VScalar>(static_cast<float>(std(c)));
const VScalar m = static_cast<VScalar>(static_cast<float>(mean(c)));
const VScalar t = static_cast<VScalar>(v * static_cast<float>(s));   // mul -> bf16
x(...) = kTwoOpBf16 ? static_cast<VScalar>(static_cast<float>(t) +
                                           static_cast<float>(m))     // add -> bf16
                    : single_rounding_form;
```

Consequences for the plan: no code change needed for R2 — instead **gate C (§6) is now the
experiment that validates it**, and its measured value becomes the new bf16 baseline. (Being
uncommitted, confirm the lane is still present when you run the test; the `mean`/`std` device
vectors are already `Scalar` so the double cast is a no-op for bf16 and exact for float.)

---

## 5. Exact element conversion (golden file → driver layout)

Golden file: `qwen_image_21_turbo_final_latents.f32`, **1 048 576 bytes = 262 144 floats**, written
by `scripts/qwen_image_21_golden.py:343-346` as `final_lat.to(torch.float32).numpy().tofile()` on a
`[1, 4096, 64]` tensor (log line `[d] final latents shape=(1, 4096, 64)`). Contents are identical to
the sibling `.npy` (verified `np.array_equal`).

* token row index `t = h*64 + w` (`h` outer, `w` inner; `pack_latents` is a plain
  `view(B,C,H*W).transpose(1,2)`), channel inner:
  `file[t*64 + c]`, `t = 0 … 4095`.
* driver layout `lat[(c*H + h)*W + w]` with `C=64, H=W=64`, so

```
lat[(c*64 + h)*64 + w]  =  file[(h*64 + w)*64 + c]
```

Equivalent one-liner (also what `--decode-latents-layout token` does in C++):

```python
z = np.load('.../qwen_image_21_turbo_final_latents.npy')            # [1,4096,64]
chw = np.ascontiguousarray(z[0].reshape(64, 64, 64).transpose(2, 0, 1), dtype='<f4')
chw.tofile('/tmp/opencode/qwen/golden_latents_chw.f32')            # 1 MiB, [C,H,W]
```

Verification that this mapping is right: the scratch harness
`/tmp/opencode/qwen21-vae-cuda/vae_full.cu:35-38` used exactly this mapping
(`lh(0,c,h,w) = raw[(h*64+w)*64+c]`), and its fp32 output reaches 61.6 dB against an independent
torch fp32 decode — a wrong mapping could not get there.

Caveat: at 1024² the grid is square, so this test does not distinguish `h↔w` mistakes (R9).

---

## 6. Acceptance metrics and thresholds

Metric definition (put in the compare script, Appendix B): both images → `u8`,
`u8 = rint(clip(x*0.5+0.5, 0, 1) * 255)`; compare the **RGB** channels; report
`PSNR = 10log10(255²/MSE)`, `max|Δ|`, `mean|Δ|`, `px>2` = fraction of pixels whose worst RGB channel
differs by >2 levels, plus alpha separately. Also run the same binary twice — the decode is
deterministic, so a non-zero self-diff means the harness is broken.

Measured reference points (all from real artifacts, Appendix A):

| comparison (u8, RGB) | PSNR | max | mean | px>2 | px>8 |
| --- | --- | --- | --- | --- | --- |
| native fp32 ← golden latents **vs torch fp32** ← golden latents | **61.55 dB** | 9 | 0.045 | 0.01 % | 0.00 % |
| native fp32 vs torch bf16 | 48.95 dB | 61 | 0.447 | 3.32 % | 0.11 % |
| torch fp32 vs torch bf16 | 48.89 dB | 59 | 0.451 | 3.39 % | 0.11 % |
| native fp32 vs **golden PNG** | **47.78 dB** | 40 | 0.571 | 4.52 % | 0.15 % |
| torch fp32 vs golden PNG | 47.75 dB | 40 | 0.573 | 4.56 % | 0.15 % |
| torch bf16, **fp32-order** affine vs golden PNG (bf16-order affine) | 46.50 dB | 61 | 0.652 | 6.49 % | 0.26 % |

Float space (no u8 quantisation): native fp32 vs torch fp32 = `max 6.8e-2`, `rms 6.0e-4`,
`9.2e-4` relative RMS (60.7 dB); per channel R 54.0, G 54.4, B 62.5, A 91.5 dB.

Proposed gates:

| gate | what | threshold | rationale |
| --- | --- | --- | --- |
| **A. semantic (primary)** | native **fp32** decode of the golden latents vs a **torch fp32** decode of the same latents (TF32 off, §7) | **PSNR ≥ 58 dB, mean ≤ 0.10, px>2 ≤ 0.05 %, max ≤ 16** | measured 61.55 dB *with* a TF32 reference; leaves ~3.5 dB for TPL/build variation (cuBLAS/cuDNN on vs off) while a single wrong op (conv flip, eps, activation, channel map) lands <25 dB |
| **B. end-to-end precision** | driver default (`DScalar=bhalf_t`) decode of the golden latents vs **golden PNG** | **PSNR ≥ 45 dB, mean ≤ 0.9, px>2 ≤ 9 %, max ≤ 80** | ~47.8 dB is the bf16-vs-fp32 ceiling; two independent bf16 decodes already sit at 46.5 dB. `max` is loose by design: isolated clamp-boundary pixels in bf16 differ by up to ~61 levels |
| **C. stretch (validates the R2 fix now in the tree)** | same as B | **PSNR ≥ 52 dB, px>2 ≤ 1 %** | predicts that the two-op bf16 affine locks the bf16 trajectory (quantiser behaviour: 1-ulp input changes mostly round the same way, so divergence becomes sparse). If not achieved, record the measured value as the bf16 baseline and keep gate B |
| **D. determinism** | same binary twice | `max\|Δ\| = 0` | catches non-deterministic dispatch |

Do **not** gate on the 1024² end-to-end image against the golden PNG at >45 dB: that mixes DiT
latent error (currently 39.4 dB) with VAE error and will always fail; treat e2e as a separate,
coarser gate.

---

## 7. Exact commands

Build (`/tmp/opencode/qwen21-main-check` already has `CMAKE_HOME_DIRECTORY=/workspace`,
`KokkosDNN_ENABLE_IMAGE=ON`, warnings-as-errors ON, all TPL dispatch OFF — the same configuration
class as the runs logged in `/tmp/opencode/qwen21-run*.log`):

```sh
cmake --build /tmp/opencode/qwen21-main-check --target qwen_image_21_generate -j
# fresh tree, if needed:
cmake -S /workspace -B /tmp/opencode/qwen21-vaeonly -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ROOT=/workspace/install/kokkos-cuda \
  -DKokkosDNN_ENABLE_IMAGE=ON -DKokkosDNN_ENABLE_TESTS=ON \
  -DKokkosDNN_ENABLE_COMPILER_WARNINGS=ON
cmake --build /tmp/opencode/qwen21-vaeonly --target qwen_image_21_generate -j
```

Prepare the latents (CPU only):

```sh
python3 - <<'EOF'
import numpy as np
z = np.load('/workspace/weights/Qwen-Image-2.1-Turbo-golden/qwen_image_21_turbo_final_latents.npy')
chw = np.ascontiguousarray(z[0].reshape(64,64,64).transpose(2,0,1), dtype='<f4')
chw.tofile('/tmp/opencode/qwen/golden_latents_chw.f32'); print(chw.shape, chw.nbytes)
EOF
```

VAE-only decode (bf16 driver path, the intended gate B; ~1 min incl. weight load):

```sh
mkdir -p /tmp/opencode/qwen
/tmp/opencode/qwen21-main-check/examples/qwen_image_21/qwen_image_21_generate \
  --decode-latents-in /tmp/opencode/qwen/golden_latents_chw.f32 \
  --height 64 --width 64 \
  --vae-weights /workspace/weights/Qwen-Image-2.1-Turbo/vae/diffusion_pytorch_model.safetensors \
  --decode-out-f32 /tmp/opencode/qwen/vae_only_bf16.f32 \
  --out /tmp/opencode/qwen/vae_only_bf16.png
```

or, with `--decode-latents-layout token`, point straight at the golden file:

```sh
  --decode-latents-in /workspace/weights/Qwen-Image-2.1-Turbo-golden/qwen_image_21_turbo_final_latents.f32 \
  --decode-latents-layout token --height 64 --width 64
```

Reproduce the scratch fp32 reference (gate A; needs the GPU briefly, ~1-2 min, 1.4 GB weights):

```sh
python3 - <<'EOF'
import numpy as np, torch
from diffusers import AutoencoderKLQwenImage21
torch.backends.cudnn.allow_tf32 = False                     # R7: true fp32 convs
torch.backends.cuda.matmul.allow_tf32 = False
vae = AutoencoderKLQwenImage21.from_pretrained(
    '/workspace/weights/Qwen-Image-2.1-Turbo/vae', torch_dtype=torch.float32).cuda().eval()
lat = np.load('/workspace/weights/Qwen-Image-2.1-Turbo-golden/qwen_image_21_turbo_final_latents.npy')
z = torch.from_numpy(lat).float().transpose(1, 2).reshape(1, 64, 1, 64, 64).cuda()
m = torch.tensor(vae.config.latents_mean, dtype=torch.float32).cuda().view(1, 64, 1, 1, 1)
s = torch.tensor(vae.config.latents_std,  dtype=torch.float32).cuda().view(1, 64, 1, 1, 1)
with torch.no_grad():
    out = vae.decode(z * s + m, return_dict=False)[0][:, :, 0].float().cpu().numpy()
out.tofile('/tmp/opencode/qwen/vae_ref_fp32.f32')           # [4,1024,1024]
EOF
```

then run the same decode with `--vae-fp32` and compare
`/tmp/opencode/qwen/vae_only_fp32.f32` against `/tmp/opencode/qwen/vae_ref_fp32.f32`.

**Control experiment (recommended, 1 min of GPU):** confirm the golden PNG is exactly reproducible
so the target is well defined — same as above but `torch_dtype=torch.bfloat16`, keep
`z` bf16 and build the affine **exactly like the pipeline** (constants cast to bf16):

```python
vae = AutoencoderKLQwenImage21.from_pretrained(
    '/workspace/weights/Qwen-Image-2.1-Turbo/vae', torch_dtype=torch.bfloat16).cuda().eval()
lat = torch.from_numpy(np.load('.../qwen_image_21_turbo_final_latents.npy')).bfloat16() \
          .transpose(1, 2).reshape(1, 64, 1, 64, 64).cuda()
m = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1).to(lat.device, lat.dtype)
s = torch.tensor(vae.config.latents_std ).view(1, 64, 1, 1, 1).to(lat.device, lat.dtype)
with torch.no_grad():
    out = vae.decode(lat * s + m, return_dict=False)[0][:, :, 0]
u8 = (out.float() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)[0].cpu().numpy()
u8 = (u8 * 255).round().astype(np.uint8)     # == image_processor.numpy_to_pil
Image.fromarray(u8).save('/tmp/opencode/qwen/vae_control_bf16.png')  # diff vs golden PNG
```

Expected: **max|Δ| ≤ 1** vs the golden PNG (deterministic kernels, same code). If it reproduces
exactly, the 46.5 dB figure above is confirmed to be *only* the affine-ordering effect, and gate C
becomes actionable; if it does not reproduce, the golden PNG must be re-generated before any
threshold is meaningful.

Compare (Appendix B):

```sh
# gate B: driver bf16 decode vs the golden PNG
python3 scripts/qwen_image_21_vae_compare.py \
  --native /tmp/opencode/qwen/vae_only_bf16.png --psnr-min 45
# gate A: native fp32 vs torch fp32 (same latents, no u8/golden-PNG in between)
python3 scripts/qwen_image_21_vae_compare.py \
  --native /tmp/opencode/qwen/vae_only_fp32.f32 --f32 \
  --golden /tmp/opencode/qwen/vae_ref_fp32.f32 --golden-f32 --psnr-min 58
```

---

## 8. Residual risks / follow-ups

1. **Non-square geometry (R9)**: add a torch reference at 1024×768 (latent 64×48) and a matching
   `--decode-latents-in` run; the current square golden cannot catch an h↔w transposition.
2. **Promote the fp32 semantic gate into CI**: it needs no 32 GB checkpoint (VAE only) and runs in
   ~1-2 min; the reference artifact is 16 MiB. Alternatively add a `tests/Test_QwenImage21Vae.cpp`
   in the style of `tests/Test_MossImageVae.cpp` using `Config::tiny_test()` (4-ch latent, 2×2 →
   4×4 image, 128 output floats) against a checked-in torch reference with a 1e-5 tolerance — that
   is a GPU-free regression test for the whole op table.
3. **R3 (attention precision)**: only worth revisiting if gate C stalls below ~50 dB after the R2
   fix; the fix is a float32 scores/probs scratch (67 MB at 1024 px).
4. **Alpha channel**: gate on RGB only; alpha matches to ≤1 level in 99.99 % of pixels and is not a
   perceptual signal (golden alpha is 254–255 for 99.6 % of pixels).
5. **Artifact provenance**: the fp32/bf16 reference `.f32` files live only in `/tmp` and were
   produced by ad-hoc scripts (`/tmp/qwen21_golden_float.py`, `/tmp/qwen21_golden_float32.py`,
   `/tmp/opencode/qwen21-vae-cuda/vae_full.cu`). If the plan above lands, move the torch reference
   generator into `scripts/` (`--vae-only` mode of `qwen_image_21_golden.py`, or a small
   `qwen_image_21_vae_ref.py`) so the numbers are reproducible.
6. The reported 39.4 dB e2e number was **not re-verified**: the 1024² e2e artifact
   (`/tmp/opencode/qwen/e2e_1024.ppm`) is not on disk; the e2e images found are all 512²
   (`/tmp/opencode/qwen21_e2e.png` mode L, `/tmp/opencode/qwen21-turbo-fix.png`, 512² RGBA,
   latent 32×32).

---

## Appendix A — artifacts used for the measured numbers

| file | what |
| --- | --- |
| `/workspace/weights/Qwen-Image-2.1-Turbo-golden/qwen_image_21_turbo_final_latents.{f32,npy}` | `[1,4096,64]` token-major bf16-exact latents (1 048 576 B) |
| `/workspace/weights/Qwen-Image-2.1-Turbo-golden/qwen_image_21_turbo_capybara.png` | 1024² RGBA torch **bf16** decode of those latents (pipeline, seed 42) |
| `/tmp/qwen21_cpp_full.f32` | native **fp32** decode of the golden latents (16 MiB, `[4,1024,1024]`, post-clamp) — from `/tmp/opencode/qwen21-vae-cuda/vae_full.cu` |
| `/tmp/qwen21_cpp_capybara.png` | the same native fp32 decode rendered to RGBA PNG (reproduces the u8 metrics above: 47.78 dB vs golden) |
| `/tmp/qwen21_ref_full_f32.f32` | torch **fp32** decode of the same latents — `/tmp/qwen21_golden_float32.py` (TF32 convs ON) |
| `/tmp/qwen21_ref_full.f32` | torch **bf16** decode of the same latents, fp32 affine — `/tmp/qwen21_golden_float.py` |

Cross-checks performed: golden `.f32` == `.npy`; all latent values bf16-exact; native/expected tensor
inventory == checkpoint keys/shapes; cross-channel PSNR matrix shows a correct RGBA mapping;
alpha is the decoded 4th channel. `/tmp/qwen21_cpp_full.f32` predates the uncommitted bf16-affine
change but is unaffected by it (the `float` path was not modified).

## Appendix B — proposed `scripts/qwen_image_21_vae_compare.py`

```python
#!/usr/bin/env python3
"""Compare a VAE-only decode against a reference PNG or raw f32 decode.

Gate A (semantic): --native vae_only_fp32.f32 --f32 \
                   --golden vae_ref_f32.f32 --golden-f32 --psnr-min 58
Gate B (bf16 driver vs the golden PNG): --native vae_only_bf16.png --psnr-min 45
"""
import argparse
import numpy as np
from PIL import Image

GOLDEN = ("/workspace/weights/Qwen-Image-2.1-Turbo-golden/"
          "qwen_image_21_turbo_capybara.png")


def load(path, f32=False, h=1024, w=1024):
    if f32:  # raw [4,H,W] float32 post-clamp decode
        a = np.fromfile(path, dtype=np.float32).reshape(4, h, w)
        return np.rint((np.clip(a, -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.asarray(Image.open(path)).transpose(2, 0, 1)[:4]


def report(name, a, b):
    d = np.abs(a[:3].astype(np.int16) - b[:3].astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse else float("inf")
    px = d.max(0)
    print(f"{name}: PSNR={psnr:.2f} dB max={d.max()} mean={d.mean():.3f} "
          f"px>2={100 * np.mean(px > 2):.2f}% px>8={100 * np.mean(px > 8):.3f}%")
    if a.shape[0] == 4 and b.shape[0] == 4:
        da = np.abs(a[3].astype(np.int16) - b[3].astype(np.int16))
        print(f"  alpha: max={da.max()} mean={da.mean():.4f}")
    return psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--f32", action="store_true",
                    help="--native is raw [4,H,W] float32 instead of a PNG")
    ap.add_argument("--golden-f32", action="store_true",
                    help="--golden is raw [4,H,W] float32 (gate A)")
    ap.add_argument("--hw", type=int, default=1024, help="image side (f32 paths)")
    ap.add_argument("--psnr-min", type=float, default=0.0)
    a = ap.parse_args()
    psnr = report(a.native, load(a.native, a.f32, a.hw, a.hw),
                  load(a.golden, a.golden_f32, a.hw, a.hw))
    if psnr < a.psnr_min:
        raise SystemExit(f"FAIL: {psnr:.2f} dB < {a.psnr_min} dB")
    print("PASS")


if __name__ == "__main__":
    main()
```

Validated against the existing artifacts: the script reproduces this report's numbers exactly
(`--native /tmp/qwen21_cpp_full.f32 --f32` → `47.78 dB max=40 mean=0.571 px>2=4.52 %`,
alpha `max=23 mean=0.2113`).
