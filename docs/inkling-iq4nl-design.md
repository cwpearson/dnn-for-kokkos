# IQ4_NL support — design

Scope: add GGUF `IQ4_NL` decode/compute coverage to
`src/KokkosDNN_GGUF.hpp` (+ optional int8-dot path in
`src/KokkosDNN_GGUFGemvInt8.hpp`) so IQ4_NL checkpoints
(e.g. `ggml-org/Inkling-2-Mini-GGUF` IQ4_NL, ~17.6 GB packed)
can run on a 24 GB RTX 3090, where BF16 (52 GB) and Q8_0
(27.8 GB) do not fit.

Status: **implemented** (batch 1 commit `321e013` + batch 4 view-portability
fixes; this doc is retained as the verification record).

## 1. Verified ggml spec (upstream llama.cpp, master)

Checked against `ggml/include/ggml.h`, `ggml/src/ggml-common.h`
(struct + `kvalues_iq4nl` table), and `ggml/src/ggml-quants.c`
(`dequantize_row_iq4_nl`, `quantize_row_iq4_nl_impl`).

| Item | Value (verified) |
|---|---|
| `GGML_TYPE_IQ4_NL` | **20** (`ggml.h` enum). Note: 24 is `I8`, already taken in our enum. The task prompt's guess of 24 is wrong; use 20. |
| `QK4_NL` (block length) | **32** elements |
| `block_iq4_nl` layout | `ggml_half d; uint8_t qs[16];` — 2 + 16 = **18 bytes** (same footprint as Q4_0, different dequant) |
| Nibble order | `qs[j]` low nibble → `y[j]`, high nibble → `y[j+16]`, `j` in 0..15 (identical split to Q4_0/Q8_0-style halves, NOT the K-quant 64-group interleave) |
| Codebook `kvalues_iq4nl[16]` (int8) | `-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113` |
| Dequant formula | `y[j] = fp16(d) * LUT[qs[j] & 0xF]`, `y[j+16] = fp16(d) * LUT[qs[j] >> 4]`. **No min/offset term** (unlike Q4_K/Q5_K `d*q − min`). No extra shift: the table values are used directly (asymmetric range, min −127 / max +113). |
| Effective bitrate | 18·8/32 = **4.5 bpw** |
| Alignment class | 18 ≡ 2 (mod 4): block `i` starts at byte `18i` → always 2-byte aligned, 4-byte aligned on even `i` only. **Same class as Q8_0 (34 B)**, not Q4_K (144 B). Consequence: 128-bit and odd-offset 32-bit loads on the packed row are unsafe; use 2-byte loads (`QuantU16x8` / `Dot::load_u16`) exactly like the Q8_0 paths. |

## 2. Current code inventory (line anchors, `src/KokkosDNN_GGUF.hpp`)

- Enum `GgmlType`: 68–87 (`I8 = 24`, `BF16 = 30`; id 20 is free).
- `ggml_block_len`: 90–112 (Q4_0-group → 32 at 98–103; K-quants → 256 at 104–109).
- `ggml_type_size`: 114–138 (Q4_0 → 18 at 124; Q8_0 → 34 at 128).
- `ggml_is_quant`: 140–142 (block_len > 1 — IQ4_NL becomes true automatically).
- `ggml_is_k_quant`: 144–148. IQ4_NL is **not** a K-quant (no 256-superblock, no sub-block scales): leave it `false`.
- `gguf_can_compute`: 156–160 (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K).
- `ggml_type_name`: 162–184; `ggml_row_bytes`: 186–195 (throws if `cols % bl != 0`).
- Block structs: `BlockQ4K` 242–248, `BlockQ80` 250–254, `BlockQ6K` 256–262, `BlockQ5K` 264–271; GEMV-mirror drift asserts 275–282.
- Full-block dequantizers: `dequant_q4k_block` 300–322, `dequant_q80_block` 324–327, `dequant_q6k_block` 329–354, `dequant_q5k_block` 362–384.
- Host `dequant_tensor_to_float`: 778–846 (per-dtype cases 786–841, `default` throw 842–844).
- Device `quant_elem`: 892–968 (Q8_0 900–904, Q4_K 905–918, Q5_K 919–935, Q6_K 936–965, `default: return 0` 966).
- Grouped dot `quant_group_dot`: 982–1094 (per-dtype group = one scale set; Q8_0 1083–1090); eligibility `quant_can_group`: 1099–1107 (Q4_K/Q5_K/Q8_0 gs=32, Q6_K gs=16).
- Decode units: `quant_unit_count` 1153–1161 (K-quants + Q8_0 → `cols/32`); `QuantU8x16` 1165–1167 (16-byte loads); `QuantU16x8` + `quant_load_u16x8` 1174–1186 (2-byte loads for 2-aligned payloads); `QuantUnitX` 1207–1209; `quant_unit_load_x` 1213–1252 (Q4_K 1216–1221, Q6_K 1222–1224, Q5_K/Q8_0 contiguous-32 at 1229–1237).
- Per-dtype unit dots: `quant_q4k_dot_x` 1258–1287, `quant_q6k_dot_x` 1330–1367, `quant_q80_dot_x` 1406–1434, `quant_q5k_dot_x` 1469–1512; runtime dispatch `quant_unit_dot` 1556–1569; compile-time `quant_unit_dot_x` 1573–1584.
- `quant_check` 1993–2006 (error string lists supported dtypes at 1998).
- MoE `block_aligned` lambda 2369–2377 (K-quants + Q8_0 require `n % 256 == 0`, else `true`).
- `quant_gemv` dispatch 2395–2462: unit path 2408, grouped 2451, per-element 2458. `quant_gemm` 2564–2593 (grouped 2582, per-element 2589).
- `dequant_to_dense` + grouped row kernels 2639–2934 (dtype switch 2861–2933; Q8_0 case 2895–2905).
- int8-dot bridge: `Impl::quant_int8_dtype` 2504–2514, `quant_gemv_int8_supported` 2523–2529, scratch sizing 2496–2499.

`src/KokkosDNN_GGUFGemvInt8.hpp` inventory:

- `Wd` tag enum 79; mirror block structs 83–105 (asserted equal in GGUF.hpp 275–282).
- `dtype_ok` 142–150 (Q4K/Q5K/Q6K need `K % 256`, Q80 needs `K % 32`).
- `mvq_load_unit_window` 281–353 (Q5K/Q80 = contiguous-32 window 302–352).
- `gemv_int8_unit_dot_win` 360–466 (Q80 branch 457–465: int dots + `Dot::load_u16` — the template for IQ4_NL).
- Launch alignment gate 739–743: `wdata` even required always; 16-byte required only for Q4K/Q5K (IQ4_NL must take the Q8_0-side rule: even only).

## 3. Patch sketch (function by function)

### 3a. `src/KokkosDNN_GGUF.hpp` (required for any IQ4_NL compute)

1. Enum (68–87): `IQ4_NL = 20,` after `Q8_K = 15,` (numeric order; 20 is the ggml id).
2. `ggml_block_len` (90–112): add `case GgmlType::IQ4_NL:` to the 32-group (with Q4_0…Q8_1).
3. `ggml_type_size` (114–138): `case GgmlType::IQ4_NL: return 18;`.
4. `ggml_type_name` (162–184): `case GgmlType::IQ4_NL: return "IQ4_NL";`.
5. `gguf_can_compute` (156–160): add `|| t == GgmlType::IQ4_NL`.
6. New struct after `BlockQ80` (~254), mirroring ggml field order:
   `struct BlockIQ4NL { uint16_t d; uint8_t qs[16]; };`
   + `static_assert(sizeof(BlockIQ4NL) == 18, ...)` + a drift assert in the 275–282 group (requires the mirror struct in the int8 header).
7. New table + block dequantizer (near 324–327):
   `KOKKOS_INLINE_FUNCTION` LUT as `constexpr int8_t kIq4NlValues[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};`
   `dequant_iq4nl_block(const BlockIQ4NL* b, float* out /*32*/)`:
   `d = fp16_to_float(b->d); for j<16 { out[j] = d*LUT[qs[j]&0xF]; out[j+16] = d*LUT[qs[j]>>4]; }`.
8. `dequant_tensor_to_float` (778–846): new `case GgmlType::IQ4_NL:` with `float tmp[32]` loop over `numel/32` (same shape as the Q8_0 case 802–811).
9. `quant_elem` (892–968): new case —
   block = `row + (k/32)*sizeof(BlockIQ4NL)`; `l = k%32`;
   `nib = l<16 ? qs[l]&0xF : qs[l-16]>>4`;
   `return fp16_to_float(b->d) * LUT[nib];`
10. Grouped path — `quant_group_dot` (982–1094) new `case Q4_K`-style branch with gs=32 (one block = one group; scale hoisted once per 32 elements, exactly like Q8_0 1083–1090); `quant_can_group` (1099–1107): `case GgmlType::IQ4_NL: gs = 32; return K % 32 == 0;`. Groups never straddle a block (32 % 32 == 0), same argument as the header comment 970–980.
11. Decode-unit path —
    `quant_unit_count` (1153–1161): add IQ4_NL → `cols/32`;
    `quant_unit_load_x` (1213–1252): route IQ4_NL with the contiguous-32 branch (same as Q8_0, `k = c*32`);
    new `quant_iq4nl_dot_x`/`quant_iq4nl_chunk_dot` modeled on `quant_q80_dot_x` (1406–1434) but with 2× `QuantU16x8` loads over the 16 `qs` bytes + LUT lookup instead of signed bytes (4 FMA chains, same structure);
    register in `quant_unit_dot` (1556–1569), `quant_unit_dot_x` (1573–1584), `quant_can_unit` (1899–1907, `K % 32 == 0`).
    Load-width note: use `quant_load_u16x8`, never `QuantU8x16` 128-bit loads — 18-byte blocks are only 2-aligned (see §1).
12. `dequant_rows_iq4nl` (new, near 2769–2788) + `dequant_to_dense` case (after Q8_0 2895–2905) with `C % 32` guard. VEC logic is unchanged (32 ≥ VEL always, so the vector path applies whenever the row is 16-aligned).
13. `quant_check` message (1998): extend list to `F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL`.
14. MoE `block_aligned` (2369–2377): add `case GgmlType::IQ4_NL: return n % 32 == 0;`. Justification: IQ4_NL blocks are fully independent 32-wide units (no cross-block scale pairing as in K-quants), so 32-alignment is sufficient for the unit path; the MoE gate/up/down kernels otherwise consume it via the same `quant_unit_count`/`quant_unit_dot` entries above. (Q8_0 keeps its existing conservative 256 rule — do not touch.)

### 3b. `src/KokkosDNN_GGUFGemvInt8.hpp` (optional second step; decode-speed parity)

1. `Wd` (79): `IQ4NL = 4`; mirror `BlockIQ4NL` (after 102–105) + drift assert in GGUF.hpp.
2. `dtype_ok` (142–150): `case Wd::IQ4NL: return K % 32 == 0;` (Q8_0 rule — no 256 requirement).
3. `mvq_load_unit_window` (281–353): contiguous-32 branch (copy the Q80 arm 338–352; the two halves are `k0`, `k0+16`).
4. `gemv_int8_unit_dot_win` (360–466): new branch modeled on Q80 (457–465): per-element LUT lookup `int8 w = kIq4NlValues[nib]` (nibble from `Dot::load_u16` words), `IntDot::dot4` against the int8 activation window, single FP32 rescale `d * xd`. Values fit int8 by construction (|max| = 127), so signed-`dp4a` semantics match the scalar path exactly (same argument as the header comment 252–256).
5. Alignment gate (739–743): IQ4_NL takes the Q8_0 rule (even `wdata` only, no 16-byte requirement). GGUF.hpp `Impl::quant_int8_dtype` (2504–2514): add the `IQ4_NL → Wd::IQ4NL` mapping.

Explicitly out of scope: quantizer (`quantize_row_*`), MoE prefill `dequant_to_dense` for other dtypes, server/models/example changes.

## 4. Test plan

1. **Table test (host-only, no Kokkos needed):** hard-code the §1 LUT in the test, dequantize a synthetic block with `d = 1.0` and `qs[j] = (j<<4)|j` (covers all 16 codes in both halves); expect `out[j] = out[j+16] = LUT[j]`. Catches nibble-swap and sign bugs.
2. **Reference-vs-ggml:** generate 8 random blocks (seeded), compare `dequant_iq4nl_block` against `dequantize_row_iq4_nl` from `ggml/src/ggml-quants.c` bit-for-bit (both are `fp16→float` + int8→float multiply; require exact equality, as the existing K-quant tests do).
3. **Tiny GEMV vs dequant-then-dense:** random 4×64 IQ4_NL matrix, random BF16/F32 activation; `quant_gemv` vs FP64 reference dot of the dequantized row; tolerance ≤ 1e-5 relative (error sources: fp16 scale rounding only — same as Q8_0 coverage).
4. **Path-consistency:** grouped (`quant_can_group`) vs unit (`quant_can_unit`) vs per-element vs `dequant_to_dense` GEMM on a 256-wide row; all pairs within FP32 reassociation (~1e-6).
5. **Loader:** synthetic GGUF (reuse `KokkosDNN_GGUF_Synthetic.hpp` pattern) with one IQ4_NL tensor: `GgufFile` parse accepts dtype 20, `load_quant_weight` row_bytes = `cols/32*18`, `quant_check` passes.
6. **int8 path (after §3b):** portable `quant_gemv` vs int8-dot `quant_gemv` on the same weights; tolerance = activation-quant noise (~1%, same bar as the existing Q4_K MMVQ parity tests), plus a `dtype_ok` K=32-minimum shape test.
7. **Negative:** dtype-20 tensor with `numel % 32 != 0` still throws in `GgufFile::parse`; unknown ids still throw.

## 5. VRAM math (resident estimate for the Inkling-2-Mini IQ4_NL file)

- Packed-weight density: 4.5 bpw (§1) → packed bytes = `numel_packed × 18/32`. For the quoted file this is **≈ 17.6 GB on disk** (take as given from the file size; verify with a tensor-inventory sum over `GgufFile::tensors()` — sketch: accumulate `ti.nbytes` per dtype).
- Resident extras on top of the 17.6 GB (all small vs the 24 GB budget):
  - norms + biases + any F32/F16/BF16 tensors: load at full width (typically < 0.5 GB total; confirm via inventory);
  - `dequant_to_dense` prefill staging: one BF16 dense copy of the routed experts per layer (transient, caller-owned);
  - KV cache + activations: standard `n_layers × n_kv × d × 2 B × ctx` BF16 terms, unchanged by weight quant;
  - int8-dot activation scratch: `act_q8_bytes(K)` per GEMV call (KBs, caller-owned, reused).
- Headroom check: 24.0 − 17.6 ≈ 6.4 GB for extras + CUDA context + fragmentation — comfortable, versus BF16 (52 GB) and Q8_0 (27.8 GB) which exceed total VRAM before any extras.
- Suggested verification command (post-patch): load the file with `GgufFile`, print per-dtype `Σ nbytes`, and assert `IQ4_NL ≈ 17.6 GB` with the residual in F32/F16/BF16.

## 6. Enum-only edit assessment

NOT applied. Rationale: even the "trivially safe" subset (enum value + `ggml_block_len`/`ggml_type_size` cases) changes `GgufFile::parse` behavior for dtype id 20 (unknown → accepted, then rejected later at `load_quant_weight` via `gguf_can_compute`). That is behavior-neutral for inference results but still a shared-header semantic change while parallel agents are active — per the task brief, the new-file report takes precedence, so this worktree has **zero code edits**.
