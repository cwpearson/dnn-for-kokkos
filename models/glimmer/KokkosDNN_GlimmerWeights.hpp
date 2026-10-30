// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GLIMMER_WEIGHTS_HPP
#define KOKKOSDNN_GLIMMER_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_GlimmerConfig.hpp"

namespace KokkosDNN {
namespace Glimmer {

// Default compute/storage scalar for the Glimmer example: bfloat16, the
// Muse-Glimmer-30B-heretic checkpoint dtype
// (Kokkos::Experimental::bhalf_t).
using DefaultScalar = Kokkos::Experimental::bhalf_t;

// One transformer block's weights: 2-D matrices stay packed
// (Gguf::QuantWeight, per-tensor GGUF/safetensors dtype, never expanded on
// the host); the four RMSNorms are dense device vectors in Scalar.
//
// Glimmer differences vs the Llama block: explicit head_dim 128 (q/k/v/o
// shapes derive from it, not from dim / n_heads), a per-layer attention
// output gate (self_attn.gate_proj), a scaleless per-head QK RMSNorm (no
// stored weight: with_scale=False in the HF module), and four RMSNorms —
// input / post-attention (post eps) / pre-FFN / post-FFN (post eps) — all of
// the Centered (1 + w) form with zero-initialized raw weights. The v
// projection carries no norm.
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  Gguf::QuantWeight<ExecSpace> wq, wk, wv, wo;
  Gguf::QuantWeight<ExecSpace> wgate_attn;
  Gguf::QuantWeight<ExecSpace> wgate, wup, wdown;
  Kokkos::View<Scalar*, ExecSpace> input_norm, post_attn_norm, pre_ffn_norm,
      post_ffn_norm;
};

// Full model weights. 2-D matrices stay packed (Gguf::QuantWeight, one entry
// per tensor, dtype fixed at load time); 1-D norms are dense device vectors
// in Scalar. The safetensors loader packs on-disk BF16/FP16/F32 preserving
// the dtype (identity packing: unquantized ggml layouts are dense
// row-major). The lm_head is untied (tie_word_embeddings=false): the
// checkpoint carries a separate lm_head tensor and the loader must NOT
// alias the embedding table (a tied config still aliases the packed embed).
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Gguf::QuantWeight<ExecSpace> embed, lm_head;
  Kokkos::View<Scalar*, ExecSpace> final_norm;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;

  void allocate(const Config& cfg);
};

// Size the blocks vector and allocate the dense norm vectors for a config.
// Packed 2-D matrices are NOT allocated here (their dtype is only known at
// load time): loaders and random_init fill each QuantWeight entry.
template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  final_norm = Kokkos::View<Scalar*, ExecSpace>("final_norm", cfg.dim);
  blocks.resize((size_t)cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b             = blocks[(size_t)l];
    b.input_norm      = Kokkos::View<Scalar*, ExecSpace>("input_norm", cfg.dim);
    b.post_attn_norm  = Kokkos::View<Scalar*, ExecSpace>("post_attn_norm", cfg.dim);
    b.pre_ffn_norm    = Kokkos::View<Scalar*, ExecSpace>("pre_ffn_norm", cfg.dim);
    b.post_ffn_norm   = Kokkos::View<Scalar*, ExecSpace>("post_ffn_norm", cfg.dim);
  }
}

// Fill with small random values (for shape/forward tests without weights).
// 2-D matrices become F32 packed QuantWeights (raw-float packing is exact,
// so quant_gemm() on them matches a dense reference bit-for-bit). Centered
// norms are filled with 1.0 (the stored (1 + w) value matching a raw zero
// init); the scaleless norms (embed/QK) store no weight.
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load HF safetensors checkpoint(s) into packed device weights.
//   files : one or more .safetensors shards (tensors are looked up across
//           shards; later shards win on duplicate names).
// On-disk dtypes BF16/FP16/F32 are packed preserving the dtype (identity
// packing; only dtypes in Gguf::gguf_can_compute() are accepted). Shards are
// mapped read-only (mmap; bounded chunked reads where mmap is unavailable)
// and each tensor streams from the mapping through a small (~1 MiB) host
// staging buffer — no full-tensor/host-F32 expansion. Loading is lazy
// per-tensor in call order (each tensor converts/copies once, shards stay
// mapped for the call). Throws on missing tensors or shape mismatch.
//
// Expected HF names (prefix `model.language_model.`, ASSUMPTION confirmed
// against modeling_muse_glimmer.py module paths up to the checkpoint root;
// the `model.language_model.` root itself follows the Gemma4 layout and the
// scaffold DESIGN.md):
//   model.language_model.embed_tokens.weight,
//   model.language_model.norm.weight, lm_head.weight
//   (also accepted: model.language_model.lm_head.weight),
//   model.language_model.layers.{l}.{input_layernorm,
//   self_attn.{q,k,v,o,gate}_proj, post_attention_layernorm,
//   pre_feedforward_layernorm, post_feedforward_layernorm,
//   mlp.{gate,up,down}_proj}.weight
// The scaleless norms (embed_norm, self_attn.qk_norm) carry no weight tensor
// (with_scale=False upstream) and are never looked up. With
// tie_word_embeddings=false a missing lm_head tensor is an error (never
// aliased). Use list_checkpoint_tensors() to inspect a partial download.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_safetensors(const Config& cfg, const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w);

// List all tensor names stored across checkpoint file(s) (header only, no
// tensor data is read). Useful to inspect partial/sharded downloads.
std::vector<std::string> list_checkpoint_tensors(
    const std::vector<std::string>& files);

// Load a llama.cpp Muse-Glimmer GGUF checkpoint (single .gguf file) into
// packed device weights (Gguf::QuantWeight, per-tensor dtype; Q4_K_M files
// stay quantized, so VRAM scales with the packed payload, not an F32
// expansion). Accepted 2-D dtypes are the Gguf::gguf_can_compute() set
// (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K; anything else throws). Throws on
// missing tensors, shape mismatch, or GGUF metadata (logit scale, softcap,
// sliding window + pattern) that disagrees with cfg.
//
// Expected GGUF names (llama.cpp `tn()` convention with the ".weight"
// suffix; bare stems are also accepted):
//   token_embd.weight, output_norm.weight, output.weight,
//   per layer blk.%d.{attn_norm, post_attention_norm, attn_q, attn_k,
//   attn_v, attn_output, attn_q_norm, attn_k_norm, attn_gate, ffn_norm,
//   post_ffw_norm, ffn_gate, ffn_down, ffn_up}.weight
//
// Conversion-trap handling (see conversion/muse_glimmer.py +
// src/models/muse-glimmer.cpp in llama.cpp):
//   * Norm shift: conversion adds +1 to the four layer norms (GGUF rms_norm
//     is x/rms*w while HF CenteredRMSNorm is x/rms*(1+w)); the final norm
//     is untouched (shift 0). The loader subtracts 1 back from the four
//     layer norms so the Model's one_plus path sees HF-centered weights.
//   * Q/K RoPE layout: conversion unpermutes q/k into ggml interleaved
//     (NORM) layout; the loader re-permutes packed rows back into the HF
//     rotate_half layout consumed by rope_neoX_inplace (whole packed-row
//     swaps, no dequantization).
//   * Synthesized QK norms: GGUF attn_q_norm (full qk_scale_factor) /
//     attn_k_norm (ones) absorb the q pre-scale into a *scaled* RMSNorm,
//     while the Model runs a *scaleless* QK norm plus the separate 3.87
//     multiply -- mathematically identical (rms(x)*3.87, rms(x)*1), same
//     eps), so the loader IGNORES both tensors (never looked up).
//   * Metadata: muse-glimmer.{logit_scale, final_logit_softcapping,
//     attention.sliding_window, attention.sliding_window_pattern} are read
//     and VALIDATED against cfg (throw on missing/mismatch), never
//     silently overriding it.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_gguf(const Config& cfg, const std::string& path,
               Weights<ExecSpace, Scalar>& w);

// List all tensor names stored in a GGUF file (header only, no tensor data
// is read). GGUF counterpart of list_checkpoint_tensors().
std::vector<std::string> list_gguf_tensors(const std::string& path);

// Raw FP32 host buffer (row-major) converted into a device matrix view.
template <class ExecSpace, class Scalar>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst);

}  // namespace Glimmer
}  // namespace KokkosDNN

#endif
