// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SMOLLM2_WEIGHTS_HPP
#define KOKKOSDNN_SMOLLM2_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "KokkosDNN_SmolLM2Config.hpp"

namespace KokkosDNN {
namespace SmolLM2 {

// Default compute/storage scalar for the SmolLM2 example: bfloat16, the
// SmolLM2-135M checkpoint dtype (Kokkos::Experimental::bhalf_t).
using DefaultScalar = Kokkos::Experimental::bhalf_t;

// One transformer block's weights (all row-major, device views).
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wq, wk, wv, wo;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wgate, wup, wdown;
  Kokkos::View<Scalar*, ExecSpace> attn_norm, ffn_norm;
};

// Full model weights. Device views in Scalar; the safetensors loader
// converts on-disk BF16/FP16/F32 to Scalar automatically.
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> embed, lm_head;
  Kokkos::View<Scalar*, ExecSpace> final_norm;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;

  void allocate(const Config& cfg);
};

// Allocate all views for a config (uninitialized contents).
template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  embed = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "embed", cfg.vocab_size, cfg.dim);
  lm_head = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "lm_head", cfg.vocab_size, cfg.dim);
  final_norm = Kokkos::View<Scalar*, ExecSpace>("final_norm", cfg.dim);
  blocks.resize(cfg.n_layers);
  const int hq = cfg.n_heads * cfg.head_dim;
  const int hkv = cfg.n_kv_heads * cfg.head_dim;
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b   = blocks[l];
    b.wq      = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wq", hq, cfg.dim);
    b.wk = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wk", hkv,
                                                                 cfg.dim);
    b.wv = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wv", hkv,
                                                                 cfg.dim);
    b.wo = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wo",
                                                                 cfg.dim, hq);
    b.wgate = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wgate", cfg.ffn_dim, cfg.dim);
    b.wup = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wup", cfg.ffn_dim, cfg.dim);
    b.wdown = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wdown", cfg.dim, cfg.ffn_dim);
    b.attn_norm = Kokkos::View<Scalar*, ExecSpace>("attn_norm", cfg.dim);
    b.ffn_norm  = Kokkos::View<Scalar*, ExecSpace>("ffn_norm", cfg.dim);
  }
}

// Fill with small random values (for shape/forward tests without weights).
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load HF safetensors checkpoint(s) into device views.
//   files : one or more .safetensors shards (multi-file checkpoints
//           are all listed; tensors are looked up across shards).
// On-disk dtypes BF16/FP16/F32 are all converted to Scalar. Shards are
// mapped read-only (mmap; bounded chunked reads where mmap is unavailable)
// and each tensor streams from the mapping through a small (~1 MiB) host
// staging buffer — no full-tensor/host-F32 expansion. Loading is lazy
// per-tensor in call order (each tensor converts/copies once, shards stay
// mapped for the call). Throws on missing tensors or shape mismatch.
// On-disk dtypes BF16/FP16/F32 are all converted to Scalar. Throws on
// missing tensors or shape mismatch. Expects HF names:
//   model.embed_tokens.weight, model.layers.{l}.{input_layernorm,
//   self_attn.{q,k,v,o}_proj, post_attention_layernorm,
//   mlp.{gate,up,down}_proj}.weight, model.norm.weight, lm_head.weight
template <class ExecSpace, class Scalar = DefaultScalar>
void load_safetensors(const Config& cfg, const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w);

// Raw FP32 host buffer (row-major) converted into a device matrix view.
template <class ExecSpace, class Scalar>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst);

}  // namespace SmolLM2
}  // namespace KokkosDNN

#endif
