#ifndef KOKKOSDNN_LLAMA_WEIGHTS_HPP
#define KOKKOSDNN_LLAMA_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "llama/KokkosDNN_LlamaConfig.hpp"

namespace KokkosDNN {
namespace Llama {

// One transformer block's weights (all row-major, FP32 device views).
template <class ExecSpace>
struct BlockWeights {
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> wq, wk, wv, wo;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> wgate, wup, wdown;
  Kokkos::View<float*, ExecSpace> attn_norm, ffn_norm;
};

// Full model weights. FP32 device views; the safetensors loader converts
// on-disk BF16/FP16/F32 to float automatically.
template <class ExecSpace>
struct Weights {
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> embed, lm_head;
  Kokkos::View<float*, ExecSpace> final_norm;
  std::vector<BlockWeights<ExecSpace>> blocks;

  void allocate(const Config& cfg);
};

// Allocate all views for a config (uninitialized contents).
template <class ExecSpace>
void Weights<ExecSpace>::allocate(const Config& cfg) {
  embed = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
      "embed", cfg.vocab_size, cfg.dim);
  lm_head = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
      "lm_head", cfg.vocab_size, cfg.dim);
  final_norm = Kokkos::View<float*, ExecSpace>("final_norm", cfg.dim);
  blocks.resize(cfg.n_layers);
  const int hq = cfg.n_heads * cfg.head_dim;
  const int hkv = cfg.n_kv_heads * cfg.head_dim;
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b   = blocks[l];
    b.wq      = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
        "wq", hq, cfg.dim);
    b.wk = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>("wk", hkv,
                                                                 cfg.dim);
    b.wv = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>("wv", hkv,
                                                                 cfg.dim);
    b.wo = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>("wo",
                                                                 cfg.dim, hq);
    b.wgate = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
        "wgate", cfg.ffn_dim, cfg.dim);
    b.wup = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
        "wup", cfg.ffn_dim, cfg.dim);
    b.wdown = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>(
        "wdown", cfg.dim, cfg.ffn_dim);
    b.attn_norm = Kokkos::View<float*, ExecSpace>("attn_norm", cfg.dim);
    b.ffn_norm  = Kokkos::View<float*, ExecSpace>("ffn_norm", cfg.dim);
  }
}

// Fill with small random values (for shape/forward tests without weights).
template <class ExecSpace>
void random_init(const Config& cfg, Weights<ExecSpace>& w, uint64_t seed = 42);

// Load HF safetensors checkpoint(s) into device views.
//   files : one or more .safetensors shards (multi-file Llama checkpoints
//           are all listed; tensors are looked up across shards).
// On-disk dtypes BF16/FP16/F32 are all converted to FP32. Throws on
// missing tensors or shape mismatch. Expects HF names:
//   model.embed_tokens.weight, model.layers.{l}.{input_layernorm,
//   self_attn.{q,k,v,o}_proj, post_attention_layernorm,
//   mlp.{gate,up,down}_proj}.weight, model.norm.weight, lm_head.weight
template <class ExecSpace>
void load_safetensors(const Config& cfg, const std::vector<std::string>& files,
                      Weights<ExecSpace>& w);

// Raw FP32 host buffer (row-major) copied into a device matrix view.
template <class ExecSpace>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& dst);

}  // namespace Llama
}  // namespace KokkosDNN

#endif
