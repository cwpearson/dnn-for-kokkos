#ifndef KOKKOSDNN_GEMMA4_WEIGHTS_HPP
#define KOKKOSDNN_GEMMA4_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "KokkosDNN_Gemma4Config.hpp"

namespace KokkosDNN {
namespace Gemma4 {

// Default compute/storage scalar for the Gemma4 example: bfloat16, the
// gemma-4-E2B-it checkpoint dtype (Kokkos::Experimental::bhalf_t).
using DefaultScalar = Kokkos::Experimental::bhalf_t;

// One transformer block's weights (all row-major, device views).
//
// Gemma4 differences vs the Llama block: per-layer head dim (256 sliding /
// 512 full), QK RMSNorms (q_norm every layer, k_norm only on non-shared
// layers), four RMSNorms (input / post-attention / pre-FFN / post-FFN),
// double-wide gated-GeLU MLP on KV-shared layers (12288 vs 6144), the
// per-layer-embedding (PLE) gate/projection pair plus its output norm, and
// a learned scalar `layer_scalar` (host float).
//
// KV-shared layers (idx >= kv_shared_start()) reuse an earlier layer's KV
// cache and never use k/v projections at runtime: wk/wv/k_norm are left
// unallocated for those layers (the checkpoint still stores them as dead
// weight; the loader skips them).
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wq, wk, wv, wo;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wgate, wup, wdown;
  Kokkos::View<Scalar*, ExecSpace> q_norm, k_norm;  // k_norm: non-shared only
  Kokkos::View<Scalar*, ExecSpace> input_norm, post_attn_norm, pre_ffn_norm,
      post_ffn_norm;
  // PLE per-layer path: gate [ple_dim, dim], projection [dim, ple_dim],
  // output norm [dim].
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> ple_gate, ple_proj;
  Kokkos::View<Scalar*, ExecSpace> ple_out_norm;
  float layer_scalar = 1.0f;
};

// Full model weights. Device views in Scalar; the safetensors loader
// converts on-disk BF16/FP16/F32 to Scalar automatically. The lm_head is
// tied: with tie_word_embeddings the loader aliases the embedding table
// (the checkpoint stores no lm_head tensor).
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> embed, lm_head;
  // PLE globals: per-layer token table [ple_vocab, n_layers*ple_dim],
  // context projection [n_layers*ple_dim, dim], context norm [ple_dim].
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> ple_embed, ple_ctx;
  Kokkos::View<Scalar*, ExecSpace> ple_ctx_norm;
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
  ple_embed = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "ple_embed", cfg.ple_vocab, (size_t)cfg.n_layers * (size_t)cfg.ple_dim);
  ple_ctx = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "ple_ctx", (size_t)cfg.n_layers * (size_t)cfg.ple_dim, cfg.dim);
  ple_ctx_norm =
      Kokkos::View<Scalar*, ExecSpace>("ple_ctx_norm", cfg.ple_dim);
  final_norm = Kokkos::View<Scalar*, ExecSpace>("final_norm", cfg.dim);
  blocks.resize(cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b      = blocks[l];
    const int d  = cfg.head_dim(l);
    const int hq = cfg.n_heads * d;
    const int fd = cfg.mlp_dim(l);
    b.wq = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wq", hq,
                                                                  cfg.dim);
    if (!cfg.is_kv_shared(l)) {
      const int hkv = cfg.n_kv_heads * d;
      b.wk          = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "wk", hkv, cfg.dim);
      b.wv = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wv", hkv,
                                                                    cfg.dim);
      b.k_norm = Kokkos::View<Scalar*, ExecSpace>("k_norm", d);
    }
    b.wo = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wo",
                                                                  cfg.dim, hq);
    b.wgate = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wgate", fd, cfg.dim);
    b.wup = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wup", fd, cfg.dim);
    b.wdown = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wdown", cfg.dim, fd);
    b.q_norm = Kokkos::View<Scalar*, ExecSpace>("q_norm", d);
    b.input_norm     = Kokkos::View<Scalar*, ExecSpace>("input_norm", cfg.dim);
    b.post_attn_norm = Kokkos::View<Scalar*, ExecSpace>("post_attn_norm",
                                                        cfg.dim);
    b.pre_ffn_norm = Kokkos::View<Scalar*, ExecSpace>("pre_ffn_norm", cfg.dim);
    b.post_ffn_norm =
        Kokkos::View<Scalar*, ExecSpace>("post_ffn_norm", cfg.dim);
    b.ple_gate = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "ple_gate", cfg.ple_dim, cfg.dim);
    b.ple_proj = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "ple_proj", cfg.dim, cfg.ple_dim);
    b.ple_out_norm =
        Kokkos::View<Scalar*, ExecSpace>("ple_out_norm", cfg.dim);
    b.layer_scalar = 1.0f;
  }
}

// Fill with small random values (for shape/forward tests without weights).
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Hugging Face tensor name for a per-layer tensor (Gemma4 names live under
// `model.language_model.`).
inline std::string layer_tensor(int l, const std::string& leaf) {
  return "model.language_model.layers." + std::to_string(l) + "." + leaf;
}

// Load HF safetensors checkpoint(s) into device views.
//   files : one or more .safetensors shards (tensors are looked up across
//           shards; later shards win on duplicate names).
// On-disk dtypes BF16/FP16/F32 are all converted to Scalar. Shards are
// mapped read-only (mmap; bounded chunked reads where mmap is unavailable)
// and each tensor streams from the mapping through a small (~1 MiB) host
// staging buffer — no full-tensor/host-F32 expansion. Loading is lazy
// per-tensor in call order (each tensor converts/copies once, shards stay
// mapped for the call).
//
// Expected HF names (prefix `model.language_model.`):
//   embed_tokens.weight, embed_tokens_per_layer.weight,
//   per_layer_model_projection.weight, per_layer_projection_norm.weight,
//   norm.weight, layers.{l}.{input_layernorm,
//   self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm,
//   post_attention_layernorm, pre_feedforward_layernorm,
//   post_feedforward_layernorm, mlp.{gate,up,down}_proj,
//   per_layer_input_gate, per_layer_projection, post_per_layer_input_norm,
//   layer_scalar}.weight (layer_scalar has no `.weight` suffix).
// With tie_word_embeddings=true a missing lm_head tensor shares the
// embedding table instead of throwing (the E2B-it checkpoint stores no
// lm_head). Dead k/v (+k_norm) tensors stored for KV-shared layers are
// skipped, not loaded. Throws on missing required tensors or shape
// mismatch; the message names the first missing tensor (use
// list_checkpoint_tensors() to inspect a partial download).
template <class ExecSpace, class Scalar = DefaultScalar>
void load_safetensors(const Config& cfg, const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w);

// List all tensor names stored across checkpoint file(s) (header only, no
// tensor data is read). Useful to inspect partial/sharded downloads.
std::vector<std::string> list_checkpoint_tensors(
    const std::vector<std::string>& files);

// Raw FP32 host buffer (row-major) converted into a device matrix view.
template <class ExecSpace, class Scalar>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst);

}  // namespace Gemma4
}  // namespace KokkosDNN

#endif
