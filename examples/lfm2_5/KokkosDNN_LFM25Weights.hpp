#ifndef KOKKOSDNN_LFM25_WEIGHTS_HPP
#define KOKKOSDNN_LFM25_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_LFM25Config.hpp"

namespace KokkosDNN {
namespace LFM25 {

// Default compute/storage scalar for the LFM2.5 example: bfloat16, the
// LFM2.5 GGUF checkpoint dtype (Kokkos::Experimental::bhalf_t).
using DefaultScalar = Kokkos::Experimental::bhalf_t;

// One hybrid block's weights.
//
// LFM2.5 blocks are either full-attention or short-convolution ("conv")
// mixer layers (Config::is_attention), each followed by either a dense
// SwiGLU FFN (leading layers, l < Config::n_dense_layers) or a sparse MoE
// FFN (remaining layers). Only the active branch is allocated:
//
//   attention branch : wq/wk/wv/wo + per-head q_norm/k_norm
//   conv branch      : in_proj [3*dim, dim], conv_dw [dim, K],
//                      out_proj [dim, dim]  (K = Config::conv_L_cache)
//   dense FFN        : wgate/wup [ffn_dim, dim], wdown [dim, ffn_dim]
//   MoE FFN          : gate_inp router [n_experts, dim], per-expert packed
//                      gate_exps/up_exps/down_exps, exp_bias [n_experts]
//
// Dense views hold dequantized values in Scalar (the GGUF loader converts
// F32/F16/BF16/Q8_0/Q4_K/Q6_K on the host). MoE experts stay packed as
// QuantWeight (one per expert) and run through quant_gemm() with on-the-fly
// dequantization, so expert weights are never expanded on the host.
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  bool is_attention = true;
  bool is_moe       = false;
  // Attention branch (valid iff is_attention).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wq, wk, wv, wo;
  Kokkos::View<Scalar*, ExecSpace> q_norm, k_norm;  // per-head, [head_dim]
  // Short-conv branch (valid iff !is_attention).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> in_proj, conv_dw,
      out_proj;
  // Shared (both kinds).
  Kokkos::View<Scalar*, ExecSpace> attn_norm, ffn_norm;
  // Dense FFN (valid iff !is_moe).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wgate, wup, wdown;
  // MoE FFN (valid iff is_moe).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> gate_inp;  // [E, dim]
  std::vector<Gguf::QuantWeight<ExecSpace>> gate_exps, up_exps,
      down_exps;  // one packed weight per expert
  Kokkos::View<Scalar*, ExecSpace> exp_bias;  // [E]; empty when unused
};

// Full model weights. Device views in Scalar (dense) / packed experts;
// the GGUF loader converts on-disk dtypes automatically.
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> embed, lm_head;
  Kokkos::View<Scalar*, ExecSpace> final_norm;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;

  void allocate(const Config& cfg);
};

// Allocate all views for a config (uninitialized contents). Branch
// selection (attention vs conv, dense vs MoE) follows cfg.is_attention(l)
// and cfg.n_dense_layers, matching llama.cpp (`is_recr`, `is_moe_layer`).
template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  cfg.validate();
  embed = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "embed", cfg.vocab_size, cfg.dim);
  lm_head = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "lm_head", cfg.vocab_size, cfg.dim);
  final_norm = Kokkos::View<Scalar*, ExecSpace>("final_norm", cfg.dim);
  blocks.resize(cfg.n_layers);
  const int hq  = cfg.n_heads * cfg.head_dim;
  const int hkv = cfg.n_kv_heads * cfg.head_dim;
  const int K   = cfg.conv_L_cache;
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b      = blocks[l];
    b.is_attention = cfg.is_attention(l);
    b.is_moe       = (l >= cfg.n_dense_layers) && (cfg.n_experts > 0);
    if (b.is_attention) {
      b.wq = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wq", hq,
                                                                    cfg.dim);
      b.wk = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wk", hkv,
                                                                    cfg.dim);
      b.wv = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("wv", hkv,
                                                                    cfg.dim);
      b.wo = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "wo", cfg.dim, hq);
      b.q_norm =
          Kokkos::View<Scalar*, ExecSpace>("q_norm", cfg.head_dim);
      b.k_norm =
          Kokkos::View<Scalar*, ExecSpace>("k_norm", cfg.head_dim);
    } else {
      b.in_proj = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "in_proj", 3 * cfg.dim, cfg.dim);
      b.conv_dw = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "conv_dw", cfg.dim, K);
      b.out_proj = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "out_proj", cfg.dim, cfg.dim);
    }
    b.attn_norm = Kokkos::View<Scalar*, ExecSpace>("attn_norm", cfg.dim);
    b.ffn_norm  = Kokkos::View<Scalar*, ExecSpace>("ffn_norm", cfg.dim);
    if (!b.is_moe) {
      b.wgate = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "wgate", cfg.ffn_dim, cfg.dim);
      b.wup = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "wup", cfg.ffn_dim, cfg.dim);
      b.wdown = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "wdown", cfg.dim, cfg.ffn_dim);
    } else {
      const size_t E = (size_t)cfg.n_experts;
      b.gate_inp = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          "gate_inp", E, cfg.dim);
      // Packed experts are sized here (empty payloads); random_init /
      // load_gguf fill them.
      b.gate_exps.resize(E);
      b.up_exps.resize(E);
      b.down_exps.resize(E);
      if (cfg.use_expert_bias)
        b.exp_bias = Kokkos::View<Scalar*, ExecSpace>("exp_bias", E);
    }
  }
}

// Fill with small random values (for shape/forward tests without weights).
// Experts become F32 packed QuantWeights (raw-float packing, exact) so the
// MoE path runs through quant_gemm() even in tests.
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load a llama.cpp LFM2/LFM2-MoE GGUF checkpoint into device views.
// Dense tensors are dequantized on the host (F32/F16/BF16/Q8_0/Q4_K/Q6_K
// all convert; anything else throws) and stored in Scalar; MoE experts
// stay packed per expert. Throws on missing tensors or shape mismatch.
// Expected GGUF names (llama.cpp `tn()` convention, ".weight"/".bias"
// suffix; the loader also accepts the bare stems):
//   token_embd, token_embd_norm (`output_norm` accepted as legacy alias),
//   output (tied fallback onto token_embd when
//     cfg.tie_word_embeddings and output is absent),
//   blk.{l}.attn_norm, blk.{l}.ffn_norm,
//   attention layers: blk.{l}.attn_q/k/v/output,
//     blk.{l}.attn_q_norm/k_norm,
//   conv layers: blk.{l}.shortconv.in_proj/conv/out_proj,
//   dense layers: blk.{l}.ffn_gate/up/down,
//   MoE layers: blk.{l}.ffn_gate_inp,
//     blk.{l}.ffn_gate_exps/up_exps/down_exps (stacked 3-D),
//     blk.{l}.exp_probs_b (bias suffix; `ffn_exp_probs_b` accepted as
//     legacy alias).
template <class ExecSpace, class Scalar = DefaultScalar>
void load_gguf(const Config& cfg, const std::string& path,
               Weights<ExecSpace, Scalar>& w);

// Raw FP32 host buffer (row-major) converted into a device matrix view.
template <class ExecSpace, class Scalar>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst);

}  // namespace LFM25
}  // namespace KokkosDNN

#endif
