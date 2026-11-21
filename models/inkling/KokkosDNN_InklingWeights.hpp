// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_WEIGHTS_HPP
#define KOKKOSDNN_INKLING_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_InklingConfig.hpp"
#include "KokkosDNN_InklingGgufMap.hpp"

namespace KokkosDNN {
namespace Inkling {

// Default compute/storage scalar for the Inkling example: bfloat16, the
// Inkling-2-Mini checkpoint dtype (Kokkos::Experimental::bhalf_t).
using DefaultScalar = Kokkos::Experimental::bhalf_t;

// One transformer block's weights. 2-D matrices stay packed
// (Gguf::QuantWeight, per-tensor GGUF/safetensors dtype, never expanded on
// the host). Dense tensors (norms, conv kernels, clips, biases, loop gates,
// res queries) are F32-origin device views in Scalar.
//
// Layout notes (see KokkosDNN_InklingGgufMap.hpp for the file inventory):
//   * attn linears/rel/sink/conv-proj stay packed (Gguf::QuantWeight);
//     MoE experts are per-expert packed weights
//     (gate_up_exps_q[e]: [2*I, D], down_exps_q[e]: [D, I]).
//   * dense 2-D views follow the Model math orientation, which for most
//     tensors equals the GGUF rows()/cols() extents (LayoutRight,
//     memcpy-compatible). Three families are stored transposed relative
//     to the file: rel_proj is [R, E] (file [E, R]), the loop control
//     downs are [2*D, R] (file [R, 2*D]) and the loop forget/gain gates
//     are [R, D] (file [D, R]). The loader transposes those on the way
//     in (verified against the HF torch shapes and llama.cpp usage).
//   * attn_res_q is [nq, d_model] with nq = 0/1/3 (layer 0 has none;
//     res-block boundary layers have three).
//   * attn_norm exists only on local layers; the four global dynamicconv
//     layers carry k/v_conv_proj + q/k/v_norm instead.
//   * the 13 loop_* tensors exist only on the loop layers [6, 18).
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  // Packed quant projections.
  Gguf::QuantWeight<ExecSpace> wgate_attn, wq, wk, wv, wo;
  Gguf::QuantWeight<ExecSpace> wrel, wsink;
  Gguf::QuantWeight<ExecSpace> k_conv_proj, v_conv_proj;
  // Per-expert packed MoE weights (expert-major slices sharing one load).
  std::vector<Gguf::QuantWeight<ExecSpace>> gate_up_exps_q, down_exps_q;
  // Dense 1-D vectors.
  Kokkos::View<Scalar*, ExecSpace> attn_norm;
  Kokkos::View<Scalar*, ExecSpace> q_norm, k_norm, v_norm;
  Kokkos::View<Scalar*, ExecSpace> post_attn_norm, router_bias;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mlp_res_q;
  // Dense 2-D matrices (LayoutRight [rows, cols]).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> k_conv, v_conv;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> k_conv_static,
      v_conv_static;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> post_attn_conv,
      post_mlp_conv;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> q_scale, rel_proj,
      rel_scale, attn_res_q;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> router;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> loop_q_down,
      loop_q_forget, loop_q_gain, loop_k_down,
      loop_k_forget, loop_k_gain, loop_v_down,
      loop_v_forget, loop_v_gain, loop_mlp_down,
      loop_mlp_forget, loop_mlp_gain, msg_query;
};

// Full model weights. Packed 2-D matrices stay packed (Gguf::QuantWeight,
// one entry per tensor, dtype fixed at load time); dense tensors are device
// views in Scalar. The lm_head is untied: the checkpoint carries a separate
// output tensor and the loader never aliases the embedding table.
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Gguf::QuantWeight<ExecSpace> tok_emb, out_w;
  Kokkos::View<Scalar*, ExecSpace> tok_norm, out_norm;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> out_res_q;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;

  void allocate(const Config& cfg);
};

// Size the blocks vector and allocate the dense views for a config. Packed
// 2-D matrices are NOT allocated here (their dtype is only known at load
// time): loaders and random_init fill each QuantWeight entry. Views for
// tensors absent on a layer (attn_norm vs inp_norms, loop gates,
// attn_res_query on layer 0) stay default-constructed (size 0).
template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  const size_t D = (size_t)cfg.d_model;
  const size_t E = (size_t)cfg.n_experts;
  tok_norm = Kokkos::View<Scalar*, ExecSpace>("tok_norm", D);
  out_norm     = Kokkos::View<Scalar*, ExecSpace>("out_norm", D);
  out_res_q =
      Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("out_res_q", 1,
                                                             D);
  blocks.resize((size_t)cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b      = blocks[(size_t)l];
    const bool lc = cfg.is_local(l);
    const bool lp = is_loop_layer(cfg, l);
    if (lc)
      b.attn_norm = Kokkos::View<Scalar*, ExecSpace>("attn_norm", D);
    else {
      b.q_norm =
          Kokkos::View<Scalar*, ExecSpace>("q_norm", D);
      b.k_norm =
          Kokkos::View<Scalar*, ExecSpace>("k_norm", D);
      b.v_norm =
          Kokkos::View<Scalar*, ExecSpace>("v_norm", D);
    }
    b.post_attn_norm      = Kokkos::View<Scalar*, ExecSpace>("post_attn_norm", D);
    b.mlp_res_q = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "mlp_res_q", 1, D);
    b.router_bias   = Kokkos::View<Scalar*, ExecSpace>("router_bias", E);
    auto mat = [&](const char* label, size_t rows, size_t cols) {
      return Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
          label, rows, cols);
    };
    const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    const size_t K   = (size_t)cfg.sconv_kernel;
    b.k_conv   = mat("k_conv", hkv, K);
    b.v_conv   = mat("v_conv", hkv, K);
    b.k_conv_static = mat("k_conv_static", K, hkv);
    b.v_conv_static = mat("v_conv_static", K, hkv);
    b.post_attn_conv = mat("post_attn_conv", D, K);
    b.q_scale =
        mat("q_scale", (size_t)cfg.n_heads, (size_t)cfg.head_dim);
    // Model orientation is [R, E] (the file stores [E, R]); the loader
    // transposes on the way in.
    b.rel_proj =
        mat("rel_proj", (size_t)cfg.d_rel, (size_t)cfg.rel_extent);
    b.rel_scale =
        mat("rel_scale", (size_t)cfg.n_heads, (size_t)cfg.d_rel);
    const int nq = attn_res_query_count(cfg, l);
    b.attn_res_q =
        mat("attn_res_q", (size_t)nq, D);
    b.router = mat("router", E, D);
    b.post_mlp_conv = mat("post_mlp_conv", D, K);
    if (lp) {
      const size_t G = (size_t)cfg.gate_rank;
      // Model orientation is the transpose of the file orientation; the
      // loader transposes on the way in.
      b.loop_q_down = mat("loop_q_down", 2 * D, G);
      b.loop_q_forget = mat("loop_q_forget", G, D);
      b.loop_q_gain   = mat("loop_q_gain", G, D);
      b.loop_k_down = mat("loop_k_down", 2 * D, G);
      b.loop_k_forget = mat("loop_k_forget", G, D);
      b.loop_k_gain   = mat("loop_k_gain", G, D);
      b.loop_v_down = mat("loop_v_down", 2 * D, G);
      b.loop_v_forget = mat("loop_v_forget", G, D);
      b.loop_v_gain   = mat("loop_v_gain", G, D);
      b.loop_mlp_down = mat("loop_mlp_down", 2 * D, G);
      b.loop_mlp_forget = mat("loop_mlp_forget", G, D);
      b.loop_mlp_gain   = mat("loop_mlp_gain", G, D);
      b.msg_query = mat("msg_query", K, D);
    }
  }
}

// Fill with small random values (for shape/forward tests without weights).
// 2-D matrices become F32 packed QuantWeights (raw-float packing is exact,
// so quant_gemm() on them matches a dense reference bit-for-bit). Dense
// 1-D/2-D views are filled with small random values.
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load an Inkling 2 Mini GGUF checkpoint (single .gguf file) into packed
// device weights (Gguf::QuantWeight, per-tensor dtype; the Q8_0/IQ4_NL file
// stays quantized, so VRAM scales with the packed payload, not an F32
// expansion). Accepted 2-D dtypes are the Gguf::gguf_can_compute() set
// (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL; anything else throws), except
// the two MoE expert tensors, which must follow the file convention
// (IQ4_NL on layers [0, n_layers - 4), Q8_0 on the last four). Dense F32
// tensors accept F32/F16/BF16 on disk. Throws on missing tensors, shape
// mismatch, or GGUF metadata that disagrees with cfg.
//
// Host footprint: the file is mapped read-only (::mmap via Gguf::GgufFile;
// mmap failure throws) and each tensor's payload is deep-copied to the
// device straight out of the mapping in row chunks (unmanaged host views),
// so the host never stages a full-tensor copy: peak extra host RAM is one
// chunk (~1 MiB), not the 17.6 GB checkpoint.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_gguf(const Config& cfg, const std::string& path,
               Weights<ExecSpace, Scalar>& w);

// List all tensor names stored in a GGUF file (header only, no tensor data
// is read).
std::vector<std::string> list_gguf_tensors(const std::string& path);

// Load HF safetensors checkpoint(s) into packed device weights.
//   files : one or more .safetensors shards (tensors are looked up across
//           shards; later shards win on duplicate names).
// On-disk dtypes BF16/FP16/F32 are packed preserving the dtype (identity
// packing; only dtypes in Gguf::gguf_can_compute() are accepted — the same
// bar as the GGUF path, so IQ4_NL/Q8_0-quantized shards load too). Expert
// tensors must be expert-major with numel E*out*in (any rank with that
// numel; the loader stores them as one packed [rows, cols] matrix per the
// GgufMap convention). Shards are mapped read-only (::mmap) and each tensor
// streams from the mapping in small (~1 MiB) host chunks — no full-tensor
// host expansion. Throws on missing tensors or shape mismatch.
//
// Expected HF names (assumed layout derived 1:1 from the GGUF leaves; there
// is no public HF safetensors reference — the GGUF is the canonical
// artifact):
//   model.embed_tokens.weight, model.embed_norm.weight,
//   model.output.weight, model.output_norm.weight,
//   model.output_res_query.weight,
//   model.layers.{l}.attn_{gate,q,k,v,output,rel,sink_proj,
//     k_conv,v_conv,post_conv,q_clip,rel_bias,rel_clip,norm,
//     q_inp_norm,k_inp_norm,v_inp_norm,res_query}.weight,
//   model.layers.{l}.attn_{k,v}_conv_proj.weight (dyn-conv layers only),
//   model.layers.{l}.ffn_{gate_up_exps,down_exps,gate_inp,norm,
//     post_conv,res_query}.weight, model.layers.{l}.exp_probs_b.bias,
//   model.layers.{l}.loop_{q,k,v,ffn}_ctrl_{down,forget,gain}.weight,
//   model.layers.{l}.loop_msg_query.weight
// Use list_checkpoint_tensors() to inspect a partial download.
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

// Raw FP32 host buffer converted into a device vector view.
template <class ExecSpace, class Scalar>
void copy_host_vec_to_device(
    const std::vector<float>& host,
    const Kokkos::View<Scalar*, ExecSpace>& dst);

}  // namespace Inkling
}  // namespace KokkosDNN

#endif
