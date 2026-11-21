// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_GGUF_MAP_HPP
#define KOKKOSDNN_INKLING_GGUF_MAP_HPP

// Tensor-name mapping for the Inkling 2 Mini GGUF checkpoint.
//
// The file (704 tensors: 2-D Q8_0 linears, 3-D IQ4_NL/Q8_0 MoE experts,
// F32 norms/convs/biases/loop gates) follows the llama.cpp `tn()` leaf
// convention with a ".weight" suffix (".bias" for the single expert-bias
// vector): globals `token_embd`, `token_embd_norm`, `output`,
// `output_norm`, `output_res_query`, and per-block `blk.{l}.{leaf}`.
//
// Per-block layout (l = 0..23), verified against the tensor inventory:
//   * every layer: attn_{gate,q,k,v,output,rel,sink_proj}, attn_{k,v}_conv,
//     attn_post_conv, attn_{q,rel}_clip, attn_rel_bias, ffn MoE +
//     {gate_inp, exp_probs_b, norm, post_conv, res_query};
//   * attn_norm on the 20 local layers; the 4 global dynamicconv layers
//     (5, 11, 17, 23 — exactly the complement of local_layer_ids) carry
//     attn_{k,v}_conv_proj + attn_{q,k,v}_inp_norm instead;
//   * attn_res_query on layers 1..23 (1 query; 3 on the res-block boundary
//     layers 5/11/17/23, i.e. (l+1) % attn_res_block == 0; absent on l=0);
//   * loop_{q,k,v,ffn}_ctrl_{down,forget,gain} + loop_msg_query on the 12
//     loop layers [loop_start, loop_end) = [6, 18);
//   * MoE experts IQ4_NL on layers 0..19, Q8_0 on the last four (20..23).
//
// Unlike Glimmer (which ships a custom prefetching mapper), the Inkling
// loader reads through Gguf::GgufFile (src/KokkosDNN_GGUF.hpp, already
// mmap-backed and streaming-friendly), so this header is a pure name <->
// (kind, layer) map plus the shape/presence rules the loader enforces.
// Row/column counts below use the GGUF (rows(), cols()) convention:
// rows = dims[1], cols = dims[0] (dims[0] fastest-varying); MoE `rows`
// already include the expert dim (rows = n_experts * out_features) since
// the payload is expert-major contiguous.

#include <cstddef>
#include <stdexcept>
#include <string>

#include "KokkosDNN_InklingConfig.hpp"

namespace KokkosDNN {
namespace Inkling {

// Every tensor leaf of the checkpoint. Block leaves keep their
// `blk.{l}.` prefix implicit (see TensorId); globals stand alone.
enum class Tensor {
  TokenEmbd,
  TokenEmbdNorm,
  Output,
  OutputNorm,
  OutputResQuery,
  // Block attention projections (packed quant linears).
  AttnGate,
  AttnQ,
  AttnK,
  AttnV,
  AttnOutput,
  AttnRel,
  AttnSinkProj,
  AttnKConvProj,
  AttnVConvProj,
  // Block attention dense (F32) tensors.
  AttnNorm,
  AttnQInpNorm,
  AttnKInpNorm,
  AttnVInpNorm,
  AttnKConv,
  AttnVConv,
  AttnPostConv,
  AttnQClip,
  AttnRelBias,
  AttnRelClip,
  AttnResQuery,
  // Block MoE (experts packed quant, rest dense F32).
  FfnGateUpExps,
  FfnDownExps,
  FfnGateInp,
  ExpProbsB,
  FfnNorm,
  FfnPostConv,
  FfnResQuery,
  // Loop gates (dense F32, loop layers only).
  LoopQCtrlDown,
  LoopQCtrlForget,
  LoopQCtrlGain,
  LoopKCtrlDown,
  LoopKCtrlForget,
  LoopKCtrlGain,
  LoopVCtrlDown,
  LoopVCtrlForget,
  LoopVCtrlGain,
  LoopFfnCtrlDown,
  LoopFfnCtrlForget,
  LoopFfnCtrlGain,
  LoopMsgQuery,
};

// A resolved tensor: `kind` plus the block index (`layer == -1` for the
// five global tensors).
struct TensorId {
  Tensor kind;
  int layer = -1;  // -1: global tensor
};

// Leaf stem without the `blk.{l}.` prefix or the suffix.
inline const char* tensor_stem(Tensor kind) {
  switch (kind) {
    case Tensor::TokenEmbd: return "token_embd";
    case Tensor::TokenEmbdNorm: return "token_embd_norm";
    case Tensor::Output: return "output";
    case Tensor::OutputNorm: return "output_norm";
    case Tensor::OutputResQuery: return "output_res_query";
    case Tensor::AttnGate: return "attn_gate";
    case Tensor::AttnQ: return "attn_q";
    case Tensor::AttnK: return "attn_k";
    case Tensor::AttnV: return "attn_v";
    case Tensor::AttnOutput: return "attn_output";
    case Tensor::AttnRel: return "attn_rel";
    case Tensor::AttnSinkProj: return "attn_sink_proj";
    case Tensor::AttnKConvProj: return "attn_k_conv_proj";
    case Tensor::AttnVConvProj: return "attn_v_conv_proj";
    case Tensor::AttnNorm: return "attn_norm";
    case Tensor::AttnQInpNorm: return "attn_q_inp_norm";
    case Tensor::AttnKInpNorm: return "attn_k_inp_norm";
    case Tensor::AttnVInpNorm: return "attn_v_inp_norm";
    case Tensor::AttnKConv: return "attn_k_conv";
    case Tensor::AttnVConv: return "attn_v_conv";
    case Tensor::AttnPostConv: return "attn_post_conv";
    case Tensor::AttnQClip: return "attn_q_clip";
    case Tensor::AttnRelBias: return "attn_rel_bias";
    case Tensor::AttnRelClip: return "attn_rel_clip";
    case Tensor::AttnResQuery: return "attn_res_query";
    case Tensor::FfnGateUpExps: return "ffn_gate_up_exps";
    case Tensor::FfnDownExps: return "ffn_down_exps";
    case Tensor::FfnGateInp: return "ffn_gate_inp";
    case Tensor::ExpProbsB: return "exp_probs_b";
    case Tensor::FfnNorm: return "ffn_norm";
    case Tensor::FfnPostConv: return "ffn_post_conv";
    case Tensor::FfnResQuery: return "ffn_res_query";
    case Tensor::LoopQCtrlDown: return "loop_q_ctrl_down";
    case Tensor::LoopQCtrlForget: return "loop_q_ctrl_forget";
    case Tensor::LoopQCtrlGain: return "loop_q_ctrl_gain";
    case Tensor::LoopKCtrlDown: return "loop_k_ctrl_down";
    case Tensor::LoopKCtrlForget: return "loop_k_ctrl_forget";
    case Tensor::LoopKCtrlGain: return "loop_k_ctrl_gain";
    case Tensor::LoopVCtrlDown: return "loop_v_ctrl_down";
    case Tensor::LoopVCtrlForget: return "loop_v_ctrl_forget";
    case Tensor::LoopVCtrlGain: return "loop_v_ctrl_gain";
    case Tensor::LoopFfnCtrlDown: return "loop_ffn_ctrl_down";
    case Tensor::LoopFfnCtrlForget: return "loop_ffn_ctrl_forget";
    case Tensor::LoopFfnCtrlGain: return "loop_ffn_ctrl_gain";
    case Tensor::LoopMsgQuery: return "loop_msg_query";
  }
  throw std::invalid_argument("tensor_stem: unknown Tensor kind");
}

// The checkpoint suffix: ".bias" only for the expert gate bias.
inline const char* tensor_suffix(Tensor kind) {
  return (kind == Tensor::ExpProbsB) ? ".bias" : ".weight";
}

namespace Impl {

inline bool parse_stem(const std::string& stem, Tensor& kind) {
  if (stem == "token_embd") return kind = Tensor::TokenEmbd, true;
  if (stem == "token_embd_norm") return kind = Tensor::TokenEmbdNorm, true;
  if (stem == "output") return kind = Tensor::Output, true;
  if (stem == "output_norm") return kind = Tensor::OutputNorm, true;
  if (stem == "output_res_query")
    return kind = Tensor::OutputResQuery, true;
  if (stem == "attn_gate") return kind = Tensor::AttnGate, true;
  if (stem == "attn_q") return kind = Tensor::AttnQ, true;
  if (stem == "attn_k") return kind = Tensor::AttnK, true;
  if (stem == "attn_v") return kind = Tensor::AttnV, true;
  if (stem == "attn_output") return kind = Tensor::AttnOutput, true;
  if (stem == "attn_rel") return kind = Tensor::AttnRel, true;
  if (stem == "attn_sink_proj") return kind = Tensor::AttnSinkProj, true;
  if (stem == "attn_k_conv_proj")
    return kind = Tensor::AttnKConvProj, true;
  if (stem == "attn_v_conv_proj")
    return kind = Tensor::AttnVConvProj, true;
  if (stem == "attn_norm") return kind = Tensor::AttnNorm, true;
  if (stem == "attn_q_inp_norm") return kind = Tensor::AttnQInpNorm, true;
  if (stem == "attn_k_inp_norm") return kind = Tensor::AttnKInpNorm, true;
  if (stem == "attn_v_inp_norm") return kind = Tensor::AttnVInpNorm, true;
  if (stem == "attn_k_conv") return kind = Tensor::AttnKConv, true;
  if (stem == "attn_v_conv") return kind = Tensor::AttnVConv, true;
  if (stem == "attn_post_conv") return kind = Tensor::AttnPostConv, true;
  if (stem == "attn_q_clip") return kind = Tensor::AttnQClip, true;
  if (stem == "attn_rel_bias") return kind = Tensor::AttnRelBias, true;
  if (stem == "attn_rel_clip") return kind = Tensor::AttnRelClip, true;
  if (stem == "attn_res_query") return kind = Tensor::AttnResQuery, true;
  if (stem == "ffn_gate_up_exps") return kind = Tensor::FfnGateUpExps, true;
  if (stem == "ffn_down_exps") return kind = Tensor::FfnDownExps, true;
  if (stem == "ffn_gate_inp") return kind = Tensor::FfnGateInp, true;
  if (stem == "exp_probs_b") return kind = Tensor::ExpProbsB, true;
  if (stem == "ffn_norm") return kind = Tensor::FfnNorm, true;
  if (stem == "ffn_post_conv") return kind = Tensor::FfnPostConv, true;
  if (stem == "ffn_res_query") return kind = Tensor::FfnResQuery, true;
  if (stem == "loop_q_ctrl_down") return kind = Tensor::LoopQCtrlDown, true;
  if (stem == "loop_q_ctrl_forget")
    return kind = Tensor::LoopQCtrlForget, true;
  if (stem == "loop_q_ctrl_gain") return kind = Tensor::LoopQCtrlGain, true;
  if (stem == "loop_k_ctrl_down") return kind = Tensor::LoopKCtrlDown, true;
  if (stem == "loop_k_ctrl_forget")
    return kind = Tensor::LoopKCtrlForget, true;
  if (stem == "loop_k_ctrl_gain") return kind = Tensor::LoopKCtrlGain, true;
  if (stem == "loop_v_ctrl_down") return kind = Tensor::LoopVCtrlDown, true;
  if (stem == "loop_v_ctrl_forget")
    return kind = Tensor::LoopVCtrlForget, true;
  if (stem == "loop_v_ctrl_gain") return kind = Tensor::LoopVCtrlGain, true;
  if (stem == "loop_ffn_ctrl_down")
    return kind = Tensor::LoopFfnCtrlDown, true;
  if (stem == "loop_ffn_ctrl_forget")
    return kind = Tensor::LoopFfnCtrlForget, true;
  if (stem == "loop_ffn_ctrl_gain")
    return kind = Tensor::LoopFfnCtrlGain, true;
  if (stem == "loop_msg_query") return kind = Tensor::LoopMsgQuery, true;
  return false;
}

}  // namespace Impl

// Map a GGUF tensor name to (kind, layer). Accepts the ".weight"/".bias"
// suffixed form and the bare stem. Returns false for unknown names (the
// loader reports those as unexpected, never silently skipping).
inline bool parse_gguf_name(const std::string& name, TensorId& out) {
  std::string stem = name;
  const std::string wsu(".weight"), bsu(".bias");
  if (stem.size() > wsu.size() &&
      stem.compare(stem.size() - wsu.size(), wsu.size(), wsu) == 0)
    stem.resize(stem.size() - wsu.size());
  else if (stem.size() > bsu.size() &&
           stem.compare(stem.size() - bsu.size(), bsu.size(), bsu) == 0)
    stem.resize(stem.size() - bsu.size());
  Tensor kind;
  if (stem.compare(0, 4, "blk.") == 0) {
    size_t dot = stem.find('.', 4);
    if (dot == std::string::npos) return false;
    int layer = 0;
    try {
      size_t n = 0;
      long v   = std::stol(stem.substr(4, dot - 4), &n);
      if (n != dot - 4 || v < 0 || v > 1000000) return false;
      layer = (int)v;
    } catch (...) {
      return false;
    }
    if (!Impl::parse_stem(stem.substr(dot + 1), kind)) return false;
    out.kind  = kind;
    out.layer = layer;
    return true;
  }
  if (!Impl::parse_stem(stem, kind)) return false;
  out.kind  = kind;
  out.layer = -1;
  return true;
}

// Canonical checkpoint name for a resolved tensor (diagnostics and the
// safetensors-name table share these stems).
inline std::string full_name(const TensorId& id) {
  std::string n;
  if (id.layer >= 0) n = "blk." + std::to_string(id.layer) + ".";
  n += tensor_stem(id.kind);
  n += tensor_suffix(id.kind);
  return n;
}

// ---- Presence rules (config-dependent) ----

// Loop layers carry the 13 loop_* tensors.
inline bool is_loop_layer(const Config& cfg, int layer) {
  return layer >= cfg.loop_start && layer < cfg.loop_end;
}

// Global dynamicconv layers carry k/v_conv_proj + q/k/v_inp_norm instead
// of attn_norm (exactly the complement of local_layer_ids).
inline bool is_dyn_conv_layer(const Config& cfg, int layer) {
  return !cfg.is_local(layer);
}

// Number of attn_res_query vectors on a layer: none on layer 0, three on
// res-block boundary layers ((l + 1) % attn_res_block == 0), one elsewhere.
inline int attn_res_query_count(const Config& cfg, int layer) {
  if (layer == 0) return 0;
  return ((layer + 1) % cfg.attn_res_block == 0) ? 3 : 1;
}

// MoE expert payload dtype convention: IQ4_NL everywhere except the last
// four layers, which stay Q8_0.
inline bool moe_experts_are_q8(const Config& cfg, int layer) {
  return layer >= cfg.n_layers - 4;
}

// True for the packed-quant tensors (linears, conv projections, experts,
// embeddings, head); false for the dense F32 tensors (norms, conv kernels,
// clips, biases, loop gates, res queries).
inline bool expect_quantized(Tensor kind) {
  switch (kind) {
    case Tensor::TokenEmbd:
    case Tensor::Output:
    case Tensor::AttnGate:
    case Tensor::AttnQ:
    case Tensor::AttnK:
    case Tensor::AttnV:
    case Tensor::AttnOutput:
    case Tensor::AttnRel:
    case Tensor::AttnSinkProj:
    case Tensor::AttnKConvProj:
    case Tensor::AttnVConvProj:
    case Tensor::FfnGateUpExps:
    case Tensor::FfnDownExps: return true;
    default: return false;
  }
}

// Expected (rows, cols) in the GGUF (rows(), cols()) convention for a
// resolved tensor. MoE `rows` already fold the expert dim over the
// dims[1] side (gate_up: E*2*I; down: E*D — the file stores per-expert
// C-order [dims[1], dims[0]], i.e. down_exps slices are [D, I]); cols is
// dims[0]. The payload is expert-major contiguous, so the loader treats
// experts as one packed [rows, cols] matrix.
inline void expected_shape(const TensorId& id, const Config& cfg,
                           size_t& rows, size_t& cols) {
  const size_t D   = (size_t)cfg.d_model;
  const size_t hq  = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  const size_t V   = (size_t)cfg.vocab_size;
  const size_t E   = (size_t)cfg.n_experts;
  const size_t I   = (size_t)cfg.intermediate_size;
  const size_t K   = (size_t)cfg.sconv_kernel;
  const size_t G   = (size_t)cfg.gate_rank;
  switch (id.kind) {
    case Tensor::TokenEmbd:
    case Tensor::Output: rows = V; cols = D; return;
    case Tensor::TokenEmbdNorm:
    case Tensor::OutputNorm:
    case Tensor::OutputResQuery: rows = 1; cols = D; return;
    case Tensor::AttnGate:
    case Tensor::AttnQ: rows = hq; cols = D; return;
    case Tensor::AttnK:
    case Tensor::AttnV: rows = hkv; cols = D; return;
    case Tensor::AttnOutput: rows = D; cols = hq; return;
    case Tensor::AttnRel:
      rows = (size_t)cfg.n_heads * (size_t)cfg.d_rel;
      cols = D;
      return;
    case Tensor::AttnSinkProj:
      rows = (size_t)cfg.n_heads;
      cols = D;
      return;
    case Tensor::AttnKConvProj:
    case Tensor::AttnVConvProj:
      rows = (size_t)cfg.n_kv_heads * (size_t)cfg.sconv_kernel;
      cols = D;
      return;
    case Tensor::AttnNorm:
    case Tensor::AttnQInpNorm:
    case Tensor::AttnKInpNorm:
    case Tensor::AttnVInpNorm: rows = 1; cols = D; return;
    case Tensor::AttnKConv:
    case Tensor::AttnVConv: rows = hkv; cols = K; return;
    case Tensor::AttnPostConv: rows = D; cols = K; return;
    case Tensor::AttnQClip:
      rows = (size_t)cfg.n_heads;
      cols = (size_t)cfg.head_dim;
      return;
    case Tensor::AttnRelBias:
      rows = hkv;
      cols = (size_t)cfg.d_rel;
      return;
    case Tensor::AttnRelClip:
      rows = (size_t)cfg.n_heads;
      cols = (size_t)cfg.d_rel;
      return;
    case Tensor::AttnResQuery:
      rows = (size_t)attn_res_query_count(cfg, id.layer);
      cols = D;
      return;
    case Tensor::FfnGateUpExps:
      // Square-ambiguous in the file (D == 2*I == 2304): recorded with the
      // out dim (2*I) as rows, matching the linears' rows=out convention.
      rows = E * 2 * I;
      cols = D;
      return;
    // File orientation is per-expert C-order [D, I] (dims "1152x2304x128"):
    // rows fold the expert dim over the D side, cols is I.
    case Tensor::FfnDownExps: rows = E * D; cols = I; return;
    case Tensor::FfnGateInp: rows = E; cols = D; return;
    case Tensor::ExpProbsB: rows = 1; cols = E; return;
    case Tensor::FfnNorm:
    case Tensor::FfnResQuery: rows = 1; cols = D; return;
    case Tensor::FfnPostConv: rows = D; cols = K; return;
    case Tensor::LoopQCtrlDown:
    case Tensor::LoopKCtrlDown:
    case Tensor::LoopVCtrlDown:
    case Tensor::LoopFfnCtrlDown: rows = G; cols = 2 * D; return;
    case Tensor::LoopQCtrlForget:
    case Tensor::LoopQCtrlGain:
    case Tensor::LoopKCtrlForget:
    case Tensor::LoopKCtrlGain:
    case Tensor::LoopVCtrlForget:
    case Tensor::LoopVCtrlGain:
    case Tensor::LoopFfnCtrlForget:
    case Tensor::LoopFfnCtrlGain: rows = D; cols = G; return;
    case Tensor::LoopMsgQuery: rows = K; cols = D; return;
  }
  throw std::invalid_argument("expected_shape: unknown Tensor kind");
}

}  // namespace Inkling
}  // namespace KokkosDNN

#endif
