// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 model library: installable component (KokkosDNN::qwen_image_21).
// Header-only pipeline (KokkosDNN_QwenImage21{Config,Dit,Encoder,Vae,Vision}.hpp);
// there are no .cpp sources, hence INTERFACE. Edit conditioning is wired at
// the encoder level (joint text+vision encode) and in the DiT forward_cond
// path ([cond|target] + img_shapes/img_mask); the VAE is decode-only
// (see those headers).
#ifndef KOKKOSDNN_QWENIMAGE21CONFIG_HPP
#define KOKKOSDNN_QWENIMAGE21CONFIG_HPP

#include "KokkosDNN_Safetensors.hpp"

// Shared Config + tiny host-side safetensors parsing used by the DiT,
// encoder, and VAE headers. Ground truth (Qwen-Image-2.1-Turbo, from HF
// research, not guessed):
//   * DiT: 32 single-stream layers, inner dim 4096, 32 heads x head_dim 128,
//     context_in_dim 4096, in/out_channels 64, patch_size 1, mlp_ratio 3
//     (SwiGLU 4096 -> 12288 -> 4096), 3-axis RoPE axes_dims [16, 56, 56],
//     causal_condition true (block-causal: text causal, image block
//     bidirectional), per-block QK RMSNorm (norm_q / norm_k), tanh-gated
//     residuals, text projection txt_in (ZeroCenterRMSNorm + Linear
//     4096 -> 4096 + GELU-tanh + Linear), timestep sinusoidal 256-d -> MLP
//     to 4096, ONE shared modulation (SiLU + Linear -> 4 x 4096, sliced per
//     block as scale + gate x2), final AdaLayerNormContinuous (scale-only)
//     + proj_out to 64 ch. ALL weights `.weight`, NO biases. Sharded 2-file
//     safetensors, index total_size 14230249472, bf16. Scheduler:
//     FlowMatchEuler, Turbo sample_sigmas [1.0, 0.978453, 0.95418, 0.926626,
//     0.89508, 0.845148, 0.704534, 0.414568] (8 steps), CFG
//     true_cfg_scale = 1.0 (guidance OFF by default), dynamic shift
//     (max_shift 1.15, max_seq_len 4096), dims rounded to multiples of 32,
//     latent = px / 16 (1024 px -> 64 x 64 = 4096 tokens, 64 ch).
//   * Encoder: Qwen3-VL-8B text tower (36 layers, hidden 4096, 32Q / 8KV
//     heads, hd 128, SwiGLU 12288, vocab 151936, RoPE theta 5M + M-RoPE
//     [24, 20, 20]); pipeline takes hidden_states[-1] with final RMSNorm
//     NEUTRALIZED (pre-norm output). Single-shard text_encoder/
//     model.safetensors, 17.5 GB bf16.
//   * VAE: AutoencoderKLQwenImage21, in/out 4 ch, z_dim 64, 16x
//     compression, base_dim 96 / dec 144, dim_mult [1, 2, 4, 8, 8],
//     per-channel latents_mean / latents_std [64] affine; single shard
//     675 MB bf16; decode -> RGBA.
//   * Tokenizer: Qwen2 byte-level BPE, vocab 151936, processor/
//     tokenizer.json 11.4 MB + chat_template.jinja + tokenizer_config.json.

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

namespace KokkosDNN {
namespace QwenImage21 {

// DiT + sampler constants for Qwen-Image-2.1-Turbo (text-to-image only).
struct Config {
  int num_layers = 32;       // single-stream transformer blocks
  int hidden_dim = 4096;     // inner dim
  int num_heads = 32;        // attention heads (head_dim = 128)
  int context_dim = 4096;    // text context in-dim (encoder hidden size)
  int latent_channels = 64;  // in/out_channels (patch_size 1: token dim)
  int mlp_dim = 12288;       // SwiGLU per-branch intermediate (gate AND up
                             // each [mlp_dim, hidden]; out [hidden, mlp_dim])
  std::array<int, 3> rope_axes = {16, 56, 56};  // 3-axis RoPE split of hd 128
  double rope_theta = 10000.0;  // RoPE base (assumption: not published for
                                // the DiT; golden validation TODO)
  double norm_eps = 1e-6;       // unweighted block RMSNorm epsilon
  bool causal_condition = true;  // block-causal conditioning (text causal,
                                 // image block bidirectional)
  // Turbo sampler: 8 FlowMatchEuler sigmas (descending).
  std::array<double, 8> sample_sigmas = {1.0,       0.978453, 0.95418,
                                         0.926626,  0.89508,  0.845148,
                                         0.704534,  0.414568};
  double true_cfg_scale = 1.0;  // guidance OFF by default
  double max_shift = 1.15;       // dynamic-shift ceiling
  int max_seq_len = 4096;        // dynamic-shift reference length
  int latent_h = 64;             // default latent grid (1024 px / 16)
  int latent_w = 64;

  int head_dim() const { return (num_heads > 0) ? hidden_dim / num_heads : 0; }
  int gate_dim() const { return mlp_dim; }  // per SwiGLU gate/up branch

  // Tiny config for unit tests (no checkpoint needed).
  static Config tiny_test() {
    Config c;
    c.num_layers = 2;
    c.hidden_dim = 32;
    c.num_heads = 4;  // head_dim 8
    c.context_dim = 16;
    c.latent_channels = 4;
    c.mlp_dim = 48;  // gate_dim 24
    c.rope_axes = {2, 2, 4};
    c.latent_h = 2;
    c.latent_w = 2;
    c.sample_sigmas = {1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0};
    return c;
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("QwenImage21::Config::validate: " + m);
    };
    if (num_layers <= 0) bad("num_layers must be > 0");
    if (hidden_dim <= 0) bad("hidden_dim must be > 0");
    if (num_heads <= 0) bad("num_heads must be > 0");
    if (hidden_dim % num_heads != 0)
      bad("hidden_dim must be divisible by num_heads");
    if (context_dim <= 0) bad("context_dim must be > 0");
    if (latent_channels <= 0) bad("latent_channels must be > 0");
    if (mlp_dim <= 0 || mlp_dim % 2 != 0)
      bad("mlp_dim must be a positive even number");
    if (rope_axes[0] + rope_axes[1] + rope_axes[2] != head_dim())
      bad("rope_axes must sum to head_dim");
    for (int a : rope_axes)
      if (a <= 0 || a % 2 != 0)
        bad("rope_axes entries must be positive even numbers");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (!(norm_eps > 0) || !std::isfinite(norm_eps))
      bad("norm_eps must be positive and finite");
    for (double s : sample_sigmas)
      if (!std::isfinite(s) || s < 0.0)
        bad("sample_sigmas must be finite and nonnegative");
    for (size_t i = 1; i < sample_sigmas.size(); ++i)
      if (!(sample_sigmas[i] <= sample_sigmas[i - 1]))
        bad("sample_sigmas must be non-increasing");
    if (!(max_shift > 0) || !std::isfinite(max_shift))
      bad("max_shift must be positive and finite");
    if (max_seq_len <= 0) bad("max_seq_len must be > 0");
    if (latent_h <= 0 || latent_w <= 0)
      bad("latent_h/latent_w must be > 0");
  }
};

namespace Impl {

// Grow-only view cache: exact-size managed views keyed by (tag, extents).
// Entries are never shrunk or evicted, so steady-state forward() performs
// no allocation; distinct shapes seen over the process lifetime bound memory.
template <class V>
struct Qwen21ViewCache {
  std::map<std::pair<std::string, std::vector<size_t>>, V> m;
  template <class... Dims>
  V get(const std::string& tag, Dims... d) {
    std::vector<size_t> key{(size_t)d...};
    auto& v = m[std::make_pair(tag, key)];
    bool fit = (v.data() != nullptr);
    if (fit) {
      size_t i = 0;
      for (size_t e : key) {
        if (v.extent(i++) != e) {
          fit = false;
          break;
        }
      }
    }
    if (!fit) v = V(tag, d...);
    return v;
  }
  size_t bytes(size_t scalar_bytes) const {
    size_t n = 0;
    for (const auto& kv : m) n += kv.second.size() * scalar_bytes;
    return n;
  }
  void clear() { m.clear(); }
};

// ---- host-side safetensors parsing (F32/BF16, header-only) ----
// Central implementation lives in src/KokkosDNN_Safetensors.hpp (next to
// KokkosDNN_GGUF.hpp); the names below are thin compatibility wrappers so
// existing call sites keep compiling with identical diagnostics.

inline uint64_t qwen21_ld64(const char* p) {
  return ::KokkosDNN::Safetensors::ld64_le(p);
}

inline float qwen21_bf16_to_float(uint16_t h) {
  return ::KokkosDNN::Safetensors::bf16_to_float(h);
}

using Qwen21SafeEntry = ::KokkosDNN::Safetensors::Entry;

inline std::string qwen21_parse_json_string(const std::string& s,
                                            size_t& pos) {
  return ::KokkosDNN::Safetensors::parse_json_string(s, pos, "QwenImage21");
}

// Minimal header-JSON scanner: only the torch-saved subset
// {"__metadata__":{...},"name":{"dtype":..,"shape":[..],"data_offsets":[a,b]}}.
inline std::map<std::string, Qwen21SafeEntry> qwen21_parse_header(
    const std::string& js, uint64_t data_base) {
  return ::KokkosDNN::Safetensors::parse_header(js, data_base, "QwenImage21");
}

inline size_t qwen21_entry_elsize(const Qwen21SafeEntry& en,
                                   const std::string& full) {
  return ::KokkosDNN::Safetensors::entry_elsize(en, full, "QwenImage21");
}

inline int64_t qwen21_entry_numel(const Qwen21SafeEntry& en,
                                  const std::string& full) {
  return ::KokkosDNN::Safetensors::entry_numel(en, full, "QwenImage21");
}

// Convert one chunk of file payload (F32/BF16) into Scalar outputs.
template <class Scalar>
void qwen21_convert_chunk(const char* raw, size_t elsize, Scalar* dst,
                          size_t m) {
  ::KokkosDNN::Safetensors::convert_chunk<Scalar>(raw, elsize, dst, m);
}

}  // namespace Impl
}  // namespace QwenImage21
}  // namespace KokkosDNN

#endif
