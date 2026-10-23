// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SANA_CONFIG_HPP
#define KOKKOSDNN_SANA_CONFIG_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Sana {

// Hyperparameters for NVlabs/Sana SanaMS_600M_P1_D28 (1024px, MultiLing).
// Reference: SanaMS Gesellschaft mit D=28 blocks, hidden 1152, patch 1,
// linear self-attention (LiteLA, 36 heads x dim 32), 16-head cross-attention,
// GLUMBConv FFN (ratio 2.5, acts silu/silu/None), adaLN-single conditioning
// (per-block [6,D] scale_shift_table, final [2,D] table), TimestepEmbedder
// (freq 256) + t_block SiLU+Linear(D -> 6D), CaptionEmbedder (2304 -> D,
// GELU-tanh MLP, token embedding [300,2304]) + scaled RMSNorm (scale 0.01),
// no self/cross qk-norm, no positional embedding (use_pe=false),
// LayerNorm eps 1e-6 (no affine), RMSNorm eps 1e-5, in/out channels 32,
// pred_sigma=false (noise prediction has in_channels, not 2x).
struct Config {
  int depth = 28;            // transformer blocks
  int hidden_size = 1152;    // token dim D
  int patch_size = 1;        // latent patch (1x1 conv == per-pixel linear)
  int num_heads = 16;        // cross-attention heads (head_dim = D / heads)
  int linear_num_heads = 36;  // LiteLA heads (D / linear_head_dim)
  int linear_head_dim = 32;   // LiteLA per-head dim
  double mlp_ratio = 2.5;     // GLUMBConv hidden = D * ratio (2880)
  int caption_channels = 2304;  // text-encoder embedding dim
  int model_max_length = 300;   // caption token capacity (y_embedding rows)
  int in_channels = 32;         // latent channels
  int out_channels = 32;        // == in_channels (pred_sigma=false)
  bool pred_sigma = false;      // variance prediction off
  int t_freq_dim = 256;         // sinusoidal timestep embedding dim
  bool use_pe = false;          // fixed sin-cos pos_embed unused (frozen zeros)
  double norm_eps = 1e-6;       // LayerNorm eps (norm1/norm2/norm_final)
  double rms_eps = 1e-5;        // RMSNorm eps (attention_y_norm): Sana's
                                // norm_eps default (yaml leaves it unset)
  bool qk_norm = false;         // LiteLA q/k RMSNorm off
  bool cross_norm = false;      // cross-attn q/k RMSNorm off
  bool y_norm = true;           // caption RMSNorm on
  double y_norm_scale = 0.01;   // RMSNorm init scale (baked into checkpoint w)
  double litela_eps = 1e-8;     // LiteLA normalization epsilon
  double timestep_norm_scale = 1.0;  // divisor applied to input timesteps

  // Factory for the released SanaMS_600M_P1_D28 checkpoint.
  static Config sana600m_1024() { return Config{}; }

  // Tiny config for unit/smoke tests (no checkpoint needed).
  static Config tiny_test() {
    Config c;
    c.depth = 2;
    c.hidden_size = 32;
    c.patch_size = 1;
    c.num_heads = 4;         // head_dim 8
    c.linear_num_heads = 4;  // 32 / 8
    c.linear_head_dim = 8;
    c.mlp_ratio = 2.0;       // hidden 64
    c.caption_channels = 16;
    c.model_max_length = 8;  // maxlen
    c.in_channels = 4;
    c.out_channels = 4;
    c.t_freq_dim = 16;
    return c;
  }

  // GLUMBConv intermediate dim: hidden * ratio (exact for the two
  // supported configs; validate() enforces integrality).
  int mlp_hidden() const {
    return (int)std::llround((double)hidden_size * mlp_ratio);
  }

  // Cross-attention head dim (hidden / num_heads).
  int cross_head_dim() const {
    return (num_heads > 0) ? hidden_size / num_heads : 0;
  }

  // Tokens per latent image: (H/patch) * (W/patch).
  int num_tokens(int h, int w) const {
    if (patch_size <= 0 || h <= 0 || w <= 0) return 0;
    if (h % patch_size != 0 || w % patch_size != 0) return 0;
    return (h / patch_size) * (w / patch_size);
  }

  // Throw std::invalid_argument on incompatible dims. Called by Weights::
  // allocate/random_init/load_torch_pth and by Model.
  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Sana::Config::validate: " + m);
    };
    if (depth <= 0) bad("depth must be > 0");
    if (hidden_size <= 0) bad("hidden_size must be > 0");
    if (patch_size <= 0) bad("patch_size must be > 0");
    if (num_heads <= 0) bad("num_heads must be > 0");
    if (linear_num_heads <= 0) bad("linear_num_heads must be > 0");
    if (linear_head_dim <= 0) bad("linear_head_dim must be > 0");
    if (!(mlp_ratio > 0) || !std::isfinite(mlp_ratio))
      bad("mlp_ratio must be positive and finite");
    if (caption_channels <= 0) bad("caption_channels must be > 0");
    if (model_max_length <= 0) bad("model_max_length must be > 0");
    if (in_channels <= 0) bad("in_channels must be > 0");
    if (out_channels <= 0) bad("out_channels must be > 0");
    if (t_freq_dim <= 0 || t_freq_dim % 2 != 0)
      bad("t_freq_dim must be a positive even number");
    if (hidden_size % num_heads != 0)
      bad("hidden_size (" + std::to_string(hidden_size) +
          ") must be divisible by num_heads (" + std::to_string(num_heads) +
          ")");
    if (hidden_size != linear_num_heads * linear_head_dim)
      bad("hidden_size (" + std::to_string(hidden_size) +
          ") must equal linear_num_heads * linear_head_dim (" +
          std::to_string(linear_num_heads * linear_head_dim) + ")");
    const double mh = (double)hidden_size * mlp_ratio;
    if (std::llround(mh) != (long long)mh)
      bad("hidden_size * mlp_ratio must be integral");
    if (mlp_hidden() % 2 != 0)
      bad("mlp hidden dim must be even (GLU chunk)");
    if (!(norm_eps > 0) || !std::isfinite(norm_eps))
      bad("norm_eps must be positive and finite");
    if (!(rms_eps > 0) || !std::isfinite(rms_eps))
      bad("rms_eps must be positive and finite");
    if (!(litela_eps > 0) || !std::isfinite(litela_eps))
      bad("litela_eps must be positive and finite");
    if (!(timestep_norm_scale > 0) || !std::isfinite(timestep_norm_scale))
      bad("timestep_norm_scale must be positive and finite");
    if (pred_sigma && out_channels != 2 * in_channels)
      bad("pred_sigma=true requires out_channels == 2 * in_channels");
    if (!pred_sigma && out_channels != in_channels)
      bad("pred_sigma=false requires out_channels == in_channels");
    if (use_pe)
      bad("use_pe=true is unsupported: this port implements the "
          "SanaMS_600M_P1_D28 setting (use_pe=false, no pos_embed)");
    if (qk_norm)
      bad("qk_norm=true is unsupported: LiteLA q/k RMSNorm is not "
          "implemented (600M checkpoint has qk_norm=false)");
    if (cross_norm)
      bad("cross_norm=true is unsupported: cross-attn q/k RMSNorm is not "
          "implemented (600M checkpoint has cross_norm=false)");
  }
};

}  // namespace Sana
}  // namespace KokkosDNN

#endif
