// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MOSS_CONFIG_HPP
#define KOKKOSDNN_MOSS_CONFIG_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Moss {

// Hyperparameters for the MOSS-Image-SFT text-to-image DiT example.
//
// Reference: MOSS-Image-SFT flow-matching DiT (joint self+cross attention,
// CFG with skip-layer guidance on the negative branch). The exact released
// checkpoint layout is not pinned here (see README "what still needs real
// weights"); this struct defines the geometry the example code implements
// and validates. Defaults mirror the driver CLI defaults:
// steps=50, cfg=3.0, shift=5.0, skip-layers=22.
struct Config {
  int depth = 24;             // transformer blocks
  int hidden_size = 1536;     // token dim D
  int num_heads = 24;         // joint attention heads (head_dim = D / heads)
  int patch_size = 2;         // latent patch (2x2 conv == per-patch linear)
  double mlp_ratio = 4.0;     // FFN hidden = D * ratio
  int caption_channels = 2048;  // text-encoder embedding dim
  int model_max_length = 512;   // caption token capacity
  int in_channels = 16;         // latent channels
  int out_channels = 16;        // noise prediction channels (== in_channels)
  int t_freq_dim = 256;         // sinusoidal timestep embedding dim
  double norm_eps = 1e-6;       // LayerNorm eps (adaLN modulation rows)
  double rms_eps = 1e-5;        // RMSNorm eps (caption projection)
  double timestep_norm_scale = 1.0;  // divisor applied to input timesteps

  // Factory for the full MOSS-Image-SFT geometry.
  static Config moss_sft() { return Config{}; }

  // Tiny config for unit/smoke tests (no checkpoint needed).
  static Config tiny_test() {
    Config c;
    c.depth = 2;
    c.hidden_size = 32;
    c.num_heads = 4;  // head dim 8
    c.patch_size = 1;
    c.mlp_ratio = 2.0;  // hidden 64
    c.caption_channels = 16;
    c.model_max_length = 8;
    c.in_channels = 4;
    c.out_channels = 4;
    c.t_freq_dim = 16;
    return c;
  }

  // FFN intermediate dim: hidden * ratio (exact for supported configs).
  int mlp_hidden() const {
    return (int)std::llround((double)hidden_size * mlp_ratio);
  }

  // Joint-attention head dim (hidden / num_heads).
  int head_dim() const {
    return (num_heads > 0) ? hidden_size / num_heads : 0;
  }

  // Tokens per latent image: (H/patch) * (W/patch).
  int num_tokens(int h, int w) const {
    if (patch_size <= 0 || h <= 0 || w <= 0) return 0;
    if (h % patch_size != 0 || w % patch_size != 0) return 0;
    return (h / patch_size) * (w / patch_size);
  }

  // Throw std::invalid_argument on incompatible dims.
  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Moss::Config::validate: " + m);
    };
    if (depth <= 0) bad("depth must be > 0");
    if (hidden_size <= 0) bad("hidden_size must be > 0");
    if (patch_size <= 0) bad("patch_size must be > 0");
    if (num_heads <= 0) bad("num_heads must be > 0");
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
    const double mh = (double)hidden_size * mlp_ratio;
    if (std::llround(mh) != (long long)mh)
      bad("hidden_size * mlp_ratio must be integral");
    if (!(norm_eps > 0) || !std::isfinite(norm_eps))
      bad("norm_eps must be positive and finite");
    if (!(rms_eps > 0) || !std::isfinite(rms_eps))
      bad("rms_eps must be positive and finite");
    if (!(timestep_norm_scale > 0) || !std::isfinite(timestep_norm_scale))
      bad("timestep_norm_scale must be positive and finite");
    if (out_channels != in_channels)
      bad("out_channels must equal in_channels (noise prediction)");
  }
};

}  // namespace Moss
}  // namespace KokkosDNN

#endif
