// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21VAE_HPP
#define KOKKOSDNN_QWENIMAGE21VAE_HPP

// AutoencoderKLQwenImage21 for Qwen-Image-2.1 (encode + decode paths).
// Ground truth (HF research): in/out 4 ch, z_dim 64, 16x compression,
// base_dim 96 / decoder base 144, ch_mult [1, 2, 4, 8, 8]; per-channel
// latents_mean / latents_std ([64] each) affine, read from vae/config.json
// AT RUNTIME (never hardcoded); single-shard 675 MB bf16; encode RGBA ->
// normalized latents ((mode-mean)/std by the edit pipeline), decode ->
// RGBA (4ch conv_out, scaled (x*0.5+0.5) by the pipeline, PNG alpha = ch 3).
//
// Encode math (diffusers autoencoder_kl_qwenimage21.py, T2I single-frame
// path with feat_cache=None, i.e. every CausalConv3d is a plain 2D conv and
// every 3D downsampler is ZeroPad-right/bottom-1 + stride-2 conv 3x3):
//   x [B,4,H,W] (pipeline-preprocessed RGBA in [-1,1]) -> conv_in 3x3
//   (4 -> 96) -> 5 ResidualDownBlocks, widths (96->96, 96->192, 192->384,
//   384->768, 768->768), stride-2 downsamplers after the first four plus
//   the AvgDown3D shortcut add (factor_t = 2 iff the stage is temporally
//   downsampled, else 1; factor_s = 2 iff downsampled, else 1), no
//   GroupNorm anywhere -> mid: ResNet(768) + single-head SDPA
//   attention(768) + ResNet(768) -> RMSNorm(768) -> SiLU -> conv_out 3x3
//   (768 -> 128) -> quant_conv 1x1 (128 -> 128) -> posterior mode (first
//   64 channels; DiagonalGaussianDistribution chunk dim=1, mode = mean) ->
//   latents [B,64,H/16,W/16] normalized as (mode-mean)/std by the caller.
// Block details: ResNet = RMSNorm(in, eps-free, gamma-only) -> SiLU ->
// conv3x3+bias -> RMSNorm(out) -> SiLU -> conv3x3+bias -> +shortcut (1x1
// conv+bias iff in != out, else identity). Attention = RMSNorm(C, gamma
// [C,1,1]) -> to_qkv 1x1 (C -> 3C, +bias) -> single-head SDPA with scale
// 1/sqrt(C), FP32 softmax -> proj 1x1 (+bias) -> +residual. Downsample =
// ZeroPad2d((0,1,0,1)) + stride-2 conv3x3+bias (the downsampler time_conv
// 1x1s exist in the checkpoint but are provably untouched on this path:
// only used with feat_cache != None, i.e. chunked video inference; they
// are presence/shape-checked, never executed). The ResidualDownBlock
// shortcut = AvgDown3D mean over (cin*factor/cout) expanded channels
// (factor = ft*fs*fs; single-frame T2I pads ft-1 zero frames at the front
// when temporally downsampled). Dropout is 0 at inference.
//
// Decode math (diffusers autoencoder_kl_qwenimage21.py, T2I single-frame
//   z [B,64,H,W] (pipeline-denormalized: z = raw*std + mean) ->
//   post_quant_conv 1x1 -> conv_in 3x3 (64 -> 1152) ->
//   mid: ResNet(1152) + single-head SDPA attention(1152) + ResNet(1152) ->
//   5 ResidualUpBlocks, 3 ResNets each, widths (1152->1152,1152->1152,
//   1152->576,576->288,288->144), nearest-2x + conv3x3 upsamplers after the
//   first four plus the DupUp3D avg_shortcut add, no GroupNorm anywhere ->
//   RMSNorm(144) -> SiLU -> conv_out 3x3 (144 -> 4) -> clamp(-1,1) ->
//   RGBA [B,4,16H,16W].
// Block details: ResNet = RMSNorm(in, eps-free, gamma-only) -> SiLU ->
// conv3x3+bias -> RMSNorm(out) -> SiLU -> conv3x3+bias -> +shortcut (1x1
// conv+bias iff in != out, else identity). Attention = RMSNorm(C, gamma
// [C,1,1]) -> to_qkv 1x1 (C -> 3C, +bias) -> single-head SDPA with scale
// 1/sqrt(C), FP32 softmax -> proj 1x1 (+bias) -> +residual. Upsampler =
// nearest-2x -> conv3x3+bias. Up-block shortcut = DupUp3D repeat-interleave
// add (factor_t = 2 iff the stage is temporally upsampled, else 1;
// factor_s = 2), sliced to the first chunk (single-frame T2I). The
// upsampler time_conv 1x1s exist in the checkpoint but are provably
// untouched on this path (only used with feat_cache != None, i.e. chunked
// video inference); they are presence/shape-checked, never executed.
// Dropout is 0 at inference.
//
// Dispatch contract (no hand-written cublas/cudnn calls anywhere here):
//   * 1x1 convs (post-quant, quant, conv shortcuts, attention qkv/proj) go
//     through gemm_bias, which owns the cuBLAS ladder.
//   * stride-1 3x3 convs go through conv3x3_forward (cuDNN ladder on CUDA,
//     MIOpen on HIP, im2col + gemm_bias / skinny kernels elsewhere).
//   * stride-2 3x3 downsample convs go through the native
//     qwen21vae_downsample_conv3x3_stride2 kernel below (no strided conv in
//     the core conv API; the pad-right/bottom-1 + stride-2 pattern is VAE
//     specific and small enough to own natively).
//   * RMSNorm, SiLU, nearest upsample, DupUp add, denormalize, and the
//   mid-block single-head SDPA (scores/softmax/AV) are native
//   backend-portable parallel_for kernels (no TPL expresses them).
// All Impl helpers are backend-portable parallel_for only. No backend
// #ifdefs in native paths.
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "KokkosDNN_Handle.hpp"

#include "KokkosDNN_Conv.hpp"
#include "KokkosDNN_GemmBias.hpp"

#include "KokkosDNN_QwenImage21Config.hpp"

namespace KokkosDNN {
namespace QwenImage21Vae {

// VAE constants for AutoencoderKLQwenImage21.
struct Config {
  int in_channels = 4;
  int out_channels = 4;
  int z_dim = 64;          // latent channels (DiT latent_channels)
  int base_dim = 96;       // encoder base channels
  int dec_base_dim = 144;  // decoder base channels
  std::vector<int> ch_mult = {1, 2, 4, 8, 8};
  int compression = 16;  // spatial downsampling factor
  int num_res_blocks = 2;  // ResidualBlocks per stage (resnets = +1)
  // Temporal-upsample flags per downsample stage (0 = spatial-only 2x,
  // 1 = temporal 2x + spatial 2x DupUp shortcut). Length must equal
  // ch_mult.size() - 1; the decoder consumes them reversed
  // (temperal_upsample). Selects the DupUp3D channel map (and whether a
  // time_conv key is expected); time_conv is never executed (see above).
  std::vector<int> temporal_downsample = {0, 1, 1, 1};
  double norm_eps = 1e-6;  // kept for API compat; RMSNorm here is eps-free
                           // (exact F.normalize semantics, see decode)

  // Per-channel affine, filled by load_vae_config_json() (NOT hardcoded).
  std::vector<float> latents_mean;  // [z_dim]
  std::vector<float> latents_std;   // [z_dim]

  static Config tiny_test() {
    Config c;
    c.in_channels = 2;
    c.out_channels = 2;
    c.z_dim = 4;
    c.base_dim = 4;
    c.dec_base_dim = 4;
    c.ch_mult = {1, 2};
    c.compression = 2;  // one 2x upsample stage (of two up blocks)
    c.num_res_blocks = 2;
    c.temporal_downsample = {0};
    c.latents_mean = std::vector<float>(4, 0.0f);
    c.latents_std = std::vector<float>(4, 1.0f);
    return c;
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("QwenImage21Vae::Config::validate: " + m);
    };
    if (in_channels <= 0) bad("in_channels must be > 0");
    if (out_channels <= 0) bad("out_channels must be > 0");
    if (z_dim <= 0) bad("z_dim must be > 0");
    if (base_dim <= 0) bad("base_dim must be > 0");
    if (dec_base_dim <= 0) bad("dec_base_dim must be > 0");
    if (ch_mult.empty()) bad("ch_mult must be nonempty");
    for (int m : ch_mult)
      if (m <= 0) bad("ch_mult entries must be > 0");
    if (compression <= 0) bad("compression must be > 0");
    if (num_res_blocks < 0) bad("num_res_blocks must be >= 0");
    if (temporal_downsample.size() != ch_mult.size() - 1)
      bad("temporal_downsample must hold ch_mult.size() - 1 entries");
    for (int t : temporal_downsample)
      if (t != 0 && t != 1) bad("temporal_downsample entries must be 0 or 1");
    if (!(norm_eps > 0) || !std::isfinite(norm_eps))
      bad("norm_eps must be positive and finite");
    if (!latents_mean.empty() && latents_mean.size() != (size_t)z_dim)
      bad("latents_mean must hold z_dim entries");
    if (!latents_std.empty() && latents_std.size() != (size_t)z_dim)
      bad("latents_std must hold z_dim entries");
    for (float s : latents_std)
      if (!(s > 0.0f) || !std::isfinite(s))
        bad("latents_std entries must be positive and finite");
  }

  bool has_affine() const {
    return latents_mean.size() == (size_t)z_dim &&
           latents_std.size() == (size_t)z_dim;
  }

  // Decoder geometry derived from ch_mult (mirrors QwenImage21Decoder3d:
  // dims = [dec_base*m_last] + reversed(ch_mult), one up block per mult).
  int num_up_blocks() const { return (int)ch_mult.size(); }
  int resnets_per_up_block() const { return num_res_blocks + 1; }
  int dim_mult_at(int i) const {
    // Mirrors QwenImage21Decoder3d: dims = [dec_base*m_last] +
    // reversed(ch_mult), i.e. [8,8,8,4,2,1] for [1,2,4,8,8].
    const int n = num_up_blocks();
    if (n <= 0 || i < 0 || i > n)
      throw std::invalid_argument(
          "QwenImage21Vae::Config::dim_mult_at: bad index");
    if (i == 0) return ch_mult[(size_t)n - 1];
    return ch_mult[(size_t)n - (size_t)i];
  }
  int width_at(int i) const { return dec_base_dim * dim_mult_at(i); }
  void block_io(int i, int& in, int& out) const {
    in = width_at(i);
    out = width_at(i + 1);
  }
  bool block_upsamples(int i) const { return i != num_up_blocks() - 1; }
  // temperal_upsample[i] = reverse(temporal_downsample)[i].
  int temporal_up(int i) const {
    const int n = num_up_blocks();
    if (n < 2 || i < 0 || i >= n - 1)
      throw std::invalid_argument(
          "QwenImage21Vae::Config::temporal_up: bad index");
    return temporal_downsample[(size_t)(n - 2 - i)];
  }

  // Encoder geometry derived from ch_mult (mirrors QwenImage21Encoder3d:
  // dims = [base] + base*ch_mult, one down block per mult; the last block
  // never downsamples and never temporally downsamples).
  int num_down_blocks() const { return (int)ch_mult.size(); }
  int resnets_per_down_block() const { return num_res_blocks; }
  int enc_width_at(int i) const {
    const int n = num_down_blocks();
    if (n <= 0 || i < 0 || i > n)
      throw std::invalid_argument(
          "QwenImage21Vae::Config::enc_width_at: bad index");
    if (i == 0) return base_dim;
    return base_dim * ch_mult[(size_t)i - 1];
  }
  void enc_block_io(int i, int& in, int& out) const {
    in = enc_width_at(i);
    out = enc_width_at(i + 1);
  }
  bool enc_downsamples(int i) const { return i != num_down_blocks() - 1; }
  // Encoder stage i temporally downsamples iff temporal_downsample[i] (the
  // decoder consumes these reversed as temperal_upsample).
  int enc_temporal_down(int i) const {
    const int n = num_down_blocks();
    if (n < 1 || i < 0 || i >= n)
      throw std::invalid_argument(
          "QwenImage21Vae::Config::enc_temporal_down: bad index");
    if (i == n - 1) return 0;
    return temporal_downsample[(size_t)i];
  }
};

namespace Impl {

// Parse one named float array (e.g. "latents_mean": [...]) from a small
// JSON document. Throws if the key is missing or malformed.
inline std::vector<float> qwen21vae_parse_float_array(const std::string& js,
                                                      const std::string& key) {
  const std::string pat = "\"" + key + "\"";
  size_t at = js.find(pat);
  if (at == std::string::npos)
    throw std::runtime_error("QwenImage21Vae: config missing key '" + key +
                             "'");
  at = js.find('[', at + pat.size());
  if (at == std::string::npos)
    throw std::runtime_error("QwenImage21Vae: bad array for '" + key + "'");
  ++at;
  std::vector<float> out;
  std::string num;
  auto flush = [&]() {
    if (!num.empty()) {
      out.push_back(std::stof(num));
      num.clear();
    }
  };
  for (size_t i = at; i < js.size(); ++i) {
    char c = js[i];
    if (c == ']') {
      flush();
      return out;
    }
    if (c == ',') {
      flush();
    } else if (!isspace((unsigned char)c)) {
      num += c;
    }
  }
  throw std::runtime_error("QwenImage21Vae: unterminated array '" + key + "'");
}

}  // namespace Impl

// Fill cfg.latents_mean / cfg.latents_std ([z_dim] each) from the HF
// vae/config.json at runtime. The file is NOT vendored: pass its path.
inline void qwen21_load_vae_config_json(const std::string& path,
                                        Config& cfg) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("QwenImage21Vae: cannot open '" + path + "'");
  std::streamoff e = f.tellg();
  if (e < 2)
    throw std::runtime_error("QwenImage21Vae: too small '" + path + "'");
  std::string js((size_t)e, '\0');
  f.seekg(0, std::ios::beg);
  f.read(js.data(), (std::streamsize)e);
  if (!f)
    throw std::runtime_error("QwenImage21Vae: short read '" + path + "'");
  cfg.latents_mean = Impl::qwen21vae_parse_float_array(js, "latents_mean");
  cfg.latents_std = Impl::qwen21vae_parse_float_array(js, "latents_std");
  cfg.validate();  // checks [z_dim] lengths + positivity
}

// Expected decoder-tensor inventory (diffusers key layout, F32/BF16).
// Used by allocate_expected/randomize and fail-fast-checked by decode().
// Covers post_quant_conv, decoder.conv_in, the mid block (2 resnets +
// 1 single-head attention), all ResidualUpBlocks (3 resnets each, 1x1
// shortcut iff in != out, nearest upsampler + time_conv keys for the
// temporally-upsampled stages), norm_out and conv_out.
struct Qwen21VaeTensorSpec {
  std::string name;
  std::vector<size_t> shape;  // on-disk row-major shape
};

inline std::vector<Qwen21VaeTensorSpec> qwen21vae_expected_decoder_tensors(
    const Config& cfg) {
  cfg.validate();
  std::vector<Qwen21VaeTensorSpec> out;
  const size_t Z = (size_t)cfg.z_dim;
  auto push = [&](const std::string& n, std::initializer_list<size_t> s) {
    out.push_back(Qwen21VaeTensorSpec{n, std::vector<size_t>(s)});
  };
  push("post_quant_conv.weight", {Z, Z, 1, 1});
  push("post_quant_conv.bias", {Z});
  const size_t D0 = (size_t)cfg.width_at(0);
  push("decoder.conv_in.weight", {D0, Z, 3, 3});
  push("decoder.conv_in.bias", {D0});
  auto res_keys = [&](const std::string& p, size_t cin, size_t cout) {
    push(p + "norm1.gamma", {cin, 1, 1, 1});
    push(p + "conv1.weight", {cout, cin, 3, 3});
    push(p + "conv1.bias", {cout});
    push(p + "norm2.gamma", {cout, 1, 1, 1});
    push(p + "conv2.weight", {cout, cout, 3, 3});
    push(p + "conv2.bias", {cout});
    if (cin != cout) {
      push(p + "conv_shortcut.weight", {cout, cin, 1, 1});
      push(p + "conv_shortcut.bias", {cout});
    }
  };
  res_keys("decoder.mid_block.resnets.0.", D0, D0);
  push("decoder.mid_block.attentions.0.norm.gamma", {D0, 1, 1});
  push("decoder.mid_block.attentions.0.to_qkv.weight", {3 * D0, D0, 1, 1});
  push("decoder.mid_block.attentions.0.to_qkv.bias", {3 * D0});
  push("decoder.mid_block.attentions.0.proj.weight", {D0, D0, 1, 1});
  push("decoder.mid_block.attentions.0.proj.bias", {D0});
  res_keys("decoder.mid_block.resnets.1.", D0, D0);
  const int NB = cfg.num_up_blocks();
  const int NR = cfg.resnets_per_up_block();
  for (int i = 0; i < NB; ++i) {
    int bin = 0, bout = 0;
    cfg.block_io(i, bin, bout);
    const size_t uin = (size_t)bin, uout = (size_t)bout;
    for (int j = 0; j < NR; ++j) {
      const size_t cin = (j == 0) ? uin : uout;
      res_keys("decoder.up_blocks." + std::to_string(i) + ".resnets." +
                   std::to_string(j) + ".",
               cin, uout);
    }
    if (cfg.block_upsamples(i)) {
      const std::string u = "decoder.up_blocks." + std::to_string(i) +
                            ".upsampler.";
      push(u + "resample.1.weight", {uout, uout, 3, 3});
      push(u + "resample.1.bias", {uout});
      if (cfg.temporal_up(i) != 0) {
        push(u + "time_conv.weight", {2 * uout, uout, 1, 1});
        push(u + "time_conv.bias", {2 * uout});
      }
    }
  }
  const size_t DL = (size_t)cfg.width_at(NB);
  push("decoder.norm_out.gamma", {DL, 1, 1, 1});
  push("decoder.conv_out.weight", {(size_t)cfg.out_channels, DL, 3, 3});
  push("decoder.conv_out.bias", {(size_t)cfg.out_channels});
  return out;
}

// Expected encoder-tensor inventory (diffusers key layout, F32/BF16).
// Used by allocate_expected_encoder/randomize and fail-fast-checked by
// encode(). Covers encoder.conv_in, all ResidualDownBlocks (num_res_blocks
// resnets each, 1x1 shortcut iff in != out, stride-2 downsampler +
// time_conv keys for the temporally-downsampled stages), the mid block
// (2 resnets + 1 single-head attention), norm_out, conv_out and quant_conv.
inline std::vector<Qwen21VaeTensorSpec> qwen21vae_expected_encoder_tensors(
    const Config& cfg) {
  cfg.validate();
  std::vector<Qwen21VaeTensorSpec> out;
  const size_t Z = (size_t)cfg.z_dim;
  auto push = [&](const std::string& n, std::initializer_list<size_t> s) {
    out.push_back(Qwen21VaeTensorSpec{n, std::vector<size_t>(s)});
  };
  auto res_keys = [&](const std::string& p, size_t cin, size_t cout) {
    push(p + "norm1.gamma", {cin, 1, 1, 1});
    push(p + "conv1.weight", {cout, cin, 3, 3});
    push(p + "conv1.bias", {cout});
    push(p + "norm2.gamma", {cout, 1, 1, 1});
    push(p + "conv2.weight", {cout, cout, 3, 3});
    push(p + "conv2.bias", {cout});
    if (cin != cout) {
      push(p + "conv_shortcut.weight", {cout, cin, 1, 1});
      push(p + "conv_shortcut.bias", {cout});
    }
  };
  const size_t E0 = (size_t)cfg.enc_width_at(0);
  push("encoder.conv_in.weight", {E0, (size_t)cfg.in_channels, 3, 3});
  push("encoder.conv_in.bias", {E0});
  const int NB = cfg.num_down_blocks();
  const int NR = cfg.resnets_per_down_block();
  for (int i = 0; i < NB; ++i) {
    int ein = 0, eout = 0;
    cfg.enc_block_io(i, ein, eout);
    const size_t uin = (size_t)ein, uout = (size_t)eout;
    for (int j = 0; j < NR; ++j) {
      const size_t cin = (j == 0) ? uin : uout;
      res_keys("encoder.down_blocks." + std::to_string(i) + ".resnets." +
                   std::to_string(j) + ".",
               cin, uout);
    }
    if (cfg.enc_downsamples(i)) {
      const std::string u =
          "encoder.down_blocks." + std::to_string(i) + ".downsampler.";
      push(u + "resample.1.weight", {uout, uout, 3, 3});
      push(u + "resample.1.bias", {uout});
      if (cfg.enc_temporal_down(i) != 0) {
        push(u + "time_conv.weight", {uout, uout, 1, 1});
        push(u + "time_conv.bias", {uout});
      }
    }
  }
  const size_t EL = (size_t)cfg.enc_width_at(NB);
  res_keys("encoder.mid_block.resnets.0.", EL, EL);
  push("encoder.mid_block.attentions.0.norm.gamma", {EL, 1, 1});
  push("encoder.mid_block.attentions.0.to_qkv.weight", {3 * EL, EL, 1, 1});
  push("encoder.mid_block.attentions.0.to_qkv.bias", {3 * EL});
  push("encoder.mid_block.attentions.0.proj.weight", {EL, EL, 1, 1});
  push("encoder.mid_block.attentions.0.proj.bias", {EL});
  res_keys("encoder.mid_block.resnets.1.", EL, EL);
  push("encoder.norm_out.gamma", {EL, 1, 1, 1});
  push("encoder.conv_out.weight", {2 * Z, EL, 3, 3});
  push("encoder.conv_out.bias", {2 * Z});
  push("quant_conv.weight", {2 * Z, 2 * Z, 1, 1});
  push("quant_conv.bias", {2 * Z});
  return out;
}

// VAE weights: agnostic device blobs keyed by checkpoint tensor name
// (encoder/decoder conv / resnet / attn / norm tensors; exact
// AutoencoderKLQwenImage21 key layout verified against the 675 MB
// single-shard checkpoint). Single-shard 675 MB bf16 (F32 also accepted).
template <class ExecSpace, class Scalar>
struct QwenImage21VaeWeights {
  using V1 = Kokkos::View<Scalar*, ExecSpace>;

  Config cfg;
  std::map<std::string, V1> blobs;  // device bytes per checkpoint tensor
  std::map<std::string, std::vector<int64_t>> shapes;
  std::map<std::string, std::string> dtypes;

  void allocate(const Config& c) {
    c.validate();
    cfg = c;
    blobs.clear();
    shapes.clear();
    dtypes.clear();
  }

  void release() {
    blobs.clear();
    shapes.clear();
    dtypes.clear();
  }

  size_t resident_bytes() const {
    size_t n = 0;
    for (const auto& kv : blobs) n += kv.second.extent(0);
    return n * sizeof(Scalar);
  }

  size_t num_tensors() const { return blobs.size(); }

  // Shared spec allocator for allocate_expected / allocate_expected_encoder.
  void allocate_specs(const std::vector<Qwen21VaeTensorSpec>& specs,
                      const std::string& what) {
    for (const Qwen21VaeTensorSpec& s : specs) {
      size_t n = 1;
      for (size_t d : s.shape) n *= d;
      if (n == 0)
        throw std::runtime_error(what + ": bad shape '" + s.name + "'");
      blobs[s.name] = V1(s.name, n);
      shapes[s.name] =
          std::vector<int64_t>(s.shape.begin(), s.shape.end());
      dtypes[s.name] =
          std::is_same_v<Scalar, float> ? "F32" : "BF16";
    }
  }

  // Load every tensor found in the shard(s) into device blobs (F32/BF16
  // file -> Scalar). Names outside both skeletons are accepted here;
  // dtype/shape/slice errors throw.
  void load_safetensors(const std::vector<std::string>& shard_paths) {
    if (shard_paths.empty())
      throw std::invalid_argument(
          "QwenImage21VaeWeights::load_safetensors: no shards given");
    static_assert(std::is_same_v<Scalar, float> ||
                      std::is_same_v<Scalar,
                                     Kokkos::Experimental::bhalf_t>,
                  "QwenImage21VaeWeights::load_safetensors: Scalar must be "
                  "float or bhalf_t");
    for (const std::string& path : shard_paths) {
      std::ifstream f;
      auto entries = QwenImage21::Impl::qwen21_open_shard(f, path);
      for (const auto& kv : entries) {
        const std::string& full = kv.first;
        const QwenImage21::Impl::Qwen21SafeEntry& en = kv.second;
        if (blobs.count(full))
          throw std::runtime_error(
              "QwenImage21VaeWeights::load_safetensors: duplicate tensor '" +
              full + "'");
        const size_t elsize = QwenImage21::Impl::qwen21_entry_elsize(en, full);
        const int64_t numel = QwenImage21::Impl::qwen21_entry_numel(en, full);
        if (numel <= 0)
          throw std::runtime_error(
              "QwenImage21VaeWeights::load_safetensors: bad shape '" + full +
              "'");
        if (en.end < en.begin ||
            en.end - en.begin != (size_t)numel * elsize)
          throw std::runtime_error(
              "QwenImage21VaeWeights::load_safetensors: tensor '" + full +
              "' slice size mismatch");
        V1 blob("qwen21vae_blob", (size_t)numel);
        QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
            f, en, (size_t)numel, elsize, blob.data());
        blobs[full] = blob;
        shapes[full] = en.shape;
        dtypes[full] = en.dtype;
      }
    }
    if (blobs.empty())
      throw std::runtime_error(
          "QwenImage21VaeWeights::load_safetensors: no tensors loaded");
  }

  void load_safetensors(const std::string& shard_path) {
    load_safetensors(std::vector<std::string>{shard_path});
  }

  // Allocate exactly the decoder tensors decode() consumes (see
  // qwen21vae_expected_decoder_tensors), zero-initialized. Lets tests
  // exercise decode() shapes without the 675 MB checkpoint; call
  // randomize() (or load_safetensors) before decoding.
  void allocate_expected(const Config& c) {
    c.validate();
    cfg = c;
    blobs.clear();
    shapes.clear();
    dtypes.clear();
    allocate_specs(qwen21vae_expected_decoder_tensors(cfg),
                   "QwenImage21VaeWeights::allocate_expected");
  }

  // Allocate exactly the encoder tensors encode() consumes (see
  // qwen21vae_expected_encoder_tensors), zero-initialized, WITHOUT
  // clearing any decoder tensors already present (so one Weights can
  // serve both paths). Call randomize() (or load_safetensors) before
  // encoding.
  void allocate_expected_encoder(const Config& c) {
    c.validate();
    if (blobs.empty()) cfg = c;
    allocate_specs(qwen21vae_expected_encoder_tensors(c),
                   "QwenImage21VaeWeights::allocate_expected_encoder");
  }

  // Fill every blob with small random values (smoke tests only, no
  // checkpoint): N(0, 0.02) for conv/linear weights and biases, 1 for
  // gamma (RMSNorm) tensors.
  void randomize(uint64_t seed) {
    cfg.validate();
    if (blobs.empty())
      throw std::runtime_error(
          "QwenImage21VaeWeights::randomize: no tensors (allocate first)");
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& kv : blobs) {
      const bool is_gamma = kv.first.find("gamma") != std::string::npos;
      auto h = Kokkos::create_mirror_view(kv.second);
      for (size_t i = 0; i < h.extent(0); ++i)
        h(i) = is_gamma ? static_cast<Scalar>(1.0f)
                        : static_cast<Scalar>(dist(rng));
      Kokkos::deep_copy(kv.second, h);
    }
  }
};

namespace Impl {

// Unmanaged device aliases over the agnostic weight blobs (zero-copy:
// the checkpoint layout is already row-major and dispatch-ready).
template <class ExecSpace, class Scalar>
using Qwen21VaeUV1 =
    Kokkos::View<Scalar*, ExecSpace, Kokkos::MemoryUnmanaged>;
template <class ExecSpace, class Scalar>
using Qwen21VaeUV2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                                  Kokkos::MemoryUnmanaged>;
template <class ExecSpace, class Scalar>
using Qwen21VaeUV4 = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace,
                                  Kokkos::MemoryUnmanaged>;

// Presence + numel check, then alias the blob storage (no copy).
template <class ExecSpace, class Scalar>
inline Qwen21VaeUV1<ExecSpace, Scalar> qwen21vae_bind_vec(
    const std::map<std::string, Kokkos::View<Scalar*, ExecSpace>>& blobs,
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const std::string& name, size_t n) {
  const auto bit = blobs.find(name);
  const auto sit = shapes.find(name);
  if (bit == blobs.end() || sit == shapes.end())
    throw std::runtime_error("QwenImage21VaeModel: missing tensor '" + name +
                             "'");
  size_t numel = 1;
  for (int64_t d : sit->second) {
    if (d < 0)
      throw std::runtime_error("QwenImage21VaeModel: bad shape '" + name +
                               "'");
    numel *= (size_t)d;
  }
  if (numel != n)
    throw std::runtime_error("QwenImage21VaeModel: tensor '" + name +
                             "' has wrong numel");
  return Qwen21VaeUV1<ExecSpace, Scalar>(bit->second.data(), n);
}

// [o,i,1,1] blob viewed as a gemm-ready [o,i] matrix.
template <class ExecSpace, class Scalar>
inline Qwen21VaeUV2<ExecSpace, Scalar> qwen21vae_bind_mat(
    const std::map<std::string, Kokkos::View<Scalar*, ExecSpace>>& blobs,
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const std::string& name, size_t o, size_t i) {
  const auto sit = shapes.find(name);
  if (sit == shapes.end() || sit->second.size() != 4 ||
      sit->second[0] != (int64_t)o || sit->second[1] != (int64_t)i ||
      sit->second[2] != 1 || sit->second[3] != 1)
    throw std::runtime_error("QwenImage21VaeModel: bad 1x1 shape '" + name +
                             "'");
  const auto bit = blobs.find(name);
  if (bit == blobs.end())
    throw std::runtime_error("QwenImage21VaeModel: missing tensor '" + name +
                             "'");
  return Qwen21VaeUV2<ExecSpace, Scalar>(bit->second.data(), o, i);
}

// [o,i,k,k] blob viewed as a 4D conv tensor.
template <class ExecSpace, class Scalar>
inline Qwen21VaeUV4<ExecSpace, Scalar> qwen21vae_bind_conv(
    const std::map<std::string, Kokkos::View<Scalar*, ExecSpace>>& blobs,
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const std::string& name, size_t o, size_t i, size_t k) {
  const auto sit = shapes.find(name);
  if (sit == shapes.end() || sit->second.size() != 4 ||
      sit->second[0] != (int64_t)o || sit->second[1] != (int64_t)i ||
      sit->second[2] != (int64_t)k || sit->second[3] != (int64_t)k)
    throw std::runtime_error("QwenImage21VaeModel: bad conv shape '" + name +
                             "'");
  const auto bit = blobs.find(name);
  if (bit == blobs.end())
    throw std::runtime_error("QwenImage21VaeModel: missing tensor '" + name +
                             "'");
  return Qwen21VaeUV4<ExecSpace, Scalar>(bit->second.data(), o, i, k, k);
}

// Presence + exact-shape check for tensors decode() never executes
// (upsampler time_conv 1x1s: only touched with feat_cache != None, i.e.
// chunked video inference, never on the single-frame T2I path).
inline void qwen21vae_require_shape(
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const std::string& name, std::initializer_list<int64_t> want) {
  const auto sit = shapes.find(name);
  if (sit == shapes.end())
    throw std::runtime_error("QwenImage21VaeModel: missing tensor '" + name +
                             "'");
  if (sit->second != std::vector<int64_t>(want))
    throw std::runtime_error("QwenImage21VaeModel: bad shape '" + name + "'");
}

// Token layout helpers: token rows [T=H*W, C] <-> NCHW sample [C,H,W].
template <class ExecSpace, class V3, class V2>
inline void qwen21vae_nchw_to_tokens(const V3& x, const V2& t,
                                     const std::string& label) {
  const size_t C = x.extent(0), H = x.extent(1), W = x.extent(2);
  if (t.extent(0) != H * W || t.extent(1) != C)
    throw std::invalid_argument(
        "qwen21vae_nchw_to_tokens: dimension mismatch");
  if (C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C * H * W),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = e / (H * W), rem = e % (H * W);
        const size_t h = rem / W, w = rem % W;
        t(h * W + w, c) = x(c, h, w);
      });
}

template <class ExecSpace, class V2, class V3>
inline void qwen21vae_tokens_to_nchw(const V2& t, const V3& x,
                                     const std::string& label) {
  const size_t C = x.extent(0), H = x.extent(1), W = x.extent(2);
  if (t.extent(0) != H * W || t.extent(1) != C)
    throw std::invalid_argument(
        "qwen21vae_tokens_to_nchw: dimension mismatch");
  if (C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C * H * W),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = e / (H * W), rem = e % (H * W);
        const size_t h = rem / W, w = rem % W;
        x(c, h, w) = t(h * W + w, c);
      });
}

template <class ExecSpace, class V2>
inline void qwen21vae_silu_inplace(const V2& t, const std::string& label) {
  const size_t T = t.extent(0), C = t.extent(1);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * C),
      KOKKOS_LAMBDA(size_t e) {
        const float x = static_cast<float>(t(e / C, e % C));
        t(e / C, e % C) = static_cast<typename V2::value_type>(
            x / (1.0f + expf(-x)));
      });
}

template <class ExecSpace, class V2>
inline void qwen21vae_clamp_inplace(const V2& t, float lo, float hi,
                                     const std::string& label) {
  const size_t T = t.extent(0), C = t.extent(1);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * C),
      KOKKOS_LAMBDA(size_t e) {
        const size_t r = e / C, c = e % C;
        const float x = static_cast<float>(t(r, c));
        const float y = x < lo ? lo : (x > hi ? hi : x);
        t(r, c) = static_cast<typename V2::value_type>(y);
      });
}

// y += x (same shape).
template <class ExecSpace, class V2>
inline void qwen21vae_add_inplace(const V2& y, const V2& x,
                                  const std::string& label) {
  if (y.extent(0) != x.extent(0) || y.extent(1) != x.extent(1))
    throw std::invalid_argument("qwen21vae_add_inplace: dimension mismatch");
  const size_t T = y.extent(0), C = y.extent(1);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * C),
      KOKKOS_LAMBDA(size_t e) {
        const size_t r = e / C, c = e % C;
        y(r, c) = static_cast<typename V2::value_type>(
            static_cast<float>(y(r, c)) + static_cast<float>(x(r, c)));
      });
}

// RMSNorm over channel fibers, exact F.normalize semantics from
// QwenImage21RMS_norm (no eps inside the norm; 1e-12 floor):
//   y = x / max(||x||_2, 1e-12) * sqrt(C) * gamma
template <class ExecSpace, class V2, class V1>
inline void qwen21vae_rmsnorm_inplace(const V2& t, const V1& gamma,
                                      const std::string& label) {
  const size_t T = t.extent(0), C = t.extent(1);
  if (gamma.extent(0) != C)
    throw std::invalid_argument(
        "qwen21vae_rmsnorm_inplace: dimension mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T),
      KOKKOS_LAMBDA(size_t r) {
        float ss = 0.0f;
        for (size_t c = 0; c < C; ++c) {
          const float v = static_cast<float>(t(r, c));
          ss += v * v;
        }
        const float n = sqrtf(ss);
        const float inv = sqrtf((float)C) / (n > 1e-12f ? n : 1e-12f);
        for (size_t c = 0; c < C; ++c) {
          const float v = static_cast<float>(t(r, c));
          t(r, c) = static_cast<typename V2::value_type>(
              v * inv * static_cast<float>(gamma(c)));
        }
      });
}

// Nearest-neighbor 2x upsample on token rows: y [4T,C] from x [T,C].
template <class ExecSpace, class V2>
inline void qwen21vae_nearest_upsample2x(const V2& x, size_t H, size_t W,
                                         const V2& y,
                                         const std::string& label) {
  const size_t C = x.extent(1), T = x.extent(0);
  if (T != H * W || y.extent(0) != 4 * T || y.extent(1) != C)
    throw std::invalid_argument(
        "qwen21vae_nearest_upsample2x: dimension mismatch");
  if (T == 0 || C == 0) return;
  const size_t W2 = 2 * W;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C * 4 * T),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = e / (4 * T), rem = e % (4 * T);
        const size_t h2 = rem / W2, w2 = rem % W2;
        y(h2 * W2 + w2, c) = x((h2 / 2) * W + (w2 / 2), c);
      });
}

// Fused DupUp3D avg_shortcut add for the single-frame T2I path
// (first_chunk=True): y [4T,cout] += DupUp(x [T,cin]) with factor_s = 2,
// factor_t = ft, repeats = cout*ft*4/cin. Row r -> (h2, w2); f1 = ft-1
// (first chunk), f2 = h2%2, f3 = w2%2; source channel
// c = ((co*ft + f1)*2 + f2)*2 + f3) / repeats, source row h*W+w.
template <class ExecSpace, class V2>
inline void qwen21vae_dupup_add(const V2& y, const V2& x, size_t H, size_t W,
                                int factor_t, size_t repeats,
                                const std::string& label) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = y.extent(1);
  if (T != H * W || y.extent(0) != 4 * T || factor_t < 1 || repeats == 0)
    throw std::invalid_argument("qwen21vae_dupup_add: dimension mismatch");
  if (T == 0 || Cout == 0) return;
  const size_t W2 = 2 * W, H2 = 2 * H;
  const size_t ft = (size_t)factor_t;
  const size_t f1 = (size_t)(factor_t - 1);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, H2 * W2 * Cout),
      KOKKOS_LAMBDA(size_t e) {
        const size_t co = e / (H2 * W2), rem = e % (H2 * W2);
        const size_t h2 = rem / W2, w2 = rem % W2;
        const size_t f2 = h2 % 2, f3 = w2 % 2;
        const size_t num = ((co * ft + f1) * 2 + f2) * 2 + f3;
        const size_t c = num / repeats;
        const float xv =
            (c < Cin) ? static_cast<float>(x((h2 / 2) * W + (w2 / 2), c))
                      : 0.0f;
        const size_t r = h2 * W2 + w2;
        y(r, co) = static_cast<typename V2::value_type>(
            static_cast<float>(y(r, co)) + xv);
      });
}

// Stride-2 3x3 downsample conv (encoder QwenImage21Resample downsample2d,
// and downsample3d with feat_cache=None where the time_conv never runs):
// ZeroPad2d((0,1,0,1)) then stride-2 conv3x3+bias. Padded H+1/W+1 read as
// zero past the right/bottom edges; H and W must be even (H2 = H/2).
template <class ExecSpace, class V2, class V4, class V1>
inline void qwen21vae_downsample_conv3x3_stride2(
    const V2& x, size_t H, size_t W, const V4& w, const V1& b, const V2& y,
    const std::string& label) {
  const size_t T = x.extent(0), Cin = x.extent(1);
  const size_t Cout = y.extent(1);
  if (H == 0 || W == 0 || (H % 2) != 0 || (W % 2) != 0)
    throw std::invalid_argument(
        "qwen21vae_downsample_conv3x3_stride2: H,W must be even and nonzero");
  const size_t H2 = H / 2, W2 = W / 2;
  if (T != H * W || y.extent(0) != H2 * W2 || w.extent(0) != Cout ||
      w.extent(1) != Cin || w.extent(2) != 3 || w.extent(3) != 3 ||
      b.extent(0) != Cout)
    throw std::invalid_argument(
        "qwen21vae_downsample_conv3x3_stride2: dimension mismatch");
  if (T == 0 || Cout == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, H2 * W2 * Cout),
      KOKKOS_LAMBDA(size_t e) {
        const size_t co = e / (H2 * W2), rem = e % (H2 * W2);
        const size_t h2 = rem / W2, w2 = rem % W2;
        float acc = static_cast<float>(b(co));
        for (size_t ci = 0; ci < Cin; ++ci)
          for (size_t kh = 0; kh < 3; ++kh)
            for (size_t kw = 0; kw < 3; ++kw) {
              const size_t ih = 2 * h2 + kh, iw = 2 * w2 + kw;
              const float xv = (ih < H && iw < W)
                                   ? static_cast<float>(x(ih * W + iw, ci))
                                   : 0.0f;
              acc += xv * static_cast<float>(w(co, ci, kh, kw));
            }
        y(h2 * W2 + w2, co) =
            static_cast<typename V2::value_type>(acc);
      });
}

// Fused AvgDown3D shortcut add for the single-frame encode path:
// y [H2*W2,cout] += AvgDown(x [H*W,cin]) with factor_s = fs (1 or 2),
// factor_t = ft (1 or 2). Single frame (T=1): temporal pad (ft-1 zero
// frames at the FRONT, matching F.pad (pad_t,0) on dim T) means source
// frame fti-pad_t is data only when it equals 0. Channel map mirrors
// QwenImage21AvgDown3D: ci2 = co*group+g, c = ci2/factor, f = ci2%factor
// with f = ((fti*fs)+fsh)*fs+fsw, averaged over the group.
template <class ExecSpace, class V2>
inline void qwen21vae_avgdown_add(const V2& y, const V2& x, size_t H, size_t W,
                                  size_t cin, size_t cout, int factor_t,
                                  int factor_s, const std::string& label) {
  const size_t T = x.extent(0);
  if (T != H * W || x.extent(1) != cin || factor_t < 1 || factor_t > 2 ||
      (factor_s != 1 && factor_s != 2))
    throw std::invalid_argument("qwen21vae_avgdown_add: dimension mismatch");
  const size_t fs = (size_t)factor_s, ft = (size_t)factor_t;
  const size_t H2 = H / fs, W2 = W / fs;
  if (H % fs != 0 || W % fs != 0 || y.extent(0) != H2 * W2 ||
      y.extent(1) != cout)
    throw std::invalid_argument("qwen21vae_avgdown_add: dimension mismatch");
  const size_t factor = ft * fs * fs;
  if (cin * factor % cout != 0)
    throw std::invalid_argument(
        "qwen21vae_avgdown_add: cin*factor must divide cout");
  const size_t repeats = cin * factor / cout;
  if (repeats == 0) return;
  // Padded-frame offset: the single-frame encode always has temporal T=1,
  // so F.pad (pad_t,0) on dim T prepends pad_t = (ft-1%ft)%ft = ft-1 zero
  // frames. (NOT a function of the H*W token count: an even token count
  // would give pad_t=0 and swap zero/data across even/odd channels.)
  const size_t pad_t = ft - 1;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, H2 * W2 * cout),
      KOKKOS_LAMBDA(size_t e) {
        const size_t co = e / (H2 * W2), rem = e % (H2 * W2);
        const size_t h2 = rem / W2, w2 = rem % W2;
        double acc = 0.0;
        for (size_t g = 0; g < repeats; ++g) {
          const size_t ci2 = co * repeats + g;
          const size_t c = ci2 / factor, f = ci2 % factor;
          const size_t fti = f / (fs * fs);
          const size_t fr = f % (fs * fs);
          const size_t fsh = fr / fs, fsw = fr % fs;
          const size_t src_frame = fti >= pad_t ? fti - pad_t : fs * fs;
          const float xv =
              (src_frame == 0 && c < cin)
                  ? static_cast<float>(x((h2 * fs + fsh) * W + (w2 * fs + fsw),
                                         c))
                  : 0.0f;
          acc += (double)xv;
        }
        const float mean = (float)(acc / (double)repeats);
        const size_t r = h2 * W2 + w2;
        y(r, co) = static_cast<typename V2::value_type>(
            static_cast<float>(y(r, co)) + mean);
      });
}

// qkv [T,3C] -> q/k/v [T,C] split.
template <class ExecSpace, class V2>
inline void qwen21vae_split_qkv(const V2& qkv, const V2& q, const V2& k,
                                const V2& v, const std::string& label) {
  const size_t T = qkv.extent(0), C = q.extent(1);
  if (qkv.extent(1) != 3 * C || k.extent(0) != T || k.extent(1) != C ||
      v.extent(0) != T || v.extent(1) != C)
    throw std::invalid_argument("qwen21vae_split_qkv: dimension mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * C),
      KOKKOS_LAMBDA(size_t e) {
        const size_t r = e / C, c = e % C;
        q(r, c) = qkv(r, c);
        k(r, c) = qkv(r, c + C);
        v(r, c) = qkv(r, c + 2 * C);
      });
}

// Single-head SDPA pieces: s[i,j] = scale * <q_i, k_j>; row softmax; o = p v.
template <class ExecSpace, class V2>
inline void qwen21vae_attn_scores(const V2& q, const V2& k, float scale,
                                  const V2& s, const std::string& label) {
  const size_t T = q.extent(0), C = q.extent(1);
  if (k.extent(0) != T || k.extent(1) != C || s.extent(0) != T ||
      s.extent(1) != T)
    throw std::invalid_argument("qwen21vae_attn_scores: dimension mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, T}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float acc = 0.0f;
        for (size_t c = 0; c < C; ++c)
          acc += static_cast<float>(q(i, c)) * static_cast<float>(k(j, c));
        s(i, j) =
            static_cast<typename V2::value_type>(static_cast<float>(scale) * acc);
      });
}

template <class ExecSpace, class V2>
inline void qwen21vae_softmax_rows_inplace(const V2& s,
                                           const std::string& label) {
  const size_t T = s.extent(0), N = s.extent(1);
  if (T == 0 || N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T),
      KOKKOS_LAMBDA(size_t i) {
        float mx = static_cast<float>(s(i, 0));
        for (size_t j = 1; j < N; ++j) {
          const float v = static_cast<float>(s(i, j));
          mx = v > mx ? v : mx;
        }
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
          const float e = expf(static_cast<float>(s(i, j)) - mx);
          s(i, j) = static_cast<typename V2::value_type>(e);
          sum += e;
        }
        const float inv = 1.0f / sum;
        for (size_t j = 0; j < N; ++j)
          s(i, j) = static_cast<typename V2::value_type>(
              static_cast<float>(s(i, j)) * inv);
      });
}

template <class ExecSpace, class V2>
inline void qwen21vae_attn_av(const V2& p, const V2& v, const V2& o,
                              const std::string& label) {
  const size_t T = p.extent(0), C = v.extent(1);
  if (p.extent(1) != T || v.extent(0) != T || o.extent(0) != T ||
      o.extent(1) != C)
    throw std::invalid_argument("qwen21vae_attn_av: dimension mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t i, size_t c) {
        float acc = 0.0f;
        for (size_t j = 0; j < T; ++j)
          acc += static_cast<float>(p(i, j)) * static_cast<float>(v(j, c));
        o(i, c) = static_cast<typename V2::value_type>(acc);
      });
}

// In-place pipeline denormalize: x = x * std + mean per channel. For bf16
// storage reproduce the reference's two-op lane exactly: diffusers casts the
// constants to the latent dtype, multiplies (round to bf16), then adds (round
// to bf16). The fp32 path keeps the single-rounding form.
template <class ExecSpace, class V4, class V1>
inline void qwen21vae_denormalize_inplace(const V4& x, const V1& mean,
                                          const V1& std,
                                          const std::string& label) {
  const size_t B = x.extent(0), C = x.extent(1), H = x.extent(2),
               W = x.extent(3);
  if (mean.extent(0) != C || std.extent(0) != C)
    throw std::invalid_argument(
        "qwen21vae_denormalize_inplace: dimension mismatch");
  if (B == 0 || C == 0 || H == 0 || W == 0) return;
  using VScalar = typename V4::value_type;
  constexpr bool kTwoOpBf16 =
      std::is_same_v<VScalar, Kokkos::Experimental::bhalf_t>;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, B * C * H * W),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = (e / (H * W)) % C;
        const size_t b = e / (C * H * W);
        const size_t rem = e % (H * W);
        const float v = static_cast<float>(x(b, c, rem / W, rem % W));
        // Keep all declarations outside the constexpr selection: NVCC's
        // extended device lambda cannot first-capture a variable inside an
        // if-constexpr context.
        const VScalar s = static_cast<VScalar>(static_cast<float>(std(c)));
        const VScalar m = static_cast<VScalar>(static_cast<float>(mean(c)));
        const VScalar t = static_cast<VScalar>(v * static_cast<float>(s));
        const VScalar single =
            static_cast<VScalar>(v * static_cast<float>(std(c)) +
                                 static_cast<float>(mean(c)));
        x(b, c, rem / W, rem % W) =
            kTwoOpBf16 ? static_cast<VScalar>(static_cast<float>(t) +
                                              static_cast<float>(m))
                       : single;
      });
}

// Posterior mode + pipeline normalize, fused: the quant_conv output holds
// 2*z_dim channels (DiagonalGaussianDistribution chunk dim=1: first half
// mean, second half logvar); the edit pipeline conditions on the mode
// (argmax) normalized as (mode-mean)/std per channel with the runtime
// affine. y [T,Z] from x [T,2Z]. For bf16 storage reproduce the
// reference's two-op lanes: constants cast to the latent dtype, then
// sub (round to bf16), then div (round to bf16); fp32 is single-rounding.
template <class ExecSpace, class V2, class V1>
inline void qwen21vae_quant_mode_normalize(const V2& x, const V1& mean,
                                           const V1& std, const V2& y,
                                           const std::string& label) {
  const size_t T = x.extent(0), Z = y.extent(1);
  if (x.extent(1) != 2 * Z || y.extent(0) != T)
    throw std::invalid_argument(
        "qwen21vae_quant_mode_normalize: dimension mismatch");
  if (mean.extent(0) != Z || std.extent(0) != Z)
    throw std::invalid_argument(
        "qwen21vae_quant_mode_normalize: affine dimension mismatch");
  if (T == 0 || Z == 0) return;
  using VScalar = typename V2::value_type;
  constexpr bool kTwoOpBf16 =
      std::is_same_v<VScalar, Kokkos::Experimental::bhalf_t>;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * Z),
      KOKKOS_LAMBDA(size_t e) {
        const size_t t = e / Z, c = e % Z;
        const float v = static_cast<float>(x(t, c));
        // Keep all declarations outside the constexpr selection: NVCC's
        // extended device lambda cannot first-capture a variable inside an
        // if-constexpr context.
        const VScalar m = static_cast<VScalar>(static_cast<float>(mean(c)));
        const VScalar s = static_cast<VScalar>(static_cast<float>(std(c)));
        const VScalar d = static_cast<VScalar>(v - static_cast<float>(m));
        const VScalar single =
            static_cast<VScalar>((v - static_cast<float>(mean(c))) /
                                 static_cast<float>(std(c)));
        y(t, c) = kTwoOpBf16 ? static_cast<VScalar>(static_cast<float>(d) /
                                                    static_cast<float>(s))
                             : single;
      });
}

}  // namespace Impl

// VAE model: owns Weights. decode() runs the full convolutional decoder:
// denormalized latents [B, z_dim, h, w] -> RGBA [B, out_channels, S*h,
// S*w] with S = compression (16). The caller denormalizes raw DiT latents
// first (z = raw * std + mean per channel, e.g. via denormalize_latents);
// decode() requires the runtime latents affine (has_affine()).
// encode() runs the full convolutional encoder: pipeline-preprocessed
// RGBA [B, in_channels, H, W] in [-1,1] -> NORMALIZED latents
// [B, z_dim, H/S, W/S] ((posterior-mode - mean)/std, the tensor the edit
// pipeline conditions on); encode() requires the same runtime affine.
template <class ExecSpace, class Scalar>
class QwenImage21VaeModel {
 public:
  using V4 = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using UV1 = Impl::Qwen21VaeUV1<ExecSpace, Scalar>;
  using UV2 = Impl::Qwen21VaeUV2<ExecSpace, Scalar>;
  using UV4 = Impl::Qwen21VaeUV4<ExecSpace, Scalar>;
  using Weights = QwenImage21VaeWeights<ExecSpace, Scalar>;

  explicit QwenImage21VaeModel(const Config& cfg = Config()) {
    cfg.validate();
    cfg_ = cfg;
    weights_.allocate(cfg_);
  }

  const Config& config() const { return cfg_; }
  Weights& weights() { return weights_; }
  const Weights& weights() const { return weights_; }

  // Raw DiT latents [B, z_dim, h, w] -> new V4 with the pipeline
  // denormalize applied (x * std + mean per channel, from the runtime
  // config.json affine). Feed the result to decode().
  static V4 denormalize_latents(const V4& raw, const Config& cfg) {
    cfg.validate();
    if (!cfg.has_affine())
      throw std::runtime_error(
          "QwenImage21VaeModel::denormalize_latents: latents affine not "
          "loaded (qwen21_load_vae_config_json first)");
    if (raw.extent(1) != (size_t)cfg.z_dim)
      throw std::invalid_argument(
          "QwenImage21VaeModel::denormalize_latents: channel mismatch");
    V4 out(Kokkos::ViewAllocateWithoutInitializing("qwen21vae_denorm"),
           raw.extent(0), raw.extent(1), raw.extent(2), raw.extent(3));
    Kokkos::deep_copy(out, raw);
    V1 mean("qwen21vae_mean", (size_t)cfg.z_dim);
    V1 std("qwen21vae_std", (size_t)cfg.z_dim);
    auto hm = Kokkos::create_mirror_view(mean);
    auto hs = Kokkos::create_mirror_view(std);
    for (int c = 0; c < cfg.z_dim; ++c) {
      hm((size_t)c) = static_cast<Scalar>(cfg.latents_mean[(size_t)c]);
      hs((size_t)c) = static_cast<Scalar>(cfg.latents_std[(size_t)c]);
    }
    Kokkos::deep_copy(mean, hm);
    Kokkos::deep_copy(std, hs);
    Impl::qwen21vae_denormalize_inplace<ExecSpace>(out, mean, std,
                                                  "qwen21vae_denorm");
    return out;
  }

  void decode(const DNNHandle<ExecSpace>& handle, const V4& latents,
              const V4& rgba) const {
    cfg_.validate();
    if (!cfg_.has_affine())
      throw std::runtime_error(
          "QwenImage21VaeModel::decode: latents affine not loaded "
          "(qwen21_load_vae_config_json first)");
    const size_t B = latents.extent(0);
    const size_t H = latents.extent(2), W = latents.extent(3);
    if (latents.extent(1) != (size_t)cfg_.z_dim)
      throw std::invalid_argument(
          "QwenImage21VaeModel::decode: latents must have " +
          std::to_string(cfg_.z_dim) + " channels");
    if (H == 0 || W == 0)
      throw std::invalid_argument(
          "QwenImage21VaeModel::decode: H,W must be > 0");
    const size_t oH = (size_t)cfg_.compression * H,
                 oW = (size_t)cfg_.compression * W;
    if (rgba.extent(0) != B || rgba.extent(1) != (size_t)cfg_.out_channels ||
        rgba.extent(2) != oH || rgba.extent(3) != oW)
      throw std::invalid_argument(
          "QwenImage21VaeModel::decode: rgba must be [B,out_channels,S*H,"
          "S*W] with S = compression");
    if (B == 0) return;
    require_decoder_tensors();

    // Scratch sizes: worst-case stage over the whole walk. All decode math
    // runs in token layout [T=H*W, C]; f0/f1 ping-pong stage in/out, f2
    // holds resnet temps (tA [T,cin], tB [T,cout], shortcut [T,cout]) plus
    // the saved up-block input for the DupUp add. Buffers are
    // zero-initialized (NOT WithoutInitializing): the native Serial GEMM
    // fallback folds beta into `beta * Y_old` without a beta==0 guard, so
    // a dirty output buffer poisons the result (0 * garbage); one memset
    // per decode is negligible next to the decode itself.
    const size_t Z = (size_t)cfg_.z_dim;
    const size_t D0 = (size_t)cfg_.width_at(0);
    const int NB = cfg_.num_up_blocks();
    size_t nmax = H * W * Z;
    size_t s2max = 0, savemax = 0;
    {
      size_t t = H * W;
      nmax = std::max(nmax, t * D0);
      s2max = std::max(s2max, t * D0 + 2 * t * D0);
      for (int i = 0; i < NB; ++i) {
        int bin = 0, bout = 0;
        cfg_.block_io(i, bin, bout);
        const size_t uin = (size_t)bin, uout = (size_t)bout;
        nmax = std::max(nmax, t * uin);
        const size_t tout = cfg_.block_upsamples(i) ? 4 * t : t;
        nmax = std::max(nmax, tout * uout);
        const int NR = cfg_.resnets_per_up_block();
        for (int j = 0; j < NR; ++j) {
          const size_t cin = (j == 0) ? uin : uout;
          s2max = std::max(s2max, t * cin + 2 * t * uout);
        }
        if (cfg_.block_upsamples(i)) savemax = std::max(savemax, t * uin);
        t = tout;
      }
    }
    V1 f0("qwen21vae_f0", nmax);
    V1 f1("qwen21vae_f1", nmax);
    V1 f2("qwen21vae_f2", s2max + savemax);
    // Mid-attention buffers (exact mid geometry).
    const size_t Tmid = H * W;
    V2 qkv("qwen21vae_qkv", Tmid, 3 * D0);
    V2 qq("qwen21vae_q", Tmid, D0);
    V2 kk("qwen21vae_k", Tmid, D0);
    V2 vv("qwen21vae_v", Tmid, D0);
    V2 av("qwen21vae_av", Tmid, D0);
    V2 scores("qwen21vae_scores", Tmid, Tmid);

    for (size_t b = 0; b < B; ++b) {
      auto lat_b = Kokkos::subview(latents, b, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      auto out_b = Kokkos::subview(rgba, b, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      decode_sample(handle, lat_b, out_b, f0, f1, f2, s2max, qkv, qq, kk, vv,
                    av, scores);
    }
  }

  // VAE encode: pipeline-preprocessed RGBA image [B, in_channels, H, W] in
  // [-1,1] -> NORMALIZED latents [B, z_dim, H/S, W/S] with S = compression
  // (16), i.e. the (posterior-mode - mean)/std tensor the edit pipeline
  // conditions on. H and W must be multiples of compression (the pipeline
  // snaps them to /32). Requires the runtime latents affine (has_affine()).
  V4 encode(const DNNHandle<ExecSpace>& handle, const V4& img) const {
    cfg_.validate();
    if (!cfg_.has_affine())
      throw std::runtime_error(
          "QwenImage21VaeModel::encode: latents affine not loaded "
          "(qwen21_load_vae_config_json first)");
    const size_t B = img.extent(0);
    const size_t H = img.extent(2), W = img.extent(3);
    if (img.extent(1) != (size_t)cfg_.in_channels)
      throw std::invalid_argument(
          "QwenImage21VaeModel::encode: img must have " +
          std::to_string(cfg_.in_channels) + " channels");
    if (H == 0 || W == 0)
      throw std::invalid_argument(
          "QwenImage21VaeModel::encode: H,W must be > 0");
    const size_t S = (size_t)cfg_.compression;
    if (H % S != 0 || W % S != 0)
      throw std::invalid_argument(
          "QwenImage21VaeModel::encode: H,W must tile compression");
    if (B == 0) {
      return V4("qwen21vae_enc_empty", 0, (size_t)cfg_.z_dim, 0, 0);
    }
    require_encoder_tensors();

    // Scratch sizes: worst-case stage over the whole walk. All encode math
    // runs in token layout [T=H*W, C]; f0/f1 ping-pong stage in/out, f2
    // holds resnet temps (tA [T,cin], tB [T,cout], shortcut [T,cout]) plus
    // the saved down-block input for the AvgDown add. Buffers are
    // zero-initialized (NOT WithoutInitializing): the native Serial GEMM
    // fallback folds beta into `beta * Y_old` without a beta==0 guard, so
    // a dirty output buffer poisons the result (0 * garbage); one memset
    // per encode is negligible next to the encode itself.
    const size_t Z = (size_t)cfg_.z_dim;
    const size_t E0 = (size_t)cfg_.enc_width_at(0);
    const size_t IC = (size_t)cfg_.in_channels;
    const int NB = cfg_.num_down_blocks();
    size_t nmax = H * W * IC;
    size_t s2max = 0, savemax = 0;
    {
      size_t t = H * W;
      nmax = std::max(nmax, t * E0);
      s2max = std::max(s2max, t * IC + 2 * t * E0);
      for (int i = 0; i < NB; ++i) {
        int ein = 0, eout = 0;
        cfg_.enc_block_io(i, ein, eout);
        const size_t uin = (size_t)ein, uout = (size_t)eout;
        nmax = std::max(nmax, t * uin);
        const int NR = cfg_.resnets_per_down_block();
        for (int j = 0; j < NR; ++j) {
          const size_t cin = (j == 0) ? uin : uout;
          s2max = std::max(s2max, t * cin + 2 * t * uout);
        }
        savemax = std::max(savemax, t * uin);  // saved block input
        nmax = std::max(nmax, t * uout);  // pre-downsample full-res stage
        if (cfg_.enc_downsamples(i)) t /= 4;
        nmax = std::max(nmax, t * uout);
      }
      nmax = std::max(nmax, t * 2 * Z);
    }
    V1 f0("qwen21vae_enc_f0", nmax);
    V1 f1("qwen21vae_enc_f1", nmax);
    V1 f2("qwen21vae_enc_f2", s2max + savemax);
    // Mid-attention buffers (exact mid geometry).
    size_t tmid = H * W;
    for (int i = 0; i < NB; ++i)
      if (cfg_.enc_downsamples(i)) tmid /= 4;
    const size_t EL = (size_t)cfg_.enc_width_at(NB);
    V2 qkv("qwen21vae_enc_qkv", tmid, 3 * EL);
    V2 qq("qwen21vae_enc_q", tmid, EL);
    V2 kk("qwen21vae_enc_k", tmid, EL);
    V2 vv("qwen21vae_enc_v", tmid, EL);
    V2 av("qwen21vae_enc_av", tmid, EL);
    V2 scores("qwen21vae_enc_scores", tmid, tmid);

    V1 mean("qwen21vae_enc_mean", Z);
    V1 std("qwen21vae_enc_std", Z);
    {
      auto hm = Kokkos::create_mirror_view(mean);
      auto hs = Kokkos::create_mirror_view(std);
      for (size_t c = 0; c < Z; ++c) {
        hm(c) = static_cast<Scalar>(cfg_.latents_mean[c]);
        hs(c) = static_cast<Scalar>(cfg_.latents_std[c]);
      }
      Kokkos::deep_copy(mean, hm);
      Kokkos::deep_copy(std, hs);
    }

    V4 lat(Kokkos::ViewAllocateWithoutInitializing("qwen21vae_enc_lat"), B,
           Z, H / S, W / S);
    for (size_t b = 0; b < B; ++b) {
      auto img_b =
          Kokkos::subview(img, b, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto lat_b =
          Kokkos::subview(lat, b, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      encode_sample(handle, img_b, lat_b, f0, f1, f2, s2max, qkv, qq, kk, vv,
                    av, scores, mean, std);
    }
    return lat;
  }

 private:
  Config cfg_;
  Weights weights_;

  void require_decoder_tensors() const {
    for (const Qwen21VaeTensorSpec& s :
         qwen21vae_expected_decoder_tensors(cfg_)) {
      const auto bit = weights_.blobs.find(s.name);
      const auto sit = weights_.shapes.find(s.name);
      if (bit == weights_.blobs.end() || sit == weights_.shapes.end())
        throw std::runtime_error(
            "QwenImage21VaeModel::decode: missing tensor '" + s.name + "'");
      size_t numel = 1;
      for (size_t d : s.shape) numel *= d;
      size_t have = 1;
      for (int64_t d : sit->second) {
        if (d < 0)
          throw std::runtime_error(
              "QwenImage21VaeModel::decode: bad shape '" + s.name + "'");
        have *= (size_t)d;
      }
      if (have != numel || bit->second.extent(0) != numel)
        throw std::runtime_error(
            "QwenImage21VaeModel::decode: tensor '" + s.name +
            "' has wrong numel");
    }
    // time_conv keys are never executed on this path but must be present
    // (they are part of the decoder checkpoint); fail fast if missing.
    const int NB = cfg_.num_up_blocks();
    for (int i = 0; i < NB; ++i) {
      if (!cfg_.block_upsamples(i) || cfg_.temporal_up(i) == 0) continue;
      int bin = 0, bout = 0;
      cfg_.block_io(i, bin, bout);
      const int64_t uo = (int64_t)bout;
      const std::string u = "decoder.up_blocks." + std::to_string(i) +
                            ".upsampler.";
      Impl::qwen21vae_require_shape(weights_.shapes, u + "time_conv.weight",
                                    {2 * uo, uo, 1, 1});
      Impl::qwen21vae_require_shape(weights_.shapes, u + "time_conv.bias",
                                    {2 * uo});
      if (weights_.blobs.find(u + "time_conv.weight") ==
              weights_.blobs.end() ||
          weights_.blobs.find(u + "time_conv.bias") == weights_.blobs.end())
        throw std::runtime_error(
            "QwenImage21VaeModel::decode: missing tensor '" + u +
            "time_conv'");
    }
  }

  void require_encoder_tensors() const {
    for (const Qwen21VaeTensorSpec& s :
         qwen21vae_expected_encoder_tensors(cfg_)) {
      const auto bit = weights_.blobs.find(s.name);
      const auto sit = weights_.shapes.find(s.name);
      if (bit == weights_.blobs.end() || sit == weights_.shapes.end())
        throw std::runtime_error(
            "QwenImage21VaeModel::encode: missing tensor '" + s.name + "'");
      size_t numel = 1;
      for (size_t d : s.shape) numel *= d;
      size_t have = 1;
      for (int64_t d : sit->second) {
        if (d < 0)
          throw std::runtime_error(
              "QwenImage21VaeModel::encode: bad shape '" + s.name + "'");
        have *= (size_t)d;
      }
      if (have != numel || bit->second.extent(0) != numel)
        throw std::runtime_error(
            "QwenImage21VaeModel::encode: tensor '" + s.name +
            "' has wrong numel");
    }
    // Downsampler time_conv keys are never executed on this path (only
    // touched with feat_cache != None, i.e. chunked video inference) but
    // must be present (they are part of the encoder checkpoint).
    const int NB = cfg_.num_down_blocks();
    for (int i = 0; i < NB; ++i) {
      if (!cfg_.enc_downsamples(i) || cfg_.enc_temporal_down(i) == 0) continue;
      int ein = 0, eout = 0;
      cfg_.enc_block_io(i, ein, eout);
      const int64_t uo = (int64_t)eout;
      const std::string u = "encoder.down_blocks." + std::to_string(i) +
                            ".downsampler.";
      Impl::qwen21vae_require_shape(weights_.shapes, u + "time_conv.weight",
                                    {uo, uo, 1, 1});
      Impl::qwen21vae_require_shape(weights_.shapes, u + "time_conv.bias",
                                    {uo});
      if (weights_.blobs.find(u + "time_conv.weight") ==
              weights_.blobs.end() ||
          weights_.blobs.find(u + "time_conv.bias") == weights_.blobs.end())
        throw std::runtime_error(
            "QwenImage21VaeModel::encode: missing tensor '" + u +
            "time_conv'");
    }
  }

  // One ResNet: x [T,cin] -> y [T,cout] (tA [T,cin], tB/tS [T,cout] scratch).
  void resnet_stage(const DNNHandle<ExecSpace>& handle, const std::string& p,
                    size_t cin, size_t cout, size_t H, size_t W, const UV2& x,
                    const UV2& y, const UV2& tA, const UV2& tB, const UV2& tS,
                    const std::string& tag) const {
    const auto& blobs = weights_.blobs;
    const auto& shapes = weights_.shapes;
    if (cin != cout) {
      KokkosDNN::gemm_bias<ExecSpace, Scalar>(
          handle, x, Impl::qwen21vae_bind_mat<ExecSpace, Scalar>(
                        blobs, shapes, p + "conv_shortcut.weight", cout, cin),
          Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(
              blobs, shapes, p + "conv_shortcut.bias", cout),
          tS, tag + "_sc");
    } else {
      Kokkos::deep_copy(tS, x);
    }
    Kokkos::deep_copy(tA, x);
    Impl::qwen21vae_rmsnorm_inplace<ExecSpace>(
        tA,
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "norm1.gamma", cin),
        tag + "_n1");
    Impl::qwen21vae_silu_inplace<ExecSpace>(tA, tag + "_s1");
    KokkosDNN::conv3x3_forward<ExecSpace, Scalar>(
        handle, tA,
        Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(blobs, shapes,
                                                    p + "conv1.weight", cout,
                                                    cin, 3),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "conv1.bias", cout),
        tB, H, W, tag + "_c1");
    Impl::qwen21vae_rmsnorm_inplace<ExecSpace>(
        tB,
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "norm2.gamma", cout),
        tag + "_n2");
    Impl::qwen21vae_silu_inplace<ExecSpace>(tB, tag + "_s2");
    KokkosDNN::conv3x3_forward<ExecSpace, Scalar>(
        handle, tB,
        Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(blobs, shapes,
                                                    p + "conv2.weight", cout,
                                                    cout, 3),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "conv2.bias", cout),
        y, H, W, tag + "_c2");
    Impl::qwen21vae_add_inplace<ExecSpace>(y, tS, tag + "_add");
  }

  // Mid-block single-head self-attention: x [T,C] -> y [T,C].
  void attn_stage(const DNNHandle<ExecSpace>& handle, const std::string& ap,
                  const UV2& x, const UV2& y, size_t H, size_t W, const V2& qkv,
                  const V2& qq, const V2& kk, const V2& vv, const V2& av,
                  const V2& scores, const std::string& tag) const {
    const auto& blobs = weights_.blobs;
    const auto& shapes = weights_.shapes;
    const std::string p(ap);
    const size_t T = x.extent(0), C = x.extent(1);
    if (H * W != T || y.extent(0) != T || y.extent(1) != C)
      throw std::invalid_argument(
          "QwenImage21VaeModel::attn_stage: dimension mismatch");
    // Attention RMSNorm (gamma [C,1,1]) into the av scratch, then the
    // qkv projection. av is reused below as the attention-value output.
    Kokkos::deep_copy(av, x);
    Impl::qwen21vae_rmsnorm_inplace<ExecSpace>(
        av,
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "norm.gamma", C),
        tag + "_norm");
    KokkosDNN::gemm_bias<ExecSpace, Scalar>(
        handle, av,
        Impl::qwen21vae_bind_mat<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "to_qkv.weight", 3 * C,
                                                   C),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "to_qkv.bias", 3 * C),
        qkv, tag + "_qkv");
    Impl::qwen21vae_split_qkv<ExecSpace>(qkv, qq, kk, vv, tag + "_split");
    Impl::qwen21vae_attn_scores<ExecSpace>(qq, kk, 1.0f / sqrtf((float)C),
                                           scores, tag + "_scores");
    Impl::qwen21vae_softmax_rows_inplace<ExecSpace>(scores, tag + "_softmax");
    Impl::qwen21vae_attn_av<ExecSpace>(scores, vv, av, tag + "_av");
    KokkosDNN::gemm_bias<ExecSpace, Scalar>(
        handle, av,
        Impl::qwen21vae_bind_mat<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "proj.weight", C, C),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   p + "proj.bias", C),
        y, tag + "_proj");
    Impl::qwen21vae_add_inplace<ExecSpace>(y, x, tag + "_add");
  }

  template <class V3>
  void decode_sample(const DNNHandle<ExecSpace>& handle, const V3& lat_b,
                     const V3& out_b, const V1& f0, const V1& f1, const V1& f2,
                     size_t s2max, const V2& qkv, const V2& qq, const V2& kk,
                     const V2& vv, const V2& av, const V2& scores) const {
    const auto& blobs = weights_.blobs;
    const auto& shapes = weights_.shapes;
    const size_t Z = (size_t)cfg_.z_dim;
    const size_t D0 = (size_t)cfg_.width_at(0);
    const int NB = cfg_.num_up_blocks();
    const int NR = cfg_.resnets_per_up_block();
    size_t H = lat_b.extent(1), W = lat_b.extent(2);
    size_t T = H * W;
    const V1* F[2] = {&f0, &f1};
    int cur = 0;
    auto stage = [&](int which, size_t t, size_t c) {
      return UV2(F[which]->data(), t, c);
    };
    // Post-quant 1x1 + conv_in 3x3.
    Impl::qwen21vae_nchw_to_tokens<ExecSpace>(lat_b, stage(cur, T, Z),
                                              "qwen21vae_lat");
    KokkosDNN::gemm_bias<ExecSpace, Scalar>(
        handle, stage(cur, T, Z),
        Impl::qwen21vae_bind_mat<ExecSpace, Scalar>(blobs, shapes,
                                                   "post_quant_conv.weight",
                                                   Z, Z),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(
            blobs, shapes, "post_quant_conv.bias", Z),
        stage(1 - cur, T, Z), "qwen21vae_pq");
    cur = 1 - cur;
    KokkosDNN::conv3x3_forward<ExecSpace, Scalar>(
        handle, stage(cur, T, Z),
        Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(
            blobs, shapes, "decoder.conv_in.weight", D0, Z, 3),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   "decoder.conv_in.bias",
                                                   D0),
        stage(1 - cur, T, D0), H, W, "qwen21vae_conv_in");
    cur = 1 - cur;
    auto tmp_alias = [&](size_t off, size_t t, size_t c) {
      return UV2(f2.data() + off, t, c);
    };
    const std::string mid_p("decoder.mid_block.resnets.");
    resnet_stage(handle, mid_p + "0.", D0, D0, H, W, stage(cur, T, D0),
                 stage(1 - cur, T, D0), tmp_alias(0, T, D0),
                 tmp_alias(T * D0, T, D0), tmp_alias(2 * T * D0, T, D0),
                 "qwen21vae_mid_a");
    cur = 1 - cur;
    attn_stage(handle, "decoder.mid_block.attentions.0.", stage(cur, T, D0),
               stage(1 - cur, T, D0), H, W, qkv, qq, kk, vv, av, scores,
               "qwen21vae_mid_attn");
    cur = 1 - cur;
    resnet_stage(handle, mid_p + "1.", D0, D0, H, W, stage(cur, T, D0),
                 stage(1 - cur, T, D0), tmp_alias(0, T, D0),
                 tmp_alias(T * D0, T, D0), tmp_alias(2 * T * D0, T, D0),
                 "qwen21vae_mid_b");
    cur = 1 - cur;
    // Up blocks.
    for (int i = 0; i < NB; ++i) {
      int bin = 0, bout = 0;
      cfg_.block_io(i, bin, bout);
      const size_t uin = (size_t)bin, uout = (size_t)bout;
      const bool up = cfg_.block_upsamples(i);
      UV2 saved = tmp_alias(s2max, T, uin);
      if (up) Kokkos::deep_copy(saved, stage(cur, T, uin));
      for (int j = 0; j < NR; ++j) {
        const size_t cin = (j == 0) ? uin : uout;
        const std::string p = "decoder.up_blocks." + std::to_string(i) +
                              ".resnets." + std::to_string(j) + ".";
        const std::string tag =
            "qwen21vae_up" + std::to_string(i) + "r" + std::to_string(j);
        resnet_stage(handle, p, cin, uout, H, W, stage(cur, T, cin),
                     stage(1 - cur, T, uout), tmp_alias(0, T, cin),
                     tmp_alias(T * cin, T, uout),
                     tmp_alias(T * cin + T * uout, T, uout), tag);
        cur = 1 - cur;
      }
      if (up) {
        const std::string u = "decoder.up_blocks." + std::to_string(i) +
                              ".upsampler.";
        Impl::qwen21vae_nearest_upsample2x<ExecSpace>(
            stage(cur, T, uout), H, W, stage(1 - cur, 4 * T, uout),
            "qwen21vae_up_nn" + std::to_string(i));
        cur = 1 - cur;
        H *= 2;
        W *= 2;
        T *= 4;
        KokkosDNN::conv3x3_forward<ExecSpace, Scalar>(
            handle, stage(cur, T, uout),
            Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(
                blobs, shapes, u + "resample.1.weight", uout, uout, 3),
            Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(
                blobs, shapes, u + "resample.1.bias", uout),
            stage(1 - cur, T, uout), H, W,
            "qwen21vae_up_conv" + std::to_string(i));
        cur = 1 - cur;
        const int ft = (cfg_.temporal_up(i) != 0) ? 2 : 1;
        const size_t repeats = uout * (size_t)ft * 4 / uin;
        if (uout * (size_t)ft * 4 != repeats * uin)
          throw std::runtime_error(
              "QwenImage21VaeModel::decode: bad DupUp repeats");
        Impl::qwen21vae_dupup_add<ExecSpace>(stage(cur, T, uout), saved, H / 2,
                                             W / 2, ft, repeats,
                                             "qwen21vae_dupup" +
                                                 std::to_string(i));
      }
    }
    // Head: RMSNorm -> SiLU -> conv_out 3x3 -> clamp -> NCHW RGBA.
    const size_t DL = (size_t)cfg_.width_at(NB);
    const size_t OC = (size_t)cfg_.out_channels;
    Impl::qwen21vae_rmsnorm_inplace<ExecSpace>(
        stage(cur, T, DL),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                   "decoder.norm_out.gamma",
                                                   DL),
        "qwen21vae_norm_out");
    Impl::qwen21vae_silu_inplace<ExecSpace>(stage(cur, T, DL),
                                            "qwen21vae_head_silu");
    KokkosDNN::conv3x3_forward<ExecSpace, Scalar>(
        handle, stage(cur, T, DL),
        Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(
            blobs, shapes, "decoder.conv_out.weight", OC, DL, 3),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                  "decoder.conv_out.bias",
                                                  OC),
        stage(1 - cur, T, OC), H, W, "qwen21vae_conv_out");
    cur = 1 - cur;
    Impl::qwen21vae_clamp_inplace<ExecSpace>(stage(cur, T, OC), -1.0f, 1.0f,
                                            "qwen21vae_clamp");
    Impl::qwen21vae_tokens_to_nchw<ExecSpace>(stage(cur, T, OC), out_b,
                                              "qwen21vae_out");
  }

  template <class V3>
  void encode_sample(const DNNHandle<ExecSpace>& handle, const V3& img_b,
                     const V3& lat_b, const V1& f0, const V1& f1, const V1& f2,
                     size_t s2max, const V2& qkv, const V2& qq, const V2& kk,
                     const V2& vv, const V2& av, const V2& scores,
                     const V1& mean, const V1& std) const {
    const auto& blobs = weights_.blobs;
    const auto& shapes = weights_.shapes;
    const size_t Z = (size_t)cfg_.z_dim;
    const size_t E0 = (size_t)cfg_.enc_width_at(0);
    const size_t IC = (size_t)cfg_.in_channels;
    const int NB = cfg_.num_down_blocks();
    const int NR = cfg_.resnets_per_down_block();
    size_t H = img_b.extent(1), W = img_b.extent(2);
    size_t T = H * W;
    const V1* F[2] = {&f0, &f1};
    int cur = 0;
    auto stage = [&](int which, size_t t, size_t c) {
      return UV2(F[which]->data(), t, c);
    };
    // conv_in 3x3.
    Impl::qwen21vae_nchw_to_tokens<ExecSpace>(img_b, stage(cur, T, IC),
                                              "qwen21vae_enc_img");
    KokkosDNN::conv3x3_forward<ExecSpace, Scalar>(
        handle, stage(cur, T, IC),
        Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(
            blobs, shapes, "encoder.conv_in.weight", E0, IC, 3),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                    "encoder.conv_in.bias",
                                                    E0),
        stage(1 - cur, T, E0), H, W, "qwen21vae_enc_conv_in");
    cur = 1 - cur;
    auto tmp_alias = [&](size_t off, size_t t, size_t c) {
      return UV2(f2.data() + off, t, c);
    };
    // Down blocks: resnets, then the stride-2 downsample + AvgDown3D
    // shortcut add when downsampling, else the plain AvgDown3D(ft=1,fs=1)
    // residual add (the block-level shortcut exists even without a
    // downsampler, mirroring QwenImage21ResidualDownBlock.forward).
    for (int i = 0; i < NB; ++i) {
      int ein = 0, eout = 0;
      cfg_.enc_block_io(i, ein, eout);
      const size_t uin = (size_t)ein, uout = (size_t)eout;
      const bool down = cfg_.enc_downsamples(i);
      UV2 saved = tmp_alias(s2max, T, uin);
      Kokkos::deep_copy(saved, stage(cur, T, uin));
      for (int j = 0; j < NR; ++j) {
        const size_t cin = (j == 0) ? uin : uout;
        const std::string p = "encoder.down_blocks." + std::to_string(i) +
                              ".resnets." + std::to_string(j) + ".";
        const std::string tag =
            "qwen21vae_enc_d" + std::to_string(i) + "r" + std::to_string(j);
        resnet_stage(handle, p, cin, uout, H, W, stage(cur, T, cin),
                     stage(1 - cur, T, uout), tmp_alias(0, T, cin),
                     tmp_alias(T * cin, T, uout),
                     tmp_alias(T * cin + T * uout, T, uout), tag);
        cur = 1 - cur;
      }
      if (down) {
        const std::string u =
            "encoder.down_blocks." + std::to_string(i) + ".downsampler.";
        Impl::qwen21vae_downsample_conv3x3_stride2<ExecSpace>(
            stage(cur, T, uout), H, W,
            Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(
                blobs, shapes, u + "resample.1.weight", uout, uout, 3),
            Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(
                blobs, shapes, u + "resample.1.bias", uout),
            stage(1 - cur, T / 4, uout), "qwen21vae_enc_down" + std::to_string(i));
        cur = 1 - cur;
        H /= 2;
        W /= 2;
        T /= 4;
        const int ft = (cfg_.enc_temporal_down(i) != 0) ? 2 : 1;
        Impl::qwen21vae_avgdown_add<ExecSpace>(stage(cur, T, uout), saved, H * 2,
                                              W * 2, uin, uout, ft, 2,
                                              "qwen21vae_enc_avgdown" +
                                                  std::to_string(i));
      } else {
        Impl::qwen21vae_avgdown_add<ExecSpace>(stage(cur, T, uout), saved, H, W,
                                              uin, uout, 1, 1,
                                              "qwen21vae_enc_avgdown" +
                                                  std::to_string(i));
      }
    }
    // Mid block.
    const size_t EL = (size_t)cfg_.enc_width_at(NB);
    const std::string mid_p("encoder.mid_block.resnets.");
    resnet_stage(handle, mid_p + "0.", EL, EL, H, W, stage(cur, T, EL),
                 stage(1 - cur, T, EL), tmp_alias(0, T, EL),
                 tmp_alias(T * EL, T, EL), tmp_alias(2 * T * EL, T, EL),
                 "qwen21vae_enc_mid_a");
    cur = 1 - cur;
    attn_stage(handle, "encoder.mid_block.attentions.0.", stage(cur, T, EL),
               stage(1 - cur, T, EL), H, W, qkv, qq, kk, vv, av, scores,
               "qwen21vae_enc_mid_attn");
    cur = 1 - cur;
    resnet_stage(handle, mid_p + "1.", EL, EL, H, W, stage(cur, T, EL),
                 stage(1 - cur, T, EL), tmp_alias(0, T, EL),
                 tmp_alias(T * EL, T, EL), tmp_alias(2 * T * EL, T, EL),
                 "qwen21vae_enc_mid_b");
    cur = 1 - cur;
    // Head: RMSNorm -> SiLU -> conv_out 3x3 -> quant 1x1 -> mode+normalize.
    Impl::qwen21vae_rmsnorm_inplace<ExecSpace>(
        stage(cur, T, EL),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                    "encoder.norm_out.gamma",
                                                    EL),
        "qwen21vae_enc_norm_out");
    Impl::qwen21vae_silu_inplace<ExecSpace>(stage(cur, T, EL),
                                            "qwen21vae_enc_head_silu");
    KokkosDNN::conv3x3_forward<ExecSpace, Scalar>(
        handle, stage(cur, T, EL),
        Impl::qwen21vae_bind_conv<ExecSpace, Scalar>(
            blobs, shapes, "encoder.conv_out.weight", 2 * Z, EL, 3),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(
            blobs, shapes, "encoder.conv_out.bias", 2 * Z),
        stage(1 - cur, T, 2 * Z), H, W, "qwen21vae_enc_conv_out");
    cur = 1 - cur;
    KokkosDNN::gemm_bias<ExecSpace, Scalar>(
        handle, stage(cur, T, 2 * Z),
        Impl::qwen21vae_bind_mat<ExecSpace, Scalar>(blobs, shapes,
                                                    "quant_conv.weight", 2 * Z,
                                                    2 * Z),
        Impl::qwen21vae_bind_vec<ExecSpace, Scalar>(blobs, shapes,
                                                    "quant_conv.bias", 2 * Z),
        stage(1 - cur, T, 2 * Z), "qwen21vae_enc_quant");
    cur = 1 - cur;
    Impl::qwen21vae_quant_mode_normalize<ExecSpace>(
        stage(cur, T, 2 * Z), mean, std, stage(1 - cur, T, Z),
        "qwen21vae_enc_norm");
    cur = 1 - cur;
    Impl::qwen21vae_tokens_to_nchw<ExecSpace>(stage(cur, T, Z), lat_b,
                                              "qwen21vae_enc_out");
  }
};

}  // namespace QwenImage21Vae
}  // namespace KokkosDNN

#endif
