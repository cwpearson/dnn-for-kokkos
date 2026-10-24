// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21VAE_HPP
#define KOKKOSDNN_QWENIMAGE21VAE_HPP

// AutoencoderKLQwenImage21 for Qwen-Image-2.1 (text-to-image decode path).
// Ground truth (HF research): in/out 4 ch, z_dim 64, 16x compression,
// base_dim 96 / decoder base 144, ch_mult [1, 2, 4, 8, 8]; per-channel
// latents_mean / latents_std ([64] each) affine, read from vae/config.json
// AT RUNTIME (never hardcoded); single-shard 675 MB bf16; decode -> RGBA
// (alpha channel appended downstream of the 4-channels... see decode TODO).
//
// This header is complete as a Config + runtime config.json reader +
// agnostic weights loader (conv/resnet/attn blobs keyed by checkpoint
// name); the convolutional decode() is a skeleton (TODO golden test) and
// currently throws.
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cctype>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Handle.hpp"

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
  double norm_eps = 1e-6;

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
    c.compression = 4;
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

// VAE weights: agnostic device blobs keyed by checkpoint tensor name
// (conv / resnet / attn / norm tensors; exact AutoencoderKLQwenImage21 key
// layout is a golden TODO). Single-shard 675 MB bf16 (F32 also accepted).
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

  // Load every tensor found in the shard(s) into device blobs (F32/BF16
  // file -> Scalar). Unknown-to-any-skeleton names are accepted here
  // (the conv key layout is still TODO); dtype/shape/slice errors throw.
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
};

// VAE model: owns Weights. decode() (conv decoder latents [B, z, h, w] ->
// RGB image, plus RGBA alpha handling) is a TODO golden test; it currently
// throws. Requires the runtime latents affine (has_affine()).
template <class ExecSpace, class Scalar>
class QwenImage21VaeModel {
 public:
  using V4 = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using Weights = QwenImage21VaeWeights<ExecSpace, Scalar>;

  explicit QwenImage21VaeModel(const Config& cfg = Config()) {
    cfg.validate();
    cfg_ = cfg;
    weights_.allocate(cfg_);
  }

  const Config& config() const { return cfg_; }
  Weights& weights() { return weights_; }
  const Weights& weights() const { return weights_; }

  // latents [B, z_dim, h, w] (already scaled: (z - mean) / std applied by
  // the caller from cfg.latents_mean/std) -> rgb [B, 3, H, W].
  // TODO: implement the convolutional decoder (golden test).
  void decode(const DNNHandle<ExecSpace>& handle, const V4& latents,
              const V4& rgb) {
    (void)handle;
    (void)latents;
    (void)rgb;
    cfg_.validate();
    if (!cfg_.has_affine())
      throw std::runtime_error(
          "QwenImage21VaeModel::decode: latents affine not loaded "
          "(qwen21_load_vae_config_json first)");
    throw std::runtime_error(
        "QwenImage21VaeModel::decode: TODO (conv decoder not yet "
        "implemented; golden test)");
  }

 private:
  Config cfg_;
  Weights weights_;
};

}  // namespace QwenImage21Vae
}  // namespace KokkosDNN

#endif
