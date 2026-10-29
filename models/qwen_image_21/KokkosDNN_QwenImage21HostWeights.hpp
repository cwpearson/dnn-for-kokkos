// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 host-staged weights: converted HostSpace weight copies
// held across runs, staged host->device per phase.
//
// Unlike QwenImage21::ShardCache (raw file bytes, so every run re-parses,
// re-converts, and re-uploads), HostWeights loads each shard ONCE into
// converted HostSpace views (warm_* reads from disk directly, with no
// device allocation). Every run then allocates fresh device weights and
// deep_copies host->device (stage_*_to), releasing the device copy when
// the phase ends. Host cost is one converted copy (~32 GB full
// checkpoint); there is no second raw-bytes copy.
//
// Single-flight use (like the drivers): warm once, then share across runs
// under the caller's mutex. A null pointer at the phase call sites keeps
// the legacy load-from-disk path.
#ifndef KOKKOSDNN_QWENIMAGE21HOSTWEIGHTS_HPP
#define KOKKOSDNN_QWENIMAGE21HOSTWEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN_QwenImage21Dit.hpp"
#include "KokkosDNN_QwenImage21Encoder.hpp"
#include "KokkosDNN_QwenImage21Vae.hpp"
#include "KokkosDNN_QwenImage21Vision.hpp"

namespace KokkosDNN {
namespace QwenImage21 {

// Host<->device weight copy (both directions: instantiate with
// DstExec/SrcExec as needed). Both sides must be allocated from the same
// config; extents are checked by deep_copy.
template <class DstExec, class SrcExec, class Scalar>
void copy_dit_weights(
    QwenImage21Dit::QwenImage21DitWeights<DstExec, Scalar>& dst,
    const QwenImage21Dit::QwenImage21DitWeights<SrcExec, Scalar>& src) {
  if (dst.blocks.size() != src.blocks.size())
    throw std::runtime_error("copy_dit_weights: block count mismatch");
  Kokkos::deep_copy(dst.txt_norm, src.txt_norm);
  Kokkos::deep_copy(dst.txt_in_W, src.txt_in_W);
  Kokkos::deep_copy(dst.txt_out_W, src.txt_out_W);
  Kokkos::deep_copy(dst.img_in_W, src.img_in_W);
  Kokkos::deep_copy(dst.time1_W, src.time1_W);
  Kokkos::deep_copy(dst.time2_W, src.time2_W);
  Kokkos::deep_copy(dst.mod_W, src.mod_W);
  Kokkos::deep_copy(dst.norm_out_W, src.norm_out_W);
  Kokkos::deep_copy(dst.proj_out_W, src.proj_out_W);
  for (size_t i = 0; i < dst.blocks.size(); ++i) {
    auto& d = dst.blocks[i];
    const auto& s = src.blocks[i];
    Kokkos::deep_copy(d.to_q, s.to_q);
    Kokkos::deep_copy(d.to_k, s.to_k);
    Kokkos::deep_copy(d.to_v, s.to_v);
    Kokkos::deep_copy(d.to_o, s.to_o);
    Kokkos::deep_copy(d.norm_q, s.norm_q);
    Kokkos::deep_copy(d.norm_k, s.norm_k);
    Kokkos::deep_copy(d.proj, s.proj);
    Kokkos::deep_copy(d.gate, s.gate);
    Kokkos::deep_copy(d.out, s.out);
  }
  Kokkos::fence();
}

template <class DstExec, class SrcExec, class Scalar>
void copy_encoder_weights(
    QwenImage21Encoder::QwenImage21EncoderWeights<DstExec, Scalar>& dst,
    const QwenImage21Encoder::QwenImage21EncoderWeights<SrcExec, Scalar>&
        src) {
  if (dst.layers.size() != src.layers.size())
    throw std::runtime_error("copy_encoder_weights: layer count mismatch");
  Kokkos::deep_copy(dst.embed, src.embed);
  Kokkos::deep_copy(dst.final_norm, src.final_norm);
  for (size_t i = 0; i < dst.layers.size(); ++i) {
    auto& d = dst.layers[i];
    const auto& s = src.layers[i];
    Kokkos::deep_copy(d.q_W, s.q_W);
    Kokkos::deep_copy(d.k_W, s.k_W);
    Kokkos::deep_copy(d.v_W, s.v_W);
    Kokkos::deep_copy(d.o_W, s.o_W);
    Kokkos::deep_copy(d.qn, s.qn);
    Kokkos::deep_copy(d.kn, s.kn);
    Kokkos::deep_copy(d.in_norm, s.in_norm);
    Kokkos::deep_copy(d.post_norm, s.post_norm);
    Kokkos::deep_copy(d.gate_W, s.gate_W);
    Kokkos::deep_copy(d.up_W, s.up_W);
    Kokkos::deep_copy(d.down_W, s.down_W);
  }
  Kokkos::fence();
}

template <class DstExec, class SrcExec, class Scalar>
void copy_vision_weights(
    QwenImage21Vision::QwenImage21VisionWeights<DstExec, Scalar>& dst,
    const QwenImage21Vision::QwenImage21VisionWeights<SrcExec, Scalar>&
        src) {
  if (dst.layers.size() != src.layers.size() ||
      dst.deep.size() != src.deep.size())
    throw std::runtime_error("copy_vision_weights: layer count mismatch");
  Kokkos::deep_copy(dst.patch_W, src.patch_W);
  Kokkos::deep_copy(dst.patch_b, src.patch_b);
  Kokkos::deep_copy(dst.pos_table, src.pos_table);
  for (size_t i = 0; i < dst.layers.size(); ++i) {
    auto& d = dst.layers[i];
    const auto& s = src.layers[i];
    Kokkos::deep_copy(d.qkv_W, s.qkv_W);
    Kokkos::deep_copy(d.qkv_b, s.qkv_b);
    Kokkos::deep_copy(d.proj_W, s.proj_W);
    Kokkos::deep_copy(d.proj_b, s.proj_b);
    Kokkos::deep_copy(d.n1_w, s.n1_w);
    Kokkos::deep_copy(d.n1_b, s.n1_b);
    Kokkos::deep_copy(d.n2_w, s.n2_w);
    Kokkos::deep_copy(d.n2_b, s.n2_b);
    Kokkos::deep_copy(d.fc1_W, s.fc1_W);
    Kokkos::deep_copy(d.fc1_b, s.fc1_b);
    Kokkos::deep_copy(d.fc2_W, s.fc2_W);
    Kokkos::deep_copy(d.fc2_b, s.fc2_b);
  }
  Kokkos::deep_copy(dst.merge_norm_w, src.merge_norm_w);
  Kokkos::deep_copy(dst.merge_norm_b, src.merge_norm_b);
  Kokkos::deep_copy(dst.merge_fc1_W, src.merge_fc1_W);
  Kokkos::deep_copy(dst.merge_fc1_b, src.merge_fc1_b);
  Kokkos::deep_copy(dst.merge_fc2_W, src.merge_fc2_W);
  Kokkos::deep_copy(dst.merge_fc2_b, src.merge_fc2_b);
  for (size_t i = 0; i < dst.deep.size(); ++i) {
    auto& d = dst.deep[i];
    const auto& s = src.deep[i];
    Kokkos::deep_copy(d.norm_w, s.norm_w);
    Kokkos::deep_copy(d.norm_b, s.norm_b);
    Kokkos::deep_copy(d.fc1_W, s.fc1_W);
    Kokkos::deep_copy(d.fc1_b, s.fc1_b);
    Kokkos::deep_copy(d.fc2_W, s.fc2_W);
    Kokkos::deep_copy(d.fc2_b, s.fc2_b);
  }
  Kokkos::fence();
}

template <class DstExec, class SrcExec, class Scalar>
void copy_vae_weights(
    QwenImage21Vae::QwenImage21VaeWeights<DstExec, Scalar>& dst,
    const QwenImage21Vae::QwenImage21VaeWeights<SrcExec, Scalar>& src) {
  dst.cfg = src.cfg;
  dst.shapes = src.shapes;
  dst.dtypes = src.dtypes;
  dst.blobs.clear();
  for (const auto& kv : src.blobs) {
    typename QwenImage21Vae::QwenImage21VaeWeights<DstExec, Scalar>::V1 dev(
        kv.first.c_str(), kv.second.extent(0));
    Kokkos::deep_copy(dev, kv.second);
    dst.blobs.emplace(kv.first, std::move(dev));
  }
  Kokkos::fence();
}

// Converted HostSpace weight copies held across runs. Warm once (disk ->
// host, no device allocation); every phase then stages host -> device via
// stage_*_to and releases the device copy when the phase ends. The vision
// and encoder towers share one text_encoder shard on disk but hold
// separate converted copies here (each ~its own resident_bytes).
template <class Scalar = Kokkos::Experimental::bhalf_t>
struct HostWeights {
  QwenImage21::Config dit_cfg;
  QwenImage21Encoder::Config enc_cfg;
  QwenImage21Vision::Config vis_cfg;
  QwenImage21Vae::Config vae_cfg;
  QwenImage21Dit::QwenImage21DitWeights<Kokkos::HostSpace, Scalar> dit;
  QwenImage21Encoder::QwenImage21EncoderWeights<Kokkos::HostSpace, Scalar>
      encoder;
  QwenImage21Vision::QwenImage21VisionWeights<Kokkos::HostSpace, Scalar>
      vision;
  QwenImage21Vae::QwenImage21VaeWeights<Kokkos::HostSpace, Scalar> vae;
  bool has_dit = false;
  bool has_encoder = false;
  bool has_vision = false;
  bool has_vae = false;

  void warm_dit(const QwenImage21::Config& cfg,
                const std::vector<std::string>& shards) {
    cfg.validate();
    dit_cfg = cfg;
    dit.allocate(cfg);
    dit.load_safetensors(shards);
    has_dit = true;
  }

  void warm_encoder(const QwenImage21Encoder::Config& cfg,
                    const std::string& shard_path) {
    cfg.validate();
    enc_cfg = cfg;
    encoder.allocate(cfg);
    encoder.load_safetensors(shard_path);
    has_encoder = true;
  }

  void warm_vision(const QwenImage21Vision::Config& cfg,
                   const std::string& shard_path) {
    cfg.validate();
    vis_cfg = cfg;
    vision.allocate(cfg);
    vision.load_safetensors(shard_path);
    has_vision = true;
  }

  void warm_vae(const QwenImage21Vae::Config& cfg,
                const std::vector<std::string>& shards) {
    cfg.validate();
    vae_cfg = cfg;
    vae.allocate(cfg);
    vae.load_safetensors(shards);
    has_vae = true;
  }

  template <class DExec>
  void stage_dit_to(
      QwenImage21Dit::QwenImage21DitWeights<DExec, Scalar>& dst) const {
    if (!has_dit)
      throw std::runtime_error("HostWeights::stage_dit_to: not warmed");
    dst.allocate(dit_cfg);
    copy_dit_weights(dst, dit);
  }

  template <class DExec>
  void stage_encoder_to(
      QwenImage21Encoder::QwenImage21EncoderWeights<DExec, Scalar>& dst)
      const {
    if (!has_encoder)
      throw std::runtime_error("HostWeights::stage_encoder_to: not warmed");
    dst.allocate(enc_cfg);
    copy_encoder_weights(dst, encoder);
  }

  template <class DExec>
  void stage_vision_to(
      QwenImage21Vision::QwenImage21VisionWeights<DExec, Scalar>& dst) const {
    if (!has_vision)
      throw std::runtime_error("HostWeights::stage_vision_to: not warmed");
    dst.allocate(vis_cfg);
    copy_vision_weights(dst, vision);
  }

  template <class DExec>
  void stage_vae_to(
      QwenImage21Vae::QwenImage21VaeWeights<DExec, Scalar>& dst) const {
    if (!has_vae)
      throw std::runtime_error("HostWeights::stage_vae_to: not warmed");
    dst.allocate(vae_cfg);
    copy_vae_weights(dst, vae);
  }

  void clear() {
    dit.release();
    encoder.release();
    vision.release();
    vae.release();
    has_dit = has_encoder = has_vision = has_vae = false;
  }

  size_t host_bytes() const {
    size_t n = 0;
    if (has_dit) n += dit.resident_bytes();
    if (has_encoder) n += encoder.resident_bytes();
    if (has_vision) n += vision.resident_bytes();
    if (has_vae) n += vae.resident_bytes();
    return n;
  }
};

}  // namespace QwenImage21
}  // namespace KokkosDNN

#endif
