// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 host-staged weights tests: fabricated minimal safetensors
// shards warmed once into HostWeights (converted HostSpace copies) then
// staged host->device per phase, bit-identical to direct disk loads for
// every weights type (DiT / encoder+vision sharing one text_encoder-style
// shard / VAE blobs), in F32 and BF16 file dtypes, plus stage-guard and
// accounting semantics.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "KokkosDNN_QwenImage21Config.hpp"
#include "KokkosDNN_QwenImage21Dit.hpp"
#include "KokkosDNN_QwenImage21Encoder.hpp"
#include "KokkosDNN_QwenImage21HostWeights.hpp"
#include "KokkosDNN_QwenImage21Vae.hpp"
#include "KokkosDNN_QwenImage21Vision.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
namespace QW = KokkosDNN::QwenImage21;
namespace QD = KokkosDNN::QwenImage21Dit;
namespace QE = KokkosDNN::QwenImage21Encoder;
namespace QV = KokkosDNN::QwenImage21Vision;
namespace QVV = KokkosDNN::QwenImage21Vae;

struct FabTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
  std::string dtype = "F32";
};

uint16_t f32_to_bf16_bits(float v) {
  uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  return (uint16_t)(u >> 16);
}

void write_shard(const std::string& path,
                 const std::vector<FabTensor>& tensors) {
  std::string js = "{\"__metadata__\":{},";
  size_t off = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const FabTensor& t = tensors[i];
    size_t numel = 1;
    for (int64_t d : t.shape) numel *= (size_t)d;
    ASSERT_EQ(t.data.size(), numel) << t.name;
    const size_t elsize = (t.dtype == "BF16") ? 2 : 4;
    if (i > 0) js += ",";
    js += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t k = 0; k < t.shape.size(); ++k)
      js += (k ? "," : "") + std::to_string(t.shape[k]);
    js += "],\"data_offsets\":[" + std::to_string(off) + "," +
          std::to_string(off + numel * elsize) + "]}";
    off += numel * elsize;
  }
  js += "}";
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE((bool)f) << path;
  const uint64_t hn = js.size();
  char hlen[8];
  for (int i = 0; i < 8; ++i) hlen[i] = (char)((hn >> (8 * i)) & 0xff);
  f.write(hlen, 8);
  f.write(js.data(), (std::streamsize)js.size());
  for (const auto& t : tensors) {
    if (t.dtype == "BF16") {
      for (float v : t.data) {
        const uint16_t b = f32_to_bf16_bits(v);
        f.write(reinterpret_cast<const char*>(&b), 2);
      }
    } else {
      f.write(reinterpret_cast<const char*>(t.data.data()),
              (std::streamsize)(t.data.size() * 4));
    }
  }
  f.close();
  ASSERT_TRUE((bool)f) << path;
}

std::string test_dir(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("qwen21_hostweights_" + std::to_string(::getpid())) / name;
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  return dir.string();
}

std::vector<float> payload(size_t numel, size_t seed, bool bf16) {
  std::vector<float> out(numel);
  for (size_t i = 0; i < numel; ++i)
    out[i] = bf16 ? (float)(((i + seed) % 64) - 32)
                  : (float)(((i + seed) % 128) - 64) * 0.25f;
  return out;
}

template <class V>
std::vector<float> to_host(const V& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  std::vector<float> out(h.size());
  for (size_t i = 0; i < h.size(); ++i)
    out[i] = static_cast<float>(h.data()[i]);
  return out;
}

using ViewMap = std::map<std::string, std::vector<float>>;

ViewMap dit_views(const QD::QwenImage21DitWeights<Exec, float>& w) {
  ViewMap m;
  m["txt_in.text_norm.weight"] = to_host(w.txt_norm);
  m["txt_in.in_layer.weight"] = to_host(w.txt_in_W);
  m["txt_in.out_layer.weight"] = to_host(w.txt_out_W);
  m["img_in.weight"] = to_host(w.img_in_W);
  m["time_text_embed.timestep_embedder.linear_1.weight"] = to_host(w.time1_W);
  m["time_text_embed.timestep_embedder.linear_2.weight"] = to_host(w.time2_W);
  m["modulation.1.weight"] = to_host(w.mod_W);
  m["norm_out.linear.weight"] = to_host(w.norm_out_W);
  m["proj_out.weight"] = to_host(w.proj_out_W);
  for (size_t i = 0; i < w.blocks.size(); ++i) {
    const std::string p = "transformer_blocks." + std::to_string(i) + ".";
    const auto& b = w.blocks[i];
    m[p + "attn.to_q.weight"] = to_host(b.to_q);
    m[p + "attn.to_k.weight"] = to_host(b.to_k);
    m[p + "attn.to_v.weight"] = to_host(b.to_v);
    m[p + "attn.to_out.0.weight"] = to_host(b.to_o);
    m[p + "attn.norm_q.weight"] = to_host(b.norm_q);
    m[p + "attn.norm_k.weight"] = to_host(b.norm_k);
    m[p + "img_mlp.proj.weight"] = to_host(b.proj);
    m[p + "img_mlp.gate_layer.weight"] = to_host(b.gate);
    m[p + "img_mlp.out.weight"] = to_host(b.out);
  }
  return m;
}

ViewMap enc_views(const QE::QwenImage21EncoderWeights<Exec, float>& w) {
  ViewMap m;
  m["model.embed_tokens.weight"] = to_host(w.embed);
  m["model.norm.weight"] = to_host(w.final_norm);
  for (size_t i = 0; i < w.layers.size(); ++i) {
    const std::string a =
        "model.layers." + std::to_string(i) + ".self_attn.";
    const std::string p = "model.layers." + std::to_string(i) + ".";
    const auto& l = w.layers[i];
    m[a + "q_proj.weight"] = to_host(l.q_W);
    m[a + "k_proj.weight"] = to_host(l.k_W);
    m[a + "v_proj.weight"] = to_host(l.v_W);
    m[a + "o_proj.weight"] = to_host(l.o_W);
    m[a + "q_norm.weight"] = to_host(l.qn);
    m[a + "k_norm.weight"] = to_host(l.kn);
    m[p + "input_layernorm.weight"] = to_host(l.in_norm);
    m[p + "post_attention_layernorm.weight"] = to_host(l.post_norm);
    m[p + "mlp.gate_proj.weight"] = to_host(l.gate_W);
    m[p + "mlp.up_proj.weight"] = to_host(l.up_W);
    m[p + "mlp.down_proj.weight"] = to_host(l.down_W);
  }
  return m;
}

ViewMap vis_views(const QV::QwenImage21VisionWeights<Exec, float>& w) {
  ViewMap m;
  m["model.visual.patch_embed.proj.weight"] = to_host(w.patch_W);
  m["model.visual.patch_embed.proj.bias"] = to_host(w.patch_b);
  m["model.visual.pos_embed.weight"] = to_host(w.pos_table);
  for (size_t i = 0; i < w.layers.size(); ++i) {
    const std::string a =
        "model.visual.blocks." + std::to_string(i) + ".";
    const auto& l = w.layers[i];
    m[a + "attn.qkv.weight"] = to_host(l.qkv_W);
    m[a + "attn.qkv.bias"] = to_host(l.qkv_b);
    m[a + "attn.proj.weight"] = to_host(l.proj_W);
    m[a + "attn.proj.bias"] = to_host(l.proj_b);
    m[a + "norm1.weight"] = to_host(l.n1_w);
    m[a + "norm1.bias"] = to_host(l.n1_b);
    m[a + "norm2.weight"] = to_host(l.n2_w);
    m[a + "norm2.bias"] = to_host(l.n2_b);
    m[a + "mlp.linear_fc1.weight"] = to_host(l.fc1_W);
    m[a + "mlp.linear_fc1.bias"] = to_host(l.fc1_b);
    m[a + "mlp.linear_fc2.weight"] = to_host(l.fc2_W);
    m[a + "mlp.linear_fc2.bias"] = to_host(l.fc2_b);
  }
  m["model.visual.merger.norm.weight"] = to_host(w.merge_norm_w);
  m["model.visual.merger.norm.bias"] = to_host(w.merge_norm_b);
  m["model.visual.merger.linear_fc1.weight"] = to_host(w.merge_fc1_W);
  m["model.visual.merger.linear_fc1.bias"] = to_host(w.merge_fc1_b);
  m["model.visual.merger.linear_fc2.weight"] = to_host(w.merge_fc2_W);
  m["model.visual.merger.linear_fc2.bias"] = to_host(w.merge_fc2_b);
  for (size_t k = 0; k < w.deep.size(); ++k) {
    const std::string d =
        "model.visual.deepstack_merger_list." + std::to_string(k) + ".";
    const auto& dd = w.deep[k];
    m[d + "norm.weight"] = to_host(dd.norm_w);
    m[d + "norm.bias"] = to_host(dd.norm_b);
    m[d + "linear_fc1.weight"] = to_host(dd.fc1_W);
    m[d + "linear_fc1.bias"] = to_host(dd.fc1_b);
    m[d + "linear_fc2.weight"] = to_host(dd.fc2_W);
    m[d + "linear_fc2.bias"] = to_host(dd.fc2_b);
  }
  return m;
}

ViewMap vae_views(const QVV::QwenImage21VaeWeights<Exec, float>& w) {
  ViewMap m;
  for (const auto& kv : w.blobs) m[kv.first] = to_host(kv.second);
  return m;
}

void expect_maps_equal(const ViewMap& a, const ViewMap& b) {
  ASSERT_EQ(a.size(), b.size());
  for (const auto& kv : a) {
    const auto it = b.find(kv.first);
    ASSERT_TRUE(it != b.end()) << kv.first;
    ASSERT_EQ(kv.second.size(), it->second.size()) << kv.first;
    EXPECT_TRUE(kv.second == it->second) << kv.first;
  }
}

template <class Spec>
std::vector<FabTensor> fab_specs(const std::vector<Spec>& specs, bool bf16) {
  std::vector<FabTensor> out;
  size_t seed = 0;
  for (const auto& s : specs) {
    FabTensor t;
    t.name = s.name;
    const size_t numel =
        s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
    t.shape = s.matrix ? std::vector<int64_t>{s.d0, s.d1}
                       : std::vector<int64_t>{s.d0};
    t.data = payload(numel, seed, bf16);
    t.dtype = bf16 ? "BF16" : "F32";
    seed += 17;
    out.push_back(std::move(t));
  }
  return out;
}

ViewMap fab_map(const std::vector<FabTensor>& tensors) {
  ViewMap m;
  for (const auto& t : tensors) m[t.name] = t.data;
  return m;
}

TEST(QwenImage21HostWeights, DitWarmStageMatchesDiskF32) {
  const QW::Config cfg = QW::Config::tiny_test();
  const auto tensors = fab_specs(QD::qwen21_dit_expected_tensors(cfg), false);
  const std::string path = test_dir("dit_f32") + "/dit.safetensors";
  write_shard(path, tensors);

  QW::HostWeights<float> hw;
  EXPECT_EQ(hw.host_bytes(), 0u);
  ASSERT_NO_THROW(hw.warm_dit(cfg, {path}));
  EXPECT_TRUE(hw.has_dit);
  EXPECT_GT(hw.host_bytes(), 0u);

  // Stage twice: repeat runs stream from the held host copy.
  QD::QwenImage21DitModel<Exec, float> m_staged(cfg), m_staged2(cfg);
  ASSERT_NO_THROW(hw.stage_dit_to(m_staged.weights()));
  ASSERT_NO_THROW(hw.stage_dit_to(m_staged2.weights()));

  QD::QwenImage21DitModel<Exec, float> m_file(cfg);
  ASSERT_NO_THROW(m_file.weights().load_safetensors(path));

  expect_maps_equal(dit_views(m_file.weights()), dit_views(m_staged.weights()));
  expect_maps_equal(dit_views(m_staged.weights()),
                    dit_views(m_staged2.weights()));
  expect_maps_equal(fab_map(tensors), dit_views(m_staged.weights()));
  EXPECT_EQ(m_file.weights().resident_bytes(), hw.host_bytes());
}

TEST(QwenImage21HostWeights, DitWarmStageMatchesDiskBF16) {
  const QW::Config cfg = QW::Config::tiny_test();
  const auto tensors = fab_specs(QD::qwen21_dit_expected_tensors(cfg), true);
  const std::string path = test_dir("dit_bf16") + "/dit.safetensors";
  write_shard(path, tensors);

  QW::HostWeights<float> hw;
  ASSERT_NO_THROW(hw.warm_dit(cfg, {path}));
  QD::QwenImage21DitModel<Exec, float> m_staged(cfg);
  ASSERT_NO_THROW(hw.stage_dit_to(m_staged.weights()));

  QD::QwenImage21DitModel<Exec, float> m_file(cfg);
  ASSERT_NO_THROW(m_file.weights().load_safetensors(path));
  expect_maps_equal(dit_views(m_file.weights()), dit_views(m_staged.weights()));
  expect_maps_equal(fab_map(tensors), dit_views(m_staged.weights()));
}

TEST(QwenImage21HostWeights, EncoderVisionWarmStageMatchesDisk) {
  const QE::Config ecfg = QE::Config::tiny_test();
  const QV::Config vcfg = QV::Config::tiny_test();
  auto tensors = fab_specs(QE::qwen21_encoder_expected_tensors(ecfg), false);
  const auto vis = fab_specs(QV::qwen21_vision_expected_tensors(vcfg), false);
  tensors.insert(tensors.end(), vis.begin(), vis.end());
  tensors.push_back(FabTensor{"lm_head.weight", {4}, payload(4, 999, false)});
  const std::string path = test_dir("encvis") + "/model.safetensors";
  write_shard(path, tensors);

  QW::HostWeights<float> hw;
  ASSERT_NO_THROW(hw.warm_encoder(ecfg, path));
  ASSERT_NO_THROW(hw.warm_vision(vcfg, path));
  EXPECT_TRUE(hw.has_encoder);
  EXPECT_TRUE(hw.has_vision);

  QE::QwenImage21EncoderModel<Exec, float> enc(ecfg);
  ASSERT_NO_THROW(hw.stage_encoder_to(enc.weights()));
  QV::QwenImage21VisionModel<Exec, float> vmodel(vcfg);
  ASSERT_NO_THROW(hw.stage_vision_to(vmodel.weights()));

  QE::QwenImage21EncoderModel<Exec, float> enc_file(ecfg);
  ASSERT_NO_THROW(enc_file.weights().load_safetensors(path));
  expect_maps_equal(enc_views(enc_file.weights()), enc_views(enc.weights()));
  QV::QwenImage21VisionModel<Exec, float> vis_file(vcfg);
  ASSERT_NO_THROW(vis_file.weights().load_safetensors(path));
  expect_maps_equal(vis_views(vis_file.weights()),
                    vis_views(vmodel.weights()));
}

TEST(QwenImage21HostWeights, VaeWarmStageMatchesDisk) {
  const std::vector<FabTensor> tensors = {
      {"enc.conv.weight", {8}, payload(8, 1, false)},
      {"enc.norm.weight", {4}, payload(4, 2, false)},
      {"dec.conv.weight", {2, 3}, payload(6, 3, false)},
      {"dec.norm.bias", {5}, payload(5, 4, false)},
  };
  const std::string path = test_dir("vae") + "/vae.safetensors";
  write_shard(path, tensors);

  QW::HostWeights<float> hw;
  QVV::Config c;
  c.validate();
  ASSERT_NO_THROW(hw.warm_vae(c, {path}));
  EXPECT_TRUE(hw.has_vae);

  QVV::QwenImage21VaeWeights<Exec, float> w_staged;
  {
    QVV::Config c2;
    c2.validate();
    w_staged.allocate(c2);
  }
  ASSERT_NO_THROW(hw.stage_vae_to(w_staged));

  QVV::QwenImage21VaeWeights<Exec, float> w_file;
  {
    QVV::Config c3;
    c3.validate();
    w_file.allocate(c3);
  }
  ASSERT_NO_THROW(w_file.load_safetensors(path));
  expect_maps_equal(vae_views(w_file), vae_views(w_staged));
  expect_maps_equal(fab_map(tensors), vae_views(w_staged));
  EXPECT_EQ(w_file.resident_bytes(), hw.host_bytes());
}

TEST(QwenImage21HostWeights, StageGuardsAndClear) {
  QW::HostWeights<float> hw;
  QD::QwenImage21DitModel<Exec, float> dit(QW::Config::tiny_test());
  EXPECT_THROW(hw.stage_dit_to(dit.weights()), std::runtime_error);
  QE::QwenImage21EncoderModel<Exec, float> enc(QE::Config::tiny_test());
  EXPECT_THROW(hw.stage_encoder_to(enc.weights()), std::runtime_error);
  QV::QwenImage21VisionModel<Exec, float> vis(QV::Config::tiny_test());
  EXPECT_THROW(hw.stage_vision_to(vis.weights()), std::runtime_error);
  QVV::QwenImage21VaeWeights<Exec, float> vae;
  {
    QVV::Config c;
    c.validate();
    vae.allocate(c);
  }
  EXPECT_THROW(hw.stage_vae_to(vae), std::runtime_error);

  const QW::Config cfg = QW::Config::tiny_test();
  const auto tensors = fab_specs(QD::qwen21_dit_expected_tensors(cfg), false);
  const std::string path = test_dir("clear") + "/dit.safetensors";
  write_shard(path, tensors);
  ASSERT_NO_THROW(hw.warm_dit(cfg, {path}));
  EXPECT_GT(hw.host_bytes(), 0u);
  hw.clear();
  EXPECT_FALSE(hw.has_dit);
  EXPECT_EQ(hw.host_bytes(), 0u);
  EXPECT_THROW(hw.stage_dit_to(dit.weights()), std::runtime_error);
}

}  // namespace
