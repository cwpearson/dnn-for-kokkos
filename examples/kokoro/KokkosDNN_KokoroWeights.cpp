// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Kokoro weight loader: torch.save -> fused device views. See header.

#include "KokkosDNN_KokoroWeights.hpp"

#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_KokoroPth.hpp"

namespace KokkosDNN {
namespace Kokoro {
namespace {

// ---------------------------------------------------------------------------
// Archive accessor: name -> validated host float blob (storage cache).
// ---------------------------------------------------------------------------
struct Loader {
  PthArchive arc;
  std::map<std::string, PthTensorMeta> by_name;
  std::map<std::string, std::vector<float>> cache;

  explicit Loader(const std::string& path) : arc(open_pth(path)) {
    for (auto& t : arc.tensors) {
      if (by_name.count(t.name))
        throw std::runtime_error("load: duplicate tensor '" + t.name + "'");
      by_name[t.name] = t;
    }
  }

  const PthTensorMeta& find(const std::string& name,
                           const std::vector<int64_t>& shape) {
    auto it = by_name.find(name);
    if (it == by_name.end())
      throw std::runtime_error("load: tensor not found '" + name + "'");
    const PthTensorMeta& m = it->second;
    if (m.shape != shape)
      throw std::runtime_error("load: shape mismatch for '" + name + "'");
    if ((int64_t)m.stride.size() != (int64_t)shape.size())
      throw std::runtime_error("load: bad strides for '" + name + "'");
    // Require row-major contiguous (true for the released checkpoint).
    int64_t expect = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      if (m.stride[(size_t)i] != expect)
        throw std::runtime_error("load: non-contiguous '" + name + "'");
      expect *= shape[(size_t)i];
    }
    if (m.dtype != "torch.FloatStorage")
      throw std::runtime_error("load: non-float storage for '" + name + "'");
    return m;
  }

  // Raw float blob for one tensor (offset-sliced from its storage).
  std::vector<float> raw(const std::string& name,
                         const std::vector<int64_t>& shape) {
    const PthTensorMeta& m = find(name, shape);
    auto it = cache.find(m.storage_key);
    if (it == cache.end()) {
      std::vector<float> stor = read_pth_storage(arc, m.storage_key);
      if ((int64_t)stor.size() != m.storage_numel)
        throw std::runtime_error("load: storage size mismatch '" +
                                 m.storage_key + "'");
      it = cache.emplace(m.storage_key, std::move(stor)).first;
    }
    int64_t numel = 1;
    for (auto d : shape) numel *= d;
    if (m.storage_offset < 0 || m.storage_offset + numel > (int64_t)it->second.size())
      throw std::runtime_error("load: storage slice OOB '" + name + "'");
    return std::vector<float>(it->second.begin() + (size_t)m.storage_offset,
                              it->second.begin() + (size_t)(m.storage_offset + numel));
  }

  // Fused weight_norm: w[o,i,k] = g[o] * v[o,i,k] / ||v[o]|| (norm over all
  // but axis 0). g shape [C,1,1] (scalar per o), v [Cout,Cin,K].
  std::vector<float> fused(const std::string& base,
                           const std::vector<int64_t>& vshape) {
    if (vshape.size() != 3)
      throw std::runtime_error("load: WN rank != 3 for '" + base + "'");
    const int64_t C = vshape[0];
    std::vector<float> g = raw(base + ".weight_g", {C, 1, 1});
    std::vector<float> v = raw(base + ".weight_v", vshape);
    int64_t rest = vshape[1] * vshape[2];
    std::vector<float> w(v.size());
    for (int64_t o = 0; o < C; ++o) {
      double ss = 0.0;
      for (int64_t j = 0; j < rest; ++j) {
        const double x = v[(size_t)(o * rest + j)];
        ss += x * x;
      }
      const float inv = 1.0f / (float)std::sqrt(ss + 1e-12);
      const float s = g[(size_t)o] * inv;
      for (int64_t j = 0; j < rest; ++j)
        w[(size_t)(o * rest + j)] = v[(size_t)(o * rest + j)] * s;
    }
    return w;
  }
};

// Host mirror fill helpers.
template <class ExecSpace, class Scalar>
void to_device(const std::vector<float>& h,
               const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& v,
               const std::string& what = "") {
  if (h.size() != v.extent(0) * v.extent(1))
    throw std::runtime_error("load: host/device size mismatch for '" + what +
                             "' (" + std::to_string(h.size()) + " vs [" +
                             std::to_string(v.extent(0)) + "," +
                             std::to_string(v.extent(1)) + "])");
  auto hm = Kokkos::create_mirror_view(v);
  size_t k = 0;
  for (size_t i = 0; i < hm.extent(0); ++i)
    for (size_t j = 0; j < hm.extent(1); ++j) hm(i, j) = (Scalar)h[k++];
  Kokkos::deep_copy(v, hm);
}
template <class ExecSpace, class Scalar>
void to_device(
    const std::vector<float>& h,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& v,
    const std::string& what = "") {
  if (h.size() != v.extent(0) * v.extent(1) * v.extent(2))
    throw std::runtime_error("load: host/device size mismatch for '" + what +
                             "' (" + std::to_string(h.size()) + " vs [" +
                             std::to_string(v.extent(0)) + "," +
                             std::to_string(v.extent(1)) + "," +
                             std::to_string(v.extent(2)) + "])");
  auto hm = Kokkos::create_mirror_view(v);
  size_t k = 0;
  for (size_t i = 0; i < hm.extent(0); ++i)
    for (size_t j = 0; j < hm.extent(1); ++j)
      for (size_t l = 0; l < hm.extent(2); ++l) hm(i, j, l) = (Scalar)h[k++];
  Kokkos::deep_copy(v, hm);
}
template <class ExecSpace, class Scalar>
void to_device(const std::vector<float>& h,
               const Kokkos::View<Scalar*, ExecSpace>& v,
               const std::string& what = "") {
  if (h.size() != v.extent(0))
    throw std::runtime_error("load: host/device size mismatch for '" + what +
                             "' (" + std::to_string(h.size()) + " vs [" +
                             std::to_string(v.extent(0)) + "])");
  auto hm = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < hm.extent(0); ++i) hm(i) = (Scalar)h[i];
  Kokkos::deep_copy(v, hm);
}

// LSTM dir-pair: {w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0} + {_reverse}.
template <class ExecSpace, class Scalar>
void load_lstm(Loader& L, const std::string& base, size_t D, size_t H,
               LstmW<ExecSpace, Scalar>& l) {
  const int64_t Hh = (int64_t)H, Dd = (int64_t)D;
  auto cat2 = [&](const std::string& a, const std::string& b,
                  const std::vector<int64_t>& s) {
    std::vector<float> x = L.raw(a, s), y = L.raw(b, s);
    x.insert(x.end(), y.begin(), y.end());
    return x;
  };
  std::vector<float> wih =
      cat2(base + ".weight_ih_l0", base + ".weight_ih_l0_reverse", {4 * Hh, Dd});
  std::vector<float> whh =
      cat2(base + ".weight_hh_l0", base + ".weight_hh_l0_reverse", {4 * Hh, Hh});
  std::vector<float> bih =
      cat2(base + ".bias_ih_l0", base + ".bias_ih_l0_reverse", {4 * Hh});
  std::vector<float> bhh =
      cat2(base + ".bias_hh_l0", base + ".bias_hh_l0_reverse", {4 * Hh});
  // Repack [2,4H,D] dir-major.
  std::vector<float> wih2(2 * 4 * H * D), whh2(2 * 4 * H * H);
  for (int d = 0; d < 2; ++d)
    for (int64_t i = 0; i < 4 * Hh * Dd; ++i)
      wih2[(size_t)(d * 4 * Hh * Dd + i)] =
          wih[(size_t)(d * 4 * Hh * Dd + i)];
  for (int d = 0; d < 2; ++d)
    for (int64_t i = 0; i < 4 * Hh * Hh; ++i)
      whh2[(size_t)(d * 4 * Hh * Hh + i)] =
          whh[(size_t)(d * 4 * Hh * Hh + i)];
  Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>
      mh_wih(wih2.data(), 2, 4 * H, D);
  Kokkos::deep_copy(l.w_ih, mh_wih);
  Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>
      mh_whh(whh2.data(), 2, 4 * H, H);
  Kokkos::deep_copy(l.w_hh, mh_whh);
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>
      mh_bih(bih.data(), 2, 4 * H);
  Kokkos::deep_copy(l.b_ih, mh_bih);
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>
      mh_bhh(bhh.data(), 2, 4 * H);
  Kokkos::deep_copy(l.b_hh, mh_bhh);
}

template <class ExecSpace, class Scalar>
void load_adain(Loader& L, const std::string& base, int din, int dout, bool up,
                size_t SD, AdainBlkW<ExecSpace, Scalar>& b) {
  auto sh3 = [&](int64_t a, int64_t c, int64_t k) {
    return std::vector<int64_t>{a, c, k};
  };
  to_device(L.fused(base + ".conv1", sh3(dout, din, 3)), b.conv1.w);
  to_device(L.raw(base + ".conv1.bias", {(int64_t)dout}), b.conv1.b);
  to_device(L.fused(base + ".conv2", sh3(dout, dout, 3)), b.conv2.w);
  to_device(L.raw(base + ".conv2.bias", {(int64_t)dout}), b.conv2.b);
  if (din != dout)
    to_device(L.fused(base + ".conv1x1", sh3(dout, din, 1)), b.conv1x1.w);
  if (up) {
    to_device(L.fused(base + ".pool", sh3(din, 1, 3)), b.pool.w);
    to_device(L.raw(base + ".pool.bias", {(int64_t)din}), b.pool.b);
  }
  to_device(L.raw(base + ".norm1.fc.weight", {(int64_t)2 * din, (int64_t)SD}),
            b.n1w);
  to_device(L.raw(base + ".norm1.fc.bias", {(int64_t)2 * din}), b.n1b);
  to_device(L.raw(base + ".norm2.fc.weight", {(int64_t)2 * dout, (int64_t)SD}),
            b.n2w);
  to_device(L.raw(base + ".norm2.fc.bias", {(int64_t)2 * dout}), b.n2b);
}

template <class ExecSpace, class Scalar>
void load_resblk(Loader& L, const std::string& base, int ch,
                 ResBlkW<ExecSpace, Scalar>& r) {
  for (int j = 0; j < 3; ++j) {
    const std::string s = std::to_string(j);
    auto sh3 = [&](int64_t k) {
      return std::vector<int64_t>{(int64_t)ch, (int64_t)ch, k};
    };
    const int64_t k = r.k;  // set by allocate()
    to_device(L.fused(base + ".convs1." + s, sh3(k)), r.c1[(size_t)j].w);
    to_device(L.raw(base + ".convs1." + s + ".bias", {(int64_t)ch}),
              r.c1[(size_t)j].b);
    to_device(L.fused(base + ".convs2." + s, sh3(k)), r.c2[(size_t)j].w);
    to_device(L.raw(base + ".convs2." + s + ".bias", {(int64_t)ch}),
              r.c2[(size_t)j].b);
    to_device(L.raw(base + ".adain1." + s + ".fc.weight",
                    {(int64_t)2 * ch, 128}),
              r.a1w[(size_t)j]);
    to_device(L.raw(base + ".adain1." + s + ".fc.bias", {(int64_t)2 * ch}),
              r.a1b[(size_t)j]);
    to_device(L.raw(base + ".adain2." + s + ".fc.weight",
                    {(int64_t)2 * ch, 128}),
              r.a2w[(size_t)j]);
    to_device(L.raw(base + ".adain2." + s + ".fc.bias", {(int64_t)2 * ch}),
              r.a2b[(size_t)j]);
    // Alpha [1,ch,1] -> [ch].
    std::vector<float> a1 = L.raw(base + ".alpha1." + s, {1, (int64_t)ch, 1});
    std::vector<float> a2 = L.raw(base + ".alpha2." + s, {1, (int64_t)ch, 1});
    to_device(a1, r.al1[(size_t)j]);
    to_device(a2, r.al2[(size_t)j]);
  }
}

}  // namespace

template <class ExecSpace, class Scalar>
void load_pth_weights(const Config& cfg, const std::string& pth_path,
                      Weights<ExecSpace, Scalar>& w) {
  cfg.validate();
  w.allocate(cfg);
  Loader L(pth_path);
  const size_t HD = (size_t)cfg.hidden_dim, SD = (size_t)cfg.style_dim;
  const size_t PH = (size_t)cfg.plbert_hidden, PE = (size_t)cfg.plbert_embed;
  const size_t PI = (size_t)cfg.plbert_intermediate;
  // --- Albert ---
  to_device(L.raw("bert.module.embeddings.word_embeddings.weight",
                  {(int64_t)cfg.n_token, (int64_t)PE}),
              w.bert.word);
  to_device(L.raw("bert.module.embeddings.position_embeddings.weight",
                  {(int64_t)cfg.max_position, (int64_t)PE}),
              w.bert.pos);
  to_device(L.raw("bert.module.embeddings.token_type_embeddings.weight",
                  {2, (int64_t)PE}),
              w.bert.tokentype);
  to_device(L.raw("bert.module.embeddings.LayerNorm.weight", {(int64_t)PE}),
              w.bert.ln_w);
  to_device(L.raw("bert.module.embeddings.LayerNorm.bias", {(int64_t)PE}),
              w.bert.ln_b);
  to_device(L.raw("bert.module.encoder.embedding_hidden_mapping_in.weight",
                  {(int64_t)PH, (int64_t)PE}),
              w.bert.map_w);
  to_device(L.raw("bert.module.encoder.embedding_hidden_mapping_in.bias",
                  {(int64_t)PH}),
              w.bert.map_b);
  const std::string AL =
      "bert.module.encoder.albert_layer_groups.0.albert_layers.0.";
  to_device(L.raw(AL + "attention.query.weight", {(int64_t)PH, (int64_t)PH}),
              w.bert.q_w);
  to_device(L.raw(AL + "attention.query.bias", {(int64_t)PH}), w.bert.q_b);
  to_device(L.raw(AL + "attention.key.weight", {(int64_t)PH, (int64_t)PH}),
              w.bert.k_w);
  to_device(L.raw(AL + "attention.key.bias", {(int64_t)PH}), w.bert.k_b);
  to_device(L.raw(AL + "attention.value.weight", {(int64_t)PH, (int64_t)PH}),
              w.bert.v_w);
  to_device(L.raw(AL + "attention.value.bias", {(int64_t)PH}), w.bert.v_b);
  to_device(L.raw(AL + "attention.dense.weight", {(int64_t)PH, (int64_t)PH}),
              w.bert.dense_w);
  to_device(L.raw(AL + "attention.dense.bias", {(int64_t)PH}), w.bert.dense_b);
  to_device(L.raw(AL + "attention.LayerNorm.weight", {(int64_t)PH}),
              w.bert.attn_ln_w);
  to_device(L.raw(AL + "attention.LayerNorm.bias", {(int64_t)PH}),
              w.bert.attn_ln_b);
  to_device(L.raw(AL + "ffn.weight", {(int64_t)PI, (int64_t)PH}), w.bert.ffn_w);
  to_device(L.raw(AL + "ffn.bias", {(int64_t)PI}), w.bert.ffn_b);
  to_device(L.raw(AL + "ffn_output.weight", {(int64_t)PH, (int64_t)PI}),
              w.bert.ffn_out_w);
  to_device(L.raw(AL + "ffn_output.bias", {(int64_t)PH}), w.bert.ffn_out_b);
  to_device(L.raw(AL + "full_layer_layer_norm.weight", {(int64_t)PH}),
              w.bert.full_ln_w);
  to_device(L.raw(AL + "full_layer_layer_norm.bias", {(int64_t)PH}),
              w.bert.full_ln_b);
  to_device(L.raw("bert_encoder.module.weight", {(int64_t)HD, (int64_t)PH}),
              w.enc_w);
  to_device(L.raw("bert_encoder.module.bias", {(int64_t)HD}), w.enc_b);
  // --- DurationEncoder: lstms interleave [LSTM0, AdaLN1, LSTM2, ...].
  const size_t DH = HD / 2, DD = HD + SD;
  for (int i = 0; i < cfg.n_layer; ++i) {
    load_lstm(L, "predictor.module.text_encoder.lstms." + std::to_string(2 * i),
              DD, DH, w.dur_lstm[(size_t)i]);
    const std::string FB = "predictor.module.text_encoder.lstms." +
                           std::to_string(2 * i + 1) + ".fc";
    to_device(L.raw(FB + ".weight", {(int64_t)2 * (int64_t)HD, (int64_t)SD}),
              w.dur_fcw[(size_t)i]);
    to_device(L.raw(FB + ".bias", {(int64_t)2 * (int64_t)HD}),
              w.dur_fcb[(size_t)i]);
  }
  // --- Predictor head ---
  load_lstm(L, "predictor.module.lstm", DD, DH, w.pred_lstm);
  to_device(L.raw("predictor.module.duration_proj.linear_layer.weight",
                  {(int64_t)cfg.max_dur, (int64_t)HD}),
            w.dur_proj_w);
  to_device(L.raw("predictor.module.duration_proj.linear_layer.bias",
                  {(int64_t)cfg.max_dur}),
            w.dur_proj_b);
  load_lstm(L, "predictor.module.shared", DD, DH, w.pred_shared);
  load_adain(L, "predictor.module.F0.0", (int)HD, (int)HD, false, SD, w.f0[0]);
  load_adain(L, "predictor.module.F0.1", (int)HD, (int)(HD / 2), true, SD,
             w.f0[1]);
  load_adain(L, "predictor.module.F0.2", (int)(HD / 2), (int)(HD / 2), false,
             SD, w.f0[2]);
  load_adain(L, "predictor.module.N.0", (int)HD, (int)HD, false, SD,
             w.nstack[0]);
  load_adain(L, "predictor.module.N.1", (int)HD, (int)(HD / 2), true, SD,
             w.nstack[1]);
  load_adain(L, "predictor.module.N.2", (int)(HD / 2), (int)(HD / 2), false,
             SD, w.nstack[2]);
  to_device(L.raw("predictor.module.F0_proj.weight",
                  {1, (int64_t)(HD / 2), 1}),
            w.f0_proj.w);
  to_device(L.raw("predictor.module.F0_proj.bias", {1}), w.f0_proj.b);
  to_device(L.raw("predictor.module.N_proj.weight", {1, (int64_t)(HD / 2), 1}),
            w.n_proj.w);
  to_device(L.raw("predictor.module.N_proj.bias", {1}), w.n_proj.b);
  // --- TextEncoder ---
  to_device(L.raw("text_encoder.module.embedding.weight",
                  {(int64_t)cfg.n_token, (int64_t)HD}),
            w.text_embed);
  for (int i = 0; i < 3; ++i) {
    const std::string B = "text_encoder.module.cnn." + std::to_string(i);
    to_device(L.fused(B + ".0", {(int64_t)HD, (int64_t)HD, (int64_t)cfg.text_kernel}),
              w.text_cnn[(size_t)i].conv.w);
    to_device(L.raw(B + ".0.bias", {(int64_t)HD}), w.text_cnn[(size_t)i].conv.b);
    to_device(L.raw(B + ".1.gamma", {(int64_t)HD}), w.text_cnn[(size_t)i].gamma);
    to_device(L.raw(B + ".1.beta", {(int64_t)HD}), w.text_cnn[(size_t)i].beta);
  }
  load_lstm(L, "text_encoder.module.lstm", HD, DH, w.text_lstm);
  // --- Decoder ---
  to_device(L.fused("decoder.module.F0_conv", {1, 1, 3}), w.f0_conv.w);
  to_device(L.raw("decoder.module.F0_conv.bias", {1}), w.f0_conv.b);
  to_device(L.fused("decoder.module.N_conv", {1, 1, 3}), w.n_conv.w);
  to_device(L.raw("decoder.module.N_conv.bias", {1}), w.n_conv.b);
  load_adain(L, "decoder.module.encode", (int)HD + 2,
             (int)cfg.decode_channels[0], false, SD, w.enc_blk);
  for (int i = 0; i < 4; ++i) {
    load_adain(L, "decoder.module.decode." + std::to_string(i),
               (int)cfg.decode_channels[0] + 64 + 2,
               (int)cfg.decode_channels[i],
               cfg.decode_upsample_last && i == 3, SD, w.dec_blk[(size_t)i]);
  }
  to_device(L.fused("decoder.module.asr_res.0", {64, (int64_t)HD, 1}),
            w.asr_res.w);
  to_device(L.raw("decoder.module.asr_res.0.bias", {64}), w.asr_res.b);
  const int UC = cfg.upsample_initial_channel;
  to_device(L.fused("decoder.module.generator.ups.0",
                    {(int64_t)UC, (int64_t)(UC / 2), 20}),
            w.gen.ups[0].w);
  to_device(L.raw("decoder.module.generator.ups.0.bias", {(int64_t)(UC / 2)}),
            w.gen.ups[0].b);
  to_device(L.fused("decoder.module.generator.ups.1",
                    {(int64_t)(UC / 2), (int64_t)(UC / 4), 12}),
            w.gen.ups[1].w);
  to_device(L.raw("decoder.module.generator.ups.1.bias", {(int64_t)(UC / 4)}),
            w.gen.ups[1].b);
  for (int i = 0; i < 6; ++i)
    load_resblk(L, "decoder.module.generator.resblocks." + std::to_string(i),
                UC / (i < 3 ? 2 : 4), w.gen.rb[(size_t)i]);
  load_resblk(L, "decoder.module.generator.noise_res.0", UC / 2, w.gen.nres[0]);
  load_resblk(L, "decoder.module.generator.noise_res.1", UC / 4, w.gen.nres[1]);
  to_device(L.raw("decoder.module.generator.noise_convs.0.weight",
                  {(int64_t)(UC / 2), (int64_t)(cfg.gen_n_fft + 2), 12}),
            w.gen.nc0.w);
  to_device(L.raw("decoder.module.generator.noise_convs.0.bias",
                  {(int64_t)(UC / 2)}),
            w.gen.nc0.b);
  to_device(L.raw("decoder.module.generator.noise_convs.1.weight",
                  {(int64_t)(UC / 4), (int64_t)(cfg.gen_n_fft + 2), 1}),
            w.gen.nc1.w);
  to_device(L.raw("decoder.module.generator.noise_convs.1.bias",
                  {(int64_t)(UC / 4)}),
            w.gen.nc1.b);
  to_device(L.fused("decoder.module.generator.conv_post",
                    {(int64_t)(cfg.gen_n_fft + 2), (int64_t)(UC / 4), 7}),
            w.gen.post.w);
  to_device(L.raw("decoder.module.generator.conv_post.bias",
                  {(int64_t)(cfg.gen_n_fft + 2)}),
            w.gen.post.b);
  to_device(L.raw("decoder.module.generator.m_source.l_linear.weight", {1, 9}),
            w.gen.lin_w);
  to_device(L.raw("decoder.module.generator.m_source.l_linear.bias", {1}),
            w.gen.lin_b);
}

VoicePack load_voice(const std::string& voice_path) {
  Loader L(voice_path);
  if (L.arc.tensors.size() != 1)
    throw std::runtime_error("load_voice: expected a single tensor in '" +
                             voice_path + "'");
  const PthTensorMeta& m = L.arc.tensors[0];
  if (m.shape.size() != 3 || m.shape[1] != 1 || m.shape[2] != 256)
    throw std::runtime_error("load_voice: expected [rows,1,256] in '" +
                             voice_path + "'");
  VoicePack v;
  v.n_rows = (int)m.shape[0];
  v.rows = L.raw(m.name.empty() ? "" : m.name,
                 {(int64_t)v.n_rows, 1, 256});
  // Squeeze dim 1: rows are contiguous [n_rows, 256] already.
  return v;
}

// ---------------------------------------------------------------------------
// random_init: small uniform fills (smoke tests only).
// ---------------------------------------------------------------------------
template <class ExecSpace, class Scalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed) {
  cfg.validate();
  w.allocate(cfg);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> u(-0.05f, 0.05f);
  auto fill = [&](auto& v) {
    auto hm = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < hm.size(); ++i) hm.data()[i] =
        static_cast<typename std::decay_t<decltype(hm)>::value_type>(u(rng));
    Kokkos::deep_copy(v, hm);
  };
  auto fill_one = [&](auto& v) {
    Kokkos::deep_copy(v, (typename std::decay_t<decltype(v)>::value_type)1);
  };
  auto fill_zero = [&](auto& v) {
    Kokkos::deep_copy(v, (typename std::decay_t<decltype(v)>::value_type)0);
  };
  fill(w.bert.word);
  fill(w.bert.pos);
  fill(w.bert.tokentype);
  fill_one(w.bert.ln_w);
  fill_zero(w.bert.ln_b);
  fill(w.bert.map_w);
  fill_zero(w.bert.map_b);
  fill(w.bert.q_w);
  fill_zero(w.bert.q_b);
  fill(w.bert.k_w);
  fill_zero(w.bert.k_b);
  fill(w.bert.v_w);
  fill_zero(w.bert.v_b);
  fill(w.bert.dense_w);
  fill_zero(w.bert.dense_b);
  fill_one(w.bert.attn_ln_w);
  fill_zero(w.bert.attn_ln_b);
  fill(w.bert.ffn_w);
  fill_zero(w.bert.ffn_b);
  fill(w.bert.ffn_out_w);
  fill_zero(w.bert.ffn_out_b);
  fill_one(w.bert.full_ln_w);
  fill_zero(w.bert.full_ln_b);
  fill(w.enc_w);
  fill_zero(w.enc_b);
  for (size_t i = 0; i < w.dur_lstm.size(); ++i) {
    fill(w.dur_lstm[i].w_ih);
    fill(w.dur_lstm[i].w_hh);
    fill_zero(w.dur_lstm[i].b_ih);
    fill_zero(w.dur_lstm[i].b_hh);
    fill_zero(w.dur_fcw[i]);
    fill_zero(w.dur_fcb[i]);
  }
  fill(w.pred_lstm.w_ih);
  fill(w.pred_lstm.w_hh);
  fill_zero(w.pred_lstm.b_ih);
  fill_zero(w.pred_lstm.b_hh);
  fill(w.pred_shared.w_ih);
  fill(w.pred_shared.w_hh);
  fill_zero(w.pred_shared.b_ih);
  fill_zero(w.pred_shared.b_hh);
  fill(w.dur_proj_w);
  fill_zero(w.dur_proj_b);
  auto fill_adain = [&](auto& b) {
    fill(b.conv1.w);
    fill_zero(b.conv1.b);
    fill(b.conv2.w);
    fill_zero(b.conv2.b);
    if (b.conv1x1.w.extent(0) != 0) fill(b.conv1x1.w);
    if (b.pool.w.extent(0) != 0) {
      fill(b.pool.w);
      fill_zero(b.pool.b);
    }
    fill_zero(b.n1w);
    fill_zero(b.n1b);
    fill_zero(b.n2w);
    fill_zero(b.n2b);
  };
  for (auto& b : w.f0) fill_adain(b);
  for (auto& b : w.nstack) fill_adain(b);
  fill(w.f0_proj.w);
  fill_zero(w.f0_proj.b);
  fill(w.n_proj.w);
  fill_zero(w.n_proj.b);
  fill(w.text_embed);
  for (auto& tc : w.text_cnn) {
    fill(tc.conv.w);
    fill_zero(tc.conv.b);
    fill_one(tc.gamma);
    fill_zero(tc.beta);
  }
  fill(w.text_lstm.w_ih);
  fill(w.text_lstm.w_hh);
  fill_zero(w.text_lstm.b_ih);
  fill_zero(w.text_lstm.b_hh);
  fill(w.f0_conv.w);
  fill_zero(w.f0_conv.b);
  fill(w.n_conv.w);
  fill_zero(w.n_conv.b);
  fill(w.asr_res.w);
  fill_zero(w.asr_res.b);
  fill_adain(w.enc_blk);
  for (auto& b : w.dec_blk) fill_adain(b);
  auto fill_res = [&](auto& r) {
    for (int j = 0; j < 3; ++j) {
      fill(r.c1[(size_t)j].w);
      fill_zero(r.c1[(size_t)j].b);
      fill(r.c2[(size_t)j].w);
      fill_zero(r.c2[(size_t)j].b);
      fill_zero(r.a1w[(size_t)j]);
      fill_zero(r.a1b[(size_t)j]);
      fill_zero(r.a2w[(size_t)j]);
      fill_zero(r.a2b[(size_t)j]);
      fill_one(r.al1[(size_t)j]);
      fill_one(r.al2[(size_t)j]);
    }
  };
  for (auto& r : w.gen.rb) fill_res(r);
  for (auto& r : w.gen.nres) fill_res(r);
  for (auto& up : w.gen.ups) {
    fill(up.w);
    fill_zero(up.b);
  }
  fill(w.gen.nc0.w);
  fill_zero(w.gen.nc0.b);
  fill(w.gen.nc1.w);
  fill_zero(w.gen.nc1.b);
  fill(w.gen.post.w);
  fill_zero(w.gen.post.b);
  fill(w.gen.lin_w);
  fill_zero(w.gen.lin_b);
}

// Explicit instantiations for the default execution space (keeps device
// code in the example library, mirroring the Sana example layout).
template void load_pth_weights<Kokkos::DefaultExecutionSpace, float>(
    const Config&, const std::string&, Weights<Kokkos::DefaultExecutionSpace, float>&);
template void random_init<Kokkos::DefaultExecutionSpace, float>(
    const Config&, Weights<Kokkos::DefaultExecutionSpace, float>&, uint64_t);

}  // namespace Kokoro
}  // namespace KokkosDNN
