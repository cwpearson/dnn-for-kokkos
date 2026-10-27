// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 VAE tests: config validation, expected encoder/decoder
// tensor inventory, and a tiny-config encode->decode roundtrip (shapes,
// finiteness, determinism) on random weights.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cmath>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_QwenImage21Vae.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
namespace QVV = KokkosDNN::QwenImage21Vae;

TEST(QwenImage21Vae, ConfigValidate) {
  QVV::Config c;
  EXPECT_NO_THROW(c.validate());
  // The default config carries no affine: latents_mean/std arrive at
  // runtime from vae/config.json (qwen21_load_vae_config_json).
  EXPECT_FALSE(c.has_affine());
  EXPECT_EQ(c.num_down_blocks(), 5);
  EXPECT_EQ(c.resnets_per_down_block(), 2);
  int ein = 0, eout = 0;
  c.enc_block_io(0, ein, eout);
  EXPECT_EQ(ein, 96);
  EXPECT_EQ(eout, 96);
  c.enc_block_io(1, ein, eout);
  EXPECT_EQ(ein, 96);
  EXPECT_EQ(eout, 192);
  c.enc_block_io(4, ein, eout);
  EXPECT_EQ(ein, 768);
  EXPECT_EQ(eout, 768);
  EXPECT_TRUE(c.enc_downsamples(0));
  EXPECT_FALSE(c.enc_downsamples(4));
  EXPECT_EQ(c.enc_temporal_down(0), 0);
  EXPECT_EQ(c.enc_temporal_down(1), 1);
  EXPECT_EQ(c.enc_temporal_down(4), 0);
  QVV::Config t = QVV::Config::tiny_test();
  EXPECT_NO_THROW(t.validate());
  EXPECT_TRUE(t.has_affine());
  EXPECT_EQ(t.num_down_blocks(), 2);
  t.enc_block_io(1, ein, eout);
  EXPECT_EQ(ein, 4);
  EXPECT_EQ(eout, 8);
  QVV::Config bad = c;
  bad.temporal_downsample.resize(2);
  EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(QwenImage21Vae, ExpectedTensors) {
  QVV::Config c;
  auto dec = QVV::qwen21vae_expected_decoder_tensors(c);
  // 134 decoder + 104 encoder = 238, the full single-shard checkpoint.
  EXPECT_EQ(dec.size(), 134u);
  auto enc = QVV::qwen21vae_expected_encoder_tensors(c);
  EXPECT_EQ(enc.size(), 104u);
  // Spot-check first/last encoder keys against the checkpoint layout.
  EXPECT_EQ(enc.front().name, "encoder.conv_in.weight");
  EXPECT_EQ(enc.front().shape, (std::vector<size_t>{96, 4, 3, 3}));
  EXPECT_EQ(enc.back().name, "quant_conv.bias");
  EXPECT_EQ(enc.back().shape, (std::vector<size_t>{128}));
  // Every temporally-downsampled stage carries a 1x1 time_conv pair.
  size_t time_keys = 0;
  for (const auto& s : enc)
    if (s.name.find("time_conv") != std::string::npos) ++time_keys;
  EXPECT_EQ(time_keys, 6u);  // 3 stages x (weight + bias)
  QVV::Config t = QVV::Config::tiny_test();
  EXPECT_EQ(QVV::qwen21vae_expected_decoder_tensors(t).size(), 64u);
  EXPECT_EQ(QVV::qwen21vae_expected_encoder_tensors(t).size(), 52u);
}

TEST(QwenImage21Vae, TinyEncodeDecode) {
  using Model = QVV::QwenImage21VaeModel<Exec, float>;
  // Encode-capable tiny config: torch's AvgDown3D requires
  // in*factor % out == 0 per block, so (like the real checkpoint, whose
  // dim_mult tail is [8,8]) the last down block keeps in == out.
  // Config::tiny_test() (ch_mult {1,2}) violates that on its last block
  // and stays decode-only; {2,1} mirrors the real tail shape instead.
  QVV::Config t;
  t.in_channels = 2;
  t.out_channels = 2;
  t.z_dim = 4;
  t.base_dim = 4;
  t.dec_base_dim = 4;
  t.ch_mult = {2, 1};
  t.compression = 2;
  t.num_res_blocks = 2;
  t.temporal_downsample = {0};
  t.latents_mean = std::vector<float>(4, 0.0f);
  t.latents_std = std::vector<float>(4, 1.0f);
  EXPECT_NO_THROW(t.validate());
  Model m(t);
  m.weights().allocate_expected(t);
  m.weights().allocate_expected_encoder(t);
  EXPECT_EQ(m.weights().num_tensors(), 64u + 54u);
  m.weights().randomize(42);
  KokkosDNN::DNNHandle<Exec> handle;
  // Deterministic 4x4 RGBA-ish pattern (in_channels=2 in tiny_test).
  typename Model::V4 img("qwen21vae_test_img", 1, 2, 4, 4);
  {
    auto h = Kokkos::create_mirror_view(img);
    for (size_t c = 0; c < 2; ++c)
      for (size_t y = 0; y < 4; ++y)
        for (size_t x = 0; x < 4; ++x)
          h(0, c, y, x) =
              (float)((int)(c * 16 + y * 4 + x) - 8) / 16.0f;
    Kokkos::deep_copy(img, h);
  }
  typename Model::V4 lat = m.encode(handle, img);
  EXPECT_EQ(lat.extent(0), 1u);
  EXPECT_EQ(lat.extent(1), 4u);  // z_dim
  EXPECT_EQ(lat.extent(2), 2u);  // 4 / compression(2)
  EXPECT_EQ(lat.extent(3), 2u);
  auto check_finite = [](const typename Model::V4& v) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t c = 0; c < h.extent(1); ++c)
        for (size_t y = 0; y < h.extent(2); ++y)
          for (size_t x = 0; x < h.extent(3); ++x)
            EXPECT_TRUE(std::isfinite(h(i, c, y, x)));
  };
  Kokkos::fence();
  check_finite(lat);
  // Determinism: a second encode matches exactly.
  typename Model::V4 lat2 = m.encode(handle, img);
  Kokkos::fence();
  {
    auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lat);
    auto h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lat2);
    for (size_t c = 0; c < 4; ++c)
      for (size_t y = 0; y < 2; ++y)
        for (size_t x = 0; x < 2; ++x)
          EXPECT_FLOAT_EQ(h1(0, c, y, x), h2(0, c, y, x));
  }
  // Decode the encoded latents back to pixels (shape + finiteness only;
  // the affine roundtrip is a torch-parity concern, not a unit one).
  typename Model::V4 rgba("qwen21vae_test_rgba", 1, 2, 4, 4);
  EXPECT_NO_THROW(m.decode(handle, lat, rgba));
  Kokkos::fence();
  check_finite(rgba);
}

}  // namespace
