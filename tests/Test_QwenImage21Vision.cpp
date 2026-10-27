// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen-Image-2.1 vision tower tests: config validation, expected-tensor
// inventory, pos-interp/position helper exactness, and a tiny-config full
// forward (shapes, finiteness, determinism) on random weights.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cmath>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_QwenImage21Encoder.hpp"
#include "KokkosDNN_QwenImage21Vision.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using bhalf = Kokkos::Experimental::bhalf_t;
namespace QV = KokkosDNN::QwenImage21Vision;
namespace QE = KokkosDNN::QwenImage21Encoder;

TEST(QwenImage21Vision, ConfigValidate) {
  QV::Config c;
  EXPECT_NO_THROW(c.validate());
  EXPECT_EQ(c.patch_dim(), 3 * 2 * 16 * 16);
  EXPECT_EQ(c.num_grid_per_side(), 48);
  QV::Config t = QV::Config::tiny_test();
  EXPECT_NO_THROW(t.validate());
  EXPECT_EQ(t.patch_dim(), 3 * 1 * 2 * 2);
  EXPECT_EQ(t.num_grid_per_side(), 4);
  QV::Config bad = c;
  bad.depth = 0;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = c;
  bad.num_position_embeddings = 101;  // not a square
  EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(QwenImage21Vision, ExpectedTensors) {
  QV::Config c;
  auto ts = QV::qwen21_vision_expected_tensors(c);
  // 27 blocks x 12 + patch(2) + pos(1) + merger(6) + 3 deepstack x 6.
  EXPECT_EQ(ts.size(), 27u * 12 + 2 + 1 + 6 + 3u * 6);
  QV::Config t = QV::Config::tiny_test();
  auto tt = QV::qwen21_vision_expected_tensors(t);
  EXPECT_EQ(tt.size(), 3u * 12 + 2 + 1 + 6 + 3u * 6);
  // Spot-check the distinctive shapes.
  bool found_patch = false, found_merge = false, found_deep = false;
  for (const auto& s : ts) {
    if (s.name == "model.visual.patch_embed.proj.weight") {
      EXPECT_EQ(s.d0, 1152);
      EXPECT_EQ(s.d1, 3 * 2 * 16 * 16);
      found_patch = true;
    }
    if (s.name == "model.visual.merger.linear_fc1.weight") {
      EXPECT_EQ(s.d0, 4608);
      EXPECT_EQ(s.d1, 4608);
      found_merge = true;
    }
    if (s.name == "model.visual.deepstack_merger_list.2.norm.bias") {
      EXPECT_EQ(s.d0, 4608);
      found_deep = true;
    }
  }
  EXPECT_TRUE(found_patch && found_merge && found_deep);
}

TEST(QwenImage21Vision, InterpPosHelpers) {
  // 4x4 grid, side 4, merge 2: weights must sum to 1 per patch, indices in
  // range, rows in block-major order.
  std::vector<int> idx;
  std::vector<float> wt;
  QV::Impl::qwen21vit_interp_pos_host(4, 4, 4, 2, idx, wt);
  ASSERT_EQ(idx.size(), 16u * 4);
  ASSERT_EQ(wt.size(), 16u * 4);
  for (size_t p = 0; p < 16; ++p) {
    double s = 0;
    for (int k = 0; k < 4; ++k) {
      EXPECT_GE(idx[p * 4 + k], 0);
      EXPECT_LT(idx[p * 4 + k], 16);
      s += wt[p * 4 + k];
    }
    EXPECT_NEAR(s, 1.0, 1e-6);
  }
  // align_corners: corner patch (0,0) hits table corner 0 exactly.
  EXPECT_EQ(idx[0], 0);
  EXPECT_NEAR(wt[0], 1.0, 1e-6);
  // Block-major positions: first merge block covers rows/cols {0,1}.
  const std::vector<int> pos =
      QV::Impl::qwen21vit_block_positions_host(4, 4, 2);
  ASSERT_EQ(pos.size(), 16u * 2);
  EXPECT_EQ(pos[0], 0);
  EXPECT_EQ(pos[1], 0);  // patch (0,0)
  EXPECT_EQ(pos[2], 0);
  EXPECT_EQ(pos[3], 1);  // patch (0,1)
  EXPECT_EQ(pos[4], 1);
  EXPECT_EQ(pos[5], 0);  // patch (1,0)
}

TEST(QwenImage21Vision, TinyForward) {
  QV::Config cfg = QV::Config::tiny_test();
  // 4x4 patches -> 16 rows, Nv = 4.
  constexpr int Hp = 4, Wp = 4, Np = 16, Nv = 4;
  QV::QwenImage21VisionModel<Exec, bhalf> model(cfg);
  model.weights().allocate_expected(cfg);
  model.weights().randomize(1234);
  KokkosDNN::DNNHandle<Exec> handle;
  Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec> pix("pix", Np,
                                                       cfg.patch_dim());
  {
    auto h = Kokkos::create_mirror_view(pix);
    for (int p = 0; p < Np; ++p)
      for (int d = 0; d < cfg.patch_dim(); ++d)
        h(p, (size_t)d) = bhalf(0.01f * (float)((p + d) % 7) - 0.03f);
    Kokkos::deep_copy(pix, h);
  }
  Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec> merged("merged", Nv,
                                                          cfg.out_hidden_size);
  std::vector<Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec>> deep;
  for (int k = 0; k < 3; ++k)
    deep.emplace_back("deep", Nv, cfg.out_hidden_size);
  model.forward(handle, pix, Hp, Wp, merged, deep);
  auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), merged);
  for (size_t i = 0; i < hm.extent(0); ++i)
    for (size_t j = 0; j < hm.extent(1); ++j)
      EXPECT_TRUE(std::isfinite((float)hm(i, j))) << i << "," << j;
  for (const auto& d : deep) {
    auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    for (size_t i = 0; i < hd.extent(0); ++i)
      for (size_t j = 0; j < hd.extent(1); ++j)
        EXPECT_TRUE(std::isfinite((float)hd(i, j)));
  }
  // Determinism: a second forward matches bit-for-bit.
  Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec> merged2("merged2", Nv,
                                                           cfg.out_hidden_size);
  std::vector<Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec>> deep2;
  for (int k = 0; k < 3; ++k)
    deep2.emplace_back("deep2", Nv, cfg.out_hidden_size);
  model.forward(handle, pix, Hp, Wp, merged2, deep2);
  auto hm2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), merged2);
  for (size_t i = 0; i < hm.extent(0); ++i)
    for (size_t j = 0; j < hm.extent(1); ++j)
      EXPECT_EQ(hm(i, j), hm2(i, j));
}

TEST(QwenImage21Vision, RejectsBadGrid) {
  QV::Config cfg = QV::Config::tiny_test();
  QV::QwenImage21VisionModel<Exec, bhalf> model(cfg);
  model.weights().allocate_expected(cfg);
  model.weights().randomize(7);
  KokkosDNN::DNNHandle<Exec> handle;
  Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec> pix("pix", 16,
                                                       cfg.patch_dim());
  Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec> merged("merged", 4,
                                                          cfg.out_hidden_size);
  std::vector<Kokkos::View<bhalf**, Kokkos::LayoutRight, Exec>> deep;
  for (int k = 0; k < 3; ++k)
    deep.emplace_back("deep", 4, cfg.out_hidden_size);
  // 3x4 grid does not tile the 2x2 merge.
  EXPECT_THROW(model.forward(handle, pix, 3, 4, merged, deep),
               std::invalid_argument);
}

TEST(QwenImage21Vision, TinyJointForward) {
  // Joint text+vision mechanics on random weights: 8-token sequence with 2
  // vision slots, true M-RoPE positions, DeepStack adds. Guards shapes and
  // finiteness of the forward_joint path (golden values need the checkpoint
  // + torch reference; see scripts/qwen_image_21_verify.sh --edit-encoder).
  QE::Config ecfg = QE::Config::tiny_test();
  QE::QwenImage21EncoderModel<Exec, bhalf> enc(ecfg);
  enc.weights().allocate(ecfg);
  enc.weights().random_init(11);
  KokkosDNN::DNNHandle<Exec> handle;
  constexpr size_t S = 8, Nv = 2, D = 32;
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> toks("toks", 1, S);
  Kokkos::View<int*, Exec> rows("rows", Nv);
  Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec> vis("vis", 1, Nv, D);
  Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec> deep[3] = {
      Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec>("deep0", 1, Nv, D),
      Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec>("deep1", 1, Nv, D),
      Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec>("deep2", 1, Nv, D)};
  Kokkos::View<int*, Exec> pos3("pos3", 3 * S);
  Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec> hid("hid", 1, S, D);
  {
    auto ht = Kokkos::create_mirror_view(toks);
    for (size_t s = 0; s < S; ++s) ht(0, s) = (int)(s * 3 + 1);  // < vocab
    Kokkos::deep_copy(toks, ht);
    auto hr = Kokkos::create_mirror_view(rows);
    hr(0) = 3;
    hr(1) = 4;
    Kokkos::deep_copy(rows, hr);
    auto hv = Kokkos::create_mirror_view(vis);
    for (size_t k = 0; k < Nv; ++k)
      for (size_t d = 0; d < D; ++d)
        hv(0, k, d) = bhalf(0.01f * (float)(k + d));
    Kokkos::deep_copy(vis, hv);
    for (int q = 0; q < 3; ++q) {
      auto hd = Kokkos::create_mirror_view(deep[q]);
      for (size_t k = 0; k < Nv; ++k)
        for (size_t d = 0; d < D; ++d)
          hd(0, k, d) = bhalf(0.001f * (float)(q + k + d));
      Kokkos::deep_copy(deep[q], hd);
    }
    auto hp = Kokkos::create_mirror_view(pos3);
    for (size_t r = 0; r < 3; ++r)
      for (size_t s = 0; s < S; ++s) hp(r * S + s) = (int)s;
    Kokkos::deep_copy(pos3, hp);
  }
  enc.forward_joint(handle, toks, rows, vis, deep, pos3, hid, /*drop_idx=*/2);
  auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hid);
  for (size_t s = 0; s < S; ++s)
    for (size_t d = 0; d < D; ++d)
      EXPECT_TRUE(std::isfinite((float)hh(0, s, d))) << s << "," << d;
}

}  // namespace
