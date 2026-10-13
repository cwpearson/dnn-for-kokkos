// Tiny-random smoke test for the Sana example (no weights, no downloads).
//
// Covers: Config::tiny_test()/validate()/sana600m_1024() geometry, and a
// random_init + synthetic-latents forward checking out shape [B,Cout,H,W],
// all-finite outputs, and determinism (two forwards agree bit-for-bit).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_SanaConfig.hpp"
#include "KokkosDNN_SanaModel.hpp"
#include "KokkosDNN_SanaWeights.hpp"

namespace {

namespace SN   = KokkosDNN::Sana;
using Exec     = Kokkos::DefaultExecutionSpace;
using Scalar   = SN::DefaultScalar;  // float
using LatR     = Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec>;
using TokR     = Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>;
using TVec     = Kokkos::View<float*, Exec>;

// Deterministic synthetic latents/captions/timesteps.
LatR make_latents(size_t B, size_t C, size_t H, size_t W) {
  LatR x("sana_synth_x", B, C, H, W);
  Kokkos::parallel_for(
      "sana_synth_x",
      Kokkos::MDRangePolicy<Exec, Kokkos::Rank<4>>({0, 0, 0, 0}, {B, C, H, W}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
        float v = static_cast<float>((h * W + w + b * 7 + c * 13) % 64) *
                      (1.0f / 64.0f) -
                  0.5f;
        x(b, c, h, w) = static_cast<Scalar>(v);
      });
  Kokkos::fence();
  return x;
}

TokR make_captions(size_t B, size_t L, size_t Cc) {
  TokR y("sana_synth_y", B, L, Cc);
  Kokkos::parallel_for(
      "sana_synth_y",
      Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>({0, 0, 0}, {B, L, Cc}),
      KOKKOS_LAMBDA(size_t b, size_t l, size_t c) {
        float v = static_cast<float>((l * Cc + c + b * 3) % 32) *
                      (1.0f / 32.0f) -
                  0.5f;
        y(b, l, c) = static_cast<Scalar>(v);
      });
  Kokkos::fence();
  return y;
}

TVec make_timesteps(size_t B) {
  TVec t("sana_synth_t", B);
  auto th = Kokkos::create_mirror_view(t);
  for (size_t b = 0; b < B; ++b) th(b) = 500.0f + 10.0f * (float)b;
  Kokkos::deep_copy(t, th);
  return t;
}

std::vector<float> copy_to_host_4d(const LatR& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  std::vector<float> out(h.extent(0) * h.extent(1) * h.extent(2) *
                         h.extent(3));
  size_t k = 0;
  for (size_t b = 0; b < h.extent(0); ++b)
    for (size_t c = 0; c < h.extent(1); ++c)
      for (size_t hh = 0; hh < h.extent(2); ++hh)
        for (size_t w = 0; w < h.extent(3); ++w)
          out[k++] = static_cast<float>(h(b, c, hh, w));
  return out;
}

}  // namespace

TEST(SanaSmoke, TinyConfigValidates) {
  SN::Config cfg = SN::Config::tiny_test();
  EXPECT_NO_THROW(cfg.validate());
  EXPECT_EQ(cfg.mlp_hidden(), 64);
  EXPECT_EQ(cfg.cross_head_dim(), 8);
  EXPECT_EQ(cfg.num_tokens(4, 4), 16);
  EXPECT_EQ(cfg.num_tokens(4, 5), 20);
  EXPECT_EQ(cfg.num_tokens(3, 4), 12);  // patch=1 tiles everything
}

TEST(SanaSmoke, FullConfigValidates) {
  SN::Config cfg = SN::Config::sana600m_1024();
  EXPECT_NO_THROW(cfg.validate());
  EXPECT_EQ(cfg.depth, 28);
  EXPECT_EQ(cfg.hidden_size, 1152);
  EXPECT_EQ(cfg.linear_num_heads, 36);
  EXPECT_EQ(cfg.linear_head_dim, 32);
  EXPECT_EQ(cfg.mlp_hidden(), 2880);
  EXPECT_EQ(cfg.cross_head_dim(), 72);
  EXPECT_FALSE(cfg.pred_sigma);
  EXPECT_EQ(cfg.out_channels, cfg.in_channels);

  SN::Config bad = cfg;
  bad.hidden_size = 100;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = cfg;
  bad.use_pe = true;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = SN::Config::tiny_test();
  bad.qk_norm = true;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(SanaSmoke, TinyRandomForwardShapeFiniteDeterministic) {
  SN::Config cfg = SN::Config::tiny_test();
  SN::Weights<Exec, Scalar> w;
  SN::random_init(cfg, w, 42);
  SN::Model<Exec, Scalar> model(cfg, w);

  const size_t B = 1, H = 4, W = 4, L = 8;
  LatR x   = make_latents(B, (size_t)cfg.in_channels, H, W);
  TVec t   = make_timesteps(B);
  TokR y   = make_captions(B, L, (size_t)cfg.caption_channels);
  LatR out("sana_out", B, (size_t)cfg.out_channels, H, W);

  model.forward(x, t, y, out);
  Kokkos::fence();
  EXPECT_EQ(out.extent(0), B);
  EXPECT_EQ(out.extent(1), (size_t)cfg.out_channels);
  EXPECT_EQ(out.extent(2), H);
  EXPECT_EQ(out.extent(3), W);

  std::vector<float> first = copy_to_host_4d(out);
  ASSERT_EQ(first.size(), B * (size_t)cfg.out_channels * H * W);
  for (float v : first) EXPECT_TRUE(std::isfinite(v));

  // Second forward must agree bit-for-bit (determinism / no aliasing bugs).
  LatR out2("sana_out2", B, (size_t)cfg.out_channels, H, W);
  model.forward(x, t, y, out2);
  Kokkos::fence();
  std::vector<float> second = copy_to_host_4d(out2);
  ASSERT_EQ(first.size(), second.size());
  for (size_t i = 0; i < first.size(); ++i)
    EXPECT_EQ(first[i], second[i]) << "forward is not deterministic at " << i;

  // Batched forward (B=2) runs and stays finite.
  const size_t B2 = 2;
  LatR xb        = make_latents(B2, (size_t)cfg.in_channels, H, W);
  TVec tb        = make_timesteps(B2);
  TokR yb        = make_captions(B2, L, (size_t)cfg.caption_channels);
  LatR ob("sana_ob", B2, (size_t)cfg.out_channels, H, W);
  model.forward(xb, tb, yb, ob);
  Kokkos::fence();
  for (float v : copy_to_host_4d(ob)) EXPECT_TRUE(std::isfinite(v));
}

TEST(SanaSmoke, LoaderRejectsMissingFile) {
  SN::Config cfg = SN::Config::tiny_test();
  SN::Weights<Exec, Scalar> w;
  EXPECT_THROW(SN::load_torch_pth(cfg, "/nonexistent/path/model.pth", w),
               std::runtime_error);
}

TEST(SanaSmoke, ForwardRejectsBadShapes) {
  SN::Config cfg = SN::Config::tiny_test();
  SN::Weights<Exec, Scalar> w;
  SN::random_init(cfg, w, 42);
  SN::Model<Exec, Scalar> model(cfg, w);

  LatR x   = make_latents(1, (size_t)cfg.in_channels, 4, 4);
  TVec t   = make_timesteps(1);
  TokR y   = make_captions(1, 8, (size_t)cfg.caption_channels);
  LatR out("sana_out", 1, (size_t)cfg.out_channels, 4, 4);

  TokR y_bad("sana_y_bad", 1, (size_t)cfg.model_max_length + 1,
             (size_t)cfg.caption_channels);
  EXPECT_THROW(model.forward(x, t, y_bad, out), std::invalid_argument);
  LatR out_bad("sana_out_bad", 1, (size_t)cfg.out_channels + 1, 4, 4);
  EXPECT_THROW(model.forward(x, t, y, out_bad), std::invalid_argument);
}
