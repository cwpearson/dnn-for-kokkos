// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// GroupNorm op + FLUX.2-small (MOSS-Image) VAE decoder tests.
// GroupNorm is checked against a host reference (float + bhalf); the
// BN-stat denormalize + depth-to-space glue is already covered by
// MossVae.DenormAndDepthToSpace in Test_MossImageOps.cpp (referenced, not
// duplicated here). The decoder smoke uses Vae::Config::tiny_test() with
// random weights (no 250MB checkpoint needed).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_MossImageVae.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using bhalf = Kokkos::Experimental::bhalf_t;

template <class Scalar>
void check_groupnorm(float tol) {
  const size_t N = 2, C = 8, H = 3, W = 4, G = 4;  // Cg = 2
  Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> x("x", N, C, H, W),
      y("y", N, C, H, W);
  Kokkos::View<Scalar*, Exec> w("w", C), b("b", C);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hb = Kokkos::create_mirror_view(b);
  for (size_t n = 0; n < N; ++n)
    for (size_t c = 0; c < C; ++c)
      for (size_t h = 0; h < H; ++h)
        for (size_t ww = 0; ww < W; ++ww)
          hx(n, c, h, ww) = static_cast<Scalar>(0.3f * n - 0.2f * c +
                                                0.11f * h + 0.07f * ww - 0.4f);
  for (size_t c = 0; c < C; ++c) {
    hw(c) = static_cast<Scalar>(0.8f + 0.05f * c);
    hb(c) = static_cast<Scalar>(-0.3f + 0.04f * c);
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  const float eps = 1e-6f;
  KokkosDNN::groupnorm_forward<Exec, Scalar>(x, w, b, G, eps, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);
  const size_t Cg = C / G;
  for (size_t n = 0; n < N; ++n)
    for (size_t g = 0; g < G; ++g) {
      double sum = 0;
      for (size_t c = 0; c < Cg; ++c)
        for (size_t h = 0; h < H; ++h)
          for (size_t ww = 0; ww < W; ++ww)
            sum += static_cast<double>(hx(n, g * Cg + c, h, ww));
      const double mean = sum / (Cg * H * W);
      double sq = 0;
      for (size_t c = 0; c < Cg; ++c)
        for (size_t h = 0; h < H; ++h)
          for (size_t ww = 0; ww < W; ++ww) {
            const double d =
                static_cast<double>(hx(n, g * Cg + c, h, ww)) - mean;
            sq += d * d;
          }
      const double inv = 1.0 / std::sqrt(sq / (Cg * H * W) + eps);
      for (size_t c = 0; c < Cg; ++c)
        for (size_t h = 0; h < H; ++h)
          for (size_t ww = 0; ww < W; ++ww) {
            const double want =
                (static_cast<double>(hx(n, g * Cg + c, h, ww)) - mean) * inv *
                    static_cast<double>(hw(g * Cg + c)) +
                static_cast<double>(hb(g * Cg + c));
            EXPECT_NEAR(static_cast<double>(got(n, g * Cg + c, h, ww)), want,
                        tol)
                << "mismatch at (" << n << "," << g * Cg + c << "," << h
                << "," << ww << ")";
          }
    }
}

TEST(GroupNorm, Float) { check_groupnorm<float>(1e-5f); }
TEST(GroupNorm, BHalf) { check_groupnorm<bhalf>(5e-2f); }

TEST(GroupNorm, InPlaceAndValidation) {
  const size_t N = 1, C = 4, H = 2, W = 2, G = 2;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> x("x", N, C, H, W),
      ref("ref", N, C, H, W);
  Kokkos::View<float*, Exec> w("w", C), b("b", C);
  auto hx = Kokkos::create_mirror_view(x);
  auto hw = Kokkos::create_mirror_view(w);
  auto hb = Kokkos::create_mirror_view(b);
  for (size_t c = 0; c < C; ++c)
    for (size_t h = 0; h < H; ++h)
      for (size_t ww = 0; ww < W; ++ww)
        hx(0, c, h, ww) = 0.5f * c - 0.3f * h + 0.2f * ww + 0.1f;
  for (size_t c = 0; c < C; ++c) {
    hw(c) = 1.0f + 0.1f * c;
    hb(c) = -0.1f * c;
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(ref, hx);
  Kokkos::deep_copy(w, hw);
  Kokkos::deep_copy(b, hb);
  // Out-of-place reference, then in-place must match it.
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> y("y", N, C, H, W);
  KokkosDNN::groupnorm_forward<Exec, float>(x, w, b, G, 1e-6f, y);
  KokkosDNN::groupnorm_forward<Exec, float>(x, w, b, G, 1e-6f, x);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto want = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (size_t c = 0; c < C; ++c)
    for (size_t h = 0; h < H; ++h)
      for (size_t ww = 0; ww < W; ++ww)
        EXPECT_FLOAT_EQ(got(0, c, h, ww), want(0, c, h, ww));
  // Validation.
  EXPECT_THROW(
      (KokkosDNN::groupnorm_forward<Exec, float>(x, w, b, 0, 1e-6f, y)),
      std::invalid_argument);  // num_groups == 0
  EXPECT_THROW(
      (KokkosDNN::groupnorm_forward<Exec, float>(x, w, b, 3, 1e-6f, y)),
      std::invalid_argument);  // C % G != 0
  EXPECT_THROW(
      (KokkosDNN::groupnorm_forward<Exec, float>(x, w, b, G, -1.0f, y)),
      std::invalid_argument);  // bad eps
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> bad("bad", N, C + 1, H,
                                                         W);
  EXPECT_THROW(
      (KokkosDNN::groupnorm_forward<Exec, float>(x, w, b, G, 1e-6f, bad)),
      std::invalid_argument);  // dim mismatch
}

TEST(MossImageVaeConfig, GroundedDefaults) {
  auto c = KokkosDNN::MossImage::Vae::Config::moss_image_sft();
  EXPECT_EQ(c.dit_latent_channels, 128);
  EXPECT_EQ(c.latent_channels, 32);
  EXPECT_EQ(c.out_channels, 3);
  EXPECT_EQ(c.decoder_widths, std::vector<int>({96, 192, 384, 384}));
  EXPECT_EQ(c.layers_per_block, 2);
  EXPECT_EQ(c.num_groups, 32);
  EXPECT_FLOAT_EQ(c.bn_eps, 1e-4f);
  EXPECT_FLOAT_EQ(c.gn_eps, 1e-6f);
  EXPECT_EQ(c.scale_factor, 16);
  EXPECT_EQ(c.mid_width(), 384);
  EXPECT_EQ(c.resnets_per_up_block(), 3);
  EXPECT_NO_THROW(c.validate());
  auto t = KokkosDNN::MossImage::Vae::Config::tiny_test();
  EXPECT_NO_THROW(t.validate());
  // Full-config tensor inventory is non-trivial (BN stats + post-quant +
  // conv_in + mid (2 ResNet + attention) + 4 up blocks + head).
  auto spec = KokkosDNN::MossImage::Vae::expected_decoder_tensors(c);
  EXPECT_GT(spec.size(), 100u);
  for (const auto& s : spec) {
    EXPECT_FALSE(s.name.empty());
    EXPECT_FALSE(s.shape.empty());
  }
}

TEST(MossImageVaeWeights, AllocateResidentRelease) {
  using W = KokkosDNN::MossImage::Vae::Weights<Exec>;
  auto cfg = KokkosDNN::MossImage::Vae::Config::tiny_test();
  W w;
  EXPECT_EQ(w.resident_bytes(), 0u);
  KokkosDNN::MossImage::Vae::random_init(cfg, w, 42u);
  EXPECT_GT(w.resident_bytes(), 0u);
  EXPECT_EQ(w.ups.size(), 2u);
  EXPECT_TRUE(w.ups[0].has_up);
  EXPECT_FALSE(w.ups[1].has_up);
  EXPECT_TRUE(w.ups[1].res[0].has_shortcut());   // 8 -> 4 shrink
  EXPECT_FALSE(w.ups[0].res[0].has_shortcut());  // 8 -> 8
  w.release();
  EXPECT_EQ(w.resident_bytes(), 0u);
}

TEST(MossImageVaeLoader, MissingFileThrows) {
  using W = KokkosDNN::MossImage::Vae::Weights<Exec>;
  auto cfg = KokkosDNN::MossImage::Vae::Config::tiny_test();
  W w;
  EXPECT_THROW(
      KokkosDNN::MossImage::Vae::load_safetensors(
          cfg, "/tmp/wt/moss-vae/definitely_not_here.safetensors", w),
      std::runtime_error);
  // Junk bytes are not a safetensors file.
  const std::string junk = "/tmp/wt/moss-vae/test_not_safetensors.bin";
  {
    std::ofstream f(junk, std::ios::binary | std::ios::trunc);
    f.write("not a safetensors file at all..............", 44);
  }
  EXPECT_THROW(KokkosDNN::MossImage::Vae::load_safetensors(cfg, junk, w),
               std::runtime_error);
  std::remove(junk.c_str());
}

TEST(MossImageVaeDecoder, TinySmoke) {
  using namespace KokkosDNN::MossImage::Vae;
  auto cfg = Config::tiny_test();
  Weights<Exec> w;
  random_init(cfg, w, 1234u);
  DecoderModel<Exec> model(cfg, w);
  // tiny_test: dit 8ch 4x4 latents -> d2s 2ch 8x8 -> up to SxS RGB.
  const size_t B = 1, H = 4, Wd = 4;
  const size_t S = (size_t)cfg.scale_factor;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> lat("lat", B, 8, H, Wd),
      rgb("rgb", B, 3, S * H, S * Wd);
  auto hlat = Kokkos::create_mirror_view(lat);
  for (size_t c = 0; c < 8; ++c)
    for (size_t h = 0; h < H; ++h)
      for (size_t ww = 0; ww < Wd; ++ww)
        hlat(0, c, h, ww) = 0.1f * c - 0.05f * h + 0.03f * ww - 0.2f;
  Kokkos::deep_copy(lat, hlat);
  model.decode(lat, rgb);
  Kokkos::fence();
  EXPECT_EQ(rgb.extent(0), B);
  EXPECT_EQ(rgb.extent(1), 3u);
  EXPECT_EQ(rgb.extent(2), S * H);
  EXPECT_EQ(rgb.extent(3), S * Wd);
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rgb);
  for (size_t c = 0; c < 3; ++c)
    for (size_t h = 0; h < S * H; ++h)
      for (size_t ww = 0; ww < S * Wd; ++ww)
        EXPECT_TRUE(std::isfinite(got(0, c, h, ww)))
            << "non-finite at (" << c << "," << h << "," << ww << ")";
  // Wrong channel count / wrong output geometry must throw.
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> bad_lat("bl", B, 7, H,
                                                             Wd),
      bad_rgb("br", B, 3, 8 * H, 8 * Wd);
  EXPECT_THROW(model.decode(bad_lat, rgb), std::invalid_argument);
  EXPECT_THROW(model.decode(lat, bad_rgb), std::invalid_argument);
}

}  // namespace
