// Tiny-random smoke test for the DINOv3 example (no weights, no downloads).
//
// Covers: Config::tiny_test()/validate()/from_json_string round-trip, and a
// random_init + synthetic-pixels forward checking token shape [1+R+Np, D],
// all-finite outputs, and determinism (two forwards agree bit-for-bit).
//
// The synthetic pixels below intentionally do NOT use the optional
// KokkosDNN image component, so this test runs with
// KokkosDNN_ENABLE_IMAGE=OFF (the default).
//
// Canonical synthetic pixel formula (shared with dinov3_encode.cpp,
// Test_Dinov3Golden.cpp and examples/dinov3/generate_golden.py --synthetic;
// the golden inputs are fed directly as pixel_values with no HF processor
// normalization on either side):
//   v(c,y,x) = ((y*S + x) % 256) / 255 + 0.01 * c   (float, CHW, C=3, SxS)
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Dinov3Config.hpp"
#include "KokkosDNN_Dinov3Model.hpp"
#include "KokkosDNN_Dinov3Weights.hpp"

namespace {

namespace D3 = KokkosDNN::Dinov3;
using Exec   = Kokkos::DefaultExecutionSpace;
using Scalar = D3::DefaultScalar;  // float
using ImgR   = Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>;

// Canonical deterministic synthetic pixels (see file doc comment).
ImgR make_synthetic_pixels(const D3::Config& cfg) {
  const size_t C = (size_t)cfg.num_channels, S = (size_t)cfg.image_size;
  ImgR pixels("dinov3_synthetic", C, S, S);
  auto dst = pixels;
  Kokkos::parallel_for(
      "dinov3_synthetic",
      Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>({0, 0, 0}, {C, S, S}),
      KOKKOS_LAMBDA(size_t c, size_t y, size_t x) {
        float v = static_cast<float>((y * S + x) % 256) * (1.0f / 255.0f) +
                  0.01f * static_cast<float>(c);
        dst(c, y, x) = static_cast<Scalar>(v);
      });
  Kokkos::fence();
  return pixels;
}

std::vector<float> copy_to_host_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec>& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  std::vector<float> out(h.extent(0) * h.extent(1));
  for (size_t i = 0; i < h.extent(0); ++i)
    for (size_t j = 0; j < h.extent(1); ++j)
      out[i * h.extent(1) + j] = static_cast<float>(h(i, j));
  return out;
}

}  // namespace

TEST(Dinov3Smoke, TinyConfigValidates) {
  D3::Config cfg = D3::Config::tiny_test();
  EXPECT_NO_THROW(cfg.validate());
  // tiny_test geometry: patch 8, image 32 -> 4x4 = 16 patch tokens,
  // 1 CLS + 1 register -> 18 tokens, D = 32.
  EXPECT_EQ(cfg.num_patch_tokens(), 16);
  EXPECT_EQ(cfg.num_tokens(), 18);
  EXPECT_EQ(cfg.num_tokens(32), 18);
  EXPECT_EQ(cfg.num_patch_tokens(32), 16);
  EXPECT_EQ(cfg.num_patch_tokens(7), 0);  // non-tiling image -> 0
}

TEST(Dinov3Smoke, ConfigJsonRoundTrip) {
  // Embedded minimal JSON (no absolute paths): overrides on the tiny base.
  const std::string js = R"({
    "patch_size": 8, "hidden_size": 32, "num_hidden_layers": 2,
    "num_attention_heads": 4, "intermediate_size": 64,
    "num_register_tokens": 1, "image_size": 32, "num_channels": 3,
    "model_type": "dinov3_vit", "hidden_act": "gelu",
    "query_bias": true, "key_bias": false, "value_bias": true
  })";
  D3::Config cfg =
      D3::Config::from_json_string(js, D3::Config::tiny_test());
  EXPECT_NO_THROW(cfg.validate());
  EXPECT_EQ(cfg.patch_size, 8);
  EXPECT_EQ(cfg.hidden_size, 32);
  EXPECT_EQ(cfg.num_hidden_layers, 2);
  EXPECT_EQ(cfg.head_dim, 8);  // derived: hidden / heads
  EXPECT_EQ(cfg.num_tokens(), 18);

  // Unknown keys are ignored; missing keys keep base values.
  D3::Config keep =
      D3::Config::from_json_string(R"({"unknown_key_xyz": 123})",
                                   D3::Config::tiny_test());
  EXPECT_EQ(keep.hidden_size, 32);

  // Validation still rejects bad geometry / unsupported variants.
  EXPECT_THROW(D3::Config::from_json_string(R"({"image_size": 30})",
                                            D3::Config::tiny_test()),
               std::invalid_argument);
  EXPECT_THROW(D3::Config::from_json_string(R"({"hidden_act": "gelu_new"})",
                                            D3::Config::tiny_test()),
               std::runtime_error);
  D3::Config bad = D3::Config::tiny_test();
  bad.num_register_tokens = -1;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(Dinov3Smoke, TinyRandomForwardShapeFiniteDeterministic) {
  D3::Config cfg = D3::Config::tiny_test();
  D3::Weights<Exec, Scalar> w;
  D3::random_init(cfg, w, 42);
  D3::Model<Exec, Scalar> model(cfg, w);

  ImgR pixels = make_synthetic_pixels(cfg);
  auto out    = model.forward(pixels);
  Kokkos::fence();
  const size_t T = out.extent(0), D = out.extent(1);
  const size_t R = (size_t)cfg.num_register_tokens;
  const size_t Np = (size_t)cfg.num_patch_tokens();
  EXPECT_EQ(T, 1 + R + Np);
  EXPECT_EQ(D, (size_t)cfg.hidden_size);
  EXPECT_EQ(model.token_count(), T);

  // Copy out before the second forward: forward() aliases persistent
  // scratch, valid only until the next forward() call.
  std::vector<float> first = copy_to_host_2d(out);
  for (float v : first) EXPECT_TRUE(std::isfinite(v));

  // Token accessors alias the documented row layout (row 0 = CLS).
  auto cls   = model.cls_token();
  auto patch = model.patch_tokens();
  EXPECT_EQ(cls.extent(0), D);
  EXPECT_EQ(patch.extent(0), Np);
  EXPECT_EQ(patch.extent(1), D);

  auto out2 = model.forward(pixels);
  Kokkos::fence();
  std::vector<float> second = copy_to_host_2d(out2);
  ASSERT_EQ(first.size(), second.size());
  for (size_t i = 0; i < first.size(); ++i)
    EXPECT_EQ(first[i], second[i]) << "forward is not deterministic at " << i;
}
