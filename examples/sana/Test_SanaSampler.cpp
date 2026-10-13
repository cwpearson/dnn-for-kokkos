// FlowEuler sampler tests for the Sana example (no weights, no downloads).
//
// Covers the Sana scheduler math against its torch reference
// (NVlabs/Sana FlowEuler over diffusers' FlowMatchEulerDiscreteScheduler,
// flow_shift=4.0): sigma schedule shape/monotonicity/terminal, CFG + Euler
// algebra, seeded Gaussian init, and a tiny random-model 2-step Euler loop
// (finite latents, deterministic, guidance fast-path equivalence).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "KokkosDNN_SanaConfig.hpp"
#include "KokkosDNN_SanaModel.hpp"
#include "KokkosDNN_SanaSampler.hpp"
#include "KokkosDNN_SanaWeights.hpp"

namespace {

namespace SN = KokkosDNN::Sana;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = SN::DefaultScalar;  // float

TEST(SanaSampler, SigmaScheduleShape) {
  // 18 steps, shift 4 (the 1024px yaml setting): 19 sigmas, sigma_0 == 1,
  // strictly decreasing, terminal exactly 0.
  const std::vector<float> sig = SN::build_sigmas(18, 4.0f);
  ASSERT_EQ(sig.size(), 19u);
  EXPECT_FLOAT_EQ(sig[0], 1.0f);
  EXPECT_FLOAT_EQ(sig.back(), 0.0f);
  for (size_t i = 0; i + 1 < sig.size(); ++i) {
    EXPECT_GT(sig[i], sig[i + 1]) << "not decreasing at " << i;
    EXPECT_TRUE(std::isfinite(sig[i]));
  }
  // Model timesteps are 1000x sigmas (scheduler.timesteps convention).
  const std::vector<float> t = SN::build_model_timesteps(sig);
  ASSERT_EQ(t.size(), sig.size());
  EXPECT_FLOAT_EQ(t[0], 1000.0f);
  EXPECT_FLOAT_EQ(t.back(), 0.0f);
  for (size_t i = 0; i < sig.size(); ++i)
    EXPECT_FLOAT_EQ(t[i], 1000.0f * sig[i]);

  // Degenerate: a single step goes straight 1 -> 0.
  const std::vector<float> one = SN::build_sigmas(1, 4.0f);
  ASSERT_EQ(one.size(), 2u);
  EXPECT_FLOAT_EQ(one[0], 1.0f);
  EXPECT_FLOAT_EQ(one[1], 0.0f);

  // Shift 1 disables time shifting (pure linspace 1 -> 0.001, then 0).
  const std::vector<float> lin = SN::build_sigmas(3, 1.0f);
  ASSERT_EQ(lin.size(), 4u);
  EXPECT_FLOAT_EQ(lin[0], 1.0f);
  EXPECT_NEAR(lin[1], 0.5005f, 1e-6f);
  EXPECT_NEAR(lin[2], 0.001f, 1e-6f);
  EXPECT_FLOAT_EQ(lin[3], 0.0f);

  EXPECT_THROW(SN::build_sigmas(0, 4.0f), std::invalid_argument);
  EXPECT_THROW(SN::build_sigmas(8, 0.5f), std::invalid_argument);
}

TEST(SanaSampler, GuidedPredictAlgebra) {
  // Hand-computed CFG: uncond + scale * (cond - uncond).
  const std::vector<float> cond{2.0f, 4.0f, -1.0f};
  const std::vector<float> uncond{0.0f, 1.0f, 3.0f};
  std::vector<float> out;
  SN::guided_predict(cond, uncond, 5.0f, out);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_FLOAT_EQ(out[0], 10.0f);
  EXPECT_FLOAT_EQ(out[1], 16.0f);
  EXPECT_FLOAT_EQ(out[2], -17.0f);
  // Scale 1 recovers cond; scale 0 recovers uncond.
  SN::guided_predict(cond, uncond, 1.0f, out);
  EXPECT_FLOAT_EQ(out[0], 2.0f);
  SN::guided_predict(cond, uncond, 0.0f, out);
  EXPECT_FLOAT_EQ(out[2], 3.0f);
  EXPECT_THROW(SN::guided_predict(cond, {1.0f}, 5.0f, out),
               std::invalid_argument);
}

TEST(SanaSampler, EulerStepAlgebra) {
  std::vector<float> x{1.0f, 2.0f, -3.0f};
  SN::euler_step(x, {0.5f, -1.0f, 2.0f}, 2.0f);
  EXPECT_FLOAT_EQ(x[0], 2.0f);
  EXPECT_FLOAT_EQ(x[1], 0.0f);
  EXPECT_FLOAT_EQ(x[2], 1.0f);
  // Negative dt (flow sigmas decrease, so dt < 0 in practice).
  SN::euler_step(x, {1.0f, 1.0f, 1.0f}, -0.5f);
  EXPECT_FLOAT_EQ(x[0], 1.5f);
  EXPECT_THROW(SN::euler_step(x, {1.0f}, 1.0f), std::invalid_argument);
}

TEST(SanaSampler, GaussianInitSeeded) {
  const std::vector<float> a = SN::sample_gaussian(1024, 42);
  const std::vector<float> b = SN::sample_gaussian(1024, 42);
  ASSERT_EQ(a.size(), 1024u);
  EXPECT_EQ(a, b);  // same seed -> identical stream
  const std::vector<float> c = SN::sample_gaussian(1024, 43);
  EXPECT_NE(a, c);
  double mean = 0.0;
  for (float v : a) {
    EXPECT_TRUE(std::isfinite(v));
    mean += v;
  }
  mean /= (double)a.size();
  double var = 0.0;
  for (float v : a) var += (v - mean) * (v - mean);
  var /= (double)a.size();
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(std::sqrt(var), 1.0, 0.05);
  // Odd count (Box-Muller tail) stays finite and sized.
  EXPECT_EQ(SN::sample_gaussian(7, 1).size(), 7u);
}

// Tiny random-model 2-step Euler loop: finite latents, exact output size,
// deterministic across runs, and the cond==uncond fast path agrees with the
// guided path (guidance is a no-op on identical branches).
TEST(SanaSampler, TinyTwoStepEulerFiniteDeterministic) {
  SN::Config cfg = SN::Config::tiny_test();
  SN::Weights<Exec, Scalar> w;
  SN::random_init(cfg, w, 7);
  SN::Model<Exec, Scalar> model(cfg, w);

  const size_t C = (size_t)cfg.in_channels, Cc = (size_t)cfg.caption_channels;
  const size_t L = (size_t)cfg.model_max_length;
  const int H = 4, W = 4;  // tiny smoke geometry: 4x4 latents
  auto host_of = [](const auto& v) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    std::vector<float> out(h.extent(0) * h.extent(1));
    size_t k = 0;
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t j = 0; j < h.extent(1); ++j) out[k++] = (float)h(i, j);
    return out;
  };
  const std::vector<float> null = host_of(w.y_embedding);  // [L,Cc]
  ASSERT_EQ(null.size(), L * Cc);
  const std::vector<float> x0 = SN::sample_gaussian(C * 4 * 4, 42);

  SN::FlowEulerConfig s;
  s.steps = 2;
  s.guidance_scale = 5.0f;
  s.flow_shift = 4.0f;
  const std::vector<float> lat =
      SN::run_flow_euler(model, cfg, x0, null, null, s, H, W);
  ASSERT_EQ(lat.size(), C * 4 * 4);
  for (float v : lat) EXPECT_TRUE(std::isfinite(v)) << "non-finite latent";

  // Determinism: same inputs -> bit-identical latents.
  const std::vector<float> lat2 =
      SN::run_flow_euler(model, cfg, x0, null, null, s, H, W);
  EXPECT_EQ(lat, lat2);

  // Guidance disabled takes the single-branch path (still finite).
  SN::FlowEulerConfig s1 = s;
  s1.guidance_scale = 1.0f;
  const std::vector<float> lat1 =
      SN::run_flow_euler(model, cfg, x0, null, null, s1, H, W);
  ASSERT_EQ(lat1.size(), lat.size());
  for (float v : lat1) EXPECT_TRUE(std::isfinite(v));
  EXPECT_EQ(lat, lat1);  // cond == uncond: guidance must be a no-op
}

TEST(SanaSampler, RejectsBadInput) {
  SN::Config cfg = SN::Config::tiny_test();
  SN::Weights<Exec, Scalar> w;
  SN::random_init(cfg, w, 7);
  SN::Model<Exec, Scalar> model(cfg, w);
  const size_t C = (size_t)cfg.in_channels, Cc = (size_t)cfg.caption_channels;
  const size_t L = (size_t)cfg.model_max_length;
  std::vector<float> x0(C * 16, 0.0f), cap(L * Cc, 0.0f);
  SN::FlowEulerConfig s;
  s.steps = 2;
  EXPECT_THROW(SN::run_flow_euler(model, cfg, {1.0f}, cap, cap, s, 4, 4),
               std::invalid_argument);
  SN::FlowEulerConfig bad = s;
  bad.steps = 0;
  EXPECT_THROW(SN::run_flow_euler(model, cfg, x0, cap, cap, bad, 4, 4),
               std::invalid_argument);
}

}  // namespace
