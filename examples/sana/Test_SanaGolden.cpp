// Golden real-checkpoint validation for the Sana example.
//
// The parser/inventory tests below run against the released
// `Sana_600M_1024px_MultiLing.pth` torch.save archive WITHOUT libtorch
// (dependency-free ZIP + pickle reader): they validate the full tensor
// inventory (order, shapes, strides, dtype, storage sizes) and the
// state_dict _metadata. They SKIP gracefully when the archive is absent
// (tests never download weights).
//
// The real-weight forward (FP32, null caption, fixed-seed latents) is
// additionally gated on KOKKOSDNN_SANA_WEIGHTS=<path-to-.pth>: a full 600M
// DiT forward is minutes-per-call on a serial CPU build, so it is opt-in
// even when the in-repo checkpoint exists. With KOKKOSDNN_SANA_GOLDEN_JSON
// pointing at examples/sana/generate_golden.py output, output mean/std are
// compared against the torch reference (else finite + print only).
//
// Env:
//   KOKKOSDNN_SANA_PTH     optional .pth override for the inventory test
//                          (default: the in-repo checkpoint, CWD-relative)
//   KOKKOSDNN_SANA_WEIGHTS .pth for the opt-in real-weight forward
//   KOKKOSDNN_SANA_GOLDEN_JSON golden JSON for mean/std comparison
//   KOKKOSDNN_SANA_TOL      optional abs tolerance override (default 2e-3)
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_SanaConfig.hpp"
#include "KokkosDNN_SanaModel.hpp"
#include "KokkosDNN_SanaPth.hpp"
#include "KokkosDNN_SanaSampler.hpp"
#include "KokkosDNN_SanaWeights.hpp"

namespace {

namespace SN = KokkosDNN::Sana;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = SN::DefaultScalar;  // float

const char* kDefaultPth =
    "Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth";

const char* getenv_str(const char* name) {
  const char* p = std::getenv(name);
  return (p && *p) ? p : nullptr;
}

bool file_readable(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

// .pth for the inventory test: explicit env override wins, else the
// CWD-relative in-repo default, else the workspace-absolute fallback
// (SANA_WORKSPACE_DIR, for ctest runs whose CWD is the build dir).
// Returns "" when nothing is readable (callers GTEST_SKIP).
std::string inventory_pth() {
  if (const char* e = getenv_str("KOKKOSDNN_SANA_PTH"))
    if (file_readable(e)) return e;
  if (file_readable(kDefaultPth)) return kDefaultPth;
#ifdef SANA_WORKSPACE_DIR
  const std::string abs = std::string(SANA_WORKSPACE_DIR) + "/" + kDefaultPth;
  if (file_readable(abs)) return abs;
#endif
  return "";
}

// Minimal {"key": number} extractor for the golden JSON (no third-party
// dependency). Returns false when the key is absent/unparseable.
bool json_number(const std::string& text, const std::string& key,
                 double& out) {
  const std::string q = "\"" + key + "\"";
  size_t p = text.find(q);
  if (p == std::string::npos) return false;
  p = text.find(':', p + q.size());
  if (p == std::string::npos) return false;
  const char* begin = text.c_str() + p + 1;
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return false;
  out = v;
  return true;
}

TEST(SanaGolden, JsonNumberExtractor) {
  const std::string js = R"({"mean": 0.5, "std": 2.25, "s": "x"})";
  double v = 0;
  EXPECT_TRUE(json_number(js, "mean", v));
  EXPECT_DOUBLE_EQ(v, 0.5);
  EXPECT_TRUE(json_number(js, "std", v));
  EXPECT_DOUBLE_EQ(v, 2.25);
  EXPECT_FALSE(json_number(js, "missing", v));
  EXPECT_FALSE(json_number("{bad", "mean", v));
}

TEST(SanaGolden, PthInventoryMatchesRelease) {
  const std::string pth = inventory_pth();
  if (pth.empty())
    GTEST_SKIP() << "Sana .pth not found (tried KOKKOSDNN_SANA_PTH and "
                 << kDefaultPth << "); skipping (tests never download "
                 << "weights).";
  SN::PthArchive ar = SN::open_pth(pth);  // throws -> FAIL (corrupt file)
  SN::Config cfg = SN::Config::sana600m_1024();

  // The released checkpoint holds exactly the SanaMS_600M_P1_D28
  // state_dict: 15 global + 15 per block x28 + 3 final = 438 tensors,
  // 593,618,720 fp32 params, all plain FloatStorage.
  EXPECT_EQ(ar.tensors.size(), 438u);
  EXPECT_EQ(SN::pth_total_params(ar), 593618720LL);
  for (const auto& t : ar.tensors) {
    EXPECT_EQ(t.dtype, "torch FloatStorage") << t.name;
    EXPECT_GT(t.numel, 0) << t.name;
  }

  // Order + shapes match the Config-derived expectation exactly.
  const auto expected = SN::Impl::sana_pth_expected(cfg, 32, 32);
  ASSERT_EQ(expected.size(), ar.tensors.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(ar.tensors[i].name, expected[i].first) << "at index " << i;
    EXPECT_EQ(ar.tensors[i].shape, expected[i].second)
        << ar.tensors[i].name;
  }

  // Every tensor's [offset, offset+numel) lies inside its storage record,
  // and every record is fully covered (offsets are all 0 here, but the
  // range check is what the loader relies on).
  for (const auto& t : ar.tensors) {
    const auto it = ar.records.find(t.storage_id);
    ASSERT_TRUE(it != ar.records.end()) << t.name;
    const int64_t rec_floats = (int64_t)it->second.second / 4;
    EXPECT_LE(t.storage_offset + t.numel, rec_floats) << t.name;
  }
  // 1:1 storage coverage: 438 records for 438 tensors.
  EXPECT_EQ(ar.records.size(), ar.tensors.size());

  // _metadata: module tree with version markers (informational).
  EXPECT_FALSE(ar.metadata.empty());
  for (const auto& kv : ar.metadata) EXPECT_EQ(kv.second, 1) << kv.first;

  std::printf(
      "[sana inventory] %s: %zu tensors, %lld params, prefix '%s', "
      "first '%s', last '%s', metadata entries %zu\n",
      pth.c_str(), ar.tensors.size(), (long long)SN::pth_total_params(ar),
      ar.prefix.c_str(), ar.tensors.front().name.c_str(),
      ar.tensors.back().name.c_str(), ar.metadata.size());
}

// Opt-in real-weight forward: single DiT call, null caption, fixed-seed
// latents; checks finite mean/std (+ torch comparison when a golden JSON
// is provided). Slow on serial CPU (minutes); hence env-gated.
TEST(SanaGolden, RealWeightsForwardFp32) {
  const char* wenv = getenv_str("KOKKOSDNN_SANA_WEIGHTS");
  if (!wenv)
    GTEST_SKIP() << "set KOKKOSDNN_SANA_WEIGHTS=<.pth> to run the "
                 << "real-weight forward (minutes on serial CPU; tests never "
                 << "download weights).";
  if (!file_readable(wenv)) GTEST_SKIP() << "weights not readable: " << wenv;

  double tol = 2e-3;
  if (const char* t = getenv_str("KOKKOSDNN_SANA_TOL")) tol = std::stod(t);

  SN::Config cfg = SN::Config::sana600m_1024();
  SN::Weights<Exec, Scalar> w;
  SN::load_torch_pth(cfg, wenv, w);  // throws -> FAIL
  SN::Model<Exec, Scalar> model(cfg, w);

  // Fixed-seed latents [1,32,32,32] + null caption [1,300,2304], t = 1000
  // (first FlowEuler step).
  using LatR = Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec>;
  using TokR = Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>;
  using TVec = Kokkos::View<float*, Exec>;
  const std::vector<float> lat = SN::sample_gaussian(32 * 32 * 32, 1234);
  LatR x("sana_golden_x", 1, 32, 32, 32);
  {
    auto xh = Kokkos::create_mirror_view(x);
    size_t k = 0;
    for (size_t c = 0; c < 32; ++c)
      for (size_t h = 0; h < 32; ++h)
        for (size_t ww = 0; ww < 32; ++ww) xh(0, c, h, ww) = (Scalar)lat[k++];
    Kokkos::deep_copy(x, xh);
  }
  TokR y("sana_golden_y", 1, 300, 2304);
  {
    auto yh = Kokkos::create_mirror_view(y);
    auto wh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  w.y_embedding);
    for (size_t l = 0; l < 300; ++l)
      for (size_t c = 0; c < 2304; ++c) yh(0, l, c) = (Scalar)wh(l, c);
    Kokkos::deep_copy(y, yh);
  }
  TVec t("sana_golden_t", 1);
  {
    float t_host = 1000.0f;
    Kokkos::View<float*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        th(&t_host, 1);
    Kokkos::deep_copy(t, th);
  }
  LatR out("sana_golden_out", 1, 32, 32, 32);
  model.forward(x, t, y, out);
  Kokkos::fence();
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double mean = 0.0, mn = oh(0, 0, 0, 0), mx = oh(0, 0, 0, 0);
  for (size_t c = 0; c < 32; ++c)
    for (size_t h = 0; h < 32; ++h)
      for (size_t ww = 0; ww < 32; ++ww) {
        const double v = (double)oh(0, c, h, ww);
        EXPECT_TRUE(std::isfinite(v));
        mean += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
  mean /= 32768.0;
  double var = 0.0;
  for (size_t c = 0; c < 32; ++c)
    for (size_t h = 0; h < 32; ++h)
      for (size_t ww = 0; ww < 32; ++ww) {
        const double d = (double)oh(0, c, h, ww) - mean;
        var += d * d;
      }
  const double std = std::sqrt(var / 32768.0);
  std::printf("[sana golden] mean=%.6f std=%.6f min=%.4f max=%.4f tol=%.3g\n",
              mean, std, mn, mx, tol);
  EXPECT_TRUE(std::isfinite(mean)) << "non-finite mean";
  EXPECT_TRUE(std::isfinite(std)) << "non-finite std";

  if (const char* g = getenv_str("KOKKOSDNN_SANA_GOLDEN_JSON")) {
    if (!file_readable(g)) GTEST_SKIP() << "golden JSON not readable: " << g;
    std::ifstream f(g, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    double rmean = 0, rstd = 0;
    if (!json_number(ss.str(), "mean", rmean) ||
        !json_number(ss.str(), "std", rstd))
      FAIL() << "golden JSON lacks mean/std: " << g;
    EXPECT_NEAR(mean, rmean, tol) << "output mean vs torch reference";
    EXPECT_NEAR(std, rstd, tol) << "output std vs torch reference";
  }
}

}  // namespace
