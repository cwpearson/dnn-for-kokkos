// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// DC-AE VAE decoder tests (example-scoped, never download weights).
//
// - ConfigValidate: factories validate; bad configs throw.
// - TinyDecodeSmoke: random-init tiny VAE decodes 4ch 2x2 latents to 3ch
//   4x4 RGB, finite and deterministic.
// - LoaderErrors: missing file / bad magic throw (no downloads).
// - VaeHeaderInventory (env-gated): KOKKOSDNN_SANA_VAE_SAFETENSORS points at
//   the diffusers-layout DC-AE safetensors; the file header must list every
//   expected_decoder_tensors() entry with the exact shape (header-only, fast,
//   no weight load).
// - GoldenDecodeVsTorch (env-gated): KOKKOSDNN_SANA_VAE_SAFETENSORS plus
//   KOKKOSDNN_SANA_VAE_GOLDEN_JSON (from generate_vae_golden.py: input
//   latents + torch reference stats). Loads real weights, decodes small
//   latents (4x4 -> 128x128), compares mean/std/min/max/first8 against the
//   torch reference (tolerance KOKKOSDNN_SANA_VAE_TOL, default 5e-3).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "KokkosDNN_SanaVAE.hpp"

namespace {

namespace VAE = KokkosDNN::Sana::VAE;
using Exec = Kokkos::DefaultExecutionSpace;

const char* getenv_str(const char* name) {
  const char* p = std::getenv(name);
  return (p && *p) ? p : nullptr;
}

bool file_readable(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

double env_tol(double dflt) {
  if (const char* e = getenv_str("KOKKOSDNN_SANA_VAE_TOL")) {
    try {
      return std::stod(e);
    } catch (...) {
    }
  }
  return dflt;
}

// Minimal JSON number extractor (same dependency-free approach as the DiT
// golden test).
bool json_number(const std::string& text, const std::string& key, double& out) {
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

// Minimal JSON float-array extractor for "latents": [...] / "first8".
bool json_array(const std::string& text, const std::string& key,
                std::vector<double>& out) {
  const std::string q = "\"" + key + "\"";
  size_t p = text.find(q);
  if (p == std::string::npos) return false;
  p = text.find('[', p + q.size());
  if (p == std::string::npos) return false;
  const size_t e = text.find(']', p + 1);
  if (e == std::string::npos) return false;
  const char* cur = text.c_str() + p + 1;
  const char* stop = text.c_str() + e;
  out.clear();
  while (cur < stop) {
    while (cur < stop && (*cur == ' ' || *cur == '\n' || *cur == ',' ||
                          *cur == '\t' || *cur == '\r'))
      ++cur;
    if (cur >= stop) break;
    char* end = nullptr;
    const double v = std::strtod(cur, &end);
    if (end == cur) break;
    out.push_back(v);
    cur = end;
  }
  return !out.empty();
}

TEST(SanaVAE, ConfigValidate) {
  EXPECT_NO_THROW(VAE::Config::dcae_f32c32_sana().validate());
  EXPECT_NO_THROW(VAE::Config::tiny_test().validate());
  const VAE::Config sana = VAE::Config::dcae_f32c32_sana();
  EXPECT_EQ(sana.num_stages(), 6);
  EXPECT_EQ(sana.compression_ratio(), 32);
  EXPECT_EQ(VAE::Config::tiny_test().compression_ratio(), 2);
  VAE::Config bad = sana;
  bad.stage_widths.back() = 30;
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = sana;
  bad.stage_blocks[0] = "Transformer";
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  bad = sana;
  bad.stage_widths.clear();
  bad.stage_depths.clear();
  bad.stage_blocks.clear();
  EXPECT_THROW(bad.validate(), std::invalid_argument);
  // 196 decoder tensors for the Sana config (diffusers layout).
  EXPECT_EQ(VAE::expected_decoder_tensors(sana).size(), 196u);
}

TEST(SanaVAE, TinyDecodeSmoke) {
  const VAE::Config cfg = VAE::Config::tiny_test();
  VAE::Weights<Exec> w;
  VAE::random_init(cfg, w, 1234);
  VAE::Decoder<Exec> dec(cfg, w);
  const size_t C = 4, H = 2, W = 2;
  std::vector<float> lat(C * H * W);
  for (size_t i = 0; i < lat.size(); ++i)
    lat[i] = (float)((int)(i * 37 % 11) - 5) * 0.25f;
  std::vector<float> rgb = dec.decode(lat, H, W);
  EXPECT_EQ(rgb.size(), 3u * 4u * 4u);
  for (float v : rgb) EXPECT_TRUE(std::isfinite(v));
  // Deterministic: decode twice, bit-identical (same kernels, same order).
  std::vector<float> rgb2 = dec.decode(lat, H, W);
  ASSERT_EQ(rgb2.size(), rgb.size());
  for (size_t i = 0; i < rgb.size(); ++i) EXPECT_EQ(rgb2[i], rgb[i]);
  // u8 conversion clamps to [0,255].
  std::vector<unsigned char> u8 = VAE::to_u8_rgb(rgb);
  EXPECT_EQ(u8.size(), rgb.size());
}

TEST(SanaVAE, LoaderErrors) {
  const VAE::Config cfg = VAE::Config::tiny_test();
  VAE::Weights<Exec> w;
  EXPECT_THROW(VAE::load_safetensors(cfg, "/nonexistent/vae.safetensors", w),
               std::runtime_error);
  // Bad magic: not a safetensors file.
  const std::string tmp = "/tmp/opencode_sana_vae_badmagic.bin";
  {
    std::ofstream f(tmp, std::ios::binary);
    f << "this is not safetensors";
  }
  EXPECT_THROW(VAE::load_safetensors(cfg, tmp, w), std::runtime_error);
}

// The diffusers-layout DC-AE file for the header/golden tests below:
//   HF mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers,
//   diffusion_pytorch_model.safetensors. Tests never download; they SKIP
//   unless KOKKOSDNN_SANA_VAE_SAFETENSORS names a readable file.
std::string vae_file() {
  if (const char* e = getenv_str("KOKKOSDNN_SANA_VAE_SAFETENSORS"))
    if (file_readable(e)) return e;
  return "";
}

TEST(SanaVAE, HeaderInventoryMatchesDiffusers) {
  const std::string path = vae_file();
  if (path.empty())
    GTEST_SKIP() << "set KOKKOSDNN_SANA_VAE_SAFETENSORS to run";
  std::ifstream f(path, std::ios::binary);
  ASSERT_TRUE(f);
  unsigned char lb[8];
  f.read(reinterpret_cast<char*>(lb), 8);
  ASSERT_TRUE(f);
  uint64_t n = 0;
  for (int i = 7; i >= 0; --i) n = (n << 8) | lb[i];
  std::string js((size_t)n, '\0');
  f.read(js.data(), (std::streamsize)n);
  ASSERT_TRUE(f);
  const VAE::Config cfg = VAE::Config::dcae_f32c32_sana();
  size_t missing = 0;
  std::string first_missing;
  for (const auto& t : VAE::expected_decoder_tensors(cfg)) {
    const std::string q = "\"" + t.name + "\"";
    const size_t p = js.find(q);
    if (p == std::string::npos) {
      if (missing == 0) first_missing = t.name;
      ++missing;
      continue;
    }
    // Shape check: the numbers following the next "shape" key must match.
    const size_t sp = js.find("\"shape\"", p);
    ASSERT_NE(sp, std::string::npos);
    const size_t a = js.find('[', sp);
    const size_t b = js.find(']', a);
    ASSERT_NE(a, std::string::npos);
    ASSERT_NE(b, std::string::npos);
    std::string inner = js.substr(a + 1, b - a - 1);
    std::vector<size_t> shape;
    size_t i = 0;
    while (i < inner.size()) {
      while (i < inner.size() && !std::isdigit(inner[i])) ++i;
      size_t j = i;
      while (j < inner.size() && std::isdigit(inner[j])) ++j;
      if (j > i) shape.push_back(std::stoull(inner.substr(i, j - i)));
      i = j;
    }
    EXPECT_EQ(shape, t.shape) << t.name;
  }
  EXPECT_EQ(missing, 0u) << "first missing: " << first_missing;
}

TEST(SanaVAE, GoldenDecodeVsTorch) {
  const std::string path = vae_file();
  const char* gj = getenv_str("KOKKOSDNN_SANA_VAE_GOLDEN_JSON");
  if (path.empty() || !gj || !file_readable(gj))
    GTEST_SKIP() << "need KOKKOSDNN_SANA_VAE_SAFETENSORS + "
                    "KOKKOSDNN_SANA_VAE_GOLDEN_JSON";
  std::ifstream gf(gj);
  std::stringstream ss;
  ss << gf.rdbuf();
  const std::string text = ss.str();
  std::vector<double> latd;
  ASSERT_TRUE(json_array(text, "latents", latd));
  double glat = 0, gH = 0;
  ASSERT_TRUE(json_number(text, "latent_channels", glat));
  ASSERT_TRUE(json_number(text, "latent_hw", gH));
  ASSERT_EQ((int)glat, 32);
  const size_t H = (size_t)gH;
  ASSERT_EQ(latd.size(), 32 * H * H);
  std::vector<float> lat(latd.size());
  for (size_t i = 0; i < latd.size(); ++i) lat[i] = (float)latd[i];

  const VAE::Config cfg = VAE::Config::dcae_f32c32_sana();
  VAE::Weights<Exec> w;
  VAE::load_safetensors(cfg, path, w);
  VAE::Decoder<Exec> dec(cfg, w);
  std::vector<float> rgb = dec.decode(lat, H, H);
  EXPECT_EQ(rgb.size(), 3u * (32 * H) * (32 * H));

  double mean = 0, mn = rgb[0], mx = rgb[0];
  for (float v : rgb) {
    mean += v;
    mn = std::min(mn, (double)v);
    mx = std::max(mx, (double)v);
  }
  mean /= (double)rgb.size();
  double var = 0;
  for (float v : rgb) var += (v - mean) * (v - mean);
  var /= (double)rgb.size();
  const double std = std::sqrt(var);
  std::printf("[vae golden] mean=%.6f std=%.6f min=%.4f max=%.4f n=%zu\n", mean,
              std, mn, mx, rgb.size());

  const double tol = env_tol(5e-3);
  double gmean = 0, gstd = 0, gmin = 0, gmax = 0;
  ASSERT_TRUE(json_number(text, "mean", gmean));
  ASSERT_TRUE(json_number(text, "std", gstd));
  ASSERT_TRUE(json_number(text, "min", gmin));
  ASSERT_TRUE(json_number(text, "max", gmax));
  EXPECT_NEAR(mean, gmean, tol);
  EXPECT_NEAR(std, gstd, tol);
  EXPECT_NEAR(mn, gmin, tol);
  EXPECT_NEAR(mx, gmax, tol);
  std::vector<double> first8;
  if (json_array(text, "first8", first8)) {
    ASSERT_LE(first8.size(), rgb.size());
    for (size_t i = 0; i < first8.size(); ++i)
      EXPECT_NEAR(rgb[i], first8[i], tol) << "i=" << i;
  }
}

}  // namespace
