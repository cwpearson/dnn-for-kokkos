// Kokoro golden test: full-config forward with real weights vs the torch
// reference (examples/kokoro/generate_kokoro_golden.py). Env-gated:
//   KOKKOSDNN_KOKORO_WEIGHTS=kokoro-v1_0.pth
//   KOKKOSDNN_KOKORO_VOICE=af_heart.pt (must match the golden's voice)
//   KOKKOSDNN_KOKORO_GOLDEN=golden_short.json
//   KOKKOSDNN_KOKORO_TOL=1e-3 (max-abs for deterministic stages)
//   KOKKOSDNN_KOKORO_AUDIO_TOL=0.02 (relative RMS tolerance; SineGen RNG
//     differs from torch by stream, so audio compares statistically)
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroModel.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = KokkosDNN::Kokoro::DefaultScalar;

const char* env(const char* k, const char* dflt = nullptr) {
  const char* v = std::getenv(k);
  return v ? v : dflt;
}

// Minimal JSON scanner for the machine-generated golden file: extracts a
// scalar ("key": number/string) or flat/nested number arrays.
struct Golden {
  std::string text;
  size_t find_key(const std::string& key) const {
    const std::string q = "\"" + key + "\"";
    size_t p = text.find(q);
    if (p == std::string::npos)
      throw std::runtime_error("golden: no key '" + key + "'");
    p = text.find(':', p);
    if (p == std::string::npos) throw std::runtime_error("golden: bad key");
    return p + 1;
  }
  static void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' ||
                            s[p] == '\t'))
      ++p;
  }
  double number_at(size_t& p) const {
    skip_ws(text, p);
    size_t q = p;
    while (q < text.size() && (std::isdigit(text[q]) || text[q] == '-' ||
                               text[q] == '+' || text[q] == '.' ||
                               text[q] == 'e' || text[q] == 'E'))
      ++q;
    double v = std::stod(text.substr(p, q - p));
    p = q;
    return v;
  }
  // Parse a possibly-nested array into flat values + shape.
  void array_at(size_t& p, std::vector<double>& out,
                std::vector<size_t>& shape) const {
    skip_ws(text, p);
    if (p >= text.size() || text[p] != '[')
      throw std::runtime_error("golden: expected array");
    ++p;
    // Detect nesting: '[' after optional whitespace.
    size_t q = p;
    skip_ws(text, q);
    if (q < text.size() && text[q] == '[') {
      shape.push_back(0);
      while (true) {
        skip_ws(text, p);
        if (p < text.size() && text[p] == ']') {
          ++p;
          break;
        }
        std::vector<size_t> sub;
        array_at(p, out, sub);
        if (shape.size() == 1) shape.push_back(sub[0]);
        shape[0]++;
        skip_ws(text, p);
        if (p < text.size() && text[p] == ',') {
          ++p;
          continue;
        }
      }
      return;
    }
    shape.push_back(0);
    while (true) {
      skip_ws(text, p);
      if (p < text.size() && text[p] == ']') {
        ++p;
        break;
      }
      out.push_back(number_at(p));
      shape[0]++;
      skip_ws(text, p);
      if (p < text.size() && text[p] == ',') {
        ++p;
        continue;
      }
    }
  }
  std::vector<double> numbers(const std::string& key) const {
    size_t p = find_key(key);
    std::vector<double> out;
    std::vector<size_t> shape;
    array_at(p, out, shape);
    return out;
  }
  std::vector<std::vector<double>> nested(const std::string& key) const {
    size_t p = find_key(key);
    skip_ws(text, p);
    if (text[p] != '[') throw std::runtime_error("golden: expected array");
    ++p;
    std::vector<std::vector<double>> rows;
    while (true) {
      skip_ws(text, p);
      if (p < text.size() && text[p] == ']') break;
      std::vector<double> row;
      std::vector<size_t> shape;
      array_at(p, row, shape);
      rows.push_back(std::move(row));
      skip_ws(text, p);
      if (p < text.size() && text[p] == ',') ++p;
    }
    return rows;
  }
  double number(const std::string& key) const {
    size_t p = find_key(key);
    return number_at(p);
  }
};

Golden load_golden(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open golden '" + path + "'");
  const long long n = (long long)f.tellg();
  f.seekg(0, std::ios::beg);
  Golden g;
  g.text.resize((size_t)n);
  f.read(g.text.data(), n);
  return g;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<double>& b) {
  EXPECT_EQ(a.size(), b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i)
    m = std::max(m, std::fabs((double)a[i] - b[i]));
  return m;
}

TEST(KokoroGolden, RealWeightsForwardFp32) {
  const char* wp = env("KOKKOSDNN_KOKORO_WEIGHTS");
  const char* vp = env("KOKKOSDNN_KOKORO_VOICE");
  const char* gp = env("KOKKOSDNN_KOKORO_GOLDEN");
  if (!wp || !vp || !gp)
    GTEST_SKIP() << "set KOKKOSDNN_KOKORO_WEIGHTS/_VOICE/_GOLDEN to run";
  const double tol = env("KOKKOSDNN_KOKORO_TOL")
                         ? std::stod(env("KOKKOSDNN_KOKORO_TOL"))
                         : 1e-3;
  const double atol = env("KOKKOSDNN_KOKORO_AUDIO_TOL")
                          ? std::stod(env("KOKKOSDNN_KOKORO_AUDIO_TOL"))
                          : 0.02;
  Golden g = load_golden(gp);
  KokkosDNN::Kokoro::Config cfg;
  KokkosDNN::Kokoro::Weights<Exec, Scalar> w;
  KokkosDNN::Kokoro::load_pth_weights(cfg, wp, w);
  KokkosDNN::Kokoro::VoicePack voice = KokkosDNN::Kokoro::load_voice(vp);

  std::vector<double> gd = g.numbers("ids");
  std::vector<int> ids(gd.begin(), gd.end());
  const int row = (int)g.number("n_codepoints") - 1;
  ASSERT_LT(row, voice.n_rows);
  std::vector<float> style(voice.rows.begin() + (size_t)row * 256,
                           voice.rows.begin() + (size_t)(row + 1) * 256);
  const float speed = (float)g.number("speed");

  KokkosDNN::Kokoro::Model<Exec, Scalar> m(cfg, w);
  KokkosDNN::Kokoro::Intermediates dbg;
  std::vector<float> audio = m.synthesize(ids, style, speed, 42, &dbg);

  // Deterministic stages: exact durations, tolerant curves/features.
  std::vector<double> gdur = g.numbers("pred_dur");
  ASSERT_EQ(dbg.pred_dur.size(), gdur.size());
  for (size_t i = 0; i < gdur.size(); ++i)
    EXPECT_EQ(dbg.pred_dur[i], (int)gdur[i]) << i;
  EXPECT_EQ(dbg.frames, (int)g.number("frames"));
  EXPECT_LE(max_abs_diff(dbg.f0, g.numbers("f0")), tol) << "F0";
  EXPECT_LE(max_abs_diff(dbg.ncurve, g.numbers("ncurve")), tol) << "N";
  {
    auto gasr = g.nested("asr");  // [512, F] torch layout
    ASSERT_EQ((int)gasr.size(), 512);
    ASSERT_EQ((int)gasr[0].size(), dbg.frames);
    double mx = 0.0;
    for (int d = 0; d < 512; ++d)
      for (int f = 0; f < dbg.frames; ++f)
        mx = std::max(mx, std::fabs((double)dbg.asr[(size_t)f * 512 +
                                                     (size_t)d] -
                                    gasr[(size_t)d][(size_t)f]));
    EXPECT_LE(mx, tol) << "asr";
  }
  // Audio: exact length, statistical energy (RNG stream differs).
  EXPECT_EQ((int)audio.size(), (int)g.number("audio_len"));
  double e = 0.0, peak = 0.0;
  for (float v : audio) {
    EXPECT_TRUE(std::isfinite(v));
    e += (double)v * v;
    peak = std::max(peak, (double)std::fabs(v));
  }
  const double rms = std::sqrt(e / (double)audio.size());
  const double grms = g.number("audio_rms");
  EXPECT_NEAR(rms, grms, atol * grms) << "rms";
  const double gpeak = g.number("audio_peak");
  EXPECT_NEAR(peak, gpeak, 0.10 * gpeak) << "peak";
}

}  // namespace
