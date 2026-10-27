// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Kokoro-TTS driver: phonemes (or naive text) in, 24 kHz WAV out.
//
//   kokoro_tts [--weights kokoro-v1_0.pth] [--voice af_heart.pt]
//              [--phonemes "hɛloʊ wɜːld"] [--text "hello"] [--speed 1]
//              [--seed 42] [--out kokoro_out.wav] [--tiny]
//
// KModel is language-blind: it consumes phoneme symbols, not graphemes.
// --phonemes is the primary input (raw IPA string, e.g. produced by the
// kokoro KPipeline with model=False, or espeak-ng -v en --ipa). --text is
// a demo-quality fallback that maps characters present in the vocab
// directly (no G2P; unmapped characters are skipped). WAV is peak-
// normalized PCM16.
//
// --tiny runs a random-weights smoke model (no downloads).

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroModel.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

namespace {

namespace KK = KokkosDNN::Kokoro;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = KK::DefaultScalar;

// Decode one UTF-8 code point; returns bytes consumed (0 on error).
size_t utf8_next(const std::string& s, size_t i, uint32_t& cp) {
  const size_t n = s.size();
  if (i >= n) return 0;
  const unsigned char c = (unsigned char)s[i];
  if (c < 0x80) {
    cp = c;
    return 1;
  }
  size_t len = 0;
  uint32_t v = 0;
  if ((c & 0xE0) == 0xC0) {
    len = 2;
    v = c & 0x1F;
  } else if ((c & 0xF0) == 0xE0) {
    len = 3;
    v = c & 0x0F;
  } else if ((c & 0xF8) == 0xF0) {
    len = 4;
    v = c & 0x07;
  } else {
    return 0;
  }
  if (i + len > n) return 0;
  for (size_t k = 1; k < len; ++k) {
    const unsigned char d = (unsigned char)s[i + k];
    if ((d & 0xC0) != 0x80) return 0;
    v = (v << 6) | (d & 0x3F);
  }
  cp = v;
  return len;
}

// Map a UTF-8 phoneme string to vocab ids (unknown code points skipped,
// mirroring KModel.forward's filter). Returns ids + code point count.
std::pair<std::vector<int>, size_t> phonemes_to_ids(
    const std::string& ps, const std::map<std::string, int>& vocab) {
  std::vector<int> ids;
  size_t ncp = 0;
  for (size_t i = 0; i < ps.size();) {
    uint32_t cp = 0;
    const size_t len = utf8_next(ps, i, cp);
    if (len == 0) {
      ++i;
      continue;
    }
    ++ncp;
    const std::string sym = ps.substr(i, len);
    auto it = vocab.find(sym);
    if (it != vocab.end()) ids.push_back(it->second);
    i += len;
  }
  return {ids, ncp};
}

void save_wav16(const std::string& path, const std::vector<float>& x,
                int rate, float& peak_out) {
  float peak = 0.0f;
  for (float v : x) peak = std::max(peak, std::fabs(v));
  peak_out = peak;
  const float g = (peak > 1e-9f) ? (0.95f / peak) : 1.0f;
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "'");
  const uint32_t n = (uint32_t)x.size();
  const uint32_t data_bytes = n * 2;
  auto w32 = [&](uint32_t v) {
    f.put((char)(v & 0xFF));
    f.put((char)((v >> 8) & 0xFF));
    f.put((char)((v >> 16) & 0xFF));
    f.put((char)((v >> 24) & 0xFF));
  };
  auto w16 = [&](uint16_t v) {
    f.put((char)(v & 0xFF));
    f.put((char)((v >> 8) & 0xFF));
  };
  f.write("RIFF", 4);
  w32(36 + data_bytes);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  w32(16);
  w16(1);
  w16(1);
  w32((uint32_t)rate);
  w32((uint32_t)rate * 2);
  w16(2);
  w16(16);
  f.write("data", 4);
  w32(data_bytes);
  for (float v : x) {
    float s = v * g;
    s = std::max(-1.0f, std::min(1.0f, s));
    w16((uint16_t)(int16_t)std::lround(s * 32767.0f));
  }
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

void usage() {
  std::printf(
      "usage: kokoro_tts [--weights kokoro-v1_0.pth] [--voice af_heart.pt]\n"
      "                   [--phonemes \"hɛloʊ wɜːld\"] [--text \"hello\"]\n"
      "                   [--speed 1.0] [--seed 42] [--out kokoro_out.wav]\n"
      "                   [--tiny]\n"
      "\n"
      "  --weights  torch.save Kokoro-82M checkpoint (default kokoro-v1_0.pth;\n"
      "             `hf download hexgrad/Kokoro-82M kokoro-v1_0.pth`)\n"
      "  --voice    voice pack (default af_heart.pt;\n"
      "             `hf download hexgrad/Kokoro-82M voices/af_heart.pt`)\n"
      "  --phonemes raw IPA string (primary input; KModel is language-blind)\n"
      "  --text     naive fallback: vocab chars mapped directly, no G2P\n"
      "  --speed    duration divisor (default 1.0)\n"
      "  --seed     SineGen RNG seed (default 42; audio differs from torch\n"
      "             by RNG stream, deterministic given seed)\n"
      "  --out      WAV path (PCM16 peak-normalized)\n"
      "  --tiny     random-weights smoke model (no files)\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "kokoro-v1_0.pth", voice = "af_heart.pt";
  std::string phonemes = "hɛloʊ wɜːld", text, out = "kokoro_out.wav";
  float speed = 1.0f;
  uint64_t seed = 42;
  bool tiny = false, have_phonemes = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](std::string& dst) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
        return false;
      }
      dst = argv[++i];
      return true;
    };
    if (a == "--weights") {
      if (!need(weights)) return 1;
    } else if (a == "--voice") {
      if (!need(voice)) return 1;
    } else if (a == "--phonemes" && i + 1 < argc) {
      phonemes = argv[++i];
      have_phonemes = true;
    } else if (a == "--text" && i + 1 < argc) {
      text = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--speed" && i + 1 < argc) {
      speed = std::stof(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      seed = (uint64_t)std::stoull(argv[++i]);
    } else if (a == "--tiny") {
      tiny = true;
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "warning: ignoring unknown arg '%s'\n", a.c_str());
    }
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    KokkosDNN::print_configuration(std::cout);
    KK::Config cfg = tiny ? KK::tiny_test_config() : KK::Config();
    std::printf("[model %s]\n", tiny ? "tiny_test" : "kokoro-82M");

    // Vocab map (only needed for real configs; tiny uses raw ids).
    std::map<std::string, int> vocab;
    for (auto& kv : KK::Config::vocab_table()) vocab[kv.first] = kv.second;

    std::vector<int> ids;
    size_t ncp = 0;
    if (!text.empty() && !have_phonemes) {
      std::printf("[text \"%s\"] naive vocab mapping (no G2P)\n",
                  text.c_str());
      std::tie(ids, ncp) = phonemes_to_ids(text, vocab);
    } else {
      if (!have_phonemes)
        std::printf("[default demo phonemes \"%s\"]\n", phonemes.c_str());
      std::tie(ids, ncp) = phonemes_to_ids(phonemes, vocab);
    }
    std::printf("[phonemes %zu code points -> %zu ids]\n", ncp, ids.size());
    if (ids.empty()) throw std::runtime_error("no phoneme ids (empty input?)");
    if (!(speed > 0.0f) || !std::isfinite(speed))
      throw std::runtime_error("--speed must be a positive finite number");

    KK::Weights<Exec, Scalar> w;
    const size_t SD = tiny ? (size_t)cfg.style_dim : 128;
    std::vector<float> style(2 * SD, 0.0f);
    if (tiny) {
      KK::random_init(cfg, w, seed);
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<float> u(-1.0f, 1.0f);
      for (auto& v : style) v = u(rng);
      ids = {1, 2, 3, 4, 5};
      if ((int)ids.size() > cfg.n_token - 2) ids.resize((size_t)cfg.n_token - 2);
      std::printf("[tiny random weights] %zu ids\n", ids.size());
    } else {
      std::printf("[loading %s]\n", weights.c_str());
      KK::load_pth_weights(cfg, weights, w);
      std::printf("[loading voice %s]\n", voice.c_str());
      KK::VoicePack vp = KK::load_voice(voice);
      const int row = (int)ncp - 1;  // KPipeline.infer: pack[len(ps)-1]
      if (row < 0 || row >= vp.n_rows)
        throw std::runtime_error("voice pack has " +
                                 std::to_string(vp.n_rows) + " rows, need " +
                                 std::to_string(row + 1));
      if (vp.rows.size() != (size_t)vp.n_rows * 2 * SD)
        throw std::runtime_error("voice pack rows are not 256-dim");
      std::copy(vp.rows.begin() + (size_t)row * 2 * SD,
                vp.rows.begin() + (size_t)(row + 1) * 2 * SD, style.begin());
      std::printf("[voice row %d of %d]\n", row, vp.n_rows);
    }

    KK::Model<Exec, Scalar> model(cfg, w);
    KK::Intermediates dbg;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> audio =
        model.synthesize(ids, style, speed, seed, &dbg);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    double mean = 0.0, var = 0.0;
    float mn = audio.empty() ? 0 : audio[0], mx = mn;
    for (float v : audio) {
      mean += v;
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    mean /= (double)std::max<size_t>(audio.size(), 1);
    for (float v : audio) var += (v - mean) * (v - mean);
    var /= (double)std::max<size_t>(audio.size(), 1);
    std::printf("[audio %zu samples @24000Hz = %.2fs] mean=%.6f rms=%.6f "
                "min=%.4f max=%.4f (synth %.2fs)\n",
                audio.size(), audio.size() / 24000.0, mean, std::sqrt(var),
                (double)mn, (double)mx, secs);
    std::printf("[durations");
    for (int d : dbg.pred_dur) {
      std::printf(" %d", d);
    }
    std::printf("] frames=%d audio=%zu (expect %zu)\n", dbg.frames,
                audio.size(), (size_t)dbg.frames * 2 * 300);

    float peak = 0.0f;
    save_wav16(out, audio, 24000, peak);
    std::printf("[wrote %s] peak=%.4f (peak-normalized PCM16)\n", out.c_str(),
                (double)peak);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
