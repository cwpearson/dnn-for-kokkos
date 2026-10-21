// Kokoro TTS driver: phonemes -> 24kHz WAV via the native Kokkos pipeline.
//
// Usage:
//   kokoro_tts --weights kokoro-v1_0.pth --voice voices/af_heart.pt \
//     --config config-kokoro-v1_0.json --phonemes "hEl@U" --output out.wav
//   kokoro_tts --tiny --phonemes "hello" --output tiny.wav
//
// Notes:
// - Pass IPA phonemes directly (--phonemes). Unknown characters are dropped;
//   boundary 0s are added automatically. Proper G2P (misaki/espeak) is out
//   of scope; for English, plain lowercase text is accepted but will sound
//   mispronounced (each letter mapped through the vocab).
// - Voice ref vector: row clamp(len(phonemes)-1) of the [N,1,256] voice pack,
//   matching KPipeline.infer (pack[len(ps)-1]).
// - Deterministic: harmonic phase accumulation starts at zero and SineGen/
//   noise draws use a fixed seed, so output differs stochastically from
//   torch but is reproducible.
// - Tiny mode: small random model, no downloads (also the unit-test path).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroModel.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

namespace {
void write_wav(const std::string& path, const std::vector<float>& pcm,
               int sample_rate = 24000) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open output " + path);
  const uint32_t n = (uint32_t)pcm.size();
  const uint32_t data_bytes = n * 2;
  f.write("RIFF", 4);
  uint32_t chunk = 36 + data_bytes;
  f.write((char*)&chunk, 4);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  uint32_t sub1 = 16;
  f.write((char*)&sub1, 4);
  uint16_t audio_fmt = 1;
  f.write((char*)&audio_fmt, 2);
  uint16_t ch = 1;
  f.write((char*)&ch, 2);
  uint32_t sr = (uint32_t)sample_rate;
  f.write((char*)&sr, 4);
  uint32_t br = sr * 2;
  f.write((char*)&br, 4);
  uint16_t ba = 2;
  f.write((char*)&ba, 2);
  uint16_t bps = 16;
  f.write((char*)&bps, 2);
  f.write("data", 4);
  f.write((char*)&data_bytes, 4);
  for (float v : pcm) {
    float c = v;
    if (!(c == c)) c = 0;
    c = std::max(-1.0f, std::min(1.0f, c));
    int16_t s = (int16_t)std::lround(c * 32767.0f);
    f.write((char*)&s, 2);
  }
}

void usage() {
  std::cout
      << "Usage: kokoro_tts [--tiny] [--weights FILE] [--voice FILE]\n"
      << "                  [--config FILE] [--phonemes STR] [--text STR]\n"
      << "                  [--output FILE] [--speed F] [--voice-index I]\n";
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::string weights, voice,
        config = "examples/kokoro/config-kokoro-v1_0.json";
    std::string phonemes, text, output = "kokoro_out.wav";
    float speed = 1.0f;
    bool tiny = false;
    int voice_index = -1;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto need = [&](std::string& dst) {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        dst = argv[++i];
      };
      if (a == "--weights")
        need(weights);
      else if (a == "--voice")
        need(voice);
      else if (a == "--config")
        need(config);
      else if (a == "--phonemes")
        need(phonemes);
      else if (a == "--text")
        need(text);
      else if (a == "--output")
        need(output);
      else if (a == "--tiny")
        tiny = true;
      else if (a == "--speed") {
        std::string s;
        need(s);
        speed = std::stof(s);
      } else if (a == "--voice-index") {
        std::string s;
        need(s);
        voice_index = std::stoi(s);
      } else if (a == "--help" || a == "-h") {
        usage();
        Kokkos::finalize();
        return 0;
      } else {
        throw std::runtime_error("unknown flag " + a);
      }
    }
    if (!text.empty() && phonemes.empty()) phonemes = text;
    if (phonemes.empty()) phonemes = "h\xc9\x9b\xcb\x88lo\xca\x8a";
    using Exec = Kokkos::DefaultExecutionSpace;
    KokkosDNN::DNNHandle<Exec> handle;
    if (tiny) {
      KokkosDNN::Kokoro::Config cfg;
      cfg.hidden_dim = 16;
      cfg.style_dim = 8;
      cfg.plbert_hidden_size = 32;
      cfg.plbert_heads = 4;
      cfg.plbert_intermediate = 64;
      cfg.plbert_layers = 2;
      cfg.upsample_initial_channel = 32;
      cfg.max_tokens = 64;
      cfg.max_frames = 256;
      cfg.max_samples = 1 << 20;
      KokkosDNN::Kokoro::Vocab vocab;
      for (int c = 0; c < 26; ++c)
        vocab[std::string(1, (char)('a' + c))] = 1 + c;
      vocab[" "] = 27;
      vocab["\xc9\x9b"] = 28;
      vocab["\xcb\x88"] = 29;
      vocab["\xca\x8a"] = 30;
      KokkosDNN::Kokoro::Weights<Exec, float> w;
      KokkosDNN::Kokoro::random_init<Exec, float>(cfg, w, 42);
      Kokkos::View<float*, Exec> ref_s("ref", cfg.style_dim);
      Kokkos::deep_copy(ref_s, 0.1f);
      auto out = KokkosDNN::Kokoro::synthesize<Exec>(cfg, handle, w, vocab,
                                                     phonemes, ref_s, speed);
      write_wav(output, out.audio, (int)cfg.sample_rate);
      std::cout << "tiny: phonemes=" << phonemes.size()
                << " frames=" << out.durations.size()
                << " samples=" << out.audio.size() << " -> " << output << "\n";
    } else {
      if (weights.empty() || voice.empty())
        throw std::runtime_error("--weights and --voice are required");
      KokkosDNN::Kokoro::Vocab vocab;
      KokkosDNN::Kokoro::Config cfg =
          KokkosDNN::Kokoro::load_config(config, vocab);
      KokkosDNN::Kokoro::Weights<Exec, float> w;
      KokkosDNN::Kokoro::load_kokoro_pth<Exec, float>(cfg, weights, w);
      int idx =
          voice_index >= 0 ? voice_index : (int)phonemes.size() - 1;
      Kokkos::View<float*, Exec> ref_s;
      KokkosDNN::Kokoro::load_voice_row<Exec, float>(voice, idx, ref_s);
      auto out = KokkosDNN::Kokoro::synthesize<Exec>(cfg, handle, w, vocab,
                                                     phonemes, ref_s, speed);
      write_wav(output, out.audio, (int)cfg.sample_rate);
      std::cout << "wrote " << output << " (" << out.audio.size()
                << " samples @ " << cfg.sample_rate << "Hz)\n";
      std::cout << "durations:";
      for (int d : out.durations) std::cout << " " << d;
      std::cout << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
