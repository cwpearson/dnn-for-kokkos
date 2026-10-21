#ifndef KOKKOSDNN_KOKORO_CONFIG_HPP
#define KOKKOSDNN_KOKORO_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Kokoro {

// Kokoro-82M (StyleTTS2 + ISTFTNet) inference config. Faithful copy of the
// HF config.json values plus PoC knobs (max sampling length). All inference
// is batch-1, float32.
struct Config {
  // Model dims (from config.json).
  int hidden_dim = 512;
  int style_dim = 128;
  int n_layer = 3;
  int n_token = 178;
  int text_encoder_kernel_size = 5;
  int max_dur = 50;
  int n_mels = 80;
  int max_conv_dim = 512;
  int dim_in = 64;
  bool multispeaker = true;
  float dropout = 0.2f;
  // PLBERT.
  int plbert_hidden_size = 768;
  int plbert_heads = 12;
  int plbert_intermediate = 2048;
  int plbert_layers = 12;
  int plbert_max_pos = 512;
  float plbert_dropout = 0.1f;
  // ISTFTNet.
  int upsample_initial_channel = 512;
  int gen_istft_n_fft = 20;
  int gen_istft_hop = 5;
  // PoC knobs.
  int max_tokens = 512;     // input phoneme tokens (incl. 2 boundary)
  int max_frames = 4096;    // decoder frames (duration-regulated length)
  int max_samples = 1 << 20;  // waveform cap (24kHz; ~43s at 1M)
  float sample_rate = 24000.0f;

  void validate() const {
    if (hidden_dim <= 0 || style_dim <= 0 || n_layer <= 0 || n_token <= 0 ||
        plbert_hidden_size <= 0 || plbert_heads <= 0 ||
        plbert_intermediate <= 0 || plbert_layers <= 0 ||
        gen_istft_n_fft <= 0 || gen_istft_hop <= 0 || max_tokens <= 2)
      throw std::invalid_argument("Kokoro Config: bad dims");
    if (plbert_hidden_size % plbert_heads != 0)
      throw std::invalid_argument("Kokoro Config: heads must divide hidden");
  }
};

// Vocab: phoneme string (UTF-8 char) -> id, from config.json "vocab".
// Token 0 is the boundary/pad id (not in the map).
using Vocab = std::map<std::string, int>;

// Load config + vocab from the HF config.json copy. Throws on I/O or parse
// errors. Minimal JSON scanner (no new dependency): reads the known keys
// and the flat "vocab" string->int map (handles \uXXXX escapes).
Config load_config(const std::string& path, Vocab& vocab);

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
