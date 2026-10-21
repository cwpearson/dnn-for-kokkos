// Kokoro config: vocab table + tiny preset. See KokkosDNN_KokoroConfig.hpp.

#include "KokkosDNN_KokoroConfig.hpp"

namespace KokkosDNN {
namespace Kokoro {

std::vector<std::pair<std::string, int>> Config::vocab_table() {
  // Ordered (symbol, id) pairs transcribed from the hexgrad/Kokoro-82M
  // config.json vocab (114 entries; ids run to 177 with gaps for symbols
  // the G2P never emits). Symbols are UTF-8; matching is byte-wise, so the
  // driver decodes input code points before lookup.
  return {
#include "KokkosDNN_KokoroVocab.inc"
  };
}

std::vector<std::string> Config::id_to_symbol() {
  std::vector<std::string> out(178, "");
  out[0] = "<pad>";
  for (auto& kv : vocab_table()) {
    if (kv.second >= 0 && (size_t)kv.second < out.size())
      out[(size_t)kv.second] = kv.first;
  }
  return out;
}

Config tiny_test_config() {
  Config c;
  c.n_token = 24;
  c.max_position = 64;
  c.plbert_embed = 16;
  c.plbert_hidden = 48;
  c.plbert_heads = 4;
  c.plbert_intermediate = 64;
  c.plbert_layers = 2;
  c.hidden_dim = 32;
  c.style_dim = 8;
  c.n_layer = 1;
  c.max_dur = 8;
  c.text_kernel = 3;
  c.upsample_initial_channel = 32;
  // Generator geometry (STFT/upsample/stride pattern) mirrors the 82M
  // checkpoint; the tiny preset only shrinks channels. Hardcoded strides in
  // synthesize (noise-conv s6, ups x10/x6, STFT hop 5) assume this geometry.
  c.gen_n_fft = 20;
  c.gen_hop = 5;
  c.decode_channels[0] = 32;
  c.decode_channels[1] = 32;
  c.decode_channels[2] = 32;
  c.decode_channels[3] = 32;  // must equal upsample_initial_channel
  c.decode_upsample_last = true;
  return c;
}

}  // namespace Kokoro
}  // namespace KokkosDNN
