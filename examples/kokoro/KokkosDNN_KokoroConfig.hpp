#ifndef KOKKOSDNN_KOKORO_CONFIG_HPP
#define KOKKOSDNN_KOKORO_CONFIG_HPP

// Kokoro-82M TTS configuration (hexgrad/Kokoro-82M, Apache-2.0).
//
// Architecture: StyleTTS2 text-to-speech (phoneme input) + ISTFTNet neural
// vocoder. The reference is the `kokoro` pip package (kokoro/model.py,
// modules.py, istftnet.py, v0.7.x): KModel.forward(phonemes, ref_s, speed).
// All sizes below come from `config.json` in the HF repo; weight shapes
// are asserted by the .pth loader, so a config/weight mismatch fails loudly
// at load time instead of silently miscomputing.
//
// Pipeline (batch size is always 1; T = phoneme tokens incl. BOS/EOS pads):
//   ids [T] -> PL-BERT Albert (128-dim embed, 768 hidden, 12 shared layers)
//     -> Linear 768->512 -> DurationEncoder (3x BiLSTM-640/AdaLN, hidden
//     512) -> LSTM-640 -> duration_proj (50-way) -> pred_dur [T] ->
//     alignment -> en [640,F] -> shared BiLSTM -> F0/N Adain stacks
//     (F0: 512,512 -> up -> 256,256 -> 256,256; N: same) -> F0/N [F'].
//   ids [T] -> TextEncoder (embed 512, 3x Conv1d k=5 + BiLSTM-512) [512,T]
//     -> align -> asr [512,F].
//   F0/N are Conv1d k=3/s=2 downsampled to [1,F]; asr stays [512,F];
//   Decoder: encode AdainResBlk(514->1024) + 4x decode blocks (last ups x2)
//     -> Generator (ups x10/x6, AdaINResBlocks, conv_post -> mag/phase) ->
//     ISTFT (n_fft 20, hop 5, Hann) -> audio [(F'-pad...) samples].
// With the reference durations F' = 2F and audio samples = 300F.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace KokkosDNN {
namespace Kokoro {

// Default compute/storage scalar: float, the kokoro-v1_0.pth dtype.
using DefaultScalar = float;

struct Config {
  // Tokenizer / text side.
  int n_token = 178;      // phoneme vocab (incl. pad id 0)
  int max_position = 512;  // PL-BERT + voice-pack context cap (ids+2 <= 512)
  // PL-BERT (CustomAlbert over transformers Albert, one layer group with a
  // single layer applied num_hidden_layers times).
  int plbert_embed = 128;
  int plbert_hidden = 768;
  int plbert_heads = 12;
  int plbert_intermediate = 2048;
  int plbert_layers = 12;
  float plbert_eps = 1e-12f;
  // Shared trunk.
  int hidden_dim = 512;  // bert_encoder out, text-encoder channels
  int style_dim = 128;   // style vector halves (s = ref_s[:,128:])
  int n_layer = 3;       // DurationEncoder LSTM/AdaLN pairs
  int max_dur = 50;      // duration_proj classes
  int text_kernel = 5;   // TextEncoder CNN kernel (pad 2)
  // ISTFTNet generator / vocoder.
  int upsample_initial_channel = 512;
  int gen_n_fft = 20;
  int gen_hop = 5;
  // Decoder AdainResBlk channel plan: encode in = hidden+2 (asr+F0+N).
  int decode_channels[4] = {1024, 1024, 1024, 512};
  bool decode_upsample_last = true;

  void validate() const {
    if (n_token <= 0 || max_position <= 0 || plbert_hidden <= 0 ||
        hidden_dim <= 0 || style_dim <= 0 || n_layer <= 0 || max_dur <= 0)
      throw std::invalid_argument("Kokoro::Config: bad dims");
    if (plbert_hidden % plbert_heads != 0)
      throw std::invalid_argument("Kokoro::Config: heads must divide hidden");
  }

  // Phoneme vocab: ordered (symbol, id) pairs from config.json, plus the
  // implicit pad id 0 (<bos>/<eos> pads).
  static std::vector<std::pair<std::string, int>> vocab_table();
  // Inverse map id -> symbol (index = id; entry 0 is "<pad>").
  static std::vector<std::string> id_to_symbol();
};

// Tiny smoke-test preset: same code paths, small channels, 1 shared Albert
// layer application. Not a released checkpoint; random_init() fills it.
Config tiny_test_config();

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
