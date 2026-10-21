#ifndef KOKKOSDNN_KOKORO_MODEL_HPP
#define KOKKOSDNN_KOKORO_MODEL_HPP

// Kokoro-82M inference (StyleTTS2 + ISTFTNet), batch-1, float.
// See KokkosDNN_KokoroModel.cpp for the pipeline implementation.

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

namespace KokkosDNN {
namespace Kokoro {

struct SynthOutput {
  std::vector<float> audio;  // 24kHz mono PCM in [-1, 1]
  std::vector<int> durations;  // per-token frames (incl. 2 boundary)
};

// Full pipeline: phoneme string -> waveform.
// phonemes: raw IPA string (each UTF-8 char looked up in vocab; unknown
//   chars dropped). ref_s: [256] voice vector. speed: duration scale.
template <class ExecSpace>
SynthOutput synthesize(const Config& cfg,
                       const DNNHandle<ExecSpace>& handle,
                       const Weights<ExecSpace, float>& w,
                       const Vocab& vocab, const std::string& phonemes,
                       const Kokkos::View<float*, ExecSpace>& ref_s,
                       float speed = 1.0f);

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
