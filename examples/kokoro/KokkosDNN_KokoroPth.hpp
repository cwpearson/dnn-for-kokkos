#ifndef KOKKOSDNN_KOKORO_PTH_HPP
#define KOKKOSDNN_KOKORO_PTH_HPP

// Minimal torch.save reader for the Kokoro example.
//
// kokoro-v1_0.pth is a ZIP archive (stored) with data.pkl (protocol-2 pickle)
// holding {section: OrderedDict} for sections bert/bert_encoder/predictor/
// decoder/text_encoder, each value being an OrderedDict of tensors via
// _rebuild_tensor_v2 + persistent FloatStorage IDs, plus one data/<i> record
// per fp32 storage blob. Voice files (voices/*.pt) hold a single tensor
// ([510,1,256]) with the same storage layout.
//
// API:
//   open_kokoro_pth(path) -> inventory of "section.key" tensors.
//   read_kokoro_storage() -> fp32 blob.
//   open_voice_pt(path)   -> single-tensor inventory ([N,1,256]).
// Supported pickle opcodes mirror the Sana reader (GLOBAL, REDUCE, BUILD,
// BINPERSID, ...); anything else throws. Float32 only.

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Kokoro {

struct KokoroTensorMeta {
  std::string name;  // "section.key" (model) or "tensor" (voice)
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  int64_t storage_offset = 0;
  int storage_id = -1;
  int64_t numel = 0;
  std::string dtype;
};

struct KokoroArchive {
  std::string path;
  std::string prefix;
  std::vector<KokoroTensorMeta> tensors;
  std::map<int, std::pair<uint64_t, uint64_t>> records;
};

KokoroArchive open_kokoro_pth(const std::string& path);
KokoroArchive open_voice_pt(const std::string& path);
int64_t kokoro_total_params(const KokoroArchive& archive);
std::vector<float> read_kokoro_storage(const KokoroArchive& archive,
                                       int storage_id);

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
