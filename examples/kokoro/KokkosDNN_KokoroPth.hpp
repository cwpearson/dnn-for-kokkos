#ifndef KOKKOSDNN_KOKORO_PTH_HPP
#define KOKKOSDNN_KOKORO_PTH_HPP

// Minimal torch.save (.pth/.pt) reader for the Kokoro example.
//
// The `kokoro-v1_0.pth` checkpoint is a ZIP archive (stored, uncompressed)
// with `data.pkl` (protocol-2 pickle of `{'bert': ..., 'bert_encoder': ...,
// 'predictor': ..., 'decoder': ..., 'text_encoder': ...}` where each group
// is an OrderedDict of float32 tensors via `_rebuild_tensor_v2` +
// persistent `('storage', FloatStorage, key, ...)`` storage IDs) and one
// `data/<key>` record per storage blob. Voice packs (`voices/*.pt`) are the
// same layout with a single tensor at top level.
//
// Supported pickle opcodes are the closed set torch.save emits here: PROTO,
// GLOBAL, REDUCE, BUILD, BINPERSID, EMPTY_DICT, EMPTY_TUPLE, TUPLE[N],
// TUPLE1/2/3, MARK, SETITEM[S], BINUNICODE, BININT[1,2], BINPUT/LONG_BINPUT,
// BINGET/LONG_BINGET, NEWFALSE/NEWTRUE, STOP (plus LONG/BINFLOAT defensively).
// Storage dtype must be a FloatStorage variant (the checkpoint is all F32);
// anything else throws.
//
// Rationale (mirrors examples/sana `KokkosDNN_SanaPth.hpp`): the checkpoint
// is ~330 MB of plain float32 blobs with a 94 KiB pickle index. Depending
// on libtorch just to memcpy those blobs would dwarf the example; a voice
// pack is a single 510x1x256 float tensor with the same framing.

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Kokoro {

// One tensor entry: torch._utils._rebuild_tensor_v2(storage,
// storage_offset, size, stride, requires_grad, backward_hooks).
struct PthTensorMeta {
  std::string name;              // dotted group path, e.g. "decoder.module.encode.conv1.weight"
  std::vector<int64_t> shape;    // tensor sizes
  std::vector<int64_t> stride;   // element strides
  int64_t storage_offset = 0;    // elements into the storage record
  std::string storage_key;       // data/<storage_key> record id
  int64_t storage_numel = 0;     // elements in the storage record
  std::string dtype;             // storage type, e.g. "torch.FloatStorage"
  int64_t numel = 0;             // elements (== product(shape))
};

// Parsed archive: ordered tensor inventory + byte ranges of each storage
// record. Tensor bytes stay on disk until read_pth_storage() is called.
struct PthArchive {
  std::string path;
  std::string prefix;  // zip entry prefix, e.g. "model" / "af_heart"
  std::vector<PthTensorMeta> tensors;  // document order
  // storage_key -> (file offset, byte size) of the data/<key> record.
  std::map<std::string, std::pair<uint64_t, uint64_t>> records;
};

// Parse the ZIP index + data.pkl. Throws std::runtime_error on I/O errors
// or an unrecognized layout (not a torch.save archive of dicts/tensors).
PthArchive open_pth(const std::string& path);

// Total parameter count over the inventory (sum of numel).
int64_t pth_total_params(const PthArchive& archive);

// Stream one storage record from disk as float32 (little-endian host
// assumed, matching the archive byteorder marker). Throws if the record is
// missing, short, or not float32.
std::vector<float> read_pth_storage(const PthArchive& archive,
                                    const std::string& storage_key);

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
