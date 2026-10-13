#ifndef KOKKOSDNN_SANA_PTH_HPP
#define KOKKOSDNN_SANA_PTH_HPP

// Minimal torch.save (.pth) reader for the Sana example.
//
// The `Sana_600M_1024px_MultiLing.pth` checkpoint is a ZIP archive (stored,
// uncompressed) with `data.pkl` (protocol-2 pickle of
// `{'state_dict': OrderedDict[...]}` using `_rebuild_tensor_v2` + persistent
// storage IDs) and one `data/<i>` record per fp32 storage blob. This header
// declares a dependency-free reader for exactly that layout:
//
//   open_pth(path)      parse ZIP central directory + unpickle data.pkl into
//                       an ordered tensor inventory (no tensor bytes loaded).
//   read_pth_storage()  stream one fp32 storage record from disk.
//   load_pth_weights()  fill Sana Weights from the archive, validating every
//                       expected name/shape (throws on any mismatch).
//
// Supported pickle opcodes are the closed set torch.save emits here (PROTO,
// GLOBAL, REDUCE, BUILD, BINPERSID, EMPTY_DICT, EMPTY_TUPLE, TUPLE[N],
// MARK, SETITEM[S], BINUNICODE, BININT[1,2], BINPUT/LONG_BINPUT,
// BINGET/LONG_BINGET, NEWFALSE, STOP); anything else throws. Supported
// storage dtype is float32 only (the checkpoint is all F32).
//
// Rationale for a hand-rolled reader: the checkpoint is ~2.4 GB of plain
// float32 blobs with a 91 KiB pickle index. Depending on libtorch (or
// converting offline) just to memcpy those blobs would dwarf the example.

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN_SanaConfig.hpp"

namespace KokkosDNN {
namespace Sana {

// One state_dict entry: torch._utils._rebuild_tensor_v2(storage,
// storage_offset, size, stride, requires_grad, backward_hooks).
struct PthTensorMeta {
  std::string name;
  std::vector<int64_t> shape;   // tensor sizes
  std::vector<int64_t> stride;  // element strides
  int64_t storage_offset = 0;   // elements into the storage record
  int storage_id = -1;          // data/<storage_id> record
  int64_t numel = 0;            // elements (== product(shape))
  std::string dtype;            // storage type, e.g. "torch.FloatStorage"
};

// Parsed archive: ordered tensor inventory + byte ranges of each storage
// record. Tensor bytes stay on disk until read_pth_storage() is called.
struct PthArchive {
  std::string path;
  std::string prefix;  // zip entry prefix, e.g. "Sana_600M_1024px_MultiLing/"
  std::vector<PthTensorMeta> tensors;  // state_dict order
  // storage_id -> (file offset, byte size) of the data/<id> record.
  std::map<int, std::pair<uint64_t, uint64_t>> records;
  // state_dict _metadata (module prefix -> version); informational only.
  std::map<std::string, int64_t> metadata;
};

// Parse the ZIP index + data.pkl. Throws std::runtime_error on I/O errors
// or an unrecognized layout (not a torch.save state_dict archive).
PthArchive open_pth(const std::string& path);

// Total parameter count over the inventory (sum of numel).
int64_t pth_total_params(const PthArchive& archive);

// Stream one storage record from disk as float32 (little-endian host
// assumed, matching the archive byteorder marker). Throws if the record is
// missing, short, or not float32.
std::vector<float> read_pth_storage(const PthArchive& archive, int storage_id);

namespace Impl {
// (name, shape) of every tensor the Sana Config requires, in load order.
// latent_h/latent_w set the expected pos_embed token count (32x32 = 1024
// for the 1024px checkpoint: latent grid = image / VAE downsample 32).
std::vector<std::pair<std::string, std::vector<int64_t>>> sana_pth_expected(
    const Config& cfg, int latent_h = 32, int latent_w = 32);
}  // namespace Impl

}  // namespace Sana
}  // namespace KokkosDNN

#endif
