#ifndef KOKKOSDNN_GGUF_HPP
#define KOKKOSDNN_GGUF_HPP

// GGUF container + K-quant weight support (header-only).
//
// Scope: read GGUFv3 files produced by llama.cpp-style quantizers, dequantize
// supported tensor dtypes to FP32 on the host, and run quantized GEMV/GEMM
// against device-resident packed weights with on-the-fly dequantization and
// FP32 accumulation. Portable: TeamPolicy kernels only, no backend ifdefs.
//
// A note on "Q4_K_M": it is a file-level quantization *recipe*, not a ggml
// storage type. Q4_K_M checkpoints store most tensors as Q4_K blocks (a few,
// typically attention V/projections, as Q6_K; norms stay F32). Since this
// module dispatches per-tensor dtype, supporting Q4_K + Q6_K (+ Q8_0, F32,
// F16, BF16) covers Q4_K_M inference. Per-tensor dtypes are read from the
// GGUF tensor infos, so mixed files work without extra plumbing.
//
// GGUF layout assumed (v3, little-endian):
//   header: magic u32 0x46554747 ("GGUF"), version u32 (3),
//           n_tensors u64, n_kv u64
//   metadata: per KV: key string, value-type u32, value
//   tensor infos: per tensor: name string, n_dims u32, dims u64[n_dims]
//           (ggml order: dims[0] is fastest-varying), dtype u32 (ggml_type),
//           offset u64 (bytes from start of the tensor-data section)
//   padding to a 32-byte boundary, then raw tensor data.
// Strings are u64 length + bytes (no NUL). Metadata value types follow
// gguf.h: 0 U8, 1 I8, 2 U16, 3 I16, 4 U32, 5 I32, 6 F32, 7 BOOL, 8 STRING,
// 9 ARRAY (elem-type u32 + len u64 + elements, nestable), 10 U64, 11 I64,
// 12 F64.
//
// Dequantization reference: ggml/src/ggml-quants.c (get_scale_min_k4,
// dequantize_row_q4_K, dequantize_row_q6_K, dequantize_row_q8_0).

#include <Kokkos_Core.hpp>

#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Gguf {

// ---------------------------------------------------------------------------
// ggml dtype enum (values match ggml_type in ggml.h).
// ---------------------------------------------------------------------------
enum class GgmlType : uint32_t {
  F32  = 0,
  F16  = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8   = 24,
  I16  = 25,
  I32  = 26,
  BF16 = 30
};

// Elements per block (ggml_blck_size). Unquantized/integer dtypes: 1.
inline constexpr uint32_t ggml_block_len(GgmlType t) {
  switch (t) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::I8:
    case GgmlType::I16:
    case GgmlType::I32: return 1;
    case GgmlType::Q4_0:
    case GgmlType::Q4_1:
    case GgmlType::Q5_0:
    case GgmlType::Q5_1:
    case GgmlType::Q8_0:
    case GgmlType::Q8_1: return 32;
    case GgmlType::Q2_K:
    case GgmlType::Q3_K:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
    case GgmlType::Q8_K: return 256;
  }
  return 1;  // unreachable; keeps device compilers quiet
}

// Bytes per block (ggml_type_size). Matches ggml-common.h static_asserts:
// Q8_0 = half + 32, Q8_1 = 2*half + 32, Q6_K = half + 16 + 192, etc.
inline constexpr size_t ggml_type_size(GgmlType t) {
  switch (t) {
    case GgmlType::F32: return 4;
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::I16: return 2;
    case GgmlType::I8: return 1;
    case GgmlType::I32: return 4;
    case GgmlType::Q4_0: return 18;
    case GgmlType::Q4_1: return 20;
    case GgmlType::Q5_0: return 22;
    case GgmlType::Q5_1: return 24;
    case GgmlType::Q8_0: return 34;
    case GgmlType::Q8_1: return 36;
    case GgmlType::Q2_K: return 84;
    case GgmlType::Q3_K: return 110;
    case GgmlType::Q4_K: return 144;
    case GgmlType::Q5_K: return 176;
    case GgmlType::Q6_K: return 210;
    case GgmlType::Q8_K: return 292;
  }
  return 0;  // unreachable
}

inline constexpr bool ggml_is_quantized(GgmlType t) {
  return ggml_block_len(t) > 1;
}

inline constexpr bool ggml_is_k_quant(GgmlType t) {
  return t == GgmlType::Q2_K || t == GgmlType::Q3_K ||
         t == GgmlType::Q4_K || t == GgmlType::Q5_K ||
         t == GgmlType::Q6_K || t == GgmlType::Q8_K;
}

// This module's quantized-kernel + host-dequant coverage.
inline constexpr bool gguf_can_compute(GgmlType t) {
  return t == GgmlType::F32 || t == GgmlType::F16 || t == GgmlType::BF16 ||
         t == GgmlType::Q8_0 || t == GgmlType::Q4_K || t == GgmlType::Q6_K;
}

inline std::string ggml_type_name(GgmlType t) {
  switch (t) {
    case GgmlType::F32: return "F32";
    case GgmlType::F16: return "F16";
    case GgmlType::Q4_0: return "Q4_0";
    case GgmlType::Q4_1: return "Q4_1";
    case GgmlType::Q5_0: return "Q5_0";
    case GgmlType::Q5_1: return "Q5_1";
    case GgmlType::Q8_0: return "Q8_0";
    case GgmlType::Q8_1: return "Q8_1";
    case GgmlType::Q2_K: return "Q2_K";
    case GgmlType::Q3_K: return "Q3_K";
    case GgmlType::Q4_K: return "Q4_K";
    case GgmlType::Q5_K: return "Q5_K";
    case GgmlType::Q6_K: return "Q6_K";
    case GgmlType::Q8_K: return "Q8_K";
    case GgmlType::I8: return "I8";
    case GgmlType::I16: return "I16";
    case GgmlType::I32: return "I32";
    case GgmlType::BF16: return "BF16";
  }
  return "UNKNOWN";
}

// Packed payload bytes for one row of `cols` elements.
inline size_t ggml_row_bytes(GgmlType t, size_t cols) {
  const uint32_t bl = ggml_block_len(t);
  if (cols % bl != 0)
    throw std::invalid_argument("ggml_row_bytes: cols " +
                                std::to_string(cols) + " not a multiple of " +
                                std::to_string(bl) + " for " +
                                ggml_type_name(t));
  return (cols / bl) * ggml_type_size(t);
}

// ---------------------------------------------------------------------------
// Half-precision conversions. Device-inline versions (usable in kernels) plus
// thin host wrappers for loader/test code.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION float fp16_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = ((uint32_t)h & 0x7C00u) >> 10;
  uint32_t mant = (uint32_t)h & 0x03FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      // Subnormal: normalize.
      exp = 1;
      while (!(mant & 0x0400u)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FFu;
      f = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);  // inf/nan
  } else {
    f = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

KOKKOS_INLINE_FUNCTION float bf16_to_float(uint16_t h) {
  uint32_t f = (uint32_t)h << 16;
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

inline float fp16_to_float_host(uint16_t h) { return fp16_to_float(h); }
inline float bf16_to_float_host(uint16_t h) { return bf16_to_float(h); }

// ---------------------------------------------------------------------------
// Packed quant blocks (field order matches ggml-common.h).
// ---------------------------------------------------------------------------
struct BlockQ4K {
  uint16_t d;          // superblock scale (fp16)
  uint16_t dmin;       // superblock min (fp16, subtracted)
  uint8_t scales[12];  // 8x (6-bit scale, 6-bit min) packed
  uint8_t qs[128];     // 256x 4-bit quants (low nibble first per pair)
};
static_assert(sizeof(BlockQ4K) == 144, "BlockQ4K must be 144 bytes");

struct BlockQ80 {
  uint16_t d;      // block scale (fp16)
  int8_t qs[32];   // quants
};
static_assert(sizeof(BlockQ80) == 34, "BlockQ80 must be 34 bytes");

struct BlockQ6K {
  uint8_t ql[128];    // low 4 bits of each 6-bit quant
  uint8_t qh[64];     // high 2 bits, packed 4 per byte
  int8_t scales[16];  // per-16-element scales (8 per 128-chunk)
  uint16_t d;         // superblock scale (fp16)
};
static_assert(sizeof(BlockQ6K) == 210, "BlockQ6K must be 210 bytes");

// Scale/min unpacking for Q4_K (and Q5_K): 8 sub-blocks of 32 elements share
// 12 bytes. j<4: 6-bit values straight out of bytes j/j+4; j>=4: 4-bit nibble
// of bytes j+4 plus the top 2 bits stashed in bytes j-4/j. Verbatim ggml.
KOKKOS_INLINE_FUNCTION void get_scale_min_k4(int j, const uint8_t* scales,
                                             uint8_t* sc, uint8_t* mn) {
  if (j < 4) {
    *sc = scales[j] & 63;
    *mn = scales[j + 4] & 63;
  } else {
    *sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
    *mn = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
  }
}

// Full-block dequantizers (host- and device-callable). `out` sizes: 256 for
// Q4_K/Q6_K, 32 for Q8_0. Layout logic mirrors ggml's dequantize_row_*.
KOKKOS_INLINE_FUNCTION void dequant_q4k_block(const BlockQ4K* b, float* out) {
  const float dall = fp16_to_float(b->d);
  const float dmin = -fp16_to_float(b->dmin);
  const uint8_t* q = b->qs;
  // ggml walks the 256 values in four 64-element groups: low nibbles of a
  // 32-byte q chunk use sub-block scale 2*j, high nibbles use 2*j+1 (NOT
  // j / j+4; the 6-bit scale bytes are laid out for that pairing).
  for (int j = 0; j < 4; ++j) {
    uint8_t sc, mn;
    get_scale_min_k4(2 * j, b->scales, &sc, &mn);
    const float d1 = dall * (float)sc;
    const float m1 = dmin * (float)mn;
    get_scale_min_k4(2 * j + 1, b->scales, &sc, &mn);
    const float d2 = dall * (float)sc;
    const float m2 = dmin * (float)mn;
    for (int l = 0; l < 32; ++l) {
      out[l]      = d1 * (float)(q[l] & 0xF) + m1;
      out[l + 32] = d2 * (float)(q[l] >> 4) + m2;
    }
    out += 64;
    q += 32;
  }
}

KOKKOS_INLINE_FUNCTION void dequant_q80_block(const BlockQ80* b, float* out) {
  const float d = fp16_to_float(b->d);
  for (int i = 0; i < 32; ++i) out[i] = d * (float)b->qs[i];
}

KOKKOS_INLINE_FUNCTION void dequant_q6k_block(const BlockQ6K* b, float* out) {
  const float d      = fp16_to_float(b->d);
  const uint8_t* ql  = b->ql;
  const uint8_t* qh  = b->qh;
  const int8_t* sc   = b->scales;
  for (int n = 0; n < 256; n += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = (int)(int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int q2 =
          (int)(int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 =
          (int)(int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 =
          (int)(int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      out[l]      = d * (float)sc[is + 0] * (float)q1;
      out[l + 32] = d * (float)sc[is + 2] * (float)q2;
      out[l + 64] = d * (float)sc[is + 4] * (float)q3;
      out[l + 96] = d * (float)sc[is + 6] * (float)q4;
    }
    out += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

// ---------------------------------------------------------------------------
// Tensor metadata.
// ---------------------------------------------------------------------------
struct GgufTensorInfo {
  std::string name;
  uint32_t n_dims             = 0;
  std::vector<uint64_t> dims;  // ggml order: dims[0] fastest-varying
  GgmlType dtype              = GgmlType::F32;
  uint64_t offset             = 0;  // bytes from start of data section
  uint64_t nbytes             = 0;  // packed payload size on disk
  uint64_t numel              = 0;

  // Row-major matrix view of the tensor: a 2-D [rows, cols] tensor is stored
  // in ggml order as dims = {cols, rows}; a 1-D tensor is one row.
  size_t rows() const {
    if (n_dims == 0) return 0;
    if (n_dims == 1) return 1;
    return (size_t)dims[1];
  }
  size_t cols() const {
    if (n_dims == 0) return 0;
    return (size_t)dims[0];
  }
};

// ---------------------------------------------------------------------------
// Host-side GGUF file reader. Loads the file into memory, parses header,
// string metadata, and tensor infos; exposes bounds-checked payload access.
// ---------------------------------------------------------------------------
class GgufFile {
 public:
  static constexpr uint32_t kMagic   = 0x46554747u;  // "GGUF" little-endian
  static constexpr uint32_t kVersion = 3;
  static constexpr size_t kAlign     = 32;

  explicit GgufFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("GgufFile: cannot open '" + path + "'");
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    if (n < 0) throw std::runtime_error("GgufFile: cannot stat '" + path + "'");
    if (n < 24)
      throw std::runtime_error("GgufFile: file too small '" + path + "'");
    f.seekg(0, std::ios::beg);
    buf_.resize((size_t)n);
    f.read(buf_.data(), n);
    if (!f) throw std::runtime_error("GgufFile: short read '" + path + "'");
    parse();
  }

  bool contains(const std::string& name) const {
    return index_.count(name) != 0;
  }

  const GgufTensorInfo& find(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end())
      throw std::runtime_error("GgufFile: tensor not found '" + name + "'");
    return tensors_[it->second];
  }

  bool has_kv(const std::string& key) const {
    return str_kv_.count(key) != 0;
  }

  std::string kv_string(const std::string& key) const {
    auto it = str_kv_.find(key);
    if (it == str_kv_.end())
      throw std::runtime_error("GgufFile: string KV not found '" + key + "'");
    return it->second;
  }

  const std::map<std::string, std::string>& kv_strings() const {
    return str_kv_;
  }

  // Bounds-checked pointer to a tensor's packed payload.
  const uint8_t* payload(const GgufTensorInfo& info) const {
    const size_t abs = data_start_ + (size_t)info.offset;
    if (abs + info.nbytes > buf_.size())
      throw std::runtime_error("GgufFile: tensor '" + info.name +
                               "' payload out of range");
    return reinterpret_cast<const uint8_t*>(buf_.data() + abs);
  }

  uint32_t version() const { return version_; }
  const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
  size_t data_start() const { return data_start_; }
  size_t file_size() const { return buf_.size(); }

 private:
  // Little-endian cursor with bounds checks (GGUF is LE-only, as is ggml).
  struct Cursor {
    const char* p;
    const char* end;
    void need(size_t n, const char* what) const {
      if ((size_t)(end - p) < n)
        throw std::runtime_error(std::string("GgufFile: truncated ") + what);
    }
    uint64_t u64() {
      need(8, "u64");
      uint64_t v = 0;
      for (int i = 7; i >= 0; --i) v = (v << 8) | (uint8_t)p[i];
      p += 8;
      return v;
    }
    uint32_t u32() {
      need(4, "u32");
      uint32_t v = 0;
      for (int i = 3; i >= 0; --i) v = (v << 8) | (uint8_t)p[i];
      p += 4;
      return v;
    }
    std::string str() {
      uint64_t n = u64();
      need((size_t)n, "string");
      std::string s(p, (size_t)n);
      p += n;
      return s;
    }
  };

  // Skip one metadata value of `type` (arrays recurse); store strings.
  void parse_value(Cursor& c, uint32_t type, const std::string& key,
                   bool store) {
    switch (type) {
      case 0: c.need(1, "u8"); c.p += 1; return;    // UINT8
      case 1: c.need(1, "i8"); c.p += 1; return;    // INT8
      case 2: c.need(2, "u16"); c.p += 2; return;   // UINT16
      case 3: c.need(2, "i16"); c.p += 2; return;   // INT16
      case 4: c.need(4, "u32"); c.p += 4; return;   // UINT32
      case 5: c.need(4, "i32"); c.p += 4; return;   // INT32
      case 6: c.need(4, "f32"); c.p += 4; return;   // FLOAT32
      case 7: c.need(1, "bool"); c.p += 1; return;  // BOOL
      case 8: {                                     // STRING
        std::string s = c.str();
        if (store) str_kv_[key] = s;
        return;
      }
      case 9: {  // ARRAY
        uint32_t et = c.u32();
        uint64_t n  = c.u64();
        for (uint64_t i = 0; i < n; ++i) parse_value(c, et, key, false);
        return;
      }
      case 10: c.need(8, "u64"); c.p += 8; return;  // UINT64
      case 11: c.need(8, "i64"); c.p += 8; return;  // INT64
      case 12: c.need(8, "f64"); c.p += 8; return;  // FLOAT64
      default: throw std::runtime_error("GgufFile: bad metadata type " +
                                        std::to_string(type));
    }
  }

  void parse() {
    Cursor c{buf_.data(), buf_.data() + buf_.size()};
    if (buf_[0] != 'G' || buf_[1] != 'G' || buf_[2] != 'U' ||
        buf_[3] != 'F')
      throw std::runtime_error("GgufFile: bad magic (not a GGUF file)");
    c.p += 4;
    version_ = c.u32();
    if (version_ != kVersion)
      throw std::runtime_error("GgufFile: unsupported version " +
                               std::to_string(version_) + " (want 3)");
    const uint64_t n_tensors = c.u64();
    const uint64_t n_kv      = c.u64();

    for (uint64_t i = 0; i < n_kv; ++i) {
      std::string key  = c.str();
      uint32_t type    = c.u32();
      parse_value(c, type, key, true);
    }

    tensors_.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
      GgufTensorInfo ti;
      ti.name             = c.str();
      ti.n_dims           = c.u32();
      if (ti.n_dims == 0 || ti.n_dims > 4)
        throw std::runtime_error("GgufFile: bad n_dims " +
                                 std::to_string(ti.n_dims) + " for '" +
                                 ti.name + "'");
      ti.dims.resize(ti.n_dims);
      ti.numel = 1;
      for (uint32_t d = 0; d < ti.n_dims; ++d) {
        ti.dims[d] = c.u64();
        ti.numel *= ti.dims[d];
      }
      const uint32_t raw_type = c.u32();
      ti.dtype                = (GgmlType)raw_type;
      // Reject unknown ggml type ids (ggml_type_size is 0 for them).
      if (ggml_type_size(ti.dtype) == 0)
        throw std::runtime_error("GgufFile: tensor '" + ti.name +
                                 "' has unknown dtype " +
                                 std::to_string(raw_type));
      const uint32_t bl = ggml_block_len(ti.dtype);
      if (ti.numel % bl != 0)
        throw std::runtime_error("GgufFile: tensor '" + ti.name +
                                 "' numel not a multiple of block length");
      ti.nbytes = (ti.numel / bl) * ggml_type_size(ti.dtype);
      ti.offset = c.u64();
      index_[ti.name] = tensors_.size();
      tensors_.push_back(std::move(ti));
    }

    const size_t off = (size_t)(c.p - buf_.data());
    data_start_      = (off + kAlign - 1) & ~(kAlign - 1);
    if (data_start_ > buf_.size())
      throw std::runtime_error("GgufFile: header runs past EOF");
    // Validate every payload range up front so later access cannot OOB.
    for (const auto& ti : tensors_) {
      const size_t abs = data_start_ + (size_t)ti.offset;
      if (abs + ti.nbytes > buf_.size())
        throw std::runtime_error("GgufFile: tensor '" + ti.name +
                                 "' payload out of range");
    }
  }

  std::vector<char> buf_;
  uint32_t version_ = 0;
  size_t data_start_ = 0;
  std::vector<GgufTensorInfo> tensors_;
  std::map<std::string, size_t> index_;
  std::map<std::string, std::string> str_kv_;
};

// Dequantize a whole tensor to FP32 (streamed block by block; peak extra RAM
// is one 256-float scratch, not a second copy of the tensor).
inline void dequant_tensor_to_float(const GgufFile& file,
                                    const GgufTensorInfo& info,
                                    std::vector<float>& out) {
  out.resize((size_t)info.numel);
  if (info.numel == 0) return;
  const uint8_t* src = file.payload(info);
  float* dst         = out.data();
  switch (info.dtype) {
    case GgmlType::F32: {
      std::memcpy(dst, src, out.size() * 4);
      return;
    }
    case GgmlType::F16: {
      const uint16_t* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < out.size(); ++i)
        dst[i] = fp16_to_float_host(h[i]);
      return;
    }
    case GgmlType::BF16: {
      const uint16_t* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < out.size(); ++i)
        dst[i] = bf16_to_float_host(h[i]);
      return;
    }
    case GgmlType::Q8_0: {
      float tmp[32];
      for (uint64_t b = 0, n = info.numel / 32; b < n; ++b) {
        BlockQ80 blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        dequant_q80_block(&blk, tmp);
        std::memcpy(dst + b * 32, tmp, sizeof(tmp));
      }
      return;
    }
    case GgmlType::Q4_K: {
      float tmp[256];
      for (uint64_t b = 0, n = info.numel / 256; b < n; ++b) {
        BlockQ4K blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        dequant_q4k_block(&blk, tmp);
        std::memcpy(dst + b * 256, tmp, sizeof(tmp));
      }
      return;
    }
    case GgmlType::Q6_K: {
      float tmp[256];
      for (uint64_t b = 0, n = info.numel / 256; b < n; ++b) {
        BlockQ6K blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        dequant_q6k_block(&blk, tmp);
        std::memcpy(dst + b * 256, tmp, sizeof(tmp));
      }
      return;
    }
    default:
      throw std::runtime_error("dequant_tensor_to_float: unsupported dtype " +
                               ggml_type_name(info.dtype));
  }
}

// ---------------------------------------------------------------------------
// Device-side quantized weights + GEMV/GEMM.
// ---------------------------------------------------------------------------
template <class ExecSpace>
struct QuantWeight {
  Kokkos::View<uint8_t*, ExecSpace> data;  // packed rows, ggml block order
  GgmlType dtype = GgmlType::F32;
  size_t rows    = 0;  // N (out_features)
  size_t cols    = 0;  // K (in_features)

  size_t row_bytes() const { return ggml_row_bytes(dtype, cols); }
  size_t bytes() const { return rows * row_bytes(); }
};

// Copy one 1-D/2-D tensor's packed payload to device. 2-D ggml dims are
// {cols, rows}, matching a row-major [rows, cols] weight matrix.
template <class ExecSpace>
QuantWeight<ExecSpace> load_quant_weight(const GgufFile& file,
                                         const GgufTensorInfo& info) {
  if (info.n_dims != 1 && info.n_dims != 2)
    throw std::runtime_error("load_quant_weight: only 1-D/2-D supported");
  if (!gguf_can_compute(info.dtype))
    throw std::runtime_error("load_quant_weight: unsupported dtype " +
                             ggml_type_name(info.dtype));
  QuantWeight<ExecSpace> w;
  w.dtype = info.dtype;
  w.rows  = info.rows();
  w.cols  = info.cols();
  if (w.rows * w.row_bytes() != info.nbytes)
    throw std::runtime_error("load_quant_weight: size mismatch for '" +
                             info.name + "'");
  w.data = Kokkos::View<uint8_t*, ExecSpace>("Gguf::weight", info.nbytes);
  Kokkos::View<uint8_t*, Kokkos::HostSpace> h("Gguf::weight_host",
                                              info.nbytes);
  std::memcpy(h.data(), file.payload(info), (size_t)info.nbytes);
  Kokkos::deep_copy(w.data, h);
  return w;
}

namespace Impl {

// Single-element decode used by the GEMV/GEMM kernels (device-callable).
// Supports the gguf_can_compute() dtype set; other dtypes are rejected on
// the host before launch.
KOKKOS_INLINE_FUNCTION float quant_elem(GgmlType dtype, const uint8_t* row,
                                        size_t k) {
  switch (dtype) {
    case GgmlType::F32: return reinterpret_cast<const float*>(row)[k];
    case GgmlType::F16:
      return fp16_to_float(reinterpret_cast<const uint16_t*>(row)[k]);
    case GgmlType::BF16:
      return bf16_to_float(reinterpret_cast<const uint16_t*>(row)[k]);
    case GgmlType::Q8_0: {
      const BlockQ80* b =
          reinterpret_cast<const BlockQ80*>(row + (k / 32) * sizeof(BlockQ80));
      return fp16_to_float(b->d) * (float)b->qs[k % 32];
    }
    case GgmlType::Q4_K: {
      const BlockQ4K* b =
          reinterpret_cast<const BlockQ4K*>(row + (k / 256) * sizeof(BlockQ4K));
      const size_t l    = k % 256;
      const size_t iter  = l / 64;
      const size_t pos   = l % 64;
      const uint8_t q    = b->qs[iter * 32 + (pos % 32)];
      const uint8_t nib  = (pos < 32) ? (uint8_t)(q & 0xF) : (uint8_t)(q >> 4);
      const int grp      = (pos < 32) ? (int)(2 * iter) : (int)(2 * iter + 1);
      uint8_t sc, mn;
      get_scale_min_k4(grp, b->scales, &sc, &mn);
      return fp16_to_float(b->d) * (float)sc * (float)nib +
             (-fp16_to_float(b->dmin)) * (float)mn;
    }
    case GgmlType::Q6_K: {
      const BlockQ6K* b =
          reinterpret_cast<const BlockQ6K*>(row + (k / 256) * sizeof(BlockQ6K));
      const size_t l     = k % 256;
      const size_t chunk = l / 128;
      const size_t r     = l % 128;
      const size_t sub   = r / 32;
      const size_t ll    = r % 32;
      const size_t is    = ll / 16;
      const uint8_t* ql  = b->ql + chunk * 64;
      const uint8_t* qh  = b->qh + chunk * 32;
      const int8_t* sc   = b->scales + chunk * 8;
      int qv, si;
      if (sub == 0) {
        qv = (int)(int8_t)((ql[ll] & 0xF) | (((qh[ll] >> 0) & 3) << 4)) - 32;
        si = (int)is;
      } else if (sub == 1) {
        qv = (int)(int8_t)((ql[ll + 32] & 0xF) | (((qh[ll] >> 2) & 3) << 4)) -
             32;
        si = (int)is + 2;
      } else if (sub == 2) {
        qv = (int)(int8_t)((ql[ll] >> 4) | (((qh[ll] >> 4) & 3) << 4)) - 32;
        si = (int)is + 4;
      } else {
        qv = (int)(int8_t)((ql[ll + 32] >> 4) | (((qh[ll] >> 6) & 3) << 4)) -
             32;
        si = (int)is + 6;
      }
      return fp16_to_float(b->d) * (float)sc[si] * (float)qv;
    }
    default: return 0.0f;  // unreachable: validated on host
  }
}

template <class ExecSpace, class Scalar>
struct QuantGemvFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar*, ExecSpace> x, y;
  const uint8_t* wdata;
  GgmlType dtype;
  size_t K;
  size_t row_bytes;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const size_t j   = (size_t)team.league_rank();
    const uint8_t* r = wdata + j * row_bytes;
    auto xx          = x;
    const GgmlType dt = dtype;
    float team_sum   = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, (int)K),
        KOKKOS_LAMBDA(int k, float& s) {
          s += quant_elem(dt, r, (size_t)k) * static_cast<float>(xx(k));
        },
        team_sum);
    if (team.team_rank() == 0) y(j) = static_cast<Scalar>(team_sum);
  }
};

template <class ExecSpace, class Scalar>
struct QuantGemmFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> X, Y;
  const uint8_t* wdata;
  GgmlType dtype;
  size_t K, N;
  size_t row_bytes;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const size_t id  = (size_t)team.league_rank();
    const size_t i   = id / N;
    const size_t j   = id % N;
    const uint8_t* r = wdata + j * row_bytes;
    auto XX          = X;
    const GgmlType dt = dtype;
    float team_sum   = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, (int)K),
        KOKKOS_LAMBDA(int k, float& s) {
          s += quant_elem(dt, r, (size_t)k) * static_cast<float>(XX(i, k));
        },
        team_sum);
    if (team.team_rank() == 0) Y(i, j) = static_cast<Scalar>(team_sum);
  }
};

inline void quant_check(GgmlType dtype, size_t rows, size_t cols,
                        size_t have_bytes, const char* what) {
  if (!gguf_can_compute(dtype))
    throw std::invalid_argument(
        std::string(what) + ": unsupported dtype " + ggml_type_name(dtype) +
        " (supported: F32/F16/BF16/Q8_0/Q4_K/Q6_K)");
  const size_t want = rows * ggml_row_bytes(dtype, cols);  // validates cols
  if (have_bytes != want)
    throw std::invalid_argument(std::string(what) + ": payload size " +
                                std::to_string(have_bytes) + " != " +
                                std::to_string(want));
  if (cols > (size_t)INT_MAX)
    throw std::invalid_argument(std::string(what) + ": cols too large");
}

}  // namespace Impl

// y[N] = W[N,K] * x[K] with packed-quant W, FP32 accumulation.
template <class ExecSpace, class Scalar>
void quant_gemv(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar*, ExecSpace>& x, const QuantWeight<ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::Gguf::quant_gemv") {
  (void)handle;  // native kernel only; kept for API symmetry with gemv()
  if (x.extent(0) != w.cols || y.extent(0) != w.rows)
    throw std::invalid_argument("quant_gemv: dimension mismatch");
  Impl::quant_check(w.dtype, w.rows, w.cols, w.data.extent(0), "quant_gemv");
  if (w.rows == 0 || w.cols == 0) return;
  Impl::QuantGemvFunctor<ExecSpace, Scalar> f{x, y, w.data.data(), w.dtype,
                                              w.cols, w.row_bytes()};
  Kokkos::TeamPolicy<ExecSpace> policy((int)w.rows, Kokkos::AUTO);
  Kokkos::parallel_for(label, policy, f);
}

template <class ExecSpace, class Scalar>
void quant_gemv(
    const ExecSpace&, const Kokkos::View<Scalar*, ExecSpace>& x,
    const QuantWeight<ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::Gguf::quant_gemv") {
  DNNHandle<ExecSpace> handle;
  quant_gemv(handle, x, w, y, label);
}

// Y[M,N] = X[M,K] * W[N,K]^T with packed-quant W, FP32 accumulation.
template <class ExecSpace, class Scalar>
void quant_gemm(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const QuantWeight<ExecSpace>& w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "KokkosDNN::Gguf::quant_gemm") {
  (void)handle;
  const size_t M = X.extent(0), K = X.extent(1), N = w.rows;
  if (w.cols != K || Y.extent(0) != M || Y.extent(1) != N)
    throw std::invalid_argument("quant_gemm: dimension mismatch");
  Impl::quant_check(w.dtype, w.rows, w.cols, w.data.extent(0), "quant_gemm");
  if (M == 0 || N == 0 || K == 0) return;
  if (M * N > (size_t)INT_MAX)
    throw std::invalid_argument("quant_gemm: M*N too large");
  Impl::QuantGemmFunctor<ExecSpace, Scalar> f{X, Y, w.data.data(), w.dtype, K,
                                              N, w.row_bytes()};
  Kokkos::TeamPolicy<ExecSpace> policy((int)(M * N), Kokkos::AUTO);
  Kokkos::parallel_for(label, policy, f);
}

template <class ExecSpace, class Scalar>
void quant_gemm(
    const ExecSpace&,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const QuantWeight<ExecSpace>& w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "KokkosDNN::Gguf::quant_gemm") {
  DNNHandle<ExecSpace> handle;
  quant_gemm(handle, X, w, Y, label);
}

// ---------------------------------------------------------------------------
// Synthetic GGUF writer (for tests: build small files without real weights).
// `dims` are in ggml order; `raw` must hold exactly the packed payload bytes.
// Each tensor's file offset is 32-byte aligned; the data section starts
// 32-aligned after the tensor infos.
// ---------------------------------------------------------------------------
struct SyntheticTensor {
  std::string name;
  GgmlType dtype = GgmlType::F32;
  std::vector<uint64_t> dims;  // ggml order
  std::vector<uint8_t> raw;    // packed payload
};

inline void write_synthetic_gguf(
    const std::string& path, const std::vector<SyntheticTensor>& tensors,
    const std::map<std::string, std::string>& string_kv = {}) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("write_synthetic_gguf: cannot open '" +
                                   path + "'");
  auto put = [&](const void* p, size_t n) {
    f.write((const char*)p, (std::streamsize)n);
  };
  auto u32 = [&](uint32_t v) { put(&v, 4); };
  auto u64 = [&](uint64_t v) { put(&v, 8); };
  auto str = [&](const std::string& s) {
    u64(s.size());
    put(s.data(), s.size());
  };
  // Header (magic bytes written explicitly; endian-independent).
  const char magic[4] = {'G', 'G', 'U', 'F'};
  put(magic, 4);
  u32(GgufFile::kVersion);
  u64(tensors.size());
  u64(string_kv.size());
  for (const auto& [k, v] : string_kv) {
    str(k);
    u32(8);  // GGUF_TYPE_STRING
    str(v);
  }
  // Tensor infos + expected payload sizes.
  std::vector<uint64_t> nbytes;
  nbytes.reserve(tensors.size());
  for (const auto& t : tensors) {
    if (t.dims.empty() || t.dims.size() > 4)
      throw std::invalid_argument("write_synthetic_gguf: bad dims for '" +
                                  t.name + "'");
    uint64_t numel = 1;
    for (auto d : t.dims) numel *= d;
    const uint32_t bl = ggml_block_len(t.dtype);
    if (numel % bl != 0)
      throw std::invalid_argument(
          "write_synthetic_gguf: numel not a multiple of block length for '" +
          t.name + "'");
    const uint64_t nb = (numel / bl) * ggml_type_size(t.dtype);
    if (t.raw.size() != nb)
      throw std::invalid_argument(
          "write_synthetic_gguf: raw size " + std::to_string(t.raw.size()) +
          " != expected " + std::to_string(nb) + " for '" + t.name + "'");
    nbytes.push_back(nb);
    str(t.name);
    u32((uint32_t)t.dims.size());
    for (auto d : t.dims) u64(d);
    u32((uint32_t)t.dtype);
    // Offset: filled in below (offsets must precede data, so buffer them).
    u64(0);  // placeholder
  }
  // Patch offsets: data starts 32-aligned after infos; each tensor 32-aligned.
  const size_t infos_end = (size_t)f.tellp();
  const size_t data_start =
      (infos_end + GgufFile::kAlign - 1) & ~(GgufFile::kAlign - 1);
  // Revisit offset fields: their file positions were recorded implicitly;
  // simplest is to recompute: header(24) + KVs + per-tensor info sizes.
  // Instead of seeking back, require no padding ambiguity by construction:
  // offsets are assigned sequentially here and written by seeking.
  size_t off_pos = 24;
  for (const auto& [k, v] : string_kv)
    off_pos += 8 + k.size() + 4 + 8 + v.size();
  std::vector<size_t> off_fields;
  for (const auto& t : tensors) {
    size_t p = off_pos + 8 + t.name.size() + 4 + 8 * t.dims.size() + 4;
    off_fields.push_back(p);
    off_pos = p + 8;
  }
  uint64_t cur = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    cur = (cur + GgufFile::kAlign - 1) & ~(GgufFile::kAlign - 1);
    f.seekp((std::streamoff)off_fields[i]);
    u64(cur);
    cur += nbytes[i];
  }
  // Data section.
  f.seekp((std::streamoff)data_start);
  cur = 0;
  const char pad[32] = {};
  for (size_t i = 0; i < tensors.size(); ++i) {
    const uint64_t want = (cur + GgufFile::kAlign - 1) & ~(GgufFile::kAlign - 1);
    if (want > cur) {
      put(pad, (size_t)(want - cur));
      cur = want;
    }
    put(tensors[i].raw.data(), tensors[i].raw.size());
    cur += nbytes[i];
  }
  f.flush();
  if (!f) throw std::runtime_error("write_synthetic_gguf: write failed '" +
                                   path + "'");
}

}  // namespace Gguf
}  // namespace KokkosDNN

#endif
