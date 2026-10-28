// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
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
// typically attention V/projections, as Q6_K; large vocab/output matrices as
// Q5_K; norms stay F32). Since this module dispatches per-tensor dtype,
// supporting Q4_K + Q5_K + Q6_K (+ Q8_0, F32, F16, BF16) covers Q4_K_M
// inference. Per-tensor dtypes are read from the GGUF tensor infos, so mixed
// files work without extra plumbing.
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
// dequantize_row_q4_K, dequantize_row_q5_K, dequantize_row_q6_K,
// dequantize_row_q8_0).

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
#include "KokkosDNN_Macros.hpp"
#include "KokkosDNN_Occupancy.hpp"
#include "KokkosDNN_SwiGLU.hpp"

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
// Q5_0/Q5_1 (legacy non-K 5-bit, 22/24-byte blocks) are intentionally NOT
// covered: no current recipe emits them (Q4_K_M uses Q5_K for the large
// matrices instead), and their scale/zero-point layout shares nothing with
// the K-quant blocks below. They keep throwing "unsupported dtype" naming
// the dtype (see the dispatch defaults) rather than failing silently.
inline constexpr bool gguf_can_compute(GgmlType t) {
  return t == GgmlType::F32 || t == GgmlType::F16 || t == GgmlType::BF16 ||
         t == GgmlType::Q8_0 || t == GgmlType::Q4_K || t == GgmlType::Q5_K ||
         t == GgmlType::Q6_K;
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

struct BlockQ5K {
  uint16_t d;          // superblock scale (fp16)
  uint16_t dmin;       // superblock min (fp16, subtracted)
  uint8_t scales[12];  // 8x (6-bit scale, 6-bit min), same packing as Q4_K
  uint8_t qh[32];      // high bit of each 5-bit quant (see dequant_q5k_block)
  uint8_t qs[128];     // low 4 bits (nibble per quant, low nibble first)
};
static_assert(sizeof(BlockQ5K) == 176, "BlockQ5K must be 176 bytes");

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

// Q5_K: same 8x32-element sub-blocks and scale/min packing as Q4_K, but each
// quant is 5 bits. The low 4 bits sit in the qs nibbles (low nibble: even
// sub-block 2*iter, high nibble: odd sub-block 2*iter+1 of the 64-element
// group `iter`); the top bit sits in qh. One qh byte covers all four groups
// -- the even/odd sub-block bits of group `iter` are bits 2*iter/2*iter+1 --
// so the select mask shifts per group. Value: d*sc*q - min*mn.
KOKKOS_INLINE_FUNCTION void dequant_q5k_block(const BlockQ5K* b, float* out) {
  const float d   = fp16_to_float(b->d);
  const float min = fp16_to_float(b->dmin);
  const uint8_t* ql = b->qs;
  const uint8_t* qh = b->qh;
  for (int iter = 0; iter < 4; ++iter) {
    uint8_t sc, mn;
    get_scale_min_k4(2 * iter, b->scales, &sc, &mn);
    const float d1 = d * (float)sc;
    const float m1 = min * (float)mn;
    get_scale_min_k4(2 * iter + 1, b->scales, &sc, &mn);
    const float d2 = d * (float)sc;
    const float m2 = min * (float)mn;
    const uint8_t u1 = (uint8_t)(1u << (2 * iter));
    const uint8_t u2 = (uint8_t)(2u << (2 * iter));
    for (int l = 0; l < 32; ++l) {
      out[l]      = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
      out[l + 32] = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
    }
    out += 64;
    ql += 32;
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

  bool has_kv_f32(const std::string& key) const {
    return f32_kv_.count(key) != 0;
  }

  float kv_f32(const std::string& key) const {
    auto it = f32_kv_.find(key);
    if (it == f32_kv_.end())
      throw std::runtime_error("GgufFile: F32 KV not found '" + key + "'");
    return it->second;
  }

  bool has_kv_u32(const std::string& key) const {
    return u32_kv_.count(key) != 0;
  }

  uint32_t kv_u32(const std::string& key) const {
    auto it = u32_kv_.find(key);
    if (it == u32_kv_.end())
      throw std::runtime_error("GgufFile: U32 KV not found '" + key + "'");
    return it->second;
  }

  bool has_kv_bool_array(const std::string& key) const {
    return bool_arr_kv_.count(key) != 0;
  }

  const std::vector<char>& kv_bool_array(const std::string& key) const {
    auto it = bool_arr_kv_.find(key);
    if (it == bool_arr_kv_.end())
      throw std::runtime_error("GgufFile: bool-array KV not found '" + key +
                               "'");
    return it->second;
  }

  // Full typed-KV maps (for suffix-based key lookup when the arch prefix
  // differs from the expected one).
  const std::map<std::string, float>& kv_f32_all() const { return f32_kv_; }
  const std::map<std::string, uint32_t>& kv_u32_all() const {
    return u32_kv_;
  }
  const std::map<std::string, std::vector<char>>& kv_bool_arrays_all() const {
    return bool_arr_kv_;
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

  // Skip one metadata value of `type` (arrays recurse); store strings,
  // float32/uint32 scalars, and bool arrays (the conversion-written
  // hyperparameters loaders validate against, e.g. logit scale).
  void parse_value(Cursor& c, uint32_t type, const std::string& key,
                   bool store) {
    switch (type) {
      case 0: c.need(1, "u8"); c.p += 1; return;    // UINT8
      case 1: c.need(1, "i8"); c.p += 1; return;    // INT8
      case 2: c.need(2, "u16"); c.p += 2; return;   // UINT16
      case 3: c.need(2, "i16"); c.p += 2; return;   // INT16
      case 4: {                                     // UINT32
        const uint32_t v = c.u32();
        if (store) u32_kv_[key] = v;
        return;
      }
      case 5: c.need(4, "i32"); c.p += 4; return;   // INT32
      case 6: {                                     // FLOAT32
        c.need(4, "f32");
        float v = 0;
        std::memcpy(&v, c.p, 4);
        c.p += 4;
        if (store) f32_kv_[key] = v;
        return;
      }
      case 7: c.need(1, "bool"); c.p += 1; return;  // BOOL
      case 8: {                                     // STRING
        std::string s = c.str();
        if (store) str_kv_[key] = s;
        return;
      }
      case 9: {  // ARRAY
        uint32_t et = c.u32();
        uint64_t n  = c.u64();
        if (store && et == 7) {  // BOOL array (e.g. sliding-window pattern)
          c.need((size_t)n, "bool array");
          bool_arr_kv_[key] = std::vector<char>(c.p, c.p + (size_t)n);
          c.p += (size_t)n;
          return;
        }
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
  std::map<std::string, float> f32_kv_;
  std::map<std::string, uint32_t> u32_kv_;
  std::map<std::string, std::vector<char>> bool_arr_kv_;
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
    case GgmlType::Q5_K: {
      float tmp[256];
      for (uint64_t b = 0, n = info.numel / 256; b < n; ++b) {
        BlockQ5K blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        dequant_q5k_block(&blk, tmp);
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
    case GgmlType::Q5_K: {
      const BlockQ5K* b =
          reinterpret_cast<const BlockQ5K*>(row + (k / 256) * sizeof(BlockQ5K));
      const size_t l    = k % 256;
      const size_t sub  = l / 32;  // 0..7: sub-block sharing one scale/min
      const size_t iter = sub / 2;
      const bool low    = (sub % 2) == 0;
      const size_t ll   = l % 32;
      const uint8_t q   = b->qs[iter * 32 + ll];
      const uint8_t nib = low ? (uint8_t)(q & 0xF) : (uint8_t)(q >> 4);
      const uint8_t u   = (uint8_t)((low ? 1u : 2u) << (2 * iter));
      const uint8_t qv  = (uint8_t)(nib + ((b->qh[ll] & u) ? 16 : 0));
      uint8_t sc, mn;
      get_scale_min_k4((int)sub, b->scales, &sc, &mn);
      return fp16_to_float(b->d) * (float)sc * (float)qv -
             fp16_to_float(b->dmin) * (float)mn;
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

// Sub-block grouped dot: GS consecutive packed-row elements starting at k0
// sharing one scale set (Q4_K: 32 sharing one (d*scale, dmin*min) pair;
// Q6_K: 16 sharing one (d, scale); Q8_0: 32 sharing one d). Groups never
// straddle a block (256 % GS == 0 for Q4_K/Q6_K, 32 % 32 == 0 for Q8_0), so
// the block pointer needs one division per group instead of per element.
// Hoists the fp16->float scale converts and the scale/min unpacking that
// quant_elem() repeats per element (32x/16x fewer), leaving one nibble
// extract + FMA per element. Same values up to FP32 reassociation (the
// callers reduce over groups instead of elements). xd/xs cover strided
// activation rows: element k reads xd[k * xs]. Q4_K/Q5_K/Q6_K/Q8_0 only;
// other dtypes stay on quant_elem().
template <class Scalar>
KOKKOS_INLINE_FUNCTION float quant_group_dot(GgmlType dtype,
                                             const uint8_t* row,
                                             const Scalar* xd, size_t xs,
                                             size_t k0) {
  switch (dtype) {
    case GgmlType::Q4_K: {
      const BlockQ4K* b = reinterpret_cast<const BlockQ4K*>(
          row + (k0 / 256) * sizeof(BlockQ4K));
      const size_t sub  = (k0 / 32) % 8;  // 2*iter + (high ? 1 : 0)
      const size_t iter = sub / 2;
      uint8_t sc, mn;
      get_scale_min_k4((int)(2 * iter + (sub % 2)), b->scales, &sc, &mn);
      const float dm = fp16_to_float(b->d) * (float)sc;
      const float mm = -fp16_to_float(b->dmin) * (float)mn;
      const uint8_t* q = b->qs + iter * 32;
      float s          = 0;
      if ((sub % 2) == 0) {
        for (size_t l = 0; l < 32; ++l)
          s += (dm * (float)(q[l] & 0xF) + mm) *
               static_cast<float>(xd[(k0 + l) * xs]);
      } else {
        for (size_t l = 0; l < 32; ++l)
          s += (dm * (float)(q[l] >> 4) + mm) *
               static_cast<float>(xd[(k0 + l) * xs]);
      }
      return s;
    }
    case GgmlType::Q5_K: {
      const BlockQ5K* b = reinterpret_cast<const BlockQ5K*>(
          row + (k0 / 256) * sizeof(BlockQ5K));
      const size_t sub  = (k0 / 32) % 8;
      const size_t iter = sub / 2;
      const bool low    = (sub % 2) == 0;
      uint8_t sc, mn;
      get_scale_min_k4((int)sub, b->scales, &sc, &mn);
      const float dm = fp16_to_float(b->d) * (float)sc;
      const float mm = -fp16_to_float(b->dmin) * (float)mn;
      const uint8_t* q = b->qs + iter * 32;
      const uint8_t u  = (uint8_t)((low ? 1u : 2u) << (2 * iter));
      float s          = 0;
      if (low) {
        for (size_t l = 0; l < 32; ++l) {
          const uint8_t qv =
              (uint8_t)((q[l] & 0xF) + ((b->qh[l] & u) ? 16 : 0));
          s += (dm * (float)qv + mm) * static_cast<float>(xd[(k0 + l) * xs]);
        }
      } else {
        for (size_t l = 0; l < 32; ++l) {
          const uint8_t qv =
              (uint8_t)((q[l] >> 4) + ((b->qh[l] & u) ? 16 : 0));
          s += (dm * (float)qv + mm) * static_cast<float>(xd[(k0 + l) * xs]);
        }
      }
      return s;
    }
    case GgmlType::Q6_K: {
      const BlockQ6K* b = reinterpret_cast<const BlockQ6K*>(
          row + (k0 / 256) * sizeof(BlockQ6K));
      const size_t r     = k0 % 256;
      const size_t chunk = r / 128;
      const size_t rr    = r % 128;
      const size_t sub   = rr / 32;
      const size_t ll    = rr % 32;  // 0 or 16: (ll + l) / 16 is constant
      const uint8_t* ql  = b->ql + chunk * 64;
      const uint8_t* qh  = b->qh + chunk * 32;
      const float ds =
          fp16_to_float(b->d) * (float)b->scales[chunk * 8 + sub * 2 + ll / 16];
      float s = 0;
      if (sub == 0) {
        for (size_t l = 0; l < 16; ++l) {
          const size_t a = ll + l;
          const int qv =
              (int)(int8_t)((ql[a] & 0xF) | (((qh[a] >> 0) & 3) << 4)) - 32;
          s += ds * (float)qv * static_cast<float>(xd[(k0 + l) * xs]);
        }
      } else if (sub == 1) {
        for (size_t l = 0; l < 16; ++l) {
          const size_t a = ll + l;
          const int qv = (int)(int8_t)((ql[a + 32] & 0xF) |
                                       (((qh[a] >> 2) & 3) << 4)) -
                         32;
          s += ds * (float)qv * static_cast<float>(xd[(k0 + l) * xs]);
        }
      } else if (sub == 2) {
        for (size_t l = 0; l < 16; ++l) {
          const size_t a = ll + l;
          const int qv =
              (int)(int8_t)((ql[a] >> 4) | (((qh[a] >> 4) & 3) << 4)) - 32;
          s += ds * (float)qv * static_cast<float>(xd[(k0 + l) * xs]);
        }
      } else {
        for (size_t l = 0; l < 16; ++l) {
          const size_t a = ll + l;
          const int qv = (int)(int8_t)((ql[a + 32] >> 4) |
                                       (((qh[a] >> 6) & 3) << 4)) -
                         32;
          s += ds * (float)qv * static_cast<float>(xd[(k0 + l) * xs]);
        }
      }
      return s;
    }
    case GgmlType::Q8_0: {
      const BlockQ80* b = reinterpret_cast<const BlockQ80*>(
          row + (k0 / 32) * sizeof(BlockQ80));
      const float d = fp16_to_float(b->d);
      float s       = 0;
      for (size_t l = 0; l < 32; ++l)
        s += d * (float)b->qs[l] * static_cast<float>(xd[(k0 + l) * xs]);
      return s;
    }
    default: return 0.0f;  // unreachable: grouped path is Q4_K/Q5_K/Q6_K/Q8_0 only
  }
}

// Grouped-path eligibility: the sub-block size GS divides K (groups are
// block-aligned, so they never straddle a block). F32/F16/BF16 stay on the
// per-element path (already one cheap convert or less per element).
inline bool quant_can_group(GgmlType dtype, size_t K, size_t& gs) {
  switch (dtype) {
    case GgmlType::Q4_K: gs = 32; return K % 32 == 0;
    case GgmlType::Q5_K: gs = 32; return K % 32 == 0;
    case GgmlType::Q6_K: gs = 16; return K % 16 == 0;
    case GgmlType::Q8_0: gs = 32; return K % 32 == 0;
    default: return false;
  }
}

// --- decode (M == 1) work units --------------------------------------------
// The sparse-MoE decode kernels below split one packed row into *units* and
// hand each unit to one thread of the row's team, so a team is a reduction
// over a packed row. A unit is 32 elements of one packed row -- the smallest
// span that covers whole quant sub-blocks and can be fetched with wide loads:
//   Q4_K: 16 packed bytes, one aligned 128-bit load (low-nibble sub-block 2i
//         at +0, high-nibble sub-block 2i+1 at +32 elements)
//   Q5_K: 32 qs + 32 qh bytes of one 32-element sub-block (four 128-bit
//         loads; only one nibble side and one qh bit are used)
//   Q6_K: 32 packed bytes, 16 low + 16 high bytes covering the 32-element span
//         at +0 and +64 elements (2-byte loads; see below)
//   Q8_0: one 34-byte block, 2-byte loads (same alignment reason)
//   F32/F16/BF16 and anything else: one element (per-element path).
// Why units and not elements: decode is weight-bandwidth bound, and the
// per-element grouped dot above issues one byte load per weight element plus
// one scalar activation load. Measured on the RTX 3090 with a 1536x2048 Q4_K
// GEMV (1.77 MB payload, L2 defeated), one thread per output row: 54 GB/s for
// the grouped dot vs 250 GB/s for this unit layout (128-bit weight loads +
// 128-bit activation loads), against a 300 GB/s read-only ceiling for the same
// shape. The GEMV was ~60% of decode time, so the load width is the whole game
// at batch 1.
// Same arithmetic per element (dm * q + mm) * x, FP32 accumulation; the only
// change is which thread accumulates which elements, i.e. FP32 reassociation.
//
// Q6_K/Q8_0 blocks are 210/34 bytes, so the packed payload is only 2-byte
// aligned per block (210 and 34 are 2 mod 4, and a row of them inherits that):
// 16-byte vector loads on those weights would fault on odd block indices and
// even 4-byte loads are safe only at the row start. They are fetched as eight
// 2-byte loads instead. That still beats one load per element by 8x on the
// packed side; the bigger win is that the activation side (2 bytes/element
// against ~0.8 packed bytes for Q6_K) becomes 128-bit loads, which is where
// most of the load instructions were.
inline size_t quant_unit_count(GgmlType dtype, size_t cols) {
  switch (dtype) {
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
    case GgmlType::Q8_0: return cols / 32;
    default: return cols;
  }
}

// 16-byte aggregate: nvcc/gcc both emit a single 128-bit load for an
// alignas(16) POD, and unlike uint4 it exists on every backend.
struct alignas(16) QuantU8x16 {
  uint8_t b[16];
};

// 16 packed bytes behind 8 x 2-byte loads, for the 2-byte-aligned payloads
// (Q6_K/Q8_0 blocks). Natural alignment keeps every backend happy: no backend
// gating and no unaligned 32/128-bit loads.
struct QuantU16x8 {
  uint16_t w[8];
};

KOKKOS_INLINE_FUNCTION QuantU16x8 quant_load_u16x8(const uint8_t* p) {
  const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
  QuantU16x8 r;
KOKKOSDNN_PRAGMA_UNROLL()
  for (int i = 0; i < 8; ++i) r.w[i] = q[i];
  return r;
}

KOKKOS_INLINE_FUNCTION uint8_t quant_u16x8_byte(const QuantU16x8& v, int i) {
  return static_cast<uint8_t>((v.w[i >> 1] >> ((i & 1) * 8)) & 0xFFu);
}

// Aligned activation window: 8 Scalars (16 B for bhalf, 32 B for float).
// Requires the activation row pointer to be 16-byte aligned.
template <class Scalar>
struct alignas(16) QuantX8 {
  Scalar v[8];
};

// One Q4_K unit: 16 packed bytes -> 16 low-nibble (sub-block 2*iter) and 16
// high-nibble (sub-block 2*iter+1) elements, each with its own (d*scale,
// -dmin*min) pair. Unit c of a row: block c/8, iter (c%8)/2, half c%2; the
// unit's first element is block*256 + iter*64 + half*16.
template <class Scalar>
KOKKOS_INLINE_FUNCTION float quant_q4k_chunk_dot(const uint8_t* row,
                                                 const Scalar* xd, size_t xs,
                                                 size_t c, bool xvec) {
  const size_t blk  = c / 8;
  const BlockQ4K* b = reinterpret_cast<const BlockQ4K*>(
      row + blk * sizeof(BlockQ4K));
  const size_t iter = (c % 8) / 2;
  const size_t half = c % 2;
  QuantU8x16 q = *reinterpret_cast<const QuantU8x16*>(
      b->qs + iter * 32 + half * 16);
  const size_t k0 = blk * 256 + iter * 64 + half * 16;
  uint8_t sc, mn;
  get_scale_min_k4(static_cast<int>(2 * iter), b->scales, &sc, &mn);
  const float dmA = fp16_to_float(b->d) * (float)sc;
  const float mmA = -fp16_to_float(b->dmin) * (float)mn;
  get_scale_min_k4(static_cast<int>(2 * iter + 1), b->scales, &sc, &mn);
  const float dmB = fp16_to_float(b->d) * (float)sc;
  const float mmB = -fp16_to_float(b->dmin) * (float)mn;
  float sa = 0, sb = 0;
  if (xvec) {
    // Four 128-bit activation loads instead of 32 scalar ones (1.4x on the
    // microbenchmark above).
    const QuantX8<Scalar>* xa =
        reinterpret_cast<const QuantX8<Scalar>*>(xd + k0);
    const QuantX8<Scalar>* xb =
        reinterpret_cast<const QuantX8<Scalar>*>(xd + k0 + 32);
    const QuantX8<Scalar> lo0 = xa[0], lo1 = xa[1];
    const QuantX8<Scalar> hi0 = xb[0], hi1 = xb[1];
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 8; ++i) {
      const float x_lo0 = static_cast<float>(lo0.v[i]);
      const float x_lo1 = static_cast<float>(lo1.v[i]);
      const float x_hi0 = static_cast<float>(hi0.v[i]);
      const float x_hi1 = static_cast<float>(hi1.v[i]);
      sa = fmaf(dmA * (float)(q.b[i] & 0xF) + mmA, x_lo0, sa);
      sa = fmaf(dmA * (float)(q.b[i + 8] & 0xF) + mmA, x_lo1, sa);
      sb = fmaf(dmB * (float)(q.b[i] >> 4) + mmB, x_hi0, sb);
      sb = fmaf(dmB * (float)(q.b[i + 8] >> 4) + mmB, x_hi1, sb);
    }
  } else {
    // Unaligned activation row: same math, scalar loads.
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 16; ++i) {
      sa = fmaf(dmA * (float)(q.b[i] & 0xF) + mmA,
                static_cast<float>(xd[(k0 + i) * xs]), sa);
      sb = fmaf(dmB * (float)(q.b[i] >> 4) + mmB,
                static_cast<float>(xd[(k0 + 32 + i) * xs]), sb);
    }
  }
  return sa + sb;
}

// One Q6_K unit: 32 elements of a 128-element block half. The low/high nibble
// bytes for the two sub-blocks sit at ql[chunk*64 + pair*32 + half*16] and the
// 2-bit highs at qh[chunk*32 + half*16]; the sub-blocks' scale indices are
// (2*pair + half) and (2*pair + half + 4) of the block's 8 scales. `pair`
// selects sub-blocks {0,2} or {1,3}, i.e. element offsets +0 and +64 from the
// unit base; `half` selects the 16-element half. Both windows are 16 elements,
// so all 32 elements of the unit are covered exactly once.
template <class Scalar>
KOKKOS_INLINE_FUNCTION float quant_q6k_chunk_dot(const uint8_t* row,
                                                 const Scalar* xd, size_t xs,
                                                 size_t c, bool xvec) {
  const size_t blk  = c / 8;
  const BlockQ6K* b = reinterpret_cast<const BlockQ6K*>(
      row + blk * sizeof(BlockQ6K));
  const size_t chunk = (c % 8) / 4;
  const size_t pair  = (c % 4) / 2;
  const size_t half  = c % 2;
  const size_t k0    = blk * 256 + chunk * 128 + pair * 32 + half * 16;
  const QuantU16x8 ql =
      quant_load_u16x8(b->ql + chunk * 64 + pair * 32 + half * 16);
  const QuantU16x8 qh = quant_load_u16x8(b->qh + chunk * 32 + half * 16);
  const float dd      = fp16_to_float(b->d);
  const float ds0     = dd * (float)b->scales[chunk * 8 + 2 * pair + half];
  const float ds1 = dd * (float)b->scales[chunk * 8 + 2 * pair + half + 4];
  const int shf   = pair ? 2 : 0;
  float sa = 0, sb = 0;
  if (xvec) {
    const QuantX8<Scalar>* xa =
        reinterpret_cast<const QuantX8<Scalar>*>(xd + k0);
    const QuantX8<Scalar>* xb =
        reinterpret_cast<const QuantX8<Scalar>*>(xd + k0 + 64);
    const QuantX8<Scalar> a0 = xa[0], a1 = xa[1];
    const QuantX8<Scalar> b0 = xb[0], b1 = xb[1];
    float xa8[16], xb8[16];
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 8; ++i) {
      xa8[i]     = static_cast<float>(a0.v[i]);
      xa8[i + 8] = static_cast<float>(a1.v[i]);
      xb8[i]     = static_cast<float>(b0.v[i]);
      xb8[i + 8] = static_cast<float>(b1.v[i]);
    }
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 16; ++i) {
      const int qa = (int)((quant_u16x8_byte(ql, i) & 0xF) |
                           (((quant_u16x8_byte(qh, i) >> shf) & 3) << 4));
      const int qb = (int)((quant_u16x8_byte(ql, i) >> 4) |
                           (((quant_u16x8_byte(qh, i) >> (shf + 4)) & 3) << 4));
      sa = fmaf(ds0 * (float)(qa - 32), xa8[i], sa);
      sb = fmaf(ds1 * (float)(qb - 32), xb8[i], sb);
    }
  } else {
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 16; ++i) {
      const int qa = (int)((quant_u16x8_byte(ql, i) & 0xF) |
                           (((quant_u16x8_byte(qh, i) >> shf) & 3) << 4));
      const int qb = (int)((quant_u16x8_byte(ql, i) >> 4) |
                           (((quant_u16x8_byte(qh, i) >> (shf + 4)) & 3) << 4));
      sa = fmaf(ds0 * (float)(qa - 32),
                static_cast<float>(xd[(k0 + i) * xs]), sa);
      sb = fmaf(ds1 * (float)(qb - 32),
                static_cast<float>(xd[(k0 + 64 + i) * xs]), sb);
    }
  }
  return sa + sb;
}

// One Q8_0 unit: a whole 34-byte block (32 elements, one scale).
template <class Scalar>
KOKKOS_INLINE_FUNCTION float quant_q80_chunk_dot(const uint8_t* row,
                                                 const Scalar* xd, size_t xs,
                                                 size_t c, bool xvec) {
  const BlockQ80* b = reinterpret_cast<const BlockQ80*>(
      row + c * sizeof(BlockQ80));
  const size_t k0 = c * 32;
  const float d   = fp16_to_float(b->d);
  // BlockQ80 is 34 bytes: 2-byte loads (qs follows a 2-byte header).
  const QuantU16x8 lo =
      quant_load_u16x8(reinterpret_cast<const uint8_t*>(b->qs));
  const QuantU16x8 hi =
      quant_load_u16x8(reinterpret_cast<const uint8_t*>(b->qs + 16));
  float s = 0;
  if (xvec) {
    const QuantX8<Scalar>* xa =
        reinterpret_cast<const QuantX8<Scalar>*>(xd + k0);
    const QuantX8<Scalar> a0 = xa[0], a1 = xa[1];
    const QuantX8<Scalar> a2 = xa[2], a3 = xa[3];
    float xf[32];
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 8; ++i) {
      xf[i]      = static_cast<float>(a0.v[i]);
      xf[i + 8]  = static_cast<float>(a1.v[i]);
      xf[i + 16] = static_cast<float>(a2.v[i]);
      xf[i + 24] = static_cast<float>(a3.v[i]);
    }
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 16; ++i) {
      const float qa = static_cast<float>(
          static_cast<int8_t>(quant_u16x8_byte(lo, i)));
      const float qb = static_cast<float>(
          static_cast<int8_t>(quant_u16x8_byte(hi, i)));
      s = fmaf(d * qa, xf[i], s);
      s = fmaf(d * qb, xf[i + 16], s);
    }
  } else {
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 32; ++i) {
      const uint8_t byte = i < 16 ? quant_u16x8_byte(lo, i)
                                  : quant_u16x8_byte(hi, i - 16);
      s = fmaf(d * static_cast<float>(static_cast<int8_t>(byte)),
               static_cast<float>(xd[(k0 + i) * xs]), s);
    }
  }
  return s;
}

// One Q5_K unit: one 32-element sub-block (32 contiguous elements sharing one
// (d*scale, -dmin*min) pair, unlike the split Q4_K/Q6_K units). Unit c of a
// row: block c/8, sub (c%8) = 2*iter + half; the unit's first element is
// block*256 + iter*64 + half*32. The 32 qs bytes come from qs + iter*32 (two
// 128-bit loads) and the 32 qh bytes from qh (two more); only the nibble side
// `half` and the one qh bit (1/2 << 2*iter) are used.
template <class Scalar>
KOKKOS_INLINE_FUNCTION float quant_q5k_chunk_dot(const uint8_t* row,
                                                  const Scalar* xd, size_t xs,
                                                  size_t c, bool xvec) {
  const size_t blk  = c / 8;
  const BlockQ5K* b = reinterpret_cast<const BlockQ5K*>(
      row + blk * sizeof(BlockQ5K));
  const size_t sub  = c % 8;
  const size_t iter = sub / 2;
  const bool low    = (sub % 2) == 0;
  const size_t k0   = blk * 256 + iter * 64 + (low ? 0 : 32);
  const QuantU8x16 ql0 =
      *reinterpret_cast<const QuantU8x16*>(b->qs + iter * 32);
  const QuantU8x16 ql1 =
      *reinterpret_cast<const QuantU8x16*>(b->qs + iter * 32 + 16);
  const QuantU8x16 qh0 = *reinterpret_cast<const QuantU8x16*>(b->qh);
  const QuantU8x16 qh1 = *reinterpret_cast<const QuantU8x16*>(b->qh + 16);
  uint8_t sc, mn;
  get_scale_min_k4(static_cast<int>(sub), b->scales, &sc, &mn);
  const float dm = fp16_to_float(b->d) * (float)sc;
  const float mm = -fp16_to_float(b->dmin) * (float)mn;
  const uint8_t u = (uint8_t)((low ? 1u : 2u) << (2 * iter));
  float s         = 0;
  if (xvec) {
    const QuantX8<Scalar>* xa =
        reinterpret_cast<const QuantX8<Scalar>*>(xd + k0);
    const QuantX8<Scalar> a0 = xa[0], a1 = xa[1], a2 = xa[2], a3 = xa[3];
    float xf[32];
KOKKOSDNN_PRAGMA_UNROLL()
    for (int i = 0; i < 8; ++i) {
      xf[i]      = static_cast<float>(a0.v[i]);
      xf[i + 8]  = static_cast<float>(a1.v[i]);
      xf[i + 16] = static_cast<float>(a2.v[i]);
      xf[i + 24] = static_cast<float>(a3.v[i]);
    }
    for (int i = 0; i < 32; ++i) {
      const uint8_t qb = i < 16 ? (low ? (uint8_t)(ql0.b[i] & 0xF)
                                       : (uint8_t)(ql0.b[i] >> 4))
                                : (low ? (uint8_t)(ql1.b[i - 16] & 0xF)
                                       : (uint8_t)(ql1.b[i - 16] >> 4));
      const uint8_t hb = i < 16 ? qh0.b[i] : qh1.b[i - 16];
      const float qf   = (float)qb + (float)((hb & u) ? 16 : 0);
      s = fmaf(dm * qf + mm, xf[i], s);
    }
  } else {
    for (int i = 0; i < 32; ++i) {
      const uint8_t qb = i < 16 ? (low ? (uint8_t)(ql0.b[i] & 0xF)
                                       : (uint8_t)(ql0.b[i] >> 4))
                                : (low ? (uint8_t)(ql1.b[i - 16] & 0xF)
                                       : (uint8_t)(ql1.b[i - 16] >> 4));
      const uint8_t hb = i < 16 ? qh0.b[i] : qh1.b[i - 16];
      const float qf   = (float)qb + (float)((hb & u) ? 16 : 0);
      s = fmaf(dm * qf + mm, static_cast<float>(xd[(k0 + (size_t)i) * xs]), s);
    }
  }
  return s;
}

// One decode unit of a packed row (see quant_unit_count above).
template <class Scalar>
KOKKOS_INLINE_FUNCTION float quant_unit_dot(GgmlType dtype,
                                            const uint8_t* row,
                                            const Scalar* xd, size_t xs,
                                            size_t c, bool xvec) {
  switch (dtype) {
    case GgmlType::Q4_K: return quant_q4k_chunk_dot(row, xd, xs, c, xvec);
    case GgmlType::Q5_K: return quant_q5k_chunk_dot(row, xd, xs, c, xvec);
    case GgmlType::Q6_K: return quant_q6k_chunk_dot(row, xd, xs, c, xvec);
    case GgmlType::Q8_0: return quant_q80_chunk_dot(row, xd, xs, c, xvec);
    default:
      return quant_elem(dtype, row, c) * static_cast<float>(xd[c * xs]);
  }
}

// Threads per output row in a decode team: 32 keeps a row's reduction inside
// one warp (a shuffle reduction, no shared-memory stage) while giving each
// thread several units of a real row to pipeline. Clamped down for rows with
// fewer units, so short rows never launch lanes that do nothing.
inline int quant_unit_threads(size_t nunits) {
  int t = 32;
  while ((size_t)t > nunits && t > 1) t /= 2;
  return t;
}

// Grouped GEMV/GEMM functors: same contracts as QuantGemvFunctor /
// QuantGemmFunctor below, but the TeamThreadRange runs over ngroups = K/GS
// sub-blocks (GS = 32 for Q4_K/Q8_0, 16 for Q6_K) with one quant_group_dot
// per group instead of K quant_elem calls.
template <class ExecSpace, class Scalar>
struct QuantGemvGroupedFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar*, ExecSpace> x, y;
  const uint8_t* wdata;
  GgmlType dtype;
  size_t ngroups;
  size_t group;
  size_t row_bytes;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const size_t j   = (size_t)team.league_rank();
    const uint8_t* r = wdata + j * row_bytes;
    const auto xx    = x;
    const GgmlType dt = dtype;
    const size_t ng   = ngroups;
    const size_t gs   = group;
    const Scalar* xd  = xx.data();
    const size_t xs   = xx.stride(0);
    float team_sum    = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, ng),
        KOKKOS_LAMBDA(size_t g, float& s) {
          s += quant_group_dot(dt, r, xd, xs, g * gs);
        },
        team_sum);
    if (team.team_rank() == 0) y(j) = static_cast<Scalar>(team_sum);
  }
};

template <class ExecSpace, class Scalar>
struct QuantGemmGroupedFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> X, Y;
  const uint8_t* wdata;
  GgmlType dtype;
  size_t N;
  size_t ngroups;
  size_t group;
  size_t row_bytes;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const size_t id  = (size_t)team.league_rank();
    const size_t i   = id / N;
    const size_t j   = id % N;
    const uint8_t* r = wdata + j * row_bytes;
    const auto XX     = X;
    const GgmlType dt = dtype;
    const size_t ng   = ngroups;
    const size_t gs   = group;
    const Scalar* xb  = XX.data() + i * XX.stride(0);
    float team_sum    = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, ng),
        KOKKOS_LAMBDA(size_t g, float& s) {
          s += quant_group_dot(dt, r, xb, size_t(1), g * gs);
        },
        team_sum);
    if (team.team_rank() == 0) Y(i, j) = static_cast<Scalar>(team_sum);
  }
};

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
        " (supported: F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K)");
  const size_t want = rows * ggml_row_bytes(dtype, cols);  // validates cols
  if (have_bytes != want)
    throw std::invalid_argument(std::string(what) + ": payload size " +
                                std::to_string(have_bytes) + " != " +
                                std::to_string(want));
  if (cols > (size_t)INT_MAX)
    throw std::invalid_argument(std::string(what) + ": cols too large");
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// Sparse-expert weight tables (device-side indexing of packed weights).
// ---------------------------------------------------------------------------
// MoE expert weights live as one QuantWeight per expert (a std::vector on the
// host), which a device kernel cannot index. A QuantWeightTable mirrors the
// per-expert payload pointers into a device view once at load time; the shape
// and dtype shared by every entry travel with it, so kernels need one pointer
// lookup and no host round trip. Entries must be homogeneous, which is what
// the loader produces for a stacked [E, N, K] expert tensor (and what the
// random_init test weights produce too).
template <class ExecSpace>
struct QuantWeightTable {
  Kokkos::View<uintptr_t*, ExecSpace> base;  // packed payload per entry
  GgmlType dtype = GgmlType::F32;
  size_t rows      = 0;  // N (out_features) of one entry
  size_t cols      = 0;  // K (in_features) of one entry
  size_t row_bytes = 0;

  size_t size() const { return base.extent(0); }
};

template <class ExecSpace>
QuantWeightTable<ExecSpace> make_quant_weight_table(
    const std::vector<QuantWeight<ExecSpace>>& ws, const std::string& what) {
  QuantWeightTable<ExecSpace> t;
  if (ws.empty()) return t;
  t.dtype = ws[0].dtype;
  t.rows  = ws[0].rows;
  t.cols  = ws[0].cols;
  for (size_t e = 0; e < ws.size(); ++e) {
    const auto& w = ws[e];
    if (w.dtype != t.dtype || w.rows != t.rows || w.cols != t.cols)
      throw std::invalid_argument(
          what + ": entries must share dtype and shape");
    Impl::quant_check(w.dtype, w.rows, w.cols, w.data.extent(0), what.c_str());
  }
  t.row_bytes = ws[0].row_bytes();
  t.base      = Kokkos::View<uintptr_t*, ExecSpace>(what + "::base", ws.size());
  auto hbase  = Kokkos::create_mirror_view(t.base);
  for (size_t e = 0; e < ws.size(); ++e)
    hbase(e) = reinterpret_cast<uintptr_t>(ws[e].data.data());
  Kokkos::deep_copy(t.base, hbase);
  return t;
}

namespace Impl {

// Fused gate/up + SwiGLU for one decoded token. One team covers
// `rows_per_team` output columns of one routed slot: the team splits into that
// many sub-teams of equal width, each reducing one output row, so the final
// combine is a short sum over sub-teams instead of a whole-team tree. The
// routing slot's expert id and weight are read on device (no host fence) and
// the kernel emits act[s, j] = silu(gate_e x)_j * (up_e x)_j directly, so the
// separate gate, up and swiglu launches disappear.
//
// Why several rows per team: at batch 1 a single output row is only K/32 units
// (64 at K=2048), i.e. 2 units per thread, and one row per team caps the grid
// at one-warp blocks. Fusing rows into one block keeps that per-thread unit
// count (the load pipeline depth) while filling the SM with wider blocks:
// measured on this RTX 3090 at K=2048/MI=1536/top-4/Q4_K, one row per
// 32-thread team 412 GB/s vs four rows per 128-thread team 490 GB/s (short
// bursts, boost clocks). The gate and up dots of a row run in the same inner
// loop, so the activation window is loaded once for both matrices (it is 2
// bytes/element for bf16 against 0.56 packed bytes per element for Q4_K: the
// activation load stream is the larger one).
template <class ExecSpace, class Scalar>
struct QuantMoeGateUpFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> X;    // [1, K]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> act;  // [S, MI]
  Kokkos::View<uintptr_t*, ExecSpace> gbase, ubase;
  Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace> idx;    // [1, S]
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> wts;  // [1, S]
  GgmlType gdtype, udtype;
  bool xvec;
  size_t g_units, u_units;  // decode units per packed row (K)
  size_t g_row_bytes, u_row_bytes;
  size_t rows;        // MI: output columns per expert
  size_t n_experts;
  int rows_per_team;  // output rows reduced inside one team

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const int ts   = team.team_size();
    const int rpt  = rows_per_team;
    const int tpr  = ts / rpt;  // threads per output row
    const int rank = team.team_rank();
    const int sub  = rank / tpr;
    const int lane = rank - sub * tpr;
    float* scratch = static_cast<float*>(
        team.team_shmem().get_shmem(sizeof(float) * 2 * (size_t)ts));
    const size_t teams_per_slot = (rows + (size_t)rpt - 1) / (size_t)rpt;
    const size_t id = static_cast<size_t>(team.league_rank());
    const size_t s  = id / teams_per_slot;
    const size_t j0 = (id - s * teams_per_slot) * (size_t)rpt;
    const size_t j  = j0 + (size_t)sub;
    const int e     = idx(0, s);
    // Routing ids come from the router kernel (always in range). The guard
    // keeps a bogus id from reading outside the expert tables and matching
    // the E == 0 / zero-weight convention: contribute nothing.
    const bool bad = (e < 0) || (static_cast<size_t>(e) >= n_experts) ||
                     j >= rows;
    float g = 0, u = 0;
    if (!bad) {
      const auto xx    = X;
      const Scalar* xd = xx.data();
      // Column stride: quant_unit_dot indexes the activation by *element*
      // within the single row (X is [1, K] LayoutRight), so it wants stride(1)
      // -- stride(0) is the row pitch and would walk off the buffer. The
      // vectorized path additionally assumes the row is contiguous
      // (stride(1) == 1), which xvec guarantees.
      const size_t xs = xx.stride(1);
      const uint8_t* gr =
          reinterpret_cast<const uint8_t*>(gbase(e)) + j * g_row_bytes;
      const uint8_t* ur =
          reinterpret_cast<const uint8_t*>(ubase(e)) + j * u_row_bytes;
      const GgmlType gdt = gdtype, udt = udtype;
      const bool xv      = xvec;
      const size_t ng = g_units, nu = u_units;
      const size_t nc = ng < nu ? ng : nu;
      for (size_t c = (size_t)lane; c < nc; c += (size_t)tpr) {
        g += quant_unit_dot(gdt, gr, xd, xs, c, xv);
        u += quant_unit_dot(udt, ur, xd, xs, c, xv);
      }
      // A mixed recipe can give gate and up different dtypes, hence different
      // unit counts; the longer matrix then finishes in its own loop.
      for (size_t c = nc + (size_t)lane; c < ng; c += (size_t)tpr)
        g += quant_unit_dot(gdt, gr, xd, xs, c, xv);
      for (size_t c = nc + (size_t)lane; c < nu; c += (size_t)tpr)
        u += quant_unit_dot(udt, ur, xd, xs, c, xv);
    }
    scratch[(size_t)rank]              = g;
    scratch[(size_t)ts + (size_t)rank] = u;
    team.team_barrier();
    if (rank < rpt) {
      const size_t jr = j0 + (size_t)rank;
      float sg = 0, su = 0;
      for (int t = rank * tpr; t < (rank + 1) * tpr; ++t) {
        sg += scratch[(size_t)t];
        su += scratch[(size_t)ts + (size_t)t];
      }
      // Each output column belongs to exactly one sub-team, so this is an
      // assignment: no ordering question and no caller-side zero fill.
      if (jr < rows) act(s, jr) = static_cast<Scalar>(silu(sg) * su);
    }
  }
};

// Down projection + weighted combine for one decoded token: one team covers
// `rows_per_team` output rows d, and the reduction over the routed slots
// happens inside it (y[d] = sum_s w_s * down_e[s][d,:] . act[s]). The slot
// loop lives in the kernel, so y is *assigned* (no caller-side zero fill, no
// per-slot weighted-add launch) and there is no atomics/ordering question.
template <class ExecSpace, class Scalar>
struct QuantMoeDownFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> act;  // [S, MI]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> Y;    // [1, K]
  Kokkos::View<uintptr_t*, ExecSpace> dbase;
  Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace> idx;    // [1, S]
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> wts;  // [1, S]
  GgmlType dtype;
  bool xvec;
  size_t n_units;    // decode units per packed row (MI)
  size_t row_bytes;  // down row stride
  size_t n_experts;
  size_t slots;
  size_t rows;  // K: output rows of the down projection
  int rows_per_team;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& team) const {
    const int ts   = team.team_size();
    const int rpt  = rows_per_team;
    const int tpr  = ts / rpt;
    const int rank = team.team_rank();
    const int sub  = rank / tpr;
    const int lane = rank - sub * tpr;
    float* scratch = static_cast<float*>(
        team.team_shmem().get_shmem(sizeof(float) * (size_t)ts));
    const size_t d0 = static_cast<size_t>(team.league_rank()) * (size_t)rpt;
    const size_t d  = d0 + (size_t)sub;
    const auto aa       = act;
    const Scalar* abase = aa.data();
    const size_t stride = aa.stride(0);
    const GgmlType dt = dtype;
    const bool xv     = xvec;
    const size_t nc   = n_units;
    const size_t rb   = row_bytes;
    const size_t ne   = n_experts;
    const size_t nu   = slots * nc;
    float acc         = 0;
    if (d < rows) {
      for (size_t uc = (size_t)lane; uc < nu; uc += (size_t)tpr) {
        const size_t s = uc / nc;
        const size_t c = uc - s * nc;
        const int e    = idx(0, s);
        if (e < 0 || static_cast<size_t>(e) >= ne) continue;
        const float w = wts(0, s);
        if (w == 0.0f) continue;
        const uint8_t* row =
            reinterpret_cast<const uint8_t*>(dbase(e)) + d * rb;
        acc += w * quant_unit_dot(dt, row, abase + s * stride, size_t(1), c,
                                  xv);
      }
    }
    scratch[(size_t)rank] = acc;
    team.team_barrier();
    if (rank < rpt) {
      float sum = 0;
      for (int t = rank * tpr; t < (rank + 1) * tpr; ++t)
        sum += scratch[(size_t)t];
      if (d < rows) Y(0, d) = static_cast<Scalar>(sum);
    }
  }
};

// 128-bit activation loads need 16-byte-aligned rows.
inline bool quant_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// Stage 1 of quant_moe_decode: act[s, j] = silu(gate_e x)_j * (up_e x)_j.
// Split from the down stage so the two launches can be timed (and reasoned
// about) separately; quant_moe_decode is the only production caller.
template <class ExecSpace, class Scalar>
void quant_moe_decode_gate_up(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const QuantWeightTable<ExecSpace>& gate,
    const QuantWeightTable<ExecSpace>& up,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& wts,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& act,
    const std::string& label) {
  const size_t K  = X.extent(1);
  const size_t MI = gate.rows;
  const size_t S  = idx.extent(1);
  const size_t ng = quant_unit_count(gate.dtype, K);
  const size_t nu = quant_unit_count(up.dtype, K);
  const bool xvec = quant_aligned16(X.data()) && X.stride(1) == 1;

  // 32 threads per output row (see quant_unit_threads) and up to 4 rows fused
  // into one 4-warp team. Both are launch-shape choices: the kernel derives
  // the sub-team width from the real team size, so a backend that cannot field
  // 128 threads (host backends cap at 1) simply gets narrower teams.
  const size_t nc  = ng > nu ? ng : nu;
  const int want_r = 4;
  const int want   = quant_unit_threads(nc) * want_r;
  QuantMoeGateUpFunctor<ExecSpace, Scalar> f{
      X,            act,        gate.base, up.base,   idx,  wts,
      gate.dtype,   up.dtype,   xvec,      ng,        nu,   gate.row_bytes,
      up.row_bytes, MI,         gate.size(), want_r};
  const std::size_t shmem = sizeof(float) * 2 * (std::size_t)want;
  // Cached occupancy probe (KokkosDNN_Occupancy.hpp): decode runs this twice
  // per MoE layer; the answer depends only on the functor type, the exec-space
  // instance and the scratch reservation.
  const int tmax =
      ::KokkosDNN::Impl::cached_team_size_max<ExecSpace>(f, shmem);
  const int ts = want < tmax ? want : (tmax > 1 ? tmax : 1);
  f.rows_per_team = ts < want_r ? ts : want_r;
  Kokkos::parallel_for(
      label + "::gate_up",
      Kokkos::TeamPolicy<ExecSpace>(
          static_cast<int>((S * MI + (size_t)f.rows_per_team - 1) /
                           (size_t)f.rows_per_team),
          ts)
          .set_scratch_size(0, Kokkos::PerTeam(shmem)),
      f);
}

// Stage 2 of quant_moe_decode: Y[0, d] = sum_s w_s * down_e[s][d, :] . act[s].
template <class ExecSpace, class Scalar>
void quant_moe_decode_down(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& act,
    const QuantWeightTable<ExecSpace>& down,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& wts,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label) {
  const size_t K  = Y.extent(1);
  const size_t MI = down.cols;
  const size_t S  = idx.extent(1);
  const size_t nd = quant_unit_count(down.dtype, MI);
  const bool avec = quant_aligned16(act.data()) &&
                    (act.stride(0) * sizeof(Scalar)) % 16 == 0;

  // 64 threads per output row here rather than 32: a down row carries S*MI/32
  // units (192 at S=4/MI=1536), so the extra width still leaves 3 units of
  // load pipelining per thread while halving the block count. Measured on this
  // RTX 3090 at that shape (Q6_K down, 10.3 MB/call, short bursts): 32
  // threads/row 432 GB/s, 64 threads/row 466 GB/s, 128 threads/row 446 GB/s.
  // Fusing rows buys nothing for down, whose per-row work is already 3x
  // gate_up's, so rows_per_team stays 1.
  const int row_threads = quant_unit_threads(S * nd);
  const int want_r      = 1;
  const int want        = row_threads < 64 ? row_threads : 64;
  QuantMoeDownFunctor<ExecSpace, Scalar> f{
      act,           Y,        down.base,     idx,   wts,
      down.dtype,    avec,     nd,            down.row_bytes, down.size(),
      S,             K, want_r};
  const std::size_t shmem = sizeof(float) * (std::size_t)want;
  const int tmax =
      ::KokkosDNN::Impl::cached_team_size_max<ExecSpace>(f, shmem);
  const int ts = want < tmax ? want : (tmax > 1 ? tmax : 1);
  f.rows_per_team = ts < want_r ? ts : want_r;
  Kokkos::parallel_for(
      label + "::down",
      Kokkos::TeamPolicy<ExecSpace>(
          static_cast<int>((K + (size_t)f.rows_per_team - 1) /
                           (size_t)f.rows_per_team),
          ts)
          .set_scratch_size(0, Kokkos::PerTeam(shmem)),
      f);
}

}  // namespace Impl

// Sparse mixture-of-experts decode (one token) against packed expert weights.
// Two launches per MoE layer, independent of top_k:
//   1. gate/up GEMV fused with SwiGLU  -> act [S, MI]
//   2. down GEMV fused with the weighted slot combine -> Y [1, K]
// The routing indices/weights are consumed on device, so this path performs no
// host synchronization at all (the per-token fence pair and the 2+2 * top_k
// launches of the per-slot dispatch are gone). act is caller-owned scratch
// ([S, MI], one row per routed slot) so steady-state decode allocates nothing.
// Numerics: same (dm * q + mm) * x terms in FP32, reassociated across threads;
// SwiGLU is applied in FP32 instead of after a Scalar round trip of gate/up.
template <class ExecSpace, class Scalar>
void quant_moe_decode(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const QuantWeightTable<ExecSpace>& gate,
    const QuantWeightTable<ExecSpace>& up,
    const QuantWeightTable<ExecSpace>& down,
    const Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>& idx,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& wts,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& act,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "KokkosDNN::Gguf::quant_moe_decode") {
  (void)handle;  // native kernels only; kept for API symmetry with gemm()
  if (X.extent(0) != 1 || Y.extent(0) != 1)
    throw std::invalid_argument("quant_moe_decode: expected one row");
  const size_t K = X.extent(1), MI = gate.rows;
  const size_t S = idx.extent(0) == 0 ? 0 : idx.extent(1);
  if (gate.cols != K || up.cols != K || up.rows != MI || down.rows != K ||
      down.cols != MI || Y.extent(1) != K)
    throw std::invalid_argument("quant_moe_decode: dimension mismatch");
  if (gate.size() == 0 || gate.size() != up.size() ||
      gate.size() != down.size())
    throw std::invalid_argument("quant_moe_decode: expert table mismatch");
  if (idx.extent(0) != 1 || wts.extent(0) != 1 || wts.extent(1) != S)
    throw std::invalid_argument("quant_moe_decode: routing dimension mismatch");
  if (act.extent(0) < S || (S > 0 && act.extent(1) != MI))
    throw std::invalid_argument("quant_moe_decode: act scratch too small");
  if (S == 0 || K == 0 || MI == 0) return;
  // Mixed K-quant recipes store gate/up and down with different dtypes
  // (Q4_K_M keeps most experts Q4_K but the down projection Q6_K), so each
  // matrix keeps its own dtype and unit count.
  // The unit layout assumes whole ggml blocks (a unit never straddles one);
  // every K-quant expert matrix in practice is block aligned. Other dtypes
  // fall back to a one-element unit and work for any width.
  auto block_aligned = [](GgmlType dt, size_t n) {
    switch (dt) {
      case GgmlType::Q4_K:
      case GgmlType::Q5_K:
      case GgmlType::Q6_K:
      case GgmlType::Q8_0: return n % 256 == 0;
      default: return true;
    }
  };
  if (!block_aligned(gate.dtype, K) || !block_aligned(up.dtype, K) ||
      !block_aligned(down.dtype, MI))
    throw std::invalid_argument(
        "quant_moe_decode: K-quant rows must be a multiple of 256 elements");
  if (Impl::quant_unit_count(gate.dtype, K) == 0 ||
      Impl::quant_unit_count(up.dtype, K) == 0 ||
      Impl::quant_unit_count(down.dtype, MI) == 0)
    throw std::invalid_argument("quant_moe_decode: bad packed row");

  Impl::quant_moe_decode_gate_up<ExecSpace, Scalar>(X, gate, up, idx, wts, act,
                                                    label);
  Impl::quant_moe_decode_down<ExecSpace, Scalar>(act, down, idx, wts, Y,
                                                 label);
}

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
  // Grouped sub-block path (one scale set per 32/16 elems) when K aligns;
  // otherwise the per-element path below.
  size_t gs = 0;
  if (Impl::quant_can_group(w.dtype, w.cols, gs)) {
    Impl::QuantGemvGroupedFunctor<ExecSpace, Scalar> f{
        x, y, w.data.data(), w.dtype, w.cols / gs, gs, w.row_bytes()};
    Kokkos::TeamPolicy<ExecSpace> policy((int)w.rows, Kokkos::AUTO);
    Kokkos::parallel_for(label, policy, f);
    return;
  }
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
  // Grouped sub-block path (one scale set per 32/16 elems) when K aligns;
  // otherwise the per-element path below.
  size_t gs = 0;
  if (Impl::quant_can_group(w.dtype, w.cols, gs)) {
    Impl::QuantGemmGroupedFunctor<ExecSpace, Scalar> fg{
        X, Y, w.data.data(), w.dtype, N, K / gs, gs, w.row_bytes()};
    Kokkos::TeamPolicy<ExecSpace> policy((int)(M * N), Kokkos::AUTO);
    Kokkos::parallel_for(label, policy, fg);
    return;
  }
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


// Dequantize a packed weight to a dense row-major matrix: Y[r,c] = w[r,c].
// Covers the gguf_can_compute() dtype set (Q4_K/Q5_K/Q6_K/Q8_0 per-element
// dequantization, F32/F16/BF16 via cast-copy). Portable elementwise kernels
// (one thread per output, MDRange over rows x cols), no backend ifdefs.
// Throws on dimension mismatch, unsupported dtype, or a column count that
// is not a multiple of the block size.
//
// Rationale: MoE prefill runs one quant_gemm per routed expert over a thin
// gathered row slice (N ~ C*K/E rows). The team-per-output-element kernel
// re-dequantizes each packed weight row once per output row (M times) and
// reloads each activation row once per output column (N times) -- on a
// 902-token prefill that is 99% of GPU kernel time (nsys). Dequantizing
// each routed expert once per layer and running the dense gemm()
// (cuBLAS/native-tiled, FP32 accumulation) removes both redundancies.
// Values match the packed weights up to the Scalar rounding that every
// other dense layer in the model already carries.
//
// K-quant notes: one thread per output (consecutive threads write
// consecutive outputs: perfect store coalescing, shared blocks through
// L1/L2). Block- or sub-block-per-thread grids stream 32-512B contiguous
// runs per thread and run at a fraction of stream bandwidth (~60-75 GB/s
// measured for both on RTX 3090 vs ~220 GB/s per-element). All intra-block
// offsets are shifts/masks (block sizes are powers of two); the MDRange
// policy carries r and c directly (no division).
template <class ExecSpace, class Scalar>
void dequant_to_dense(
    const QuantWeight<ExecSpace>& w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "KokkosDNN::Gguf::dequant_to_dense") {
  if (Y.extent(0) != w.rows || Y.extent(1) != w.cols)
    throw std::invalid_argument("dequant_to_dense: dimension mismatch");
  Impl::quant_check(w.dtype, w.rows, w.cols, w.data.extent(0),
                    "dequant_to_dense");
  if (w.rows == 0 || w.cols == 0) return;
  const size_t R = w.rows, C = w.cols;
  const size_t rb = w.row_bytes();
  const uint8_t* wdata = w.data.data();
  auto dst             = Y;
  switch (w.dtype) {
    case GgmlType::Q4_K: {
      if (C % 256 != 0)
        throw std::invalid_argument(
            "dequant_to_dense: Q4_K cols must be a multiple of 256");
      // Flat 1-D grid: consecutive threads take consecutive columns
      // (perfect store coalescing). The default MDRangePolicy maps
      // threadIdx.x to the row axis, striding stores by C (~200+ us here).
      Kokkos::parallel_for(
          label, Kokkos::RangePolicy<ExecSpace>(0, R * C),
          KOKKOS_LAMBDA(size_t k) {
            const size_t r = k / C;
            const size_t c = k - r * C;
            const BlockQ4K* b = reinterpret_cast<const BlockQ4K*>(
                wdata + r * rb + (c >> 8) * sizeof(BlockQ4K));
            const size_t sub  = (c >> 5) & 7;
            const size_t iter = sub >> 1;
            const int j       = (int)((iter << 1) + (sub & 1));
            uint8_t sc, mn;
            get_scale_min_k4(j, b->scales, &sc, &mn);
            const float dm = fp16_to_float(b->d) * (float)sc;
            const float mm = -fp16_to_float(b->dmin) * (float)mn;
            const uint8_t qv = b->qs[iter * 32 + (c & 31)];
            const float qf =
                (float)(((sub & 1) == 0) ? (qv & 0xF) : (qv >> 4));
            dst(r, c) = static_cast<Scalar>(dm * qf + mm);
          });
      return;
    }
    case GgmlType::Q5_K: {
      if (C % 256 != 0)
        throw std::invalid_argument(
            "dequant_to_dense: Q5_K cols must be a multiple of 256");
      Kokkos::parallel_for(
          label, Kokkos::RangePolicy<ExecSpace>(0, R * C),
          KOKKOS_LAMBDA(size_t k) {
            const size_t r = k / C;
            const size_t c = k - r * C;
            const BlockQ5K* b = reinterpret_cast<const BlockQ5K*>(
                wdata + r * rb + (c >> 8) * sizeof(BlockQ5K));
            const size_t sub  = (c >> 5) & 7;
            const size_t iter = sub >> 1;
            const bool low    = (sub & 1) == 0;
            const size_t ll   = c & 31;
            uint8_t sc, mn;
            get_scale_min_k4((int)sub, b->scales, &sc, &mn);
            const float dm = fp16_to_float(b->d) * (float)sc;
            const float mm = -fp16_to_float(b->dmin) * (float)mn;
            const uint8_t qv = b->qs[iter * 32 + ll];
            const uint8_t u  = (uint8_t)((low ? 1u : 2u) << (2 * iter));
            const float qf =
                (float)(low ? (qv & 0xF) : (qv >> 4)) +
                (float)(((b->qh[ll] & u) != 0) ? 16 : 0);
            dst(r, c) = static_cast<Scalar>(dm * qf + mm);
          });
      return;
    }
    case GgmlType::Q6_K: {
      if (C % 256 != 0)
        throw std::invalid_argument(
            "dequant_to_dense: Q6_K cols must be a multiple of 256");
      Kokkos::parallel_for(
          label, Kokkos::RangePolicy<ExecSpace>(0, R * C),
          KOKKOS_LAMBDA(size_t k) {
            const size_t r = k / C;
            const size_t c = k - r * C;
            const BlockQ6K* b = reinterpret_cast<const BlockQ6K*>(
                wdata + r * rb + (c >> 8) * sizeof(BlockQ6K));
            const size_t l  = c & 255;  // position inside the 256-block
            const size_t nh = l >> 7;   // 128-element half
            const size_t m  = (l >> 5) & 3;  // which of the 4 32-groups
            const size_t ll = l & 31;
            const uint8_t* ql = b->ql + nh * 64;
            const uint8_t* qh = b->qh + nh * 32;
            const int8_t* sc  = b->scales + nh * 8;
            const size_t is = (ll >> 4);  // scale selector within the half
            int qv;
            if (m == 0)
              qv = (int)(int8_t)((ql[ll] & 0xF) | (((qh[ll] >> 0) & 3) << 4)) -
                   32;
            else if (m == 1)
              qv = (int)(int8_t)((ql[ll + 32] & 0xF) |
                                 (((qh[ll] >> 2) & 3) << 4)) -
                   32;
            else if (m == 2)
              qv = (int)(int8_t)((ql[ll] >> 4) | (((qh[ll] >> 4) & 3) << 4)) -
                   32;
            else
              qv = (int)(int8_t)((ql[ll + 32] >> 4) |
                                 (((qh[ll] >> 6) & 3) << 4)) -
                   32;
            const float ds = fp16_to_float(b->d) * (float)sc[is + 2 * m];
            dst(r, c) = static_cast<Scalar>(ds * (float)qv);
          });
      return;
    }
    case GgmlType::Q8_0: {
      if (C % 32 != 0)
        throw std::invalid_argument(
            "dequant_to_dense: Q8_0 cols must be a multiple of 32");
      Kokkos::parallel_for(
          label, Kokkos::RangePolicy<ExecSpace>(0, R * C),
          KOKKOS_LAMBDA(size_t k) {
            const size_t r = k / C;
            const size_t c = k - r * C;
            const BlockQ80* b = reinterpret_cast<const BlockQ80*>(
                wdata + r * rb + (c >> 5) * sizeof(BlockQ80));
            const float d = fp16_to_float(b->d);
            dst(r, c) = static_cast<Scalar>(d * (float)b->qs[c & 31]);
          });
      return;
    }
    case GgmlType::F32: {
      Kokkos::View<const float**, Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          src(reinterpret_cast<const float*>(wdata), R, C);
      Kokkos::parallel_for(
          label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                                   {R, C}),
          KOKKOS_LAMBDA(size_t r, size_t c) {
            dst(r, c) = static_cast<Scalar>(src(r, c));
          });
      return;
    }
    case GgmlType::F16: {
      Kokkos::View<const uint16_t**, Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          src(reinterpret_cast<const uint16_t*>(wdata), R, C);
      Kokkos::parallel_for(
          label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                                   {R, C}),
          KOKKOS_LAMBDA(size_t r, size_t c) {
            dst(r, c) = static_cast<Scalar>(fp16_to_float(src(r, c)));
          });
      return;
    }
    case GgmlType::BF16: {
      Kokkos::View<const uint16_t**, Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          src(reinterpret_cast<const uint16_t*>(wdata), R, C);
      Kokkos::parallel_for(
          label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                                   {R, C}),
          KOKKOS_LAMBDA(size_t r, size_t c) {
            dst(r, c) = static_cast<Scalar>(bf16_to_float(src(r, c)));
          });
      return;
    }
    default:
      throw std::invalid_argument(
          "dequant_to_dense: unsupported dtype (unreachable: validated)");
  }
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

// Typed key-value metadata for write_synthetic_gguf. Implicitly constructible
// from a string map so existing (path, tensors, string_map) calls keep
// working; the writer emits F32/U32/BOOL-array entries after the strings.
struct SyntheticKV {
  std::map<std::string, std::string> strings;
  std::map<std::string, float> f32;
  std::map<std::string, uint32_t> u32;
  std::map<std::string, std::vector<char>> bool_arrays;  // 0/1 bytes
  SyntheticKV() = default;
  SyntheticKV(const std::map<std::string, std::string>& s) : strings(s) {}
};

inline void write_synthetic_gguf(
    const std::string& path, const std::vector<SyntheticTensor>& tensors,
    const SyntheticKV& kv = {}) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("write_synthetic_gguf: cannot open '" +
                                   path + "'");
  const std::map<std::string, std::string>& string_kv = kv.strings;
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
  const uint64_t n_kv = (uint64_t)(string_kv.size() + kv.f32.size() +
                                   kv.u32.size() + kv.bool_arrays.size());
  u64(n_kv);
  for (const auto& [k, v] : string_kv) {
    str(k);
    u32(8);  // GGUF_TYPE_STRING
    str(v);
  }
  for (const auto& [k, v] : kv.f32) {
    str(k);
    u32(6);  // GGUF_TYPE_FLOAT32
    put(&v, 4);
  }
  for (const auto& [k, v] : kv.u32) {
    str(k);
    u32(4);  // GGUF_TYPE_UINT32
    uint32_t x = v;
    put(&x, 4);
  }
  for (const auto& [k, v] : kv.bool_arrays) {
    str(k);
    u32(9);  // GGUF_TYPE_ARRAY
    u32(7);  // GGUF_TYPE_BOOL elements
    u64(v.size());
    if (!v.empty()) put(v.data(), v.size());
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
  for (const auto& [k, v] : kv.f32) {
    (void)v;
    off_pos += 8 + k.size() + 4 + 4;
  }
  for (const auto& [k, v] : kv.u32) {
    (void)v;
    off_pos += 8 + k.size() + 4 + 4;
  }
  for (const auto& [k, v] : kv.bool_arrays)
    off_pos += 8 + k.size() + 4 + 4 + 8 + v.size();
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

// Backward-compatible string-KV-only overload (no default argument, so
// two-argument calls unambiguously select the SyntheticKV overload above).
inline void write_synthetic_gguf(
    const std::string& path, const std::vector<SyntheticTensor>& tensors,
    const std::map<std::string, std::string>& string_kv) {
  SyntheticKV kv;
  kv.strings = string_kv;
  write_synthetic_gguf(path, tensors, kv);
}

}  // namespace Gguf
}  // namespace KokkosDNN

#endif
