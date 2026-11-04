// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GGUF_SYNTHETIC_HPP
#define KOKKOSDNN_GGUF_SYNTHETIC_HPP

// Synthetic GGUF writer for tests (header-only).
//
// Why split: the writer needs <fstream>, which is expensive to parse and
// needlessly widens every translation unit that includes the core GGUF
// container header. Test-only file creation lives here; production loaders
// include KokkosDNN_GGUF.hpp alone and stay free of <fstream>.

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_GGUF.hpp"

namespace KokkosDNN {
namespace Gguf {

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

#endif  // KOKKOSDNN_GGUF_SYNTHETIC_HPP
