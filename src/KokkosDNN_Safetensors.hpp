// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SAFETENSORS_HPP
#define KOKKOSDNN_SAFETENSORS_HPP

// Central safetensors header reader (header-only, next to KokkosDNN_GGUF.hpp).
//
// Consolidates the byte-identical torch-subset header scanners previously
// duplicated in:
//   models/moss_image/KokkosDNN_MossImageDit.hpp (dit_ld64/dit_parse_header,
//     DitMappedFile),
//   models/qwen_image_21/KokkosDNN_QwenImage21Config.hpp (qwen21_ld64/
//     qwen21_parse_header) + KokkosDNN_QwenImage21SafetensorsMap.hpp
//     (Qwen21MappedShard mmap/parse/validate),
//   models/moss_image/KokkosDNN_MossImageEncoder.hpp (MossMmap +
//     read_safetensors_header DOM parser),
//   models/moss_image/KokkosDNN_MossImageVae.hpp (read_le64/parse_entry/
//     SafeFile substring finder),
//   examples/moss_image/KokkosDNN_MossWeights.cpp (parse_json_string/
//     parse_safetensors_header/find_entry).
//
// Scope: safetensors container framing only -- `[0,8)` u64 LE header length,
// `[8,8+hn)` JSON, `data_base = 8+hn`, per-tensor
// `{dtype, shape, data_offsets}` with absolute `[begin,end)` slices.
// No model-specific names, dtypes beyond F32/BF16 element sizes, or weight
// layouts here; per-model loaders keep their expected-tensor inventories,
// suffix/exact matching policies, and device fill logic.
//
// The JSON scanner accepts only the torch-saved subset
// `{"__metadata__":{...},"name":{"dtype":..,"shape":[..],
// "data_offsets":[a,b]}}`. `__metadata__` is skipped with string-aware
// depth counting; unknown per-tensor fields are skipped (string/array/
// object aware). Error messages take a caller `what` prefix so historical
// diagnostics are preserved.
//
// No dependency on models/common (dependency direction is models -> core):
// the JSON string decoder below is self-contained (full escapes including
// \u with surrogate pairs; valid safetensors headers are ASCII, where it
// agrees with Common::JsonLite::json_parse_string).
//
// Portable: POSIX mmap (MAP_PRIVATE) with sequential read-ahead; mmap
// failure is a hard error; there is no read() fallback. Little-endian host
// assumed for the u64 header length (matches every previous loader, which
// memcpy'd the length on LE hosts).

#include <Kokkos_Core.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Safetensors {

namespace Impl {

// Decode a JSON string; s[q] must be the opening quote. Returns the decoded
// value and advances q past the closing quote. Full escapes including \uXXXX
// with UTF-16 surrogate pairs (superset of the old ASCII-only scanners).
inline std::string parse_json_string_impl(const std::string& s, size_t& q,
                                          const std::string& what) {
  ++q;  // opening quote
  std::string out;
  while (q < s.size()) {
    char c = s[q++];
    if (c == '"') return out;
    if (c != '\\') {
      out += c;
      continue;
    }
    if (q >= s.size()) break;
    char e = s[q++];
    switch (e) {
      case 'n': out += '\n'; break;
      case 't': out += '\t'; break;
      case 'r': out += '\r'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case '/': out += '/'; break;
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case 'u': {
        if (q + 4 > s.size())
          throw std::runtime_error(what + ": bad \\u escape");
        unsigned cp = 0;
        for (int i = 0; i < 4; ++i) {
          char h = s[q++];
          cp <<= 4;
          if (h >= '0' && h <= '9') cp += (unsigned)(h - '0');
          else if (h >= 'a' && h <= 'f') cp += (unsigned)(h - 'a') + 10;
          else if (h >= 'A' && h <= 'F') cp += (unsigned)(h - 'A') + 10;
          else throw std::runtime_error(what + ": bad \\u escape");
        }
        // Surrogate pair.
        if (cp >= 0xD800 && cp <= 0xDBFF && q + 6 <= s.size() && s[q] == '\\' &&
            s[q + 1] == 'u') {
          unsigned lo = 0;
          for (int i = 0; i < 4; ++i) {
            char h = s[q + 2 + i];
            lo <<= 4;
            if (h >= '0' && h <= '9') lo += (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') lo += (unsigned)(h - 'a') + 10;
            else if (h >= 'A' && h <= 'F') lo += (unsigned)(h - 'A') + 10;
            else break;
          }
          if (lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            q += 6;
          }
        }
        if (cp < 0x80) {
          out += (char)cp;
        } else if (cp < 0x800) {
          out += (char)(0xC0 | (cp >> 6));
          out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
          out += (char)(0xE0 | (cp >> 12));
          out += (char)(0x80 | ((cp >> 6) & 0x3F));
          out += (char)(0x80 | (cp & 0x3F));
        } else {
          out += (char)(0xF0 | (cp >> 18));
          out += (char)(0x80 | ((cp >> 12) & 0x3F));
          out += (char)(0x80 | ((cp >> 6) & 0x3F));
          out += (char)(0x80 | (cp & 0x3F));
        }
        break;
      }
      default: out += e; break;
    }
  }
  throw std::runtime_error(what + ": malformed safetensors header (string)");
}

// Skip a JSON value starting at pos (string/array/object/scalar aware).
// Used for unknown per-tensor fields. Stops at ',' or '}' at depth 0.
inline void skip_json_value(const std::string& js, size_t& pos,
                            const std::string& what, const std::string& name) {
  if (pos >= js.size())
    throw std::runtime_error(what + ": truncated entry " + name);
  if (js[pos] == '"') {
    (void)parse_json_string_impl(js, pos, what);
    return;
  }
  if (js[pos] == '[' || js[pos] == '{') {
    char oc = js[pos], cc = (oc == '[') ? ']' : '}';
    int depth = 0;
    bool in_str = false;
    while (pos < js.size()) {
      char c = js[pos++];
      if (in_str) {
        if (c == '\\') {
          if (pos < js.size()) ++pos;
        } else if (c == '"') {
          in_str = false;
        }
      } else {
        if (c == '"') in_str = true;
        if (c == oc) ++depth;
        if (c == cc && --depth == 0) return;
      }
    }
    throw std::runtime_error(what + ": truncated entry " + name);
  }
  while (pos < js.size() && js[pos] != ',' && js[pos] != '}') ++pos;
}

}  // namespace Impl

// ---- little-endian load ----
inline uint64_t ld64_le(const char* p) {
  uint64_t v = 0;
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

// ---- BF16 helpers ----
inline float bf16_to_float(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round-to-nearest-even float32 -> bf16, preserving Inf/NaN payloads.
inline uint16_t float_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7F800000u) == 0x7F800000u) {
    uint16_t h = (uint16_t)(u >> 16);
    if ((u & 0x007FFFFFu) != 0) h |= (uint16_t)0x0040u;  // keep NaN quiet
    return h;
  }
  const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)((u + bias) >> 16);
}

// One tensor entry: absolute file offsets of the data slice.
struct Entry {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // absolute file offsets
};

inline size_t entry_numel_shape(const std::vector<int64_t>& shape) {
  size_t n = 1;
  for (int64_t d : shape) n *= (size_t)d;
  return n;
}

// Decode a JSON string with caller `what` error prefix. s[pos] must be the
// opening quote; preserves the historical trailing-quote diagnostic.
inline std::string parse_json_string(const std::string& s, size_t& pos,
                                     const std::string& what = "safetensors") {
  std::string out = Impl::parse_json_string_impl(s, pos, what);
  if (pos == 0 || pos > s.size() || s[pos - 1] != '"')
    throw std::runtime_error(what + ": malformed safetensors header");
  return out;
}

// Minimal header-JSON scanner: only the torch-saved subset
// {"__metadata__":{...},"name":{"dtype":..,"shape":[..],"data_offsets":[a,b]}}.
inline std::map<std::string, Entry> parse_header(
    const std::string& js, uint64_t data_base,
    const std::string& what = "safetensors") {
  std::map<std::string, Entry> out;
  size_t pos = 0;
  auto skip_ws = [&]() {
    while (pos < js.size() && (js[pos] == ' ' || js[pos] == '\t' ||
                               js[pos] == '\n' || js[pos] == '\r'))
      ++pos;
  };
  skip_ws();
  if (pos >= js.size() || js[pos] != '{')
    throw std::runtime_error(what + ": malformed safetensors header");
  ++pos;
  while (true) {
    skip_ws();
    if (pos >= js.size())
      throw std::runtime_error(what + ": truncated safetensors header");
    if (js[pos] == '}') break;
    if (js[pos] != '"')
      throw std::runtime_error(what + ": bad header key");
    std::string name = parse_json_string(js, pos, what);
    skip_ws();
    if (pos >= js.size() || js[pos] != ':')
      throw std::runtime_error(what + ": bad header separator");
    ++pos;
    skip_ws();
    if (name == "__metadata__") {
      if (pos >= js.size() || js[pos] != '{')
        throw std::runtime_error(what + ": bad __metadata__");
      int depth = 0;
      bool in_str = false;
      while (pos < js.size()) {
        char c = js[pos++];
        if (in_str) {
          if (c == '\\') {
            if (pos < js.size()) ++pos;
          } else if (c == '"') {
            in_str = false;
          }
        } else {
          if (c == '"') in_str = true;
          if (c == '{') ++depth;
          if (c == '}') {
            if (--depth == 0) break;
          }
        }
      }
    } else {
      if (pos >= js.size() || js[pos] != '{')
        throw std::runtime_error(what + ": bad entry for " + name);
      ++pos;
      Entry e;
      bool have_dtype = false, have_shape = false, have_off = false;
      while (true) {
        skip_ws();
        if (pos >= js.size())
          throw std::runtime_error(what + ": truncated entry " + name);
        if (js[pos] == '}') {
          ++pos;
          break;
        }
        if (js[pos] != '"')
          throw std::runtime_error(what + ": bad entry key " + name);
        std::string k = parse_json_string(js, pos, what);
        skip_ws();
        if (pos >= js.size() || js[pos] != ':')
          throw std::runtime_error(what + ": bad entry sep " + name);
        ++pos;
        skip_ws();
        if (k == "dtype") {
          if (pos >= js.size() || js[pos] != '"')
            throw std::runtime_error(what + ": bad dtype " + name);
          e.dtype = parse_json_string(js, pos, what);
          have_dtype = true;
        } else if (k == "shape") {
          if (pos >= js.size() || js[pos] != '[')
            throw std::runtime_error(what + ": bad shape " + name);
          ++pos;
          e.shape.clear();
          while (true) {
            skip_ws();
            if (pos < js.size() && js[pos] == ']') {
              ++pos;
              break;
            }
            size_t st = pos;
            while (pos < js.size() &&
                   (isdigit((unsigned char)js[pos]) || js[pos] == '-'))
              ++pos;
            if (st == pos)
              throw std::runtime_error(what + ": bad shape num " + name);
            e.shape.push_back(std::stoll(js.substr(st, pos - st)));
            skip_ws();
            if (pos < js.size() && js[pos] == ',') ++pos;
          }
          have_shape = true;
        } else if (k == "data_offsets") {
          if (pos >= js.size() || js[pos] != '[')
            throw std::runtime_error(what + ": bad offsets " + name);
          ++pos;
          auto num = [&]() {
            skip_ws();
            size_t st = pos;
            while (pos < js.size() && isdigit((unsigned char)js[pos])) ++pos;
            if (st == pos)
              throw std::runtime_error(what + ": bad offset " + name);
            uint64_t v = std::stoull(js.substr(st, pos - st));
            skip_ws();
            return v;
          };
          uint64_t a = num();
          if (pos >= js.size() || js[pos] != ',')
            throw std::runtime_error(what + ": bad offsets " + name);
          ++pos;
          uint64_t b = num();
          skip_ws();
          if (pos >= js.size() || js[pos] != ']')
            throw std::runtime_error(what + ": bad offsets " + name);
          ++pos;
          if (b < a) throw std::runtime_error(what + ": bad offsets " + name);
          e.begin = data_base + a;
          e.end = data_base + b;
          have_off = true;
        } else {
          Impl::skip_json_value(js, pos, what, name);
        }
        skip_ws();
        if (pos < js.size() && js[pos] == ',') ++pos;
      }
      if (!have_dtype || !have_shape || !have_off)
        throw std::runtime_error(what + ": incomplete entry " + name);
      out[name] = e;
    }
    skip_ws();
    if (pos < js.size() && js[pos] == ',') ++pos;
  }
  return out;
}

inline size_t entry_elsize(const Entry& en, const std::string& full,
                           const std::string& what = "safetensors") {
  if (en.dtype == "F32") return 4;
  if (en.dtype == "BF16") return 2;
  throw std::runtime_error(what + ": tensor '" + full + "' has dtype " +
                           en.dtype + " (only F32/BF16 supported)");
}

inline int64_t entry_numel(const Entry& en, const std::string& full,
                           const std::string& what = "safetensors") {
  int64_t numel = 1;
  for (int64_t d : en.shape) {
    if (d < 0) throw std::runtime_error(what + ": bad shape '" + full + "'");
    numel *= d;
  }
  return numel;
}

// Convert one chunk of file payload (F32/BF16) into Scalar outputs.
template <class Scalar>
void convert_chunk(const char* raw, size_t elsize, Scalar* dst, size_t m) {
  const bool fast_f32 = std::is_same_v<Scalar, float> && elsize == 4;
  if (fast_f32) {
    std::memcpy(dst, raw, m * 4);
  } else if (elsize == 4) {  // F32 file -> Scalar (float or bhalf_t)
    const float* src = reinterpret_cast<const float*>(raw);
    for (size_t i = 0; i < m; ++i) dst[i] = static_cast<Scalar>(src[i]);
  } else {  // BF16 file -> Scalar via FP32
    const uint16_t* src = reinterpret_cast<const uint16_t*>(raw);
    for (size_t i = 0; i < m; ++i)
      dst[i] = static_cast<Scalar>(bf16_to_float(src[i]));
  }
}

// Stream n elements from a mapping through a bounded HostSpace chunk with a
// single deep_copy (no 2x device residency).
template <class ExecSpace, class Scalar>
void stream_into(const char* base, size_t n, size_t elsize, Scalar* dev_ptr) {
  using HostV = Kokkos::View<Scalar*, Kokkos::HostSpace>;
  HostV host("safetensors_load", n);
  constexpr size_t kChunk = size_t{1} << 18;
  for (size_t o = 0; o < n;) {
    const size_t m = std::min(n - o, kChunk);
    convert_chunk<Scalar>(base + o * elsize, elsize, host.data() + o, m);
    o += m;
  }
  Kokkos::View<Scalar*, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      dev(dev_ptr, n);
  Kokkos::deep_copy(dev, host);
}

// ---- mmap-only shard view (POSIX MAP_PRIVATE, no fallback) ----
// Read-only mapping of one safetensors shard; the OS page cache is the host
// staging. mmap failure is a hard error. Non-copyable; munmap on
// destruction. Header entries carry absolute file offsets. Public map/size
// mirror the historical MossMmap/DitMappedFile/FileReader field names so
// call sites keep working.
struct MappedFile {
  std::string path;
  const char* map = nullptr;
  size_t size = 0;

  explicit MappedFile(const std::string& p,
                      const std::string& what = "safetensors")
      : path(p) {
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error(what + ": cannot open '" + p + "'");
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error(what + ": cannot stat '" + p + "'");
    }
    if (st.st_size < 8) {
      ::close(fd);
      throw std::runtime_error(what + ": too small '" + p + "'");
    }
    size = (size_t)st.st_size;
    void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED)
      throw std::runtime_error(what + ": mmap failed for '" + p + "'");
    map = static_cast<const char*>(m);
#if defined(MADV_SEQUENTIAL)
    ::madvise(const_cast<char*>(map), size, MADV_SEQUENTIAL);
#endif
    const uint64_t hn = ld64_le(map);
    if (hn > (uint64_t{1} << 32) || 8 + hn > size) {
      ::munmap(const_cast<char*>(map), size);
      map = nullptr;
      throw std::runtime_error(what + ": bad header in '" + p + "'");
    }
    entries_ = parse_header(std::string(map + 8, map + 8 + (size_t)hn), 8 + hn,
                            what);
    for (const auto& kv : entries_) {
      if (kv.second.begin > kv.second.end || kv.second.end > size) {
        ::munmap(const_cast<char*>(map), size);
        map = nullptr;
        throw std::runtime_error(what + ": tensor '" + kv.first +
                                 "' out of range in '" + p + "'");
      }
    }
  }

  ~MappedFile() {
    if (map != nullptr) ::munmap(const_cast<char*>(map), size);
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept
      : path(std::move(o.path)),
        map(o.map),
        size(o.size),
        entries_(std::move(o.entries_)) {
    o.map = nullptr;
    o.size = 0;
  }
  MappedFile& operator=(MappedFile&&) = delete;

  const std::map<std::string, Entry>& entries() const { return entries_; }

  // Bounds-checked pointer to a tensor's payload.
  const char* data_ptr(const Entry& en, const std::string& what = "safetensors") const {
    if (en.begin > en.end || en.end > size)
      throw std::runtime_error(what + ": tensor slice out of mapped range in '" +
                               path + "'");
    return map + en.begin;
  }

  std::string read_str(uint64_t off, size_t n,
                       const std::string& what = "safetensors") const {
    if (off + n > size)
      throw std::runtime_error(what + ": read out of range in '" + path + "'");
    return std::string(map + off, n);
  }

  void read_bytes(uint64_t off, char* dst, size_t n,
                  const std::string& what = "safetensors") const {
    if (off + n > size)
      throw std::runtime_error(what + ": read out of range in '" + path + "'");
    std::memcpy(dst, map + off, n);
  }

 private:
  std::map<std::string, Entry> entries_;
};

}  // namespace Safetensors
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_SAFETENSORS_HPP
