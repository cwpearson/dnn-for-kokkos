// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_ZIP_HPP
#define KOKKOSDNN_ZIP_HPP

// Central ZIP central-directory reader for torch.save (.pth/.pt) checkpoints
// (stored entries only), header-only, next to KokkosDNN_GGUF.hpp.
//
// Consolidates the byte-identical ZIP parsers previously duplicated in:
//   examples/moss_image/KokkosDNN_MossZip.hpp (ld16/ld32/ld64, FileReader,
//     ZipEntry/ZipArchive),
//   examples/moss_image/KokkosDNN_MossWeights.cpp (FileReader/ZipArchive use
//     for the .pth path),
//   examples/sana/SanaWeightsPth.cpp (MappedFile/read_central_directory/
//     record_range) + examples/sana/KokkosDNN_SanaWeights.cpp (ZipEntry map),
//   examples/kokoro/KokkosDNN_KokoroPth.cpp (rd16/rd32, mmap_file, zip_index,
//     zip_data_range).
//
// Scope: stored-entry ZIP framing only (EOCD tail scan over the last
// 65535+22 bytes, central-directory walk, local-header data offsets).
// No pickle/object parsing here; the per-example pickle VMs (int- vs
// string-keyed storages) stay in examples/.
//
// Central-directory layout (APPNOTE.TXT 4.3.12, offsets within the 46-byte
// central header):
//   +10: compression method (u16)
//   +20: compressed size (u32)
//   +24: uncompressed size (u32)
//   +28: file name length `nl` (u16)
//   +30: extra field length `el` (u16)
//   +32: file comment length `cl` (u16)
//   +42: relative offset of local header (u32)
// EOCD (22 bytes): +8 records-this-disk (u16), +10 total records (u16),
// +12 central-dir size (u32), +16 central-dir offset (u32).
//
// Portable: POSIX mmap (MAP_PRIVATE); mmap failure is a hard error; there
// is no read() fallback.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Zip {

// ---- little-endian loads ----
inline uint16_t ld16(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  return (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
}
inline uint32_t ld32(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  return b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
}
inline uint64_t ld64(const char* p) {
  uint64_t lo = (uint64_t)ld32(p);
  uint64_t hi = (uint64_t)ld32(p + 4);
  return lo | (hi << 32);
}

// ---- file reader: mmap window ----
struct MappedFile {
  std::string path;
  size_t size = 0;
  const char* map = nullptr;
  explicit MappedFile(const std::string& p,
                      const std::string& what = "zip")
      : path(p) {
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error(what + ": cannot open " + p);
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error(what + ": cannot stat " + p);
    }
    if (st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error(what + ": empty file: " + p);
    }
    size = (size_t)st.st_size;
    void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED)
      throw std::runtime_error(what + ": mmap failed: " + p);
    map = static_cast<const char*>(m);
#if defined(MADV_SEQUENTIAL)
    ::madvise(const_cast<char*>(map), size, MADV_SEQUENTIAL);
#endif
  }
  ~MappedFile() {
    if (map) ::munmap(const_cast<char*>(map), size);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept
      : path(std::move(o.path)), size(o.size), map(o.map) {
    o.map = nullptr;
    o.size = 0;
  }
  MappedFile& operator=(MappedFile&&) = delete;

  void read_bytes(uint64_t off, char* dst, size_t n,
                  const std::string& what = "zip") const {
    if (off + n > size)
      throw std::runtime_error(what + ": read out of range");
    std::memcpy(dst, map + off, n);
  }
  std::string read_str(uint64_t off, size_t n,
                       const std::string& what = "zip") const {
    std::string s(n, '\0');
    if (n) read_bytes(off, s.data(), n, what);
    return s;
  }
};

// ---- minimal ZIP central-directory reader (stored entries) ----
struct Entry {
  std::string name;
  uint16_t method = 0;
  uint64_t comp_size = 0, uncomp_size = 0, local_off = 0;
};

struct Archive {
  const MappedFile& fr;
  std::string what;
  std::map<std::string, Entry> entries;
  explicit Archive(const MappedFile& r, const std::string& w = "zip")
      : fr(r), what(w) {
    if (fr.size < 22)
      throw std::runtime_error(what + ": not a zip file");
    size_t eocd = 0;
    bool found = false;
    const size_t tail = std::min<size_t>(fr.size, (size_t)65535 + 22);
    std::string t = fr.read_str(fr.size - tail, tail, what);
    for (size_t i = tail; i-- > 0;) {
      if (t[i] == 'P' && i + 22 <= tail && t[i + 1] == 'K' &&
          t[i + 2] == '\x05' && t[i + 3] == '\x06') {
        eocd = fr.size - tail + i;
        found = true;
        break;
      }
      if (i == 0) break;
    }
    if (!found)
      throw std::runtime_error(what + ": EOCD not found (not a .pth?)");
    std::string e = fr.read_str(eocd, 22, what);
    uint64_t cd_size = ld32(e.data() + 12);
    uint64_t cd_off = ld32(e.data() + 16);
    size_t pos = (size_t)cd_off, end = (size_t)(cd_off + cd_size);
    if (end > fr.size)
      throw std::runtime_error(what + ": bad central directory");
    while (pos + 46 <= end) {
      std::string h = fr.read_str(pos, 46, what);
      if (!(h[0] == 'P' && h[1] == 'K' && h[2] == '\x01' && h[3] == '\x02'))
        break;
      uint16_t method = ld16(h.data() + 10);
      uint64_t cs = ld32(h.data() + 20);
      uint64_t us = ld32(h.data() + 24);
      uint16_t nl = ld16(h.data() + 28);
      uint16_t el = ld16(h.data() + 30);
      uint16_t cl = ld16(h.data() + 32);
      uint64_t lh = ld32(h.data() + 42);
      std::string name = fr.read_str(pos + 46, nl, what);
      Entry ze;
      ze.name = name;
      ze.method = method;
      ze.comp_size = cs;
      ze.uncomp_size = us;
      ze.local_off = lh;
      entries[name] = ze;
      pos += 46 + nl + el + cl;
    }
    if (entries.empty())
      throw std::runtime_error(what + ": no zip entries (not a .pth?)");
  }

  // Data offset of a stored entry (follows the local header).
  uint64_t data_off(const Entry& ze) const {
    std::string h = fr.read_str((size_t)ze.local_off, 30, what);
    if (!(h[0] == 'P' && h[1] == 'K' && h[2] == '\x03' && h[3] == '\x04'))
      throw std::runtime_error(what + ": bad local header for '" + ze.name +
                               "'");
    uint16_t nl = ld16(h.data() + 26);
    uint16_t el = ld16(h.data() + 28);
    uint64_t start = ze.local_off + 30 + nl + el;
    if (start + ze.comp_size > fr.size)
      throw std::runtime_error(what + ": entry '" + ze.name +
                               "' overruns file");
    return start;
  }

  // (offset, size) range of a stored record's payload.
  std::pair<uint64_t, uint64_t> record_range(const std::string& name) const {
    auto it = entries.find(name);
    if (it == entries.end())
      throw std::runtime_error(what + ": missing record '" + name + "'");
    const Entry& e = it->second;
    if (e.method != 0)
      throw std::runtime_error(what + ": entry '" + name +
                               "' is compressed (only stored supported)");
    return {data_off(e), e.comp_size};
  }

  std::string read_stored(const std::string& name) const {
    auto it = entries.find(name);
    if (it == entries.end())
      throw std::runtime_error(what + ": missing zip entry " + name);
    if (it->second.method != 0)
      throw std::runtime_error(what + ": compressed zip entry unsupported: " +
                               name);
    return fr.read_str((size_t)data_off(it->second),
                       (size_t)it->second.uncomp_size, what);
  }
};

}  // namespace Zip
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_ZIP_HPP
