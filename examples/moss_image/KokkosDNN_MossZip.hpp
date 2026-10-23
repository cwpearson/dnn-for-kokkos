#ifndef KOKKOSDNN_MOSS_ZIP_HPP
#define KOKKOSDNN_MOSS_ZIP_HPP

// Minimal ZIP central-directory reader for torch.save (.pth/.pt/.bin)
// checkpoints (stored entries only). Extracted from
// KokkosDNN_MossWeights.cpp so the parser is unit-testable without a real
// checkpoint; the weights loader includes this header.
//
// Central-directory layout (APPNOTE.TXT 4.3.12):
//   +10: compression method (u16)
//   +20: compressed size (u32)
//   +24: uncompressed size (u32)
//   +28: file name length `nl` (u16)
//   +30: extra field length `el` (u16)
//   +32: file comment length `cl` (u16)
//   +42: relative offset of local header (u32)

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

// mmap-backed reads (POSIX). Fallback: bounded ifstream reads with the same
// chunked staging when mmap is unavailable (mirrors the Sana loaders).
#if !defined(KOKKOSDNN_HAS_MMAP)
#if defined(__unix__) || defined(__unix) || defined(__APPLE__) || \
    defined(__linux__)
#define KOKKOSDNN_HAS_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define KOKKOSDNN_HAS_MMAP 0
#endif
#endif

namespace KokkosDNN {
namespace Moss {
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

// ---- file reader: mmap window or bounded ifstream seeks ----
struct FileReader {
  std::string path;
  size_t size = 0;
#if KOKKOSDNN_HAS_MMAP
  const char* map = nullptr;
#endif
  explicit FileReader(const std::string& p) : path(p) {
#if KOKKOSDNN_HAS_MMAP
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("moss: cannot open " + p);
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("moss: cannot stat " + p);
    }
    if (st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("moss: empty file: " + p);
    }
    size = (size_t)st.st_size;
    void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("moss: mmap failed: " + p);
    map = static_cast<const char*>(m);
#else
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("moss: cannot open " + p);
    std::streamoff e = f.tellg();
    if (e <= 0) throw std::runtime_error("moss: empty file: " + p);
    size = (size_t)e;
#endif
  }
  ~FileReader() {
#if KOKKOSDNN_HAS_MMAP
    if (map) ::munmap(const_cast<char*>(map), size);
#endif
  }
  FileReader(const FileReader&) = delete;
  FileReader& operator=(const FileReader&) = delete;

  void read_bytes(uint64_t off, char* dst, size_t n) const {
    if (off + n > size) throw std::runtime_error("moss: read out of range");
#if KOKKOSDNN_HAS_MMAP
    std::memcpy(dst, map + off, n);
#else
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("moss: cannot open " + path);
    f.seekg((std::streamoff)off);
    f.read(dst, (std::streamsize)n);
    if (!f) throw std::runtime_error("moss: truncated read");
#endif
  }
  std::string read_str(uint64_t off, size_t n) const {
    std::string s(n, '\0');
    if (n) read_bytes(off, s.data(), n);
    return s;
  }
};

// ---- minimal ZIP central-directory reader (stored entries) ----
struct ZipEntry {
  uint16_t method = 0;
  uint64_t comp_size = 0, uncomp_size = 0, local_off = 0;
};

struct ZipArchive {
  const FileReader& fr;
  std::map<std::string, ZipEntry> entries;
  explicit ZipArchive(const FileReader& r) : fr(r) {
    if (fr.size < 22) throw std::runtime_error("moss: not a zip file");
    size_t eocd = 0;
    bool found = false;
    const size_t tail =
        std::min<size_t>(fr.size, (size_t)65535 + 22);
    std::string t = fr.read_str(fr.size - tail, tail);
    for (size_t i = tail; i-- > 0;) {
      if (t[i] == 'P' && i + 22 <= tail && t[i + 1] == 'K' &&
          t[i + 2] == '\x05' && t[i + 3] == '\x06') {
        eocd = fr.size - tail + i;
        found = true;
        break;
      }
      if (i == 0) break;
    }
    if (!found) throw std::runtime_error("moss: EOCD not found (not a .pth?)");
    std::string e = fr.read_str(eocd, 22);
    uint64_t cd_size = ld32(e.data() + 12);
    uint64_t cd_off = ld32(e.data() + 16);
    size_t pos = (size_t)cd_off, end = (size_t)(cd_off + cd_size);
    if (end > fr.size) throw std::runtime_error("moss: bad central directory");
    while (pos + 46 <= end) {
      std::string h = fr.read_str(pos, 46);
      if (!(h[0] == 'P' && h[1] == 'K' && h[2] == '\x01' && h[3] == '\x02'))
        break;
      uint16_t method = ld16(h.data() + 10);
      uint64_t cs = ld32(h.data() + 20);
      uint64_t us = ld32(h.data() + 24);
      uint16_t nl = ld16(h.data() + 28);
      uint16_t el = ld16(h.data() + 30);
      uint16_t cl = ld16(h.data() + 32);
      uint64_t lh = ld32(h.data() + 42);
      std::string name = fr.read_str(pos + 46, nl);
      ZipEntry ze;
      ze.method = method;
      ze.comp_size = cs;
      ze.uncomp_size = us;
      ze.local_off = lh;
      entries[name] = ze;
      pos += 46 + nl + el + cl;
    }
    if (entries.empty())
      throw std::runtime_error("moss: no zip entries (not a .pth?)");
  }

  // Data offset of a stored entry (follows the local header).
  uint64_t data_off(const ZipEntry& ze) const {
    std::string h = fr.read_str((size_t)ze.local_off, 30);
    uint16_t nl = ld16(h.data() + 26);
    uint16_t el = ld16(h.data() + 28);
    return ze.local_off + 30 + nl + el;
  }

  std::string read_stored(const std::string& name) const {
    auto it = entries.find(name);
    if (it == entries.end())
      throw std::runtime_error("moss: missing zip entry " + name);
    if (it->second.method != 0)
      throw std::runtime_error("moss: compressed zip entry unsupported: " +
                               name);
    return fr.read_str((size_t)data_off(it->second),
                       (size_t)it->second.uncomp_size);
  }
};

}  // namespace Zip
}  // namespace Moss
}  // namespace KokkosDNN

#endif
