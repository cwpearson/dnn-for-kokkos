// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GLIMMER_GGUF_MAP_HPP
#define KOKKOSDNN_GLIMMER_GGUF_MAP_HPP

// Read-only mmap view of a GGUF checkpoint for the Glimmer loader.
//
// Why this exists: Gguf::GgufFile (src/KokkosDNN_GGUF.hpp) reads the whole
// file into a std::vector first, which for the 17 GB Q4_K_M Glimmer
// checkpoint doubles both the I/O and the peak host footprint before a
// single tensor is copied to the device. MappedGgufFile instead maps the
// file and lets the loader copy each tensor's payload straight from the
// mapping into its device view, so the host never holds a second copy of
// the checkpoint.
//
// The header parse mirrors GgufFile (same magic/version/alignment rules,
// same up-front payload range validation, same typed-KV storage for the
// keys the loader validates) so the two readers accept and reject exactly
// the same files. It is intentionally local to the Glimmer component: the
// shared reader is owned elsewhere, and the persisted tensor bytes are
// untouched by either path.
//
// Portability: the mapping is POSIX mmap with a sequential-read-ahead
// prefetcher. Non-POSIX builds report KOKKOSDNN_GLIMMER_HAS_MMAP == 0 and
// the loader falls back to Gguf::GgufFile (bounded ifstream reads).

#include <KokkosDNN_GGUF.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(KOKKOSDNN_GLIMMER_HAS_MMAP)
#if defined(__unix__) || defined(__unix) || defined(__APPLE__) || \
    defined(__linux__)
#define KOKKOSDNN_GLIMMER_HAS_MMAP 1
#else
#define KOKKOSDNN_GLIMMER_HAS_MMAP 0
#endif
#endif

#if KOKKOSDNN_GLIMMER_HAS_MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace KokkosDNN {
namespace Glimmer {
namespace Impl {

#if KOKKOSDNN_GLIMMER_HAS_MMAP

// Sink for the prefetcher's page touches, so the compiler cannot drop the
// reads. Atomic because several prefetch threads report into it.
inline void prefetch_sink_byte(unsigned char v) {
  static std::atomic<unsigned char> sink{0};
  sink.fetch_xor(v, std::memory_order_relaxed);
}

// Sequential read-ahead threads that fault the mapping into the page cache
// while the loader copies earlier tensors to the device. Disk reads and
// H2D copies then overlap instead of serializing, which is where the
// startup time goes for a checkpoint much larger than RAM.
class MappedGgufPrefetcher {
 public:
  MappedGgufPrefetcher() = default;
  ~MappedGgufPrefetcher() { stop(); }
  MappedGgufPrefetcher(const MappedGgufPrefetcher&)            = delete;
  MappedGgufPrefetcher& operator=(const MappedGgufPrefetcher&) = delete;

  // Split [begin, end) across a few threads and touch one byte per page in
  // order. Page faults (and the kernel's read-ahead) then pull the file in
  // at roughly the parallel-IO rate of the backing device.
  void start(const char* begin, const char* end) {
    stop();
    stop_.store(false, std::memory_order_relaxed);
    const size_t nbytes = (size_t)(end - begin);
    if (nbytes == 0) return;
    unsigned hw       = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const size_t nthreads = std::min<size_t>(4, hw);
    const size_t chunk    = nbytes / nthreads + 1;
    workers_.reserve(nthreads);
    for (size_t i = 0; i < nthreads; ++i) {
      const char* lo = begin + i * chunk;
      if (lo >= end) break;
      const char* hi = std::min(end, lo + chunk);
      workers_.emplace_back([this, lo, hi] { run(lo, hi); });
    }
  }

  void stop() {
    stop_.store(true, std::memory_order_relaxed);
    for (auto& t : workers_)
      if (t.joinable()) t.join();
    workers_.clear();
  }

 private:
  void run(const char* lo, const char* hi) {
    constexpr size_t kPage = 4096;  // conservative: <= every page size
    unsigned char acc      = 0;
    for (const char* p = lo; p < hi && !stop_.load(std::memory_order_relaxed);
         p += kPage)
      acc = (unsigned char)(acc + *reinterpret_cast<const unsigned char*>(p));
    prefetch_sink_byte(acc);
  }

  std::atomic<bool> stop_{true};
  std::vector<std::thread> workers_;
};

// Read-only mmap-backed GGUF file with the GgufFile access surface used by
// the Glimmer loader. Throws std::runtime_error with the same diagnostics
// for missing/corrupt/truncated files.
class MappedGgufFile {
 public:
  static constexpr size_t kAlign = 32;  // GGUF data-section alignment

  explicit MappedGgufFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("GgufFile: cannot open '" + path + "'");
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("GgufFile: cannot stat '" + path + "'");
    }
    if (st.st_size < 24) {
      ::close(fd);
      throw std::runtime_error("GgufFile: file too small '" + path + "'");
    }
    size_        = (size_t)st.st_size;
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping keeps the file alive
    if (mapped == MAP_FAILED)
      throw std::runtime_error("GgufFile: mmap failed for '" + path + "'");
    map_ = static_cast<const char*>(mapped);
#if defined(MADV_SEQUENTIAL)
    ::madvise(const_cast<char*>(map_), size_, MADV_SEQUENTIAL);
#endif
    parse();
  }

  ~MappedGgufFile() {
    prefetch_.stop();
    if (map_ != nullptr) ::munmap(const_cast<char*>(map_), size_);
  }

  MappedGgufFile(const MappedGgufFile&)            = delete;
  MappedGgufFile& operator=(const MappedGgufFile&) = delete;

  bool contains(const std::string& name) const {
    return index_.count(name) != 0;
  }

  const Gguf::GgufTensorInfo& find(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end())
      throw std::runtime_error("GgufFile: tensor not found '" + name + "'");
    return tensors_[it->second];
  }

  const std::vector<Gguf::GgufTensorInfo>& tensors() const { return tensors_; }

  // Bounds-checked pointer to a tensor's packed payload (valid until the
  // mapping is released, i.e. for the lifetime of this object).
  const uint8_t* payload(const Gguf::GgufTensorInfo& info) const {
    const size_t abs = data_start_ + (size_t)info.offset;
    if (abs + info.nbytes > size_)
      throw std::runtime_error("GgufFile: tensor '" + info.name +
                               "' payload out of range");
    return reinterpret_cast<const uint8_t*>(map_ + abs);
  }

  uint32_t version() const { return version_; }
  size_t data_start() const { return data_start_; }
  size_t file_size() const { return size_; }

  const std::map<std::string, float>& kv_f32_all() const { return f32_kv_; }
  const std::map<std::string, uint32_t>& kv_u32_all() const { return u32_kv_; }
  const std::map<std::string, std::vector<char>>& kv_bool_arrays_all() const {
    return bool_arr_kv_;
  }

  // Start/stop sequential read-ahead over the whole mapping. Idempotent;
  // the prefetcher is always stopped (and joined) by the destructor.
  void start_prefetch() { prefetch_.start(map_, map_ + size_); }
  void stop_prefetch() { prefetch_.stop(); }

 private:
  // Little-endian cursor with bounds checks; GGUF is LE-only, as is ggml.
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
      const uint64_t n = u64();
      need((size_t)n, "string");
      std::string s(p, (size_t)n);
      p += n;
      return s;
    }
  };

  // Skip/store one metadata value; arrays recurse. Strings, f32/u32 scalars
  // and bool arrays are kept: those are the hyperparameters validating
  // loaders compare against.
  void parse_value(Cursor& c, uint32_t type, const std::string& key,
                   bool store) {
    switch (type) {
      case 0: c.need(1, "u8"); c.p += 1; return;
      case 1: c.need(1, "i8"); c.p += 1; return;
      case 2: c.need(2, "u16"); c.p += 2; return;
      case 3: c.need(2, "i16"); c.p += 2; return;
      case 4: {
        const uint32_t v = c.u32();
        if (store) u32_kv_[key] = v;
        return;
      }
      case 5: c.need(4, "i32"); c.p += 4; return;
      case 6: {
        c.need(4, "f32");
        float v = 0;
        std::memcpy(&v, c.p, 4);
        c.p += 4;
        if (store) f32_kv_[key] = v;
        return;
      }
      case 7: c.need(1, "bool"); c.p += 1; return;
      case 8: {
        std::string s = c.str();
        if (store) str_kv_[key] = std::move(s);
        return;
      }
      case 9: {
        const uint32_t et = c.u32();
        const uint64_t n  = c.u64();
        if (store && et == 7) {  // BOOL array (e.g. sliding-window pattern)
          c.need((size_t)n, "bool array");
          bool_arr_kv_[key] = std::vector<char>(c.p, c.p + (size_t)n);
          c.p += (size_t)n;
          return;
        }
        for (uint64_t i = 0; i < n; ++i) parse_value(c, et, key, false);
        return;
      }
      case 10: c.need(8, "u64"); c.p += 8; return;
      case 11: c.need(8, "i64"); c.p += 8; return;
      case 12: c.need(8, "f64"); c.p += 8; return;
      default:
        throw std::runtime_error("GgufFile: bad metadata type " +
                                 std::to_string(type));
    }
  }

  void parse() {
    Cursor c{map_, map_ + size_};
    if (map_[0] != 'G' || map_[1] != 'G' || map_[2] != 'U' || map_[3] != 'F')
      throw std::runtime_error("GgufFile: bad magic (not a GGUF file)");
    c.p += 4;
    version_ = c.u32();
    if (version_ != 3)
      throw std::runtime_error("GgufFile: unsupported version " +
                               std::to_string(version_) + " (want 3)");
    const uint64_t n_tensors = c.u64();
    const uint64_t n_kv      = c.u64();

    for (uint64_t i = 0; i < n_kv; ++i) {
      std::string key  = c.str();
      const uint32_t t = c.u32();
      parse_value(c, t, key, true);
    }

    tensors_.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
      Gguf::GgufTensorInfo ti;
      ti.name   = c.str();
      ti.n_dims = c.u32();
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
      ti.dtype                = (Gguf::GgmlType)raw_type;
      // Reject unknown ggml type ids (ggml_type_size is 0 for them).
      if (Gguf::ggml_type_size(ti.dtype) == 0)
        throw std::runtime_error("GgufFile: tensor '" + ti.name +
                                 "' has unknown dtype " +
                                 std::to_string(raw_type));
      const uint32_t bl = Gguf::ggml_block_len(ti.dtype);
      if (ti.numel % bl != 0)
        throw std::runtime_error("GgufFile: tensor '" + ti.name +
                                 "' numel not a multiple of block length");
      ti.nbytes = (ti.numel / bl) * Gguf::ggml_type_size(ti.dtype);
      ti.offset = c.u64();
      index_[ti.name] = tensors_.size();
      tensors_.push_back(std::move(ti));
    }

    const size_t off = (size_t)(c.p - map_);
    data_start_      = (off + kAlign - 1) & ~(kAlign - 1);
    if (data_start_ > size_)
      throw std::runtime_error("GgufFile: header runs past EOF");
    // Validate every payload range up front so later access cannot OOB.
    for (const auto& ti : tensors_) {
      const size_t abs = data_start_ + (size_t)ti.offset;
      if (abs + ti.nbytes > size_)
        throw std::runtime_error("GgufFile: tensor '" + ti.name +
                                 "' payload out of range");
    }
  }

  const char* map_   = nullptr;
  size_t size_       = 0;
  uint32_t version_  = 0;
  size_t data_start_ = 0;
  std::vector<Gguf::GgufTensorInfo> tensors_;
  std::map<std::string, size_t> index_;
  std::map<std::string, std::string> str_kv_;
  std::map<std::string, float> f32_kv_;
  std::map<std::string, uint32_t> u32_kv_;
  std::map<std::string, std::vector<char>> bool_arr_kv_;
  MappedGgufPrefetcher prefetch_;
};

#endif  // KOKKOSDNN_GLIMMER_HAS_MMAP

}  // namespace Impl
}  // namespace Glimmer
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_GLIMMER_GGUF_MAP_HPP
