// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21SAFETENSORS_MAP_HPP
#define KOKKOSDNN_QWENIMAGE21SAFETENSORS_MAP_HPP

// Read-only mmap view of safetensors shards for the Qwen-Image-2.1 loaders.
//
// Why this exists: the previous staging story kept a second copy of the
// checkpoint in host RAM (first raw file bytes, then converted HostSpace
// views: ~32 GB on top of the ~32 GB checkpoint). This header replaces both
// with persistent mmap'd safetensors shards: the OS page cache IS the host
// staging. Drivers warm() the shard set once (open/fstat/mmap/parse, no
// payload reads, no device allocation); every phase then streams tensors
// straight from the mapping into freshly allocated device views and releases
// the device copy when the phase ends, so the encoder and DiT are never
// co-resident and the 24 GB fit is preserved with zero host-side weight
// copies beyond the page cache.
//
// The header parse reuses qwen21_parse_header (same F32/BF16 subset), and
// every tensor [begin, end) range is validated up front so later access
// cannot read out of bounds. A shard whose file changed under the mapping
// (size/mtime) stops matching find() until the next warm().
//
// Portability: the mapping is POSIX mmap with sequential read-ahead;
// mmap failure throws and there is no fallback path.

#include "KokkosDNN_QwenImage21Config.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace KokkosDNN {
namespace QwenImage21 {
namespace Impl {


// Sink for the prefetcher's page touches, so the compiler cannot drop the
// reads. Atomic because several prefetch threads report into it.
inline void qwen21_prefetch_sink_byte(unsigned char v) {
  static std::atomic<unsigned char> sink{0};
  sink.fetch_xor(v, std::memory_order_relaxed);
}

// Sequential read-ahead threads that fault the mapping into the page cache
// while the loader copies earlier tensors to the device. Disk reads and
// H2D copies then overlap instead of serializing, which is where the
// startup time goes for a checkpoint much larger than RAM.
class Qwen21MappedPrefetcher {
 public:
  Qwen21MappedPrefetcher() = default;
  ~Qwen21MappedPrefetcher() { stop(); }
  Qwen21MappedPrefetcher(const Qwen21MappedPrefetcher&) = delete;
  Qwen21MappedPrefetcher& operator=(const Qwen21MappedPrefetcher&) = delete;

  // Split [begin, end) across a few threads and touch one byte per page in
  // order. Page faults (and the kernel's read-ahead) then pull the file in
  // at roughly the parallel-IO rate of the backing device.
  void start(const char* begin, const char* end) {
    stop();
    stop_.store(false, std::memory_order_relaxed);
    const size_t nbytes = (size_t)(end - begin);
    if (nbytes == 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const size_t nthreads = std::min<size_t>(4, hw);
    const size_t chunk = nbytes / nthreads + 1;
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
    unsigned char acc = 0;
    for (const char* p = lo; p < hi && !stop_.load(std::memory_order_relaxed);
         p += kPage)
      acc = (unsigned char)(acc + *reinterpret_cast<const unsigned char*>(p));
    qwen21_prefetch_sink_byte(acc);
  }

  std::atomic<bool> stop_{true};
  std::vector<std::thread> workers_;
};


// One read-only safetensors shard: POSIX mmap (entries parsed once up
// front, data_ptr() bounds-checked). Non-copyable (the mapping is
// released in the destructor); held by unique_ptr in Qwen21MappedShardSet.
class Qwen21MappedShard {
 public:
  explicit Qwen21MappedShard(const std::string& path) : path_(path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("QwenImage21: cannot open '" + path + "'");
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("QwenImage21: cannot stat '" + path + "'");
    }
    if (st.st_size < 8) {
      ::close(fd);
      throw std::runtime_error("QwenImage21: too small '" + path + "'");
    }
    size_ = (size_t)st.st_size;
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping keeps the file alive
    if (mapped == MAP_FAILED)
      throw std::runtime_error("QwenImage21: mmap failed for '" + path + "'");
    map_ = static_cast<const char*>(mapped);
#if defined(MADV_SEQUENTIAL)
    ::madvise(const_cast<char*>(map_), size_, MADV_SEQUENTIAL);
#endif
    const uint64_t hn = qwen21_ld64(map_);
    if (8 + hn > size_) {
      ::munmap(const_cast<char*>(map_), size_);
      map_ = nullptr;
      throw std::runtime_error("QwenImage21: bad header in '" + path + "'");
    }
    entries_ = qwen21_parse_header(
        std::string(map_ + 8, map_ + 8 + (size_t)hn), 8 + hn);
    std::error_code ec;
    const auto mt = std::filesystem::last_write_time(path, ec);
    if (!ec) mtime_ = mt;
    // Validate every payload range up front so later access cannot OOB.
    for (const auto& kv : entries_) {
      if (kv.second.begin > kv.second.end || kv.second.end > size_)
        throw std::runtime_error("QwenImage21: tensor '" + kv.first +
                                 "' out of range in '" + path + "'");
    }
  }

  ~Qwen21MappedShard() {
    prefetch_.stop();
    if (map_ != nullptr) ::munmap(const_cast<char*>(map_), size_);
  }

  Qwen21MappedShard(const Qwen21MappedShard&) = delete;
  Qwen21MappedShard& operator=(const Qwen21MappedShard&) = delete;

  const std::string& path() const { return path_; }
  size_t file_size() const { return size_; }
  const std::map<std::string, Qwen21SafeEntry>& entries() const {
    return entries_;
  }

  // Bounds-checked pointer to a tensor's payload (valid for the lifetime
  // of this object). Throws when the slice reaches outside the shard
  // (shard changed under the mapping after a stale find()).
  const char* data_ptr(const Qwen21SafeEntry& en) const {
    if (en.begin > en.end || en.end > size_)
      throw std::runtime_error(
          "QwenImage21: tensor slice out of mapped range in '" + path_ +
          "' (shard changed under the mapping?)");
    return base() + en.begin;
  }

  // True when the file on disk still matches this mapping (size + mtime).
  // Never throws: stat failures report stale (fail open to a disk read).
  bool fresh() const noexcept {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path_, ec);
    if (ec || sz != (uintmax_t)size_) return false;
    const auto mt = std::filesystem::last_write_time(path_, ec);
    if (ec || mt != mtime_) return false;
    return true;
  }

  // Start/stop sequential read-ahead over the whole shard. Idempotent; the
  // prefetcher is always stopped (and joined) by the destructor.
  void start_prefetch() {
    prefetch_.start(map_, map_ + size_);
  }
  void stop_prefetch() {
    prefetch_.stop();
  }

 private:
  const char* base() const noexcept {
    return map_;
  }

  std::string path_;
  size_t size_ = 0;
  std::map<std::string, Qwen21SafeEntry> entries_;
  std::filesystem::file_time_type mtime_{};
  const char* map_ = nullptr;
  Qwen21MappedPrefetcher prefetch_;
};

// Persistent mmap'd shard set for repeat loads. A long-lived process (or a
// REPL driver) warms the checkpoint shards once; later
// Weights::load_safetensors(paths, &maps) calls stream tensors straight
// from the mappings (the OS page cache) into freshly allocated device
// views. Device staging is unaffected (encoder and DiT are still never
// co-resident); only the disk re-reads and the old host-side weight copies
// go away. Entries whose file changed under them stop matching (find()
// returns nullptr, so loads fall back to disk) until the next warm(). Not
// thread-safe (single-flight use, like the drivers).
class Qwen21MappedShardSet {
 public:
  // Map every listed shard, skipping files that are already mapped and
  // fresh. Returns the number of (re)mapped files. Throws with the path on
  // the first unreadable file (earlier files in this call stay mapped).
  size_t warm(const std::vector<std::string>& paths) {
    size_t newly = 0;
    for (const auto& p : paths) {
      const auto it = shards_.find(p);
      if (it != shards_.end() && it->second->fresh()) continue;
      shards_.insert_or_assign(p, std::make_unique<Qwen21MappedShard>(p));
      ++newly;
    }
    return newly;
  }

  // Mapped shard for path, or nullptr when absent/stale (callers fall back
  // to a disk read). Never throws.
  const Qwen21MappedShard* find(const std::string& path) const noexcept {
    const auto it = shards_.find(path);
    if (it == shards_.end() || !it->second->fresh()) return nullptr;
    return it->second.get();
  }

  // Sequential read-ahead over every mapped shard (page-cache warmup while
  // the first phase loads). Idempotent; stop joins the workers.
  void start_prefetch_all() {
    for (auto& kv : shards_) kv.second->start_prefetch();
  }
  void stop_prefetch_all() {
    for (auto& kv : shards_) kv.second->stop_prefetch();
  }

  void clear() { shards_.clear(); }
  size_t shard_count() const { return shards_.size(); }
  unsigned long long mapped_bytes() const {
    unsigned long long n = 0;
    for (const auto& kv : shards_) n += kv.second->file_size();
    return n;
  }

 private:
  std::map<std::string, std::unique_ptr<Qwen21MappedShard>> shards_;
};

// Mapping twin of the file streamer in Config.hpp: same chunking,
// conversion, and single deep_copy, but sourced from a Qwen21MappedShard
// (the OS page cache, no disk I/O, no host-side weight copy).
template <class ExecSpace, class Scalar>
void qwen21_stream_into(const Qwen21MappedShard& shard,
                        const Qwen21SafeEntry& en, size_t n, size_t elsize,
                        Scalar* dev_ptr) {
  const char* base = shard.data_ptr(en);  // range-checked
  using HostV = Kokkos::View<Scalar*, Kokkos::HostSpace>;
  HostV host("qwen21_load", n);
  constexpr size_t kChunk = size_t{1} << 18;
  for (size_t o = 0; o < n;) {
    const size_t m = std::min(n - o, kChunk);
    qwen21_convert_chunk<Scalar>(base + o * elsize, elsize, host.data() + o,
                                 m);
    o += m;
  }
  Kokkos::View<Scalar*, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dev(
      dev_ptr, n);
  Kokkos::deep_copy(dev, host);
}

}  // namespace Impl

// Persistent mmap'd shard set for repeat loads (see Impl above).
// Long-lived processes hold one, warm() it once, and pass it to every
// Weights::load_safetensors(paths, &maps) call; per-phase loads then stream
// from the OS page cache while only the phase's device weights are
// resident.
using MappedShardSet = Impl::Qwen21MappedShardSet;
}  // namespace QwenImage21
}  // namespace KokkosDNN

#endif
