// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include "KokkosDNN_GlimmerWeights.hpp"

#include "KokkosDNN_GlimmerGgufMap.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

// mmap-backed zero-copy checkpoint reads (POSIX). Fallback: chunked
// ifstream reads with the same bounded host staging when mmap is
// unavailable (e.g. non-POSIX builds).
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
namespace Glimmer {
namespace {

// ---- half-precision conversions (host side, loader only) ----
#if !KOKKOSDNN_HAS_MMAP
// Only the non-mmap (ifstream) checkpoint path uses this; guard the
// definition so mmap builds do not see an unused function.
uint64_t read_le64(std::ifstream& f) {
  unsigned char b[8];
  f.read(reinterpret_cast<char*>(b), 8);
  if (!f) throw std::runtime_error("safetensors: truncated header length");
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}
#endif

static uint64_t load_le64(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  uint64_t v   = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

// Minimal parser for the safetensors header JSON. We only need, per tensor:
// name -> {dtype, shape[], data_offsets[]}. Rather than a full JSON
// parser, scan for the known "dtype"/"shape"/"data_offsets" keys that the
// safetensors spec emits. Robust for HF checkpoints; not general JSON.
struct TensorMeta {
  std::string dtype;
  std::vector<size_t> shape;
  size_t begin = 0, end = 0;
};

// Find matching closing brace/bracket from open position (no strings-aware
// nesting beyond quotes).
size_t match_close(const std::string& s, size_t open, char oc, char cc) {
  int depth = 0;
  bool in_str = false, esc = false;
  for (size_t i = open; i < s.size(); ++i) {
    char c = s[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
    } else {
      if (c == '"') in_str = true;
      else if (c == oc) depth++;
      else if (c == cc) {
        if (--depth == 0) return i;
      }
    }
  }
  throw std::runtime_error("safetensors: unbalanced JSON");
}

// Extract raw string value for "key" within object substring.
std::string find_string_val(const std::string& obj, const std::string& key) {
  std::string q = "\"" + key + "\"";
  size_t p = obj.find(q);
  if (p == std::string::npos) return "";
  p = obj.find(':', p);
  size_t a = obj.find('"', p + 1);
  size_t b = obj.find('"', a + 1);
  return obj.substr(a + 1, b - a - 1);
}

std::vector<size_t> find_size_list(const std::string& obj,
                                   const std::string& key) {
  std::vector<size_t> out;
  std::string q = "\"" + key + "\"";
  size_t p = obj.find(q);
  if (p == std::string::npos) return out;
  p = obj.find('[', p);
  size_t e = obj.find(']', p);
  std::string inner = obj.substr(p + 1, e - p - 1);
  size_t i = 0;
  while (i < inner.size()) {
    while (i < inner.size() && !std::isdigit(inner[i])) i++;
    size_t j = i;
    while (j < inner.size() && std::isdigit(inner[j])) j++;
    if (j > i) out.push_back(std::stoull(inner.substr(i, j - i)));
    i = j;
  }
  return out;
}

std::map<std::string, TensorMeta> parse_header(const std::string& js) {
  std::map<std::string, TensorMeta> metas;
  // Top level: "name": {...}, skip __metadata__.
  size_t i = 0;
  while (i < js.size()) {
    size_t a = js.find('"', i);
    if (a == std::string::npos) break;
    size_t b = a + 1;
    // find closing quote
    std::string name;
    {
      std::string tmp;
      size_t k = a + 1;
      bool e = false;
      while (k < js.size()) {
        char c = js[k];
        if (e) {
          tmp += c;
          e = false;
        } else if (c == '\\') {
          e = true;
        } else if (c == '"') {
          break;
        } else {
          tmp += c;
        }
        k++;
      }
      b = k;
      name = tmp;
    }
    size_t colon = js.find(':', b);
    size_t ob = js.find('{', colon);
    if (ob == std::string::npos || (colon != std::string::npos && ob > js.find('"', colon + 1) && false)) {
      i = b + 1;
      continue;
    }
    // Heuristic: value is object only for tensor entries / __metadata__.
    size_t ws = colon + 1;
    while (ws < js.size() && std::isspace((unsigned char)js[ws])) ws++;
    if (ws >= js.size() || js[ws] != '{') {
      i = b + 1;
      continue;
    }
    size_t cb = match_close(js, ws, '{', '}');
    std::string obj = js.substr(ws, cb - ws + 1);
    if (name != "__metadata__" && obj.find("\"dtype\"") != std::string::npos) {
      TensorMeta m;
      m.dtype = find_string_val(obj, "dtype");
      m.shape = find_size_list(obj, "shape");
      auto off = find_size_list(obj, "data_offsets");
      if (off.size() == 2) {
        m.begin = off[0];
        m.end   = off[1];
      }
      metas[name] = m;
    }
    i = cb + 1;
    (void)a;
  }
  return metas;
}

size_t numel(const std::vector<size_t>& shape) {
  size_t n = 1;
  for (auto s : shape) n *= s;
  return n;
}

std::string shape_str(const std::vector<size_t>& s) {
  std::string o = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) o += ",";
    o += std::to_string(s[i]);
  }
  return o + "]";
}

size_t dtype_elem_bytes(const std::string& dtype, const std::string& name) {
  if (dtype == "F32") return 4;
  if (dtype == "F16" || dtype == "BF16") return 2;
  throw std::runtime_error("unsupported safetensors dtype '" + dtype +
                           "' for tensor '" + name + "'");
}

// One mapped (or fallback-readable) checkpoint shard. The tensor data
// section is never copied wholesale: conversions read directly from the
// mmap window (or bounded ifstream chunk reads on non-mmap builds) and
// stream into small host staging buffers, so peak extra host RAM is one
// chunk (~1 MiB), not a full-model F32 expansion.
struct ShardFile {
  std::string path;
  std::map<std::string, TensorMeta> metas;
  size_t data_start = 0;
  size_t file_size  = 0;
#if KOKKOSDNN_HAS_MMAP
  const char* map = nullptr;
#endif

  explicit ShardFile(const std::string& p) : path(p) {
#if KOKKOSDNN_HAS_MMAP
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + p);
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("cannot stat " + p);
    }
    if (st.st_size < 8) {
      ::close(fd);
      throw std::runtime_error("safetensors: file too small: " + p);
    }
    file_size      = (size_t)st.st_size;
    void* m        = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("mmap failed for " + p);
    map = static_cast<const char*>(m);
    uint64_t n = load_le64(map);
    if (8 + n > file_size) {
      ::munmap(const_cast<char*>(map), file_size);
      map = nullptr;
      throw std::runtime_error("truncated JSON in " + p);
    }
    data_start = (size_t)(8 + n);
    metas      = parse_header(std::string(map + 8, (size_t)n));
#else
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    uint64_t n = read_le64(f);
    std::string js(n, '\0');
    f.read(js.data(), (std::streamsize)n);
    if (!f) throw std::runtime_error("truncated JSON in " + p);
    f.seekg(0, std::ios::end);
    std::streamoff end = f.tellg();
    if (end < 0) throw std::runtime_error("cannot stat " + p);
    file_size  = (size_t)end;
    data_start = (size_t)(8 + n);
    if (data_start > file_size)
      throw std::runtime_error("truncated JSON in " + p);
    metas = parse_header(js);
#endif
  }

  ~ShardFile() {
#if KOKKOSDNN_HAS_MMAP
    if (map) ::munmap(const_cast<char*>(map), file_size);
#endif
  }

  ShardFile(const ShardFile&)            = delete;
  ShardFile& operator=(const ShardFile&) = delete;
  ShardFile(ShardFile&& o) noexcept
      : path(std::move(o.path)),
        metas(std::move(o.metas)),
        data_start(o.data_start),
        file_size(o.file_size)
#if KOKKOSDNN_HAS_MMAP
        ,
        map(o.map)
#endif
  {
#if KOKKOSDNN_HAS_MMAP
    o.map       = nullptr;
#endif
    o.file_size = 0;
  }
  ShardFile& operator=(ShardFile&&) = delete;

  // Read n raw bytes at data-section offset `off` into dst. Only used by
  // the non-mmap fallback streaming paths below; the mmap path uses
  // data_ptr() instead.
#if !KOKKOSDNN_HAS_MMAP
  void read_bytes(size_t off, char* dst, size_t n) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg((std::streamoff)(data_start + off));
    f.read(dst, (std::streamsize)n);
    if (!f) throw std::runtime_error("safetensors: truncated tensor data");
  }
#endif

#if KOKKOSDNN_HAS_MMAP
  // Direct pointer into the mapped window (bounds-checked by caller).
  const char* data_ptr(size_t off) const { return map + data_start + off; }
#endif
};

// Streaming chunk size (elements per host staging buffer). Bounds extra
// host RAM to ~chunk * sizeof(Scalar) regardless of model size.
inline constexpr size_t kStreamChunkElems = size_t{1} << 18;

// Validate dtype/shape/offsets of one tensor against its expected dense
// shape. Returns the on-disk element size.
size_t check_tensor(const ShardFile& sh, const std::string& name,
                    const TensorMeta& m, const std::vector<size_t>& want) {
  size_t es = dtype_elem_bytes(m.dtype, name);
  if (m.shape != want)
    throw std::runtime_error("shape mismatch for '" + name +
                             "': checkpoint has " + shape_str(m.shape) +
                             ", expected " + shape_str(want));
  if (m.end < m.begin)
    throw std::runtime_error("bad data_offsets for '" + name + "'");
  size_t nbytes = m.end - m.begin;
  size_t expect = numel(want) * es;
  if (nbytes != expect)
    throw std::runtime_error("'" + m.dtype + "' size mismatch for '" + name +
                             "': got " + std::to_string(nbytes) + " bytes" +
                             ", expected " + std::to_string(expect));
  if (sh.data_start + m.end > sh.file_size)
    throw std::runtime_error("tensor data out of range for '" + name + "'");
  return es;
}

// Convert n on-disk elements at raw pointer `src` into host staging `dst`
// (dense 1-D norm path). BF16->bhalf_t and F32->float are bit-exact memcpys;
// other pairs go through an exact F32 intermediate (BF16/F16->F32 is exact,
// then a single host-side cast to Scalar, same rounding as before).
template <class Scalar>
void convert_block(const char* src, const std::string& dtype, Scalar* dst,
                   size_t n) {
  if constexpr (std::is_same_v<Scalar, float>) {
    if (dtype == "F32") {
      std::memcpy(dst, src, n * 4);
    } else if (dtype == "BF16") {
      auto* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < n; ++i) dst[i] = Gguf::bf16_to_float_host(h[i]);
    } else {  // F16
      auto* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < n; ++i) dst[i] = Gguf::fp16_to_float_host(h[i]);
    }
  } else {
    static_assert(sizeof(Scalar) == 2, "half-precision Scalar must be 2B");
    if (dtype == "BF16") {
      std::memcpy(dst, src, n * 2);  // zero-copy, bit-exact
    } else if (dtype == "F32") {
      const float* f = reinterpret_cast<const float*>(src);
      for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<Scalar>(f[i]);
    } else {  // F16 via exact F32 intermediate
      auto* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<Scalar>(Gguf::fp16_to_float_host(h[i]));
    }
  }
}

// Map a safetensors dtype tag to the packed ggml dtype (identity packing:
// F32/F16/BF16 payloads are already dense row-major, which is exactly the
// packed layout for unquantized dtypes).
Gguf::GgmlType packed_dtype_for(const std::string& dtype,
                                const std::string& name) {
  if (dtype == "F32") return Gguf::GgmlType::F32;
  if (dtype == "F16") return Gguf::GgmlType::F16;
  if (dtype == "BF16") return Gguf::GgmlType::BF16;
  throw std::runtime_error("unsupported safetensors dtype '" + dtype +
                           "' for tensor '" + name + "'");
}

// Stream one [rows, cols] tensor into a packed device weight, preserving the
// on-disk dtype (no conversion: the packed layout of F32/F16/BF16 is dense
// row-major). Chunked like the 1-D norm path below so at most ~1 MiB of host
// staging plus one target-sized device deep_copy chunk is live at a time.
template <class ExecSpace>
void stream_to_packed(const ShardFile& sh, const std::string& name,
                      const TensorMeta& m, size_t rows, size_t cols,
                      Gguf::QuantWeight<ExecSpace>& dst) {
  const size_t es = check_tensor(sh, name, m, {rows, cols});
  const Gguf::GgmlType dt = packed_dtype_for(m.dtype, name);
  const size_t rb         = Gguf::ggml_row_bytes(dt, cols);
  if (rb != cols * es)
    throw std::runtime_error("packing mismatch for '" + name + "'");
  dst.dtype = dt;
  dst.rows  = rows;
  dst.cols  = cols;
  dst.data =
      Kokkos::View<uint8_t*, ExecSpace>("safetensors_packed", rows * rb);
  if (rows == 0 || cols == 0) return;
  const size_t rows_per_chunk =
      std::max<size_t>(1, kStreamChunkElems / std::max<size_t>(cols, 1));
  for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
    const size_t r1 = std::min(rows, r0 + rows_per_chunk);
    const size_t n  = (r1 - r0) * cols * es;
    auto sub        = Kokkos::subview(
        dst.data, std::make_pair(r0 * rb, r1 * rb));
#if KOKKOSDNN_HAS_MMAP
    const auto* bytes = reinterpret_cast<const uint8_t*>(
        sh.data_ptr(m.begin + r0 * cols * es));
    Kokkos::View<const uint8_t*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        stage(bytes, n);
    Kokkos::deep_copy(sub, stage);
#else
    std::vector<char> raw(n);
    sh.read_bytes(m.begin + r0 * cols * es, raw.data(), n);
    Kokkos::View<uint8_t*, Kokkos::HostSpace> stage("safetensors_chunk", n);
    std::memcpy(stage.data(), raw.data(), n);
    Kokkos::deep_copy(sub, stage);
#endif
  }
}

// Stream one [n] tensor into a device vector view in chunks.
template <class ExecSpace, class Scalar>
void stream_to_vec(const ShardFile& sh, const std::string& name,
                   const TensorMeta& m,
                   const Kokkos::View<Scalar*, ExecSpace>& dst) {
  size_t n = dst.extent(0);
  check_tensor(sh, name, m, {n});
  if (n == 0) return;
#if KOKKOSDNN_HAS_MMAP
  for (size_t o = 0; o < n; o += kStreamChunkElems) {
    size_t c = std::min(kStreamChunkElems, n - o);
    Kokkos::View<Scalar*, Kokkos::HostSpace> stage("safetensors_chunk", c);
    convert_block<Scalar>(
        sh.data_ptr(m.begin + o * dtype_elem_bytes(m.dtype, name)), m.dtype,
        stage.data(), c);
    auto sub = Kokkos::subview(
        dst, std::make_pair(o, o + c));
    Kokkos::deep_copy(sub, stage);
  }
#else
  size_t es = dtype_elem_bytes(m.dtype, name);
  std::vector<char> raw;
  for (size_t o = 0; o < n; o += kStreamChunkElems) {
    size_t c = std::min(kStreamChunkElems, n - o);
    raw.resize(c * es);
    sh.read_bytes(m.begin + o * es, raw.data(), c * es);
    Kokkos::View<Scalar*, Kokkos::HostSpace> stage("safetensors_chunk", c);
    convert_block<Scalar>(raw.data(), m.dtype, stage.data(), c);
    auto sub = Kokkos::subview(
        dst, std::make_pair(o, o + c));
    Kokkos::deep_copy(sub, stage);
  }
#endif
}

}  // namespace

template <class ExecSpace, class Scalar>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  if (dst.extent(0) != rows || dst.extent(1) != cols)
    throw std::runtime_error("copy_host_to_device: shape mismatch");
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      h(i, j) = static_cast<Scalar>(host[i * cols + j]);
  Kokkos::deep_copy(dst, h);
}

template <class ExecSpace, class Scalar>
void copy_host_vec_to_device(
    const std::vector<float>& host,
    const Kokkos::View<Scalar*, ExecSpace>& dst) {
  if (dst.extent(0) != host.size())
    throw std::runtime_error("copy_host_vec_to_device: shape mismatch");
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < host.size(); ++i)
    h(i) = static_cast<Scalar>(host[i]);
  Kokkos::deep_copy(dst, h);
}

template <class ExecSpace, class Scalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed) {
  w.allocate(cfg);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  // Packed F32 2-D weight with small random values (raw-float packing is
  // exact, so quant_gemm() on it matches a dense reference bit-for-bit).
  auto fill_packed = [&](Gguf::QuantWeight<ExecSpace>& qw, size_t rows,
                         size_t cols) {
    qw.dtype = Gguf::GgmlType::F32;
    qw.rows  = rows;
    qw.cols  = cols;
    const size_t nbytes = qw.bytes();
    qw.data = Kokkos::View<uint8_t*, ExecSpace>("random_init", nbytes);
    Kokkos::View<uint8_t*, Kokkos::HostSpace> h("random_init_host", nbytes);
    float* f = reinterpret_cast<float*>(h.data());
    for (size_t i = 0; i < rows * cols; ++i) f[i] = dist(rng);
    Kokkos::deep_copy(qw.data, h);
  };
  auto fill_vec = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      h(i) = static_cast<Scalar>(1.0f);
    Kokkos::deep_copy(v, h);
  };
  const size_t D = (size_t)cfg.dim, V = (size_t)cfg.vocab_size;
  const size_t hq = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  const size_t FF  = (size_t)cfg.ffn_dim;
  fill_packed(w.embed, V, D);
  fill_packed(w.lm_head, V, D);
  fill_vec(w.final_norm);
  for (auto& b : w.blocks) {
    fill_packed(b.wq, hq, D);
    fill_packed(b.wk, hkv, D);
    fill_packed(b.wv, hkv, D);
    fill_packed(b.wo, D, hq);
    fill_packed(b.wgate_attn, hq, D);
    fill_packed(b.wgate, FF, D);
    fill_packed(b.wup, FF, D);
    fill_packed(b.wdown, D, FF);
    fill_vec(b.input_norm);
    fill_vec(b.post_attn_norm);
    fill_vec(b.pre_ffn_norm);
    fill_vec(b.post_ffn_norm);
  }
}

std::vector<std::string> list_checkpoint_tensors(
    const std::vector<std::string>& files) {
  if (files.empty()) throw std::runtime_error("no checkpoint files given");
  std::vector<std::string> out;
  for (auto& fp : files) {
    ShardFile sh(fp);
    for (auto& [name, m] : sh.metas) {
      (void)m;
      out.push_back(name);
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

template <class ExecSpace, class Scalar>
void load_safetensors(const Config& cfg,
                      const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w) {
  w.allocate(cfg);
  // Map every shard read-only (mmap; bounded ifstream reads where mmap is
  // unavailable) and index tensors across shards. Tensors stream per-tensor
  // from the mapping in chunks, so at most one ~1 MiB staging buffer plus
  // one target-sized device deep_copy chunk is live at a time — no
  // full-model (or full-tensor F32) host expansion. Later shards win on
  // duplicate names, as before.
  if (files.empty()) throw std::runtime_error("no checkpoint files given");
  std::vector<ShardFile> shards;
  shards.reserve(files.size());
  for (auto& fp : files) shards.emplace_back(fp);
  struct Loc {
    size_t shard;
    const TensorMeta* meta;
  };
  std::map<std::string, Loc> found;
  for (size_t s = 0; s < shards.size(); ++s)
    for (auto& [name, m] : shards[s].metas) found[name] = {s, &m};

  auto locate = [&](const std::string& name) -> Loc {
    auto it = found.find(name);
    if (it == found.end())
      throw std::runtime_error("tensor not found in checkpoint: " + name);
    return it->second;
  };
  auto load_mat = [&](const std::string& name, size_t rows, size_t cols,
                      Gguf::QuantWeight<ExecSpace>& dst) {
    Loc l = locate(name);
    stream_to_packed<ExecSpace>(shards[l.shard], name, *l.meta, rows, cols,
                               dst);
  };
  auto load_vec = [&](const std::string& name,
                      const Kokkos::View<Scalar*, ExecSpace>& dst) {
    Loc l = locate(name);
    stream_to_vec<ExecSpace, Scalar>(shards[l.shard], name, *l.meta, dst);
  };

  load_mat("model.language_model.embed_tokens.weight", (size_t)cfg.vocab_size,
           (size_t)cfg.dim, w.embed);
  load_vec("model.language_model.norm.weight", w.final_norm);
  // Untied head (tie_word_embeddings=false): the checkpoint carries a
  // separate lm_head tensor; a missing tensor is an error (never aliased).
  // Accept both the outer "lm_head.weight" (nn.Linear on the causal-LM
  // wrapper) and the nested "model.language_model.lm_head.weight" spellings.
  if (found.count("lm_head.weight")) {
    load_mat("lm_head.weight", (size_t)cfg.vocab_size, (size_t)cfg.dim,
             w.lm_head);
  } else if (found.count("model.language_model.lm_head.weight")) {
    load_mat("model.language_model.lm_head.weight", (size_t)cfg.vocab_size,
             (size_t)cfg.dim, w.lm_head);
  } else {
    locate("lm_head.weight");  // rethrow the informative missing-tensor error
  }
  const int hq  = cfg.n_heads * cfg.head_dim;
  const int hkv = cfg.n_kv_heads * cfg.head_dim;
  for (int l = 0; l < cfg.n_layers; ++l) {
    std::string p =
        "model.language_model.layers." + std::to_string(l) + ".";
    auto& b = w.blocks[(size_t)l];
    load_mat(p + "self_attn.q_proj.weight", (size_t)hq, (size_t)cfg.dim, b.wq);
    load_mat(p + "self_attn.k_proj.weight", (size_t)hkv, (size_t)cfg.dim,
             b.wk);
    load_mat(p + "self_attn.v_proj.weight", (size_t)hkv, (size_t)cfg.dim,
             b.wv);
    load_mat(p + "self_attn.o_proj.weight", (size_t)cfg.dim, (size_t)hq, b.wo);
    load_mat(p + "self_attn.gate_proj.weight", (size_t)hq, (size_t)cfg.dim,
             b.wgate_attn);
    load_mat(p + "mlp.gate_proj.weight", (size_t)cfg.ffn_dim, (size_t)cfg.dim,
             b.wgate);
    load_mat(p + "mlp.up_proj.weight", (size_t)cfg.ffn_dim, (size_t)cfg.dim,
             b.wup);
    load_mat(p + "mlp.down_proj.weight", (size_t)cfg.dim,
             (size_t)cfg.ffn_dim, b.wdown);
    load_vec(p + "input_layernorm.weight", b.input_norm);
    load_vec(p + "post_attention_layernorm.weight", b.post_attn_norm);
    load_vec(p + "pre_feedforward_layernorm.weight", b.pre_ffn_norm);
    load_vec(p + "post_feedforward_layernorm.weight", b.post_ffn_norm);
    // NOTE: no self_attn.qk_norm.weight exists upstream (with_scale=False);
    // embed_norm is likewise weightless. Neither is looked up.
  }
}

// ---- GGUF loading (llama.cpp `tn()` tensor naming) ----

namespace {

// Checkpoint reader for the GGUF branch: the mapped, zero-copy reader where
// POSIX mmap is available, the shared full-file reader otherwise.
#if KOKKOSDNN_GLIMMER_HAS_MMAP
using CheckpointFile = Impl::MappedGgufFile;
#else
using CheckpointFile = Gguf::GgufFile;
#endif

// Opt-in checkpoint-load diagnostics: KOKKOSDNN_LOAD_TIMING=1 makes the
// loaders below print per-phase wall times (header parse, per-tensor host
// staging, device allocation, H2D copy) to stderr. Off by default; the
// steady_clock reads are skipped entirely when the variable is unset.
bool load_timing_enabled() {
  static const bool on = std::getenv("KOKKOSDNN_LOAD_TIMING") != nullptr;
  return on;
}

using LoadClock = std::chrono::steady_clock;

inline double since_s(LoadClock::time_point t0) {
  return std::chrono::duration<double>(LoadClock::now() - t0).count();
}

// Scoped wall-time report for one phase of a loader.
class LoadPhase {
 public:
  explicit LoadPhase(const char* what) : what_(what) {
    if (load_timing_enabled()) start_ = LoadClock::now();
  }
  LoadPhase(const LoadPhase&)            = delete;
  LoadPhase& operator=(const LoadPhase&) = delete;
  ~LoadPhase() {
    if (load_timing_enabled())
      std::fprintf(stderr, "[load] %-24s %8.3f s\n", what_, since_s(start_));
  }

 private:
  const char* what_;
  LoadClock::time_point start_{};
};

// Sums one phase's wall time into a counter (per-tensor helpers below).
class LoadAccum {
 public:
  explicit LoadAccum(double& slot) : slot_(slot) {
    if (load_timing_enabled()) start_ = LoadClock::now();
  }
  LoadAccum(const LoadAccum&)            = delete;
  LoadAccum& operator=(const LoadAccum&) = delete;
  ~LoadAccum() {
    if (load_timing_enabled()) slot_ += since_s(start_);
  }

 private:
  double& slot_;
  LoadClock::time_point start_{};
};

// Aggregate per-tensor counters (summed over every tensor of a checkpoint).
struct LoadStats {
  double host_stage_s   = 0;  // host-side staging/permutation of bytes
  double device_alloc_s = 0;
  double h2d_s          = 0;  // host -> device copy
  size_t tensors        = 0;
  size_t bytes          = 0;
};
LoadStats g_load_stats;

void report_load_stats(const char* tag) {
  if (!load_timing_enabled()) return;
  std::fprintf(stderr,
               "[load] %-24s %8.3f s (%zu tensors, %.2f GiB)\n", tag,
               g_load_stats.h2d_s + g_load_stats.host_stage_s +
                   g_load_stats.device_alloc_s,
               g_load_stats.tensors,
               (double)g_load_stats.bytes / (1024.0 * 1024.0 * 1024.0));
  std::fprintf(stderr, "[load]   host staging         %8.3f s\n",
               g_load_stats.host_stage_s);
  std::fprintf(stderr, "[load]   device alloc         %8.3f s\n",
               g_load_stats.device_alloc_s);
  std::fprintf(stderr, "[load]   H2D copy             %8.3f s\n",
               g_load_stats.h2d_s);
  g_load_stats = LoadStats{};
}

// GGUF leaf names follow llama.cpp's `tn()` convention: per-layer tensors
// live under "blk.{l}." with a ".weight" suffix.
std::string gguf_blk(int l, const std::string& leaf) {
  return "blk." + std::to_string(l) + "." + leaf;
}

std::string gguf_shape_str(const Gguf::GgufTensorInfo& info) {
  std::string o = "[";
  if (info.n_dims == 1) {
    o += std::to_string((size_t)info.numel);
  } else if (info.n_dims == 2) {
    o += std::to_string(info.rows()) + "," + std::to_string(info.cols());
  } else {
    for (uint32_t d = 0; d < info.n_dims; ++d) {
      if (d != 0) o += ",";
      o += std::to_string(info.dims[d]);
    }
  }
  return o + "]:" + Gguf::ggml_type_name(info.dtype);
}

// Resolve stem+".weight", falling back to the bare stem. Returns nullptr
// when neither is present (caller reports the missing tensor). Templated on
// the checkpoint reader (mapped or Gguf::GgufFile) — both expose the same
// contains()/find() surface.
template <class File>
const Gguf::GgufTensorInfo* gguf_resolve(const File& file,
                                         const std::string& stem) {
  if (file.contains(stem + ".weight")) return &file.find(stem + ".weight");
  if (file.contains(stem)) return &file.find(stem);
  return nullptr;
}

template <class File>
const Gguf::GgufTensorInfo& gguf_require(const File& file,
                                         const std::string& stem) {
  if (const auto* info = gguf_resolve(file, stem)) return *info;
  throw std::runtime_error("tensor not found in checkpoint: '" + stem +
                           ".weight' (also tried '" + stem + "')");
}

namespace Impl {

// Invert the llama.cpp conversion _unpermute_for_rope
// (conversion/muse_glimmer.py) on PACKED rows: GGUF q/k rows are stored in
// ggml interleaved (NORM) layout while the model consumes the HF permuted
// (rotate_half) layout matched by rope_neoX_inplace. This is transformers'
// _permute_for_rope:
//   out[h*hd + i*(hd/2)+j, :] = in[h*hd + j*2+i, :], i in {0,1}.
// Whole packed rows of `row_bytes` each are copied through a scratch buffer
// (no dequantization). head_dim must be even (Config enforces it).
//
// Rows [first_row, first_row + nrows) of the destination are produced from
// the same rows of `src`, so a caller can stream the tensor in head-aligned
// chunks instead of staging all of it on the host.
inline void permute_packed_rows_for_rope_range(
    const uint8_t* src, uint8_t* dst, size_t first_row, size_t nrows,
    size_t total_rows, size_t row_bytes, size_t n_heads, size_t head_dim) {
  if (n_heads * head_dim != total_rows || row_bytes == 0 || head_dim == 0 ||
      head_dim % 2 != 0 || first_row % head_dim != 0 ||
      nrows % head_dim != 0 || nrows == 0)
    throw std::runtime_error("permute_packed_rows_for_rope: bad dims");
  const size_t half = head_dim / 2;
  for (size_t r = 0; r < nrows; ++r) {
    const size_t g = first_row + r;  // destination row within the tensor
    const size_t h = g / head_dim;
    const size_t i = (g % head_dim) / half;  // 0 = first half, 1 = second
    const size_t j = g % half;
    const size_t s = h * head_dim + j * 2 + i;
    std::memcpy(dst + r * row_bytes, src + s * row_bytes, row_bytes);
  }
}

}  // namespace Impl

// Device copy of `n` raw bytes from a host-readable pointer (the checkpoint
// mapping, or the fallback reader's buffer). The host side is a temporary
// unmanaged view, so no second copy of the bytes is materialized.
template <class DstView>
void gguf_copy_bytes_to_device(const DstView& dst, const uint8_t* src,
                               size_t nbytes) {
  if (nbytes == 0) return;
  Kokkos::View<const uint8_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      stage(src, nbytes);
  Kokkos::deep_copy(dst, stage);
}

// Dequantize one packed payload to host FP32 with an informative error when
// the dtype is outside the compute set (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K).
// Same block decoders as Gguf::dequant_tensor_to_float, sourced from a raw
// payload pointer so it works for any checkpoint reader. Only used for the
// 1-D norms, which are a few KiB each.
void gguf_dequant_payload_to_float(const uint8_t* src,
                                   const Gguf::GgufTensorInfo& info,
                                   std::vector<float>& out) {
  out.resize((size_t)info.numel);
  if (info.numel == 0) return;
  float* dst = out.data();
  switch (info.dtype) {
    case Gguf::GgmlType::F32:
      std::memcpy(dst, src, out.size() * 4);
      return;
    case Gguf::GgmlType::F16: {
      const uint16_t* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < out.size(); ++i)
        dst[i] = Gguf::fp16_to_float_host(h[i]);
      return;
    }
    case Gguf::GgmlType::BF16: {
      const uint16_t* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < out.size(); ++i)
        dst[i] = Gguf::bf16_to_float_host(h[i]);
      return;
    }
    case Gguf::GgmlType::Q8_0: {
      float tmp[32];
      for (uint64_t b = 0, n = info.numel / 32; b < n; ++b) {
        Gguf::BlockQ80 blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        Gguf::dequant_q80_block(&blk, tmp);
        std::memcpy(dst + b * 32, tmp, sizeof(tmp));
      }
      return;
    }
    case Gguf::GgmlType::Q4_K: {
      float tmp[256];
      for (uint64_t b = 0, n = info.numel / 256; b < n; ++b) {
        Gguf::BlockQ4K blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        Gguf::dequant_q4k_block(&blk, tmp);
        std::memcpy(dst + b * 256, tmp, sizeof(tmp));
      }
      return;
    }
    case Gguf::GgmlType::Q5_K: {
      float tmp[256];
      for (uint64_t b = 0, n = info.numel / 256; b < n; ++b) {
        Gguf::BlockQ5K blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        Gguf::dequant_q5k_block(&blk, tmp);
        std::memcpy(dst + b * 256, tmp, sizeof(tmp));
      }
      return;
    }
    case Gguf::GgmlType::Q6_K: {
      float tmp[256];
      for (uint64_t b = 0, n = info.numel / 256; b < n; ++b) {
        Gguf::BlockQ6K blk;
        std::memcpy(&blk, src + b * sizeof(blk), sizeof(blk));
        Gguf::dequant_q6k_block(&blk, tmp);
        std::memcpy(dst + b * 256, tmp, sizeof(tmp));
      }
      return;
    }
    default:
      throw std::runtime_error("unsupported dtype " +
                               Gguf::ggml_type_name(info.dtype));
  }
}

// Load one 2-D tensor into a packed device weight (no host dequantization:
// the payload moves verbatim and stays quantized for quant_gemv/quant_gemm).
// Throws when the dtype is outside the compute set, with the tensor name in
// the message (mirrors the old dequant path's error context).
//
// The bytes go straight from the checkpoint mapping into the device view:
// an unmanaged host view over the payload is the deep_copy source, so the
// loader never stages a copy of the tensor in host RAM. The q/k rope
// re-permutation is the one case that needs a host scratch buffer; it is
// streamed in head-aligned chunks bounded by kPermuteChunkBytes.
template <class ExecSpace, class File>
void gguf_load_mat(const File& file, const std::string& stem, size_t rows,
                   size_t cols, Gguf::QuantWeight<ExecSpace>& dst,
                   bool permute_rope = false, size_t rope_heads = 0,
                   size_t rope_head_dim = 0) {
  const auto& info = gguf_require(file, stem);
  if (info.n_dims != 2 || info.rows() != rows || info.cols() != cols)
    throw std::runtime_error("shape mismatch for '" + info.name +
                             "': checkpoint has " + gguf_shape_str(info) +
                             ", expected [" + std::to_string(rows) + "," +
                             std::to_string(cols) + "]");
  try {
    Gguf::Impl::quant_check(info.dtype, rows, cols, info.nbytes,
                            "gguf_load_mat");
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot decode '" + info.name + "': " + e.what());
  }
  const uint8_t* src     = file.payload(info);
  const size_t row_bytes = Gguf::ggml_row_bytes(info.dtype, cols);
  Gguf::QuantWeight<ExecSpace> w;
  w.dtype = info.dtype;
  w.rows  = rows;
  w.cols  = cols;
  {
    LoadAccum acc(g_load_stats.device_alloc_s);
    w.data = Kokkos::View<uint8_t*, ExecSpace>("Glimmer::weight", info.nbytes);
  }
  if (info.nbytes != 0) {
    if (!permute_rope) {
      LoadAccum acc(g_load_stats.h2d_s);
      gguf_copy_bytes_to_device(w.data, src, (size_t)info.nbytes);
    } else {
      if (rope_head_dim == 0)
        throw std::runtime_error("gguf_load_mat: rope_head_dim required for '"
                                 + info.name + "'");
      // Chunk by whole heads so every chunk is independently permutable.
      constexpr size_t kPermuteChunkBytes = size_t{64} << 20;
      const size_t bytes_per_head = row_bytes * rope_head_dim;
      const size_t heads_per_chunk =
          std::max<size_t>(1, kPermuteChunkBytes /
                                  std::max<size_t>(bytes_per_head, 1));
      const size_t rows_per_chunk = heads_per_chunk * rope_head_dim;
      std::vector<uint8_t> stage;
      stage.reserve(
          std::min(rows, rows_per_chunk) * std::max<size_t>(row_bytes, 1));
      for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
        const size_t r1 = std::min(rows, r0 + rows_per_chunk);
        stage.resize((r1 - r0) * row_bytes);
        {
          LoadAccum acc(g_load_stats.host_stage_s);
          Impl::permute_packed_rows_for_rope_range(
              src, stage.data(), r0, r1 - r0, rows, row_bytes, rope_heads,
              rope_head_dim);
        }
        auto sub = Kokkos::subview(w.data, std::make_pair(r0 * row_bytes,
                                                         r1 * row_bytes));
        LoadAccum acc(g_load_stats.h2d_s);
        gguf_copy_bytes_to_device(sub, stage.data(), stage.size());
      }
    }
  }
  g_load_stats.tensors += 1;
  g_load_stats.bytes += (size_t)info.nbytes;
  dst = w;
}

// Load one 1-D norm tensor, subtracting the conversion-time norm shift.
// Conversion adds +shift to HF-centered weights (GGUF rms_norm is x/rms*w
// while HF CenteredRMSNorm is x/rms*(1+w)); shift is 1 for the four layer
// norms and 0 for the final norm (norm_shift in conversion/muse_glimmer.py).
template <class ExecSpace, class Scalar, class File>
void gguf_load_norm(const File& file, const std::string& stem,
                    const Kokkos::View<Scalar*, ExecSpace>& dst, float shift) {
  const auto& info = gguf_require(file, stem);
  if (info.numel != dst.extent(0))
    throw std::runtime_error("shape mismatch for '" + info.name +
                             "': checkpoint has " + gguf_shape_str(info) +
                             ", expected [" + std::to_string(dst.extent(0)) +
                             "]");
  std::vector<float> tmp;
  try {
    gguf_dequant_payload_to_float(file.payload(info), info, tmp);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot decode '" + info.name + "': " + e.what());
  }
  if (shift != 0.0f)
    for (auto& v : tmp) v -= shift;
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < tmp.size(); ++i)
    h(i) = static_cast<Scalar>(tmp[i]);
  Kokkos::deep_copy(dst, h);
}

// Find a typed KV by exact key, else by unique suffix match (tolerates a
// renamed arch prefix). Throws when missing or ambiguous.
template <class Map>
typename Map::mapped_type gguf_find_kv(const Map& all,
                                       const std::string& exact,
                                       const std::string& suffix,
                                       const std::string& what) {
  auto it = all.find(exact);
  if (it != all.end()) return it->second;
  const typename Map::mapped_type* hit = nullptr;
  std::string hit_key;
  for (const auto& [k, v] : all) {
    if (k.size() >= suffix.size() &&
        k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0) {
      if (hit != nullptr)
        throw std::runtime_error("GGUF metadata: ambiguous " + what +
                                 " keys '" + hit_key + "' and '" + k + "'");
      hit     = &v;
      hit_key = k;
    }
  }
  if (hit == nullptr)
    throw std::runtime_error("GGUF metadata missing: '" + exact +
                             "' (no key ending in '" + suffix + "')");
  return *hit;
}

// Validate conversion-written hyperparameters against cfg. The GGUF file is
// authoritative about its own bytes, but the Model runs off cfg: any
// disagreement is a hard error, never a silent override.
template <class File>
void gguf_validate_metadata(const File& file, const Config& cfg) {
  constexpr const char* arch = "muse-glimmer";
  constexpr double rel_tol   = 1e-6;
  auto close = [&](const std::string& what, double have, double want) {
    const double tol = rel_tol * std::max(1.0, std::fabs(want));
    if (std::fabs(have - want) > tol)
      throw std::runtime_error("GGUF metadata mismatch: '" + what + "' is " +
                               std::to_string(have) + " but Config expects " +
                               std::to_string(want));
  };
  const std::string prefix(arch);
  const float logit_scale =
      gguf_find_kv(file.kv_f32_all(), prefix + ".logit_scale", ".logit_scale",
                   "logit scale");
  close("logit_scale", (double)logit_scale, cfg.output_multiplier);
  const float softcap = gguf_find_kv(file.kv_f32_all(),
                                     prefix + ".final_logit_softcapping",
                                     ".final_logit_softcapping", "softcap");
  close("final_logit_softcapping", (double)softcap,
        (double)cfg.final_logit_softcap);
  const uint32_t swin = gguf_find_kv(file.kv_u32_all(),
                                     prefix + ".attention.sliding_window",
                                     ".attention.sliding_window",
                                     "sliding window");
  if (swin != (uint32_t)cfg.sliding_window)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attention.sliding_window' is " +
        std::to_string(swin) + " but Config expects " +
        std::to_string(cfg.sliding_window));
  const std::vector<char> pat =
      gguf_find_kv(file.kv_bool_arrays_all(),
                   prefix + ".attention.sliding_window_pattern",
                   ".attention.sliding_window_pattern", "s_w pattern");
  if (pat.size() != (size_t)cfg.n_layers)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attention.sliding_window_pattern' has " +
        std::to_string(pat.size()) + " entries but Config has " +
        std::to_string(cfg.n_layers) + " layers");
  for (int l = 0; l < cfg.n_layers; ++l) {
    const char b = pat[(size_t)l];
    if (b != 0 && b != 1)
      throw std::runtime_error(
          "GGUF metadata mismatch: 'attention.sliding_window_pattern'[" +
          std::to_string(l) + "] is " + std::to_string((int)b) +
          " (expected 0/1)");
    const bool expect_sliding = !cfg.is_full(l);
    if ((b != 0) != expect_sliding)
      throw std::runtime_error(
          "GGUF metadata mismatch: 'attention.sliding_window_pattern'[" +
          std::to_string(l) + "] marks a " +
          std::string(b != 0 ? "sliding" : "full") +
          " layer but Config marks it " +
          std::string(expect_sliding ? "sliding" : "full"));
  }
}

}  // namespace

template <class ExecSpace, class Scalar>
void load_gguf(const Config& cfg, const std::string& path,
               Weights<ExecSpace, Scalar>& w) {
  LoadClock::time_point t_total;
  if (load_timing_enabled()) t_total = LoadClock::now();
  w.allocate(cfg);
  // Map the checkpoint read-only where the platform supports it (POSIX), so
  // tensor payloads are copied to the device straight out of the mapping
  // instead of through a full-file host buffer. Elsewhere the shared reader
  // (bounded ifstream reads) takes over.
  std::unique_ptr<CheckpointFile> file_owner;
  {
    LoadPhase p("GGUF map/read + parse");
    file_owner = std::make_unique<CheckpointFile>(path);
  }
  const CheckpointFile& file = *file_owner;  // throws on corrupt file
#if KOKKOSDNN_GLIMMER_HAS_MMAP
  // Read the rest of the file into the page cache in the background while
  // the tensors below are copied to the device: disk reads and H2D copies
  // then overlap instead of serializing.
  file_owner->start_prefetch();
#endif
  {
    LoadPhase p("GGUF metadata check");
    gguf_validate_metadata(file, cfg);
  }
  const size_t D = (size_t)cfg.dim, V = (size_t)cfg.vocab_size;
  gguf_load_mat<ExecSpace>(file, "token_embd", V, D, w.embed);
  // Final norm: the conversion shift is 0 (only *layernorm names shift).
  gguf_load_norm<ExecSpace, Scalar>(file, "output_norm", w.final_norm, 0.0f);
  // Untied head (tie_word_embeddings=false): the checkpoint carries a
  // separate output tensor; a missing tensor is an error (never aliased).
  // A tied config still falls back to aliasing the embedding table.
  if (gguf_resolve(file, "output") != nullptr) {
    gguf_load_mat<ExecSpace>(file, "output", V, D, w.lm_head);
  } else if (cfg.tie_word_embeddings) {
    w.lm_head = w.embed;  // tied: alias the embedding table
  } else {
    gguf_require(file, "output");  // rethrow the missing-tensor error
  }
  const size_t hq  = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  const size_t HD  = (size_t)cfg.head_dim;
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b = w.blocks[(size_t)l];
    gguf_load_norm<ExecSpace, Scalar>(file, gguf_blk(l, "attn_norm"),
                                      b.input_norm, 1.0f);
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "attn_q"), hq, D, b.wq,
                                     true, (size_t)cfg.n_heads, HD);
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "attn_k"), hkv, D,
                                     b.wk, true, (size_t)cfg.n_kv_heads, HD);
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "attn_v"), hkv, D,
                                     b.wv);
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "attn_output"), D, hq,
                                     b.wo);
    // Attention output gate: sigmoid(gate) * attn_out before o_proj.
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "attn_gate"), hq, D,
                                     b.wgate_attn);
    gguf_load_norm<ExecSpace, Scalar>(file, gguf_blk(l, "post_attention_norm"),
                                      b.post_attn_norm, 1.0f);
    gguf_load_norm<ExecSpace, Scalar>(file, gguf_blk(l, "ffn_norm"),
                                      b.pre_ffn_norm, 1.0f);
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "ffn_gate"),
                                     (size_t)cfg.ffn_dim, D, b.wgate);
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "ffn_down"), D,
                                     (size_t)cfg.ffn_dim, b.wdown);
    gguf_load_mat<ExecSpace>(file, gguf_blk(l, "ffn_up"),
                                     (size_t)cfg.ffn_dim, D, b.wup);
    gguf_load_norm<ExecSpace, Scalar>(file, gguf_blk(l, "post_ffw_norm"),
                                      b.post_ffn_norm, 1.0f);
    // NOTE: blk.{l}.attn_q_norm / attn_k_norm are intentionally never
    // looked up. Conversion synthesizes them (full qk_scale_factor / ones)
    // to absorb the q pre-scale into a *scaled* RMSNorm, while the Model
    // runs a *scaleless* QK norm (ones stand-in) plus the separate 3.87
    // multiply: qk_rmsnorm(q2, ones, eps)*3.87 == rms(q2, eps)*3.87 and
    // qk_rmsnorm(k2, ones, eps) == rms(k2, eps) with the same eps, so
    // loading the synthesized tensors would be exactly equivalent. Files
    // with or without them load identically.
  }
#if KOKKOSDNN_GLIMMER_HAS_MMAP
  // Every payload copy above reads straight out of the mapping, so make sure
  // all transfers have completed before the mapping is released (and before
  // the prefetcher threads are joined).
  Kokkos::fence();
  file_owner->stop_prefetch();
#endif
  report_load_stats("tensor load total");
  if (load_timing_enabled())
    std::fprintf(stderr, "[load] %-24s %8.3f s\n", "load_gguf end-to-end",
                 since_s(t_total));
}

std::vector<std::string> list_gguf_tensors(const std::string& path) {
  CheckpointFile file(path);  // header only: no tensor data is touched
  std::vector<std::string> out;
  for (const auto& t : file.tensors()) out.push_back(t.name);
  std::sort(out.begin(), out.end());
  return out;
}

// Explicit instantiations for the default execution space (defined here so
// the .cpp compiles them). All callers use DefaultExecutionSpace, so no
// backend needs enumerating here: whichever space Kokkos was configured with
// (Serial, OpenMP, Cuda, HIP, ...) is covered. DefaultHostExecutionSpace is
// deliberately not instantiated: nothing calls these functions on a host
// space, and on CPU-only builds it aliases DefaultExecutionSpace, where a
// second instantiation would be a duplicate-definition error.
// On builds without a native bhalf type Kokkos aliases bhalf_t to float
// (KOKKOS_BHALF_T_IS_FLOAT), so the bhalf_t lines are skipped there to
// avoid duplicate explicit instantiations.
template void random_init<Kokkos::DefaultExecutionSpace, float>(const Config&, Weights<Kokkos::DefaultExecutionSpace, float>&, uint64_t);
template void load_safetensors<Kokkos::DefaultExecutionSpace, float>(const Config&, const std::vector<std::string>&, Weights<Kokkos::DefaultExecutionSpace, float>&);
template void load_gguf<Kokkos::DefaultExecutionSpace, float>(const Config&, const std::string&, Weights<Kokkos::DefaultExecutionSpace, float>&);
template void copy_host_to_device<Kokkos::DefaultExecutionSpace, float>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>&);
#if !KOKKOS_BHALF_T_IS_FLOAT
template void random_init<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&, uint64_t);
template void load_safetensors<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, const std::vector<std::string>&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&);
template void load_gguf<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, const std::string&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&);
template void copy_host_to_device<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<Kokkos::Experimental::bhalf_t**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>&);
#endif

}  // namespace Glimmer
}  // namespace KokkosDNN
