// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include "KokkosDNN_GlimmerWeights.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
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
float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp  = (h & 0x7C00u) >> 10;
  uint32_t mant = (h & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      // subnormal -> normalize
      exp = 1;
      while (!(mant & 0x0400u)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FFu;
      f = sign | ((exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);  // inf/nan
  } else {
    f = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

inline float bf16_to_fp32(uint16_t h) {
  uint32_t f = (uint32_t)h << 16;
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

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

// Convert n on-disk elements at raw pointer `src` (dtype `is_f32`) into
// host staging `dst`. BF16->bhalf_t and F32->float are bit-exact memcpys;
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
      for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(h[i]);
    } else {  // F16
      auto* h = reinterpret_cast<const uint16_t*>(src);
      for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(h[i]);
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
        dst[i] = static_cast<Scalar>(fp16_to_fp32(h[i]));
    }
  }
}

// Stream one [rows, cols] tensor into a device matrix view in row chunks.
template <class ExecSpace, class Scalar>
void stream_to_mat(const ShardFile& sh, const std::string& name,
                   const TensorMeta& m, size_t rows, size_t cols,
                   const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>&
                       dst) {
  if (dst.extent(0) != rows || dst.extent(1) != cols)
    throw std::runtime_error("copy_host_to_device: shape mismatch");
  check_tensor(sh, name, m, {rows, cols});
  if (rows == 0 || cols == 0) return;
  size_t rows_per_chunk =
      std::max<size_t>(1, kStreamChunkElems / std::max<size_t>(cols, 1));
#if KOKKOSDNN_HAS_MMAP
  for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
    size_t r1 = std::min(rows, r0 + rows_per_chunk);
    size_t n  = (r1 - r0) * cols;
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace> stage(
        "safetensors_chunk", r1 - r0, cols);
    convert_block<Scalar>(sh.data_ptr(m.begin + r0 * cols *
                                                  dtype_elem_bytes(m.dtype,
                                                                   name)),
                          m.dtype, stage.data(), n);
    auto sub = Kokkos::subview(
        dst, std::make_pair(r0, r1),
        Kokkos::ALL());
    Kokkos::deep_copy(sub, stage);
  }
#else
  size_t es = dtype_elem_bytes(m.dtype, name);
  std::vector<char> raw;
  for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
    size_t r1 = std::min(rows, r0 + rows_per_chunk);
    size_t n  = (r1 - r0) * cols;
    raw.resize(n * es);
    sh.read_bytes(m.begin + r0 * cols * es, raw.data(), n * es);
    Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace> stage(
        "safetensors_chunk", r1 - r0, cols);
    convert_block<Scalar>(raw.data(), m.dtype, stage.data(), n);
    auto sub = Kokkos::subview(
        dst, std::make_pair(r0, r1),
        Kokkos::ALL());
    Kokkos::deep_copy(sub, stage);
  }
#endif
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
  auto fill_mat = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t j = 0; j < h.extent(1); ++j)
        h(i, j) = static_cast<Scalar>(dist(rng));
    Kokkos::deep_copy(v, h);
  };
  auto fill_vec = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      h(i) = static_cast<Scalar>(1.0f);
    Kokkos::deep_copy(v, h);
  };
  fill_mat(w.embed);
  fill_mat(w.lm_head);
  fill_vec(w.final_norm);
  for (auto& b : w.blocks) {
    fill_mat(b.wq);
    fill_mat(b.wk);
    fill_mat(b.wv);
    fill_mat(b.wo);
    fill_mat(b.wgate_attn);
    fill_mat(b.wgate);
    fill_mat(b.wup);
    fill_mat(b.wdown);
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
                      const Kokkos::View<Scalar**, Kokkos::LayoutRight,
                                         ExecSpace>& dst) {
    Loc l = locate(name);
    stream_to_mat<ExecSpace, Scalar>(shards[l.shard], name, *l.meta, rows,
                                    cols, dst);
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
template void copy_host_to_device<Kokkos::DefaultExecutionSpace, float>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>&);
#if !KOKKOS_BHALF_T_IS_FLOAT
template void random_init<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&, uint64_t);
template void load_safetensors<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, const std::vector<std::string>&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&);
template void copy_host_to_device<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<Kokkos::Experimental::bhalf_t**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>&);
#endif

}  // namespace Glimmer
}  // namespace KokkosDNN
