// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#include "KokkosDNN_InklingWeights.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

// mmap-backed checkpoint reads (::mmap; Linux and macOS only).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace KokkosDNN {
namespace Inkling {
namespace {

// ---- host-side helpers (loader only) ----
static uint64_t load_le64(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  uint64_t v   = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

// Minimal parser for the safetensors header JSON. We only need, per tensor:
// name -> {dtype, shape[], data_offsets[]}.
struct TensorMeta {
  std::string dtype;
  std::vector<size_t> shape;
  size_t begin = 0, end = 0;
};

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
    while (i < inner.size() && !(inner[i] >= '0' && inner[i] <= '9')) i++;
    size_t j = i;
    while (j < inner.size() && inner[j] >= '0' && inner[j] <= '9') j++;
    if (j > i) out.push_back((size_t)std::stoull(inner.substr(i, j - i)));
    i = j;
  }
  return out;
}

std::map<std::string, TensorMeta> parse_header(const std::string& js) {
  std::map<std::string, TensorMeta> metas;
  size_t i = 0;
  while (i < js.size()) {
    size_t a = js.find('"', i);
    if (a == std::string::npos) break;
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
      name = tmp;
      i = k + 1;
    }
    size_t colon = js.find(':', a);
    if (colon == std::string::npos) break;
    size_t ws = colon + 1;
    while (ws < js.size() && (js[ws] == ' ' || js[ws] == '\t' ||
                              js[ws] == '\n' || js[ws] == '\r'))
      ws++;
    if (ws >= js.size() || js[ws] != '{') continue;
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

// One mapped checkpoint shard. The tensor data section is never copied
// wholesale: tensors stream from the mmap window in chunks, so peak extra
// host RAM is one chunk (~1 MiB), not a full-model expansion.
struct ShardFile {
  std::string path;
  std::map<std::string, TensorMeta> metas;
  size_t data_start = 0;
  size_t file_size  = 0;
  const char* map = nullptr;

  explicit ShardFile(const std::string& p) : path(p) {
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
  }

  ~ShardFile() {
    if (map) ::munmap(const_cast<char*>(map), file_size);
  }

  ShardFile(const ShardFile&)            = delete;
  ShardFile& operator=(const ShardFile&) = delete;
  ShardFile(ShardFile&& o) noexcept
      : path(std::move(o.path)),
        metas(std::move(o.metas)),
        data_start(o.data_start),
        file_size(o.file_size),
        map(o.map) {
    o.map       = nullptr;
    o.file_size = 0;
  }
  ShardFile& operator=(ShardFile&&) = delete;

  const char* data_ptr(size_t off) const { return map + data_start + off; }
};

inline constexpr size_t kStreamChunkElems = size_t{1} << 18;

size_t dtype_elem_bytes(const std::string& dtype, const std::string& name) {
  if (dtype == "F32") return 4;
  if (dtype == "F16" || dtype == "BF16") return 2;
  throw std::runtime_error("unsupported safetensors dtype '" + dtype +
                           "' for tensor '" + name + "'");
}

// Map a safetensors dtype tag to the packed ggml dtype (identity packing:
// F32/F16/BF16 payloads are already dense row-major, which is exactly the
// packed layout for unquantized dtypes; quantized payloads are opaque
// packed blocks copied verbatim).
Gguf::GgmlType packed_dtype_for(const std::string& dtype,
                                const std::string& name) {
  if (dtype == "F32") return Gguf::GgmlType::F32;
  if (dtype == "F16") return Gguf::GgmlType::F16;
  if (dtype == "BF16") return Gguf::GgmlType::BF16;
  if (dtype == "Q8_0") return Gguf::GgmlType::Q8_0;
  if (dtype == "Q4_K") return Gguf::GgmlType::Q4_K;
  if (dtype == "Q5_K") return Gguf::GgmlType::Q5_K;
  if (dtype == "Q6_K") return Gguf::GgmlType::Q6_K;
  if (dtype == "IQ4_NL") return Gguf::GgmlType::IQ4_NL;
  throw std::runtime_error("unsupported safetensors dtype '" + dtype +
                           "' for tensor '" + name + "'");
}

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

// Convert n on-disk elements at `src` into host staging `dst` (dense path).
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
      std::memcpy(dst, src, n * 2);
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

// Stream one [rows, cols] tensor into a packed device weight, preserving
// the on-disk dtype (identity packing for F32/F16/BF16; verbatim packed
// blocks for quantized dtypes). Chunked so at most ~1 MiB of host staging
// plus one target-sized device deep_copy chunk is live at a time.
template <class ExecSpace>
void stream_to_packed(const ShardFile& sh, const std::string& name,
                      const TensorMeta& m, size_t rows, size_t cols,
                      Gguf::QuantWeight<ExecSpace>& dst) {
  const Gguf::GgmlType dt = packed_dtype_for(m.dtype, name);
  if (!Gguf::gguf_can_compute(dt))
    throw std::runtime_error("unsupported dtype '" + m.dtype +
                             "' for tensor '" + name + "'");
  const size_t nbytes =
      (numel(m.shape) / Gguf::ggml_block_len(dt)) * Gguf::ggml_type_size(dt);
  if (numel(m.shape) != rows * cols)
    throw std::runtime_error("shape mismatch for '" + name +
                             "': checkpoint has " + shape_str(m.shape) +
                             ", expected [" + std::to_string(rows) + "," +
                             std::to_string(cols) + "]");
  if (m.end < m.begin || m.end - m.begin != nbytes)
    throw std::runtime_error("bad data_offsets/size for '" + name + "'");
  if (sh.data_start + m.end > sh.file_size)
    throw std::runtime_error("tensor data out of range for '" + name + "'");
  const size_t rb = Gguf::ggml_row_bytes(dt, cols);
  if (rows * rb != nbytes)
    throw std::runtime_error("packing mismatch for '" + name + "'");
  dst.dtype = dt;
  dst.rows  = rows;
  dst.cols  = cols;
  dst.data =
      Kokkos::View<uint8_t*, ExecSpace>("safetensors_packed", nbytes);
  if (nbytes == 0) return;
  const size_t rows_per_chunk =
      std::max<size_t>(1, (size_t{64} << 20) / std::max<size_t>(rb, 1));
  for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
    const size_t r1 = std::min(rows, r0 + rows_per_chunk);
    const size_t n  = (r1 - r0) * rb;
    auto sub =
        Kokkos::subview(dst.data, std::make_pair(r0 * rb, r1 * rb));
    const auto* bytes = reinterpret_cast<const uint8_t*>(
        sh.data_ptr(m.begin + r0 * rb));
    Kokkos::View<const uint8_t*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        stage(bytes, n);
    Kokkos::deep_copy(sub, stage);
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
  for (size_t o = 0; o < n; o += kStreamChunkElems) {
    size_t c = std::min(kStreamChunkElems, n - o);
    Kokkos::View<Scalar*, Kokkos::HostSpace> stage("safetensors_chunk", c);
    convert_block<Scalar>(
        sh.data_ptr(m.begin + o * dtype_elem_bytes(m.dtype, name)), m.dtype,
        stage.data(), c);
    auto sub = Kokkos::subview(dst, std::make_pair(o, o + c));
    Kokkos::deep_copy(sub, stage);
  }
}

// Stream one [rows, cols] dense tensor into a device matrix view in chunks.
template <class ExecSpace, class Scalar>
void stream_to_mat(
    const ShardFile& sh, const std::string& name, const TensorMeta& m,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  const size_t rows = dst.extent(0), cols = dst.extent(1);
  // Mirror the GGUF convention: single-query res-query tensors may persist
  // as 1-D [cols] (same bytes as [1, cols]).
  std::vector<size_t> want = {rows, cols};
  if (rows == 1 && m.shape.size() == 1 && m.shape[0] == cols) want = m.shape;
  const size_t es = check_tensor(sh, name, m, want);
  if (rows == 0 || cols == 0) return;
  const size_t elems_per_chunk = std::max<size_t>(1, kStreamChunkElems);
  const size_t rows_per_chunk =
      std::max<size_t>(1, elems_per_chunk / std::max<size_t>(cols, 1));
  for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
    const size_t r1 = std::min(rows, r0 + rows_per_chunk);
    const size_t c  = (r1 - r0) * cols;
    Kokkos::View<Scalar*, Kokkos::HostSpace> stage("safetensors_chunk", c);
    convert_block<Scalar>(sh.data_ptr(m.begin + r0 * cols * es), m.dtype,
                          stage.data(), c);
    auto sub = Kokkos::subview(dst, std::make_pair(r0, r1), Kokkos::ALL);
    auto h = Kokkos::create_mirror_view(sub);
    for (size_t i = 0; i < r1 - r0; ++i)
      for (size_t j = 0; j < cols; ++j) h(i, j) = stage[i * cols + j];
    Kokkos::deep_copy(sub, h);
  }
}

// Transposed dense load: the checkpoint holds [rows, cols] (torch order)
// but the Model math wants [cols, rows] (loop control downs/forget/gain,
// rel_proj — see the GGUF transpose loader). Small tensors: materialize
// the whole tensor on the host once, then write transposed.
template <class ExecSpace, class Scalar>
void stream_to_mat_T(
    const ShardFile& sh, const std::string& name, const TensorMeta& m,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  const size_t rows = dst.extent(0), cols = dst.extent(1);
  if (rows == 0 || cols == 0) return;
  // Checkpoint shape is the transpose of dst.
  const size_t es = check_tensor(sh, name, m, {cols, rows});
  Kokkos::View<Scalar*, Kokkos::HostSpace> stage("safetensors_chunk",
                                                 rows * cols);
  convert_block<Scalar>(sh.data_ptr(m.begin), m.dtype, stage.data(),
                        rows * cols);
  (void)es;
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j) h(i, j) = stage[j * rows + i];
  Kokkos::deep_copy(dst, h);
}

}  // namespace

// Device transpose via host mirrors (loader-time only; the transposed
// tensors are KBs). dst must be [src.cols, src.rows].
template <class ExecSpace, class Scalar>
void transpose_into(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& src,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  if (dst.extent(0) != src.extent(1) || dst.extent(1) != src.extent(0))
    throw std::runtime_error("transpose_into: shape mismatch");
  if (src.size() == 0) return;
  auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src);
  auto hd = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < src.extent(0); ++i)
    for (size_t j = 0; j < src.extent(1); ++j) hd(j, i) = hs(i, j);
  Kokkos::deep_copy(dst, hd);
}

// Split one folded expert-major packed weight ([experts*per, cols]) into
// per-expert QuantWeight views sharing the same device payload. Slice
// starts are row-aligned (expert boundaries are multiples of whole rows),
// so every slice keeps the dtype's row-packing alignment.
template <class ExecSpace>
void slice_experts(const Gguf::QuantWeight<ExecSpace>& folded, size_t experts,
                   std::vector<Gguf::QuantWeight<ExecSpace>>& per) {
  if (experts == 0 || folded.rows % experts != 0)
    throw std::runtime_error("slice_experts: bad expert folding");
  const size_t per_rows = folded.rows / experts;
  const size_t rb       = folded.row_bytes();
  per.resize(experts);
  for (size_t e = 0; e < experts; ++e) {
    per[e].dtype = folded.dtype;
    per[e].rows  = per_rows;
    per[e].cols  = folded.cols;
    per[e].data  = Kokkos::View<uint8_t*, ExecSpace>(Kokkos::subview(
        folded.data, std::make_pair(e * per_rows * rb, (e + 1) * per_rows * rb)));
  }
}

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
      h(i) = static_cast<Scalar>(dist(rng));
    Kokkos::deep_copy(v, h);
  };
  auto fill_mat = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t j = 0; j < h.extent(1); ++j)
        h(i, j) = static_cast<Scalar>(dist(rng));
    Kokkos::deep_copy(v, h);
  };
  const size_t D = (size_t)cfg.d_model, V = (size_t)cfg.vocab_size;
  const size_t hq = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  fill_packed(w.tok_emb, V, D);
  fill_packed(w.out_w, V, D);
  fill_vec(w.tok_norm);
  fill_vec(w.out_norm);
  fill_mat(w.out_res_q);
  for (auto& b : w.blocks) {
    fill_packed(b.wgate_attn, hq, D);
    fill_packed(b.wq, hq, D);
    fill_packed(b.wk, hkv, D);
    fill_packed(b.wv, hkv, D);
    fill_packed(b.wo, D, hq);
    fill_packed(b.wrel, (size_t)cfg.n_heads * (size_t)cfg.d_rel, D);
    fill_packed(b.wsink, (size_t)cfg.n_heads, D);
    if (is_dyn_conv_layer(cfg, (int)(&b - w.blocks.data()))) {
      const size_t dk = (size_t)cfg.n_kv_heads * (size_t)cfg.sconv_kernel;
      fill_packed(b.k_conv_proj, dk, D);
      fill_packed(b.v_conv_proj, dk, D);
    }
    size_t erows = 0, ecols = 0;
    {
      TensorId id{Tensor::FfnGateUpExps, 0};
      expected_shape(id, cfg, erows, ecols);
      const size_t per_gu = erows / (size_t)cfg.n_experts;
      b.gate_up_exps_q.resize((size_t)cfg.n_experts);
      for (auto& q : b.gate_up_exps_q) fill_packed(q, per_gu, ecols);
      expected_shape(TensorId{Tensor::FfnDownExps, 0}, cfg, erows, ecols);
      const size_t per_d = erows / (size_t)cfg.n_experts;
      b.down_exps_q.resize((size_t)cfg.n_experts);
      for (auto& q : b.down_exps_q) fill_packed(q, per_d, ecols);
    }
    fill_vec(b.attn_norm);
    fill_vec(b.q_norm);
    fill_vec(b.k_norm);
    fill_vec(b.v_norm);
    fill_vec(b.post_attn_norm);
    fill_mat(b.mlp_res_q);
    fill_vec(b.router_bias);
    fill_mat(b.k_conv);
    fill_mat(b.v_conv);
    transpose_into(b.k_conv, b.k_conv_static);
    transpose_into(b.v_conv, b.v_conv_static);
    fill_mat(b.post_attn_conv);
    fill_mat(b.q_scale);
    fill_mat(b.rel_proj);
    fill_mat(b.rel_scale);
    fill_mat(b.attn_res_q);
    fill_mat(b.router);
    fill_mat(b.post_mlp_conv);
    fill_mat(b.loop_q_down);
    fill_mat(b.loop_q_forget);
    fill_mat(b.loop_q_gain);
    fill_mat(b.loop_k_down);
    fill_mat(b.loop_k_forget);
    fill_mat(b.loop_k_gain);
    fill_mat(b.loop_v_down);
    fill_mat(b.loop_v_forget);
    fill_mat(b.loop_v_gain);
    fill_mat(b.loop_mlp_down);
    fill_mat(b.loop_mlp_forget);
    fill_mat(b.loop_mlp_gain);
    fill_mat(b.msg_query);
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

// Assumed HF safetensors name for a resolved tensor (mechanical 1:1 mapping
// of the GGUF leaves; see the header for the caveat).
static std::string hf_name(const TensorId& id) {
  std::string n;
  if (id.layer >= 0)
    n = "model.layers." + std::to_string(id.layer) + ".";
  else
    n = "model.";
  n += tensor_stem(id.kind);
  n += tensor_suffix(id.kind);
  return n;
}

template <class ExecSpace, class Scalar>
void load_safetensors(const Config& cfg,
                      const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w) {
  w.allocate(cfg);
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
  auto load_mat = [&](const TensorId& id, size_t rows, size_t cols,
                      Gguf::QuantWeight<ExecSpace>& dst) {
    const std::string name = hf_name(id);
    Loc l = locate(name);
    stream_to_packed<ExecSpace>(shards[l.shard], name, *l.meta, rows, cols,
                                dst);
  };
  auto load_vec = [&](const TensorId& id,
                      const Kokkos::View<Scalar*, ExecSpace>& dst) {
    if (dst.size() == 0) return;
    const std::string name = hf_name(id);
    Loc l = locate(name);
    stream_to_vec<ExecSpace, Scalar>(shards[l.shard], name, *l.meta, dst);
  };
  auto load_dense = [&](const TensorId& id,
                        const Kokkos::View<Scalar**, Kokkos::LayoutRight,
                                           ExecSpace>& dst) {
    if (dst.size() == 0) return;
    const std::string name = hf_name(id);
    Loc l = locate(name);
    stream_to_mat<ExecSpace, Scalar>(shards[l.shard], name, *l.meta, dst);
  };
  auto load_dense_T = [&](const TensorId& id,
                          const Kokkos::View<Scalar**, Kokkos::LayoutRight,
                                             ExecSpace>& dst) {
    if (dst.size() == 0) return;
    const std::string name = hf_name(id);
    Loc l = locate(name);
    stream_to_mat_T<ExecSpace, Scalar>(shards[l.shard], name, *l.meta, dst);
  };

  size_t rows = 0, cols = 0;
  TensorId id{Tensor::TokenEmbd, -1};
  expected_shape(id, cfg, rows, cols);
  load_mat(id, rows, cols, w.tok_emb);
  load_vec({Tensor::TokenEmbdNorm, -1}, w.tok_norm);
  load_mat({Tensor::Output, -1}, rows, cols, w.out_w);
  load_vec({Tensor::OutputNorm, -1}, w.out_norm);
  load_dense({Tensor::OutputResQuery, -1}, w.out_res_q);
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b = w.blocks[(size_t)l];
    auto need = [&](Tensor t, size_t r, size_t c,
                    Gguf::QuantWeight<ExecSpace>& dst) {
      load_mat({t, l}, r, c, dst);
    };
    expected_shape({Tensor::AttnQ, l}, cfg, rows, cols);
    const size_t hq = rows;
    need(Tensor::AttnGate, hq, (size_t)cfg.d_model, b.wgate_attn);
    need(Tensor::AttnQ, hq, (size_t)cfg.d_model, b.wq);
    expected_shape({Tensor::AttnK, l}, cfg, rows, cols);
    need(Tensor::AttnK, rows, (size_t)cfg.d_model, b.wk);
    need(Tensor::AttnV, rows, (size_t)cfg.d_model, b.wv);
    need(Tensor::AttnOutput, (size_t)cfg.d_model, hq, b.wo);
    expected_shape({Tensor::AttnRel, l}, cfg, rows, cols);
    need(Tensor::AttnRel, rows, (size_t)cfg.d_model, b.wrel);
    expected_shape({Tensor::AttnSinkProj, l}, cfg, rows, cols);
    need(Tensor::AttnSinkProj, rows, (size_t)cfg.d_model, b.wsink);
    if (!is_dyn_conv_layer(cfg, l)) {
      load_vec({Tensor::AttnNorm, l}, b.attn_norm);
    } else {
      expected_shape({Tensor::AttnKConvProj, l}, cfg, rows, cols);
      need(Tensor::AttnKConvProj, rows, (size_t)cfg.d_model,
           b.k_conv_proj);
      need(Tensor::AttnVConvProj, rows, (size_t)cfg.d_model,
           b.v_conv_proj);
      load_vec({Tensor::AttnQInpNorm, l}, b.q_norm);
      load_vec({Tensor::AttnKInpNorm, l}, b.k_norm);
      load_vec({Tensor::AttnVInpNorm, l}, b.v_norm);
    }
    expected_shape({Tensor::FfnGateUpExps, l}, cfg, rows, cols);
    {
      Gguf::QuantWeight<ExecSpace> folded;
      need(Tensor::FfnGateUpExps, rows, cols, folded);
      slice_experts(folded, (size_t)cfg.n_experts, b.gate_up_exps_q);
    }
    expected_shape({Tensor::FfnDownExps, l}, cfg, rows, cols);
    {
      Gguf::QuantWeight<ExecSpace> folded;
      need(Tensor::FfnDownExps, rows, cols, folded);
      slice_experts(folded, (size_t)cfg.n_experts, b.down_exps_q);
    }
    load_vec({Tensor::FfnNorm, l}, b.post_attn_norm);
    load_dense({Tensor::FfnResQuery, l}, b.mlp_res_q);
    load_vec({Tensor::ExpProbsB, l}, b.router_bias);
    load_dense({Tensor::AttnKConv, l}, b.k_conv);
    load_dense({Tensor::AttnVConv, l}, b.v_conv);
    transpose_into(b.k_conv, b.k_conv_static);
    transpose_into(b.v_conv, b.v_conv_static);
    load_dense({Tensor::AttnPostConv, l}, b.post_attn_conv);
    load_dense({Tensor::AttnQClip, l}, b.q_scale);
    load_dense_T({Tensor::AttnRelBias, l}, b.rel_proj);
    load_dense({Tensor::AttnRelClip, l}, b.rel_scale);
    load_dense({Tensor::AttnResQuery, l}, b.attn_res_q);
    load_dense({Tensor::FfnGateInp, l}, b.router);
    load_dense({Tensor::FfnPostConv, l}, b.post_mlp_conv);
    if (is_loop_layer(cfg, l)) {
      load_dense_T({Tensor::LoopQCtrlDown, l}, b.loop_q_down);
      load_dense_T({Tensor::LoopQCtrlForget, l}, b.loop_q_forget);
      load_dense_T({Tensor::LoopQCtrlGain, l}, b.loop_q_gain);
      load_dense_T({Tensor::LoopKCtrlDown, l}, b.loop_k_down);
      load_dense_T({Tensor::LoopKCtrlForget, l}, b.loop_k_forget);
      load_dense_T({Tensor::LoopKCtrlGain, l}, b.loop_k_gain);
      load_dense_T({Tensor::LoopVCtrlDown, l}, b.loop_v_down);
      load_dense_T({Tensor::LoopVCtrlForget, l}, b.loop_v_forget);
      load_dense_T({Tensor::LoopVCtrlGain, l}, b.loop_v_gain);
      load_dense_T({Tensor::LoopFfnCtrlDown, l}, b.loop_mlp_down);
      load_dense_T({Tensor::LoopFfnCtrlForget, l}, b.loop_mlp_forget);
      load_dense_T({Tensor::LoopFfnCtrlGain, l}, b.loop_mlp_gain);
      load_dense({Tensor::LoopMsgQuery, l}, b.msg_query);
    }
  }
  Kokkos::fence();
}

// ---- GGUF loading (llama.cpp `tn()` tensor naming) ----

namespace {

// Opt-in checkpoint-load diagnostics: KOKKOSDNN_LOAD_TIMING=1 makes the
// loader below print per-phase wall times to stderr. Off by default.
bool load_timing_enabled() {
  static const bool on = std::getenv("KOKKOSDNN_LOAD_TIMING") != nullptr;
  return on;
}

using LoadClock = std::chrono::steady_clock;

inline double since_s(LoadClock::time_point t0) {
  return std::chrono::duration<double>(LoadClock::now() - t0).count();
}

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

std::string gguf_shape_str(const Gguf::GgufTensorInfo& info) {
  std::string o = "[";
  if (info.n_dims == 1) {
    o += std::to_string((size_t)info.numel);
  } else {
    for (uint32_t d = 0; d < info.n_dims; ++d) {
      if (d != 0) o += "x";
      o += std::to_string(info.dims[d]);
    }
  }
  return o + "]:" + Gguf::ggml_type_name(info.dtype);
}

// Resolve stem+".weight" (or stem+".bias" for the expert bias), falling
// back to the bare stem. Returns nullptr when neither is present.
const Gguf::GgufTensorInfo* gguf_resolve(const Gguf::GgufFile& file,
                                         const TensorId& id) {
  const std::string suffixed = full_name(id);
  if (file.contains(suffixed)) return &file.find(suffixed);
  std::string bare =
      (id.layer >= 0 ? "blk." + std::to_string(id.layer) + "." : "") +
      tensor_stem(id.kind);
  if (file.contains(bare)) return &file.find(bare);
  return nullptr;
}

const Gguf::GgufTensorInfo& gguf_require(const Gguf::GgufFile& file,
                                         const TensorId& id) {
  if (const auto* info = gguf_resolve(file, id)) return *info;
  throw std::runtime_error("tensor not found in checkpoint: '" +
                           full_name(id) + "'");
}

// Device copy of `n` raw bytes from a host-readable pointer (the checkpoint
// mapping). The host side is a temporary unmanaged view, so no second copy
// of the bytes is materialized. One tensor per call: the loader streams
// tensors one at a time, never holding a full-file copy.
template <class DstView>
void gguf_copy_bytes_to_device(const DstView& dst, const uint8_t* src,
                               size_t nbytes) {
  if (nbytes == 0) return;
  Kokkos::View<const uint8_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      stage(src, nbytes);
  Kokkos::deep_copy(dst, stage);
}

// Dequantize one packed payload to host FP32. Only used for the dense F32
// tensors (norms, convs, clips, gates), which are small; the accepted disk
// dtypes are F32/F16/BF16.
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
    default:
      throw std::runtime_error("unsupported dense dtype " +
                               Gguf::ggml_type_name(info.dtype));
  }
}

// Load one 2-D tensor into a packed device weight (no host dequantization:
// the payload moves verbatim and stays quantized). Throws when the dtype
// is outside the compute set, with the tensor name in the message.
template <class ExecSpace>
void gguf_load_mat(const Gguf::GgufFile& file, const TensorId& id,
                   size_t rows, size_t cols,
                   Gguf::QuantWeight<ExecSpace>& dst) {
  const auto& info = gguf_require(file, id);
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
  Gguf::QuantWeight<ExecSpace> w;
  w.dtype = info.dtype;
  w.rows  = rows;
  w.cols  = cols;
  w.data = Kokkos::View<uint8_t*, ExecSpace>("Inkling::weight", info.nbytes);
  gguf_copy_bytes_to_device(w.data, file.payload(info), (size_t)info.nbytes);
  dst = w;
}

// Load one MoE expert tensor ([experts, out, in] 3-D, expert-major) into a
// packed device weight with rows = experts*out. `want_q8` selects the
// expected payload dtype (Q8_0 on the last four layers, IQ4_NL elsewhere).
template <class ExecSpace>
void gguf_load_experts(const Gguf::GgufFile& file, const TensorId& id,
                       size_t rows, size_t cols, size_t experts,
                       bool want_q8, Gguf::QuantWeight<ExecSpace>& dst) {
  const auto& info = gguf_require(file, id);
  const Gguf::GgmlType want =
      want_q8 ? Gguf::GgmlType::Q8_0 : Gguf::GgmlType::IQ4_NL;
  if (info.dtype != want)
    throw std::runtime_error("dtype mismatch for '" + info.name +
                             "': checkpoint has " +
                             Gguf::ggml_type_name(info.dtype) +
                             ", expected " + Gguf::ggml_type_name(want));
  if (experts == 0 || rows % experts != 0)
    throw std::runtime_error("bad expert folding for '" + info.name + "'");
  const size_t per = rows / experts;
  if (info.n_dims != 3 || info.dims[2] != experts || info.dims[1] != per ||
      info.dims[0] != cols)
    throw std::runtime_error(
        "shape mismatch for '" + info.name + "': checkpoint has " +
        gguf_shape_str(info) + ", expected [" + std::to_string(cols) + "x" +
        std::to_string(per) + "x" + std::to_string(experts) + "]");
  const size_t expect =
      (info.numel / Gguf::ggml_block_len(want)) * Gguf::ggml_type_size(want);
  if (info.nbytes != expect)
    throw std::runtime_error("bad payload size for '" + info.name + "'");
  Gguf::QuantWeight<ExecSpace> w;
  w.dtype = want;
  w.rows  = rows;
  w.cols  = cols;
  w.data = Kokkos::View<uint8_t*, ExecSpace>("Inkling::experts", info.nbytes);
  gguf_copy_bytes_to_device(w.data, file.payload(info), (size_t)info.nbytes);
  dst = w;
}

// Load one dense tensor transposed: the file holds [rows, cols] but the
// Model math wants [cols, rows] (loop control downs [2D,R] vs file [R,2D],
// forget/gain [R,D] vs file [D,R], rel_proj [R,E] vs file [E,R] — verified
// against the HF torch shapes and the llama.cpp build). The payload is
// dequantized to host FP32 once (these tensors are KBs), then written
// transposed.
template <class ExecSpace, class Scalar>
void gguf_load_dense_mat_T(
    const Gguf::GgufFile& file, const TensorId& id,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  if (dst.size() == 0) return;
  const auto& info = gguf_require(file, id);
  const size_t rows = dst.extent(0), cols = dst.extent(1);
  if (!(info.n_dims == 2 && info.rows() == cols && info.cols() == rows))
    throw std::runtime_error("shape mismatch for '" + info.name +
                             "': checkpoint has " + gguf_shape_str(info) +
                             ", expected transpose of [" +
                             std::to_string(rows) + "," +
                             std::to_string(cols) + "]");
  std::vector<float> tmp;
  try {
    gguf_dequant_payload_to_float(file.payload(info), info, tmp);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot decode '" + info.name + "': " + e.what());
  }
  if (tmp.size() != rows * cols)
    throw std::runtime_error("bad payload size for '" + info.name + "'");
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      h(i, j) = static_cast<Scalar>(tmp[j * rows + i]);
  Kokkos::deep_copy(dst, h);
}
// Load one dense tensor (F32/F16/BF16 on disk) into a device view. 1-D
// tensors land in a vector view; 2-D tensors in a [rows, cols] matrix view.
template <class ExecSpace, class Scalar>
void gguf_load_dense_vec(const Gguf::GgufFile& file, const TensorId& id,
                         const Kokkos::View<Scalar*, ExecSpace>& dst) {
  if (dst.size() == 0) return;
  const auto& info = gguf_require(file, id);
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
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < tmp.size(); ++i)
    h(i) = static_cast<Scalar>(tmp[i]);
  Kokkos::deep_copy(dst, h);
}

template <class ExecSpace, class Scalar>
void gguf_load_dense_mat(
    const Gguf::GgufFile& file, const TensorId& id,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst) {
  if (dst.size() == 0) return;
  const auto& info = gguf_require(file, id);
  const size_t rows = dst.extent(0), cols = dst.extent(1);
  // Single-query res-query layers persist as 1-D [d_model] in the file;
  // accept that spelling when rows == 1 (same bytes either way).
  const bool two_d =
      (info.n_dims == 2 && info.rows() == rows && info.cols() == cols);
  const bool one_d =
      (info.n_dims == 1 && rows == 1 && info.numel == cols);
  if (!two_d && !one_d)
    throw std::runtime_error("shape mismatch for '" + info.name +
                             "': checkpoint has " + gguf_shape_str(info) +
                             ", expected [" + std::to_string(rows) + "," +
                             std::to_string(cols) + "]");
  std::vector<float> tmp;
  try {
    gguf_dequant_payload_to_float(file.payload(info), info, tmp);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot decode '" + info.name + "': " + e.what());
  }
  if (tmp.size() != rows * cols)
    throw std::runtime_error("bad payload size for '" + info.name + "'");
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      h(i, j) = static_cast<Scalar>(tmp[i * cols + j]);
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
void gguf_validate_metadata(const Gguf::GgufFile& file, const Config& cfg) {
  constexpr const char* arch = "inkling2_loop";
  constexpr double rel_tol   = 1e-6;
  auto close = [&](const std::string& what, double have, double want) {
    const double tol = rel_tol * std::max(1.0, std::fabs(want));
    if (std::fabs(have - want) > tol)
      throw std::runtime_error("GGUF metadata mismatch: '" + what + "' is " +
                               std::to_string(have) + " but Config expects " +
                               std::to_string(want));
  };
  auto u32 = [&](const char* key, const char* what) {
    return gguf_find_kv(file.kv_u32_all(), arch + std::string(".") + key,
                        std::string(".") + key, what);
  };
  auto f32 = [&](const char* key, const char* what) {
    return gguf_find_kv(file.kv_f32_all(), arch + std::string(".") + key,
                        std::string(".") + key, what);
  };
  const std::string prefix(arch);
  if (file.has_kv("general.architecture") &&
      file.kv_string("general.architecture") != arch)
    throw std::runtime_error("GGUF metadata mismatch: general.architecture is '" +
                             file.kv_string("general.architecture") +
                             "' (expected '" + prefix + "')");
  if (u32("block_count", "block count") != (uint32_t)cfg.n_layers)
    throw std::runtime_error("GGUF metadata mismatch: 'block_count' disagrees");
  if (u32("embedding_length", "embedding length") != (uint32_t)cfg.d_model)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'embedding_length' disagrees");
  if (u32("vocab_size", "vocab size") != (uint32_t)cfg.vocab_size)
    throw std::runtime_error("GGUF metadata mismatch: 'vocab_size' disagrees");
  if (u32("vocab_size_unpadded", "unpadded vocab") !=
      (uint32_t)cfg.unpadded_vocab)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'vocab_size_unpadded' disagrees");
  if (u32("expert_count", "expert count") != (uint32_t)cfg.n_experts)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'expert_count' disagrees");
  if (u32("expert_used_count", "top-k") != (uint32_t)cfg.top_k)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'expert_used_count' disagrees");
  if (u32("attention.key_length", "key length") != (uint32_t)cfg.head_dim)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attention.key_length' disagrees");
  if (u32("attention.sliding_window", "sliding window") !=
      (uint32_t)cfg.swa_window)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attention.sliding_window' disagrees");
  if (u32("shortconv.l_cache", "shortconv kernel") !=
      (uint32_t)cfg.sconv_kernel)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'shortconv.l_cache' disagrees");
  if (u32("attn_res.block_size", "res block") !=
      (uint32_t)cfg.attn_res_block)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attn_res.block_size' disagrees");
  if (u32("attention.relative_length", "rel length") !=
      (uint32_t)cfg.d_rel)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attention.relative_length' disagrees");
  if (u32("attention.relative_extent", "rel extent") !=
      (uint32_t)cfg.rel_extent)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attention.relative_extent' disagrees");
  if (u32("attention.log_scale_floor", "log floor") !=
      (uint32_t)cfg.log_floor)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'attention.log_scale_floor' disagrees");
  if (u32("loop.start", "loop start") != (uint32_t)cfg.loop_start)
    throw std::runtime_error("GGUF metadata mismatch: 'loop.start' disagrees");
  if (u32("loop.end", "loop end") != (uint32_t)cfg.loop_end)
    throw std::runtime_error("GGUF metadata mismatch: 'loop.end' disagrees");
  if (u32("loop.count", "loop passes") != (uint32_t)cfg.loop_passes)
    throw std::runtime_error("GGUF metadata mismatch: 'loop.count' disagrees");
  if (u32("loop.gate_rank", "gate rank") != (uint32_t)cfg.gate_rank)
    throw std::runtime_error(
        "GGUF metadata mismatch: 'loop.gate_rank' disagrees");
  close("attention.layer_norm_rms_epsilon",
        (double)f32("attention.layer_norm_rms_epsilon", "rms eps"),
        (double)cfg.eps);
  close("expert_weights_scale", (double)f32("expert_weights_scale",
                                            "route scale"),
        (double)cfg.route_scale);
  close("final_logit_softcapping",
        (double)f32("final_logit_softcapping", "softcap"),
        (double)cfg.final_softcap);
  close("attention.qk_clip", (double)f32("attention.qk_clip", "qk clip"),
        (double)cfg.qk_clip);
  close("attention.relative_clip",
        (double)f32("attention.relative_clip", "rel clip"),
        (double)cfg.rel_clip);
  close("attention.log_scale_alpha",
        (double)f32("attention.log_scale_alpha", "log alpha"),
        (double)cfg.log_alpha);
  const std::vector<char> pat = gguf_find_kv(
      file.kv_bool_arrays_all(), prefix + ".attention.sliding_window_pattern",
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
    const bool expect_sliding = cfg.is_local(l);
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
  // Map the checkpoint read-only (::mmap via Gguf::GgufFile), so tensor
  // payloads are copied to the device straight out of the mapping instead
  // of through a full-file host buffer.
  std::unique_ptr<Gguf::GgufFile> file_owner;
  {
    LoadPhase p("GGUF map/read + parse");
    file_owner = std::make_unique<Gguf::GgufFile>(path);
  }
  const Gguf::GgufFile& file = *file_owner;  // throws on corrupt file
  file_owner->start_prefetch();
  {
    LoadPhase p("GGUF metadata check");
    gguf_validate_metadata(file, cfg);
  }
  size_t rows = 0, cols = 0;
  expected_shape({Tensor::TokenEmbd, -1}, cfg, rows, cols);
  gguf_load_mat<ExecSpace>(file, {Tensor::TokenEmbd, -1}, rows, cols, w.tok_emb);
  gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::TokenEmbdNorm, -1},
                                         w.tok_norm);
  gguf_load_mat<ExecSpace>(file, {Tensor::Output, -1}, rows, cols, w.out_w);
  gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::OutputNorm, -1},
                                         w.out_norm);
  gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::OutputResQuery, -1},
                                         w.out_res_q);
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto& b = w.blocks[(size_t)l];
    const size_t D   = (size_t)cfg.d_model;
    const size_t hq  = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
    const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    gguf_load_mat<ExecSpace>(file, {Tensor::AttnGate, l}, hq, D, b.wgate_attn);
    gguf_load_mat<ExecSpace>(file, {Tensor::AttnQ, l}, hq, D, b.wq);
    gguf_load_mat<ExecSpace>(file, {Tensor::AttnK, l}, hkv, D, b.wk);
    gguf_load_mat<ExecSpace>(file, {Tensor::AttnV, l}, hkv, D, b.wv);
    gguf_load_mat<ExecSpace>(file, {Tensor::AttnOutput, l}, D, hq,
                             b.wo);
    size_t rr = 0, rc = 0;
    expected_shape({Tensor::AttnRel, l}, cfg, rr, rc);
    gguf_load_mat<ExecSpace>(file, {Tensor::AttnRel, l}, rr, rc, b.wrel);
    size_t sr = 0, sc = 0;
    expected_shape({Tensor::AttnSinkProj, l}, cfg, sr, sc);
    gguf_load_mat<ExecSpace>(file, {Tensor::AttnSinkProj, l}, sr, sc,
                             b.wsink);
    if (!is_dyn_conv_layer(cfg, l)) {
      gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::AttnNorm, l},
                                             b.attn_norm);
    } else {
      size_t pr = 0, pc = 0;
      expected_shape({Tensor::AttnKConvProj, l}, cfg, pr, pc);
      gguf_load_mat<ExecSpace>(file, {Tensor::AttnKConvProj, l}, pr, pc,
                               b.k_conv_proj);
      gguf_load_mat<ExecSpace>(file, {Tensor::AttnVConvProj, l}, pr, pc,
                               b.v_conv_proj);
      gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::AttnQInpNorm, l},
                                             b.q_norm);
      gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::AttnKInpNorm, l},
                                             b.k_norm);
      gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::AttnVInpNorm, l},
                                             b.v_norm);
    }
    const bool want_q8 = moe_experts_are_q8(cfg, l);
    size_t gr = 0, gc = 0;
    expected_shape({Tensor::FfnGateUpExps, l}, cfg, gr, gc);
    {
      Gguf::QuantWeight<ExecSpace> folded;
      gguf_load_experts<ExecSpace>(file, {Tensor::FfnGateUpExps, l}, gr, gc,
                                   (size_t)cfg.n_experts, want_q8, folded);
      slice_experts(folded, (size_t)cfg.n_experts, b.gate_up_exps_q);
    }
    size_t dr = 0, dc = 0;
    expected_shape({Tensor::FfnDownExps, l}, cfg, dr, dc);
    {
      Gguf::QuantWeight<ExecSpace> folded;
      gguf_load_experts<ExecSpace>(file, {Tensor::FfnDownExps, l}, dr, dc,
                                   (size_t)cfg.n_experts, want_q8, folded);
      slice_experts(folded, (size_t)cfg.n_experts, b.down_exps_q);
    }
    gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::FfnNorm, l},
                                           b.post_attn_norm);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::FfnResQuery, l},
                                           b.mlp_res_q);
    gguf_load_dense_vec<ExecSpace, Scalar>(file, {Tensor::ExpProbsB, l},
                                           b.router_bias);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::AttnKConv, l},
                                           b.k_conv);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::AttnVConv, l},
                                           b.v_conv);
    // The full-attention dynamic-conv path iterates the kernel
    // [K, Ckv]-major; persist the transpose alongside the sliding [Ckv, K].
    transpose_into(b.k_conv, b.k_conv_static);
    transpose_into(b.v_conv, b.v_conv_static);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::AttnPostConv, l},
                                           b.post_attn_conv);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::AttnQClip, l},
                                           b.q_scale);
    gguf_load_dense_mat_T<ExecSpace, Scalar>(file, {Tensor::AttnRelBias, l},
                                           b.rel_proj);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::AttnRelClip, l},
                                           b.rel_scale);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::AttnResQuery, l},
                                           b.attn_res_q);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::FfnGateInp, l},
                                           b.router);
    gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::FfnPostConv, l},
                                           b.post_mlp_conv);
    if (is_loop_layer(cfg, l)) {
      gguf_load_dense_mat_T<ExecSpace, Scalar>(file, {Tensor::LoopQCtrlDown, l},
                                             b.loop_q_down);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(
          file, {Tensor::LoopQCtrlForget, l}, b.loop_q_forget);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(file, {Tensor::LoopQCtrlGain, l},
                                             b.loop_q_gain);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(file, {Tensor::LoopKCtrlDown, l},
                                             b.loop_k_down);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(
          file, {Tensor::LoopKCtrlForget, l}, b.loop_k_forget);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(file, {Tensor::LoopKCtrlGain, l},
                                             b.loop_k_gain);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(file, {Tensor::LoopVCtrlDown, l},
                                             b.loop_v_down);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(
          file, {Tensor::LoopVCtrlForget, l}, b.loop_v_forget);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(file, {Tensor::LoopVCtrlGain, l},
                                             b.loop_v_gain);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(
          file, {Tensor::LoopFfnCtrlDown, l}, b.loop_mlp_down);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(
          file, {Tensor::LoopFfnCtrlForget, l}, b.loop_mlp_forget);
      gguf_load_dense_mat_T<ExecSpace, Scalar>(
          file, {Tensor::LoopFfnCtrlGain, l}, b.loop_mlp_gain);
      gguf_load_dense_mat<ExecSpace, Scalar>(file, {Tensor::LoopMsgQuery, l},
                                             b.msg_query);
    }
    // Keep host RAM bounded: at most one tensor payload is referenced at a
    // time (quant payloads copy straight out of the mapping; dense tensors
    // convert through small host temporaries above).
  }
  // Every payload copy above reads straight out of the mapping, so make sure
  // all transfers have completed before the mapping is released.
  Kokkos::fence();
  file_owner->stop_prefetch();
  if (load_timing_enabled())
    std::fprintf(stderr, "[load] %-24s %8.3f s\n", "load_gguf end-to-end",
                 since_s(t_total));
}

std::vector<std::string> list_gguf_tensors(const std::string& path) {
  Gguf::GgufFile file(path);  // header only: no tensor data is touched
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
template void copy_host_vec_to_device<Kokkos::DefaultExecutionSpace, float>(const std::vector<float>&,
    const Kokkos::View<float*, Kokkos::DefaultExecutionSpace>&);
#if !KOKKOS_BHALF_T_IS_FLOAT
template void random_init<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&, uint64_t);
template void load_safetensors<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, const std::vector<std::string>&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&);
template void load_gguf<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const Config&, const std::string&, Weights<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>&);
template void copy_host_to_device<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<Kokkos::Experimental::bhalf_t**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>&);
template void copy_host_vec_to_device<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::bhalf_t>(const std::vector<float>&,
    const Kokkos::View<Kokkos::Experimental::bhalf_t*, Kokkos::DefaultExecutionSpace>&);
#endif

}  // namespace Inkling
}  // namespace KokkosDNN
