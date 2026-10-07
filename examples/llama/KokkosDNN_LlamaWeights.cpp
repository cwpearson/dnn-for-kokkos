#include "KokkosDNN_LlamaWeights.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace KokkosDNN {
namespace Llama {
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

uint64_t read_le64(std::ifstream& f) {
  unsigned char b[8];
  f.read(reinterpret_cast<char*>(b), 8);
  if (!f) throw std::runtime_error("safetensors: truncated header length");
  uint64_t v = 0;
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
    size_t b = a + 1, esc = false;
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

// Read + convert one tensor from open file (data base = data_start).
std::vector<float> read_tensor(std::ifstream& f, size_t data_start,
                               const TensorMeta& m) {
  size_t n = numel(m.shape);
  size_t nbytes = m.end - m.begin;
  std::vector<char> raw(nbytes);
  f.seekg((std::streamoff)(data_start + m.begin));
  f.read(raw.data(), (std::streamsize)nbytes);
  if (!f) throw std::runtime_error("safetensors: truncated tensor data");
  std::vector<float> out(n);
  if (m.dtype == "F32") {
    if (nbytes != n * 4) throw std::runtime_error("F32 size mismatch");
    std::memcpy(out.data(), raw.data(), nbytes);
  } else if (m.dtype == "F16") {
    if (nbytes != n * 2) throw std::runtime_error("F16 size mismatch");
    auto* h = reinterpret_cast<uint16_t*>(raw.data());
    for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(h[i]);
  } else if (m.dtype == "BF16") {
    if (nbytes != n * 2) throw std::runtime_error("BF16 size mismatch");
    auto* h = reinterpret_cast<uint16_t*>(raw.data());
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(h[i]);
  } else {
    throw std::runtime_error("unsupported safetensors dtype: " + m.dtype);
  }
  return out;
}

}  // namespace

template <class ExecSpace>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& dst) {
  if (dst.extent(0) != rows || dst.extent(1) != cols)
    throw std::runtime_error("copy_host_to_device: shape mismatch");
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j) h(i, j) = host[i * cols + j];
  Kokkos::deep_copy(dst, h);
}

template <class ExecSpace>
void copy_host_vec_to_device(
    const std::vector<float>& host,
    const Kokkos::View<float*, ExecSpace>& dst) {
  if (dst.extent(0) != host.size())
    throw std::runtime_error("copy_host_vec_to_device: shape mismatch");
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < host.size(); ++i) h(i) = host[i];
  Kokkos::deep_copy(dst, h);
}

template <class ExecSpace>
void random_init(const Config& cfg, Weights<ExecSpace>& w, uint64_t seed) {
  w.allocate(cfg);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  auto fill_mat = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i)
      for (size_t j = 0; j < h.extent(1); ++j) h(i, j) = dist(rng);
    Kokkos::deep_copy(v, h);
  };
  auto fill_vec = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.extent(0); ++i) h(i) = 1.0f;
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
    fill_mat(b.wgate);
    fill_mat(b.wup);
    fill_mat(b.wdown);
    fill_vec(b.attn_norm);
    fill_vec(b.ffn_norm);
  }
}

template <class ExecSpace>
void load_safetensors(const Config& cfg,
                      const std::vector<std::string>& files,
                      Weights<ExecSpace>& w) {
  w.allocate(cfg);
  // Gather metas across shards.
  struct Loc {
    std::string file;
    TensorMeta meta;
  };
  std::map<std::string, Loc> found;
  std::map<std::string, size_t> data_start;  // file -> data section offset
  std::map<std::string, std::map<std::string, TensorMeta>> per_file;
  for (auto& fp : files) {
    std::ifstream f(fp, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + fp);
    uint64_t n = read_le64(f);
    std::string js(n, '\0');
    f.read(js.data(), (std::streamsize)n);
    if (!f) throw std::runtime_error("truncated JSON in " + fp);
    data_start[fp] = (size_t)(8 + n);
    per_file[fp]   = parse_header(js);
    for (auto& [name, m] : per_file[fp]) found[name] = {fp, m};
  }
  auto need = [&](const std::string& name) -> std::vector<float> {
    auto it = found.find(name);
    if (it == found.end())
      throw std::runtime_error("tensor not found in checkpoint: " + name);
    std::ifstream f(it->second.file, std::ios::binary);
    return read_tensor(f, data_start[it->second.file], it->second.meta);
  };
  auto expect_shape = [&](const std::vector<float>& v,
                          const std::vector<size_t>& want,
                          const std::string& name) {
    if (v.size() != numel(want))
      throw std::runtime_error("shape mismatch for " + name);
  };

  copy_host_to_device<ExecSpace>(
      need("model.embed_tokens.weight"),
      cfg.vocab_size, cfg.dim, w.embed);
  {
    auto v = need("model.norm.weight");
    expect_shape(v, {(size_t)cfg.dim}, "model.norm.weight");
    copy_host_vec_to_device<ExecSpace>(v, w.final_norm);
  }
  // lm_head may be named "lm_head.weight" (HF) — accept both.
  {
    std::vector<float> v;
    try {
      v = need("lm_head.weight");
    } catch (...) {
      v = need("model.lm_head.weight");
    }
    copy_host_to_device<ExecSpace>(v, cfg.vocab_size, cfg.dim, w.lm_head);
  }
  const int hq  = cfg.n_heads * cfg.head_dim;
  const int hkv = cfg.n_kv_heads * cfg.head_dim;
  for (int l = 0; l < cfg.n_layers; ++l) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    auto& b = w.blocks[l];
    copy_host_to_device<ExecSpace>(need(p + "self_attn.q_proj.weight"), hq,
                                   cfg.dim, b.wq);
    copy_host_to_device<ExecSpace>(need(p + "self_attn.k_proj.weight"), hkv,
                                   cfg.dim, b.wk);
    copy_host_to_device<ExecSpace>(need(p + "self_attn.v_proj.weight"), hkv,
                                   cfg.dim, b.wv);
    copy_host_to_device<ExecSpace>(need(p + "self_attn.o_proj.weight"),
                                   cfg.dim, hq, b.wo);
    copy_host_to_device<ExecSpace>(need(p + "mlp.gate_proj.weight"),
                                   cfg.ffn_dim, cfg.dim, b.wgate);
    copy_host_to_device<ExecSpace>(need(p + "mlp.up_proj.weight"),
                                   cfg.ffn_dim, cfg.dim, b.wup);
    copy_host_to_device<ExecSpace>(need(p + "mlp.down_proj.weight"), cfg.dim,
                                   cfg.ffn_dim, b.wdown);
    {
      auto v = need(p + "input_layernorm.weight");
      expect_shape(v, {(size_t)cfg.dim}, p + "input_layernorm.weight");
      copy_host_vec_to_device<ExecSpace>(v, b.attn_norm);
    }
    {
      auto v = need(p + "post_attention_layernorm.weight");
      expect_shape(v, {(size_t)cfg.dim}, p + "post_attention_layernorm");
      copy_host_vec_to_device<ExecSpace>(v, b.ffn_norm);
    }
  }
}

// Explicit instantiation declarations for common spaces (defined here so
// the .cpp compiles them; other ExecSpaces instantiate implicitly from
// the header since functions are templates — the definitions below force
// Serial emit for faster test linking).
template void random_init<Kokkos::Serial>(const Config&, Weights<Kokkos::Serial>&, uint64_t);
template void load_safetensors<Kokkos::Serial>(const Config&, const std::vector<std::string>&, Weights<Kokkos::Serial>&);
template void copy_host_to_device<Kokkos::Serial>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::Serial>&);
#ifdef KOKKOS_ENABLE_OPENMP
template void random_init<Kokkos::OpenMP>(const Config&, Weights<Kokkos::OpenMP>&, uint64_t);
template void load_safetensors<Kokkos::OpenMP>(const Config&, const std::vector<std::string>&, Weights<Kokkos::OpenMP>&);
template void copy_host_to_device<Kokkos::OpenMP>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::OpenMP>&);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template void random_init<Kokkos::Cuda>(const Config&, Weights<Kokkos::Cuda>&, uint64_t);
template void load_safetensors<Kokkos::Cuda>(const Config&, const std::vector<std::string>&, Weights<Kokkos::Cuda>&);
template void copy_host_to_device<Kokkos::Cuda>(const std::vector<float>&, size_t, size_t,
    const Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::Cuda>&);
#endif

}  // namespace Llama
}  // namespace KokkosDNN
