#include "KokkosDNN_SanaTextEncoder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

#include "KokkosDNN_Activation.hpp"
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_RoPE.hpp"

#if defined(__unix__) || defined(__unix) || defined(__APPLE__) || \
    defined(__linux__)
#define SANAGEMMA_HAS_DIRENT 1
#include <dirent.h>
#else
#define SANAGEMMA_HAS_DIRENT 0
#endif

// mmap-backed checkpoint reads (POSIX) with bounded-ifstream fallback.
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
namespace SanaGemma {
namespace {

// ---------------------------------------------------------------------------
// Minimal JSON helpers (dependency-free, schema-targeted)
// ---------------------------------------------------------------------------

void skip_ws(const std::string& s, size_t& p) {
  while (p < s.size() &&
         (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
    ++p;
}

std::string parse_json_string(const std::string& s, size_t& p) {
  if (p >= s.size() || s[p] != '"')
    throw std::runtime_error("sanagemma config: expected string");
  ++p;
  std::string out;
  while (true) {
    if (p >= s.size())
      throw std::runtime_error("sanagemma config: unterminated string");
    char c = s[p++];
    if (c == '"') break;
    if (c != '\\') {
      out += c;
      continue;
    }
    if (p >= s.size())
      throw std::runtime_error("sanagemma config: bad escape");
    char e = s[p++];
    switch (e) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        if (p + 4 > s.size())
          throw std::runtime_error("sanagemma config: bad \\u");
        unsigned cp = 0;
        for (int i = 0; i < 4; ++i) {
          char h = s[p++];
          cp <<= 4;
          if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
          else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
          else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
          else throw std::runtime_error("sanagemma config: bad \\u");
        }
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) {
          out += (char)(0xC0 | (cp >> 6));
          out += (char)(0x80 | (cp & 0x3F));
        } else {
          out += (char)(0xE0 | (cp >> 12));
          out += (char)(0x80 | ((cp >> 6) & 0x3F));
          out += (char)(0x80 | (cp & 0x3F));
        }
        break;
      }
      default: throw std::runtime_error("sanagemma config: bad escape");
    }
  }
  return out;
}

void skip_json_value(const std::string& s, size_t& p) {
  skip_ws(s, p);
  if (p >= s.size())
    throw std::runtime_error("sanagemma config: unexpected end");
  char c = s[p];
  if (c == '"') {
    (void)parse_json_string(s, p);
    return;
  }
  if (c == '{' || c == '[') {
    char open = c, close = (c == '{') ? '}' : ']';
    int depth = 0;
    bool in_str = false, esc = false;
    for (; p < s.size(); ++p) {
      char k = s[p];
      if (in_str) {
        if (esc) esc = false;
        else if (k == '\\') esc = true;
        else if (k == '"') in_str = false;
      } else {
        if (k == '"') in_str = true;
        else if (k == open) ++depth;
        else if (k == close && --depth == 0) {
          ++p;
          return;
        }
      }
    }
    throw std::runtime_error("sanagemma config: unbalanced JSON");
  }
  while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;
}

bool find_key(const std::string& s, const std::string& key, size_t& vs,
              size_t& ve) {
  size_t p = 0;
  skip_ws(s, p);
  if (p >= s.size() || s[p] != '{')
    throw std::runtime_error("sanagemma config: root must be an object");
  ++p;
  skip_ws(s, p);
  if (p < s.size() && s[p] == '}') return false;
  while (true) {
    skip_ws(s, p);
    if (p >= s.size() || s[p] != '"')
      throw std::runtime_error("sanagemma config: expected string key");
    std::string k = parse_json_string(s, p);
    skip_ws(s, p);
    if (p >= s.size() || s[p] != ':')
      throw std::runtime_error("sanagemma config: expected ':'");
    ++p;
    size_t v = p;
    skip_json_value(s, p);
    if (k == key) {
      vs = v;
      ve = p;
      return true;
    }
    skip_ws(s, p);
    if (p >= s.size())
      throw std::runtime_error("sanagemma config: unexpected end");
    if (s[p] == ',') {
      ++p;
      continue;
    }
    if (s[p] == '}') return false;
    throw std::runtime_error("sanagemma config: expected ',' or '}'");
  }
}

std::string trim_ws(const std::string& t) {
  size_t a = 0;
  while (a < t.size() && std::isspace((unsigned char)t[a])) ++a;
  size_t b = t.size();
  while (b > a && std::isspace((unsigned char)t[b - 1])) --b;
  return t.substr(a, b - a);
}

bool get_raw(const std::string& s, const std::string& key, std::string& out) {
  size_t vs = 0, ve = 0;
  if (!find_key(s, key, vs, ve)) return false;
  out = trim_ws(s.substr(vs, ve - vs));
  return true;
}

double parse_num_val(const std::string& v, const std::string& key) {
  try {
    size_t n = 0;
    double d = std::stod(v, &n);
    if (n != v.size()) throw std::runtime_error("trailing characters");
    if (!std::isfinite(d)) throw std::runtime_error("non-finite value");
    return d;
  } catch (const std::runtime_error&) {
    throw;
  } catch (const std::exception&) {
    throw std::runtime_error("sanagemma config: '" + key +
                             "' is not a number");
  }
}

bool get_num(const std::string& s, const std::string& key, double& out) {
  std::string v;
  if (!get_raw(s, key, v)) return false;
  if (v == "null")
    throw std::runtime_error("sanagemma config: '" + key +
                             "' must not be null");
  out = parse_num_val(v, key);
  return true;
}

bool get_int(const std::string& s, const std::string& key, long long& out) {
  double d = 0;
  if (!get_num(s, key, d)) return false;
  if (std::floor(d) != d)
    throw std::runtime_error("sanagemma config: '" + key +
                             "' must be an integer");
  out = std::llround(d);
  return true;
}

bool get_bool(const std::string& s, const std::string& key, bool& out) {
  std::string v;
  if (!get_raw(s, key, v)) return false;
  if (v == "true") {
    out = true;
    return true;
  }
  if (v == "false") {
    out = false;
    return true;
  }
  throw std::runtime_error("sanagemma config: '" + key +
                           "' must be true/false");
}

bool get_string(const std::string& s, const std::string& key,
                std::string& out) {
  std::string v;
  if (!get_raw(s, key, v)) return false;
  if (v == "null")
    throw std::runtime_error("sanagemma config: '" + key +
                             "' must not be null");
  if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
    size_t p = 0;
    out = parse_json_string(v, p);
    return true;
  }
  throw std::runtime_error("sanagemma config: '" + key + "' must be a string");
}

int checked_int(long long v, const std::string& key) {
  if (v < 0 || v > 1000000000LL)
    throw std::runtime_error("sanagemma config: '" + key + "' out of range");
  return (int)v;
}

std::string to_lower_str(std::string t) {
  for (auto& ch : t) ch = (char)std::tolower((unsigned char)ch);
  return t;
}

// ---------------------------------------------------------------------------
// Half-precision conversions (host side, loader only)
// ---------------------------------------------------------------------------

float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h & 0x7C00u) >> 10;
  uint32_t mant = (h & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while (!(mant & 0x0400u)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FFu;
      f = sign | ((exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);
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

// ---------------------------------------------------------------------------
// Minimal safetensors header parser (name -> dtype/shape/data_offsets)
// ---------------------------------------------------------------------------

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
      else if (c == cc && --depth == 0) return i;
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
    while (i < inner.size() && !std::isdigit((unsigned char)inner[i])) i++;
    size_t j = i;
    while (j < inner.size() && std::isdigit((unsigned char)inner[j])) j++;
    if (j > i) out.push_back(std::stoull(inner.substr(i, j - i)));
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
    size_t b;
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
    if (colon == std::string::npos) break;
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
        m.end = off[1];
      }
      metas[name] = m;
    }
    i = cb + 1;
  }
  return metas;
}

size_t numel_shape(const std::vector<size_t>& shape) {
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

}  // namespace

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

float Config::attn_scale() const {
  if (!(query_pre_attn_scalar > 0) || !std::isfinite(query_pre_attn_scalar))
    throw std::invalid_argument(
        "Config::attn_scale: query_pre_attn_scalar must be positive");
  return 1.0f / std::sqrt(query_pre_attn_scalar);
}

void Config::validate() const {
  auto bad = [](const std::string& m) {
    throw std::invalid_argument("SanaGemma Config::validate: " + m);
  };
  if (hidden_size <= 0) bad("hidden_size must be > 0");
  if (num_hidden_layers <= 0) bad("num_hidden_layers must be > 0");
  if (num_attention_heads <= 0) bad("num_attention_heads must be > 0");
  if (num_key_value_heads <= 0) bad("num_key_value_heads must be > 0");
  if (num_attention_heads % num_key_value_heads != 0)
    bad("num_attention_heads must be a multiple of num_key_value_heads");
  if (head_dim <= 0 || head_dim % 2 != 0)
    bad("head_dim must be positive and even");
  if (intermediate_size <= 0) bad("intermediate_size must be > 0");
  if (vocab_size <= 0) bad("vocab_size must be > 0");
  if (!(rms_norm_eps > 0) || !std::isfinite(rms_norm_eps))
    bad("rms_norm_eps must be positive and finite");
  if (!(rope_theta > 0) || !std::isfinite(rope_theta))
    bad("rope_theta must be positive and finite");
  if (sliding_window <= 0) bad("sliding_window must be > 0");
  if (!(query_pre_attn_scalar > 0) || !std::isfinite(query_pre_attn_scalar))
    bad("query_pre_attn_scalar must be positive and finite");
  if (attn_logit_softcapping < 0 || !std::isfinite(attn_logit_softcapping))
    bad("attn_logit_softcapping must be >= 0 and finite");
  if (is_full_layer.size() != (size_t)num_hidden_layers)
    bad("layer_types length must equal num_hidden_layers");
  if (max_position_embeddings <= 0)
    bad("max_position_embeddings must be > 0");
  if (bos_token_id < 0 || eos_token_id < 0 || pad_token_id < 0)
    bad("bos/eos/pad token ids must be >= 0");
  if (to_lower_str(hidden_activation) != "gelu_pytorch_tanh")
    throw std::runtime_error(
        "SanaGemma Config::validate: hidden_activation '" + hidden_activation +
        "' unsupported: only 'gelu_pytorch_tanh' is supported (Model "
        "hardcodes gated-GeLU with the tanh approximation)");
  if (attention_bias)
    throw std::runtime_error(
        "SanaGemma Config::validate: 'attention_bias' must be false "
        "(Model has no bias path)");
}

Config Config::from_json_string(const std::string& text, const Config& base) {
  Config c = base;
  long long iv = 0;
  double dv = 0;
  bool bv = false;
  std::string v;
  if (get_int(text, "hidden_size", iv))
    c.hidden_size = checked_int(iv, "hidden_size");
  if (get_int(text, "num_hidden_layers", iv)) {
    c.num_hidden_layers = checked_int(iv, "num_hidden_layers");
    c.is_full_layer = default_layer_types(c.num_hidden_layers);
  }
  if (get_int(text, "num_attention_heads", iv))
    c.num_attention_heads = checked_int(iv, "num_attention_heads");
  if (get_int(text, "num_key_value_heads", iv))
    c.num_key_value_heads = checked_int(iv, "num_key_value_heads");
  if (get_int(text, "head_dim", iv)) {
    if (iv <= 0 || iv > 1000000000LL || iv % 2 != 0)
      throw std::runtime_error(
          "sanagemma config: 'head_dim' must be positive and even");
    c.head_dim = (int)iv;
  }
  if (get_int(text, "intermediate_size", iv))
    c.intermediate_size = checked_int(iv, "intermediate_size");
  if (get_int(text, "vocab_size", iv))
    c.vocab_size = checked_int(iv, "vocab_size");
  if (get_int(text, "sliding_window", iv))
    c.sliding_window = checked_int(iv, "sliding_window");
  if (get_num(text, "query_pre_attn_scalar", dv))
    c.query_pre_attn_scalar = (float)dv;
  if (get_num(text, "attn_logit_softcapping", dv))
    c.attn_logit_softcapping = (float)dv;
  if (get_raw(text, "layer_types", v)) {
    std::string t = trim_ws(v);
    if (t == "null")
      throw std::runtime_error(
          "sanagemma config: 'layer_types' must not be null");
    if (t.empty() || t.front() != '[')
      throw std::runtime_error(
          "sanagemma config: 'layer_types' must be an array");
    std::vector<char> lt;
    size_t p = 1;
    while (true) {
      skip_ws(t, p);
      if (p >= t.size())
        throw std::runtime_error(
            "sanagemma config: unbalanced 'layer_types'");
      if (t[p] == ']') break;
      if (t[p] != '"')
        throw std::runtime_error(
            "sanagemma config: 'layer_types' entries must be strings");
      std::string e = parse_json_string(t, p);
      std::string le = to_lower_str(e);
      if (le == "full_attention")
        lt.push_back(1);
      else if (le == "sliding_attention" || le == "sliding_window_attention")
        lt.push_back(0);
      else
        throw std::runtime_error(
            "sanagemma config: unknown layer_types entry '" + e + "'");
      skip_ws(t, p);
      if (p < t.size() && t[p] == ',') {
        ++p;
        continue;
      }
      if (p < t.size() && t[p] == ']') break;
      throw std::runtime_error(
          "sanagemma config: expected ',' or ']' in layer_types");
    }
    c.is_full_layer = std::move(lt);
  }
  // Legacy top-level rope_theta first, then the newer rope_parameters form
  // (which wins when both are present).
  if (get_num(text, "rope_theta", dv)) c.rope_theta = (float)dv;
  if (get_raw(text, "rope_parameters", v)) {
    std::string t = trim_ws(v);
    if (t == "null")
      throw std::runtime_error(
          "sanagemma config: 'rope_parameters' must not be null");
    if (t.empty() || t.front() != '{')
      throw std::runtime_error(
          "sanagemma config: 'rope_parameters' must be an object");
    double th = 0;
    if (get_num(t, "rope_theta", th)) c.rope_theta = (float)th;
    std::string rt;
    if (get_string(t, "rope_type", rt) && to_lower_str(rt) != "default")
      throw std::runtime_error("sanagemma config: rope_type '" + rt +
                               "' unsupported: only 'default' is supported");
  }
  if (get_string(text, "hidden_activation", v)) {
    if (to_lower_str(v) != "gelu_pytorch_tanh")
      throw std::runtime_error("sanagemma config: 'hidden_activation' '" + v +
                               "' unsupported: only 'gelu_pytorch_tanh' is "
                               "supported");
    c.hidden_activation = v;
  }
  if (get_string(text, "model_type", v)) {
    if (to_lower_str(v) != "gemma2")
      throw std::runtime_error("sanagemma config: 'model_type' '" + v +
                               "' unsupported: only 'gemma2' is supported");
  }
  if (get_num(text, "rms_norm_eps", dv)) c.rms_norm_eps = (float)dv;
  if (get_int(text, "max_position_embeddings", iv))
    c.max_position_embeddings = checked_int(iv, "max_position_embeddings");
  if (get_int(text, "bos_token_id", iv))
    c.bos_token_id = checked_int(iv, "bos_token_id");
  if (get_int(text, "eos_token_id", iv))
    c.eos_token_id = checked_int(iv, "eos_token_id");
  if (get_int(text, "pad_token_id", iv))
    c.pad_token_id = checked_int(iv, "pad_token_id");
  if (get_bool(text, "attention_bias", bv)) {
    c.attention_bias = bv;
    if (bv)
      throw std::runtime_error(
          "sanagemma config: 'attention_bias' must be false (Model has no "
          "bias path)");
  }
  if (get_bool(text, "tie_word_embeddings", bv)) c.tie_word_embeddings = bv;
  c.validate();
  return c;
}

Config Config::from_json_file(const std::string& path, const Config& base) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("sanagemma config: cannot open '" + path + "'");
  std::ostringstream ss;
  ss << f.rdbuf();
  return from_json_string(ss.str(), base);
}

namespace {

uint64_t load_le64(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

// One mapped (or fallback-readable) checkpoint shard. Tensors stream
// per-tensor through a small host staging buffer; peak extra host RAM is
// one chunk, not a full-model expansion.
struct ShardFile {
  std::string path;
  std::map<std::string, TensorMeta> metas;
  size_t data_start = 0;
  size_t file_size = 0;
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
    file_size = (size_t)st.st_size;
    void* m = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
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
    metas = parse_header(std::string(map + 8, (size_t)n));
#else
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    unsigned char lb[8];
    f.read(reinterpret_cast<char*>(lb), 8);
    if (!f) throw std::runtime_error("safetensors: truncated header length");
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | lb[i];
    std::string js((size_t)n, '\0');
    f.read(js.data(), (std::streamsize)n);
    if (!f) throw std::runtime_error("truncated JSON in " + p);
    f.seekg(0, std::ios::end);
    std::streamoff end = f.tellg();
    if (end < 0) throw std::runtime_error("cannot stat " + p);
    file_size = (size_t)end;
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

  ShardFile(const ShardFile&) = delete;
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
    o.map = nullptr;
#endif
    o.file_size = 0;
  }
  ShardFile& operator=(ShardFile&&) = delete;

#if KOKKOSDNN_HAS_MMAP
  const char* data_ptr(size_t off) const { return map + data_start + off; }
#else
  void read_bytes(size_t off, char* dst, size_t n) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg((std::streamoff)(data_start + off));
    f.read(dst, (std::streamsize)n);
    if (!f) throw std::runtime_error("safetensors: truncated tensor data");
  }
#endif
};

inline constexpr size_t kStreamChunkElems = size_t{1} << 18;

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
  size_t expect = numel_shape(want) * es;
  if (nbytes != expect)
    throw std::runtime_error("'" + m.dtype + "' size mismatch for '" + name +
                             "': got " + std::to_string(nbytes) +
                             " bytes, expected " + std::to_string(expect));
  if (sh.data_start + m.end > sh.file_size)
    throw std::runtime_error("tensor data out of range for '" + name + "'");
  return es;
}

// Convert n on-disk elements (dtype F32/F16/BF16) to float staging.
void convert_block_to_f32(const char* src, const std::string& dtype,
                          float* dst, size_t n) {
  if (dtype == "F32") {
    std::memcpy(dst, src, n * 4);
  } else if (dtype == "BF16") {
    auto* h = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(h[i]);
  } else {  // F16
    auto* h = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(h[i]);
  }
}

template <class ExecSpace, class Scalar>
void stream_to_mat(const ShardFile& sh, const std::string& name,
                   const TensorMeta& m, size_t rows, size_t cols,
                   const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>&
                       dst) {
  if (dst.extent(0) != rows || dst.extent(1) != cols)
    throw std::runtime_error("stream_to_mat: view shape mismatch");
  check_tensor(sh, name, m, {rows, cols});
  if (rows == 0 || cols == 0) return;
  size_t rows_per_chunk =
      std::max<size_t>(1, kStreamChunkElems / std::max<size_t>(cols, 1));
#if KOKKOSDNN_HAS_MMAP
  for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
    size_t r1 = std::min(rows, r0 + rows_per_chunk);
    size_t n = (r1 - r0) * cols;
    std::vector<float> stage(n);
    convert_block_to_f32(
        sh.data_ptr(m.begin + r0 * cols * dtype_elem_bytes(m.dtype, name)),
        m.dtype, stage.data(), n);
    auto h = Kokkos::create_mirror_view(
        Kokkos::subview(dst, std::make_pair(r0, r1), Kokkos::ALL()));
    for (size_t i = 0; i < r1 - r0; ++i)
      for (size_t j = 0; j < cols; ++j)
        h(i, j) = static_cast<Scalar>(stage[i * cols + j]);
    Kokkos::deep_copy(
        Kokkos::subview(dst, std::make_pair(r0, r1), Kokkos::ALL()), h);
  }
#else
  size_t es = dtype_elem_bytes(m.dtype, name);
  std::vector<char> raw;
  std::vector<float> stage;
  for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
    size_t r1 = std::min(rows, r0 + rows_per_chunk);
    size_t n = (r1 - r0) * cols;
    raw.resize(n * es);
    sh.read_bytes(m.begin + r0 * cols * es, raw.data(), n * es);
    stage.resize(n);
    convert_block_to_f32(raw.data(), m.dtype, stage.data(), n);
    auto h = Kokkos::create_mirror_view(
        Kokkos::subview(dst, std::make_pair(r0, r1), Kokkos::ALL()));
    for (size_t i = 0; i < r1 - r0; ++i)
      for (size_t j = 0; j < cols; ++j)
        h(i, j) = static_cast<Scalar>(stage[i * cols + j]);
    Kokkos::deep_copy(
        Kokkos::subview(dst, std::make_pair(r0, r1), Kokkos::ALL()), h);
  }
#endif
}

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
    std::vector<float> stage(c);
    convert_block_to_f32(
        sh.data_ptr(m.begin + o * dtype_elem_bytes(m.dtype, name)), m.dtype,
        stage.data(), c);
    auto h = Kokkos::create_mirror_view(
        Kokkos::subview(dst, std::make_pair(o, o + c)));
    for (size_t i = 0; i < c; ++i) h(i) = static_cast<Scalar>(stage[i]);
    Kokkos::deep_copy(Kokkos::subview(dst, std::make_pair(o, o + c)), h);
  }
#else
  size_t es = dtype_elem_bytes(m.dtype, name);
  std::vector<char> raw;
  std::vector<float> stage;
  for (size_t o = 0; o < n; o += kStreamChunkElems) {
    size_t c = std::min(kStreamChunkElems, n - o);
    raw.resize(c * es);
    sh.read_bytes(m.begin + o * es, raw.data(), c * es);
    stage.resize(c);
    convert_block_to_f32(raw.data(), m.dtype, stage.data(), c);
    auto h = Kokkos::create_mirror_view(
        Kokkos::subview(dst, std::make_pair(o, o + c)));
    for (size_t i = 0; i < c; ++i) h(i) = static_cast<Scalar>(stage[i]);
    Kokkos::deep_copy(Kokkos::subview(dst, std::make_pair(o, o + c)), h);
  }
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

std::vector<std::string> list_checkpoint_tensors(
    const std::vector<std::string>& files) {
  if (files.empty()) throw std::runtime_error("no checkpoint files given");
  std::vector<std::string> out;
  for (auto& fp : files) {
    ShardFile sh(fp);
    for (auto& kv : sh.metas) out.push_back(kv.first);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  embed = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sg_embed", cfg.vocab_size, cfg.hidden_size);
  final_norm =
      Kokkos::View<Scalar*, ExecSpace>("sg_final_norm", cfg.hidden_size);
  blocks.resize((size_t)cfg.num_hidden_layers);
  const int hq = cfg.num_attention_heads * cfg.head_dim;
  const int hkv = cfg.num_key_value_heads * cfg.head_dim;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    auto& b = blocks[(size_t)l];
    b.wq = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sg_wq", hq, cfg.hidden_size);
    b.wk = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sg_wk", hkv, cfg.hidden_size);
    b.wv = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sg_wv", hkv, cfg.hidden_size);
    b.wo = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sg_wo", cfg.hidden_size, hq);
    b.wgate = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sg_wgate", cfg.intermediate_size, cfg.hidden_size);
    b.wup = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sg_wup", cfg.intermediate_size, cfg.hidden_size);
    b.wdown = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sg_wdown", cfg.hidden_size, cfg.intermediate_size);
    b.input_norm =
        Kokkos::View<Scalar*, ExecSpace>("sg_in", cfg.hidden_size);
    b.post_attn_norm =
        Kokkos::View<Scalar*, ExecSpace>("sg_post_attn", cfg.hidden_size);
    b.pre_ffn_norm =
        Kokkos::View<Scalar*, ExecSpace>("sg_pre_ffn", cfg.hidden_size);
    b.post_ffn_norm =
        Kokkos::View<Scalar*, ExecSpace>("sg_post_ffn", cfg.hidden_size);
  }
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
      h(i) = static_cast<Scalar>(0.0f);  // Gemma2 norm weights init at zero
    Kokkos::deep_copy(v, h);
  };
  fill_mat(w.embed);
  fill_vec(w.final_norm);
  for (auto& b : w.blocks) {
    fill_mat(b.wq);
    fill_mat(b.wk);
    fill_mat(b.wv);
    fill_mat(b.wo);
    fill_mat(b.wgate);
    fill_mat(b.wup);
    fill_mat(b.wdown);
    fill_vec(b.input_norm);
    fill_vec(b.post_attn_norm);
    fill_vec(b.pre_ffn_norm);
    fill_vec(b.post_ffn_norm);
  }
}

template <class ExecSpace, class Scalar>
void load_safetensors(const Config& cfg,
                      const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w) {
  cfg.validate();
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
    for (auto& kv : shards[s].metas) found[kv.first] = {s, &kv.second};

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

  const size_t D = (size_t)cfg.hidden_size;
  const size_t F = (size_t)cfg.intermediate_size;
  const size_t Hq = (size_t)cfg.num_attention_heads * (size_t)cfg.head_dim;
  const size_t Hkv = (size_t)cfg.num_key_value_heads * (size_t)cfg.head_dim;
  load_mat("model.embed_tokens.weight", cfg.vocab_size, D, w.embed);
  load_vec("model.norm.weight", w.final_norm);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    auto& b = w.blocks[(size_t)l];
    load_mat(p + "self_attn.q_proj.weight", Hq, D, b.wq);
    load_mat(p + "self_attn.k_proj.weight", Hkv, D, b.wk);
    load_mat(p + "self_attn.v_proj.weight", Hkv, D, b.wv);
    load_mat(p + "self_attn.o_proj.weight", D, Hq, b.wo);
    load_mat(p + "mlp.gate_proj.weight", F, D, b.wgate);
    load_mat(p + "mlp.up_proj.weight", F, D, b.wup);
    load_mat(p + "mlp.down_proj.weight", D, F, b.wdown);
    load_vec(p + "input_layernorm.weight", b.input_norm);
    load_vec(p + "post_attention_layernorm.weight", b.post_attn_norm);
    load_vec(p + "pre_feedforward_layernorm.weight", b.pre_ffn_norm);
    load_vec(p + "post_feedforward_layernorm.weight", b.post_ffn_norm);
  }
}

namespace {

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + p);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

size_t tok_skip_ws(const std::string& s, size_t p) {
  while (p < s.size() &&
         (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
    ++p;
  return p;
}

void utf8_encode_cp(uint32_t cp, std::string& out) {
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
}

std::string json_parse_string(const std::string& s, size_t& q) {
  ++q;
  std::string out;
  while (q < s.size()) {
    char c = s[q++];
    if (c == '"') break;
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
          throw std::runtime_error("tokenizer: bad \\u escape");
        uint32_t cp = std::stoul(s.substr(q, 4), nullptr, 16);
        q += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && q + 6 <= s.size() &&
            s[q] == '\\' && s[q + 1] == 'u') {
          uint32_t lo = std::stoul(s.substr(q + 2, 4), nullptr, 16);
          if (lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            q += 6;
          }
        }
        utf8_encode_cp(cp, out);
        break;
      }
      default: out += e; break;
    }
  }
  return out;
}

size_t find_key_value(const std::string& js, const char* key, size_t from) {
  std::string pat = std::string("\"") + key + "\"";
  size_t p = from;
  while ((p = js.find(pat, p)) != std::string::npos) {
    size_t c = tok_skip_ws(js, p + pat.size());
    if (c < js.size() && js[c] == ':') return c + 1;
    p += pat.size();
  }
  return std::string::npos;
}

size_t match_bracket(const std::string& js, size_t open, char oc, char cc) {
  bool in_str = false, esc = false;
  int depth = 0;
  for (size_t i = open; i < js.size(); ++i) {
    char c = js[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
    } else {
      if (c == '"') in_str = true;
      else if (c == oc) ++depth;
      else if (c == cc && --depth == 0) return i;
    }
  }
  return std::string::npos;
}

std::string object_body(const std::string& js, const char* key, size_t from,
                        const char* what) {
  size_t v = find_key_value(js, key, from);
  if (v == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: no ") + what);
  v = tok_skip_ws(js, v);
  if (v >= js.size() || js[v] != '{')
    throw std::runtime_error(std::string("tokenizer: ") + what +
                             " is not an object");
  size_t close = match_bracket(js, v, '{', '}');
  if (close == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: unbalanced ") + what);
  return js.substr(v + 1, close - v - 1);
}

std::string array_body(const std::string& js, const char* key, size_t from,
                       const char* what) {
  size_t v = find_key_value(js, key, from);
  if (v == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: no ") + what);
  v = tok_skip_ws(js, v);
  if (v >= js.size() || js[v] != '[')
    throw std::runtime_error(std::string("tokenizer: ") + what +
                             " is not an array");
  size_t close = match_bracket(js, v, '[', ']');
  if (close == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: unbalanced ") + what);
  return js.substr(v + 1, close - v - 1);
}

long parse_number(const std::string& s, size_t& p) {
  p = tok_skip_ws(s, p);
  size_t e = p;
  if (e < s.size() && (s[e] == '-' || s[e] == '+')) ++e;
  while (e < s.size() && std::isdigit((unsigned char)s[e])) ++e;
  if (e == p) throw std::runtime_error("tokenizer: expected number");
  long v = std::stol(s.substr(p, e - p));
  p = e;
  return v;
}

inline const char* sp_space() { return "\xe2\x96\x81"; }

std::string normalize_spaces(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == ' ')
      out += sp_space();
    else
      out += c;
  }
  return out;
}

std::string byte_token(unsigned b) {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "<0x%02X>", b & 0xFF);
  return std::string(buf);
}

int parse_byte_token(const std::string& tok) {
  if (tok.size() != 6 || tok[0] != '<' || tok[1] != '0' ||
      (tok[2] != 'x' && tok[2] != 'X') || tok[5] != '>')
    return -1;
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  int hi = hex(tok[3]), lo = hex(tok[4]);
  if (hi < 0 || lo < 0) return -1;
  return (hi << 4) | lo;
}

size_t utf8_len_here(const std::string& s, size_t i) {
  unsigned char c = (unsigned char)s[i];
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6 && i + 1 < s.size() &&
      (((unsigned char)s[i + 1] & 0xC0) == 0x80))
    return 2;
  if ((c >> 4) == 0xE && i + 2 < s.size() &&
      (((unsigned char)s[i + 1] & 0xC0) == 0x80) &&
      (((unsigned char)s[i + 2] & 0xC0) == 0x80))
    return 3;
  if ((c >> 3) == 0x1E && i + 3 < s.size() &&
      (((unsigned char)s[i + 1] & 0xC0) == 0x80) &&
      (((unsigned char)s[i + 2] & 0xC0) == 0x80) &&
      (((unsigned char)s[i + 3] & 0xC0) == 0x80))
    return 4;
  return 1;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

void Tokenizer::load(const std::string& path) {
  path_ = path;
  std::string js = read_file(path);

  token_to_id_.clear();
  merge_rank_.clear();
  added_tokens_.clear();
  special_ids_.clear();
  special_content_.clear();

  size_t model_pos = find_key_value(js, "model", 0);
  if (model_pos == std::string::npos)
    throw std::runtime_error("tokenizer: no model object");
  model_pos = tok_skip_ws(js, model_pos);
  if (model_pos >= js.size() || js[model_pos] != '{')
    throw std::runtime_error("tokenizer: model is not an object");
  size_t model_close = match_bracket(js, model_pos, '{', '}');
  if (model_close == std::string::npos)
    throw std::runtime_error("tokenizer: unbalanced model object");

  std::string vocab = object_body(js, "vocab", model_pos, "model.vocab");
  {
    std::unordered_map<int, std::string> idmap;
    int max_id = -1;
    size_t i = 0;
    while (i < vocab.size()) {
      size_t q = vocab.find('"', i);
      if (q == std::string::npos) break;
      std::string tok = json_parse_string(vocab, q);
      size_t colon = vocab.find(':', q);
      if (colon == std::string::npos) break;
      size_t p = colon + 1;
      long id = parse_number(vocab, p);
      token_to_id_[tok] = (int)id;
      idmap[(int)id] = tok;
      if (id > max_id) max_id = (int)id;
      i = p;
    }
    if (max_id < 0) throw std::runtime_error("tokenizer: empty vocab");
    vocab_size_ = max_id + 1;
    id_to_token_.assign((size_t)vocab_size_, "");
    for (auto& kv : idmap) id_to_token_[(size_t)kv.first] = kv.second;
  }

  {
    std::string merges = array_body(js, "merges", model_pos, "model.merges");
    size_t j = 0;
    int rank = 0;
    while (j < merges.size()) {
      j = tok_skip_ws(merges, j);
      if (j >= merges.size()) break;
      if (merges[j] == ']') break;
      if (merges[j] == ',') {
        ++j;
        continue;
      }
      if (merges[j] == '[') {
        size_t close = match_bracket(merges, j, '[', ']');
        if (close == std::string::npos)
          throw std::runtime_error("tokenizer: unbalanced merge pair");
        std::string pair = merges.substr(j + 1, close - j - 1);
        size_t q = tok_skip_ws(pair, 0);
        if (q >= pair.size() || pair[q] != '"')
          throw std::runtime_error("tokenizer: bad merge pair");
        std::string a = json_parse_string(pair, q);
        q = tok_skip_ws(pair, q);
        if (q < pair.size() && pair[q] == ',') ++q;
        q = tok_skip_ws(pair, q);
        if (q >= pair.size() || pair[q] != '"')
          throw std::runtime_error("tokenizer: bad merge pair");
        std::string b = json_parse_string(pair, q);
        merge_rank_[a + " " + b] = rank++;
        j = close + 1;
      } else if (merges[j] == '"') {
        size_t q = j;
        std::string m = json_parse_string(merges, q);
        merge_rank_[m] = rank++;
        j = q;
      } else {
        throw std::runtime_error("tokenizer: bad merges entry");
      }
    }
  }

  {
    std::string added = array_body(js, "added_tokens", 0, "added_tokens");
    size_t j = 0;
    while (j < added.size()) {
      size_t o = added.find('{', j);
      if (o == std::string::npos) break;
      size_t c = match_bracket(added, o, '{', '}');
      if (c == std::string::npos)
        throw std::runtime_error("tokenizer: unbalanced added_token");
      std::string obj = added.substr(o, c - o + 1);
      size_t pv = find_key_value(obj, "id", 0);
      size_t cv = find_key_value(obj, "content", 0);
      if (pv == std::string::npos || cv == std::string::npos) {
        j = c + 1;
        continue;
      }
      size_t p = pv;
      long id = parse_number(obj, p);
      size_t q = tok_skip_ws(obj, cv);
      std::string content;
      if (q < obj.size() && obj[q] == '"')
        content = json_parse_string(obj, q);
      else {
        j = c + 1;
        continue;
      }
      bool special = true;
      size_t sv = find_key_value(obj, "special", 0);
      if (sv != std::string::npos) {
        size_t t = tok_skip_ws(obj, sv);
        if (obj.compare(t, 4, "true") == 0) special = true;
        else if (obj.compare(t, 5, "false") == 0) special = false;
      }
      if (id >= 0 && !content.empty()) {
        auto it = token_to_id_.find(content);
        if (it == token_to_id_.end()) {
          token_to_id_[content] = (int)id;
          if (id >= (int)id_to_token_.size()) id_to_token_.resize(id + 1);
          id_to_token_[id] = content;
          if (id + 1 > vocab_size_) vocab_size_ = (int)id + 1;
        }
        added_tokens_.push_back({(int)id, content, special});
        if (special) {
          special_ids_.insert((int)id);
          special_content_[(int)id] = content;
        }
      }
      j = c + 1;
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const AddedToken& a, const AddedToken& b) {
                return a.content.size() > b.content.size();
              });
  }

  auto resolve = [&](const char* content, int fallback) {
    for (auto& t : added_tokens_)
      if (t.content == content && t.special) return t.id;
    auto it = token_to_id_.find(content);
    if (it != token_to_id_.end()) return it->second;
    return fallback;
  };
  bos_id_ = resolve("<bos>", 2);
  eos_id_ = resolve("<eos>", 1);
  pad_id_ = resolve("<pad>", 0);
  unk_id_ = resolve("<unk>", 3);

  {
    size_t slash = path_.find_last_of('/');
    if (slash != std::string::npos) {
      try {
        std::string tcfg =
            read_file(path_.substr(0, slash + 1) + "tokenizer_config.json");
        auto override_from_config = [&](const char* key, int& target) {
          size_t kv = find_key_value(tcfg, key, 0);
          if (kv == std::string::npos) return;
          size_t q = tok_skip_ws(tcfg, kv);
          if (q >= tcfg.size() || tcfg[q] != '"') return;
          std::string tok = json_parse_string(tcfg, q);
          auto it = token_to_id_.find(tok);
          if (it != token_to_id_.end()) target = it->second;
        };
        override_from_config("bos_token", bos_id_);
        override_from_config("eos_token", eos_id_);
      } catch (...) {
      }
    }
  }
}

void Tokenizer::bpe_piece(const std::string& piece,
                          std::vector<int>& out) const {
  if (piece.empty()) return;
  std::vector<std::string> parts;
  for (size_t i = 0; i < piece.size();) {
    size_t len = utf8_len_here(piece, i);
    std::string ch = piece.substr(i, len);
    if (token_to_id_.find(ch) != token_to_id_.end()) {
      parts.push_back(ch);
    } else {
      for (size_t k = 0; k < len; ++k)
        parts.push_back(byte_token((unsigned char)piece[i + k]));
    }
    i += len;
  }
  while (parts.size() > 1) {
    int best_rank = -1;
    size_t best = 0;
    for (size_t k = 0; k + 1 < parts.size(); ++k) {
      auto it = merge_rank_.find(parts[k] + " " + parts[k + 1]);
      if (it != merge_rank_.end() &&
          (best_rank < 0 || it->second < best_rank)) {
        best_rank = it->second;
        best = k;
      }
    }
    if (best_rank < 0) break;
    parts[best] += parts[best + 1];
    parts.erase(parts.begin() + (long)best + 1);
  }
  for (auto& s : parts) {
    auto it = token_to_id_.find(s);
    if (it == token_to_id_.end())
      out.push_back(unk_id_);
    else
      out.push_back(it->second);
  }
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                   bool add_bos) const {
  std::vector<int> ids;
  if (add_bos) ids.push_back(bos_id_);
  if (text.empty() || !loaded()) return ids;

  auto special_at = [&](size_t pos) -> const AddedToken* {
    if (text[pos] != '<') return nullptr;
    for (auto& t : added_tokens_) {
      if (t.special && text.compare(pos, t.content.size(), t.content) == 0)
        return &t;
    }
    return nullptr;
  };

  std::string buf;
  auto flush = [&]() {
    if (buf.empty()) return;
    bpe_piece(normalize_spaces(buf), ids);
    buf.clear();
  };

  size_t i = 0;
  while (i < text.size()) {
    if (const AddedToken* t = special_at(i)) {
      flush();
      ids.push_back(t->id);
      i += t->content.size();
    } else {
      buf += text[i++];
    }
  }
  flush();
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids,
                              bool skip_special_tokens) const {
  std::string acc;
  for (int id : ids) {
    if (is_special(id)) {
      if (!skip_special_tokens) {
        auto it = special_content_.find(id);
        if (it != special_content_.end()) acc += it->second;
      }
      continue;
    }
    if (id < 0 || id >= (int)id_to_token_.size() ||
        id_to_token_[(size_t)id].empty())
      throw std::runtime_error("tokenizer: invalid id in decode");
    const std::string& tok = id_to_token_[(size_t)id];
    int b = parse_byte_token(tok);
    if (b >= 0)
      acc += (char)b;
    else
      acc += tok;
  }
  std::string out;
  out.reserve(acc.size());
  const char* sp = sp_space();
  for (size_t i = 0; i < acc.size();) {
    if (acc.compare(i, 3, sp) == 0) {
      out += ' ';
      i += 3;
    } else {
      out += acc[i++];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

template <class ExecSpace, class Scalar>
Model<ExecSpace, Scalar>::Model(const Config& c,
                                const Weights<ExecSpace, Scalar>& w)
    : cfg(c), weights(w) {
  cfg.validate();
  if (cfg.hidden_size != (int)weights.embed.extent(1))
    throw std::invalid_argument("SanaGemma Model: weights/config mismatch");
  inv_freq_ = rope_inv_freq<ExecSpace>(cfg.head_dim, cfg.rope_theta);
}

template <class ExecSpace, class Scalar>
void Model<ExecSpace, Scalar>::ensure_scratch(size_t T) {
  if (T <= scratch_cap_ && h_.extent(0) >= T) return;
  const size_t D = (size_t)cfg.hidden_size;
  const size_t F = (size_t)cfg.intermediate_size;
  const size_t NH = (size_t)cfg.num_attention_heads;
  const size_t NKV = (size_t)cfg.num_key_value_heads;
  const size_t DH = (size_t)cfg.head_dim;
  h_ = MatR("sg_h", T, D);
  tmp_ = MatR("sg_tmp", T, D);
  tmp2_ = MatR("sg_tmp2", T, D);
  q2_ = MatR("sg_q2", T, NH * DH);
  k2_ = MatR("sg_k2", T, NKV * DH);
  v2_ = MatR("sg_v2", T, NKV * DH);
  af_ = MatR("sg_af", T, NH * DH);
  proj_ = MatR("sg_proj", T, D);
  gate_ = MatR("sg_gate", T, F);
  up_ = MatR("sg_up", T, F);
  act_ = MatR("sg_act", T, F);
  down_ = MatR("sg_down", T, D);
  q3_ = View3("sg_q3", T, NH, DH);
  k3_ = View3("sg_k3", T, NKV, DH);
  v3_ = View3("sg_v3", T, NKV, DH);
  ao_ = View3("sg_ao", T, NH, DH);
  pos_ = ViewInt("sg_pos", T);
  tok_ = ViewInt("sg_tok", T);
  valid_ = ViewChar("sg_valid", T);
  scratch_cap_ = T;
}

template <class ExecSpace, class Scalar>
std::vector<float> Model<ExecSpace, Scalar>::forward(
    const std::vector<int>& ids, const std::vector<char>& key_valid) {
  const size_t T = ids.size();
  if (T == 0) throw std::invalid_argument("SanaGemma Model::forward: no ids");
  if (key_valid.size() != T)
    throw std::invalid_argument(
        "SanaGemma Model::forward: ids/mask length mismatch");
  if (T > (size_t)cfg.max_position_embeddings)
    throw std::runtime_error(
        "SanaGemma Model::forward: sequence exceeds max_position_embeddings");
  const int D = cfg.hidden_size, NH = cfg.num_attention_heads,
            NKV = cfg.num_key_value_heads, DH = cfg.head_dim,
            NL = cfg.num_hidden_layers;
  const float EPS = cfg.rms_norm_eps;
  const float SCALE = cfg.attn_scale();
  const float SOFTCAP = cfg.attn_logit_softcapping;
  const float EMBED_SCALE = cfg.embed_scale();
  const int HqD = NH * DH;
  ensure_scratch(T);

  // Carve T-row working views out of the capacity buffers (row-range
  // subviews share the allocation and keep the LayoutRight layout, so none
  // of this allocates; repeated calls with different T stay exact).
  auto rows = std::make_pair(size_t(0), T);
  auto all = Kokkos::ALL();
  auto h = Kokkos::subview(h_, rows, all);
  auto tmp = Kokkos::subview(tmp_, rows, all);
  auto tmp2 = Kokkos::subview(tmp2_, rows, all);
  auto q2 = Kokkos::subview(q2_, rows, all);
  auto k2 = Kokkos::subview(k2_, rows, all);
  auto v2 = Kokkos::subview(v2_, rows, all);
  auto af = Kokkos::subview(af_, rows, all);
  auto proj = Kokkos::subview(proj_, rows, all);
  auto gate = Kokkos::subview(gate_, rows, all);
  auto up = Kokkos::subview(up_, rows, all);
  auto act = Kokkos::subview(act_, rows, all);
  auto down = Kokkos::subview(down_, rows, all);
  auto q3 = Kokkos::subview(q3_, rows, all, all);
  auto k3 = Kokkos::subview(k3_, rows, all, all);
  auto v3 = Kokkos::subview(v3_, rows, all, all);
  auto ao = Kokkos::subview(ao_, rows, all, all);
  auto pos = Kokkos::subview(pos_, rows);
  auto tok = Kokkos::subview(tok_, rows);
  auto valid = Kokkos::subview(valid_, rows);
  auto inv = inv_freq_;
  const DNNHandle<ExecSpace>& handle = handle_;

  {
    auto th = Kokkos::create_mirror_view(tok);
    auto vh = Kokkos::create_mirror_view(valid);
    auto ph = Kokkos::create_mirror_view(pos);
    for (size_t t = 0; t < T; ++t) {
      if (ids[t] < 0 || ids[t] >= cfg.vocab_size)
        throw std::invalid_argument("SanaGemma Model::forward: id " +
                                    std::to_string(ids[t]) + " out of range");
      th(t) = ids[t];
      vh(t) = key_valid[t] ? 1 : 0;
      ph(t) = (int)t;
    }
    Kokkos::deep_copy(tok, th);
    Kokkos::deep_copy(valid, vh);
    Kokkos::deep_copy(pos, ph);
  }

  embedding_gather_scaled<ExecSpace, Scalar>(weights.embed, tok, EMBED_SCALE,
                                            h);
  for (int l = 0; l < NL; ++l) {
    auto& b = weights.blocks[(size_t)l];
    auto wq = b.wq, wk = b.wk, wv = b.wv, wo = b.wo;
    auto wg = b.wgate, wu = b.wup, wd = b.wdown;
    auto in_n = b.input_norm, po_n = b.post_attn_norm;
    auto pf_n = b.pre_ffn_norm, qf_n = b.post_ffn_norm;

    rmsnorm_one_plus<ExecSpace, Scalar>(h, in_n, EPS, tmp);
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wq,
                            static_cast<Scalar>(0), q2);
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wk,
                            static_cast<Scalar>(0), k2);
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wv,
                            static_cast<Scalar>(0), v2);
    Kokkos::parallel_for(
        "sg_reshape_q",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, (size_t)HqD}),
        KOKKOS_LAMBDA(size_t t, size_t i) {
          q3(t, i / (size_t)DH, i % (size_t)DH) = q2(t, i);
        });
    Kokkos::parallel_for(
        "sg_reshape_kv",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {T, (size_t)NKV * (size_t)DH}),
        KOKKOS_LAMBDA(size_t t, size_t i) {
          k3(t, i / (size_t)DH, i % (size_t)DH) = k2(t, i);
          v3(t, i / (size_t)DH, i % (size_t)DH) = v2(t, i);
        });
    rope_neoX_inplace<ExecSpace, Scalar>(q3, k3, inv, DH, pos);
    attention_gqa_single_softcap<ExecSpace, Scalar>(handle, q3, k3, v3, valid,
                                                       0, ao, SCALE, SOFTCAP);
    Kokkos::parallel_for(
        "sg_flatten",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, (size_t)HqD}),
        KOKKOS_LAMBDA(size_t t, size_t i) {
          af(t, i) = ao(t, i / (size_t)DH, i % (size_t)DH);
        });
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), af, wo,
                            static_cast<Scalar>(0), proj);
    rmsnorm_one_plus<ExecSpace, Scalar>(proj, po_n, EPS, tmp2);
    Kokkos::parallel_for(
        "sg_resid1",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, (size_t)D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += tmp2(t, d); });

    rmsnorm_one_plus<ExecSpace, Scalar>(h, pf_n, EPS, tmp);
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wg,
                            static_cast<Scalar>(0), gate);
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), tmp, wu,
                            static_cast<Scalar>(0), up);
    gated_gelu_multiply<ExecSpace, Scalar>(gate, up, act, GeluVariant::Tanh);
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1), act, wd,
                            static_cast<Scalar>(0), down);
    rmsnorm_one_plus<ExecSpace, Scalar>(down, qf_n, EPS, tmp2);
    Kokkos::parallel_for(
        "sg_resid2",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, (size_t)D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { h(t, d) += tmp2(t, d); });
  }
  rmsnorm_one_plus<ExecSpace, Scalar>(h, weights.final_norm, EPS, tmp);
  Kokkos::fence();

  std::vector<float> out(T * (size_t)D);
  auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tmp);
  size_t k = 0;
  for (size_t t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) out[k++] = static_cast<float>(hh(t, d));
  return out;
}

// ---------------------------------------------------------------------------
// Sana caption helpers + checkpoint path helpers
// ---------------------------------------------------------------------------

std::vector<int> sana_select_index(int length_all, int max_length) {
  if (length_all <= 0 || max_length <= 0)
    throw std::invalid_argument("sana_select_index: lengths must be > 0");
  if (length_all < max_length)
    throw std::invalid_argument(
        "sana_select_index: length_all < max_length (tokenize must pad to "
        "at least max_length)");
  std::vector<int> sel;
  sel.reserve((size_t)max_length);
  sel.push_back(0);
  for (int i = length_all - max_length + 1; i < length_all; ++i)
    sel.push_back(i);
  return sel;
}

namespace {

}  // namespace

std::vector<std::string> collect_safetensors(const std::string& path) {
  // Single file?
  {
    std::ifstream f(path, std::ios::binary);
    if (f) return {path};
  }
#if SANAGEMMA_HAS_DIRENT
  DIR* dp = ::opendir(path.c_str());
  if (dp) {
    std::vector<std::string> out;
    while (dirent* e = ::readdir(dp)) {
      std::string n = e->d_name;
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0)
        out.push_back(path + "/" + n);
    }
    ::closedir(dp);
    if (!out.empty()) {
      std::sort(out.begin(), out.end());
      return out;
    }
    throw std::runtime_error("collect_safetensors: directory '" + path +
                             "' holds no *.safetensors files");
  }
#endif
  throw std::runtime_error(
      "collect_safetensors: '" + path +
      "' is not a readable file; pass a .safetensors file or a directory "
      "containing them (multi-shard: shards load in sorted order)");
}

std::string sibling_file(const std::string& weights_path,
                         const std::string& name) {
  std::string dir;
  {
    std::ifstream f(weights_path, std::ios::binary);
    if (f) {
      size_t slash = weights_path.find_last_of("/\\");
      dir = (slash == std::string::npos) ? "." : weights_path.substr(0, slash);
    } else {
      dir = weights_path;  // treat as a directory
    }
  }
  std::string cand = dir + "/" + name;
  std::ifstream g(cand, std::ios::binary);
  if (g) return cand;
  return "";
}

// Explicit instantiations (float only; Serial always, others when enabled).
template void Weights<Kokkos::Serial, float>::allocate(const Config&);
template void random_init<Kokkos::Serial, float>(const Config&,
                                                Weights<Kokkos::Serial, float>&,
                                                uint64_t);
template void load_safetensors<Kokkos::Serial, float>(
    const Config&, const std::vector<std::string>&,
    Weights<Kokkos::Serial, float>&);
template class Model<Kokkos::Serial, float>;
#ifdef KOKKOS_ENABLE_OPENMP
template void Weights<Kokkos::OpenMP, float>::allocate(const Config&);
template void random_init<Kokkos::OpenMP, float>(const Config&,
                                                Weights<Kokkos::OpenMP, float>&,
                                                uint64_t);
template void load_safetensors<Kokkos::OpenMP, float>(
    const Config&, const std::vector<std::string>&,
    Weights<Kokkos::OpenMP, float>&);
template class Model<Kokkos::OpenMP, float>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template void Weights<Kokkos::Cuda, float>::allocate(const Config&);
template void random_init<Kokkos::Cuda, float>(const Config&,
                                              Weights<Kokkos::Cuda, float>&,
                                              uint64_t);
template void load_safetensors<Kokkos::Cuda, float>(
    const Config&, const std::vector<std::string>&,
    Weights<Kokkos::Cuda, float>&);
template class Model<Kokkos::Cuda, float>;
#endif

}  // namespace SanaGemma
}  // namespace KokkosDNN
