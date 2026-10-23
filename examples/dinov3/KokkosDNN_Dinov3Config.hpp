// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_DINOV3_CONFIG_HPP
#define KOKKOSDNN_DINOV3_CONFIG_HPP

#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Dinov3 {

namespace config_detail {

// ---- Minimal JSON lookup (dependency-free) ----
// Flat key lookup on the root object, string-aware (keys inside nested
// strings/objects are not confused). Follows the minimal-parser style used
// by examples/llama/KokkosDNN_LlamaConfig.hpp. Only the value span of a
// requested top-level key is extracted; callers interpret it.
inline void skip_ws(const std::string& s, size_t& p) {
  while (p < s.size() &&
         (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
    ++p;
}

// Parse a JSON string literal starting at s[p] == '"'. Returns decoded
// string, advances p past the closing quote. Handles common escapes and
// \uXXXX (BMP; astral pairs via surrogate combination).
inline std::string parse_string(const std::string& s, size_t& p) {
  if (p >= s.size() || s[p] != '"')
    throw std::runtime_error("config: expected string");
  ++p;
  std::string out;
  while (true) {
    if (p >= s.size()) throw std::runtime_error("config: unterminated string");
    char c = s[p++];
    if (c == '"') break;
    if (c != '\\') {
      out += c;
      continue;
    }
    if (p >= s.size()) throw std::runtime_error("config: bad escape");
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
        if (p + 4 > s.size()) throw std::runtime_error("config: bad \\u");
        unsigned cp = 0;
        for (int i = 0; i < 4; ++i) {
          char h = s[p++];
          cp <<= 4;
          if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
          else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
          else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
          else throw std::runtime_error("config: bad \\u");
        }
        if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= s.size() &&
            s[p] == '\\' && s[p + 1] == 'u') {
          p += 2;
          unsigned lo = 0;
          for (int i = 0; i < 4; ++i) {
            char h = s[p++];
            lo <<= 4;
            if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
            else throw std::runtime_error("config: bad \\u");
          }
          if (lo >= 0xDC00 && lo <= 0xDFFF)
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        }
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) {
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
      default: throw std::runtime_error("config: bad escape");
    }
  }
  return out;
}

// Advance p past one JSON value starting at p (string-aware for {} [] "").
inline void skip_value(const std::string& s, size_t& p) {
  skip_ws(s, p);
  if (p >= s.size()) throw std::runtime_error("config: unexpected end");
  char c = s[p];
  if (c == '"') {
    (void)parse_string(s, p);
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
        else if (k == close) {
          if (--depth == 0) {
            ++p;
            return;
          }
        }
      }
    }
    throw std::runtime_error("config: unbalanced JSON");
  }
  // Literal / number: run to the next structural character.
  while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;
}

// Find the value span [vs, ve) of top-level `key` in root object `s`.
// Returns false when absent. Throws on malformed JSON.
inline bool find_key(const std::string& s, const std::string& key,
                     size_t& vs, size_t& ve) {
  size_t p = 0;
  skip_ws(s, p);
  if (p >= s.size() || s[p] != '{')
    throw std::runtime_error("config: root must be a JSON object");
  ++p;
  skip_ws(s, p);
  if (p < s.size() && s[p] == '}') return false;  // empty object
  while (true) {
    skip_ws(s, p);
    if (p >= s.size() || s[p] != '"')
      throw std::runtime_error("config: expected string key");
    std::string k = parse_string(s, p);
    skip_ws(s, p);
    if (p >= s.size() || s[p] != ':')
      throw std::runtime_error("config: expected ':' after key");
    ++p;
    size_t v = p;
    skip_value(s, p);
    if (k == key) {
      vs = v;
      ve = p;
      return true;
    }
    skip_ws(s, p);
    if (p >= s.size()) throw std::runtime_error("config: unexpected end");
    if (s[p] == ',') {
      ++p;
      continue;
    }
    if (s[p] == '}') return false;
    throw std::runtime_error("config: expected ',' or '}'");
  }
}

inline std::string trim(const std::string& t) {
  size_t a = 0;
  while (a < t.size() && (t[a] == ' ' || t[a] == '\t' || t[a] == '\n' ||
                          t[a] == '\r'))
    ++a;
  size_t b = t.size();
  while (b > a && (t[b - 1] == ' ' || t[b - 1] == '\t' || t[b - 1] == '\n' ||
                   t[b - 1] == '\r'))
    --b;
  return t.substr(a, b - a);
}

// Get raw value text of `key` ("null" included). Returns false if absent.
inline bool get_raw(const std::string& s, const std::string& key,
                    std::string& out) {
  size_t vs = 0, ve = 0;
  if (!find_key(s, key, vs, ve)) return false;
  out = trim(s.substr(vs, ve - vs));
  return true;
}

inline bool is_null(const std::string& v) { return v == "null"; }

// Parse a JSON number (no trailing garbage). Throws on bad input.
inline double parse_num(const std::string& v, const std::string& key) {
  try {
    size_t n = 0;
    double d = std::stod(v, &n);
    if (n != v.size())
      throw std::runtime_error("trailing characters");
    if (!std::isfinite(d))
      throw std::runtime_error("non-finite value");
    return d;
  } catch (const std::runtime_error&) {
    throw;
  } catch (const std::exception&) {
    throw std::runtime_error("config: '" + key + "' is not a number");
  }
}

inline bool get_num(const std::string& s, const std::string& key, double& out) {
  std::string v;
  if (!get_raw(s, key, v)) return false;
  if (is_null(v))
    throw std::runtime_error("config: '" + key + "' must not be null");
  out = parse_num(v, key);
  return true;
}

// Integer field: JSON numbers only (quoted strings rejected).
inline bool get_int(const std::string& s, const std::string& key,
                    long long& out) {
  double d = 0;
  if (!get_num(s, key, d)) return false;
  if (std::floor(d) != d)
    throw std::runtime_error("config: '" + key + "' must be an integer");
  out = std::llround(d);
  return true;
}

inline bool get_bool(const std::string& s, const std::string& key, bool& out) {
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
  if (v == "0" || v == "1") {  // tolerate numeric booleans
    out = (v == "1");
    return true;
  }
  throw std::runtime_error("config: '" + key +
                           "' must be true/false (boolean)");
}

// Decode a JSON string field (quoted string expected; bare tokens passed
// through for a clearer downstream error). Throws on null.
inline bool get_string(const std::string& s, const std::string& key,
                       std::string& out) {
  std::string v;
  if (!get_raw(s, key, v)) return false;
  std::string t = trim(v);
  if (is_null(t))
    throw std::runtime_error("config: '" + key + "' must not be null");
  if (t.size() >= 2 && t.front() == '"' && t.back() == '"') {
    size_t p = 0;
    out      = parse_string(t, p);
  } else {
    out = t;
  }
  return true;
}

inline int checked_int(long long v, const std::string& key) {
  if (v < 0 || v > 1000000000LL)
    throw std::runtime_error("config: '" + key + "' out of range");
  return (int)v;
}

// Case-insensitive string compare (for hidden_act / model_type).
inline std::string to_lower_str(const std::string& t) {
  std::string out = t;
  for (auto& ch : out) ch = (char)std::tolower((unsigned char)ch);
  return out;
}

inline bool iequals(const std::string& a, const std::string& b) {
  return to_lower_str(a) == to_lower_str(b);
}

}  // namespace config_detail

// Hyperparameters for HF `dinov3_vit` (DINOv3 ViT-B/16 pretrain-lvd1689m).
// Reference: facebook/dinov3-vitb16-pretrain-lvd1689m config.json: patch 16,
// hidden 768, 12 layers, 12 heads (head_dim 64, MHA), MLP 3072 with `gelu`
// (no SwiGLU/gating), 4 register tokens (+1 CLS assembled at runtime),
// image 224, 3 channels, biased q/v/proj/mlp linears (k unbiased),
// layer_norm_eps 1e-5, rope_theta 100 (2D axial RoPE), layerscale_value 1.0
// (scalar init for random_init; the checkpoint's per-channel lambdas
// override at load), use_gated_mlp false, model_type "dinov3_vit".
struct Config {
  int patch_size          = 16;
  int hidden_size         = 768;
  int num_hidden_layers   = 12;
  int num_attention_heads = 12;
  int head_dim            = 64;  // hidden_size / num_attention_heads
  int intermediate_size   = 3072;
  int num_register_tokens = 4;
  int image_size          = 224;
  int num_channels        = 3;
  bool query_bias = true;
  bool key_bias   = false;
  bool value_bias = true;
  bool proj_bias  = true;
  bool mlp_bias   = true;
  double layer_norm_eps   = 1e-5;
  double rope_theta       = 100.0;
  double layerscale_value = 1.0;
  std::string hidden_act = "gelu";
  bool use_gated_mlp     = false;
  std::string model_type = "dinov3_vit";

  static Config vitb16() { return Config{}; }

  // Tiny config for unit tests (no weights needed).
  static Config tiny_test() {
    Config c;
    c.patch_size          = 8;
    c.hidden_size         = 32;
    c.num_hidden_layers   = 2;
    c.num_attention_heads = 4;
    c.head_dim            = 8;  // 32 / 4
    c.intermediate_size   = 64;
    c.num_register_tokens = 1;
    c.image_size          = 32;
    c.num_channels        = 3;
    return c;
  }

  // Recompute head_dim from hidden_size / num_attention_heads. Only applies
  // when the division is exact; otherwise validate() reports the mismatch.
  void recompute_derived() {
    if (num_attention_heads > 0 && hidden_size > 0 &&
        hidden_size % num_attention_heads == 0) {
      head_dim = hidden_size / num_attention_heads;
    }
  }

  // Patch tokens for a square image of `img` px: (img/patch)^2. Returns 0
  // when dims are not usable (validate() rejects those configs).
  int num_patch_tokens(int img) const {
    if (patch_size <= 0 || img <= 0) return 0;
    if (img % patch_size != 0) return 0;
    const int h = img / patch_size;
    return h * h;
  }
  int num_patch_tokens() const { return num_patch_tokens(image_size); }

  // Full sequence length: 1 CLS + registers + patches.
  int num_tokens(int img) const {
    return 1 + num_register_tokens + num_patch_tokens(img);
  }
  int num_tokens() const { return num_tokens(image_size); }

  // Throw std::invalid_argument/std::runtime_error on incompatible dims or
  // unsupported architecture variants. Called by from_json_* and by the
  // future encoder driver.
  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Config::validate: " + m);
    };
    if (patch_size <= 0) bad("patch_size must be > 0");
    if (hidden_size <= 0) bad("hidden_size must be > 0");
    if (num_hidden_layers <= 0) bad("num_hidden_layers must be > 0");
    if (num_attention_heads <= 0) bad("num_attention_heads must be > 0");
    if (head_dim <= 0) bad("head_dim must be > 0");
    if (intermediate_size <= 0) bad("intermediate_size must be > 0");
    if (num_register_tokens < 0) bad("num_register_tokens must be >= 0");
    if (image_size <= 0) bad("image_size must be > 0");
    if (num_channels <= 0) bad("num_channels must be > 0");
    if (hidden_size % num_attention_heads != 0)
      bad("hidden_size (" + std::to_string(hidden_size) +
          ") must be divisible by num_attention_heads (" +
          std::to_string(num_attention_heads) + ")");
    if (head_dim != hidden_size / num_attention_heads)
      bad("head_dim (" + std::to_string(head_dim) +
          ") must equal hidden_size / num_attention_heads (" +
          std::to_string(hidden_size / num_attention_heads) + ")");
    if (image_size % patch_size != 0)
      bad("image_size (" + std::to_string(image_size) +
          ") must be divisible by patch_size (" + std::to_string(patch_size) +
          ")");
    if (!(layer_norm_eps > 0) || !std::isfinite(layer_norm_eps))
      bad("layer_norm_eps must be positive and finite");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (!std::isfinite(layerscale_value))
      bad("layerscale_value must be finite");
    if (!config_detail::iequals(hidden_act, "gelu"))
      throw std::runtime_error(
          "Config::validate: hidden_act '" + hidden_act +
          "' is not supported: only 'gelu' is supported");
    if (use_gated_mlp)
      throw std::runtime_error(
          "Config::validate: 'use_gated_mlp' must be false: gated/SwiGLU MLP "
          "is unsupported (dinov3_vit uses plain gelu MLP)");
    if (!config_detail::iequals(model_type, "dinov3_vit"))
      throw std::runtime_error(
          "Config::validate: model_type '" + model_type +
          "' is not supported: only 'dinov3_vit' is supported");
  }

  // Parse an HF dinov3_vit config.json object. Keys present override `base`;
  // missing keys keep base values (HF vitb16 defaults via vitb16()).
  // Unknown keys are ignored. Throws on malformed JSON or validation
  // failure. `head_dim` is derived (hidden/heads), never read from JSON.
  static Config from_json_string(const std::string& text,
                                 const Config& base = vitb16()) {
    using namespace config_detail;
    Config c = base;
    long long iv = 0;
    double dv    = 0;
    bool bv      = false;
    std::string sv;
    if (get_int(text, "patch_size", iv)) c.patch_size = checked_int(iv, "patch_size");
    if (get_int(text, "hidden_size", iv))
      c.hidden_size = checked_int(iv, "hidden_size");
    if (get_int(text, "num_hidden_layers", iv))
      c.num_hidden_layers = checked_int(iv, "num_hidden_layers");
    if (get_int(text, "num_attention_heads", iv))
      c.num_attention_heads = checked_int(iv, "num_attention_heads");
    if (get_int(text, "intermediate_size", iv))
      c.intermediate_size = checked_int(iv, "intermediate_size");
    if (get_int(text, "num_register_tokens", iv))
      c.num_register_tokens = checked_int(iv, "num_register_tokens");
    if (get_int(text, "image_size", iv))
      c.image_size = checked_int(iv, "image_size");
    if (get_int(text, "num_channels", iv))
      c.num_channels = checked_int(iv, "num_channels");
    if (get_bool(text, "query_bias", bv)) c.query_bias = bv;
    if (get_bool(text, "key_bias", bv)) c.key_bias = bv;
    if (get_bool(text, "value_bias", bv)) c.value_bias = bv;
    if (get_bool(text, "proj_bias", bv)) c.proj_bias = bv;
    if (get_bool(text, "mlp_bias", bv)) c.mlp_bias = bv;
    if (get_num(text, "layer_norm_eps", dv)) c.layer_norm_eps = dv;
    if (get_num(text, "rope_theta", dv)) c.rope_theta = dv;
    if (get_num(text, "layerscale_value", dv)) c.layerscale_value = dv;
    if (get_string(text, "hidden_act", sv)) {
      if (!iequals(sv, "gelu"))
        throw std::runtime_error(
            "config: 'hidden_act' '" + sv +
            "' is not supported: only 'gelu' is supported");
      c.hidden_act = sv;
    }
    if (get_bool(text, "use_gated_mlp", bv)) {
      c.use_gated_mlp = bv;
      if (bv)
        throw std::runtime_error(
            "config: 'use_gated_mlp' must be false: gated/SwiGLU MLP is "
            "unsupported (dinov3_vit uses plain gelu MLP)");
    }
    if (get_string(text, "model_type", sv)) {
      if (!iequals(sv, "dinov3_vit"))
        throw std::runtime_error(
            "config: 'model_type' '" + sv +
            "' is not supported: only 'dinov3_vit' is supported");
      c.model_type = sv;
    }
    c.recompute_derived();
    c.validate();
    return c;
  }

  static Config from_json_file(const std::string& path,
                               const Config& base = vitb16()) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("config: cannot open config file '" + path +
                               "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_string(ss.str(), base);
  }
};

}  // namespace Dinov3
}  // namespace KokkosDNN

#endif
