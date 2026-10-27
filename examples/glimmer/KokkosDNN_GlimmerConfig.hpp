// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GLIMMER_CONFIG_HPP
#define KOKKOSDNN_GLIMMER_CONFIG_HPP

#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Glimmer {

// Supported KV-cache context window. The attention kernel streams over keys
// with O(1) extra memory and prefill is chunked, so the architecture window
// (Muse Glimmer text advertises max_position_embeddings = 131072) is
// supported; the bound below caps the rented KV-cache allocation. The PoC
// *default* stays at 8192 so a plain run does not rent the full-window cache
// by accident; raise via config.json or --max-seq-len.
inline constexpr int kMaxSupportedSeqLen = 131072;
inline constexpr int kMinSupportedSeqLen = 1;

namespace config_detail {

// ---- Minimal JSON lookup (dependency-free) ----
// Flat key lookup on an object, string-aware (keys inside nested
// strings/objects are not confused). Follows the minimal-parser style used
// by the safetensors loader. Only the value span of a requested key is
// extracted; callers interpret it.
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

// Find the value span [vs, ve) of `key` in object `s`.
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

inline int checked_int(long long v, const std::string& key) {
  if (v < 0 || v > 1000000000LL)
    throw std::runtime_error("config: '" + key + "' out of range");
  return (int)v;
}

// Case-insensitive string compare (for hidden_activation / model_type).
inline std::string to_lower_str(const std::string& t) {
  std::string out = t;
  for (auto& ch : out) ch = (char)std::tolower((unsigned char)ch);
  return out;
}

inline bool iequals(const std::string& a, const std::string& b) {
  return to_lower_str(a) == to_lower_str(b);
}

// Decode a JSON string value (quoted) into `out`. Returns false when the
// raw value is absent; throws when present but not a quoted string.
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
    return true;
  }
  throw std::runtime_error("config: '" + key + "' must be a string");
}

}  // namespace config_detail

// Hyperparameters for the Muse Glimmer 30B text path (2026).
// Reference: HF darkc0de/Muse-Glimmer-30B-heretic config.json `text_config`:
// dim 6656, 52 layers, 32 Q / 2 KV heads, EXPLICIT head_dim 128 (note:
// 6656 / 32 = 208, so head_dim is NOT hidden_size / num_attention_heads —
// q_proj is [32*128, 6656], k/v_proj [2*128, 6656], o_proj [6656, 32*128]),
// SwiGLU ffn 19968, 3x-sliding + 1x-full repeat (full layers at idx 3, 7,
// ..., 51; 13 full, 39 sliding), sliding window 2048, per-layer
// layer_rope_theta (500000 on sliding layers, 0 on full layers = use the
// global rope_theta 500000), qk_scale_factor 3.87, output_multiplier
// ~0.19611, final_logit_softcapping 20, rms_norm_eps 1e-5 plus a tighter
// post_norm_eps 1e-8, tie_word_embeddings false (separate lm_head),
// vocab 202048, bos/eos 200000/200001, BF16 checkpoint dtype.
struct Config {
  int dim            = 6656;    // hidden size
  int n_layers       = 52;
  int n_heads        = 32;      // query heads
  int n_kv_heads     = 2;       // GQA kv heads (all layers)
  // Explicit per-head dimension. Unlike Llama (where head_dim is derived),
  // this is an independent architectural value: it must NOT be recomputed
  // from dim / n_heads (which would give 208, not 128).
  int head_dim       = 128;
  int vocab_size     = 202048;
  int ffn_dim        = 19968;   // SwiGLU intermediate
  float rms_norm_eps = 1e-5f;
  // Tighter epsilon for the post-attention / post-FFN RMSNorms.
  float post_norm_eps = 1e-8f;
  // Global RoPE base frequency. Full-attention layers (whose
  // layer_rope_theta entry is 0) use this value.
  float rope_theta   = 500000.0f;
  int sliding_window = 2048;    // sliding-layer causal window
  // Per-layer attention type: false = sliding, true = full. Default is the
  // 3x-sliding + 1x-full repeat (full layers at idx 3, 7, ..., 51).
  std::vector<char> is_full_layer;
  // Per-layer RoPE theta, parallel to is_full_layer. A 0 entry means "use
  // the global rope_theta" (this is how the HF checkpoint marks the 13
  // full-attention layers); positive entries are used directly.
  std::vector<double> layer_rope_theta;
  float final_logit_softcap = 20.0f;
  // Multiplicative factor applied to the QK-normalized attention scores.
  float qk_scale_factor = 3.87f;
  // Scalar multiplier on the final hidden states (before the head).
  double output_multiplier = 0.19611613513818404;
  int max_seq_len    = 8192;  // PoC runtime cap; arch supports 131072
  int bos_token      = 200000;
  int eos_token      = 200001;
  // HF pad_token_id is null (absent); -1 marks "no pad token".
  int pad_token      = -1;
  // Chunked prefill: forward() processes prompts in chunks of this many
  // tokens so activations stay O(chunk·dim) instead of O(T·dim). <= 0
  // means single-shot (one chunk of T).
  int prefill_chunk_size = 2048;
  // HF text_config: tie_word_embeddings (Glimmer 30B: false). The checkpoint
  // carries a separate lm_head tensor; a loader must NOT alias the
  // embedding table (unlike the tied Gemma4/Llama-3.2-1B case).
  bool tie_word_embeddings = false;
  // HF text_config: hidden_activation (Glimmer: "silu"). The Model hardcodes
  // SwiGLU, so only that value (case-insensitive) is accepted; anything
  // else throws at parse/validate time.
  std::string hidden_activation = "silu";
  // HF text_config: attention_bias (Glimmer: false). The Model has no bias
  // path, so true is rejected.
  bool attention_bias = false;

  Config() {
    is_full_layer   = default_layer_types(n_layers);
    layer_rope_theta = default_layer_rope_theta(n_layers, (double)rope_theta);
  }

  // Default 3x-sliding + 1x-full repeat: full layers at idx 3,7,...,51
  // for 52 layers (13 full, 39 sliding).
  static std::vector<char> default_layer_types(int num_layers) {
    std::vector<char> t((size_t)num_layers, 0);
    for (int i = 0; i < num_layers; ++i)
      if (i % 4 == 3) t[(size_t)i] = 1;
    return t;
  }

  // Default per-layer RoPE theta: sliding layers use `theta`, full layers
  // are 0 (meaning "use the global rope_theta"), matching the HF
  // layer_rope_theta array.
  static std::vector<double> default_layer_rope_theta(int num_layers,
                                                      double theta) {
    std::vector<double> t((size_t)num_layers, theta);
    for (int i = 0; i < num_layers; ++i)
      if (i % 4 == 3) t[(size_t)i] = 0.0;
    return t;
  }

  static Config glimmer_30b() { return Config{}; }

  // Tiny config for unit tests (no 60GB download needed). Keeps the Glimmer
  // structure (explicit head_dim != dim/heads, mixed layer types, per-layer
  // rope theta with a 0-marked full layer) at small dims.
  static Config tiny_test() {
    Config c;
    c.dim = 64; c.n_layers = 4; c.n_heads = 4; c.n_kv_heads = 2;
    c.head_dim = 8;  // explicit: dim / n_heads would be 16
    c.vocab_size = 256; c.ffn_dim = 128;
    c.sliding_window = 8;
    c.is_full_layer   = {0, 0, 0, 1};
    c.layer_rope_theta = {500000.0, 500000.0, 500000.0, 0.0};
    c.max_seq_len = 32;
    return c;
  }

  bool is_full(int layer) const { return is_full_layer[(size_t)layer] != 0; }
  // Effective RoPE theta for a layer: the per-layer entry when positive,
  // otherwise the global rope_theta (HF marks full layers with 0).
  double rope_theta_for_layer(int layer) const {
    double per_layer = layer_rope_theta[(size_t)layer];
    return (per_layer > 0.0) ? per_layer : (double)rope_theta;
  }

  // head_dim is EXPLICIT for Glimmer (128 != 6656/32), so there is nothing
  // to recompute from dim / n_heads. Kept for API symmetry with the Llama
  // Config; intentionally a no-op.
  void recompute_derived() {}

  // Total K+V bytes across all layers for the full context window, given
  // the unquantized element size (e.g. sizeof(Scalar)). Dense cache:
  // every layer rents its own [max_seq_len, n_kv_heads, head_dim] K and V.
  size_t kv_cache_bytes(size_t elem_bytes) const {
    const size_t toks   = (size_t)max_seq_len;
    const size_t per_tok = (size_t)n_kv_heads * (size_t)head_dim * elem_bytes;
    return (size_t)n_layers * toks * per_tok * 2;  // K + V
  }

  // Throw std::invalid_argument/std::runtime_error on incompatible dims or
  // out-of-range context. Called by from_json_* and by Model (which rents
  // the [max_seq_len, n_kv_heads, head_dim] KV cache from these values).
  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Config::validate: " + m);
    };
    if (dim <= 0) bad("hidden_size (dim) must be > 0");
    if (n_layers <= 0) bad("num_hidden_layers (n_layers) must be > 0");
    if (n_heads <= 0) bad("num_attention_heads (n_heads) must be > 0");
    if (n_kv_heads <= 0) bad("num_key_value_heads (n_kv_heads) must be > 0");
    if (vocab_size <= 0) bad("vocab_size must be > 0");
    if (ffn_dim <= 0) bad("intermediate_size (ffn_dim) must be > 0");
    if (n_kv_heads > n_heads)
      bad("num_key_value_heads (" + std::to_string(n_kv_heads) +
          ") must not exceed num_attention_heads (" + std::to_string(n_heads) +
          ")");
    if (n_heads % n_kv_heads != 0)
      bad("num_attention_heads (" + std::to_string(n_heads) +
          ") must be a multiple of num_key_value_heads (" +
          std::to_string(n_kv_heads) + ") for grouped-query attention");
    // NOTE: no dim % n_heads / head_dim == dim / n_heads requirement here.
    // head_dim is explicit (128 vs dim/n_heads = 208); only positivity and
    // evenness (RoPE pairs) are required.
    if (head_dim <= 0 || head_dim % 2 != 0)
      bad("head_dim must be positive and even");
    if (!(rms_norm_eps > 0) || !std::isfinite(rms_norm_eps))
      bad("rms_norm_eps must be positive and finite");
    if (!(post_norm_eps > 0) || !std::isfinite(post_norm_eps))
      bad("post_norm_eps must be positive and finite");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (sliding_window <= 0) bad("sliding_window must be > 0");
    if (is_full_layer.size() != (size_t)n_layers)
      bad("layer_types length (" + std::to_string(is_full_layer.size()) +
          ") must equal num_hidden_layers (" + std::to_string(n_layers) + ")");
    if (layer_rope_theta.size() != (size_t)n_layers)
      bad("layer_rope_theta length (" +
          std::to_string(layer_rope_theta.size()) +
          ") must equal num_hidden_layers (" + std::to_string(n_layers) + ")");
    for (int l = 0; l < n_layers; ++l) {
      double theta = layer_rope_theta[(size_t)l];
      if (!(theta >= 0.0) || !std::isfinite(theta))
        bad("layer_rope_theta[" + std::to_string(l) + "] must be >= 0 " +
            "(0 selects the global rope_theta) and finite");
    }
    if (!(final_logit_softcap > 0) || !std::isfinite(final_logit_softcap))
      bad("final_logit_softcapping must be positive and finite");
    if (!(qk_scale_factor > 0) || !std::isfinite(qk_scale_factor))
      bad("qk_scale_factor must be positive and finite");
    if (!(output_multiplier > 0) || !std::isfinite(output_multiplier))
      bad("output_multiplier must be positive and finite");
    if (!config_detail::iequals(hidden_activation, "silu"))
      throw std::runtime_error(
          "Config::validate: hidden_activation '" + hidden_activation +
          "' is not supported: only 'silu' is supported (Model hardcodes "
          "SwiGLU)");
    if (attention_bias)
      throw std::runtime_error(
          "Config::validate: 'attention_bias' must be false: bias "
          "projections are unsupported (Model has no bias path)");
    if (max_seq_len < kMinSupportedSeqLen ||
        max_seq_len > kMaxSupportedSeqLen)
      bad("max_seq_len (" + std::to_string(max_seq_len) +
          ") out of supported range [" +
          std::to_string(kMinSupportedSeqLen) + ", " +
          std::to_string(kMaxSupportedSeqLen) +
          "]; the KV cache rents [max_seq_len, n_kv_heads, head_dim] per "
          "layer, so cap it to what fits in memory");
    if (bos_token < 0) bad("bos_token_id must be >= 0");
    if (eos_token < 0) bad("eos_token_id must be >= 0");
    if (pad_token < -1) bad("pad_token_id must be >= 0 or absent (-1)");
  }

  // Parse a text_config object scope (the `text` scope already narrowed to
  // the text_config object when the file is a full config.json).
  static Config from_json_scope(const std::string& text,
                                const Config& base = glimmer_30b()) {
    using namespace config_detail;
    Config c = base;
    long long iv = 0;
    double dv    = 0;
    bool bv      = false;
    std::string v;
    if (get_int(text, "hidden_size", iv)) c.dim = checked_int(iv, "hidden_size");
    if (get_int(text, "num_hidden_layers", iv)) {
      c.n_layers      = checked_int(iv, "num_hidden_layers");
      c.is_full_layer = default_layer_types(c.n_layers);
    }
    if (get_int(text, "num_attention_heads", iv))
      c.n_heads = checked_int(iv, "num_attention_heads");
    if (get_int(text, "num_key_value_heads", iv))
      c.n_kv_heads = checked_int(iv, "num_key_value_heads");
    // head_dim is EXPLICIT: recorded as-is (positive, even). Unlike Llama,
    // it is never cross-checked against hidden_size / num_attention_heads.
    if (get_int(text, "head_dim", iv)) {
      if (iv <= 0 || iv > 1000000000LL || iv % 2 != 0)
        throw std::runtime_error(
            "config: 'head_dim' must be positive and even");
      c.head_dim = (int)iv;
    }
    if (get_int(text, "intermediate_size", iv))
      c.ffn_dim = checked_int(iv, "intermediate_size");
    if (get_int(text, "vocab_size", iv))
      c.vocab_size = checked_int(iv, "vocab_size");
    if (get_int(text, "sliding_window", iv))
      c.sliding_window = checked_int(iv, "sliding_window");
    // layer_types: ["sliding_attention", ..., "full_attention", ...].
    // Parsed when present (length must match num_hidden_layers); otherwise
    // the 3+1 default from the constructor/base stands.
    if (get_raw(text, "layer_types", v)) {
      std::string t = trim(v);
      if (is_null(t))
        throw std::runtime_error("config: 'layer_types' must not be null");
      if (t.empty() || t.front() != '[')
        throw std::runtime_error("config: 'layer_types' must be an array");
      std::vector<char> lt;
      size_t p = 1;
      while (true) {
        skip_ws(t, p);
        if (p >= t.size())
          throw std::runtime_error("config: unbalanced 'layer_types'");
        if (t[p] == ']') break;
        if (t[p] != '"')
          throw std::runtime_error(
              "config: 'layer_types' entries must be strings");
        std::string e = parse_string(t, p);
        if (iequals(e, "full_attention")) lt.push_back(1);
        else if (iequals(e, "sliding_attention") ||
                 iequals(e, "sliding_window_attention"))
          lt.push_back(0);
        else
          throw std::runtime_error("config: unknown layer_types entry '" + e +
                                   "' (expected 'sliding_attention' or "
                                   "'full_attention')");
        skip_ws(t, p);
        if (p < t.size() && t[p] == ',') {
          ++p;
          continue;
        }
        if (p < t.size() && t[p] == ']') break;
        throw std::runtime_error("config: expected ',' or ']' in layer_types");
      }
      c.is_full_layer = std::move(lt);
    }
    // layer_rope_theta: [500000.0, ..., 0, ...], parallel to layer_types.
    // A 0 entry marks "use the global rope_theta" (the HF convention for
    // full-attention layers). Length must match num_hidden_layers when
    // present; otherwise defaults are rebuilt from the final rope_theta
    // after parsing (see below).
    bool have_layer_rope_theta = false;
    if (get_raw(text, "layer_rope_theta", v)) {
      std::string t = trim(v);
      if (is_null(t))
        throw std::runtime_error(
            "config: 'layer_rope_theta' must not be null");
      if (t.empty() || t.front() != '[')
        throw std::runtime_error(
            "config: 'layer_rope_theta' must be an array");
      std::vector<double> rt;
      size_t p = 1;
      while (true) {
        skip_ws(t, p);
        if (p >= t.size())
          throw std::runtime_error("config: unbalanced 'layer_rope_theta'");
        if (t[p] == ']') break;
        size_t e = p;
        while (e < t.size() && t[e] != ',' && t[e] != ']') ++e;
        std::string num = trim(t.substr(p, e - p));
        double theta    = parse_num(num, "layer_rope_theta");
        if (!(theta >= 0.0) || !std::isfinite(theta))
          throw std::runtime_error(
              "config: 'layer_rope_theta' entries must be >= 0 (0 selects "
              "the global rope_theta) and finite");
        rt.push_back(theta);
        p = e;
        skip_ws(t, p);
        if (p < t.size() && t[p] == ',') {
          ++p;
          continue;
        }
        if (p < t.size() && t[p] == ']') break;
        throw std::runtime_error(
            "config: expected ',' or ']' in layer_rope_theta");
      }
      c.layer_rope_theta    = std::move(rt);
      have_layer_rope_theta = true;
    }
    if (get_num(text, "final_logit_softcapping", dv))
      c.final_logit_softcap = (float)dv;
    if (get_num(text, "qk_scale_factor", dv)) c.qk_scale_factor = (float)dv;
    if (get_num(text, "output_multiplier", dv)) c.output_multiplier = dv;
    if (get_num(text, "rms_norm_eps", dv)) c.rms_norm_eps = (float)dv;
    if (get_num(text, "post_norm_eps", dv)) c.post_norm_eps = (float)dv;
    // rope_parameters: {"rope_theta": 500000.0, "rope_type": "default"}.
    if (get_raw(text, "rope_parameters", v)) {
      std::string t = trim(v);
      if (!is_null(t)) {
        if (t.empty() || t.front() != '{')
          throw std::runtime_error(
              "config: 'rope_parameters' must be an object");
        double theta = 0;
        if (get_num(t, "rope_theta", theta)) c.rope_theta = (float)theta;
        std::string rope_type;
        if (get_string(t, "rope_type", rope_type) &&
            !iequals(rope_type, "default"))
          throw std::runtime_error("config: rope_type '" + rope_type +
                                   "' is not supported: only 'default' is "
                                   "supported");
      }
    }
    // Non-HF alias: a top-level rope_theta wins over rope_parameters when
    // both are present.
    if (get_num(text, "rope_theta", dv)) c.rope_theta = (float)dv;
    // hidden_activation: the Model hardcodes SwiGLU (silu).
    if (get_string(text, "hidden_activation", v)) {
      if (!iequals(v, "silu"))
        throw std::runtime_error(
            "config: 'hidden_activation' '" + v +
            "' is not supported: only 'silu' is supported (Model hardcodes "
            "SwiGLU)");
      c.hidden_activation = v;
    }
    if (get_bool(text, "attention_bias", bv)) {
      c.attention_bias = bv;
      if (bv)
        throw std::runtime_error(
            "config: 'attention_bias' must be false: bias projections are "
            "unsupported (Model has no bias path)");
    }
    // model_type: accept "muse_glimmer_text" (and "muse_glimmer").
    if (get_string(text, "model_type", v)) {
      if (!iequals(v, "muse_glimmer_text") && !iequals(v, "muse_glimmer"))
        throw std::runtime_error("config: 'model_type' '" + v +
                                 "' is not supported: only "
                                 "'muse_glimmer_text' is supported");
    }
    if (get_int(text, "max_position_embeddings", iv))
      c.max_seq_len = checked_int(iv, "max_position_embeddings");
    if (get_int(text, "max_seq_len", iv))  // non-HF alias wins if both
      c.max_seq_len = checked_int(iv, "max_seq_len");
    if (get_int(text, "bos_token_id", iv))
      c.bos_token = checked_int(iv, "bos_token_id");
    // eos_token_id is an int in config.json but a list in
    // generation_config.json; accept either (first element wins).
    if (get_raw(text, "eos_token_id", v)) {
      if (is_null(v))
        throw std::runtime_error("config: 'eos_token_id' must not be null");
      if (!v.empty() && v.front() == '[') {
        // Parse [a, b, ...]; use the first id.
        size_t p = 1;
        skip_ws(v, p);
        if (p < v.size() && v[p] == ']')
          throw std::runtime_error("config: 'eos_token_id' array is empty");
        size_t e = p;
        while (e < v.size() && v[e] != ',' && v[e] != ']') ++e;
        c.eos_token = checked_int(std::llround(parse_num(trim(v.substr(p, e - p)),
                                                          "eos_token_id")),
                                  "eos_token_id");
      } else {
        c.eos_token =
            checked_int(std::llround(parse_num(v, "eos_token_id")),
                        "eos_token_id");
      }
    }
    // pad_token_id is null in the HF checkpoint (no pad token): null keeps
    // the -1 "absent" marker.
    if (get_raw(text, "pad_token_id", v)) {
      std::string t = trim(v);
      if (!is_null(t)) c.pad_token = checked_int(std::llround(parse_num(t,
        "pad_token_id")), "pad_token_id");
    }
    if (get_bool(text, "tie_word_embeddings", bv)) c.tie_word_embeddings = bv;
    if (get_int(text, "prefill_chunk_size", iv)) {
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'prefill_chunk_size' out of range");
      c.prefill_chunk_size = (int)iv;
    }
    // With no explicit per-layer thetas, rebuild the 3+1 defaults from the
    // final global rope_theta (a config-level rope_theta override must apply
    // to sliding layers).
    if (!have_layer_rope_theta)
      c.layer_rope_theta = default_layer_rope_theta(c.n_layers,
                                                    (double)c.rope_theta);
    c.validate();
    return c;
  }

  // Parse an HF config.json object. Accepts either the full Muse Glimmer
  // config.json (text fields nested under "text_config") or a bare
  // text_config object. Keys present override `base`; missing keys keep
  // base values (HF defaults via glimmer_30b()). Unknown keys are
  // ignored. Throws on malformed JSON or incompatible dims.
  static Config from_json_string(const std::string& text,
                                 const Config& base = glimmer_30b()) {
    using namespace config_detail;
    std::string scope = text;
    std::string sub;
    if (get_raw(text, "text_config", sub)) {
      std::string t = trim(sub);
      if (is_null(t))
        throw std::runtime_error("config: 'text_config' must not be null");
      if (t.empty() || t.front() != '{')
        throw std::runtime_error("config: 'text_config' must be an object");
      scope = t;
    }
    return from_json_scope(scope, base);
  }

  static Config from_json_file(const std::string& path,
                               const Config& base = glimmer_30b()) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("config: cannot open config file '" + path +
                               "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_string(ss.str(), base);
  }
};

}  // namespace Glimmer
}  // namespace KokkosDNN

#endif
