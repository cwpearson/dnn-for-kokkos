#ifndef KOKKOSDNN_GEMMA4_CONFIG_HPP
#define KOKKOSDNN_GEMMA4_CONFIG_HPP

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
namespace Gemma4 {

// Supported KV-cache context window. The attention kernel streams over keys
// with O(1) extra memory and prefill is chunked, so the architecture window
// (Gemma4 text advertises max_position_embeddings = 131072) is supported;
// the bound below caps the rented KV-cache allocation. The PoC *default*
// stays at 8192 so a plain run does not rent the full-window cache by
// accident; raise via config.json or --max-seq-len.
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

// Hyperparameters for the Gemma4 text path (E2B-it, 2026).
// Reference: HF google/gemma-4-E2B-it config.json `text_config`: dim 1536,
// 35 layers, 8 Q / 1 KV heads, sliding head_dim 256 / full (global) 512,
// 4x sliding + 1x full repeating (full layers at idx 4,9,14,19,24,29,34),
// sliding window 512, sliding RoPE default theta 10k, full RoPE proportional
// theta 1e6 with partial_rotary_factor 0.25, gelu_pytorch_tanh,
// use_double_wide_mlp (layers >= 15 use 12288-wide MLP), 20 trailing
// KV-shared layers (first shared idx 15), PLE width 256,
// final_logit_softcapping 30, tie_word_embeddings, vocab 262144,
// rms_norm_eps 1e-6, BF16 checkpoint dtype.
struct Config {
  int dim            = 1536;    // hidden size
  int n_layers       = 35;
  int n_heads        = 8;       // query heads
  int n_kv_heads     = 1;       // GQA kv heads (all layers)
  int vocab_size     = 262144;
  int ffn_dim        = 6144;    // MLP intermediate (non-shared layers)
  int head_dim_sliding = 256;   // sliding-layer head dim
  int head_dim_full    = 512;   // full-layer (global) head dim
  float rms_norm_eps = 1e-6f;
  float rope_theta_sliding = 10000.0f;
  float rope_theta_full    = 1000000.0f;
  float partial_rotary_factor = 0.25f;  // full-attention p-RoPE fraction
  float attn_scale   = 1.0f;    // QK-normed attention: scale is 1.0, NOT 1/sqrt(d)
  int sliding_window = 512;     // sliding-layer causal window
  // Per-layer attention type: false = sliding, true = full. Default is the
  // 4x-sliding + 1x-full repeat with the last layer forced full.
  std::vector<char> is_full_layer;
  int num_kv_shared_layers = 20;  // trailing KV-shared layers
  bool use_double_wide_mlp = true;
  int ple_dim        = 256;     // hidden_size_per_layer_input
  int ple_vocab      = 262144;  // vocab_size_per_layer_input
  float final_logit_softcap = 30.0f;
  int max_seq_len    = 8192;  // PoC runtime cap; arch supports 131072
  int bos_token      = 2;     // <bos>
  int eos_token      = 1;     // <eos>
  int pad_token      = 0;     // <pad>
  // HF generation_config.json eos_token_id [1, 106, 50] (<eos> + two
  // turn/channel markers); used as the default stop set by the driver.
  std::vector<int> default_stop_ids = {1, 106, 50};
  // Chunked prefill: forward() processes prompts in chunks of this many
  // tokens so activations stay O(chunk·dim) instead of O(T·dim). <= 0
  // means single-shot (one chunk of T). Default 2048 bounds a full prefill
  // to ~1 GiB of BF16 activations regardless of context length.
  int prefill_chunk_size = 2048;
  // HF text_config: tie_word_embeddings (E2B-it: true). When true and the
  // checkpoint has no lm_head tensor, the loader shares the embedding
  // table as lm_head instead of throwing.
  bool tie_word_embeddings = true;
  // HF text_config: hidden_activation (Gemma4: "gelu_pytorch_tanh"). The
  // Model hardcodes gated-GeLU(tanh), so only that value (case-insensitive)
  // is accepted; anything else throws at parse/validate time.
  std::string hidden_activation = "gelu_pytorch_tanh";
  // HF text_config: attention_bias (E2B-it: false). The Model has no bias
  // path, so true is rejected.
  bool attention_bias = false;

  Config() : is_full_layer(default_layer_types(35)) {}

  // Default 4x-sliding + 1x-full repeat with the last layer forced full:
  // full layers at idx 4,9,14,19,24,29,34 for 35 layers.
  static std::vector<char> default_layer_types(int n_layers) {
    std::vector<char> t((size_t)n_layers, 0);
    for (int i = 0; i < n_layers; ++i)
      if (i % 5 == 4 || i == n_layers - 1) t[(size_t)i] = 1;
    return t;
  }

  static Config gemma4_e2b_it() { return Config{}; }

  // Tiny config for unit tests (no 10GB download needed). Keeps the Gemma4
  // structure (mixed layer types, KV sharing, PLE, double-wide) at small
  // dims: 4 layers, full layers at idx 1 and 3, first shared idx 2.
  static Config tiny_test() {
    Config c;
    c.dim = 32; c.n_layers = 4; c.n_heads = 4; c.n_kv_heads = 1;
    c.vocab_size = 256; c.ffn_dim = 64;
    c.head_dim_sliding = 8; c.head_dim_full = 16;
    c.sliding_window = 8;
    c.is_full_layer = {0, 1, 0, 1};
    c.num_kv_shared_layers = 2;  // layers 2..3 reuse L0 (sliding) / L1 (full)
    c.ple_dim = 8; c.ple_vocab = 256;
    c.max_seq_len = 32;
    return c;
  }

  bool is_full(int layer) const { return is_full_layer[(size_t)layer] != 0; }
  int head_dim(int layer) const {
    return is_full(layer) ? head_dim_full : head_dim_sliding;
  }
  // First KV-shared layer index (layers >= this reuse an earlier cache).
  int kv_shared_start() const { return n_layers - num_kv_shared_layers; }
  bool is_kv_shared(int layer) const { return layer >= kv_shared_start(); }
  // MLP width: double-wide (2x) on KV-shared layers when enabled.
  int mlp_dim(int layer) const {
    return (use_double_wide_mlp && is_kv_shared(layer)) ? 2 * ffn_dim
                                                        : ffn_dim;
  }
  float embed_scale() const { return std::sqrt((float)dim); }
  float ple_token_scale() const { return std::sqrt((float)ple_dim); }
  float ple_ctx_scale() const { return 1.0f / std::sqrt((float)dim); }
  static float ple_mix_scale() {
    return 0.7071067811865476f;  // 1/sqrt(2)
  }

  // Number of dense KV caches and their shapes: every non-shared layer
  // rents its own [max_seq, n_kv, head_dim] cache; shared layers reuse the
  // last non-shared cache of the same type.
  int num_sliding_caches() const {
    int n = 0;
    for (int l = 0; l < kv_shared_start(); ++l)
      if (!is_full(l)) ++n;
    return n;
  }
  int num_full_caches() const {
    int n = 0;
    for (int l = 0; l < kv_shared_start(); ++l)
      if (is_full(l)) ++n;
    return n;
  }
  // Total K+V bytes across all caches for the full context window, given
  // the unquantized element size (e.g. sizeof(Scalar)).
  size_t kv_cache_bytes(size_t elem_bytes) const {
    const size_t toks = (size_t)max_seq_len;
    size_t per_tok = (size_t)num_sliding_caches() * (size_t)head_dim_sliding +
                     (size_t)num_full_caches() * (size_t)head_dim_full;
    per_tok *= (size_t)n_kv_heads * elem_bytes;
    return per_tok * toks * 2;  // K + V
  }

  // Throw std::invalid_argument/std::runtime_error on incompatible dims or
  // out-of-range context. Called by from_json_* and by Model (which rents
  // the KV caches from these values).
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
    if (head_dim_sliding <= 0 || head_dim_sliding % 2 != 0)
      bad("head_dim (sliding) must be positive and even");
    if (head_dim_full <= 0 || head_dim_full % 2 != 0)
      bad("global_head_dim (full) must be positive and even");
    if (!(rms_norm_eps > 0) || !std::isfinite(rms_norm_eps))
      bad("rms_norm_eps must be positive and finite");
    if (!(rope_theta_sliding > 0) || !std::isfinite(rope_theta_sliding))
      bad("sliding rope_theta must be positive and finite");
    if (!(rope_theta_full > 0) || !std::isfinite(rope_theta_full))
      bad("full rope_theta must be positive and finite");
    if (!(partial_rotary_factor > 0) || !(partial_rotary_factor <= 1))
      bad("partial_rotary_factor must be in (0, 1]");
    if (sliding_window <= 0) bad("sliding_window must be > 0");
    if (is_full_layer.size() != (size_t)n_layers)
      bad("layer_types length (" + std::to_string(is_full_layer.size()) +
          ") must equal num_hidden_layers (" + std::to_string(n_layers) + ")");
    if (num_kv_shared_layers < 0 || num_kv_shared_layers > n_layers)
      bad("num_kv_shared_layers (" + std::to_string(num_kv_shared_layers) +
          ") must be in [0, num_hidden_layers]");
    // Every shared layer needs a non-shared donor of the same type below
    // the sharing threshold; otherwise KV sharing is ill-defined.
    for (int l = kv_shared_start(); l < n_layers; ++l) {
      bool found = false;
      for (int d = 0; d < kv_shared_start(); ++d)
        if (is_full(d) == is_full(l)) {
          found = true;
          break;
        }
      if (!found)
        bad("KV-shared layer " + std::to_string(l) + " has no non-shared " +
            std::string(is_full(l) ? "full" : "sliding") + " donor layer");
    }
    if (ple_dim <= 0) bad("hidden_size_per_layer_input (ple_dim) must be > 0");
    if (ple_vocab <= 0) bad("vocab_size_per_layer_input must be > 0");
    if (!(final_logit_softcap > 0) || !std::isfinite(final_logit_softcap))
      bad("final_logit_softcapping must be positive and finite");
    if (!config_detail::iequals(hidden_activation, "gelu_pytorch_tanh"))
      throw std::runtime_error(
          "Config::validate: hidden_activation '" + hidden_activation +
          "' is not supported: only 'gelu_pytorch_tanh' is supported (Model "
          "hardcodes gated-GeLU with the tanh approximation)");
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
          "]; the KV caches rent [max_seq_len, n_kv_heads, head_dim] per "
          "non-shared layer, so cap it to what fits in memory");
    if (bos_token < 0) bad("bos_token_id must be >= 0");
    if (eos_token < 0) bad("eos_token_id must be >= 0");
    if (pad_token < 0) bad("pad_token_id must be >= 0");
  }

  // Parse a text_config object scope (the `text` scope already narrowed to
  // the text_config object when the file is a full config.json).
  static Config from_json_scope(const std::string& text,
                                const Config& base = gemma4_e2b_it()) {
    using namespace config_detail;
    Config c = base;
    long long iv = 0;
    double dv    = 0;
    bool bv      = false;
    std::string v;
    if (get_int(text, "hidden_size", iv)) c.dim = checked_int(iv, "hidden_size");
    if (get_int(text, "num_hidden_layers", iv)) {
      c.n_layers = checked_int(iv, "num_hidden_layers");
      c.is_full_layer = default_layer_types(c.n_layers);
    }
    if (get_int(text, "num_attention_heads", iv))
      c.n_heads = checked_int(iv, "num_attention_heads");
    if (get_int(text, "num_key_value_heads", iv))
      c.n_kv_heads = checked_int(iv, "num_key_value_heads");
    if (get_int(text, "head_dim", iv)) {
      if (iv <= 0 || iv > 1000000000LL || iv % 2 != 0)
        throw std::runtime_error(
            "config: 'head_dim' must be positive and even");
      c.head_dim_sliding = (int)iv;
    }
    if (get_int(text, "global_head_dim", iv)) {
      if (iv <= 0 || iv > 1000000000LL || iv % 2 != 0)
        throw std::runtime_error(
            "config: 'global_head_dim' must be positive and even");
      c.head_dim_full = (int)iv;
    }
    if (get_int(text, "intermediate_size", iv))
      c.ffn_dim = checked_int(iv, "intermediate_size");
    if (get_int(text, "vocab_size", iv))
      c.vocab_size = checked_int(iv, "vocab_size");
    if (get_int(text, "sliding_window", iv))
      c.sliding_window = checked_int(iv, "sliding_window");
    if (get_int(text, "num_kv_shared_layers", iv)) {
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'num_kv_shared_layers' out of range");
      c.num_kv_shared_layers = (int)iv;
    }
    if (get_bool(text, "use_double_wide_mlp", bv)) c.use_double_wide_mlp = bv;
    if (get_int(text, "hidden_size_per_layer_input", iv))
      c.ple_dim = checked_int(iv, "hidden_size_per_layer_input");
    if (get_int(text, "vocab_size_per_layer_input", iv))
      c.ple_vocab = checked_int(iv, "vocab_size_per_layer_input");
    if (get_num(text, "final_logit_softcapping", dv))
      c.final_logit_softcap = (float)dv;
    // layer_types: ["sliding_attention", ..., "full_attention", ...].
    // Parsed when present (length must match num_hidden_layers); otherwise
    // the 4+1 default from the constructor/base stands.
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
    // rope_parameters: {"sliding_attention": {rope_theta, rope_type},
    //                   "full_attention": {rope_theta, rope_type,
    //                                      partial_rotary_factor}}.
    if (get_raw(text, "rope_parameters", v)) {
      std::string t = trim(v);
      if (!is_null(t)) {
        if (t.empty() || t.front() != '{')
          throw std::runtime_error(
              "config: 'rope_parameters' must be an object");
        std::string sub;
        if (get_raw(t, "sliding_attention", sub)) {
          std::string st = trim(sub);
          if (!is_null(st)) {
            double th = 0;
            if (get_num(st, "rope_theta", th)) c.rope_theta_sliding = (float)th;
            std::string rt;
            if (get_string(st, "rope_type", rt) && !iequals(rt, "default"))
              throw std::runtime_error(
                  "config: sliding rope_type '" + rt +
                  "' is not supported: only 'default' is supported");
          }
        }
        if (get_raw(t, "full_attention", sub)) {
          std::string st = trim(sub);
          if (!is_null(st)) {
            double th = 0;
            if (get_num(st, "rope_theta", th)) c.rope_theta_full = (float)th;
            std::string rt;
            if (get_string(st, "rope_type", rt) &&
                !iequals(rt, "proportional"))
              throw std::runtime_error(
                  "config: full rope_type '" + rt +
                  "' is not supported: only 'proportional' is supported");
            if (get_num(st, "partial_rotary_factor", th))
              c.partial_rotary_factor = (float)th;
          }
        }
      }
    }
    // hidden_activation: the Model hardcodes gated-GeLU(tanh).
    if (get_string(text, "hidden_activation", v)) {
      if (!iequals(v, "gelu_pytorch_tanh"))
        throw std::runtime_error(
            "config: 'hidden_activation' '" + v +
            "' is not supported: only 'gelu_pytorch_tanh' is supported "
            "(Model hardcodes gated-GeLU with the tanh approximation)");
      c.hidden_activation = v;
    }
    if (get_bool(text, "attention_bias", bv)) {
      c.attention_bias = bv;
      if (bv)
        throw std::runtime_error(
            "config: 'attention_bias' must be false: bias projections are "
            "unsupported (Model has no bias path)");
    }
    // model_type: accept "gemma4_text" (and "gemma4", case-insensitive).
    if (get_string(text, "model_type", v)) {
      if (!iequals(v, "gemma4_text") && !iequals(v, "gemma4"))
        throw std::runtime_error("config: 'model_type' '" + v +
                                 "' is not supported: only 'gemma4_text' is "
                                 "supported");
    }
    if (get_num(text, "rms_norm_eps", dv)) c.rms_norm_eps = (float)dv;
    if (get_int(text, "max_position_embeddings", iv))
      c.max_seq_len = checked_int(iv, "max_position_embeddings");
    if (get_int(text, "max_seq_len", iv))  // non-HF alias wins if both
      c.max_seq_len = checked_int(iv, "max_seq_len");
    if (get_int(text, "bos_token_id", iv))
      c.bos_token = checked_int(iv, "bos_token_id");
    if (get_int(text, "eos_token_id", iv))
      c.eos_token = checked_int(iv, "eos_token_id");
    if (get_int(text, "pad_token_id", iv))
      c.pad_token = checked_int(iv, "pad_token_id");
    if (get_bool(text, "tie_word_embeddings", bv)) c.tie_word_embeddings = bv;
    if (get_int(text, "prefill_chunk_size", iv)) {
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'prefill_chunk_size' out of range");
      c.prefill_chunk_size = (int)iv;
    }
    c.validate();
    return c;
  }

  // Parse an HF config.json object. Accepts either the full Gemma4
  // config.json (text fields nested under "text_config") or a bare
  // text_config object. Keys present override `base`; missing keys keep
  // base values (HF defaults via gemma4_e2b_it()). Unknown keys are
  // ignored. Throws on malformed JSON or incompatible dims.
  static Config from_json_string(const std::string& text,
                                 const Config& base = gemma4_e2b_it()) {
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
                               const Config& base = gemma4_e2b_it()) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("config: cannot open config file '" + path +
                               "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_string(ss.str(), base);
  }
};

}  // namespace Gemma4
}  // namespace KokkosDNN

#endif
