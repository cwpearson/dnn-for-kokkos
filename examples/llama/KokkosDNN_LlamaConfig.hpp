#ifndef KOKKOSDNN_LLAMA_CONFIG_HPP
#define KOKKOSDNN_LLAMA_CONFIG_HPP

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
namespace Llama {

// Supported KV-cache context window. The attention kernel streams over keys
// with O(1) extra memory and prefill is chunked, so the architecture
// supports 128k contexts (Llama 3.1 8B-Instruct advertises max_position_embeddings =
// 131072); the bound below is on the rented KV-cache allocation, not on the
// kernel. The PoC *default* stays at 8192 so a plain 8B run does not rent a
// ~17 GiB BF16 KV cache by accident; raise via config.json or --max-seq-len.
inline constexpr int kMaxSupportedSeqLen = 131072;
inline constexpr int kMinSupportedSeqLen = 1;

namespace config_detail {

// ---- Minimal JSON lookup (dependency-free) ----
// Flat key lookup on the root object, string-aware (keys inside nested
// strings/objects are not confused). Follows the minimal-parser style used
// by the safetensors loader and Test_LlamaGolden.cpp. Only the value span
// of a requested top-level key is extracted; callers interpret it.
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

// Hyperparameters for Llama 3.1 8B-Instruct (Meta, July 2024).
// Reference: HF meta-llama/Llama-3.1-8B-Instruct config.json: dim 4096,
// 32 layers, 32 heads, 8 kv heads, 14336 ffn dim, 128256 vocab,
// rms_norm_eps 1e-5, rope_theta 500000, max_position_embeddings 131072,
// tie_word_embeddings false, BF16 checkpoint dtype.
struct Config {
  int dim            = 4096;    // hidden size
  int n_layers       = 32;
  int n_heads        = 32;      // query heads
  int n_kv_heads     = 8;       // GQA kv heads
  int vocab_size     = 128256;
  int ffn_dim        = 14336;   // SwiGLU intermediate
  int head_dim       = 128;     // dim / n_heads
  float rms_norm_eps = 1e-5f;
  float rope_theta   = 500000.0f;
  float attn_scale   = 0.0883883476f;  // 1/sqrt(128)
  int max_seq_len    = 8192;  // PoC runtime cap; arch supports 131072
  int bos_token      = 128000;
  int eos_token      = 128001;
  // Chunked prefill: forward() processes prompts in chunks of this many
  // tokens so activations stay O(chunk·dim) instead of O(T·dim). <= 0
  // means single-shot (one chunk of T). Default 2048 bounds a full 8B
  // prefill to ~360 MiB of BF16 activations regardless of context length.
  int prefill_chunk_size = 2048;
  // Instruct stop tokens (HF generation_config eos_token_id
  // [128001, 128008, 128009]): <|eom_id|> and <|eot_id|>.
  int eom_token      = 128008;
  int eot_token      = 128009;
  // KV-cache layout/precision. kv_page_size == 0 selects the legacy dense
  // contiguous cache ([max_seq_len, n_kv, head_dim] per layer, rented
  // upfront); > 0 selects paged KV with that block size (pools are
  // [n_pages, page, n_kv, head_dim], addressed through an explicit block
  // table, grown on demand so short prompts on a long-context config only
  // rent the pages they touch). kv_int8 selects symmetric int8 KV with
  // per-token per-head FP32 scales (~2x fewer KV bytes plus a small scale
  // table, small numerical drift); false keeps the model Scalar. The two
  // options combine (paged int8). JSON keys: "kv_page_size" (int),
  // "kv_int8" / "kv_quantized" (bool), "kv_quant" ("int8"/"none").
  int kv_page_size = 0;
  bool kv_int8     = false;
  // HF config.json: tie_word_embeddings (Llama 3.1 8B-Instruct: false). When true
  // and the checkpoint has no lm_head tensor, the loader shares the
  // embedding table as lm_head instead of throwing.
  bool tie_word_embeddings = false;
  // HF config.json: hidden_act (Llama 3.x: "silu"). The Model hardcodes
  // SwiGLU, so only "silu" (case-insensitive) is accepted; anything else
  // throws at parse/validate time.
  std::string hidden_act = "silu";
  // HF config.json: attention_bias / mlp_bias (Llama 3.x: false). The Model
  // has no bias path, so true is rejected: bias projections are unsupported.
  bool attention_bias = false;
  bool mlp_bias       = false;
  // rope_scaling bookkeeping. Supports plain NeoX RoPE (null/absent,
  // type "") and Llama 3.1 llama3-type scaling. Other types (linear,
  // dynamic, yarn, ...) are rejected with a clear error at parse/validate
  // time. Fields record what was seen for frequency computation / errors.
  std::string rope_scaling_type  = "";  // empty == none (null/absent)
  double rope_scaling_factor     = 1.0;
  double rope_scaling_low_freq_factor  = 1.0;
  double rope_scaling_high_freq_factor = 4.0;
  int rope_scaling_original_max  = 0;

  static Config llama_3_1_8b() { return Config{}; }

  // Hyperparameters for Llama 3.2 1B (Meta, Sept 2024).
  // Reference: HF meta-llama/Llama-3.2-1B config.json: dim 2048, 16 layers,
  // 32 heads, 8 kv heads, 8192 ffn dim, 128256 vocab, rms_norm_eps 1e-5,
  // rope_theta 500000, max_position_embeddings 131072,
  // tie_word_embeddings true, head_dim 64, hidden_act silu,
  // attention_bias/mlp_bias false, rope_scaling llama3-type factor 32
  // (low_freq_factor 1.0, high_freq_factor 4.0,
  // original_max_position_embeddings 8192).
  // max_seq_len stays at the 8192 PoC cap (same policy as the 8B default);
  // the arch max is 131072 via config.json -- raise via config.json or
  // --max-seq-len.
  static Config llama_3_2_1b() {
    Config c;
    c.dim          = 2048;
    c.n_layers     = 16;
    c.n_heads      = 32;
    c.n_kv_heads   = 8;
    c.vocab_size   = 128256;
    c.ffn_dim      = 8192;
    c.head_dim     = 64;      // dim / n_heads
    c.rms_norm_eps = 1e-5f;
    c.rope_theta   = 500000.0f;
    c.attn_scale   = 0.125f;  // 1/sqrt(64)
    c.max_seq_len  = 8192;    // PoC runtime cap; arch supports 131072
    c.bos_token    = 128000;
    c.eos_token    = 128001;
    c.eom_token    = 128008;
    c.eot_token    = 128009;
    c.tie_word_embeddings = true;
    c.hidden_act          = "silu";
    c.attention_bias      = false;
    c.mlp_bias            = false;
    c.rope_scaling_type         = "llama3";
    c.rope_scaling_factor       = 32.0;
    c.rope_scaling_low_freq_factor  = 1.0;
    c.rope_scaling_high_freq_factor = 4.0;
    c.rope_scaling_original_max     = 8192;
    return c;
  }

  // Tiny config for unit tests (no 16GB download needed).
  static Config tiny_test() {
    Config c;
    c.dim = 64; c.n_layers = 2; c.n_heads = 8; c.n_kv_heads = 4;
    c.vocab_size = 256; c.ffn_dim = 128; c.head_dim = 8;
    c.max_seq_len = 32;
    return c;
  }

  // Recompute head_dim/attn_scale from dim/n_heads. Only applies when the
  // division is exact; otherwise validate() reports the incompatibility.
  void recompute_derived() {
    if (n_heads > 0 && dim > 0 && dim % n_heads == 0) {
      head_dim   = dim / n_heads;
      attn_scale = 1.0f / std::sqrt((float)head_dim);
    }
  }

  bool use_paged_kv() const { return kv_page_size > 0; }
  bool use_quant_kv() const { return kv_int8; }
  // Logical pages covering max_seq_len (paged) or 0 (dense).
  size_t kv_num_pages() const {
    if (!use_paged_kv()) return 0;
    return ((size_t)max_seq_len + (size_t)kv_page_size - 1) /
           (size_t)kv_page_size;
  }
  // Resident token slots (padded up to full pages when paged).
  size_t kv_padded_tokens() const {
    if (!use_paged_kv()) return (size_t)max_seq_len;
    return kv_num_pages() * (size_t)kv_page_size;
  }
  // Total K+V bytes across all layers for the full context window, given
  // the unquantized element size (e.g. sizeof(Scalar)). Quantized caches
  // store int8 values plus one FP32 scale per token per KV head per K/V.
  // Block tables (~4 B per page per layer) are negligible and excluded.
  size_t kv_cache_bytes(size_t elem_bytes) const {
    const size_t toks = kv_padded_tokens();
    const size_t per_tok =
        (size_t)n_kv_heads * (size_t)head_dim * (kv_int8 ? 1 : elem_bytes);
    size_t bytes = (size_t)n_layers * toks * per_tok * 2;  // K + V
    if (kv_int8)
      bytes += (size_t)n_layers * toks * (size_t)n_kv_heads * sizeof(float) * 2;
    return bytes;
  }

  // Throw std::invalid_argument/std::runtime_error on incompatible dims or
  // out-of-range context. Called by from_json_* and by Model (which rents
  // the [max_seq_len, n_kv, head_dim] KV cache from these values).
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
    if (dim % n_heads != 0)
      bad("hidden_size (" + std::to_string(dim) +
          ") must be divisible by num_attention_heads (" +
          std::to_string(n_heads) + ")");
    if (head_dim != dim / n_heads)
      bad("head_dim (" + std::to_string(head_dim) +
          ") must equal hidden_size / num_attention_heads (" +
          std::to_string(dim / n_heads) + ")");
    if (n_kv_heads > n_heads)
      bad("num_key_value_heads (" + std::to_string(n_kv_heads) +
          ") must not exceed num_attention_heads (" + std::to_string(n_heads) +
          ")");
    if (n_heads % n_kv_heads != 0)
      bad("num_attention_heads (" + std::to_string(n_heads) +
          ") must be a multiple of num_key_value_heads (" +
          std::to_string(n_kv_heads) + ") for grouped-query attention");
    if (!(rms_norm_eps > 0) || !std::isfinite(rms_norm_eps))
      bad("rms_norm_eps must be positive and finite");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (!config_detail::iequals(hidden_act, "silu"))
      throw std::runtime_error(
          "Config::validate: hidden_act '" + hidden_act +
          "' is not supported: only 'silu' is supported (Model hardcodes "
          "SwiGLU)");
    if (attention_bias)
      throw std::runtime_error(
          "Config::validate: 'attention_bias' must be false: bias "
          "projections are unsupported (Model has no bias path)");
    if (mlp_bias)
      throw std::runtime_error(
          "Config::validate: 'mlp_bias' must be false: bias projections are "
          "unsupported (Model has no bias path)");
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
    if (kv_page_size < 0) bad("kv_page_size must be >= 0 (0 selects dense KV)");
    if (kv_page_size > 0) {
      if (max_seq_len > 0 && kv_page_size > max_seq_len)
        bad("kv_page_size (" + std::to_string(kv_page_size) +
            ") must not exceed max_seq_len (" + std::to_string(max_seq_len) +
            ")");
      if (kv_page_size > 1024)
        bad("kv_page_size (" + std::to_string(kv_page_size) +
            ") looks excessive (expected 1..1024)");
    }
    if (rope_scaling_type.empty()) {
      // none: plain NeoX RoPE, no further checks.
    } else if (rope_scaling_type == "llama3") {
      if (!std::isfinite(rope_scaling_factor) ||
          !(rope_scaling_factor >= 1.0))
        bad("rope_scaling factor must be >= 1 for type 'llama3'");
      if (!std::isfinite(rope_scaling_low_freq_factor) ||
          !std::isfinite(rope_scaling_high_freq_factor) ||
          !(rope_scaling_high_freq_factor > rope_scaling_low_freq_factor))
        bad("rope_scaling high_freq_factor must exceed low_freq_factor for "
            "type 'llama3'");
      if (rope_scaling_original_max <= 0)
        bad("rope_scaling original_max_position_embeddings must be > 0 for "
            "type 'llama3'");
    } else {
      throw std::runtime_error(
          "Config::validate: rope_scaling type '" + rope_scaling_type +
          "' (factor " + std::to_string(rope_scaling_factor) +
          ") is not supported: supported types are null/absent (none) and "
          "'llama3' (linear/dynamic/yarn/etc would silently shift positions)");
    }
  }

  // Parse an HF config.json object. Keys present override `base`; missing
  // keys keep base values (HF defaults via llama_3_1_8b()). Unknown keys
  // are ignored. Throws on malformed JSON, incompatible dims, or
  // unsupported rope_scaling types (only null/absent and "llama3" allowed).
  static Config from_json_string(const std::string& text,
                                 const Config& base = llama_3_1_8b()) {
    using namespace config_detail;
    Config c = base;
    long long iv = 0;
    double dv    = 0;
    bool bv      = false;
    std::string v;
    if (get_int(text, "hidden_size", iv)) c.dim = checked_int(iv, "hidden_size");
    if (get_int(text, "num_hidden_layers", iv))
      c.n_layers = checked_int(iv, "num_hidden_layers");
    if (get_int(text, "num_attention_heads", iv))
      c.n_heads = checked_int(iv, "num_attention_heads");
    if (get_int(text, "num_key_value_heads", iv))
      c.n_kv_heads = checked_int(iv, "num_key_value_heads");
    if (get_int(text, "intermediate_size", iv))
      c.ffn_dim = checked_int(iv, "intermediate_size");
    if (get_int(text, "vocab_size", iv))
      c.vocab_size = checked_int(iv, "vocab_size");
    // head_dim (explicit opt-in): recorded here, checked against
    // hidden_size / num_attention_heads at the end (validate instead of
    // overwrite). When absent, head_dim/attn_scale are recomputed as before.
    bool have_head_dim = false;
    if (get_int(text, "head_dim", iv)) {
      have_head_dim = true;
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'head_dim' out of range");
      c.head_dim = (int)iv;
    }
    // hidden_act: the Model hardcodes SwiGLU, so only "silu"
    // (case-insensitive) is accepted.
    if (get_raw(text, "hidden_act", v)) {
      std::string t = trim(v);
      if (is_null(t))
        throw std::runtime_error("config: 'hidden_act' must not be null");
      std::string act;
      if (t.size() >= 2 && t.front() == '"' && t.back() == '"') {
        size_t p = 0;
        act      = parse_string(t, p);
      } else {
        act = t;
      }
      if (!iequals(act, "silu"))
        throw std::runtime_error(
            "config: 'hidden_act' '" + act +
            "' is not supported: only 'silu' is supported (Model hardcodes "
            "SwiGLU)");
      c.hidden_act = act;
    }
    // attention_bias / mlp_bias: the Model has no bias path, so true is
    // rejected (message names the flag); false (or absent) is fine.
    if (get_bool(text, "attention_bias", bv)) {
      c.attention_bias = bv;
      if (bv)
        throw std::runtime_error(
            "config: 'attention_bias' must be false: bias projections are "
            "unsupported (Model has no bias path)");
    }
    if (get_bool(text, "mlp_bias", bv)) {
      c.mlp_bias = bv;
      if (bv)
        throw std::runtime_error(
            "config: 'mlp_bias' must be false: bias projections are "
            "unsupported (Model has no bias path)");
    }
    // model_type: accept "llama" (case-insensitive); reject anything else.
    if (get_raw(text, "model_type", v)) {
      std::string t = trim(v);
      std::string mt;
      if (t.size() >= 2 && t.front() == '"' && t.back() == '"') {
        size_t p = 0;
        mt       = parse_string(t, p);
      } else {
        mt = t;
      }
      if (!iequals(mt, "llama"))
        throw std::runtime_error(
            "config: 'model_type' '" + mt +
            "' is not supported: only 'llama' is supported");
    }
    if (get_num(text, "rms_norm_eps", dv)) c.rms_norm_eps = (float)dv;
    if (get_num(text, "rope_theta", dv)) c.rope_theta = (float)dv;
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
    if (get_bool(text, "tie_word_embeddings", bv)) c.tie_word_embeddings = bv;
    if (get_int(text, "kv_page_size", iv)) {
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'kv_page_size' out of range");
      c.kv_page_size = (int)iv;
    }
    if (get_bool(text, "kv_int8", bv)) c.kv_int8 = bv;
    if (get_bool(text, "kv_quantized", bv)) c.kv_int8 = c.kv_int8 || bv;
    if (get_raw(text, "kv_quant", v)) {
      std::string t = trim(v);
      if (t.size() >= 2 && t.front() == '"' && t.back() == '"') {
        size_t p = 0;
        t = parse_string(t, p);
      }
      if (t == "int8" || t == "INT8" || t == "kv_int8" || t == "true" ||
          t == "1")
        c.kv_int8 = true;
      else if (t == "none" || t == "fp" || t == "false" || t == "0" ||
               t == "null")
        c.kv_int8 = false;
      else
        throw std::runtime_error(
            "config: 'kv_quant' must be \"int8\" or \"none\"");
    }
    if (get_raw(text, "rope_scaling", v)) {
      if (!is_null(v)) {
        if (v.empty() || v.front() != '{')
          throw std::runtime_error(
              "config: 'rope_scaling' must be null or an object");
        std::string rtype, rraw;
        if (get_raw(v, "rope_type", rraw) || get_raw(v, "type", rraw)) {
          // Quoted JSON string expected.
          std::string t = trim(rraw);
          if (t.size() >= 2 && t.front() == '"' && t.back() == '"') {
            size_t p = 0;
            rtype = parse_string(t, p);
          } else {
            rtype = t;
          }
        } else {
          rtype = "unknown";
        }
        double factor = c.rope_scaling_factor;
        double low = c.rope_scaling_low_freq_factor;
        double high = c.rope_scaling_high_freq_factor;
        long long orig = c.rope_scaling_original_max;
        bool have_orig = false;
        double tmpd = 0;
        if (get_num(v, "factor", tmpd)) {
          factor = tmpd;
        } else if (get_num(v, "rope_factor", tmpd)) {
          factor = tmpd;  // legacy alias
        }
        if (get_num(v, "low_freq_factor", tmpd)) low = tmpd;
        if (get_num(v, "high_freq_factor", tmpd)) high = tmpd;
        if (get_int(v, "original_max_position_embeddings", orig)) {
          have_orig = true;
        }
        c.rope_scaling_type   = rtype;
        c.rope_scaling_factor = factor;
        c.rope_scaling_low_freq_factor  = low;
        c.rope_scaling_high_freq_factor = high;
        if (have_orig)
          c.rope_scaling_original_max = checked_int(
              orig, "original_max_position_embeddings");
        if (rtype != "llama3")
          throw std::runtime_error(
              "config: rope_scaling type '" + rtype + "' (factor " +
              std::to_string(factor) +
              ") is not supported: supported types are null/absent (none) "
              "and 'llama3'; use a checkpoint with rope_scaling=null or "
              "llama3-type scaling (e.g. Llama 3.1 8B-Instruct)");
      } else {
        c.rope_scaling_type   = "";
        c.rope_scaling_factor = 1.0;
        c.rope_scaling_low_freq_factor  = 1.0;
        c.rope_scaling_high_freq_factor = 4.0;
        c.rope_scaling_original_max     = 0;
      }
    }
    if (have_head_dim) {
      // Explicit head_dim: validate instead of overwrite.
      if (c.n_heads <= 0 || c.dim <= 0 || c.dim % c.n_heads != 0 ||
          c.head_dim != c.dim / c.n_heads)
        throw std::runtime_error(
            "config: 'head_dim' (" + std::to_string(c.head_dim) +
            ") must equal hidden_size / num_attention_heads (" +
            std::to_string(c.dim) + " / " + std::to_string(c.n_heads) + ")");
      c.attn_scale = 1.0f / std::sqrt((float)c.head_dim);
    } else {
      c.recompute_derived();
    }
    c.validate();
    return c;
  }

  static Config from_json_file(const std::string& path,
                               const Config& base = llama_3_1_8b()) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("config: cannot open config file '" + path +
                               "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_string(ss.str(), base);
  }
};

}  // namespace Llama
}  // namespace KokkosDNN

#endif
