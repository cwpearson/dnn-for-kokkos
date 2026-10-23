// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_LFM25_CONFIG_HPP
#define KOKKOSDNN_LFM25_CONFIG_HPP

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
namespace LFM25 {

// Supported KV-cache context window. The architecture advertises
// max_position_embeddings = 131072; the bound below is on the rented
// KV-cache allocation, not on any kernel. The PoC *default* stays at 8192
// so a plain 24B-A2B run does not rent a large KV cache by accident; raise
// via config JSON ("max_position_embeddings" / "max_seq_len") or the
// equivalent runtime flag. Only full-attention layers rent KV cache;
// short-convolution ("conv") layers hold a small fixed state of
// conv_L_cache positions instead.
inline constexpr int kMaxSupportedSeqLen = 131072;
inline constexpr int kMinSupportedSeqLen = 1;

namespace config_detail {

// ---- Minimal JSON lookup (dependency-free) ----
// Flat key lookup on the root object, string-aware (keys inside nested
// strings/objects are not confused). Follows the minimal-parser style used
// by the safetensors loader and the Llama config. Only the value span
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

// Case-insensitive string compare (for hidden_act / model_type /
// layer_types entries).
inline std::string to_lower_str(const std::string& t) {
  std::string out = t;
  for (auto& ch : out) ch = (char)std::tolower((unsigned char)ch);
  return out;
}

inline bool iequals(const std::string& a, const std::string& b) {
  return to_lower_str(a) == to_lower_str(b);
}

}  // namespace config_detail

// Hyperparameters for LFM2.5-24B-A2B (Liquid AI): hybrid short-convolution
// + full-attention decoder with mixture-of-experts feedforwards.
// Reference values (HF config.json): hidden 2048, 40 layers, 32 query heads,
// 8 kv heads (head_dim 64), 11776 dense FFN dim, 1536 MoE intermediate, 64
// experts with top-4 routing, first 2 layers dense, norm_eps 1e-5,
// rope_theta 10M, vocab 128000, tie_word_embeddings true, silu activation.
// Layer rhythm: every 4th layer starting at index 2 is full attention
// (2,6,...,38 -> 10 attention layers); the other 30 are short-conv layers
// with a fixed state of conv_L_cache = 3 positions.
struct Config {
  int dim               = 2048;   // hidden size
  int n_layers          = 40;
  int n_heads           = 32;     // query heads
  int n_kv_heads        = 8;      // GQA kv heads (attention layers only)
  int head_dim          = 64;     // dim / n_heads
  int vocab_size        = 128000;
  int ffn_dim           = 11776;  // dense FFN intermediate (first layers)
  int moe_intermediate  = 1536;   // per-expert intermediate size
  int n_experts         = 64;
  int top_k             = 4;      // num_experts_per_tok
  int n_dense_layers    = 2;      // leading dense-FFN layers (rest are MoE)
  float rms_norm_eps    = 1e-5f;  // HF key "norm_eps"
  float rope_theta      = 10000000.0f;
  float attn_scale      = 0.125f;  // 1/sqrt(64)
  bool norm_topk_prob   = true;   // normalize the top-k router probabilities
  bool use_expert_bias  = true;   // auxiliary-loss-free bias routing
  float routed_scaling_factor = 1.0f;
  int conv_L_cache      = 3;      // short-conv state length (conv layers)
  bool conv_bias        = false;  // short-conv layers carry no bias
  // Per-layer kind: nonzero == full attention, zero == short conv.
  // vector<char> (not vector<bool>: no bit-packed proxy, stable refs).
  // Default rhythm: attention at indices 2,6,...,38 (10 of 40).
  std::vector<char> is_attention_layer = {
      0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0};
  int max_seq_len = 8192;  // PoC runtime cap; arch supports 131072
  int bos         = 124894;  // <|startoftext|>
  int eos         = 124900;  // <|im_end|>
  int pad         = 124893;
  // Chunked prefill: forward() processes prompts in chunks of this many
  // tokens so activations stay O(chunk·dim) instead of O(T·dim). <= 0
  // means single-shot (one chunk of T).
  int prefill_chunk_size = 2048;
  // KV-cache layout/precision (attention layers only). kv_page_size == 0
  // selects the legacy dense contiguous cache rented upfront; > 0 selects
  // paged KV with that block size. kv_int8 selects symmetric int8 KV with
  // per-token per-head FP32 scales. JSON keys: "kv_page_size" (int),
  // "kv_int8" / "kv_quantized" (bool), "kv_quant" ("int8"/"none").
  int kv_page_size = 0;
  bool kv_int8     = false;
  // HF config.json: tie_word_embeddings (LFM2.5: true). When true and the
  // checkpoint has no lm_head tensor, the loader may share the embedding
  // table as lm_head instead of throwing.
  bool tie_word_embeddings = true;
  // HF config.json: hidden_act (LFM2.5: "silu"). Only "silu"
  // (case-insensitive) is accepted; anything else throws at parse/validate
  // time.
  std::string hidden_act = "silu";
  // HF config.json: model_type (LFM2.5: "lfm2_moe"). Only "lfm2_moe"
  // (case-insensitive) is accepted.
  std::string model_type = "lfm2_moe";

  static Config lfm2_5_24b_a2b() { return Config{}; }

  // Tiny config for unit tests (no weight download needed). 4 layers with
  // [conv, conv, attn, conv] rhythm (attention at index 2, matching the
  // full-model i%4==2 pattern).
  static Config tiny_test() {
    Config c;
    c.dim             = 64;
    c.n_layers        = 4;
    c.n_heads         = 8;
    c.n_kv_heads      = 4;
    c.head_dim        = 8;  // dim / n_heads
    c.vocab_size      = 256;
    c.ffn_dim         = 128;
    c.moe_intermediate = 32;
    c.n_experts       = 8;
    c.top_k           = 2;
    c.n_dense_layers  = 1;
    c.is_attention_layer = {0, 0, 1, 0};
    c.max_seq_len     = 32;
    c.recompute_derived();
    return c;
  }

  // Default hybrid rhythm: attention at indices i with i % 4 == 2.
  static std::vector<char> default_layer_pattern(int n_layers) {
    std::vector<char> v((size_t)(n_layers > 0 ? n_layers : 0), 0);
    for (int i = 0; i < n_layers; ++i)
      v[(size_t)i] = (i % 4 == 2) ? (char)1 : (char)0;
    return v;
  }

  // Recompute head_dim/attn_scale from dim/n_heads. Only applies when the
  // division is exact; otherwise validate() reports the incompatibility.
  void recompute_derived() {
    if (n_heads > 0 && dim > 0 && dim % n_heads == 0) {
      head_dim   = dim / n_heads;
      attn_scale = 1.0f / std::sqrt((float)head_dim);
    }
  }

  bool is_attention(int layer) const {
    return is_attention_layer.at((size_t)layer) != 0;
  }
  bool is_conv(int layer) const { return !is_attention(layer); }
  int num_attention_layers() const {
    int n = 0;
    for (char a : is_attention_layer) n += (a != 0);
    return n;
  }
  int num_conv_layers() const { return n_layers - num_attention_layers(); }

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
  // Total K+V bytes across ATTENTION layers only for the full context
  // window, given the unquantized element size (e.g. sizeof(Scalar)).
  // Short-conv layers hold a fixed conv_L_cache state instead and rent no
  // KV cache. Quantized caches store int8 values plus one FP32 scale per
  // token per KV head per K/V. Block tables (~4 B per page per layer) are
  // negligible and excluded.
  size_t kv_cache_bytes(size_t elem_bytes) const {
    const size_t toks = kv_padded_tokens();
    const size_t per_tok =
        (size_t)n_kv_heads * (size_t)head_dim * (kv_int8 ? 1 : elem_bytes);
    size_t bytes =
        (size_t)num_attention_layers() * toks * per_tok * 2;  // K + V
    if (kv_int8)
      bytes += (size_t)num_attention_layers() * toks * (size_t)n_kv_heads *
               sizeof(float) * 2;
    return bytes;
  }

  // Throw std::invalid_argument/std::runtime_error on incompatible dims or
  // out-of-range context. Called by from_json_* and by Model (which rents
  // the KV cache from these values).
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
    if (moe_intermediate <= 0)
      bad("moe_intermediate_size (moe_intermediate) must be > 0");
    if (n_experts <= 0) bad("num_experts must be > 0");
    if (top_k <= 0 || top_k > n_experts)
      bad("num_experts_per_tok (top_k) must satisfy 1 <= top_k <= num_experts "
          "(" +
          std::to_string(n_experts) + ")");
    if (n_dense_layers < 0 || n_dense_layers > n_layers)
      bad("num_dense_layers must satisfy 0 <= n_dense_layers <= n_layers");
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
      bad("norm_eps (rms_norm_eps) must be positive and finite");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (!std::isfinite(routed_scaling_factor))
      bad("routed_scaling_factor must be finite");
    if (conv_L_cache <= 0) bad("conv_L_cache must be > 0");
    if ((int)is_attention_layer.size() != n_layers)
      bad("is_attention_layer size (" +
          std::to_string(is_attention_layer.size()) +
          ") must equal num_hidden_layers (" + std::to_string(n_layers) + ")");
    if (num_attention_layers() == 0)
      bad("layer_types must contain at least one full-attention layer");
    if (!config_detail::iequals(hidden_act, "silu"))
      throw std::runtime_error(
          "Config::validate: hidden_act '" + hidden_act +
          "' is not supported: only 'silu' is supported");
    if (!config_detail::iequals(model_type, "lfm2_moe"))
      throw std::runtime_error(
          "Config::validate: model_type '" + model_type +
          "' is not supported: only 'lfm2_moe' is supported");
    if (max_seq_len < kMinSupportedSeqLen ||
        max_seq_len > kMaxSupportedSeqLen)
      bad("max_seq_len (" + std::to_string(max_seq_len) +
          ") out of supported range [" +
          std::to_string(kMinSupportedSeqLen) + ", " +
          std::to_string(kMaxSupportedSeqLen) +
          "]; the KV cache rents [max_seq_len, n_kv_heads, head_dim] per "
          "attention layer, so cap it to what fits in memory");
    if (bos < 0) bad("bos id must be >= 0");
    if (eos < 0) bad("eos id must be >= 0");
    if (pad < 0) bad("pad id must be >= 0");
    if (prefill_chunk_size < 0)
      bad("prefill_chunk_size must be >= 0 (0 selects single-shot prefill)");
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
  }

  // Parse an HF config.json object. Keys present override `base`; missing
  // keys keep base values (HF defaults via lfm2_5_24b_a2b()). Unknown keys
  // are ignored. Throws on malformed JSON, incompatible dims, unknown
  // layer_types entries, or unsupported hidden_act/model_type.
  static Config from_json_string(const std::string& text,
                                 const Config& base = lfm2_5_24b_a2b()) {
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
    if (get_int(text, "moe_intermediate_size", iv))
      c.moe_intermediate = checked_int(iv, "moe_intermediate_size");
    if (get_int(text, "num_experts", iv))
      c.n_experts = checked_int(iv, "num_experts");
    if (get_int(text, "num_experts_per_tok", iv))
      c.top_k = checked_int(iv, "num_experts_per_tok");
    if (get_int(text, "num_dense_layers", iv))
      c.n_dense_layers = checked_int(iv, "num_dense_layers");
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
    // hidden_act: only "silu" (case-insensitive) is accepted.
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
            "' is not supported: only 'silu' is supported");
      c.hidden_act = act;
    }
    // model_type: accept "lfm2_moe" (case-insensitive); reject anything else.
    if (get_raw(text, "model_type", v)) {
      std::string t = trim(v);
      std::string mt;
      if (t.size() >= 2 && t.front() == '"' && t.back() == '"') {
        size_t p = 0;
        mt       = parse_string(t, p);
      } else {
        mt = t;
      }
      if (!iequals(mt, "lfm2_moe"))
        throw std::runtime_error(
            "config: 'model_type' '" + mt +
            "' is not supported: only 'lfm2_moe' is supported");
      c.model_type = mt;
    }
    // HF key is "norm_eps" ("rms_norm_eps" accepted as an alias).
    if (get_num(text, "norm_eps", dv)) c.rms_norm_eps = (float)dv;
    if (get_num(text, "rms_norm_eps", dv)) c.rms_norm_eps = (float)dv;
    // Nested rope_parameters object wins for rope_theta; a top-level
    // "rope_theta" is accepted as a fallback/override when present.
    if (get_raw(text, "rope_parameters", v)) {
      std::string t = trim(v);
      if (is_null(t))
        throw std::runtime_error(
            "config: 'rope_parameters' must not be null");
      if (t.empty() || t.front() != '{')
        throw std::runtime_error(
            "config: 'rope_parameters' must be an object");
      double th = 0;
      if (get_num(t, "rope_theta", th)) c.rope_theta = (float)th;
    }
    if (get_num(text, "rope_theta", dv)) c.rope_theta = (float)dv;
    if (get_num(text, "routed_scaling_factor", dv))
      c.routed_scaling_factor = (float)dv;
    if (get_bool(text, "norm_topk_prob", bv)) c.norm_topk_prob = bv;
    if (get_bool(text, "use_expert_bias", bv)) c.use_expert_bias = bv;
    if (get_int(text, "conv_L_cache", iv)) {
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'conv_L_cache' out of range");
      c.conv_L_cache = (int)iv;
    }
    if (get_bool(text, "conv_bias", bv)) c.conv_bias = bv;
    // layer_types: array of per-layer kind strings, e.g.
    // ["conv", "conv", "full_attention", ...]. Entries containing
    // "attention" (case-insensitive) select full attention, entries
    // containing "conv" select short conv; anything else throws. Length
    // must equal num_hidden_layers.
    if (get_raw(text, "layer_types", v)) {
      std::string t = trim(v);
      if (is_null(t))
        throw std::runtime_error("config: 'layer_types' must not be null");
      if (t.empty() || t.front() != '[' || t.back() != ']')
        throw std::runtime_error(
            "config: 'layer_types' must be an array of strings");
      std::vector<char> layers;
      size_t p = 1;  // past '['
      skip_ws(t, p);
      if (p < t.size() && t[p] == ']') {
        ++p;  // empty array: caught by size check below
      } else {
        while (true) {
          skip_ws(t, p);
          if (p >= t.size() || t[p] != '"')
            throw std::runtime_error(
                "config: 'layer_types' entries must be strings");
          std::string e = parse_string(t, p);
          std::string le = to_lower_str(e);
          if (le.find("attention") != std::string::npos)
            layers.push_back(1);
          else if (le.find("conv") != std::string::npos)
            layers.push_back(0);
          else
            throw std::runtime_error(
                "config: 'layer_types' entry '" + e +
                "' is not supported: expected 'conv' or 'full_attention'");
          skip_ws(t, p);
          if (p >= t.size())
            throw std::runtime_error(
                "config: 'layer_types' array is unterminated");
          if (t[p] == ',') {
            ++p;
            continue;
          }
          if (t[p] == ']') break;
          throw std::runtime_error(
              "config: expected ',' or ']' in 'layer_types'");
        }
      }
      if ((int)layers.size() != c.n_layers)
        throw std::runtime_error(
            "config: 'layer_types' length (" + std::to_string(layers.size()) +
            ") must equal num_hidden_layers (" + std::to_string(c.n_layers) +
            ")");
      c.is_attention_layer = std::move(layers);
    } else if ((int)c.is_attention_layer.size() != c.n_layers) {
      // num_hidden_layers overridden without layer_types: rebuild the
      // default hybrid rhythm for the new depth.
      c.is_attention_layer = default_layer_pattern(c.n_layers);
    }
    if (get_int(text, "max_position_embeddings", iv))
      c.max_seq_len = checked_int(iv, "max_position_embeddings");
    if (get_int(text, "max_seq_len", iv))  // non-HF alias wins if both
      c.max_seq_len = checked_int(iv, "max_seq_len");
    if (get_int(text, "prefill_chunk_size", iv)) {
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'prefill_chunk_size' out of range");
      c.prefill_chunk_size = (int)iv;
    }
    if (get_int(text, "bos_token_id", iv))
      c.bos = checked_int(iv, "bos_token_id");
    // eos_token_id is an int in config.json but may be a list elsewhere;
    // accept either (first element wins).
    if (get_raw(text, "eos_token_id", v)) {
      if (is_null(v))
        throw std::runtime_error("config: 'eos_token_id' must not be null");
      if (!v.empty() && v.front() == '[') {
        size_t p = 1;
        skip_ws(v, p);
        if (p < v.size() && v[p] == ']')
          throw std::runtime_error("config: 'eos_token_id' array is empty");
        size_t e = p;
        while (e < v.size() && v[e] != ',' && v[e] != ']') ++e;
        c.eos = checked_int(std::llround(parse_num(trim(v.substr(p, e - p)),
                                                         "eos_token_id")),
                                  "eos_token_id");
      } else {
        c.eos =
            checked_int(std::llround(parse_num(v, "eos_token_id")),
                        "eos_token_id");
      }
    }
    if (get_int(text, "pad_token_id", iv))
      c.pad = checked_int(iv, "pad_token_id");
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
                               const Config& base = lfm2_5_24b_a2b()) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("config: cannot open config file '" + path +
                               "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_string(ss.str(), base);
  }
};

}  // namespace LFM25
}  // namespace KokkosDNN

#endif
