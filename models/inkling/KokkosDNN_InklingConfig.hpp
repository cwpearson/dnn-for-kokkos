// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_CONFIG_HPP
#define KOKKOSDNN_INKLING_CONFIG_HPP

// Hyperparameters for the Thinking Machines Inkling 2 Mini text path.
// Reference: HF thinkingmachines/Inkling-2-Mini config.json `text_config`:
// dim 2304, 24 layers, 24 Q / 8 KV heads with EXPLICIT head_dim 128 (q/k/v
// projections use n_heads*head_dim = 3072 / n_kv_heads*head_dim = 1024),
// vocab 201024 (unpadded 200058), 128 routed MoE experts with top-8 routing
// (intermediate 1152, gate/up softcaps 4/25), shortconv kernel 4, sliding
// window 1024 on the 20 local layers (local_layer_ids lists them; the 4
// global layers 5/11/17/23 use dynamicconv projections instead of attn_norm),
// a 6-layer attention residual block with normalization, a 12-layer loop
// core [6, 18) with 3 passes and rank-64 gates, RMS epsilon 1e-6, QK/rel
// L2 clips 22.627417/8.0, log-scale floor 8192 / alpha 0.1, final logit
// softcap 30, route scale 8, gate bias on, BF16 checkpoint dtype.
// The runtime window defaults to 32768 (well under the 1048576
// model_max_length); raise via config file or --max-seq-len.

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace KokkosDNN {
namespace Inkling {

// Supported KV-cache context window. The bound caps the rented KV-cache
// allocation; the default stays at 32768 so a plain run does not rent the
// full million-token window by accident.
inline constexpr int kMaxSupportedSeqLen = 1048576;
inline constexpr int kMinSupportedSeqLen = 1;

namespace Impl {

// ---- Minimal JSON lookup (dependency-free) ----
// Flat key lookup on an object, string-aware (keys inside nested
// strings/objects are not confused). Only the value span of a requested key
// is extracted; callers interpret it.
inline void skip_ws(const std::string& s, size_t& p) {
  while (p < s.size() &&
         (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
    ++p;
}

// Parse a JSON string literal starting at s[p] == '"'. Returns the decoded
// value and advances p past the closing quote. Handles the common escapes
// and \uXXXX (BMP; astral pairs via surrogate combination).
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
  while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;
}

// Find the value span [vs, ve) of `key` in object `s`. Returns false when
// absent. Throws on malformed JSON.
inline bool find_key(const std::string& s, const std::string& key,
                     size_t& vs, size_t& ve) {
  size_t p = 0;
  skip_ws(s, p);
  if (p >= s.size() || s[p] != '{')
    throw std::runtime_error("config: root must be a JSON object");
  ++p;
  skip_ws(s, p);
  if (p < s.size() && s[p] == '}') return false;
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

inline bool get_raw(const std::string& s, const std::string& key,
                    std::string& out) {
  size_t vs = 0, ve = 0;
  if (!find_key(s, key, vs, ve)) return false;
  out = trim(s.substr(vs, ve - vs));
  return true;
}

inline bool is_null(const std::string& v) { return v == "null"; }

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

inline bool get_num(const std::string& s, const std::string& key,
                    double& out) {
  std::string v;
  if (!get_raw(s, key, v)) return false;
  if (is_null(v))
    throw std::runtime_error("config: '" + key + "' must not be null");
  out = parse_num(v, key);
  return true;
}

inline bool get_int(const std::string& s, const std::string& key,
                    long long& out) {
  double d = 0;
  if (!get_num(s, key, d)) return false;
  if (std::floor(d) != d)
    throw std::runtime_error("config: '" + key + "' must be an integer");
  out = std::llround(d);
  return true;
}

inline bool get_bool(const std::string& s, const std::string& key,
                     bool& out) {
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
  throw std::runtime_error("config: '" + key +
                           "' must be true/false (boolean)");
}

inline int checked_int(long long v, const std::string& key) {
  if (v < 0 || v > 1000000000LL)
    throw std::runtime_error("config: '" + key + "' out of range");
  return (int)v;
}

// Parse a JSON array of integers (e.g. local_layer_ids) into `out`.
inline bool get_int_array(const std::string& s, const std::string& key,
                          std::vector<int>& out) {
  std::string v;
  if (!get_raw(s, key, v)) return false;
  std::string t = trim(v);
  if (is_null(t))
    throw std::runtime_error("config: '" + key + "' must not be null");
  if (t.empty() || t.front() != '[')
    throw std::runtime_error("config: '" + key + "' must be an array");
  std::vector<int> vals;
  size_t p = 1;
  while (true) {
    skip_ws(t, p);
    if (p >= t.size())
      throw std::runtime_error("config: unbalanced '" + key + "'");
    if (t[p] == ']') break;
    size_t e = p;
    while (e < t.size() && t[e] != ',' && t[e] != ']') ++e;
    vals.push_back(
        checked_int(std::llround(parse_num(trim(t.substr(p, e - p)), key)),
                    key));
    p = e;
    skip_ws(t, p);
    if (p < t.size() && t[p] == ',') {
      ++p;
      continue;
    }
    if (p < t.size() && t[p] == ']') break;
    throw std::runtime_error("config: expected ',' or ']' in '" + key + "'");
  }
  out = std::move(vals);
  return true;
}

}  // namespace Impl

struct Config {
  int n_layers         = 24;
  int d_model          = 2304;
  int n_heads          = 24;   // query heads
  int n_kv_heads       = 8;    // GQA KV heads (all layers)
  // Explicit per-head dimension: q/k/v/o shapes derive from it
  // (n_heads*head_dim = 3072, n_kv_heads*head_dim = 1024), never from
  // d_model / n_heads.
  int head_dim         = 128;
  int d_rel            = 16;   // relative-position feature dim
  int rel_extent       = 1024; // relative-position extent
  int vocab_size       = 201024;
  int unpadded_vocab   = 200058;
  int n_experts        = 128;  // routed MoE experts per layer
  int top_k            = 8;    // experts per token
  int intermediate_size = 1152;  // MoE expert FFN dim
  int sconv_kernel     = 4;    // shortconv kernel size
  int swa_window       = 1024; // sliding-window causal width (local layers)
  int loop_start       = 6;    // loop core: layers [loop_start, loop_end)
  int loop_end         = 18;
  int loop_passes      = 3;
  int gate_rank        = 64;   // loop gate rank
  int attn_res_block   = 6;    // layers per attention residual block
  float eps            = 1e-6f;
  float qk_clip        = 22.627417f;
  float rel_clip       = 8.0f;
  int log_floor        = 8192;  // log-scale floor
  float log_alpha      = 0.1f;  // log-scale alpha
  float gate_softcap   = 4.0f;  // MoE gate softcap
  float up_softcap     = 25.0f; // MoE up-projection softcap
  float final_softcap  = 30.0f; // final logit softcap
  float route_scale    = 8.0f;   // expert route scale
  bool use_gate_bias   = true;
  bool attn_res_normalize = true;
  // Sliding (local) layer ids; the remaining layers (5, 11, 17, 23) are
  // global dynamicconv layers carrying k/v_conv_proj + q/k/v_inp_norm
  // instead of attn_norm.
  std::vector<int> local_layer_ids = {0, 1, 2,  3,  4,  6,  7,  8,  9, 10,
                                      12, 13, 14, 15, 16, 18, 19, 20, 21, 22};
  int max_seq_len = 32768;  // PoC runtime cap; allocation cap is 1048576
  // Chunked prefill: forward() processes prompts in chunks of this many
  // tokens so activations stay O(chunk*d_model) instead of O(T*d_model).
  // <= 0 means single-shot (one chunk of T).
  int prefill_chunk_size = 2048;
  // KV-cache precision. kv_int8 selects symmetric int8 KV with per-token
  // per-head FP32 scales; false keeps the model Scalar (dense FP).
  bool kv_int8    = false;
  int eos_token   = 200006;  // <|content_model_end_sampling|>

  static Config inkling_2_mini() { return Config{}; }

  bool is_local(int layer) const {
    for (int l : local_layer_ids)
      if (l == layer) return true;
    return false;
  }

  bool use_quant_kv() const { return kv_int8; }

  // Total K+V bytes across all layers for the full context window, given
  // the unquantized element size (e.g. sizeof(Scalar)). Dense bound: every
  // layer rents [max_seq_len, n_kv_heads, head_dim] K and V (local layers
  // need only the sliding window at runtime; this stays conservative).
  // Quantized cache: int8 values plus one FP32 scale per token per KV head
  // per K/V.
  size_t kv_cache_bytes(size_t elem_bytes) const {
    const size_t toks   = (size_t)max_seq_len;
    const size_t per_tok =
        (size_t)n_kv_heads * (size_t)head_dim * (kv_int8 ? 1 : elem_bytes);
    size_t bytes = (size_t)n_layers * toks * per_tok * 2;  // K + V
    if (kv_int8)
      bytes += (size_t)n_layers * toks * (size_t)n_kv_heads * sizeof(float) *
               2;  // K + V scale tables
    return bytes;
  }

  // Throw std::invalid_argument on incompatible dims or out-of-range
  // context. Called by from_json_* and by the Model (which rents the
  // [max_seq_len, n_kv_heads, head_dim] KV cache from these values).
  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Config::validate: " + m);
    };
    if (d_model <= 0) bad("d_model (hidden_size) must be > 0");
    if (n_layers <= 0) bad("n_layers (num_hidden_layers) must be > 0");
    if (n_heads <= 0) bad("n_heads (num_attention_heads) must be > 0");
    if (n_kv_heads <= 0) bad("n_kv_heads (num_key_value_heads) must be > 0");
    if (n_kv_heads > n_heads)
      bad("num_key_value_heads (" + std::to_string(n_kv_heads) +
          ") must not exceed num_attention_heads (" + std::to_string(n_heads) +
          ")");
    if (n_heads % n_kv_heads != 0)
      bad("num_attention_heads (" + std::to_string(n_heads) +
          ") must be a multiple of num_key_value_heads (" +
          std::to_string(n_kv_heads) + ") for grouped-query attention");
    if (head_dim <= 0 || head_dim % 2 != 0)
      bad("head_dim must be positive and even");
    if (d_rel <= 0) bad("d_rel must be > 0");
    if (rel_extent <= 0) bad("rel_extent must be > 0");
    if (vocab_size <= 0) bad("vocab_size must be > 0");
    if (unpadded_vocab <= 0) bad("unpadded_vocab must be > 0");
    if (unpadded_vocab > vocab_size)
      bad("unpadded_vocab (" + std::to_string(unpadded_vocab) +
          ") must not exceed vocab_size (" + std::to_string(vocab_size) + ")");
    if (n_experts <= 0) bad("n_experts (n_routed_experts) must be > 0");
    if (top_k <= 0) bad("top_k (num_experts_per_tok) must be > 0");
    if (top_k > n_experts)
      bad("num_experts_per_tok (" + std::to_string(top_k) +
          ") must not exceed n_routed_experts (" + std::to_string(n_experts) +
          ")");
    if (intermediate_size <= 0)
      bad("intermediate_size must be > 0");
    if (sconv_kernel <= 0) bad("sconv_kernel (sconv_kernel_size) must be > 0");
    if (swa_window <= 0) bad("swa_window (sliding_window_size) must be > 0");
    if (loop_start < 0 || loop_end > n_layers || loop_start >= loop_end)
      bad("loop range [loop_start, loop_end) must satisfy 0 <= start < end "
          "<= n_layers");
    if (loop_passes < 1) bad("loop_passes (loop num_passes) must be >= 1");
    if (gate_rank <= 0) bad("gate_rank (loop gate_rank) must be > 0");
    if (attn_res_block <= 0)
      bad("attn_res_block (n_layers_per_attn_res_block) must be > 0");
    if (n_layers % attn_res_block != 0)
      bad("num_hidden_layers (" + std::to_string(n_layers) +
          ") must be a multiple of n_layers_per_attn_res_block (" +
          std::to_string(attn_res_block) + ")");
    if (!(eps > 0) || !std::isfinite(eps))
      bad("rms_norm_eps (eps) must be positive and finite");
    if (!(qk_clip > 0) || !std::isfinite(qk_clip))
      bad("qk_l2_clip_threshold (qk_clip) must be positive and finite");
    if (!(rel_clip > 0) || !std::isfinite(rel_clip))
      bad("rel_l2_clip_threshold (rel_clip) must be positive and finite");
    if (log_floor <= 0) bad("log_scaling_n_floor (log_floor) must be > 0");
    if (!(log_alpha >= 0) || !std::isfinite(log_alpha))
      bad("log_scaling_alpha (log_alpha) must be >= 0 and finite");
    if (!(gate_softcap > 0) || !std::isfinite(gate_softcap))
      bad("mlp_gate_softcap (gate_softcap) must be positive and finite");
    if (!(up_softcap > 0) || !std::isfinite(up_softcap))
      bad("mlp_up_softcap (up_softcap) must be positive and finite");
    if (!(final_softcap > 0) || !std::isfinite(final_softcap))
      bad("final_logit_softcapping (final_softcap) must be positive and "
          "finite");
    if (!(route_scale > 0) || !std::isfinite(route_scale))
      bad("route_scale must be positive and finite");
    if (local_layer_ids.empty()) bad("local_layer_ids must not be empty");
    for (size_t i = 0; i < local_layer_ids.size(); ++i) {
      int l = local_layer_ids[i];
      if (l < 0 || l >= n_layers)
        bad("local_layer_ids[" + std::to_string(i) + "] (" +
            std::to_string(l) + ") out of [0, num_hidden_layers)");
      if (i > 0 && l <= local_layer_ids[i - 1])
        bad("local_layer_ids must be strictly increasing");
    }
    if (max_seq_len < kMinSupportedSeqLen ||
        max_seq_len > kMaxSupportedSeqLen)
      bad("max_seq_len (" + std::to_string(max_seq_len) +
          ") out of supported range [" +
          std::to_string(kMinSupportedSeqLen) + ", " +
          std::to_string(kMaxSupportedSeqLen) +
          "]; the KV cache rents [max_seq_len, n_kv_heads, head_dim] per "
          "layer, so cap it to what fits in memory");
    if (prefill_chunk_size < 0)
      bad("prefill_chunk_size must be >= 0 (0 selects single-shot prefill)");
    if (eos_token < 0) bad("eos_token_id (eos_token) must be >= 0");
  }

  // Parse a text_config object scope (the `text` scope already narrowed to
  // the text_config object when the file is a full HF config.json).
  static Config from_json_scope(const std::string& text) {
    using namespace Impl;
    Config c;
    long long iv = 0;
    double dv    = 0;
    bool bv      = false;
    std::string v;
    if (get_int(text, "hidden_size", iv)) c.d_model = checked_int(iv, "hidden_size");
    if (get_int(text, "num_hidden_layers", iv))
      c.n_layers = checked_int(iv, "num_hidden_layers");
    if (get_int(text, "num_attention_heads", iv))
      c.n_heads = checked_int(iv, "num_attention_heads");
    if (get_int(text, "num_key_value_heads", iv))
      c.n_kv_heads = checked_int(iv, "num_key_value_heads");
    if (get_int(text, "head_dim", iv)) {
      if (iv <= 0 || iv > 1000000000LL || iv % 2 != 0)
        throw std::runtime_error(
            "config: 'head_dim' must be positive and even");
      c.head_dim = (int)iv;
    }
    if (get_int(text, "d_rel", iv)) c.d_rel = checked_int(iv, "d_rel");
    if (get_int(text, "rel_extent", iv))
      c.rel_extent = checked_int(iv, "rel_extent");
    if (get_int(text, "vocab_size", iv))
      c.vocab_size = checked_int(iv, "vocab_size");
    if (get_int(text, "unpadded_vocab_size", iv))
      c.unpadded_vocab = checked_int(iv, "unpadded_vocab_size");
    if (get_int(text, "n_routed_experts", iv))
      c.n_experts = checked_int(iv, "n_routed_experts");
    if (get_int(text, "num_experts_per_tok", iv))
      c.top_k = checked_int(iv, "num_experts_per_tok");
    if (get_int(text, "intermediate_size", iv))
      c.intermediate_size = checked_int(iv, "intermediate_size");
    if (get_int(text, "sconv_kernel_size", iv))
      c.sconv_kernel = checked_int(iv, "sconv_kernel_size");
    if (get_int(text, "sliding_window_size", iv))
      c.swa_window = checked_int(iv, "sliding_window_size");
    if (get_int(text, "n_layers_per_attn_res_block", iv))
      c.attn_res_block = checked_int(iv, "n_layers_per_attn_res_block");
    if (get_num(text, "rms_norm_eps", dv)) c.eps = (float)dv;
    if (get_num(text, "qk_l2_clip_threshold", dv)) c.qk_clip = (float)dv;
    if (get_num(text, "rel_l2_clip_threshold", dv)) c.rel_clip = (float)dv;
    if (get_int(text, "log_scaling_n_floor", iv))
      c.log_floor = checked_int(iv, "log_scaling_n_floor");
    if (get_num(text, "log_scaling_alpha", dv)) c.log_alpha = (float)dv;
    if (get_num(text, "mlp_gate_softcap", dv)) c.gate_softcap = (float)dv;
    if (get_num(text, "mlp_up_softcap", dv)) c.up_softcap = (float)dv;
    if (get_num(text, "final_logit_softcapping", dv))
      c.final_softcap = (float)dv;
    if (get_num(text, "route_scale", dv)) c.route_scale = (float)dv;
    if (get_bool(text, "use_gate_bias", bv)) c.use_gate_bias = bv;
    if (get_bool(text, "attn_res_normalize_values", bv))
      c.attn_res_normalize = bv;
    if (get_int_array(text, "local_layer_ids", c.local_layer_ids)) {
    }
    // loop_config: {core_start, core_end, num_passes, gate_rank}.
    if (get_raw(text, "loop_config", v)) {
      std::string t = trim(v);
      if (!is_null(t)) {
        if (t.empty() || t.front() != '{')
          throw std::runtime_error("config: 'loop_config' must be an object");
        if (get_int(t, "core_start", iv))
          c.loop_start = checked_int(iv, "loop_config.core_start");
        if (get_int(t, "core_end", iv))
          c.loop_end = checked_int(iv, "loop_config.core_end");
        if (get_int(t, "num_passes", iv))
          c.loop_passes = checked_int(iv, "loop_config.num_passes");
        if (get_int(t, "gate_rank", iv))
          c.gate_rank = checked_int(iv, "loop_config.gate_rank");
      }
    }
    // eos_token_id is an int in config.json but a list in
    // generation_config.json; accept either (first element wins).
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
        c.eos_token = checked_int(
            std::llround(parse_num(trim(v.substr(p, e - p)), "eos_token_id")),
            "eos_token_id");
      } else {
        c.eos_token = checked_int(
            std::llround(parse_num(v, "eos_token_id")), "eos_token_id");
      }
    }
    // Runtime (non-HF) keys. model_max_length is intentionally NOT read:
    // it is the architecture limit (1048576), not the rented window.
    if (get_int(text, "max_seq_len", iv))
      c.max_seq_len = checked_int(iv, "max_seq_len");
    if (get_int(text, "prefill_chunk_size", iv)) {
      if (iv < 0 || iv > 1000000000LL)
        throw std::runtime_error("config: 'prefill_chunk_size' out of range");
      c.prefill_chunk_size = (int)iv;
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
    c.validate();
    return c;
  }

  // Parse an HF config.json object. Accepts either the full Inkling 2 Mini
  // config.json (text fields nested under "text_config") or a bare
  // text_config object. Keys present override the inkling_2_mini() defaults;
  // missing keys keep them. A top-level "eos_token_id" applies when the
  // text_config scope carries none. Unknown keys are ignored. Throws on
  // malformed JSON or incompatible dims.
  static Config from_json_string(const std::string& text) {
    using namespace Impl;
    std::string scope = text;
    bool narrowed     = false;
    std::string sub;
    if (get_raw(text, "text_config", sub)) {
      std::string t = trim(sub);
      if (is_null(t))
        throw std::runtime_error("config: 'text_config' must not be null");
      if (t.empty() || t.front() != '{')
        throw std::runtime_error("config: 'text_config' must be an object");
      scope   = t;
      narrowed = true;
    }
    Config c = from_json_scope(scope);
    // Top-level eos fallback (the full config.json carries eos_token_id
    // next to text_config, not inside it). Only fills in when the scope
    // had none.
    if (narrowed) {
      std::string v;
      bool scope_had_eos = get_raw(scope, "eos_token_id", v);
      if (!scope_had_eos && get_raw(text, "eos_token_id", v)) {
        if (!is_null(v) && !v.empty() && v.front() != '[')
          c.eos_token = checked_int(
              std::llround(parse_num(v, "eos_token_id")), "eos_token_id");
      }
      c.validate();
    }
    return c;
  }

  static Config from_json_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("config: cannot open config file '" + path +
                               "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_string(ss.str());
  }
};

}  // namespace Inkling
}  // namespace KokkosDNN

#endif
