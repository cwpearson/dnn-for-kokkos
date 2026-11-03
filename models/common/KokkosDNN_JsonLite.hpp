// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_JSONLITE_HPP
#define KOKKOSDNN_JSONLITE_HPP

// Minimal dependency-free JSON helpers for tokenizer.json-style files
// (models/common).
//
// Consolidates the byte-identical helper blocks previously duplicated in
//   examples/llama/KokkosDNN_Tokenizer.cpp,
//   examples/smollm2/KokkosDNN_SmolLM2Tokenizer.cpp,
//   models/lfm2_5/KokkosDNN_LFM25Tokenizer.cpp,
//   models/glimmer/KokkosDNN_GlimmerTokenizer.cpp,
//   examples/gemma4/KokkosDNN_Gemma4Tokenizer.cpp,
//   examples/sana/KokkosDNN_SanaTextEncoder.cpp (tok_* renames),
// plus the renamed safetensors-header string scanners
//   models/moss_image/KokkosDNN_MossImageDit.hpp (dit_parse_json_string),
//   models/qwen_image_21/KokkosDNN_QwenImage21Config.hpp
//     (qwen21_parse_json_string).
// Those two are strict subsets (ASCII headers, no \u handling); their call
// sites now use the full json_parse_string below, which agrees on ASCII
// input and additionally decodes escapes correctly.
//
// Scope: flat schema-targeted lookup only (no DOM, no UTF-16 outside
// json_parse_string). Error messages keep the historical "tokenizer: ..."
// prefix so existing diagnostics are unchanged.
//
// No Kokkos dependency: <string>/<fstream>/<sstream>/<stdexcept>/<cctype>.

#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Common {
namespace JsonLite {

inline std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + p);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

inline size_t skip_ws(const std::string& s, size_t p) {
  while (p < s.size() &&
         (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
    ++p;
  return p;
}

inline void utf8_encode_cp(uint32_t cp, std::string& out) {
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

// Decode a JSON string; s[q] must be the opening quote. Returns the decoded
// value and advances q past the closing quote. Handles all standard escapes
// including \uXXXX with UTF-16 surrogate pairs.
inline std::string json_parse_string(const std::string& s, size_t& q) {
  ++q;  // opening quote
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
        // surrogate pair?
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

// Find `"key"` used as an object key at/after `from`: returns the position
// just past the ':' or npos. A match only counts when followed by ':'.
inline size_t find_key_value(const std::string& js, const char* key,
                             size_t from) {
  std::string pat = std::string("\"") + key + "\"";
  size_t p = from;
  while ((p = js.find(pat, p)) != std::string::npos) {
    size_t c = skip_ws(js, p + pat.size());
    if (c < js.size() && js[c] == ':') return c + 1;
    p += pat.size();
  }
  return std::string::npos;
}

// Given the position of an opening bracket, find the matching closer,
// respecting JSON strings. Returns npos on unbalanced input.
inline size_t match_bracket(const std::string& js, size_t open, char oc,
                            char cc) {
  bool in_str = false, esc = false;
  int depth = 0;
  for (size_t i = open; i < js.size(); ++i) {
    char c = js[i];
    if (in_str) {
      if (esc)
        esc = false;
      else if (c == '\\')
        esc = true;
      else if (c == '"')
        in_str = false;
    } else {
      if (c == '"')
        in_str = true;
      else if (c == oc)
        ++depth;
      else if (c == cc && --depth == 0)
        return i;
    }
  }
  return std::string::npos;
}

// Extract the body (without outer braces) of the object value of `key`
// searched from `from`. Throws when absent/unbalanced.
inline std::string object_body(const std::string& js, const char* key,
                               size_t from, const char* what) {
  size_t v = find_key_value(js, key, from);
  if (v == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: no ") + what);
  v = skip_ws(js, v);
  if (v >= js.size() || js[v] != '{')
    throw std::runtime_error(std::string("tokenizer: ") + what +
                             " is not an object");
  size_t close = match_bracket(js, v, '{', '}');
  if (close == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: unbalanced ") + what);
  return js.substr(v + 1, close - v - 1);
}

// Same for an array value.
inline std::string array_body(const std::string& js, const char* key,
                              size_t from, const char* what) {
  size_t v = find_key_value(js, key, from);
  if (v == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: no ") + what);
  v = skip_ws(js, v);
  if (v >= js.size() || js[v] != '[')
    throw std::runtime_error(std::string("tokenizer: ") + what +
                             " is not an array");
  size_t close = match_bracket(js, v, '[', ']');
  if (close == std::string::npos)
    throw std::runtime_error(std::string("tokenizer: unbalanced ") + what);
  return js.substr(v + 1, close - v - 1);
}

inline long parse_number(const std::string& s, size_t& p) {
  p = skip_ws(s, p);
  size_t e = p;
  if (e < s.size() && (s[e] == '-' || s[e] == '+')) ++e;
  while (e < s.size() && std::isdigit((unsigned char)s[e])) ++e;
  if (e == p) throw std::runtime_error("tokenizer: expected number");
  long v = std::stol(s.substr(p, e - p));
  p = e;
  return v;
}

}  // namespace JsonLite
}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_JSONLITE_HPP
