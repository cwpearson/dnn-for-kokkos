#include "KokkosDNN_Gemma4Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace KokkosDNN {
namespace Gemma4 {
namespace {

// ---------------------------------------------------------------------------
// File + JSON helpers (schema-targeted, no third-party JSON dependency).
// Same style as the Llama/SmolLM2 tokenizers.
// ---------------------------------------------------------------------------

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + p);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

size_t skip_ws(const std::string& s, size_t p) {
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

// Decode a JSON string; s[q] must be the opening quote. Returns the decoded
// value and advances q past the closing quote. Handles all standard escapes
// including \uXXXX with UTF-16 surrogate pairs.
std::string json_parse_string(const std::string& s, size_t& q) {
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
size_t find_key_value(const std::string& js, const char* key, size_t from) {
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

// Extract the body (without outer braces) of the object value of `key`
// searched from `from`. Throws when absent/unbalanced.
std::string object_body(const std::string& js, const char* key, size_t from,
                        const char* what) {
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
std::string array_body(const std::string& js, const char* key, size_t from,
                       const char* what) {
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

long parse_number(const std::string& s, size_t& p) {
  p = skip_ws(s, p);
  size_t e = p;
  if (e < s.size() && (s[e] == '-' || s[e] == '+')) ++e;
  while (e < s.size() && std::isdigit((unsigned char)s[e])) ++e;
  if (e == p) throw std::runtime_error("tokenizer: expected number");
  long v = std::stol(s.substr(p, e - p));
  p = e;
  return v;
}

// ---------------------------------------------------------------------------
// SentencePiece-▁ helpers.
// ---------------------------------------------------------------------------

// U+2581 LOWER ONE EIGHTH BLOCK in UTF-8 (the SentencePiece space marker).
inline const char* sp_space() { return "\xe2\x96\x81"; }

// Normalizer (Replace " " -> ▁): every ASCII space becomes U+2581.
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

// "<0xXX>" byte token for byte b (uppercase hex, matching the vocab).
std::string byte_token(unsigned b) {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "<0x%02X>", b & 0xFF);
  return std::string(buf);
}

// Parse "<0xXX>" into the byte value; returns -1 when not a byte token.
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

// Length of the UTF-8 sequence starting at s[i] (lenient: invalid
// sequences decode as single bytes so byte_fallback always terminates).
size_t utf8_len_here(const std::string& s, size_t i) {
  unsigned char c = s[i];
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

void Tokenizer::load(const std::string& path) {
  path_ = path;
  std::string js = read_file(path);

  token_to_id_.clear();
  merge_rank_.clear();
  added_tokens_.clear();
  special_ids_.clear();
  special_content_.clear();

  // model.{vocab,merges}: scope the search inside "model" so same-named
  // keys elsewhere cannot interfere.
  size_t model_pos = find_key_value(js, "model", 0);
  if (model_pos == std::string::npos)
    throw std::runtime_error("tokenizer: no model object");
  model_pos = skip_ws(js, model_pos);
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

  // merges: Gemma ships [["a", "b"], ...] pairs; accept the legacy
  // single-string "a b" form as well. The rank key is "a b".
  {
    std::string merges = array_body(js, "merges", model_pos, "model.merges");
    size_t j = 0;
    int rank = 0;
    while (j < merges.size()) {
      j = skip_ws(merges, j);
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
        size_t q = skip_ws(pair, 0);
        if (q >= pair.size() || pair[q] != '"')
          throw std::runtime_error("tokenizer: bad merge pair");
        std::string a = json_parse_string(pair, q);
        q = skip_ws(pair, q);
        if (q < pair.size() && pair[q] == ',') ++q;
        q = skip_ws(pair, q);
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

  // added_tokens: [{id, content, special, ...}] in any field order.
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
      size_t q = skip_ws(obj, cv);
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
        size_t t = skip_ws(obj, sv);
        if (obj.compare(t, 4, "true") == 0) special = true;
        else if (obj.compare(t, 5, "false") == 0) special = false;
      }
      if (id >= 0 && !content.empty()) {
        // added_tokens may extend the vocab (262144+ for Gemma4).
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
    // longest content first for greedy longest-match splitting
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const AddedToken& a, const AddedToken& b) {
                return a.content.size() > b.content.size();
              });
  }

  // Resolve well-known specials by content, else keep the Gemma
  // conventions (bos=2 <bos>, eos=1 <eos>, pad=0 <pad>, unk=3 <unk>).
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

  // Sibling tokenizer_config.json override (best-effort, never throws):
  // records the bos/eos string-name choice. Only attempted when the
  // tokenizer path contains a directory.
  {
    size_t slash = path_.find_last_of('/');
    if (slash != std::string::npos) {
      try {
        std::string cfg =
            read_file(path_.substr(0, slash + 1) + "tokenizer_config.json");
        auto override_from_config = [&](const char* key, int& target) {
          size_t kv = find_key_value(cfg, key, 0);
          if (kv == std::string::npos) return;
          size_t q = skip_ws(cfg, kv);
          if (q >= cfg.size() || cfg[q] != '"') return;  // null/non-string
          std::string tok = json_parse_string(cfg, q);
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
  // Start symbols: single Unicode characters when in-vocab, else the
  // character's UTF-8 bytes as <0xXX> byte tokens (byte_fallback=true).
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
  // Greedily merge the lowest-rank adjacent pair.
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
    parts.erase(parts.begin() + best + 1);
  }
  // Every merged piece must resolve: merged vocab entries, single chars,
  // or byte tokens. Anything else falls back to <unk> (fuse_unk=true).
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

  // Split out literal special tokens first (longest match wins; Gemma4
  // specials start with '<', e.g. <eos>, <|turn|>).
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
    // Normalizer (space -> ▁) + Split(" ", MergedWithPrevious): no literal
    // space survives normalization, so the segment is a single BPE piece.
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
  auto append_token = [&](const std::string& tok) {
    // ByteFallback: <0xXX> tokens become raw bytes.
    int b = parse_byte_token(tok);
    if (b >= 0)
      acc += (char)b;
    else
      acc += tok;
  };
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
    append_token(id_to_token_[(size_t)id]);
  }
  // Decoder Replace(▁ -> " ") + Fuse: the concatenation already fuses.
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

std::vector<int> ensure_bos(const std::vector<int>& ids, int bos_id,
                            bool want_bos) {
  size_t k = 0;
  while (k < ids.size() && ids[k] == bos_id) ++k;
  if (want_bos) {
    std::vector<int> out;
    out.reserve(ids.size() - k + 1);
    out.push_back(bos_id);
    out.insert(out.end(), ids.begin() + (long)k, ids.end());
    return out;
  }
  return std::vector<int>(ids.begin() + (long)k, ids.end());
}

size_t find_stop_string(const std::string& text,
                        const std::vector<std::string>& stop_strings,
                        std::string* matched) {
  size_t best = std::string::npos;
  const std::string* which = nullptr;
  for (const auto& s : stop_strings) {
    if (s.empty()) continue;
    size_t p = text.find(s);
    if (p != std::string::npos && (best == std::string::npos || p < best)) {
      best  = p;
      which = &s;
    }
  }
  if (matched) *matched = (which ? *which : std::string());
  return best;
}

std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings) {
  size_t p = find_stop_string(text, stop_strings);
  if (p == std::string::npos) return text;
  return text.substr(0, p);
}

}  // namespace Gemma4
}  // namespace KokkosDNN
