#include "llama/KokkosDNN_Tokenizer.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace KokkosDNN {
namespace Llama {
namespace {

// Read whole file.
std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + p);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Decode a JSON string starting just after opening quote; handles
// \" \\ \n \t \uXXXX (BMP only, sufficient for tokenizer.json).
std::string json_unescape(const std::string& s, size_t& i) {
  std::string out;
  while (i < s.size()) {
    char c = s[i++];
    if (c == '"') break;
    if (c == '\\' && i < s.size()) {
      char e = s[i++];
      if (e == 'n') out += '\n';
      else if (e == 't') out += '\t';
      else if (e == 'r') out += '\r';
      else if (e == 'u' && i + 4 <= s.size()) {
        unsigned code = std::stoul(s.substr(i, 4), nullptr, 16);
        i += 4;
        // encode as UTF-8 (BMP)
        if (code < 0x80) out += (char)code;
        else if (code < 0x800) {
          out += (char)(0xC0 | (code >> 6));
          out += (char)(0x80 | (code & 0x3F));
        } else {
          out += (char)(0xE0 | (code >> 12));
          out += (char)(0x80 | ((code >> 6) & 0x3F));
          out += (char)(0x80 | (code & 0x3F));
        }
      } else {
        out += e;  // \" \\ \/ etc.
      }
    } else {
      out += c;
    }
  }
  return out;
}

// Byte-to-unicode table (GPT-2 style) so raw bytes map to printable
// single-char tokens when no explicit vocab entry exists.
std::map<int, std::string> byte_encoder() {
  std::map<int, std::string> m;
  auto chr = [](int c) { return std::string(1, (char)c); };
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if ((b >= '!' && b <= '~') || (b >= 0xA1)) {
      // representable directly
    } else {
      (void)n;
    }
    (void)chr;
  }
  // Simple identity fallback: byte -> single char string.
  for (int b = 0; b < 256; ++b) m[b] = std::string(1, (char)(unsigned char)b);
  return m;
}

}  // namespace

void Tokenizer::load(const std::string& path) {
  path_ = path;
  std::string js = read_file(path);

  // --- vocab: find "vocab": { ... } ---
  token_to_id_.clear();
  id_to_token_.clear();
  merge_rank_.clear();
  size_t vp = js.find("\"vocab\"");
  if (vp == std::string::npos) throw std::runtime_error("tokenizer: no vocab");
  size_t ob = js.find('{', vp);
  int depth = 0;
  bool in_str = false, esc = false;
  size_t cb = ob;
  for (; cb < js.size(); ++cb) {
    char c = js[cb];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
    } else {
      if (c == '"') in_str = true;
      else if (c == '{') depth++;
      else if (c == '}') {
        if (--depth == 0) break;
      }
    }
  }
  std::string vocab = js.substr(ob + 1, cb - ob - 1);
  // entries: "token": id
  size_t i = 0, max_id = 0;
  std::map<int, std::string> idmap;
  while (i < vocab.size()) {
    size_t q = vocab.find('"', i);
    if (q == std::string::npos) break;
    size_t k = q + 1;
    std::string tok = json_unescape(vocab, k);
    size_t colon = vocab.find(':', k);
    size_t p = colon + 1;
    while (p < vocab.size() && std::isspace((unsigned char)vocab[p])) p++;
    size_t e = p;
    while (e < vocab.size() && std::isdigit((unsigned char)vocab[e])) e++;
    int id = std::stoi(vocab.substr(p, e - p));
    token_to_id_[tok] = id;
    idmap[id] = tok;
    if ((size_t)id > max_id) max_id = id;
    i = e;
  }
  vocab_size_ = (int)max_id + 1;
  id_to_token_.assign(vocab_size_, "");
  for (auto& [id, tok] : idmap) id_to_token_[id] = tok;

  // --- merges: "merges": [ "a b", ... ] ---
  size_t mp = js.find("\"merges\"");
  if (mp != std::string::npos) {
    size_t ab = js.find('[', mp);
    size_t ae = ab + 1;
    // find matching ] respecting strings
    {
      bool s = false, es = false;
      int dd = 0;
      for (; ae < js.size(); ++ae) {
        char c = js[ae];
        if (s) {
          if (es) es = false;
          else if (c == '\\') es = true;
          else if (c == '"') s = false;
        } else {
          if (c == '"') s = true;
          else if (c == '[') dd++;
          else if (c == ']') {
            if (dd == 0) break;
            dd--;
          }
        }
      }
    }
    std::string body = js.substr(ab + 1, ae - ab - 1);
    size_t j = 0;
    int rank = 0;
    while (j < body.size()) {
      size_t q = body.find('"', j);
      if (q == std::string::npos) break;
      size_t k = q + 1;
      std::string m = json_unescape(body, k);
      merge_rank_[m] = rank++;
      j = k;
    }
  }

  // --- added tokens may extend vocab beyond "vocab" (id/content pairs) ---
  size_t ap = 0;
  while ((ap = js.find("\"id\"", ap)) != std::string::npos) {
    size_t colon = js.find(':', ap);
    size_t p = colon + 1;
    while (p < js.size() && std::isspace((unsigned char)js[p])) p++;
    size_t e = p;
    while (e < js.size() && std::isdigit((unsigned char)js[e])) e++;
    int id = std::stoi(js.substr(p, e - p));
    size_t cp = js.find("\"content\"", e);
    if (cp == std::string::npos || cp > ap + 500) {
      ap = e;
      continue;
    }
    size_t cq = js.find('"', js.find(':', cp) + 1) + 1;
    size_t k = cq;
    std::string tok = json_unescape(js, k);
    if (id >= 0 && token_to_id_.find(tok) == token_to_id_.end()) {
      token_to_id_[tok] = id;
      if (id >= (int)id_to_token_.size()) id_to_token_.resize(id + 1);
      id_to_token_[id] = tok;
      if (id + 1 > vocab_size_) vocab_size_ = id + 1;
    }
    ap = k;
  }
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                   bool add_bos) const {
  std::vector<int> ids;
  if (add_bos) ids.push_back(bos());
  if (text.empty() || !loaded()) return ids;
  // Start from UTF-8 bytes; greedily merge lowest-rank adjacent pair.
  std::vector<std::string> parts;
  for (unsigned char c : text) parts.emplace_back(1, (char)c);
  // Map raw bytes through vocab when byte-tokens exist (tiktoken uses
  // explicit byte entries); our identity fallback keeps single chars.
  while (true) {
    int best_rank = -1;
    size_t best = 0;
    for (size_t k = 0; k + 1 < parts.size(); ++k) {
      auto it = merge_rank_.find(parts[k] + " " + parts[k + 1]);
      if (it != merge_rank_.end() && (best_rank < 0 || it->second < best_rank)) {
        best_rank = it->second;
        best = k;
      }
    }
    if (best_rank < 0) break;
    parts[best] += parts[best + 1];
    parts.erase(parts.begin() + best + 1);
  }
  for (auto& s : parts) {
    auto it = token_to_id_.find(s);
    if (it != token_to_id_.end()) {
      ids.push_back(it->second);
    } else {
      // unknown: fall back to per-byte vocab ids when available
      for (unsigned char c : s) {
        std::string b(1, (char)c);
        auto jt = token_to_id_.find(b);
        if (jt != token_to_id_.end()) ids.push_back(jt->second);
      }
    }
  }
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for (int id : ids) {
    if (id == bos() || id < 0) continue;
    if (id < (int)id_to_token_.size()) out += id_to_token_[id];
  }
  return out;
}

}  // namespace Llama
}  // namespace KokkosDNN
