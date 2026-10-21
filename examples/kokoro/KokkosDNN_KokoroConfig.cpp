#include "KokkosDNN_KokoroConfig.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace KokkosDNN {
namespace Kokoro {
namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Kokoro load_config: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

long get_int(const std::string& s, const std::string& key, long dflt) {
  const std::string q = "\"" + key + "\"";
  size_t p = s.find(q);
  if (p == std::string::npos) return dflt;
  p = s.find(':', p + q.size());
  if (p == std::string::npos) return dflt;
  size_t e = p + 1;
  while (e < s.size() && std::isspace((unsigned char)s[e])) ++e;
  bool neg = false;
  if (e < s.size() && (s[e] == '-' || s[e] == '+')) {
    neg = s[e] == '-';
    ++e;
  }
  long v = 0;
  bool any = false;
  while (e < s.size() && std::isdigit((unsigned char)s[e])) {
    v = v * 10 + (s[e] - '0');
    ++e;
    any = true;
  }
  if (!any) return dflt;
  return neg ? -v : v;
}

bool get_bool(const std::string& s, const std::string& key, bool dflt) {
  const std::string q = "\"" + key + "\"";
  size_t p = s.find(q);
  if (p == std::string::npos) return dflt;
  p = s.find(':', p + q.size());
  if (p == std::string::npos) return dflt;
  size_t e = p + 1;
  while (e < s.size() && std::isspace((unsigned char)s[e])) ++e;
  if (s.compare(e, 4, "true") == 0) return true;
  if (s.compare(e, 5, "false") == 0) return false;
  return dflt;
}

// Parse a JSON string literal starting at s[p]=='"'. Returns decoded bytes
// and the index one past the closing quote. Handles \" \\ \/ \b \f \n \r
// \t \uXXXX (BMP only, re-encoded as UTF-8).
std::pair<std::string, size_t> parse_json_string(const std::string& s,
                                                 size_t p) {
  std::string out;
  size_t i = p + 1;
  while (i < s.size()) {
    char c = s[i];
    if (c == '"') return {out, i + 1};
    if (c == '\\') {
      if (i + 1 >= s.size())
        throw std::runtime_error("Kokoro load_config: bad escape");
      char e = s[i + 1];
      if (e == '"' || e == '\\' || e == '/') {
        out.push_back(e);
        i += 2;
      } else if (e == 'b') {
        out.push_back('\b');
        i += 2;
      } else if (e == 'f') {
        out.push_back('\f');
        i += 2;
      } else if (e == 'n') {
        out.push_back('\n');
        i += 2;
      } else if (e == 'r') {
        out.push_back('\r');
        i += 2;
      } else if (e == 't') {
        out.push_back('\t');
        i += 2;
      } else if (e == 'u') {
        if (i + 5 >= s.size())
          throw std::runtime_error("Kokoro load_config: bad \\u");
        unsigned cp = 0;
        for (int k = 0; k < 4; ++k) {
          char h = s[i + 2 + k];
          cp <<= 4;
          if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
          else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
          else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
          else
            throw std::runtime_error("Kokoro load_config: bad \\u hex");
        }
        if (cp < 0x80) {
          out.push_back((char)cp);
        } else if (cp < 0x800) {
          out.push_back((char)(0xC0 | (cp >> 6)));
          out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
          out.push_back((char)(0xE0 | (cp >> 12)));
          out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
          out.push_back((char)(0x80 | (cp & 0x3F)));
        }
        i += 6;
      } else {
        throw std::runtime_error("Kokoro load_config: bad escape");
      }
    } else {
      out.push_back(c);
      ++i;
    }
  }
  throw std::runtime_error("Kokoro load_config: unterminated string");
}

}  // namespace

Config load_config(const std::string& path, Vocab& vocab) {
  const std::string s = read_file(path);
  Config c;
  c.hidden_dim = (int)get_int(s, "hidden_dim", c.hidden_dim);
  c.style_dim = (int)get_int(s, "style_dim", c.style_dim);
  c.n_layer = (int)get_int(s, "n_layer", c.n_layer);
  c.n_token = (int)get_int(s, "n_token", c.n_token);
  c.text_encoder_kernel_size =
      (int)get_int(s, "text_encoder_kernel_size", c.text_encoder_kernel_size);
  c.max_dur = (int)get_int(s, "max_dur", c.max_dur);
  c.n_mels = (int)get_int(s, "n_mels", c.n_mels);
  c.max_conv_dim = (int)get_int(s, "max_conv_dim", c.max_conv_dim);
  c.dim_in = (int)get_int(s, "dim_in", c.dim_in);
  c.multispeaker = get_bool(s, "multispeaker", c.multispeaker);
  c.plbert_hidden_size = (int)get_int(s, "hidden_size", c.plbert_hidden_size);
  c.plbert_heads = (int)get_int(s, "num_attention_heads", c.plbert_heads);
  c.plbert_intermediate =
      (int)get_int(s, "intermediate_size", c.plbert_intermediate);
  c.plbert_layers = (int)get_int(s, "num_hidden_layers", c.plbert_layers);
  c.plbert_max_pos =
      (int)get_int(s, "max_position_embeddings", c.plbert_max_pos);
  c.upsample_initial_channel =
      (int)get_int(s, "upsample_initial_channel", c.upsample_initial_channel);
  c.gen_istft_n_fft = (int)get_int(s, "gen_istft_n_fft", c.gen_istft_n_fft);
  c.gen_istft_hop = (int)get_int(s, "gen_istft_hop_size", c.gen_istft_hop);
  vocab.clear();
  size_t vp = s.find("\"vocab\"");
  if (vp == std::string::npos)
    throw std::runtime_error("Kokoro load_config: no vocab in " + path);
  vp = s.find('{', vp);
  if (vp == std::string::npos)
    throw std::runtime_error("Kokoro load_config: bad vocab in " + path);
  size_t i = vp + 1;
  int depth = 1;
  size_t vend = i;
  bool in_str = false, esc = false;
  for (; vend < s.size(); ++vend) {
    char ch = s[vend];
    if (in_str) {
      if (esc)
        esc = false;
      else if (ch == '\\')
        esc = true;
      else if (ch == '"')
        in_str = false;
    } else {
      if (ch == '"')
        in_str = true;
      else if (ch == '{')
        ++depth;
      else if (ch == '}') {
        if (--depth == 0) break;
      }
    }
  }
  size_t p = i;
  while (p < vend) {
    while (p < vend && s[p] != '"') ++p;
    if (p >= vend) break;
    auto ks = parse_json_string(s, p);
    p = ks.second;
    while (p < vend && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' ||
                        s[p] == '\r' || s[p] == ':'))
      ++p;
    bool neg = false;
    if (p < vend && (s[p] == '-' || s[p] == '+')) {
      neg = s[p] == '-';
      ++p;
    }
    long v = 0;
    bool any = false;
    while (p < vend && std::isdigit((unsigned char)s[p])) {
      v = v * 10 + (s[p] - '0');
      ++p;
      any = true;
    }
    if (any) vocab[ks.first] = (int)(neg ? -v : v);
    while (p < vend && s[p] != '"' && s[p] != ',') ++p;
    if (p < vend && s[p] == ',') ++p;
  }
  if (vocab.empty())
    throw std::runtime_error("Kokoro load_config: empty vocab in " + path);
  c.validate();
  return c;
}

}  // namespace Kokoro
}  // namespace KokkosDNN
