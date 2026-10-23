// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENTOKENIZER_HPP
#define KOKKOSDNN_QWENTOKENIZER_HPP

// Byte-level BPE tokenizer for the Qwen2/Qwen3 family (MOSS-Image-SFT text
// encoder input), plus the single-turn chat-template formatting used by the
// reference `encode_prompt`.
//
// Scope: host-side preprocessing only (std::string, no Kokkos). It consumes
// `tokenizer.json` (vocab + merges + added specials + pre-tokenizer shape,
// all verified against the MOSS-Image-SFT `tokenizer/` artifacts) and
// produces the token ids consumed by `Qwen35::EncoderModel::forward`.
//
// Fidelity contract (verified against transformers 5.12.1, see
// examples/moss_image/QWEN_NOTES.md and tests/Test_QwenTokenizer.cpp
// goldens):
//   formatted = "<|im_start|>user\n" + strip(prompt) + "<|im_end|>\n"
//               "<|im_start|>assistant\n"
//               + (thinking ? "<think>\n" : "<think>\n\n</think>\n\n")
//   ids = BPE(NFC(formatted)) truncated to max_length (right truncation),
//   with the 33 added specials matched as atomic tokens.
// The strip is Python str.strip(); NFC is full canonical
// (decompose + CCC reorder + compose, Hangul algorithmic) driven by the
// generated tables in KokkosDNN_QwenTokenizerTables.inc; the pre-tokenizer
// is the Qwen2 GPT-2-style regex executed as an ordered manual scan;
// merges come from tokenizer.json ranks. Only the single user-turn path of
// chat_template.jinja is implemented (that is the only shape the reference
// pipeline ever feeds the encoder); multi-turn/tool content is rejected
// with an exception rather than silently misformatted.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "KokkosDNN_QwenTokenizerTables.inc"

namespace KokkosDNN {
namespace QwenTokenizer {

namespace Impl {

// ---- UTF-8 ---------------------------------------------------------------
// Decode UTF-8 into codepoints; invalid sequences become U+FFFD (one per
// maximal subpart, matching Python's errors='replace' on decode).
inline std::vector<uint32_t> utf8_decode(const std::string& s) {
  std::vector<uint32_t> out;
  out.reserve(s.size());
  size_t i = 0;
  const size_t n = s.size();
  auto bad = [&]() { out.push_back(0xFFFDu); ++i; };
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      out.push_back(c);
      ++i;
    } else if (c < 0xC2) {
      bad();  // stray continuation / overlong lead
    } else if (c < 0xE0) {
      if (i + 1 >= n) {
        out.push_back(0xFFFDu);
        break;
      }
      const unsigned char d = static_cast<unsigned char>(s[i + 1]);
      if ((d & 0xC0) != 0x80) {
        bad();
        continue;
      }
      out.push_back(((uint32_t)(c & 0x1F) << 6) | (d & 0x3F));
      i += 2;
    } else if (c < 0xF0) {
      if (i + 2 >= n) {
        out.push_back(0xFFFDu);
        break;
      }
      const unsigned char d = static_cast<unsigned char>(s[i + 1]);
      const unsigned char e = static_cast<unsigned char>(s[i + 2]);
      const uint32_t cp =
          ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(d & 0x3F) << 6) | (e & 0x3F);
      if ((d & 0xC0) != 0x80 || (e & 0xC0) != 0x80 || cp < 0x800 ||
          (cp >= 0xD800 && cp < 0xE000)) {
        bad();  // overlong / surrogate / bad continuation: resync by one
        continue;
      }
      out.push_back(cp);
      i += 3;
    } else if (c < 0xF5) {
      if (i + 3 >= n) {
        out.push_back(0xFFFDu);
        break;
      }
      const unsigned char d = static_cast<unsigned char>(s[i + 1]);
      const unsigned char e = static_cast<unsigned char>(s[i + 2]);
      const unsigned char f = static_cast<unsigned char>(s[i + 3]);
      const uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                          ((uint32_t)(d & 0x3F) << 12) |
                          ((uint32_t)(e & 0x3F) << 6) | (f & 0x3F);
      if ((d & 0xC0) != 0x80 || (e & 0xC0) != 0x80 || (f & 0xC0) != 0x80 ||
          cp < 0x10000 || cp > 0x10FFFF) {
        bad();
        continue;
      }
      out.push_back(cp);
      i += 4;
    } else {
      bad();  // > U+10FFFF lead
    }
  }
  return out;
}

inline void utf8_encode(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back((char)cp);
  } else if (cp < 0x800) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

inline std::string utf8_encode_str(const std::vector<uint32_t>& cps) {
  std::string out;
  out.reserve(cps.size() * 2);
  for (uint32_t cp : cps) utf8_encode(cp, out);
  return out;
}

// ---- table lookup ----------------------------------------------------------
inline bool in_ranges(const Range* tbl, size_t n, uint32_t cp) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (tbl[mid].hi < cp)
      lo = mid + 1;
    else if (tbl[mid].lo > cp)
      hi = mid;
    else
      return true;
  }
  return false;
}

inline bool is_letter(uint32_t cp) {
  return in_ranges(kLetterRanges,
                   sizeof(kLetterRanges) / sizeof(kLetterRanges[0]), cp);
}
inline bool is_number(uint32_t cp) {
  return in_ranges(kNumberRanges,
                   sizeof(kNumberRanges) / sizeof(kNumberRanges[0]), cp);
}
inline bool is_space(uint32_t cp) {
  // kSpaceCps is a sorted codepoint list, not ranges.
  const size_t n = sizeof(kSpaceCps) / sizeof(kSpaceCps[0]);
  size_t lo = 0, hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (kSpaceCps[mid] < cp)
      lo = mid + 1;
    else if (kSpaceCps[mid] > cp)
      hi = mid;
    else
      return true;
  }
  return false;
}
inline bool is_strip(uint32_t cp) {
  // kStripCps is a sorted codepoint list, not ranges.
  const size_t n = sizeof(kStripCps) / sizeof(kStripCps[0]);
  size_t lo = 0, hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (kStripCps[mid] < cp)
      lo = mid + 1;
    else if (kStripCps[mid] > cp)
      hi = mid;
    else
      return true;
  }
  return false;
}
inline uint8_t ccc_of(uint32_t cp) {
  const size_t n = sizeof(kCcc) / sizeof(kCcc[0]);
  size_t lo = 0, hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (kCcc[mid].cp < cp)
      lo = mid + 1;
    else if (kCcc[mid].cp > cp)
      hi = mid;
    else
      return kCcc[mid].ccc;
  }
  return 0;
}

// ---- NFC -------------------------------------------------------------------
// Hangul algorithmic constants (UAX #15 §3.12).
constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161,
                   kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = 588;

// Fully expand one codepoint: Hangul syllable -> Jamo, else table lookup.
inline void decompose_char(uint32_t cp, std::vector<uint32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kLCount * kNCount) {
    const uint32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    const uint32_t t = kTBase + s % kTCount;
    if (t != kTBase) out.push_back(t);
    return;
  }
  const size_t n = sizeof(kDecomp) / sizeof(kDecomp[0]);
  size_t lo = 0, hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (kDecomp[mid].cp < cp)
      lo = mid + 1;
    else if (kDecomp[mid].cp > cp)
      hi = mid;
    else {
      for (uint32_t k = 0; k < kDecomp[mid].len; ++k)
        out.push_back(kDecompPool[kDecomp[mid].off + k]);
      return;
    }
  }
  out.push_back(cp);
}

// Canonical composition of one starter+combiner pair. Hangul is
// algorithmic; the rest is the probed kCompose table.
inline bool compose_pair(uint32_t starter, uint32_t comb,
                         uint32_t& composite) {
  // L + V -> LV.
  if (starter >= kLBase && starter < kLBase + kLCount && comb >= kVBase &&
      comb < kVBase + kVCount) {
    composite =
        kSBase + (starter - kLBase) * kNCount + (comb - kVBase) * kTCount;
    return true;
  }
  // LV + T -> LVT.
  if (starter >= kSBase && starter < kSBase + kLCount * kNCount &&
      (starter - kSBase) % kTCount == 0 && comb > kTBase &&
      comb < kTBase + kTCount) {
    composite = starter + (comb - kTBase);
    return true;
  }
  const uint64_t key = ((uint64_t)starter << 21) | comb;
  const size_t n = sizeof(kCompose) / sizeof(kCompose[0]);
  size_t lo = 0, hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (kCompose[mid].key < key)
      lo = mid + 1;
    else if (kCompose[mid].key > key)
      hi = mid;
    else {
      composite = kCompose[mid].composite;
      return true;
    }
  }
  return false;
}

// In-place NFC (UAX #15): decompose + CCC reorder + compose with blocking.
inline void nfc_inplace(std::vector<uint32_t>& s) {
  std::vector<uint32_t> d;
  d.reserve(s.size() * 2);
  for (uint32_t cp : s) decompose_char(cp, d);
  // Canonical ordering: bubble each non-starter left past strictly
  // greater CCCs (starters never move).
  for (size_t i = 1; i < d.size(); ++i) {
    if (ccc_of(d[i]) == 0) continue;
    size_t j = i;
    while (j > 0 && ccc_of(d[j - 1]) > ccc_of(d[j])) {
      const uint32_t t = d[j - 1];
      d[j - 1] = d[j];
      d[j] = t;
      --j;
    }
  }
  // Composition with blocking.
  std::vector<uint32_t> out;
  out.reserve(d.size());
  for (uint32_t c : d) {
    if (ccc_of(c) == 0) {
      // Starter: composes only with the immediately preceding character
      // (anything earlier is blocked by definition), and only if that
      // character is itself a starter. compose_pair is exact: its table
      // holds only probed-composable pairs, and Hangul pairs are
      // structural, so a blind lookup cannot misfire.
      uint32_t comp = 0;
      if (!out.empty() && ccc_of(out.back()) == 0 &&
          compose_pair(out.back(), c, comp)) {
        out.back() = comp;
        continue;
      }
      out.push_back(c);
      continue;
    }
    // Non-starter: find the last starter; blocked if any intervening
    // character has ccc 0 or ccc >= ccc(c).
    long ls = (long)out.size() - 1;
    while (ls >= 0 && ccc_of(out[(size_t)ls]) != 0) --ls;
    if (ls < 0) {
      out.push_back(c);  // no starter to compose with
      continue;
    }
    bool blocked = false;
    for (size_t m = (size_t)ls + 1; m < out.size(); ++m)
      if (ccc_of(out[m]) == 0 || ccc_of(out[m]) >= ccc_of(c)) {
        blocked = true;
        break;
      }
    uint32_t comp = 0;
    if (!blocked && compose_pair(out[(size_t)ls], c, comp)) {
      out[(size_t)ls] = comp;
      continue;
    }
    out.push_back(c);
  }
  s.swap(out);
}

// Python str.strip() over the exact CPython whitespace set.
inline std::vector<uint32_t> strip_ws(const std::vector<uint32_t>& s) {
  size_t b = 0, e = s.size();
  while (b < e && is_strip(s[b])) ++b;
  while (e > b && is_strip(s[e - 1])) --e;
  return std::vector<uint32_t>(s.begin() + (long)b, s.begin() + (long)e);
}

// ---- pre-tokenizer -----------------------------------------------------------
// Ordered manual scan of the Qwen2 Split regex (behavior Isolated: every
// match is its own pre-token; the 7 alternatives cover all codepoints):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   | ' '?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
// Case-insensitive ASCII literal match at pos; returns length or 0.
inline size_t match_ci(const std::vector<uint32_t>& s, size_t pos,
                       const char* lit) {
  size_t k = 0;
  while (lit[k]) {
    if (pos + k >= s.size()) return 0;
    uint32_t c = s[pos + k];
    if (c >= 'A' && c <= 'Z') c += 32;
    if (c != (uint32_t)(unsigned char)lit[k]) return 0;
    ++k;
  }
  return k;
}

inline std::vector<std::vector<uint32_t>> pretokenize(
    const std::vector<uint32_t>& s) {
  static const char* kContractions[] = {"'s", "'t", "'re",
                                        "'ve", "'m", "'ll", "'d"};
  std::vector<std::vector<uint32_t>> toks;
  size_t i = 0;
  const size_t n = s.size();
  auto is_nl = [](uint32_t c) { return c == '\r' || c == '\n'; };
  while (i < n) {
    const uint32_t c = s[i];
    // 1. contractions (case-insensitive).
    bool done = false;
    for (const char* lit : kContractions) {
      const size_t m = match_ci(s, i, lit);
      if (m) {
        toks.emplace_back(s.begin() + (long)i, s.begin() + (long)(i + m));
        i += m;
        done = true;
        break;
      }
    }
    if (done) continue;
    // 2. [^\r\nL N]? L+. The optional prefix is any char but \r,\n,L,N
    // (spaces included); it only participates when a letter follows.
    if (is_letter(c) || (!is_nl(c) && !is_letter(c) && !is_number(c) &&
                         i + 1 < n && is_letter(s[i + 1]))) {
      size_t j = i;
      if (!is_letter(s[j])) ++j;  // the optional prefix char
      while (j < n && is_letter(s[j])) ++j;
      toks.emplace_back(s.begin() + (long)i, s.begin() + (long)j);
      i = j;
      continue;
    }
    // 3. single N.
    if (is_number(c)) {
      toks.emplace_back(s.begin() + (long)i, s.begin() + (long)(i + 1));
      ++i;
      continue;
    }
    // 4. optional-space + symbol run + trailing newlines. The space prefix
    // participates only if at least one symbol follows (regex
    // backtracking); a lone space falls through to whitespace below.
    if (c == ' ' || (!is_space(c) && !is_letter(c) && !is_number(c))) {
      size_t j = i;
      if (s[j] == ' ') ++j;
      size_t k = j;
      while (k < n && !is_space(s[k]) && !is_letter(s[k]) && !is_number(s[k]))
        ++k;
      if (k > j) {
        while (k < n && is_nl(s[k])) ++k;
        toks.emplace_back(s.begin() + (long)i, s.begin() + (long)k);
        i = k;
        continue;
      }
      if (c != ' ') {
        // Unreachable: c itself is a symbol, so k > j always.
        throw std::logic_error("QwenTokenizer::pretokenize: alt4");
      }
      // Lone space: fall through to the whitespace rules.
    }
    // 5/6/7. whitespace runs. Ordered semantics matter:
    //   alt 5 (\s*[\r\n]+) wins when the run contains a newline and spans
    //     [i, last_newline + 1) by greedy backtracking;
    //   alt 6 (\s+(?!\S)) takes a newline-free run whole at end-of-input,
    //     else all but the last char (the final space reattaches to the
    //     next pre-token, e.g. GPT-2 "Ġword"); length-1 runs fall to
    //   alt 7 (\s+), which takes the single char.
    if (is_space(c)) {
      size_t j = i;
      while (j < n && is_space(s[j])) ++j;
      size_t last_nl = n;
      for (size_t k = i; k < j; ++k)
        if (is_nl(s[k])) last_nl = k;
      if (last_nl < n) {
        toks.emplace_back(s.begin() + (long)i,
                          s.begin() + (long)(last_nl + 1));
        i = last_nl + 1;
        continue;
      }
      if (j >= n || j - i == 1) {
        toks.emplace_back(s.begin() + (long)i, s.begin() + (long)j);
        i = j;
        continue;
      }
      toks.emplace_back(s.begin() + (long)i, s.begin() + (long)(j - 1));
      i = j - 1;
      continue;
    }
    // Unreachable: every codepoint is L, N, space, or other.
    throw std::logic_error("QwenTokenizer::pretokenize: uncovered char");
  }
  return toks;
}

// ---- byte-level BPE ------------------------------------------------------------
// GPT-2 bytes_to_unicode, computed (no table): printable/latin-1 bytes map
// to themselves, the rest to U+0100+.
inline uint32_t byte_to_unicode(unsigned b) {
  if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
      (b >= 0xAE && b <= 0xFF))
    return b;
  unsigned n = 0;
  for (unsigned k = 0; k < 256; ++k) {
    const bool plain = (k >= 0x21 && k <= 0x7E) || (k >= 0xA1 && k <= 0xAC) ||
                       (k >= 0xAE && k <= 0xFF);
    if (!plain) {
      if (k == b) return 256 + n;
      ++n;
    }
  }
  throw std::logic_error("QwenTokenizer::byte_to_unicode: bad byte");
}

// Standard rank-merge BPE over UTF-8 symbol strings.
inline std::vector<std::string> bpe_merge(
    const std::vector<std::string>& pieces,
    const std::unordered_map<std::string, int>& ranks) {
  std::vector<std::string> cur = pieces;
  if (cur.size() < 2) return cur;
  for (;;) {
    int best_rank = -1;
    size_t best = 0;
    for (size_t i = 0; i + 1 < cur.size(); ++i) {
      const auto it = ranks.find(cur[i] + " " + cur[i + 1]);
      if (it != ranks.end() && (best_rank < 0 || it->second < best_rank)) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best_rank < 0) return cur;
    cur[best] += cur[best + 1];
    cur.erase(cur.begin() + (long)(best + 1));
  }
}

// Encode one pre-token (codepoints) to vocab ids.
inline void bpe_encode_pretoken(
    const std::vector<uint32_t>& pre,
    const std::unordered_map<std::string, int>& vocab,
    const std::unordered_map<std::string, int>& ranks,
    std::vector<int>& ids) {
  const std::string bytes = utf8_encode_str(pre);
  std::vector<std::string> pieces;
  pieces.reserve(bytes.size());
  for (unsigned char b : bytes) {
    std::string sym;
    utf8_encode(byte_to_unicode(b), sym);
    pieces.push_back(sym);
  }
  for (const std::string& sym : bpe_merge(pieces, ranks)) {
    const auto it = vocab.find(sym);
    if (it == vocab.end())
      throw std::runtime_error(
          "QwenTokenizer::encode: BPE symbol missing from vocab: '" + sym +
          "'");
    ids.push_back(it->second);
  }
}

// ---- tokenizer.json loading --------------------------------------------------
// Targeted single-pass scanner: extracts model.vocab (string->id),
// model.merges (ranked "l r" strings) and added_tokens (surface->id)
// without building a DOM for the 20MB file.
struct JsonCursor {
  const char* p;
  const char* end;
  const char* base;
  [[noreturn]] void fail(const std::string& m) {
    std::ostringstream os;
    os << "QwenTokenizer::load: " << m << " (offset " << (p - base) << ")";
    throw std::runtime_error(os.str());
  }
  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
      ++p;
  }
  char peek() {
    if (p >= end) fail("unexpected end of file");
    return *p;
  }
  // Parse a JSON string into UTF-8 (handles all escapes incl. \u pairs).
  std::string parse_string() {
    ws();
    if (p >= end || *p != '"') fail("expected string");
    ++p;
    std::string out;
    while (true) {
      if (p >= end) fail("unterminated string");
      const char c = *p++;
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (p >= end) fail("bad escape");
      const char e = *p++;
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (end - p < 4) fail("bad \\u escape");
          unsigned cp = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = *p++;
            cp <<= 4;
            if (h >= '0' && h <= '9')
              cp += (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f')
              cp += (unsigned)(h - 'a') + 10;
            else if (h >= 'A' && h <= 'F')
              cp += (unsigned)(h - 'A') + 10;
            else
              fail("bad \\u escape");
          }
          // Surrogate pair.
          if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' &&
              p[1] == 'u') {
            unsigned lo = 0;
            bool ok = true;
            for (int i = 2; i < 6; ++i) {
              const char h = p[i];
              lo <<= 4;
              if (h >= '0' && h <= '9')
                lo += (unsigned)(h - '0');
              else if (h >= 'a' && h <= 'f')
                lo += (unsigned)(h - 'a') + 10;
              else if (h >= 'A' && h <= 'F')
                lo += (unsigned)(h - 'A') + 10;
              else
                ok = false;
            }
            if (ok && lo >= 0xDC00 && lo < 0xE000) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              p += 6;
            }
          }
          if (cp >= 0xD800 && cp < 0xE000) cp = 0xFFFDu;  // lone surrogate
          utf8_encode(cp, out);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }
  long long parse_int() {
    ws();
    const char* b = p;
    if (p < end && (*p == '-' || *p == '+')) ++p;
    if (p >= end || *p < '0' || *p > '9') fail("expected integer");
    while (p < end && *p >= '0' && *p <= '9') ++p;
    // Vocab ids are plain integers; reject anything fancier.
    if (p < end && (*p == '.' || *p == 'e' || *p == 'E')) fail("bad integer");
    return std::stoll(std::string(b, p));
  }
  // Skip one JSON value (string/object/array/number/literal).
  void skip_value() {
    ws();
    const char c = peek();
    if (c == '"') {
      parse_string();
    } else if (c == '{') {
      ++p;
      ws();
      if (peek() == '}') {
        ++p;
        return;
      }
      while (true) {
        ws();
        parse_string();
        ws();
        if (peek() != ':') fail("expected ':'");
        ++p;
        skip_value();
        ws();
        const char d = peek();
        if (d == ',') {
          ++p;
          continue;
        }
        if (d == '}') {
          ++p;
          break;
        }
        fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      ++p;
      ws();
      if (peek() == ']') {
        ++p;
        return;
      }
      while (true) {
        skip_value();
        ws();
        const char d = peek();
        if (d == ',') {
          ++p;
          continue;
        }
        if (d == ']') {
          ++p;
          break;
        }
        fail("expected ',' or ']'");
      }
    } else {
      // number or literal: consume the atom.
      while (p < end && *p != ',' && *p != ']' && *p != '}' && *p != ' ' &&
             *p != '\t' && *p != '\n' && *p != '\r')
        ++p;
    }
  }
};

}  // namespace Impl

// Qwen2-style byte-level BPE tokenizer loaded from tokenizer.json.
class Tokenizer {
 public:
  Tokenizer() = default;

  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("QwenTokenizer::load: cannot open '" + path +
                               "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string js = ss.str();
    Impl::JsonCursor cur{js.data(), js.data() + js.size(), js.data()};
    cur.ws();
    if (cur.peek() != '{') cur.fail("top level must be an object");
    ++cur.p;
    bool have_vocab = false;
    while (true) {
      cur.ws();
      if (cur.peek() == '}') {
        ++cur.p;
        break;
      }
      std::string key = cur.parse_string();
      cur.ws();
      if (cur.peek() != ':') cur.fail("expected ':'");
      ++cur.p;
      if (key == "model") {
        parse_model(cur);
        have_vocab = true;
      } else if (key == "added_tokens") {
        parse_added(cur);
      } else {
        cur.skip_value();
      }
      cur.ws();
      const char d = cur.peek();
      if (d == ',') {
        ++cur.p;
        continue;
      }
      if (d == '}') {
        ++cur.p;
        break;
      }
      cur.fail("expected ',' or '}'");
    }
    if (!have_vocab || vocab_.empty())
      throw std::runtime_error("QwenTokenizer::load: no model.vocab in '" +
                               path + "'");
    if (ranks_.empty())
      throw std::runtime_error("QwenTokenizer::load: no model.merges in '" +
                               path + "'");
    if (specials_.empty())
      throw std::runtime_error("QwenTokenizer::load: no added_tokens in '" +
                               path + "'");
    // Longest-match order for added-token splitting.
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) {
                return a.first.size() > b.first.size();
              });
    vocab_size_ = (size_t)(top_id_ + 1);
    max_special_len_ = 0;
    for (const auto& kv : specials_)
      max_special_len_ = std::max(max_special_len_, kv.first.size());
  }

  size_t vocab_size() const { return vocab_size_; }
  // Reference single-turn formatting:
  //   <|im_start|>user\n{strip(prompt)}<|im_end|>\n
  //   <|im_start|>assistant\n[<think>\n | <think>\n\n</think>\n\n]
  std::string apply_chat_template(const std::string& prompt,
                                  bool thinking = true) const {
    require_loaded("apply_chat_template");
    const std::string content =
        Impl::utf8_encode_str(Impl::strip_ws(Impl::utf8_decode(prompt)));
    std::string out = "<|im_start|>user\n" + content +
                      "<|im_end|>\n<|im_start|>assistant\n";
    out += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return out;
  }

  // Full pipeline: template -> added-token split -> NFC -> pre-tokenize ->
  // byte-level BPE -> ids, right-truncated to max_length.
  std::vector<int> encode(const std::string& prompt, bool thinking,
                          size_t max_length) const {
    require_loaded("encode");
    if (max_length == 0)
      throw std::invalid_argument("QwenTokenizer::encode: max_length == 0");
    const std::string fmt = apply_chat_template(prompt, thinking);
    std::vector<int> ids;
    // Left-to-right longest-match split on added-token surfaces.
    size_t i = 0;
    const size_t n = fmt.size();
    size_t seg_start = 0;
    auto flush_gap = [&](size_t end) {
      if (end == seg_start) return;
      const std::string gap = fmt.substr(seg_start, end - seg_start);
      std::vector<uint32_t> cps = Impl::utf8_decode(gap);
      Impl::nfc_inplace(cps);
      for (const auto& pre : Impl::pretokenize(cps))
        Impl::bpe_encode_pretoken(pre, vocab_, ranks_, ids);
    };
    while (i < n) {
      size_t hit_len = 0;
      int hit_id = -1;
      // Bounded lookahead: no special is longer than max_special_len_.
      for (const auto& kv : specials_) {
        const size_t L = kv.first.size();
        if (L > hit_len && i + L <= n && fmt.compare(i, L, kv.first) == 0) {
          hit_len = L;
          hit_id = kv.second;
          if (L == max_special_len_) break;
        }
      }
      if (hit_len) {
        flush_gap(i);
        ids.push_back(hit_id);
        i += hit_len;
        seg_start = i;
      } else {
        ++i;
      }
    }
    flush_gap(n);
    if (ids.size() > max_length) ids.resize(max_length);
    return ids;
  }

 private:
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> ranks_;
  std::vector<std::pair<std::string, int>> specials_;
  size_t max_special_len_ = 0;
  long long top_id_ = -1;
  size_t vocab_size_ = 0;

  void require_loaded(const char* what) const {
    if (vocab_.empty())
      throw std::logic_error(std::string("QwenTokenizer::") + what +
                             ": no tokenizer loaded (call load() first)");
  }

  void parse_model(Impl::JsonCursor& cur) {
    cur.ws();
    if (cur.peek() != '{') cur.fail("model must be an object");
    ++cur.p;
    while (true) {
      cur.ws();
      if (cur.peek() == '}') {
        ++cur.p;
        break;
      }
      std::string key = cur.parse_string();
      cur.ws();
      if (cur.peek() != ':') cur.fail("expected ':'");
      ++cur.p;
      if (key == "vocab") {
        parse_vocab(cur);
      } else if (key == "merges") {
        parse_merges(cur);
      } else {
        cur.skip_value();
      }
      cur.ws();
      const char d = cur.peek();
      if (d == ',') {
        ++cur.p;
        continue;
      }
      if (d == '}') {
        ++cur.p;
        break;
      }
      cur.fail("expected ',' or '}'");
    }
  }

  void parse_vocab(Impl::JsonCursor& cur) {
    cur.ws();
    if (cur.peek() != '{') cur.fail("vocab must be an object");
    ++cur.p;
    vocab_.reserve(250000);
    while (true) {
      cur.ws();
      if (cur.peek() == '}') {
        ++cur.p;
        break;
      }
      std::string tok = cur.parse_string();
      cur.ws();
      if (cur.peek() != ':') cur.fail("expected ':'");
      ++cur.p;
      const long long id = cur.parse_int();
      if (id < 0 || id > 10000000) cur.fail("vocab id out of range");
      vocab_[tok] = (int)id;
      top_id_ = std::max(top_id_, id);
      cur.ws();
      const char d = cur.peek();
      if (d == ',') {
        ++cur.p;
        continue;
      }
      if (d == '}') {
        ++cur.p;
        break;
      }
      cur.fail("expected ',' or '}'");
    }
  }

  void parse_merges(Impl::JsonCursor& cur) {
    cur.ws();
    if (cur.peek() != '[') cur.fail("merges must be an array");
    ++cur.p;
    int rank = 0;
    while (true) {
      cur.ws();
      if (cur.peek() == ']') {
        ++cur.p;
        break;
      }
      // tokenizer.json stores merges as 2-element arrays ["l", "r"].
      if (cur.peek() != '[') cur.fail("merge entry must be an array");
      ++cur.p;
      cur.ws();
      const std::string left = cur.parse_string();
      cur.ws();
      if (cur.peek() != ',') cur.fail("expected ',' in merge entry");
      ++cur.p;
      const std::string right = cur.parse_string();
      cur.ws();
      if (cur.peek() != ']') cur.fail("expected ']' in merge entry");
      ++cur.p;
      // BPE symbols never contain a raw space (0x20 maps to U+0120), so
      // "l r" splits back into exactly two parts.
      if (left.empty() || right.empty() || left.find(' ') != std::string::npos ||
          right.find(' ') != std::string::npos)
        cur.fail("malformed merge entry");
      ranks_[left + " " + right] = rank++;
      cur.ws();
      const char d = cur.peek();
      if (d == ',') {
        ++cur.p;
        continue;
      }
      if (d == ']') {
        ++cur.p;
        break;
      }
      cur.fail("expected ',' or ']'");
    }
  }

  void parse_added(Impl::JsonCursor& cur) {
    cur.ws();
    if (cur.peek() != '[') cur.fail("added_tokens must be an array");
    ++cur.p;
    while (true) {
      cur.ws();
      if (cur.peek() == ']') {
        ++cur.p;
        break;
      }
      if (cur.peek() != '{') cur.fail("added token must be an object");
      ++cur.p;
      std::string content;
      long long id = -1;
      while (true) {
        cur.ws();
        if (cur.peek() == '}') {
          ++cur.p;
          break;
        }
        std::string key = cur.parse_string();
        cur.ws();
        if (cur.peek() != ':') cur.fail("expected ':'");
        ++cur.p;
        if (key == "content") {
          content = cur.parse_string();
        } else if (key == "id") {
          id = cur.parse_int();
        } else {
          cur.skip_value();
        }
        cur.ws();
        const char d = cur.peek();
        if (d == ',') {
          ++cur.p;
          continue;
        }
        if (d == '}') {
          ++cur.p;
          break;
        }
        cur.fail("expected ',' or '}'");
      }
      if (content.empty() || id < 0) cur.fail("bad added token entry");
      specials_.emplace_back(content, (int)id);
      top_id_ = std::max(top_id_, id);
      cur.ws();
      const char d = cur.peek();
      if (d == ',') {
        ++cur.p;
        continue;
      }
      if (d == ']') {
        ++cur.p;
        break;
      }
      cur.fail("expected ',' or ']'");
    }
  }
};

}  // namespace QwenTokenizer
}  // namespace KokkosDNN

#endif
