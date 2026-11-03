// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_BPECORE_HPP
#define KOKKOSDNN_BPECORE_HPP

// Byte-level BPE core for GPT-2/tiktoken-style tokenizers (models/common).
//
// Consolidates the byte-identical blocks previously duplicated in
//   examples/llama/KokkosDNN_Tokenizer.cpp,
//   examples/smollm2/KokkosDNN_SmolLM2Tokenizer.cpp,
//   models/lfm2_5/KokkosDNN_LFM25Tokenizer.cpp,
//   models/glimmer/KokkosDNN_GlimmerTokenizer.cpp:
// GPT-2 bytes_to_unicode table, its inverse, and the greedy lowest-rank
// BPE merge loop over one pre-token piece.
//
// SentencePiece-style tokenizers (examples/gemma4, examples/sana text
// encoder) keep their own byte-fallback/unk paths and are intentionally
// out of scope: only the shared byte-level core lives here.
//
// No Kokkos dependency; header-only.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "KokkosDNN_JsonLite.hpp"

namespace KokkosDNN {
namespace Common {
namespace Bpe {

struct ByteMaps {
  std::string enc[256];  // byte -> UTF-8 of mapped char
  std::unordered_map<std::string, unsigned char> dec;  // UTF-8 -> byte
  ByteMaps() {
    // bs = printable ASCII + Latin-1 supplement ranges, then the rest.
    bool in_bs[256] = {};
    for (int b = 0x21; b <= 0x7E; ++b) in_bs[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) in_bs[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) in_bs[b] = true;
    std::vector<int> bs, cs;
    for (int b = 0; b < 256; ++b)
      if (in_bs[b]) {
        bs.push_back(b);
        cs.push_back(b);
      }
    int m = 0;
    for (int b = 0; b < 256; ++b)
      if (!in_bs[b]) {
        bs.push_back(b);
        cs.push_back(0x100 + m++);
      }
    for (size_t k = 0; k < bs.size(); ++k) {
      JsonLite::utf8_encode_cp((uint32_t)cs[k], enc[bs[k]]);
      dec[enc[bs[k]]] = (unsigned char)bs[k];
    }
  }
};

inline const ByteMaps& byte_maps() {
  static ByteMaps m;
  return m;
}

inline std::string bytes_to_unicode(const std::string& bytes) {
  const ByteMaps& m = byte_maps();
  std::string out;
  for (unsigned char b : bytes) out += m.enc[b];
  return out;
}

// Inverse: mapped unicode chars -> original bytes. Chars outside the table
// (should not occur) pass through as UTF-8.
inline std::string unicode_to_bytes(const std::string& s) {
  const ByteMaps& m = byte_maps();
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if (c >= 0x80) {
      if ((c >> 5) == 0x6)
        len = 2;
      else if ((c >> 4) == 0xE)
        len = 3;
      else if ((c >> 3) == 0x1E)
        len = 4;
    }
    std::string u = s.substr(i, len);
    auto it = m.dec.find(u);
    if (it != m.dec.end()) {
      out += (char)it->second;
    } else {
      out += u;  // lenient
    }
    i += len;
  }
  return out;
}

// Split one byte-mapped piece into single unicode chars (UTF-8 aware).
inline void bpe_split_chars(const std::string& piece,
                            std::vector<std::string>& parts) {
  parts.clear();
  for (size_t i = 0; i < piece.size();) {
    unsigned char c = piece[i];
    size_t len = 1;
    if (c >= 0x80) {
      if ((c >> 5) == 0x6)
        len = 2;
      else if ((c >> 4) == 0xE)
        len = 3;
      else if ((c >> 3) == 0x1E)
        len = 4;
    }
    parts.push_back(piece.substr(i, len));
    i += len;
  }
}

// Greedily merge the lowest-rank adjacent pair until no pair merges.
inline void bpe_merge_greedy(
    std::vector<std::string>& parts,
    const std::unordered_map<std::string, int>& merge_rank) {
  while (parts.size() > 1) {
    int best_rank = -1;
    size_t best = 0;
    for (size_t k = 0; k + 1 < parts.size(); ++k) {
      auto it = merge_rank.find(parts[k] + " " + parts[k + 1]);
      if (it != merge_rank.end() &&
          (best_rank < 0 || it->second < best_rank)) {
        best_rank = it->second;
        best = k;
      }
    }
    if (best_rank < 0) break;
    parts[best] += parts[best + 1];
    parts.erase(parts.begin() + best + 1);
  }
}

// Full byte-level BPE over one byte-mapped pre-token piece; appends
// resulting ids. Throws on tokens missing from vocab (same diagnostic as
// the historical per-tokenizer copies).
inline void bpe_piece_bytelevel(
    const std::string& piece,
    const std::unordered_map<std::string, int>& token_to_id,
    const std::unordered_map<std::string, int>& merge_rank,
    std::vector<int>& out) {
  if (piece.empty()) return;
  std::vector<std::string> parts;
  bpe_split_chars(piece, parts);
  bpe_merge_greedy(parts, merge_rank);
  for (auto& s : parts) {
    auto it = token_to_id.find(s);
    if (it == token_to_id.end())
      throw std::runtime_error("tokenizer: BPE token missing from vocab: " +
                               unicode_to_bytes(s));
    out.push_back(it->second);
  }
}

}  // namespace Bpe
}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_BPECORE_HPP
