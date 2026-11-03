// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CHATSTOP_HPP
#define KOKKOSDNN_CHATSTOP_HPP

// Shared chat/stop-string helpers for text tokenizers (models/common).
//
// Consolidates the byte-identical functions previously duplicated in
//   examples/llama/KokkosDNN_Tokenizer.{hpp,cpp},
//   examples/smollm2/KokkosDNN_SmolLM2Tokenizer.{hpp,cpp},
//   models/lfm2_5/KokkosDNN_LFM25Tokenizer.{hpp,cpp},
//   models/glimmer/KokkosDNN_GlimmerTokenizer.{hpp,cpp},
//   examples/gemma4/KokkosDNN_Gemma4Tokenizer.{hpp,cpp}:
// find_stop_string, truncate_at_stop_strings, ensure_bos, plus generic
// make_chat_messages / encode_rendered_chat templates.
//
// Per-model renderers (render_chat_template, render_chatml_template,
// render_lfm25_chatml, ...) stay per-namespace: their markup differs.
// Per-namespace ChatMessage types also stay (LFM25/Glimmer carry
// thinking/tool-call fields); the generic make_chat_messages below works
// for any aggregate whose first two fields are (role, content) and whose
// trailing fields are default-constructible, so each namespace keeps a
// thin wrapper preserving its public API.
//
// No Kokkos dependency; header-only.

#include <string>
#include <utility>
#include <vector>

namespace KokkosDNN {
namespace Common {
namespace Chat {

// Stop-strings (checked on decoded text): earliest occurrence wins;
// `matched` receives the winning string (or "" when none).
inline size_t find_stop_string(const std::string& text,
                               const std::vector<std::string>& stop_strings,
                               std::string* matched = nullptr) {
  size_t best = std::string::npos;
  const std::string* which = nullptr;
  for (const auto& s : stop_strings) {
    if (s.empty()) continue;
    size_t p = text.find(s);
    if (p != std::string::npos && (best == std::string::npos || p < best)) {
      best = p;
      which = &s;
    }
  }
  if (matched) *matched = (which ? *which : std::string());
  return best;
}

inline std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings) {
  size_t p = find_stop_string(text, stop_strings);
  if (p == std::string::npos) return text;
  return text.substr(0, p);
}

// BOS handling: collapse any run of leading `bos_id` to exactly one when
// want_bos (or strip all leading copies when !want_bos).
inline std::vector<int> ensure_bos(const std::vector<int>& ids, int bos_id,
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

// Build the standard single-turn messages for a raw prompt: optional
// system message first, then one `role` (default "user") turn. Message must
// expose mutable `role`/`content` string fields (all per-namespace
// ChatMessage types do); trailing fields (thinking/tool calls/...) stay
// default-constructed. Field assignment (not aggregate init) keeps
// -Wmissing-field-initializers clean under warnings-as-errors.
template <typename Message>
inline std::vector<Message> make_chat_messages(const std::string& prompt,
                                              const std::string& system = "",
                                              const std::string& role = "user") {
  std::vector<Message> msgs;
  if (!system.empty()) {
    Message m{};
    m.role = "system";
    m.content = system;
    msgs.push_back(std::move(m));
  }
  {
    Message m{};
    m.role = role.empty() ? "user" : role;
    m.content = prompt;
    msgs.push_back(std::move(m));
  }
  return msgs;
}

// Encode an already-rendered chat string with exactly-one-BOS semantics:
// encodes with add_bos=false (literal specials flow through the
// special-token path), then applies ensure_bos(). Tokenizer must expose
// encode(string, bool) and bos().
template <typename Tokenizer>
inline std::vector<int> encode_rendered_chat(const Tokenizer& tok,
                                            const std::string& rendered,
                                            bool add_bos = true) {
  std::vector<int> body = tok.encode(rendered, false);
  return ensure_bos(body, tok.bos(), add_bos);
}

}  // namespace Chat
}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_CHATSTOP_HPP
