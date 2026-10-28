// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GLIMMER_TOKENIZER_HPP
#define KOKKOSDNN_GLIMMER_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KokkosDNN {
namespace Glimmer {

// Hugging Face `tokenizer.json` loader for Muse Glimmer 30B (tiktoken-style
// BPE, 200000 base + 2048 added = 202048).
//
// Faithful to the `tokenizers` Rust library pipeline used by the real
// `darkc0de/Muse-Glimmer-30B-heretic` tokenizer (verified against the file:
// normalizer null, pre_tokenizer Sequence[Split(o200k pattern, Isolated),
// ByteLevel(add_prefix_space=false)], BPE model with 200000-entry vocab and
// [a, b]-pair merges, ByteLevel decoder):
//   1. literal special/added tokens are split out first (longest match,
//      never broken up by BPE);
//   2. remaining text is pre-tokenized with the Split regex
//      [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*
//        [\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
//      [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+
//        [\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
//      \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//      (fine Unicode classes, leftmost-first alternation with greedy
//      backtracking, mirroring the `regex` crate the Rust library uses);
//   3. each piece is byte-mapped with the GPT-2 bytes_to_unicode table;
//   4. BPE merges apply per piece (lowest rank first);
//   5. decode inverts the byte mapping and skips special tokens unless
//      asked otherwise.
//
// Special tokens: BOS <|begin_of_text|> = 200000,
// EOS <|end_of_text|> = 200001, EOM <|eom|> = 200007,
// EOT <|eot|> = 200008, PAD <|finetune_right_pad|> = 200018 (from
// tokenizer_config.json / generation_config.json; config.json itself carries
// pad_token_id null), plus reserved tokens in 200002..202047. Ids are
// resolved by content from `added_tokens`, falling back to the well-known
// values above when absent. A sibling tokenizer_config.json bos/eos string
// override is applied best-effort (never throws) when the tokenizer path
// contains a directory, mirroring the Gemma4 loader.
class Tokenizer {
 public:
  Tokenizer() = default;
  // Throws on missing/corrupt file.
  explicit Tokenizer(const std::string& tokenizer_json_path) { load(path_ = tokenizer_json_path); }
  void load(const std::string& tokenizer_json_path);

  bool loaded() const { return !id_to_token_.empty(); }
  int vocab_size() const { return vocab_size_; }
  // Explicit BOS/EOS overrides (e.g. when no tokenizer_config.json is
  // available to read them from).
  void set_bos(int id) { bos_id_ = id; }
  void set_eos(int id) { eos_id_ = id; }

  // Encode text -> ids. add_bos prepends <|begin_of_text|> (default true,
  // matching the file's TemplateProcessing post-processor).
  std::vector<int> encode(const std::string& text, bool add_bos = true) const;
  // skip_special_tokens=true (HF decode default) drops added/special ids.
  std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

  int bos() const { return bos_id_; }
  int eos() const { return eos_id_; }
  int eom() const { return eom_id_; }
  int eot() const { return eot_id_; }
  int pad() const { return pad_id_; }
  bool is_special(int id) const { return special_ids_.count(id) != 0; }
 private:
  struct AddedToken {
    int id = -1;
    std::string content;
    bool special = true;
  };

  std::string path_;
  int vocab_size_ = 202048;
  int bos_id_ = 200000, eos_id_ = 200001, eom_id_ = 200007, eot_id_ = 200008;
  int pad_id_ = 200018;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, int> merge_rank_;
  std::vector<AddedToken> added_tokens_;  // longest content first
  std::unordered_set<int> special_ids_;
  std::unordered_map<int, std::string> special_content_;

  // BPE over one byte-mapped pre-token piece; appends resulting ids.
  void bpe_piece(const std::string& piece, std::vector<int>& out) const;
};

// ---------------------------------------------------------------------------
// Muse Glimmer chat template (ASSUMPTION: no chat_template ships in the
// upstream tokenizer_config.json, so the driver cannot read one; the shape
// below is inferred from the tokenizer_config.json `response_template`
// evidence: `start_anchor` "<|start|>assistant", message opens
// "to=user<|message|>" / "to=self<|message|>", closes "<|eot|>" / "<|eom|>").
//
// Renders one "<|start|>{role}<|message|>{content}<|eot|>" block per message
// (role verbatim: "system" | "user" | "assistant"), preceded by
// <|begin_of_text|>; with add_generation_prompt=true an open
// "<|start|>assistant<|message|>" block (no <|eot|>) is appended so the model
// continues as assistant. The `to=<channel>` routing prefixes seen in the
// response_template (assistant reasoning vs user-facing spans) are NOT
// emitted: the user-turn framing is unconfirmed upstream, so the template
// keeps the minimal uniform framing and documents the gap. Revisit against
// the upstream chat template when one is published.
// ---------------------------------------------------------------------------
// A single tool invocation. `arguments` is a JSON object string (OpenAI
// `function.arguments` wire form). Glimmer has no native call markers, so
// parsed completions never produce these; the type exists so generic
// server code compiles against Glimmer unchanged.
struct ToolCall {
  std::string id;         // OpenAI tool_call id ("" when unknown/unparsed)
  std::string name;       // function name
  std::string arguments;  // JSON object string ("" or "{}" when none)
};

struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant"
  std::string content;
  // Agentic-history interchange (kept for API parity with the LFM2.5 chat
  // helpers so server code can stay generic; Glimmer publishes no native
  // tool-call markers, so the _with_tools renderer ignores per-message
  // tool_calls and the tool parser below always returns none).
  // Defaulted so the existing {"role", "content"} aggregate init keeps
  // working.
  std::string thinking;  // unused by the renderer (kept for parity)
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
};

std::string render_chat_template(const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt = true);

// Build the standard single-turn messages for a raw prompt: optional
// system message first, then one `role` (default "user") turn.
std::vector<ChatMessage> make_chat_messages(const std::string& prompt,
                                            const std::string& system = "",
                                            const std::string& role = "user");

// BOS handling: collapse any run of leading `bos_id` to exactly one when
// want_bos (or strip all leading copies when !want_bos). Guarantees the
// "exactly one BOS" invariant even when a chat-rendered string already
// starts with <|begin_of_text|> and encode(add_bos=true) would prepend a
// second copy, or when raw text itself starts with the BOS literal.
std::vector<int> ensure_bos(const std::vector<int>& ids, int bos_id,
                            bool want_bos);

// Encode an already-rendered chat string with exactly-one-BOS semantics:
// encodes with add_bos=false (the literal <|begin_of_text|> in `rendered`
// flows through the special-token path), then applies ensure_bos().
std::vector<int> encode_rendered_chat(const Tokenizer& tok,
                                      const std::string& rendered,
                                      bool add_bos = true);

// Render + encode in one step (same BOS semantics as encode_rendered_chat).
std::vector<int> encode_chat(const Tokenizer& tok,
                             const std::vector<ChatMessage>& messages,
                             bool add_generation_prompt = true,
                             bool add_bos = true);

// ---------------------------------------------------------------------------
// Tool-aware entry points (API parity with LFM25's
// render_lfm25_chatml_with_tools / encode_chatml_with_tools).
//
// Glimmer publishes no native tool-call markers (no <|tool_call_start|>
// equivalent) and no tool catalogue convention, so:
//   - tools_json (compact-serialized OpenAI `tools` array, "" = none) is
//     appended to the system block as "\nList of tools: <json>" (a system
//     block is injected when no system message exists), exactly like the
//     LFM25 renderer;
//   - system_suffix ("" = none): extra instruction sentence(s) appended to
//     the system block (tool_choice required/named directives, JSON-mode
//     bias);
//   - per-message tool_calls / tool_call_id / thinking render as nothing
//     (only `content` is emitted); a "tool"-role message renders under that
//     role verbatim.
// Tool results therefore only round-trip as plain text; structured
// tool_calls never parse back out (see parse_tool_calls below).
// ---------------------------------------------------------------------------
std::string render_glimmer_chatml_with_tools(
    const std::vector<ChatMessage>& messages, const std::string& tools_json,
    const std::string& system_suffix = "", bool add_generation_prompt = true,
    const std::string& default_system = "");

// Render + encode with tool support (same BOS semantics as
// encode_rendered_chat).
std::vector<int> encode_chatml_with_tools(
    const Tokenizer& tok, const std::vector<ChatMessage>& messages,
    const std::string& tools_json, const std::string& system_suffix = "",
    bool add_generation_prompt = true, bool add_bos = true);

// ---------------------------------------------------------------------------
// Stop-strings (checked on decoded text): truncate the generated text
// before the earliest occurrence of any non-empty stop string.
// ---------------------------------------------------------------------------
size_t find_stop_string(const std::string& text,
                        const std::vector<std::string>& stop_strings,
                        std::string* matched = nullptr);
std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings);

// Default stop strings: the chat turn terminator <|eot|> plus the
// end-of-message marker <|eom|> (the generation-config eos id 200001 stays
// covered by the stop-id side of the Session policy).
std::vector<std::string> default_stop_strings();

// ---------------------------------------------------------------------------
// Plain-text tool shim (API parity with LFM25's parse_tool_calls /
// ToolCallStreamParser so generic server code compiles unchanged).
//
// Glimmer defines no native tool-call markers, so there is nothing to
// extract: parse_tool_calls() truncates the raw completion at the first
// stop string and returns the remainder as display text with no calls and
// no reasoning. The streaming parser below is the incremental form of the
// same policy: it holds back only a bounded stop-prefix tail (plus partial
// UTF-8, mirroring IncrementalDetokenizer safety) and emits everything
// else as display deltas. Never throws.
// ---------------------------------------------------------------------------
struct ToolParseResult {
  std::string text;
  std::string reasoning;  // always "" (Glimmer has no thinking trace)
  std::vector<ToolCall> calls;  // always empty (no native markers)
};

ToolParseResult parse_tool_calls(
    const std::string& raw, const std::vector<std::string>& stops = {});

struct StreamDeltas {
  std::string display;
  std::string reasoning;  // always "" (no thinking trace)
};

class ToolCallStreamParser {
 public:
  explicit ToolCallStreamParser(
      const std::vector<std::string>& stop_strings = {});
  // One parser step for one raw piece: returns display text newly safe to
  // stream (may be empty = "hold"). Prefer this over push() on hot paths.
  StreamDeltas feed(const std::string& raw_piece);
  // Returns display text newly safe to stream (may be empty = "hold").
  std::string push(const std::string& raw_piece);
  // Glimmer has no reasoning channel: always returns "".
  std::string poll_reasoning();
  // Tail text (not yet emitted); calls is always empty.
  ToolParseResult finish();

 private:
  std::string buf_;  // undecided tail (bounded by the longest stop string)
  std::string matched_;  // stop string that fired ("" until it does)
  bool stopped_ = false;
  std::vector<std::string> stops_;  // configured stop strings (no empties)
  size_t max_stop_len_ = 1;
};

}  // namespace Glimmer
}  // namespace KokkosDNN

#endif
