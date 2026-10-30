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
// Muse Glimmer chat template.
//
// The plain render_chat_template() below keeps the original minimal framing
// (one "<|start|>{role}<|message|>{content}<|eot|>" block per message plus
// an open assistant header); it is retained for the driver/raw-prompt path.
//
// The tool-aware render_glimmer_chatml_with_tools() instead follows the
// upstream ATEM chat template (meta-models/Muse-Glimmer-30B
// `chat_template.jinja`, confirmed via the Model API prompting guide):
// role-tagged turns with `to=` recipients (`to=self` private reasoning,
// `to=<tool>` tool calls, `to=user` replies), the tool catalogue + valid
// recipient list in the system block, `<|eom|>` (message continues) vs
// `<|eot|>` (turn ends) closers, and ATEM tool-call blocks:
//   <atem:function_calls>
//   <atem:invoke name="NAME">
//   <atem:parameter name="K">V</atem:parameter>
//   ...
//   </atem:invoke>
//   </atem:function_calls>
// Tool results render as `<|start|>tool NAME<|message|><tool_output
// name="NAME">...</tool_output><|eot|>` turns. The server path (and any
// agentic client) must use the tool-aware renderer; the plain renderer
// cannot carry tool calls, tool results, or reasoning traces.
// ---------------------------------------------------------------------------
// A single tool invocation. `arguments` is a JSON object string (OpenAI
// `function.arguments` wire form).
struct ToolCall {
  std::string id;         // OpenAI tool_call id ("" when unknown/unparsed)
  std::string name;       // function name
  std::string arguments;  // JSON object string ("" or "{}" when none)
};

struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant" | "tool"
  std::string content;
  // Agentic-history interchange (API parity with the LFM2.5 chat helpers so
  // server code stays generic): assistant turns may carry a reasoning trace
  // (`thinking`, rendered as the `to=self` channel and parsed back out of
  // completions) and structured tool calls (rendered as `to=<tool>` ATEM
  // channels); "tool" turns carry the function result in `content` and name
  // the tool via `name` (preferred) or `tool_call_id` (resolved against
  // earlier assistant `tool_calls` when `name` is empty, mirroring the
  // upstream template). Defaulted so the existing {"role", "content"}
  // aggregate init keeps working.
  std::string thinking;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
  std::string name;  // tool message function name (may be empty)
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
// render_lfm25_chatml_with_tools / encode_chatml_with_tools), following the
// upstream ATEM chat template.
//
//   - tools_json (compact-serialized OpenAI `tools` array, "" = none) is
//     rendered into the system block as the upstream tool-catalogue text
//     (instruction preamble + per-namespace metadata + per-function JSON
//     schemas), exactly as `apply_chat_template(..., tools=...)` does. A
//     system block is always emitted: when no system message exists the
//     upstream default ("You are a helpful AI assistant.\nKnowledge cutoff:
//     2026-01-04.", overridable via default_system) is used.
//   - system_suffix ("" = none): extra instruction sentence(s) appended to
//     the system block (tool_choice required/named directives, JSON-mode
//     bias). "Reasoning strength: high." is ensured exactly once ahead of
//     the catalogue (skipped when the system text already carries a
//     reasoning-strength line in any casing).
//   - assistant `thinking` renders as the `to=self` reasoning channel;
//     assistant `tool_calls` render as `to=<tool>` ATEM channels (an
//     assistant turn carrying `tool_calls` drops its plain `content`, as the
//     upstream template does); "tool"-role messages render as `<tool_output>`
//     turns under their resolved tool name.
//   - A single tool call per turn is the upstream norm; multiple tool calls
//     in one turn render (each closed with `<|eom|>` except the last, which
//     takes the turn-closer) and parse back, and the server truncates to the
//     first when the request sets `parallel_tool_calls:false`.
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

// Render one tool call as its upstream ATEM block (one
// `<atem:function_calls>` wrapper per call). `arguments` is the OpenAI JSON
// object string; values re-materialize per the template rules (JSON strings
// as raw text, numbers/booleans/null/objects/arrays as JSON).
std::string atem_invoke_block(const ToolCall& call);

// Parse every complete `<atem:invoke name="...">...</atem:invoke>` region in
// `text` into calls (in order; ids are empty, the server assigns them).
// Unclosed trailing regions are dropped (never calls, never display).
// Never throws on model output (malformed entries are skipped).
std::vector<ToolCall> parse_atem_invokes(const std::string& text);

// Convert an OpenAI arguments JSON object string to the inner ATEM
// `<atem:parameter>` lines ("" when empty/"{}"/unparseable). Never throws.
std::string args_json_to_atem(const std::string& args_json);

// Convert one raw ATEM parameter value to its JSON representation: valid
// JSON values (objects, arrays, numbers, true/false/null, quoted strings)
// pass through, everything else becomes a JSON string. Never throws.
std::string atem_value_to_json(const std::string& value);

// ---------------------------------------------------------------------------
// Stop-strings (checked on decoded text): truncate the generated text
// before the earliest occurrence of any non-empty stop string.
// ---------------------------------------------------------------------------
size_t find_stop_string(const std::string& text,
                        const std::vector<std::string>& stop_strings,
                        std::string* matched = nullptr);
std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings);

// Default stop strings: the chat turn terminator <|eot|> (the
// generation-config eos id 200001 stays covered by the stop-id side of the
// Session policy). `<|eom|>` is deliberately NOT a stop: it ends a message
// (e.g. the `to=self` reasoning channel) while the turn continues, so
// stopping there would cut agentic turns before the tool call. The channel
// parser treats `<|eom|>` structurally regardless of the stop list.
std::vector<std::string> default_stop_strings();

// ---------------------------------------------------------------------------
// ATEM tool parsing (API parity with LFM25's parse_tool_calls /
// ToolCallStreamParser so generic server code compiles unchanged).
//
// parse_tool_calls() splits a raw completion (decoded with
// skip_special_tokens=false) into display text, the `to=self` reasoning
// trace, and tool calls parsed from `<atem:invoke>` blocks (see
// parse_atem_invokes). Channel headers (`<|start|>assistant to=...`) and
// footers (`<|eom|>`/`<|eot|>`) never leak into either text view; the
// display text is additionally cut at the first stop string (with
// `<|eot|>` always applying, independent of `stops`). A length-capped turn
// cut mid-marker has its longest trailing partial marker head stripped, so
// fragments never leak into `content`. Never throws on model output.
// ---------------------------------------------------------------------------
struct ToolParseResult {
  std::string text;
  std::string reasoning;  // `to=self` trace ("" when none)
  std::vector<ToolCall> calls;
};

ToolParseResult parse_tool_calls(
    const std::string& raw, const std::vector<std::string>& stops = {});

struct StreamDeltas {
  std::string display;
  std::string reasoning;
};

class ToolCallStreamParser {
 public:
  explicit ToolCallStreamParser(
      const std::vector<std::string>& stop_strings = {});
  // One parser step for one raw piece: returns display + reasoning text
  // newly safe to stream (either may be empty = "hold"). Prefer this over
  // push() on hot paths.
  StreamDeltas feed(const std::string& raw_piece);
  // Returns display text newly safe to stream (may be empty = "hold").
  // Stashes the reasoning delta for the next poll_reasoning() call.
  std::string push(const std::string& raw_piece);
  // Returns reasoning text accumulated since the last poll (may be empty).
  std::string poll_reasoning();
  // Tail text + tail reasoning + parsed calls. Idempotent.
  ToolParseResult finish();

 private:
  enum class Channel { kDisplay, kSelf, kTool };

  // Incremental channel state. Raw pieces append to hold_; complete
  // channel headers/footers and ATEM markup resolve out of it while
  // channel bodies stream live into acc_d_/acc_r_ (display stop hold-backs
  // and partial UTF-8 excepted). rel_d_/rel_r_ are the settled frontiers,
  // emitted_d_/emitted_r_ what feed()/push() already returned, so deltas
  // stay monotone; calls_ collects completed invokes. finish() finalizes
  // once and caches the tail (repeat calls agree).
  Channel cur_ = Channel::kDisplay;
  std::string hold_;       // undecided bytes (bounded, see drive())
  bool in_hdr_ = false;    // inside a `<|start|>assistant...` header
  bool in_atem_ = false;   // inside an `<atem:invoke>` element
  std::string atem_buf_;   // bytes of the current invoke element
  std::string acc_d_;      // display bytes (append-only)
  std::string acc_r_;      // reasoning bytes (append-only)
  size_t rel_d_ = 0;       // settled display frontier
  size_t rel_r_ = 0;       // settled reasoning frontier
  size_t emitted_d_ = 0;   // display chars already returned
  size_t emitted_r_ = 0;   // reasoning chars already returned
  std::string tool_pend_;  // all-whitespace TOOL text (structural; dropped)
  std::string stash_r_;   // reasoning delta held for poll_reasoning()
  std::vector<ToolCall> calls_;
  bool stopped_ = false;  // display cut (stop found or eot seen)
  bool eot_seen_ = false;  // structural turn end seen (drop all)
  size_t eot_at_ = std::string::npos;
  bool finished_ = false;
  ToolParseResult tail_;  // cached finish() result
  std::vector<std::string> stops_;  // configured stop strings (no empties)
  size_t max_stop_len_ = 1;
  std::vector<std::string> marker_hold_;  // release hold-back markers

  void drive(bool end_of_input);
  void route_text(const std::string& s);
  void switch_to(const std::string& recip);
  void close_atem();
  void release_display();
  void release_reasoning();
  StreamDeltas emit_deltas();
};

}  // namespace Glimmer
}  // namespace KokkosDNN

#endif
