// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_LFM25_TOKENIZER_HPP
#define KOKKOSDNN_LFM25_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KokkosDNN {
namespace LFM25 {

// LFM2.5 special-token ids and literal contents.
//
//   pad 124893, bos <|startoftext|> 124894, <|im_start|> 124899,
//   eos/stop <|im_end|> 124900, think pair 124901/124902,
//   tool-list pair 124903/124904, tool-call pair 124905/124906.
//
// The loader resolves every id by literal content from `added_tokens` and
// falls back to the numeric values below when an entry is absent, so a
// tokenizer.json that renames a special still works as long as the content
// matches. The think contents ("<think>"/"</think>") are the PoC
// convention used by render_lfm25_chatml; files that spell the think pair
// differently keep working because resolution is content-driven with the
// numeric fallback only as a last resort.
inline constexpr int kPadId        = 124893;
inline constexpr int kBosId        = 124894;
inline constexpr int kImStartId    = 124899;
inline constexpr int kEosId        = 124900;
inline constexpr int kThinkStartId = 124901;
inline constexpr int kThinkEndId   = 124902;
inline constexpr int kToolListStartId = 124903;
inline constexpr int kToolListEndId   = 124904;
inline constexpr int kToolCallStartId = 124905;
inline constexpr int kToolCallEndId   = 124906;

inline constexpr const char* kPadToken        = "<|pad|>";
inline constexpr const char* kBosToken        = "<|startoftext|>";
inline constexpr const char* kImStartToken    = "<|im_start|>";
inline constexpr const char* kEosToken        = "<|im_end|>";
inline constexpr const char* kThinkStartToken = "<think>";
inline constexpr const char* kThinkEndToken   = "</think>";
inline constexpr const char* kToolListStartToken = "<|tool_list_start|>";
inline constexpr const char* kToolListEndToken   = "<|tool_list_end|>";
inline constexpr const char* kToolCallStartToken = "<|tool_call_start|>";
inline constexpr const char* kToolCallEndToken   = "<|tool_call_end|>";
// Exact assistant tool-call wrapper used by the Instruct template:
//   <|tool_call_start|>[name(k="v", ...), ...]<|tool_call_end|>
//
// Tolerated alternate spellings. The canonical model emits the native pair
// above and the <think> trace, but fine-tunes and long degenerate turns
// sometimes slide into the LFM3 / qwen3-coder grammar
//   <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
// or a hybrid of the two (an XML opener with the native end token, and/or a
// `function=` prefix on pythonic items). The parser accepts those as
// aliases so a drifting turn still becomes a tool call instead of leaking
// markup into the thinking trace. Rendering always uses the native pair.
inline constexpr const char* kThinkStartAltToken = "<thinking>";
inline constexpr const char* kThinkEndAltToken   = "</thinking>";
inline constexpr const char* kToolCallStartAltToken = "<tool_call>";
inline constexpr const char* kToolCallEndAltToken   = "</tool_call>";

// Hugging Face `tokenizer.json` loader for LFM2.5 (byte-level BPE, 128k).
//
// Same dependency-free pipeline as the SmolLM2 tokenizer, whose pattern is
// mirrored here (this header is standalone and shares no code with it):
//   1. literal special/added tokens are split out first (longest match,
//      never broken up by BPE);
//   2. remaining text is pre-tokenized with the LFM2 ByteLevel regex
//      (use_regex=true, in the tokenizer.json Split step):
//        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|
//        \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//      (Unicode-aware \p{L}/\p{N}, leftmost-first alternation;
//      contractions are case-INSENSITIVE, the optional prefix on a letter
//      run is any single non-CR/LF/letter/number char, numbers group in
//      runs of up to three, symbol runs absorb trailing CR/LF, and a
//      whitespace run containing a newline breaks after the last newline);
//   3. each piece is byte-mapped with the GPT-2 bytes_to_unicode table;
//   4. BPE merges apply per piece (lowest rank first); pieces with no
//      applicable merge fall back to single-byte token ids (byte-fallback
//      encode path: every input byte maps to exactly one base vocab id);
//   5. decode inverts the byte mapping and skips special tokens unless
//      asked otherwise.
//
// BOS/EOS conventions: bos=<|startoftext|> (id 124894),
// eos=stop=<|im_end|> (id 124900). The loader defaults to that convention
// and then applies a sibling tokenizer_config.json override (bos_token /
// eos_token string names, best-effort, never throws) when the tokenizer
// path contains a directory; set_bos()/set_eos() override explicitly.
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

  // Encode text -> ids. add_bos prepends the BOS id (default true).
  std::vector<int> encode(const std::string& text, bool add_bos = true) const;
  // skip_special_tokens=true (HF decode default) drops added/special ids.
  std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

  int bos() const { return bos_id_; }
  int eos() const { return eos_id_; }
  int eot() const { return eos_id_; }  // Instruct stop == <|im_end|>
  int pad() const { return pad_id_; }
  int im_start() const { return im_start_id_; }
  int think_start() const { return think_start_id_; }
  int think_end() const { return think_end_id_; }
  bool is_special(int id) const { return special_ids_.count(id) != 0; }
 private:
  struct AddedToken {
    int id = -1;
    std::string content;
    bool special = true;
  };

  std::string path_;
  int vocab_size_ = 128000;
  int bos_id_ = kBosId, eos_id_ = kEosId, pad_id_ = kPadId;
  int im_start_id_ = kImStartId;
  int think_start_id_ = kThinkStartId, think_end_id_ = kThinkEndId;
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
// LFM2.5-Instruct ChatML template with thinking support.
//
// Renders the ChatML-style format:
//
//   <|im_start|>system\n{system}<|im_end|>\n
//   <|im_start|>user\n{user}<|im_end|>\n
//   <|im_start|>assistant\n[<think>{thinking}</think>]{content}<|im_end|>\n
//   <|im_start|>assistant\n<think>   <- generation prompt (open thinker)
//
// The generation prompt carries the open <think> (matching the HF
// chat_template.jinja, which appends "<|im_start|>assistant\n<think>") so
// the model starts inside the thinking trace: raw completions normally
// contain only the closing </think>, never the opener. History thinking
// blocks render as "<think>{thinking}</think>" directly followed by the
// content (no separating newline), also matching HF.
//
// A thinking block is emitted inside an assistant turn only when that
// turn carries non-empty `thinking` AND `preserve_thinking` is true;
// otherwise the thinking trace is dropped (content is always kept).
// Thinking on non-assistant roles is ignored. When no system message is
// present and `default_system` is non-empty, a default system block is
// injected first (empty by default: no injection). With
// add_generation_prompt=true an open assistant header is appended so the
// model continues as assistant. The output is plain text with literal
// special tokens, so Tokenizer::encode handles it via the special-token
// split path. The Instruct stop string is "<|im_end|>" (Tokenizer::eot()).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// One function invocation. `arguments` is a JSON object string
// (OpenAI `function.arguments` wire form, e.g. `{"location":"SF"}`);
// the renderer converts it to the native pythonic form and the parser
// converts the native form back, so both directions stay lossless for
// JSON-representable values.
struct ToolCall {
  std::string id;         // OpenAI tool_call id ("" when unknown/unparsed)
  std::string name;       // function name
  std::string arguments;  // JSON object string ("" or "{}" when none)
};

struct ChatMessage {
  std::string role;      // "system" | "user" | "assistant" | "tool"
  std::string content;
  std::string thinking;  // assistant reasoning trace (may be empty)
  // Agentic history: assistant turns may carry structured tool calls
  // (reconstructed into the native <|tool_call_start|>...<|tool_call_end|>
  // form by the renderer); "tool" turns carry the function result in
  // `content` and optionally link it via `tool_call_id`. Defaulted so the
  // existing {"role", "content", "thinking"} aggregate init keeps working.
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
};

// Build the standard single-turn messages for a raw prompt: optional
// system message first, then one `role` (default "user") turn.
std::vector<ChatMessage> make_chat_messages(const std::string& prompt,
                                            const std::string& system = "",
                                            const std::string& role = "user");

std::string render_lfm25_chatml(const std::vector<ChatMessage>& messages,
                                bool add_generation_prompt = true,
                                bool preserve_thinking = true,
                                const std::string& default_system = "");

// BOS handling: collapse any run of leading `bos_id` to exactly one when
// want_bos (or strip all leading copies when !want_bos). Guarantees the
// "exactly one BOS" invariant even when a chat-rendered string already
// starts with the BOS literal and encode(add_bos=true) would prepend a
// second copy, or when raw text itself starts with the BOS literal.
std::vector<int> ensure_bos(const std::vector<int>& ids, int bos_id,
                            bool want_bos);

// Encode an already-rendered chat string with exactly-one-BOS semantics:
// encodes with add_bos=false (the literal specials in `rendered` flow
// through the special-token path), then applies ensure_bos().
std::vector<int> encode_rendered_chat(const Tokenizer& tok,
                                      const std::string& rendered,
                                      bool add_bos = true);

// Render + encode in one step (same BOS semantics as encode_rendered_chat,
// but with the ChatML renderer).
std::vector<int> encode_chatml(const Tokenizer& tok,
                               const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt = true,
                               bool add_bos = true,
                               bool preserve_thinking = true);

// ---------------------------------------------------------------------------
// Native tool-use (function calling) support.
//
// LFM2.5-Instruct emits tool calls in the native pythonic form
//   <|tool_call_start|>[name(k="v", ...), ...]<|tool_call_end|>
// with the tool catalogue supplied as `List of tools: [...]` JSON text in
// the system prompt (this mirrors apply_chat_template(tools=...)); tool
// results arrive as <|im_start|>tool turns. Everything here is plain-text
// translation with no JSON-library dependency: the OpenAI-protocol mapping
// (tools/tool_choice/message tool_calls/JSON arguments) lives in server/.
// ---------------------------------------------------------------------------

// Like render_lfm25_chatml, plus native tool support:
//  - tools_json (compact-serialized OpenAI `tools` array, "" = none):
//    appended to the system prompt as "\nList of tools: <json>" (a system
//    block is injected when no system message exists).
//  - system_suffix ("" = none): extra instruction sentence(s) appended to
//    the system block (tool_choice required/named directives, JSON-mode
//    bias). The server builds these from the OpenAI request.
//  - role "tool" renders as <|im_start|>tool\n{content}<|im_end|>\n.
//  - assistant messages with tool_calls render text content followed by the
//    native call block (arguments converted JSON -> pythonic).
std::string render_lfm25_chatml_with_tools(
    const std::vector<ChatMessage>& messages, const std::string& tools_json,
    const std::string& system_suffix = "", bool add_generation_prompt = true,
    bool preserve_thinking = true, const std::string& default_system = "");

// Render + encode with tool support (same BOS semantics as
// encode_rendered_chat).
std::vector<int> encode_chatml_with_tools(
    const Tokenizer& tok, const std::vector<ChatMessage>& messages,
    const std::string& tools_json, const std::string& system_suffix = "",
    bool add_generation_prompt = true, bool add_bos = true,
    bool preserve_thinking = true);

// Split raw completion text (decoded with skip_special_tokens=false) into
// display text + tool calls + reasoning. Every complete
// <|tool_call_start|>...<|tool_call_end|> region becomes calls (in order;
// ids are empty, the server assigns them); surrounding text, truncated at
// the first stop string, is the display content (the <|im_end|> turn
// terminator always applies, independent of `stops`). A length-capped
// input ending mid-marker has its trailing partial control-marker head
// stripped up front, so fragments never leak into either view. <think>...</think> reasoning blocks (the LFM2.5-Instruct
// thinking trace) are extracted into
// `reasoning` and removed from the display text so OpenAI-compatible
// clients can surface them as `reasoning_content` instead of leaking
// literal <think> tags into `content`. The generation prompt ends with an
// open <think>, so raw completions normally open implicitly: when
// `start_in_reasoning` is true, the leading segment up to the first
// </think> / tool-call start / turn terminator is reasoning even without
// an explicit <think> opener (a redundant leading opener is stripped and
// leading whitespace belongs to the trace). A turn that reaches none of
// those terminators is treated as cut mid-thought, so all of it is
// reasoning and `text` is empty — the streaming view cannot wait for a
// terminator that may never arrive, so the whole-text parse has to agree.
// An explicit opener later in the turn only counts while no non-whitespace
// display content precedes it (tool-call blocks excluded, so thinking
// after a tool call still counts); model text that merely mentions the
// tag stays content. A block closes at the first </think>; a tool call
// start or the turn terminator also ends the block implicitly. An
// unclosed trailing explicit <think> treats the rest as reasoning; a lone
// </think> without an opener passes through into the text unless
// `start_in_reasoning` covers the leading segment. Unclosed
// trailing tool markers (a turn cut by
// max_tokens/length inside or just before a call block) are recovered
// best-effort: complete calls (closed parens) in the truncated list become
// calls instead of leaking call markup into the display text; a tail cut
// mid-arguments recovers nothing and stays text. Recovery carries no
// truncation marker, so validate arguments before executing.
// Never throws on model output (malformed blocks are skipped).
//
// Tolerated spellings: <thinking>...</thinking> is accepted wherever
// <think>/</think> is, and <tool_call>...</tool_call> (the LFM3 /
// qwen3-coder wrapper) is accepted as a tool-call block with either the
// native or the LFM3 body grammar — including hybrids such as an XML
// opener closed by <|tool_call_end|> or pythonic items prefixed with
// `function=`. These are parse-only aliases; the renderer always emits
// the canonical native markers.
struct ToolParseResult {
  std::string text;
  std::string reasoning;  // trace contents, openers excluded ("" when none)
  std::vector<ToolCall> calls;
};
ToolParseResult parse_tool_calls(
    const std::string& raw, const std::vector<std::string>& stops = {},
    bool start_in_reasoning = false);

// Convert one call-block body into calls. The native form is
// `[f(a="1"), g(x=2)]` (brackets optional), with arguments converted from
// the pythonic form (strings, numbers, True/False/None, lists, dicts, bare
// words) to a JSON object string. Also accepts the LFM3 / qwen3-coder body
// `<function=f><parameter=a>1</parameter></function>` and hybrids that
// prefix pythonic items with `function=`. Never throws; malformed entries
// are skipped.
std::vector<ToolCall> parse_tool_call_block(const std::string& block);

// Incremental variant of parse_tool_calls for SSE streaming: feed raw
// pieces (each sampled id decoded with skip_special_tokens=false) via
// feed(); each call returns the display and reasoning text newly safe to
// stream since the last call (either may be empty = "hold"). push() /
// poll_reasoning() split the same work across two calls for callers that
// consume the channels separately (call poll once per push); finish()
// flushes the tail and returns the parsed calls. The turn terminator is
// always in the stop set (unioned in from the constructor argument), so a
// complete <|im_end|> never leaks into content however the caller
// configures stops. Holding (rather than emitting) retractable bytes
// keeps both views monotone in the input, and every entry point rebases
// instead of throwing, so streaming never throws. Reasoning and display
// are disjoint views: total display text = concatenated display deltas +
// finish().text, total reasoning = concatenated reasoning deltas +
// finish().reasoning. Totaled streaming output agrees with the whole-text
// parse_tool_calls() on the same input (with the same stops and the same
// start_in_reasoning), including the unclosed-trailing recovery.
//
// Cost: the parser is a single-pass incremental machine. Each input byte
// is classified once (think split, call excision, stop cut run on the
// newly arrived bytes plus a bounded marker/stop overlap window — at most
// one trailing head, the longest matching prefix, is ever held), so a
// turn costs O(n) total, not O(n^2): long thinking traces and long
// trailing `<` runs stream in amortized O(1) per token.
struct StreamDeltas {
  std::string display;
  std::string reasoning;
};

class ToolCallStreamParser {
 public:
  // `start_in_reasoning` mirrors parse_tool_calls()'s third argument: set
  // it when the generation prompt already opened the thinker, so the raw
  // pieces are a bare trace followed by </think> (or cut mid-thought).
  explicit ToolCallStreamParser(
      const std::vector<std::string>& stop_strings = {},
      bool start_in_reasoning = false);
  // One parser step for one raw piece: returns display + reasoning deltas
  // newly safe to stream. A single call per token keeps the per-turn cost
  // linear; prefer this over push()+poll_reasoning() on hot paths.
  StreamDeltas feed(const std::string& raw_piece);
  // Returns display text newly safe to stream (may be empty = "hold").
  // Stashes the reasoning delta for the next poll_reasoning() call.
  std::string push(const std::string& raw_piece);
  // Returns reasoning text accumulated since the last poll (may be empty).
  std::string poll_reasoning();
  // Tail text (not yet emitted) + tail reasoning + parsed calls.
  ToolParseResult finish();

 private:
  // Layer 1: scan raw_ for tool-call block structure; deliver the settled
  // prefix (everything before the first unclosed block start, minus a
  // trailing partial control-marker head) to l2_consume() in order.
  void l1_drive();
  // Layer 2: think split over the settled stream (stateful across calls);
  // display bytes flow to l3_consume(), reasoning bytes to out_r_.
  void l2_consume(const std::string& piece);
  // Layer 3: tool-call block excision over the display stream; outside
  // bytes accumulate in out_e_.
  void l3_consume(const std::string& piece);
  // Layer 4: advance the stop search over out_e_ and carve the display /
  // reasoning deltas out of out_e_ / out_r_ with hold-backs for partial
  // UTF-8, control-marker heads, and custom stop heads.
  StreamDeltas emit_deltas();

  std::string raw_;
  // Layer 1 (settled delivery) state.
  size_t l1_delivered_ = 0;  // raw_ bytes delivered to layer 2
  bool l1_in_block_ = false;
  size_t l1_block_start_ = 0;  // raw_ offset of the open block's start
  size_t l1_end_search_ = 0;   // end-marker search cursor (while in block)
  std::string l1_prehold_;  // held head bytes preceding the open block
  // Layer 2 (think split) state.
  // l2_lead_ covers the prompt-provided open thinker: the generation
  // prompt ends with "<think>", so the turn opens inside reasoning and the
  // raw completion normally carries no opener. While set, settled bytes
  // stream live as reasoning until the first decided closer / tool-call
  // start / turn terminator (a redundant leading opener is stripped once,
  // while l2_lead_fresh_ holds).
  bool l2_lead_ = false;
  bool l2_lead_fresh_ = true;  // only whitespace routed to the trace so far
  bool l2_in_think_ = false;
  bool l2_ws_ = true;  // display emitted so far is all whitespace
                       // (tool-call blocks excluded: thinking after a tool
                       // call still counts as leading)
  std::string l2_pend_;  // undecided settled tail (bounded by marker lens)
  // Layer 3 (call excision) state.
  bool l3_in_call_ = false;
  std::string l3_pend_;  // undecided display tail (bounded by marker lens)
  // Layer 4 (stop cut + emission) state.
  std::string out_e_;  // post-excision display (append-only)
  std::string out_r_;  // firm reasoning (append-only)
  size_t l4_stop_search_ = 0;  // stop search cursor over out_e_
  bool l4_stopped_ = false;
  size_t l4_cut_ = 0;  // display cut position (valid when stopped)
  std::string stash_r_;  // reasoning delta held for poll_reasoning()
  size_t emitted_ = 0;  // chars of display text already returned
  size_t emitted_r_ = 0;  // chars of reasoning text already returned
  std::vector<std::string> stops_;
  size_t max_stop_len_ = 1;
  bool start_in_reasoning_ = false;
};

// Argument conversion between the native pythonic form and JSON.
// args_pythonic_to_json("a=\"x\", n=3, ok=True") -> {"a":"x","n":3,"ok":true}
// args_json_to_pythonic("{\"a\":\"x\",\"n\":3}") -> a="x", n=3
// Both are lenient and never throw: unparseable pythonic input yields "{}";
// unparseable JSON input is carried as a single raw string value.
std::string args_pythonic_to_json(const std::string& args);
std::string args_json_to_pythonic(const std::string& args_json);

// ---------------------------------------------------------------------------
// Stop-strings (checked on decoded text): truncate the generated text
// before the earliest occurrence of any non-empty stop string.
// ---------------------------------------------------------------------------
size_t find_stop_string(const std::string& text,
                        const std::vector<std::string>& stop_strings,
                        std::string* matched = nullptr);
std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings);

// Default LFM2.5-Instruct stop strings: the <|im_end|> turn terminator.
std::vector<std::string> default_stop_strings();

}  // namespace LFM25
}  // namespace KokkosDNN

#endif
