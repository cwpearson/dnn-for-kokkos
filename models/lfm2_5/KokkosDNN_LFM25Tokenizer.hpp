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
//   tool-call markers 124905/124906.
//
// The loader resolves every id by literal content from `added_tokens` and
// falls back to the numeric values below when an entry is absent, so a
// tokenizer.json that renames a special still works as long as the content
// matches. The think contents ("<think>"/"</think>") are the PoC
// convention used by render_lfm25_chatml; files that spell the think pair
// differently keep working because resolution is content-driven with the
// numeric fallback only as a last resort.
//
// NOTE: the tool-call markers are `special: false` in the HF tokenizer,
// so Tokenizer::decode keeps them even with skip_special_tokens=true.
// parse_tool_calls() relies on that: tool calls are parsed from decoded
// text.
inline constexpr int kPadId        = 124893;
inline constexpr int kBosId        = 124894;
inline constexpr int kImStartId    = 124899;
inline constexpr int kEosId        = 124900;
inline constexpr int kThinkStartId = 124901;
inline constexpr int kThinkEndId   = 124902;
inline constexpr int kToolCallStartId = 124905;
inline constexpr int kToolCallEndId   = 124906;

inline constexpr const char* kPadToken        = "<|pad|>";
inline constexpr const char* kBosToken        = "<|startoftext|>";
inline constexpr const char* kImStartToken    = "<|im_start|>";
inline constexpr const char* kEosToken        = "<|im_end|>";
inline constexpr const char* kThinkStartToken = "<think>";
inline constexpr const char* kThinkEndToken   = "</think>";
inline constexpr const char* kToolCallStartToken = "<|tool_call_start|>";
inline constexpr const char* kToolCallEndToken   = "<|tool_call_end|>";

// Hugging Face `tokenizer.json` loader for LFM2.5 (byte-level BPE, 128k).
//
// Same dependency-free pipeline as the SmolLM2 tokenizer, whose pattern is
// mirrored here (this header is standalone and shares no code with it):
//   1. literal special/added tokens are split out first (longest match,
//      never broken up by BPE);
//   2. remaining text is split by HF Digits (cut before and after every
//      ASCII digit) and then pre-tokenized with the GPT-2 ByteLevel regex
//      (use_regex=true):
//        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|
//        \s+(?!\S)|\s+
//      (Unicode-aware \p{L}/\p{N}, leftmost-first alternation;
//      contractions are case-SENSITIVE, the optional prefix is a single
//      literal U+0020 space, numbers group greedily);
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
// LFM2.5-Instruct ChatML template with thinking and tool-call support.
//
// Renders the ChatML-style format:
//
//   <|im_start|>system\n{system}<|im_end|>\n
//   <|im_start|>user\n{user}<|im_end|>\n
//   <|im_start|>assistant\n[<think>{thinking}</think>\n]{content}[tool calls]<|im_end|>\n
//   <|im_start|>tool\n{result}<|im_end|>\n
//   <|im_start|>assistant\n            <- generation prompt
//
// Tool definitions (`tools_json`, one JSON object string per tool, e.g.
// `{"name":"...","description":"...","parameters":{...}}`) are appended to
// the system prompt as `List of tools: [...]`, matching the official LFM2.5
// chat template: an existing first system message gets the suffix, otherwise
// a system block is injected (after `default_system` when set).
//
// Assistant tool calls render in the native LFM format:
//
//   <|tool_call_start|>[name(arg='val', ...), ...]<|tool_call_end|>
//
// with Pythonic argument values (strings single-quoted, True/False/None,
// nested structures as JSON). Tool results use the `tool` role block.
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
struct ToolCall {
  std::string id;         // OpenAI call id (may be empty in history)
  std::string name;       // function name
  std::string arguments;  // JSON object string, OpenAI wire format (may be
                          // empty, meaning no arguments)
};

struct ChatMessage {
  std::string role;      // "system" | "user" | "assistant" | "tool"
  std::string content;
  std::string thinking;  // assistant reasoning trace (may be empty)
  std::vector<ToolCall> tool_calls;  // assistant tool calls (else empty)
  std::string tool_call_id;  // tool result id (role "tool", may be empty)
};

// Build the standard single-turn messages for a raw prompt: optional
// system message first, then one `role` (default "user") turn.
std::vector<ChatMessage> make_chat_messages(const std::string& prompt,
                                            const std::string& system = "",
                                            const std::string& role = "user");

std::string render_lfm25_chatml(const std::vector<ChatMessage>& messages,
                                bool add_generation_prompt = true,
                                bool preserve_thinking = true,
                                const std::string& default_system = "",
                                const std::vector<std::string>& tools_json = {});

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
                               bool preserve_thinking = true,
                               const std::vector<std::string>& tools_json = {});

// Render one native tool call: `name(arg='val', ...)` with Pythonic values
// converted from the JSON `arguments_json` object string. Empty/absence
// ("", "{}", "null") renders as `name()`. Unparseable input falls back to
// `name()` rather than emitting syntax the model was not trained on.
std::string render_tool_call(const std::string& name,
                             const std::string& arguments_json);

// Parse native tool calls out of generated text. Every
// `<|tool_call_start|>[...]<|tool_call_end|>` block that yields at least one
// well-formed call contributes its calls (in order, `id` left empty for the
// server to allocate); the block is removed from the returned text
// (non-call text trailing the call list is kept as reply text).
// Blocks without a well-formed call are left in place, and when no block
// parses the input is returned unchanged with empty `calls`. The returned
// text is trimmed of leading/trailing whitespace.
struct ParsedToolCalls {
  std::string text;
  std::vector<ToolCall> calls;
};

ParsedToolCalls parse_tool_calls(const std::string& text);

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
