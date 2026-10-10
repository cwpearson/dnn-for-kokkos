#ifndef KOKKOSDNN_TOKENIZER_HPP
#define KOKKOSDNN_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KokkosDNN {
namespace Llama {

// Hugging Face `tokenizer.json` loader for Llama 3 (tiktoken BPE, 128k).
//
// Faithful to the `tokenizers` Rust library pipeline used by the real
// `meta-llama/Llama-3.1-8B-Instruct` tokenizer:
//   1. literal special/added tokens are split out first (longest match,
//      never broken up by BPE);
//   2. remaining text is pre-tokenized with the tiktoken Split regex
//      (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//       ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//      (Unicode-aware \p{L}/\p{N}, leftmost-first alternation);
//   3. each piece is byte-mapped with the GPT-2 bytes_to_unicode table;
//   4. BPE merges apply per piece (lowest rank first);
//   5. decode inverts the byte mapping and skips special tokens unless
//      asked otherwise.
//
// Special tokens (Llama 3): BOS <|begin_of_text|> = 128000,
// EOS <|end_of_text|> = 128001, EOM <|eom_id|> = 128008,
// EOT <|eot_id|> = 128009 (the actual stop
// token of the Instruct chat template), plus <|start_header_id|> /
// <|end_header_id|> and reserved tokens in 128002..128255. Ids are
// resolved by content from `added_tokens`, falling back to the
// well-known values above when absent.
class Tokenizer {
 public:
  Tokenizer() = default;
  // Throws on missing/corrupt file.
  explicit Tokenizer(const std::string& tokenizer_json_path) { load(path_ = tokenizer_json_path); }
  void load(const std::string& tokenizer_json_path);

  bool loaded() const { return !id_to_token_.empty(); }
  int vocab_size() const { return vocab_size_; }

  // Encode text -> ids. add_bos prepends <|begin_of_text|> (default true,
  // matching the file's TemplateProcessing post-processor).
  std::vector<int> encode(const std::string& text, bool add_bos = true) const;
  // skip_special_tokens=true (HF decode default) drops added/special ids.
  std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

  int bos() const { return bos_id_; }
  int eos() const { return eos_id_; }
  int eom() const { return eom_id_; }
  int eot() const { return eot_id_; }
  bool is_special(int id) const { return special_ids_.count(id) != 0; }
 private:
  struct AddedToken {
    int id = -1;
    std::string content;
    bool special = true;
  };

  std::string path_;
  int vocab_size_ = 128256;
  int bos_id_ = 128000, eos_id_ = 128001, eom_id_ = 128008, eot_id_ = 128009;
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
// Llama 3.x Instruct chat template (README TODO "Instruct chat template").
//
// Renders the Meta format used by Llama-3.1-8B-Instruct:
//
//   <|begin_of_text|>
//   <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
//   <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
//   <|start_header_id|>assistant<|end_header_id|>\n\n          <- generation prompt
//
// Each message contributes one header/content/<|eot_id|> block; with
// add_generation_prompt=true an open assistant header (no <|eot_id|>) is
// appended so the model continues as assistant. Content is used verbatim
// (no trimming); callers that want HF `| trim` semantics trim beforehand.
// The rendered string is plain text containing literal special tokens, so
// Tokenizer::encode handles it via the existing special-token split path
// (faithful BPE path untouched).
// ---------------------------------------------------------------------------
struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant"
  std::string content;
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
// Stop-strings (checked on decoded text): truncate the generated text
// before the earliest occurrence of any non-empty stop string.
// ---------------------------------------------------------------------------
size_t find_stop_string(const std::string& text,
                        const std::vector<std::string>& stop_strings,
                        std::string* matched = nullptr);
std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings);

}  // namespace Llama
}  // namespace KokkosDNN

#endif
