// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SMOLLM2_TOKENIZER_HPP
#define KOKKOSDNN_SMOLLM2_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KokkosDNN {
namespace SmolLM2 {

// Hugging Face `tokenizer.json` loader for SmolLM2 (GPT-2 BPE, 49152).
//
// SmolLM2-135M ships a GPT-2-style tokenizer (vocab.json/merges.txt
// folded into tokenizer.json, vocab 49152, specials <|endoftext|>=0,
// <|im_start|>=1, <|im_end|>=2 within added-tokens ids 0..16) whose
// `pre_tokenizer` is Sequence[Digits(individual_digits=true),
// ByteLevel(add_prefix_space=false, use_regex=true)]. The pipeline is:
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
// BOS/EOS conventions: base model bos=eos=<|endoftext|> (id 0);
// Instruct bos=<|im_start|> (id 1), eos=pad=<|im_end|> (id 2). The loader
// defaults to the base convention (bos=eos=<|endoftext|>, eot=<|im_end|>
// as the Instruct stop token, eom aliases eos) and then applies a sibling
// tokenizer_config.json override (bos_token/eos_token string names,
// best-effort, never throws) when the tokenizer path contains a
// directory; set_bos()/set_eos() override explicitly (e.g. Instruct
// bos=1 eos=2 when no config file is available).
//
// Standalone: shares no headers with the Llama tokenizer; the small
// byte-mapping/BPE core (bytes_to_unicode, per-piece merge loop) plus
// the JSON/UTF-8 helpers and the \p{L}/\p{N} tables needed by the GPT-2
// regex are duplicated here.
class Tokenizer {
 public:
  Tokenizer() = default;
  // Throws on missing/corrupt file.
  explicit Tokenizer(const std::string& tokenizer_json_path) { load(path_ = tokenizer_json_path); }
  void load(const std::string& tokenizer_json_path);

  bool loaded() const { return !id_to_token_.empty(); }
  int vocab_size() const { return vocab_size_; }
  // Explicit BOS/EOS overrides (e.g. SmolLM2-Instruct bos=1 eos=2 when no
  // tokenizer_config.json is available to read them from).
  void set_bos(int id) { bos_id_ = id; }
  void set_eos(int id) { eos_id_ = id; }

  // Encode text -> ids. add_bos prepends the BOS id (default true).
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
  int vocab_size_ = 49152;
  int bos_id_ = 0, eos_id_ = 0, eom_id_ = 0, eot_id_ = 2;
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
// SmolLM2-Instruct ChatML template.
//
// Renders the HuggingFaceTB/SmolLM2-135M-Instruct format:
//
//   <|im_start|>system\n{system}<|im_end|>\n
//   <|im_start|>user\n{user}<|im_end|>\n
//   <|im_start|>assistant\n            <- generation prompt
//
// When no system message is present and `default_system` is non-empty, a
// default system block is injected first (mirroring the HF template's
// fallback). With add_generation_prompt=true an open assistant header is
// appended so the model continues as assistant. The output is plain text
// with literal special tokens, so Tokenizer::encode handles it via the
// special-token split path. The Instruct stop string is "<|im_end|>"
// (Tokenizer::eot()).
// ---------------------------------------------------------------------------
struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant"
  std::string content;
};

// Build the standard single-turn messages for a raw prompt: optional
// system message first, then one `role` (default "user") turn.
std::vector<ChatMessage> make_chat_messages(const std::string& prompt,
                                            const std::string& system = "",
                                            const std::string& role = "user");

std::string render_chatml_template(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt = true,
    const std::string& default_system = "You are a helpful AI assistant.");

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

}  // namespace SmolLM2
}  // namespace KokkosDNN

#endif
