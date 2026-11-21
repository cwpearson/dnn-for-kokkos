// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_TOKENIZER_HPP
#define KOKKOSDNN_INKLING_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KokkosDNN {
namespace Inkling {

// Hugging Face `tokenizer.json` loader for Inkling 2 Mini (tiktoken-style
// BPE, 199998 base + 60 added = 200058).
//
// Faithful to the `tokenizers` Rust library pipeline used by the real
// tokenizer (verified against the file: normalizer null, pre_tokenizer
// Sequence[Split(o200k pattern, Isolated), ByteLevel(add_prefix_space=false)],
// BPE model with a 199998-entry vocab and [a, b]-pair merges, ByteLevel
// decoder with add_prefix_space=true):
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
// Special tokens (<|message_user|> 200000, <|message_model|> 200001,
// <|message_system|> 200002, <|message_tool|> 200003, <|content_text|>
// 200004, <|content_image|> 200005, <|content_model_end_sampling|> 200006
// (eos), <|content_thinking|> 200008, <|end_message|> 200010,
// <|content_tool_error|> 200022, <|content_xml|> 200024,
// <|begin_of_text|> 200028, <|content_invoke_tool_json|> 200049,
// <|unused_200054|> (image placeholder), <|content_invoke_tool_text|>
// 200057, <|endoftext|> 199999 (pad), plus <|unused...|> fillers) come from
// `added_tokens`, resolved by content with the well-known values above as
// fallbacks. Upstream adds no BOS/EOS (tokenizer_config has no bos_token,
// GGUF tokenizer.ggml.add_bos_token/add_eos_token are false), so encode()
// defaults to add_bos=false.
class Tokenizer {
 public:
  Tokenizer() = default;
  // Throws on missing/corrupt file.
  explicit Tokenizer(const std::string& tokenizer_json_path) { load(path_ = tokenizer_json_path); }
  void load(const std::string& tokenizer_json_path);

  bool loaded() const { return !id_to_token_.empty(); }
  int vocab_size() const { return vocab_size_; }
  // Explicit BOS/EOS overrides.
  void set_bos(int id) { bos_id_ = id; }
  void set_eos(int id) { eos_id_ = id; }

  // Encode text -> ids. add_bos prepends <|begin_of_text|> (default false:
  // upstream never emits it; the chat template carries no BOS).
  std::vector<int> encode(const std::string& text,
                          bool add_bos = false) const;
  // skip_special_tokens=true (HF decode default) drops added/special ids.
  std::string decode(const std::vector<int>& ids,
                     bool skip_special_tokens = true) const;

  int bos() const { return bos_id_; }
  int eos() const { return eos_id_; }
  int pad() const { return pad_id_; }
  bool is_special(int id) const { return special_ids_.count(id) != 0; }

 private:
  struct AddedToken {
    int id = -1;
    std::string content;
    bool special = true;
  };

  std::string path_;
  int vocab_size_ = 200058;
  int bos_id_ = 200028, eos_id_ = 200006, pad_id_ = 199999;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, int> merge_rank_;
  std::vector<AddedToken> added_tokens_;  // longest content first
  std::unordered_set<int> special_ids_;
  std::unordered_map<int, std::string> special_content_;

  // BPE over one byte-mapped pre-token piece; appends resulting ids.
  void bpe_piece(const std::string& piece, std::vector<int>& out) const;
};

}  // namespace Inkling
}  // namespace KokkosDNN

#endif
