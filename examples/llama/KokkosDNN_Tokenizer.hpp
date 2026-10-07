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
// `meta-llama/Llama-3.3-8B-Instruct` tokenizer:
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

}  // namespace Llama
}  // namespace KokkosDNN

#endif
