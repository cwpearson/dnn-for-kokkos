#ifndef KOKKOSDNN_GEMMA4_TOKENIZER_HPP
#define KOKKOSDNN_GEMMA4_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KokkosDNN {
namespace Gemma4 {

// Hugging Face `tokenizer.json` loader for Gemma4 (SentencePiece-▁ BPE,
// 262144 + 24 added tokens).
//
// Faithful to the `tokenizers` Rust library pipeline used by the real
// `google/gemma-4-E2B-it` tokenizer:
//   1. literal special/added tokens are split out first (longest match,
//      never broken up by BPE);
//   2. remaining text is normalized with the Replace(" " -> "▁", U+2581)
//      normalizer, then pre-tokenized with Split(" ", MergedWithPrevious)
//      (after normalization no literal space remains outside added tokens,
//      so each segment is a single piece);
//   3. each piece is split into Unicode characters; characters missing from
//      the vocab fall back to their UTF-8 bytes as <0xXX> (uppercase hex)
//      byte tokens (byte_fallback=true; the vocab holds all 256);
//   4. BPE merges (model.merges [a, b] pairs, lowest rank first) apply per
//      piece; applied greedily until no pair merges;
//   5. decode concatenates token strings, expands <0xXX> byte tokens,
//      replaces ▁ with a space, and skips special tokens unless asked
//      otherwise (decoder sequence Replace(▁ -> " ") + ByteFallback +
//      Fuse).
//
// BOS/EOS conventions (tokenizer_config.json + text_config):
// bos=<bos> (id 2), eos=<eos> (id 1), pad=<pad> (id 0), unk=<unk> (id 3).
// The loader resolves bos/eos by content from `added_tokens` (falling back
// to ids 2/1), then applies a sibling tokenizer_config.json override
// (bos_token/eos_token string names, best-effort, never throws) when the
// tokenizer path contains a directory.
//
// Standalone: shares no headers with the Llama/SmolLM2 tokenizers; the
// JSON/UTF-8 helpers and the BPE core are duplicated here in the same
// style (only the core src/ ops are shared).
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
  int pad() const { return pad_id_; }
  int unk() const { return unk_id_; }
  bool is_special(int id) const { return special_ids_.count(id) != 0; }
 private:
  struct AddedToken {
    int id = -1;
    std::string content;
    bool special = true;
  };

  std::string path_;
  int vocab_size_ = 262144;
  int bos_id_ = 2, eos_id_ = 1, pad_id_ = 0, unk_id_ = 3;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, int> merge_rank_;
  std::vector<AddedToken> added_tokens_;  // longest content first
  std::unordered_set<int> special_ids_;
  std::unordered_map<int, std::string> special_content_;

  // BPE over one normalized pre-token piece; appends resulting ids.
  void bpe_piece(const std::string& piece, std::vector<int>& out) const;
};

// BOS handling: collapse any run of leading `bos_id` to exactly one when
// want_bos (or strip all leading copies when !want_bos). Guarantees the
// "exactly one BOS" invariant even when raw text itself starts with the
// BOS literal.
std::vector<int> ensure_bos(const std::vector<int>& ids, int bos_id,
                            bool want_bos);

// ---------------------------------------------------------------------------
// Stop-strings (checked on decoded text): truncate the generated text
// before the earliest occurrence of any non-empty stop string.
// ---------------------------------------------------------------------------
size_t find_stop_string(const std::string& text,
                        const std::vector<std::string>& stop_strings,
                        std::string* matched = nullptr);
std::string truncate_at_stop_strings(
    const std::string& text, const std::vector<std::string>& stop_strings);

}  // namespace Gemma4
}  // namespace KokkosDNN

#endif
