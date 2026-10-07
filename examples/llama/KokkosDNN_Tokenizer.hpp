#ifndef KOKKOSDNN_TOKENIZER_HPP
#define KOKKOSDNN_TOKENIZER_HPP

#include <map>
#include <string>
#include <vector>

namespace KokkosDNN {
namespace Llama {

// Minimal HF tokenizer.json loader for Llama 3 (tiktoken BPE, 128k).
// Loads "vocab" (token->id) and "merges" (ranked "a b" pairs) plus
// added tokens. encode() implements byte-level BPE with BOS handling;
// decode() concatenates token strings with byte-fallback repair.
//
// Special tokens (Llama 3): BOS=128000, EOS=128001/128009, plus
// <|start_header_id|> etc. in 128006..128010. Only BOS/EOS are treated
// specially here; the rest round-trip as regular vocab entries when
// present in tokenizer.json.
class Tokenizer {
 public:
  Tokenizer() = default;
  // Throws on missing/corrupt file.
  explicit Tokenizer(const std::string& tokenizer_json_path) { load(path_ = tokenizer_json_path); }
  void load(const std::string& tokenizer_json_path);

  bool loaded() const { return !id_to_token_.empty(); }
  int vocab_size() const { return vocab_size_; }

  // Encode text -> ids. add_bos prepends 128000 (default true, Llama style).
  std::vector<int> encode(const std::string& text, bool add_bos = true) const;
  std::string decode(const std::vector<int>& ids) const;

  int bos() const { return 128000; }
  int eos() const { return 128001; }

 private:
  std::string path_;
  int vocab_size_ = 128256;
  std::vector<std::string> id_to_token_;
  std::map<std::string, int> token_to_id_;
  std::map<std::string, int> merge_rank_;
};

}  // namespace Llama
}  // namespace KokkosDNN

#endif
