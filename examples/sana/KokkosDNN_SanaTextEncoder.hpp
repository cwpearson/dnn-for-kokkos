#ifndef KOKKOSDNN_SANA_TEXT_ENCODER_HPP
#define KOKKOSDNN_SANA_TEXT_ENCODER_HPP

// Native Gemma-2-2B-IT text encoder for the Sana example (FP32, Serial CPU
// correctness first; no KV cache — a single prompt encode per run).
//
// Reference: NVlabs/Sana `app/sana_pipeline.py` + `scripts/inference.py`
// (tokenize with padding/truncation to `model_max_length`, Gemma2Model
// last-hidden states, `select_index = [0] + [-299..-1]` gather) and HF
// transformers `modeling_gemma2.py` (exact layer math). Upstream checkpoint:
// `google/gemma-2-2b-it` (gated; safetensors + tokenizer.json + config.json).
//
// Gemma-2-2B-IT architecture (26 layers, hidden 2304):
//   * token embedding scaled by sqrt(2304) = 48
//   * per layer: input RMSNorm -> GQA self-attn (8 Q / 4 KV heads, head_dim
//     256, unbiased proj, scale = 256^-0.5, tanh logit softcap 50, RoPE
//     theta 10000, alternating sliding-4096 / full causal starting with
//     sliding) -> post-attention RMSNorm -> +residual -> pre-FFN RMSNorm ->
//     gated-GeLU(tanh) MLP (9216, unbiased) -> post-FFN RMSNorm -> +residual
//   * final RMSNorm; all norms are the Gemma2 `(1 + w)` variant, eps 1e-6
//   * vocab 256000, tie_word_embeddings (lm_head unused here)
// Sliding vs full attention is exact-identical for Sana's L = 300 tokens
// (300 < 4096 window), so every layer runs the same full-causal kernel;
// the padding mask is still honored (pad keys are never attended), matching
// the HF `attention_mask` path.
//
// Differences vs Gemma4-E2B (examples/gemma4): no PLE, no KV sharing, no
// double-wide MLP, no QK norms, no partial/proportional RoPE, attention
// scale is 1/sqrt(256) (not 1.0), single RoPE theta for all layers, and the
// MLP is unbiased gated-GeLU rather than Gemma4's biased projections.

#include <Kokkos_Core.hpp>

#include "KokkosDNN_Handle.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KokkosDNN {
namespace SanaGemma {

// FP32 throughout: the Sana DiT consumes float caption features and the
// encoder runs on Serial CPU, so no half-precision path is provided.
using DefaultScalar = float;

// Sana caption geometry produced by encode_prompt().
inline constexpr int kCaptionLength = 300;
inline constexpr int kCaptionDim = 2304;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Config {
  int hidden_size = 2304;
  int num_hidden_layers = 26;
  int num_attention_heads = 8;      // query heads
  int num_key_value_heads = 4;      // GQA kv heads
  int head_dim = 256;
  int intermediate_size = 9216;     // MLP width
  int vocab_size = 256000;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.0f;
  int sliding_window = 4096;        // even layers (0-based); odd layers full
  float query_pre_attn_scalar = 256.0f;  // attention scale = scalar^-0.5
  float attn_logit_softcapping = 50.0f;  // 0 disables (not the HF default)
  int max_position_embeddings = 8192;
  int bos_token_id = 2;
  int eos_token_id = 1;
  int pad_token_id = 0;
  bool attention_bias = false;      // must be false (no bias path in Model)
  bool tie_word_embeddings = true;  // informational (lm_head is not loaded)
  std::string hidden_activation = "gelu_pytorch_tanh";
  // Per-layer attention type: false = sliding, true = full. Default is the
  // Gemma2 alternation starting with sliding (layer 0 sliding).
  std::vector<char> is_full_layer;

  Config() : is_full_layer(default_layer_types(26)) {}

  static std::vector<char> default_layer_types(int n_layers) {
    std::vector<char> t((size_t)n_layers, 0);
    for (int i = 0; i < n_layers; ++i)
      if (i % 2 == 1) t[(size_t)i] = 1;
    return t;
  }

  static Config gemma2_2b_it() { return Config{}; }

  // Tiny config for unit tests (no checkpoint needed).
  static Config tiny_test() {
    Config c;
    c.hidden_size = 32;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.head_dim = 8;
    c.intermediate_size = 64;
    c.vocab_size = 256;
    c.sliding_window = 64;
    c.is_full_layer = {0, 1};
    c.max_position_embeddings = 64;
    return c;
  }

  bool is_full(int layer) const { return is_full_layer[(size_t)layer] != 0; }
  // Scaled token embedding (HF Gemma2TextScaledWordEmbedding): sqrt(hidden).
  // Exact in FP32 for hidden 2304 (48.0).
  float embed_scale() const { return std::sqrt((float)hidden_size); }
  float attn_scale() const;                     // query_pre_attn_scalar^-0.5

  void validate() const;

  // Parse an HF config.json object (flat Gemma2 keys). Accepts both the
  // legacy top-level `rope_theta` and the newer
  // `rope_parameters: {rope_theta, rope_type}` form. Unknown keys ignored.
  static Config from_json_string(const std::string& text,
                                 const Config& base = gemma2_2b_it());
  static Config from_json_file(const std::string& path,
                               const Config& base = gemma2_2b_it());
};

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wq, wk, wv, wo;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wgate, wup, wdown;
  Kokkos::View<Scalar*, ExecSpace> input_norm, post_attn_norm, pre_ffn_norm,
      post_ffn_norm;
};

template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> embed;
  Kokkos::View<Scalar*, ExecSpace> final_norm;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;

  void allocate(const Config& cfg);
};

template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load HF safetensors checkpoint(s) into device views (F32/F16/BF16 on disk
// are all converted to float). `files` holds one or more shards; later
// shards win on duplicate names. Throws on missing tensors or shape
// mismatch; the message names the first missing tensor (use
// list_checkpoint_tensors() to inspect a partial download).
//
// Expected HF names: `model.embed_tokens.weight`, `model.norm.weight`,
// `model.layers.{l}.{input_layernorm, post_attention_layernorm,
// pre_feedforward_layernorm, post_feedforward_layernorm}.weight`,
// `model.layers.{l}.self_attn.{q,k,v,o}_proj.weight`,
// `model.layers.{l}.mlp.{gate,up,down}_proj.weight`. No lm_head is needed
// (tied; the encoder only produces hidden states).
template <class ExecSpace, class Scalar = DefaultScalar>
void load_safetensors(const Config& cfg, const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w);

std::vector<std::string> list_checkpoint_tensors(
    const std::vector<std::string>& files);

// ---------------------------------------------------------------------------
// Tokenizer (HF `tokenizer.json`, SentencePiece-▁ BPE + byte fallback)
// ---------------------------------------------------------------------------
// Same pipeline as the `tokenizers` Rust library for the Gemma2 tokenizer:
// added/special tokens split out first (longest match), remaining text
// normalized (" " -> "▁" U+2581), BPE merges applied per piece with
// per-character UTF-8 start symbols and `<0xXX>` byte fallback. encode()
// with add_bos=true matches the Sana pipeline's
// `tokenizer(text, ..., add_special_tokens=True)` call (BOS id 2 prepended).
class Tokenizer {
 public:
  Tokenizer() = default;
  explicit Tokenizer(const std::string& tokenizer_json_path) {
    load(tokenizer_json_path);
  }
  void load(const std::string& tokenizer_json_path);

  bool loaded() const { return !id_to_token_.empty(); }
  int vocab_size() const { return vocab_size_; }

  std::vector<int> encode(const std::string& text,
                          bool add_bos = true) const;
  std::string decode(const std::vector<int>& ids,
                     bool skip_special_tokens = true) const;

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
  void bpe_piece(const std::string& piece, std::vector<int>& out) const;

  std::string path_;
  int vocab_size_ = 256000;
  int bos_id_ = 2, eos_id_ = 1, pad_id_ = 0, unk_id_ = 3;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, int> merge_rank_;
  std::vector<AddedToken> added_tokens_;  // longest content first
  std::unordered_set<int> special_ids_;
  std::unordered_map<int, std::string> special_content_;
};

// ---------------------------------------------------------------------------
// Model (single prompt encode, no KV cache)
// ---------------------------------------------------------------------------
template <class ExecSpace, class Scalar = DefaultScalar>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using ViewChar = Kokkos::View<char*, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;

  Model() = default;
  Model(const Config& c, const Weights<ExecSpace, Scalar>& w);

  // Forward over T token ids with a key-valid mask (1 = real token, 0 =
  // pad; pad keys are never attended, matching the HF attention_mask path).
  // Returns row-major [T, hidden] last-hidden states on host. Scratch grows
  // on demand and is reused across calls (encode the prompt and the null
  // prompt with the same Model).
  std::vector<float> forward(const std::vector<int>& ids,
                             const std::vector<char>& key_valid);

 private:
  void ensure_scratch(size_t T);
  DNNHandle<ExecSpace> handle_;
  Kokkos::View<float*, ExecSpace> inv_freq_;  // [head_dim/2], theta
  MatR h_, tmp_, tmp2_, q2_, k2_, v2_, af_, proj_, gate_, up_, act_, down_;
  View3 q3_, k3_, v3_, ao_;
  ViewInt pos_, tok_;
  ViewChar valid_;
  size_t scratch_cap_ = 0;
};

// Sana's caption gather: keep token 0 (BOS) plus the last (max_length - 1)
// tokens of a length_all tokenize, i.e. [0] + [-(max_length-1)..-1].
// For length_all == max_length this is the identity [0..max_length).
std::vector<int> sana_select_index(int length_all, int max_length);

// Full Sana text-encoder path for one prompt: tokenize with right-padding /
// right-truncation to max_length_all (300, or chi-adjusted when chi_prompt
// is non-empty), Gemma forward with the padding mask, then the
// select_index gather. Returns row-major [max_length, hidden] features
// matching the pipeline's `caption_embs[:, 0, select_index, :]` layout.
template <class ExecSpace>
std::vector<float> encode_prompt(const Config& cfg, Tokenizer& tok,
                                 Model<ExecSpace>& model,
                                 const std::string& prompt,
                                 int max_length = kCaptionLength,
                                 const std::string& chi_prompt = "") {
  if (max_length <= 0)
    throw std::invalid_argument("encode_prompt: max_length must be > 0");
  if ((int)model.cfg.hidden_size != cfg.hidden_size)
    throw std::invalid_argument(
        "encode_prompt: model/config hidden_size mismatch");
  // Sana chi_prompt path (app/sana_pipeline.py): prepend "\n"-joined
  // instructions and widen the tokenize window by the chi token count
  // (tokenizer.encode includes BOS; magic -2 drops [bos] and one spare).
  int max_length_all = max_length;
  std::string full = prompt;
  if (!chi_prompt.empty()) {
    full = chi_prompt + prompt;
    const std::vector<int> chi_ids = tok.encode(chi_prompt, true);
    max_length_all = (int)chi_ids.size() + max_length - 2;
    if (max_length_all < max_length) max_length_all = max_length;
  }
  // Tokenize with right-truncation / right-padding to max_length_all,
  // mirroring tokenizer(..., padding="max_length", truncation=True).
  std::vector<int> ids = tok.encode(full, true);
  if ((int)ids.size() > max_length_all)
    ids.resize((size_t)max_length_all);
  const int length_all = (int)ids.size();
  std::vector<char> key_valid((size_t)length_all, 1);
  ids.resize((size_t)max_length_all, tok.pad());
  key_valid.resize((size_t)max_length_all, 0);
  std::vector<float> hidden =
      model.forward(ids, key_valid);  // [max_length_all, hidden]
  const std::vector<int> sel = sana_select_index(max_length_all, max_length);
  std::vector<float> out((size_t)max_length * (size_t)cfg.hidden_size);
  for (int i = 0; i < max_length; ++i) {
    const float* src =
        hidden.data() + (size_t)sel[(size_t)i] * (size_t)cfg.hidden_size;
    float* dst = out.data() + (size_t)i * (size_t)cfg.hidden_size;
    for (int d = 0; d < cfg.hidden_size; ++d) dst[d] = src[d];
  }
  return out;
}

// Checkpoint path helpers: `weights_path` may be a single .safetensors file
// or a directory (all `*.safetensors` inside, sorted). sibling_file() looks
// next to a weights file (or inside a weights dir) for config/tokenizer.
std::vector<std::string> collect_safetensors(const std::string& path);
std::string sibling_file(const std::string& weights_path,
                         const std::string& name);

}  // namespace SanaGemma
}  // namespace KokkosDNN

#endif
