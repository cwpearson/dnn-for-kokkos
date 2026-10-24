// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21ENCODER_HPP
#define KOKKOSDNN_QWENIMAGE21ENCODER_HPP

// Qwen3-VL-8B text-tower encoder for Qwen-Image-2.1 (text-to-image prompt
// conditioning). Ground truth (HF research): 36 layers, hidden 4096,
// 32 Q heads / 8 KV heads (GQA), head_dim 128, SwiGLU intermediate 12288,
// vocab 151936, RoPE theta 5000000 with M-RoPE sections [24, 20, 20]; the
// pipeline consumes hidden_states[-1] with the final RMSNorm NEUTRALIZED
// (pre-norm output). text_encoder/model.safetensors is a single 17.5 GB
// bf16 shard. T2I prompt template (NOT apply_chat_template):
//   <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>
//   \n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
// with left-padding; the first _drop_idx (system-token count) hidden states
// are dropped before conditioning (see qwen21_t2i_head() below).
//
// Tokenizer note: the in-tree tables generator under models/moss_image is
// MOSS-specific (vocab 248320) and MUST NOT be copied. The Qwen2 byte-level
// BPE tables (vocab 151936) must be regenerated from
// processor/tokenizer.json (11.4 MB) + tokenizer_config.json with the same
// tablegen procedure; stub-owned TODO until then.
//
// This header is complete as a Config + weights loader; the 36-layer
// forward (M-RoPE + GQA + SwiGLU, mirroring the MossImageEncoder block
// structure) is marked TODO and currently throws.
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <array>
#include <cstddef>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Handle.hpp"

#include "KokkosDNN_QwenImage21Config.hpp"

namespace KokkosDNN {
namespace QwenImage21Encoder {

// Qwen3-VL-8B text-tower constants.
struct Config {
  int num_layers = 36;
  int hidden_dim = 4096;
  int num_q_heads = 32;
  int num_kv_heads = 8;  // GQA
  int intermediate_dim = 12288;  // SwiGLU total (2 x gate branches)
  int vocab_size = 151936;
  double rope_theta = 5000000.0;
  std::array<int, 3> mrope_sections = {24, 20, 20};
  double norm_eps = 1e-6;

  int head_dim() const {
    return (num_q_heads > 0) ? hidden_dim / num_q_heads : 0;
  }
  int gate_dim() const { return intermediate_dim / 2; }

  static Config tiny_test() {
    Config c;
    c.num_layers = 2;
    c.hidden_dim = 32;
    c.num_q_heads = 4;  // head_dim 8
    c.num_kv_heads = 2;
    c.intermediate_dim = 48;
    c.vocab_size = 256;
    return c;
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("QwenImage21Encoder::Config::validate: " +
                                  m);
    };
    if (num_layers <= 0) bad("num_layers must be > 0");
    if (hidden_dim <= 0) bad("hidden_dim must be > 0");
    if (num_q_heads <= 0) bad("num_q_heads must be > 0");
    if (hidden_dim % num_q_heads != 0)
      bad("hidden_dim must be divisible by num_q_heads");
    if (num_kv_heads <= 0) bad("num_kv_heads must be > 0");
    if (num_q_heads % num_kv_heads != 0)
      bad("num_q_heads must be divisible by num_kv_heads");
    if (intermediate_dim <= 0 || intermediate_dim % 2 != 0)
      bad("intermediate_dim must be a positive even number");
    if (vocab_size <= 0) bad("vocab_size must be > 0");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (mrope_sections[0] + mrope_sections[1] + mrope_sections[2] !=
        head_dim())
      bad("mrope_sections must sum to head_dim");
    if (!(norm_eps > 0) || !std::isfinite(norm_eps))
      bad("norm_eps must be positive and finite");
  }
};

// ---- T2I prompt template (REAL) ----

// Fixed head: tokenize this alone (with the Qwen2 tokenizer, left-padded)
// and its token count is _drop_idx: the number of leading hidden states to
// drop (system-token count) before conditioning the DiT.
inline std::string qwen21_t2i_head() {
  return "<|im_start|>system\nComprehend and analyze the provided "
         "prompt.<|im_end|>\n<|im_start|>user\n";
}

// Fixed tail appended after the user prompt.
inline std::string qwen21_t2i_tail() {
  return "<|im_end|>\n<|im_start|>assistant\n";
}

// Full T2I prompt (raw template, NOT apply_chat_template).
inline std::string qwen21_format_prompt_t2i(const std::string& prompt) {
  return qwen21_t2i_head() + prompt + qwen21_t2i_tail();
}

// One expected checkpoint tensor: bare (suffix-matched) name + shape.
struct EncoderTensorSpec {
  std::string name;
  int64_t d0 = 0;
  int64_t d1 = 1;
  bool matrix = false;
};

// Canonical Qwen3-VL text-tower weight list for a Config. Key layout
// (suffix-matched; exact HF export prefixes may vary):
//   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight,
//   model.layers.{i}.self_attn.{q,k}_norm.weight,
//   model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight,
//   model.layers.{i}.mlp.{gate,up,down}_proj.weight,
//   model.embed_tokens.weight, model.norm.weight.
// (lm_head is tied/absent for the encoder path and not required.)
inline std::vector<EncoderTensorSpec> qwen21_encoder_expected_tensors(
    const Config& cfg) {
  cfg.validate();
  const int64_t D = cfg.hidden_dim, G = cfg.gate_dim(),
                Hd = cfg.head_dim(), Kv = (int64_t)cfg.num_kv_heads * Hd,
                V = cfg.vocab_size;
  std::vector<EncoderTensorSpec> ts;
  auto mat = [&](const std::string& n, int64_t r, int64_t c) {
    ts.push_back({n, r, c, true});
  };
  auto vec = [&](const std::string& n, int64_t r) {
    ts.push_back({n, r, 1, false});
  };
  mat("model.embed_tokens.weight", V, D);
  vec("model.norm.weight", D);
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string a =
        "model.layers." + std::to_string(i) + ".self_attn.";
    mat(a + "q_proj.weight", D, D);
    mat(a + "k_proj.weight", Kv, D);
    mat(a + "v_proj.weight", Kv, D);
    mat(a + "o_proj.weight", D, D);
    vec(a + "q_norm.weight", Hd);
    vec(a + "k_norm.weight", Hd);
    const std::string p = "model.layers." + std::to_string(i) + ".";
    vec(p + "input_layernorm.weight", D);
    vec(p + "post_attention_layernorm.weight", D);
    mat(p + "mlp.gate_proj.weight", G, D);
    mat(p + "mlp.up_proj.weight", G, D);
    mat(p + "mlp.down_proj.weight", D, G);
  }
  return ts;
}

template <class ExecSpace, class Scalar>
struct QwenImage21EncoderLayerWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> q_W;  // [D, D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> k_W;  // [Kv, D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> v_W;  // [Kv, D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> o_W;  // [D, D]
  Kokkos::View<Scalar*, ExecSpace> qn, kn;                     // [Hd]
  Kokkos::View<Scalar*, ExecSpace> in_norm, post_norm;         // [D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> gate_W;  // [G, D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> up_W;    // [G, D]
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> down_W;  // [D, G]
};

// Qwen3-VL text-tower weights: embed + final norm + 36 GQA/SwiGLU layers.
template <class ExecSpace, class Scalar>
struct QwenImage21EncoderWeights {
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;

  Config cfg;
  V2 embed;      // [V, D]
  V1 final_norm;  // [D] (NEUTRALIZED by the pipeline: pre-norm output used)
  std::vector<QwenImage21EncoderLayerWeights<ExecSpace, Scalar>> layers;

  void allocate(const Config& c) {
    c.validate();
    cfg = c;
    const size_t D = (size_t)c.hidden_dim, G = (size_t)c.gate_dim(),
                 Hd = (size_t)c.head_dim(),
                 Kv = (size_t)c.num_kv_heads * Hd, V = (size_t)c.vocab_size;
    auto mat = [&](const char* lb, size_t r, size_t k) {
      return V2(lb, r, k);
    };
    auto vec = [&](const char* lb, size_t n) { return V1(lb, n); };
    embed = mat("qwen21enc_embed", V, D);
    final_norm = vec("qwen21enc_final", D);
    layers.resize((size_t)c.num_layers);
    for (auto& l : layers) {
      l.q_W = mat("qwen21enc_q", D, D);
      l.k_W = mat("qwen21enc_k", Kv, D);
      l.v_W = mat("qwen21enc_v", Kv, D);
      l.o_W = mat("qwen21enc_o", D, D);
      l.qn = vec("qwen21enc_qn", Hd);
      l.kn = vec("qwen21enc_kn", Hd);
      l.in_norm = vec("qwen21enc_in", D);
      l.post_norm = vec("qwen21enc_post", D);
      l.gate_W = mat("qwen21enc_gate", G, D);
      l.up_W = mat("qwen21enc_up", G, D);
      l.down_W = mat("qwen21enc_down", D, G);
    }
  }

  void release() {
    embed = {};
    final_norm = {};
    layers.clear();
    layers.shrink_to_fit();
  }

  size_t resident_bytes() const {
    size_t n = 0;
    n += embed.extent(0) * embed.extent(1);
    n += final_norm.extent(0);
    for (const auto& l : layers) {
      n += l.q_W.extent(0) * l.q_W.extent(1);
      n += l.k_W.extent(0) * l.k_W.extent(1);
      n += l.v_W.extent(0) * l.v_W.extent(1);
      n += l.o_W.extent(0) * l.o_W.extent(1);
      n += l.qn.extent(0) + l.kn.extent(0);
      n += l.in_norm.extent(0) + l.post_norm.extent(0);
      n += l.gate_W.extent(0) * l.gate_W.extent(1);
      n += l.up_W.extent(0) * l.up_W.extent(1);
      n += l.down_W.extent(0) * l.down_W.extent(1);
    }
    return n * sizeof(Scalar);
  }

  void random_init(uint64_t seed = 7) {
    if (embed.extent(0) == 0)
      throw std::runtime_error(
          "QwenImage21EncoderWeights::random_init: allocate?");
    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    auto rnd2 = [&](auto& v) {
      auto h = Kokkos::create_mirror_view(Kokkos::HostSpace(), v);
      for (size_t i = 0; i < v.extent(0); ++i)
        for (size_t j = 0; j < v.extent(1); ++j)
          h(i, j) = static_cast<Scalar>(dist(eng));
      Kokkos::deep_copy(v, h);
    };
    auto ones1 = [&](auto& v) {
      Kokkos::deep_copy(v, static_cast<Scalar>(1.0f));
    };
    rnd2(embed);
    ones1(final_norm);
    for (auto& l : layers) {
      rnd2(l.q_W);
      rnd2(l.k_W);
      rnd2(l.v_W);
      rnd2(l.o_W);
      rnd2(l.gate_W);
      rnd2(l.up_W);
      rnd2(l.down_W);
      ones1(l.qn);
      ones1(l.kn);
      ones1(l.in_norm);
      ones1(l.post_norm);
    }
  }

  // Load the REAL single-shard text_encoder/model.safetensors (F32/BF16).
  // Suffix-matched names; throws on missing/unknown/dtype/shape mismatch.
  // Needs allocate() first.
  void load_safetensors(const std::string& shard_path) {
    if (embed.extent(0) == 0)
      throw std::runtime_error(
          "QwenImage21EncoderWeights::load_safetensors: allocate first");
    static_assert(std::is_same_v<Scalar, float> ||
                      std::is_same_v<Scalar,
                                     Kokkos::Experimental::bhalf_t>,
                  "QwenImage21EncoderWeights::load_safetensors: Scalar must "
                  "be float or bhalf_t");
    const std::vector<EncoderTensorSpec> specs =
        qwen21_encoder_expected_tensors(cfg);
    std::map<std::string, bool> found;
    for (const auto& s : specs) found[s.name] = false;
    std::ifstream f;
    auto entries = QwenImage21::Impl::qwen21_open_shard(f, shard_path);
    for (const auto& kv : entries) {
      const std::string& full = kv.first;
      const QwenImage21::Impl::Qwen21SafeEntry& en = kv.second;
      const EncoderTensorSpec* spec = nullptr;
      for (const auto& s : specs) {
        if (full.size() >= s.name.size() &&
            full.compare(full.size() - s.name.size(), s.name.size(), s.name) ==
                0 &&
            (full.size() == s.name.size() ||
             full[full.size() - s.name.size() - 1] == '.')) {
          if (!spec || s.name.size() > spec->name.size()) spec = &s;
        }
      }
      if (!spec)
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: unknown tensor '" +
            full + "' in '" + shard_path + "'");
      if (found[spec->name])
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: duplicate tensor '" +
            spec->name + "'");
      load_one_tensor(f, full, en, *spec);
      found[spec->name] = true;
    }
    for (const auto& s : specs) {
      if (!found[s.name])
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: missing tensor '" +
            s.name + "'");
    }
  }

 private:
  void load_one_tensor(std::ifstream& f, const std::string& full,
                       const QwenImage21::Impl::Qwen21SafeEntry& en,
                       const EncoderTensorSpec& s) {
    const size_t elsize = QwenImage21::Impl::qwen21_entry_elsize(en, full);
    const int64_t numel = QwenImage21::Impl::qwen21_entry_numel(en, full);
    const size_t want =
        s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
    if ((size_t)numel != want)
      throw std::runtime_error(
          "QwenImage21EncoderWeights::load_safetensors: tensor '" + full +
          "' has numel " + std::to_string(numel) + ", expected " +
          std::to_string(want) + " (" + s.name + ")");
    if (en.end < en.begin || en.end - en.begin != want * elsize)
      throw std::runtime_error(
          "QwenImage21EncoderWeights::load_safetensors: tensor '" + full +
          "' slice size mismatch");
    V2* dst2 = nullptr;
    V1* dst1 = nullptr;
    resolve_tensor(s.name, dst2, dst1);
    if (s.matrix) {
      if (!dst2 || (*dst2).extent(0) != (size_t)s.d0 ||
          (*dst2).extent(1) != (size_t)s.d1)
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: view mismatch '" +
            s.name + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          f, en, want, elsize, (*dst2).data());
    } else {
      if (!dst1 || (*dst1).extent(0) != (size_t)s.d0)
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: view mismatch '" +
            s.name + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          f, en, want, elsize, (*dst1).data());
    }
  }

  void resolve_tensor(const std::string& name, V2*& out2, V1*& out1) {
    out2 = nullptr;
    out1 = nullptr;
    if (name == "model.embed_tokens.weight") {
      out2 = &embed;
      return;
    }
    if (name == "model.norm.weight") {
      out1 = &final_norm;
      return;
    }
    const std::string lb = "model.layers.";
    if (name.compare(0, lb.size(), lb) == 0) {
      size_t dot = name.find('.', lb.size());
      int li = std::stoi(name.substr(lb.size(), dot - lb.size()));
      if (li < 0 || (size_t)li >= layers.size())
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: bad layer '" + name +
            "'");
      auto& l = layers[(size_t)li];
      std::string tail = name.substr(dot + 1);
      if (tail == "self_attn.q_proj.weight") {
        out2 = &l.q_W;
        return;
      }
      if (tail == "self_attn.k_proj.weight") {
        out2 = &l.k_W;
        return;
      }
      if (tail == "self_attn.v_proj.weight") {
        out2 = &l.v_W;
        return;
      }
      if (tail == "self_attn.o_proj.weight") {
        out2 = &l.o_W;
        return;
      }
      if (tail == "self_attn.q_norm.weight") {
        out1 = &l.qn;
        return;
      }
      if (tail == "self_attn.k_norm.weight") {
        out1 = &l.kn;
        return;
      }
      if (tail == "input_layernorm.weight") {
        out1 = &l.in_norm;
        return;
      }
      if (tail == "post_attention_layernorm.weight") {
        out1 = &l.post_norm;
        return;
      }
      if (tail == "mlp.gate_proj.weight") {
        out2 = &l.gate_W;
        return;
      }
      if (tail == "mlp.up_proj.weight") {
        out2 = &l.up_W;
        return;
      }
      if (tail == "mlp.down_proj.weight") {
        out2 = &l.down_W;
        return;
      }
    }
    throw std::runtime_error(
        "QwenImage21EncoderWeights::load_safetensors: cannot resolve '" + name +
        "'");
  }
};

// Encoder model: owns Weights. forward() (36-layer M-RoPE + GQA + SwiGLU
// over left-padded token ids, final norm NEUTRALIZED, drop first _drop_idx
// rows) is a TODO mirroring the MossImageEncoder block structure; it
// currently throws.
template <class ExecSpace, class Scalar>
class QwenImage21EncoderModel {
 public:
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using I1 = Kokkos::View<int*, ExecSpace>;
  using Weights = QwenImage21EncoderWeights<ExecSpace, Scalar>;

  explicit QwenImage21EncoderModel(const Config& cfg = Config()) {
    cfg.validate();
    cfg_ = cfg;
    weights_.allocate(cfg_);
  }

  const Config& config() const { return cfg_; }
  Weights& weights() { return weights_; }
  const Weights& weights() const { return weights_; }

  // tokens [B, S] int ids (left-padded) -> hidden [B, S, D] pre-norm output
  // (final RMSNorm neutralized); caller drops the first drop_idx rows.
  // TODO: implement the 36-layer M-RoPE/GQA/SwiGLU forward (golden test).
  void forward(const DNNHandle<ExecSpace>& handle, const I1& tokens,
               const V2& hidden, int drop_idx) {
    (void)handle;
    (void)tokens;
    (void)hidden;
    (void)drop_idx;
    cfg_.validate();
    throw std::runtime_error(
        "QwenImage21EncoderModel::forward: TODO (36-layer M-RoPE/GQA/SwiGLU "
        "forward not yet implemented; see MossImageEncoder pattern)");
  }

 private:
  Config cfg_;
  Weights weights_;
};

}  // namespace QwenImage21Encoder
}  // namespace KokkosDNN

#endif
