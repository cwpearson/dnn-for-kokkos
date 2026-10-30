// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21ENCODER_HPP
#define KOKKOSDNN_QWENIMAGE21ENCODER_HPP

// Qwen3-VL-8B text-tower encoder for Qwen-Image-2.1 (text-to-image prompt
// conditioning). Ground truth (HF research): 36 layers, hidden 4096,
// 32 Q heads / 8 KV heads (GQA), head_dim 128, SwiGLU intermediate 12288,
// vocab 151936, RoPE theta 5000000 with M-RoPE sections [24, 20, 20]; the
// pipeline consumes hidden_states[-1] PRE-NORM (the last decoder-layer
// output; the model's final RMSNorm is not applied in this environment).
// Skipping the final norm here is therefore correct, not a neutralized
// approximation. text_encoder/model.safetensors is a single 17.5 GB
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
// This header is complete as a Config + weights loader + the 36-layer
// forward (M-RoPE-as-plain-RoPE for text-only input + GQA + SwiGLU,
// mirroring the MossImageEncoder block structure and verified against
// modeling_qwen3_vl.py).
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Embedding.hpp"
#include "KokkosDNN_Gemm.hpp"
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
  int intermediate_dim = 12288;  // SwiGLU hidden (gate/up: [I, D])
  int vocab_size = 151936;
  double rope_theta = 5000000.0;
  std::array<int, 3> mrope_sections = {24, 20, 20};  // sums to head_dim/2
  double norm_eps = 1e-6;

  int head_dim() const {
    return (num_q_heads > 0) ? hidden_dim / num_q_heads : 0;
  }
  int gate_dim() const { return intermediate_dim; }

  static Config tiny_test() {
    Config c;
    c.num_layers = 2;
    c.hidden_dim = 32;
    c.num_q_heads = 4;  // head_dim 8
    c.num_kv_heads = 2;
    c.intermediate_dim = 48;
    c.vocab_size = 256;
    c.mrope_sections = {2, 1, 1};  // sums to head_dim/2 = 4
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
    if (head_dim() % 2 != 0) bad("head_dim must be even (rotary pairs)");
    if (intermediate_dim <= 0 || intermediate_dim % 2 != 0)
      bad("intermediate_dim must be a positive even number");
    if (vocab_size <= 0) bad("vocab_size must be > 0");
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (mrope_sections[0] + mrope_sections[1] + mrope_sections[2] !=
        head_dim() / 2)
      bad("mrope_sections must sum to head_dim/2");
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

// Canonical Qwen3-VL text-tower weight list for a Config. Checkpoint keys
// live under model.language_model.* (normalized at load); the extra shard
// members (lm_head.weight, model.visual.*) are skipped, not required here.
//   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight,
//   model.layers.{i}.self_attn.{q,k}_norm.weight,
//   model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight,
//   model.layers.{i}.mlp.{gate,up,down}_proj.weight,
//   model.embed_tokens.weight, model.norm.weight.
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
  // The shard holds the full Qwen3VL checkpoint: text tensors live under
  // "model.language_model.*", plus "lm_head.weight" and the vision tower
  // ("model.visual.*"). Names are matched after stripping an optional
  // ".language_model" infix; lm_head/visual keys are skipped (the T2I text
  // path never reads them). Throws on missing/duplicate/dtype/shape
  // mismatch, and on any other unknown tensor. Needs allocate() first.
  // Pass a warmed QwenImage21::ShardCache to stream a staged shard from
  // host RAM instead of disk (repeat loads); the default nullptr reads
  // from disk, so existing one-argument callers are unchanged.
  void load_safetensors(const std::string& shard_path,
                        QwenImage21::ShardCache* cache = nullptr) {
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
    const QwenImage21::Impl::Qwen21ShardBlob* blob =
        (cache != nullptr) ? cache->find(shard_path) : nullptr;
    const std::map<std::string, QwenImage21::Impl::Qwen21SafeEntry> entries =
        (blob != nullptr)
            ? QwenImage21::Impl::qwen21_parse_shard_blob(blob->bytes,
                                                        shard_path)
            : QwenImage21::Impl::qwen21_open_shard(f, shard_path);
    for (const auto& kv : entries) {
      const std::string& full = kv.first;
      const QwenImage21::Impl::Qwen21SafeEntry& en = kv.second;
      const std::string norm = normalize_key(full);
      const EncoderTensorSpec* spec = nullptr;
      for (const auto& s : specs) {
        if (norm == s.name) {
          spec = &s;
          break;
        }
      }
      if (!spec) {
        if (ignorable_key(full)) continue;
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: unknown tensor '" +
            full + "' in '" + shard_path + "'");
      }
      if (found[spec->name])
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: duplicate tensor '" +
            spec->name + "'");
      if (blob != nullptr) {
        load_one_tensor(*blob, full, en, *spec);
      } else {
        load_one_tensor(f, full, en, *spec);
      }
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
  // "model.language_model.layers.N...." -> "model.layers.N...." so the
  // canonical spec names match the real checkpoint layout.
  static std::string normalize_key(const std::string& full) {
    const std::string infix = ".language_model.";
    const size_t p = full.find(infix);
    if (p == std::string::npos) return full;
    return full.substr(0, p + 1) + full.substr(p + infix.size());
  }

  // Shard members the T2I text path never reads: the tied-absent lm head
  // and the whole vision tower.
  static bool ignorable_key(const std::string& full) {
    if (full == "lm_head.weight") return true;
    return full.compare(0, 13, "model.visual.") == 0;
  }

  // Stream one file tensor into its device view (shape-checked). Source is
  // either a disk stream or a staged shard image (overload resolution on
  // the first argument picks the matching qwen21_stream_into).
  template <class Source>
  void load_one_tensor(Source& src, const std::string& full,
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
          src, en, want, elsize, (*dst2).data());
    } else {
      if (!dst1 || (*dst1).extent(0) != (size_t)s.d0)
        throw std::runtime_error(
            "QwenImage21EncoderWeights::load_safetensors: view mismatch '" +
            s.name + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          src, en, want, elsize, (*dst1).data());
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

namespace Impl {

// Host inv_freq table: inv[j] = theta^(-2j/Hd), j < Hd/2. Matches
// Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters with
// dim = head_dim (denominator is the full head_dim, not Hd/2).
inline std::vector<float> qwen21_rope_inv_host(int head_dim, double theta) {
  const size_t Rh = static_cast<size_t>(head_dim) / 2;
  std::vector<float> out(Rh);
  const double Hd = static_cast<double>(head_dim);
  for (size_t j = 0; j < Rh; ++j)
    out[j] =
        static_cast<float>(1.0 / std::pow(theta, (2.0 * (double)j) / Hd));
  return out;
}

// Full-dim rotary apply, in place, HF Qwen3VL half-rotation pairing:
// (a, b) = (x[d], x[d + Hd/2]), angle = pos * inv[d], pos = token index.
// For text-only input all M-RoPE rows (T/H/W) equal arange(S), so the
// interleaved recomposition reproduces these exact frequencies and this
// plain RoPE is bit-faithful (verified: identical angles to the torch
// rotary_emb forward at equal row positions). x: [T, H, Hd] contiguous.
template <class ExecSpace, class XView, class FView>
void qwen21_rope_inplace(const XView& x, const FView& inv_freq,
                         const std::string& label = "Qwen21Enc::rope") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (D % 2 != 0)
    throw std::invalid_argument("qwen21_rope_inplace: head_dim must be even");
  const size_t Rh = D / 2;
  if (inv_freq.extent(0) != Rh)
    throw std::invalid_argument("qwen21_rope_inplace: inv_freq dim");
  if (T == 0 || H == 0) return;
  using Scalar = std::remove_const_t<typename XView::value_type>;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const float pos = static_cast<float>(t);
        for (size_t j = 0; j < Rh; ++j) {
          const float ang = pos * inv_freq(j);
          const float c = Kokkos::cosf(ang), s = Kokkos::sinf(ang);
          const float a = static_cast<float>(x(t, h, j));
          const float b = static_cast<float>(x(t, h, j + Rh));
          x(t, h, j) = static_cast<Scalar>(a * c - b * s);
          x(t, h, j + Rh) = static_cast<Scalar>(a * s + b * c);
        }
      });
}

// True M-RoPE apply for joint text+vision input: per-token (t, h, w)
// positions select the frequency row by interleaved section (reference
// recomposition: freqs_thw starts as the T row; H overwrites idx 1,4,...
// up to 3*mrope_section[1], W overwrites idx 2,5,... up to
// 3*mrope_section[2]; slots at/above the cutoffs stay T, so with
// [24,20,20] the tail slots 60..63 all read T). Same half-rotation pairing
// as qwen21_rope_inplace. x: [S, Hq, Hd]; pos: [3, S] int (t/h/w rows).
template <class ExecSpace, class XView, class PView, class FView>
void qwen21_mrope_inplace(const XView& x, const PView& pos,
                          const FView& inv_freq,
                          const std::array<int, 3>& sections,
                          const std::string& label = "Qwen21Enc::mrope") {
  const size_t S = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (D % 2 != 0)
    throw std::invalid_argument("qwen21_mrope_inplace: head_dim must be even");
  const size_t Rh = D / 2;
  if (inv_freq.extent(0) != Rh)
    throw std::invalid_argument("qwen21_mrope_inplace: inv_freq dim");
  if (pos.extent(0) != 3 || pos.extent(1) != S)
    throw std::invalid_argument("qwen21_mrope_inplace: pos must be [3, S]");
  if (sections[0] + sections[1] + sections[2] != static_cast<int>(Rh))
    throw std::invalid_argument("qwen21_mrope_inplace: sections must sum to "
                                "head_dim/2");
  if (S == 0 || H == 0) return;
  using Scalar = std::remove_const_t<typename XView::value_type>;
  const int s1 = sections[1];
  const int s2 = sections[2];
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {S, H}),
      KOKKOS_LAMBDA(size_t s, size_t h) {
        for (size_t j = 0; j < Rh; ++j) {
          const size_t m = j % 3;
          const int jj = static_cast<int>(j);
          size_t r = 0;
          if (m == 1 && jj < 3 * s1)
            r = 1;
          else if (m == 2 && jj < 3 * s2)
            r = 2;
          const float p = static_cast<float>(pos(r, s));
          const float ang = p * inv_freq(j);
          const float c = Kokkos::cosf(ang), s_ = Kokkos::sinf(ang);
          const float a = static_cast<float>(x(s, h, j));
          const float b = static_cast<float>(x(s, h, j + Rh));
          x(s, h, j) = static_cast<Scalar>(a * c - b * s_);
          x(s, h, j + Rh) = static_cast<Scalar>(a * s_ + b * c);
        }
      });
}

// Torch-exact RMSNorm: replicates Qwen3VLTextRMSNorm.forward's bf16
// intermediate rounding (hidden.to(bf16) BEFORE the weight multiply, i.e.
// two roundings) instead of the single-FP32-rounding core op. For float
// Scalar this is numerically identical to KokkosDNN::rmsnorm; for bf16 it
// matches the reference bit-for-bit on the rounding, which matters: the
// layer-6 attention sink (~1e4, bf16 ulp 64) amplifies a last-bit norm
// difference into macroscopic row-sum drift (verified: single-layer
// maxabs 64.0 -> 0.125 on the golden prompt).
template <class ExecSpace, class Scalar>
void qwen21_rmsnorm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "Qwen21Enc::rmsnorm") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (y.extent(0) != T || y.extent(1) != D || w.extent(0) != D)
    throw std::invalid_argument("qwen21_rmsnorm: dimension mismatch");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        float ss = 0;
        for (size_t k = 0; k < D; ++k) {
          float v = static_cast<float>(x(t, k));
          ss += v * v;
        }
        float inv = 1.0f / Kokkos::sqrtf(ss / (float)D + eps);
        Scalar nb =
            static_cast<Scalar>(static_cast<float>(x(t, d)) * inv);
        y(t, d) = static_cast<Scalar>(static_cast<float>(nb) *
                                      static_cast<float>(w(d)));
      });
}

// Torch-exact per-head QK RMSNorm: same double rounding as above,
// out[t,h,d] = w[d] * Scalar(flat[t,h*D+d] * inv(t,h)).
template <class ExecSpace, class Scalar>
void qwen21_qk_rmsnorm(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& flat,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "Qwen21Enc::qk_norm") {
  const size_t T = out.extent(0), H = out.extent(1), D = out.extent(2);
  if (flat.extent(0) != T || flat.extent(1) != H * D || w.extent(0) != D)
    throw std::invalid_argument("qwen21_qk_rmsnorm: dimension mismatch");
  if (T == 0 || H == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(flat(t, h * D + d));
          ss += v * v;
        }
        float inv = 1.0f / Kokkos::sqrtf(ss / (float)D + eps);
        for (size_t d = 0; d < D; ++d) {
          Scalar nb = static_cast<Scalar>(
              static_cast<float>(flat(t, h * D + d)) * inv);
          out(t, h, d) = static_cast<Scalar>(static_cast<float>(nb) *
                                             static_cast<float>(w(d)));
        }
      });
}

// Torch-exact SwiGLU: the elementwise bf16 SiLU rounds the sigmoid before
// the gate*up product (two roundings), out = Scalar(Scalar(silu(g)) * u).
// For float Scalar identical to KokkosDNN::swiglu.
template <class ExecSpace, class Scalar>
void qwen21_swiglu(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "Qwen21Enc::swiglu") {
  const size_t T = gate.extent(0), F = gate.extent(1);
  if (up.extent(0) != T || up.extent(1) != F || out.extent(0) != T ||
      out.extent(1) != F)
    throw std::invalid_argument("qwen21_swiglu: dimension mismatch");
  if (T == 0 || F == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, F}),
      KOKKOS_LAMBDA(size_t t, size_t f) {
        float g = static_cast<float>(gate(t, f));
        Scalar sg = static_cast<Scalar>(g / (1.0f + Kokkos::expf(-g)));
        out(t, f) = static_cast<Scalar>(static_cast<float>(sg) *
                                        static_cast<float>(up(t, f)));
      });
}

// Residual add, in place on x: x += branch. Same FP32-accumulate idiom as
// the MossImageEncoder block.
template <class ExecSpace, class XView, class BView>
void qwen21_residual_add(const XView& x, const BView& branch,
                         const std::string& label = "Qwen21Enc::residual") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (branch.extent(0) != T || branch.extent(1) != D)
    throw std::invalid_argument("qwen21_residual_add: dim mismatch");
  if (T == 0 || D == 0) return;
  using Scalar = std::remove_const_t<typename XView::value_type>;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        x(t, d) = static_cast<Scalar>(static_cast<float>(x(t, d)) +
                                      static_cast<float>(branch(t, d)));
      });
}

}  // namespace Impl

// Encoder model: owns Weights. forward() runs the 36-layer M-RoPE + GQA +
// SwiGLU text tower over token ids (T2I path: no visual slots; the
// image_pad mask is all-False and needs no plumbing here). forward_joint()
// below adds the edit path: vision-slot embedding replacement, true M-RoPE
// positions, and DeepStack adds (Qwen3-VL joint text+vision conditioning).
//
// Reference math (modeling_qwen3_vl.py, Qwen3VLText*):
//   h = embed[ids]
//   repeat per layer:
//     h += o_proj(GQA(rope(qn(q_proj(n)), kn(k_proj(n))), v_proj(n))))
//        with n = RMSNorm(h, in), qn/kn = per-head RMSNorm(Hd),
//        rope = half-rotation pairs, scale = 1/sqrt(Hd), causal, KV heads
//        repeated x num_q_heads/num_kv_heads; no attention bias.
//     h += down_proj(silu(gate_proj(m)) * up_proj(m))
//        with m = RMSNorm(h, post).
//   NO final RMSNorm: the pipeline neutralizes model.norm and consumes the
//   pre-norm output (transformers>=5.0 ties hidden_states[-1] to the
//   post-norm value, ~1/3 the magnitude — not what the DiT was trained on).
template <class ExecSpace, class Scalar>
class QwenImage21EncoderModel {
 public:
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using V3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using I1 = Kokkos::View<int*, ExecSpace>;
  using I2 = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace>;
  using Weights = QwenImage21EncoderWeights<ExecSpace, Scalar>;

  explicit QwenImage21EncoderModel(const Config& cfg = Config()) {
    cfg.validate();
    cfg_ = cfg;
    weights_.allocate(cfg_);
    auto hv = Impl::qwen21_rope_inv_host(cfg_.head_dim(), cfg_.rope_theta);
    rope_inv_ = Kokkos::View<float*, ExecSpace>("qwen21enc_rope_inv", hv.size());
    auto hm = Kokkos::create_mirror_view(rope_inv_);
    for (size_t i = 0; i < hv.size(); ++i) hm(i) = hv[i];
    Kokkos::deep_copy(rope_inv_, hm);
  }

  const Config& config() const { return cfg_; }
  Weights& weights() { return weights_; }
  const Weights& weights() const { return weights_; }

  // tokens [B, S] int ids (left-padded per the pipeline; every row is
  // consumed as-is with positions 0..S-1, exactly the text-only positions
  // the reference builds) -> hidden [B, S, D] pre-norm output (the caller
  // drops the first drop_idx rows). Exact for unpadded batches (the T2I
  // golden is B=1, no pads); left-pad rows are NOT masked, so padded
  // multi-prompt batches would see pad keys — pad-mask plumbing is an
  // explicit follow-up, not wired here.
  void forward(const DNNHandle<ExecSpace>& handle, const I2& tokens,
               const V3& hidden, int drop_idx) {
    cfg_.validate();
    const size_t B = tokens.extent(0), S = tokens.extent(1);
    const size_t D = (size_t)cfg_.hidden_dim;
    const size_t Nq = (size_t)cfg_.num_q_heads, Nk = (size_t)cfg_.num_kv_heads;
    const size_t Hd = (size_t)cfg_.head_dim(), G = (size_t)cfg_.gate_dim();
    if (hidden.extent(0) != B || hidden.extent(1) != S ||
        hidden.extent(2) != D)
      throw std::invalid_argument("QwenImage21EncoderModel::forward: hidden "
                                  "must be [B, S, D]");
    if (drop_idx < 0 || (size_t)drop_idx > S)
      throw std::invalid_argument(
          "QwenImage21EncoderModel::forward: drop_idx out of range");
    if (weights_.embed.extent(0) == 0)
      throw std::logic_error(
          "QwenImage21EncoderModel::forward: weights empty");
    if (B == 0 || S == 0) return;
    // Host-side id range check (one mirror per forward; negligible vs a
    // multi-GB encoder pass, and required for gather safety).
    {
      auto hm = Kokkos::create_mirror_view(tokens);
      Kokkos::deep_copy(hm, tokens);
      for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
          if (hm(b, s) < 0 || hm(b, s) >= cfg_.vocab_size)
            throw std::invalid_argument(
                "QwenImage21EncoderModel::forward: token id out of range");
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(Hd));
    const float eps = static_cast<float>(cfg_.norm_eps);
    using UView3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                                Kokkos::MemoryUnmanaged>;
    I1 row_ids("qwen21enc_ids", S);
    V2 h("qwen21enc_h", S, D);
    V2 n("qwen21enc_n", S, D);
    V2 qf("qwen21enc_q", S, Nq * Hd);
    V2 kf("qwen21enc_k", S, Nk * Hd);
    V2 vf("qwen21enc_v", S, Nk * Hd);
    V2 ao("qwen21enc_ao", S, D);
    V2 o("qwen21enc_o", S, D);
    V2 g("qwen21enc_g", S, G);
    V2 u("qwen21enc_u", S, G);
    V2 d("qwen21enc_d", S, D);
    UView3 q3(qf.data(), S, Nq, Hd);
    UView3 k3(kf.data(), S, Nk, Hd);
    UView3 v3(vf.data(), S, Nk, Hd);
    UView3 ao3(ao.data(), S, Nq, Hd);
    for (size_t b = 0; b < B; ++b) {
      Kokkos::View<int*, ExecSpace, Kokkos::MemoryUnmanaged> rowpre(
          tokens.data() + b * S, S);
      Kokkos::deep_copy(row_ids, rowpre);
      KokkosDNN::embedding_gather<ExecSpace, Scalar>(weights_.embed, row_ids,
                                                     h, "Qwen21Enc::embed");
      for (int l = 0; l < cfg_.num_layers; ++l) {
        const auto& L = weights_.layers[(size_t)l];
        Impl::qwen21_rmsnorm<ExecSpace, Scalar>(h, L.in_norm, eps, n,
                                     "Qwen21Enc::in_norm");
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.q_W,
                                           Scalar(0), qf, "Qwen21Enc::q");
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.k_W,
                                           Scalar(0), kf, "Qwen21Enc::k");
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.v_W,
                                           Scalar(0), vf, "Qwen21Enc::v");
        Impl::qwen21_qk_rmsnorm<ExecSpace, Scalar>(qf, L.qn, eps, q3,
                                        "Qwen21Enc::q_norm");
        Impl::qwen21_qk_rmsnorm<ExecSpace, Scalar>(kf, L.kn, eps, k3,
                                        "Qwen21Enc::k_norm");
        Impl::qwen21_rope_inplace<ExecSpace>(q3, rope_inv_, "Qwen21Enc::rope_q");
        Impl::qwen21_rope_inplace<ExecSpace>(k3, rope_inv_, "Qwen21Enc::rope_k");
        KokkosDNN::attention_gqa_single<ExecSpace, Scalar>(
            handle, q3, k3, v3, size_t{0}, ao3, scale, "Qwen21Enc::attn");
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), ao, L.o_W,
                                           Scalar(0), o, "Qwen21Enc::o");
        Impl::qwen21_residual_add<ExecSpace>(h, o, "Qwen21Enc::res1");
        Impl::qwen21_rmsnorm<ExecSpace, Scalar>(h, L.post_norm, eps, n,
                                     "Qwen21Enc::post_norm");
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.gate_W,
                                           Scalar(0), g, "Qwen21Enc::gate");
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.up_W,
                                           Scalar(0), u, "Qwen21Enc::up");
        Impl::qwen21_swiglu<ExecSpace, Scalar>(g, u, g, "Qwen21Enc::swiglu");
        KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), g, L.down_W,
                                           Scalar(0), d, "Qwen21Enc::down");
        Impl::qwen21_residual_add<ExecSpace>(h, d, "Qwen21Enc::res2");
      }
      // Final RMSNorm NEUTRALIZED: copy the pre-norm output (pipeline hook
      // parity). drop_idx rows are left for the caller to drop.
      Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryUnmanaged>
          dst(hidden.data() + b * S * D, S, D);
      Kokkos::deep_copy(dst, h);
    }
    Kokkos::fence();
  }

  // Joint text+vision forward for edit conditioning (B=1): token ids with
  // Nv vision slots at vis_rows (ascending S positions, count == Nv) get
  // their embeddings replaced by vis_embeds[0, k] in order; text layers
  // 0..2 add deep_embeds[k][0, k] at the same rows (DeepStack,
  // modeling_qwen3_vl.py _deepstack_process); positions are true M-RoPE
  // (pos3 flattened row-major [3, S], caller-built per get_rope_index, B=1
  // unpadded so every row is consumed as-is). Output is the pre-norm
  // hidden states (pipeline hook parity); the caller drops the first
  // drop_idx rows. The text-only forward() above is untouched (T2I path).
  void forward_joint(const DNNHandle<ExecSpace>& handle, const I2& tokens,
                     const I1& vis_rows, const V3& vis_embeds,
                     const V3 deep_embeds[3], const I1& pos3_flat,
                     const V3& hidden, int drop_idx) {
    cfg_.validate();
    const size_t B = tokens.extent(0), S = tokens.extent(1);
    const size_t D = (size_t)cfg_.hidden_dim;
    const size_t Nq = (size_t)cfg_.num_q_heads, Nk = (size_t)cfg_.num_kv_heads;
    const size_t Hd = (size_t)cfg_.head_dim(), G = (size_t)cfg_.gate_dim();
    if (B != 1)
      throw std::invalid_argument(
          "QwenImage21EncoderModel::forward_joint: batch must be 1");
    const size_t Nv = vis_rows.extent(0);
    if (vis_embeds.extent(0) != 1 || vis_embeds.extent(1) != Nv ||
        vis_embeds.extent(2) != D)
      throw std::invalid_argument(
          "QwenImage21EncoderModel::forward_joint: vis_embeds must be "
          "[1, Nv, D]");
    for (int k = 0; k < 3; ++k)
      if (deep_embeds[k].extent(0) != 1 || deep_embeds[k].extent(1) != Nv ||
          deep_embeds[k].extent(2) != D)
        throw std::invalid_argument(
            "QwenImage21EncoderModel::forward_joint: deep_embeds must be "
            "[1, Nv, D]");
    if (pos3_flat.extent(0) != 3 * S)
      throw std::invalid_argument(
          "QwenImage21EncoderModel::forward_joint: pos3 must hold 3*S ints");
    if (hidden.extent(0) != 1 || hidden.extent(1) != S ||
        hidden.extent(2) != D)
      throw std::invalid_argument(
          "QwenImage21EncoderModel::forward_joint: hidden must be [1, S, D]");
    if (drop_idx < 0 || (size_t)drop_idx > S)
      throw std::invalid_argument(
          "QwenImage21EncoderModel::forward_joint: drop_idx out of range");
    if (weights_.embed.extent(0) == 0)
      throw std::logic_error(
          "QwenImage21EncoderModel::forward_joint: weights empty");
    if (S == 0) return;
    {
      auto hm = Kokkos::create_mirror_view(tokens);
      Kokkos::deep_copy(hm, tokens);
      for (size_t s = 0; s < S; ++s)
        if (hm(0, s) < 0 || hm(0, s) >= cfg_.vocab_size)
          throw std::invalid_argument(
              "QwenImage21EncoderModel::forward_joint: token id out of range");
    }
    // Vision-slot rows on host (one mirror; ascending order required so
    // vis_embeds rows map in sequence).
    std::vector<size_t> rows;
    {
      auto hr = Kokkos::create_mirror_view(vis_rows);
      Kokkos::deep_copy(hr, vis_rows);
      for (size_t k = 0; k < Nv; ++k) {
        if ((size_t)hr(k) >= S)
          throw std::invalid_argument(
              "QwenImage21EncoderModel::forward_joint: vis row out of range");
        if (k > 0 && (size_t)hr(k) <= rows.back())
          throw std::invalid_argument(
              "QwenImage21EncoderModel::forward_joint: vis rows must ascend");
        rows.push_back((size_t)hr(k));
      }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(Hd));
    const float eps = static_cast<float>(cfg_.norm_eps);
    using UView3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                                Kokkos::MemoryUnmanaged>;
    using UView2 = Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace,
                                Kokkos::MemoryUnmanaged>;
    UView2 pos3(const_cast<int*>(pos3_flat.data()), 3, S);
    I1 row_ids("qwen21encj_ids", S);
    V2 h("qwen21encj_h", S, D);
    V2 n("qwen21encj_n", S, D);
    V2 qf("qwen21encj_q", S, Nq * Hd);
    V2 kf("qwen21encj_k", S, Nk * Hd);
    V2 vf("qwen21encj_v", S, Nk * Hd);
    V2 ao("qwen21encj_ao", S, D);
    V2 o("qwen21encj_o", S, D);
    V2 g("qwen21encj_g", S, G);
    V2 u("qwen21encj_u", S, G);
    V2 d("qwen21encj_d", S, D);
    UView3 q3(qf.data(), S, Nq, Hd);
    UView3 k3(kf.data(), S, Nk, Hd);
    UView3 v3(vf.data(), S, Nk, Hd);
    UView3 ao3(ao.data(), S, Nq, Hd);
    {
      Kokkos::View<int*, ExecSpace, Kokkos::MemoryUnmanaged> rowpre(
          tokens.data(), S);
      Kokkos::deep_copy(row_ids, rowpre);
      KokkosDNN::embedding_gather<ExecSpace, Scalar>(weights_.embed, row_ids,
                                                    h, "Qwen21EncJ::embed");
      // Replace vision-slot embeddings with the merger output rows.
      if (Nv > 0) {
        I1 d_rows("qwen21encj_rows", Nv);
        {
          auto m = Kokkos::create_mirror_view(d_rows);
          for (size_t k = 0; k < Nv; ++k) m(k) = (int)rows[k];
          Kokkos::deep_copy(d_rows, m);
        }
        UView3 ve(const_cast<Scalar*>(vis_embeds.data()), 1, Nv, D);
        Kokkos::parallel_for(
            "Qwen21EncJ::vis_replace",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Nv, D}),
            KOKKOS_LAMBDA(size_t k, size_t c) {
              h(d_rows(k), c) = ve(0, k, c);
            });
      }
    }
    for (int l = 0; l < cfg_.num_layers; ++l) {
      const auto& L = weights_.layers[(size_t)l];
      Impl::qwen21_rmsnorm<ExecSpace, Scalar>(h, L.in_norm, eps, n,
                                             "Qwen21EncJ::in_norm");
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.q_W,
                                         Scalar(0), qf, "Qwen21EncJ::q");
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.k_W,
                                         Scalar(0), kf, "Qwen21EncJ::k");
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.v_W,
                                         Scalar(0), vf, "Qwen21EncJ::v");
      Impl::qwen21_qk_rmsnorm<ExecSpace, Scalar>(qf, L.qn, eps, q3,
                                                "Qwen21EncJ::q_norm");
      Impl::qwen21_qk_rmsnorm<ExecSpace, Scalar>(kf, L.kn, eps, k3,
                                                "Qwen21EncJ::k_norm");
      Impl::qwen21_mrope_inplace<ExecSpace>(q3, pos3, rope_inv_,
                                             cfg_.mrope_sections,
                                             "Qwen21EncJ::rope_q");
      Impl::qwen21_mrope_inplace<ExecSpace>(k3, pos3, rope_inv_,
                                             cfg_.mrope_sections,
                                             "Qwen21EncJ::rope_k");
      KokkosDNN::attention_gqa_single<ExecSpace, Scalar>(
          handle, q3, k3, v3, size_t{0}, ao3, scale, "Qwen21EncJ::attn");
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), ao, L.o_W,
                                         Scalar(0), o, "Qwen21EncJ::o");
      Impl::qwen21_residual_add<ExecSpace>(h, o, "Qwen21EncJ::res1");
      Impl::qwen21_rmsnorm<ExecSpace, Scalar>(h, L.post_norm, eps, n,
                                             "Qwen21EncJ::post_norm");
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.gate_W,
                                         Scalar(0), g, "Qwen21EncJ::gate");
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), n, L.up_W,
                                         Scalar(0), u, "Qwen21EncJ::up");
      Impl::qwen21_swiglu<ExecSpace, Scalar>(g, u, g, "Qwen21EncJ::swiglu");
      KokkosDNN::gemm<ExecSpace, Scalar>(handle, Scalar(1), g, L.down_W,
                                         Scalar(0), d, "Qwen21EncJ::down");
      Impl::qwen21_residual_add<ExecSpace>(h, d, "Qwen21EncJ::res2");
      // DeepStack: text layers 0..2 absorb the vision-tower taps at the
      // vision rows at the end of the full decoder layer (attn + MLP),
      // matching the reference (hidden = layer_outputs; layer_idx in
      // range(len(deep)): hidden[mask] += embeds).
      if (l < 3 && Nv > 0) {
        UView3 de(const_cast<Scalar*>(deep_embeds[l].data()), 1, Nv, D);
        Kokkos::View<int*, ExecSpace, Kokkos::MemoryUnmanaged> rr(
            vis_rows.data(), Nv);
        Kokkos::parallel_for(
            "Qwen21EncJ::deepstack",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                             {Nv, D}),
            KOKKOS_LAMBDA(size_t k, size_t c) {
              const size_t s = (size_t)rr(k);
              h(s, c) = static_cast<Scalar>(static_cast<float>(h(s, c)) +
                                            static_cast<float>(de(0, k, c)));
            });
      }
    }
    // Final RMSNorm NEUTRALIZED (pipeline hook parity); drop_idx rows are
    // left for the caller to drop.
    Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                 Kokkos::MemoryUnmanaged>
        dst(hidden.data(), S, D);
    Kokkos::deep_copy(dst, h);
    Kokkos::fence();
  }

 private:
  Config cfg_;
  Weights weights_;
  Kokkos::View<float*, ExecSpace> rope_inv_;  // [Hd/2]
};

}  // namespace QwenImage21Encoder
}  // namespace KokkosDNN

#endif
