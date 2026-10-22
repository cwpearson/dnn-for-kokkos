// LFM25 GGUF loader regression test: llama.cpp tensor naming.
//
// The real LFM2.5-24B-A2B-Q4_K_M checkpoint names the final norm
// `token_embd_norm` (not `output_norm`), the expert bias
// `blk.{l}.exp_probs_b` (not `blk.{l}.ffn_exp_probs_b`), and ships no
// `output` tensor (tied embeddings). The loader must accept the real names
// while keeping the legacy aliases working. All files here are synthesized
// into /tmp by write_synthetic_gguf; no downloads needed.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Weights.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using namespace KokkosDNN;
using namespace KokkosDNN::LFM25;

std::string tmp_path(const std::string& name) {
  return "/tmp/test_lfm25_" + name + ".gguf";
}

// F32 synthetic tensor filled with a constant (exactly representable).
Gguf::SyntheticTensor const_f32(const std::string& name,
                                std::vector<uint64_t> dims, float c) {
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::F32;
  t.dims  = std::move(dims);
  uint64_t numel = 1;
  for (auto d : t.dims) numel *= d;
  std::vector<float> vals((size_t)numel, c);
  t.raw.resize(vals.size() * 4);
  std::memcpy(t.raw.data(), vals.data(), t.raw.size());
  return t;
}

struct Naming {
  std::string final_norm;  // "token_embd_norm" (real) or "output_norm"
  std::string exp_bias;    // "exp_probs_b" (real) or "ffn_exp_probs_b"
  bool with_output = true;
};

// Full tiny_test() tensor set in ggml dim order ({cols, rows}[, experts]).
std::vector<Gguf::SyntheticTensor> build_tensors(const Config& cfg,
                                                 const Naming& nm) {
  const size_t D = (size_t)cfg.dim, V = (size_t)cfg.vocab_size;
  const size_t hq = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  const size_t HD = (size_t)cfg.head_dim, K = (size_t)cfg.conv_L_cache;
  const size_t FF = (size_t)cfg.ffn_dim, MI = (size_t)cfg.moe_intermediate;
  const size_t E = (size_t)cfg.n_experts;
  std::vector<Gguf::SyntheticTensor> ts;
  float c = 1.0f;
  auto add = [&](const std::string& n, std::vector<uint64_t> d) {
    ts.push_back(const_f32(n, std::move(d), c));
    c += 0.5f;
  };
  add("token_embd.weight", {D, V});
  add(nm.final_norm + ".weight", {D});
  if (nm.with_output) add("output.weight", {D, V});
  for (int l = 0; l < cfg.n_layers; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    add(b + "attn_norm.weight", {D});
    add(b + "ffn_norm.weight", {D});
    if (cfg.is_attention(l)) {
      add(b + "attn_q.weight", {D, hq});
      add(b + "attn_k.weight", {D, hkv});
      add(b + "attn_v.weight", {D, hkv});
      add(b + "attn_output.weight", {hq, D});
      add(b + "attn_q_norm.weight", {HD});
      add(b + "attn_k_norm.weight", {HD});
    } else {
      add(b + "shortconv.in_proj.weight", {D, 3 * D});
      add(b + "shortconv.conv.weight", {K, D});
      add(b + "shortconv.out_proj.weight", {D, D});
    }
    if (l < cfg.n_dense_layers) {
      add(b + "ffn_gate.weight", {D, FF});
      add(b + "ffn_up.weight", {D, FF});
      add(b + "ffn_down.weight", {FF, D});
    } else {
      add(b + "ffn_gate_inp.weight", {D, E});
      add(b + "ffn_gate_exps.weight", {D, MI, E});
      add(b + "ffn_up_exps.weight", {D, MI, E});
      add(b + "ffn_down_exps.weight", {MI, D, E});
      add(b + nm.exp_bias + ".bias", {E});
    }
  }
  return ts;
}

float host_vec0(const Kokkos::View<float*, Exec>& v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  return h(0);
}

float host_mat00(
    const Kokkos::View<float**, Kokkos::LayoutRight, Exec>& m) {
  auto h = Kokkos::create_mirror_view(m);
  Kokkos::deep_copy(h, m);
  return h(0, 0);
}

float expert_first_float(const Gguf::QuantWeight<Exec>& qw) {
  auto sub = Kokkos::subview(qw.data, std::make_pair((size_t)0, (size_t)4));
  Kokkos::View<uint8_t*, Kokkos::HostSpace> h("exp_host", 4);
  Kokkos::deep_copy(h, sub);
  float f = 0;
  std::memcpy(&f, h.data(), 4);
  return f;
}

void check_loads_with_values(const Config& cfg, const std::string& path) {
  Weights<Exec, float> w;
  EXPECT_NO_THROW((load_gguf<Exec, float>(cfg, path, w)));
  // Tensor order in build_tensors: token_embd c=1.0, norm c=1.5, output
  // c=2.0 (when present). Spot-check a few constants survived exactly.
  EXPECT_FLOAT_EQ(host_mat00(w.embed), 1.0f);
  EXPECT_FLOAT_EQ(host_vec0(w.final_norm), 1.5f);
  if (cfg.tie_word_embeddings) {
    // Present-output case: lm_head carries the output constant.
    Gguf::GgufFile f(path);
    if (f.contains("output.weight"))
      EXPECT_FLOAT_EQ(host_mat00(w.lm_head), 2.0f);
    else
      EXPECT_EQ(w.lm_head.data(), w.embed.data());  // tied: shared table
  }
  // Layer 0 is conv+dense in tiny_test(); its attn_norm is the 4th tensor.
  EXPECT_FLOAT_EQ(host_vec0(w.blocks[0].attn_norm), 2.5f);
  // MoE expert payloads stay packed F32; first expert's first float must
  // match the constant written for that tensor (exact memcpy path).
  {
    Gguf::GgufFile f(path);
    const int moe_layer = cfg.n_dense_layers + 1;  // layer 2 in tiny_test
    const std::string nm =
        "blk." + std::to_string(moe_layer) + ".ffn_gate_exps.weight";
    const auto& info = f.find(nm);
    std::vector<float> raw;
    Gguf::dequant_tensor_to_float(f, info, raw);
    ASSERT_FALSE(raw.empty());
    const auto& qw = w.blocks[(size_t)moe_layer].gate_exps[0];
    EXPECT_EQ(qw.dtype, Gguf::GgmlType::F32);
    EXPECT_FLOAT_EQ(expert_first_float(qw), raw[0]);
  }
}

}  // namespace

// Real llama.cpp naming loads and round-trips values.
TEST(LFM25WeightsGGUF, RealNaming) {
  const auto cfg = Config::tiny_test();
  const std::string path =
      tmp_path("real_naming");  // token_embd_norm + exp_probs_b + output
  Naming nm;
  nm.final_norm = "token_embd_norm";
  nm.exp_bias   = "exp_probs_b";
  nm.with_output = true;
  Gguf::write_synthetic_gguf(path, build_tensors(cfg, nm));
  check_loads_with_values(cfg, path);
  std::remove(path.c_str());
}

// Legacy aliases (output_norm, ffn_exp_probs_b) keep loading.
TEST(LFM25WeightsGGUF, LegacyAliases) {
  const auto cfg = Config::tiny_test();
  const std::string path = tmp_path("legacy_alias");
  Naming nm;
  nm.final_norm = "output_norm";
  nm.exp_bias   = "ffn_exp_probs_b";
  nm.with_output = true;
  Gguf::write_synthetic_gguf(path, build_tensors(cfg, nm));
  Weights<Exec, float> w;
  EXPECT_NO_THROW((load_gguf<Exec, float>(cfg, path, w)));
  EXPECT_FLOAT_EQ(host_vec0(w.final_norm), 1.5f);
  std::remove(path.c_str());
}

// Absent lm_head + tie_word_embeddings shares the embedding table.
TEST(LFM25WeightsGGUF, TiedOutputFallback) {
  auto cfg                  = Config::tiny_test();
  cfg.tie_word_embeddings = true;
  const std::string path  = tmp_path("tied");
  Naming nm;
  nm.final_norm  = "token_embd_norm";
  nm.exp_bias    = "exp_probs_b";
  nm.with_output = false;
  Gguf::write_synthetic_gguf(path, build_tensors(cfg, nm));
  Weights<Exec, float> w;
  EXPECT_NO_THROW((load_gguf<Exec, float>(cfg, path, w)));
  EXPECT_EQ(w.lm_head.data(), w.embed.data());
  std::remove(path.c_str());
}

// Neither final-norm alias present: loader must throw, not segfault.
TEST(LFM25WeightsGGUF, MissingNormThrows) {
  const auto cfg = Config::tiny_test();
  const std::string path = tmp_path("missing_norm");
  Naming nm;
  nm.final_norm = "token_embd_norm";
  nm.exp_bias   = "exp_probs_b";
  auto ts       = build_tensors(cfg, nm);
  ts.erase(ts.begin() + 1);  // drop the final norm tensor
  Gguf::write_synthetic_gguf(path, ts);
  Weights<Exec, float> w;
  EXPECT_THROW((load_gguf<Exec, float>(cfg, path, w)), std::runtime_error);
  std::remove(path.c_str());
}

// Real 24B checkpoint header (when present locally): token_embd_norm exists,
// output_norm does not, expert bias is blk.{l}.exp_probs_b. Skips cleanly
// when the weights were never downloaded.
TEST(LFM25WeightsGGUF, RealCheckpointHeader) {
  const char* env = std::getenv("LFM25_GGUF_PATH");
  const std::string path =
      env ? env
          : "/workspace/dnn-for-kokkos/weights/lfm2_5/"
            "LFM2.5-24B-A2B-Q4_K_M.gguf";
  std::ifstream probe(path, std::ios::binary);
  if (!probe) GTEST_SKIP() << "real checkpoint absent: " << path;
  Gguf::GgufFile f(path);
  EXPECT_TRUE(f.contains("token_embd.weight"));
  EXPECT_TRUE(f.contains("token_embd_norm.weight"));
  EXPECT_FALSE(f.contains("output_norm.weight"));
  // Every MoE layer (l >= n_dense_layers) carries exp_probs_b, none carry
  // the old ffn_exp_probs_b stem.
  bool saw_exp = false;
  for (const auto& t : f.tensors()) {
    EXPECT_TRUE(t.name.find("ffn_exp_probs_b") == std::string::npos)
        << t.name;
    if (t.name.find("exp_probs_b") != std::string::npos) saw_exp = true;
  }
  EXPECT_TRUE(saw_exp);
}
