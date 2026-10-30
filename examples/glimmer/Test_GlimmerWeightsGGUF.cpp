// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Glimmer GGUF loader regression test: llama.cpp tensor naming plus the
// three conversion-trap handlings (norm shift, Q/K re-permute, synthesized
// QK norms) and metadata validation. All files are synthesized into /tmp by
// write_synthetic_gguf; no downloads needed.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "KokkosDNN_GGUF.hpp"
#include "KokkosDNN_GlimmerConfig.hpp"
#include "KokkosDNN_GlimmerModel.hpp"
#include "KokkosDNN_GlimmerWeights.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using namespace KokkosDNN;
using namespace KokkosDNN::Glimmer;

std::string tmp_path(const std::string& name) {
  return "/tmp/test_glimmer_" + name + ".gguf";
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

Gguf::SyntheticTensor f32_tensor(const std::string& name,
                                std::vector<uint64_t> dims,
                                const std::vector<float>& vals) {
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::F32;
  t.dims  = std::move(dims);
  t.raw.resize(vals.size() * 4);
  std::memcpy(t.raw.data(), vals.data(), t.raw.size());
  return t;
}

uint16_t float_to_fp16(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp   = (int32_t)((x >> 23) & 0xFFu);
  const uint32_t mant = x & 0x7FFFFFu;
  if (exp == 255)
    return (uint16_t)(sign | 0x7C00u | (mant != 0u ? 0x0200u : 0u));
  int32_t e = exp - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7C00u);
  if (e <= 0) return (uint16_t)sign;  // test constants never take this path
  uint32_t m = mant + 0x1000u;        // round (exact inputs unaffected)
  if (m >= 0x800000u) {
    m = 0;
    e += 1;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
  }
  return (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
}

uint16_t float_to_bf16(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const uint32_t r = x + 0x7FFFu + ((x >> 16) & 1u);  // round to nearest even
  return (uint16_t)(r >> 16);
}

Gguf::SyntheticTensor f16_tensor(const std::string& name,
                                 std::vector<uint64_t> dims,
                                 const std::vector<float>& vals) {
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::F16;
  t.dims  = std::move(dims);
  t.raw.resize(vals.size() * 2);
  auto* out = reinterpret_cast<uint16_t*>(t.raw.data());
  for (size_t i = 0; i < vals.size(); ++i) out[i] = float_to_fp16(vals[i]);
  return t;
}

Gguf::SyntheticTensor bf16_tensor(const std::string& name,
                                  std::vector<uint64_t> dims,
                                  const std::vector<float>& vals) {
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::BF16;
  t.dims  = std::move(dims);
  t.raw.resize(vals.size() * 2);
  auto* out = reinterpret_cast<uint16_t*>(t.raw.data());
  for (size_t i = 0; i < vals.size(); ++i) out[i] = float_to_bf16(vals[i]);
  return t;
}

// Q8_0 tensor from exact integer values in [-127, 127]: one fp16 scale of
// 1.0 per 32-element block, so dequantization is exact.
Gguf::SyntheticTensor q80_tensor(const std::string& name,
                                std::vector<uint64_t> dims,
                                const std::vector<float>& vals) {
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::Q8_0;
  t.dims  = std::move(dims);
  if (vals.size() % 32 != 0)
    throw std::runtime_error("q80_tensor: numel not a multiple of 32");
  t.raw.resize(vals.size() / 32 * sizeof(Gguf::BlockQ80));
  for (size_t b = 0; b < vals.size() / 32; ++b) {
    Gguf::BlockQ80 blk{};
    blk.d = 0x3C00;  // fp16 1.0
    for (size_t i = 0; i < 32; ++i) {
      const float v = vals[b * 32 + i];
      if (v < -127.0f || v > 127.0f || v != (float)(int)v)
        throw std::runtime_error("q80_tensor: value not an exact int8");
      blk.qs[i] = (int8_t)(int)v;
    }
    std::memcpy(t.raw.data() + b * sizeof(blk), &blk, sizeof(blk));
  }
  return t;
}

// Q4_K tensor from exact integer values in [0, 15]: superblock scale 1.0,
// zero min, all sub-block scales 1, so dequantization returns the nibbles
// exactly. numel must be a multiple of 256.
Gguf::SyntheticTensor q4k_tensor(const std::string& name,
                                std::vector<uint64_t> dims,
                                const std::vector<float>& vals) {
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::Q4_K;
  t.dims  = std::move(dims);
  if (vals.size() % 256 != 0)
    throw std::runtime_error("q4k_tensor: numel not a multiple of 256");
  t.raw.resize(vals.size() / 256 * sizeof(Gguf::BlockQ4K));
  for (size_t b = 0; b < vals.size() / 256; ++b) {
    Gguf::BlockQ4K blk{};
    blk.d    = 0x3C00;  // fp16 1.0
    blk.dmin = 0x0000;  // fp16 0.0 (min contribution vanishes)
    // Sub-block scales all 1: get_scale_min_k4(j) reads scales[j]&63 for
    // j<4 and (scales[j+4]&0xF)|((scales[j-4]>>6)<<4) for j>=4.
    blk.scales[0] = blk.scales[1] = blk.scales[2] = blk.scales[3] = 1;
    blk.scales[8] = blk.scales[9] = blk.scales[10] = blk.scales[11] = 1;
    for (size_t i = 0; i < 256; ++i) {
      const float v = vals[b * 256 + i];
      if (v < 0.0f || v > 15.0f || v != (float)(int)v)
        throw std::runtime_error("q4k_tensor: value not an exact nibble");
      // ggml layout: values[i] and values[i+32] share byte
      // chunk*32 + (i%32) (low nibble = values[i], high = values[i+32]).
      const uint8_t q = (uint8_t)(int)v;
      const size_t byte = (i / 64) * 32 + (i % 32);
      if ((i % 64) < 32)
        blk.qs[byte] = (uint8_t)((blk.qs[byte] & 0xF0) | q);
      else
        blk.qs[byte] = (uint8_t)((blk.qs[byte] & 0x0F) | (q << 4));
    }
    std::memcpy(t.raw.data() + b * sizeof(blk), &blk, sizeof(blk));
  }
  return t;
}

// Q6_K tensor holding one constant integer in [-32, 31] everywhere:
// superblock scale 1.0, all 16 sub-block scales 1, quants packed from
// q = value + 32. numel must be a multiple of 256.
Gguf::SyntheticTensor q6k_const_tensor(const std::string& name,
                                       std::vector<uint64_t> dims, int v) {
  if (v < -32 || v > 31)
    throw std::runtime_error("q6k_const_tensor: value out of range");
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::Q6_K;
  t.dims  = std::move(dims);
  uint64_t numel = 1;
  for (auto d : t.dims) numel *= d;
  if (numel % 256 != 0)
    throw std::runtime_error("q6k_const_tensor: numel not multiple of 256");
  const uint8_t q  = (uint8_t)(v + 32);
  const uint8_t lo = (uint8_t)(q & 0xF);
  const uint8_t hi = (uint8_t)((q >> 4) & 0x3);
  t.raw.resize((size_t)numel / 256 * sizeof(Gguf::BlockQ6K));
  for (size_t b = 0; b < (size_t)numel / 256; ++b) {
    Gguf::BlockQ6K blk{};
    blk.d = 0x3C00;  // fp16 1.0
    for (int i = 0; i < 16; ++i) blk.scales[i] = 1;
    for (int i = 0; i < 128; ++i) blk.ql[i] = (uint8_t)(lo | (lo << 4));
    for (int i = 0; i < 64; ++i)
      blk.qh[i] = (uint8_t)(hi | (hi << 2) | (hi << 4) | (hi << 6));
    std::memcpy(t.raw.data() + b * sizeof(blk), &blk, sizeof(blk));
  }
  return t;
}

// Test-local copy of the llama.cpp conversion _unpermute_for_rope
// (conversion/muse_glimmer.py): HF permuted (rotate_half) -> GGUF
// interleaved (NORM) layout. The loader inverts this; the exact-F32 test
// below checks loader(unpermute(hf)) == hf through the public load_gguf.
std::vector<float> unpermute_for_rope(const std::vector<float>& hf,
                                      size_t n_heads, size_t head_dim,
                                      size_t cols) {
  const size_t half = head_dim / 2;
  std::vector<float> out(hf.size());
  for (size_t h = 0; h < n_heads; ++h)
    for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < half; ++j) {
        const size_t dst = h * head_dim + j * 2 + i;
        const size_t src = h * head_dim + i * half + j;
        for (size_t k = 0; k < cols; ++k)
          out[dst * cols + k] = hf[src * cols + k];
      }
  return out;
}

// Inverse direction (GGUF interleaved -> HF permuted), mirroring the
// loader's re-permute; used only for the known-vector roundtrip check.
std::vector<float> permute_for_rope(const std::vector<float>& gguf,
                                     size_t n_heads, size_t head_dim,
                                     size_t cols) {
  const size_t half = head_dim / 2;
  std::vector<float> out(gguf.size());
  for (size_t h = 0; h < n_heads; ++h)
    for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < half; ++j) {
        const size_t dst = h * head_dim + i * half + j;
        const size_t src = h * head_dim + j * 2 + i;
        for (size_t k = 0; k < cols; ++k)
          out[dst * cols + k] = gguf[src * cols + k];
      }
  return out;
}

std::string gguf_layer(int l, const std::string& leaf) {
  return "blk." + std::to_string(l) + "." + leaf;
}

// Conversion-written hyperparameters for cfg (what a real GGUF carries).
Gguf::SyntheticKV default_kv(const Config& cfg) {
  Gguf::SyntheticKV kv;
  kv.f32["muse-glimmer.logit_scale"] =
      static_cast<float>(cfg.output_multiplier);
  kv.f32["muse-glimmer.final_logit_softcapping"] = cfg.final_logit_softcap;
  kv.u32["muse-glimmer.attention.sliding_window"] =
      static_cast<uint32_t>(cfg.sliding_window);
  std::vector<char> pat((size_t)cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l)
    pat[(size_t)l] = cfg.is_full(l) ? (char)0 : (char)1;
  kv.bool_arrays["muse-glimmer.attention.sliding_window_pattern"] = pat;
  return kv;
}

// Full tiny_test() tensor set in ggml dim order ({cols, rows}).
// Norms are written GGUF-style (HF-centered value + 1); q/k are written
// conversion-style (unpermuted from the known HF data in hf_q/hf_k).
struct BuiltTensors {
  std::vector<Gguf::SyntheticTensor> ts;
  // Known HF-layout q/k data per layer (what the loader must reproduce).
  std::vector<std::vector<float>> hf_q, hf_k;
};

BuiltTensors build_exact_tensors(const Config& cfg, bool with_qk_norms) {
  const size_t D  = (size_t)cfg.dim, V = (size_t)cfg.vocab_size;
  const size_t hq = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
  const size_t HD = (size_t)cfg.head_dim, FF = (size_t)cfg.ffn_dim;
  BuiltTensors out;
  float c = 1.0f;
  auto add = [&](const std::string& n, std::vector<uint64_t> d) {
    out.ts.push_back(const_f32(n, std::move(d), c));
    c += 0.5f;
  };
  add("token_embd.weight", {D, V});
  add("output_norm.weight", {D});
  add("output.weight", {D, V});
  out.hf_q.resize((size_t)cfg.n_layers);
  out.hf_k.resize((size_t)cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    const std::string b = gguf_layer(l, "");
    // Known HF-layout data: distinct per element.
    out.hf_q[(size_t)l].resize(hq * D);
    out.hf_k[(size_t)l].resize(hkv * D);
    for (size_t i = 0; i < hq * D; ++i)
      out.hf_q[(size_t)l][i] = (float)(i + 1) + 0.25f * (float)l;
    for (size_t i = 0; i < hkv * D; ++i)
      out.hf_k[(size_t)l][i] = (float)(i + 1) + 0.125f * (float)l;
    out.ts.push_back(f32_tensor(b + "attn_q.weight", {D, hq},
                                unpermute_for_rope(out.hf_q[(size_t)l],
                                                   (size_t)cfg.n_heads, HD,
                                                   D)));
    out.ts.push_back(f32_tensor(b + "attn_k.weight", {D, hkv},
                                unpermute_for_rope(out.hf_k[(size_t)l],
                                                   (size_t)cfg.n_kv_heads, HD,
                                                   D)));
    add(b + "attn_v.weight", {D, hkv});
    add(b + "attn_output.weight", {hq, D});
    add(b + "attn_gate.weight", {D, hq});
    // GGUF norms carry the conversion +1 shift on top of HF-centered 0.5.
    out.ts.push_back(const_f32(b + "attn_norm.weight", {D}, 1.5f));
    out.ts.push_back(const_f32(b + "post_attention_norm.weight", {D}, 1.5f));
    out.ts.push_back(const_f32(b + "ffn_norm.weight", {D}, 1.5f));
    out.ts.push_back(const_f32(b + "post_ffw_norm.weight", {D}, 1.5f));
    add(b + "ffn_gate.weight", {D, FF});
    add(b + "ffn_up.weight", {D, FF});
    add(b + "ffn_down.weight", {FF, D});
    if (with_qk_norms) {
      // Synthesized conversion weights: absorbed qk_scale_factor / ones.
      out.ts.push_back(const_f32(b + "attn_q_norm.weight", {HD}, 3.87f));
      out.ts.push_back(const_f32(b + "attn_k_norm.weight", {HD}, 1.0f));
    }
  }
  return out;
}

float host_vec_elem(const Kokkos::View<float*, Exec>& v, size_t i) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  return h(i);
}

void check_all_vec(const Kokkos::View<float*, Exec>& v, float want) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  for (size_t i = 0; i < h.extent(0); ++i)
    EXPECT_FLOAT_EQ(h(i), want) << "at " << i;
}

// Packed-weight host read: dequantize once via Gguf::dequant_to_dense,
// mirror, flatten row-major. All packed assertions below go through this
// (one kernel launch per tensor, not per element).
std::vector<float> packed_host(const Gguf::QuantWeight<Exec>& qw) {
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> Y("packed_host", qw.rows,
                                                     qw.cols);
  Gguf::dequant_to_dense(qw, Y);
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(h, Y);
  std::vector<float> out(qw.rows * qw.cols);
  for (size_t r = 0; r < qw.rows; ++r)
    for (size_t c = 0; c < qw.cols; ++c) out[r * qw.cols + c] = h(r, c);
  return out;
}

void check_packed_all(const Gguf::QuantWeight<Exec>& qw, float want) {
  const auto v = packed_host(qw);
  for (size_t i = 0; i < v.size(); ++i)
    EXPECT_FLOAT_EQ(v[i], want) << "at " << i;
}

}  // namespace

// The two directions invert each other on a known vector, including a
// hand-computed interleave ([0,1,2,3] -> [0,2,1,3] for 1 head of dim 4).
TEST(GlimmerWeightsGGUF, RopePermuteRoundtrip) {
  const std::vector<float> hf = {0.0f, 1.0f, 2.0f, 3.0f};
  const std::vector<float> unperm = unpermute_for_rope(hf, 1, 4, 1);
  ASSERT_EQ(unperm.size(), 4u);
  EXPECT_FLOAT_EQ(unperm[0], 0.0f);
  EXPECT_FLOAT_EQ(unperm[1], 2.0f);
  EXPECT_FLOAT_EQ(unperm[2], 1.0f);
  EXPECT_FLOAT_EQ(unperm[3], 3.0f);
  EXPECT_EQ(permute_for_rope(unperm, 1, 4, 1), hf);
  // Larger known vector, 2 heads x dim 4 x 3 cols.
  std::vector<float> big(2 * 4 * 3);
  for (size_t i = 0; i < big.size(); ++i) big[i] = (float)i * 0.5f - 3.0f;
  EXPECT_EQ(permute_for_rope(unpermute_for_rope(big, 2, 4, 3), 2, 4, 3),
            big);
}

// Full tiny_test() set loads and round-trips values: norms shed the
// conversion +1, q/k come back in HF (re-permuted) layout, and the
// synthesized QK norms are accepted but ignored.
TEST(GlimmerWeightsGGUF, ExactRoundtrip) {
  const auto cfg = Config::tiny_test();
  const std::string path = tmp_path("exact");
  BuiltTensors built = build_exact_tensors(cfg, true);
  Gguf::write_synthetic_gguf(path, built.ts, default_kv(cfg));
  Weights<Exec, float> w;
  EXPECT_NO_THROW((load_gguf<Exec, float>(cfg, path, w)));
  EXPECT_FLOAT_EQ(packed_host(w.embed)[0], 1.0f);
  EXPECT_FLOAT_EQ(host_vec_elem(w.final_norm, 0), 1.5f);  // shift 0: as-is
  EXPECT_FLOAT_EQ(packed_host(w.lm_head)[0], 2.0f);
  // c-counter: token_embd 1.0, output_norm 1.5, output 2.5; layer 0 norms
  // are the constant 1.5 file value minus the loader's -1 shift.
  for (int l = 0; l < cfg.n_layers; ++l) {
    const auto& b = w.blocks[(size_t)l];
    check_all_vec(b.input_norm, 0.5f);
    check_all_vec(b.post_attn_norm, 0.5f);
    check_all_vec(b.pre_ffn_norm, 0.5f);
    check_all_vec(b.post_ffn_norm, 0.5f);
    // q/k match the known HF-layout data (re-permuted by the loader).
    const auto& hq = built.hf_q[(size_t)l];
    const auto& hk = built.hf_k[(size_t)l];
    const size_t D = (size_t)cfg.dim;
    const size_t nq = (size_t)cfg.n_heads * (size_t)cfg.head_dim;
    const size_t nk = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;
    const auto wq = packed_host(b.wq);
    for (size_t i = 0; i < nq * D; ++i)
      EXPECT_FLOAT_EQ(wq[i], hq[i]) << "layer " << l << " q elem " << i;
    const auto wk = packed_host(b.wk);
    for (size_t i = 0; i < nk * D; ++i)
      EXPECT_FLOAT_EQ(wk[i], hk[i]) << "layer " << l << " k elem " << i;
  }
  // Untied head: separate tensor, never aliased.
  EXPECT_NE(w.lm_head.data.data(), w.embed.data.data());
  // Forward smoke on the loaded weights (finite, softcap-bounded logits).
  Model<Exec, float> model(cfg, w);
  Kokkos::View<int*, Exec> toks("toks", 2);
  {
    auto h = Kokkos::create_mirror_view(toks);
    h(0) = 2;
    h(1) = 10;
    Kokkos::deep_copy(toks, h);
  }
  auto lg = model.forward(toks);
  ASSERT_EQ(lg.extent(0), (size_t)cfg.vocab_size);
  {
    auto h = Kokkos::create_mirror_view(lg);
    Kokkos::deep_copy(h, lg);
    for (size_t i = 0; i < h.extent(0); ++i) {
      const float v = h(i);
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LE(v, 20.0f + 1e-3f);
      EXPECT_GE(v, -20.0f - 1e-3f);
    }
  }
  // bhalf_t instantiation loads the same file (values BF16-rounded).
  Weights<Exec, Kokkos::Experimental::bhalf_t> wb;
  EXPECT_NO_THROW(
      (load_gguf<Exec, Kokkos::Experimental::bhalf_t>(cfg, path, wb)));
  {
    EXPECT_NEAR(packed_host(wb.embed)[0], 1.0f, 1e-2f);
    auto hn = Kokkos::create_mirror_view(wb.blocks[0].input_norm);
    Kokkos::deep_copy(hn, wb.blocks[0].input_norm);
    EXPECT_NEAR(static_cast<float>(hn(0)), 0.5f, 1e-2f);
  }
  std::remove(path.c_str());
}

// The synthesized QK norms are ignored, not required: a file without them
// loads identically.
TEST(GlimmerWeightsGGUF, SynthesizedNormsAbsent) {
  const auto cfg = Config::tiny_test();
  const std::string path = tmp_path("no_qk_norms");
  BuiltTensors built = build_exact_tensors(cfg, false);
  Gguf::write_synthetic_gguf(path, built.ts, default_kv(cfg));
  Weights<Exec, float> w;
  EXPECT_NO_THROW((load_gguf<Exec, float>(cfg, path, w)));
  EXPECT_FLOAT_EQ(packed_host(w.embed)[0], 1.0f);
  check_all_vec(w.blocks[0].input_norm, 0.5f);
  const auto& hq = built.hf_q[0];
  const auto wq0 = packed_host(w.blocks[0].wq);
  for (size_t i = 0; i < hq.size(); ++i) EXPECT_FLOAT_EQ(wq0[i], hq[i]);
  std::remove(path.c_str());
}

// Wide-dim (dim 256) mixed-dtype file: F16/BF16/Q8_0/Q4_K/Q6_K all decode
// exactly for crafted values; q/k stay F32 so the permute check is exact.
TEST(GlimmerWeightsGGUF, QuantizedMixed) {
  Config cfg;
  cfg.dim = 256;
  cfg.n_layers = 2;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.head_dim = 8;
  cfg.vocab_size = 64;
  cfg.ffn_dim = 64;
  cfg.sliding_window = 8;
  cfg.is_full_layer = {0, 1};
  cfg.layer_rope_theta = {500000.0, 0.0};
  cfg.max_seq_len = 32;
  cfg.validate();
  const size_t D  = (size_t)cfg.dim, V = (size_t)cfg.vocab_size;
  const size_t hq = (size_t)cfg.n_heads * (size_t)cfg.head_dim;    // 32
  const size_t hkv = (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim;  // 16
  const size_t HD = (size_t)cfg.head_dim, FF = (size_t)cfg.ffn_dim;

  std::vector<Gguf::SyntheticTensor> ts;
  std::vector<std::vector<float>> hf_q(2), hf_k(2);
  ts.push_back(const_f32("token_embd.weight", {D, V}, 1.0f));
  ts.push_back(const_f32("output_norm.weight", {D}, 1.5f));
  {
    std::vector<float> v(V * D, 2.5f);
    ts.push_back(bf16_tensor("output.weight", {D, V}, v));
  }
  for (int l = 0; l < cfg.n_layers; ++l) {
    const std::string b = gguf_layer(l, "");
    hf_q[(size_t)l].resize(hq * D);
    hf_k[(size_t)l].resize(hkv * D);
    for (size_t i = 0; i < hq * D; ++i)
      hf_q[(size_t)l][i] = (float)(i % 97) * 0.5f;
    for (size_t i = 0; i < hkv * D; ++i)
      hf_k[(size_t)l][i] = (float)(i % 53) * 0.25f;
    ts.push_back(f32_tensor(b + "attn_q.weight", {D, hq},
                            unpermute_for_rope(hf_q[(size_t)l], 4, HD, D)));
    ts.push_back(f32_tensor(b + "attn_k.weight", {D, hkv},
                            unpermute_for_rope(hf_k[(size_t)l], 2, HD, D)));
    ts.push_back(q6k_const_tensor(b + "attn_v.weight", {D, hkv}, -32));
    {
      std::vector<float> v(D * hq);
      for (size_t i = 0; i < v.size(); ++i)
        v[i] = (float)(i % 32) - 16.0f;
      ts.push_back(q80_tensor(b + "attn_output.weight", {hq, D}, v));
    }
    ts.push_back(const_f32(b + "attn_gate.weight", {D, hq}, 4.0f));
    // Shifted norms: file carries HF-centered 0.5 + conversion 1.0.
    ts.push_back(f16_tensor(b + "attn_norm.weight", {D},
                            std::vector<float>(D, 1.5f)));
    ts.push_back(f16_tensor(b + "post_attention_norm.weight", {D},
                            std::vector<float>(D, 1.5f)));
    ts.push_back(f16_tensor(b + "ffn_norm.weight", {D},
                            std::vector<float>(D, 1.5f)));
    ts.push_back(f16_tensor(b + "post_ffw_norm.weight", {D},
                            std::vector<float>(D, 1.5f)));
    {
      // Q4_K on ffn_gate ([FF, D]: cols 256 is a K-quant multiple).
      std::vector<float> v(FF * D);
      for (size_t i = 0; i < v.size(); ++i) v[i] = (float)(i % 16);
      ts.push_back(q4k_tensor(b + "ffn_gate.weight", {D, FF}, v));
    }
    {
      // Q8_0 on ffn_up with a repeating signed pattern.
      std::vector<float> v(FF * D);
      for (size_t i = 0; i < v.size(); ++i)
        v[i] = (float)((int)(i % 32) - 16);
      ts.push_back(q80_tensor(b + "ffn_up.weight", {D, FF}, v));
    }
    ts.push_back(const_f32(b + "ffn_down.weight", {FF, D}, 5.0f));
    ts.push_back(const_f32(b + "attn_q_norm.weight", {HD}, 3.87f));
    ts.push_back(const_f32(b + "attn_k_norm.weight", {HD}, 1.0f));
  }
  const std::string path = tmp_path("quant_mixed");
  Gguf::write_synthetic_gguf(path, ts, default_kv(cfg));
  Weights<Exec, float> w;
  EXPECT_NO_THROW((load_gguf<Exec, float>(cfg, path, w)));
  EXPECT_FLOAT_EQ(packed_host(w.embed)[0], 1.0f);
  EXPECT_FLOAT_EQ(host_vec_elem(w.final_norm, 3), 1.5f);
  EXPECT_FLOAT_EQ(packed_host(w.lm_head)[1 * D + 2], 2.5f);  // BF16 exact
  for (int l = 0; l < cfg.n_layers; ++l) {
    const auto& b = w.blocks[(size_t)l];
    check_all_vec(b.input_norm, 0.5f);  // F16 file 1.5, shift removed
    check_all_vec(b.post_attn_norm, 0.5f);
    check_all_vec(b.pre_ffn_norm, 0.5f);
    check_all_vec(b.post_ffn_norm, 0.5f);
    const auto wq = packed_host(b.wq);
    for (size_t i = 0; i < hq * D; ++i)
      EXPECT_FLOAT_EQ(wq[i], hf_q[(size_t)l][i]);
    const auto wk = packed_host(b.wk);
    for (size_t i = 0; i < hkv * D; ++i)
      EXPECT_FLOAT_EQ(wk[i], hf_k[(size_t)l][i]);
    check_packed_all(b.wv, -32.0f);  // Q6_K constant
    const auto wo = packed_host(b.wo);
    for (size_t i = 0; i < D * hq; ++i)
      EXPECT_FLOAT_EQ(wo[i], (float)(i % 32) - 16.0f);  // Q8_0
    check_packed_all(b.wgate_attn, 4.0f);
    // Q4_K nibble pattern on ffn_gate; Q8_0 signed pattern on ffn_up.
    const auto wgate = packed_host(b.wgate);
    const auto wup = packed_host(b.wup);
    for (size_t i = 0; i < FF * D; ++i) {
      EXPECT_FLOAT_EQ(wgate[i], (float)(i % 16));
      EXPECT_FLOAT_EQ(wup[i], (float)((int)(i % 32) - 16));
    }
    check_packed_all(b.wdown, 5.0f);
  }
  std::remove(path.c_str());
}

// A missing required tensor is a hard error, not a silent zero-fill.
TEST(GlimmerWeightsGGUF, MissingTensorThrows) {
  const auto cfg = Config::tiny_test();
  const std::string path = tmp_path("missing");
  BuiltTensors built = build_exact_tensors(cfg, true);
  // Drop one required tensor (ffn_up of layer 1).
  auto& ts = built.ts;
  ts.erase(std::remove_if(ts.begin(), ts.end(),
                          [](const Gguf::SyntheticTensor& t) {
                            return t.name == "blk.1.ffn_up.weight";
                          }),
           ts.end());
  Gguf::write_synthetic_gguf(path, ts, default_kv(cfg));
  Weights<Exec, float> w;
  EXPECT_THROW((load_gguf<Exec, float>(cfg, path, w)), std::runtime_error);
  std::remove(path.c_str());
}

// Metadata disagreements throw: wrong logit scale, wrong sliding window,
// wrong sliding-window pattern length/content. The loader validates, never
// silently overrides cfg.
TEST(GlimmerWeightsGGUF, MetadataMismatchThrows) {
  const auto cfg = Config::tiny_test();
  BuiltTensors built = build_exact_tensors(cfg, true);
  {
    const std::string path = tmp_path("bad_scale");
    Gguf::SyntheticKV kv = default_kv(cfg);
    kv.f32["muse-glimmer.logit_scale"] = 1.0f;
    Gguf::write_synthetic_gguf(path, built.ts, kv);
    Weights<Exec, float> w;
    EXPECT_THROW((load_gguf<Exec, float>(cfg, path, w)), std::runtime_error);
    std::remove(path.c_str());
  }
  {
    const std::string path = tmp_path("bad_window");
    Gguf::SyntheticKV kv = default_kv(cfg);
    kv.u32["muse-glimmer.attention.sliding_window"] = 1024;
    Gguf::write_synthetic_gguf(path, built.ts, kv);
    Weights<Exec, float> w;
    EXPECT_THROW((load_gguf<Exec, float>(cfg, path, w)), std::runtime_error);
    std::remove(path.c_str());
  }
  {
    const std::string path = tmp_path("bad_pattern");
    Gguf::SyntheticKV kv = default_kv(cfg);
    // Layer 3 is full-attention in tiny_test; marking it sliding must fail.
    kv.bool_arrays["muse-glimmer.attention.sliding_window_pattern"] =
        std::vector<char>({1, 1, 1, 1});
    Gguf::write_synthetic_gguf(path, built.ts, kv);
    Weights<Exec, float> w;
    EXPECT_THROW((load_gguf<Exec, float>(cfg, path, w)), std::runtime_error);
    std::remove(path.c_str());
  }
}

// list_gguf_tensors reports the GGUF names without reading tensor data.
TEST(GlimmerWeightsGGUF, ListTensors) {
  const auto cfg = Config::tiny_test();
  const std::string path = tmp_path("list");
  BuiltTensors built = build_exact_tensors(cfg, true);
  Gguf::write_synthetic_gguf(path, built.ts, default_kv(cfg));
  auto names = list_gguf_tensors(path);
  EXPECT_EQ(names.size(), built.ts.size());
  EXPECT_TRUE(std::find(names.begin(), names.end(), "token_embd.weight") !=
              names.end());
  EXPECT_TRUE(std::find(names.begin(), names.end(),
                        "blk.0.attn_q_norm.weight") != names.end());
  EXPECT_TRUE(std::find(names.begin(), names.end(),
                        "blk.3.post_ffw_norm.weight") != names.end());
  std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Packed-weights validation track (MERGE NOTE).
//
// Context: the GGUF loader above dequantizes everything to dense views
// (~56 GB for the 30B Q4_K_M checkpoint: OOM on 24 GB VRAM / 31 GB RAM).
// The conversion in progress replaces dense BlockWeights matrices with
// Gguf::QuantWeight packed payloads + quant_gemv/quant_gemm. These tests are
// ADDITIVE: they compile and pass both before and after the Weights struct
// change (they build packed payloads directly, never touching the struct),
// and they pin the contracts the new loader must satisfy:
//
//   PackedParity* : quant_gemv / quant_gemm / dequant_to_dense agree with a
//     host-dequantized (double-accumulation) reference on exact-value
//     synthetic packings. Mirrors the tests/Test_GGUF.cpp bit-parity
//     approach (GgufQuantGemm.*, GgufDequantToDense.*).
//   PackedPermute*: the Q/K RoPE layout fixup is a ROW permutation of the
//     [rows, cols] matrix (see gguf_permute_qk_for_rope in
//     models/glimmer/KokkosDNN_GlimmerWeights.cpp), so on packed row-major
//     payloads it must be a memcpy of row_bytes chunks -- never a full-tensor
//     dequantize. The new loader must permute packed rows, not dense values.
//   PackedRealFileGate: header-only gate for the 17 GB real GGUF (no tensor
//     data is read, so it cannot OOM). Post-merge end-to-end validation runs
//     the real greedy decode; the expected IDs are recorded there.
//
// Do NOT delete or weaken ExactRoundtrip / QuantizedMixed for the struct
// change: update them only if the Weights struct change requires it.
// ---------------------------------------------------------------------------

namespace packed_track {

// Q5_K tensor with unit scale / zero min (mirrors the Q4_K helper above):
// superblock scale 1.0, min 0.0, all sub-block (scale, min) = (1, 0), so the
// dequantized value is the 5-bit quant (0..31). numel must be a multiple
// of 256.
Gguf::SyntheticTensor q5k_tensor(const std::string& name,
                                 std::vector<uint64_t> dims,
                                 const std::vector<float>& vals) {
  Gguf::SyntheticTensor t;
  t.name  = name;
  t.dtype = Gguf::GgmlType::Q5_K;
  t.dims  = std::move(dims);
  if (vals.size() % 256 != 0)
    throw std::runtime_error("q5k_tensor: numel not a multiple of 256");
  t.raw.resize(vals.size() / 256 * sizeof(Gguf::BlockQ5K));
  for (size_t b = 0; b < vals.size() / 256; ++b) {
    Gguf::BlockQ5K blk{};
    blk.d    = 0x3C00;  // fp16 1.0
    blk.dmin = 0x0000;  // fp16 0.0
    blk.scales[0] = blk.scales[1] = blk.scales[2] = blk.scales[3] = 1;
    blk.scales[8] = blk.scales[9] = blk.scales[10] = blk.scales[11] = 1;
    for (int iter = 0; iter < 4; ++iter)
      for (int l = 0; l < 32; ++l) {
        const float v0 = vals[b * 256 + (size_t)iter * 64 + (size_t)l];
        const float v1 = vals[b * 256 + (size_t)iter * 64 + 32 + (size_t)l];
        if (v0 < 0.0f || v0 > 31.0f || v0 != (float)(int)v0 ||
            v1 < 0.0f || v1 > 31.0f || v1 != (float)(int)v1)
          throw std::runtime_error("q5k_tensor: value not an exact 5-bit int");
        blk.qs[iter * 32 + l] =
            (uint8_t)(((int)v0 & 0xF) | (((int)v1 & 0xF) << 4));
        blk.qh[l] = (uint8_t)(blk.qh[l] | ((((int)v0 >> 4) << (2 * iter)) |
                                           (((int)v1 >> 4) << (2 * iter + 1))));
      }
    std::memcpy(t.raw.data() + b * sizeof(blk), &blk, sizeof(blk));
  }
  return t;
}

// Device-resident packed weight from a synthetic tensor's raw payload.
Gguf::QuantWeight<Exec> make_packed(const Gguf::SyntheticTensor& t,
                                    size_t rows, size_t cols) {
  Gguf::QuantWeight<Exec> w;
  w.dtype = t.dtype;
  w.rows  = rows;
  w.cols  = cols;
  EXPECT_EQ(t.raw.size(), w.bytes());
  w.data = Kokkos::View<uint8_t*, Exec>("packed_track", t.raw.size());
  Kokkos::View<uint8_t*, Kokkos::HostSpace> h("packed_track_host",
                                              t.raw.size());
  std::memcpy(h.data(), t.raw.data(), t.raw.size());
  Kokkos::deep_copy(w.data, h);
  return w;
}

// Host dequantize of a packed row-major payload (independent of the device
// kernels under test; uses the header's block fns).
std::vector<float> host_dequant(Gguf::GgmlType dtype,
                                const std::vector<uint8_t>& raw, size_t numel) {
  std::vector<float> out(numel);
  switch (dtype) {
    case Gguf::GgmlType::F32:
      std::memcpy(out.data(), raw.data(), numel * 4);
      return out;
    case Gguf::GgmlType::Q8_0:
      for (size_t b = 0; b < numel / 32; ++b)
        Gguf::dequant_q80_block(
            reinterpret_cast<const Gguf::BlockQ80*>(raw.data() + b * 34),
            out.data() + b * 32);
      return out;
    case Gguf::GgmlType::Q4_K:
      for (size_t b = 0; b < numel / 256; ++b)
        Gguf::dequant_q4k_block(
            reinterpret_cast<const Gguf::BlockQ4K*>(raw.data() + b * 144),
            out.data() + b * 256);
      return out;
    case Gguf::GgmlType::Q5_K:
      for (size_t b = 0; b < numel / 256; ++b)
        Gguf::dequant_q5k_block(
            reinterpret_cast<const Gguf::BlockQ5K*>(raw.data() + b * 176),
            out.data() + b * 256);
      return out;
    case Gguf::GgmlType::Q6_K:
      for (size_t b = 0; b < numel / 256; ++b)
        Gguf::dequant_q6k_block(
            reinterpret_cast<const Gguf::BlockQ6K*>(raw.data() + b * 210),
            out.data() + b * 256);
      return out;
    default: throw std::runtime_error("host_dequant: unsupported dtype");
  }
}

// Byte-level row reorder of a packed payload: out row r = in row perm[r].
// Rows are independent packed runs, so this is the operation the packed
// Q/K loader must use for the RoPE re-permute (no dequantization).
std::vector<uint8_t> permute_packed_rows(const std::vector<uint8_t>& raw,
                                         size_t rows, size_t row_bytes,
                                         const std::vector<size_t>& perm) {
  std::vector<uint8_t> out(raw.size());
  for (size_t r = 0; r < rows; ++r)
    std::memcpy(out.data() + r * row_bytes, raw.data() + perm[r] * row_bytes,
                row_bytes);
  return out;
}

// quant_gemv + quant_gemm + dequant_to_dense vs the host-dequantized
// reference (double accumulation, exact-value packings so quantization
// error is zero by construction).
void check_packed_parity(Gguf::GgmlType dtype, size_t rows, size_t cols,
                         const std::vector<uint8_t>& raw,
                         const std::vector<float>& ref_vals) {
  const std::vector<float> wdq = host_dequant(dtype, raw, rows * cols);
  ASSERT_EQ(wdq.size(), ref_vals.size());
  for (size_t i = 0; i < wdq.size(); ++i)
    EXPECT_FLOAT_EQ(wdq[i], ref_vals[i]) << "packing not exact at " << i;

  Gguf::SyntheticTensor t;
  t.dtype = dtype;
  t.raw   = raw;
  Gguf::QuantWeight<Exec> w = make_packed(t, rows, cols);

  // dequant_to_dense reproduces the host reference bit-for-bit (float).
  {
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> Y("Y", rows, cols);
    Gguf::dequant_to_dense(w, Y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(Y);
    Kokkos::deep_copy(got, Y);
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c)
        EXPECT_FLOAT_EQ(got(r, c), wdq[r * cols + c])
            << "dequant_to_dense (" << r << "," << c << ")";
  }

  // Input vector / matrix: exact-binary values.
  std::vector<float> xf(cols);
  for (size_t k = 0; k < cols; ++k)
    xf[k] = 0.25f * (float)((int)(k % 5) - 2);

  // quant_gemv vs double-accumulation reference on the dequantized weights.
  {
    Kokkos::View<float*, Exec> x("x", cols), y("y", rows);
    auto hx = Kokkos::create_mirror_view(x);
    for (size_t k = 0; k < cols; ++k) hx(k) = xf[k];
    Kokkos::deep_copy(x, hx);
    Gguf::quant_gemv(Exec{}, x, w, y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(got, y);
    for (size_t r = 0; r < rows; ++r) {
      double acc = 0;
      for (size_t k = 0; k < cols; ++k)
        acc += (double)wdq[r * cols + k] * (double)xf[k];
      EXPECT_NEAR(got(r), (float)acc, 1e-3f) << "quant_gemv row " << r;
    }
  }

  // quant_gemm (M=2) vs the same reference.
  {
    const size_t M = 2;
    Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", M, cols), Y("Y", M,
                                                                        rows);
    auto hx = Kokkos::create_mirror_view(X);
    for (size_t i = 0; i < M; ++i)
      for (size_t k = 0; k < cols; ++k)
        hx(i, k) = 0.25f * (float)((int)((i + k) % 5) - 2);
    Kokkos::deep_copy(X, hx);
    Gguf::quant_gemm(Exec{}, X, w, Y);
    Kokkos::fence();
    auto got = Kokkos::create_mirror_view(Y);
    Kokkos::deep_copy(got, Y);
    for (size_t i = 0; i < M; ++i)
      for (size_t r = 0; r < rows; ++r) {
        double acc = 0;
        for (size_t k = 0; k < cols; ++k)
          acc += (double)wdq[r * cols + k] * (double)(float)hx(i, k);
        EXPECT_NEAR(got(i, r), (float)acc, 1e-3f)
            << "quant_gemm (" << i << "," << r << ")";
      }
  }
}

}  // namespace packed_track

TEST(GlimmerWeightsGGUF, PackedParityQ4K) {
  // 2 rows x 256 cols, one K-block per row, exact nibble values.
  const size_t rows = 2, cols = 256;
  std::vector<float> vals(rows * cols);
  for (size_t i = 0; i < vals.size(); ++i) vals[i] = (float)(i % 16);
  auto t = q4k_tensor("wq", {cols, rows}, vals);
  packed_track::check_packed_parity(Gguf::GgmlType::Q4_K, rows, cols, t.raw,
                                    vals);
}

TEST(GlimmerWeightsGGUF, PackedParityQ5K) {
  // 2 rows x 256 cols, exact 5-bit values (Q4_K_M uses Q5_K for the large
  // vocab/output matrices).
  const size_t rows = 2, cols = 256;
  std::vector<float> vals(rows * cols);
  for (size_t i = 0; i < vals.size(); ++i) vals[i] = (float)(i % 32);
  auto t = packed_track::q5k_tensor("wq", {cols, rows}, vals);
  packed_track::check_packed_parity(Gguf::GgmlType::Q5_K, rows, cols, t.raw,
                                    vals);
}

TEST(GlimmerWeightsGGUF, PackedParityQ6K) {
  // 2 rows x 256 cols, distinct exact constants per row (attention V /
  // projections in the Q4_K_M recipe ship Q6_K).
  const size_t rows = 2, cols = 256;
  std::vector<float> vals(rows * cols);
  for (size_t c = 0; c < cols; ++c) {
    vals[c]        = -32.0f;
    vals[cols + c] = 5.0f;
  }
  std::vector<Gguf::SyntheticTensor> ts;
  {
    // Row 0 constant -32, row 1 constant 5: build per-row blocks.
    Gguf::SyntheticTensor t;
    t.name  = "wv";
    t.dtype = Gguf::GgmlType::Q6_K;
    t.dims  = {cols, rows};
    t.raw.resize(rows * sizeof(Gguf::BlockQ6K));
    for (size_t r = 0; r < rows; ++r) {
      std::vector<float> row(cols, r == 0 ? -32.0f : 5.0f);
      auto one = q6k_const_tensor("row", {cols, 1}, (int)row[0]);
      std::memcpy(t.raw.data() + r * sizeof(Gguf::BlockQ6K), one.raw.data(),
                  sizeof(Gguf::BlockQ6K));
    }
    ts.push_back(t);
  }
  packed_track::check_packed_parity(Gguf::GgmlType::Q6_K, rows, cols,
                                    ts[0].raw, vals);
}

TEST(GlimmerWeightsGGUF, PackedParityQ80) {
  // 2 rows x 64 cols, exact int8-range values (scale 1.0 per 32-block).
  const size_t rows = 2, cols = 64;
  std::vector<float> vals(rows * cols);
  for (size_t i = 0; i < vals.size(); ++i)
    vals[i] = (float)((int)(i % 32) - 16);
  auto t = q80_tensor("wo", {cols, rows}, vals);
  packed_track::check_packed_parity(Gguf::GgmlType::Q8_0, rows, cols, t.raw,
                                    vals);
}

// The packed Q/K RoPE fixup is a row permutation done with memcpys: build a
// packed multi-row tensor with distinct per-row constants, permute the packed
// rows with the loader's rope index map, dequantize, and check the result
// equals the dense-permuted reference. Covers Q4_K (K-quant row_bytes) and
// Q8_0 (non-K-quant row_bytes); the loader must do this instead of
// dequantize -> permute -> (repack) for every q/k tensor.
TEST(GlimmerWeightsGGUF, PackedPermuteQK) {
  const size_t n_heads = 1, head_dim = 8, cols = 256;
  const size_t rows = n_heads * head_dim;
  // Each packed row holds its own index as the exact constant.
  std::vector<float> vals(rows * cols);
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c) vals[r * cols + c] = (float)r;
  auto t = q4k_tensor("q", {cols, rows}, vals);

  // Row permutation the dense loader applies (GGUF interleave -> HF): derive
  // the index map by permuting row ids through the same helper.
  std::vector<float> ids(rows * cols);
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c) ids[r * cols + c] = (float)r;
  std::vector<float> permuted = permute_for_rope(ids, n_heads, head_dim, cols);
  std::vector<size_t> perm(rows);
  for (size_t r = 0; r < rows; ++r) perm[r] = (size_t)permuted[r * cols];

  const size_t row_bytes = Gguf::ggml_row_bytes(Gguf::GgmlType::Q4_K, cols);
  std::vector<uint8_t> packed_perm =
      packed_track::permute_packed_rows(t.raw, rows, row_bytes, perm);
  std::vector<float> got =
      packed_track::host_dequant(Gguf::GgmlType::Q4_K, packed_perm, rows * cols);
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
      EXPECT_FLOAT_EQ(got[r * cols + c], (float)perm[r])
          << "packed Q4_K permute (" << r << "," << c << ")";

  // Same contract for Q8_0 (different row_bytes, same memcpy logic).
  {
    const size_t c8 = 64;
    std::vector<float> v8(rows * c8);
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < c8; ++c) v8[r * c8 + c] = (float)((int)r - 4);
    auto t8 = q80_tensor("q8", {c8, rows}, v8);
    const size_t rb8 = Gguf::ggml_row_bytes(Gguf::GgmlType::Q8_0, c8);
    std::vector<uint8_t> pp8 =
        packed_track::permute_packed_rows(t8.raw, rows, rb8, perm);
    std::vector<float> g8 = packed_track::host_dequant(Gguf::GgmlType::Q8_0,
                                                       pp8, rows * c8);
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < c8; ++c)
        EXPECT_FLOAT_EQ(g8[r * c8 + c], (float)((int)perm[r] - 4))
            << "packed Q8_0 permute (" << r << "," << c << ")";
  }
}

// Real-file gate (header-only: reads metadata + tensor infos, never tensor
// data, so it cannot OOM pre-fix). Set KOKKOSDNN_GGUF_PATH to the 17 GB
// Q4_K_M file to run; skips otherwise. Post-merge this gate grows into the
// end-to-end greedy check -- see the expected-IDs comment below.
//
// POST-MERGE VALIDATION COMMAND LIST (24 GB GPU machine):
//   1. Header:  KOKKOSDNN_GGUF_PATH=<17GB Q4_K_M file> test_glimmer_weights_gguf
//                 --gtest_filter='GlimmerWeightsGGUF.PackedRealFileGate'
//   2. Names:   list tensors via list_gguf_tensors (expect token_embd.weight,
//               output_norm.weight, output.weight, blk.{0..51}.attn_{q,k,v,
//               output,gate,norm...}, ffn_*, post_ffw_norm; Q4_K/Q6_K/Q5_K
//               dtypes; NO full dequantize)
//   3. Tiny run (synthetic, any machine):
//               test_glimmer_weights_gguf --gtest_filter='GlimmerWeightsGGUF.PackedParity*:GlimmerWeightsGGUF.PackedPermute*'
//   4. Real greedy (post-merge, packed weights resident, ~17 GB):
//               KOKKOSDNN_TOKENIZER_JSON=<tokenizer.json> glimmer_generate
//                 --gguf <file> --prompt "Hello" --temp 0 -n 8
//               EXPECT: prompt ids [200000, 19873] (llama-tokenize oracle),
//               first 8 new token ids [24, 372, 1045, 10016, 328, 2170, 262, 5091]
//               i.e. text "Hello, I am trying to make a simple..." (llama.cpp
//               CUDA greedy reference, --temp 0, -no-cnv). Any deviation is a
//               conversion/numerics bug, not sampling noise (temp 0 = argmax).
TEST(GlimmerWeightsGGUF, PackedRealFileGate) {
  const char* p = std::getenv("KOKKOSDNN_GGUF_PATH");
  if (!p || !*p)
    GTEST_SKIP() << "set KOKKOSDNN_GGUF_PATH to the 17 GB Q4_K_M GGUF to run "
                    "(header-only; reads no tensor data).";
  Gguf::GgufFile f(p);  // throws -> FAIL (missing/corrupt file)
  // Spot-check the llama.cpp tensor naming the packed loader depends on.
  EXPECT_TRUE(f.contains("token_embd.weight"));
  EXPECT_TRUE(f.contains("output_norm.weight"));
  EXPECT_TRUE(f.contains("output.weight"));
  EXPECT_TRUE(f.contains("blk.0.attn_q.weight"));
  EXPECT_TRUE(f.contains("blk.0.attn_k.weight"));
  // Q4_K_M recipe: most tensors Q4_K, some Q6_K (V/projections), large
  // vocab/output matrices Q5_K; norms stay F32.
  bool saw_q4k = false, saw_q6k = false, saw_q5k = false, saw_f32 = false;
  for (const auto& t : f.tensors()) {
    if (t.dtype == Gguf::GgmlType::Q4_K) saw_q4k = true;
    if (t.dtype == Gguf::GgmlType::Q6_K) saw_q6k = true;
    if (t.dtype == Gguf::GgmlType::Q5_K) saw_q5k = true;
    if (t.dtype == Gguf::GgmlType::F32) saw_f32 = true;
  }
  EXPECT_TRUE(saw_q4k);
  EXPECT_TRUE(saw_q6k);
  EXPECT_TRUE(saw_q5k);
  EXPECT_TRUE(saw_f32);
  std::printf("[glimmer:packed-gate] %s: %zu tensors, Q4_K/Q6_K/Q5_K/F32=%d/%d/%d/%d\n",
              p, f.tensors().size(), (int)saw_q4k, (int)saw_q6k,
              (int)saw_q5k, (int)saw_f32);
}
