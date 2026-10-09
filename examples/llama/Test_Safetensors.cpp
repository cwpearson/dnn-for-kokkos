// Safetensors round-trip: write a tiny F32/BF16 checkpoint, load it back
// into both fp32 and bfloat16 weights.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"

namespace {

uint16_t fp32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)(u >> 16);
}

uint16_t fp32_to_fp16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp  = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);  // overflow -> max
  if (exp <= 0) return (uint16_t)sign;               // underflow -> 0
  // round-to-nearest-even on the dropped 13 bits
  uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
  uint32_t rest = mant & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half;
  return (uint16_t)half;
}

float fp16_to_fp32_ref(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp  = (h & 0x7C00u) >> 10;
  uint32_t mant = (h & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while (!(mant & 0x0400u)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FFu;
      f = sign | ((exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

void write_safetensors_tiny(const std::string& path) {
  // Tensors for tiny_test with dim=8 to keep file small (override shapes).
  // We write only the tensors that loader expects for a custom cfg below.
  // For simplicity this test uses dim=8,n_layers=1,n_heads=4,n_kv=2,
  // head_dim=2,ffn=16,vocab=16.
  const int D = 8, Hq = 8, Hkv = 4, F = 16, V = 16;
  auto bf16 = [&](std::vector<float> f) {
    std::vector<char> b(f.size() * 2);
    for (size_t i = 0; i < f.size(); ++i) {
      uint16_t h = fp32_to_bf16(f[i]);
      std::memcpy(b.data() + 2 * i, &h, 2);
    }
    return b;
  };
  auto f32 = [&](std::vector<float> f) {
    std::vector<char> b(f.size() * 4);
    std::memcpy(b.data(), f.data(), b.size());
    return b;
  };
  std::vector<float> ones_D(D, 1.0f);
  std::vector<float> seqHqD(Hq * D), seqHkvD(Hkv * D), seqDHq(D * Hq),
      seqF(F), gateF(F * D), downF(D * F), embF(V * D);
  for (size_t i = 0; i < seqHqD.size(); ++i) seqHqD[i] = 0.01f * (i + 1);
  for (size_t i = 0; i < seqHkvD.size(); ++i) seqHkvD[i] = 0.02f * (i + 1);
  for (size_t i = 0; i < seqDHq.size(); ++i) seqDHq[i] = 0.03f * (i + 1);
  for (size_t i = 0; i < gateF.size(); ++i) gateF[i] = 0.04f * (i + 1);
  for (size_t i = 0; i < downF.size(); ++i) downF[i] = 0.05f * (i + 1);
  for (size_t i = 0; i < embF.size(); ++i) embF[i] = 0.06f * (i + 1);
  for (size_t i = 0; i < seqF.size(); ++i) seqF[i] = 0.07f * (i + 1);

  struct E {
    std::string name, dtype, shape;
    std::vector<char> data;
  };
  // Mix dtypes: linears BF16, norms/embed F32 — exercises conversion.
  std::vector<E> es = {
      {"model.embed_tokens.weight", "F32", "[16,8]", f32(embF)},
      {"model.norm.weight", "F32", "[8]", f32(ones_D)},
      {"lm_head.weight", "BF16", "[16,8]", bf16(embF)},
      {"model.layers.0.self_attn.q_proj.weight", "BF16", "[8,8]",
       bf16(seqHqD)},
      {"model.layers.0.self_attn.k_proj.weight", "BF16", "[4,8]",
       bf16(seqHkvD)},
      {"model.layers.0.self_attn.v_proj.weight", "BF16", "[4,8]",
       bf16(seqHkvD)},
      {"model.layers.0.self_attn.o_proj.weight", "BF16", "[8,8]",
       bf16(seqDHq)},
      {"model.layers.0.mlp.gate_proj.weight", "BF16", "[16,8]", bf16(gateF)},
      {"model.layers.0.mlp.up_proj.weight", "BF16", "[16,8]", bf16(gateF)},
      {"model.layers.0.mlp.down_proj.weight", "BF16", "[8,16]", bf16(downF)},
      {"model.layers.0.input_layernorm.weight", "F32", "[8]", f32(ones_D)},
      {"model.layers.0.post_attention_layernorm.weight", "F32", "[8]",
       f32(ones_D)},
  };
  std::string js = "{";
  size_t off     = 0;
  for (size_t i = 0; i < es.size(); ++i) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%zu,"
                  "%zu]}%s",
                  es[i].name.c_str(), es[i].dtype.c_str(), es[i].shape.c_str(),
                  off, off + es[i].data.size(),
                  i + 1 < es.size() ? "," : "");
    js += buf;
    off += es[i].data.size();
  }
  js += "}";
  std::ofstream f(path, std::ios::binary);
  uint64_t n = js.size();
  unsigned char hb[8];
  for (int i = 0; i < 8; ++i) hb[i] = (n >> (8 * i)) & 0xFF;
  f.write((char*)hb, 8);
  f.write(js.data(), js.size());
  for (auto& e : es) f.write(e.data.data(), e.data.size());
}

KokkosDNN::Llama::Config tiny_safetensors_config() {
  KokkosDNN::Llama::Config cfg;
  cfg.dim        = 8;
  cfg.n_layers   = 1;
  cfg.n_heads    = 4;
  cfg.n_kv_heads = 2;
  cfg.vocab_size = 16;
  cfg.ffn_dim    = 16;
  cfg.head_dim   = 2;
  cfg.max_seq_len = 16;
  return cfg;
}

const char* tiny_safetensors_path() {
  return "/tmp/opencode_test_tiny.safetensors";
}

}  // namespace

TEST(Safetensors, Fp32) {
  using Exec = Kokkos::DefaultExecutionSpace;
  write_safetensors_tiny(tiny_safetensors_path());
  auto cfg = tiny_safetensors_config();
  KokkosDNN::Llama::Weights<Exec, float> w;
  KokkosDNN::Llama::load_safetensors(cfg, {tiny_safetensors_path()}, w);
  // Spot-check: embed[0,0] == 0.06, q[0,0] ~= 0.01 (BF16 rounds to ~0.00999)
  auto he = Kokkos::create_mirror_view(w.embed);
  Kokkos::deep_copy(he, w.embed);
  EXPECT_NEAR(he(0, 0), 0.06f, 1e-3f);
  auto hq = Kokkos::create_mirror_view(w.blocks[0].wq);
  Kokkos::deep_copy(hq, w.blocks[0].wq);
  EXPECT_NEAR(hq(0, 0), 0.01f, 2e-3f);
}

TEST(Safetensors, BHalf) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = Kokkos::Experimental::bhalf_t;
  write_safetensors_tiny(tiny_safetensors_path());
  auto cfg = tiny_safetensors_config();
  // BF16 weights: native storage, no FP32 upcast.
  KokkosDNN::Llama::Weights<Exec, Scalar> wb;
  KokkosDNN::Llama::load_safetensors(cfg, {tiny_safetensors_path()}, wb);
  auto heb = Kokkos::create_mirror_view(wb.embed);
  Kokkos::deep_copy(heb, wb.embed);
  EXPECT_NEAR(static_cast<float>(heb(0, 0)), 0.06f, 1e-2f);
  auto hqb = Kokkos::create_mirror_view(wb.blocks[0].wq);
  Kokkos::deep_copy(hqb, wb.blocks[0].wq);
  EXPECT_NEAR(static_cast<float>(hqb(0, 0)), 0.01f, 1e-2f);
  // BF16 forward smoke: prefill tokens through the loaded weights.
  KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, wb);
  Kokkos::View<int*, Exec> toks("toks", 2);
  {
    auto ht = Kokkos::create_mirror_view(toks);
    ht(0)   = 1;
    ht(1)   = 2;
    Kokkos::deep_copy(toks, ht);
  }
  auto logits = model.forward(toks);
  EXPECT_EQ(logits.extent(0), (size_t)cfg.vocab_size);
}

TEST(Safetensors, MissingFileThrows) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto cfg   = tiny_safetensors_config();
  KokkosDNN::Llama::Weights<Exec, float> w2;
  EXPECT_THROW(
      KokkosDNN::Llama::load_safetensors(cfg, {"/nonexistent.safetensors"},
                                         w2),
      std::runtime_error);
}

// ---- mmap/streaming loader tests: synthetic tiny checkpoints ----

struct RawTensor {
  std::string name, dtype, shape;
  std::vector<char> data;
};

std::vector<char> encode_vals(const std::vector<float>& f,
                              const std::string& dtype) {
  std::vector<char> b;
  if (dtype == "F32") {
    b.resize(f.size() * 4);
    std::memcpy(b.data(), f.data(), b.size());
  } else if (dtype == "BF16") {
    b.resize(f.size() * 2);
    for (size_t i = 0; i < f.size(); ++i) {
      uint16_t h = fp32_to_bf16(f[i]);
      std::memcpy(b.data() + 2 * i, &h, 2);
    }
  } else if (dtype == "F16") {
    b.resize(f.size() * 2);
    for (size_t i = 0; i < f.size(); ++i) {
      uint16_t h = fp32_to_fp16(f[i]);
      std::memcpy(b.data() + 2 * i, &h, 2);
    }
  } else {
    throw std::runtime_error("bad test dtype");
  }
  return b;
}

void write_raw_safetensors(const std::string& path,
                           const std::vector<RawTensor>& es) {
  std::string js = "{";
  size_t off     = 0;
  for (size_t i = 0; i < es.size(); ++i) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%zu,"
                  "%zu]}%s",
                  es[i].name.c_str(), es[i].dtype.c_str(), es[i].shape.c_str(),
                  off, off + es[i].data.size(),
                  i + 1 < es.size() ? "," : "");
    js += buf;
    off += es[i].data.size();
  }
  js += "}";
  std::ofstream f(path, std::ios::binary);
  uint64_t n = js.size();
  unsigned char hb[8];
  for (int i = 0; i < 8; ++i) hb[i] = (n >> (8 * i)) & 0xFF;
  f.write((char*)hb, 8);
  f.write(js.data(), js.size());
  for (auto& e : es) f.write(e.data.data(), e.data.size());
}

// Full tiny tensor set (same shapes as write_safetensors_tiny) with
// selectable on-disk dtypes; optionally omits lm_head (tied-weights test)
// and optionally splits output across two shard files.
std::vector<RawTensor> tiny_tensors(const std::string& linear_dtype,
                                    const std::string& norm_dtype,
                                    bool include_lm_head = true) {
  const int D = 8, Hq = 8, Hkv = 4, F = 16, V = 16;
  std::vector<float> ones_D(D, 1.0f);
  std::vector<float> seqHqD(Hq * D), seqHkvD(Hkv * D), seqDHq(D * Hq),
      gateF(F * D), downF(D * F), embF(V * D);
  for (size_t i = 0; i < seqHqD.size(); ++i) seqHqD[i] = 0.01f * (i + 1);
  for (size_t i = 0; i < seqHkvD.size(); ++i) seqHkvD[i] = 0.02f * (i + 1);
  for (size_t i = 0; i < seqDHq.size(); ++i) seqDHq[i] = 0.03f * (i + 1);
  for (size_t i = 0; i < gateF.size(); ++i) gateF[i] = 0.04f * (i + 1);
  for (size_t i = 0; i < downF.size(); ++i) downF[i] = 0.05f * (i + 1);
  for (size_t i = 0; i < embF.size(); ++i) embF[i] = 0.06f * (i + 1);
  std::vector<RawTensor> es = {
      {"model.embed_tokens.weight", norm_dtype, "[16,8]",
       encode_vals(embF, norm_dtype)},
      {"model.norm.weight", norm_dtype, "[8]", encode_vals(ones_D, norm_dtype)},
      {"model.layers.0.self_attn.q_proj.weight", linear_dtype, "[8,8]",
       encode_vals(seqHqD, linear_dtype)},
      {"model.layers.0.self_attn.k_proj.weight", linear_dtype, "[4,8]",
       encode_vals(seqHkvD, linear_dtype)},
      {"model.layers.0.self_attn.v_proj.weight", linear_dtype, "[4,8]",
       encode_vals(seqHkvD, linear_dtype)},
      {"model.layers.0.self_attn.o_proj.weight", linear_dtype, "[8,8]",
       encode_vals(seqDHq, linear_dtype)},
      {"model.layers.0.mlp.gate_proj.weight", linear_dtype, "[16,8]",
       encode_vals(gateF, linear_dtype)},
      {"model.layers.0.mlp.up_proj.weight", linear_dtype, "[16,8]",
       encode_vals(gateF, linear_dtype)},
      {"model.layers.0.mlp.down_proj.weight", linear_dtype, "[8,16]",
       encode_vals(downF, linear_dtype)},
      {"model.layers.0.input_layernorm.weight", norm_dtype, "[8]",
       encode_vals(ones_D, norm_dtype)},
      {"model.layers.0.post_attention_layernorm.weight", norm_dtype, "[8]",
       encode_vals(ones_D, norm_dtype)},
  };
  if (include_lm_head)
    es.insert(es.begin() + 2, {"lm_head.weight", linear_dtype, "[16,8]",
                               encode_vals(embF, linear_dtype)});
  return es;
}

template <class Scalar>
Scalar host_at(const Kokkos::View<Scalar**, Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>& v, size_t i,
               size_t j) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  return h(i, j);
}

TEST(Safetensors, F16OnDiskBothTargets) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using BHalf = Kokkos::Experimental::bhalf_t;
  const char* p = "/tmp/opencode_test_tiny_f16.safetensors";
  write_raw_safetensors(p, tiny_tensors("F16", "F16"));
  auto cfg = tiny_safetensors_config();
  // float target: q[0,0] ~= fp16(0.01)
  {
    KokkosDNN::Llama::Weights<Exec, float> w;
    KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
    float got = host_at<float>(w.blocks[0].wq, 0, 0);
    float want =
        fp16_to_fp32_ref(fp32_to_fp16(0.01f));  // bit-exact loader contract
    uint32_t gb, wb;
    std::memcpy(&gb, &got, 4);
    std::memcpy(&wb, &want, 4);
    EXPECT_EQ(gb, wb);
    auto he = Kokkos::create_mirror_view(w.final_norm);
    Kokkos::deep_copy(he, w.final_norm);
    EXPECT_EQ(he(0), 1.0f);  // fp16(1.0) is exact
  }
  // bhalf target: bit-exact vs host-side cast of the same F16 value
  // (when bhalf_t is a native 16-bit type; otherwise it aliases float and
  // we check values with tolerance).
  {
    KokkosDNN::Llama::Weights<Exec, BHalf> w;
    KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
    BHalf got = host_at<BHalf>(w.blocks[0].wq, 0, 0);
#if !KOKKOS_BHALF_T_IS_FLOAT
    BHalf want =
        static_cast<BHalf>(fp16_to_fp32_ref(fp32_to_fp16(0.01f)));
    uint16_t gb, wb;
    std::memcpy(&gb, &got, 2);
    std::memcpy(&wb, &want, 2);
    EXPECT_EQ(gb, wb);
#else
    EXPECT_NEAR(static_cast<float>(got),
                fp16_to_fp32_ref(fp32_to_fp16(0.01f)), 1e-6f);
#endif
  }
}

TEST(Safetensors, BitExactConversions) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using BHalf = Kokkos::Experimental::bhalf_t;
  // Hand-picked on-disk bit patterns (zeros, subnormals, inf, extremes).
  std::vector<uint16_t> bf16bits = {0x0000, 0x8000, 0x3F80, 0xBF80,
                                    0x4000, 0x3DCC, 0x0001, 0x7F80};
  std::vector<uint16_t> f16bits = {0x3C00, 0xBC00, 0x7BFF, 0x0001,
                                   0x3555, 0x7C00, 0x0400, 0x8400};
  std::vector<float> f32vals = {0.0f, -0.0f, 1.0f, -1.0f,
                                2.5f, 100.0f, 1e-30f, 3.14159265f};
  auto bits_to_bytes = [](const std::vector<uint16_t>& v) {
    std::vector<char> b(v.size() * 2);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
  };
  auto f32_to_bytes = [](const std::vector<float>& v) {
    std::vector<char> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
  };
  // Pad each pattern vector to the 64-element q_proj [8,8] shape.
  auto pad8 = [](std::vector<uint16_t> v) {
    v.resize(64, 0x0000);
    return v;
  };
  auto padf = [](std::vector<float> v) {
    v.resize(64, 0.0f);
    return v;
  };
  bf16bits = pad8(bf16bits);
  f16bits  = pad8(f16bits);
  f32vals  = padf(f32vals);
  std::vector<float> ones_D(8, 1.0f);
  const char* p = "/tmp/opencode_test_bits.safetensors";
  std::vector<RawTensor> es = tiny_tensors("BF16", "F32");
  for (auto& e : es) {
    if (e.name == "model.layers.0.self_attn.q_proj.weight") {
      e.dtype = "BF16";
      e.data  = bits_to_bytes(bf16bits);
    }
  }
  write_raw_safetensors(p, es);
  std::vector<uint16_t> orig_bf16 = {0x0000, 0x8000, 0x3F80, 0xBF80,
                                     0x4000, 0x3DCC, 0x0001, 0x7F80};
  // BF16 on disk -> bhalf target must be a bit-exact memcpy (when
  // bhalf_t is a native 16-bit type; when it aliases float the loader
  // takes the float path, so check values instead).
  {
    KokkosDNN::Llama::Weights<Exec, BHalf> w;
    auto cfg = tiny_safetensors_config();
    KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
    auto hq = Kokkos::create_mirror_view(w.blocks[0].wq);
    Kokkos::deep_copy(hq, w.blocks[0].wq);
#if !KOKKOS_BHALF_T_IS_FLOAT
    for (size_t j = 0; j < orig_bf16.size(); ++j) {
      uint16_t got;
      std::memcpy(&got, &hq(0, j), 2);
      EXPECT_EQ(got, orig_bf16[j]) << "col " << j;
    }
#else
    for (size_t j = 0; j < orig_bf16.size(); ++j) {
      uint32_t f = (uint32_t)orig_bf16[j] << 16;
      float want;
      std::memcpy(&want, &f, 4);
      EXPECT_FLOAT_EQ(static_cast<float>(hq(0, j)), want) << "col " << j;
    }
#endif
  }
  // BF16 on disk -> float target must equal bit-shift expansion exactly.
  {
    KokkosDNN::Llama::Weights<Exec, float> w;
    auto cfg = tiny_safetensors_config();
    KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
    auto hq = Kokkos::create_mirror_view(w.blocks[0].wq);
    Kokkos::deep_copy(hq, w.blocks[0].wq);
    for (size_t j = 0; j < orig_bf16.size(); ++j) {
      uint32_t f = (uint32_t)orig_bf16[j] << 16;
      float want;
      std::memcpy(&want, &f, 4);
      if (orig_bf16[j] == 0x7F80) {  // inf: exact bits too
        uint32_t gb, wb;
        std::memcpy(&gb, &hq(0, j), 4);
        std::memcpy(&wb, &want, 4);
        EXPECT_EQ(gb, wb);
      } else {
        EXPECT_FLOAT_EQ(hq(0, j), want) << "col " << j;
      }
    }
  }
  // F16 on disk -> float target bit-exact vs reference decoder.
  {
    for (auto& e : es) {
      if (e.name == "model.layers.0.self_attn.q_proj.weight") {
        e.dtype = "F16";
        e.data  = bits_to_bytes(f16bits);
      }
    }
    write_raw_safetensors(p, es);
    KokkosDNN::Llama::Weights<Exec, float> w;
    auto cfg = tiny_safetensors_config();
    KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
    auto hq = Kokkos::create_mirror_view(w.blocks[0].wq);
    Kokkos::deep_copy(hq, w.blocks[0].wq);
    std::vector<uint16_t> orig_f16 = {0x3C00, 0xBC00, 0x7BFF, 0x0001,
                                      0x3555, 0x7C00, 0x0400, 0x8400};
    for (size_t j = 0; j < orig_f16.size(); ++j) {
      float want = fp16_to_fp32_ref(orig_f16[j]);
      uint32_t gb, wb;
      std::memcpy(&gb, &hq(0, j), 4);
      std::memcpy(&wb, &want, 4);
      EXPECT_EQ(gb, wb) << "col " << j;
    }
  }
  // F32 on disk -> both targets bit-exact.
  {
    for (auto& e : es) {
      if (e.name == "model.layers.0.self_attn.q_proj.weight") {
        e.dtype = "F32";
        e.data  = f32_to_bytes(f32vals);
      }
    }
    write_raw_safetensors(p, es);
    auto cfg = tiny_safetensors_config();
    KokkosDNN::Llama::Weights<Exec, float> wf;
    KokkosDNN::Llama::load_safetensors(cfg, {p}, wf);
    auto hqf = Kokkos::create_mirror_view(wf.blocks[0].wq);
    Kokkos::deep_copy(hqf, wf.blocks[0].wq);
    for (size_t j = 0; j < f32vals.size(); ++j)
      EXPECT_FLOAT_EQ(hqf(0, j), f32vals[j]) << "col " << j;
    KokkosDNN::Llama::Weights<Exec, BHalf> wb;
    KokkosDNN::Llama::load_safetensors(cfg, {p}, wb);
    auto hqb = Kokkos::create_mirror_view(wb.blocks[0].wq);
    Kokkos::deep_copy(hqb, wb.blocks[0].wq);
    for (size_t j = 0; j < f32vals.size(); ++j) {
#if !KOKKOS_BHALF_T_IS_FLOAT
      BHalf want = static_cast<BHalf>(f32vals[j]);
      uint16_t gb, wbts;
      std::memcpy(&gb, &hqb(0, j), 2);
      std::memcpy(&wbts, &want, 2);
      EXPECT_EQ(gb, wbts) << "col " << j;
#else
      EXPECT_FLOAT_EQ(static_cast<float>(hqb(0, j)), f32vals[j])
          << "col " << j;
#endif
    }
  }
}

TEST(Safetensors, MultiShard) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using BHalf = Kokkos::Experimental::bhalf_t;
  auto es = tiny_tensors("BF16", "F32");
  // Split: embedding + half the layers in shard 0, rest in shard 1.
  std::vector<RawTensor> a(es.begin(), es.begin() + 6);
  std::vector<RawTensor> b(es.begin() + 6, es.end());
  const char* pa = "/tmp/opencode_test_shard0.safetensors";
  const char* pb = "/tmp/opencode_test_shard1.safetensors";
  write_raw_safetensors(pa, a);
  write_raw_safetensors(pb, b);
  auto cfg = tiny_safetensors_config();
  KokkosDNN::Llama::Weights<Exec, float> w;
  KokkosDNN::Llama::load_safetensors(cfg, {pa, pb}, w);
  EXPECT_NEAR(host_at<float>(w.embed, 0, 0), 0.06f, 1e-6f);
  EXPECT_NEAR(host_at<float>(w.blocks[0].wq, 0, 0), 0.01f, 2e-3f);
  // bhalf target across shards as well
  KokkosDNN::Llama::Weights<Exec, BHalf> wb;
  KokkosDNN::Llama::load_safetensors(cfg, {pa, pb}, wb);
  EXPECT_NEAR(static_cast<float>(host_at<BHalf>(wb.embed, 0, 0)), 0.06f,
              1e-2f);
}

TEST(Safetensors, TiedHeadAlias) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const char* p = "/tmp/opencode_test_tied.safetensors";
  write_raw_safetensors(p, tiny_tensors("BF16", "F32",
                                        /*include_lm_head=*/false));
  // tie=true: lm_head aliases embed instead of throwing.
  {
    auto cfg                  = tiny_safetensors_config();
    cfg.tie_word_embeddings = true;
    KokkosDNN::Llama::Weights<Exec, float> w;
    KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
    EXPECT_EQ(w.lm_head.data(), w.embed.data());
    EXPECT_NEAR(host_at<float>(w.lm_head, 0, 0), 0.06f, 1e-6f);
  }
  // tie=false: missing lm_head still throws the informative error.
  {
    auto cfg = tiny_safetensors_config();
    EXPECT_FALSE(cfg.tie_word_embeddings);
    KokkosDNN::Llama::Weights<Exec, float> w;
    try {
      KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
      FAIL() << "expected missing lm_head to throw";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("lm_head.weight"),
                std::string::npos);
    }
  }
}

TEST(Safetensors, ShapeMismatchThrows) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto es    = tiny_tensors("BF16", "F32");
  for (auto& e : es)
    if (e.name == "model.norm.weight") e.shape = "[7]";  // wrong, same bytes
  const char* p = "/tmp/opencode_test_badshape.safetensors";
  write_raw_safetensors(p, es);
  auto cfg = tiny_safetensors_config();
  KokkosDNN::Llama::Weights<Exec, float> w;
  EXPECT_THROW(KokkosDNN::Llama::load_safetensors(cfg, {p}, w),
               std::runtime_error);
}

TEST(Safetensors, UnsupportedDtypeThrows) {
  using Exec = Kokkos::DefaultExecutionSpace;
  auto es    = tiny_tensors("BF16", "F32");
  for (auto& e : es)
    if (e.name == "model.norm.weight") {
      e.dtype = "I32";
      e.data  = std::vector<char>(8 * 4, 0);
    }
  const char* p = "/tmp/opencode_test_baddtype.safetensors";
  write_raw_safetensors(p, es);
  auto cfg = tiny_safetensors_config();
  KokkosDNN::Llama::Weights<Exec, float> w;
  EXPECT_THROW(KokkosDNN::Llama::load_safetensors(cfg, {p}, w),
               std::runtime_error);
}

TEST(Safetensors, MultiChunkStreaming) {
  // embed [2048,256] = 524288 BF16 elems forces >1 streaming chunk
  // (chunk = 2^18 elems), exercising chunk-boundary correctness.
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::Llama::Config cfg;
  cfg.dim         = 256;
  cfg.n_layers    = 1;
  cfg.n_heads     = 8;
  cfg.n_kv_heads  = 2;
  cfg.vocab_size  = 2048;
  cfg.ffn_dim     = 512;
  cfg.head_dim    = 32;
  cfg.max_seq_len = 16;
  const int D = 256, Hq = 256, Hkv = 64, F = 512, V = 2048;
  auto seq = [](size_t n, float s) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = s * (i + 1);
    return v;
  };
  std::vector<RawTensor> es = {
      {"model.embed_tokens.weight", "BF16", "[2048,256]",
       encode_vals(seq((size_t)V * D, 1e-4f), "BF16")},
      {"model.norm.weight", "F32", "[256]",
       encode_vals(std::vector<float>(D, 1.0f), "F32")},
      {"lm_head.weight", "BF16", "[2048,256]",
       encode_vals(seq((size_t)V * D, 1e-4f), "BF16")},
      {"model.layers.0.self_attn.q_proj.weight", "F32", "[256,256]",
       encode_vals(seq((size_t)Hq * D, 1e-4f), "F32")},
      {"model.layers.0.self_attn.k_proj.weight", "F32", "[64,256]",
       encode_vals(seq((size_t)Hkv * D, 1e-4f), "F32")},
      {"model.layers.0.self_attn.v_proj.weight", "F32", "[64,256]",
       encode_vals(seq((size_t)Hkv * D, 1e-4f), "F32")},
      {"model.layers.0.self_attn.o_proj.weight", "F32", "[256,256]",
       encode_vals(seq((size_t)D * Hq, 1e-4f), "F32")},
      {"model.layers.0.mlp.gate_proj.weight", "F32", "[512,256]",
       encode_vals(seq((size_t)F * D, 1e-4f), "F32")},
      {"model.layers.0.mlp.up_proj.weight", "F32", "[512,256]",
       encode_vals(seq((size_t)F * D, 1e-4f), "F32")},
      {"model.layers.0.mlp.down_proj.weight", "F32", "[256,512]",
       encode_vals(seq((size_t)D * F, 1e-4f), "F32")},
      {"model.layers.0.input_layernorm.weight", "F32", "[256]",
       encode_vals(std::vector<float>(D, 1.0f), "F32")},
      {"model.layers.0.post_attention_layernorm.weight", "F32", "[256]",
       encode_vals(std::vector<float>(D, 1.0f), "F32")},
  };
  const char* p = "/tmp/opencode_test_chunks.safetensors";
  write_raw_safetensors(p, es);
  KokkosDNN::Llama::Weights<Exec, float> w;
  KokkosDNN::Llama::load_safetensors(cfg, {p}, w);
  auto want = [&](size_t i, size_t j) {
    float v = 1e-4f * (i * (size_t)D + j + 1);
    uint16_t h = fp32_to_bf16(v);
    uint32_t f = (uint32_t)h << 16;
    float o;
    std::memcpy(&o, &f, 4);
    return o;
  };
  // rows 1023/1024 straddle the 1024-row chunk boundary (chunk = 2^18
  // elems / 256 cols).
  EXPECT_FLOAT_EQ(host_at<float>(w.embed, 0, 0), want(0, 0));
  EXPECT_FLOAT_EQ(host_at<float>(w.embed, 1023, 255), want(1023, 255));
  EXPECT_FLOAT_EQ(host_at<float>(w.embed, 1024, 0), want(1024, 0));
  EXPECT_FLOAT_EQ(host_at<float>(w.embed, 2047, 255), want(2047, 255));
}
