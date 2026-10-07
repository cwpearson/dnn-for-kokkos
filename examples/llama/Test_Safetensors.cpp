// Safetensors round-trip: write a tiny F32/BF16 checkpoint, load it back
// into both fp32 and bfloat16 weights.
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"

static uint16_t fp32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)(u >> 16);
}

static void write_safetensors_tiny(const std::string& path) {
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
  size_t off = 0;
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

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    const std::string path = "/tmp/opencode_test_tiny.safetensors";
    write_safetensors_tiny(path);
    KokkosDNN::Llama::Config cfg;
    cfg.dim = 8;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.vocab_size = 16;
    cfg.ffn_dim = 16;
    cfg.head_dim = 2;
    cfg.max_seq_len = 16;
    KokkosDNN::Llama::Weights<Exec, float> w;
    KokkosDNN::Llama::load_safetensors(cfg, {path}, w);
    // Spot-check: embed[0,0] == 0.06, q[0,0] ~= 0.01 (BF16 rounds to ~0.00999)
    auto he = Kokkos::create_mirror_view(w.embed);
    Kokkos::deep_copy(he, w.embed);
    if (std::abs(he(0, 0) - 0.06f) > 1e-3f) {
      std::printf("FAIL safetensors[fp32] embed %f\n", he(0, 0));
      return 1;
    }
    auto hq = Kokkos::create_mirror_view(w.blocks[0].wq);
    Kokkos::deep_copy(hq, w.blocks[0].wq);
    if (std::abs(hq(0, 0) - 0.01f) > 2e-3f) {
      std::printf("FAIL safetensors[fp32] bf16 %f\n", hq(0, 0));
      return 1;
    }
    // BF16 weights: native storage, no FP32 upcast.
    {
      using Scalar = Kokkos::Experimental::bhalf_t;
      KokkosDNN::Llama::Weights<Exec, Scalar> wb;
      KokkosDNN::Llama::load_safetensors(cfg, {path}, wb);
      auto heb = Kokkos::create_mirror_view(wb.embed);
      Kokkos::deep_copy(heb, wb.embed);
      if (std::abs(static_cast<float>(heb(0, 0)) - 0.06f) > 1e-2f) {
        std::printf("FAIL safetensors[bf16] embed %f\n",
                    static_cast<float>(heb(0, 0)));
        return 1;
      }
      auto hqb = Kokkos::create_mirror_view(wb.blocks[0].wq);
      Kokkos::deep_copy(hqb, wb.blocks[0].wq);
      if (std::abs(static_cast<float>(hqb(0, 0)) - 0.01f) > 1e-2f) {
        std::printf("FAIL safetensors[bf16] bf16 %f\n",
                    static_cast<float>(hqb(0, 0)));
        return 1;
      }
      // BF16 forward smoke: prefill tokens through the loaded weights.
      KokkosDNN::Llama::Model<Exec, Scalar> model(cfg, wb);
      Kokkos::View<int*, Exec> toks("toks", 2);
      {
        auto ht = Kokkos::create_mirror_view(toks);
        ht(0) = 1;
        ht(1) = 2;
        Kokkos::deep_copy(toks, ht);
      }
      auto logits = model.forward(toks);
      if (logits.extent(0) != (size_t)cfg.vocab_size) {
        std::printf("FAIL safetensors[bf16] forward shape\n");
        return 1;
      }
    }
    // Missing tensor must throw.
    bool threw = false;
    try {
      KokkosDNN::Llama::Weights<Exec, float> w2;
      KokkosDNN::Llama::load_safetensors(cfg, {"/nonexistent.safetensors"},
                                         w2);
    } catch (...) {
      threw = true;
    }
    if (!threw) {
      std::printf("FAIL safetensors missing-file\n");
      return 1;
    }
    std::printf("PASS test_safetensors\n");
  }
  Kokkos::finalize();
  return 0;
}
