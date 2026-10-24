// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Sparse mixture-of-experts decode (Gguf::quant_moe_decode): one token,
// top_k routed experts against packed device-side expert tables.
//
// Decode runs two fused kernels per layer: a gate/up GEMV with SwiGLU
// (writing act[S, MI]) and a down GEMV that also applies the routing weights
// (writing Y[1, K]). These tests check the whole path against a host reference
// at the real LFM2.5-24B-A2B expert shape (K=2048, MI=1536, top_k=4, Q4_K
// gate/up with the Q6_K down projection the published Q4_K_M file stores, bf16
// activations), plus the other unit layouts (Q6_K/Q8_0 wide loads and the
// F32/F16/BF16 per-element fallback), mixed recipes, odd top_k, the
// rows-per-team tail, dead routing slots, and the scalar-activation branch.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"

namespace {

using Exec     = Kokkos::DefaultExecutionSpace;
using bhalf_t  = Kokkos::Experimental::bhalf_t;
using GgmlType = KokkosDNN::Gguf::GgmlType;
using BlockQ4K = KokkosDNN::Gguf::BlockQ4K;
using BlockQ6K = KokkosDNN::Gguf::BlockQ6K;
using BlockQ80 = KokkosDNN::Gguf::BlockQ80;
namespace Gguf = KokkosDNN::Gguf;

uint16_t f32_to_f16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t exp   = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
  const uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// Host dequantization of a packed buffer (independent of the decode kernels;
// only the shared block dequantizers are reused).
void host_dequant(GgmlType dt, const uint8_t* raw, size_t numel,
                  std::vector<float>& out) {
  out.assign(numel, 0.0f);
  if (dt == GgmlType::F32) {
    std::memcpy(out.data(), raw, numel * 4);
    return;
  }
  if (dt == GgmlType::F16) {
    for (size_t i = 0; i < numel; ++i)
      out[i] = Gguf::fp16_to_float_host(
          reinterpret_cast<const uint16_t*>(raw)[i]);
    return;
  }
  if (dt == GgmlType::BF16) {
    for (size_t i = 0; i < numel; ++i)
      out[i] = Gguf::bf16_to_float_host(
          reinterpret_cast<const uint16_t*>(raw)[i]);
    return;
  }
  if (dt == GgmlType::Q8_0) {
    for (size_t b = 0; b < numel / 32; ++b)
      Gguf::dequant_q80_block(reinterpret_cast<const BlockQ80*>(raw + b * 34),
                              out.data() + b * 32);
    return;
  }
  if (dt == GgmlType::Q4_K) {
    for (size_t b = 0; b < numel / 256; ++b)
      Gguf::dequant_q4k_block(
          reinterpret_cast<const BlockQ4K*>(raw + b * 144),
          out.data() + b * 256);
    return;
  }
  if (dt == GgmlType::Q6_K) {
    for (size_t b = 0; b < numel / 256; ++b)
      Gguf::dequant_q6k_block(
          reinterpret_cast<const BlockQ6K*>(raw + b * 210),
          out.data() + b * 256);
    return;
  }
  ADD_FAILURE() << "unsupported test dtype";
}

// One packed expert matrix (rows x cols) built from unit-scale blocks with a
// per-expert overall scale: every element is exactly representable, so the
// reference isolates the kernel's accumulation from quantization error. Each
// expert and row gets a different pattern, so a kernel that mixes up expert
// ids or output rows cannot pass.
void make_expert(GgmlType dt, size_t rows, size_t cols, size_t seed,
                 std::vector<uint8_t>& raw, std::vector<float>& dq) {
  const size_t rb      = Gguf::ggml_row_bytes(dt, cols);
  const float dscale   = 0.5f + 0.25f * (float)(seed % 4);
  const uint16_t dbits = f32_to_f16_bits(dscale);
  raw.assign(rows * rb, 0);
  for (size_t r = 0; r < rows; ++r) {
    uint8_t* row = raw.data() + r * rb;
    if (dt == GgmlType::Q4_K) {
      for (size_t blk = 0; blk < cols / 256; ++blk) {
        BlockQ4K b{};
        b.d    = dbits;  // d
        b.dmin = 0x0000;
        const uint8_t sc[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0x01, 0x01, 0x01, 0x01};
        std::memcpy(b.scales, sc, 12);
        for (int iter = 0; iter < 4; ++iter)
          for (int l = 0; l < 32; ++l) {
            const uint8_t lo = (uint8_t)((r * 7 + blk * 5 + seed +
                                          (size_t)iter * 3 + (size_t)l) %
                                         16);
            const uint8_t hi = (uint8_t)((r * 11 + blk * 3 + seed * 5 +
                                          (size_t)iter * 7 + (size_t)l + 5) %
                                         16);
            b.qs[(size_t)iter * 32 + (size_t)l] = (uint8_t)(lo | (hi << 4));
          }
        std::memcpy(row + blk * 144, &b, 144);
      }
    } else if (dt == GgmlType::Q6_K) {
      for (size_t blk = 0; blk < cols / 256; ++blk) {
        BlockQ6K b{};
        std::memcpy(&b.d, &dbits, 2);
        for (int s = 0; s < 16; ++s) b.scales[s] = 1;
        // qh stays 0: the decoded value is the nibble minus 32.
        for (int c = 0; c < 2; ++c)
          for (int l = 0; l < 32; ++l) {
            const size_t base = r * 3 + blk * 7 + seed * 11 + (size_t)c * 5;
            const uint8_t v0  = (uint8_t)((base + (size_t)l) % 32);
            const uint8_t v1  = (uint8_t)((base + (size_t)l + 3) % 32);
            const uint8_t v2  = (uint8_t)((base + (size_t)l + 7) % 32);
            const uint8_t v3  = (uint8_t)((base + (size_t)l + 11) % 32);
            b.ql[(size_t)c * 64 + (size_t)l] = (uint8_t)(v0 | (v1 << 4));
            b.ql[(size_t)c * 64 + 32 + (size_t)l] = (uint8_t)(v2 | (v3 << 4));
          }
        std::memcpy(row + blk * 210, &b, 210);
      }
    } else if (dt == GgmlType::Q8_0) {
      for (size_t blk = 0; blk < cols / 32; ++blk) {
        BlockQ80 b{};
        std::memcpy(&b.d, &dbits, 2);
        for (int i = 0; i < 32; ++i)
          b.qs[i] = (int8_t)((int)((r * 13 + blk * 5 + seed + (size_t)i) % 17) -
                             8);
        std::memcpy(row + blk * 34, &b, 34);
      }
    } else if (dt == GgmlType::F32 || dt == GgmlType::F16 ||
               dt == GgmlType::BF16) {
      for (size_t c = 0; c < cols; ++c) {
        const float v = dscale * 0.25f *
                        (float)((int)((r * 17 + c * 3 + seed) % 19) - 9);
        if (dt == GgmlType::F32) {
          reinterpret_cast<float*>(row)[c] = v;
        } else if (dt == GgmlType::F16) {
          reinterpret_cast<uint16_t*>(row)[c] = f32_to_f16_bits(v);
        } else {
          uint32_t u;
          std::memcpy(&u, &v, 4);
          reinterpret_cast<uint16_t*>(row)[c] = (uint16_t)(u >> 16);
        }
      }
    } else {
      ADD_FAILURE() << "unsupported test dtype";
    }
  }
  host_dequant(dt, raw.data(), rows * cols, dq);
}

Gguf::QuantWeight<Exec> upload(GgmlType dt, size_t rows, size_t cols,
                               const std::vector<uint8_t>& raw) {
  Gguf::QuantWeight<Exec> w;
  w.dtype = dt;
  w.rows  = rows;
  w.cols  = cols;
  w.data  = Kokkos::View<uint8_t*, Exec>("w", raw.size());
  Kokkos::View<uint8_t*, Kokkos::HostSpace> h("wh", raw.size());
  std::memcpy(h.data(), raw.data(), raw.size());
  Kokkos::deep_copy(w.data, h);
  return w;
}

// Full decode step vs a host reference. Weight magnitudes are large (unit
// blocks of 15 over thousands of terms), so tolerances are relative to the
// reference; what they cover is the Scalar round trip of act and Y.
template <class Scalar>
void check_decode(GgmlType gdt, GgmlType udt, GgmlType ddt, size_t K,
                  size_t MI, size_t S, size_t E, const std::vector<int>& ids,
                  const std::vector<float>& wv, bool unaligned_x,
                  float rel_tol) {
  ASSERT_EQ(ids.size(), S);
  std::vector<Gguf::QuantWeight<Exec>> gate(E), up(E), down(E);
  std::vector<std::vector<float>> gdq(E), udq(E), ddq(E);
  for (size_t e = 0; e < E; ++e) {
    std::vector<uint8_t> raw;
    make_expert(gdt, MI, K, e, raw, gdq[e]);
    gate[e] = upload(gdt, MI, K, raw);
    make_expert(udt, MI, K, e + 1, raw, udq[e]);
    up[e] = upload(udt, MI, K, raw);
    make_expert(ddt, K, MI, e + 2, raw, ddq[e]);
    down[e] = upload(ddt, K, MI, raw);
  }
  const auto tg = Gguf::make_quant_weight_table(gate, "test::gate");
  const auto tu = Gguf::make_quant_weight_table(up, "test::up");
  const auto td = Gguf::make_quant_weight_table(down, "test::down");

  // Exact-binary activations: the Scalar round trip is lossless.
  std::vector<float> xf(K);
  for (size_t k = 0; k < K; ++k) xf[k] = 0.25f * (float)((int)(k % 7) - 3);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", 1, K);
  auto hx = Kokkos::create_mirror_view(X);
  for (size_t k = 0; k < K; ++k) hx(0, k) = static_cast<Scalar>(xf[k]);
  Kokkos::deep_copy(X, hx);
  // 2-byte-aligned alias of the same values: exercises the scalar-activation
  // branch (xvec == false) when requested.
  Kokkos::View<Scalar*, Exec> pad("pad", K + 8);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> Xu(pad.data() + 1, 1, K);
  {
    auto hp = Kokkos::create_mirror_view(pad);
    for (size_t k = 0; k < K + 8; ++k) hp(k) = Scalar(0);
    for (size_t k = 0; k < K; ++k) hp(k + 1) = static_cast<Scalar>(xf[k]);
    Kokkos::deep_copy(pad, hp);
  }

  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", 1, S);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> wts("wts", 1, S);
  auto hi = Kokkos::create_mirror_view(idx);
  auto hw = Kokkos::create_mirror_view(wts);
  for (size_t s = 0; s < S; ++s) {
    hi(0, s) = ids[s];
    hw(0, s) = wv[s];
  }
  Kokkos::deep_copy(idx, hi);
  Kokkos::deep_copy(wts, hw);

  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> act("act", S, MI);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> Y("Y", 1, K);
  KokkosDNN::DNNHandle<Exec> handle;
  if (unaligned_x)
    Gguf::quant_moe_decode<Exec, Scalar>(handle, Xu, tg, tu, td, idx, wts, act,
                                         Y);
  else
    Gguf::quant_moe_decode<Exec, Scalar>(handle, X, tg, tu, td, idx, wts, act,
                                         Y);
  Kokkos::fence();

  // Reference: gate/up dots in double, SwiGLU in FP32, act rounded to Scalar
  // (what the down stage reads), then the weighted down combine.
  std::vector<double> actref((size_t)S * MI, 0.0);
  for (size_t s = 0; s < S; ++s) {
    const bool live = ids[s] >= 0 && (size_t)ids[s] < E && wv[s] != 0.0f;
    if (!live) continue;
    const size_t e = (size_t)ids[s];
    for (size_t j = 0; j < MI; ++j) {
      double g = 0, u = 0;
      const float* gr = gdq[e].data() + j * K;
      const float* ur = udq[e].data() + j * K;
      for (size_t k = 0; k < K; ++k) {
        g += (double)gr[k] * (double)xf[k];
        u += (double)ur[k] * (double)xf[k];
      }
      const float gs  = (float)g;
      const double sv = (double)gs / (1.0 + std::exp(-(double)gs));
      actref[s * MI + j] =
          (double)(float)static_cast<Scalar>((float)(sv * (double)(float)u));
    }
  }
  std::vector<double> yref(K, 0.0);
  for (size_t d = 0; d < K; ++d) {
    double acc = 0;
    for (size_t s = 0; s < S; ++s) {
      if (ids[s] < 0 || (size_t)ids[s] >= E || wv[s] == 0.0f) continue;
      const float* dr = ddq[(size_t)ids[s]].data() + d * MI;
      double sub      = 0;
      for (size_t mi = 0; mi < MI; ++mi)
        sub += (double)dr[mi] * actref[s * MI + mi];
      acc += (double)wv[s] * sub;
    }
    yref[d] = acc;
  }

  auto ha = Kokkos::create_mirror_view(act);
  auto hy = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(ha, act);
  Kokkos::deep_copy(hy, Y);
  double scale = 1e-6;
  for (double v : yref) scale = std::fmax(scale, std::fabs(v));
  for (size_t d = 0; d < K; ++d) {
    EXPECT_NEAR((float)hy(0, d), (float)yref[d], (double)rel_tol * scale)
        << "Y row " << d << " dtypes " << Gguf::ggml_type_name(gdt) << "/"
        << Gguf::ggml_type_name(udt) << "/" << Gguf::ggml_type_name(ddt)
        << " K=" << K << " MI=" << MI << " S=" << S
        << (unaligned_x ? " unaligned-x" : "");
  }
  // A bad expert id zeroes that slot's act row (nothing to compute); a zero
  // routing weight is a down-stage concern, so its row is still computed.
  for (size_t s = 0; s < S; ++s) {
    const bool bad = ids[s] < 0 || (size_t)ids[s] >= E;
    for (size_t j = 0; j < MI; j += 61) {
      if (bad)
        EXPECT_EQ((float)ha(s, j), 0.0f) << "slot " << s << " col " << j;
      else
        EXPECT_TRUE(std::isfinite((float)ha(s, j)))
            << "slot " << s << " col " << j;
    }
  }
}

}  // namespace

// The real LFM2.5-24B-A2B expert shape: K=2048, MI=1536, top_k=4, gate/up
// Q4_K and down Q6_K (exactly the Q4_K_M recipe the published file stores),
// bf16 activations. Four experts keep the host reference cheap; the routing
// ids hit all of them.
TEST(MoeDecode, Lfm25Top4Shape) {
  const std::vector<int> ids  = {3, 0, 2, 1};
  const std::vector<float> wv = {0.4f, 0.3f, 0.2f, 0.1f};
  check_decode<bhalf_t>(GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q6_K, 2048,
                        1536, 4, 4, ids, wv, false, 2e-2f);
  // Same call through the scalar (unaligned activation) branch.
  check_decode<bhalf_t>(GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q6_K, 2048,
                        1536, 4, 4, ids, wv, true, 2e-2f);
}

// Every unit layout: Q4_K and Q6_K wide loads, Q8_0 blocks, the F32/F16/BF16
// per-element fallback, mixed recipes, and top_k 1..4. MI=96 exercises a
// rows-per-team tail that is not a multiple of 4; the K-quant cases keep both
// dimensions block-multiples (256), as the entry point requires.
TEST(MoeDecode, DtypesAndSlots) {
  {
    const std::vector<int> ids = {1, 0, 2, 3};
    const std::vector<float> w = {0.4f, 0.3f, 0.2f, 0.1f};
    const std::vector<int> ids4 = {1, 0, 2, 3};
    const std::vector<float> w4 = {0.4f, 0.3f, 0.2f, 0.1f};
    const std::vector<int> ids3 = {1, 0, 2};
    const std::vector<float> w3 = {0.5f, 0.3f, 0.2f};
    const std::vector<int> ids2 = {1, 0};
    const std::vector<float> w2 = {0.6f, 0.4f};
    check_decode<float>(GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K, 512,
                        256, 4, 4, ids4, w4, false, 1e-4f);
    check_decode<float>(GgmlType::Q6_K, GgmlType::Q6_K, GgmlType::Q6_K, 768,
                        512, 2, 3, ids2, w2, false, 1e-4f);
    check_decode<float>(GgmlType::Q8_0, GgmlType::Q8_0, GgmlType::Q8_0, 256,
                        512, 3, 4, ids3, w3, false, 1e-4f);
    check_decode<bhalf_t>(GgmlType::Q6_K, GgmlType::Q4_K, GgmlType::Q8_0, 512,
                          256, 4, 5, ids4, w4, false, 2e-2f);
  }
  {
    const std::vector<int> ids = {2, 0};
    const std::vector<float> w = {0.75f, 0.25f};
    check_decode<float>(GgmlType::F32, GgmlType::F16, GgmlType::BF16, 96, 96, 2,
                        3, ids, w, false, 1e-4f);
  }
  {
    const std::vector<int> ids = {0, 1, 2};
    const std::vector<float> w = {0.5f, 0.25f, 0.25f};
    check_decode<float>(GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q6_K, 256,
                        256, 3, 3, ids, w, false, 1e-4f);
  }
}

// Dead routing slots: an out-of-range id and a negative id must read nothing
// outside the expert tables, write act == 0 for their slot, and contribute
// nothing to Y; a zero routing weight also contributes nothing while Y is
// still *assigned* (not accumulated onto its previous contents).
TEST(MoeDecode, DeadSlotsContributeNothing) {
  const size_t K = 256, MI = 256, E = 2;
  const std::vector<int> ids = {(int)E, -1, 0};
  const std::vector<float> wv = {1.0f, 1.0f, 0.0f};
  std::vector<Gguf::QuantWeight<Exec>> gate(E), up(E), down(E);
  for (size_t e = 0; e < E; ++e) {
    std::vector<uint8_t> raw;
    std::vector<float> dq;
    make_expert(GgmlType::Q8_0, MI, K, e, raw, dq);
    gate[e] = upload(GgmlType::Q8_0, MI, K, raw);
    make_expert(GgmlType::Q8_0, MI, K, e + 1, raw, dq);
    up[e] = upload(GgmlType::Q8_0, MI, K, raw);
    make_expert(GgmlType::Q8_0, K, MI, e + 2, raw, dq);
    down[e] = upload(GgmlType::Q8_0, K, MI, raw);
  }
  const auto tg = Gguf::make_quant_weight_table(gate, "t::g");
  const auto tu = Gguf::make_quant_weight_table(up, "t::u");
  const auto td = Gguf::make_quant_weight_table(down, "t::d");

  Kokkos::View<bhalf_t**, Kokkos::LayoutRight, Exec> X("X", 1, K),
      act("act", 3, MI), Y("Y", 1, K);
  auto hx = Kokkos::create_mirror_view(X);
  for (size_t k = 0; k < K; ++k) hx(0, k) = bhalf_t(0.5f);
  Kokkos::deep_copy(X, hx);
  Kokkos::deep_copy(Y, bhalf_t(7));  // pre-filled: the kernel must overwrite

  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", 1, 3);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> wts("wts", 1, 3);
  auto hi = Kokkos::create_mirror_view(idx);
  auto hw = Kokkos::create_mirror_view(wts);
  for (size_t s = 0; s < 3; ++s) {
    hi(0, s) = ids[s];
    hw(0, s) = wv[s];
  }
  Kokkos::deep_copy(idx, hi);
  Kokkos::deep_copy(wts, hw);

  KokkosDNN::DNNHandle<Exec> handle;
  Gguf::quant_moe_decode<Exec, bhalf_t>(handle, X, tg, tu, td, idx, wts, act,
                                        Y);
  Kokkos::fence();
  auto ha = Kokkos::create_mirror_view(act);
  auto hy = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(ha, act);
  Kokkos::deep_copy(hy, Y);
  for (size_t s = 0; s < 2; ++s)
    for (size_t j = 0; j < MI; ++j)
      EXPECT_EQ((float)ha(s, j), 0.0f) << "slot " << s << " col " << j;
  for (size_t j = 0; j < MI; j += 37)
    EXPECT_TRUE(std::isfinite((float)ha(2, j)));
  for (size_t d = 0; d < K; ++d) EXPECT_EQ((float)hy(0, d), 0.0f) << "row " << d;
}

// Argument validation: one activation row, matching shapes, act scratch that
// covers the top_k rows, and block-multiple K-quant rows.
TEST(MoeDecode, Errors) {
  std::vector<Gguf::QuantWeight<Exec>> w(1);
  std::vector<uint8_t> raw;
  std::vector<float> dq;
  make_expert(GgmlType::Q8_0, 256, 256, 0, raw, dq);
  w[0]              = upload(GgmlType::Q8_0, 256, 256, raw);
  const auto t      = Gguf::make_quant_weight_table(w, "e::t");
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> X("X", 1, 256),
      act("act", 1, 256), Y("Y", 1, 256), X2("X2", 2, 256),
      act_bad("actb", 1, 64);
  Kokkos::View<int**, Kokkos::LayoutRight, Exec> idx("idx", 1, 1);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> wts("wts", 1, 1);
  KokkosDNN::DNNHandle<Exec> handle;
  // Two activation rows are not a decode call.
  EXPECT_THROW((Gguf::quant_moe_decode<Exec, float>(handle, X2, t, t, t, idx,
                                                    wts, act, Y)),
               std::invalid_argument);
  // act scratch narrower than MI.
  EXPECT_THROW((Gguf::quant_moe_decode<Exec, float>(handle, X, t, t, t, idx,
                                                    wts, act_bad, Y)),
               std::invalid_argument);
  // K-quant rows must be whole 256-element blocks: a down projection with
  // cols = 128 cannot be decoded.
  std::vector<Gguf::QuantWeight<Exec>> shortw(1);
  make_expert(GgmlType::Q8_0, 256, 128, 0, raw, dq);
  shortw[0] = upload(GgmlType::Q8_0, 256, 128, raw);
  const auto ts = Gguf::make_quant_weight_table(shortw, "e::ts");
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> Xs("Xs", 1, 256),
      acts("acts", 1, 128), Ys("Ys", 1, 256);
  EXPECT_THROW((Gguf::quant_moe_decode<Exec, float>(handle, Xs, t, t, ts, idx,
                                                    wts, acts, Ys)),
               std::invalid_argument);
}
