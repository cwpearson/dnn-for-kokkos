// GGUF + K-quant (Q4_K_M family) tests. No real weights are used: every file
// is synthesized into /tmp by write_synthetic_gguf and removed afterwards.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "KokkosDNN_GGUF.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using namespace KokkosDNN::Gguf;

// ---------------------------------------------------------------------------
// Test-local float <-> half bit converters (reference for building payloads).
// ---------------------------------------------------------------------------
uint16_t f32_to_f16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t e    = (u >> 23) & 0xFFu;
  const uint32_t m    = u & 0x7FFFFFu;
  if (e == 0xFFu) return (uint16_t)(sign | 0x7BFFu);  // tests avoid inf/nan
  const int hexp = (int)e - 127 + 15;
  if (hexp >= 31) return (uint16_t)(sign | 0x7BFFu);
  if (hexp <= 0) {
    if (hexp < -10) return (uint16_t)sign;
    const uint32_t mm    = m | 0x800000u;
    const uint32_t shift = (uint32_t)(14 - hexp);
    uint32_t h           = mm >> shift;
    if (((mm >> (shift - 1)) & 1u) != 0) h += 1;  // round-half-up
    return (uint16_t)(sign | (h & 0x3FFu));
  }
  uint32_t h = m + 0x1000u;  // 0.5 ulp
  int he     = hexp;
  if ((h & 0x800000u) != 0) {
    h  = 0;
    he += 1;
  }
  if (he >= 31) return (uint16_t)(sign | 0x7BFFu);
  return (uint16_t)(sign | ((uint32_t)he << 10) | (h >> 13));
}

uint16_t f32_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if (((u >> 23) & 0xFFu) == 0xFFu)
    return (uint16_t)(0x7FC0u | ((u >> 16) & 0x8000u));
  const uint32_t r = u + 0x7FFFu + ((u >> 16) & 1u);  // round-to-nearest-even
  return (uint16_t)(r >> 16);
}

// Reference Q8_0 quantizer (round-to-nearest, one 32-block).
void quant_q80_block(const float* x, BlockQ80* b) {
  float amax = 0;
  for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[i]));
  const float d = (amax == 0.0f) ? 1.0f : amax / 127.0f;
  b->d          = f32_to_f16_bits(d);
  for (int i = 0; i < 32; ++i) {
    long q = std::lround(x[i] / d);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    b->qs[i] = (int8_t)q;
  }
}

// Q4_K block with unit scale / zero min: dequantized value == nibble (0..15).
// scales: j<4 -> (1,0); j>=4 -> (1,0); dall=1.0, dmin=0.0.
BlockQ4K make_unit_q4k() {
  BlockQ4K b{};
  b.d    = 0x3C00;  // 1.0
  b.dmin = 0x0000;  // 0.0
  const uint8_t s[12] = {1, 1, 1, 1, 0, 0, 0, 0, 0x01, 0x01, 0x01, 0x01};
  std::memcpy(b.scales, s, 12);
  return b;
}

template <class ExecSpace>
QuantWeight<ExecSpace> make_qw(GgmlType dt, size_t rows, size_t cols,
                               const std::vector<uint8_t>& bytes) {
  QuantWeight<ExecSpace> w;
  w.dtype = dt;
  w.rows  = rows;
  w.cols  = cols;
  EXPECT_EQ(bytes.size(), w.bytes());
  w.data = Kokkos::View<uint8_t*, ExecSpace>("qw", bytes.size());
  Kokkos::View<uint8_t*, Kokkos::HostSpace> h("qwh", bytes.size());
  std::memcpy(h.data(), bytes.data(), bytes.size());
  Kokkos::deep_copy(w.data, h);
  return w;
}

// Host dequant dispatcher for test references (uses the header's block fns).
void host_dequant(GgmlType dt, const uint8_t* raw, size_t numel,
                  std::vector<float>& out) {
  out.resize(numel);
  switch (dt) {
    case GgmlType::F32:
      std::memcpy(out.data(), raw, numel * 4);
      return;
    case GgmlType::F16:
      for (size_t i = 0; i < numel; ++i)
        out[i] = fp16_to_float_host(
            reinterpret_cast<const uint16_t*>(raw)[i]);
      return;
    case GgmlType::BF16:
      for (size_t i = 0; i < numel; ++i)
        out[i] = bf16_to_float_host(
            reinterpret_cast<const uint16_t*>(raw)[i]);
      return;
    case GgmlType::Q8_0:
      for (size_t b = 0; b < numel / 32; ++b)
        dequant_q80_block(reinterpret_cast<const BlockQ80*>(raw + b * 34),
                          out.data() + b * 32);
      return;
    case GgmlType::Q4_K:
      for (size_t b = 0; b < numel / 256; ++b)
        dequant_q4k_block(reinterpret_cast<const BlockQ4K*>(raw + b * 144),
                          out.data() + b * 256);
      return;
    case GgmlType::Q6_K:
      for (size_t b = 0; b < numel / 256; ++b)
        dequant_q6k_block(reinterpret_cast<const BlockQ6K*>(raw + b * 210),
                          out.data() + b * 256);
      return;
    default: throw std::runtime_error("host_dequant: unsupported dtype");
  }
}

const char* kRtPath  = "/tmp/kokkosdnn_gguf_roundtrip.gguf";
const char* kBadPath = "/tmp/kokkosdnn_gguf_bad.gguf";

}  // namespace

// ---------------------------------------------------------------------------
// (a) Q4_K dequant vs hand-computed values.
// ---------------------------------------------------------------------------
TEST(GgufDequant, Q4KHandComputed) {
  BlockQ4K b{};
  b.d    = 0x3C00;  // dall = 1.0
  b.dmin = 0x4000;  // 2.0 -> subtracted term is -2.0 * m
  // scales: j<4 -> sc {1,2,3,4}, mn {0,1,0,2}; j>=4 -> sc {1,2,1,4},
  // mn {3,0,1,2}.
  const uint8_t s[12] = {1, 2, 3, 4, 0, 1, 0, 2, 0x31, 0x02, 0x11, 0x24};
  std::memcpy(b.scales, s, 12);
  for (int l = 0; l < 128; ++l) b.qs[l] = (uint8_t)l;

  // get_scale_min_k4 unit checks (all 8 groups).
  const uint8_t want_sc[8] = {1, 2, 3, 4, 1, 2, 1, 4};
  const uint8_t want_mn[8] = {0, 1, 0, 2, 3, 0, 1, 2};
  for (int j = 0; j < 8; ++j) {
    uint8_t sc = 0, mn = 0;
    get_scale_min_k4(j, b.scales, &sc, &mn);
    EXPECT_EQ(sc, want_sc[j]) << "j=" << j;
    EXPECT_EQ(mn, want_mn[j]) << "j=" << j;
  }

  float out[256];
  dequant_q4k_block(&b, out);

  // Hand-computed spots: val = dall*sc*nib + (-2.0)*mn.
  EXPECT_FLOAT_EQ(out[0], 0.0f);    // iter0 low, grp0: 1*1*0 - 2*0
  EXPECT_FLOAT_EQ(out[5], 5.0f);    // iter0 low, grp0: 1*1*5
  EXPECT_FLOAT_EQ(out[32], -6.0f);  // iter0 high, grp4: 1*1*0 - 2*3
  EXPECT_FLOAT_EQ(out[49], -5.0f);  // iter0 high l=17: 1*1*1 - 6
  EXPECT_FLOAT_EQ(out[67], 4.0f);   // iter1 low l=3: 2*3 - 2*1
  EXPECT_FLOAT_EQ(out[103], 4.0f);  // iter1 high l=7: 2*2 - 0
  EXPECT_FLOAT_EQ(out[159], 45.0f);  // iter2 low l=31: 3*15 - 0
  EXPECT_FLOAT_EQ(out[160], 2.0f);   // iter2 high l=0: 1*4 - 2*1
  EXPECT_FLOAT_EQ(out[202], 36.0f);  // iter3 low l=10: 4*10 - 2*2
  EXPECT_FLOAT_EQ(out[244], 24.0f);  // iter3 high l=20: 4*7 - 4

  // Full-block check with an independent per-lane re-derivation.
  for (int i = 0; i < 256; ++i) {
    const int iter    = i / 64;
    const int pos     = i % 64;
    const uint8_t q   = b.qs[iter * 32 + (pos % 32)];
    const uint8_t nib = (pos < 32) ? (uint8_t)(q & 0xF) : (uint8_t)(q >> 4);
    const int grp     = (pos < 32) ? iter : iter + 4;
    uint8_t sc, mn;
    if (grp < 4) {
      sc = s[grp] & 63;
      mn = s[grp + 4] & 63;
    } else {
      sc = (s[grp + 4] & 0xF) | ((s[grp - 4] >> 6) << 4);
      mn = (s[grp + 4] >> 4) | ((s[grp] >> 6) << 4);
    }
    const float want = 1.0f * (float)sc * (float)nib + (-2.0f) * (float)mn;
    EXPECT_FLOAT_EQ(out[i], want) << "i=" << i;
  }
}

// ---------------------------------------------------------------------------
// (b) Q8_0 / Q6_K / F16 spot checks.
// ---------------------------------------------------------------------------
TEST(GgufDequant, Q80Spot) {
  BlockQ80 b{};
  b.d = 0x3800;  // 0.5
  for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t)(i * 3 - 20);
  float out[32];
  dequant_q80_block(&b, out);
  for (int i = 0; i < 32; ++i)
    EXPECT_FLOAT_EQ(out[i], 0.5f * (float)(i * 3 - 20)) << "i=" << i;
}

TEST(GgufDequant, Q6KSpot) {
  BlockQ6K b{};
  b.d = 0x3C00;  // 1.0
  for (int i = 0; i < 16; ++i) b.scales[i] = 1;
  // Chunk 0, l=5: exercise all four qh bit-pairs of one byte.
  b.ql[5]     = 0xA3;
  b.ql[5 + 32] = 0x7C;
  b.qh[5]     = 0xE4;  // pairs (lo..hi): 0, 1, 2, 3
  // Chunk 1 (offset 128), l=9.
  b.ql[64 + 9]      = 0x51;
  b.ql[64 + 9 + 32] = 0x29;
  b.qh[32 + 9]      = 0x1B;  // pairs: 3, 2, 1, 0
  // l=20 uses scales[1] (=2) to cover the is = l/16 = 1 path.
  b.scales[1] = 2;

  float out[256];
  dequant_q6k_block(&b, out);
  // Chunk 0: q1 = (0x3|0)-32, q2 = (0xC|0x10)-32, q3 = (0xA|0x20)-32,
  // q4 = (0x7|0x30)-32.
  EXPECT_FLOAT_EQ(out[5], -29.0f);
  EXPECT_FLOAT_EQ(out[37], -4.0f);
  EXPECT_FLOAT_EQ(out[69], 10.0f);
  EXPECT_FLOAT_EQ(out[101], 23.0f);
  // Chunk 1: q1 = (0x1|0x30)-32, q2 = (0x9|0x20)-32, q3 = (0x5|0x10)-32,
  // q4 = (0x2|0x00)-32.
  EXPECT_FLOAT_EQ(out[128 + 9], 17.0f);
  EXPECT_FLOAT_EQ(out[128 + 41], 9.0f);
  EXPECT_FLOAT_EQ(out[128 + 73], -11.0f);
  EXPECT_FLOAT_EQ(out[128 + 105], -30.0f);
  // scales[1] = 2 path: q = (0|0) - 32, scale 2.
  EXPECT_FLOAT_EQ(out[20], -64.0f);
  // Every untouched lane has ql = qh = 0, scale 1 -> 1*1*(0-32).
  EXPECT_FLOAT_EQ(out[0], -32.0f);
  EXPECT_FLOAT_EQ(out[200], -32.0f);
  EXPECT_FLOAT_EQ(out[255], -32.0f);
}

TEST(GgufDequant, Float16) {
  EXPECT_FLOAT_EQ(fp16_to_float_host(0x3C00), 1.0f);
  EXPECT_FLOAT_EQ(fp16_to_float_host(0xBC00), -1.0f);
  EXPECT_FLOAT_EQ(fp16_to_float_host(0x3800), 0.5f);
  EXPECT_FLOAT_EQ(fp16_to_float_host(0x0000), 0.0f);
  EXPECT_FLOAT_EQ(fp16_to_float_host(0x7BFF), 65504.0f);
  EXPECT_FLOAT_EQ(fp16_to_float_host(0x0001), 5.960464477539063e-08f);
  EXPECT_FLOAT_EQ(bf16_to_float_host(0x3F80), 1.0f);
  EXPECT_FLOAT_EQ(bf16_to_float_host(0xBF80), -1.0f);
  EXPECT_FLOAT_EQ(bf16_to_float_host(0x4000), 2.0f);
  EXPECT_FLOAT_EQ(bf16_to_float_host(0x0000), 0.0f);
}

// ---------------------------------------------------------------------------
// (c) Synthetic GGUF roundtrip: write small file, read back, dequant.
// ---------------------------------------------------------------------------
TEST(GgufFile, Roundtrip) {
  std::remove(kRtPath);

  // F32 1-D tensor.
  std::vector<float> f32v = {0.25f, -1.5f, 3.0f, 0.0f,
                             7.25f, -0.5f, 2.0f, 100.0f};
  // F16 2-D tensor [rows=2, cols=4] (ggml dims {4, 2}).
  std::vector<float> f16v;
  for (int i = 0; i < 8; ++i) f16v.push_back(0.25f * (float)(i - 3));
  // Q8_0 [rows=2, cols=64].
  std::vector<float> q80v;
  for (int i = 0; i < 128; ++i)
    q80v.push_back(0.05f * (float)((i * 7) % 11) - 0.2f);
  // Q4_K [rows=1, cols=256] unit block: values are the nibbles.
  std::vector<float> q4kv(256);
  for (int i = 0; i < 256; ++i) q4kv[i] = (float)(i % 16);

  std::vector<SyntheticTensor> ts;
  SyntheticTensor t;
  t.name  = "t_f32";
  t.dtype = GgmlType::F32;
  t.dims  = {8};
  t.raw.resize(32);
  std::memcpy(t.raw.data(), f32v.data(), 32);
  ts.push_back(t);

  t.name  = "t_f16";
  t.dtype = GgmlType::F16;
  t.dims  = {4, 2};
  t.raw.resize(16);
  for (int i = 0; i < 8; ++i) {
    const uint16_t h = f32_to_f16_bits(f16v[i]);
    std::memcpy(t.raw.data() + 2 * i, &h, 2);
  }
  ts.push_back(t);

  t.name  = "t_q80";
  t.dtype = GgmlType::Q8_0;
  t.dims  = {64, 2};
  t.raw.resize(2 * 2 * 34);
  for (int r = 0; r < 2; ++r)
    for (int blk = 0; blk < 2; ++blk) {
      BlockQ80 b{};
      quant_q80_block(q80v.data() + r * 64 + blk * 32, &b);
      std::memcpy(t.raw.data() + (r * 2 + blk) * 34, &b, 34);
    }
  ts.push_back(t);

  t.name  = "t_q4k";
  t.dtype = GgmlType::Q4_K;
  t.dims  = {256, 1};
  {
    BlockQ4K b = make_unit_q4k();
    for (int iter = 0; iter < 4; ++iter)
      for (int l = 0; l < 32; ++l) {
        const uint8_t lo = (uint8_t)q4kv[iter * 64 + l];
        const uint8_t hi = (uint8_t)q4kv[iter * 64 + 32 + l];
        b.qs[iter * 32 + l] = (uint8_t)(lo | (hi << 4));
      }
    t.raw.resize(144);
    std::memcpy(t.raw.data(), &b, 144);
  }
  ts.push_back(t);

  write_synthetic_gguf(kRtPath, ts,
                       {{"general.architecture", "test-arch"},
                        {"general.name", "gguf-unit-test"}});

  GgufFile f(kRtPath);
  EXPECT_EQ(f.version(), 3u);
  EXPECT_EQ(f.tensors().size(), 4u);
  EXPECT_TRUE(f.contains("t_f32"));
  EXPECT_FALSE(f.contains("nope"));
  EXPECT_THROW(f.find("nope"), std::runtime_error);
  EXPECT_EQ(f.kv_string("general.architecture"), "test-arch");
  EXPECT_THROW(f.kv_string("missing"), std::runtime_error);

  const GgufTensorInfo& q4k = f.find("t_q4k");
  EXPECT_EQ(q4k.dtype, GgmlType::Q4_K);
  EXPECT_EQ(q4k.numel, 256u);
  EXPECT_EQ(q4k.nbytes, 144u);
  EXPECT_EQ(q4k.rows(), 1u);
  EXPECT_EQ(q4k.cols(), 256u);
  const GgufTensorInfo& q80 = f.find("t_q80");
  EXPECT_EQ(q80.rows(), 2u);
  EXPECT_EQ(q80.cols(), 64u);

  std::vector<float> out;
  dequant_tensor_to_float(f, f.find("t_f32"), out);
  ASSERT_EQ(out.size(), 8u);
  for (int i = 0; i < 8; ++i) EXPECT_FLOAT_EQ(out[i], f32v[i]) << "i=" << i;

  dequant_tensor_to_float(f, f.find("t_f16"), out);
  ASSERT_EQ(out.size(), 8u);
  for (int i = 0; i < 8; ++i) EXPECT_FLOAT_EQ(out[i], f16v[i]) << "i=" << i;

  dequant_tensor_to_float(f, f.find("t_q80"), out);
  ASSERT_EQ(out.size(), 128u);
  for (int r = 0; r < 2; ++r)
    for (int blk = 0; blk < 2; ++blk) {
      float amax = 0;
      for (int i = 0; i < 32; ++i)
        amax = std::fmax(amax, std::fabs(q80v[r * 64 + blk * 32 + i]));
      // Half-ulp rounding plus the fp16 rounding of the block scale itself
      // (|q| <= 127 amplifies it by up to ~amax/1024).
      const float tol = amax / 127.0f / 2.0f + amax / 1024.0f + 1e-6f;
      for (int i = 0; i < 32; ++i)
        EXPECT_NEAR(out[r * 64 + blk * 32 + i],
                    q80v[r * 64 + blk * 32 + i], tol)
            << "r=" << r << " i=" << i;
    }

  dequant_tensor_to_float(f, f.find("t_q4k"), out);
  ASSERT_EQ(out.size(), 256u);
  for (int i = 0; i < 256; ++i) EXPECT_FLOAT_EQ(out[i], q4kv[i]) << "i=" << i;

  // load_quant_weight path (used by the GEMV kernels below).
  {
    KokkosDNN::DNNHandle<Exec> h;
    QuantWeight<Exec> w = load_quant_weight<Exec>(f, q80);
    EXPECT_EQ(w.rows, 2u);
    EXPECT_EQ(w.cols, 64u);
    EXPECT_EQ(w.data.extent(0), 2u * 2u * 34u);
  }

  // Corrupt inputs are rejected, not misparsed.
  {
    std::FILE* g = std::fopen(kBadPath, "wb");
    ASSERT_NE(g, nullptr);
    std::fwrite("NOPE\1\0\0\0\0\0\0\0\0\0\0\0\0", 1, 20, g);
    std::fclose(g);
    EXPECT_THROW(
        {
          GgufFile bad(kBadPath);
          (void)bad;
        },
        std::runtime_error);
  }

  std::remove(kRtPath);
  std::remove(kBadPath);
}

// ---------------------------------------------------------------------------
// (d) quant_gemv / quant_gemm vs dense reference.
// ---------------------------------------------------------------------------
namespace {

struct CaseCfg {
  GgmlType dtype;
  size_t K;
  bool crafted_exact;  // weights exactly representable (tight tol vs orig)
};

// Build one row-major [N, K] reference matrix + packed raw bytes.
void build_case(GgmlType dtype, size_t N, size_t K, std::vector<float>& ref,
                std::vector<uint8_t>& raw) {
  ref.resize(N * K);
  if (dtype == GgmlType::Q4_K) {
    for (size_t i = 0; i < N; ++i)
      for (size_t k = 0; k < K; ++k)
        ref[i * K + k] = (float)((i * K + k) % 16);
    raw.resize(N * 144);
    for (size_t i = 0; i < N; ++i) {
      BlockQ4K b = make_unit_q4k();
      for (int iter = 0; iter < 4; ++iter)
        for (int l = 0; l < 32; ++l) {
          const uint8_t lo = (uint8_t)ref[i * K + iter * 64 + l];
          const uint8_t hi = (uint8_t)ref[i * K + iter * 64 + 32 + l];
          b.qs[iter * 32 + l] = (uint8_t)(lo | (hi << 4));
        }
      std::memcpy(raw.data() + i * 144, &b, 144);
    }
    return;
  }
  if (dtype == GgmlType::Q6_K) {
    for (size_t i = 0; i < N; ++i)
      for (size_t k = 0; k < K; ++k)
        ref[i * K + k] = -32.0f + (float)((i * K + k) % 16);
    raw.resize(N * 210);
    for (size_t i = 0; i < N; ++i) {
      BlockQ6K b{};
      b.d = 0x3C00;
      for (int s = 0; s < 16; ++s) b.scales[s] = 1;
      // qh = 0: decoded q = nibble - 32, so store v + 32 in the nibbles.
      for (int c = 0; c < 2; ++c)
        for (int l = 0; l < 32; ++l) {
          const uint8_t v0 =
              (uint8_t)(ref[i * K + c * 128 + l] + 32.0f);  // y[l]
          const uint8_t v1 =
              (uint8_t)(ref[i * K + c * 128 + 64 + l] + 32.0f);  // y[l+64]
          const uint8_t v2 =
              (uint8_t)(ref[i * K + c * 128 + 32 + l] + 32.0f);  // y[l+32]
          const uint8_t v3 =
              (uint8_t)(ref[i * K + c * 128 + 96 + l] + 32.0f);  // y[l+96]
          b.ql[c * 64 + l]      = (uint8_t)(v0 | (v1 << 4));
          b.ql[c * 64 + 32 + l] = (uint8_t)(v2 | (v3 << 4));
        }
      std::memcpy(raw.data() + i * 210, &b, 210);
    }
    return;
  }
  // Small exact-binary values for the F-variants and Q8_0.
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < K; ++k)
      ref[i * K + k] = 0.125f * (float)((i * K + k) % 5) - 0.25f;
  if (dtype == GgmlType::F32) {
    raw.resize(N * K * 4);
    std::memcpy(raw.data(), ref.data(), raw.size());
  } else if (dtype == GgmlType::F16) {
    raw.resize(N * K * 2);
    for (size_t i = 0; i < N * K; ++i) {
      const uint16_t h = f32_to_f16_bits(ref[i]);
      std::memcpy(raw.data() + 2 * i, &h, 2);
    }
  } else if (dtype == GgmlType::BF16) {
    raw.resize(N * K * 2);
    for (size_t i = 0; i < N * K; ++i) {
      const uint16_t h = f32_to_bf16_bits(ref[i]);
      std::memcpy(raw.data() + 2 * i, &h, 2);
    }
  } else if (dtype == GgmlType::Q8_0) {
    raw.resize(N * (K / 32) * 34);
    for (size_t i = 0; i < N; ++i)
      for (size_t blk = 0; blk < K / 32; ++blk) {
        BlockQ80 b{};
        quant_q80_block(ref.data() + i * K + blk * 32, &b);
        std::memcpy(raw.data() + (i * (K / 32) + blk) * 34, &b, 34);
      }
  } else {
    throw std::runtime_error("build_case: unsupported dtype");
  }
}

template <class Scalar>
void check_quant_gemv(GgmlType dtype, size_t K, float tol) {
  const size_t N = 3;
  std::vector<float> ref;
  std::vector<uint8_t> raw;
  build_case(dtype, N, K, ref, raw);
  std::vector<float> wdq;
  host_dequant(dtype, raw.data(), N * K, wdq);

  // Input vector: exact-binary values (bf16/f16-safe).
  const float step = (K == 64) ? 0.125f : 0.25f;
  std::vector<float> xf(K);
  for (size_t k = 0; k < K; ++k) xf[k] = step * (float)(((int)(k % 5)) - 2);

  Kokkos::View<Scalar*, Exec> x("x", K), y("y", N);
  auto hx = Kokkos::create_mirror_view(x);
  for (size_t k = 0; k < K; ++k) hx(k) = static_cast<Scalar>(xf[k]);
  Kokkos::deep_copy(x, hx);
  QuantWeight<Exec> w = make_qw<Exec>(dtype, N, K, raw);

  KokkosDNN::DNNHandle<Exec> h;
  quant_gemv(h, x, w, y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(got, y);

  for (size_t j = 0; j < N; ++j) {
    // Reference on the *dequantized* weights (double accumulation): checks
    // the kernel independently of quantization error.
    double acc = 0;
    for (size_t k = 0; k < K; ++k)
      acc += (double)wdq[j * K + k] * (double)(float)hx(k);
    EXPECT_NEAR((float)got(j), (float)acc, tol)
        << "kernel mismatch dtype=" << ggml_type_name(dtype) << " row=" << j;
    // Sanity vs the original floats: exact for F32/crafted, bounded for
    // the rest.
    double acc0 = 0;
    for (size_t k = 0; k < K; ++k)
      acc0 += (double)ref[j * K + k] * (double)xf[k];
    const float loose =
        (dtype == GgmlType::F32 || dtype == GgmlType::Q4_K ||
         dtype == GgmlType::Q6_K)
            ? tol
            : 5e-2f;
    EXPECT_NEAR((float)got(j), (float)acc0, loose)
        << "quant sanity dtype=" << ggml_type_name(dtype) << " row=" << j;
  }

  // ExecSpace (no-handle) overload runs too.
  quant_gemv(Exec{}, x, w, y);
  Kokkos::fence();
}

template <class Scalar>
void check_quant_gemm(GgmlType dtype, size_t K, float tol) {
  const size_t M = 3, N = 4;
  std::vector<float> ref;
  std::vector<uint8_t> raw;
  build_case(dtype, N, K, ref, raw);
  std::vector<float> wdq;
  host_dequant(dtype, raw.data(), N * K, wdq);

  const float step = (K == 64) ? 0.125f : 0.25f;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> X("X", M, K), Y("Y", M, N);
  auto hx = Kokkos::create_mirror_view(X);
  for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k)
      hx(i, k) = static_cast<Scalar>(step * (float)(((int)((i + k) % 5)) - 2));
  Kokkos::deep_copy(X, hx);
  QuantWeight<Exec> w = make_qw<Exec>(dtype, N, K, raw);

  KokkosDNN::DNNHandle<Exec> h;
  quant_gemm(h, X, w, Y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(Y);
  Kokkos::deep_copy(got, Y);

  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j) {
      double acc = 0;
      for (size_t k = 0; k < K; ++k)
        acc += (double)wdq[j * K + k] * (double)(float)hx(i, k);
      EXPECT_NEAR((float)got(i, j), (float)acc, tol)
          << "kernel mismatch dtype=" << ggml_type_name(dtype) << " (" << i
          << "," << j << ")";
    }
  quant_gemm(Exec{}, X, w, Y);
  Kokkos::fence();
}

}  // namespace

TEST(GgufQuantGemm, GemvFloat) {
  check_quant_gemv<float>(GgmlType::F32, 64, 1e-4f);
  check_quant_gemv<float>(GgmlType::F16, 64, 1e-4f);
  check_quant_gemv<float>(GgmlType::BF16, 64, 1e-4f);
  check_quant_gemv<float>(GgmlType::Q8_0, 64, 1e-4f);
  check_quant_gemv<float>(GgmlType::Q4_K, 256, 1e-4f);
  check_quant_gemv<float>(GgmlType::Q6_K, 256, 1e-4f);
}

TEST(GgufQuantGemm, GemvBHalf) {
  using bhalf_t = Kokkos::Experimental::bhalf_t;
  check_quant_gemv<bhalf_t>(GgmlType::F32, 64, 5e-2f);
  check_quant_gemv<bhalf_t>(GgmlType::F16, 64, 5e-2f);
  check_quant_gemv<bhalf_t>(GgmlType::BF16, 64, 5e-2f);
  check_quant_gemv<bhalf_t>(GgmlType::Q8_0, 64, 5e-2f);
}

TEST(GgufQuantGemm, GemmFloat) {
  check_quant_gemm<float>(GgmlType::F32, 64, 1e-4f);
  check_quant_gemm<float>(GgmlType::F16, 64, 1e-4f);
  check_quant_gemm<float>(GgmlType::BF16, 64, 1e-4f);
  check_quant_gemm<float>(GgmlType::Q8_0, 64, 1e-4f);
  check_quant_gemm<float>(GgmlType::Q4_K, 256, 1e-4f);
  check_quant_gemm<float>(GgmlType::Q6_K, 256, 1e-4f);
}

TEST(GgufQuant, Errors) {
  KokkosDNN::DNNHandle<Exec> h;
  // Unsupported dtype is rejected before any kernel launch.
  {
    std::vector<uint8_t> bytes(18, 0);
    auto w = make_qw<Exec>(GgmlType::Q4_0, 1, 32, bytes);
    Kokkos::View<float*, Exec> x("x", 32), y("y", 1);
    EXPECT_THROW(quant_gemv(h, x, w, y), std::invalid_argument);
  }
  // Dimension mismatches are rejected.
  {
    std::vector<uint8_t> bytes(2 * 34, 0);
    auto w = make_qw<Exec>(GgmlType::Q8_0, 1, 64, bytes);
    Kokkos::View<float*, Exec> x("x", 32), y("y", 1);
    EXPECT_THROW(quant_gemv(h, x, w, y), std::invalid_argument);
  }
  // cols must be a whole number of blocks.
  {
    QuantWeight<Exec> w;
    w.dtype = GgmlType::Q8_0;
    w.rows  = 1;
    w.cols  = 17;
    w.data  = Kokkos::View<uint8_t*, Exec>("w", 34);
    Kokkos::View<float*, Exec> x("x", 17), y("y", 1);
    EXPECT_THROW(quant_gemv(h, x, w, y), std::invalid_argument);
  }
  // load_quant_weight rejects dtypes outside the compute set.
  {
    std::remove(kRtPath);
    SyntheticTensor t;
    t.name  = "w_q81";
    t.dtype = GgmlType::Q8_1;
    t.dims  = {32};
    t.raw.assign(36, 0);
    write_synthetic_gguf(kRtPath, {t});
    GgufFile f(kRtPath);
    EXPECT_THROW(load_quant_weight<Exec>(f, f.find("w_q81")),
                 std::runtime_error);
    std::remove(kRtPath);
  }
}
