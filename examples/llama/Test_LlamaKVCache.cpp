// Paged + int8-quantized KV cache: parity, accuracy, config, and growth.
// Always-on, tiny config, no weight downloads.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_LlamaConfig.hpp"
#include "KokkosDNN_LlamaModel.hpp"
#include "KokkosDNN_LlamaWeights.hpp"

namespace {

using KokkosDNN::Llama::Config;

template <class Scalar>
Kokkos::View<int*, Kokkos::DefaultExecutionSpace> to_device(
    const std::vector<int>& ids) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<int*, Exec> v("toks", ids.size());
  auto h = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
  Kokkos::deep_copy(v, h);
  return v;
}

template <class Scalar>
std::vector<float> forward_logits(
    KokkosDNN::Llama::Model<Kokkos::DefaultExecutionSpace, Scalar>& m,
    const std::vector<int>& ids) {
  auto lg = m.forward(to_device<Scalar>(ids));
  auto h = Kokkos::create_mirror_view(lg);
  Kokkos::deep_copy(h, lg);
  std::vector<float> out(h.extent(0));
  for (size_t i = 0; i < h.extent(0); ++i)
    out[i] = static_cast<float>(h(i));
  return out;
}

template <class Scalar>
void expect_near(const std::vector<float>& a, const std::vector<float>& b,
                 float tol) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_NEAR(a[i], b[i], tol) << "logit mismatch at " << i;
}

template <class Scalar>
void expect_exact(const std::vector<float>& a, const std::vector<float>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_FLOAT_EQ(a[i], b[i]) << "logit mismatch at " << i;
}

}  // namespace

// Paged dense KV is bit-exact vs dense: same values, same accumulation
// order, only the addressing differs (page/slot via block table).
template <class Scalar>
void check_paged_dense_exact() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  std::vector<int> prompt = {1, 2, 3, 4, 5, 6, 7};
  for (int page : {1, 3, 4, 16}) {
    for (int chunk : {0, 2}) {
      auto base = Config::tiny_test();
      Weights<Exec, Scalar> w;
      random_init(base, w, 1234);
      Model<Exec, Scalar> dense(base, w);
      dense.cfg.prefill_chunk_size = chunk;
      auto cfg2 = base;
      cfg2.kv_page_size = page;
      Model<Exec, Scalar> paged(cfg2, w);
      paged.cfg.prefill_chunk_size = chunk;
      expect_exact<Scalar>(forward_logits(paged, prompt),
                           forward_logits(dense, prompt));
      // End-to-end generate agrees exactly too.
      Model<Exec, Scalar> d2(base, w), p2(cfg2, w);
      EXPECT_EQ(p2.generate(prompt, 4), d2.generate(prompt, 4));
    }
  }
}

TEST(LlamaPagedKV, DenseExactFloat) { check_paged_dense_exact<float>(); }

TEST(LlamaPagedKV, DenseExactBHalf) {
  check_paged_dense_exact<Kokkos::Experimental::bhalf_t>();
}

// Dense int8 vs dense FP: bounded drift, valid + deterministic decode.
template <class Scalar>
void check_dense_quant_close(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto base = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(base, w, 1234);
  std::vector<int> prompt = {1, 2, 3, 4, 5, 6};
  Model<Exec, Scalar> dense(base, w);
  auto cfgq = base;
  cfgq.kv_int8 = true;
  Model<Exec, Scalar> quant(cfgq, w);
  expect_near<Scalar>(forward_logits(quant, prompt),
                      forward_logits(dense, prompt), tol);
  // Quantized decode: valid ids, full length, deterministic.
  Model<Exec, Scalar> q1(cfgq, w), q2(cfgq, w);
  auto o1 = q1.generate(prompt, 4);
  auto o2 = q2.generate(prompt, 4);
  EXPECT_EQ(o1, o2);
  ASSERT_EQ(o1.size(), prompt.size() + 4);
  for (auto id : o1) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, base.vocab_size);
  }
  // Full-window bytes shrink (int8 values + scales < FP values).
  EXPECT_LT(cfgq.kv_cache_bytes(sizeof(Scalar)),
            base.kv_cache_bytes(sizeof(Scalar)));
}

TEST(LlamaQuantKV, DenseCloseFloat) { check_dense_quant_close<float>(0.5f); }

TEST(LlamaQuantKV, DenseCloseBHalf) {
  check_dense_quant_close<Kokkos::Experimental::bhalf_t>(1.0f);
}

// Paged int8 matches dense int8 exactly; both drift slightly vs dense FP.
template <class Scalar>
void check_paged_quant() {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto base = Config::tiny_test();
  Weights<Exec, Scalar> w;
  random_init(base, w, 1234);
  std::vector<int> prompt = {1, 2, 3, 4, 5, 6};
  auto cfgq = base;
  cfgq.kv_int8 = true;
  auto cfgpq = base;
  cfgpq.kv_int8 = true;
  cfgpq.kv_page_size = 4;
  Model<Exec, Scalar> dq(cfgq, w), pq(cfgpq, w), dense(base, w);
  auto lq = forward_logits(dq, prompt);
  auto lpq = forward_logits(pq, prompt);
  expect_exact<Scalar>(lpq, lq);  // same quant math, different addressing
  // Paged-quant decode is deterministic with valid ids.
  Model<Exec, Scalar> p1(cfgpq, w), p2(cfgpq, w);
  EXPECT_EQ(p1.generate(prompt, 4), p2.generate(prompt, 4));
  // And stays near FP (looser than dense-quant-vs-FP: same bound is fine).
  expect_near<Scalar>(lpq, forward_logits(dense, prompt),
                      std::is_same_v<Scalar, float> ? 0.5f : 1.0f);
}

TEST(LlamaPagedQuantKV, MatchesDenseQuantFloat) {
  check_paged_quant<float>();
}

TEST(LlamaPagedQuantKV, MatchesDenseQuantBHalf) {
  check_paged_quant<Kokkos::Experimental::bhalf_t>();
}

// Paged pools grow on demand: short prompts rent touched pages only, reset
// keeps the allocation, and growth preserves exactness.
TEST(LlamaPagedKV, OnDemandGrowth) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  cfg.max_seq_len = 64;
  cfg.kv_page_size = 16;
  Weights<Exec, float> w;
  random_init(cfg, w, 1234);
  Model<Exec, float> m(cfg, w);
  EXPECT_EQ(m.kv_allocated_bytes(), 0u);  // nothing touched yet
  (void)forward_logits(m, {1, 2, 3, 4});
  const size_t one_page =
      (size_t)cfg.n_layers * 16 * (size_t)cfg.n_kv_heads *
      (size_t)cfg.head_dim * sizeof(float) * 2;
  EXPECT_EQ(m.kv_allocated_bytes(), one_page);
  EXPECT_LT(m.kv_allocated_bytes(), m.kv_cache_bytes());
  m.reset();
  EXPECT_EQ(m.kv_allocated_bytes(), one_page);  // reset keeps pages (O(1))
  // Deterministic across the reset boundary.
  Model<Exec, float> fresh(cfg, w);
  expect_exact<float>(forward_logits(m, {1, 2, 3, 4}),
                      forward_logits(fresh, {1, 2, 3, 4}));
  // Crossing a page boundary grows and stays exact vs dense.
  std::vector<int> long_prompt(20);
  for (size_t i = 0; i < long_prompt.size(); ++i)
    long_prompt[i] = (int)(i + 1);
  m.reset();
  auto got = forward_logits(m, long_prompt);
  EXPECT_EQ(m.kv_allocated_bytes(), 2 * one_page);  // 20 tokens, page 16
  auto dense_cfg = Config::tiny_test();
  dense_cfg.max_seq_len = 64;
  Weights<Exec, float> w2;
  random_init(dense_cfg, w2, 1234);
  // Same weights object reused: dense model over the same weights.
  Model<Exec, float> d(dense_cfg, w);
  expect_exact<float>(got, forward_logits(d, long_prompt));
}

// Quantized paged pools also grow on demand and report shrunken bytes.
TEST(LlamaPagedQuantKV, GrowthReportsShrunkenBytes) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using namespace KokkosDNN::Llama;
  auto cfg = Config::tiny_test();
  cfg.max_seq_len = 64;
  cfg.kv_page_size = 8;
  cfg.kv_int8 = true;
  Weights<Exec, float> w;
  random_init(cfg, w, 7);
  Model<Exec, float> m(cfg, w);
  EXPECT_EQ(m.kv_allocated_bytes(), 0u);
  (void)forward_logits(m, {1, 2, 3});
  const size_t toks = 8;  // one page resident
  const size_t expect =
      (size_t)cfg.n_layers * toks * (size_t)cfg.n_kv_heads *
      ((size_t)cfg.head_dim * 1 + sizeof(float)) * 2;
  EXPECT_EQ(m.kv_allocated_bytes(), expect);
  auto dense = Config::tiny_test();
  dense.max_seq_len = 64;
  EXPECT_LT(cfg.kv_cache_bytes(sizeof(float)),
            dense.kv_cache_bytes(sizeof(float)));
}

// Config surface: JSON keys, aliases, and validation.
TEST(LlamaKVConfig, JsonAndValidate) {
  using KokkosDNN::Llama::Config;
  auto base = Config::tiny_test();
  // Defaults: dense FP.
  EXPECT_EQ(base.kv_page_size, 0);
  EXPECT_FALSE(base.kv_int8);
  EXPECT_FALSE(base.use_paged_kv());
  // Direct keys.
  auto p = Config::from_json_string(R"({"kv_page_size": 16})", base);
  EXPECT_EQ(p.kv_page_size, 16);
  EXPECT_TRUE(p.use_paged_kv());
  EXPECT_EQ(p.kv_num_pages(), (size_t)(p.max_seq_len + 15) / 16);
  auto q = Config::from_json_string(R"({"kv_int8": true})", base);
  EXPECT_TRUE(q.kv_int8);
  // Aliases: kv_quantized bool, kv_quant string.
  auto qa = Config::from_json_string(R"({"kv_quantized": true})", base);
  EXPECT_TRUE(qa.kv_int8);
  auto qs =
      Config::from_json_string(R"({"kv_quant": "int8"})", base);
  EXPECT_TRUE(qs.kv_int8);
  auto qn =
      Config::from_json_string(R"({"kv_quant": "none"})", base);
  EXPECT_FALSE(qn.kv_int8);
  EXPECT_THROW(Config::from_json_string(R"({"kv_quant": "fp8"})", base),
               std::runtime_error);
  // Validation: negative / oversized pages rejected.
  EXPECT_THROW(Config::from_json_string(R"({"kv_page_size": -1})", base),
               std::exception);
  auto bad = base;
  bad.kv_page_size = base.max_seq_len + 1;
  EXPECT_THROW(bad.validate(), std::exception);
  bad.kv_page_size = 4096;  // > 1024 guard
  EXPECT_THROW(bad.validate(), std::exception);
  auto ok = base;
  ok.kv_page_size = 8;
  ok.kv_int8 = true;
  EXPECT_NO_THROW(ok.validate());
  // Byte accounting: paged pads up to full pages; int8 shrinks.
  auto pg = base;
  pg.max_seq_len = 32;
  pg.kv_page_size = 16;  // exact division: no padding
  EXPECT_EQ(pg.kv_cache_bytes(2), base.kv_cache_bytes(2));
  pg.kv_page_size = 24;  // pads 32 -> 48
  EXPECT_GT(pg.kv_cache_bytes(2), base.kv_cache_bytes(2));
}

// Standalone paged-dense attention matches the dense kernel across page
// sizes (including pages that never divide the key length).
TEST(PagedAttention, MatchesDense) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 3, Hq = 4, Hkv = 2, D = 8, Ktot = 7;
  const size_t q_off = 4;  // klen(t) = 5,6,7
  const float scale = 1.0f / 8.0f;
  Kokkos::View<float***, Exec> q("q", T, Hq, D), out_d("od", T, Hq, D),
      out_p("op", T, Hq, D);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) = 0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f;
  for (int n = 0; n < Ktot; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) = 0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f;
        hv(n, hh, d) = 0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f;
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  KokkosDNN::attention_gqa_single<Exec, float>(h, q, k, v, q_off, out_d,
                                               scale);
  for (int page : {1, 3, 4, 16}) {
    const size_t npages = ((size_t)Ktot + page - 1) / page;
    Kokkos::View<float****, Kokkos::LayoutRight, Exec> kp("kp", npages, page,
                                                          Hkv, D),
        vp("vp", npages, page, Hkv, D);
    Kokkos::View<int*, Exec> tab("tab", npages);
    auto hkp = Kokkos::create_mirror_view(kp);
    auto hvp = Kokkos::create_mirror_view(vp);
    auto htab = Kokkos::create_mirror_view(tab);
    for (size_t pg = 0; pg < npages; ++pg) htab(pg) = (int)pg;
    for (int n = 0; n < Ktot; ++n)
      for (int hh = 0; hh < Hkv; ++hh)
        for (int d = 0; d < D; ++d) {
          hkp(n / page, n % page, hh, d) = hk(n, hh, d);
          hvp(n / page, n % page, hh, d) = hv(n, hh, d);
        }
    Kokkos::deep_copy(kp, hkp);
    Kokkos::deep_copy(vp, hvp);
    Kokkos::deep_copy(tab, htab);
    KokkosDNN::attention_gqa_paged<Exec, float>(h, q, kp, vp, tab, page,
                                                q_off, out_p, scale);
    Kokkos::fence();
    auto gd = Kokkos::create_mirror_view(out_d);
    auto gp = Kokkos::create_mirror_view(out_p);
    Kokkos::deep_copy(gd, out_d);
    Kokkos::deep_copy(gp, out_p);
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < Hq; ++hh)
        for (int d = 0; d < D; ++d)
          EXPECT_FLOAT_EQ(gp(t, hh, d), gd(t, hh, d))
              << "paged/dense mismatch page=" << page << " at (" << t << ","
              << hh << "," << d << ")";
  }
}

// Standalone quantized attention tracks dense within a tight bound.
TEST(QuantAttention, CloseToDense) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 3, Hq = 4, Hkv = 2, D = 8, Ktot = 7;
  const size_t q_off = 4;
  const float scale = 1.0f / 8.0f;
  Kokkos::View<float***, Exec> q("q", T, Hq, D), out_d("od", T, Hq, D),
      out_q("oq", T, Hq, D);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D);
  Kokkos::View<int8_t***, Kokkos::LayoutRight, Exec> kq("kq", Ktot, Hkv, D),
      vq("vq", Ktot, Hkv, D);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> ks("ks", Ktot, Hkv),
      vs("vs", Ktot, Hkv);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) = 0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f;
  for (int n = 0; n < Ktot; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) = 0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f;
        hv(n, hh, d) = 0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f;
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  // Reference dense.
  KokkosDNN::attention_gqa_single<Exec, float>(h, q, k, v, q_off, out_d,
                                               scale);
  // Quantize on host with the same symmetric scheme, then run the kernel.
  auto hkq = Kokkos::create_mirror_view(kq);
  auto hvq = Kokkos::create_mirror_view(vq);
  auto hks = Kokkos::create_mirror_view(ks);
  auto hvs = Kokkos::create_mirror_view(vs);
  for (int n = 0; n < Ktot; ++n)
    for (int hh = 0; hh < Hkv; ++hh) {
      float kmax = 0, vmax = 0;
      for (int d = 0; d < D; ++d) {
        kmax = std::max(kmax, std::fabs(hk(n, hh, d)));
        vmax = std::max(vmax, std::fabs(hv(n, hh, d)));
      }
      float ksc = kmax > 0 ? kmax / 127.0f : 1.0f;
      float vsc = vmax > 0 ? vmax / 127.0f : 1.0f;
      hks(n, hh) = ksc;
      hvs(n, hh) = vsc;
      for (int d = 0; d < D; ++d) {
        int kqi = (int)std::round(hk(n, hh, d) / ksc);
        int vqi = (int)std::round(hv(n, hh, d) / vsc);
        kqi = std::max(-127, std::min(127, kqi));
        vqi = std::max(-127, std::min(127, vqi));
        hkq(n, hh, d) = (int8_t)kqi;
        hvq(n, hh, d) = (int8_t)vqi;
      }
    }
  Kokkos::deep_copy(kq, hkq);
  Kokkos::deep_copy(vq, hvq);
  Kokkos::deep_copy(ks, hks);
  Kokkos::deep_copy(vs, hvs);
  KokkosDNN::attention_gqa_quantized<Exec, float>(h, q, kq, ks, vq, vs, q_off,
                                                  out_q, scale);
  Kokkos::fence();
  auto gd = Kokkos::create_mirror_view(out_d);
  auto gq = Kokkos::create_mirror_view(out_q);
  Kokkos::deep_copy(gd, out_d);
  Kokkos::deep_copy(gq, out_q);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(gq(t, hh, d), gd(t, hh, d), 2e-2f)
            << "quant/dense mismatch at (" << t << "," << hh << "," << d
            << ")";
}
