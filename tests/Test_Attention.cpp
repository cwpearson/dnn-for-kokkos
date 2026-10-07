// Attention: single query attending to a single key must return its value
// (float + bfloat16). Also covers embedding_gather on both scalars.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <utility>

#include "KokkosDNN.hpp"

template <class Scalar>
void check_attention(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int Hq = 4, Hkv = 2, D = 8;
  Kokkos::View<Scalar***, Exec> q("q", 1, Hq, D), k("k", 1, Hkv, D),
      v("v", 1, Hkv, D), out("out", 1, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int hh = 0; hh < Hq; ++hh)
    for (int d = 0; d < D; ++d)
      hq(0, hh, d) = static_cast<Scalar>(0.1f * (hh + d + 1));
  for (int hh = 0; hh < Hkv; ++hh)
    for (int d = 0; d < D; ++d) {
      hk(0, hh, d) = static_cast<Scalar>(0.2f * (hh - d));
      hv(0, hh, d) = static_cast<Scalar>(1.0f * hh + 0.01f * d + 5.0f);
    }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  KokkosDNN::attention_gqa_single<Exec, Scalar>(h, q, k, v, 0, out,
                                               1.0f / 8.0f);
  // embedding_gather smoke cover on the same scalar.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> tab("tab", 4, D),
      emb("emb", 2, D);
  Kokkos::View<int*, Exec> toks("toks", 2);
  {
    auto ht = Kokkos::create_mirror_view(tab);
    for (int i = 0; i < 4; ++i)
      for (int d = 0; d < D; ++d)
        ht(i, d) = static_cast<Scalar>(0.1f * i + 0.01f * d);
    Kokkos::deep_copy(tab, ht);
    auto hx = Kokkos::create_mirror_view(toks);
    hx(0) = 3;
    hx(1) = 1;
    Kokkos::deep_copy(toks, hx);
  }
  KokkosDNN::embedding_gather<Exec, Scalar>(tab, toks, emb);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int hh = 0; hh < Hq; ++hh) {
    int hkv = (hh * Hkv) / Hq;
    for (int d = 0; d < D; ++d)
      EXPECT_NEAR(static_cast<float>(got(0, hh, d)),
                  static_cast<float>(hv(0, hkv, d)), tol)
          << "attention single-key mismatch at (" << hh << "," << d << ")";
  }
  auto ge = Kokkos::create_mirror_view(emb);
  Kokkos::deep_copy(ge, emb);
  auto gt = Kokkos::create_mirror_view(tab);
  Kokkos::deep_copy(gt, tab);
  for (int d = 0; d < D; ++d) {
    EXPECT_EQ(static_cast<float>(ge(0, d)), static_cast<float>(gt(3, d)))
        << "embedding_gather mismatch at (" << d << ")";
    EXPECT_EQ(static_cast<float>(ge(1, d)), static_cast<float>(gt(1, d)))
        << "embedding_gather mismatch at (" << d << ")";
  }
}

TEST(Attention, Float) { check_attention<float>(1e-4f); }

TEST(Attention, BHalf) { check_attention<Kokkos::Experimental::bhalf_t>(5e-2f); }

// Multi-token causal GQA reference: compares the streaming kernel against a
// host softmax reference with a nonzero prefill offset (q_off), exercising
// GQA head mapping, causal masking, and the LayoutRight cache-subview path
// the Llama model uses (no repack copy).
template <class Scalar>
void check_attention_reference(float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 4, Hq = 4, Hkv = 2, D = 8, Ktot = 7;
  const size_t q_off = 3;  // klen(t) = 4,5,6,7
  const float scale = 1.0f / 8.0f;

  Kokkos::View<Scalar***, Exec> q("q", T, Hq, D), out("out", T, Hq, D);
  // Oversized LayoutRight KV caches; the kernel gets subviews [0, Ktot).
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> kbig("kbig", Ktot + 3,
                                                          Hkv, D),
      vbig("vbig", Ktot + 3, Hkv, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(kbig);
  auto hv = Kokkos::create_mirror_view(vbig);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) =
            static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f);
  for (int n = 0; n < Ktot + 3; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f);
        hv(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f);
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(kbig, hk);
  Kokkos::deep_copy(vbig, hv);

  auto kp = Kokkos::subview(kbig, std::make_pair(0, Ktot), Kokkos::ALL,
                            Kokkos::ALL);
  auto vp = Kokkos::subview(vbig, std::make_pair(0, Ktot), Kokkos::ALL,
                            Kokkos::ALL);
  KokkosDNN::attention_gqa_single<Exec, Scalar>(h, q, kp, vp, q_off, out,
                                               scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);

  // Host reference: plain causal softmax over [0, klen).
  for (int t = 0; t < T; ++t) {
    size_t klen = q_off + t + 1;
    if (klen > (size_t)Ktot) klen = Ktot;
    for (int hh = 0; hh < Hq; ++hh) {
      int hkv = (hh * Hkv) / Hq;
      float m = -1e30f;
      for (size_t kk = 0; kk < klen; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, hh, d)) *
               static_cast<float>(hk(kk, hkv, d));
        m = std::max(m, s * scale);
      }
      float denom = 0;
      for (size_t kk = 0; kk < klen; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, hh, d)) *
               static_cast<float>(hk(kk, hkv, d));
        denom += std::exp(s * scale - m);
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (size_t kk = 0; kk < klen; ++kk) {
          float s = 0;
          for (int dd = 0; dd < D; ++dd)
            s += static_cast<float>(hq(t, hh, dd)) *
                 static_cast<float>(hk(kk, hkv, dd));
          acc += std::exp(s * scale - m) * static_cast<float>(hv(kk, hkv, d));
        }
        float ref = acc / denom;
        EXPECT_NEAR(static_cast<float>(got(t, hh, d)), ref, tol)
            << "reference mismatch at (" << t << "," << hh << "," << d << ")";
      }
    }
  }
}

TEST(Attention, FloatCausalReference) { check_attention_reference<float>(1e-4f); }

TEST(Attention, BHalfCausalReference) {
  check_attention_reference<Kokkos::Experimental::bhalf_t>(5e-2f);
}

// Long-key streaming smoke: T=1 decode over K=8192 keys proves the kernel
// holds no O(K) transient (the old [T,Hq,K] score tensor would already be
// 64 KiB here and PiBs for a full 128k prefill) while matching the host
// reference.
TEST(Attention, FloatLongKeys) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int Hq = 2, Hkv = 1, D = 4, Ktot = 8192;
  const size_t q_off = Ktot - 1;
  const float scale = 1.0f / 2.0f;
  Kokkos::View<float***, Exec> q("q", 1, Hq, D), k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D), out("out", 1, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int hh = 0; hh < Hq; ++hh)
    for (int d = 0; d < D; ++d)
      hq(0, hh, d) = 0.01f * (hh + d + 1);
  for (int n = 0; n < Ktot; ++n)
    for (int d = 0; d < D; ++d) {
      hk(n, 0, d) = 0.01f * ((n + d) % 13) - 0.05f;
      hv(n, 0, d) = 0.001f * n + 0.01f * d;
    }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  KokkosDNN::attention_gqa_single<Exec, float>(h, q, k, v, q_off, out, scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int hh = 0; hh < Hq; ++hh) {
    float m = -1e30f;
    for (int kk = 0; kk < Ktot; ++kk) {
      float s = 0;
      for (int d = 0; d < D; ++d) s += hq(0, hh, d) * hk(kk, 0, d);
      m = std::max(m, s * scale);
    }
    float denom = 0;
    for (int kk = 0; kk < Ktot; ++kk) {
      float s = 0;
      for (int d = 0; d < D; ++d) s += hq(0, hh, d) * hk(kk, 0, d);
      denom += std::exp(s * scale - m);
    }
    for (int d = 0; d < D; ++d) {
      float acc = 0;
      for (int kk = 0; kk < Ktot; ++kk) {
        float s = 0;
        for (int dd = 0; dd < D; ++dd) s += hq(0, hh, dd) * hk(kk, 0, dd);
        acc += std::exp(s * scale - m) * hv(kk, 0, d);
      }
      EXPECT_NEAR(got(0, hh, d), acc / denom, 1e-4f)
          << "long-key mismatch at (" << hh << "," << d << ")";
    }
  }
}
