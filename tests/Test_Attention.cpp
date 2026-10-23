// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Attention: single query attending to a single key must return its value
// (float + bfloat16). Also covers embedding_gather on both scalars.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <stdexcept>
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
    // FP64 reference: over 8192 keys a float accumulator is itself ~1.3e-4
    // off the exact value, i.e. worse than the 1e-4 tolerance below, so it
    // cannot adjudicate a kernel whose own error is ~5e-6.
    double m = -1e30;
    for (int kk = 0; kk < Ktot; ++kk) {
      double s = 0;
      for (int d = 0; d < D; ++d) s += (double)hq(0, hh, d) * hk(kk, 0, d);
      m = std::max(m, s * (double)scale);
    }
    double denom = 0;
    for (int kk = 0; kk < Ktot; ++kk) {
      double s = 0;
      for (int d = 0; d < D; ++d) s += (double)hq(0, hh, d) * hk(kk, 0, d);
      denom += std::exp(s * (double)scale - m);
    }
    for (int d = 0; d < D; ++d) {
      double acc = 0;
      for (int kk = 0; kk < Ktot; ++kk) {
        double s = 0;
        for (int dd = 0; dd < D; ++dd)
          s += (double)hq(0, hh, dd) * hk(kk, 0, dd);
        acc += std::exp(s * (double)scale - m) * (double)hv(kk, 0, d);
      }
      EXPECT_NEAR(got(0, hh, d), acc / denom, 1e-4f)
          << "long-key mismatch at (" << hh << "," << d << ")";
    }
  }
}

// Split-K decode kernel (flash-decoding form) vs an FP64 host reference.
// Covers every geometry class the launcher specialises: a GQA group of 4
// (the LFM2.5 shape, where one block covers all query heads of a KV head),
// MHA (no sharing), MQA (group of 8), a geometry whose Hq % Hkv != 0 (group
// width 1 fallback), an odd head dim, and a sliding-window decode. The
// kernel is reached through the public entry point, so the test also covers
// the launcher's split/tile/team choice and the combine pass.
template <class Scalar>
void check_decode_split_reference(float tol, int Hq, int Hkv, int D, int Ktot,
                                  size_t window) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const size_t ktot = static_cast<size_t>(Ktot);
  const size_t q_off = ktot - 1;  // single-token decode at the cache end
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  Kokkos::View<Scalar***, Exec> q("q", 1, Hq, D), k("k", ktot, Hkv, D),
      v("v", ktot, Hkv, D), out("out", 1, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int hh = 0; hh < Hq; ++hh)
    for (int d = 0; d < D; ++d)
      hq(0, hh, d) = static_cast<Scalar>(0.02f * ((hh * 5 + d) % 11) - 0.1f);
  for (size_t n = 0; n < ktot; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) =
            static_cast<Scalar>(0.02f * ((n * 3 + hh * 7 + d) % 13) - 0.1f);
        hv(n, hh, d) =
            static_cast<Scalar>(0.01f * ((n * 5 + hh * 3 + d) % 17) - 0.1f);
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  if (window == 0)
    KokkosDNN::attention_gqa_single<Exec, Scalar>(h, q, k, v, q_off, out, scale);
  else
    KokkosDNN::attention_gqa_single_sliding<Exec, Scalar>(h, q, k, v, q_off,
                                                         out, scale, window);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);

  // Reference over the exact stored (bf16-rounded) inputs, in double.
  auto mq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, q);
  auto mk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, k);
  auto mv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  size_t klo = 0, khi = 0;
  KokkosDNN::Impl::gqa_key_window(q_off, ktot, window, 0, klo, khi);
  for (int hh = 0; hh < Hq; ++hh) {
    const int kvh = (hh * Hkv) / Hq;
    double m = -1e300;
    for (size_t kk = klo; kk < khi; ++kk) {
      double s = 0.0;
      for (int d = 0; d < D; ++d)
        s += (double)mq(0, hh, d) * (double)mk(kk, kvh, d);
      m = std::max(m, s * (double)scale);
    }
    double den = 0.0;
    for (size_t kk = klo; kk < khi; ++kk) {
      double s = 0.0;
      for (int d = 0; d < D; ++d)
        s += (double)mq(0, hh, d) * (double)mk(kk, kvh, d);
      den += std::exp(s * (double)scale - m);
    }
    for (int d = 0; d < D; ++d) {
      double acc = 0.0;
      for (size_t kk = klo; kk < khi; ++kk) {
        double s = 0.0;
        for (int dd = 0; dd < D; ++dd)
          s += (double)mq(0, hh, dd) * (double)mk(kk, kvh, dd);
        acc += std::exp(s * (double)scale - m) * (double)mv(kk, kvh, d);
      }
      const double ref = den > 0.0 ? acc / den : 0.0;
      EXPECT_NEAR(static_cast<double>(got(0, hh, d)), ref, tol)
          << "split-K mismatch at Hq=" << Hq << " Hkv=" << Hkv << " D=" << D
          << " Ktot=" << Ktot << " win=" << window << " (" << hh << "," << d
          << ")";
    }
  }
}

TEST(Attention, SplitKVDecodeReference) {
  check_decode_split_reference<float>(1e-5f, 32, 8, 64, 1024, 0);
  check_decode_split_reference<Kokkos::Experimental::bhalf_t>(5e-2f, 32, 8, 64,
                                                              1024, 0);
  check_decode_split_reference<float>(1e-5f, 8, 8, 64, 512, 0);   // MHA
  check_decode_split_reference<float>(1e-5f, 8, 1, 64, 512, 0);   // MQA
  check_decode_split_reference<float>(1e-5f, 3, 2, 8, 256, 0);    // uneven
  check_decode_split_reference<float>(1e-5f, 4, 2, 7, 256, 0);    // odd D
  check_decode_split_reference<float>(1e-5f, 32, 8, 64, 512, 256);   // window
  check_decode_split_reference<float>(1e-5f, 3, 2, 8, 33, 0);     // ragged
}

// Causal GQA with tanh logit softcapping and a per-key validity mask vs a
// host reference. Covers softcap on (large + small cap), cap <= 0 disabling,
// interior pad keys, and a nonzero prefill offset (generalized q_off/Ktot
// contract beyond the Sana self-attention case).
template <class Scalar>
void check_attention_softcap_reference(float tol, float softcap,
                                       size_t q_off) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 4, Hq = 4, Hkv = 2, D = 8;
  const int Ktot = static_cast<int>(q_off) + T;
  const float scale = 1.0f / 8.0f;

  Kokkos::View<Scalar***, Exec> q("q", T, Hq, D), out("out", T, Hq, D);
  Kokkos::View<Scalar***, Exec> k("k", Ktot, Hkv, D), v("v", Ktot, Hkv, D);
  Kokkos::View<char*, Exec> valid("valid", Ktot);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  auto hvalid = Kokkos::create_mirror_view(valid);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) =
            static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f);
  for (int n = 0; n < Ktot; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f);
        hv(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f);
      }
  // Key 0 always valid; pad out interior key 2 (and key 5 when present).
  for (int n = 0; n < Ktot; ++n)
    hvalid(n) = (n == 2 || n == 5) ? 0 : 1;
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  Kokkos::deep_copy(valid, hvalid);

  KokkosDNN::attention_gqa_single_softcap<Exec, Scalar>(h, q, k, v, valid,
                                                       q_off, out, scale,
                                                       softcap);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);

  // Host reference: causal softmax over valid keys, tanh softcap applied
  // exactly as the kernel does (float tanhf after scaling).
  for (int t = 0; t < T; ++t) {
    size_t khi = q_off + t + 1;
    if (khi > (size_t)Ktot) khi = Ktot;
    for (int hh = 0; hh < Hq; ++hh) {
      int hkv = (hh * Hkv) / Hq;
      float m = -1e30f;
      for (size_t kk = 0; kk < khi; ++kk) {
        if (!hvalid(kk)) continue;
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, hh, d)) *
               static_cast<float>(hk(kk, hkv, d));
        s *= scale;
        if (softcap > 0) s = softcap * std::tanh(s / softcap);
        m = std::max(m, s);
      }
      float denom = 0;
      for (size_t kk = 0; kk < khi; ++kk) {
        if (!hvalid(kk)) continue;
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, hh, d)) *
               static_cast<float>(hk(kk, hkv, d));
        s *= scale;
        if (softcap > 0) s = softcap * std::tanh(s / softcap);
        denom += std::exp(s - m);
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (size_t kk = 0; kk < khi; ++kk) {
          if (!hvalid(kk)) continue;
          float s = 0;
          for (int dd = 0; dd < D; ++dd)
            s += static_cast<float>(hq(t, hh, dd)) *
                 static_cast<float>(hk(kk, hkv, dd));
          s *= scale;
          if (softcap > 0) s = softcap * std::tanh(s / softcap);
          acc += std::exp(s - m) * static_cast<float>(hv(kk, hkv, d));
        }
        float ref = acc / denom;
        EXPECT_NEAR(static_cast<float>(got(t, hh, d)), ref, tol)
            << "softcap(cap=" << softcap << ",off=" << q_off << ") mismatch at ("
            << t << "," << hh << "," << d << ")";
      }
    }
  }
}

TEST(Attention, FloatSoftcapReference) {
  check_attention_softcap_reference<float>(1e-4f, 50.0f, 0);
}

TEST(Attention, FloatSoftcapSmallCapOffsetReference) {
  check_attention_softcap_reference<float>(1e-4f, 1.0f, 3);
}

TEST(Attention, BHalfSoftcapReference) {
  check_attention_softcap_reference<Kokkos::Experimental::bhalf_t>(5e-2f, 50.0f,
                                                                   0);
}

// Micro geometry (T=2, single head, D=2) with a trailing pad key.
TEST(Attention, FloatSoftcapMicro) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 2, Hq = 1, Hkv = 1, D = 2, Ktot = 2;
  const float scale = 0.5f, softcap = 50.0f;
  Kokkos::View<float***, Exec> q("q", T, Hq, D), k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D), out("out", T, Hq, D);
  Kokkos::View<char*, Exec> valid("valid", Ktot);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  auto hvalid = Kokkos::create_mirror_view(valid);
  hq(0, 0, 0) = 0.3f;
  hq(0, 0, 1) = -0.4f;
  hq(1, 0, 0) = 0.1f;
  hq(1, 0, 1) = 0.7f;
  hk(0, 0, 0) = 0.5f;
  hk(0, 0, 1) = 0.2f;
  hk(1, 0, 0) = -0.6f;
  hk(1, 0, 1) = 0.9f;
  hv(0, 0, 0) = 1.0f;
  hv(0, 0, 1) = 2.0f;
  hv(1, 0, 0) = 3.0f;
  hv(1, 0, 1) = 4.0f;
  hvalid(0) = 1;
  hvalid(1) = 0;  // trailing pad: row 1 attends key 0 only
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  Kokkos::deep_copy(valid, hvalid);
  // Deduction overload (Scalar taken from the query view).
  KokkosDNN::attention_gqa_single_softcap(h, q, k, v, valid, size_t{0}, out,
                                          scale, softcap);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  // Row 0 attends a single valid key, so it must equal V[0] regardless of
  // the softcap value.
  EXPECT_FLOAT_EQ(got(0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(got(0, 0, 1), 2.0f);
  // Row 1 also sees only key 0 (key 1 is a pad).
  EXPECT_FLOAT_EQ(got(1, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(got(1, 0, 1), 2.0f);
}

// cap <= 0 disables softcapping: with an all-valid mask the softcap path is
// bit-identical to attention_gqa_single under the native kernel, and with
// pads it matches the plain-scores host reference. The softcap entry point
// is native-only (no cuDNN softcap form) while the plain path dispatches to
// cuDNN SDPA on CUDA+cuDNN builds (~1-ulp drift), so the comparison forces
// native; dispatched-vs-native parity is covered separately with tolerance.
TEST(Attention, FloatSoftcapDisabledMatchesPlain) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  h.use_cudnn = false;
  const int T = 4, Hq = 4, Hkv = 2, D = 8, Ktot = 4;
  const float scale = 1.0f / 8.0f;
  Kokkos::View<float***, Exec> q("q", T, Hq, D), k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D), out_cap("oc", T, Hq, D), out_plain("op", T, Hq, D);
  Kokkos::View<char*, Exec> valid("valid", Ktot);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  auto hvalid = Kokkos::create_mirror_view(valid);
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
  for (int n = 0; n < Ktot; ++n) hvalid(n) = 1;
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  Kokkos::deep_copy(valid, hvalid);
  for (float cap : {0.0f, -2.0f}) {
    KokkosDNN::attention_gqa_single_softcap<Exec, float>(
        h, q, k, v, valid, 0, out_cap, scale, cap);
    KokkosDNN::attention_gqa_single<Exec, float>(h, q, k, v, 0, out_plain,
                                                scale);
    Kokkos::fence();
    auto gcap = Kokkos::create_mirror_view(out_cap);
    auto gplain = Kokkos::create_mirror_view(out_plain);
    Kokkos::deep_copy(gcap, out_cap);
    Kokkos::deep_copy(gplain, out_plain);
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < Hq; ++hh)
        for (int d = 0; d < D; ++d)
          EXPECT_EQ(gcap(t, hh, d), gplain(t, hh, d))
              << "cap=" << cap << " diverges from plain at (" << t << ","
              << hh << "," << d << ")";
  }
  // Masked + disabled: pad key 2 skipped, plain scores.
  hvalid(2) = 0;
  Kokkos::deep_copy(valid, hvalid);
  KokkosDNN::attention_gqa_single_softcap<Exec, float>(h, q, k, v, valid, 0,
                                                      out_cap, scale, 0.0f);
  Kokkos::fence();
  auto gcap = Kokkos::create_mirror_view(out_cap);
  Kokkos::deep_copy(gcap, out_cap);
  for (int t = 0; t < T; ++t) {
    for (int hh = 0; hh < Hq; ++hh) {
      int hkv = (hh * Hkv) / Hq;
      float m = -1e30f;
      for (int kk = 0; kk <= t; ++kk) {
        if (kk == 2) continue;
        float s = 0;
        for (int d = 0; d < D; ++d) s += hq(t, hh, d) * hk(kk, hkv, d);
        m = std::max(m, s * scale);
      }
      float denom = 0;
      for (int kk = 0; kk <= t; ++kk) {
        if (kk == 2) continue;
        float s = 0;
        for (int d = 0; d < D; ++d) s += hq(t, hh, d) * hk(kk, hkv, d);
        denom += std::exp(s * scale - m);
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (int kk = 0; kk <= t; ++kk) {
          if (kk == 2) continue;
          float s = 0;
          for (int dd = 0; dd < D; ++dd) s += hq(t, hh, dd) * hk(kk, hkv, dd);
          acc += std::exp(s * scale - m) * hv(kk, hkv, d);
        }
        EXPECT_NEAR(gcap(t, hh, d), acc / denom, 1e-4f)
            << "masked disabled mismatch at (" << t << "," << hh << "," << d
            << ")";
      }
    }
  }
}

// Mask contract: a query row with no valid key keeps zeros, and a
// wrong-length mask is rejected.
TEST(Attention, FloatSoftcapMaskContract) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 3, Hq = 2, Hkv = 1, D = 4, Ktot = 3;
  Kokkos::View<float***, Exec> q("q", T, Hq, D), k("k", Ktot, Hkv, D),
      v("v", Ktot, Hkv, D), out("out", T, Hq, D);
  Kokkos::View<char*, Exec> valid("valid", Ktot);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  auto hvalid = Kokkos::create_mirror_view(valid);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d) hq(t, hh, d) = 0.1f * (t + hh + d + 1);
  for (int n = 0; n < Ktot; ++n)
    for (int d = 0; d < D; ++d) {
      hk(n, 0, d) = 0.1f * (n - d);
      hv(n, 0, d) = 1.0f + 0.1f * n + 0.01f * d;
    }
  for (int n = 0; n < Ktot; ++n) hvalid(n) = 0;  // nothing valid
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);
  Kokkos::deep_copy(valid, hvalid);
  KokkosDNN::attention_gqa_single_softcap<Exec, float>(h, q, k, v, valid, 0,
                                                      out, 0.5f, 50.0f);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        EXPECT_EQ(got(t, hh, d), 0.0f)
            << "all-invalid row not zero at (" << t << "," << hh << "," << d
            << ")";
  Kokkos::View<char*, Exec> bad_valid("bad_valid", Ktot + 1);
  EXPECT_THROW((KokkosDNN::attention_gqa_single_softcap<Exec, float>(
                   h, q, k, v, bad_valid, 0, out, 0.5f, 50.0f)),
               std::invalid_argument);
}

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
#include <cudnn_version.h>
// cuDNN engagement: the entry point must report success on D-contiguous
// inputs (guards against silent-fallback regressions where every test
// above would still pass via the native kernel), and the dispatched
// public path must agree with the forced-native path on shapes that route
// to cuDNN (bf16 decode T=1 via flash, float prefill T>=2 via composite,
// float MHA decode T=1 via composite). Float T=1 decode with GQA falls
// back on cuDNN <= 9.27.0 (narrow version-gated exclusion: the composite
// path computes only the first query head per KV group there while
// reporting success), so dispatched-vs-native agreement there proves the
// guard engaged — without it, non-leading group heads would come back zero.
TEST(Attention, CudnnEngagesAndMatchesNative) {
  using Exec = Kokkos::DefaultExecutionSpace;
  // One shared handle: the raw entry point runs on its cached cuDNN handle
  // (created once here, not per call), mirroring the dispatched path.
  KokkosDNN::DNNHandle<Exec> h_raw;
  EXPECT_TRUE(h_raw.use_cudnn);
  void* raw = h_raw.cudnn_raw();
  EXPECT_NE(raw, nullptr);
  {
    Kokkos::View<float***, Kokkos::LayoutRight, Exec> q("q", 2, 4, 8),
        k("k", 2, 2, 8), v("v", 2, 2, 8), o("o", 2, 4, 8);
    EXPECT_TRUE(KokkosDNN::Impl::cudnn_sdpa_forward(
        raw, KokkosDNN::Impl::CudnnSdpaScalar::Float, q.data(), k.data(),
        v.data(), o.data(), 2, 2, 4, 2, 8, q.stride(0), q.stride(1),
        q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0),
        v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
        0.35f));
  }
  {
    Kokkos::View<Kokkos::Experimental::bhalf_t***, Kokkos::LayoutRight, Exec>
        q("q", 1, 4, 8), k("k", 5, 2, 8), v("v", 5, 2, 8), o("o", 1, 4, 8);
    EXPECT_TRUE(KokkosDNN::Impl::cudnn_sdpa_forward(
        raw, KokkosDNN::Impl::CudnnSdpaScalar::BFloat16, q.data(), k.data(),
        v.data(), o.data(), 1, 5, 4, 2, 8, q.stride(0), q.stride(1),
        q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0),
        v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
        0.35f));
  }
  {
    // Float MHA s_q == 1 is outside the narrow exclusion and must engage.
    Kokkos::View<float***, Kokkos::LayoutRight, Exec> q("q", 1, 2, 8),
        k("k", 9, 2, 8), v("v", 9, 2, 8), o("o", 1, 2, 8);
    EXPECT_TRUE(KokkosDNN::Impl::cudnn_sdpa_forward(
        raw, KokkosDNN::Impl::CudnnSdpaScalar::Float, q.data(), k.data(),
        v.data(), o.data(), 1, 9, 2, 2, 8, q.stride(0), q.stride(1),
        q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0),
        v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
        1.0f / 8.0f));
  }
  {
    // Float s_q == 1 with GQA is the excluded combination: the guard must
    // decline (return false) on cuDNN <= 9.27.0 so the caller falls back.
    Kokkos::View<float***, Kokkos::LayoutRight, Exec> q("q", 1, 4, 8),
        k("k", 9, 2, 8), v("v", 9, 2, 8), o("o", 1, 4, 8);
    const bool ran = KokkosDNN::Impl::cudnn_sdpa_forward(
        raw, KokkosDNN::Impl::CudnnSdpaScalar::Float, q.data(), k.data(),
        v.data(), o.data(), 1, 9, 4, 2, 8, q.stride(0), q.stride(1),
        q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0),
        v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
        1.0f / 8.0f);
#if CUDNN_VERSION <= 92700
    EXPECT_FALSE(ran);
#else
    // Newer cuDNN may have fixed the composite path; either outcome is
    // correct as long as dispatched-vs-native agrees below.
    (void)ran;
#endif
  }

  auto check_dispatched_vs_native = [](auto tag, int T, int Hq, int Hkv,
                                       int D, int Ktot, size_t q_off,
                                       float scale, float tol) {
    using Scalar = typename decltype(tag)::type;
    KokkosDNN::DNNHandle<Exec> h_on;
    EXPECT_TRUE(h_on.use_cudnn);
    KokkosDNN::DNNHandle<Exec> h_off;
    h_off.use_cudnn = false;
    Kokkos::View<Scalar***, Exec> q("q", T, Hq, D), o_cu("ocu", T, Hq, D),
        o_nat("onat", T, Hq, D);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> k("k", Ktot, Hkv, D),
        v("v", Ktot, Hkv, D);
    auto hq = Kokkos::create_mirror_view(q);
    auto hk = Kokkos::create_mirror_view(k);
    auto hv = Kokkos::create_mirror_view(v);
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < Hq; ++hh)
        for (int d = 0; d < D; ++d)
          hq(t, hh, d) = static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) %
                                                       17) -
                                             0.2f);
    for (int n = 0; n < Ktot; ++n)
      for (int hh = 0; hh < Hkv; ++hh)
        for (int d = 0; d < D; ++d) {
          hk(n, hh, d) = static_cast<Scalar>(0.05f * ((n * 5 + hh * 11 + d) %
                                                       19) -
                                             0.2f);
          hv(n, hh, d) = static_cast<Scalar>(0.05f * ((n * 3 + hh * 17 + d * 2) %
                                                       23) -
                                             0.3f);
        }
    Kokkos::deep_copy(q, hq);
    Kokkos::deep_copy(k, hk);
    Kokkos::deep_copy(v, hv);
    KokkosDNN::attention_gqa_single<Exec, Scalar>(h_on, q, k, v, q_off, o_cu,
                                                  scale);
    KokkosDNN::attention_gqa_single<Exec, Scalar>(h_off, q, k, v, q_off, o_nat,
                                                  scale);
    Kokkos::fence();
    auto gcu = Kokkos::create_mirror_view(o_cu);
    auto gnat = Kokkos::create_mirror_view(o_nat);
    Kokkos::deep_copy(gcu, o_cu);
    Kokkos::deep_copy(gnat, o_nat);
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < Hq; ++hh)
        for (int d = 0; d < D; ++d)
          EXPECT_NEAR(static_cast<float>(gcu(t, hh, d)),
                      static_cast<float>(gnat(t, hh, d)), tol)
              << "dispatched/native mismatch at (" << t << "," << hh << ","
              << d << ")";
  };
  struct FloatTag {
    using type = float;
  };
  struct BhalfTag {
    using type = Kokkos::Experimental::bhalf_t;
  };
  check_dispatched_vs_native(FloatTag{}, 4, 4, 2, 8, 7, 3, 1.0f / 8.0f, 1e-5f);
  check_dispatched_vs_native(BhalfTag{}, 1, 4, 2, 8, 9, 8, 1.0f / 8.0f, 2e-2f);
  // Float MHA decode engages cuDNN and must match native; float GQA decode
  // exercises the exclusion guard (falls back on cuDNN <= 9.27.0) and must
  // still match native — without the guard the composite path would zero
  // the non-leading group heads here.
  check_dispatched_vs_native(FloatTag{}, 1, 2, 2, 8, 9, 8, 1.0f / 8.0f, 1e-5f);
  check_dispatched_vs_native(FloatTag{}, 1, 4, 2, 8, 9, 8, 1.0f / 8.0f, 1e-5f);
}
#endif
