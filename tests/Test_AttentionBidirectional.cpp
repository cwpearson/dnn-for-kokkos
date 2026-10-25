// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Bidirectional (non-causal) MHA: host-reference agreement (float +
// bfloat16, MHA + GQA mappings), and the
// non-causal property (early outputs depend on future tokens, unlike the
// causal kernel).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include "KokkosDNN.hpp"

// Host reference: dense softmax over ALL keys, FP32 accumulation.
template <class Scalar, class HQ, class HK, class HV>
void host_bidir_ref(const HQ& hq, const HK& hk, const HV& hv, float scale,
                    int T, int K, int Hq, int Hkv, int D, float* ref) {
  auto at = [=](float* b, int t, int h, int d) -> float& {
    return b[(size_t)(t * Hq + h) * D + d];
  };
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < Hq; ++h) {
      int hkv = (h * Hkv) / Hq;
      float m = -1e30f;
      for (int kk = 0; kk < K; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, h, d)) *
               static_cast<float>(hk(kk, hkv, d));
        m = std::max(m, s * scale);
      }
      float denom = 0;
      for (int kk = 0; kk < K; ++kk) {
        float s = 0;
        for (int d = 0; d < D; ++d)
          s += static_cast<float>(hq(t, h, d)) *
               static_cast<float>(hk(kk, hkv, d));
        denom += std::exp(s * scale - m);
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (int kk = 0; kk < K; ++kk) {
          float s = 0;
          for (int dd = 0; dd < D; ++dd)
            s += static_cast<float>(hq(t, h, dd)) *
                 static_cast<float>(hk(kk, hkv, dd));
          acc += std::exp(s * scale - m) * static_cast<float>(hv(kk, hkv, d));
        }
        at(ref, t, h, d) = acc / denom;
      }
    }
}

template <class Scalar>
void check_bidir_reference(int T, int K, int Hq, int Hkv, int D, float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  Kokkos::View<Scalar***, Exec> q("q", T, Hq, D), k("k", K, Hkv, D),
      v("v", K, Hkv, D), out("out", T, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) =
            static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f);
  for (int n = 0; n < K; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f);
        hv(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f);
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);

  // Deduction overload (Scalar from query view).
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, out, scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);

  std::vector<float> ref((size_t)T * Hq * D);
  host_bidir_ref<Scalar>(hq, hk, hv, scale, T, K, Hq, Hkv, D, ref.data());
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(static_cast<float>(got(t, hh, d)),
                    ref[(size_t)(t * Hq + hh) * D + d], tol)
            << "bidir reference mismatch at (" << t << "," << hh << "," << d
            << ")";
}

// ViT-B/16 MHA shape (12 heads, head_dim 64), self-attention K == T.
TEST(AttentionBidirectional, FloatMHAReference) {
  check_bidir_reference<float>(/*T=*/6, /*K=*/6, /*Hq=*/4, /*Hkv=*/4,
                               /*D=*/8, 1e-4f);
}

TEST(AttentionBidirectional, FloatGQAReference) {
  check_bidir_reference<float>(/*T=*/5, /*K=*/7, /*Hq=*/4, /*Hkv=*/2,
                               /*D=*/8, 1e-4f);
}

TEST(AttentionBidirectional, BHalfMHAReference) {
  check_bidir_reference<Kokkos::Experimental::bhalf_t>(
      /*T=*/5, /*K=*/5, /*Hq=*/2, /*Hkv=*/2, /*D=*/8, 5e-2f);
}

// VAE mid-block geometry: a single head with head_dim 384, past the fused
// cuDNN SDPA node's engine limit (D <= 128), so CUDA+CUDNN builds serve it
// from the TPL math path while other builds run the native kernel. Small
// T/K keeps the host reference cheap; asymmetric K exercises cross shapes.
TEST(AttentionBidirectional, SingleHeadWideDimReference) {
  check_bidir_reference<float>(/*T=*/16, /*K=*/12, /*Hq=*/1, /*Hkv=*/1,
                               /*D=*/384, 1e-4f);
}

// DiT joint-attention geometry (MOSS-Image: 24 heads, head_dim 128, dense
// T == K == HW + Sv): cond-vs-neg text-length variants must both match the
// host reference. Small HW/Sv keep the host reference cheap; LayoutRight
// mirrors the DiT scratch buffers (D-contiguous, so cuDNN takes them with
// no repack, exactly like the per-block calls in MossImageDitModel).
template <class Scalar>
void check_dit_shape_reference(int HW, int Sv, float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = HW + Sv, K = HW + Sv, Hq = 24, Hkv = 24, D = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> q("q", T, Hq, D),
      k("k", K, Hkv, D), v("v", K, Hkv, D), out("out", T, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) =
            static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f);
  for (int n = 0; n < K; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f);
        hv(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f);
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);

  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, out, scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(got, out);

  std::vector<float> ref((size_t)T * Hq * D);
  host_bidir_ref<Scalar>(hq, hk, hv, scale, T, K, Hq, Hkv, D, ref.data());
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        EXPECT_NEAR(static_cast<float>(got(t, hh, d)),
                    ref[(size_t)(t * Hq + hh) * D + d], tol)
            << "DiT-shape bidir mismatch at (" << t << "," << hh << "," << d
            << ") HW=" << HW << " Sv=" << Sv;
}

TEST(AttentionBidirectional, DitJointCondShapeReference) {
  check_dit_shape_reference<float>(/*HW=*/48, /*Sv=*/16, 1e-4f);
}

TEST(AttentionBidirectional, DitJointNegShapeReference) {
  check_dit_shape_reference<float>(/*HW=*/48, /*Sv=*/23, 1e-4f);
}

TEST(AttentionBidirectional, DitJointBHalfReference) {
  check_dit_shape_reference<Kokkos::Experimental::bhalf_t>(
      /*HW=*/48, /*Sv=*/16, 5e-2f);
}

// Zero-copy reshape path: the DiT consumes GEMM outputs through unmanaged
// [T,H,Dh] aliases over contiguous [T,H*Dh] buffers (see dit_alias_*).
// Aliased views must agree with managed views bit-exactly (same kernels,
// same bytes, no repack on either side).
TEST(AttentionBidirectional, UnmanagedAliasViewsMatchManaged) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 32, K = 32, Hq = 8, Hkv = 8, D = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  Kokkos::View<float**, Kokkos::LayoutRight, Exec> qf("qf", T, Hq * D),
      kf("kf", K, Hkv * D), vf("vf", K, Hkv * D), of("of", T, Hq * D);
  auto hqf = Kokkos::create_mirror_view(qf);
  auto hkf = Kokkos::create_mirror_view(kf);
  auto hvf = Kokkos::create_mirror_view(vf);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < Hq * D; ++d)
      hqf(t, d) = 0.05f * ((t * 7 + d * 3) % 17) - 0.2f;
  for (int n = 0; n < K; ++n)
    for (int d = 0; d < Hkv * D; ++d) {
      hkf(n, d) = 0.05f * ((n * 5 + d) % 19) - 0.2f;
      hvf(n, d) = 0.05f * ((n * 3 + d * 2) % 23) - 0.3f;
    }
  Kokkos::deep_copy(qf, hqf);
  Kokkos::deep_copy(kf, hkf);
  Kokkos::deep_copy(vf, hvf);

  using U3 = Kokkos::View<float***, Kokkos::LayoutRight, Exec,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  U3 q(qf.data(), T, Hq, D), k(kf.data(), K, Hkv, D), v(vf.data(), K, Hkv, D),
      out(of.data(), T, Hq, D);
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, out, scale);
  Kokkos::fence();

  Kokkos::View<float***, Kokkos::LayoutRight, Exec> qm("qm", T, Hq, D),
      km("km", K, Hkv, D), vm("vm", K, Hkv, D), om("om", T, Hq, D);
  Kokkos::deep_copy(qm, q);
  Kokkos::deep_copy(km, k);
  Kokkos::deep_copy(vm, v);
  KokkosDNN::attention_mha_bidirectional<Exec>(h, qm, km, vm, om, scale);
  Kokkos::fence();
  auto got_alias = Kokkos::create_mirror_view(of);
  auto got_managed = Kokkos::create_mirror_view(om);
  Kokkos::deep_copy(got_alias, of);
  Kokkos::deep_copy(got_managed, om);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        EXPECT_EQ(got_alias(t, hh * D + d), got_managed(t, hh, d))
            << "alias/managed divergence at (" << t << "," << hh << "," << d
            << ")";
}

#if defined(KOKKOSDNN_ENABLE_CUDNN) && defined(KOKKOS_ENABLE_CUDA)
#include "tpl/cudnn/KokkosDNN_CuDNN_Forward.hpp"

// Plan-cache reuse across DiT blocks: after warming the cond and neg joint
// shapes, every repeat call must be a plan-cache hit (no rebuild miss, no
// known-bad skip). Shapes repeat across the 42 blocks and across steps —
// only the text length varies cond vs neg — so a miss per block would be
// pure host-side overhead (~45 ms per new-shape build).
TEST(AttentionBidirectional, DitPlanCacheHitsAcrossBlocks) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int Hq = 8, Hkv = 8, D = 64;
  const int Tcond = 48, Tneg = 55;  // cond-vs-neg text-length diversity
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> q("q", Tneg, Hq, D),
      k("k", Tneg, Hkv, D), v("v", Tneg, Hkv, D), out("out", Tneg, Hq, D);
  Kokkos::deep_copy(q, 0.1f);
  Kokkos::deep_copy(k, 0.1f);
  Kokkos::deep_copy(v, 0.1f);
  auto sub = [](auto& w, int T) {
    return Kokkos::subview(w, std::pair<int, int>(0, T), Kokkos::ALL(),
                           Kokkos::ALL());
  };
  // Warm both shapes (absorbs the one-time builds).
  KokkosDNN::attention_mha_bidirectional<Exec>(h, sub(q, Tcond), sub(k, Tcond),
                                               sub(v, Tcond), sub(out, Tcond),
                                               scale);
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, out, scale);
  Kokkos::fence();
  uint64_t h0, m0, b0;
  KokkosDNN::Impl::sdpa_plan_cache_stats(h0, m0, b0);
  // 12 alternating cond/neg calls, as the per-sample loop issues them.
  for (int i = 0; i < 12; ++i) {
    if (i % 2 == 0)
      KokkosDNN::attention_mha_bidirectional<Exec>(
          h, sub(q, Tcond), sub(k, Tcond), sub(v, Tcond), sub(out, Tcond),
          scale);
    else
      KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, out, scale);
  }
  Kokkos::fence();
  uint64_t h1, m1, b1;
  KokkosDNN::Impl::sdpa_plan_cache_stats(h1, m1, b1);
  // One clipped subview row-range keeps D-contiguity, so every repeat must
  // hit the cached plan for its shape.
  EXPECT_EQ(h1 - h0, 12u) << "expected 12 plan-cache hits (6 cond + 6 neg)";
  EXPECT_EQ(m1 - m0, 0u) << "repeat shapes must not rebuild plans";
  EXPECT_EQ(b1 - b0, 0u) << "DiT shapes must not be known-bad";
}
#endif

// Non-causal property: perturbing a FUTURE key/value (last token) must
// change the FIRST query's output under the bidirectional kernel, while
// the causal kernel's first output (q_off=0, attending only to key 0)
// must be bit-identical.
TEST(AttentionBidirectional, FutureTokensAffectEarlyOutputs) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const int T = 4, Hq = 2, Hkv = 2, D = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  Kokkos::View<float***, Exec> q("q", T, Hq, D), k("k", T, Hkv, D),
      v("v", T, Hkv, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) = 0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f;
  for (int n = 0; n < T; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) = 0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f;
        hv(n, hh, d) = 0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f;
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);

  Kokkos::View<float***, Exec> ob1("ob1", T, Hq, D), oc1("oc1", T, Hq, D);
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, ob1, scale);
  KokkosDNN::attention_gqa_single<Exec>(h, q, k, v, /*q_off=*/0, oc1, scale);
  Kokkos::fence();

  // Perturb only the LAST token's K/V (strictly future for query 0).
  {
    auto hk2 = Kokkos::create_mirror_view(k);
    auto hv2 = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(hk2, k);
    Kokkos::deep_copy(hv2, v);
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk2(T - 1, hh, d) += 2.5f;
        hv2(T - 1, hh, d) += 2.5f;
      }
    Kokkos::deep_copy(k, hk2);
    Kokkos::deep_copy(v, hv2);
  }

  Kokkos::View<float***, Exec> ob2("ob2", T, Hq, D), oc2("oc2", T, Hq, D);
  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, ob2, scale);
  KokkosDNN::attention_gqa_single<Exec>(h, q, k, v, /*q_off=*/0, oc2, scale);
  Kokkos::fence();
  auto gb1 = Kokkos::create_mirror_view(ob1);
  auto gb2 = Kokkos::create_mirror_view(ob2);
  auto gc1 = Kokkos::create_mirror_view(oc1);
  auto gc2 = Kokkos::create_mirror_view(oc2);
  Kokkos::deep_copy(gb1, ob1);
  Kokkos::deep_copy(gb2, ob2);
  Kokkos::deep_copy(gc1, oc1);
  Kokkos::deep_copy(gc2, oc2);

  float bidir_delta = 0, causal_delta = 0;
  for (int hh = 0; hh < Hq; ++hh)
    for (int d = 0; d < D; ++d) {
      bidir_delta =
          std::max(bidir_delta, std::abs(gb2(0, hh, d) - gb1(0, hh, d)));
      causal_delta =
          std::max(causal_delta, std::abs(gc2(0, hh, d) - gc1(0, hh, d)));
    }
  EXPECT_GT(bidir_delta, 1e-3f)
      << "bidirectional out(0) ignores future tokens (looks causal)";
  EXPECT_EQ(causal_delta, 0.0f)
      << "causal out(0) changed under future-token perturbation";
}

// CUDA tensor-core fused bidirectional SDPA
// (src/KokkosDNN_AttentionTensorCuda.hpp): bhalf_t D in {64, 128}
// D-contiguous shapes past the small-size floor run the wmma kernel on
// SM80+ CUDA and the portable kernel elsewhere, so one numerics contract
// covers both. Tolerances are absolute on O(0.1) outputs: tight enough to
// catch transpose/tiling bugs (which showed up as O(0.1)-O(1) deviations
// during development) with margin over measured tensor-vs-portable
// rounding (~1e-3 max on these shapes). Covers full tiles, T/K tails, the
// D=64/128 engines, and a GQA mapping, vs both the host FP32 reference and
// the forced-portable kernel.
template <class Scalar>
void check_bidir_tensor_cores(int T, int K, int Hq, int Hkv, int D,
                              float tol_host, float tol_portable) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  h.use_cublas = false;
  h.use_cudnn = false;
  h.use_tensor_cores = true;  // native path; portable fallback elsewhere
  KokkosDNN::DNNHandle<Exec> hp;
  hp.use_cublas = false;
  hp.use_cudnn = false;
  hp.use_tensor_cores = false;  // forced portable reference
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> q("q", T, Hq, D),
      k("k", K, Hkv, D), v("v", K, Hkv, D), ot("ot", T, Hq, D),
      op("op", T, Hq, D);
  auto hq = Kokkos::create_mirror_view(q);
  auto hk = Kokkos::create_mirror_view(k);
  auto hv = Kokkos::create_mirror_view(v);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d)
        hq(t, hh, d) =
            static_cast<Scalar>(0.05f * ((t * 7 + hh * 13 + d * 3) % 17) - 0.2f);
  for (int n = 0; n < K; ++n)
    for (int hh = 0; hh < Hkv; ++hh)
      for (int d = 0; d < D; ++d) {
        hk(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 5 + hh * 11 + d) % 19) - 0.2f);
        hv(n, hh, d) =
            static_cast<Scalar>(0.05f * ((n * 3 + hh * 17 + d * 2) % 23) - 0.3f);
      }
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(v, hv);

  KokkosDNN::attention_mha_bidirectional<Exec>(h, q, k, v, ot, scale);
  KokkosDNN::attention_mha_bidirectional<Exec>(hp, q, k, v, op, scale);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view(ot);
  auto gop = Kokkos::create_mirror_view(op);
  Kokkos::deep_copy(got, ot);
  Kokkos::deep_copy(gop, op);

  std::vector<float> ref((size_t)T * Hq * D);
  host_bidir_ref<Scalar>(hq, hk, hv, scale, T, K, Hq, Hkv, D, ref.data());
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < Hq; ++hh)
      for (int d = 0; d < D; ++d) {
        const float a = static_cast<float>(got(t, hh, d));
        const float b = static_cast<float>(gop(t, hh, d));
        const float c = ref[(size_t)(t * Hq + hh) * D + d];
        EXPECT_NEAR(a, c, tol_host)
            << "tensor bidir vs host at (" << t << "," << hh << "," << d
            << ") T=" << T << " K=" << K << " Hq=" << Hq << " Hkv=" << Hkv
            << " D=" << D;
        EXPECT_NEAR(a, b, tol_portable)
            << "tensor-vs-portable mismatch at (" << t << "," << hh << ","
            << d << ") T=" << T << " K=" << K << " Hq=" << Hq << " Hkv=" << Hkv
            << " D=" << D;
      }
}

TEST(AttentionBidirectional, TensorCoresMHA128Tails) {
  check_bidir_tensor_cores<Kokkos::Experimental::bhalf_t>(
      /*T=*/150, /*K=*/140, /*Hq=*/8, /*Hkv=*/8, /*D=*/128,
      /*tol_host=*/5e-2f, /*tol_portable=*/5e-3f);
}

TEST(AttentionBidirectional, TensorCoresMHA64Tail) {
  check_bidir_tensor_cores<Kokkos::Experimental::bhalf_t>(
      /*T=*/128, /*K=*/96, /*Hq=*/8, /*Hkv=*/8, /*D=*/64,
      /*tol_host=*/5e-2f, /*tol_portable=*/5e-3f);
}

TEST(AttentionBidirectional, TensorCoresGQA128) {
  check_bidir_tensor_cores<Kokkos::Experimental::bhalf_t>(
      /*T=*/96, /*K=*/112, /*Hq=*/8, /*Hkv=*/2, /*D=*/128,
      /*tol_host=*/5e-2f, /*tol_portable=*/5e-3f);
}
