// layerscale_residual_forward (out = x + gamma*branch, gamma [D] broadcast)
// and layerscale_forward (out = gamma*branch): batched [B,T,D] vs host
// reference, single-sequence [T,D], rank-1 [D], gamma=1/0 degenerates,
// in-place aliasing, and argument validation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <stdexcept>

#include "KokkosDNN.hpp"

namespace {

// NOTE: host snapshots use explicit HostSpace views (not
// create_mirror_view): on a host backend such as Serial the mirror may
// alias device memory, which would let later kernels mutate the reference
// inputs out from under the checks.

template <class Scalar>
using HostView1 = Kokkos::View<Scalar*, Kokkos::HostSpace>;
template <class Scalar>
using HostView2R = Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace>;
template <class Scalar>
using HostView3R =
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Kokkos::HostSpace>;

float x_val(int b, int t, int d) {
  return 0.1f * (float)(b * 61 + t * 13 + d * 3 + 1) - 0.4f;
}
float br_val(int b, int t, int d) {
  return 0.07f * (float)(b * 37 + t * 11 + d * 5 + 2) - 0.2f;
}
float g_val(int d) { return 1e-3f * (float)(d + 1); }  // small-init style

}  // namespace

// Batched [B,T,D] fused residual vs a host loop reference with known gamma.
TEST(LayerScale, BatchedMatchesHostReference) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, T = 5, D = 6;

  HostView3R<Scalar> hx("hx", B, T, D);
  HostView3R<Scalar> hb("hb", B, T, D);
  HostView1<Scalar> hg("hg", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        hx(b, t, d) = x_val((int)b, (int)t, (int)d);
        hb(b, t, d) = br_val((int)b, (int)t, (int)d);
      }
  for (size_t d = 0; d < D; ++d) hg(d) = g_val((int)d);

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(br, hb);
  Kokkos::deep_copy(gamma, hg);

  KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        float ref = (float)hx(b, t, d) + (float)hg(d) * (float)hb(b, t, d);
        EXPECT_FLOAT_EQ(got(b, t, d), ref)
            << "mismatch at (" << b << "," << t << "," << d << ")";
      }
}

// Standalone multiply [B,T,D]: out = gamma*branch vs host reference.
TEST(LayerScale, MultiplyBatchedMatchesHostReference) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, T = 4, D = 5;

  HostView3R<Scalar> hb("hb", B, T, D);
  HostView1<Scalar> hg("hg", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) hb(b, t, d) = br_val((int)b, (int)t, (int)d);
  for (size_t d = 0; d < D; ++d) hg(d) = g_val((int)d);

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(br, hb);
  Kokkos::deep_copy(gamma, hg);

  KokkosDNN::layerscale_forward<Exec, Scalar>(br, gamma, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d)
        EXPECT_FLOAT_EQ(got(b, t, d), (float)hg(d) * (float)hb(b, t, d))
            << "mismatch at (" << b << "," << t << "," << d << ")";
}

// Single-sequence [T,D] fused residual vs host reference.
TEST(LayerScale, SingleSequenceMatchesHostReference) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t T = 6, D = 4;

  HostView2R<Scalar> hx("hx", T, D);
  HostView2R<Scalar> hb("hb", T, D);
  HostView1<Scalar> hg("hg", D);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d) {
      hx(t, d) = x_val(0, (int)t, (int)d);
      hb(t, d) = br_val(0, (int)t, (int)d);
    }
  for (size_t d = 0; d < D; ++d) hg(d) = g_val((int)d);

  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> x("x", T, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> br("br", T, D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> out("out", T, D);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(br, hb);
  Kokkos::deep_copy(gamma, hg);

  KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, out);
  Kokkos::fence();

  HostView2R<Scalar> got("got", T, D);
  Kokkos::deep_copy(got, out);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d) {
      float ref = (float)hx(t, d) + (float)hg(d) * (float)hb(t, d);
      EXPECT_FLOAT_EQ(got(t, d), ref) << "mismatch at (" << t << "," << d << ")";
    }

  // Standalone single-seq multiply agrees elementwise with the fused path
  // minus x.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> mout("mout", T, D);
  KokkosDNN::layerscale_forward<Exec, Scalar>(br, gamma, mout);
  Kokkos::fence();
  HostView2R<Scalar> mgot("mgot", T, D);
  Kokkos::deep_copy(mgot, mout);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d)
      EXPECT_FLOAT_EQ(mgot(t, d), (float)hg(d) * (float)hb(t, d))
          << "multiply mismatch at (" << t << "," << d << ")";
}

// Rank-1 [D] fused residual vs host reference.
TEST(LayerScale, VectorMatchesHostReference) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t D = 8;

  HostView1<Scalar> hx("hx", D);
  HostView1<Scalar> hb("hb", D);
  HostView1<Scalar> hg("hg", D);
  for (size_t d = 0; d < D; ++d) {
    hx(d) = x_val(0, 0, (int)d);
    hb(d) = br_val(0, 0, (int)d);
    hg(d) = g_val((int)d);
  }

  Kokkos::View<Scalar*, Exec> x("x", D);
  Kokkos::View<Scalar*, Exec> br("br", D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar*, Exec> out("out", D);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(br, hb);
  Kokkos::deep_copy(gamma, hg);

  KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, out);
  Kokkos::fence();

  HostView1<Scalar> got("got", D);
  Kokkos::deep_copy(got, out);
  for (size_t d = 0; d < D; ++d)
    EXPECT_FLOAT_EQ(got(d), (float)hx(d) + (float)hg(d) * (float)hb(d))
        << "mismatch at (" << d << ")";
}

// gamma=1 reduces to a plain residual add (out = x + branch).
TEST(LayerScale, GammaOneIsResidualAdd) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, T = 3, D = 4;

  HostView3R<Scalar> hx("hx", B, T, D);
  HostView3R<Scalar> hb("hb", B, T, D);
  HostView1<Scalar> hg("hg", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        hx(b, t, d) = x_val((int)b, (int)t, (int)d);
        hb(b, t, d) = br_val((int)b, (int)t, (int)d);
      }
  for (size_t d = 0; d < D; ++d) hg(d) = 1.0f;

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(br, hb);
  Kokkos::deep_copy(gamma, hg);

  KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d)
        EXPECT_FLOAT_EQ(got(b, t, d), (float)hx(b, t, d) + (float)hb(b, t, d))
            << "mismatch at (" << b << "," << t << "," << d << ")";
}

// gamma=0 returns x (blocked branch contributes nothing).
TEST(LayerScale, GammaZeroReturnsX) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, T = 3, D = 4;

  HostView3R<Scalar> hx("hx", B, T, D);
  HostView3R<Scalar> hb("hb", B, T, D);
  HostView1<Scalar> hg("hg", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        hx(b, t, d) = x_val((int)b, (int)t, (int)d);
        hb(b, t, d) = br_val((int)b, (int)t, (int)d);
      }
  for (size_t d = 0; d < D; ++d) hg(d) = 0.0f;

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(br, hb);
  Kokkos::deep_copy(gamma, hg);

  KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d)
        EXPECT_FLOAT_EQ(got(b, t, d), (float)hx(b, t, d))
            << "mismatch at (" << b << "," << t << "," << d << ")";
}

// In-place: out aliasing x, aliasing branch, and aliasing both give the
// same result as the out-of-place reference.
TEST(LayerScale, InPlaceAliasing) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, T = 4, D = 5;

  HostView3R<Scalar> hx("hx", B, T, D);
  HostView3R<Scalar> hb("hb", B, T, D);
  HostView1<Scalar> hg("hg", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        hx(b, t, d) = x_val((int)b, (int)t, (int)d);
        hb(b, t, d) = br_val((int)b, (int)t, (int)d);
      }
  for (size_t d = 0; d < D; ++d) hg(d) = g_val((int)d);
  auto ref_at = [&](size_t b, size_t t, size_t d) {
    return (float)hx(b, t, d) + (float)hg(d) * (float)hb(b, t, d);
  };

  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::deep_copy(gamma, hg);

  // out aliases x.
  {
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(br, hb);
    KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, x);
    Kokkos::fence();
    HostView3R<Scalar> got("got", B, T, D);
    Kokkos::deep_copy(got, x);
    for (size_t b = 0; b < B; ++b)
      for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d)
          EXPECT_FLOAT_EQ(got(b, t, d), ref_at(b, t, d))
              << "out==x mismatch at (" << b << "," << t << "," << d << ")";
  }

  // out aliases branch.
  {
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(br, hb);
    KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, br);
    Kokkos::fence();
    HostView3R<Scalar> got("got", B, T, D);
    Kokkos::deep_copy(got, br);
    for (size_t b = 0; b < B; ++b)
      for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d)
          EXPECT_FLOAT_EQ(got(b, t, d), ref_at(b, t, d))
              << "out==branch mismatch at (" << b << "," << t << "," << d << ")";
  }

  // x, branch, and out all the same buffer: out = x + gamma*x.
  {
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> buf("buf", B, T, D);
    Kokkos::deep_copy(buf, hx);
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bref("bref", B, T, D);
    Kokkos::deep_copy(bref, hx);
    // Reference uses hx as both inputs.
    KokkosDNN::layerscale_residual_forward<Exec, Scalar>(buf, buf, gamma, buf);
    Kokkos::fence();
    HostView3R<Scalar> got("got", B, T, D);
    Kokkos::deep_copy(got, buf);
    for (size_t b = 0; b < B; ++b)
      for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
          float ref = (float)hx(b, t, d) + (float)hg(d) * (float)hx(b, t, d);
          EXPECT_FLOAT_EQ(got(b, t, d), ref)
              << "full-alias mismatch at (" << b << "," << t << "," << d << ")";
        }
  }

  // Standalone multiply in-place (out aliases branch).
  {
    Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
    Kokkos::deep_copy(br, hb);
    KokkosDNN::layerscale_forward<Exec, Scalar>(br, gamma, br);
    Kokkos::fence();
    HostView3R<Scalar> got("got", B, T, D);
    Kokkos::deep_copy(got, br);
    for (size_t b = 0; b < B; ++b)
      for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d)
          EXPECT_FLOAT_EQ(got(b, t, d), (float)hg(d) * (float)hb(b, t, d))
              << "multiply in-place mismatch at (" << b << "," << t << ","
              << d << ")";
  }
}

// Scalar-agnostic smoke: bhalf_t fused residual matches the FP32 reference
// cast back through bhalf_t.
TEST(LayerScale, BHalfSmoke) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = Kokkos::Experimental::bhalf_t;
  const size_t B = 2, T = 3, D = 4;

  HostView3R<Scalar> hx("hx", B, T, D);
  HostView3R<Scalar> hb("hb", B, T, D);
  HostView1<Scalar> hg("hg", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        hx(b, t, d) = static_cast<Scalar>(x_val((int)b, (int)t, (int)d));
        hb(b, t, d) = static_cast<Scalar>(br_val((int)b, (int)t, (int)d));
      }
  for (size_t d = 0; d < D; ++d) hg(d) = static_cast<Scalar>(g_val((int)d));

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(br, hb);
  Kokkos::deep_copy(gamma, hg);

  KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        float ref = static_cast<float>(static_cast<Scalar>(
            static_cast<float>(hx(b, t, d)) +
            static_cast<float>(hg(d)) * static_cast<float>(hb(b, t, d))));
        EXPECT_NEAR(static_cast<float>(got(b, t, d)), ref, 1e-6f)
            << "mismatch at (" << b << "," << t << "," << d << ")";
      }
}

TEST(LayerScale, RejectsBadArgs) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, T = 4, D = 5;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> x("x", B, T, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> br("br", B, T, D);
  Kokkos::View<Scalar*, Exec> gamma("gamma", D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  EXPECT_NO_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma, out)));
  EXPECT_NO_THROW((KokkosDNN::layerscale_forward<Exec, Scalar>(br, gamma, out)));

  // gamma length mismatch.
  Kokkos::View<Scalar*, Exec> bad_gamma("bad_gamma", D + 1);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, bad_gamma,
                                                            out)),
      std::invalid_argument);
  EXPECT_THROW((KokkosDNN::layerscale_forward<Exec, Scalar>(br, bad_gamma, out)),
               std::invalid_argument);

  // branch shape mismatch (batch / tokens / channels).
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_br_b("bad_br_b", B + 1,
                                                             T, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_br_t("bad_br_t", B,
                                                             T + 1, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_br_d("bad_br_d", B, T,
                                                             D + 1);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, bad_br_b, gamma,
                                                            out)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, bad_br_t, gamma,
                                                            out)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, bad_br_d, gamma,
                                                            out)),
      std::invalid_argument);

  // output shape mismatch.
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_out("bad_out", B, T,
                                                            D + 1);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(x, br, gamma,
                                                            bad_out)),
      std::invalid_argument);
  EXPECT_THROW((KokkosDNN::layerscale_forward<Exec, Scalar>(br, gamma, bad_out)),
               std::invalid_argument);

  // Single-sequence: gamma / branch / output mismatches.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> sx("sx", T, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> sbr("sbr", T, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> sout("sout", T, D);
  EXPECT_NO_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(sx, sbr, gamma,
                                                            sout)));
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad_sbr("bad_sbr", T, D + 1);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad_sout("bad_sout", T + 1,
                                                            D);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(sx, bad_sbr, gamma,
                                                            sout)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(sx, sbr, gamma,
                                                            bad_sout)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(sx, sbr, bad_gamma,
                                                            sout)),
      std::invalid_argument);

  // Rank-1: gamma / branch / output mismatches.
  Kokkos::View<Scalar*, Exec> vx("vx", D);
  Kokkos::View<Scalar*, Exec> vbr("vbr", D);
  Kokkos::View<Scalar*, Exec> vout("vout", D);
  EXPECT_NO_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(vx, vbr, gamma,
                                                            vout)));
  Kokkos::View<Scalar*, Exec> bad_v("bad_v", D + 1);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(vx, bad_v, gamma,
                                                            vout)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(vx, vbr, bad_gamma,
                                                            vout)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::layerscale_residual_forward<Exec, Scalar>(vx, vbr, gamma,
                                                            bad_v)),
      std::invalid_argument);
}
