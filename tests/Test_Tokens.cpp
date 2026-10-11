// tokens_assemble (ViT token assembly, cat[cls,registers,patches]): batched
// vs host reference, shared-prefix broadcast ([D] and [1,D] CLS forms),
// single-sequence path, and argument validation.
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

float fill_val(int b, int t, int d) {
  return 0.05f * (float)(b * 131 + t * 17 + d + 1) - 0.3f;
}

}  // namespace

// Fully batched assembly vs a host loop reference over [B,T,D].
TEST(Tokens, BatchedMatchesHostReference) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, R = 2, Np = 3, D = 4, T = 1 + R + Np;

  HostView3R<Scalar> hcls("hcls", B, 1, D);
  HostView3R<Scalar> hreg("hreg", B, R, D);
  HostView3R<Scalar> hpat("hpat", B, Np, D);
  for (size_t b = 0; b < B; ++b) {
    for (size_t d = 0; d < D; ++d) hcls(b, 0, d) = fill_val((int)b, 0, (int)d);
    for (size_t r = 0; r < R; ++r)
      for (size_t d = 0; d < D; ++d)
        hreg(b, r, d) = fill_val((int)b, 10 + (int)r, (int)d);
    for (size_t p = 0; p < Np; ++p)
      for (size_t d = 0; d < D; ++d)
        hpat(b, p, d) = fill_val((int)b, 20 + (int)p, (int)d);
  }

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> cls("cls", B, 1, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> reg("reg", B, R, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pat("pat", B, Np, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(cls, hcls);
  Kokkos::deep_copy(reg, hreg);
  Kokkos::deep_copy(pat, hpat);

  KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        float ref = (t == 0) ? (float)hcls(b, 0, d)
                    : (t < 1 + R)
                        ? (float)hreg(b, t - 1, d)
                        : (float)hpat(b, t - 1 - R, d);
        EXPECT_FLOAT_EQ(got(b, t, d), ref)
            << "mismatch at (" << b << "," << t << "," << d << ")";
      }
}

// Shared prefix ([D] CLS + [R,D] registers) is broadcast across the batch.
TEST(Tokens, BroadcastPrefixMatchesHostReference) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 3, R = 2, Np = 2, D = 4, T = 1 + R + Np;

  HostView1<Scalar> hcls("hcls", D);
  HostView2R<Scalar> hreg("hreg", R, D);
  HostView3R<Scalar> hpat("hpat", B, Np, D);
  for (size_t d = 0; d < D; ++d) hcls(d) = fill_val(0, 0, (int)d);
  for (size_t r = 0; r < R; ++r)
    for (size_t d = 0; d < D; ++d) hreg(r, d) = fill_val(0, 10 + (int)r, (int)d);
  for (size_t b = 0; b < B; ++b)
    for (size_t p = 0; p < Np; ++p)
      for (size_t d = 0; d < D; ++d)
        hpat(b, p, d) = fill_val((int)b, 20 + (int)p, (int)d);

  Kokkos::View<Scalar*, Exec> cls("cls", D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> reg("reg", R, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pat("pat", B, Np, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(cls, hcls);
  Kokkos::deep_copy(reg, hreg);
  Kokkos::deep_copy(pat, hpat);

  KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        float ref = (t == 0) ? (float)hcls(d)
                    : (t < 1 + R)
                        ? (float)hreg(t - 1, d)
                        : (float)hpat(b, t - 1 - R, d);
        EXPECT_FLOAT_EQ(got(b, t, d), ref)
            << "mismatch at (" << b << "," << t << "," << d << ")";
      }
}

// [1,D] CLS matrix form broadcasts identically to the [D] vector form.
TEST(Tokens, BroadcastClsMatrixForm) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 2, R = 1, Np = 2, D = 3, T = 1 + R + Np;

  HostView2R<Scalar> hcls("hcls", 1, D);
  HostView2R<Scalar> hreg("hreg", R, D);
  HostView3R<Scalar> hpat("hpat", B, Np, D);
  for (size_t d = 0; d < D; ++d) hcls(0, d) = fill_val(0, 0, (int)d);
  for (size_t d = 0; d < D; ++d) hreg(0, d) = fill_val(0, 10, (int)d);
  for (size_t b = 0; b < B; ++b)
    for (size_t p = 0; p < Np; ++p)
      for (size_t d = 0; d < D; ++d)
        hpat(b, p, d) = fill_val((int)b, 20 + (int)p, (int)d);

  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> cls("cls", 1, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> reg("reg", R, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pat("pat", B, Np, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(cls, hcls);
  Kokkos::deep_copy(reg, hreg);
  Kokkos::deep_copy(pat, hpat);

  KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat, out);
  Kokkos::fence();

  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        float ref = (t == 0) ? (float)hcls(0, d)
                    : (t < 1 + R)
                        ? (float)hreg(t - 1, d)
                        : (float)hpat(b, t - 1 - R, d);
        EXPECT_FLOAT_EQ(got(b, t, d), ref)
            << "mismatch at (" << b << "," << t << "," << d << ")";
      }
}

// Single-sequence path: [Np,D] -> [T,D], both CLS forms.
TEST(Tokens, SingleSequenceMatchesHostReference) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t R = 2, Np = 3, D = 4, T = 1 + R + Np;

  HostView1<Scalar> hcls("hcls", D);
  HostView2R<Scalar> hreg("hreg", R, D);
  HostView2R<Scalar> hpat("hpat", Np, D);
  for (size_t d = 0; d < D; ++d) hcls(d) = fill_val(0, 0, (int)d);
  for (size_t r = 0; r < R; ++r)
    for (size_t d = 0; d < D; ++d) hreg(r, d) = fill_val(0, 10 + (int)r, (int)d);
  for (size_t p = 0; p < Np; ++p)
    for (size_t d = 0; d < D; ++d)
      hpat(p, d) = fill_val(0, 20 + (int)p, (int)d);
  auto ref_at = [&](size_t t, size_t d) {
    return (t == 0) ? (float)hcls(d)
           : (t < 1 + R)
               ? (float)hreg(t - 1, d)
               : (float)hpat(t - 1 - R, d);
  };

  Kokkos::View<Scalar*, Exec> cls("cls", D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> reg("reg", R, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> pat("pat", Np, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> out("out", T, D);
  Kokkos::deep_copy(cls, hcls);
  Kokkos::deep_copy(reg, hreg);
  Kokkos::deep_copy(pat, hpat);

  KokkosDNN::tokens_assemble_single<Exec, Scalar>(cls, reg, pat, out);
  Kokkos::fence();
  HostView2R<Scalar> got("got", T, D);
  Kokkos::deep_copy(got, out);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d)
      EXPECT_FLOAT_EQ(got(t, d), ref_at(t, d))
          << "mismatch at (" << t << "," << d << ")";

  // [1,D] CLS form agrees with the [D] form.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> clsm("clsm", 1, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> out2("out2", T, D);
  HostView2R<Scalar> hclsm("hclsm", 1, D);
  for (size_t d = 0; d < D; ++d) hclsm(0, d) = hcls(d);
  Kokkos::deep_copy(clsm, hclsm);
  KokkosDNN::tokens_assemble_single<Exec, Scalar>(clsm, reg, pat, out2);
  Kokkos::fence();
  HostView2R<Scalar> got2("got2", T, D);
  Kokkos::deep_copy(got2, out2);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d)
      EXPECT_FLOAT_EQ(got2(t, d), ref_at(t, d))
          << "matrix-cls mismatch at (" << t << "," << d << ")";
}

// Zero registers (R=0) degrades to CLS + patches.
TEST(Tokens, ZeroRegisters) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  const size_t B = 1, R = 0, Np = 2, D = 3, T = 1 + Np;

  HostView1<Scalar> hcls("hcls", D);
  HostView3R<Scalar> hpat("hpat", B, Np, D);
  for (size_t d = 0; d < D; ++d) hcls(d) = fill_val(0, 0, (int)d);
  for (size_t p = 0; p < Np; ++p)
    for (size_t d = 0; d < D; ++d) hpat(0, p, d) = fill_val(0, 20 + (int)p, (int)d);

  Kokkos::View<Scalar*, Exec> cls("cls", D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> reg("reg", R, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pat("pat", B, Np, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(cls, hcls);
  Kokkos::deep_copy(pat, hpat);

  KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat, out);
  Kokkos::fence();
  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t t = 0; t < T; ++t)
    for (size_t d = 0; d < D; ++d) {
      float ref = (t == 0) ? (float)hcls(d) : (float)hpat(0, t - 1, d);
      EXPECT_FLOAT_EQ(got(0, t, d), ref) << "mismatch at (" << t << "," << d << ")";
    }
}

// Scalar-agnostic smoke: bhalf_t broadcast path round-trips exactly
// (plain copies, no arithmetic).
TEST(Tokens, BHalfSmoke) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = Kokkos::Experimental::bhalf_t;
  const size_t B = 2, R = 1, Np = 2, D = 4, T = 1 + R + Np;

  HostView1<Scalar> hcls("hcls", D);
  HostView2R<Scalar> hreg("hreg", R, D);
  HostView3R<Scalar> hpat("hpat", B, Np, D);
  for (size_t d = 0; d < D; ++d)
    hcls(d) = static_cast<Scalar>(fill_val(0, 0, (int)d));
  for (size_t d = 0; d < D; ++d)
    hreg(0, d) = static_cast<Scalar>(fill_val(0, 10, (int)d));
  for (size_t b = 0; b < B; ++b)
    for (size_t p = 0; p < Np; ++p)
      for (size_t d = 0; d < D; ++d)
        hpat(b, p, d) = static_cast<Scalar>(fill_val((int)b, 20 + (int)p, (int)d));

  Kokkos::View<Scalar*, Exec> cls("cls", D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> reg("reg", R, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pat("pat", B, Np, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::deep_copy(cls, hcls);
  Kokkos::deep_copy(reg, hreg);
  Kokkos::deep_copy(pat, hpat);

  KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat, out);
  Kokkos::fence();
  HostView3R<Scalar> got("got", B, T, D);
  Kokkos::deep_copy(got, out);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        float ref = (t == 0) ? static_cast<float>(hcls(d))
                    : (t < 1 + R)
                        ? static_cast<float>(hreg(t - 1, d))
                        : static_cast<float>(hpat(b, t - 1 - R, d));
        EXPECT_FLOAT_EQ(static_cast<float>(got(b, t, d)), ref)
            << "mismatch at (" << b << "," << t << "," << d << ")";
      }
}

TEST(Tokens, RejectsBadArgs) {
  using Exec   = Kokkos::DefaultExecutionSpace;
  using Scalar = float;
  // Base valid config: B=2, R=2, Np=3, D=4, T=6.
  const size_t B = 2, R = 2, Np = 3, D = 4, T = 1 + R + Np;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> cls("cls", B, 1, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> reg("reg", B, R, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> pat("pat", B, Np, D);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, T, D);
  Kokkos::View<Scalar*, Exec> clsv("clsv", D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> regm("regm", R, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> clsm("clsm", 1, D);
  EXPECT_NO_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat, out)));
  EXPECT_NO_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(clsv, regm, pat, out)));
  EXPECT_NO_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(clsm, regm, pat, out)));

  // Batched: CLS must be [B,1,D].
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_cls_b("bad_cls_b", 1,
                                                              1, D);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(bad_cls_b, reg, pat,
                                                        out)),
      std::invalid_argument);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_cls_t("bad_cls_t", B,
                                                              2, D);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(bad_cls_t, reg, pat,
                                                        out)),
      std::invalid_argument);
  // Batched: registers batch / channel mismatch.
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_reg_b("bad_reg_b", 1,
                                                              R, D);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, bad_reg_b, pat,
                                                        out)),
      std::invalid_argument);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_reg_d("bad_reg_d", B,
                                                              R, D + 1);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, bad_reg_d, pat,
                                                        out)),
      std::invalid_argument);
  // Output token count / batch / channel mismatch.
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_out_t("bad_out_t", B,
                                                              T - 1, D);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat,
                                                        bad_out_t)),
      std::invalid_argument);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_out_b("bad_out_b",
                                                              B + 1, T, D);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat,
                                                        bad_out_b)),
      std::invalid_argument);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> bad_out_d("bad_out_d", B,
                                                              T, D + 1);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(cls, reg, pat,
                                                        bad_out_d)),
      std::invalid_argument);

  // Broadcast: CLS length / matrix shape, registers channel.
  Kokkos::View<Scalar*, Exec> bad_clsv("bad_clsv", D + 1);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(bad_clsv, regm, pat,
                                                        out)),
      std::invalid_argument);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad_clsm("bad_clsm", 2, D);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(bad_clsm, regm, pat,
                                                        out)),
      std::invalid_argument);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad_regm("bad_regm", R,
                                                             D + 1);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(clsv, bad_regm, pat,
                                                        out)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_forward<Exec, Scalar>(clsv, regm, pat,
                                                        bad_out_t)),
      std::invalid_argument);

  // Single-sequence: output rows must be 1+R+Np; CLS / registers checked.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> spat("spat", Np, D);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> sout("sout", T, D);
  EXPECT_NO_THROW(
      (KokkosDNN::tokens_assemble_single<Exec, Scalar>(clsv, regm, spat, sout)));
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> bad_sout("bad_sout", T - 1,
                                                             D);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_single<Exec, Scalar>(clsv, regm, spat,
                                                       bad_sout)),
      std::invalid_argument);
  Kokkos::View<Scalar*, Exec> bad_scls("bad_scls", D + 1);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_single<Exec, Scalar>(bad_scls, regm, spat,
                                                       sout)),
      std::invalid_argument);
  EXPECT_THROW(
      (KokkosDNN::tokens_assemble_single<Exec, Scalar>(clsv, bad_regm, spat,
                                                       sout)),
      std::invalid_argument);
}
