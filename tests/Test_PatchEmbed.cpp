// patchembed_forward (ViT patch embed, im2col + gemm_bias) vs naive host
// conv reference: non-overlapping stride path, overlapping K/S, bias add,
// unbiased empty-bias path, and argument validation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <stdexcept>

#include "KokkosDNN.hpp"

// Naive host Conv2d reference: out[b, ph*Wp+pw, d] =
//   bias[d] + sum_{c,kh,kw} img[b,c,ph*S+kh,pw*S+kw] * wt[d, c*K*K+kh*K+kw].
// Host views live in HostSpace explicitly (see Test_RoPE2D note: on Serial
// a mirror may alias device memory).
template <class Scalar>
void host_conv_ref(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, Kokkos::HostSpace>& img,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace>& wt,
    const Kokkos::View<Scalar*, Kokkos::HostSpace>& bs, bool biased,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, Kokkos::HostSpace>& ref,
    int K, int S) {
  const size_t B = img.extent(0), C = img.extent(1);
  const size_t H = img.extent(2), W = img.extent(3);
  const size_t D = wt.extent(0);
  const size_t Hp = (H - (size_t)K) / (size_t)S + 1;
  const size_t Wp = (W - (size_t)K) / (size_t)S + 1;
  for (size_t b = 0; b < B; ++b)
    for (size_t ph = 0; ph < Hp; ++ph)
      for (size_t pw = 0; pw < Wp; ++pw) {
        const size_t p = ph * Wp + pw;
        for (size_t d = 0; d < D; ++d) {
          float acc = biased ? static_cast<float>(bs(d)) : 0.0f;
          for (size_t c = 0; c < C; ++c)
            for (int kh = 0; kh < K; ++kh)
              for (int kw = 0; kw < K; ++kw)
                acc += static_cast<float>(img(b, c, ph * (size_t)S + (size_t)kh,
                                              pw * (size_t)S + (size_t)kw)) *
                       static_cast<float>(wt(d, c * (size_t)(K * K) +
                                                (size_t)(kh * K + kw)));
          ref(b, p, d) = static_cast<Scalar>(acc);
        }
      }
}

template <class Scalar>
void fill_patchembed_inputs(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, Kokkos::HostSpace>& img,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Kokkos::HostSpace>& wt,
    const Kokkos::View<Scalar*, Kokkos::HostSpace>& bs) {
  for (size_t b = 0; b < img.extent(0); ++b)
    for (size_t c = 0; c < img.extent(1); ++c)
      for (size_t h = 0; h < img.extent(2); ++h)
        for (size_t w = 0; w < img.extent(3); ++w)
          img(b, c, h, w) = static_cast<Scalar>(
              0.05f * (float)(b * 61 + c * 17 + h * 5 + w + 1) - 0.2f);
  for (size_t d = 0; d < wt.extent(0); ++d)
    for (size_t k = 0; k < wt.extent(1); ++k)
      wt(d, k) = static_cast<Scalar>(0.03f * (float)(d * 7 + k + 1) - 0.1f);
  for (size_t d = 0; d < bs.extent(0); ++d)
    bs(d) = static_cast<Scalar>(-0.5f + 0.25f * (float)d);
}

// One config end to end: device forward vs host conv reference.
template <class Scalar>
void check_patchembed(int B, int C, int H, int W, int D, int K, int S,
                      bool biased, float tol) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using Host = Kokkos::HostSpace;
  KokkosDNN::DNNHandle<Exec> h;
  const size_t Hp = ((size_t)H - (size_t)K) / (size_t)S + 1;
  const size_t Wp = ((size_t)W - (size_t)K) / (size_t)S + 1;
  const size_t Np = Hp * Wp, CKK = (size_t)C * (size_t)K * (size_t)K;

  Kokkos::View<Scalar****, Kokkos::LayoutRight, Host> himg("himg", B, C, H, W);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Host> hwt("hwt", D, CKK);
  Kokkos::View<Scalar*, Host> hbs("hbs", D);
  fill_patchembed_inputs(himg, hwt, hbs);

  Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> images("images", B, C, H,
                                                             W);
  Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec> weight("weight", D, CKK);
  Kokkos::View<Scalar*, Exec> bias("bias", biased ? D : 0);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec> out("out", B, Np, D);
  Kokkos::deep_copy(images, himg);
  Kokkos::deep_copy(weight, hwt);
  if (biased) Kokkos::deep_copy(bias, hbs);

  KokkosDNN::patchembed_forward(h, images, weight, bias, out, K, S);
  Kokkos::fence();

  Kokkos::View<Scalar***, Kokkos::LayoutRight, Host> got("got", B, Np, D);
  Kokkos::deep_copy(got, out);
  Kokkos::View<Scalar***, Kokkos::LayoutRight, Host> ref("ref", B, Np, D);
  host_conv_ref(himg, hwt, hbs, biased, ref, K, S);
  for (size_t b = 0; b < (size_t)B; ++b)
    for (size_t p = 0; p < Np; ++p)
      for (size_t d = 0; d < (size_t)D; ++d)
        EXPECT_NEAR(static_cast<float>(got(b, p, d)),
                    static_cast<float>(ref(b, p, d)), tol)
            << "mismatch at (" << b << "," << p << "," << d << ")";
}

// Non-overlapping stride path (the ViT-B/16 K=S tiling), batched, biased.
TEST(PatchEmbed, NonOverlappingMatchesHostConv) {
  check_patchembed<float>(/*B=*/2, /*C=*/2, /*H=*/6, /*W=*/6, /*D=*/3,
                          /*K=*/2, /*S=*/2, /*biased=*/true, 1e-4f);
}

// Smallest non-trivial tiling: C=3 like RGB, K=4 over 8x8 -> 2x2 patches.
TEST(PatchEmbed, RGBLikeKernel4) {
  check_patchembed<float>(/*B=*/1, /*C=*/3, /*H=*/8, /*W=*/8, /*D=*/4,
                          /*K=*/4, /*S=*/4, /*biased=*/true, 1e-4f);
}

// Overlapping patches (S < K) exercise the general im2col offsets.
TEST(PatchEmbed, OverlappingStrideMatchesHostConv) {
  check_patchembed<float>(/*B=*/2, /*C=*/2, /*H=*/7, /*W=*/7, /*D=*/2,
                          /*K=*/3, /*S=*/2, /*biased=*/true, 1e-4f);
}

// Empty (extent-0) bias = unbiased path: matches the bias-free reference.
TEST(PatchEmbed, UnbiasedEmptyBias) {
  check_patchembed<float>(/*B=*/2, /*C=*/1, /*H=*/6, /*W=*/4, /*D=*/3,
                          /*K=*/2, /*S=*/2, /*biased=*/false, 1e-4f);
}

// Non-square image / rectangular grid.
TEST(PatchEmbed, RectangularGrid) {
  check_patchembed<float>(/*B=*/1, /*C=*/2, /*H=*/6, /*W=*/8, /*D=*/2,
                          /*K=*/2, /*S=*/2, /*biased=*/true, 1e-4f);
}

// Scalar-agnostic smoke: bhalf_t forward matches host conv at low precision.
TEST(PatchEmbed, BHalfSmoke) {
  check_patchembed<Kokkos::Experimental::bhalf_t>(
      /*B=*/1, /*C=*/2, /*H=*/4, /*W=*/4, /*D=*/2, /*K=*/2, /*S=*/2,
      /*biased=*/true, 5e-2f);
}

TEST(PatchEmbed, RejectsBadArgs) {
  using Exec = Kokkos::DefaultExecutionSpace;
  KokkosDNN::DNNHandle<Exec> h;
  // Base valid config: B=1, C=1, H=4, W=4, D=2, K=2, S=2 -> Np=4.
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> images("images", 1, 1, 4,
                                                            4);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> weight("weight", 2, 4);
  Kokkos::View<float*, Exec> bias("bias", 2);
  Kokkos::View<float*, Exec> empty_b;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> out("out", 1, 4, 2);
  EXPECT_NO_THROW(
      (KokkosDNN::patchembed_forward(h, images, weight, empty_b, out, 2, 2)));
  // Non-positive K/S.
  EXPECT_THROW((KokkosDNN::patchembed_forward(h, images, weight, bias, out, 0,
                                              2)),
               std::invalid_argument);
  EXPECT_THROW((KokkosDNN::patchembed_forward(h, images, weight, bias, out, 2,
                                              0)),
               std::invalid_argument);
  // Image not exactly tiled: 5x5 with K=2/S=2 leaves a remainder.
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> bad_img("bad_img", 1, 1,
                                                             5, 5);
  EXPECT_THROW(
      (KokkosDNN::patchembed_forward(h, bad_img, weight, bias, out, 2, 2)),
      std::invalid_argument);
  // Kernel bigger than image.
  EXPECT_THROW(
      (KokkosDNN::patchembed_forward(h, images, weight, bias, out, 8, 8)),
      std::invalid_argument);
  // Weight width != C*K*K.
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_w("bad_w", 2, 3);
  EXPECT_THROW(
      (KokkosDNN::patchembed_forward(h, images, bad_w, bias, out, 2, 2)),
      std::invalid_argument);
  // Output patch count / channel mismatch.
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_out("bad_out", 1, 3,
                                                            2);
  EXPECT_THROW(
      (KokkosDNN::patchembed_forward(h, images, weight, bias, bad_out, 2, 2)),
      std::invalid_argument);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_out_d("bad_out_d", 1,
                                                              4, 3);
  EXPECT_THROW(
      (KokkosDNN::patchembed_forward(h, images, weight, bias, bad_out_d, 2,
                                      2)),
      std::invalid_argument);
  // Bias length mismatch (non-empty, wrong size).
  Kokkos::View<float*, Exec> bad_b("bad_b", 3);
  EXPECT_THROW(
      (KokkosDNN::patchembed_forward(h, images, weight, bad_b, out, 2, 2)),
      std::invalid_argument);
}

TEST(PatchEmbedK1S1, MatchesHost1x1Conv) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using Host = Kokkos::HostSpace;
  auto probe = [](size_t a, size_t b, size_t c = 0) {
    return 0.06f * static_cast<float>((a * 31 + b * 17 + c * 13) % 19) - 0.5f;
  };
  KokkosDNN::DNNHandle<Exec> h;
  const size_t B = 2, C = 3, H = 4, W = 4, T = H * W, D = 5;
  Kokkos::View<float****, Kokkos::LayoutRight, Host> himg("himg", B, C, H, W);
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hwt("hwt", D, C);
  Kokkos::View<float*, Host> hbs("hbs", D);
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < C; ++c)
      for (size_t hh = 0; hh < H; ++hh)
        for (size_t w = 0; w < W; ++w)
          himg(b, c, hh, w) = probe(b * C * H * W + c * H * W + hh * W + w, c);
  for (size_t d = 0; d < D; ++d) {
    for (size_t c = 0; c < C; ++c) hwt(d, c) = probe(d, c, 3);
    hbs(d) = 0.2f * (float)d - 0.3f;
  }
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> images("images", B, C, H,
                                                            W);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> weight("weight", D, C);
  Kokkos::View<float*, Exec> bias("bias", D), empty_b;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> tok("tok", B, T, D),
      tok_u("tok_u", B, T, D);
  Kokkos::deep_copy(images, himg);
  Kokkos::deep_copy(weight, hwt);
  Kokkos::deep_copy(bias, hbs);
  KokkosDNN::patchembed_forward(h, images, weight, bias, tok, 1, 1);
  KokkosDNN::patchembed_forward(h, images, weight, empty_b, tok_u, 1, 1);
  Kokkos::fence();
  Kokkos::View<float***, Kokkos::LayoutRight, Host> got("got", B, T, D),
      got_u("got_u", B, T, D), ref("ref", B, T, D), ref_u("ref_u", B, T, D);
  Kokkos::deep_copy(got, tok);
  Kokkos::deep_copy(got_u, tok_u);
  // Host 1x1 reference: tok[b, h*W+w, d] = bias[d] + sum_c img[b,c,h,w]*wt[d,c].
  auto host_patch1x1 = [&](const Kokkos::View<float*, Host>& bs, bool biased,
                           const Kokkos::View<float***, Kokkos::LayoutRight, Host>& out) {
    for (size_t b = 0; b < B; ++b)
      for (size_t hh = 0; hh < H; ++hh)
        for (size_t w = 0; w < W; ++w)
          for (size_t d = 0; d < D; ++d) {
            double acc = biased ? bs(d) : 0.0;
            for (size_t c = 0; c < C; ++c)
              acc += (double)himg(b, c, hh, w) * (double)hwt(d, c);
            out(b, hh * W + w, d) = static_cast<float>(acc);
          }
  };
  host_patch1x1(hbs, true, ref);
  Kokkos::View<float*, Host> hb;
  host_patch1x1(hb, false, ref_u);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t d = 0; d < D; ++d) {
        EXPECT_NEAR(got(b, t, d), ref(b, t, d), 1e-4f)
            << " biased at (" << b << "," << t << "," << d << ")";
        EXPECT_NEAR(got_u(b, t, d), ref_u(b, t, d), 1e-4f)
            << " unbiased at (" << b << "," << t << "," << d << ")";
      }
}

TEST(Unpatchify, MatchesHostTranspose) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using Host = Kokkos::HostSpace;
  auto probe = [](size_t a, size_t b, size_t c = 0) {
    return 0.06f * static_cast<float>((a * 31 + b * 17 + c * 13) % 19) - 0.5f;
  };
  const size_t B = 2, H = 3, W = 3, T = H * W, C = 2;
  Kokkos::View<float**, Kokkos::LayoutRight, Host> htok("htok", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Host> htok3("htok3", B, T, C);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c) {
        const float v = probe(b * T + t, c);
        htok3(b, t, c) = v;
        if (b == 0) htok(t, c) = v;
      }
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> tok("tok", T, C);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> img("img", C, H, W),
      tok3("tok3", B, T, C);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> imgs("imgs", B, C, H, W);
  Kokkos::deep_copy(tok, htok);
  Kokkos::deep_copy(tok3, htok3);
  KokkosDNN::unpatchify_forward(tok, img, H, W);
  KokkosDNN::unpatchify_forward(tok3, imgs, H, W);
  Kokkos::fence();
  Kokkos::View<float***, Kokkos::LayoutRight, Host> got("got", C, H, W);
  Kokkos::View<float****, Kokkos::LayoutRight, Host> got3("got3", B, C, H, W);
  Kokkos::deep_copy(got, img);
  Kokkos::deep_copy(got3, imgs);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < C; ++c)
      EXPECT_NEAR(got(c, t / W, t % W), htok(t, c), 1e-6f)
          << " rank-2 at (" << t << "," << c << ")";
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c)
        EXPECT_NEAR(got3(b, c, t / W, t % W), htok3(b, t, c), 1e-6f)
            << " rank-3 at (" << b << "," << t << "," << c << ")";
  // Validation.
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> bad_img("bad_img", C, H,
                                                            W + 1);
  EXPECT_THROW(KokkosDNN::unpatchify_forward(tok, bad_img, H, W),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::unpatchify_forward(tok, img, 0, W),
               std::invalid_argument);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> bad_tok("bad_tok", T + 1, C);
  EXPECT_THROW(KokkosDNN::unpatchify_forward(bad_tok, img, H, W),
               std::invalid_argument);
}

// End-to-end: identity 1x1 patch embed followed by unpatchify recovers the
// VAE-latent images exactly (Sana patch_size=1 is a pure reshape under an
// identity projection).
TEST(PatchUnpatchRoundTrip, Identity1x1RecoversImages) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using Host = Kokkos::HostSpace;
  auto probe = [](size_t a, size_t b, size_t c = 0) {
    return 0.06f * static_cast<float>((a * 31 + b * 17 + c * 13) % 19) - 0.5f;
  };
  KokkosDNN::DNNHandle<Exec> h;
  const size_t B = 2, C = 4, H = 4, W = 4, T = H * W;
  Kokkos::View<float****, Kokkos::LayoutRight, Host> himg("himg", B, C, H, W);
  Kokkos::View<float**, Kokkos::LayoutRight, Host> hwt("hwt", C, C);
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < C; ++c)
      for (size_t hh = 0; hh < H; ++hh)
        for (size_t w = 0; w < W; ++w)
          himg(b, c, hh, w) = probe(b * 97 + c * 13 + hh * W + w, 7);
  for (size_t i = 0; i < C; ++i)
    for (size_t j = 0; j < C; ++j) hwt(i, j) = (i == j) ? 1.0f : 0.0f;
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> images("images", B, C, H,
                                                            W),
      back("back", B, C, H, W);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> weight("weight", C, C);
  Kokkos::View<float*, Exec> empty_b;
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> tok("tok", B, T, C);
  Kokkos::deep_copy(images, himg);
  Kokkos::deep_copy(weight, hwt);
  KokkosDNN::patchembed_forward(h, images, weight, empty_b, tok, 1, 1);
  KokkosDNN::unpatchify_forward(tok, back, H, W);
  Kokkos::fence();
  Kokkos::View<float****, Kokkos::LayoutRight, Host> got("got", B, C, H, W);
  Kokkos::deep_copy(got, back);
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < C; ++c)
      for (size_t hh = 0; hh < H; ++hh)
        for (size_t w = 0; w < W; ++w)
          EXPECT_NEAR(got(b, c, hh, w), himg(b, c, hh, w), 1e-5f)
              << " at (" << b << "," << c << "," << hh << "," << w << ")";
}
