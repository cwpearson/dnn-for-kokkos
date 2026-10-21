#ifndef KOKKOSDNN_LSTM_HPP
#define KOKKOSDNN_LSTM_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {

// Batch-first LSTM forward (inference only, eval mode).
//
//   X   : [B, T, I] inputs (LayoutRight)
//   Wih : [4H, I] input weights, Whh : [4H, H] recurrent weights
//   bih, bhh : [4H] biases (either may be empty (extent 0) for no bias)
//   Y   : [B, T, H] outputs (uni) or [B, T, 2H] (bidir, fwd||bwd concat)
//   lengths : [B] valid lengths (device view) or empty for full-T.
//
// PyTorch gate order (i, f, g, o):
//   ig = sigmoid(X*Wii + h*Whi + bi + bh)
//   fg = sigmoid(X*Wif + h*Whf + ...)
//   gg = tanh   (X*Wig + h*Whg + ...)
//   og = sigmoid(X*Wio + h*Who + ...)
//   c = fg*c + ig*gg ; h = og*tanh(nc)
// h0=c0=0. For t >= lengths[b] the output row is zero and (h,c) are frozen
// (matches pack_padded_sequence + pad_packed_sequence with enforce_sorted).
// Reverse direction runs t = len-1 .. 0 and writes Y[b][t].
//
// TPL: input projections (X*Wih^T over flattened [B*T, I]) ride the gemm
// ladder (cuBLAS/hipBLAS on device builds); the recurrent step is a native
// per-step GEMV + elementwise (B is 1 in TTS, sequential by construction).
// Scalar-agnostic (float, bhalf_t); accumulation FP32.

template <class ExecSpace, class Scalar>
void lstm_forward_uni(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Wih,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Whh,
    const Kokkos::View<Scalar*, ExecSpace>& bih,
    const Kokkos::View<Scalar*, ExecSpace>& bhh,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& Y,
    const Kokkos::View<int*, ExecSpace>& lengths =
        Kokkos::View<int*, ExecSpace>(),
    const std::string& label = "KokkosDNN::lstm_uni") {
  const size_t B = X.extent(0), T = X.extent(1), I = X.extent(2);
  const size_t H4 = Wih.extent(0);
  if (H4 % 4 != 0) throw std::invalid_argument("lstm_forward_uni: 4H mismatch");
  const size_t H = H4 / 4;
  if (Wih.extent(1) != I || Whh.extent(0) != H4 || Whh.extent(1) != H ||
      Y.extent(0) != B || Y.extent(1) != T || Y.extent(2) != H)
    throw std::invalid_argument("lstm_forward_uni: dimension mismatch");
  const bool has_bih = bih.extent(0) != 0, has_bhh = bhh.extent(0) != 0;
  if ((has_bih && bih.extent(0) != H4) || (has_bhh && bhh.extent(0) != H4))
    throw std::invalid_argument("lstm_forward_uni: bias mismatch");
  if (B == 0 || T == 0 || H == 0) return;
  using Mat2u =
      Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryUnmanaged>;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> Xf("lstm_xf", B * T,
                                                             I);
  Kokkos::parallel_for(
      label + "_flatten",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, I}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t i) {
        Xf(b * T + t, i) = X(b, t, i);
      });
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> Pi(
      "lstm_xproj", B * T, H4);
  gemm(handle, Scalar(1), Xf, Wih, Scalar(0), Pi, label + "_xproj");
  std::vector<int> hlen(B, (int)T);
  if (lengths.extent(0) != 0) {
    if (lengths.extent(0) != B)
      throw std::invalid_argument("lstm_forward_uni: lengths mismatch");
    auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                  lengths);
    for (size_t b = 0; b < B; ++b) hlen[b] = hl(b);
  }
  auto Whh_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Whh);
  Kokkos::View<Scalar*, Kokkos::HostSpace> bih_h, bhh_h;
  if (has_bih)
    bih_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bih);
  if (has_bhh)
    bhh_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bhh);
  auto Pi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Pi);
  auto Y_h = Kokkos::create_mirror_view(Y);
  // Per-batch (h,c) state: the host loop below is t-major, so state must
  // be indexed by batch to avoid leaking batch 0 into batch 1+.
  std::vector<float> hv(B * H, 0.0f), cv(B * H, 0.0f);
  std::vector<float> pre(4 * H);
  for (size_t t = 0; t < T; ++t) {
    for (size_t b = 0; b < B; ++b) {
      float* hvb = hv.data() + b * H;
      float* cvb = cv.data() + b * H;
      if ((int)t >= hlen[b]) {
        for (size_t h = 0; h < H; ++h) Y_h(b, t, h) = Scalar(0);
        continue;
      }
      for (size_t g = 0; g < 4; ++g)
        for (size_t h = 0; h < H; ++h) {
          float acc = static_cast<float>(Pi_h(b * T + t, g * H + h));
          for (size_t k = 0; k < H; ++k)
            acc += static_cast<float>(Whh_h(g * H + h, k)) * hvb[k];
          if (has_bih) acc += static_cast<float>(bih_h(g * H + h));
          if (has_bhh) acc += static_cast<float>(bhh_h(g * H + h));
          pre[g * H + h] = acc;
        }
      for (size_t h = 0; h < H; ++h) {
        const float i = 1.0f / (1.0f + expf(-pre[0 * H + h]));
        const float f = 1.0f / (1.0f + expf(-pre[1 * H + h]));
        const float g = tanhf(pre[2 * H + h]);
        const float o = 1.0f / (1.0f + expf(-pre[3 * H + h]));
        const float nc = f * cvb[h] + i * g;
        const float nh = o * tanhf(nc);
        cvb[h] = nc;
        hvb[h] = nh;
        Y_h(b, t, h) = static_cast<Scalar>(nh);
      }
    }
  }
  Kokkos::deep_copy(Y, Y_h);
}

// Bidirectional LSTM: fwd weights (*_f) + bwd weights (*_r), Y [B,T,2H] as
// [fwd || bwd]. Lengths semantics match the uni path per direction (bwd runs
// t = len-1 .. 0, padded rows zero).
template <class ExecSpace, class Scalar>
void lstm_forward_bidir(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Wih_f,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Whh_f,
    const Kokkos::View<Scalar*, ExecSpace>& bif_f,
    const Kokkos::View<Scalar*, ExecSpace>& bhf_f,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Wih_r,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Whh_r,
    const Kokkos::View<Scalar*, ExecSpace>& bif_r,
    const Kokkos::View<Scalar*, ExecSpace>& bhf_r,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& Y,
    const Kokkos::View<int*, ExecSpace>& lengths =
        Kokkos::View<int*, ExecSpace>(),
    const std::string& label = "KokkosDNN::lstm_bidir") {
  const size_t B = X.extent(0), T = X.extent(1), I = X.extent(2);
  const size_t H4 = Wih_f.extent(0);
  if (H4 % 4 != 0) throw std::invalid_argument("lstm_bidir: 4H mismatch");
  const size_t H = H4 / 4;
  if (Wih_f.extent(1) != I || Whh_f.extent(0) != H4 || Whh_f.extent(1) != H ||
      Wih_r.extent(0) != H4 || Wih_r.extent(1) != I ||
      Whh_r.extent(0) != H4 || Whh_r.extent(1) != H || Y.extent(0) != B ||
      Y.extent(1) != T || Y.extent(2) != 2 * H)
    throw std::invalid_argument("lstm_bidir: dimension mismatch");
  if (B == 0 || T == 0 || H == 0) return;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> Yf("lstm_yf", B, T,
                                                             H),
      Yr("lstm_yr", B, T, H);
  lstm_forward_uni(handle, X, Wih_f, Whh_f, bif_f, bhf_f, Yf, lengths,
                   label + "_fwd");
  std::vector<int> hlen(B, (int)T);
  if (lengths.extent(0) != 0) {
    auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                  lengths);
    for (size_t b = 0; b < B; ++b) hlen[b] = hl(b);
  }
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> Xr("lstm_xr", B, T,
                                                             I);
  {
    auto Xh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X);
    auto Xrh = Kokkos::create_mirror_view(Xr);
    for (size_t b = 0; b < B; ++b)
      for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < I; ++i) {
          const int L = hlen[b];
          float v = 0.0f;
          if ((int)t < L) v = (float)Xh(b, (size_t)(L - 1 - (int)t), i);
          Xrh(b, t, i) = static_cast<Scalar>(v);
        }
    Kokkos::deep_copy(Xr, Xrh);
  }
  Kokkos::View<int*, ExecSpace> rlen("lstm_rlen", B);
  {
    auto rl = Kokkos::create_mirror_view(rlen);
    for (size_t b = 0; b < B; ++b) rl(b) = hlen[b];
    Kokkos::deep_copy(rlen, rl);
  }
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> Yrr("lstm_yrr", B,
                                                              T, H);
  lstm_forward_uni(handle, Xr, Wih_r, Whh_r, bif_r, bhf_r, Yrr, rlen,
                   label + "_bwd");
  {
    auto Yrrh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Yrr);
    auto Yrh = Kokkos::create_mirror_view(Yr);
    for (size_t b = 0; b < B; ++b)
      for (size_t t = 0; t < T; ++t)
        for (size_t h = 0; h < H; ++h) {
          const int L = hlen[b];
          float v = 0.0f;
          if ((int)t < L) v = (float)Yrrh(b, (size_t)(L - 1 - (int)t), h);
          Yrh(b, t, h) = static_cast<Scalar>(v);
        }
    Kokkos::deep_copy(Yr, Yrh);
  }
  auto Yh = Kokkos::create_mirror_view(Y);
  auto Yfh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Yf);
  auto Yrh2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Yr);
  for (size_t b = 0; b < B; ++b)
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) Yh(b, t, h) = Yfh(b, t, h);
      for (size_t h = 0; h < H; ++h) Yh(b, t, H + h) = Yrh2(b, t, h);
    }
  Kokkos::deep_copy(Y, Yh);
}

}  // namespace KokkosDNN

#endif
