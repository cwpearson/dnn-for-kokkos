// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_LSTM_HPP
#define KOKKOSDNN_LSTM_HPP

// Single-layer bidirectional LSTM, batch-first, with per-batch lengths.
//
// Covers the StyleTTS2/Kokoro recurrence stack (DurationEncoder 3x
// LSTM(640 -> 256 bidir), predictor lstm/shared LSTM(640 -> 256 bidir),
// TextEncoder LSTM(512 -> 256 bidir)): all single-layer, batch_first=True,
// bidirectional=True, bias=True, dropout inert (num_layers == 1).
//
// Math (PyTorch nn.LSTM gate order i, f, g(cell), o):
//   pre = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh          ([4H], FP32)
//   i,f,g,o = split(pre); i=sig, f=sig, g=tanh, o=sig
//   c = f * c_prev + i * g;  h = o * tanh(c)
//   y[b,t] = concat(h_fwd, h_rev)                        ([2H])
// Lengths semantics match pack_padded_sequence(enforce_sorted=False) +
// pad_packed_sequence for the single-layer case: steps t >= len[b] are
// skipped (states freeze) and outputs there are zero. h0/c0 are zero.
//
// Scalar-agnostic I/O (float, bhalf_t); accumulation in FP32. Portable: one
// team per (batch, direction), TeamThreadRange over the gate/hidden rows,
// no backend #ifdefs, runs on Serial.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Impl {

template <class ExecSpace>
struct LstmBidirStepFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> x;  // [B,T,D]
  Kokkos::View<int*, ExecSpace> lengths;                     // [B]
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> wih;  // [2,4H,D]
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> whh;  // [2,4H,H]
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> bia;   // [2,4H]
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> gates;  // [2B,4H]
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> hc;     // [2B,2H]
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> y;     // [B,T,2H]
  size_t B, T, D, H;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& m) const {
    // Local handles: the KOKKOS_LAMBDAs below capture by value, so bind
    // members to locals to avoid an implicit `this` capture (deprecated
    // in C++20; an error under -Werror on the nvcc path).
    const auto bia_ = bia;
    const auto wih_ = wih;
    const auto whh_ = whh;
    const auto hc_ = hc;
    const auto x_ = x;
    const auto y_ = y;
    const auto gates_ = gates;
    const size_t H_ = H, D_ = D;
    const size_t r = (size_t)m.league_rank();
    const size_t b = r / 2;
    const size_t dir = r % 2;
    const size_t len = (size_t)lengths(b);
    const size_t H4 = 4 * H;
    for (size_t s = 0; s < len; ++s) {
      const size_t t = (dir == 0) ? s : (len - 1 - s);
      // Gate pre-activations: input part + bias + recurrent part.
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(m, H4), KOKKOS_LAMBDA(size_t k) {
            float acc = bia_(dir, k);
            for (size_t j = 0; j < D_; ++j) acc += wih_(dir, k, j) * x_(b, t, j);
            for (size_t j = 0; j < H_; ++j) acc += whh_(dir, k, j) * hc_(r, H_ + j);
            gates_(r, k) = acc;
          });
      m.team_barrier();
      // Recurrence update + output write, parallel over H (each h touches
      // only its own 4 gates and its own state/output slots).
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(m, H), KOKKOS_LAMBDA(size_t h) {
            const float i = 1.0f / (1.0f + expf(-gates_(r, h)));
            const float f = 1.0f / (1.0f + expf(-gates_(r, H_ + h)));
            const float g = tanhf(gates_(r, 2 * H_ + h));
            const float o = 1.0f / (1.0f + expf(-gates_(r, 3 * H_ + h)));
            const float c = f * hc_(r, h) + i * g;
            const float hh = o * tanhf(c);
            hc_(r, h) = c;
            hc_(r, H_ + h) = hh;
            y_(b, t, dir * H_ + h) = hh;
          });
      m.team_barrier();
    }
  }
};

}  // namespace Impl

// Single-layer bidirectional LSTM forward, batch-first:
//   x       : [B, T, D] inputs (any Scalar; widened to FP32)
//   lengths : [B] valid steps per batch (0 <= len <= T)
//   w_ih    : [2, 4H, D] input weights (dir 0 = fwd, 1 = rev)
//   w_hh    : [2, 4H, H] recurrent weights
//   b_ih    : [2, 4H] input biases, b_hh : [2, 4H] recurrent biases
//   y       : [B, T, 2H] outputs (fwd/rev concat); rows t >= lengths[b]
//             are zero.
// FP32 accumulation; the only Scalar rounding is the final write.
// Throws on dimension mismatch.
template <class ExecSpace, class Scalar>
void lstm_bidirectional(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<int*, ExecSpace>& lengths,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& w_ih,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& w_hh,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& b_ih,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& b_hh,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::lstm_bidirectional") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  const size_t H2 = y.extent(2);
  if (H2 % 2 != 0)
    throw std::invalid_argument("lstm_bidirectional: y dim must be even");
  const size_t H = H2 / 2;
  if (y.extent(0) != B || y.extent(1) != T || lengths.extent(0) != B)
    throw std::invalid_argument("lstm_bidirectional: shape mismatch");
  if (w_ih.extent(0) != 2 || w_ih.extent(1) != 4 * H || w_ih.extent(2) != D)
    throw std::invalid_argument("lstm_bidirectional: w_ih shape mismatch");
  if (w_hh.extent(0) != 2 || w_hh.extent(1) != 4 * H || w_hh.extent(2) != H)
    throw std::invalid_argument("lstm_bidirectional: w_hh shape mismatch");
  if (b_ih.extent(0) != 2 || b_ih.extent(1) != 4 * H || b_hh.extent(0) != 2 ||
      b_hh.extent(1) != 4 * H)
    throw std::invalid_argument("lstm_bidirectional: bias shape mismatch");
  if (B == 0 || T == 0 || D == 0 || H == 0) return;
  // Fuse b_ih + b_hh into one [2, 4H] bias.
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> bia("lstm::bia", 2,
                                                            4 * H);
  Kokkos::parallel_for(
      label + "_bias", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                           {size_t(0), size_t(0)}, {size_t(2), 4 * H}),
      KOKKOS_LAMBDA(size_t d, size_t k) {
        bia(d, k) = b_ih(d, k) + b_hh(d, k);
      });
  // Widen I/O to FP32 (float aliases zero-copy via unmanaged views).
  using V3f = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;
  V3f xf("lstm::xf", B, T, D);
  V3f yf("lstm::yf", B, T, 2 * H);
  Kokkos::parallel_for(
      label + "_cast_in",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, D}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t d) {
        xf(b, t, d) = static_cast<float>(x(b, t, d));
      });
  Kokkos::deep_copy(yf, 0.0f);
  // Team-lane scratch: gates [2B,4H], states [2B,2H] (zero init).
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> gates2("lstm::gates",
                                                               2 * B, 4 * H);
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> hc("lstm::hc", 2 * B,
                                                           2 * H);
  Kokkos::deep_copy(hc, 0.0f);
  Impl::LstmBidirStepFunctor<ExecSpace> fn{xf,  lengths, w_ih, w_hh, bia,
                                          gates2, hc,   yf,  B,   T,
                                          D,      H};
  Kokkos::parallel_for(label, Kokkos::TeamPolicy<ExecSpace>(B * 2, Kokkos::AUTO),
                       fn);
  Kokkos::parallel_for(
      label + "_cast_out",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                       {B, T, 2 * H}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t h) {
        y(b, t, h) = static_cast<Scalar>(yf(b, t, h));
      });
}

// Rank-2 convenience: single batch [T, D] -> [T, 2H], lengths [1].
// Explicit copies (no unmanaged reinterpret casts over const data).
template <class ExecSpace, class Scalar>
void lstm_bidirectional(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x_in,
    const Kokkos::View<int*, ExecSpace>& lengths,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& w_ih,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& w_hh,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& b_ih,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& b_hh,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y_in,
    const std::string& label = "KokkosDNN::lstm_bidirectional") {
  const size_t T = x_in.extent(0);
  if (lengths.extent(0) != 1)
    throw std::invalid_argument("lstm_bidirectional: rank-2 needs 1 length");
  using V3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  V3 x3("lstm::x3", 1, x_in.extent(0), x_in.extent(1));
  V3 y3("lstm::y3", 1, y_in.extent(0), y_in.extent(1));
  Kokkos::parallel_for(
      label + "_lift",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
          {size_t(0), size_t(0)}, {x_in.extent(0), x_in.extent(1)}),
      KOKKOS_LAMBDA(size_t t, size_t d) { x3(0, t, d) = x_in(t, d); });
  lstm_bidirectional<ExecSpace, Scalar>(x3, lengths, w_ih, w_hh, b_ih, b_hh,
                                       y3, label);
  Kokkos::parallel_for(
      label + "_drop",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
          {size_t(0), size_t(0)}, {y_in.extent(0), y_in.extent(1)}),
      KOKKOS_LAMBDA(size_t t, size_t h) { y_in(t, h) = y3(0, t, h); });
  (void)T;
}

}  // namespace KokkosDNN

#endif
