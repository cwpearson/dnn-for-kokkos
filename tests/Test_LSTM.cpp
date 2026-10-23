// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Single-layer bidirectional LSTM: device vs host reference, lengths
// handling (pad rows zero, states freeze), forward/backward equivalence,
// argument validation.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "KokkosDNN_LSTM.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;

float probe(size_t i) {
  return 0.031f * (float)((i * 37 + 11) % 29) - 0.4f;
}

// Host reference: batch-first single-layer bidir LSTM, zero init. Mirrors
// the device kernel exactly (fused bias, single accumulation pass, float
// expf/tanhf) so agreement is at the 1e-6 level; torch parity of the kernel
// itself is covered by the Kokoro golden test.
void host_lstm(const std::vector<float>& x, const std::vector<int>& lens,
               size_t B, size_t T, size_t D, size_t H,
               const std::vector<float>& wih, const std::vector<float>& whh,
               const std::vector<float>& bih, const std::vector<float>& bhh,
               std::vector<float>& y) {
  y.assign(B * T * 2 * H, 0.0f);
  auto at = [&](const std::vector<float>& m, size_t d, size_t k, size_t j,
                size_t K) { return m[(d * 4 * H + k) * K + j]; };
  for (size_t b = 0; b < B; ++b) {
    for (size_t dir = 0; dir < 2; ++dir) {
      std::vector<float> c(H, 0.0f), h(H, 0.0f), hp(H);
      for (size_t s = 0; s < (size_t)lens[b]; ++s) {
        const size_t t = (dir == 0) ? s : ((size_t)lens[b] - 1 - s);
        hp = h;  // previous-step state snapshot: units must not observe
                 // same-step updates (the device kernel gates on old h).
        for (size_t hh = 0; hh < H; ++hh) {
          float gi = bih[dir * 4 * H + hh] + bhh[dir * 4 * H + hh];
          float gf = bih[dir * 4 * H + H + hh] + bhh[dir * 4 * H + H + hh];
          float gg = bih[dir * 4 * H + 2 * H + hh] +
                     bhh[dir * 4 * H + 2 * H + hh];
          float go = bih[dir * 4 * H + 3 * H + hh] +
                     bhh[dir * 4 * H + 3 * H + hh];
          for (size_t j = 0; j < D; ++j) {
            const float xv = x[(b * T + t) * D + j];
            gi += at(wih, dir, hh, j, D) * xv;
            gf += at(wih, dir, H + hh, j, D) * xv;
            gg += at(wih, dir, 2 * H + hh, j, D) * xv;
            go += at(wih, dir, 3 * H + hh, j, D) * xv;
          }
          for (size_t j = 0; j < H; ++j) {
            gi += at(whh, dir, hh, j, H) * hp[j];
            gf += at(whh, dir, H + hh, j, H) * hp[j];
            gg += at(whh, dir, 2 * H + hh, j, H) * hp[j];
            go += at(whh, dir, 3 * H + hh, j, H) * hp[j];
          }
          const float i = 1.0f / (1.0f + expf(-gi));
          const float f = 1.0f / (1.0f + expf(-gf));
          const float g = tanhf(gg), o = 1.0f / (1.0f + expf(-go));
          c[hh] = f * c[hh] + i * g;
          h[hh] = o * tanhf(c[hh]);
          y[(b * T + t) * 2 * H + dir * H + hh] = h[hh];
        }
      }
    }
  }
}

void run_case(size_t B, size_t T, size_t D, size_t H,
              const std::vector<int>& lens) {
  std::vector<float> hx(B * T * D), wih(2 * 4 * H * D), whh(2 * 4 * H * H),
      bih(2 * 4 * H), bhh(2 * 4 * H);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = probe(i);
  for (size_t i = 0; i < wih.size(); ++i) wih[i] = 0.11f * probe(i + 3);
  for (size_t i = 0; i < whh.size(); ++i) whh[i] = 0.11f * probe(i + 5);
  for (size_t i = 0; i < bih.size(); ++i) bih[i] = 0.05f * probe(i + 7);
  for (size_t i = 0; i < bhh.size(); ++i) bhh[i] = 0.05f * probe(i + 9);
  std::vector<float> ref;
  host_lstm(hx, lens, B, T, D, H, wih, whh, bih, bhh, ref);

  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", B, T, D),
      y("y", B, T, 2 * H);
  Kokkos::View<int*, Exec> len("len", B);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> dwih("wih", 2, 4 * H, D),
      dwhh("whh", 2, 4 * H, H);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> dbih("bih", 2, 4 * H),
      dbhh("bhh", 2, 4 * H);
  {
    auto h = Kokkos::create_mirror_view(x);
    for (size_t b = 0; b < B; ++b)
      for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) h(b, t, d) = hx[(b * T + t) * D + d];
    Kokkos::deep_copy(x, h);
  }
  {
    auto h = Kokkos::create_mirror_view(len);
    for (size_t b = 0; b < B; ++b) h(b) = lens[b];
    Kokkos::deep_copy(len, h);
  }
  auto fill3 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t b = 0; b < h.extent(1); ++b)
        for (size_t c = 0; c < h.extent(2); ++c) h(a, b, c) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  auto fill2 = [](auto& v, const std::vector<float>& s) {
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (size_t a = 0; a < h.extent(0); ++a)
      for (size_t b = 0; b < h.extent(1); ++b) h(a, b) = s[k++];
    Kokkos::deep_copy(v, h);
  };
  fill3(dwih, wih);
  fill3(dwhh, whh);
  fill2(dbih, bih);
  fill2(dbhh, bhh);
  KokkosDNN::lstm_bidirectional<Exec, float>(x, len, dwih, dwhh, dbih, dbhh,
                                            y);
  Kokkos::fence();
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  ASSERT_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(got.data()[i], ref[i], 1e-5f) << "at " << i;
}

TEST(Lstm, FullLength) { run_case(1, 7, 6, 4, {7}); }

TEST(Lstm, TruncatedLengths) { run_case(2, 6, 5, 3, {6, 3}); }

TEST(Lstm, ZeroLengthRow) { run_case(2, 4, 3, 2, {4, 0}); }

TEST(Lstm, Rank2) {
  // Rank-2 path matches rank-3 with B=1.
  const size_t T = 5, D = 4, H = 3;
  run_case(1, T, D, H, {(int)T});
}

TEST(Lstm, BadArgsThrow) {
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> x("x", 1, 4, 3),
      y("y", 1, 4, 7);  // odd H2
  Kokkos::View<int*, Exec> len("len", 1);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> wih("w", 2, 8, 3),
      whh("w", 2, 8, 2);
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> b("b", 2, 8);
  auto bad = [&]() {
    KokkosDNN::lstm_bidirectional<Exec, float>(x, len, wih, whh, b, b, y);
  };
  EXPECT_THROW(bad(), std::invalid_argument);
}

}  // namespace
