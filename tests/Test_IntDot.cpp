// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Impl::IntDot: exactness of the portable fallback and of every backend
// specialization against a host-computed reference, including signed extremes
// (-128/127) where signedness handling matters.
#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "KokkosDNN_IntDot.hpp"

#include <gtest/gtest.h>

using Exec = Kokkos::DefaultExecutionSpace;

namespace {

// Device launches live in free functions: extended lambdas cannot appear
// directly inside TEST() bodies (private TestBody context).
void run_dot4(const Kokkos::View<int32_t*, Exec>& da,
              const Kokkos::View<int32_t*, Exec>& db,
              const Kokkos::View<int32_t*, Exec>& dacc,
              const Kokkos::View<int32_t*, Exec>& dout) {
  using Dot = KokkosDNN::Impl::IntDot<Exec>;
  Kokkos::parallel_for(
      "intdot_check", Kokkos::RangePolicy<Exec>(0, (int)da.extent(0)),
      KOKKOS_LAMBDA(int i) { dout(i) = Dot::dot4(da(i), db(i), dacc(i)); });
  Kokkos::fence();
}

void run_acc(const Kokkos::View<int, Exec>& out) {
  using Dot = KokkosDNN::Impl::IntDot<Exec>;
  Kokkos::parallel_for(
      "intdot_acc", Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
        // 0x01010101 . 0x02020202 = 4 * (1*2) = 8, plus incoming 100.
        out() = Dot::dot4(0x01010101, 0x02020202, 100);
      });
  Kokkos::fence();
}

void run_load(const Kokkos::View<int32_t*, Exec>& d,
              const Kokkos::View<int32_t*, Exec>& out) {
  using Dot = KokkosDNN::Impl::IntDot<Exec>;
  Kokkos::parallel_for(
      "intdot_load", Kokkos::RangePolicy<Exec>(0, (int)out.extent(0)),
      KOKKOS_LAMBDA(int i) { out(i) = Dot::load(&d(i)); });
  Kokkos::fence();
}

void run_load_u(const Kokkos::View<uint8_t*, Exec>& d,
                const Kokkos::View<int32_t*, Exec>& out) {
  using Dot = KokkosDNN::Impl::IntDot<Exec>;
  Kokkos::parallel_for(
      "intdot_load_u", Kokkos::RangePolicy<Exec>(0, (int)out.extent(0)),
      KOKKOS_LAMBDA(int i) { out(i) = Dot::load_u(d.data() + i); });
  Kokkos::fence();
}

void run_load_u16(const Kokkos::View<uint8_t*, Exec>& d,
                  const Kokkos::View<int32_t*, Exec>& out) {
  using Dot = KokkosDNN::Impl::IntDot<Exec>;
  Kokkos::parallel_for(
      "intdot_load_u16", Kokkos::RangePolicy<Exec>(0, (int)out.extent(0)),
      KOKKOS_LAMBDA(int i) { out(i) = Dot::load_u16(d.data() + 2 * i); });
  Kokkos::fence();
}

TEST(IntDot, Dot4MatchesScalarReference) {
  constexpr int nwords = 4096;
  std::mt19937 rng(0x1D07u);
  std::vector<int32_t> ha(nwords), hb(nwords);
  std::vector<int32_t> hacc(nwords), href(nwords);
  for (int i = 0; i < nwords; ++i) {
    uint32_t wa = 0, wb = 0;
    const int32_t acc = (int32_t)(rng() & 0xFFFFu) - 32768;
    int32_t ref = acc;
    for (int l = 0; l < 4; ++l) {
      // Bias toward signed extremes so the signed-byte path is exercised.
      const int menu[6] = {-128, -1, 0, 1, 127, (int)(rng() & 0xFFu) - 128};
      const int8_t a = (int8_t)menu[rng() % 6];
      const int8_t b = (int8_t)menu[rng() % 6];
      wa |= (uint32_t)(uint8_t)a << (l * 8);
      wb |= (uint32_t)(uint8_t)b << (l * 8);
      ref += (int32_t)a * (int32_t)b;
    }
    ha[i] = (int32_t)wa;
    hb[i] = (int32_t)wb;
    hacc[i] = acc;
    href[i] = ref;
  }
  Kokkos::View<int32_t*, Exec> da("da", nwords), db("db", nwords),
      dacc("dacc", nwords), dout("dout", nwords);
  auto mha = Kokkos::create_mirror_view(da);
  auto mhb = Kokkos::create_mirror_view(db);
  auto mhacc = Kokkos::create_mirror_view(dacc);
  for (int i = 0; i < nwords; ++i) {
    mha(i) = ha[i];
    mhb(i) = hb[i];
    mhacc(i) = hacc[i];
  }
  Kokkos::deep_copy(da, mha);
  Kokkos::deep_copy(db, mhb);
  Kokkos::deep_copy(dacc, mhacc);
  run_dot4(da, db, dacc, dout);
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dout);
  for (int i = 0; i < nwords; ++i) EXPECT_EQ(got(i), href[i]) << "word " << i;
}

TEST(IntDot, Dot4AccumulatesIncomingAcc) {
  Kokkos::View<int, Exec> out("out");
  run_acc(out);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  EXPECT_EQ(h(), 108);
}

TEST(IntDot, LoadRoundTripsAlignedWords) {
  // `load` needs 4-byte alignment: an int32 view's data satisfies the
  // contract by construction.
  constexpr int n = 64;
  Kokkos::View<int32_t*, Exec> d("d", n);
  auto mh = Kokkos::create_mirror_view(d);
  std::mt19937 rng(0xBEEFu);
  for (int i = 0; i < n; ++i) mh(i) = (int32_t)rng();
  Kokkos::deep_copy(d, mh);
  Kokkos::View<int32_t*, Exec> out("out", n);
  run_load(d, out);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) EXPECT_EQ(h(i), mh(i)) << "word " << i;
}

TEST(IntDot, LoadURoundTripsUnalignedBytes) {
  // `load_u` is safe at any alignment: 7-byte payload, word loads at offsets
  // 0..3 all overlap and the last three are misaligned.
  const uint8_t bytes[7] = {0x00, 0xFF, 0x80, 0x7F, 0x01, 0x80, 0xFF};
  Kokkos::View<uint8_t*, Exec> d("d", 7);
  auto mh = Kokkos::create_mirror_view(d);
  for (int i = 0; i < 7; ++i) mh(i) = bytes[i];
  Kokkos::deep_copy(d, mh);
  Kokkos::View<int32_t*, Exec> out("out", 4);
  run_load_u(d, out);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < 4; ++i) {
    int32_t want;
    std::memcpy(&want, bytes + i, 4);
    EXPECT_EQ(h(i), want) << "offset " << i;
  }
}

TEST(IntDot, LoadU16RoundTripsTwoByteAlignedWords) {
  // `load_u16` requires only 2-byte alignment (Q6_K/Q8_0 block payloads).
  const uint8_t bytes[8] = {0x00, 0xFF, 0x80, 0x7F,
                            0x01, 0x80, 0xFF, 0x3C};
  Kokkos::View<uint8_t*, Exec> d("d", 8);
  auto mh = Kokkos::create_mirror_view(d);
  for (int i = 0; i < 8; ++i) mh(i) = bytes[i];
  Kokkos::deep_copy(d, mh);
  Kokkos::View<int32_t*, Exec> out("out", 3);
  run_load_u16(d, out);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < 3; ++i) {
    int32_t want;
    std::memcpy(&want, bytes + 2 * i, 4);
    EXPECT_EQ(h(i), want) << "offset " << 2 * i;
  }
}

}  // namespace
