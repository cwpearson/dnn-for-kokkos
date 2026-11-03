// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Tests for Common::generate_impl / normalize_stop_ids (models/common).
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "KokkosDNN_Generate.hpp"

namespace {

// Minimal DecoderModel: forward() returns fixed logits whose argmax is
// next_id, advances seq_len, and records forward call sizes.
struct FakeModel {
  using ViewInt = Kokkos::View<int*, Kokkos::DefaultExecutionSpace>;
  struct FakeCfg {
    int max_seq_len = 32;
  };
  FakeCfg cfg;
  size_t seq_len = 0;
  int next_id = 3;
  std::vector<size_t> calls;

  void reset() { seq_len = 0; }

  Kokkos::View<float*, Kokkos::DefaultExecutionSpace> forward(
      const ViewInt& toks) {
    const size_t t = toks.extent(0);
    calls.push_back(t);
    seq_len += t;
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> lg("lg", 8);
    auto h = Kokkos::create_mirror_view(lg);
    for (size_t i = 0; i < 8; ++i) h(i) = -1.0f;
    h(static_cast<size_t>(next_id)) = 10.0f;
    Kokkos::deep_copy(lg, h);
    return lg;
  }
};

TEST(Generate, NormalizeStopIds) {
  EXPECT_EQ(KokkosDNN::Common::normalize_stop_ids({3, 1, 3, 2}),
            (std::vector<int>{1, 2, 3}));
  EXPECT_TRUE(KokkosDNN::Common::normalize_stop_ids({}).empty());
}

TEST(Generate, PrefillDecodeLoop) {
  FakeModel m;
  m.cfg.max_seq_len = 32;
  m.next_id = 3;
  const std::vector<int> prompt = {1, 2};
  int cb_count = 0;
  auto out = KokkosDNN::Common::generate_impl(
      m, prompt, 4, std::vector<int>{9},
      KokkosDNN::SampleOptions{}, nullptr,
      [&](int) { ++cb_count; });
  // prompt (2) + 4 generated.
  EXPECT_EQ(out.size(), 6u);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 2);
  for (size_t i = 2; i < out.size(); ++i) EXPECT_EQ(out[i], 3);
  EXPECT_EQ(cb_count, 4);
  // One prefill (T=2) + three single-token decodes: the 4th new token is
  // sampled from the 3rd decode forward, so 4 forwards cover 5 tokens.
  ASSERT_EQ(m.calls.size(), 4u);
  EXPECT_EQ(m.calls[0], 2u);
  EXPECT_EQ(m.seq_len, 5u);
}

TEST(Generate, StopIdBreaksBeforeForward) {
  FakeModel m;
  m.cfg.max_seq_len = 32;
  m.next_id = 5;
  // 5 is a stop id: prefill samples it, the loop breaks without another
  // forward, and the stop token is included.
  auto out = KokkosDNN::Common::generate_impl(
      m, {1}, 4, std::vector<int>{5}, KokkosDNN::SampleOptions{});
  EXPECT_EQ(out, (std::vector<int>{1, 5}));
  ASSERT_EQ(m.calls.size(), 1u);
}

TEST(Generate, MaxSeqLenCap) {
  FakeModel m;
  m.cfg.max_seq_len = 4;
  m.next_id = 3;
  auto out = KokkosDNN::Common::generate_impl(
      m, {1, 2}, 10, std::vector<int>{}, KokkosDNN::SampleOptions{});
  EXPECT_LE((int)out.size(), 4);
}

}  // namespace
