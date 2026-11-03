// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Tests for the shared session/prefix-cache vocabulary (models/common)
// and its per-arch aliases.
#include <gtest/gtest.h>

#include <type_traits>

#include "KokkosDNN_PrefixCacheCore.hpp"
#include "KokkosDNN_SessionTypes.hpp"

namespace {

TEST(SessionTypes, Defaults) {
  KokkosDNN::Common::SessionOptions o;
  EXPECT_TRUE(o.stop_ids.empty());
  EXPECT_TRUE(o.stop_strings.empty());
  EXPECT_EQ(o.max_new, 256);
  EXPECT_FALSE(o.seedProvided);
  KokkosDNN::Common::StepOutcome s;
  EXPECT_EQ(s.id, -1);
  EXPECT_TRUE(s.delta.empty());
  EXPECT_FALSE(s.stop_hit);
  EXPECT_FALSE(s.finished);
  EXPECT_EQ(s.finish, KokkosDNN::Common::FinishReason::Stop);
  KokkosDNN::Common::CachedPrefillOutcome c;
  EXPECT_FALSE(c.hit);
  EXPECT_EQ(c.reused, 0u);
}

TEST(PrefixCacheCore, FindPromote) {
  KokkosDNN::Common::PrefixCacheCore core(2);
  EXPECT_TRUE(core.empty());
  size_t n = 0;
  EXPECT_EQ(core.find({1, 2, 3}, 3, n), -1);
  EXPECT_EQ(n, 0u);
}

// The per-arch aliases must stay identical to the shared definitions.
#ifdef KOKKOSDNN_TEST_HAVE_LFM25
static_assert(std::is_same_v<KokkosDNN::LFM25::SessionOptions,
                             KokkosDNN::Common::SessionOptions>);
static_assert(std::is_same_v<KokkosDNN::LFM25::StepOutcome,
                             KokkosDNN::Common::StepOutcome>);
static_assert(std::is_same_v<KokkosDNN::LFM25::PrefixCacheCore,
                             KokkosDNN::Common::PrefixCacheCore>);
#endif
#ifdef KOKKOSDNN_TEST_HAVE_GLIMMER
static_assert(std::is_same_v<KokkosDNN::Glimmer::SessionOptions,
                             KokkosDNN::Common::SessionOptions>);
static_assert(std::is_same_v<KokkosDNN::Glimmer::StepOutcome,
                             KokkosDNN::Common::StepOutcome>);
#endif

}  // namespace
