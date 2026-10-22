// Tests for Common::Usage / Common::FinishReason / to_string().
#include <gtest/gtest.h>

#include "KokkosDNN_Usage.hpp"

namespace {

using KokkosDNN::Common::FinishReason;
using KokkosDNN::Common::Usage;
using KokkosDNN::Common::to_string;

TEST(Usage, Total) {
  Usage u;
  EXPECT_EQ(u.total(), 0u);
  u.prompt_tokens = 7;
  u.completion_tokens = 13;
  EXPECT_EQ(u.total(), 20u);
}

TEST(Usage, FinishReasonStrings) {
  EXPECT_STREQ(to_string(FinishReason::Stop), "stop");
  EXPECT_STREQ(to_string(FinishReason::Length), "length");
  EXPECT_STREQ(to_string(FinishReason::Error), "error");
}
}  // namespace
