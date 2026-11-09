// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Tests for Common::Cli::try_parse_shared / resolve_random_seed
// (models/common). Untouched fields are preset to sentinel values so the
// tests confirm "parsed what was given, preserved the rest" without
// freezing the struct's defaults.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "KokkosDNN_CliOptions.hpp"

namespace {

using KokkosDNN::Common::Cli::ParseAllow;
using KokkosDNN::Common::Cli::TextGenOptions;
using KokkosDNN::Common::Cli::resolve_random_seed;
using KokkosDNN::Common::Cli::try_parse_shared;

// Mimic a driver argv loop over synthetic args (args[0] is the program
// name, parsing starts at 1), optionally into preset options.
TextGenOptions run_parse(const std::vector<std::string>& args,
                         ParseAllow allow = ParseAllow{},
                         TextGenOptions opts = TextGenOptions{}) {
  std::vector<std::string> storage = args;  // mutable token storage
  std::vector<char*> argv;
  argv.reserve(storage.size());
  for (auto& s : storage) argv.push_back(s.data());
  const int argc = static_cast<int>(argv.size());
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (try_parse_shared(a, argv.data(), i, argc, opts, allow)) continue;
  }
  return opts;
}

TEST(CliOptions, FullSweep) {
  const auto o = run_parse(
      {"prog", "--weights", "a", "b", "--tokenizer", "t.json", "--prompt",
       "hi", "--max-new", "7", "--stop", "1", "2", "--seed", "5", "--tiny",
       "--add-bos", "--stop-string", "x", "y z", "--prefill-chunk", "3",
       "--config", "c.json", "--max-seq-len", "11", "--fp32", "--temperature",
       "0.5", "--top-k", "4", "--top-p", "0.8", "--min-p", "0.1",
       "--repetition-penalty", "1.2", "--seed", "42", "--chat", "--system",
       "sys", "--role", "r", "--messages", "m"});
  EXPECT_EQ(o.weights, (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(o.tokenizer_path, "t.json");
  EXPECT_EQ(o.prompt, "m");  // --messages overwrites --prompt, implies chat
  EXPECT_EQ(o.max_new, 7);
  // --stop ends at the next flag (its value is parsed, not swallowed).
  EXPECT_EQ(o.extra_stops, (std::vector<int>{1, 2}));
  EXPECT_EQ(o.stop_strings, (std::vector<std::string>{"x", "y z"}));
  EXPECT_EQ(o.prefill_chunk, 3);
  EXPECT_EQ(o.config_path, "c.json");
  EXPECT_EQ(o.max_seq_len, 11);
  EXPECT_TRUE(o.fp32);
  EXPECT_TRUE(o.tiny);
  EXPECT_TRUE(o.tiny_explicit);
  EXPECT_TRUE(o.want_bos);
  EXPECT_FLOAT_EQ(o.samp.temperature, 0.5f);
  EXPECT_EQ(o.samp.top_k, 4);
  EXPECT_FLOAT_EQ(o.samp.top_p, 0.8f);
  EXPECT_FLOAT_EQ(o.samp.min_p, 0.1f);
  EXPECT_FLOAT_EQ(o.samp.repetition_penalty, 1.2f);
  EXPECT_EQ(o.samp.seed, 42u);  // later --seed wins over the earlier one
  EXPECT_TRUE(o.seed_given);
  EXPECT_TRUE(o.chat_mode);
  EXPECT_EQ(o.system_prompt, "sys");
  EXPECT_EQ(o.role, "r");
}

TEST(CliOptions, WeightsWithoutFilesStillOptsOutOfTiny) {
  // Historical quirk: the branch is taken whenever a token follows
  // --weights, even when no file token does. Preserve it.
  const auto o = run_parse({"prog", "--weights", "--fp32"});
  EXPECT_TRUE(o.weights.empty());
  EXPECT_FALSE(o.tiny);
  EXPECT_TRUE(o.fp32);
}

TEST(CliOptions, ParseAllowGatesChatAndMessages) {
  TextGenOptions sentinel;
  sentinel.prompt = "SENTINEL";
  const auto o = run_parse({"prog", "--chat", "--system", "x", "--messages",
                            "z"},
                           ParseAllow{false, false}, sentinel);
  EXPECT_FALSE(o.chat_mode);
  EXPECT_TRUE(o.system_prompt.empty());
  EXPECT_EQ(o.prompt, "SENTINEL");

  const auto chat_only = run_parse({"prog", "--chat", "--messages", "z"},
                                   ParseAllow{true, false}, sentinel);
  EXPECT_TRUE(chat_only.chat_mode);
  EXPECT_EQ(chat_only.prompt, "SENTINEL");
}

TEST(CliOptions, UnknownFlagAndMissingValueAreLeftAlone) {
  TextGenOptions opts;
  opts.prompt = "SENTINEL";
  std::vector<std::string> storage = {"prog", "--bogus", "--prompt"};
  std::vector<char*> argv;
  for (auto& s : storage) argv.push_back(s.data());
  const int argc = static_cast<int>(argv.size());
  int i = 1;  // unknown flags are left for the driver's own branches
  EXPECT_FALSE(
      try_parse_shared(argv[i], argv.data(), i, argc, opts, ParseAllow{}));
  EXPECT_EQ(i, 1);
  // A trailing flag with no value is not consumed either.
  i = 2;
  EXPECT_FALSE(
      try_parse_shared(argv[i], argv.data(), i, argc, opts, ParseAllow{}));
  EXPECT_EQ(i, 2);
  EXPECT_EQ(opts.prompt, "SENTINEL");
}

TEST(CliOptions, ResolveRandomSeed) {
  // An explicit --seed is preserved exactly.
  TextGenOptions given = run_parse({"prog", "--seed", "1234"});
  resolve_random_seed(given);
  EXPECT_EQ(given.samp.seed, 1234u);
  EXPECT_TRUE(given.seed_given);
  // Without --seed the call fills something in and stays unmarked.
  TextGenOptions fresh;
  resolve_random_seed(fresh);
  EXPECT_FALSE(fresh.seed_given);
}

}  // namespace
