// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Qwen byte-level BPE tokenizer tests.
//
// Two layers:
//   (1) data-free unit tests of the Impl pieces (NFC, strip, pre-tokenizer
//       spans, byte map) — always run;
//   (2) golden tests of apply_chat_template/encode against transformers 5.12.1
//       outputs (see /tmp/opencode/gen_goldens.py). These need the real
//       tokenizer.json via KOKKOSDNN_QWEN_TOKENIZER_JSON and GTEST_SKIP()
//       otherwise, so the suite stays weight-free.
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "KokkosDNN_QwenTokenizer.hpp"

namespace {

namespace QT = KokkosDNN::QwenTokenizer;
namespace QI = KokkosDNN::QwenTokenizer::Impl;

std::vector<uint32_t> cp(const std::string& s) {
  return QI::utf8_decode(s);
}
std::string str(const std::vector<uint32_t>& cps) {
  return QI::utf8_encode_str(cps);
}

TEST(QwenNfc, CanonicalCompose) {
  std::vector<uint32_t> s = cp("cafe\xCC\x81");  // e + combining acute
  QI::nfc_inplace(s);
  EXPECT_EQ(str(s), "caf\xC3\xA9");  // precomposed é
  // Already-composed input is a fixed point.
  std::vector<uint32_t> t = cp("caf\xC3\xA9");
  QI::nfc_inplace(t);
  EXPECT_EQ(str(t), "caf\xC3\xA9");
}

TEST(QwenNfc, CompatibilityUnchanged) {
  // Compatibility decompositions are NOT applied by NFC.
  std::vector<uint32_t> s = cp("\xEF\xAC\x81");  // U+FB01 LATIN SMALL LIGATURE FI
  QI::nfc_inplace(s);
  EXPECT_EQ(str(s), "\xEF\xAC\x81");
  std::vector<uint32_t> n = cp("\xE2\x91\xA0");  // U+2460 CIRCLED DIGIT ONE
  QI::nfc_inplace(n);
  EXPECT_EQ(str(n), "\xE2\x91\xA0");
}

TEST(QwenNfc, ReorderAndBlock) {
  // U+0301 (ccc 230) before U+0327 (ccc 202) must reorder to 202,230; then
  // U+0301 composes with 'a' across U+0327 (202 < 230: not blocked).
  std::vector<uint32_t> s = cp("a\xCC\x81\xCC\xA7");
  QI::nfc_inplace(s);
  EXPECT_EQ(str(s), "\xC3\xA1\xCC\xA7");  // U+00E1 U+0327
}

TEST(QwenNfc, HangulRoundtrip) {
  // U+AC00 decomposes to Jamo L+V and recomposes to the syllable.
  std::vector<uint32_t> s = cp("\xEA\xB0\x80");
  QI::nfc_inplace(s);
  EXPECT_EQ(str(s), "\xEA\xB0\x80");
  // Jamo L+V compose to the LV syllable.
  std::vector<uint32_t> j = cp("\xE1\x84\x80\xE1\x85\xA1");
  QI::nfc_inplace(j);
  EXPECT_EQ(str(j), "\xEA\xB0\x80");
}

TEST(QwenStrip, PythonSemantics) {
  EXPECT_EQ(str(QI::strip_ws(cp("  padded with spaces   "))),
            "padded with spaces");
  EXPECT_EQ(str(QI::strip_ws(cp("\t\n x \r\n"))), "x");
  EXPECT_EQ(str(QI::strip_ws(cp(""))), "");
  EXPECT_EQ(str(QI::strip_ws(cp("   "))), "");
}

TEST(QwenPretokenize, Spans) {
  auto spans = [](const std::string& s) {
    std::vector<std::string> out;
    for (const auto& pre : QI::pretokenize(cp(s))) out.push_back(str(pre));
    return out;
  };
  // "hello  world": double space splits as ["hello", " ", " world"]
  // (alt-6 backtrack: all but the last space, then space-prefixed word).
  EXPECT_EQ(spans("hello  world"),
            (std::vector<std::string>{"hello", " ", " world"}));
  // Trailing-space run at end of input stays whole.
  EXPECT_EQ(spans("x  "), (std::vector<std::string>{"x", "  "}));
  // Newline run: alt-5 spans to the last newline (greedy backtracking).
  EXPECT_EQ(spans(" \n "), (std::vector<std::string>{" \n", " "}));
  // Contraction prefers the earliest alternative.
  EXPECT_EQ(spans("it's"), (std::vector<std::string>{"it", "'s"}));
  EXPECT_EQ(spans("a  b"), (std::vector<std::string>{"a", " ", " b"}));
}

TEST(QwenByteMap, Gpt2Values) {
  EXPECT_EQ(QI::byte_to_unicode(0x20), 0x120u);  // ' ' -> U+0120
  EXPECT_EQ(QI::byte_to_unicode(0x41), 0x41u);   // 'A' maps to itself
  EXPECT_EQ(QI::byte_to_unicode(0x00), 0x100u);
  EXPECT_EQ(QI::byte_to_unicode(0xFF), 0xFFu);  // 0xAE-0xFF map to themselves
}

// ---- golden tests (need tokenizer.json) ------------------------------------
const char* tokenizer_json_path() {
  return std::getenv("KOKKOSDNN_QWEN_TOKENIZER_JSON");
}

struct Golden {
  const char* prompt;
  bool thinking;
  const char* formatted;
  std::vector<int> ids;
};

std::vector<Golden> goldens() {
  return {
      {"a red panda reading a newspaper", true,
       "<|im_start|>user\n"
       "a red panda reading a newspaper<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 64, 2438, 85183, 5122, 264, 16311, 248046, 198,
        248045, 74455, 198, 248068, 198}},
      {"a red panda reading a newspaper", false,
       "<|im_start|>user\n"
       "a red panda reading a newspaper<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n\n</think>\n\n",
       {248045, 846, 198, 64, 2438, 85183, 5122, 264, 16311, 248046, 198,
        248045, 74455, 198, 248068, 271, 248069, 271}},
      {"it's a lovely day", true,
       "<|im_start|>user\nit's a lovely day<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 275, 579, 264, 16200, 1834, 248046, 198, 248045,
        74455, 198, 248068, 198}},
      // Precomposed é (NFC of the next case must match it exactly).
      {"caf\xC3\xA9", true,
       "<|im_start|>user\ncaf\xC3\xA9<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 895, 56868, 248046, 198, 248045, 74455, 198, 248068,
        198}},
      {"", true,
       "<|im_start|>user\n<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198}},
      {"  padded with spaces   ", true,
       "<|im_start|>user\npadded with spaces<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 79, 16336, 440, 12258, 248046, 198, 248045, 74455,
        198, 248068, 198}},
      // Emoji exercises the byte-level fallback (no emoji in vocab).
      {"\xF0\x9F\x90\xBC red panda", true,
       "<|im_start|>user\n\xF0\x9F\x90\xBC red panda<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 9008, 238, 120, 2438, 85183, 248046, 198, 248045,
        74455, 198, 248068, 198}},
      {"hello  world", true,
       "<|im_start|>user\nhello  world<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 14556, 220, 1814, 248046, 198, 248045, 74455, 198,
        248068, 198}},
      {"na\xC3\xAFve fa\xC3\xA7"
       "ade",
       true,
       "<|im_start|>user\nna\xC3\xAFve fa\xC3\xA7"
       "ade<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 3267, 36328, 571, 218894, 248046, 198, 248045, 74455,
        198, 248068, 198}},
      {"mix: a  b\nc   d", true,
       "<|im_start|>user\nmix: a  b\nc   d<|im_end|>\n"
       "<|im_start|>assistant\n"
       "<think>\n",
       {248045, 846, 198, 33911, 25, 264, 220, 292, 198, 66, 256, 293, 248046,
        198, 248045, 74455, 198, 248068, 198}},
  };
}

TEST(QwenTemplate, GoldenFormat) {
  const char* p = tokenizer_json_path();
  if (!p) GTEST_SKIP() << "set KOKKOSDNN_QWEN_TOKENIZER_JSON for goldens";
  QT::Tokenizer tok;
  tok.load(p);
  for (const auto& g : goldens())
    EXPECT_EQ(tok.apply_chat_template(g.prompt, g.thinking), g.formatted)
        << "prompt=" << g.prompt;
}

TEST(QwenEncode, GoldenIds) {
  const char* p = tokenizer_json_path();
  if (!p) GTEST_SKIP() << "set KOKKOSDNN_QWEN_TOKENIZER_JSON for goldens";
  QT::Tokenizer tok;
  tok.load(p);
  EXPECT_EQ(tok.vocab_size(), 248077u);
  for (const auto& g : goldens())
    EXPECT_EQ(tok.encode(g.prompt, g.thinking, 512), g.ids)
        << "prompt=" << g.prompt;
}

TEST(QwenEncode, NfcEquality) {
  const char* p = tokenizer_json_path();
  if (!p) GTEST_SKIP() << "set KOKKOSDNN_QWEN_TOKENIZER_JSON for goldens";
  QT::Tokenizer tok;
  tok.load(p);
  // Decomposed e + combining acute must tokenize like precomposed é.
  EXPECT_EQ(tok.encode("cafe\xCC\x81", true, 512),
            tok.encode("caf\xC3\xA9", true, 512));
}

TEST(QwenEncode, Truncation) {
  const char* p = tokenizer_json_path();
  if (!p) GTEST_SKIP() << "set KOKKOSDNN_QWEN_TOKENIZER_JSON for goldens";
  QT::Tokenizer tok;
  tok.load(p);
  const std::string prompt(2000, 'a');
  const std::vector<int> ids = tok.encode(prompt, true, 32);
  ASSERT_EQ(ids.size(), 32u);
  // Right truncation keeps the head: 248045,846,198 then 29x 67089.
  EXPECT_EQ(ids[0], 248045);
  EXPECT_EQ(ids[1], 846);
  EXPECT_EQ(ids[2], 198);
  for (size_t i = 3; i < 32; ++i) EXPECT_EQ(ids[i], 67089) << "i=" << i;
}

TEST(QwenEncode, ThinkingFlagOnlyAffectsSuffix) {
  const char* p = tokenizer_json_path();
  if (!p) GTEST_SKIP() << "set KOKKOSDNN_QWEN_TOKENIZER_JSON for goldens";
  QT::Tokenizer tok;
  tok.load(p);
  const std::vector<int> t = tok.encode("a red panda", true, 512);
  const std::vector<int> f = tok.encode("a red panda", false, 512);
  ASSERT_GT(t.size(), 3u);
  ASSERT_GT(f.size(), 5u);
  // Shared head through "<think>" (248068); suffix differs.
  for (size_t i = 0; i + 1 < t.size(); ++i) EXPECT_EQ(t[i], f[i]);
  EXPECT_EQ(t.back(), 198);
  EXPECT_EQ((std::vector<int>{f[f.size() - 3], f[f.size() - 2], f.back()}),
            (std::vector<int>{271, 248069, 271}));
}

}  // namespace
