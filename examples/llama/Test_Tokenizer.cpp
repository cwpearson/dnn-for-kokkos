// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Tokenizer regression tests: synthetic fixture (always runs) + real-file
// oracle vectors (runs when KOKKOSDNN_TOKENIZER_JSON points at a Llama 3
// tokenizer.json; expected ids generated with Hugging Face `tokenizers`).
//
// NOTE: this file is pure ASCII by design. Non-ASCII appears only as JSON
// \uXXXX escapes (fixture) and C++ \xNN escapes (expectations), so the
// \u-escape parser itself is covered too.
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "KokkosDNN_Tokenizer.hpp"

namespace {

// Minimal BPE fixture exercising: specials split, Split pre-tokenizer
// (letters/digits/contractions), GPT-2 byte mapping (space -> U+0120, byte
// 0xC3 -> U+00C3), per-piece BPE, and decode incl. multibyte UTF-8. The
// post_processor decoy ("id" with a *string* value) guards the parser
// against the real file's TemplateProcessing block that crashed stoi.
std::string fixture_json() {
  return R"({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "c": 2, "d": 3, "f": 4, "o": 5, "n": 6,
    "1": 10, "2": 11, "3": 12, "4": 13, "12": 14, "123": 15,
    "ab": 20, "abc": 21, "don": 22, "'t": 23,
    "\u00c3": 30, "\u00a9": 31, "\u0120": 40
  }, "merges": ["a b", "ab c", "' t", "1 2", "12 3", "d o", "do n"]},
  "post_processor": {"special_tokens": {"<|bot|>": {"id": "<|bot|>"}}},
  "added_tokens": [
    {"id": 100, "content": "<|bot|>", "special": true},
    {"id": 101, "content": "<|eot|>", "special": true}
  ]
})";
}

std::string write_tmp(const std::string& body) {
  std::string p = "/tmp/kokkosdnn_tok_fixture_" +
                  std::to_string(::getpid()) + ".json";
  std::ofstream f(p, std::ios::binary);
  f << body;
  f.close();
  return p;
}

}  // namespace

TEST(Tokenizer, FixtureLoad) {
  KokkosDNN::Llama::Tokenizer tok(write_tmp(fixture_json()));
  EXPECT_TRUE(tok.loaded());
  EXPECT_EQ(tok.vocab_size(), 102);  // max id 101 (specials extend vocab)
  EXPECT_EQ(tok.bos(), 128000);      // fallback (no <|begin_of_text|> here)
  EXPECT_EQ(tok.eos(), 128001);
  EXPECT_EQ(tok.eot(), 128009);
  EXPECT_TRUE(tok.is_special(100));
  EXPECT_TRUE(tok.is_special(101));
  EXPECT_FALSE(tok.is_special(0));
}

TEST(Tokenizer, FixtureBpe) {
  KokkosDNN::Llama::Tokenizer tok(write_tmp(fixture_json()));
  EXPECT_EQ(tok.encode("abc", false), (std::vector<int>{21}));
  EXPECT_EQ(tok.encode("ab", false), (std::vector<int>{20}));
  // " a" stays one Split piece; space byte-maps to U+0120, no merge for
  // that pair in the fixture, so per-char fallback ids:
  EXPECT_EQ(tok.encode("a b", false), (std::vector<int>{0, 40, 1}));
  // \p{N}{1,3} splits "1234" into "123" + "4"
  EXPECT_EQ(tok.encode("123", false), (std::vector<int>{15}));
  EXPECT_EQ(tok.encode("12", false), (std::vector<int>{14}));
  EXPECT_EQ(tok.encode("1234", false), (std::vector<int>{15, 13}));
}

TEST(Tokenizer, FixturePretokenizer) {
  KokkosDNN::Llama::Tokenizer tok(write_tmp(fixture_json()));
  // contraction splits to ["don", "'t"]
  EXPECT_EQ(tok.encode("don't", false), (std::vector<int>{22, 23}));
  // specials pass through untouched
  EXPECT_EQ(tok.encode("a<|eot|>b", false), (std::vector<int>{0, 101, 1}));
  // add_bos prepends the (fallback) BOS id
  EXPECT_EQ(tok.encode("a", true), (std::vector<int>{128000, 0}));
}

TEST(Tokenizer, FixtureByteLevel) {
  KokkosDNN::Llama::Tokenizer tok(write_tmp(fixture_json()));
  // ASCII maps to itself
  EXPECT_EQ(tok.decode({4, 5, 5}), "foo");
  // U+00C3 U+00A9 decode back to UTF-8 bytes C3 A9 (U+00E9, "e-acute")
  EXPECT_EQ(tok.decode({30, 31}), "\xc3\xa9");
  // specials skipped by default, kept on request
  EXPECT_EQ(tok.decode({100, 0, 101, 1}), "ab");
  EXPECT_EQ(tok.decode({100, 0, 101, 1}, false), "<|bot|>a<|eot|>b");
}

TEST(Tokenizer, MissingFileThrows) {
  EXPECT_THROW(KokkosDNN::Llama::Tokenizer("/nonexistent/tok.json"),
               std::runtime_error);
}

// Real Llama 3 tokenizer.json oracle vectors (HF `tokenizers` output).
TEST(Tokenizer, RealFileOracle) {
  const char* p = std::getenv("KOKKOSDNN_TOKENIZER_JSON");
  if (!p || !*p) GTEST_SKIP() << "set KOKKOSDNN_TOKENIZER_JSON to run";
  KokkosDNN::Llama::Tokenizer tok(p);
  EXPECT_EQ(tok.vocab_size(), 128256);
  EXPECT_EQ(tok.bos(), 128000);
  EXPECT_EQ(tok.eos(), 128001);
  EXPECT_EQ(tok.eot(), 128009);
  const std::vector<std::pair<std::string, std::vector<int>>> cases = {
      {"Hello", {128000, 9906}},
      {"Hello world", {128000, 9906, 1917}},
      {"Hello, I am", {128000, 9906, 11, 358, 1097}},
      {"Explain Kokkos in one sentence.",
       {128000, 849, 21435, 65815, 57778, 304, 832, 11914, 13}},
      {"don't can't won't", {128000, 15357, 956, 649, 956, 2834, 956}},
      {"123 4567 89", {128000, 4513, 220, 10961, 22, 220, 4578}},
      {"caf" "\xc3\xa9" " na" "\xc3\xaf" "ve " "\xc3\x9c" "n" "\xc3\xaf" "cod" "\xc3\xa9",
       {128000, 936, 59958, 95980, 588, 105766, 38672, 20720, 978}},
      {"\xf0\x9f\x98\x80" " emoji", {128000, 76460, 222, 43465}},
      {"a  b", {128000, 64, 220, 293}},
      {"a\n b", {128000, 64, 198, 293}},
      {"\xc2\xa0" "nbsp" "\xc2\xa0", {128000, 4194, 5792, 4194}},
  };
  for (auto& [text, want] : cases) {
    EXPECT_EQ(tok.encode(text), want) << "text: " << text;
  }
  // Plain-text cases roundtrip exactly ...
  for (auto& [text, want] : cases) {
    if (text.find("<|") != std::string::npos) continue;  // specials below
    EXPECT_EQ(tok.decode(tok.encode(text)), text) << "roundtrip: " << text;
  }
  // specials skipped on decode (HF default), kept on request
  EXPECT_EQ(tok.decode({128000, 9906, 1917}), "Hello world");
  EXPECT_EQ(tok.decode({128000, 9906, 128001}, false),
            "<|begin_of_text|>Hello<|end_of_text|>");
  // ... while inputs containing specials decode to text-only by default
  // and roundtrip exactly when specials are kept.
  const std::string chat =
      "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>";
  auto chat_ids = tok.encode(chat);
  EXPECT_EQ(chat_ids,
            (std::vector<int>{128000, 128006, 882, 128007, 271, 13347,
                              128009}));
  EXPECT_EQ(tok.decode(chat_ids), "user\n\nHi");
  // keep-specials also renders the add_bos <|begin_of_text|>
  EXPECT_EQ(tok.decode(chat_ids, false), "<|begin_of_text|>" + chat);
}
