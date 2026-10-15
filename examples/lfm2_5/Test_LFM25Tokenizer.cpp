// LFM2.5 tokenizer regression tests (synthetic GPT-2-style fixture only,
// no model downloads). Covers: LFM2.5 special-token conventions, BPE
// roundtrip, special-token atomicity (<|im_start|> is never split by BPE),
// the ChatML render/encode path with thinking blocks, and ensure_bos.
//
// NOTE: this file is pure ASCII by design. Non-ASCII appears only as JSON
// \uXXXX escapes (fixture), so the \u-escape parser itself is covered too.
#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "KokkosDNN_LFM25Tokenizer.hpp"

namespace {

// Minimal GPT-2-style fixture: single lowercase chars + byte-mapped space
// (U+0120) + two merges; specials mirror LFM2.5 (<|startoftext|> = BOS,
// <|im_end|> = EOS/stop, <|pad|>, <think> pair).
std::string lfm25_fixture_json() {
  return R"({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "d": 2, "e": 3, "h": 4, "i": 5, "l": 6, "m": 7,
    "n": 8, "o": 9, "r": 10, "s": 11, "t": 12, "u": 13, "w": 14, "y": 15,
    "H": 16,
    "1": 30, "2": 31,
    "hi": 20, "th": 21,
    "\u0120": 17, "\u010a": 18
  }, "merges": ["h i", "t h"]},
  "added_tokens": [
    {"id": 100, "content": "<|startoftext|>", "special": true},
    {"id": 101, "content": "<|im_start|>", "special": true},
    {"id": 102, "content": "<|im_end|>", "special": true},
    {"id": 103, "content": "<think>", "special": true},
    {"id": 104, "content": "</think>", "special": true},
    {"id": 105, "content": "<|pad|>", "special": true}
  ]
})";
}

std::string write_tmp(const std::string& body) {
  std::string p =
      "/tmp/kokkosdnn_lfm25_tok_fixture_" + std::to_string(::getpid()) + ".json";
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("cannot write " + p);
  f << body;
  f.close();
  return p;
}

}  // namespace

TEST(LFM25Tokenizer, SpecialConventions) {
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json()));
  EXPECT_EQ(tok.vocab_size(), 106);  // max id 105 (specials extend vocab)
  // LFM2.5 convention: bos=<|startoftext|>, eos/stop=<|im_end|>.
  EXPECT_EQ(tok.bos(), 100);
  EXPECT_EQ(tok.eos(), 102);
  EXPECT_EQ(tok.eot(), 102);
  EXPECT_EQ(tok.pad(), 105);
  EXPECT_EQ(tok.im_start(), 101);
  EXPECT_EQ(tok.think_start(), 103);
  EXPECT_EQ(tok.think_end(), 104);
  for (int id = 100; id <= 105; ++id) EXPECT_TRUE(tok.is_special(id));
  EXPECT_FALSE(tok.is_special(0));
}

TEST(LFM25Tokenizer, Roundtrip) {
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json()));
  // Merges fire per piece: "hi" is one id.
  EXPECT_EQ(tok.encode("hi", false), (std::vector<int>{20}));
  EXPECT_EQ(tok.encode("the", false), (std::vector<int>{21, 3}));
  // Digits are isolated (individual-digits splitting).
  EXPECT_EQ(tok.encode("12", false), (std::vector<int>{30, 31}));
  // Space handling via the GPT-2 byte mapping (U+0120).
  EXPECT_EQ(tok.encode("hi there", false),
            (std::vector<int>{20, 17, 21, 3, 10, 3}));
  // Exact roundtrips through the byte mapping.
  EXPECT_EQ(tok.decode(tok.encode("hello", false)), "hello");
  EXPECT_EQ(tok.decode(tok.encode("hi there", false)), "hi there");
  EXPECT_EQ(tok.decode(tok.encode("he1", false)), "he1");
  // BOS prepend + skip-specials decode.
  EXPECT_EQ(tok.encode("he", true), (std::vector<int>{100, 4, 3}));
  EXPECT_EQ(tok.decode({100, 4, 3, 102}), "he");
  EXPECT_EQ(tok.decode({100, 4, 3, 102}, false),
            "<|startoftext|>he<|im_end|>");
}

TEST(LFM25Tokenizer, SpecialAtomicity) {
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json()));
  // <|im_start|> is emitted as one atomic id, never split into BPE pieces.
  EXPECT_EQ(tok.encode("a<|im_start|>b", false),
            (std::vector<int>{0, 101, 1}));
  EXPECT_EQ(tok.encode("<|im_start|>user", false)[0], 101);
  // The think pair is atomic too (longest-match special split).
  EXPECT_EQ(tok.encode("<think>hi</think>", false),
            (std::vector<int>{103, 20, 104}));
  // Adjacent specials stay separate ids (no accidental fusion).
  EXPECT_EQ(tok.encode("<|im_start|><|im_end|>", false),
            (std::vector<int>{101, 102}));
}

TEST(LFM25Tokenizer, ChatRenderGolden) {
  using KokkosDNN::LFM25::ChatMessage;
  using KokkosDNN::LFM25::render_lfm25_chatml;

  // System + user + open assistant header (generation prompt).
  EXPECT_EQ(render_lfm25_chatml({{"system", "Be brief.", ""},
                                 {"user", "Hi", ""}},
                                true),
            "<|im_start|>system\nBe brief.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n");

  // Thinking block: emitted inside an assistant turn only when that turn
  // carries non-empty thinking and preserve_thinking is true.
  std::vector<ChatMessage> think = {{"assistant", "reply", "plan"}};
  EXPECT_EQ(render_lfm25_chatml(think, false, true),
            "<|im_start|>assistant\n<think>plan</think>\nreply<|im_end|>\n");
  // preserve_thinking=false drops the trace but keeps the content.
  EXPECT_EQ(render_lfm25_chatml(think, false, false),
            "<|im_start|>assistant\nreply<|im_end|>\n");
  // Thinking on non-assistant roles is ignored.
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", "nope"}}, false, true),
            "<|im_start|>user\nHi<|im_end|>\n");
  // No generation prompt on request; default-system injection when asked.
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", ""}}, false),
            "<|im_start|>user\nHi<|im_end|>\n");
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", ""}}, false, true, "Sys."),
            "<|im_start|>system\nSys.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n");
  // Empty default disables injection.
  EXPECT_EQ(render_lfm25_chatml({}, true),
            "<|im_start|>assistant\n");
}

TEST(LFM25Tokenizer, ChatEncodeRoundtrip) {
  using KokkosDNN::LFM25::ChatMessage;
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json()));
  std::vector<ChatMessage> msgs = {{"system", "be bold", ""},
                                   {"user", "hi there", ""}};
  std::vector<int> ids =
      KokkosDNN::LFM25::encode_chatml(tok, msgs, true, true);
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids[0], 100);  // exactly-one-BOS (<|startoftext|>)
  size_t lead = 0;
  while (lead < ids.size() && ids[lead] == 100) ++lead;
  EXPECT_EQ(lead, 1u);
  // Literal specials survive the special-token split path.
  std::string raw = tok.decode(ids, false);
  EXPECT_EQ(raw.substr(0, 28), "<|startoftext|><|im_start|>s");
  EXPECT_NE(raw.find("<|im_start|>assistant\n"), std::string::npos);
  // Skipping specials recovers the plain text.
  std::string back = tok.decode(ids);
  EXPECT_NE(back.find("hi there"), std::string::npos);
  EXPECT_NE(back.find("be bold"), std::string::npos);
}

TEST(LFM25Tokenizer, EnsureBos) {
  using KokkosDNN::LFM25::ensure_bos;
  // Collapse runs to exactly one / strip / empty edge.
  EXPECT_EQ(ensure_bos({100, 100, 5}, 100, true), (std::vector<int>{100, 5}));
  EXPECT_EQ(ensure_bos({5, 100}, 100, true), (std::vector<int>{100, 5, 100}));
  EXPECT_EQ(ensure_bos({100, 100}, 100, true), (std::vector<int>{100}));
  EXPECT_EQ(ensure_bos({}, 100, true), (std::vector<int>{100}));
  EXPECT_EQ(ensure_bos({100, 100, 5}, 100, false), (std::vector<int>{5}));
  EXPECT_EQ(ensure_bos({5}, 100, false), (std::vector<int>{5}));

  // Through the fixture tokenizer: the chat render carries no BOS literal
  // (BOS comes from encode), so encode(add_bos=true) yields exactly one;
  // encode_rendered_chat keeps that invariant even when the raw text
  // starts with the BOS literal.
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json()));
  EXPECT_EQ(tok.bos(), 100);
  auto dup = KokkosDNN::LFM25::encode_rendered_chat(
      tok, "<|startoftext|>he", true);
  EXPECT_EQ(dup.front(), 100);
  size_t d = 0;
  while (d < dup.size() && dup[d] == 100) ++d;
  EXPECT_EQ(d, 1u);
  auto stripped = KokkosDNN::LFM25::encode_rendered_chat(
      tok, "<|startoftext|>he", false);
  EXPECT_EQ(stripped, (std::vector<int>{4, 3}));  // 'h' 'e', BOS stripped
}

TEST(LFM25Tokenizer, StopStrings) {
  using KokkosDNN::LFM25::default_stop_strings;
  using KokkosDNN::LFM25::truncate_at_stop_strings;
  EXPECT_EQ(default_stop_strings(),
            (std::vector<std::string>{"<|im_end|>"}));
  EXPECT_EQ(truncate_at_stop_strings("reply<|im_end|>extra",
                                     default_stop_strings()),
            "reply");
  EXPECT_EQ(truncate_at_stop_strings("clean", default_stop_strings()),
            "clean");
}
