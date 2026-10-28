// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Glimmer tokenizer tests: synthetic fixture (always runs) covering the
// o200k-style Split pre-tokenizer (letter runs, contraction suffix,
// digits, slash, whitespace), [a, b]-pair merges, specials, BOS handling
// and the Muse-style chat template; plus a real-file oracle (runs when
// KOKKOSDNN_TOKENIZER_JSON points at the
// darkc0de/Muse-Glimmer-30B-heretic tokenizer.json; expected ids generated
// with Hugging Face `tokenizers`).
//
// NOTE: this file is pure ASCII by design. Non-ASCII appears only as JSON
// \uXXXX escapes (fixture) and C++ \xNN escapes (expectations).
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "KokkosDNN_GlimmerTokenizer.hpp"

namespace {

// Minimal BPE fixture exercising: specials split, the o200k Split pieces
// (lowercase run + contraction suffix as ONE piece, uppercase-run alt2,
// N{1,3} digit split, '/' inside alt4, space gluing), GPT-2 byte mapping
// (space -> U+0120, byte 0xC3 -> U+00C3), per-piece BPE from [a, b]-pair
// merges (plus one legacy "e f" string merge), and decode. The
// post_processor decoy guards the parser against TemplateProcessing blocks
// with non-numeric ids.
std::string fixture_json() {
  return R"({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "c": 2, "d": 3, "f": 4, "o": 5, "n": 6, "t": 7,
    "A": 8, "B": 9, "C": 10, "'": 11, "/": 12, " ": 13,
    "w": 14, "s": 15, "e": 16, ":": 17,
    "1": 20, "2": 21, "3": 22, "4": 23, "12": 24, "123": 25,
    "ef": 38,
    "ab": 30, "abc": 31, "don": 32, "don't": 33, "ABC": 34, "ABCdef": 35,
    "'t": 36, "was": 37,
    "\u00c3": 60, "\u00a9": 61, "\u0120": 50, "\u0120b": 51
  }, "merges": [["a", "b"], ["ab", "c"], ["d", "o"], ["do", "n"],
                ["'", "t"], ["don", "'t"], ["w", "a"], ["wa", "s"],
                ["A", "B"], ["AB", "C"], ["d", "e"], ["de", "f"],
                ["ABC", "def"], ["1", "2"], ["12", "3"], "e f",
                ["\u0120", "b"]]},
  "post_processor": {"special_tokens": {"<|begin_of_text|>": {"id": "<|begin_of_text|>"}}},
  "added_tokens": [
    {"id": 200000, "content": "<|begin_of_text|>", "special": true},
    {"id": 200001, "content": "<|end_of_text|>", "special": true}
  ]
})";
}

std::string write_tmp(const std::string& body) {
  std::string p = "/tmp/kokkosdnn_glimmer_tok_fixture_" +
                  std::to_string(::getpid()) + ".json";
  std::ofstream f(p, std::ios::binary);
  f << body;
  f.close();
  return p;
}

}  // namespace

TEST(GlimmerTokenizer, FixtureLoad) {
  KokkosDNN::Glimmer::Tokenizer tok(write_tmp(fixture_json()));
  EXPECT_TRUE(tok.loaded());
  EXPECT_EQ(tok.vocab_size(), 200002);  // max id 200001 (specials extend vocab)
  EXPECT_EQ(tok.bos(), 200000);
  EXPECT_EQ(tok.eos(), 200001);
  EXPECT_EQ(tok.eom(), 200007);  // fallback (no <|eom|> here)
  EXPECT_EQ(tok.eot(), 200008);  // fallback (no <|eot|> here)
  EXPECT_TRUE(tok.is_special(200000));
  EXPECT_TRUE(tok.is_special(200001));
  EXPECT_FALSE(tok.is_special(0));
}

TEST(GlimmerTokenizer, FixtureBpe) {
  KokkosDNN::Glimmer::Tokenizer tok(write_tmp(fixture_json()));
  // pair merges build "abc" from chars
  EXPECT_EQ(tok.encode("abc", false), (std::vector<int>{31}));
  EXPECT_EQ(tok.encode("ab", false), (std::vector<int>{30}));
  // contraction suffix stays in ONE pre-token piece, then BPEs to one id
  EXPECT_EQ(tok.encode("don't", false), (std::vector<int>{33}));
  EXPECT_EQ(tok.encode("don", false), (std::vector<int>{32}));
  // legacy single-string merge form
  EXPECT_EQ(tok.encode("ef", false), (std::vector<int>{38}));
  // " b" stays one Split piece ("Ġb"); the fixture's ("Ġ", "b") merge
  // fuses it into one id (mirroring the real file's "Ġb" fusion)
  EXPECT_EQ(tok.encode("a b", false), (std::vector<int>{0, 51}));
  EXPECT_EQ(tok.encode("a  b", false), (std::vector<int>{0, 50, 51}));
  // \p{N}{1,3} splits "1234" into "123" + "4"
  EXPECT_EQ(tok.encode("123", false), (std::vector<int>{25}));
  EXPECT_EQ(tok.encode("12", false), (std::vector<int>{24}));
  EXPECT_EQ(tok.encode("1234", false), (std::vector<int>{25, 23}));
}

TEST(GlimmerTokenizer, FixturePretokenizer) {
  KokkosDNN::Glimmer::Tokenizer tok(write_tmp(fixture_json()));
  // uppercase run (alt2) then lowercase tail in one piece
  EXPECT_EQ(tok.encode("ABC", false), (std::vector<int>{34}));
  EXPECT_EQ(tok.encode("ABCdef", false), (std::vector<int>{35}));
  // "'" prefixes the piece (longest-match special scan misses it), then BPE
  EXPECT_EQ(tok.encode("'twas", false), (std::vector<int>{36, 37}));
  // '/' glues to the following run via the prefix slot
  EXPECT_EQ(tok.encode("a/b", false), (std::vector<int>{0, 12, 1}));
  EXPECT_EQ(tok.encode("://", false), (std::vector<int>{17, 12, 12}));
  // specials pass through untouched
  EXPECT_EQ(tok.encode("a<|end_of_text|>b", false),
            (std::vector<int>{0, 200001, 1}));
  // add_bos prepends the BOS id
  EXPECT_EQ(tok.encode("a", true), (std::vector<int>{200000, 0}));
}

TEST(GlimmerTokenizer, FixtureByteLevel) {
  KokkosDNN::Glimmer::Tokenizer tok(write_tmp(fixture_json()));
  // U+00C3 U+00A9 decode back to UTF-8 bytes C3 A9 (U+00E9, "e-acute")
  EXPECT_EQ(tok.decode({60, 61}), "\xc3\xa9");
  // specials skipped by default, kept on request
  EXPECT_EQ(tok.decode({200000, 0, 200001, 1}), "ab");
  EXPECT_EQ(tok.decode({200000, 0, 200001, 1}, false),
            "<|begin_of_text|>a<|end_of_text|>b");
}

TEST(GlimmerTokenizer, FixtureChat) {
  using namespace KokkosDNN::Glimmer;
  // Muse-style framing (assumption: no upstream chat_template; see header).
  auto msgs = make_chat_messages("abc", "sys", "user");
  ASSERT_EQ(msgs.size(), 2u);
  std::string rendered = render_chat_template(msgs, true);
  EXPECT_EQ(rendered,
            "<|begin_of_text|><|start|>system<|message|>sys<|eot|>"
            "<|start|>user<|message|>abc<|eot|>"
            "<|start|>assistant<|message|>");
  EXPECT_EQ(render_chat_template({msgs[1]}, false),
            "<|begin_of_text|><|start|>user<|message|>abc<|eot|>");
  // Literal BOS in a rendered string collapses to exactly one BOS id.
  Tokenizer tok(write_tmp(fixture_json()));
  EXPECT_EQ(encode_rendered_chat(tok, "<|begin_of_text|>a", true),
            (std::vector<int>{200000, 0}));
  EXPECT_EQ(encode_rendered_chat(tok, "<|begin_of_text|>a", false),
            (std::vector<int>{0}));
  // ensure_bos invariant both directions.
  EXPECT_EQ(ensure_bos({200000, 200000, 0}, 200000, true),
            (std::vector<int>{200000, 0}));
  EXPECT_EQ(ensure_bos({200000, 0}, 200000, false), (std::vector<int>{0}));
}

TEST(GlimmerTokenizer, ChatWithToolsParity) {
  using namespace KokkosDNN::Glimmer;
  // No tools/system extras: byte-identical to the plain template.
  auto msgs = make_chat_messages("abc", "sys", "user");
  EXPECT_EQ(render_glimmer_chatml_with_tools(msgs, ""),
            render_chat_template(msgs, true));
  EXPECT_EQ(render_glimmer_chatml_with_tools({msgs[1]}, "", "", false),
            render_chat_template({msgs[1]}, false));
  // Tools catalogue + suffix fold into the system block.
  std::string rendered = render_glimmer_chatml_with_tools(
      msgs, R"([{"type":"function"}])", "Always answer.", true);
  EXPECT_EQ(rendered,
            "<|begin_of_text|>"
            "<|start|>system<|message|>sys\n"
            "List of tools: [{\"type\":\"function\"}]\n"
            "Always answer.<|eot|>"
            "<|start|>user<|message|>abc<|eot|>"
            "<|start|>assistant<|message|>");
  // System injected when no system message exists.
  std::string injected = render_glimmer_chatml_with_tools(
      {msgs[1]}, R"([{"type":"function"}])", "", true, "be nice");
  EXPECT_EQ(injected,
            "<|begin_of_text|>"
            "<|start|>system<|message|>be nice\n"
            "List of tools: [{\"type\":\"function\"}]<|eot|>"
            "<|start|>user<|message|>abc<|eot|>"
            "<|start|>assistant<|message|>");
  // Encode path matches the render path: the framing literals must split
  // as specials, so the check runs on a fixture extended with them (the
  // base fixture only carries begin/end_of_text) and fixture-safe content.
  // The role words also need their missing letters as base singletons.
  std::string chat_fixture = fixture_json();
  const std::string vocab_anchor = "\"e\": 16, \":\": 17,";
  ASSERT_NE(chat_fixture.find(vocab_anchor), std::string::npos);
  chat_fixture.replace(chat_fixture.find(vocab_anchor), vocab_anchor.size(),
                       "\"e\": 16, \":\": 17, \"y\": 100, \"u\": 101, "
                       "\"r\": 102, \"i\": 103, \"n\": 104, \"m\": 105,");
  const std::string anchor =
      "{\"id\": 200001, \"content\": \"<|end_of_text|>\", \"special\": true}";
  ASSERT_NE(chat_fixture.find(anchor), std::string::npos);
  chat_fixture.replace(
      chat_fixture.find(anchor), anchor.size(),
      anchor +
          ",\n    {\"id\": 200007, \"content\": \"<|eom|>\", "
          "\"special\": true},\n"
          "    {\"id\": 200008, \"content\": \"<|eot|>\", \"special\": true},\n"
          "    {\"id\": 200010, \"content\": \"<|start|>\", "
          "\"special\": true},\n"
          "    {\"id\": 200011, \"content\": \"<|message|>\", "
          "\"special\": true}");
  Tokenizer tok(write_tmp(chat_fixture));
  auto enc_msgs = make_chat_messages("abc", "cab", "user");
  EXPECT_EQ(encode_chatml_with_tools(tok, enc_msgs, "", "", true, true),
            encode_chat(tok, enc_msgs, true, true));
}

TEST(GlimmerTokenizer, DefaultStopsAndPlainTextTools) {
  using namespace KokkosDNN::Glimmer;
  EXPECT_EQ(default_stop_strings(),
            (std::vector<std::string>{"<|eot|>", "<|eom|>"}));
  // No native markers: everything before the stop cut is display text,
  // calls and reasoning stay empty.
  ToolParseResult r = parse_tool_calls("hello <|eot|> tail",
                                       default_stop_strings());
  EXPECT_EQ(r.text, "hello ");
  EXPECT_TRUE(r.reasoning.empty());
  EXPECT_TRUE(r.calls.empty());

  // Streaming shim agrees with the whole-text parse.
  ToolCallStreamParser p(default_stop_strings());
  std::string streamed;
  for (const char* piece : {"hel", "lo ", "<|e", "ot|> tail"})
    streamed += p.feed(piece).display;
  ToolParseResult tail = p.finish();
  EXPECT_EQ(streamed + tail.text, "hello ");
  EXPECT_TRUE(tail.calls.empty());
  EXPECT_EQ(p.poll_reasoning(), "");
}

TEST(GlimmerTokenizer, MissingFileThrows) {
  EXPECT_THROW(KokkosDNN::Glimmer::Tokenizer("/nonexistent/tok.json"),
               std::runtime_error);
}

TEST(GlimmerTokenizer, StopStrings) {
  using namespace KokkosDNN::Glimmer;
  EXPECT_EQ(truncate_at_stop_strings("hello world", {"world"}), "hello ");
  EXPECT_EQ(truncate_at_stop_strings("hello world", {"xyz"}), "hello world");
  EXPECT_EQ(truncate_at_stop_strings("abc", {""}), "abc");
  std::string hit;
  EXPECT_EQ(find_stop_string("aab", {"b", "ab"}, &hit), 1u);
  EXPECT_EQ(hit, "ab");  // earliest match wins over the longer-id alternative
  EXPECT_EQ(find_stop_string("abc", {"xyz"}), std::string::npos);
}

// Real Glimmer tokenizer.json oracle vectors (HF `tokenizers` output for
// darkc0de/Muse-Glimmer-30B-heretic).
TEST(GlimmerTokenizer, RealFileOracle) {
  const char* p = std::getenv("KOKKOSDNN_TOKENIZER_JSON");
  if (!p || !*p) GTEST_SKIP() << "set KOKKOSDNN_TOKENIZER_JSON to run";
  KokkosDNN::Glimmer::Tokenizer tok(p);
  EXPECT_EQ(tok.vocab_size(), 202048);
  EXPECT_EQ(tok.bos(), 200000);
  EXPECT_EQ(tok.eos(), 200001);
  EXPECT_EQ(tok.eom(), 200007);
  EXPECT_EQ(tok.eot(), 200008);
  const std::vector<std::pair<std::string, std::vector<int>>> cases = {
      {"Hello", {200000, 19873}},
      {"Hello world", {200000, 19873, 3817}},
      {"Hello, I am", {200000, 19873, 24, 372, 1045}},
      {"don't can't won't", {200000, 127161, 12337, 21094}},
      {"123 4567 89", {200000, 7235, 220, 19596, 35, 220, 6189}},
      {"ABCdef GHI", {200000, 34271, 1251, 477, 31862}},
      {"a  b", {200000, 77, 220, 297}},
      {"a\n b", {200000, 77, 198, 297}},
      {"caf" "\xc3\xa9" " na" "\xc3\xaf" "ve",
       {200000, 1349, 53389, 1181, 164301}},
      {"\xf0\x9f\x98\x80" " emoji", {200000, 80883, 222, 88230}},
      {"'twas the night", {200000, 2050, 12521, 290, 9331}},
      {"http://example.com/a/b",
       {200000, 3303, 1050, 15860, 1127, 11159, 4790}},
      {"Explain Kokkos in one sentence.",
       {200000, 51015, 75518, 52298, 310, 1085, 17705, 26}},
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
  EXPECT_EQ(tok.decode({200000, 19873, 200001}), "Hello");
  EXPECT_EQ(tok.decode({200000, 19873, 200001}, false),
            "<|begin_of_text|>Hello<|end_of_text|>");
  // ... while inputs containing specials decode to text-only by default
  // and roundtrip exactly when specials are kept.
  const std::string chat =
      "<|start|>user<|message|>Hi<|eot|><|start|>assistant<|message|>";
  auto chat_ids = tok.encode(chat);
  EXPECT_EQ(chat_ids, (std::vector<int>{200000, 200022, 1556, 200023, 25181,
                                        200008, 200022, 140680, 200023}));
  EXPECT_EQ(tok.decode(chat_ids), "userHiassistant");
  // keep-specials also renders the add_bos <|begin_of_text|>
  EXPECT_EQ(tok.decode(chat_ids, false), "<|begin_of_text|>" + chat);
}
