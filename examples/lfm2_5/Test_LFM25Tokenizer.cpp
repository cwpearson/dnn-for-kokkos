// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
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


// Same fixture with the transformers 5.x merge spelling: each merge is a
// two-element array instead of an "a b" string. Adds digit tokens so the
// LFM2 pre-tokenizer's \p{N}{1,3} grouping is observable.
std::string lfm25_fixture_json_pairs() {
  return R"({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "d": 2, "e": 3, "h": 4, "i": 5, "l": 6, "m": 7,
    "n": 8, "o": 9, "r": 10, "s": 11, "t": 12, "u": 13, "w": 14, "y": 15,
    "H": 16,
    "1": 30, "2": 31, "3": 32, "12": 33, "123": 34,
    "hi": 20, "th": 21,
    "\u0120": 17, "\u010a": 18
  }, "merges": [["h","i"], ["t","h"], ["1","2"], ["12","3"]]},
  "added_tokens": [
    {"id": 100, "content": "<|startoftext|>", "special": true},
    {"id": 101, "content": "<|im_start|>", "special": true},
    {"id": 102, "content": "<|im_end|>", "special": true},
    {"id": 103, "content": " thinking", "special": true},
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


// transformers 5.x writes merges as ["a", "b"] pairs instead of the legacy
// "a b" strings. Both spellings must produce identical encodings; parsing
// only the string form silently disables every merge (pure byte fallback).
TEST(LFM25Tokenizer, PairFormMerges) {
  KokkosDNN::LFM25::Tokenizer strings_tok(write_tmp(lfm25_fixture_json()));
  KokkosDNN::LFM25::Tokenizer pairs_tok(
      write_tmp(lfm25_fixture_json_pairs()));
  for (const char* text : {"hi", "the", "hi there", "he1"}) {
    EXPECT_EQ(pairs_tok.encode(text, false), strings_tok.encode(text, false))
        << "text='" << text << "'";
  }
  EXPECT_EQ(pairs_tok.encode("hi", false), (std::vector<int>{20}));
  EXPECT_EQ(pairs_tok.encode("the", false), (std::vector<int>{21, 3}));
}

// LFM2 pre-tokenizer regex (not the GPT-2 one): number runs group up to
// three digits, and a whitespace run that contains a newline breaks after
// its last newline.
TEST(LFM25Tokenizer, PretokenizerRegex) {
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json_pairs()));
  // "123" is one pre-token, so the two merges fuse it to a single id
  // (digit-splitting would have emitted 1, 2, 3).
  EXPECT_EQ(tok.encode("123", false), (std::vector<int>{34}));
  // Symbol/space handling is unchanged from the byte-level mapping.
  EXPECT_EQ(tok.encode("hi there", false),
            (std::vector<int>{20, 17, 21, 3, 10, 3}));
  // Two spaces: the run keeps all but the last space, which prefixes the
  // following word (\s+(?!\S) backtracking).
  EXPECT_EQ(tok.encode("hi  there", false),
            (std::vector<int>{20, 17, 17, 21, 3, 10, 3}));
  // "\n\n" is one pre-token (\s*[\r\n]+); the fixture has no "\n\n"
  // merge, so it still encodes as two newline ids.
  EXPECT_EQ(tok.encode("hi\n\nthere", false),
            (std::vector<int>{20, 18, 18, 21, 3, 10, 3}));
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
  EXPECT_EQ(render_lfm25_chatml({{"system", "Be brief.", "", {}, ""},
                                 {"user", "Hi", "", {}, ""}},
                                true),
            "<|im_start|>system\nBe brief.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n");

  // Thinking block: emitted inside an assistant turn only when that turn
  // carries non-empty thinking and preserve_thinking is true.
  std::vector<ChatMessage> think = {{"assistant", "reply", "plan", {}, ""}};
  EXPECT_EQ(render_lfm25_chatml(think, false, true),
            "<|im_start|>assistant\n<think>plan</think>\nreply<|im_end|>\n");
  // preserve_thinking=false drops the trace but keeps the content.
  EXPECT_EQ(render_lfm25_chatml(think, false, false),
            "<|im_start|>assistant\nreply<|im_end|>\n");
  // Thinking on non-assistant roles is ignored.
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", "nope", {}, ""}}, false, true),
            "<|im_start|>user\nHi<|im_end|>\n");
  // No generation prompt on request; default-system injection when asked.
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", "", {}, ""}}, false),
            "<|im_start|>user\nHi<|im_end|>\n");
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", "", {}, ""}}, false, true, "Sys."),
            "<|im_start|>system\nSys.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n");
  // Empty default disables injection.
  EXPECT_EQ(render_lfm25_chatml({}, true),
            "<|im_start|>assistant\n");
}

TEST(LFM25Tokenizer, ChatEncodeRoundtrip) {
  using KokkosDNN::LFM25::ChatMessage;
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json()));
  std::vector<ChatMessage> msgs = {{"system", "be bold", "", {}, ""},
                                   {"user", "hi there", "", {}, ""}};
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

TEST(LFM25Tokenizer, ToolRenderGolden) {
  using KokkosDNN::LFM25::ChatMessage;
  using KokkosDNN::LFM25::ToolCall;
  using KokkosDNN::LFM25::render_lfm25_chatml_with_tools;

  // No tools + plain history renders exactly like the base renderer.
  const std::vector<ChatMessage> plain = {{"system", "Be brief.", "", {}, ""},
                                            {"user", "Hi", "", {}, ""}};
  EXPECT_EQ(render_lfm25_chatml_with_tools(plain, "", "", true, true),
            KokkosDNN::LFM25::render_lfm25_chatml(plain, true, true));

  // Tools catalogue is injected into the system block.
  const std::string tools =
      "[{\"type\":\"function\",\"function\":{\"name\":"
      "\"get_weather\"}}]";
  EXPECT_EQ(render_lfm25_chatml_with_tools({{"user", "Hi", "", {}, ""}}, tools, "",
                                           true, true),
            "<|im_start|>system\nList of tools: " + tools +
                "<|im_end|>\n"
                "<|im_start|>user\nHi<|im_end|>\n"
                "<|im_start|>assistant\n");
  const std::vector<ChatMessage> sys_user = {{"system", "Sys.", "", {}, ""},
                                                  {"user", "Hi", "", {}, ""}};
  EXPECT_EQ(render_lfm25_chatml_with_tools(sys_user, tools, "Do X.", true,
                                           true),
      "<|im_start|>system\nSys.\nList of tools: " + tools +
          "\nDo X.<|im_end|>\n"
          "<|im_start|>user\nHi<|im_end|>\n"
          "<|im_start|>assistant\n");

  // Assistant history tool_calls reconstruct natively; tool results
  // render as tool turns.
  ToolCall c;
  c.name = "get_weather";
  c.arguments = "{\"location\":\"SF\"}";
  ChatMessage a;
  a.role = "assistant";
  a.content = "Checking.";
  a.tool_calls.push_back(c);
  ChatMessage tool_msg;
  tool_msg.role = "tool";
  tool_msg.content = "{\"temp\": 72}";
  EXPECT_EQ(
      render_lfm25_chatml_with_tools({a, tool_msg}, "", "", false, true),
      "<|im_start|>assistant\nChecking.\n"
      "<|tool_call_start|>[get_weather(location=\"SF\")]"
      "<|tool_call_end|><|im_end|>\n"
      "<|im_start|>tool\n{\"temp\": 72}<|im_end|>\n");
}

// The server's prefix cache keys on the history encoding without the
// trailing generation prompt and requires it to be an exact token prefix
// of the full encoding (specials split first, so the appended generation
// header never merges with history bytes). The same holds across an
// appended agentic turn: the earlier full prompt's history part stays a
// prefix of the extended prompt.
TEST(LFM25Tokenizer, ContentIsExactPrefixOfFull) {
  using KokkosDNN::LFM25::ChatMessage;
  KokkosDNN::LFM25::Tokenizer tok(write_tmp(lfm25_fixture_json()));
  auto is_prefix = [](const std::vector<int>& pre,
                      const std::vector<int>& full) {
    if (pre.size() > full.size()) return false;
    for (size_t i = 0; i < pre.size(); ++i)
      if (pre[i] != full[i]) return false;
    return true;
  };
  // Fixture-safe text only (the fixture vocab is lowercase fragments).
  const std::vector<ChatMessage> msgs = {{"system", "be bold", "", {}, ""},
                                         {"user", "hi there", "", {}, ""}};
  const std::vector<int> content =
      KokkosDNN::LFM25::encode_chatml(tok, msgs, false, true);
  const std::vector<int> full =
      KokkosDNN::LFM25::encode_chatml(tok, msgs, true, true);
  EXPECT_LT(content.size(), full.size());
  EXPECT_TRUE(is_prefix(content, full));

  // Agentic growth: an assistant turn plus a follow-up user turn appended.
  // The first turn's content stays an exact prefix of the extended prompt.
  const std::vector<ChatMessage> grown = {
      {"system", "be bold", "", {}, ""},
      {"user", "hi there", "", {}, ""},
      {"assistant", "hi", "", {}, ""},
      {"user", "the", "", {}, ""}};
  const std::vector<int> grown_full =
      KokkosDNN::LFM25::encode_chatml(tok, grown, true, true);
  EXPECT_LT(full.size(), grown_full.size());
  EXPECT_TRUE(is_prefix(content, grown_full));

  // Tool-bearing growth holds at the render level (tool names/JSON are
  // outside the fixture vocab, so this half stays string-level): history
  // renders identically up to the appended turns.
  using KokkosDNN::LFM25::ToolCall;
  ToolCall c;
  c.name = "get_weather";
  c.arguments = "{\"location\":\"SF\"}";
  ChatMessage a;
  a.role = "assistant";
  a.content = "Checking.";
  a.tool_calls.push_back(c);
  ChatMessage tool_msg;
  tool_msg.role = "tool";
  tool_msg.content = "{\"temp\": 72}";
  const std::vector<ChatMessage> tool_grown = {
      {"system", "be bold", "", {}, ""},
      {"user", "hi there", "", {}, ""}, a, tool_msg};
  const std::string base =
      KokkosDNN::LFM25::render_lfm25_chatml(msgs, false, true);
  const std::string grown_render =
      KokkosDNN::LFM25::render_lfm25_chatml_with_tools(tool_grown, "", "",
                                                       true, true);
  EXPECT_EQ(grown_render.substr(0, base.size()), base);
}

TEST(LFM25Tokenizer, ToolParse) {  using KokkosDNN::LFM25::kEosToken;
  using KokkosDNN::LFM25::kToolCallEndToken;
  using KokkosDNN::LFM25::kToolCallStartToken;
  using KokkosDNN::LFM25::parse_tool_call_block;
  using KokkosDNN::LFM25::parse_tool_calls;
  using KokkosDNN::LFM25::ToolParseResult;

  auto v = parse_tool_call_block("[f(a=1), g(x=\"y\", z=None)]");
  ASSERT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0].name, "f");
  EXPECT_EQ(v[0].arguments, "{\"a\":1}");
  EXPECT_EQ(v[1].arguments, "{\"x\":\"y\",\"z\":null}");
  EXPECT_TRUE(parse_tool_call_block("").empty());
  EXPECT_TRUE(parse_tool_call_block("no-parens").empty());
  // A max_tokens cut usually drops the trailing ']' but keeps '[': each
  // bracket is stripped independently so mid-list truncations still
  // recover the complete calls.
  {
    auto m1 = parse_tool_call_block("[f(a=1, b=2)");
    ASSERT_EQ(m1.size(), 1u);
    EXPECT_EQ(m1[0].name, "f");
    auto m2 = parse_tool_call_block("[f(a=1), g(b=2)");
    ASSERT_EQ(m2.size(), 2u);
    EXPECT_EQ(m2[0].name, "f");
    EXPECT_EQ(m2[1].name, "g");
    // First call complete, second cut mid-args: only the first recovers.
    auto m3 = parse_tool_call_block("[f(a=1), g(b=2");
    ASSERT_EQ(m3.size(), 1u);
    EXPECT_EQ(m3[0].name, "f");
    // Cut inside the only call's args: nothing recoverable.
    EXPECT_TRUE(parse_tool_call_block("[f(a=").empty());
  }

  const std::string raw =
      std::string("Hi ") + kToolCallStartToken +
      "[get_weather(location=\"SF\")]" + kToolCallEndToken + " bye" +
      kEosToken + "junk";
  ToolParseResult r = parse_tool_calls(raw);
  ASSERT_EQ(r.calls.size(), 1u);
  EXPECT_EQ(r.calls[0].name, "get_weather");
  EXPECT_EQ(r.calls[0].arguments, "{\"location\":\"SF\"}");
  EXPECT_EQ(r.text, "Hi  bye");  // turn ends at <|im_end|>

  ToolParseResult plain = parse_tool_calls("just text");
  EXPECT_TRUE(plain.calls.empty());
  EXPECT_EQ(plain.text, "just text");
  EXPECT_TRUE(plain.reasoning.empty());

  // Unclosed trailing call block (length-capped): recovered best-effort so
  // a cut-off call still executes instead of leaking markup. Mid-list cuts
  // (trailing ']' missing) recover the complete calls.
  ToolParseResult open =
      parse_tool_calls(std::string("a") + kToolCallStartToken + "[f(x=1)]");
  ASSERT_EQ(open.calls.size(), 1u);
  EXPECT_EQ(open.calls[0].name, "f");
  EXPECT_EQ(open.text, "a");
  ToolParseResult open_list = parse_tool_calls(
      std::string("a") + kToolCallStartToken + "[f(a=1), g(b=2)");
  ASSERT_EQ(open_list.calls.size(), 2u);
  EXPECT_EQ(open_list.calls[0].name, "f");
  EXPECT_EQ(open_list.calls[1].name, "g");
  EXPECT_EQ(open_list.text, "a");
  // Unparseable trailing bytes stay text, never a call.
  ToolParseResult junk =
      parse_tool_calls(std::string("a") + kToolCallStartToken + "oops");
  EXPECT_TRUE(junk.calls.empty());
  EXPECT_EQ(junk.text, "aoops");

  // <think> blocks split into reasoning, never into display.
  ToolParseResult th = parse_tool_calls(
      std::string("<think>hmm</think>answer") + kToolCallStartToken +
      "[f(x=1)]" + kToolCallEndToken);
  EXPECT_EQ(th.reasoning, "hmm");
  EXPECT_EQ(th.text, "answer");
  ASSERT_EQ(th.calls.size(), 1u);
  EXPECT_EQ(th.calls[0].name, "f");
  // Unclosed trailing think at turn start: the rest is reasoning.
  ToolParseResult th_open = parse_tool_calls(std::string("<think>thinking"));
  EXPECT_EQ(th_open.reasoning, "thinking");
  EXPECT_EQ(th_open.text, "");
  // A mid-turn <think> is literal content (thinking leads the turn).
  ToolParseResult th_mid =
      parse_tool_calls(std::string("pre <think>thinking"));
  EXPECT_TRUE(th_mid.reasoning.empty());
  EXPECT_EQ(th_mid.text, "pre <think>thinking");
  // The turn terminator cuts regardless of custom stops (the parser sees
  // specials): no <|im_end|> leak with a custom stop list.
  EXPECT_EQ(parse_tool_calls("a<|im_end|>b", {"STOP"}).text, "a");
  EXPECT_EQ(parse_tool_calls("The answer is 42<|im_end|>", {"STOP"}).text,
            "The answer is 42");
  // Length-capped mid-marker inputs strip the fragment in whole-text too.
  EXPECT_EQ(parse_tool_calls("hello <|im_e").text, "hello ");
  EXPECT_EQ(parse_tool_calls("hello <|tool_call_st").text, "hello ");
  // Only the single longest head is stripped: an earlier '<' followed by
  // '<' is unambiguous content.
  EXPECT_EQ(parse_tool_calls("hello <<").text, "hello <");
  EXPECT_EQ(parse_tool_calls("hello <<<").text, "hello <<");
  EXPECT_EQ(parse_tool_calls("<think>abc<<").reasoning, "abc<");
}

TEST(LFM25Tokenizer, ToolArgsConvert) {
  using KokkosDNN::LFM25::args_json_to_pythonic;
  using KokkosDNN::LFM25::args_pythonic_to_json;
  EXPECT_EQ(args_pythonic_to_json(""), "{}");
  EXPECT_EQ(args_pythonic_to_json(
                "location=\"SF\", n=3, ok=True, m=None"),
            "{\"location\":\"SF\",\"n\":3,\"ok\":true,"
            "\"m\":null}");
  EXPECT_EQ(args_pythonic_to_json("unit=celsius"),
            "{\"unit\":\"celsius\"}");
  const std::string js =
      args_pythonic_to_json("a='q', b=[1, \"x\", None]");
  EXPECT_EQ(args_json_to_pythonic(js), "a=\"q\", b=[1, \"x\", None]");
  EXPECT_EQ(args_json_to_pythonic(""), "");
  EXPECT_EQ(args_json_to_pythonic("{}"), "");
  EXPECT_EQ(args_json_to_pythonic("oops"), "");
}

TEST(LFM25Tokenizer, ToolStreamParser) {
  using KokkosDNN::LFM25::kEosToken;
  using KokkosDNN::LFM25::kToolCallEndToken;
  using KokkosDNN::LFM25::kToolCallStartToken;
  using KokkosDNN::LFM25::parse_tool_calls;
  using KokkosDNN::LFM25::ToolCallStreamParser;
  using KokkosDNN::LFM25::ToolParseResult;
  const std::string raw =
      std::string("Hi there ") + kToolCallStartToken +
      "[f(a=1), g(x=\"y\")]" + kToolCallEndToken + " done" + kEosToken;
  // Byte-at-a-time feeding agrees with the whole-text parse.
  ToolCallStreamParser fed;
  std::string acc, racc;
  for (char ch : raw) {
    acc += fed.push(std::string(1, ch));
    racc += fed.poll_reasoning();
  }
  ToolParseResult tail = fed.finish();
  acc += tail.text;
  racc += tail.reasoning;
  const ToolParseResult whole = parse_tool_calls(raw);
  EXPECT_EQ(acc, whole.text);
  EXPECT_EQ(racc, whole.reasoning);
  ASSERT_EQ(tail.calls.size(), 2u);
  EXPECT_EQ(tail.calls[0].name, "f");
  EXPECT_EQ(tail.calls[1].arguments, "{\"x\":\"y\"}");
  // Plain text streams through untouched (utf-8 safe).
  ToolCallStreamParser plain;
  const std::string snow = "snow \xe2\x9b\x84 end";
  std::string out;
  for (size_t i = 0; i < snow.size(); ++i) out += plain.push(snow.substr(i, 1));
  out += plain.finish().text;
  EXPECT_EQ(out, snow);
}

TEST(LFM25Tokenizer, ToolStreamCustomStop) {
  using KokkosDNN::LFM25::ToolCallStreamParser;
  using KokkosDNN::LFM25::ToolParseResult;
  using KokkosDNN::LFM25::parse_tool_calls;
  // Byte-at-a-time over a completing custom stop: partial heads are held
  // (never emitted and retracted), totals agree, and the tail does not
  // leak the stop back in. Used to throw std::out_of_range from push().
  ToolCallStreamParser fed({"STOP"});
  std::string acc, racc;
  const std::string raw = "hello STOP world";
  for (char ch : raw) {
    acc += fed.push(std::string(1, ch));
    racc += fed.poll_reasoning();
    EXPECT_EQ(acc.find("STO"), std::string::npos);
  }
  ToolParseResult tail = fed.finish();
  acc += tail.text;
  racc += tail.reasoning;
  const ToolParseResult whole = parse_tool_calls(raw, {"STOP"});
  EXPECT_EQ(acc, whole.text);
  EXPECT_EQ(acc, "hello ");
  EXPECT_EQ(racc, whole.reasoning);
  EXPECT_TRUE(tail.calls.empty());
  EXPECT_TRUE(tail.text.empty());
}

TEST(LFM25Tokenizer, ToolStreamReasoningFragments) {
  using KokkosDNN::LFM25::parse_tool_calls;
  using KokkosDNN::LFM25::ToolCallStreamParser;
  using KokkosDNN::LFM25::ToolParseResult;
  // Length-capped turns can end mid-marker inside an unclosed <think> trace.
  // The whole-text parse strips the trailing fragment; the streaming view
  // must hold (never emit) the same bytes so finish() cannot drop a byte
  // the client has already seen as reasoning_content. Byte-at-a-time here
  // also covers chunk boundaries splitting a marker.
  const std::vector<std::string> frags = {
      "<",          "<|tool_call_st", "<|tool_call_end", "<|im_e",
      "<thi",       "</thi",          "<<",              "<<<",
      "<|tool_call_start|><"};
  for (const std::string& frag : frags) {
    const std::string raw = std::string("<think>hello ") + frag;
    const ToolParseResult whole = parse_tool_calls(raw);
    // No fragment leaks into either view. Only the single longest trailing
    // head is stripped: an earlier '<' followed by '<' is unambiguous
    // content, so "<<"/"<<<" keep all but the last '<' as reasoning.
    std::string expect_reasoning = "hello ";
    if (frag == "<<")
      expect_reasoning = "hello <";
    else if (frag == "<<<")
      expect_reasoning = "hello <<";
    EXPECT_EQ(whole.text, "") << "frag=" << frag;
    EXPECT_EQ(whole.reasoning, expect_reasoning) << "frag=" << frag;
    ToolCallStreamParser fed;
    std::string acc, racc;
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
      // Monotone: already-streamed bytes stay a prefix of the whole-text
      // result (finish() cannot retract them).
      EXPECT_TRUE(whole.text.find(acc) == 0) << "frag=" << frag;
      EXPECT_TRUE(whole.reasoning.find(racc) == 0) << "frag=" << frag;
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    EXPECT_EQ(acc, whole.text) << "frag=" << frag;
    EXPECT_EQ(racc, whole.reasoning) << "frag=" << frag;
    EXPECT_TRUE(tail.calls.empty()) << "frag=" << frag;
  }
  // A disproved fragment head streams once complete (no permanent hold).
  {
    const std::string raw = std::string("<think>a < b</think>hi");
    const ToolParseResult whole = parse_tool_calls(raw);
    EXPECT_EQ(whole.reasoning, "a < b");
    EXPECT_EQ(whole.text, "hi");
    ToolCallStreamParser fed;
    std::string acc, racc;
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
      EXPECT_TRUE(whole.text.find(acc) == 0);
      EXPECT_TRUE(whole.reasoning.find(racc) == 0);
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    EXPECT_EQ(acc, whole.text);
    EXPECT_EQ(racc, whole.reasoning);
  }
}

TEST(LFM25Tokenizer, ToolStreamTrailingLtRun) {
  using KokkosDNN::LFM25::parse_tool_calls;
  using KokkosDNN::LFM25::ToolCallStreamParser;
  using KokkosDNN::LFM25::ToolParseResult;
  // Only the single longest head is held: "hello <<" streams "hello <"
  // with the last '<' held, and a long '<' run never stalls the stream
  // (the fixpoint hold kept the whole run back and rescanned it per push).
  const std::string raw = std::string("hello ") + std::string(2000, '<');
  const ToolParseResult whole = parse_tool_calls(raw);
  EXPECT_EQ(whole.text, std::string("hello ") + std::string(1999, '<'));
  ToolCallStreamParser fed;
  std::string acc, racc;
  size_t fed_n = 0;
  for (size_t p = 0; p < raw.size(); p += 8) {
    const std::string chunk = raw.substr(p, 8);
    acc += fed.push(chunk);
    racc += fed.poll_reasoning();
    fed_n += chunk.size();
    // Prefix of the whole-text result with bounded lag (one head per
    // layer): the stream keeps up instead of stalling to the end.
    EXPECT_TRUE(whole.text.find(acc) == 0);
    EXPECT_LE(fed_n - acc.size(), 64u) << "p=" << p;
    EXPECT_TRUE(racc.empty());
  }
  // finish() is idempotent: the tail is computed, not drained.
  const ToolParseResult tail1 = fed.finish();
  const ToolParseResult tail2 = fed.finish();
  EXPECT_EQ(tail1.text, tail2.text);
  EXPECT_EQ(tail1.reasoning, tail2.reasoning);
  EXPECT_EQ(tail1.calls.size(), tail2.calls.size());
  acc += tail1.text;
  racc += tail1.reasoning;
  EXPECT_EQ(acc, whole.text);
  EXPECT_TRUE(tail1.calls.empty());
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
