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
// <|im_end|> = EOS/stop, <|pad|>, <think> pair, tool-call markers). The
// think pair and tool markers are `special: false` in the real tokenizer,
// exactly as here, so they still encode atomically but survive
// skip_special_tokens decoding.
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
    {"id": 103, "content": "<think>", "special": false},
    {"id": 104, "content": "</think>", "special": false},
    {"id": 105, "content": "<|pad|>", "special": true},
    {"id": 106, "content": "<|tool_list_start|>", "special": true},
    {"id": 107, "content": "<|tool_list_end|>", "special": true},
    {"id": 108, "content": "<|tool_call_start|>", "special": false},
    {"id": 109, "content": "<|tool_call_end|>", "special": false}
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
  EXPECT_EQ(tok.vocab_size(), 110);  // max id 109 (added tokens extend vocab)
  // LFM2.5 convention: bos=<|startoftext|>, eos/stop=<|im_end|>.
  EXPECT_EQ(tok.bos(), 100);
  EXPECT_EQ(tok.eos(), 102);
  EXPECT_EQ(tok.eot(), 102);
  EXPECT_EQ(tok.pad(), 105);
  EXPECT_EQ(tok.im_start(), 101);
  EXPECT_EQ(tok.think_start(), 103);
  EXPECT_EQ(tok.think_end(), 104);
  for (int id : {100, 101, 102, 105, 106, 107})
    EXPECT_TRUE(tok.is_special(id));
  // The think pair and tool-call markers are `special: false` (as in the
  // real tokenizer): atomic in encode, but kept by skip-specials decode so
  // parse_tool_calls sees them.
  for (int id : {103, 104, 108, 109}) EXPECT_FALSE(tok.is_special(id));
  EXPECT_EQ(tok.decode({108, 109}), "<|tool_call_start|><|tool_call_end|>");
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
  // The think pair is atomic too (longest-match added-token split),
  // including as `special: false` entries.
  EXPECT_EQ(tok.encode("<think>hi</think>", false),
            (std::vector<int>{103, 20, 104}));
  // Tool-call markers are atomic single ids as well.
  EXPECT_EQ(tok.encode("a<|tool_call_start|>b<|tool_call_end|>d", false),
            (std::vector<int>{0, 108, 1, 109, 2}));
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

TEST(LFM25Tokenizer, ToolRenderGolden) {
  using KokkosDNN::LFM25::ChatMessage;
  using KokkosDNN::LFM25::ToolCall;
  using KokkosDNN::LFM25::render_lfm25_chatml;
  using KokkosDNN::LFM25::render_tool_call;

  // JSON arguments render Pythonic (official template style).
  EXPECT_EQ(render_tool_call("get_weather", R"({"location":"Boston"})"),
            "get_weather(location='Boston')");
  EXPECT_EQ(render_tool_call("f", "{}"), "f()");
  EXPECT_EQ(render_tool_call("f", ""), "f()");
  EXPECT_EQ(render_tool_call("f", R"({"a":1,"b":true,"c":null})"),
            "f(a=1, b=True, c=None)");
  EXPECT_EQ(render_tool_call("f", "not-json"), "f()");

  const std::vector<std::string> tools = {
      R"({"name":"get_weather","parameters":{"type":"object"}})"};
  // Tools append to an existing system block.
  EXPECT_EQ(
      render_lfm25_chatml({{"system", "Be brief.", ""}, {"user", "Hi", ""}},
                           false, true, "", tools),
      "<|im_start|>system\nBe brief.\n"
      "List of tools: "
      "[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}]"
      "<|im_end|>\n"
      "<|im_start|>user\nHi<|im_end|>\n");
  // Without a system block, a system block is injected for the tools.
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", ""}}, false, true, "", tools),
            "<|im_start|>system\n"
            "List of tools: "
            "[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}]"
            "<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n");
  // No tools: rendering is unchanged (no system injection).
  EXPECT_EQ(render_lfm25_chatml({{"user", "Hi", ""}}, false),
            "<|im_start|>user\nHi<|im_end|>\n");

  // Assistant history with tool calls renders natively; tool results use
  // the tool role block.
  ChatMessage a;
  a.role = "assistant";
  a.content = "Checking.";
  a.tool_calls.push_back(ToolCall{"", "get_weather", "{\"b\":1}"});
  ChatMessage t;
  t.role = "tool";
  t.content = "[{\"b\": 1}]";
  ChatMessage u;
  u.role = "user";
  u.content = "Hi";
  EXPECT_EQ(
      render_lfm25_chatml({u, a, t}, false, true, "", tools),
      "<|im_start|>system\n"
      "List of tools: "
      "[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}]"
      "<|im_end|>\n"
      "<|im_start|>user\nHi<|im_end|>\n"
      "<|im_start|>assistant\nChecking."
      "<|tool_call_start|>[get_weather(b=1)]<|tool_call_end|><|im_end|>\n"
      "<|im_start|>tool\n[{\"b\": 1}]<|im_end|>\n");
}

TEST(LFM25Tokenizer, ToolParse) {
  using KokkosDNN::LFM25::parse_tool_calls;
  using KokkosDNN::LFM25::ParsedToolCalls;

  // Single call with surrounding text: text kept, call extracted as JSON.
  ParsedToolCalls p = parse_tool_calls(
      "Sure <|tool_call_start|>[get_weather(location='Boston')]"
      "<|tool_call_end|> done");
  ASSERT_EQ(p.calls.size(), 1u);
  EXPECT_EQ(p.calls[0].name, "get_weather");
  EXPECT_EQ(p.calls[0].arguments, "{\"location\":\"Boston\"}");
  EXPECT_EQ(p.text, "Sure  done");

  // Multiple calls, mixed Python/JSON values.
  p = parse_tool_calls(
      "<|tool_call_start|>[f(a=1, b=True, c=None, d='x y'), "
      "g(xs=[1, 'a'], o={\"k\": 2})]<|tool_call_end|>");
  ASSERT_EQ(p.calls.size(), 2u);
  EXPECT_EQ(p.calls[0].arguments,
            "{\"a\":1,\"b\":true,\"c\":null,\"d\":\"x y\"}");
  EXPECT_EQ(p.calls[1].name, "g");
  EXPECT_EQ(p.calls[1].arguments, "{\"xs\":[1,\"a\"],\"o\":{\"k\":2}}");
  EXPECT_TRUE(p.text.empty());

  // Empty args, double quotes, and commas inside quoted strings.
  p = parse_tool_calls("<|tool_call_start|>[ping()]<|tool_call_end|>");
  ASSERT_EQ(p.calls.size(), 1u);
  EXPECT_EQ(p.calls[0].arguments, "{}");
  p = parse_tool_calls(
      "<|tool_call_start|>[f(a=\"hi, here\") Nest]<|tool_call_end|>");
  ASSERT_EQ(p.calls.size(), 0u);  // trailing prose invalidates the call
  // Trailing punctuation after the list is kept as reply text.
  p = parse_tool_calls(
      "<|tool_call_start|>[f(a=\"hi, here\")]. Nice.<|tool_call_end|> tail");
  ASSERT_EQ(p.calls.size(), 1u);
  EXPECT_EQ(p.calls[0].arguments, "{\"a\":\"hi, here\"}");
  EXPECT_EQ(p.text, ". Nice. tail");
  p = parse_tool_calls(
      "<|tool_call_start|>[f(a=\"hi, here\") Watts]<|tool_call_end|> "
      "and <|tool_call_start|>[ok(x=1)]<|tool_call_end|>");
  ASSERT_EQ(p.calls.size(), 1u);
  EXPECT_EQ(p.calls[0].name, "ok");
  p = parse_tool_calls(
      "<|tool_call_start|>[f(a=\"hi, here\")]<|tool_call_end|>");
  ASSERT_EQ(p.calls.size(), 1u);
  EXPECT_EQ(p.calls[0].arguments, "{\"a\":\"hi, here\"}");

  // No markers: input passes through unchanged.
  p = parse_tool_calls("just text");
  EXPECT_TRUE(p.calls.empty());
  EXPECT_EQ(p.text, "just text");

  // Malformed blocks are left in place (never corrupt the reply).
  p = parse_tool_calls("a <|tool_call_start|>oops<|tool_call_end|> b");
  EXPECT_TRUE(p.calls.empty());
  EXPECT_EQ(p.text, "a <|tool_call_start|>oops<|tool_call_end|> b");
  p = parse_tool_calls("unterminated <|tool_call_start|>[f(a=1)");
  EXPECT_TRUE(p.calls.empty());
}
