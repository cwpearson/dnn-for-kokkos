// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Parsing unit tests for kokkosdnn_server: exercise the pure OpenAI/JSON
// translation helpers (parse_common, parse_messages, openai_error) plus the
// arch registry and per-backend render paths by including server_main.cpp
// with KOKKOSDNN_SERVER_NO_MAIN. No weights, no tokenizer file, no network,
// no Kokkos runtime init needed (helpers only touch JSON mapping plus tiny
// in-memory configs; the LFM25 render tests use header-only defaults).
//
// Assert-based (no gtest): any failure aborts nonzero. Wired in
// server/CMakeLists.txt behind KokkosDNN_ENABLE_TESTS.

#define KOKKOSDNN_SERVER_NO_MAIN
#include "server_main.cpp"

#include <cassert>
#include <iostream>

#ifdef KOKKOSDNN_SERVER_HAVE_LFM25
#include <KokkosDNN_LFM25Config.hpp>
#include <KokkosDNN_LFM25Tokenizer.hpp>
#endif
#ifdef KOKKOSDNN_SERVER_HAVE_GLIMMER
#include <KokkosDNN_GlimmerTokenizer.hpp>
#endif

namespace {
using KokkosDNN::Server::ChatBackend;
using KokkosDNN::Server::ServerChatMessage;
using KokkosDNN::Server::ServerToolCall;
using json = nlohmann::json;

int g_checks = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    ++g_checks;                                                  \
    if (!(cond)) {                                               \
      std::cerr << "FAIL " << __LINE__ << ": " #cond "\n"; \
      return 1;                                                  \
    }                                                            \
  } while (0)

// Arch-agnostic default stop seeding (stand-in eos id; the real value comes
// from backend->default_stop_ids()).
bool parse_ok(const json& body, ParsedRequest& pr, std::string& err,
              const std::vector<int>& stops = {124900},
              const std::string& model = "lfm2_5") {
  return parse_common(body, stops, model, pr, err);
}

int test_messages() {
  std::vector<ServerChatMessage> out;
  std::string err;
  // Happy path + developer alias.
  CHECK(parse_messages(json::parse(R"({"messages":[
      {"role":"system","content":"sys"},
      {"role":"developer","content":"dev"},
      {"role":"user","content":"hi"},
      {"role":"assistant","content":"yo"}]})"),
                       out, err));
  CHECK(out.size() == 4);
  CHECK(out[0].role == "system" && out[0].content == "sys");
  CHECK(out[1].role == "user" && out[1].content == "dev");
  // Empty / missing / non-array messages.
  CHECK(!parse_messages(json::parse(R"({"messages":[]})"), out, err));
  CHECK(!parse_messages(json::parse(R"({})"), out, err));
  // Bad roles rejected; tool + legacy function alias accepted.
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"robot","content":"x"}]})"), out,
      err));
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"tool","content":"42",
        "tool_call_id":"c1"}]})"),
      out, err));
  CHECK(out.back().role == "tool" && out.back().content == "42" &&
        out.back().tool_call_id == "c1");
  // Tool function name is captured when present (Glimmer renders
  // <tool_output> turns under it).
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"tool","content":"ok",
        "tool_call_id":"c2","name":"f"}]})"),
      out, err));
  CHECK(out.back().name == "f" && out.back().tool_call_id == "c2");
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"tool","content":"ok",
        "name":42}]})"),
      out, err));
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"function","content":"42"}]})"),
      out, err));
  CHECK(out.back().role == "tool");
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"tool","content":42}]})"), out,
      err));
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"tool"}]})"), out, err));
  // Assistant tool_calls: content may be null; args string or object.
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"assistant","content":null,
        "tool_calls":[{"id":"c1","type":"function",
          "function":{"name":"get_weather","arguments":"{\"a\":1}"}}]}]})"),
      out, err));
  CHECK(out.back().tool_calls.size() == 1 &&
        out.back().tool_calls[0].id == "c1" &&
        out.back().tool_calls[0].name == "get_weather" &&
        out.back().tool_calls[0].arguments == "{\"a\":1}" &&
        out.back().content.empty());
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"assistant","content":"x",
        "tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]}]})"),
      out, err));
  CHECK(out.back().tool_calls[0].arguments == "{\"a\":1}");
  // Malformed tool_calls rejected.
  for (const char* f :
       {R"({"messages":[{"role":"assistant","content":"x","tool_calls":[]}]})",
        R"({"messages":[{"role":"assistant","content":"x","tool_calls":[42]}]})",
        R"({"messages":[{"role":"assistant","content":"x",
          "tool_calls":[{"type":"code","function":{"name":"f"}}]}]})",
        R"({"messages":[{"role":"assistant","content":"x",
          "tool_calls":[{"function":{"arguments":"{}"}}]}]})",
        R"({"messages":[{"role":"assistant","content":"x",
          "tool_calls":[{"function":{"name":"","arguments":"{}"}}]}]})",
        R"({"messages":[{"role":"assistant","content":"x",
          "tool_calls":[{"function":{"name":"f","arguments":42}}]}]})",
        R"({"messages":[{"role":"user","content":"x",
          "tool_calls":[{"function":{"name":"f"}}]}]})",
        R"({"messages":[{"role":"assistant","content":null}]})",
        R"({"messages":[{"role":"assistant","content":"x",
          "function_call":{"name":"f"}}]})"}) {
    CHECK(!parse_messages(json::parse(f), out, err));
  }
  // Text-part arrays join; non-text parts rejected.
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"user","content":[
        {"type":"text","text":"a"},{"type":"text","text":"b"}]}]})"),
      out, err));
  CHECK(out.back().content == "ab");
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{}}]}]})"),
      out, err));
  // Non-string content rejected.
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"user","content":42}]})"), out,
      err));
  // Thinking replay: thinking / reasoning / reasoning_content land in
  // ChatMessage.thinking (first non-empty wins); other roles ignore them.
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"assistant","content":"x",
        "reasoning_content":"trace"}]})"),
      out, err));
  CHECK(out.back().thinking == "trace");
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"assistant","content":"x",
        "thinking":"t1","reasoning_content":"t2"}]})"),
      out, err));
  CHECK(out.back().thinking == "t1");
  CHECK(parse_messages(
      json::parse(R"({"messages":[{"role":"user","content":"x",
        "reasoning_content":"ignored"}]})"),
      out, err));
  CHECK(out.back().thinking.empty());
  return 0;
}

int test_common() {
  ParsedRequest pr;
  std::string err;
  // Minimal body passes with defaults.
  CHECK(parse_ok(json::parse(R"({})"), pr, err));
  CHECK(pr.req_max_tokens == -1 && !pr.has_seed && !pr.stream);
  CHECK(pr.opts.stop_ids == std::vector<int>{124900});
  // Unknown model id rejected; absent ok.
  CHECK(!parse_ok(json::parse(R"({"model":"other"})"), pr, err));
  CHECK(parse_ok(json::parse(R"({"model":"lfm2_5"})"), pr, err));
  // Unsupported features rejected (legacy function calling retired).
  for (const char* f :
       {R"({"n":2})", R"({"functions":[]})",
        R"({"function_call":"auto"})"}) {
    CHECK(!parse_ok(json::parse(f), pr, err));
  }
  // tools accepted + validated; empty array = no-op.
  CHECK(parse_ok(json::parse(R"({"tools":[]})"), pr, err));
  CHECK(!pr.tools_active && pr.tools_json.empty());
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"type":"function",
        "function":{"name":"get_weather","description":"d",
          "parameters":{"type":"object"},"strict":false}}]})"),
      pr, err));
  CHECK(pr.tools_active);
  CHECK(pr.tools_json.find("get_weather") != std::string::npos);
  for (const char* f :
       {R"({"tools":{}})",
        R"({"tools":[42]})",
        R"({"tools":[{"type":"code","function":{"name":"f"}}]})",
        R"({"tools":[{"function":{}}]})",
        R"({"tools":[{"function":{"name":""}}]})",
        R"({"tools":[{"function":{"name":"f","parameters":[]}}]})",
        R"({"tools":[{"function":{"name":"f","strict":"yes"}}]})"}) {
    CHECK(!parse_ok(json::parse(f), pr, err));
  }
  // tool_choice: none/auto/required/named + validation.
  pr = ParsedRequest{};
  CHECK(parse_ok(json::parse(R"({"tool_choice":"auto"})"), pr, err));
  CHECK(!pr.tools_active && pr.tool_directive.empty());
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"function":{"name":"f"}}],
        "tool_choice":"none"})"),
      pr, err));
  CHECK(!pr.tools_active);
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"function":{"name":"f"}}],
        "tool_choice":"required"})"),
      pr, err));
  CHECK(pr.tools_active && !pr.tool_directive.empty());
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"function":{"name":"f"}}],
        "tool_choice":{"type":"function","function":{"name":"f"}}})"),
      pr, err));
  CHECK(pr.tool_directive.find("f") != std::string::npos);
  pr = ParsedRequest{};
  for (const char* f :
       {R"({"tool_choice":"sometimes"})", R"({"tool_choice":42})",
        R"({"tool_choice":{"type":"code","function":{"name":"f"}}})",
        R"({"tool_choice":{"function":{}}})",
        R"({"tool_choice":{"function":{"name":""}}})",
        R"({"tool_choice":{"function":{"name":"f"}}})",
        R"({"tool_choice":"required"})"}) {
    CHECK(!parse_ok(json::parse(f), pr, err));
  }
  // parallel_tool_calls must be boolean.
  CHECK(parse_ok(json::parse(R"({"parallel_tool_calls":false})"), pr,
                 err));
  CHECK(!pr.parallel_tool_calls);
  CHECK(!parse_ok(json::parse(R"({"parallel_tool_calls":"yes"})"), pr,
                  err));
  // response_format: text/json_object/json_schema accepted + validated.
  CHECK(parse_ok(json::parse(R"({"response_format":{"type":"text"}})"), pr,
                 err));
  CHECK(!pr.json_mode);
  CHECK(parse_ok(json::parse(R"({"response_format":{"type":"json_object"}})"),
                 pr, err));
  CHECK(pr.json_mode && !pr.json_directive.empty());
  CHECK(parse_ok(
      json::parse(R"({"response_format":{"type":"json_schema",
        "json_schema":{"type":"object"}}})"),
      pr, err));
  CHECK(pr.json_mode);
  for (const char* f :
       {R"({"response_format":{"type":"json_schema"}})",
        R"({"response_format":{"type":"yaml"}})",
        R"({"response_format":{}})",
        R"({"response_format":"json"})"}) {
    CHECK(!parse_ok(json::parse(f), pr, err));
  }
  CHECK(!pr.tools_active);
  CHECK(parse_ok(json::parse(R"({"n":1})"), pr, err));  // n=1 ok
  // Sampling mapping.
  CHECK(parse_ok(
      json::parse(R"({"temperature":0.7,"top_p":0.9,"top_k":40,"min_p":0.05,
        "repetition_penalty":1.1,"frequency_penalty":0.5,"presence_penalty":-0.2,
        "logit_bias":{"100":2.0,"200":-1.0},
        "logprobs":true,"top_logprobs":3,"seed":123,"stream":true,
        "max_tokens":64})"),
      pr, err));
  CHECK(pr.opts.sampling.temperature == 0.7f);
  CHECK(pr.opts.sampling.top_p == 0.9f);
  CHECK(pr.opts.sampling.top_k == 40);
  CHECK(pr.opts.sampling.min_p == 0.05f);
  CHECK(pr.opts.sampling.repetition_penalty == 1.1f);
  CHECK(pr.opts.sampling.frequency_penalty == 0.5f);
  CHECK(pr.opts.sampling.presence_penalty == -0.2f);
  CHECK(pr.opts.sampling.logit_bias.size() == 2);
  CHECK(pr.opts.sampling.logprobs == 3);
  CHECK(pr.has_seed && pr.seed == 123u);
  CHECK(pr.stream);
  CHECK(pr.req_max_tokens == 64);
  // logprobs:true without count defaults to 5; integer form passes through.
  CHECK(parse_ok(json::parse(R"({"logprobs":true})"), pr, err));
  CHECK(pr.opts.sampling.logprobs == 5);
  CHECK(parse_ok(json::parse(R"({"logprobs":2})"), pr, err));
  CHECK(pr.opts.sampling.logprobs == 2);
  // stop mapping: numeric strings -> ids, rest -> strings.
  CHECK(parse_ok(json::parse(R"({"stop":["<|im_end|>","42"]})"), pr,
                 err));
  CHECK(pr.opts.stop_strings.size() == 1);
  CHECK(pr.opts.stop_strings[0] == "<|im_end|>");
  CHECK(pr.opts.stop_ids.size() == 2);
  CHECK(pr.opts.stop_ids[0] == 124900 && pr.opts.stop_ids[1] == 42);
  CHECK(parse_ok(json::parse(R"({"stop":"halt"})"), pr, err));
  CHECK(pr.opts.stop_strings == std::vector<std::string>{"halt"});
  // Bad types rejected.
  CHECK(!parse_ok(json::parse(R"({"temperature":"hot"})"), pr, err));
  CHECK(!parse_ok(json::parse(R"({"seed":1.5})"), pr, err));
  CHECK(!parse_ok(json::parse(R"({"max_tokens":0})"), pr, err));
  CHECK(!parse_ok(json::parse(R"({"stop":[42.5]})"), pr, err));
  CHECK(!parse_ok(json::parse(R"({"logit_bias":{"x":1.0}})"), pr, err));
  // max_completion_tokens wins over max_tokens.
  CHECK(parse_ok(
      json::parse(R"({"max_tokens":10,"max_completion_tokens":20})"), pr,
      err));
  CHECK(pr.req_max_tokens == 20);
  // Unknown fields ignored.
  CHECK(parse_ok(json::parse(R"({"bogus_field":[1,2,3]})"), pr, err));
  return 0;
}

#ifdef KOKKOSDNN_SERVER_HAVE_LFM25
int test_lfm25_tool_wire() {
  using KokkosDNN::LFM25::args_json_to_pythonic;
  using KokkosDNN::LFM25::args_pythonic_to_json;
  using KokkosDNN::LFM25::ChatMessage;
  using KokkosDNN::LFM25::encode_chatml_with_tools;
  using KokkosDNN::LFM25::parse_tool_call_block;
  using KokkosDNN::LFM25::parse_tool_calls;
  using KokkosDNN::LFM25::render_lfm25_chatml;
  using KokkosDNN::LFM25::render_lfm25_chatml_with_tools;
  using KokkosDNN::LFM25::ToolCall;
  using KokkosDNN::LFM25::ToolCallStreamParser;
  using KokkosDNN::LFM25::ToolParseResult;
  // Argument conversion round-trips.
  CHECK(args_pythonic_to_json("") == "{}");
  CHECK(args_pythonic_to_json(
            "location=\"SF\", n=3, ok=True, m=None") ==
        "{\"location\":\"SF\",\"n\":3,\"ok\":true,\"m\":null}");
  CHECK(args_pythonic_to_json("unit=celsius") == "{\"unit\":\"celsius\"}");
  const std::string js =
      args_pythonic_to_json("a='q', b=[1, \"x\", None]");
  CHECK(args_json_to_pythonic(js) == "a=\"q\", b=[1, \"x\", None]");
  CHECK(args_json_to_pythonic("") == "");
  CHECK(args_json_to_pythonic("{}") == "");
  CHECK(args_json_to_pythonic("oops") == "");
  // Block + full-text parsing.
  {
    auto v = parse_tool_call_block("[get_weather(location=\"SF\")]");
    CHECK(v.size() == 1 && v[0].name == "get_weather" &&
          v[0].arguments == "{\"location\":\"SF\"}");
  }
  CHECK(parse_tool_call_block("oops-no-parens").empty());
  {
    auto m1 = parse_tool_call_block("[f(a=1, b=2)");
    CHECK(m1.size() == 1 && m1[0].name == "f");
    auto m2 = parse_tool_call_block("[f(a=1), g(b=2)");
    CHECK(m2.size() == 2 && m2[0].name == "f" && m2[1].name == "g");
    auto m3 = parse_tool_call_block("[f(a=1), g(b=2");
    CHECK(m3.size() == 1 && m3[0].name == "f");
    CHECK(parse_tool_call_block("[f(a=").empty());
  }
  {
    const std::string raw =
        std::string("Hi ") +
        KokkosDNN::LFM25::kToolCallStartToken +
        "[get_weather(location=\"SF\")]" +
        KokkosDNN::LFM25::kToolCallEndToken + " bye" +
        KokkosDNN::LFM25::kEosToken + "junk";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.calls.size() == 1 && r.calls[0].name == "get_weather" &&
          r.calls[0].arguments == "{\"location\":\"SF\"}");
    CHECK(r.text == "Hi  bye");
  }
  // Unclosed trailing call block (length-capped): recovered as calls
  // instead of leaking markup into the display text. Mid-list cuts
  // (trailing ']' missing) recover the complete calls.
  {
    const std::string raw =
        std::string("a") + KokkosDNN::LFM25::kToolCallStartToken + "[f(x=1)]";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.calls.size() == 1 && r.calls[0].name == "f");
    CHECK(r.text == "a");
  }
  {
    const std::string raw = std::string("a") +
                            KokkosDNN::LFM25::kToolCallStartToken +
                            "[f(a=1), g(b=2)";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.calls.size() == 2 && r.calls[0].name == "f" &&
          r.calls[1].name == "g");
    CHECK(r.text == "a");
  }
  // Unparseable trailing bytes stay text, never a call.
  {
    const std::string raw =
        std::string("a") + KokkosDNN::LFM25::kToolCallStartToken + "oops";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.calls.empty() && r.text == "aoops");
  }
  // Render -> parse round trip with tools injected + history.
  {
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
    tool_msg.tool_call_id = "c1";
    const std::string tools =
        "[{\"type\":\"function\",\"function\":{\"name\":"
        "\"get_weather\"}}]";
    const std::string rendered = render_lfm25_chatml_with_tools(
        {a, tool_msg}, tools, "", true, true);
    CHECK(rendered.find("List of tools: " + tools) != std::string::npos);
    CHECK(rendered.find(
              std::string(
                  KokkosDNN::LFM25::kToolCallStartToken) +
              "[get_weather(location=\"SF\")]" +
              KokkosDNN::LFM25::kToolCallEndToken) != std::string::npos);
    CHECK(rendered.find("<|im_start|>tool\n{\"temp\": 72}<|im_end|>") !=
          std::string::npos);
    // History tool call re-parses losslessly.
    const size_t s = rendered.find(KokkosDNN::LFM25::kToolCallStartToken);
    const size_t e = rendered.find(KokkosDNN::LFM25::kToolCallEndToken);
    CHECK(s != std::string::npos && e != std::string::npos);
    const ToolParseResult back = parse_tool_calls(rendered.substr(s));
    CHECK(back.calls.size() == 1 &&
          back.calls[0].arguments == "{\"location\":\"SF\"}");
  }
  // No-tool render matches the plain renderer byte-for-byte.
  {
    const std::vector<ChatMessage> msgs = {{{"system", "be bold", "", {}, ""},
                                            {"user", "hi there", "", {}, ""}}};
    CHECK(render_lfm25_chatml_with_tools(msgs, "", "", true, true) ==
          KokkosDNN::LFM25::render_lfm25_chatml(msgs, true, true));
  }
  // Incremental stream parser agrees with whole-text parse, even fed
  // one byte at a time (markers split across pushes included).
  {
    const std::string raw =
        std::string("Hi there ") +
        KokkosDNN::LFM25::kToolCallStartToken +
        "[f(a=1), g(x=\"y\")]" + KokkosDNN::LFM25::kToolCallEndToken +
        " done" + KokkosDNN::LFM25::kEosToken;
    ToolCallStreamParser fed;
    std::string acc;
    for (char ch : raw) acc += fed.push(std::string(1, ch));
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    const ToolParseResult whole = parse_tool_calls(raw);
    CHECK(acc == whole.text);
    CHECK(tail.calls.size() == 2 && tail.calls[0].name == "f" &&
          tail.calls[1].arguments == "{\"x\":\"y\"}");
  }
  // Plain text streams through untouched (utf-8 safe incl. emoji).
  {
    ToolCallStreamParser fed;
    const std::string snow = "snow \xe2\x9b\x84 end";
    std::string acc;
    for (size_t i = 0; i < snow.size(); ++i)
      acc += fed.push(snow.substr(i, 1));
    acc += fed.finish().text;
    CHECK(acc == snow);
    CHECK(fed.finish().calls.empty());
    CHECK(fed.finish().reasoning.empty());
  }
  // Thinking blocks split into reasoning, never into display.
  {
    const std::string raw =
        std::string(KokkosDNN::LFM25::kThinkStartToken) + "hmm, let me see" +
        KokkosDNN::LFM25::kThinkEndToken + "The answer" +
        KokkosDNN::LFM25::kToolCallStartToken + "[f(x=1)]" +
        KokkosDNN::LFM25::kToolCallEndToken;
    const ToolParseResult whole = parse_tool_calls(raw);
    CHECK(whole.reasoning == "hmm, let me see");
    CHECK(whole.text == "The answer");
    CHECK(whole.calls.size() == 1 && whole.calls[0].name == "f");
    // Streaming agrees byte-for-byte, reasoning deltas included.
    ToolCallStreamParser fed;
    std::string acc, racc;
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(acc == whole.text);
    CHECK(racc == whole.reasoning);
    CHECK(tail.calls.size() == 1 && tail.calls[0].name == "f");
    CHECK(tail.calls[0].arguments == "{\"x\":1}");
  }
  // Unclosed trailing think at turn start: the rest is reasoning.
  {
    const std::string raw =
        std::string(KokkosDNN::LFM25::kThinkStartToken) + "thinking";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.reasoning == "thinking");
    CHECK(r.text.empty());
    ToolCallStreamParser fed;
    std::string acc, racc;
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(acc == r.text);
    CHECK(racc == r.reasoning);
  }
  // Implicit leading trace: the generation prompt already opened the
  // thinker, so a turn-2 completion (after a tool result) carries only
  // "</think>" and would otherwise leak the whole trace into `content`.
  // The server constructs every parser with start_in_reasoning=true.
  {
    const std::string raw =
        std::string("second thoughts") + KokkosDNN::LFM25::kThinkEndToken +
        KokkosDNN::LFM25::kToolCallStartToken + "[g(y=2)]" +
        KokkosDNN::LFM25::kToolCallEndToken;
    ToolCallStreamParser fed({}, /*start_in_reasoning=*/true);
    std::string acc, racc;
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(racc == "second thoughts");
    CHECK(acc.empty());
    CHECK(tail.calls.size() == 1 && tail.calls[0].name == "g");
  }
  // A mid-turn <think> is literal content (the template emits thinking
  // first), so answers that merely mention the tag are never swallowed.
  {
    const std::string raw =
        std::string("pre ") + KokkosDNN::LFM25::kThinkStartToken + "thinking";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.reasoning.empty());
    CHECK(r.text == raw);
  }
  // System suffix lands in the LFM25 system block; directives combine.
  {
    ParsedRequest pr;
    pr.tools_json = "[t]";
    pr.tool_directive = "MUST-A.";
    pr.json_mode = true;
    pr.json_directive = "JSON-B.";
    CHECK(make_system_suffix(pr) == "MUST-A.\nJSON-B.");
    const std::string r = render_lfm25_chatml_with_tools(
        {{{"user", "hi", "", {}, ""}}}, pr.tools_json, make_system_suffix(pr), true,
        true);
    CHECK(r.find("List of tools: [t]\nMUST-A.\nJSON-B.") !=
          std::string::npos);
  }
  return 0;
}
#endif  // KOKKOSDNN_SERVER_HAVE_LFM25

#ifdef KOKKOSDNN_SERVER_HAVE_LFM25
// Tiny-config LFM25 render check (pure string mapping, no weights): the
// tools render path matches the plain renderer byte-for-byte without tools.
int test_lfm25_render_tiny() {
  const std::vector<KokkosDNN::LFM25::ChatMessage> msgs = {
      {{"system", "be bold", "", {}, ""}, {"user", "hi there", "", {}, ""}}};
  CHECK(KokkosDNN::LFM25::render_lfm25_chatml_with_tools(msgs, "", "", true,
                                                         true) ==
        KokkosDNN::LFM25::render_lfm25_chatml(msgs, true, true));
  return 0;
}
#endif  // KOKKOSDNN_SERVER_HAVE_LFM25

#ifdef KOKKOSDNN_SERVER_HAVE_GLIMMER
// Glimmer render check (pure string mapping, no weights): one user turn
// renders the Muse-style framing with an open assistant header.
int test_glimmer_render_tiny() {
  const std::vector<KokkosDNN::Glimmer::ChatMessage> msgs =
      KokkosDNN::Glimmer::make_chat_messages("hi", "", "user");
  const std::string r =
      KokkosDNN::Glimmer::render_chat_template(msgs, true);
  CHECK(r.find("<|begin_of_text|>") == 0);
  CHECK(r.find("<|start|>user<|message|>hi<|eot|>") != std::string::npos);
  CHECK(r.rfind("<|start|>assistant<|message|>") ==
        r.size() - std::string("<|start|>assistant<|message|>").size());
  return 0;
}
#endif  // KOKKOSDNN_SERVER_HAVE_GLIMMER

#ifdef KOKKOSDNN_SERVER_HAVE_GLIMMER
// Glimmer ATEM tool wire: argument conversion, invoke blocks, channel
// render/parse round trips, and streaming agreement.
int test_glimmer_tool_wire() {
  using KokkosDNN::Glimmer::args_json_to_atem;
  using KokkosDNN::Glimmer::atem_invoke_block;
  using KokkosDNN::Glimmer::atem_value_to_json;
  using KokkosDNN::Glimmer::ChatMessage;
  using KokkosDNN::Glimmer::default_stop_strings;
  using KokkosDNN::Glimmer::parse_atem_invokes;
  using KokkosDNN::Glimmer::parse_tool_calls;
  using KokkosDNN::Glimmer::render_glimmer_chatml_with_tools;
  using KokkosDNN::Glimmer::StreamDeltas;
  using KokkosDNN::Glimmer::ToolCall;
  using KokkosDNN::Glimmer::ToolCallStreamParser;
  using KokkosDNN::Glimmer::ToolParseResult;
  // The turn terminator is <|eot|> only: <|eom|> ends a message while the
  // turn continues (reasoning channel), so it must not stop the turn.
  CHECK(default_stop_strings() == std::vector<std::string>{"<|eot|>"});
  // ATEM value conversion: JSON passes through, bare words become strings.
  CHECK(atem_value_to_json("123") == "123");
  CHECK(atem_value_to_json("-1.5e3") == "-1.5e3");
  CHECK(atem_value_to_json("true") == "true");
  CHECK(atem_value_to_json("null") == "null");
  CHECK(atem_value_to_json("{\"a\":1}") == "{\"a\":1}");
  CHECK(atem_value_to_json("[1, \"x\"]") == "[1, \"x\"]");
  CHECK(atem_value_to_json("\"s\"") == "\"s\"");
  CHECK(atem_value_to_json("Tokyo") == "\"Tokyo\"");
  CHECK(atem_value_to_json("") == "\"\"");
  // args object -> ATEM params.
  CHECK(args_json_to_atem("") == "");
  CHECK(args_json_to_atem("{}") == "");
  CHECK(args_json_to_atem("oops") == "");
  {
    const std::string p =
        args_json_to_atem("{\"city\":\"Tokyo\",\"n\":3,\"ok\":true}");
    CHECK(p.find("<atem:parameter name=\"city\">Tokyo</atem:parameter>") !=
          std::string::npos);
    CHECK(p.find("<atem:parameter name=\"n\">3</atem:parameter>") !=
          std::string::npos);
    CHECK(p.find("<atem:parameter name=\"ok\">true</atem:parameter>") !=
          std::string::npos);
  }
  // Nested JSON values round-trip verbatim.
  {
    const std::string args = "{\"a\":{\"b\":[1,2]},\"s\":\"x y\"}";
    const std::string p = args_json_to_atem(args);
    CHECK(p.find("<atem:parameter name=\"a\">{\"b\":[1,2]}</atem:parameter>") !=
          std::string::npos);
    CHECK(p.find("<atem:parameter name=\"s\">x y</atem:parameter>") !=
          std::string::npos);
  }
  // Invoke block shape + parse back.
  {
    ToolCall c;
    c.name = "get_weather";
    c.arguments = "{\"city\":\"Tokyo\"}";
    const std::string b = atem_invoke_block(c);
    CHECK(b.find("<atem:invoke name=\"get_weather\">") != std::string::npos);
    const std::vector<ToolCall> v = parse_atem_invokes(b);
    CHECK(v.size() == 1 && v[0].name == "get_weather" &&
          v[0].arguments == "{\"city\":\"Tokyo\"}");
  }
  // Unclosed trailing invoke: dropped, never a call.
  CHECK(parse_atem_invokes("x <atem:invoke name=\"f\">").empty());
  // Full completion parse: reasoning + tool + (no) display channels.
  {
    const std::string raw =
        std::string(" to=self<|message|>let me check<|eom|>") +
        "<|start|>assistant to=get_weather<|message|>" +
        "<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n"
        "<atem:parameter name=\"city\">Tokyo</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls><|eot|>";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.calls.size() == 1 && r.calls[0].name == "get_weather");
    CHECK(r.calls[0].arguments == "{\"city\":\"Tokyo\"}");
    CHECK(r.reasoning == "let me check");
    CHECK(r.text.empty());
  }
  // Plain reply channel.
  {
    const ToolParseResult r =
        parse_tool_calls(" to=user<|message|>Hi there<|eot|>junk");
    CHECK(r.text == "Hi there");
    CHECK(r.reasoning.empty() && r.calls.empty());
  }
  // No channels at all: plain display, cut at the turn end.
  {
    const ToolParseResult r = parse_tool_calls("Hello<|eot|>junk");
    CHECK(r.text == "Hello" && r.calls.empty());
  }
  // Tool result + reasoning history render in the upstream framing, and
  // the tool name resolves through the matching tool_call_id.
  {
    ToolCall c;
    c.id = "c1";
    c.name = "get_weather";
    c.arguments = "{\"city\":\"Tokyo\"}";
    ChatMessage sys;
    sys.role = "system";
    sys.content = "Be brief.";
    ChatMessage a;
    a.role = "assistant";
    a.thinking = "check weather";
    a.tool_calls.push_back(c);
    ChatMessage tool_msg;
    tool_msg.role = "tool";
    tool_msg.content = "{\"temp\": 72}";
    tool_msg.tool_call_id = "c1";
    const std::string tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"d\",\"parameters\":{\"type\":\"object\"}}}]";
    const std::string r = render_glimmer_chatml_with_tools(
        {sys, a, tool_msg}, tools, "", true);
    CHECK(r.find("Reasoning strength: high.") != std::string::npos);
    CHECK(r.find("# Valid recipients: \"self\", \"get_weather.*\", "
                 "\"user\".") != std::string::npos);
    CHECK(r.find("In this environment you have access to a set of tools") !=
          std::string::npos);
    CHECK(r.find("<|start|>assistant to=self<|message|>check "
                 "weather<|eom|>") != std::string::npos);
    CHECK(r.find("<|start|>assistant to=get_weather<|message|>") !=
          std::string::npos);
    CHECK(r.find("<tool_output name=\"get_weather\">\n{\"temp\": "
                 "72}\n</tool_output>") != std::string::npos);
    // History tool call re-parses losslessly.
    const size_t s = r.find("<|start|>assistant to=get_weather");
    CHECK(s != std::string::npos);
    const ToolParseResult back = parse_tool_calls(r.substr(s));
    CHECK(back.calls.size() == 1 &&
          back.calls[0].arguments == "{\"city\":\"Tokyo\"}");
  }
  // Unclosed trailing invoke test block.
  {
    const std::string raw = std::string(" to=self<|message|>hmm<|eom|>");
    const ToolParseResult whole = parse_tool_calls(raw);
    CHECK(whole.calls.empty());
    CHECK(whole.reasoning == "hmm");
    CHECK(whole.text.empty());
  }
  // Streaming agrees with the whole-text parse, even fed one byte at a
  // time (headers, markers and stops split across pushes included).
  {
    const std::string raw =
        std::string(" to=self<|message|>thinking...<|eom|>") +
        "<|start|>assistant to=user<|message|>The answer is 42<|eot|>";
    const ToolParseResult whole = parse_tool_calls(raw);
    CHECK(whole.text == "The answer is 42");
    CHECK(whole.reasoning == "thinking...");
    ToolCallStreamParser fed;
    std::string acc, racc;
    for (char ch : raw) {
      StreamDeltas d = fed.feed(std::string(1, ch));
      acc += d.display;
      racc += d.reasoning;
      CHECK(whole.text.find(acc) == 0);
      CHECK(whole.reasoning.find(racc) == 0);
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(acc == whole.text);
    CHECK(racc == whole.reasoning);
    CHECK(tail.calls.empty());
    CHECK(fed.finish().text == tail.text);
    (void)tail;
  }
  return 0;
}
#endif

int test_backends() {
  using KokkosDNN::Server::available_backends;
  using KokkosDNN::Server::create_backend;
  using KokkosDNN::Server::normalize_arch_name;
  // Registry: both arch names resolve (real or stub backend), unknown does
  // not. No weights/Kokkos needed: construction allocates nothing.
  {
    const std::vector<std::string> names = available_backends();
    bool lfm = false, gli = false;
    for (const auto& n : names) {
      lfm = lfm || n == "lfm2.5";
      gli = gli || n == "glimmer";
    }
    CHECK(lfm && gli);
  }
  {
    std::unique_ptr<ChatBackend> lfm = create_backend("lfm2.5");
    std::unique_ptr<ChatBackend> gli = create_backend("glimmer");
    CHECK(lfm && gli);
    CHECK(!create_backend("llama"));
    CHECK(lfm->arch() == "lfm2.5");
    CHECK(gli->arch() == "glimmer");
    CHECK(lfm->default_model_id() == "lfm2_5");
    CHECK(gli->default_model_id() == "glimmer-30b");
    CHECK(lfm->supports_tools());
    CHECK(gli->supports_tools());
    // Load-free defaults (no weights/tokenizer): parse_common seeds from
    // these, so they must be sane before load().
    CHECK(!lfm->default_stop_ids().empty());
    CHECK(!lfm->default_stop_strings().empty());
    CHECK(!gli->default_stop_ids().empty());
    CHECK(!gli->default_stop_strings().empty());
  }
  // CLI arch aliases (case-insensitive, driver-style).
  CHECK(normalize_arch_name("lfm2.5") == "lfm2.5");
  CHECK(normalize_arch_name("LFM2_5") == "lfm2.5");
  CHECK(normalize_arch_name("lfm25") == "lfm2.5");
  CHECK(normalize_arch_name("glimmer") == "glimmer");
  CHECK(normalize_arch_name("Glimmer-30B") == "glimmer");
  CHECK(normalize_arch_name("muse-glimmer") == "glimmer");
  CHECK(normalize_arch_name("muse-glimmer-30b") == "glimmer");
  CHECK(normalize_arch_name("darkc0de-muse-glimmer-30b") == "glimmer");
  CHECK(normalize_arch_name("llama") == "");
  CHECK(normalize_arch_name("") == "");
  // Tools gate: both archs accept (LFM25 native grammar, Glimmer ATEM).
  {
    std::unique_ptr<ChatBackend> lfm = create_backend("lfm2.5");
    std::unique_ptr<ChatBackend> gli = create_backend("glimmer");
    ParsedRequest pr;
    pr.tools_active = true;
    pr.tools_json = "[{}]";
    std::string err;
    CHECK(check_tools_support(*lfm, pr, err));
    CHECK(check_tools_support(*gli, pr, err));
    pr.tools_active = false;
    pr.tools_json.clear();
    ServerToolCall c;
    c.name = "f";
    ServerChatMessage m;
    m.role = "assistant";
    m.content = "x";
    m.tool_calls.push_back(c);
    pr.messages.push_back(m);
    CHECK(check_tools_support(*lfm, pr, err));
    CHECK(check_tools_support(*gli, pr, err));
    pr.messages.clear();
    CHECK(check_tools_support(*gli, pr, err));
  }
  return 0;
}

int test_tool_helpers() {
  // System suffix assembly (arch-agnostic) + tool_calls_json shape +
  // fence stripping. No model code involved.
  {
    ParsedRequest pr;
    pr.tools_json = "[t]";
    pr.tool_directive = "MUST-A.";
    pr.json_mode = true;
    pr.json_directive = "JSON-B.";
    CHECK(make_system_suffix(pr) == "MUST-A.\nJSON-B.");
  }
  // tool_calls_json shape: fresh ids, function envelope, {} default.
  // (Default-constructed ServerState performs no Kokkos allocation, so
  // no runtime init is needed; only the id counter is touched.)
  {
    ServerState st;
    ServerToolCall c;
    c.name = "f";
    c.arguments = "";
    const json tcs = tool_calls_json(st, {c});
    CHECK(tcs.size() == 1 && tcs[0]["type"] == "function" &&
          tcs[0]["function"]["name"] == "f" &&
          tcs[0]["function"]["arguments"] == "{}" &&
          tcs[0]["id"].get<std::string>().rfind("call-", 0) == 0);
  }
  // strip_json_fences: fences removed, plain text untouched.
  CHECK(strip_json_fences("{\"a\":1}") == "{\"a\":1}");
  CHECK(strip_json_fences("```json\n{\"a\":1}\n```") == "{\"a\":1}");
  CHECK(strip_json_fences("  ```\n[1, 2]\n```  ") == "[1, 2]");
  CHECK(strip_json_fences("no fences") == "no fences");
  return 0;
}

int test_assemble() {
  // (Default-constructed ServerState performs no Kokkos allocation, so
  // no runtime init is needed; only the id counter is touched.)
  ServerState st;
  ParsedRequest pr;  // plain request: no tools, no JSON mode
  {
    AssembledMessage m =
        assemble_chat_message(st, pr, "hi", "", {}, "stop");
    CHECK(m.message["role"] == "assistant");
    CHECK(m.message["content"] == "hi");
    CHECK(!m.message.contains("tool_calls"));
    CHECK(!m.message.contains("reasoning_content"));
    CHECK(m.finish == "stop");
  }
  // Reasoning travels as reasoning_content, never inside content.
  {
    AssembledMessage m =
        assemble_chat_message(st, pr, "answer", "trace", {}, "stop");
    CHECK(m.message["content"] == "answer");
    CHECK(m.message["reasoning_content"] == "trace");
    CHECK(m.finish == "stop");
  }
  // Tools, no calls: parser text used verbatim, session finish kept.
  pr.tools_active = true;
  {
    AssembledMessage m =
        assemble_chat_message(st, pr, "thinking...", "", {}, "length");
    CHECK(m.message["content"] == "thinking...");
    CHECK(!m.message.contains("tool_calls"));
    CHECK(m.finish == "length");
  }
  // Tools with calls: tool_calls payload, finish tool_calls.
  {
    ServerToolCall c;
    c.name = "get_weather";
    c.arguments = "{\"location\":\"SF\"}";
    AssembledMessage m =
        assemble_chat_message(st, pr, "checking", "", {c}, "stop");
    CHECK(m.message["content"] == "checking");
    CHECK(m.message["tool_calls"].size() == 1);
    CHECK(m.message["tool_calls"][0]["type"] == "function");
    CHECK(m.message["tool_calls"][0]["function"]["name"] == "get_weather");
    CHECK(m.message["tool_calls"][0]["function"]["arguments"] ==
          "{\"location\":\"SF\"}");
    CHECK(m.message["tool_calls"][0]["id"].get<std::string>().rfind(
              "call-", 0) == 0);
    CHECK(m.finish == "tool_calls");
  }
  // Empty display text with calls -> content null.
  {
    ServerToolCall c;
    c.name = "f";
    c.arguments = "";
    AssembledMessage m = assemble_chat_message(st, pr, "", "", {c}, "stop");
    CHECK(m.message["content"].is_null());
    CHECK(m.message["tool_calls"][0]["function"]["arguments"] == "{}");
    CHECK(m.finish == "tool_calls");
  }
  // parallel_tool_calls=false keeps the first call only.
  pr.parallel_tool_calls = false;
  {
    ServerToolCall a, b;
    a.name = "a";
    a.arguments = "{}";
    b.name = "b";
    b.arguments = "{}";
    AssembledMessage m =
        assemble_chat_message(st, pr, "t", "", {a, b}, "stop");
    CHECK(m.message["tool_calls"].size() == 1);
    CHECK(m.message["tool_calls"][0]["function"]["name"] == "a");
  }
  pr.parallel_tool_calls = true;
  // json_mode strips fences from content (calls or not).
  pr.json_mode = true;
  {
    AssembledMessage m =
        assemble_chat_message(st, pr, "```json\n{\"a\":1}\n```", "", {},
                              "stop");
    CHECK(m.message["content"] == "{\"a\":1}");
  }
  pr.json_mode = false;
  pr.tools_active = false;
  // Streaming delta shape: index/id/type/function per call.
  {
    ServerToolCall c;
    c.name = "f";
    c.arguments = "{\"x\":2}";
    const json d = tool_call_delta(0, "call-9", c);
    CHECK(d["tool_calls"].size() == 1);
    CHECK(d["tool_calls"][0]["index"] == 0);
    CHECK(d["tool_calls"][0]["id"] == "call-9");
    CHECK(d["tool_calls"][0]["function"]["arguments"] == "{\"x\":2}");
  }
  return 0;
}

int test_cli_glimmer_flags() {
  // Weight-free CLI checks: --kv-*/--prefill-chunk parse into CliOptions
  // with example-identical semantics (dense default, last explicit wins).
  auto run = [](const std::vector<std::string>& args, CliOptions& out) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (const auto& s : args) argv.push_back(const_cast<char*>(s.c_str()));
    return parse_cli(static_cast<int>(argv.size()), argv.data(), out);
  };
  {
    CliOptions o;
    CHECK(run({"prog"}, o) == 0);
    CHECK(o.kv_quant_override == -1);
    CHECK(o.prefill_chunk_override == -1);
    CHECK(o.max_seq_len == 0);
    CHECK(!o.prefix_cache_host);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--kv-int8"}, o) == 0);
    CHECK(o.kv_quant_override == 1);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--kv-fp"}, o) == 0);
    CHECK(o.kv_quant_override == 0);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--kv-quant", "int8"}, o) == 0);
    CHECK(o.kv_quant_override == 1);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--kv-quant", "KV_INT8"}, o) == 0);
    CHECK(o.kv_quant_override == 1);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--kv-quant", "none"}, o) == 0);
    CHECK(o.kv_quant_override == 0);
  }
  // Last explicit source wins (mirrors the glimmer_generate driver).
  for (const auto& args :
       {std::vector<std::string>{"prog", "--kv-int8", "--kv-fp"},
        std::vector<std::string>{"prog", "--kv-quant", "int8", "--kv-fp"}}) {
    CliOptions o;
    CHECK(run(args, o) == 0);
    CHECK(o.kv_quant_override == 0);
  }
  for (const auto& args :
       {std::vector<std::string>{"prog", "--kv-fp", "--kv-int8"},
        std::vector<std::string>{"prog", "--kv-quant", "none", "--kv-int8"},
        std::vector<std::string>{"prog", "--kv-int8", "--kv-quant", "none",
                                 "--kv-quant", "int8"}}) {
    CliOptions o;
    CHECK(run(args, o) == 0);
    CHECK(o.kv_quant_override == 1);
  }
  // Bad/missing values are startup errors, not silent defaults.
  {
    CliOptions o;
    CHECK(run({"prog", "--kv-quant", "fp8"}, o) == 1);
    CHECK(run({"prog", "--kv-quant"}, o) == 1);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--prefill-chunk", "512"}, o) == 0);
    CHECK(o.prefill_chunk_override == 512);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--prefill-chunk", "0"}, o) == 0);
    CHECK(o.prefill_chunk_override == 0);
  }
  {
    CliOptions o;
    CHECK(run({"prog", "--prefill-chunk", "-1"}, o) == 1);
    CHECK(run({"prog", "--prefill-chunk"}, o) == 1);
  }
  // Combined with the existing overrides.
  {
    CliOptions o;
    CHECK(run({"prog", "--max-seq-len", "4096", "--kv-int8",
               "--prefill-chunk", "512", "--prefix-cache-host"},
              o) == 0);
    CHECK(o.max_seq_len == 4096);
    CHECK(o.kv_quant_override == 1);
    CHECK(o.prefill_chunk_override == 512);
    CHECK(o.prefix_cache_host);
  }
  // LoadOptions carries the same unset sentinel (config file wins).
  {
    Server::LoadOptions lo;
    CHECK(lo.kv_quant_override == -1);
    CHECK(lo.prefill_chunk_override == -1);
  }
  return 0;
}

int test_error_shape() {
  const json e = openai_error("boom");
  CHECK(e.contains("error") && e["error"]["message"] == "boom");
  CHECK(e["error"]["type"] == "invalid_request_error");
  return 0;
}

// Streaming robustness: partial stops/markers must be held, never emitted
// and retracted (a shrink below the emitted count used to throw
// std::out_of_range out of push()/poll_reasoning(), killing the turn).
#ifdef KOKKOSDNN_SERVER_HAVE_LFM25
int test_stream_robustness() {
  using KokkosDNN::LFM25::ToolCallStreamParser;
  using KokkosDNN::LFM25::ToolParseResult;
  using KokkosDNN::LFM25::parse_tool_calls;
  // Reviewer repro: byte-at-a-time over a completing custom stop.
  {
    ToolCallStreamParser fed({"STOP"});
    std::string acc, racc;
    const std::string raw = "hello STOP world";
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
      // A partial stop head must never be emitted...
      CHECK(acc.find("STO") == std::string::npos);
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    const ToolParseResult whole = parse_tool_calls(raw, {"STOP"});
    CHECK(acc == whole.text);
    CHECK(acc == "hello ");
    CHECK(racc == whole.reasoning);
    CHECK(tail.calls.empty());
    CHECK(tail.text.empty());  // tail must not leak the stop back in
  }
  // Multi-char stop with newlines (the common stop-parameter shape).
  {
    ToolCallStreamParser fed({"\n\nUser:"});
    std::string acc;
    const std::string raw = "answer\n\nUser: next";
    for (char ch : raw) acc += fed.push(std::string(1, ch));
    acc += fed.finish().text;
    CHECK(acc == "answer");
    CHECK(acc == parse_tool_calls(raw, {"\n\nUser:"}).text);
  }
  // Disproved stop head releases as content.
  {
    ToolCallStreamParser fed({"STOP"});
    std::string acc;
    const std::string raw = "hello STOp tail";
    for (char ch : raw) acc += fed.push(std::string(1, ch));
    acc += fed.finish().text;
    CHECK(acc == raw);
  }
  // Turn terminator never leaks, however stops are configured: the parser
  // decodes with specials visible, so <|im_end|> must be cut even when the
  // request sets custom stops (previously it streamed literally).
  {
    ToolCallStreamParser fed({"STOP"});
    std::string acc, racc;
    const std::string raw = "The answer is 42<|im_end|>";
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(acc == "The answer is 42");
    CHECK(acc.find("<|im_end|>") == std::string::npos);
    CHECK(racc.empty());
    CHECK(tail.text.empty());
    const ToolParseResult whole = parse_tool_calls(raw, {"STOP"});
    CHECK(whole.text == acc);
  }
  // Turn ending exactly at a custom stop: nothing after it, tail empty.
  {
    ToolCallStreamParser fed({"STOP"});
    std::string acc;
    const std::string raw = "hello STOP";
    for (char ch : raw) acc += fed.push(std::string(1, ch));
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    CHECK(acc == "hello ");
    CHECK(tail.text.empty());
  }
  // Length-capped mid-marker turns: trailing partial heads are stripped,
  // never flushed into content by finish(). Only the single longest head
  // is held: "hello <<" keeps one '<' as content.
  for (const char* frag : {"hello <|im_e", "hello <|tool_call_st",
                           "hello <thi", "hello <|tool_call_end"}) {
    ToolCallStreamParser fed;
    std::string acc, racc;
    const std::string raw(frag);
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(acc == "hello ");
    CHECK(racc.empty());
    CHECK(parse_tool_calls(raw).text == "hello ");
  }
  {
    CHECK(parse_tool_calls("hello <<").text == "hello <");
    CHECK(parse_tool_calls("hello <<<").text == "hello <<");
    CHECK(parse_tool_calls("<think>abc<<").reasoning == "abc<");
    ToolCallStreamParser fed;
    std::string acc;
    const std::string raw = "hello <<";
    for (char ch : raw) acc += fed.push(std::string(1, ch));
    acc += fed.finish().text;
    CHECK(acc == "hello <");
  }
  // Long '<' run in 8-byte pushes: only the single longest head is held,
  // so the stream keeps up (at most one byte back) instead of stalling to
  // the end. finish() is idempotent (computed, not drained).
  {
    const std::string raw = std::string("hello ") + std::string(2000, '<');
    const ToolParseResult whole = parse_tool_calls(raw);
    CHECK(whole.text == std::string("hello ") + std::string(1999, '<'));
    ToolCallStreamParser fed;
    std::string acc, racc;
    size_t fed_n = 0;
    for (size_t p = 0; p < raw.size(); p += 8) {
      const std::string chunk = raw.substr(p, 8);
      acc += fed.push(chunk);
      racc += fed.poll_reasoning();
      fed_n += chunk.size();
      CHECK(whole.text.find(acc) == 0);
      CHECK(fed_n - acc.size() <= 64);
      CHECK(racc.empty());
    }
    const ToolParseResult tail1 = fed.finish();
    const ToolParseResult tail2 = fed.finish();
    CHECK(tail1.text == tail2.text);
    CHECK(tail1.reasoning == tail2.reasoning);
    CHECK(tail1.calls.size() == tail2.calls.size());
    acc += tail1.text;
    CHECK(acc == whole.text);
    CHECK(tail1.calls.empty());
  }
  // Length-capped mid-marker inside an unclosed <think> trace: the fragment
  // is stripped from the whole-text reasoning, and the streaming view must
  // hold (never emit) the same bytes so finish() cannot drop a byte the
  // client already saw as reasoning_content. Byte-at-a-time also covers
  // chunk boundaries splitting a marker; the prefix checks catch any
  // emit-then-retract.
  for (const char* frag :
       {"<", "<|tool_call_st", "<|tool_call_end", "<|im_e", "<thi",
        "</thi", "<<", "<<<", "<|tool_call_start|><"}) {
    const std::string raw = std::string("<think>hello ") + frag;
    const ToolParseResult whole = parse_tool_calls(raw);
    CHECK(whole.text.empty());
    // Single-head strip: "<<" keeps one '<', "<<<" keeps two.
    std::string expect_reasoning = "hello ";
    const std::string frag_s(frag);
    if (frag_s == "<<")
      expect_reasoning = "hello <";
    else if (frag_s == "<<<")
      expect_reasoning = "hello <<";
    CHECK(whole.reasoning == expect_reasoning);
    ToolCallStreamParser fed;
    std::string acc, racc;
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
      CHECK(whole.text.find(acc) == 0);
      CHECK(whole.reasoning.find(racc) == 0);
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(acc == whole.text);
    CHECK(racc == whole.reasoning);
    CHECK(tail.calls.empty());
  }
  // Think + tool + custom stop combined.
  {
    const std::vector<std::string> stops{"STOP"};
    ToolCallStreamParser fed(stops);
    std::string acc, racc;
    const std::string raw =
        std::string("<think>t</think>hi ") +
        KokkosDNN::LFM25::kToolCallStartToken + "[f(x=1)]" +
        KokkosDNN::LFM25::kToolCallEndToken + "STOP bye";
    for (char ch : raw) {
      acc += fed.push(std::string(1, ch));
      racc += fed.poll_reasoning();
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    const ToolParseResult whole = parse_tool_calls(raw, stops);
    CHECK(acc == whole.text);
    CHECK(racc == whole.reasoning);
    CHECK(tail.calls.size() == 1 && tail.calls[0].name == "f");
  }
  // Deterministic fuzz: random marker-ish soup in random chunk sizes must
  // never throw, and streaming totals must agree with the whole parse.
  {
    const char* toks[] = {"<",    ">",    "|",    "/",    "t",   "hink",
                          "<think>", "</think>", "<|tool_call_start|>",
                          "<|tool_call_end|>", "<|im_end|>", "[f(x=1)]",
                          "<thinking>", "</thinking>", "<tool_call>",
                          "</tool_call>", "<function=f(x=1)>",
                          "<parameter=a>", "</parameter>", "</function>",
                          "function=f(x=1)", "<function>", " ",    "a",
                          "STOP", "\n", "(",   ")"};
    const size_t ntoks = sizeof(toks) / sizeof(toks[0]);
    uint64_t rng = 0x12345678u;
    auto next = [&]() {
      rng = rng * 6364136223846793005u + 1442695040888963407u;
      return (uint32_t)(rng >> 33);
    };
    const std::vector<std::string> stops{"STOP", "\n\n"};
    for (int iter = 0; iter < 3000; ++iter) {
      std::string raw;
      const size_t nlen = next() % 48;
      for (size_t i = 0; i < nlen; ++i) raw += toks[next() % ntoks];
      const ToolParseResult whole = parse_tool_calls(raw, stops);
      ToolCallStreamParser fed(stops);
      std::string acc, racc;
      size_t p = 0;
      while (p < raw.size()) {
        size_t k = 1 + next() % 4;
        if (k > raw.size() - p) k = raw.size() - p;
        acc += fed.push(raw.substr(p, k));
        racc += fed.poll_reasoning();
        // Monotone: already-streamed bytes stay a prefix of the whole-text
        // result (finish() cannot retract them).
        CHECK(whole.text.find(acc) == 0);
        CHECK(whole.reasoning.find(racc) == 0);
        p += k;
      }
      ToolParseResult tail = fed.finish();
      acc += tail.text;
      racc += tail.reasoning;
      CHECK(acc == whole.text);
      CHECK(racc == whole.reasoning);
      CHECK(tail.calls.size() == whole.calls.size());
      for (size_t i = 0; i < tail.calls.size(); ++i) {
        CHECK(tail.calls[i].name == whole.calls[i].name);
        CHECK(tail.calls[i].arguments == whole.calls[i].arguments);
      }
    }
    // Same sweep with default stops (turn-terminator only).
    for (int iter = 0; iter < 2000; ++iter) {
      std::string raw;
      const size_t nlen = next() % 48;
      for (size_t i = 0; i < nlen; ++i) raw += toks[next() % ntoks];
      ToolCallStreamParser fed;
      std::string acc, racc;
      size_t p = 0;
      while (p < raw.size()) {
        size_t k = 1 + next() % 3;
        if (k > raw.size() - p) k = raw.size() - p;
        acc += fed.push(raw.substr(p, k));
        racc += fed.poll_reasoning();
        p += k;
      }
      ToolParseResult tail = fed.finish();
      acc += tail.text;
      racc += tail.reasoning;
      const ToolParseResult whole = parse_tool_calls(raw);
      CHECK(acc == whole.text);
      CHECK(racc == whole.reasoning);
      CHECK(tail.calls.size() == whole.calls.size());
    }
  }
  return 0;
}
#endif  // KOKKOSDNN_SERVER_HAVE_LFM25

// Single-call streaming step: one feed() per token returns both channels
// and must agree with push()/poll_reasoning() and the whole-text parse —
// including over a long thinking trace fed in token-sized chunks, where
// per-step full re-scans would be quadratic.
#ifdef KOKKOSDNN_SERVER_HAVE_LFM25
int test_stream_feed() {
  using KokkosDNN::LFM25::StreamDeltas;
  using KokkosDNN::LFM25::ToolCallStreamParser;
  using KokkosDNN::LFM25::ToolParseResult;
  using KokkosDNN::LFM25::parse_tool_calls;
  // feed() and push()/poll() agree on mixed content, random chunkings.
  {
    const std::vector<std::string> stops{"STOP"};
    const std::string raw =
        std::string("<think>let me think...</think>The answer is ") +
        KokkosDNN::LFM25::kToolCallStartToken + "[f(x=1)]" +
        KokkosDNN::LFM25::kToolCallEndToken + " done STOP tail";
    const ToolParseResult whole = parse_tool_calls(raw, stops);
    for (size_t chunk : {1, 3, 7}) {
      ToolCallStreamParser via_feed(stops), via_split(stops);
      std::string fd, fr, pd, pr;
      for (size_t p = 0; p < raw.size();) {
        const size_t k = std::min(chunk, raw.size() - p);
        StreamDeltas d = via_feed.feed(raw.substr(p, k));
        fd += d.display;
        fr += d.reasoning;
        pd += via_split.push(raw.substr(p, k));
        pr += via_split.poll_reasoning();
        // Monotone on both channels for both entry points.
        CHECK(whole.text.find(fd) == 0);
        CHECK(whole.reasoning.find(fr) == 0);
        CHECK(whole.text.find(pd) == 0);
        CHECK(whole.reasoning.find(pr) == 0);
        p += k;
      }
      ToolParseResult tf = via_feed.finish(), tp = via_split.finish();
      fd += tf.text;
      fr += tf.reasoning;
      pd += tp.text;
      pr += tp.reasoning;
      CHECK(fd == whole.text);
      CHECK(fr == whole.reasoning);
      CHECK(pd == whole.text);
      CHECK(pr == whole.reasoning);
      CHECK(fd == pd);
      CHECK(fr == pr);
      CHECK(tf.calls.size() == tp.calls.size());
      CHECK(tf.calls.size() == whole.calls.size());
    }
  }
  // Long thinking trace (~60k chars) in token-sized chunks: totals agree
  // with the whole-text parse and the streamed prefixes never retract.
  // (Spot-check monotonicity: a per-step prefix scan would itself be
  // quadratic in the test.)
  {
    std::string think;
    for (int i = 0; i < 1500; ++i)
      think += "thought number " + std::to_string(i) + ": considering... ";
    const std::string raw =
        std::string("<think>") + think + "</think>" +
        "The final answer is 42." + KokkosDNN::LFM25::kToolCallStartToken +
        "[get_weather(location=\"SF\")]" +
        KokkosDNN::LFM25::kToolCallEndToken;
    const ToolParseResult whole = parse_tool_calls(raw);
    CHECK(whole.reasoning == think);
    CHECK(whole.calls.size() == 1);
    ToolCallStreamParser fed;
    std::string acc, racc;
    uint64_t rng = 0x9e3779b9u;
    size_t p = 0, steps = 0;
    while (p < raw.size()) {
      rng = rng * 6364136223846793005u + 1442695040888963407u;
      size_t k = 1 + (uint32_t)(rng >> 33) % 13;
      if (k > raw.size() - p) k = raw.size() - p;
      StreamDeltas d = fed.feed(raw.substr(p, k));
      acc += d.display;
      racc += d.reasoning;
      if (++steps % 997 == 0) {
        CHECK(whole.text.find(acc) == 0);
        CHECK(whole.reasoning.find(racc) == 0);
      }
      p += k;
    }
    ToolParseResult tail = fed.finish();
    acc += tail.text;
    racc += tail.reasoning;
    CHECK(acc == whole.text);
    CHECK(racc == whole.reasoning);
    CHECK(tail.calls.size() == 1 && tail.calls[0].name == "get_weather");
  }
  return 0;
}
#endif  // KOKKOSDNN_SERVER_HAVE_LFM25

}  // namespace

int main() {
  if (test_messages()) return 1;
  if (test_common()) return 1;
  if (test_tool_helpers()) return 1;
  if (test_backends()) return 1;
  if (test_cli_glimmer_flags()) return 1;
  if (test_assemble()) return 1;
#ifdef KOKKOSDNN_SERVER_HAVE_LFM25
  if (test_lfm25_tool_wire()) return 1;
  if (test_lfm25_render_tiny()) return 1;
  if (test_stream_robustness()) return 1;
  if (test_stream_feed()) return 1;
#endif
#ifdef KOKKOSDNN_SERVER_HAVE_GLIMMER
  if (test_glimmer_render_tiny()) return 1;
  if (test_glimmer_tool_wire()) return 1;
#endif
  if (test_error_shape()) return 1;
  std::cout << "server parsing tests passed (" << g_checks << " checks)\n";
  return 0;
}
