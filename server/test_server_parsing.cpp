// Parsing unit tests for kokkosdnn_server: exercise the pure OpenAI/JSON
// translation helpers (parse_common, parse_messages, openai_error) by
// including server_main.cpp with KOKKOSDNN_SERVER_NO_MAIN. No weights, no
// tokenizer file, no network, no Kokkos runtime init needed (helpers only
// touch Config/SessionOptions JSON mapping; Config::tiny_test() is
// header-only defaults).
//
// Assert-based (no gtest): any failure aborts nonzero. Wired in
// server/CMakeLists.txt behind KokkosDNN_ENABLE_TESTS.

#define KOKKOSDNN_SERVER_NO_MAIN
#include "server_main.cpp"

#include <cassert>
#include <iostream>

namespace {

using KokkosDNN::LFM25::Config;
using KokkosDNN::LFM25::ToolCall;
using KokkosDNN::LFM25::ToolParseResult;
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

bool parse_ok(const json& body, const Config& cfg, ParsedRequest& pr,
              std::string& err, const std::string& model = "lfm2_5") {
  return parse_common(body, cfg, model, pr, err);
}

int test_messages() {
  std::vector<ChatMessage> out;
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
  return 0;
}

int test_common() {
  const Config cfg = Config::tiny_test();
  ParsedRequest pr;
  std::string err;
  // Minimal body passes with defaults.
  CHECK(parse_ok(json::parse(R"({})"), cfg, pr, err));
  CHECK(pr.req_max_tokens == -1 && !pr.has_seed && !pr.stream);
  CHECK(pr.opts.stop_ids == std::vector<int>{cfg.eos});
  // Unknown model id rejected; absent ok.
  CHECK(!parse_ok(json::parse(R"({"model":"other"})"), cfg, pr, err));
  CHECK(parse_ok(json::parse(R"({"model":"lfm2_5"})"), cfg, pr, err));
  // Unsupported features rejected (legacy function calling retired).
  for (const char* f :
       {R"({"n":2})", R"({"functions":[]})",
        R"({"function_call":"auto"})"}) {
    CHECK(!parse_ok(json::parse(f), cfg, pr, err));
  }
  // tools accepted + validated; empty array = no-op.
  CHECK(parse_ok(json::parse(R"({"tools":[]})"), cfg, pr, err));
  CHECK(!pr.tools_active && pr.tools_json.empty());
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"type":"function",
        "function":{"name":"get_weather","description":"d",
          "parameters":{"type":"object"},"strict":false}}]})"),
      cfg, pr, err));
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
    CHECK(!parse_ok(json::parse(f), cfg, pr, err));
  }
  // tool_choice: none/auto/required/named + validation.
  pr = ParsedRequest{};
  CHECK(parse_ok(json::parse(R"({"tool_choice":"auto"})"), cfg, pr, err));
  CHECK(!pr.tools_active && pr.tool_directive.empty());
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"function":{"name":"f"}}],
        "tool_choice":"none"})"),
      cfg, pr, err));
  CHECK(!pr.tools_active);
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"function":{"name":"f"}}],
        "tool_choice":"required"})"),
      cfg, pr, err));
  CHECK(pr.tools_active && !pr.tool_directive.empty());
  CHECK(parse_ok(
      json::parse(R"({"tools":[{"function":{"name":"f"}}],
        "tool_choice":{"type":"function","function":{"name":"f"}}})"),
      cfg, pr, err));
  CHECK(pr.tool_directive.find("f") != std::string::npos);
  pr = ParsedRequest{};
  for (const char* f :
       {R"({"tool_choice":"sometimes"})", R"({"tool_choice":42})",
        R"({"tool_choice":{"type":"code","function":{"name":"f"}}})",
        R"({"tool_choice":{"function":{}}})",
        R"({"tool_choice":{"function":{"name":""}}})",
        R"({"tool_choice":{"function":{"name":"f"}}})",
        R"({"tool_choice":"required"})"}) {
    CHECK(!parse_ok(json::parse(f), cfg, pr, err));
  }
  // parallel_tool_calls must be boolean.
  CHECK(parse_ok(json::parse(R"({"parallel_tool_calls":false})"), cfg, pr,
                 err));
  CHECK(!pr.parallel_tool_calls);
  CHECK(!parse_ok(json::parse(R"({"parallel_tool_calls":"yes"})"), cfg, pr,
                  err));
  // response_format: text/json_object/json_schema accepted + validated.
  CHECK(parse_ok(json::parse(R"({"response_format":{"type":"text"}})"), cfg,
                 pr, err));
  CHECK(!pr.json_mode);
  CHECK(parse_ok(json::parse(R"({"response_format":{"type":"json_object"}})"),
                 cfg, pr, err));
  CHECK(pr.json_mode && !pr.json_directive.empty());
  CHECK(parse_ok(
      json::parse(R"({"response_format":{"type":"json_schema",
        "json_schema":{"type":"object"}}})"),
      cfg, pr, err));
  CHECK(pr.json_mode);
  for (const char* f :
       {R"({"response_format":{"type":"json_schema"}})",
        R"({"response_format":{"type":"yaml"}})",
        R"({"response_format":{}})",
        R"({"response_format":"json"})"}) {
    CHECK(!parse_ok(json::parse(f), cfg, pr, err));
  }
  CHECK(!pr.tools_active);
  CHECK(parse_ok(json::parse(R"({"n":1})"), cfg, pr, err));  // n=1 ok
  // Sampling mapping.
  CHECK(parse_ok(
      json::parse(R"({"temperature":0.7,"top_p":0.9,"top_k":40,"min_p":0.05,
        "repetition_penalty":1.1,"frequency_penalty":0.5,"presence_penalty":-0.2,
        "logit_bias":{"100":2.0,"200":-1.0},
        "logprobs":true,"top_logprobs":3,"seed":123,"stream":true,
        "max_tokens":64})"),
      cfg, pr, err));
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
  CHECK(parse_ok(json::parse(R"({"logprobs":true})"), cfg, pr, err));
  CHECK(pr.opts.sampling.logprobs == 5);
  CHECK(parse_ok(json::parse(R"({"logprobs":2})"), cfg, pr, err));
  CHECK(pr.opts.sampling.logprobs == 2);
  // stop mapping: numeric strings -> ids, rest -> strings.
  CHECK(parse_ok(json::parse(R"({"stop":["<|im_end|>","42"]})"), cfg, pr,
                 err));
  CHECK(pr.opts.stop_strings.size() == 1);
  CHECK(pr.opts.stop_strings[0] == "<|im_end|>");
  CHECK(pr.opts.stop_ids.size() == 2);
  CHECK(pr.opts.stop_ids[0] == cfg.eos && pr.opts.stop_ids[1] == 42);
  CHECK(parse_ok(json::parse(R"({"stop":"halt"})"), cfg, pr, err));
  CHECK(pr.opts.stop_strings == std::vector<std::string>{"halt"});
  // Bad types rejected.
  CHECK(!parse_ok(json::parse(R"({"temperature":"hot"})"), cfg, pr, err));
  CHECK(!parse_ok(json::parse(R"({"seed":1.5})"), cfg, pr, err));
  CHECK(!parse_ok(json::parse(R"({"max_tokens":0})"), cfg, pr, err));
  CHECK(!parse_ok(json::parse(R"({"stop":[42.5]})"), cfg, pr, err));
  CHECK(!parse_ok(json::parse(R"({"logit_bias":{"x":1.0}})"), cfg, pr, err));
  // max_completion_tokens wins over max_tokens.
  CHECK(parse_ok(
      json::parse(R"({"max_tokens":10,"max_completion_tokens":20})"), cfg, pr,
      err));
  CHECK(pr.req_max_tokens == 20);
  // Unknown fields ignored.
  CHECK(parse_ok(json::parse(R"({"bogus_field":[1,2,3]})"), cfg, pr, err));
  return 0;
}

int test_tool_wire() {
  using KokkosDNN::LFM25::args_json_to_pythonic;
  using KokkosDNN::LFM25::args_pythonic_to_json;
  using KokkosDNN::LFM25::encode_chatml_with_tools;
  using KokkosDNN::LFM25::parse_tool_call_block;
  using KokkosDNN::LFM25::parse_tool_calls;
  using KokkosDNN::LFM25::render_lfm25_chatml_with_tools;
  using KokkosDNN::LFM25::ToolCallStreamParser;
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
  // Unclosed markers are text, not calls.
  {
    const std::string raw =
        std::string("a") + KokkosDNN::LFM25::kToolCallStartToken + "[f(x=1)";
    const ToolParseResult r = parse_tool_calls(raw);
    CHECK(r.calls.empty() && r.text == "a[f(x=1)");
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
  }
  // System suffix lands in the system block; directives combine.
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
  // tool_calls_json shape: fresh ids, function envelope, {} default.
  // (Default-constructed ServerState performs no Kokkos allocation, so
  // no runtime init is needed; only the id counter is touched.)
  {
    ServerState st;
    ToolCall c;
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
        assemble_chat_message(st, pr, "hi", {}, "stop");
    CHECK(m.message["role"] == "assistant");
    CHECK(m.message["content"] == "hi");
    CHECK(!m.message.contains("tool_calls"));
    CHECK(m.finish == "stop");
  }
  // Tools, no calls: parser text used verbatim, session finish kept.
  pr.tools_active = true;
  {
    AssembledMessage m =
        assemble_chat_message(st, pr, "thinking...", {}, "length");
    CHECK(m.message["content"] == "thinking...");
    CHECK(!m.message.contains("tool_calls"));
    CHECK(m.finish == "length");
  }
  // Tools with calls: tool_calls payload, finish tool_calls.
  {
    ToolCall c;
    c.name = "get_weather";
    c.arguments = "{\"location\":\"SF\"}";
    AssembledMessage m =
        assemble_chat_message(st, pr, "checking", {c}, "stop");
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
    ToolCall c;
    c.name = "f";
    c.arguments = "";
    AssembledMessage m = assemble_chat_message(st, pr, "", {c}, "stop");
    CHECK(m.message["content"].is_null());
    CHECK(m.message["tool_calls"][0]["function"]["arguments"] == "{}");
    CHECK(m.finish == "tool_calls");
  }
  // parallel_tool_calls=false keeps the first call only.
  pr.parallel_tool_calls = false;
  {
    ToolCall a, b;
    a.name = "a";
    a.arguments = "{}";
    b.name = "b";
    b.arguments = "{}";
    AssembledMessage m =
        assemble_chat_message(st, pr, "t", {a, b}, "stop");
    CHECK(m.message["tool_calls"].size() == 1);
    CHECK(m.message["tool_calls"][0]["function"]["name"] == "a");
  }
  pr.parallel_tool_calls = true;
  // json_mode strips fences from content (calls or not).
  pr.json_mode = true;
  {
    AssembledMessage m =
        assemble_chat_message(st, pr, "```json\n{\"a\":1}\n```", {},
                              "stop");
    CHECK(m.message["content"] == "{\"a\":1}");
  }
  pr.json_mode = false;
  pr.tools_active = false;
  // Streaming delta shape: index/id/type/function per call.
  {
    ToolCall c;
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

int test_error_shape() {
  const json e = openai_error("boom");
  CHECK(e.contains("error") && e["error"]["message"] == "boom");
  CHECK(e["error"]["type"] == "invalid_request_error");
  return 0;
}

}  // namespace

int main() {
  if (test_messages()) return 1;
  if (test_common()) return 1;
  if (test_tool_wire()) return 1;
  if (test_assemble()) return 1;
  if (test_error_shape()) return 1;
  std::cout << "server parsing tests passed (" << g_checks << " checks)\n";
  return 0;
}
