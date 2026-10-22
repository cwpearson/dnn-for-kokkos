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
  // Bad roles incl. tool/function.
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"tool","content":"x"}]})"), out,
      err));
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"function","content":"x"}]})"), out,
      err));
  // tool_calls payload rejected.
  CHECK(!parse_messages(
      json::parse(R"({"messages":[{"role":"assistant","content":"x",
        "tool_calls":[]}]})"),
      out, err));
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
  // Unsupported features rejected.
  for (const char* f :
       {R"({"n":2})", R"({"tools":[]})", R"({"tool_choice":"auto"})",
        R"({"response_format":{"type":"json_object"}})"}) {
    CHECK(!parse_ok(json::parse(f), cfg, pr, err));
  }
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
  if (test_error_shape()) return 1;
  std::cout << "server parsing tests passed (" << g_checks << " checks)\n";
  return 0;
}
