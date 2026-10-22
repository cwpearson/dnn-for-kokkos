// kokkosdnn_server: OpenAI-compatible HTTP front-end for KokkosDNN (LFM2.5).
//
// Translation only: HTTP/JSON <-> LFM25 Session calls. No sampling, chat
// templating, detokenization, or model math lives here — all of that belongs
// to models/* (Session, Tokenizer, Sampler, Detokenizer). This file owns
// Kokkos::initialize/finalize, CLI parsing, weight/tokenizer loading, the
// cpp-httplib route table, and OpenAI request/response mapping.
//
// Concurrency: one global mutex guards all inference (single in-flight
// request; a second concurrent request blocks until the first finishes). A
// fresh Session is built per request under that lock so per-request
// sampling/stop/seed options apply without mutating shared state; weights
// and the tokenizer are shared read-only.

#include <Kokkos_Core.hpp>

#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Session.hpp"
#include "KokkosDNN_LFM25Tokenizer.hpp"
#include "KokkosDNN_LFM25Weights.hpp"

#include <httplib.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

using KokkosDNN::LFM25::ChatMessage;
using KokkosDNN::LFM25::Config;
using KokkosDNN::LFM25::ToolCall;
using KokkosDNN::LFM25::ToolCallStreamParser;
using KokkosDNN::LFM25::ToolParseResult;
using KokkosDNN::LFM25::DefaultScalar;
using KokkosDNN::LFM25::Session;
using KokkosDNN::LFM25::SessionOptions;
using KokkosDNN::LFM25::StepOutcome;
using KokkosDNN::LFM25::Tokenizer;
using KokkosDNN::LFM25::Weights;
namespace Common = KokkosDNN::Common;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = DefaultScalar;
using ServerSession = Session<Exec, Scalar>;
using json = nlohmann::json;

constexpr const char* kVersion = "0.1.0";
constexpr int kDefaultPort = 8000;
constexpr const char* kDefaultModel = "lfm2_5";
constexpr int kDefaultMaxTokens = 256;

// The server runs as PID 1 in its container image, where SIGINT/SIGTERM
// revert to "ignore" unless the process installs a handler — that's why
// `podman run` ctrl-C and `podman stop` used to hang until SIGKILL. The
// handler below asks the httplib listener to stop so shutdown is prompt.
std::atomic<httplib::Server*> g_listening_server{nullptr};
std::atomic<bool> g_stop_requested{false};

void handle_stop_signal(int) {
  g_stop_requested.store(true, std::memory_order_relaxed);
  if (auto* svr = g_listening_server.load(std::memory_order_acquire)) {
    svr->stop();
  }
}

void install_stop_handlers() {
  std::signal(SIGINT, handle_stop_signal);
  std::signal(SIGTERM, handle_stop_signal);
}

// Startup progress goes to stderr (unbuffered enough for `podman logs` to
// show it while weights load) with an explicit flush per line. Per-request
// traffic stays silent by design.
void startup_log(const std::string& msg) {
  std::fprintf(stderr, "%s\n", msg.c_str());
  std::fflush(stderr);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliOptions {
  int port = kDefaultPort;
  std::string model = kDefaultModel;
  std::string weights;
  std::string tokenizer;
  std::string config_path;
};

// Returns an exit code when the process should stop after parse_cli
// (0 = ok to continue serving, anything else = exit with that code).
int parse_cli(int argc, char* argv[], CliOptions& out) {
  auto help = [&]() {
    std::printf(
        "Usage: %s [--port N] [--model ID] [--weights PATH] [--tokenizer "
        "PATH]\n"
        "             [--config PATH] [--help] [--version]\n"
        "\n"
        "OpenAI-compatible inference server for KokkosDNN (LFM2.5).\n"
        "\n"
        "Options:\n"
        "  --port N          TCP port to listen on (default: %d)\n"
        "  --model ID        model identifier to serve (default: %s)\n"
        "  --weights PATH    GGUF checkpoint to load (required to serve)\n"
        "  --tokenizer PATH  tokenizer.json path or directory (required)\n"
        "  --config PATH     optional model config JSON overlay (base: HF\n"
        "                    lfm2_5_24b_a2b defaults)\n"
        "  --help            print this message and exit\n"
        "  --version         print the server version and exit\n"
        "\n"
        "Routes:\n"
        "  GET  /v1/models\n"
        "  POST /v1/chat/completions\n"
        "  POST /v1/completions\n",
        argv[0], kDefaultPort, kDefaultModel);
  };
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto need_value = [&](const char* flag, std::string& dst) -> bool {
      if (i + 1 >= argc) {
        std::printf("Missing value for %s\n", flag);
        return false;
      }
      dst = argv[++i];
      return true;
    };
    if (std::strcmp(a, "--help") == 0) {
      help();
      return 2;  // --help / --version: already printed, caller exits 0
    }
    if (std::strcmp(a, "--version") == 0) {
      std::printf("kokkosdnn_server %s\n", kVersion);
      return 2;
    }
    if (std::strcmp(a, "--port") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value for --port\n");
        return 1;
      }
      out.port = std::atoi(argv[++i]);
    } else if (std::strcmp(a, "--model") == 0) {
      if (!need_value("--model", out.model)) return 1;
    } else if (std::strcmp(a, "--weights") == 0) {
      if (!need_value("--weights", out.weights)) return 1;
    } else if (std::strcmp(a, "--tokenizer") == 0) {
      if (!need_value("--tokenizer", out.tokenizer)) return 1;
    } else if (std::strcmp(a, "--config") == 0) {
      if (!need_value("--config", out.config_path)) return 1;
    } else {
      std::printf("Unknown argument: %s\n", a);
      help();
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Shared server state
// ---------------------------------------------------------------------------

struct ServerState {
  std::string model_id;
  Config cfg;
  Weights<Exec, Scalar> weights;
  Tokenizer tok;
  std::mutex mu;  // single in-flight request; others block
  std::atomic<unsigned long long> next_id{1};
};

// Loads Config (HF defaults + optional JSON overlay), GGUF weights, and the
// tokenizer. Requires Kokkos to be initialized (allocates device views).
// Returns true on success; on failure `err` describes the problem.
bool load_model(const CliOptions& o, ServerState& st, std::string& err) {
  if (o.weights.empty()) {
    err = "missing required --weights PATH (GGUF checkpoint)";
    return false;
  }
  if (o.tokenizer.empty()) {
    err = "missing required --tokenizer PATH (tokenizer.json)";
    return false;
  }
  try {
    st.model_id = o.model;
    st.cfg = o.config_path.empty()
                 ? Config::lfm2_5_24b_a2b()
                 : Config::from_json_file(o.config_path);
    st.weights.allocate(st.cfg);
    KokkosDNN::LFM25::load_gguf(st.cfg, o.weights, st.weights);
    st.tok = Tokenizer(o.tokenizer);
  } catch (const std::exception& e) {
    err = std::string("failed to load model: ") + e.what();
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// OpenAI error/response helpers
// ---------------------------------------------------------------------------

json openai_error(const std::string& message,
                  const std::string& type = "invalid_request_error") {
  return json{{"error", {{"message", message}, {"type", type}}}};
}

void send_json(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

void send_bad_request(httplib::Response& res, const std::string& message) {
  send_json(res, 400, openai_error(message));
}

void send_server_error(httplib::Response& res, const std::string& message) {
  send_json(res, 500, openai_error(message, "server_error"));
}

std::string new_id(const std::string& prefix, ServerState& st) {
  return prefix + "-kdnn-" +
         std::to_string(st.next_id.fetch_add(1, std::memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// Request parsing (pure JSON -> Session inputs; no model math)
// ---------------------------------------------------------------------------

struct ParsedRequest {
  std::vector<ChatMessage> messages;
  SessionOptions opts;  // sampling + stop_ids/stop_strings; max_new filled in
                        // after the prompt is encoded (needs prompt length)
  int req_max_tokens = -1;  // -1 = client did not ask; default applies later
  bool has_seed = false;
  uint64_t seed = 0;
  bool stream = false;
  // Agentic extensions (OpenAI `tools` / `tool_choice` / `response_format`).
  // tools_json: compact-serialized `tools` array for the List-of-tools
  // prompt block ("" = none). tools_active: inject tools into the prompt
  // and parse native tool calls out of the completion.
  std::string tools_json;
  std::string tool_directive;  // "required"/named tool_choice instruction
  std::string json_directive;  // response_format JSON-mode instruction
  bool tools_active = false;
  bool json_mode = false;
  bool parallel_tool_calls = true;
};

// Shared fields: model id, unsupported-feature gates, sampling options,
// logit_bias, logprobs, stop, seed, stream, max tokens. Returns true on
// success; on failure `err` carries the 400 message.
bool parse_common(const json& body, const Config& cfg,
                  const std::string& served_model, ParsedRequest& out,
                  std::string& err) {
  if (!body.is_object()) {
    err = "request body must be a JSON object";
    return false;
  }
  // Unknown fields are ignored, except the gated features below.
  if (body.contains("model") && !body["model"].is_null()) {
    if (!body["model"].is_string() ||
        body["model"].get<std::string>() != served_model) {
      err = "unknown model id '" +
            (body["model"].is_string() ? body["model"].get<std::string>()
                                       : body["model"].dump()) +
            "' (serving '" + served_model + "')";
      return false;
    }
  }
  // Unsupported OpenAI features fail loudly instead of being silently
  // ignored (a wrong answer is worse than an error).
  if (body.contains("n") && !body["n"].is_null()) {
    if (!body["n"].is_number_integer() || body["n"].get<int>() != 1) {
      err = "only n=1 is supported";
      return false;
    }
  }
  // Legacy function calling stays retired: modern agents use tools.
  for (const char* k : {"functions", "function_call"}) {
    if (body.contains(k) && !body[k].is_null()) {
      err = std::string("unsupported field '") + k +
            "' (legacy function calling is retired; use 'tools'/'tool_choice')";
      return false;
    }
  }
  // OpenAI `tools`: [{type:"function",
  // function:{name, description?, parameters?, strict?}}]. Validated here,
  // injected into the prompt as `List of tools: [...]` at render time.
  if (body.contains("tools") && !body["tools"].is_null()) {
    const auto& tools = body["tools"];
    if (!tools.is_array()) {
      err = "'tools' must be an array of function tool definitions";
      return false;
    }
    for (const auto& tool : tools) {
      if (!tool.is_object()) {
        err = "each entry of 'tools' must be an object";
        return false;
      }
      if (tool.value("type", "function") != "function") {
        err = "only 'function' tools are supported";
        return false;
      }
      auto fit = tool.find("function");
      if (fit == tool.end() || !fit->is_object()) {
        err = "each tool must carry an object 'function'";
        return false;
      }
      auto nit = fit->find("name");
      if (nit == fit->end() || !nit->is_string() ||
          nit->get<std::string>().empty()) {
        err = "each tool function must carry a non-empty string 'name'";
        return false;
      }
      const std::string name = nit->get<std::string>();
      if (fit->contains("description") && !(*fit)["description"].is_null() &&
          !(*fit)["description"].is_string()) {
        err = "'description' of tool '" + name + "' must be a string";
        return false;
      }
      if (fit->contains("parameters") && !(*fit)["parameters"].is_null() &&
          !(*fit)["parameters"].is_object()) {
        err = "'parameters' of tool '" + name + "' must be an object";
        return false;
      }
      if (fit->contains("strict") && !(*fit)["strict"].is_null() &&
          !(*fit)["strict"].is_boolean()) {
        err = "'strict' of tool '" + name + "' must be a boolean";
        return false;
      }
    }
    if (!tools.empty()) {
      out.tools_json = tools.dump();
      out.tools_active = true;
    }
  }
  // tool_choice: "none" | "auto" | "required" |
  // {"type":"function","function":{"name": "..."}}. "none" disables tool
  // injection for this request (history tool turns still render natively).
  // required/named add a best-effort prompt directive: without
  // grammar-constrained decoding a missing call is not an error.
  if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
    const auto& tc = body["tool_choice"];
    if (tc.is_string()) {
      const std::string v = tc.get<std::string>();
      if (v == "none") {
        out.tools_json.clear();
        out.tools_active = false;
      } else if (v == "auto") {
      } else if (v == "required") {
        if (out.tools_json.empty()) {
          err = "'tool_choice' is 'required' but no 'tools' were provided";
          return false;
        }
        out.tool_directive =
            "You MUST call one of the provided tools in this turn. "
            "Do not answer with plain text instead of calling a tool.";
      } else {
        err = "'tool_choice' string must be one of none, auto, required";
        return false;
      }
    } else if (tc.is_object()) {
      if (tc.value("type", "function") != "function") {
        err = "only 'function' tool_choice is supported";
        return false;
      }
      auto fit = tc.find("function");
      if (fit == tc.end() || !fit->is_object()) {
        err = "'tool_choice' must carry an object 'function'";
        return false;
      }
      auto nit = fit->find("name");
      if (nit == fit->end() || !nit->is_string() ||
          nit->get<std::string>().empty()) {
        err = "'tool_choice.function' must carry a non-empty string 'name'";
        return false;
      }
      if (out.tools_json.empty()) {
        err = "'tool_choice' names a function but no 'tools' were provided";
        return false;
      }
      out.tool_directive = "You MUST call the function " +
                           nit->get<std::string>() +
                           " in this turn. Do not call any other function.";
    } else {
      err = "'tool_choice' must be a string or an object";
      return false;
    }
  }
  if (body.contains("parallel_tool_calls") &&
      !body["parallel_tool_calls"].is_null()) {
    if (!body["parallel_tool_calls"].is_boolean()) {
      err = "'parallel_tool_calls' must be a boolean";
      return false;
    }
    out.parallel_tool_calls = body["parallel_tool_calls"].get<bool>();
  }
  // response_format: {"type":"text"} (default) | {"type":"json_object"} |
  // {"type":"json_schema","json_schema":{...}}. Soft enforcement: a JSON
  // instruction joins the system prompt and markdown fences are stripped
  // from the reply. The schema itself is recorded but NOT enforced (the
  // sampler has no grammar-constrained decoding); validate client-side.
  if (body.contains("response_format") &&
      !body["response_format"].is_null()) {
    const auto& rf = body["response_format"];
    if (!rf.is_object() || !rf.contains("type") || !rf["type"].is_string()) {
      err = "'response_format' must be an object with a string 'type'";
      return false;
    }
    const std::string rtype = rf["type"].get<std::string>();
    if (rtype == "text") {
    } else if (rtype == "json_object") {
      out.json_mode = true;
      out.json_directive =
          "You must respond with valid JSON only, without any explanation, "
          "markdown fences, or additional text.";
    } else if (rtype == "json_schema") {
      auto it = rf.find("json_schema");
      if (it == rf.end() || !it->is_object()) {
        err =
            "'response_format' of type 'json_schema' requires an object "
            "'json_schema'";
        return false;
      }
      out.json_mode = true;
      out.json_directive =
          "You must respond with valid JSON only, conforming to the provided "
          "JSON schema, without any explanation, markdown fences, or "
          "additional text.";
    } else {
      err = "unsupported 'response_format' type '" + rtype +
            "' (expected text, json_object, or json_schema)";
      return false;
    }
  }

  auto& s = out.opts.sampling;
  auto get_float = [&](const char* k, float& dst) -> bool {
    if (body.contains(k) && !body[k].is_null()) {
      if (!body[k].is_number()) {
        err = std::string("'") + k + "' must be a number";
        return false;
      }
      dst = body[k].get<float>();
    }
    return true;
  };
  if (!get_float("temperature", s.temperature)) return false;
  if (!get_float("top_p", s.top_p)) return false;
  if (!get_float("min_p", s.min_p)) return false;
  if (!get_float("repetition_penalty", s.repetition_penalty)) return false;
  if (!get_float("frequency_penalty", s.frequency_penalty)) return false;
  if (!get_float("presence_penalty", s.presence_penalty)) return false;
  if (body.contains("top_k") && !body["top_k"].is_null()) {
    if (!body["top_k"].is_number_integer()) {
      err = "'top_k' must be an integer";
      return false;
    }
    s.top_k = body["top_k"].get<int>();
  }
  if (body.contains("logit_bias") && !body["logit_bias"].is_null()) {
    const auto& lb = body["logit_bias"];
    if (!lb.is_object()) {
      err = "'logit_bias' must be an object mapping token ids to biases";
      return false;
    }
    for (auto it = lb.begin(); it != lb.end(); ++it) {
      int id = 0;
      try {
        size_t pos = 0;
        id = std::stoi(it.key(), &pos);
        if (pos != it.key().size()) throw std::invalid_argument("trailing");
      } catch (...) {
        err = "'logit_bias' keys must be integer token ids";
        return false;
      }
      if (!it.value().is_number()) {
        err = "'logit_bias' values must be numbers";
        return false;
      }
      s.logit_bias.emplace_back(id, it.value().get<float>());
    }
  }
  // OpenAI chat uses logprobs(bool) + top_logprobs(int); accept a plain
  // integer logprobs too (completions style / extensions).
  if ((body.contains("logprobs") && !body["logprobs"].is_null()) ||
      (body.contains("top_logprobs") && !body["top_logprobs"].is_null())) {
    int n = 0;
    bool want = false;
    if (body.contains("logprobs") && !body["logprobs"].is_null()) {
      const auto& lp = body["logprobs"];
      if (lp.is_boolean()) {
        want = lp.get<bool>();
      } else if (lp.is_number_integer()) {
        want = lp.get<int>() > 0;
        n = lp.get<int>();
      } else {
        err = "'logprobs' must be a boolean or integer";
        return false;
      }
    }
    if (body.contains("top_logprobs") && !body["top_logprobs"].is_null()) {
      if (!body["top_logprobs"].is_number_integer()) {
        err = "'top_logprobs' must be an integer";
        return false;
      }
      n = body["top_logprobs"].get<int>();
      want = want || n > 0;
    }
    if (want && n <= 0) n = 5;  // logprobs:true without an explicit count
    if (n < 0) n = 0;
    s.logprobs = n;
  }

  // stop: string | number | array thereof. Integer-valued entries become
  // stop ids; everything else becomes a stop string.
  out.opts.stop_ids = {cfg.eos};
  out.opts.stop_strings.clear();
  if (body.contains("stop") && !body["stop"].is_null()) {
    const auto& stop = body["stop"];
    auto add_entry = [&](const json& e) -> bool {
      if (e.is_string()) {
        const std::string v = e.get<std::string>();
        try {
          size_t pos = 0;
          int id = std::stoi(v, &pos);
          if (pos == v.size()) {
            out.opts.stop_ids.push_back(id);
            return true;
          }
        } catch (...) {
          // Not an integer: fall through to stop string.
        }
        out.opts.stop_strings.push_back(v);
        return true;
      }
      if (e.is_number_integer()) {
        out.opts.stop_ids.push_back(e.get<int>());
        return true;
      }
      err = "'stop' entries must be strings";
      return false;
    };
    if (stop.is_string() || stop.is_number_integer()) {
      if (!add_entry(stop)) return false;
    } else if (stop.is_array()) {
      for (const auto& e : stop)
        if (!add_entry(e)) return false;
    } else {
      err = "'stop' must be a string or an array of strings";
      return false;
    }
  }

  if (body.contains("seed") && !body["seed"].is_null()) {
    if (!body["seed"].is_number_integer()) {
      err = "'seed' must be an integer";
      return false;
    }
    out.has_seed = true;
    out.seed = static_cast<uint64_t>(body["seed"].get<long long>());
  }
  if (body.contains("stream") && !body["stream"].is_null()) {
    if (!body["stream"].is_boolean()) {
      err = "'stream' must be a boolean";
      return false;
    }
    out.stream = body["stream"].get<bool>();
  }
  // max_completion_tokens wins over the legacy max_tokens alias.
  const json* maxp = nullptr;
  if (body.contains("max_completion_tokens") &&
      !body["max_completion_tokens"].is_null())
    maxp = &body["max_completion_tokens"];
  else if (body.contains("max_tokens") && !body["max_tokens"].is_null())
    maxp = &body["max_tokens"];
  if (maxp) {
    if (!maxp->is_number_integer()) {
      err = "'max_tokens' must be an integer";
      return false;
    }
    out.req_max_tokens = maxp->get<int>();
    if (out.req_max_tokens <= 0) {
      err = "'max_tokens' must be positive";
      return false;
    }
  }
  return true;
}

// messages -> vector<ChatMessage>. "developer" maps to "user";
// "function" (legacy role) maps to "tool". Assistant turns may carry
// OpenAI `tool_calls` (validated, arguments string or object); "tool"
// turns carry the function result plus an optional `tool_call_id` link.
// Assistant content may be null when tool_calls are present, and is
// required otherwise; every other role requires string content.
// Non-text content parts and message-level `function_call` are rejected
// (400) rather than silently dropped.
bool parse_messages(const json& body, std::vector<ChatMessage>& out,
                    std::string& err) {
  auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty()) {
    err = "'messages' must be a non-empty array";
    return false;
  }
  // String content or text-part arrays joined; null/missing yields "".
  auto text_content = [&](const json& m, std::string& content) -> bool {
    auto cit = m.find("content");
    if (cit == m.end() || cit->is_null()) {
      content.clear();
      return true;
    }
    if (cit->is_string()) {
      content = cit->get<std::string>();
      return true;
    }
    if (cit->is_array()) {
      // Content-part arrays: join text parts, reject anything else.
      content.clear();
      for (const auto& part : *cit) {
        if (!part.is_object() || part.value("type", "") != "text" ||
            !part.contains("text") || !part["text"].is_string()) {
          err = "only text content parts are supported";
          return false;
        }
        content += part["text"].get<std::string>();
      }
      return true;
    }
    err = "each message must carry a string 'content'";
    return false;
  };
  for (const auto& m : *it) {
    if (!m.is_object()) {
      err = "each entry of 'messages' must be an object";
      return false;
    }
    if (!m.contains("role") || !m["role"].is_string()) {
      err = "each message must carry a string 'role'";
      return false;
    }
    std::string role = m["role"].get<std::string>();
    if (role == "developer") role = "user";  // OpenAI reasoning-role alias
    if (role == "function") role = "tool";   // legacy role alias
    if (role != "system" && role != "user" && role != "assistant" &&
        role != "tool") {
      err = "unsupported message role '" + role +
            "' (expected system, user, assistant, or tool)";
      return false;
    }
    std::string content;
    if (!text_content(m, content)) return false;
    ChatMessage msg;
    msg.role = std::move(role);
    msg.content = std::move(content);
    const bool has_content =
        m.find("content") != m.end() && !m["content"].is_null();
    if (msg.role == "assistant") {
      auto tcit = m.find("tool_calls");
      if (tcit != m.end() && !tcit->is_null()) {
        if (!tcit->is_array() || tcit->empty()) {
          err = "'tool_calls' must be a non-empty array";
          return false;
        }
        for (const auto& tc : *tcit) {
          if (!tc.is_object()) {
            err = "each entry of 'tool_calls' must be an object";
            return false;
          }
          if (tc.value("type", "function") != "function") {
            err = "only 'function' tool calls are supported";
            return false;
          }
          auto fit = tc.find("function");
          if (fit == tc.end() || !fit->is_object()) {
            err = "each tool call must carry an object 'function'";
            return false;
          }
          auto nit = fit->find("name");
          if (nit == fit->end() || !nit->is_string() ||
              nit->get<std::string>().empty()) {
            err =
                "each tool call function must carry a non-empty string 'name'";
            return false;
          }
          std::string args = "{}";
          auto ait = fit->find("arguments");
          if (ait != fit->end() && !ait->is_null()) {
            if (ait->is_string()) {
              args = ait->get<std::string>();
            } else if (ait->is_object()) {
              args = ait->dump();
            } else {
              err = "tool call 'arguments' must be a JSON string or object";
              return false;
            }
          }
          ToolCall c;
          auto idit = tc.find("id");
          c.id = (idit != tc.end() && idit->is_string())
                     ? idit->get<std::string>()
                     : "";
          c.name = nit->get<std::string>();
          c.arguments = std::move(args);
          msg.tool_calls.push_back(std::move(c));
        }
      }
      if (m.contains("function_call") && !m["function_call"].is_null()) {
        err =
            "unsupported field 'function_call' in messages (legacy function "
            "calling is retired; use 'tool_calls')";
        return false;
      }
      if (!has_content && msg.tool_calls.empty()) {
        err = "assistant messages need 'content' or 'tool_calls'";
        return false;
      }
    } else {
      if (!has_content) {
        err = "each " + msg.role + " message must carry a string 'content'";
        return false;
      }
      if (m.contains("tool_calls") && !m["tool_calls"].is_null()) {
        err = "'tool_calls' is only valid on assistant messages";
        return false;
      }
      if (msg.role == "tool") {
        auto idit = m.find("tool_call_id");
        if (idit != m.end() && !idit->is_null()) {
          if (!idit->is_string()) {
            err = "'tool_call_id' must be a string";
            return false;
          }
          msg.tool_call_id = idit->get<std::string>();
        }
      }
    }
    out.push_back(std::move(msg));
  }
  return true;
}

// Stop strings actually enforced for this request (mirrors Session's
// empty-means-default resolution) for the tool stream parser.
std::vector<std::string> resolved_stops(const ParsedRequest& pr) {
  if (!pr.opts.stop_strings.empty()) return pr.opts.stop_strings;
  return KokkosDNN::LFM25::default_stop_strings();
}

// System-prompt directives assembled from tool_choice (required/named)
// and response_format (JSON bias). Rendered into the system block; the
// model owns no other notion of these OpenAI fields.
std::string make_system_suffix(const ParsedRequest& pr) {
  std::string s = pr.tool_directive;
  if (pr.json_mode) {
    if (!s.empty()) s += "\n";
    s += pr.json_directive;
  }
  return s;
}

// Remove a single surrounding ```json ... ``` / ``` ... ``` fence and trim
// surrounding whitespace. Best-effort presentation cleanup for the
// (soft-enforced) response_format JSON modes; operates on display text.
std::string strip_json_fences(std::string s) {
  auto trim = [](std::string v) {
    size_t b = 0;
    while (b < v.size() && std::isspace((unsigned char)v[b])) ++b;
    size_t e = v.size();
    while (e > b && std::isspace((unsigned char)v[e - 1])) --e;
    return v.substr(b, e - b);
  };
  s = trim(std::move(s));
  if (s.size() < 6 || s.compare(0, 3, "```") != 0) return s;
  const size_t nl = s.find('\n');
  if (nl == std::string::npos) return s;
  s = s.substr(nl + 1);
  const size_t fence = s.rfind("```");
  if (fence == std::string::npos) return s;
  return trim(s.substr(0, fence));
}

// Build the OpenAI `tool_calls` message payload, assigning fresh ids.
json tool_calls_json(ServerState& st, const std::vector<ToolCall>& calls) {
  json tcs = json::array();
  for (const auto& c : calls) {
    tcs.push_back({{"id", new_id("call", st)},
                   {"type", "function"},
                   {"function",
                    {{"name", c.name},
                     {"arguments",
                      c.arguments.empty() ? "{}" : c.arguments}}}});
  }
  return tcs;
}

// Assembled non-streaming turn: OpenAI message payload + finish reason.
// Pure JSON mapping (no inference): `text` is the parser display text in
// tools mode, the detokenized text otherwise; `calls` the parsed native
// calls (post parallel_tool_calls truncation); `session_finish` the
// Session finish string. With calls the message carries `tool_calls`
// (fresh ids), `content` null when no display text remains, and finish
// "tool_calls"; otherwise a plain content message. json_mode strips
// markdown fences from the display text in all cases.
struct AssembledMessage {
  json message;
  std::string finish;
};

AssembledMessage assemble_chat_message(ServerState& st,
                                       const ParsedRequest& pr,
                                       std::string text,
                                       std::vector<ToolCall> calls,
                                       const std::string& session_finish) {
  AssembledMessage out;
  if (pr.json_mode) text = strip_json_fences(std::move(text));
  if (pr.tools_active && !pr.parallel_tool_calls && calls.size() > 1)
    calls.resize(1);
  out.message = {{"role", "assistant"}};
  if (pr.tools_active && !calls.empty()) {
    out.message["content"] =
        text.empty() ? json(nullptr) : json(std::move(text));
    out.message["tool_calls"] = tool_calls_json(st, calls);
    out.finish = Common::to_string(Common::FinishReason::ToolCalls);
  } else {
    out.message["content"] = std::move(text);
    out.finish = session_finish;
  }
  return out;
}

// One SSE `tool_calls` delta payload for call `i` (OpenAI streams each
// call as its own delta chunk carrying index/id/type/function).
json tool_call_delta(size_t i, const std::string& id, const ToolCall& c) {
  return {{"tool_calls",
           json::array(
               {{{"index", i},
                 {"id", id},
                 {"type", "function"},
                 {"function",
                  {{"name", c.name},
                   {"arguments",
                    c.arguments.empty() ? "{}" : c.arguments}}}}})}};
}

// Feed one sampled token through the tool stream parser in tools mode:
// decode the raw piece (specials visible) and return releasable display
// text. Never throws (undecodable ids contribute nothing).
std::string tool_feed(ServerState& st, ToolCallStreamParser& parser,
                      const StepOutcome& s) {
  std::string raw_piece;
  try {
    raw_piece = st.tok.decode(std::vector<int>{s.id}, false);
  } catch (...) {
    raw_piece.clear();
  }
  return parser.push(raw_piece);
}

// ---------------------------------------------------------------------------
// Generation driver (render/encode/prefill/step loop; no model math of its
// own — everything runs through Session).
// ---------------------------------------------------------------------------

struct GenResult {
  std::string text;
  std::string finish;  // OpenAI finish_reason: "stop" | "length"
  size_t prompt_tokens = 0;
  size_t completion_tokens = 0;
};

// Encodes the prompt, caps max_new against the context window, then runs
// prefill + one-token steps to completion. Throws std::invalid_argument on
// over-context prompts (caller maps to 400); other exceptions propagate to
// the 500 path.
GenResult run_session(ServerState& st, const std::vector<int>& prompt_ids,
                      SessionOptions opts, bool has_seed, uint64_t seed,
                      int req_max_tokens,
                      const std::function<void(const StepOutcome&)>& on_step) {
  if (prompt_ids.size() >= static_cast<size_t>(st.cfg.max_seq_len))
    throw std::invalid_argument(
        "prompt is longer than the model's context window (" +
        std::to_string(prompt_ids.size()) + " tokens, max_seq_len=" +
        std::to_string(st.cfg.max_seq_len) + ")");
  const size_t budget =
      static_cast<size_t>(st.cfg.max_seq_len) - prompt_ids.size();
  int max_new = req_max_tokens < 0 ? kDefaultMaxTokens : req_max_tokens;
  if (static_cast<size_t>(max_new) > budget)
    max_new = static_cast<int>(budget);
  opts.max_new = max_new;

  ServerSession sess(st.cfg, st.weights, st.tok, opts);
  if (has_seed)
    sess.prefill(prompt_ids, seed, /*hasSeed=*/true);
  else
    sess.prefill(prompt_ids);
  std::string text;
  while (!sess.finished()) {
    StepOutcome s = sess.step();
    if (!s.delta.empty()) text += s.delta;
    if (on_step) on_step(s);
  }
  text += sess.flush();
  const auto u = sess.usage();
  GenResult r;
  r.text = std::move(text);
  r.finish = Common::to_string(sess.finish());
  r.prompt_tokens = u.prompt_tokens;
  r.completion_tokens = u.completion_tokens;
  return r;
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

void sse_send(httplib::DataSink& sink, const json& payload) {
  const std::string frame = "data: " + payload.dump() + "\n\n";
  sink.write(frame.c_str(), frame.size());
}

void sse_done(httplib::DataSink& sink) {
  static const char kDone[] = "data: [DONE]\n\n";
  sink.write(kDone, sizeof(kDone) - 1);
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

void models_handler(ServerState& st, httplib::Response& res) {
  send_json(res, 200,
            json{{"object", "list"},
                 {"data",
                  json::array(
                      {{{"id", st.model_id},
                        {"object", "model"},
                        {"created", 0},
                        {"owned_by", "kokkosdnn"}}})}});
}

void chat_handler(ServerState& st, const httplib::Request& req,
                  httplib::Response& res) {
  json body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception&) {
    send_bad_request(res, "request body must be valid JSON");
    return;
  }
  ParsedRequest pr;
  std::string err;
  if (!parse_common(body, st.cfg, st.model_id, pr, err) ||
      !parse_messages(body, pr.messages, err)) {
    send_bad_request(res, err);
    return;
  }

  const std::string id = new_id("chatcmpl", st);
  const std::time_t created = std::time(nullptr);

  if (!pr.stream) {
    std::lock_guard<std::mutex> lk(st.mu);
    try {
      const std::vector<int> prompt_ids =
          KokkosDNN::LFM25::encode_chatml_with_tools(
              st.tok, pr.messages, pr.tools_json, make_system_suffix(pr),
              true, true);
      // Tools mode feeds raw per-step pieces (specials visible) through
      // the stream parser so native call blocks become `tool_calls`
      // instead of leaking into content; plain mode keeps Session's
      // detokenized text byte-for-byte.
      ToolCallStreamParser tool_parser(resolved_stops(pr));
      std::string tool_text;
      std::function<void(const StepOutcome&)> on_step;
      if (pr.tools_active) {
        on_step = [&](const StepOutcome& s) {
          tool_text += tool_feed(st, tool_parser, s);
        };
      }
      GenResult r = run_session(st, prompt_ids, pr.opts, pr.has_seed, pr.seed,
                                pr.req_max_tokens, on_step);
      std::string text = r.text;
      std::vector<ToolCall> calls;
      if (pr.tools_active) {
        ToolParseResult tp = tool_parser.finish();
        tool_text += tp.text;
        text = std::move(tool_text);
        calls = std::move(tp.calls);
      }
      AssembledMessage turn =
          assemble_chat_message(st, pr, std::move(text), std::move(calls),
                                r.finish);
      send_json(res, 200,
                json{{"id", id},
                     {"object", "chat.completion"},
                     {"created", created},
                     {"model", st.model_id},
                     {"choices",
                      json::array({{{"index", 0},
                                    {"message", std::move(turn.message)},
                                    {"finish_reason", turn.finish}}})},
                     {"usage",
                      {{"prompt_tokens", r.prompt_tokens},
                       {"completion_tokens", r.completion_tokens},
                       {"total_tokens",
                        r.prompt_tokens + r.completion_tokens}}}});
    } catch (const std::invalid_argument& e) {
      send_bad_request(res, e.what());  // over-context prompts
    } catch (const std::exception& e) {
      send_server_error(res, e.what());
    }
    return;
  }

  // Streaming: headers + chunked provider. Inference runs inside the
  // provider (still under the single-flight mutex); session errors surface
  // as an SSE error event since headers are already sent.
  res.set_header("Cache-Control", "no-cache");
  res.set_header("X-Accel-Buffering", "no");
  res.set_chunked_content_provider(
      "text/event-stream",
      [&st, body = req.body, id, created](size_t /*offset*/,
                                          httplib::DataSink& sink) -> bool {
        std::lock_guard<std::mutex> lk(st.mu);
        try {
          const json b = json::parse(body);  // re-parse: already validated
          ParsedRequest p;
          std::string e;
          if (!parse_common(b, st.cfg, st.model_id, p, e) ||
              !parse_messages(b, p.messages, e)) {
            sse_send(sink, openai_error(e));
            return false;
          }
          const std::vector<int> prompt_ids =
              KokkosDNN::LFM25::encode_chatml_with_tools(
                  st.tok, p.messages, p.tools_json, make_system_suffix(p),
                  true, true);
          auto chunk = [&](const json& delta, const json& finish) {
            json c{{"id", id},
                   {"object", "chat.completion.chunk"},
                   {"created", created},
                   {"model", st.model_id},
                   {"choices",
                    json::array(
                        {{{"index", 0},
                          {"delta", delta},
                          {"finish_reason", finish}}})}};
            sse_send(sink, c);
          };
          chunk({{"role", "assistant"}}, nullptr);
          ToolCallStreamParser tool_parser(resolved_stops(p));
          GenResult r = run_session(
              st, prompt_ids, p.opts, p.has_seed, p.seed, p.req_max_tokens,
              [&](const StepOutcome& s) {
                if (!p.tools_active) {
                  if (!s.delta.empty())
                    chunk({{"content", s.delta}}, nullptr);
                  return;
                }
                std::string d = tool_feed(st, tool_parser, s);
                if (!d.empty()) chunk({{"content", d}}, nullptr);
              });
          // (display text was already streamed piece-wise via on_step.)
          std::string finish = r.finish;
          if (p.tools_active) {
            // (json_mode fence stripping is a non-streaming affordance;
            // streamed deltas arrive unmodified.)
            ToolParseResult tp = tool_parser.finish();
            if (!tp.text.empty()) chunk({{"content", tp.text}}, nullptr);
            std::vector<ToolCall> calls = std::move(tp.calls);
            if (!p.parallel_tool_calls && calls.size() > 1) calls.resize(1);
            if (!calls.empty()) {
              for (size_t i = 0; i < calls.size(); ++i)
                chunk(tool_call_delta(i, new_id("call", st), calls[i]),
                      nullptr);
              finish = Common::to_string(Common::FinishReason::ToolCalls);
            }
          }
          json final = json{{"id", id},
                            {"object", "chat.completion.chunk"},
                            {"created", created},
                            {"model", st.model_id},
                            {"choices",
                             json::array({{{"index", 0},
                                           {"delta", json::object()},
                                           {"finish_reason", finish}}})},
                            {"usage",
                             {{"prompt_tokens", r.prompt_tokens},
                              {"completion_tokens", r.completion_tokens},
                              {"total_tokens",
                               r.prompt_tokens + r.completion_tokens}}}};
          sse_send(sink, final);
          sse_done(sink);
        } catch (const std::exception& ex) {
          sse_send(sink, openai_error(ex.what(), "server_error"));
        }
        return false;  // single-shot provider: done after one pass
      });
}

// POST /v1/completions: legacy alias. {prompt: string} becomes a single
// user message; the rest of the pipeline (sampling, stop, seed, max tokens,
// streaming envelope excepted) is shared with chat.
void completions_handler(ServerState& st, const httplib::Request& req,
                         httplib::Response& res) {
  json body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception&) {
    send_bad_request(res, "request body must be valid JSON");
    return;
  }
  ParsedRequest pr;
  std::string err;
  if (!parse_common(body, st.cfg, st.model_id, pr, err)) {
    send_bad_request(res, err);
    return;
  }
  // The legacy text endpoint has no tool/structured-output protocol.
  for (const char* k :
       {"tools", "tool_choice", "parallel_tool_calls", "response_format"}) {
    if (body.contains(k) && !body[k].is_null()) {
      send_bad_request(res, std::string("'") + k +
                                "' is only supported for "
                                "/v1/chat/completions");
      return;
    }
  }
  auto pit = body.find("prompt");
  if (pit == body.end() || !pit->is_string()) {
    send_bad_request(res, "'prompt' must be a string");
    return;
  }
  pr.messages = KokkosDNN::LFM25::make_chat_messages(
      pit->get<std::string>(), /*system=*/"", "user");

  const std::string id = new_id("cmpl", st);
  const std::time_t created = std::time(nullptr);

  if (!pr.stream) {
    std::lock_guard<std::mutex> lk(st.mu);
    try {
      const std::vector<int> prompt_ids =
          KokkosDNN::LFM25::encode_chatml(st.tok, pr.messages, true, true);
      GenResult r = run_session(st, prompt_ids, pr.opts, pr.has_seed, pr.seed,
                                pr.req_max_tokens, nullptr);
      send_json(res, 200,
                json{{"id", id},
                     {"object", "text_completion"},
                     {"created", created},
                     {"model", st.model_id},
                     {"choices",
                      json::array({{{"text", r.text},
                                    {"index", 0},
                                    {"finish_reason", r.finish}}})},
                     {"usage",
                      {{"prompt_tokens", r.prompt_tokens},
                       {"completion_tokens", r.completion_tokens},
                       {"total_tokens",
                        r.prompt_tokens + r.completion_tokens}}}});
    } catch (const std::invalid_argument& e) {
      send_bad_request(res, e.what());
    } catch (const std::exception& e) {
      send_server_error(res, e.what());
    }
    return;
  }

  res.set_header("Cache-Control", "no-cache");
  res.set_header("X-Accel-Buffering", "no");
  res.set_chunked_content_provider(
      "text/event-stream",
      [&st, body = req.body, id, created](size_t /*offset*/,
                                          httplib::DataSink& sink) -> bool {
        std::lock_guard<std::mutex> lk(st.mu);
        try {
          const json b = json::parse(body);
          ParsedRequest p;
          std::string e;
          if (!parse_common(b, st.cfg, st.model_id, p, e)) {
            sse_send(sink, openai_error(e));
            return false;
          }
          auto it = b.find("prompt");
          if (it == b.end() || !it->is_string()) {
            sse_send(sink, openai_error("'prompt' must be a string"));
            return false;
          }
          p.messages = KokkosDNN::LFM25::make_chat_messages(
              it->get<std::string>(), "", "user");
          const std::vector<int> prompt_ids =
              KokkosDNN::LFM25::encode_chatml(st.tok, p.messages, true, true);
          GenResult r = run_session(
              st, prompt_ids, p.opts, p.has_seed, p.seed, p.req_max_tokens,
              [&](const StepOutcome& s) {
                if (s.delta.empty()) return;
                sse_send(sink,
                         json{{"id", id},
                              {"object", "text_completion.chunk"},
                              {"created", created},
                              {"model", st.model_id},
                              {"choices",
                               json::array({{{"text", s.delta},
                                             {"index", 0},
                                             {"finish_reason", nullptr}}})}});
              });
          sse_send(sink,
                   json{{"id", id},
                        {"object", "text_completion.chunk"},
                        {"created", created},
                        {"model", st.model_id},
                        {"choices",
                         json::array({{{"text", ""},
                                       {"index", 0},
                                       {"finish_reason", r.finish}}})},
                        {"usage",
                         {{"prompt_tokens", r.prompt_tokens},
                          {"completion_tokens", r.completion_tokens},
                          {"total_tokens",
                           r.prompt_tokens + r.completion_tokens}}}});
          sse_done(sink);
        } catch (const std::exception& ex) {
          sse_send(sink, openai_error(ex.what(), "server_error"));
        }
        return false;
      });
}

}  // namespace

// Unit tests (server/test_server_parsing.cpp) define
// KOKKOSDNN_SERVER_NO_MAIN and #include this file to exercise the pure
// JSON helpers above without starting the server.
#ifndef KOKKOSDNN_SERVER_NO_MAIN
int main(int argc, char* argv[]) {
  CliOptions opt;
  {
    int rc = parse_cli(argc, argv, opt);
    if (rc == 2) return 0;  // --help / --version already printed
    if (rc != 0) return rc;
  }

  Kokkos::initialize(argc, argv);
  install_stop_handlers();
  int exit_code = 0;
  {
    startup_log(std::string("kokkosdnn_server ") + kVersion + " starting...");
    startup_log("  model: " + opt.model);
    startup_log("  weights: " +
                (opt.weights.empty() ? std::string("(none)") : opt.weights));
    startup_log("  tokenizer: " +
                (opt.tokenizer.empty() ? std::string("(none)")
                                       : opt.tokenizer));
    startup_log("  config: " +
                (opt.config_path.empty() ? std::string("(baked-in defaults)")
                                         : opt.config_path));
    startup_log("loading weights...");
    ServerState st;
    std::string err;
    if (!load_model(opt, st, err)) {
      startup_log(std::string("kokkosdnn_server: ") + err);
      exit_code = 1;
    } else if (g_stop_requested.load(std::memory_order_relaxed)) {
      startup_log("kokkosdnn_server: shutdown requested during load, exiting");
      exit_code = 0;
    } else {
      startup_log("model loaded: " + st.model_id + " (max_seq_len=" +
                  std::to_string(st.cfg.max_seq_len) + ")");
      startup_log("listening on 0.0.0.0:" + std::to_string(opt.port) +
                  " — ready to serve");
      httplib::Server svr;
      svr.Get("/v1/models", [&](const httplib::Request&,
                                httplib::Response& res) {
        models_handler(st, res);
      });
      svr.Post("/v1/chat/completions",
               [&](const httplib::Request& req, httplib::Response& res) {
                 chat_handler(st, req, res);
               });
      svr.Post("/v1/completions",
               [&](const httplib::Request& req, httplib::Response& res) {
                 completions_handler(st, req, res);
               });
      g_listening_server.store(&svr, std::memory_order_release);
      const bool ok = svr.listen("0.0.0.0", opt.port);
      g_listening_server.store(nullptr, std::memory_order_release);
      if (g_stop_requested.load(std::memory_order_relaxed)) {
        startup_log("kokkosdnn_server: received shutdown signal, stopped");
        exit_code = 0;
      } else if (!ok) {
        startup_log("kokkosdnn_server: failed to bind port " +
                    std::to_string(opt.port));
        exit_code = 1;
      } else {
        startup_log("kokkosdnn_server: stopped");
      }
    }
  }
  Kokkos::finalize();
  return exit_code;
}
#endif  // KOKKOSDNN_SERVER_NO_MAIN
