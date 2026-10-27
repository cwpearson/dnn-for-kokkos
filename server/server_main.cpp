// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
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
// and the tokenizer are shared read-only. A streaming client that goes away
// mid-response is detected from sink.write() and its turn stops at the next
// token instead of burning the remaining max_new tokens under that lock.

#include <Kokkos_Core.hpp>

#include <KokkosDNN.hpp>

#include "KokkosDNN_config.h"
#include "KokkosDNN_LFM25Config.hpp"
#include "KokkosDNN_LFM25Session.hpp"
#include "KokkosDNN_LFM25Tokenizer.hpp"
#include "KokkosDNN_LFM25Weights.hpp"

#include <httplib.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using KokkosDNN::LFM25::ChatMessage;
using KokkosDNN::LFM25::Config;
using KokkosDNN::LFM25::ToolCall;
using KokkosDNN::LFM25::ToolCallStreamParser;
using KokkosDNN::LFM25::StreamDeltas;
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
using ServerSessionHost = Session<Exec, Scalar, Kokkos::HostSpace>;
using json = nlohmann::json;

// Single-sourced from the top-level project(VERSION ...): the generated
// KokkosDNN_config.h carries KOKKOSDNN_VERSION, so bumps only touch
// CMakeLists.txt.
constexpr const char* kVersion = KOKKOSDNN_VERSION;
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

[[maybe_unused]] void install_stop_handlers() {
  std::signal(SIGINT, handle_stop_signal);
  std::signal(SIGTERM, handle_stop_signal);
}

// Startup progress goes to stderr (unbuffered enough for `podman logs` to
// show it while weights load) with an explicit flush per line. Per-request
// traffic is logged to stderr too (method + endpoint + body size + status;
// the server `error.message` for non-2xx) via the httplib logger installed
// in main() plus explicit SSE error lines from the streaming providers, and
// one timings line per completed turn (lock/encode/prefill/decode/SSE ms +
// decode tok/s) from log_turn().
[[maybe_unused]] void startup_log(const std::string& msg) {
  std::fprintf(stderr, "%s\n", msg.c_str());
  std::fflush(stderr);
}

// Startup hardware/dispatch report. Without it a mis-provisioned deployment
// (a MIG slice instead of the full device, a lower-arch build JIT-ed forward,
// TPL dispatch compiled out) is indistinguishable from a slow kernel in the
// per-turn timings line. The dispatch list comes from the library's
// print_configuration(); the device block is CUDA-only because SM count and
// clocks have no portable Kokkos query.
[[maybe_unused]] void log_device_info() {
  const std::string es_name = Exec::name();
  startup_log(std::string("  execution space: ") + es_name +
              " (concurrency=" + std::to_string(Exec().concurrency()) + ")");
#ifdef KOKKOS_ENABLE_CUDA
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return;
  // CUDA 13 dropped clockRate/memoryClockRate from cudaDeviceProp; the
  // attributes are still queryable and tell an operator whether the device
  // is clocked/limited as expected.
  int sm_clock_khz = 0, mem_clock_khz = 0;
  cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  char line[512];
  std::snprintf(
      line, sizeof(line),
      "  device %d: %s (compute %d.%d, %d SMs, %.1f GiB, SM clock %d MHz, "
      "memory %d MHz x %d-bit)",
      dev, p.name, p.major, p.minor, p.multiProcessorCount,
      static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
      sm_clock_khz / 1000, mem_clock_khz / 1000, p.memoryBusWidth);
  startup_log(line);
  // A MIG slice presents fewer SMs and a proportional share of bandwidth;
  // batch-1 decode scales with both (grids are small at T == 1), so this is
  // the first thing to check when tok/s lands far below the GPU's class.
  if (p.multiProcessorCount < 64)
    startup_log(
        "  note: <64 SMs visible; if this is not the whole GPU (MIG slice or "
        "device busy), decode throughput will be a fraction of full-device");
#endif
}

// Single-line stderr logging: control chars that would split lines are
// blanked and long details truncated so one request stays one line.
[[maybe_unused]] std::string log_sanitize(std::string s,
                                          size_t max_len = 300) {
  for (char& c : s) {
    if (c == '\n' || c == '\r' || c == '\t') c = ' ';
  }
  if (s.size() > max_len) s = s.substr(0, max_len) + "...";
  return s;
}

// Extracts the `error.message` carried by send_bad_request/send_server_error
// (and the 404 error_handler) so request logs name the failure cause.
// Falls back to the raw body snippet when it is not the expected JSON.
[[maybe_unused]] std::string request_error_detail(
    const httplib::Response& res) {
  if (res.body.empty()) return "";
  try {
    const json j = json::parse(res.body);
    const auto it = j.find("error");
    if (it != j.end() && it->is_object()) {
      const auto mit = it->find("message");
      if (mit != it->end() && mit->is_string())
        return mit->get<std::string>();
    }
  } catch (...) {
  }
  return res.body;
}

// httplib logger: every request (including unmatched routes via the 404
// error_handler) logs `METHOD target body_bytes=N -> STATUS` with
// `error="..."` appended for non-2xx. Single fprintf per line; httplib
// serializes logger calls and fprintf itself is thread-safe.
[[maybe_unused]] void log_request(const httplib::Request& req,
                                  const httplib::Response& res) {
  const std::string target = req.target.empty() ? req.path : req.target;
  std::string line = req.method + " " + target +
                     " body_bytes=" + std::to_string(req.body.size()) +
                     " -> " + std::to_string(res.status);
  if (res.status < 200 || res.status >= 300) {
    const std::string detail = log_sanitize(request_error_detail(res));
    if (!detail.empty()) line += " error=\"" + detail + "\"";
  }
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

// Streaming providers answer 200 before generation runs, so mid-stream
// failures travel as SSE `data: {"error":...}` events the logger (which runs
// before the streamed body) cannot see. Each SSE error site logs its own
// line with the same endpoint/body-size prefix.
[[maybe_unused]] void log_stream_error(const std::string& method,
                                       const std::string& target,
                                       size_t body_bytes,
                                       const std::string& err) {
  std::string line = method + " " + target +
                     " body_bytes=" + std::to_string(body_bytes) +
                     " -> sse-error error=\"" + log_sanitize(err) + "\"";
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

// ---------------------------------------------------------------------------
// Per-turn timing
// ---------------------------------------------------------------------------
// Where a request's wall time goes, filled in by run_session()/the handlers.
// Reported as one stderr line per completed turn (log_turn) so an operator can
// tell model time (prefill_ms + decode_ms) from server time (lock_ms +
// encode_ms + sse_ms) and read the decode rate the client actually sees.
struct TurnTimings {
  double lock_ms = 0.0;     // waiting for the single-flight inference mutex
  double encode_ms = 0.0;   // render + encode the full message history
  double prefill_ms = 0.0;  // suffix prefill after prefix-cache reuse
  double decode_ms = 0.0;   // sum of Session::step() (model forward + detok)
  double sse_ms = 0.0;      // SSE framing + socket writes inside the loop
  bool aborted = false;     // peer disconnected; generation stopped early
  bool prefix_hit = false;  // a cached content prefix was reused
  size_t prefix_reused = 0;  // reused prefix length in tokens (0 on miss)
};

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

// One line per completed turn: prompt/completion counts, the split above, and
// decode tok/s (completion tokens over decode time — the rate a streaming
// client observes, prefill excluded).
void log_turn(const std::string& method, const std::string& target, int status,
              const TurnTimings& tm, size_t prompt_tokens,
              size_t completion_tokens) {
  const double tok_s =
      tm.decode_ms > 0.0 ? completion_tokens * 1000.0 / tm.decode_ms : 0.0;
  char tail[384];
  std::snprintf(tail, sizeof(tail),
                " prompt_tokens=%zu completion_tokens=%zu lock_ms=%.1f"
                " encode_ms=%.1f prefill_ms=%.1f decode_ms=%.1f sse_ms=%.1f"
                " decode_tok_s=%.1f prefix_hit=%d prefix_reused=%zu%s",
                prompt_tokens, completion_tokens, tm.lock_ms, tm.encode_ms,
                tm.prefill_ms, tm.decode_ms, tm.sse_ms, tok_s,
                tm.prefix_hit ? 1 : 0, tm.prefix_reused,
                tm.aborted ? " aborted=client-disconnect" : "");
  std::fprintf(stderr, "%s %s -> %d%s\n", method.c_str(), target.c_str(),
               status, tail);
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
  int max_seq_len = 0;  // 0 => keep the config file / preset value
  bool prefix_cache_host = false;  // keep the KV-cache entry in host memory
};

// Returns an exit code when the process should stop after parse_cli
// (0 = ok to continue serving, anything else = exit with that code).
[[maybe_unused]] int parse_cli(int argc, char* argv[], CliOptions& out) {
  auto help = [&]() {
    std::printf(
        "Usage: %s [--port N] [--model ID] [--weights PATH] [--tokenizer "
        "PATH]\n"
        "             [--config PATH] [--max-seq-len N] [--help] "
        "[--version]\n"
        "             [--prefix-cache-host]\n"
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
        "  --max-seq-len N   context window to serve (overrides the config;\n"
        "                    the KV cache is allocated for N tokens)\n"
        "  --prefix-cache-host\n"
        "                    keep the single prefix-cache KV entry in host\n"
        "                    memory instead of device memory (saves VRAM at\n"
        "                    the cost of host<->device copies per turn)\n"
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
    } else if (std::strcmp(a, "--max-seq-len") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value for --max-seq-len\n");
        return 1;
      }
      out.max_seq_len = std::atoi(argv[++i]);
      if (out.max_seq_len <= 0) {
        std::printf("--max-seq-len must be a positive integer\n");
        return 1;
      }
    } else if (std::strcmp(a, "--prefix-cache-host") == 0) {
      out.prefix_cache_host = true;
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
  // Long-lived decoding session reused across requests (guarded by mu).
  // Per-request construction re-rents the full KV cache and deep-copies
  // the tokenizer (~170ms TTFT); Session::prefill_cached() instead keeps a
  // single-entry prefix cache of content states across turns (exact-prefix
  // reuse for agentic loops) while still resetting all turn state, and
  // run_session() re-applies per-request options + seed. The concrete
  // Session instantiation (device- vs host-resident cache entry) is picked
  // from prefix_cache_host at construction; the interface erases it.
  bool prefix_cache_host = false;
  std::unique_ptr<KokkosDNN::LFM25::ISession> sess;
};

// Loads Config (HF defaults + optional JSON overlay), GGUF weights, and the
// tokenizer. Requires Kokkos to be initialized (allocates device views).
// Returns true on success; on failure `err` describes the problem.
[[maybe_unused]] bool load_model(const CliOptions& o, ServerState& st, std::string& err) {
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
    if (o.max_seq_len > 0) st.cfg.max_seq_len = o.max_seq_len;
    st.cfg.validate();
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
// OpenAI `tool_calls` (validated, arguments string or object) plus a
// `thinking` / `reasoning` / `reasoning_content` trace (first non-empty
// wins, replayed into the prompt's <think> block). "tool" turns carry the
// function result plus an optional `tool_call_id` link; assistant
// `content` may be null when tool_calls are present, and is required otherwise; every
// other role requires string content.
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
      // Thinking replay: Zed (interleaved_reasoning) and the LFM2.5
      // chat template itself (`thinking or reasoning or
      // reasoning_content`) carry the trace outside `content`. Accept all
      // three spellings so multi-turn history preserves the trace for the
      // prompt renderer; first non-empty wins.
      for (const char* k : {"thinking", "reasoning", "reasoning_content"}) {
        auto tit = m.find(k);
        if (tit != m.end() && tit->is_string() &&
            !tit->get<std::string>().empty()) {
          msg.thinking = tit->get<std::string>();
          break;
        }
      }
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
// every mode (never Session's raw detokenized text); `reasoning` is the parsed
// <think> trace (emitted as `reasoning_content`, the DeepSeek/OpenRouter
// convention Zed and other OpenAI-compatible clients read); `calls` the
// parsed native calls (post parallel_tool_calls truncation);
// `session_finish` the Session finish string. With calls the message
// carries `tool_calls` (fresh ids), `content` null when no display text
// remains, and finish "tool_calls" — even when `session_finish` was
// "length": a recovered length-capped turn hides the cap, so callers must
// treat the arguments as possibly truncated. Otherwise a plain content
// message.
// json_mode strips markdown fences from the display text in all cases.
struct AssembledMessage {
  json message;
  std::string finish;
};

AssembledMessage assemble_chat_message(ServerState& st,
                                       const ParsedRequest& pr,
                                       std::string text,
                                       std::string reasoning,
                                       std::vector<ToolCall> calls,
                                       const std::string& session_finish) {
  AssembledMessage out;
  if (pr.json_mode) text = strip_json_fences(std::move(text));
  if (pr.tools_active && !pr.parallel_tool_calls && calls.size() > 1)
    calls.resize(1);
  out.message = {{"role", "assistant"}};
  // The trace travels out-of-band so it never pollutes `content` (Zed
  // renders `reasoning_content` in its thinking trace, not the answer).
  if (!reasoning.empty()) out.message["reasoning_content"] = std::move(reasoning);
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

// Feed one sampled token through the stream parser: decode the raw piece
// (specials visible) and return the display + reasoning newly releasable.
// The parser holds back <think> traces and tool-call regions so neither
// leaks into `content`. Used in every mode so the thinking trace never
// pollutes the answer with or without tools. One parser step per token
// keeps the per-turn cost linear. Never throws: an undecodable id
// contributes nothing, and a parser failure degrades to a held delta
// (recovered at finish) instead of killing the turn with a 500 or a
// mid-stream SSE error.
StreamDeltas tool_feed(ServerState& st, ToolCallStreamParser& parser,
                       const StepOutcome& s) {
  try {
    // Persistent 1-elem staging: tools-mode decode already costs a map
    // lookup + string build per token; no need for a vector malloc too.
    thread_local std::vector<int> idbuf{0};
    idbuf[0] = s.id;
    return parser.feed(st.tok.decode(idbuf, false));
  } catch (...) {
    return {};
  }
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
//
// full_ids is the whole prompt (history + trailing generation prompt);
// content_ids is the cacheable history prefix (same encoding without the
// generation prompt, expected to be an exact prefix of full_ids). The
// single-entry prefix cache resumes from the cached content on exact
// matches; a mismatch falls back to a full prefill internally.
//
// on_step may return false to stop generation early (the streaming providers
// do this when sink.write() reports the peer is gone: the single-flight
// inference lock makes generating for a dead client steal a whole completion
// from the next queued request).
GenResult run_session(ServerState& st, const std::vector<int>& full_ids,
                      const std::vector<int>& content_ids, SessionOptions opts,
                      bool has_seed, uint64_t seed, int req_max_tokens,
                      const std::function<bool(const StepOutcome&)>& on_step,
                      TurnTimings* tm) {
  if (full_ids.size() >= static_cast<size_t>(st.cfg.max_seq_len))
    throw std::invalid_argument(
        "prompt is longer than the model's context window (" +
        std::to_string(full_ids.size()) + " tokens, max_seq_len=" +
        std::to_string(st.cfg.max_seq_len) + ")");
  const size_t budget =
      static_cast<size_t>(st.cfg.max_seq_len) - full_ids.size();
  int max_new = req_max_tokens < 0 ? kDefaultMaxTokens : req_max_tokens;
  if (static_cast<size_t>(max_new) > budget)
    max_new = static_cast<int>(budget);
  opts.max_new = max_new;

  // Reuse one long-lived Session: avoids re-renting the KV cache and
  // re-copying the tokenizer on every request. set_options() +
  // prefill_cached() reset all turn state; the forced reseed reproduces
  // fresh-session RNG behavior exactly (construction seeds from opts.seed,
  // an explicit request seed wins, unseeded requests deterministically
  // restart the default stream). The cache-entry placement (device vs host
  // memory) is fixed at construction from --prefix-cache-host.
  if (!st.sess) {
    if (st.prefix_cache_host)
      st.sess = std::make_unique<ServerSessionHost>(
          st.cfg, st.weights, st.tok, opts);
    else
      st.sess =
          std::make_unique<ServerSession>(st.cfg, st.weights, st.tok, opts);
  } else {
    st.sess->set_options(opts);
  }
  KokkosDNN::LFM25::ISession& sess = *st.sess;
  const double t_prefill = now_ms();
  const KokkosDNN::LFM25::CachedPrefillOutcome pc =
      sess.prefill_cached(full_ids, content_ids, has_seed ? seed : opts.seed,
                          /*hasSeed=*/true);
  if (tm) {
    tm->prefill_ms = now_ms() - t_prefill;
    tm->prefix_hit = pc.hit;
    tm->prefix_reused = pc.reused;
  }
  std::string text;
  while (!sess.finished()) {
    const double t_step = now_ms();
    StepOutcome s = sess.step();
    const double t_step_done = now_ms();
    if (!s.delta.empty()) text += s.delta;
    const bool keep_going = on_step ? on_step(s) : true;
    if (tm) {
      tm->decode_ms += t_step_done - t_step;
      tm->sse_ms += now_ms() - t_step_done;
      if (!keep_going) tm->aborted = true;
    }
    if (!keep_going) break;
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
// cpp-httplib v0.63.0 does not buffer a chunked provider's writes: each
// sink.write() emits one chunk header + payload + CRLF and sends it on the
// socket immediately (plus a poll() for writability), so these calls are the
// only transport hop between a decoded token and the client. The return value
// is the liveness signal — false once the peer is gone.
bool sse_send(httplib::DataSink& sink, const json& payload) {
  const std::string frame = "data: " + payload.dump() + "\n\n";
  return sink.write(frame.data(), frame.size());
}

bool sse_done(httplib::DataSink& sink) {
  static const char kDone[] = "data: [DONE]\n\n";
  return sink.write(kDone, sizeof(kDone) - 1);
}

// Peer liveness for a streaming response. A write only reports an error once
// the transport gives up (RST); a half-closed peer (FIN) can absorb many
// small SSE frames first, so a long generation can keep running for a client
// that is already gone. is_writable() is httplib's non-destructive
// select()+MSG_PEEK on the socket, which notices the FIN immediately.
//
// Polled every kLivenessPollSteps decoded tokens: one syscall per 16 tokens
// is negligible next to a forward pass, and polling per token would add one
// to every step. The single-flight inference lock means a dead client that is
// not detected costs the *next* queued request the whole completion, which is
// exactly the multi-second queueing seen in deployed agentic loops.
constexpr size_t kLivenessPollSteps = 16;

bool sse_peer_alive(httplib::DataSink& sink) { return sink.is_writable(); }

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

[[maybe_unused]] void models_handler(ServerState& st, httplib::Response& res) {
  send_json(res, 200,
            json{{"object", "list"},
                 {"data",
                  json::array(
                      {{{"id", st.model_id},
                        {"object", "model"},
                        {"created", 0},
                        {"owned_by", "kokkosdnn"}}})}});
}

[[maybe_unused]] void chat_handler(ServerState& st, const httplib::Request& req,
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
    const double t_lock = now_ms();
    std::lock_guard<std::mutex> lk(st.mu);
    TurnTimings tm;
    tm.lock_ms = now_ms() - t_lock;
    try {
      const double t_encode = now_ms();
      const std::string sys_suffix = make_system_suffix(pr);
      // Full prompt plus the cacheable history prefix (same encoding
      // without the trailing generation prompt). The prefix is an exact
      // token prefix of the full prompt, so the next turn resumes from it.
      const std::vector<int> content_ids =
          KokkosDNN::LFM25::encode_chatml_with_tools(
              st.tok, pr.messages, pr.tools_json, sys_suffix,
              /*add_generation_prompt=*/false, true);
      const std::vector<int> full_ids =
          KokkosDNN::LFM25::encode_chatml_with_tools(
              st.tok, pr.messages, pr.tools_json, sys_suffix, true, true);
      tm.encode_ms = now_ms() - t_encode;
      // Every turn feeds raw per-step pieces (specials visible) through
      // the stream parser: native call blocks become `tool_calls` and the
      // <think> trace becomes `reasoning_content` instead of leaking into
      // `content`. The parser display replaces Session's detokenized text
      // in every mode (equal in plain-text turns containing only handled
      // markers; any other special token stays raw in the parser display).
      // Without tools, parsed calls are
      // dropped — the client asked for text, not function calls.
      // The prompt ends with an open <think>, so the trace opens
      // implicitly and its bytes carry no opener.
      ToolCallStreamParser tool_parser(resolved_stops(pr),
                                       /*start_in_reasoning=*/true);
      std::string tool_text, tool_reasoning;
      std::function<bool(const StepOutcome&)> on_step =
          [&](const StepOutcome& s) {
            StreamDeltas d = tool_feed(st, tool_parser, s);
            tool_text += std::move(d.display);
            tool_reasoning += std::move(d.reasoning);
            return true;
          };
      GenResult r = run_session(st, full_ids, content_ids, pr.opts, pr.has_seed,
                                pr.seed, pr.req_max_tokens, on_step, &tm);
      ToolParseResult tp = tool_parser.finish();
      tool_text += tp.text;
      tool_reasoning += tp.reasoning;
      std::vector<ToolCall> calls;
      if (pr.tools_active) calls = std::move(tp.calls);
      AssembledMessage turn = assemble_chat_message(
          st, pr, std::move(tool_text), std::move(tool_reasoning),
          std::move(calls), r.finish);
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
      log_turn(req.method, req.target.empty() ? req.path : req.target, 200, tm,
               r.prompt_tokens, r.completion_tokens);
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
      [&st, method = req.method,
       target = req.target.empty() ? req.path : req.target,
       body = req.body, id, created](size_t /*offset*/,
                                     httplib::DataSink& sink) -> bool {
        const double t_lock = now_ms();
        std::lock_guard<std::mutex> lk(st.mu);
        TurnTimings tm;
        tm.lock_ms = now_ms() - t_lock;
        try {
          const json b = json::parse(body);  // re-parse: already validated
          ParsedRequest p;
          std::string e;
          if (!parse_common(b, st.cfg, st.model_id, p, e) ||
              !parse_messages(b, p.messages, e)) {
            sse_send(sink, openai_error(e));
            log_stream_error(method, target, body.size(), e);
            sink.done();
            return true;
          }
          const double t_encode = now_ms();
          const std::string sys_suffix = make_system_suffix(p);
          const std::vector<int> content_ids =
              KokkosDNN::LFM25::encode_chatml_with_tools(
                  st.tok, p.messages, p.tools_json, sys_suffix,
                  /*add_generation_prompt=*/false, true);
          const std::vector<int> full_ids =
              KokkosDNN::LFM25::encode_chatml_with_tools(
                  st.tok, p.messages, p.tools_json, sys_suffix, true, true);
          tm.encode_ms = now_ms() - t_encode;
          bool sse_ok = true;
          size_t steps_since_poll = 0;
          auto chunk = [&](const json& delta, const json& finish) {
            if (!sse_ok) return;
            json c{{"id", id},
                   {"object", "chat.completion.chunk"},
                   {"created", created},
                   {"model", st.model_id},
                   {"choices",
                    json::array(
                        {{{"index", 0},
                          {"delta", delta},
                          {"finish_reason", finish}}})}};
            sse_ok = sse_send(sink, c);
          };
          chunk({{"role", "assistant"}}, nullptr);
          // Nothing has been generated yet: if the peer is already gone, do
          // not pay for this turn's prefill (seconds at long context) under
          // the single-flight lock.
          if (!sse_ok || !sse_peer_alive(sink)) {
            tm.aborted = true;
            log_turn(method, target, 200, tm, 0, 0);
            sink.done();
            return true;
          }
          ToolCallStreamParser tool_parser(resolved_stops(p),
                                           /*start_in_reasoning=*/true);
          GenResult r = run_session(
              st, full_ids, content_ids, p.opts, p.has_seed, p.seed,
              p.req_max_tokens,
              [&](const StepOutcome& s) {
                // Cheap periodic liveness check: picks up a client that
                // disconnected without the socket write failing yet.
                if (++steps_since_poll >= kLivenessPollSteps) {
                  steps_since_poll = 0;
                  if (!sse_peer_alive(sink)) sse_ok = false;
                }
                if (!sse_ok) return false;
                // <think> traces stream as `reasoning_content` (Zed renders
                // them in the thinking trace, not the answer); display text
                // as `content`. Never the twain shall leak.
                StreamDeltas d = tool_feed(st, tool_parser, s);
                if (!d.display.empty())
                  chunk({{"content", std::move(d.display)}}, nullptr);
                if (!d.reasoning.empty())
                  chunk({{"reasoning_content", std::move(d.reasoning)}},
                        nullptr);
                return sse_ok;
              },
              &tm);
          // (display + reasoning were already streamed piece-wise via
          // on_step.)
          std::string finish = r.finish;
          if (sse_ok) {
            // (json_mode fence stripping is a non-streaming affordance;
            // streamed deltas arrive unmodified.)
            ToolParseResult tp = tool_parser.finish();
            if (!tp.text.empty()) chunk({{"content", tp.text}}, nullptr);
            if (!tp.reasoning.empty())
              chunk({{"reasoning_content", tp.reasoning}}, nullptr);
            // Without tools the client asked for text: parsed calls are
            // dropped (their blocks were already excised from `content`).
            std::vector<ToolCall> calls;
            if (p.tools_active) calls = std::move(tp.calls);
            if (!p.parallel_tool_calls && calls.size() > 1) calls.resize(1);
            if (!calls.empty()) {
              for (size_t i = 0; i < calls.size(); ++i)
                chunk(tool_call_delta(i, new_id("call", st), calls[i]),
                      nullptr);
              finish = Common::to_string(Common::FinishReason::ToolCalls);
            }
          }
          if (sse_ok) {
            json final =
                json{{"id", id},
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
          }
          log_turn(method, target, 200, tm, r.prompt_tokens,
                   r.completion_tokens);
        } catch (const std::exception& ex) {
          sse_send(sink, openai_error(ex.what(), "server_error"));
          log_stream_error(method, target, body.size(), ex.what());
        }
        // Terminate the chunked body cleanly: sink.done() emits the
        // trailing 0-length chunk; returning true reports success. Returning
        // false here would abort without the terminator and clients see
        // "curl: (18) transfer closed with outstanding read data remaining".
        sink.done();
        return true;
      });
}

// POST /v1/completions: legacy alias. {prompt: string} becomes a single
// user message; the rest of the pipeline (sampling, stop, seed, max tokens,
// streaming envelope excepted) is shared with chat.
[[maybe_unused]] void completions_handler(ServerState& st, const httplib::Request& req,
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
    const double t_lock = now_ms();
    std::lock_guard<std::mutex> lk(st.mu);
    TurnTimings tm;
    tm.lock_ms = now_ms() - t_lock;
    try {
      const double t_encode = now_ms();
      const std::vector<int> content_ids =
          KokkosDNN::LFM25::encode_chatml(st.tok, pr.messages, false, true);
      const std::vector<int> full_ids =
          KokkosDNN::LFM25::encode_chatml(st.tok, pr.messages, true, true);
      tm.encode_ms = now_ms() - t_encode;
      // Legacy envelope has no reasoning/tool channels: the <think> trace
      // and any call blocks are stripped from the text (never leaked).
      // Only the display channel is consumed (no reasoning poll).
      // The prompt ends with an open <think>, so the trace opens
      // implicitly and its bytes carry no opener.
      ToolCallStreamParser tool_parser(resolved_stops(pr),
                                       /*start_in_reasoning=*/true);
      std::string tool_text;
      std::function<bool(const StepOutcome&)> on_step =
          [&](const StepOutcome& s) {
            tool_text += tool_feed(st, tool_parser, s).display;
            return true;
          };
      GenResult r = run_session(st, full_ids, content_ids, pr.opts, pr.has_seed,
                                pr.seed, pr.req_max_tokens, on_step, &tm);
      tool_text += tool_parser.finish().text;
      send_json(res, 200,
                json{{"id", id},
                     {"object", "text_completion"},
                     {"created", created},
                     {"model", st.model_id},
                     {"choices",
                      json::array({{{"text", tool_text},
                                    {"index", 0},
                                    {"finish_reason", r.finish}}})},
                     {"usage",
                      {{"prompt_tokens", r.prompt_tokens},
                       {"completion_tokens", r.completion_tokens},
                       {"total_tokens",
                        r.prompt_tokens + r.completion_tokens}}}});
      log_turn(req.method, req.target.empty() ? req.path : req.target, 200, tm,
               r.prompt_tokens, r.completion_tokens);
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
      [&st, method = req.method,
       target = req.target.empty() ? req.path : req.target,
       body = req.body, id, created](size_t /*offset*/,
                                     httplib::DataSink& sink) -> bool {
        const double t_lock = now_ms();
        std::lock_guard<std::mutex> lk(st.mu);
        TurnTimings tm;
        tm.lock_ms = now_ms() - t_lock;
        try {
          const json b = json::parse(body);
          ParsedRequest p;
          std::string e;
          if (!parse_common(b, st.cfg, st.model_id, p, e)) {
            sse_send(sink, openai_error(e));
            log_stream_error(method, target, body.size(), e);
            sink.done();
            return true;
          }
          auto it = b.find("prompt");
          if (it == b.end() || !it->is_string()) {
            sse_send(sink, openai_error("'prompt' must be a string"));
            log_stream_error(method, target, body.size(),
                             "'prompt' must be a string");
            sink.done();
            return true;
          }
          p.messages = KokkosDNN::LFM25::make_chat_messages(
              it->get<std::string>(), "", "user");
          // This endpoint has no opening chunk, so this is the last chance to
          // notice a peer that left before paying for the prefill.
          if (!sse_peer_alive(sink)) {
            tm.aborted = true;
            log_turn(method, target, 200, tm, 0, 0);
            sink.done();
            return true;
          }
          size_t steps_since_poll = 0;
          const double t_encode = now_ms();
          const std::vector<int> content_ids =
              KokkosDNN::LFM25::encode_chatml(st.tok, p.messages, false, true);
          const std::vector<int> full_ids =
              KokkosDNN::LFM25::encode_chatml(st.tok, p.messages, true, true);
          tm.encode_ms = now_ms() - t_encode;
          bool sse_ok = true;
          // Legacy envelope has no reasoning/tool channels: stream display
          // text only (the <think> trace and call blocks never leak).
          ToolCallStreamParser tool_parser(resolved_stops(p),
                                           /*start_in_reasoning=*/true);
          GenResult r = run_session(
              st, full_ids, content_ids, p.opts, p.has_seed, p.seed,
              p.req_max_tokens,
              [&](const StepOutcome& s) {
                if (++steps_since_poll >= kLivenessPollSteps) {
                  steps_since_poll = 0;
                  if (!sse_peer_alive(sink)) sse_ok = false;
                }
                if (!sse_ok) return false;
                const std::string d =
                    tool_feed(st, tool_parser, s).display;
                if (d.empty()) return sse_ok;
                sse_ok = sse_send(
                    sink, json{{"id", id},
                               {"object", "text_completion.chunk"},
                               {"created", created},
                               {"model", st.model_id},
                               {"choices",
                                json::array({{{"text", d},
                                              {"index", 0},
                                              {"finish_reason", nullptr}}})}});
                return sse_ok;
              },
              &tm);
          if (sse_ok) {
            const std::string tail = tool_parser.finish().text;
            if (!tail.empty())
              sse_send(sink,
                       json{{"id", id},
                            {"object", "text_completion.chunk"},
                            {"created", created},
                            {"model", st.model_id},
                            {"choices",
                             json::array({{{"text", tail},
                                           {"index", 0},
                                           {"finish_reason", nullptr}}})}});
          }
          if (sse_ok) {
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
          }
          log_turn(method, target, 200, tm, r.prompt_tokens,
                   r.completion_tokens);
        } catch (const std::exception& ex) {
          sse_send(sink, openai_error(ex.what(), "server_error"));
          log_stream_error(method, target, body.size(), ex.what());
        }
        // See chat_handler: sink.done() + true terminates the chunked body
        // cleanly; returning false would truncate it (curl error 18).
        sink.done();
        return true;
      });
}

}  // namespace

// Unit tests (server/test_server_parsing.cpp) define
// KOKKOSDNN_SERVER_NO_MAIN and #include this file to exercise the pure
// JSON helpers above without starting the server. main() and the route
// handlers it alone references are therefore unreferenced in that TU;
// they carry [[maybe_unused]] so the test build stays -Werror-clean.
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
    startup_log(std::string("  prefix cache: single entry in ") +
                (opt.prefix_cache_host ? "host memory (--prefix-cache-host)"
                                       : "device memory"));
    log_device_info();
    {
      std::ostringstream os;
      KokkosDNN::print_configuration(os);
      std::istringstream lines(os.str());
      std::string line;
      while (std::getline(lines, line)) startup_log("  " + line);
    }
    startup_log("loading weights...");
    ServerState st;
    st.prefix_cache_host = opt.prefix_cache_host;
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
      // SSE is one small write per token. cpp-httplib defaults to
      // CPPHTTPLIB_TCP_NODELAY=false, i.e. Nagle stays enabled on accepted
      // sockets; on a non-loopback path (proxy, WAN client) Nagle + delayed
      // ACK can hold a frame back ~40 ms. Nagle costs nothing to disable and
      // the stream is latency-, not bandwidth-bound.
      svr.set_tcp_nodelay(true);
      // Bound one SSE/proxy write. cpp-httplib defaults to 5 s; a stalled
      // reader otherwise holds the single-flight inference lock for up to
      // that long per frame. SSE frames are ~100 B, so 2 s is already
      // generous for a healthy client and bounds a dead-but-not-reset peer.
      svr.set_write_timeout(2, 0);
      // Log every request: METHOD target body_bytes=N -> STATUS plus
      // error="..." for non-2xx (extracted from the JSON error body).
      svr.set_logger([](const httplib::Request& req,
                        const httplib::Response& res) { log_request(req, res); });
      // Unmatched routes would otherwise reply with httplib's plain-text
      // 404; return the OpenAI error shape instead so the logger above can
      // report the cause. Handler-set error bodies are left untouched.
      svr.set_error_handler([](const httplib::Request& req,
                               httplib::Response& res) {
        if (!res.body.empty())
          return httplib::Server::HandlerResponse::Unhandled;
        const std::string target = req.target.empty() ? req.path : req.target;
        send_json(res, 404,
                  openai_error("unknown endpoint '" + target + "'"));
        return httplib::Server::HandlerResponse::Handled;
      });
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
