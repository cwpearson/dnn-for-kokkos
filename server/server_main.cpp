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
  for (const char* k : {"tools", "functions", "function_call", "tool_choice"}) {
    if (body.contains(k) && !body[k].is_null()) {
      err = std::string("unsupported field '") + k + "' (no tool use)";
      return false;
    }
  }
  if (body.contains("response_format") && !body["response_format"].is_null()) {
    err = "unsupported field 'response_format'";
    return false;
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

// messages -> vector<ChatMessage>. "developer" maps to "user". Tool/function
// roles, tool_calls payloads, and non-text content parts are rejected
// (400) rather than silently dropped.
bool parse_messages(const json& body, std::vector<ChatMessage>& out,
                    std::string& err) {
  auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty()) {
    err = "'messages' must be a non-empty array";
    return false;
  }
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
    if (role != "system" && role != "user" && role != "assistant") {
      err = "unsupported message role '" + role +
            "' (expected system, user, or assistant)";
      return false;
    }
    if (m.contains("tool_calls") || m.contains("function_call")) {
      err = "tool/function content is not supported";
      return false;
    }
    auto cit = m.find("content");
    if (cit == m.end() || cit->is_null()) {
      err = "each message must carry a string 'content'";
      return false;
    }
    std::string content;
    if (cit->is_string()) {
      content = cit->get<std::string>();
    } else if (cit->is_array()) {
      // Content-part arrays: join text parts, reject anything else.
      for (const auto& part : *cit) {
        if (!part.is_object() || part.value("type", "") != "text" ||
            !part.contains("text") || !part["text"].is_string()) {
          err = "only text content parts are supported";
          return false;
        }
        content += part["text"].get<std::string>();
      }
    } else {
      err = "each message must carry a string 'content'";
      return false;
    }
    out.push_back(ChatMessage{role, content, ""});
  }
  return true;
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
          KokkosDNN::LFM25::encode_chatml(st.tok, pr.messages, true, true);
      GenResult r = run_session(st, prompt_ids, pr.opts, pr.has_seed, pr.seed,
                                pr.req_max_tokens, nullptr);
      send_json(res, 200,
                json{{"id", id},
                     {"object", "chat.completion"},
                     {"created", created},
                     {"model", st.model_id},
                     {"choices",
                      json::array({{{"index", 0},
                                    {"message",
                                     {{"role", "assistant"},
                                      {"content", r.text}}},
                                    {"finish_reason", r.finish}}})},
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
              KokkosDNN::LFM25::encode_chatml(st.tok, p.messages, true, true);
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
          GenResult r = run_session(
              st, prompt_ids, p.opts, p.has_seed, p.seed, p.req_max_tokens,
              [&](const StepOutcome& s) {
                if (!s.delta.empty()) chunk({{"content", s.delta}}, nullptr);
              });
          // (r.text was already streamed piece-wise via on_step.)
          json final = json{{"id", id},
                            {"object", "chat.completion.chunk"},
                            {"created", created},
                            {"model", st.model_id},
                            {"choices",
                             json::array({{{"index", 0},
                                           {"delta", json::object()},
                                           {"finish_reason", r.finish}}})},
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
  int exit_code = 0;
  {
    ServerState st;
    std::string err;
    if (!load_model(opt, st, err)) {
      std::printf("kokkosdnn_server: %s\n", err.c_str());
      exit_code = 1;
    } else {
      std::printf("kokkosdnn_server %s\n", kVersion);
      std::printf("  model: %s (max_seq_len=%d)\n", st.model_id.c_str(),
                  st.cfg.max_seq_len);
      std::printf("  listening on port %d\n", opt.port);
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
      if (!svr.listen("0.0.0.0", opt.port)) {
        std::printf("kokkosdnn_server: failed to bind port %d\n", opt.port);
        exit_code = 1;
      }
    }
  }
  Kokkos::finalize();
  return exit_code;
}
#endif  // KOKKOSDNN_SERVER_NO_MAIN
