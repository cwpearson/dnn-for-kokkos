// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// server/Backend.hpp: extensible multi-architecture model backend interface.
//
// The HTTP layer (server_main.cpp) talks to models only through
// KokkosDNN::Server::ChatBackend. Architecture specifics — chat template,
// stop defaults, tool-call grammar, weight formats, session mechanics —
// live in per-arch translation units (BackendLFM25.cpp, BackendGlimmer.cpp)
// that register themselves by name. Adding a third architecture is a new
// .cpp implementing ChatBackend plus a static register_backend() call (see
// the recipe in server/README.md); no handler or CLI edits are needed.
//
// This header is deliberately free of Kokkos/httplib/nlohmann dependencies
// so backends and parsing tests can include it anywhere.
#ifndef KOKKOSDNN_SERVER_BACKEND_HPP
#define KOKKOSDNN_SERVER_BACKEND_HPP

#include <KokkosDNN_Sampler.hpp>
#include <KokkosDNN_Usage.hpp>

#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace KokkosDNN {
namespace Server {

// Canonical architecture names served by the registry.
inline constexpr const char* kArchLFM25 = "lfm2.5";
inline constexpr const char* kArchGlimmer = "glimmer";

// ---------------------------------------------------------------------------
// Neutral request/session vocabulary (no per-arch types leak into handlers)
// ---------------------------------------------------------------------------

struct ServerToolCall {
  std::string id;         // OpenAI tool_call id ("" when unknown/unparsed)
  std::string name;       // function name
  std::string arguments;  // JSON object string ("" or "{}" when none)
};

struct ServerChatMessage {
  std::string role;      // "system" | "user" | "assistant" | "tool"
  std::string content;
  std::string thinking;  // assistant reasoning trace (may be empty)
  std::vector<ServerToolCall> tool_calls;
  std::string tool_call_id;
};

// Per-turn sampling/stop/seed policy. Mirrors LFM25::SessionOptions field
// for field so backends convert mechanically; empty stop_ids/strings mean
// "arch defaults" (resolved per backend).
struct ServerSessionOptions {
  ::KokkosDNN::SampleOptions sampling;
  std::vector<int> stop_ids;
  std::vector<std::string> stop_strings;
  int max_new = 256;  // completion-token cap per turn
  uint64_t seed = 0;
  bool seedProvided = false;
};

// One sampled token. Display text is NOT carried here: handlers re-decode
// the id with specials visible (decode_piece) and feed it through the
// backend's StreamParser, so tool-call markup and thinking traces never
// leak into `content` in any mode.
struct BackendStepOutcome {
  int id = -1;  // sampled token (-1 on terminal outcomes that sample nothing)
  bool finished = false;
  // OpenAI finish vocabulary ("stop" | "length"; Common::to_string values).
  std::string finish = "stop";
};

// Per-turn prefix-cache outcome for logging/observability.
struct CachedPrefill {
  bool hit = false;
  size_t reused = 0;  // reused prefix length in tokens (0 on miss)
};

struct StepDeltas {
  std::string display;
  std::string reasoning;
};

struct ParsedTail {
  std::string text;
  std::string reasoning;
  std::vector<ServerToolCall> calls;
};

// Incremental display/reasoning splitter for SSE streaming. One feed() per
// sampled token keeps the per-turn cost linear; finish() flushes the tail
// and returns the parsed calls. Backends without a native tool grammar
// (e.g. Glimmer) implement the pass-through (display = raw pieces).
class StreamParser {
 public:
  virtual ~StreamParser() = default;
  virtual StepDeltas feed(const std::string& raw_piece) = 0;
  virtual ParsedTail finish() = 0;
};

// Everything load() needs: resolved arch + model id, optional config
// overlay, one (LFM25 GGUF) or more (Glimmer sharded safetensors / globs)
// weight paths, tokenizer path or directory, context override, and the
// prefix-cache placement flag (backends without a prefix cache ignore it).
struct LoadOptions {
  std::string arch = kArchLFM25;
  std::string model_id;
  std::string config_path;
  std::vector<std::string> weights;
  std::string tokenizer;
  int max_seq_len = 0;  // 0 => keep the config file / preset value
  bool prefix_cache_host = false;
};

// ---------------------------------------------------------------------------
// ChatBackend: one instance serves all requests (guarded by the server's
// single-flight mutex). Not thread-safe; set_turn_options() + the
// prefill/step/flush sequence run under that lock.
// ---------------------------------------------------------------------------
class ChatBackend {
 public:
  virtual ~ChatBackend() = default;

  // Registry name ("lfm2.5" | "glimmer").
  virtual const std::string& arch() const = 0;
  // Served model id when --model is absent (e.g. "lfm2_5", "glimmer-30b").
  virtual std::string default_model_id() const = 0;
  // Load config + weights + tokenizer. Returns false with err set on
  // failure (missing files, shape mismatch, arch/config mismatch).
  virtual bool load(const LoadOptions& opts, std::string& err) = 0;
  virtual const std::string& model_id() const = 0;
  virtual int max_seq_len() const = 0;
  // Seed stop id for parse_common (always included, like LFM25's cfg.eos).
  virtual int eos_id() const = 0;
  // Full default stop-id set seeded when the request carries no stop list
  // (superset of {eos_id()}; Glimmer adds its turn-end id).
  virtual std::vector<int> default_stop_ids() const = 0;
  // Default stop strings (turn terminator etc.).
  virtual std::vector<std::string> default_stop_strings() const = 0;
  // False when the arch has no tool grammar (Glimmer): handlers reject
  // tools/tool_calls for it instead of silently dropping them.
  virtual bool supports_tools() const = 0;

  // Single user turn for POST /v1/completions.
  virtual std::vector<ServerChatMessage> make_single_turn(
      const std::string& prompt) const = 0;
  // Render + encode. `tools_json` is the compact OpenAI tools array ("" =
  // none); `system_suffix` carries tool_choice/response_format directives.
  // add_generation_prompt=false encodes the cacheable history prefix (same
  // encoding without the trailing generation prompt).
  virtual std::vector<int> encode(const std::vector<ServerChatMessage>& messages,
                                  const std::string& tools_json,
                                  const std::string& system_suffix,
                                  bool add_generation_prompt) const = 0;

  virtual void set_turn_options(const ServerSessionOptions& opts) = 0;
  // Throws std::invalid_argument on over-context prompts (mapped to 400).
  virtual CachedPrefill prefill_cached(const std::vector<int>& full,
                                       const std::vector<int>& content,
                                       uint64_t seed, bool hasSeed) = 0;
  virtual BackendStepOutcome step() = 0;
  virtual std::string flush() = 0;
  virtual ::KokkosDNN::Common::Usage usage() const = 0;
  virtual std::string finish() const = 0;

  // Decode one id with specials visible for the stream parser. Never
  // throws (undecodable ids contribute "").
  virtual std::string decode_piece(int id) const = 0;
  virtual std::unique_ptr<StreamParser> new_stream_parser(
      const std::vector<std::string>& stops,
      bool start_in_reasoning) const = 0;
};

namespace Impl {
// Registry storage. Inline with a function-local static so Backend.hpp
// stays header-only (one map program-wide, no extra TU needed).
using BackendFactory = std::function<std::unique_ptr<ChatBackend>()>;
inline std::map<std::string, BackendFactory>& backend_registry() {
  static std::map<std::string, BackendFactory> r;
  return r;
}
}  // namespace Impl

// Register a backend under `name` (called once per backend TU via a static
// registrar). Later registrations for the same name win.
inline void register_backend(const std::string& name,
                             Impl::BackendFactory factory) {
  Impl::backend_registry()[name] = std::move(factory);
}

inline std::vector<std::string> available_backends() {
  std::vector<std::string> out;
  for (const auto& kv : Impl::backend_registry()) out.push_back(kv.first);
  return out;
}

// Null when `name` names no registered backend.
inline std::unique_ptr<ChatBackend> create_backend(const std::string& name) {
  auto& r = Impl::backend_registry();
  auto it = r.find(name);
  if (it == r.end()) return nullptr;
  return it->second();
}

// ---------------------------------------------------------------------------
// Architecture name helpers (CLI + config sniffing)
// ---------------------------------------------------------------------------

// Canonicalize --arch (case-insensitive; driver-style aliases). Returns ""
// for unknown names. "lfm2.5" aliases: lfm2_5, lfm25. "glimmer" aliases:
// glimmer-30b, muse-glimmer[-30b][-heretic], optional darkc0de- prefix.
inline std::string normalize_arch_name(std::string name) {
  for (auto& ch : name)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (name == kArchLFM25 || name == "lfm2_5" || name == "lfm25")
    return kArchLFM25;
  const std::string dark = "darkc0de-";
  if (name.rfind(dark, 0) == 0) name = name.substr(dark.size());
  const std::string muse = "muse-";
  if (name.rfind(muse, 0) == 0) name = name.substr(muse.size());
  if (name == kArchGlimmer || name == "glimmer-30b" || name == "glimmer30b" ||
      name == "glimmer-30b-heretic")
    return kArchGlimmer;
  return "";
}

// Best-effort arch sniff of a model config file. Returns "lfm2.5",
// "glimmer", or "" when the file carries no discriminating marker (e.g. a
// minimal {"max_seq_len": ...} overlay) or cannot be read. Markers:
//   glimmer: "text_config" (full HF config), "layer_rope_theta",
//            "qk_scale_factor", "muse_glimmer".
//   lfm2.5:  "lfm2_moe" (model_type), "num_experts", "hidden_act".
inline std::string detect_config_arch(const std::string& path) {
  if (path.empty()) return "";
  std::string text;
  try {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    text = ss.str();
  } catch (...) {
    return "";
  }
  const bool glimmer = text.find("\"text_config\"") != std::string::npos ||
                       text.find("\"layer_rope_theta\"") != std::string::npos ||
                       text.find("\"qk_scale_factor\"") != std::string::npos ||
                       text.find("muse_glimmer") != std::string::npos;
  const bool lfm25 = text.find("lfm2_moe") != std::string::npos ||
                     text.find("\"num_experts\"") != std::string::npos ||
                     text.find("\"hidden_act\"") != std::string::npos;
  if (glimmer && !lfm25) return kArchGlimmer;
  if (lfm25 && !glimmer) return kArchLFM25;
  return "";
}

}  // namespace Server
}  // namespace KokkosDNN

#endif
