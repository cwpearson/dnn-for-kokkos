// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// server/BackendInkling.cpp: ChatBackend for the Thinking Machines
// Inkling 2 Mini text path (vision out of scope).
//
// Weight formats: sharded safetensors (glob patterns accepted, e.g.
// "model-*.safetensors") or a single llama.cpp .gguf file — mirroring the
// glimmer_generate driver and BackendGlimmer.cpp.
//
// Chat template (upstream thinkingmachines/Inkling-2-Mini chat_template.jinja,
// ground truth): role tokens user-><|message_user|>, assistant-><|message_model|>,
// system-><|message_system|>, tool-><|message_tool|> (tool messages append the
// tool name, resolved via tool_call_id against prior assistant tool_calls).
// Per-message bodies use <|content_text|>, closed by <|end_message|>.
// Assistant reasoning_content replays as
// '<|message_model|><|content_thinking|>...<|end_message|>' (response-template
// reasoning open_pattern '(?:<|message_model|>...)?<|content_thinking|>',
// close '<|end_message|>'). A thinking-effort system message
// ('<|message_system|><|content_text|>Thinking effort level: N<|end_message|>')
// is emitted before the first non-system message; reasoning_effort maps
// none/minimal/low/medium/high/xhigh/max -> {0.0,0.1,0.2,0.7,0.9,0.99,0.99}
// (default 0.9); unknown values raise in the template -> the encode path must
// throw std::invalid_argument so the server maps it to 400 (mirrors Glimmer's
// reasoning_strength handling, but with this numeric rendering and value set).
// Tools are declared as
// '<|message_system|>tool_declare<|content_xml|>{json}<|end_message|>'
// (sorted-keys compact JSON specs); calls render as
// '<|message_model|>NAME<|content_invoke_tool_json|>{"name":...,"args":{sorted}}
// <|end_message|>' (args must be an object with sorted keys, NOT a JSON
// string — canonicalize upstream; response-template tool open_pattern
// '<|content_invoke_tool_json|>' with transform name={content.name}
// arguments={content.args}). Every assistant message appends
// '<|content_model_end_sampling|>' (id 200006 = generation-config eos_token_id
// and tokenizer_config eos_token); <|end_message|> (200010) is a per-message
// structural boundary, never a stop. Tokenizer ids of note: 200006
// content_model_end_sampling (eos), 200010 end_message, 200008
// content_thinking, 200049 content_invoke_tool_json, 200054 unused/image
// placeholder. Default stops: ids {200006}, strings
// ['<|content_model_end_sampling|>'].
// Generation prompt: none beyond the template end (assistant messages are
// wrapped in generation blocks); generation_config.json carries no prompt
// suffix (only do_sample/temperature/eos 200006/pad 199999), so the
// models/inkling encode must sniff whether the processor appends a trailing
// '<|message_model|>' and mirror it when add_generation_prompt=true.
// Vision is out of scope: image parts would render
// '<|content_image|><|unused_200054|>' per image, but non-text content parts
// are rejected with 400 in server_main.cpp parse_messages (arch-agnostic,
// same as the other backends) — no backend change needed.
//
// Session mechanics: TODO — no models/inkling component exists yet. All
// `Inkling::` names below (Config, Weights, Tokenizer, Session,
// SessionOptions, ISession, make_chat_messages, encode_chatml_with_tools,
// default_stop_strings, CachedPrefillOutcome, StepOutcome) are PLACEHOLDERS
// mirroring the models/glimmer API surface (Glimmer::Session behind
// Glimmer::ISession); adjust to the real component API when it lands. This
// file only translates the neutral server vocabulary to/from it, mirroring
// BackendGlimmer.cpp (long-lived session, device/host cache placement from
// --prefix-cache-host).
//
// Without KOKKOSDNN_SERVER_HAVE_INKLING this TU compiles to a stub backend
// whose load() fails with a clear message, so "inkling" still resolves in
// --arch listings and --help.
#include "Backend.hpp"

#ifdef KOKKOSDNN_SERVER_HAVE_INKLING

// TODO(models/inkling): these headers do not exist yet. Create
// models/inkling with KokkosDNN_Inkling{Config,Session,Tokenizer,Weights}.hpp
// exposing the placeholder API used below, then wire the
// KokkosDNN::inkling target + KOKKOSDNN_SERVER_HAVE_INKLING define in
// server/CMakeLists.txt (see the wiring patch in this file's report).
#include <KokkosDNN_InklingConfig.hpp>
#include <KokkosDNN_InklingSession.hpp>
#include <KokkosDNN_InklingTokenizer.hpp>
#include <KokkosDNN_InklingWeights.hpp>

#include <Kokkos_Core.hpp>

#include <KokkosDNN_Detokenizer.hpp>

#if defined(__unix__) || defined(__unix) || defined(__APPLE__) || \
    defined(__linux__)
#include <glob.h>
#define KOKKOSDNN_SERVER_INKLING_HAS_GLOB 1
#else
#define KOKKOSDNN_SERVER_INKLING_HAS_GLOB 0
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace KokkosDNN {
namespace Server {
namespace Impl {

namespace Inkling = ::KokkosDNN::Inkling;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = Inkling::DefaultScalar;

// Expand shell-style --weights patterns (e.g. "model-*.safetensors").
// Mirrors the glimmer_generate driver: patterns matching nothing are kept
// literally so the loader reports the missing file by name. Results are
// sorted and deduplicated.
std::vector<std::string> expand_inkling_weight_patterns(
    const std::vector<std::string>& pats) {
  std::vector<std::string> out;
#if KOKKOSDNN_SERVER_INKLING_HAS_GLOB
  for (const auto& p : pats) {
    glob_t g;
    const int rc = ::glob(p.c_str(), 0, nullptr, &g);
    if (rc != 0 || g.gl_pathc == 0) {
      out.push_back(p);
    } else {
      for (size_t i = 0; i < g.gl_pathc; ++i) out.emplace_back(g.gl_pathv[i]);
    }
    ::globfree(&g);
  }
#else
  out = pats;
#endif
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// A single .gguf file (case-insensitive suffix) selects the GGUF loading
// path; anything else is a safetensors shard.
bool is_inkling_gguf_path(const std::string& p) {
  if (p.size() < 5) return false;
  std::string ext = p.substr(p.size() - 5);
  for (auto& ch : ext) ch = static_cast<char>(std::tolower(
                              static_cast<unsigned char>(ch)));
  return ext == ".gguf";
}

// Validates the reasoning_effort cue and returns its rendered float string
// ("0".."0.99"). Mirrors Glimmer's normalize_reasoning_strength: unknown
// values throw std::invalid_argument, which the server maps to 400. The
// upstream effort map is {none:0.0, minimal:0.1, low:0.2, medium:0.7,
// high:0.9, xhigh:0.99, max:0.99}; empty/absent means the 0.9 default.
std::string normalize_inkling_reasoning_effort(const std::string& v) {
  std::string k = v;
  for (auto& ch : k)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  // Numeric pass-through: the template accepts a float in [0.0, 0.99].
  if (!k.empty()) {
    try {
      const double f = std::stod(k);
      if (f >= 0.0 && f <= 0.99) return k;
    } catch (...) {
    }
  }
  if (k.empty() || k == "high") return "0.9";
  if (k == "none") return "0";
  if (k == "minimal") return "0.1";
  if (k == "low") return "0.2";
  if (k == "medium") return "0.7";
  if (k == "xhigh" || k == "max") return "0.99";
  throw std::invalid_argument("unknown reasoning_effort '" + v +
                              "' (expected none, minimal, low, medium, high,"
                              " xhigh, max, or a float in [0.0, 0.99])");
}

// Pass-through channel splitter: no tool grammar is wired yet (the upstream
// <|content_invoke_tool_json|> invoke blocks are NOT parsed here), so every
// raw piece streams as display text and finish() returns it verbatim with
// no reasoning trace and no tool calls. TODO: replace with a real
// <|content_thinking|> / <|content_invoke_tool_json|> channel parser (plus
// supports_tools()=true) once the invoke-JSON grammar is confirmed against
// the real tokenizer/component.
class InklingStreamParser final : public StreamParser {
 public:
  explicit InklingStreamParser(const std::vector<std::string>& /*stops*/) {}
  StepDeltas feed(const std::string& raw_piece) override {
    acc_ += raw_piece;
    return {raw_piece, ""};
  }
  ParsedTail finish() override {
    ParsedTail out;
    out.text = std::move(acc_);
    out.reasoning = "";
    return out;
  }

 private:
  std::string acc_;
};

class InklingBackend final : public ChatBackend {
 public:
  const std::string& arch() const override {
    // TODO: return kArchInkling once Backend.hpp declares it (wiring patch
    // below). Literal keeps this TU compilable without that header edit.
    static const std::string a = "inkling";
    return a;
  }
  std::string default_model_id() const override { return "inkling-2-mini"; }

  bool load(const LoadOptions& opts, std::string& err) override {
    if (opts.weights.empty()) {
      err =
          "missing required --weights PATH (one .gguf file or one or more "
          "safetensors shards/globs)";
      return false;
    }
    if (opts.tokenizer.empty()) {
      err = "missing required --tokenizer PATH (tokenizer.json)";
      return false;
    }
    try {
      model_id_ = opts.model_id;
      // TODO(models/inkling): confirm the preset constructor name and the
      // config-file schema (expect max_seq_len up to 1048576 per
      // tokenizer_config.json model_max_length).
      cfg_ = opts.config_path.empty() ? Inkling::Config::inkling_2_mini()
                                      : Inkling::Config::from_json_file(
                                            opts.config_path);
      if (opts.max_seq_len > 0) cfg_.max_seq_len = opts.max_seq_len;
      if (opts.prefill_chunk_override >= 0)
        cfg_.prefill_chunk_size = opts.prefill_chunk_override;
      if (opts.kv_quant_override >= 0)
        cfg_.kv_int8 = (opts.kv_quant_override != 0);
      cfg_.validate();
      {
        const double mib =
            static_cast<double>(cfg_.kv_cache_bytes(sizeof(Scalar))) /
            (1024.0 * 1024.0);
        std::fprintf(stderr,
                     "  inkling kv cache: %s, full-window %.1f MiB "
                     "(max_seq_len=%d, prefill_chunk=%d)\n",
                     cfg_.kv_int8 ? "int8" : "dense", mib, cfg_.max_seq_len,
                     cfg_.prefill_chunk_size);
      }
      weights_ = Inkling::Weights<Exec, Scalar>{};
      weights_.allocate(cfg_);
      const std::vector<std::string> files =
          expand_inkling_weight_patterns(opts.weights);
      const auto t_load = std::chrono::steady_clock::now();
      if (files.size() == 1 && is_inkling_gguf_path(files[0])) {
        Inkling::load_gguf(cfg_, files[0], weights_);
      } else {
        for (const auto& fp : files)
          if (is_inkling_gguf_path(fp))
            throw std::runtime_error(
                "GGUF loading takes a single .gguf file: got '" + fp +
                "' among " + std::to_string(files.size()) +
                " --weights entries (mixing .gguf with safetensors shards is "
                "unsupported)");
        Inkling::load_safetensors(cfg_, files, weights_);
      }
      const auto t_tok = std::chrono::steady_clock::now();
      tok_ = Inkling::Tokenizer(opts.tokenizer);
      const auto t_end = std::chrono::steady_clock::now();
      if (std::getenv("KOKKOSDNN_LOAD_TIMING") != nullptr) {
        const auto secs = [](auto a, auto b) {
          return std::chrono::duration<double>(b - a).count();
        };
        std::fprintf(stderr, "[load] %-24s %8.3f s\n", "weights (backend)",
                     secs(t_load, t_tok));
        std::fprintf(stderr, "[load] %-24s %8.3f s\n", "tokenizer.json",
                     secs(t_tok, t_end));
      }
      prefix_cache_host_ = opts.prefix_cache_host;
      sess_.reset();
    } catch (const std::exception& e) {
      err = std::string("failed to load model: ") + e.what();
      return false;
    }
    return true;
  }

  const std::string& model_id() const override { return model_id_; }
  int max_seq_len() const override { return cfg_.max_seq_len; }
  // Upstream generation_config.json eos_token_id:
  // <|content_model_end_sampling|> == 200006.
  int eos_id() const override { return cfg_.eos_token; }
  std::vector<int> default_stop_ids() const override {
    // The turn terminator is the eos id alone; <|end_message|> (200010) is a
    // per-message structural boundary (like Glimmer's <|eom|>), never a stop.
    std::vector<int> s{cfg_.eos_token};
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
  }
  std::vector<std::string> default_stop_strings() const override {
    // TODO(models/inkling): confirm against the real component. Terminator
    // text as a belt-and-braces backup for the eos id, mirroring Glimmer's
    // {"<|eot|>"}; <|end_message|> must NOT be added (structural).
    return Inkling::default_stop_strings();
  }
  // TODO: flip to true with the invoke-JSON tool grammar (template declares
  // tool_declare + content_invoke_tool_json blocks; the stream parser above
  // is still pass-through). While false, handlers reject tools/tool_calls
  // for this arch instead of silently dropping them.
  bool supports_tools() const override { return false; }

  std::vector<ServerChatMessage> make_single_turn(
      const std::string& prompt) const override {
    std::vector<ServerChatMessage> out;
    for (const auto& m : Inkling::make_chat_messages(prompt, "", "user")) {
      ServerChatMessage n;
      n.role = m.role;
      n.content = m.content;
      out.push_back(std::move(n));
    }
    return out;
  }

  std::vector<int> encode(const std::vector<ServerChatMessage>& messages,
                          const std::string& tools_json,
                          const std::string& system_suffix,
                          bool add_generation_prompt,
                          const std::string& reasoning_strength = "high") const override {
    // Validate the effort cue here so unknown values are a 400 even before
    // the models/inkling component lands (mirrors Glimmer's
    // reasoning_strength handling; the component's own encode must also
    // raise on unknown values per the template's raise_exception).
    (void)normalize_inkling_reasoning_effort(reasoning_strength);
    // TODO(models/inkling): confirm the encode entry point. Upstream effort
    // keys (none/minimal/low/medium/high/xhigh/max) already cover the
    // server's reasoning_strength vocabulary (default "high"), so the value
    // forwards verbatim like Glimmer's RLed cue; the component renders the
    // numeric 'Thinking effort level: N' message and the tool_declare /
    // content_invoke_tool_json blocks per the contract above, and must sniff
    // whether the processor appends a trailing '<|message_model|>' for
    // add_generation_prompt=true.
    return Inkling::encode_chatml_with_tools(tok_, to_native(messages),
                                             tools_json, system_suffix,
                                             add_generation_prompt, true,
                                             reasoning_strength);
  }

  void set_turn_options(const ServerSessionOptions& opts) override {
    Inkling::SessionOptions n;
    n.sampling = opts.sampling;
    n.stop_ids = opts.stop_ids;
    n.stop_strings = opts.stop_strings;
    n.max_new = opts.max_new;
    n.seed = opts.seed;
    n.seedProvided = opts.seedProvided;
    pending_opts_ = std::move(n);
    has_pending_opts_ = true;
    if (sess_) sess_->set_options(pending_opts_);
  }

  CachedPrefill prefill_cached(const std::vector<int>& full,
                               const std::vector<int>& content, uint64_t seed,
                               bool hasSeed) override {
    ensure_session();
    const Inkling::CachedPrefillOutcome o =
        sess_->prefill_cached(full, content, seed, hasSeed);
    return {o.hit, o.reused};
  }

  BackendStepOutcome step() override {
    const Inkling::StepOutcome s = sess_->step();
    BackendStepOutcome out;
    out.id = s.id;
    out.finished = s.finished;
    out.finish = ::KokkosDNN::Common::to_string(s.finish);
    return out;
  }
  std::string flush() override { return sess_->flush(); }
  ::KokkosDNN::Common::Usage usage() const override { return sess_->usage(); }
  std::string finish() const override {
    return ::KokkosDNN::Common::to_string(sess_->finish());
  }

  std::string decode_piece(int id) const override {
    try {
      thread_local std::vector<int> idbuf{0};
      idbuf[0] = id;
      return tok_.decode(idbuf, false);
    } catch (...) {
      return "";
    }
  }

  std::unique_ptr<StreamParser> new_stream_parser(
      const std::vector<std::string>& stops,
      bool /*start_in_reasoning*/) const override {
    // No prompt-provided open thinker to mirror: the pass-through parser
    // streams raw pieces as display until the real channel parser lands.
    return std::make_unique<InklingStreamParser>(stops);
  }

 private:
  static Inkling::ChatMessage to_native_msg(const ServerChatMessage& m) {
    Inkling::ChatMessage n;
    n.role = m.role;
    n.content = m.content;
    n.thinking = m.thinking;
    n.tool_calls.reserve(m.tool_calls.size());
    for (const auto& c : m.tool_calls) {
      Inkling::ToolCall t;
      t.id = c.id;
      t.name = c.name;
      t.arguments = c.arguments;
      n.tool_calls.push_back(std::move(t));
    }
    n.tool_call_id = m.tool_call_id;
    n.name = m.name;
    return n;
  }

  static std::vector<Inkling::ChatMessage> to_native(
      const std::vector<ServerChatMessage>& v) {
    std::vector<Inkling::ChatMessage> out;
    out.reserve(v.size());
    for (const auto& m : v) out.push_back(to_native_msg(m));
    return out;
  }

  // Reuse one long-lived Session: avoids re-renting the KV cache and
  // re-copying the tokenizer on every request. set_turn_options() +
  // prefill_cached() reset all turn state; the reseed below reproduces
  // fresh-session RNG behavior exactly. The cache entry placement (device
  // vs host memory) is fixed at load from --prefix-cache-host.
  void ensure_session() {
    if (sess_) {
      if (has_pending_opts_) {
        sess_->set_options(pending_opts_);
        has_pending_opts_ = false;
      }
      return;
    }
    const Inkling::SessionOptions o =
        has_pending_opts_ ? pending_opts_ : Inkling::SessionOptions{};
    has_pending_opts_ = false;
    if (prefix_cache_host_)
      sess_ =
          std::make_unique<Inkling::Session<Exec, Scalar, Kokkos::HostSpace>>(
              cfg_, weights_, tok_, o);
    else
      sess_ = std::make_unique<Inkling::Session<Exec, Scalar>>(cfg_, weights_,
                                                               tok_, o);
  }

  std::string model_id_;
  Inkling::Config cfg_;
  Inkling::Weights<Exec, Scalar> weights_;
  Inkling::Tokenizer tok_;
  bool prefix_cache_host_ = false;
  std::unique_ptr<Inkling::ISession> sess_;
  Inkling::SessionOptions pending_opts_;
  bool has_pending_opts_ = false;
};

namespace {
struct InklingRegistrar {
  InklingRegistrar() {
    // TODO: register under kArchInkling once Backend.hpp declares it.
    register_backend("inkling",
                     []() { return std::make_unique<InklingBackend>(); });
  }
};
InklingRegistrar g_inkling_registrar;
}  // namespace

}  // namespace Impl
}  // namespace Server
}  // namespace KokkosDNN

#else  // !KOKKOSDNN_SERVER_HAVE_INKLING

namespace KokkosDNN {
namespace Server {
namespace Impl {

class InklingUnavailable final : public ChatBackend {
 public:
  const std::string& arch() const override {
    // TODO: return kArchInkling once Backend.hpp declares it.
    static const std::string a = "inkling";
    return a;
  }
  std::string default_model_id() const override { return "inkling-2-mini"; }
  bool load(const LoadOptions&, std::string& err) override {
    err =
        "arch 'inkling' was disabled at configure time "
        "(rebuild with -DKokkosDNN_ENABLE_MODEL_INKLING=ON once the "
        "models/inkling component exists)";
    return false;
  }
  const std::string& model_id() const override {
    static const std::string m;
    return m;
  }
  int max_seq_len() const override { return 0; }
  // Well-known upstream ids (verified against tokenizer_config.json /
  // generation_config.json): eos <|content_model_end_sampling|> 200006, so
  // pre-load metadata stays sane; load() still fails without the component.
  int eos_id() const override { return 200006; }
  std::vector<int> default_stop_ids() const override { return {200006}; }
  std::vector<std::string> default_stop_strings() const override {
    return {"<|content_model_end_sampling|>"};
  }
  bool supports_tools() const override { return false; }
  std::vector<ServerChatMessage> make_single_turn(
      const std::string&) const override {
    return {};
  }
  std::vector<int> encode(const std::vector<ServerChatMessage>&,
                          const std::string&, const std::string&, bool,
                          const std::string& = "high") const override {
    return {};
  }
  void set_turn_options(const ServerSessionOptions&) override {}
  CachedPrefill prefill_cached(const std::vector<int>&, const std::vector<int>&,
                               uint64_t, bool) override {
    return {};
  }
  BackendStepOutcome step() override { return {}; }
  std::string flush() override { return ""; }
  ::KokkosDNN::Common::Usage usage() const override { return {}; }
  std::string finish() const override { return "stop"; }
  std::string decode_piece(int) const override { return ""; }
  std::unique_ptr<StreamParser> new_stream_parser(
      const std::vector<std::string>&, bool) const override {
    return nullptr;
  }
};

namespace {
struct InklingRegistrar {
  InklingRegistrar() {
    // TODO: register under kArchInkling once Backend.hpp declares it.
    register_backend("inkling",
                     []() { return std::make_unique<InklingUnavailable>(); });
  }
};
InklingRegistrar g_inkling_registrar;
}  // namespace

}  // namespace Impl
}  // namespace Server
}  // namespace KokkosDNN

#endif
