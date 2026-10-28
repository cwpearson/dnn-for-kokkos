// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// server/BackendGlimmer.cpp: ChatBackend for the Muse Glimmer 30B text path.
//
// Weight formats: sharded safetensors (glob patterns accepted, e.g.
// "model-*.safetensors") or a single llama.cpp .gguf file — mirroring the
// glimmer_generate driver. Chat template: the Muse-style framing from
// KokkosDNN_GlimmerTokenizer.hpp (<|begin_of_text|> + <|start|>role
// <|message|> blocks + open assistant header; an ASSUMPTION documented
// there — no upstream chat_template ships in tokenizer_config.json).
// Glimmer has no native tool grammar: supports_tools() is false and the
// handlers reject tools/tool_calls for it; the stream parser is a
// stop-cutting pass-through (Common::IncrementalDetokenizer).
//
// Session mechanics: all inference, sampling, and chat logic runs through
// models/glimmer (Glimmer::Session behind Glimmer::ISession); this file
// only translates the neutral server vocabulary to/from it, mirroring
// BackendLFM25.cpp (long-lived session, device/host cache placement from
// --prefix-cache-host). Caveat: the promoted Session's empty-stop policy
// is {cfg.eos} + default_stop_strings() ("<|eot|>", "<|eom|>"); the server
// always seeds stop_ids from default_stop_ids() ({eos, eot}), and an empty
// request stop list resolves stop strings to default_stop_strings(), so
// the session and the stream parser agree.
//
// Without KOKKOSDNN_SERVER_HAVE_GLIMMER this TU compiles to a stub backend
// whose load() fails with a clear message, so "glimmer" still resolves in
// --arch listings and --help.
#include "Backend.hpp"

#ifdef KOKKOSDNN_SERVER_HAVE_GLIMMER

#include <KokkosDNN_GlimmerConfig.hpp>
#include <KokkosDNN_GlimmerSession.hpp>
#include <KokkosDNN_GlimmerTokenizer.hpp>
#include <KokkosDNN_GlimmerWeights.hpp>

#include <Kokkos_Core.hpp>

#include <KokkosDNN_Detokenizer.hpp>

#if defined(__unix__) || defined(__unix) || defined(__APPLE__) || \
    defined(__linux__)
#include <glob.h>
#define KOKKOSDNN_SERVER_GLIMMER_HAS_GLOB 1
#else
#define KOKKOSDNN_SERVER_GLIMMER_HAS_GLOB 0
#endif

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace KokkosDNN {
namespace Server {
namespace Impl {

namespace Glimmer = ::KokkosDNN::Glimmer;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = Glimmer::DefaultScalar;

// Expand shell-style --weights patterns (e.g. "model-*.safetensors").
// Mirrors the glimmer_generate driver: patterns matching nothing are kept
// literally so the loader reports the missing file by name. Results are
// sorted and deduplicated.
std::vector<std::string> expand_weight_patterns(
    const std::vector<std::string>& pats) {
  std::vector<std::string> out;
#if KOKKOSDNN_SERVER_GLIMMER_HAS_GLOB
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
bool is_gguf_path(const std::string& p) {
  if (p.size() < 5) return false;
  std::string ext = p.substr(p.size() - 5);
  for (auto& ch : ext) ch = static_cast<char>(std::tolower(
                             static_cast<unsigned char>(ch)));
  return ext == ".gguf";
}

// Stop-cutting pass-through parser: no native tool grammar, so display is
// the raw pieces cut at the request stops (the turn terminator is unioned
// in at construction). Backed by Common::IncrementalDetokenizer for
// stop/UTF-8 safety, like the LFM25 parser's stop layer.
class PassthroughParser final : public StreamParser {
 public:
  explicit PassthroughParser(const std::vector<std::string>& stops)
      : detok_(stops) {}
  StepDeltas feed(const std::string& raw_piece) override {
    const auto s = detok_.push(0, raw_piece);
    return {std::move(s.delta), ""};
  }
  ParsedTail finish() override {
    ParsedTail out;
    out.text = detok_.flush();
    return out;
  }

 private:
  ::KokkosDNN::Common::IncrementalDetokenizer detok_;
};

class GlimmerBackend final : public ChatBackend {
 public:
  const std::string& arch() const override {
    static const std::string a = kArchGlimmer;
    return a;
  }
  std::string default_model_id() const override { return "glimmer-30b"; }

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
      cfg_ = opts.config_path.empty() ? Glimmer::Config::glimmer_30b()
                                      : Glimmer::Config::from_json_file(
                                            opts.config_path);
      if (opts.max_seq_len > 0) cfg_.max_seq_len = opts.max_seq_len;
      cfg_.validate();
      weights_ = Glimmer::Weights<Exec, Scalar>{};
      weights_.allocate(cfg_);
      const std::vector<std::string> files =
          expand_weight_patterns(opts.weights);
      if (files.size() == 1 && is_gguf_path(files[0])) {
        Glimmer::load_gguf(cfg_, files[0], weights_);
      } else {
        for (const auto& fp : files)
          if (is_gguf_path(fp))
            throw std::runtime_error(
                "GGUF loading takes a single .gguf file: got '" + fp +
                "' among " + std::to_string(files.size()) +
                " --weights entries (mixing .gguf with safetensors shards is "
                "unsupported)");
        Glimmer::load_safetensors(cfg_, files, weights_);
      }
      tok_ = Glimmer::Tokenizer(opts.tokenizer);
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
  int eos_id() const override { return cfg_.eos_token; }
  std::vector<int> default_stop_ids() const override {
    std::vector<int> s{cfg_.eos_token, tok_.eot()};
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
  }
  std::vector<std::string> default_stop_strings() const override {
    return Glimmer::default_stop_strings();
  }
  bool supports_tools() const override { return false; }

  std::vector<ServerChatMessage> make_single_turn(
      const std::string& prompt) const override {
    std::vector<ServerChatMessage> out;
    for (const auto& m : Glimmer::make_chat_messages(prompt, "", "user")) {
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
                          bool add_generation_prompt) const override {
    if (!tools_json.empty())
      throw std::invalid_argument(
          "'tools' is not supported for arch 'glimmer' (no native tool "
          "grammar)");
    std::vector<Glimmer::ChatMessage> native;
    native.reserve(messages.size());
    for (const auto& m : messages) {
      if (!m.tool_calls.empty() || m.role == "tool")
        throw std::invalid_argument(
            "'tool_calls'/'tool' history turns are not supported for arch "
            "'glimmer' (no native tool grammar)");
      Glimmer::ChatMessage n;
      n.role = m.role;
      n.content = m.content;  // thinking traces have no Glimmer rendering
      native.push_back(std::move(n));
    }
    if (!system_suffix.empty()) {
      bool found = false;
      for (auto& n : native) {
        if (n.role == "system") {
          n.content += "\n" + system_suffix;
          found = true;
          break;
        }
      }
      if (!found) {
        Glimmer::ChatMessage sys;
        sys.role = "system";
        sys.content = system_suffix;
        native.insert(native.begin(), std::move(sys));
      }
    }
    return Glimmer::encode_chat(tok_, native, add_generation_prompt, true);
  }

  void set_turn_options(const ServerSessionOptions& opts) override {
    Glimmer::SessionOptions n;
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
    const Glimmer::CachedPrefillOutcome o =
        sess_->prefill_cached(full, content, seed, hasSeed);
    return {o.hit, o.reused};
  }

  BackendStepOutcome step() override {
    const Glimmer::StepOutcome s = sess_->step();
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
    // Glimmer has no thinking trace: every byte is display.
    return std::make_unique<PassthroughParser>(stops);
  }

 private:
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
    const Glimmer::SessionOptions o =
        has_pending_opts_ ? pending_opts_ : Glimmer::SessionOptions{};
    has_pending_opts_ = false;
    if (prefix_cache_host_)
      sess_ =
          std::make_unique<Glimmer::Session<Exec, Scalar, Kokkos::HostSpace>>(
              cfg_, weights_, tok_, o);
    else
      sess_ = std::make_unique<Glimmer::Session<Exec, Scalar>>(cfg_, weights_,
                                                               tok_, o);
  }

  std::string model_id_;
  Glimmer::Config cfg_;
  Glimmer::Weights<Exec, Scalar> weights_;
  Glimmer::Tokenizer tok_;
  bool prefix_cache_host_ = false;
  std::unique_ptr<Glimmer::ISession> sess_;
  Glimmer::SessionOptions pending_opts_;
  bool has_pending_opts_ = false;
};

namespace {
struct GlimmerRegistrar {
  GlimmerRegistrar() {
    register_backend(kArchGlimmer,
                     []() { return std::make_unique<GlimmerBackend>(); });
  }
};
GlimmerRegistrar g_glimmer_registrar;
}  // namespace

}  // namespace Impl
}  // namespace Server
}  // namespace KokkosDNN

#else  // !KOKKOSDNN_SERVER_HAVE_GLIMMER

namespace KokkosDNN {
namespace Server {
namespace Impl {

class GlimmerUnavailable final : public ChatBackend {
 public:
  const std::string& arch() const override {
    static const std::string a = kArchGlimmer;
    return a;
  }
  std::string default_model_id() const override { return "glimmer-30b"; }
  bool load(const LoadOptions&, std::string& err) override {
    err =
        "arch 'glimmer' was disabled at configure time "
        "(rebuild with -DKokkosDNN_ENABLE_MODEL_GLIMMER=ON)";
    return false;
  }
  const std::string& model_id() const override {
    static const std::string m;
    return m;
  }
  int max_seq_len() const override { return 0; }
  // Well-known Glimmer ids (stable upstream: bos 200000, eos 200001,
  // turn-end <|eot|> 200008) so pre-load metadata stays sane; load() still
  // fails without the Glimmer component.
  int eos_id() const override { return 200001; }
  std::vector<int> default_stop_ids() const override {
    return {200001, 200008};
  }
  std::vector<std::string> default_stop_strings() const override {
    return {"<|eot|>"};
  }
  bool supports_tools() const override { return false; }
  std::vector<ServerChatMessage> make_single_turn(
      const std::string&) const override {
    return {};
  }
  std::vector<int> encode(const std::vector<ServerChatMessage>&,
                          const std::string&, const std::string&,
                          bool) const override {
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
struct GlimmerRegistrar {
  GlimmerRegistrar() {
    register_backend(kArchGlimmer,
                     []() { return std::make_unique<GlimmerUnavailable>(); });
  }
};
GlimmerRegistrar g_glimmer_registrar;
}  // namespace

}  // namespace Impl
}  // namespace Server
}  // namespace KokkosDNN

#endif
