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
// Session mechanics: examples/glimmer/ ships Model::generate() (one-shot)
// but no multi-turn Session, so this TU carries a minimal session wrapper
// (Impl::GlimmerSession, same prefill/step/flush/usage shape as
// LFM25::Session, sampling via the core DeviceSampler). TODO: promote it
// to models/glimmer as KokkosDNN::Glimmer::Session/ISession together with
// the models/glimmer promotion (see GlimmerShim.hpp) and switch this
// backend to that type; the wrapper's behavior contract is the LFM25 one.
//
// Without KOKKOSDNN_SERVER_HAVE_GLIMMER this TU compiles to a stub backend
// whose load() fails with a clear message, so "glimmer" still resolves in
// --arch listings and --help.
#include "Backend.hpp"

#ifdef KOKKOSDNN_SERVER_HAVE_GLIMMER

#include "GlimmerShim.hpp"

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

// Minimal multi-turn session over Glimmer::Model. Lifecycle mirrors
// LFM25::Session: set_turn_options() replaces the per-request policy,
// prefill(ids[, seed, reseed]) resets + forwards the prompt, step()
// samples one token per call, flush() releases the detokenizer tail.
// Stop policy: empty stop_ids selects {cfg.eos_token, tok.eot()}; empty
// stop_strings selects the backend-provided defaults. Not thread-safe.
template <class ExecSpace, class GlimmerScalar = Glimmer::DefaultScalar>
class GlimmerSession {
 public:
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using HostInt = typename ViewInt::host_mirror_type;
  using View1 = Kokkos::View<GlimmerScalar*, ExecSpace>;

  GlimmerSession(const Glimmer::Config& cfg,
                 const Glimmer::Weights<ExecSpace, GlimmerScalar>& w,
                 Glimmer::Tokenizer tok, ServerSessionOptions opts,
                 std::vector<std::string> default_stops)
      : model_(cfg, w),
        tok_(std::move(tok)),
        opts_(std::move(opts)),
        default_stops_(std::move(default_stops)),
        rng_(opts_.seed),
        stop_ids_(resolve_ids(opts_, model_.cfg, tok_)),
        detok_(resolve_strings(opts_, default_stops_)),
        one_("KokkosDNN::Server::GlimmerSession::one", 1) {}

  void set_options(ServerSessionOptions opts) {
    opts_ = std::move(opts);
    stop_ids_ = resolve_ids(opts_, model_.cfg, tok_);
    detok_ = ::KokkosDNN::Common::IncrementalDetokenizer(
        resolve_strings(opts_, default_stops_));
  }

  void prefill(const std::vector<int>& ids, uint64_t seed, bool hasSeed) {
    if (ids.empty())
      throw std::invalid_argument("GlimmerSession::prefill: empty prompt");
    if (hasSeed) rng_.seed(seed);
    model_.reset();
    forward_full(ids);
    has_pending_ = true;
    prev_ = -1;
    ids_ = ids;
    usage_ = ::KokkosDNN::Common::Usage{};
    usage_.prompt_tokens = ids.size();
    detok_.reset();
    finished_ = false;
    finish_ = ::KokkosDNN::Common::FinishReason::Stop;
    last_ = BackendStepOutcome{};
  }

  // Glimmer has no KV prefix cache (dense per-layer caches are reset and
  // re-prefilled every turn); report a miss so the per-turn log is honest.
  CachedPrefill prefill_cached(const std::vector<int>& full,
                               const std::vector<int>& /*content*/,
                               uint64_t seed, bool hasSeed) {
    prefill(full, seed, hasSeed);
    return {false, 0};
  }

  BackendStepOutcome step() {
    if (finished_) return last_;
    BackendStepOutcome out;
    if (static_cast<int>(model_.seq_len) >= model_.cfg.max_seq_len) {
      out.finished = true;
      out.finish =
          ::KokkosDNN::Common::to_string(::KokkosDNN::Common::FinishReason::Length);
      finished_ = true;
      finish_ = ::KokkosDNN::Common::FinishReason::Length;
      last_ = out;
      return out;
    }
    View1 lg;
    if (has_pending_) {
      lg = pend_;
      has_pending_ = false;
    } else {
      auto one = one_;
      const int tok = prev_;
      Kokkos::parallel_for(
          "KokkosDNN::Server::GlimmerSession::fill_one",
          Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int i) {
            (void)i;
            one(0) = tok;
          });
      lg = model_.forward(one_);
    }
    const ::KokkosDNN::SampleResult r =
        sampler_.sample(lg, opts_.sampling, ids_, rng_);
    ids_.push_back(r.id);
    prev_ = r.id;
    usage_.completion_tokens++;
    out.id = r.id;
    std::string piece;
    try {
      idbuf_[0] = r.id;  // persistent 1-elem staging: no vector alloc/token
      piece = tok_.decode(idbuf_);
    } catch (...) {
      piece.clear();  // undecodable id: stream "" rather than throw
    }
    const ::KokkosDNN::Common::IncrementalDetokenizer::Step ds =
        detok_.push(r.id, piece);
    const bool id_stop = std::binary_search(stop_ids_.begin(), stop_ids_.end(),
                                            r.id);
    if (id_stop || ds.stop_hit) {
      out.finished = true;
      out.finish =
          ::KokkosDNN::Common::to_string(::KokkosDNN::Common::FinishReason::Stop);
    } else if (static_cast<int>(usage_.completion_tokens) >= opts_.max_new) {
      out.finished = true;
      out.finish =
          ::KokkosDNN::Common::to_string(::KokkosDNN::Common::FinishReason::Length);
    }
    if (out.finished) {
      finished_ = true;
      finish_ = out.finish == ::KokkosDNN::Common::to_string(
                                  ::KokkosDNN::Common::FinishReason::Length)
                    ? ::KokkosDNN::Common::FinishReason::Length
                    : ::KokkosDNN::Common::FinishReason::Stop;
    }
    last_ = out;
    return out;
  }

  std::string flush() { return detok_.flush(); }

  ::KokkosDNN::Common::Usage usage() const { return usage_; }
  std::string finish() const {
    return ::KokkosDNN::Common::to_string(finish_);
  }
  bool finished() const { return finished_; }

 private:
  static std::vector<int> resolve_ids(const ServerSessionOptions& o,
                                      const Glimmer::Config& c,
                                      const Glimmer::Tokenizer& tok) {
    if (o.stop_ids.empty()) {
      std::vector<int> s{c.eos_token, tok.eot()};
      std::sort(s.begin(), s.end());
      s.erase(std::unique(s.begin(), s.end()), s.end());
      return s;
    }
    std::vector<int> s = o.stop_ids;
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
  }

  static std::vector<std::string> resolve_strings(
      const ServerSessionOptions& o, const std::vector<std::string>& def) {
    if (o.stop_strings.empty()) return def;
    return o.stop_strings;
  }

  void forward_full(const std::vector<int>& ids) {
    ViewInt toks("KokkosDNN::Server::GlimmerSession::prefill", ids.size());
    HostInt h = Kokkos::create_mirror_view(toks);
    for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
    Kokkos::deep_copy(toks, h);
    pend_ = model_.forward(toks);
  }

  Glimmer::Model<ExecSpace, GlimmerScalar> model_;
  Glimmer::Tokenizer tok_;
  ServerSessionOptions opts_;
  std::vector<std::string> default_stops_;
  ::KokkosDNN::HostRNG rng_;
  std::vector<int> stop_ids_;
  ::KokkosDNN::Common::IncrementalDetokenizer detok_;
  ::KokkosDNN::Impl::DeviceSampler<ExecSpace> sampler_;
  ViewInt one_;
  View1 pend_;
  std::vector<int> ids_;
  std::vector<int> idbuf_{0};  // single-id decode staging (no per-token alloc)
  ::KokkosDNN::Common::Usage usage_;
  ::KokkosDNN::Common::FinishReason finish_ =
      ::KokkosDNN::Common::FinishReason::Stop;
  BackendStepOutcome last_;
  bool has_pending_ = false;
  bool finished_ = false;
  int prev_ = -1;
};

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
      // Default stop string: the decoded turn-end token when the tokenizer
      // file carries it, else the well-known literal.
      try {
        const std::vector<int> eot{tok_.eot()};
        const std::string lit = tok_.decode(eot, false);
        default_stops_ = lit.empty() ? std::vector<std::string>{"<|eot|>"}
                                     : std::vector<std::string>{lit};
      } catch (...) {
        default_stops_ = {"<|eot|>"};
      }
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
    return default_stops_.empty() ? std::vector<std::string>{"<|eot|>"}
                                  : default_stops_;
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
    pending_opts_ = opts;
    has_pending_opts_ = true;
    if (sess_) sess_->set_options(pending_opts_);
  }

  CachedPrefill prefill_cached(const std::vector<int>& full,
                               const std::vector<int>& content, uint64_t seed,
                               bool hasSeed) override {
    ensure_session();
    return sess_->prefill_cached(full, content, seed, hasSeed);
  }

  BackendStepOutcome step() override { return sess_->step(); }
  std::string flush() override { return sess_->flush(); }
  ::KokkosDNN::Common::Usage usage() const override { return sess_->usage(); }
  std::string finish() const override { return sess_->finish(); }

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
  void ensure_session() {
    if (sess_) {
      if (has_pending_opts_) {
        sess_->set_options(pending_opts_);
        has_pending_opts_ = false;
      }
      return;
    }
    const ServerSessionOptions o =
        has_pending_opts_ ? pending_opts_ : ServerSessionOptions{};
    has_pending_opts_ = false;
    sess_ = std::make_unique<GlimmerSession<Exec>>(
        cfg_, weights_, tok_, o,
        default_stops_.empty() ? std::vector<std::string>{"<|eot|>"}
                               : default_stops_);
  }

  std::string model_id_;
  Glimmer::Config cfg_;
  Glimmer::Weights<Exec, Scalar> weights_;
  Glimmer::Tokenizer tok_;
  std::vector<std::string> default_stops_{"<|eot|>"};
  std::unique_ptr<GlimmerSession<Exec>> sess_;
  ServerSessionOptions pending_opts_;
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
        "arch 'glimmer' needs the models/glimmer promotion "
        "(-DKokkosDNN_ENABLE_MODEL_GLIMMER=ON once the KokkosDNN::glimmer "
        "target exists; see server/GlimmerShim.hpp)";
    return false;
  }
  const std::string& model_id() const override {
    static const std::string m;
    return m;
  }
  int max_seq_len() const override { return 0; }
  // Well-known Glimmer ids (stable upstream: bos 200000, eos 200001,
  // turn-end <|eot|> 200008) so pre-load metadata stays sane; load() still
  // fails without the promoted library.
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
