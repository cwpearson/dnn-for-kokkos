// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// server/BackendLFM25.cpp: ChatBackend for the LFM2.5 architecture.
//
// Owns the LFM25 Config/Weights/Tokenizer plus one long-lived ISession
// reused across requests (the KV-cache entry placement — device vs host —
// is fixed at load from --prefix-cache-host). All inference, sampling, and
// chat logic runs through LFM25::Session; this file only translates the
// neutral server vocabulary to/from it.
//
// Without KOKKOSDNN_SERVER_HAVE_LFM25 (server configured with
// KokkosDNN_ENABLE_MODEL_LFM25=OFF) this TU compiles to a stub backend
// whose load() fails with a clear message, so "lfm2.5" still resolves in
// --arch/--help listings.
#include "Backend.hpp"

#ifdef KOKKOSDNN_SERVER_HAVE_LFM25

#include <KokkosDNN_LFM25Config.hpp>
#include <KokkosDNN_LFM25Session.hpp>
#include <KokkosDNN_LFM25Tokenizer.hpp>
#include <KokkosDNN_LFM25Weights.hpp>

#include <algorithm>
#include <stdexcept>

namespace KokkosDNN {
namespace Server {
namespace Impl {

namespace LFM25 = ::KokkosDNN::LFM25;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = LFM25::DefaultScalar;

class LFM25StreamParser final : public StreamParser {
 public:
  LFM25StreamParser(const std::vector<std::string>& stops,
                    bool start_in_reasoning)
      : p_(stops, start_in_reasoning) {}
  StepDeltas feed(const std::string& raw_piece) override {
    const LFM25::StreamDeltas d = p_.feed(raw_piece);
    return {std::move(d.display), std::move(d.reasoning)};
  }
  ParsedTail finish() override {
    const LFM25::ToolParseResult t = p_.finish();
    ParsedTail out;
    out.text = std::move(t.text);
    out.reasoning = std::move(t.reasoning);
    out.calls.reserve(t.calls.size());
    for (const auto& c : t.calls) out.calls.push_back({c.id, c.name, c.arguments});
    return out;
  }

 private:
  LFM25::ToolCallStreamParser p_;
};

class LFM25Backend final : public ChatBackend {
 public:
  const std::string& arch() const override {
    static const std::string a = kArchLFM25;
    return a;
  }
  std::string default_model_id() const override { return "lfm2_5"; }

  bool load(const LoadOptions& opts, std::string& err) override {
    if (opts.weights.size() != 1) {
      err = "arch lfm2.5 needs exactly one --weights GGUF file (got " +
            std::to_string(opts.weights.size()) + ")";
      return false;
    }
    if (opts.tokenizer.empty()) {
      err = "missing required --tokenizer PATH (tokenizer.json)";
      return false;
    }
    try {
      model_id_ = opts.model_id;
      cfg_ = opts.config_path.empty() ? LFM25::Config::lfm2_5_24b_a2b()
                                      : LFM25::Config::from_json_file(opts.config_path);
      if (opts.max_seq_len > 0) cfg_.max_seq_len = opts.max_seq_len;
      cfg_.validate();
      weights_ = LFM25::Weights<Exec, Scalar>{};
      weights_.allocate(cfg_);
      LFM25::load_gguf(cfg_, opts.weights[0], weights_);
      tok_ = LFM25::Tokenizer(opts.tokenizer);
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
  int eos_id() const override { return cfg_.eos; }
  std::vector<int> default_stop_ids() const override { return {cfg_.eos}; }
  std::vector<std::string> default_stop_strings() const override {
    return LFM25::default_stop_strings();
  }
  bool supports_tools() const override { return true; }

  std::vector<ServerChatMessage> make_single_turn(
      const std::string& prompt) const override {
    std::vector<ServerChatMessage> out;
    // LFM25::make_chat_messages builds the optional-system + one-user-turn
    // shape; convert back to the neutral message.
    for (const auto& m : LFM25::make_chat_messages(prompt, "", "user")) {
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
    return LFM25::encode_chatml_with_tools(tok_, to_native(messages),
                                           tools_json, system_suffix,
                                           add_generation_prompt, true);
  }

  void set_turn_options(const ServerSessionOptions& opts) override {
    LFM25::SessionOptions n;
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
    const LFM25::CachedPrefillOutcome o =
        sess_->prefill_cached(full, content, seed, hasSeed);
    return {o.hit, o.reused};
  }

  BackendStepOutcome step() override {
    const LFM25::StepOutcome s = sess_->step();
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
      bool start_in_reasoning) const override {
    return std::make_unique<LFM25StreamParser>(stops, start_in_reasoning);
  }

 private:
  static LFM25::ChatMessage to_native_msg(const ServerChatMessage& m) {
    LFM25::ChatMessage n;
    n.role = m.role;
    n.content = m.content;
    n.thinking = m.thinking;
    n.tool_calls.reserve(m.tool_calls.size());
    for (const auto& c : m.tool_calls) {
      LFM25::ToolCall t;
      t.id = c.id;
      t.name = c.name;
      t.arguments = c.arguments;
      n.tool_calls.push_back(std::move(t));
    }
    n.tool_call_id = m.tool_call_id;
    return n;
  }

  static std::vector<LFM25::ChatMessage> to_native(
      const std::vector<ServerChatMessage>& v) {
    std::vector<LFM25::ChatMessage> out;
    out.reserve(v.size());
    for (const auto& m : v) out.push_back(to_native_msg(m));
    return out;
  }

  // Reuse one long-lived Session: avoids re-renting the KV cache and
  // re-copying the tokenizer on every request. set_turn_options() +
  // prefill_cached() reset all turn state; the reseed below reproduces
  // fresh-session RNG behavior exactly.
  void ensure_session() {
    if (sess_) {
      if (has_pending_opts_) {
        sess_->set_options(pending_opts_);
        has_pending_opts_ = false;
      }
      return;
    }
    const LFM25::SessionOptions o =
        has_pending_opts_ ? pending_opts_ : LFM25::SessionOptions{};
    has_pending_opts_ = false;
    if (prefix_cache_host_)
      sess_ = std::make_unique<LFM25::Session<Exec, Scalar, Kokkos::HostSpace>>(
          cfg_, weights_, tok_, o);
    else
      sess_ = std::make_unique<LFM25::Session<Exec, Scalar>>(cfg_, weights_,
                                                             tok_, o);
  }

  std::string model_id_;
  LFM25::Config cfg_;
  LFM25::Weights<Exec, Scalar> weights_;
  LFM25::Tokenizer tok_;
  bool prefix_cache_host_ = false;
  std::unique_ptr<LFM25::ISession> sess_;
  LFM25::SessionOptions pending_opts_;
  bool has_pending_opts_ = false;
};

namespace {
struct LFM25Registrar {
  LFM25Registrar() {
    register_backend(kArchLFM25,
                     []() { return std::make_unique<LFM25Backend>(); });
  }
};
LFM25Registrar g_lfm25_registrar;
}  // namespace

}  // namespace Impl
}  // namespace Server
}  // namespace KokkosDNN

#else  // !KOKKOSDNN_SERVER_HAVE_LFM25

namespace KokkosDNN {
namespace Server {
namespace Impl {

class LFM25Unavailable final : public ChatBackend {
 public:
  const std::string& arch() const override {
    static const std::string a = kArchLFM25;
    return a;
  }
  std::string default_model_id() const override { return "lfm2_5"; }
  bool load(const LoadOptions&, std::string& err) override {
    err =
        "arch 'lfm2.5' was disabled at configure time "
        "(rebuild with -DKokkosDNN_ENABLE_MODEL_LFM25=ON)";
    return false;
  }
  const std::string& model_id() const override {
    static const std::string m;
    return m;
  }
  int max_seq_len() const override { return 0; }
  int eos_id() const override { return 124900; }  // <|im_end|>
  std::vector<int> default_stop_ids() const override { return {124900}; }
  std::vector<std::string> default_stop_strings() const override {
    return {"<|im_end|>"};
  }
  bool supports_tools() const override { return true; }
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
struct LFM25Registrar {
  LFM25Registrar() {
    register_backend(kArchLFM25,
                     []() { return std::make_unique<LFM25Unavailable>(); });
  }
};
LFM25Registrar g_lfm25_registrar;
}  // namespace

}  // namespace Impl
}  // namespace Server
}  // namespace KokkosDNN

#endif
