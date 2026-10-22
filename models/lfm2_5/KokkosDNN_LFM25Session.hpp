#ifndef KOKKOSDNN_LFM25_SESSION_HPP
#define KOKKOSDNN_LFM25_SESSION_HPP

// Persistent multi-step decoding session over Model (models/lfm2_5).
//
// Model::generate() is one-shot: it reset()s, prefills, and decodes in a
// single call. A server (or any multi-turn / streaming caller) instead
// needs the same state machine across calls: prefill once, then pull one
// token per step() while usage counters, the RNG stream, the stop policy,
// and the streaming detokenizer carry over. That is what Session owns.
//
// Lifecycle per turn:
//   Session s(cfg, weights, tok, opts);   // rents KV/conv state, seeds RNG
//   s.prefill(prompt_ids);                // reset + chunked forward(prompt)
//   while (!s.finished()) {               // pull tokens one at a time
//     StepOutcome st = s.step();
//     ... emit st.delta ...
//   }
//   s.flush();                            // end-of-stream detok tail
//
// Multi-turn: calling prefill() again after a turn finished starts a new
// turn reusing the same weights and rented state. The new prefill rewrites
// KV/conv state from position 0 (model.reset() + forward), so turns are
// exact but share no prefix cache: keeping the old prefix resident across
// turns (prefix-reuse / prompt-cache) is future work.
//
// Determinism: the session owns one HostRNG. Construction seeds it from
// SessionOptions::seed; prefill(ids) reseeds from opts.seed iff
// opts.seedProvided, otherwise the stream continues from the previous
// turn. prefill(ids, seed, true) forces a reseed. Same seed + same prompt
// + same options therefore reproduces the same step sequence bit-for-bit
// (greedy draws never touch the RNG at all; sampled draws consume it in
// call order, exactly like Model::generate).
//
// Stop policy: empty opts.stop_ids selects {cfg.eos}; empty
// opts.stop_strings selects default_stop_strings() ("<|im_end|>"). A
// sampled stop id ends the turn with finish=Stop (the stop token IS
// counted as a completion token and its (usually empty) piece still flows
// through the detokenizer, mirroring generate() which includes the stop
// token in its output). A stop string fires inside the detokenizer and
// likewise ends the turn with Stop. Exhausting max_new generated tokens,
// or reaching cfg.max_seq_len, ends the turn with finish=Length.
// step() checks seq_len BEFORE forwarding and returns a terminal Length
// outcome without forwarding past the cap; a step that exactly fills the
// window still returns its token normally and the *next* step reports
// Length. Calling step() on a finished session returns the terminal
// outcome again (sticky), it never forwards.
//
// Logprobs: passed through from sample_token_result() when
// opts.sampling.logprobs > 0, otherwise left empty. Per-id decode uses
// Tokenizer::decode({id}); an id the tokenizer cannot decode (e.g. an
// out-of-range id under a fixture vocab) contributes an empty piece
// rather than throwing, so streaming never dies on undecodable ids.
//
// Threads/ownership: Session owns its Model (KV + conv state), its
// staging views, its Tokenizer copy, and its detokenizer. Not
// thread-safe; one session per request thread.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN_Detokenizer.hpp"
#include "KokkosDNN_LFM25Model.hpp"
#include "KokkosDNN_LFM25Tokenizer.hpp"
#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_Usage.hpp"

namespace KokkosDNN {
namespace LFM25 {

struct SessionOptions {
  ::KokkosDNN::SampleOptions sampling;
  std::vector<int> stop_ids;              // empty -> {cfg.eos}
  std::vector<std::string> stop_strings;  // empty -> default_stop_strings()
  int max_new = 256;                      // completion-token cap per turn
  uint64_t seed = 0;
  bool seedProvided = false;  // prefill(ids) reseeds from seed iff true
};

struct StepOutcome {
  int id = -1;  // sampled token (-1 on terminal outcomes that sample nothing)
  std::string delta;  // newly releasable text (may be empty = "hold")
  std::vector<float> logprobs;  // top-N logprobs (empty unless logprobs > 0)
  std::vector<int> top_ids;     // top-N ids matching logprobs
  bool stop_hit = false;  // stop id OR stop string fired on this step
  std::string matched;    // stop string that matched ("" for pure id-stop)
  ::KokkosDNN::Common::FinishReason finish =
      ::KokkosDNN::Common::FinishReason::Stop;
  bool finished = false;  // turn is over; step() goes sticky past this
};

template <class ExecSpace, class Scalar = DefaultScalar>
class Session {
 public:
  using ViewInt = Kokkos::View<int*, ExecSpace>;
  using HostInt = typename ViewInt::host_mirror_type;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;

  Session(const Config& cfg, const Weights<ExecSpace, Scalar>& w,
          Tokenizer tok, SessionOptions opts = SessionOptions{})
      : model_(cfg, w),
        tok_(std::move(tok)),
        opts_(std::move(opts)),
        rng_(opts_.seed),
        stop_ids_(resolve_stop_ids(opts_, model_.cfg)),
        detok_(resolve_stop_strings(opts_)),
        one_("KokkosDNN::LFM25::Session::one", 1) {}

  // Start (or restart) a turn: reset state, forward the prompt with the
  // model's chunked prefill, stash the last-prompt-token logits for the
  // first step(). Reseeds the RNG from opts.seed iff opts.seedProvided.
  void prefill(const std::vector<int>& ids) {
    prefill(ids, opts_.seed, opts_.seedProvided);
  }

  // prefill() with an explicit reseed (hasSeed=true) or a continued RNG
  // stream (hasSeed=false).
  void prefill(const std::vector<int>& ids, uint64_t seed, bool hasSeed) {
    if (ids.empty())
      throw std::invalid_argument("Session::prefill: empty prompt");
    if (hasSeed) rng_.seed(seed);
    model_.reset();
    ensure_prompt(ids.size());
    auto toks =
        Kokkos::subview(prompt_buf_, std::make_pair(size_t(0), ids.size()));
    {
      auto h =
          Kokkos::subview(prompt_host_, std::make_pair(size_t(0), ids.size()));
      for (size_t i = 0; i < ids.size(); ++i) h(i) = ids[i];
      Kokkos::deep_copy(toks, h);
    }
    auto lg = model_.forward(toks);
    // The returned view aliases model scratch (overwritten by the next
    // forward), so keep a private device copy for the first step().
    if (pend_.extent(0) != lg.extent(0))
      pend_ = View1("KokkosDNN::LFM25::Session::pending", lg.extent(0));
    Kokkos::deep_copy(pend_, lg);
    has_pending_ = true;
    prev_ = -1;
    ids_ = ids;
    usage_ = ::KokkosDNN::Common::Usage{};
    usage_.prompt_tokens = ids.size();
    detok_.reset();
    finished_ = false;
    finish_ = ::KokkosDNN::Common::FinishReason::Stop;
    last_ = StepOutcome{};
  }

  // Sample exactly one token. The first step after prefill() samples from
  // the stashed prefill logits (no forward); later steps forward the
  // previously sampled token, then sample. Never forwards past
  // cfg.max_seq_len: at the cap it returns a terminal Length outcome.
  StepOutcome step() {
    if (finished_) return last_;
    StepOutcome out;
    const int max_seq = model_.cfg.max_seq_len;
    if ((int)model_.seq_len >= max_seq) {
      out.id = -1;
      out.finished = true;
      out.finish = ::KokkosDNN::Common::FinishReason::Length;
      finished_ = true;
      finish_ = out.finish;
      last_ = out;
      return out;
    }
    View1 lg;
    if (has_pending_) {
      lg = pend_;
      has_pending_ = false;
    } else {
      // Device-side single-token staging (no host mirror + deep_copy per
      // token), mirroring Model::generate()'s decode loop. Hoisted to a
      // local: device lambdas must not capture `this`.
      auto one = one_;
      const int tok = prev_;
      Kokkos::parallel_for(
          "KokkosDNN::LFM25::Session::fill_one",
          Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(int i) {
            (void)i;
            one(0) = tok;
          });
      lg = model_.forward(one_);
    }
    ::KokkosDNN::SampleResult r =
        ::KokkosDNN::sample_token_result<ExecSpace, Scalar>(
            lg, opts_.sampling, ids_, rng_);
    ids_.push_back(r.id);
    prev_ = r.id;
    usage_.completion_tokens++;
    out.id = r.id;
    if (opts_.sampling.logprobs > 0) {
      out.logprobs = r.logprobs;
      out.top_ids = r.top_ids;
    }
    std::string piece;
    try {
      piece = tok_.decode(std::vector<int>{r.id});
    } catch (...) {
      piece.clear();  // undecodable id: stream "" rather than throw
    }
    ::KokkosDNN::Common::IncrementalDetokenizer::Step ds =
        detok_.push(r.id, piece);
    out.delta = std::move(ds.delta);
    const bool id_stop = std::binary_search(stop_ids_.begin(),
                                            stop_ids_.end(), r.id);
    if (id_stop || ds.stop_hit) {
      out.stop_hit = true;
      out.matched = ds.stop_hit ? ds.matched : std::string{};
      out.finished = true;
      out.finish = ::KokkosDNN::Common::FinishReason::Stop;
    } else if ((int)usage_.completion_tokens >= opts_.max_new) {
      out.finished = true;
      out.finish = ::KokkosDNN::Common::FinishReason::Length;
    }
    if (out.finished) {
      finished_ = true;
      finish_ = out.finish;
    }
    last_ = out;
    return out;
  }

  // End-of-stream: release whatever the detokenizer still holds back
  // (a stop-prefix tail or partial UTF-8). After a stop-string hit the
  // buffer was discarded, so this returns "" until the next prefill().
  std::string flush() { return detok_.flush(); }

  // Back to the fresh-constructed state (keeps weights, options, RNG
  // stream position). Call prefill() to start a turn afterwards.
  void reset() {
    model_.reset();
    detok_.reset();
    usage_ = ::KokkosDNN::Common::Usage{};
    ids_.clear();
    prev_ = -1;
    has_pending_ = false;
    finished_ = false;
    finish_ = ::KokkosDNN::Common::FinishReason::Stop;
    last_ = StepOutcome{};
  }

  ::KokkosDNN::Common::Usage usage() const { return usage_; }
  ::KokkosDNN::Common::FinishReason finish() const { return finish_; }
  bool finished() const { return finished_; }
  size_t generated() const { return usage_.completion_tokens; }
  const Config& config() const { return model_.cfg; }

 private:
  static std::vector<int> resolve_stop_ids(const SessionOptions& o,
                                           const Config& c) {
    if (o.stop_ids.empty()) return std::vector<int>{c.eos};
    std::vector<int> s = o.stop_ids;
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
  }

  static std::vector<std::string> resolve_stop_strings(
      const SessionOptions& o) {
    if (o.stop_strings.empty()) return default_stop_strings();
    return o.stop_strings;
  }

  void ensure_prompt(size_t n) {
    if (n <= prompt_cap_ && prompt_buf_.extent(0) >= n &&
        prompt_host_.extent(0) >= n)
      return;
    prompt_buf_ = ViewInt("KokkosDNN::LFM25::Session::prompt", n);
    prompt_host_ = HostInt("KokkosDNN::LFM25::Session::prompt_host", n);
    prompt_cap_ = n;
  }

  Model<ExecSpace, Scalar> model_;
  Tokenizer tok_;
  SessionOptions opts_;
  ::KokkosDNN::HostRNG rng_;
  std::vector<int> stop_ids_;  // sorted, deduped
  ::KokkosDNN::Common::IncrementalDetokenizer detok_;
  ViewInt prompt_buf_;
  HostInt prompt_host_;
  size_t prompt_cap_ = 0;
  ViewInt one_;  // single-token decode staging (device-filled)
  View1 pend_;   // stashed prefill logits for the first step()
  bool has_pending_ = false;
  int prev_ = -1;
  std::vector<int> ids_;  // prompt + sampled ids (sampler prior)
  ::KokkosDNN::Common::Usage usage_;
  bool finished_ = false;
  ::KokkosDNN::Common::FinishReason finish_ =
      ::KokkosDNN::Common::FinishReason::Stop;
  StepOutcome last_;  // sticky terminal outcome
};

}  // namespace LFM25
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_LFM25_SESSION_HPP
