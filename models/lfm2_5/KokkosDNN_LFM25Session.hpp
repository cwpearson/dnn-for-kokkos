// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
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
// exact but share no prefix cache. For agentic loops where each turn
// extends the previous history, use prefill_cached(full, content): a
// single-entry exact-prefix cache (see KokkosDNN_LFM25PrefixCache.hpp) resumes
// from the longest cached content prefix and forwards only the suffix,
// bit-identical to a full prefill.
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
#include "KokkosDNN_LFM25PrefixCache.hpp"
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

// Per-turn prefix-cache outcome for logging/observability.
struct CachedPrefillOutcome {
  bool hit = false;   // a cached prefix was reused
  size_t reused = 0;  // reused prefix length in tokens (0 on miss)
};

// Runtime-polymorphic Session front-end. The cache storage space (device
// vs host memory) is a compile-time Session template parameter, but the
// server picks it from a CLI flag at startup; this interface lets the
// server hold either instantiation behind one pointer. All cross-cutting
// types are non-templated, so no templated virtuals are needed. The
// per-step virtual dispatch is negligible next to a model forward.
struct ISession {
  virtual ~ISession() = default;
  virtual void set_options(SessionOptions opts) = 0;
  virtual void prefill(const std::vector<int>& ids) = 0;
  virtual void prefill(const std::vector<int>& ids, uint64_t seed,
                       bool hasSeed) = 0;
  virtual CachedPrefillOutcome prefill_cached(
      const std::vector<int>& full, const std::vector<int>& content) = 0;
  virtual CachedPrefillOutcome prefill_cached(const std::vector<int>& full,
                                             const std::vector<int>& content,
                                             uint64_t seed, bool hasSeed) = 0;
  virtual StepOutcome step() = 0;
  virtual std::string flush() = 0;
  virtual void reset() = 0;
  virtual void clear_prefix_cache() = 0;
  virtual ::KokkosDNN::Common::Usage usage() const = 0;
  virtual ::KokkosDNN::Common::FinishReason finish() const = 0;
  virtual bool finished() const = 0;
  virtual size_t generated() const = 0;
  virtual size_t prefix_cache_size() const = 0;
};

template <class ExecSpace, class Scalar = DefaultScalar,
          class StoreSpace = typename ExecSpace::memory_space>
class Session : public ISession {
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

  // Replace the request-local sampling/stop/seed policy (e.g. when a
  // long-lived Session is reused across requests with different options).
  // Re-derives the stop policy; the next prefill() resets all turn state.
  void set_options(SessionOptions opts) override {
    opts_ = std::move(opts);
    stop_ids_ = resolve_stop_ids(opts_, model_.cfg);
    detok_ = ::KokkosDNN::Common::IncrementalDetokenizer(
        resolve_stop_strings(opts_));
  }

  void prefill(const std::vector<int>& ids) override {
    prefill(ids, opts_.seed, opts_.seedProvided);
  }

  // Start (or restart) a turn: reset state, forward the prompt with the
  // model's chunked prefill, stash the last-prompt-token logits for the
  // first step(). Reseeds the RNG from opts.seed iff opts.seedProvided.
  // prefill() with an explicit reseed (hasSeed=true) or a continued RNG
  // stream (hasSeed=false).
  void prefill(const std::vector<int>& ids, uint64_t seed,
               bool hasSeed) override {
    if (ids.empty())
      throw std::invalid_argument("Session::prefill: empty prompt");
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
    last_ = StepOutcome{};
  }

  // Single-entry prefix-cache capacity (see KokkosDNN_LFM25PrefixCache.hpp).
  static constexpr size_t kPrefixCacheCapacity = 1;

  CachedPrefillOutcome prefill_cached(const std::vector<int>& full,
                                     const std::vector<int>& content) override {
    return prefill_cached(full, content, opts_.seed, opts_.seedProvided);
  }

  // Cached prefill for history-growing callers (e.g. the server's agentic
  // loop): `full` is the whole prompt to generate from (including the
  // trailing generation prompt), `content` is the cacheable history prefix
  // (everything before that prompt, expected to be an exact prefix of
  // `full`). Resumes from the longest cached content prefix that `full`
  // starts with and forwards only the suffix; bit-identical to
  // prefill(full). New content states are snapshotted for future turns
  // (a miss replaces the single cached entry). When `content` is not an exact prefix of `full`
  // (e.g. tokenizer boundary drift), falls back to an uncached full
  // prefill without touching the cache.
  CachedPrefillOutcome prefill_cached(const std::vector<int>& full,
                                     const std::vector<int>& content,
                                     uint64_t seed, bool hasSeed) override {
    if (full.empty())
      throw std::invalid_argument("Session::prefill_cached: empty prompt");
    if (content.size() > full.size() || !is_prefix_of(content, full)) {
      prefill(full, seed, hasSeed);
      return CachedPrefillOutcome{};
    }
    if (hasSeed) rng_.seed(seed);
    size_t hit_len = 0;
    const int hit =
        prefix_cache_.find(full, content.size(), hit_len);
    if (hit >= 0) {
      prefix_cache_.restore(static_cast<size_t>(hit), model_);
      prefix_cache_.promote(static_cast<size_t>(hit));
    } else {
      model_.reset();
    }
    if (hit_len < content.size()) {
      // Forward the uncached content tail, then snapshot the content state
      // for future turns before forwarding the generation-prompt suffix.
      View1 lg = forward_range(full, hit_len, content.size());
      std::vector<int> key(full.begin(),
                           full.begin() + static_cast<long>(content.size()));
      prefix_cache_.save(key, model_, lg);
      if (content.size() < full.size())
        forward_full_from(content.size(), full);
      else
        stash_logits(lg);
    } else if (content.size() < full.size()) {
      // Exact content hit: the content state is already resident.
      forward_full_from(content.size(), full);
    } else {
      // Repeated identical prompt (full == content == cached): no suffix
      // to forward; reuse the stashed last-token logits directly.
      stash_entry_logits(static_cast<size_t>(hit));
    }
    has_pending_ = true;
    prev_ = -1;
    ids_ = full;
    usage_ = ::KokkosDNN::Common::Usage{};
    usage_.prompt_tokens = full.size();
    detok_.reset();
    finished_ = false;
    finish_ = ::KokkosDNN::Common::FinishReason::Stop;
    last_ = StepOutcome{};
    return CachedPrefillOutcome{hit >= 0, hit_len};
  }

  // Drop all cached prefixes (e.g. after weights/config changes).
  void clear_prefix_cache() override { prefix_cache_.clear(); }
  size_t prefix_cache_size() const override { return prefix_cache_.size(); }

  // Sample exactly one token. The first step after prefill() samples from
  // the stashed prefill logits (no forward); later steps forward the
  // previously sampled token, then sample. Never forwards past
  // cfg.max_seq_len: at the cap it returns a terminal Length outcome.
  StepOutcome step() override {
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
    ::KokkosDNN::SampleResult r;
    // Same routing as sample_token_result(): the neutral greedy path stays
    // on device (parallel_reduce argmax, no host mirror); everything else
    // mirrors once into persistent session scratch (no per-token
    // create_mirror_view/host-vector mallocs) and runs the shared host
    // pipeline. Bit-identical to sample_token_result() by construction:
    // same device argmax, same host entry point, same predicates over the
    // same public SampleOptions fields.
    const auto& so = opts_.sampling;
    const bool need_host =
        !so.is_greedy() || so.repetition_penalty != 1.0f ||
        so.frequency_penalty != 0.0f || so.presence_penalty != 0.0f ||
        !so.logit_bias.empty() || so.logprobs > 0;
    if (!need_host && so.logprobs <= 0) {
      r.id = ::KokkosDNN::device_argmax<ExecSpace, Scalar>(lg);
    } else {
      const size_t V = lg.extent(0);
      if (V == 0)
        throw std::invalid_argument("sample_token_result: empty logits");
      ensure_samp(V);
      Kokkos::deep_copy(samp_host_, lg);
      for (size_t i = 0; i < V; ++i)
        samp_f_[i] = static_cast<float>(samp_host_(i));
      r = ::KokkosDNN::sample_host_logits_result(samp_f_.data(), V, so, ids_,
                                                rng_);
    }
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
      idbuf_[0] = r.id;  // persistent 1-elem staging: no vector alloc/token
      piece = tok_.decode(idbuf_);
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
  std::string flush() override { return detok_.flush(); }

  // Back to the fresh-constructed state (keeps weights, options, RNG
  // stream position). Call prefill() to start a turn afterwards.
  void reset() override {
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

  ::KokkosDNN::Common::Usage usage() const override { return usage_; }
  ::KokkosDNN::Common::FinishReason finish() const override { return finish_; }
  bool finished() const override { return finished_; }
  size_t generated() const override { return usage_.completion_tokens; }
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

  // True when `pre` is an exact prefix of `full`.
  static bool is_prefix_of(const std::vector<int>& pre,
                           const std::vector<int>& full) {
    if (pre.size() > full.size()) return false;
    for (size_t i = 0; i < pre.size(); ++i) {
      if (pre[i] != full[i]) return false;
    }
    return true;
  }

  // Forward full[a:b) (model.seq_len must equal a). Returns the last-token
  // logits view (aliases model scratch: the caller must copy before the
  // next forward).
  View1 forward_range(const std::vector<int>& full, size_t a, size_t b) {
    ensure_prompt(b - a);
    auto toks = Kokkos::subview(prompt_buf_, std::make_pair(size_t(0), b - a));
    {
      auto h = Kokkos::subview(prompt_host_, std::make_pair(size_t(0), b - a));
      for (size_t i = a; i < b; ++i) h(i - a) = full[i];
      Kokkos::deep_copy(toks, h);
    }
    return model_.forward(toks);
  }

  // Keep a private device copy of last-token logits for the first step()
  // (the returned view aliases model scratch, overwritten by next forward).
  void stash_logits(const View1& lg) {
    if (pend_.extent(0) != lg.extent(0))
      pend_ = View1("KokkosDNN::LFM25::Session::pending", lg.extent(0));
    Kokkos::deep_copy(pend_, lg);
  }

  // Uncached full forward + stash.
  void forward_full(const std::vector<int>& ids) {
    stash_logits(forward_range(ids, 0, ids.size()));
  }

  // Forward the suffix full[from:) and stash its logits.
  void forward_full_from(size_t from, const std::vector<int>& full) {
    stash_logits(forward_range(full, from, full.size()));
  }

  // Exact-hit path with an empty suffix: reuse the entry's stashed logits.
  void stash_entry_logits(size_t idx) {
    const size_t v = prefix_cache_.entry_logits_size(idx);
    if (pend_.extent(0) != v)
      pend_ = View1("KokkosDNN::LFM25::Session::pending", v);
    prefix_cache_.restore_logits(idx, pend_);
  }

  // Grow-once host staging for the sampled (non-greedy) path: the logits
  // mirror and its float conversion buffer persist across steps so sampled
  // tokens cost one deep_copy plus O(V) compute, with no per-token host
  // allocations.
  void ensure_samp(size_t n) {
    if (samp_host_.extent(0) < n)
      samp_host_ = typename View1::host_mirror_type(
          "KokkosDNN::LFM25::Session::samp_host", n);
    if (samp_f_.size() < n) samp_f_.resize(n);
  }

  Model<ExecSpace, Scalar> model_;
  Tokenizer tok_;
  SessionOptions opts_;
  ::KokkosDNN::HostRNG rng_;
  std::vector<int> stop_ids_;  // sorted, deduped
  ::KokkosDNN::Common::IncrementalDetokenizer detok_;
  // Exact-prefix KV cache across turns (single entry). Plain prefill()
  // leaves it untouched; prefill_cached() resumes from it and snapshots new
  // content states into it. Entry storage lives in StoreSpace (device
  // memory by default; HostSpace when the server is started with
  // --prefix-cache-host).
  PrefixCache<ExecSpace, Scalar, StoreSpace> prefix_cache_{
      kPrefixCacheCapacity};
  ViewInt prompt_buf_;
  HostInt prompt_host_;
  size_t prompt_cap_ = 0;
  ViewInt one_;  // single-token decode staging (device-filled)
  View1 pend_;   // stashed prefill logits for the first step()
  bool has_pending_ = false;
  int prev_ = -1;
  std::vector<int> ids_;  // prompt + sampled ids (sampler prior)
  std::vector<int> idbuf_{0};  // single-id decode staging (no per-token alloc)
  typename View1::host_mirror_type samp_host_;  // persistent logits mirror
  std::vector<float> samp_f_;  // persistent mirror-to-float conversion buffer
  ::KokkosDNN::Common::Usage usage_;
  bool finished_ = false;
  ::KokkosDNN::Common::FinishReason finish_ =
      ::KokkosDNN::Common::FinishReason::Stop;
  StepOutcome last_;  // sticky terminal outcome
};

}  // namespace LFM25
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_LFM25_SESSION_HPP
