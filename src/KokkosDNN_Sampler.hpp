// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SAMPLER_HPP
#define KOKKOSDNN_SAMPLER_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace KokkosDNN {

// Sampling options for autoregressive decode.
//
//  temperature == 0 (or <= 0): greedy argmax (temperature/top-p/min-p ignored).
//  top_k == 1: greedy argmax regardless of temperature.
//  top_k <= 0 or >= vocab: no top-k filtering.
//  top_p >= 1: no nucleus filtering. top_p <= 0 keeps top-1 only.
//  min_p <= 0: no min-p filtering. min_p >= 1 keeps argmax only.
//  repetition_penalty == 1: disabled. > 1 suppresses prior tokens (HF style:
//    positive penalized logits are divided, negative multiplied).
//  seed: used when the caller lets generate() construct its own RNG; an
//    explicitly passed HostRNG always wins over this field.
struct SampleOptions {
  float temperature = 0.0f;
  int top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 0;
  // OpenAI-mapped sampling fields (all neutral by default: existing paths
  // behave exactly as before when these are untouched).
  //
  //  frequency_penalty: OpenAI-style logit penalty proportional to how
  //    often each id already appears in `prior`:
  //    logits[id] -= frequency_penalty * count(id).
  //    == 0: disabled. > 0 suppresses repeated ids; < 0 encourages them.
  //  presence_penalty: binary variant applied once per distinct id seen:
  //    logits[id] -= presence_penalty  (iff count(id) > 0).
  //    == 0: disabled.
  //  logit_bias: per-id additive bias, logits[id] += bias, applied to
  //    every decode step. Out-of-range ids are ignored. Empty: disabled.
  //  logprobs: 0 (default) means off — no logprob bookkeeping, zero
  //    overhead. > 0 requests the top-N token logprobs alongside the
  //    sampled token (clamped to vocab size).
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  std::vector<std::pair<int, float>> logit_bias;
  int logprobs = 0;

  bool is_greedy() const { return temperature <= 0.0f || top_k == 1; }

  // Meta's recommended sampling for Llama 3.1 Instruct (matches the
  // upstream generation_config.json: do_sample=true, temperature=0.6,
  // top_p=0.9; no top_k / repetition_penalty specified, so those stay
  // neutral). Same values appear in Meta's chat-completion docs and the
  // vLLM Llama 3.1 Instruct examples.
  static SampleOptions llama_3_1_8b_instruct() {
    SampleOptions o;
    o.temperature = 0.6f;
    o.top_p = 0.9f;
    return o;
  }
};

inline bool is_greedy_sample(const SampleOptions& opt) { return opt.is_greedy(); }

// Seeded host RNG. Deterministic for a given seed: same seed + same call
// sequence => identical outputs on every platform (mt19937_64 is fully
// specified; the [0,1) conversion uses the top 53 bits, matching the
// standard uniform mapping without depending on std::uniform_real_distribution,
// whose exact mapping is implementation-defined).
class HostRNG {
 public:
  explicit HostRNG(uint64_t seed = 0) : eng_(seed) {}

  void seed(uint64_t s) { eng_.seed(s); }

  uint64_t next_u64() { return eng_(); }

  // Uniform double in [0, 1). Never returns exactly 1.0.
  double next_uniform() {
    return (static_cast<double>(eng_() >> 11)) *
           (1.0 / 9007199254740992.0);  // 2^53
  }

  float next_uniform_f() { return static_cast<float>(next_uniform()); }

 private:
  std::mt19937_64 eng_;
};

namespace Impl {

// Softmax (max-subtracted) over vals. Returns empty on empty input.
inline std::vector<float> softmax(const std::vector<float>& vals) {
  std::vector<float> p(vals.size());
  if (vals.empty()) return p;
  float m = *std::max_element(vals.begin(), vals.end());
  double sum = 0.0;
  for (size_t i = 0; i < vals.size(); ++i) {
    p[i] = std::exp(static_cast<double>(vals[i]) - m);
    sum += p[i];
  }
  if (!(sum > 0.0)) {  // all -inf: fall back to uniform
    float u = 1.0f / static_cast<float>(vals.size());
    std::fill(p.begin(), p.end(), u);
    return p;
  }
  for (auto& v : p) v = static_cast<float>(v / sum);
  return p;
}

// Order indices of vals in descending order (full sort; used on the already
// top-k-truncated candidate set, so this is small in the common case).
inline std::vector<int> argsort_desc(const std::vector<float>& vals) {
  std::vector<int> idx(vals.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return vals[a] > vals[b]; });
  return idx;
}

// Categorical draw over unnormalized weights. Deterministic given rng state;
// returns 0 on empty input, first positive-weight index when u lands on 0.
inline int categorical(const std::vector<float>& weights, HostRNG& rng) {
  if (weights.empty()) return 0;
  double total = 0.0;
  for (float w : weights) total += w > 0 ? w : 0;
  if (!(total > 0.0)) return 0;
  double u = rng.next_uniform() * total;
  double acc = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (weights[i] > 0) acc += weights[i];
    if (u < acc) return static_cast<int>(i);
  }
  // Rounding edge: return last positive weight.
  for (size_t i = weights.size(); i-- > 0;)
    if (weights[i] > 0) return static_cast<int>(i);
  return 0;
}

}  // namespace Impl

// Divide positive penalized logits by `penalty`, multiply negative ones
// (HF / llama.cpp convention). Each distinct prior id is penalized once,
// regardless of how often it occurs. Out-of-range ids are ignored.
// penalty <= 1 (or NaN-safe <= 1 check fails) is a no-op.
inline void apply_repetition_penalty(std::vector<float>& logits,
                                     const std::vector<int>& prior,
                                     float penalty) {
  if (!(penalty > 1.0f) || logits.empty() || prior.empty()) return;
  std::vector<char> seen(logits.size(), 0);
  for (int id : prior) {
    if (id < 0 || (size_t)id >= logits.size() || seen[(size_t)id]) continue;
    seen[(size_t)id] = 1;
    float& v = logits[(size_t)id];
    if (v > 0.0f)
      v /= penalty;
    else if (v < 0.0f)
      v *= penalty;
    // exact 0 stays 0.
  }
}

// Per-id additive logit bias (OpenAI `logit_bias`): logits[id] += bias.
// Applied after the repetition penalty, before frequency/presence
// penalties and temperature scaling. Out-of-range ids are ignored.
// Empty bias list is a no-op.
inline void apply_logit_bias(
    std::vector<float>& logits,
    const std::vector<std::pair<int, float>>& bias) {
  if (bias.empty() || logits.empty()) return;
  for (const auto& kv : bias) {
    int id = kv.first;
    if (id < 0 || (size_t)id >= logits.size()) continue;
    logits[(size_t)id] += kv.second;
  }
}

// OpenAI-style frequency + presence penalties on LOGITS (pre-softmax):
//   logits[id] -= frequency_penalty * count(id) +
//                 (count(id) > 0 ? presence_penalty : 0),
// where count(id) is the number of occurrences of id in `prior`.
// Both zero is a no-op. Negative values encourage (rather than suppress)
// repetition. Out-of-range prior ids are ignored. Counts are built
// internally from `prior` so call sites stay simple.
inline void apply_frequency_presence(std::vector<float>& logits,
                                     const std::vector<int>& prior,
                                     float frequency_penalty,
                                     float presence_penalty) {
  if ((frequency_penalty == 0.0f && presence_penalty == 0.0f) ||
      logits.empty() || prior.empty())
    return;
  std::vector<int> counts(logits.size(), 0);
  for (int id : prior) {
    if (id < 0 || (size_t)id >= logits.size()) continue;
    counts[(size_t)id]++;
  }
  for (size_t i = 0; i < logits.size(); ++i) {
    if (counts[i] == 0) continue;
    logits[i] -= frequency_penalty * static_cast<float>(counts[i]) +
                 presence_penalty;
  }
}

// Sampled token plus (optionally) the top-N token logprobs.
// `logprobs`/`top_ids` are empty when `SampleOptions::logprobs <= 0;
// otherwise they hold the top-N entries (N = min(logprobs, V)) of the
// temperature-scaled full-vocab distribution, in descending-logprob order,
// with `logprobs[i]` the natural-log probability of `top_ids[i]`.
struct SampleResult {
  int id = 0;
  std::vector<float> logprobs;
  std::vector<int> top_ids;
};

namespace Impl {

// Full-vocab log-softmax (max-subtracted, natural log). Empty on empty
// input; all -inf falls back to uniform (log(1/V)).
inline std::vector<float> log_softmax(const std::vector<float>& vals) {
  std::vector<float> lp(vals.size());
  if (vals.empty()) return lp;
  float m = *std::max_element(vals.begin(), vals.end());
  double sum = 0.0;
  for (size_t i = 0; i < vals.size(); ++i) {
    double e = std::exp(static_cast<double>(vals[i]) - m);
    sum += e;
    lp[i] = static_cast<float>(e);  // reuse as scratch for exp values
  }
  if (!(sum > 0.0)) {  // all -inf: uniform
    float u = -std::log(static_cast<double>(vals.size()));
    std::fill(lp.begin(), lp.end(), u);
    return lp;
  }
  double log_sum = std::log(sum);
  for (size_t i = 0; i < vals.size(); ++i)
    lp[i] = static_cast<float>(
        (static_cast<double>(vals[i]) - m) - log_sum);
  return lp;
}

// Top-N selection over log-probs: O(V) average via nth_element, then sort
// only the top N. Descending-logprob order; ties broken by smaller id
// (deterministic). N is clamped to [0, V].
inline void top_logprobs(const std::vector<float>& logp, int n,
                         std::vector<int>& top_ids,
                         std::vector<float>& top_lp) {
  top_ids.clear();
  top_lp.clear();
  const size_t V = logp.size();
  if (V == 0 || n <= 0) return;
  size_t N = (size_t)n < V ? (size_t)n : V;
  std::vector<int> ids(V);
  std::iota(ids.begin(), ids.end(), 0);
  std::nth_element(ids.begin(), ids.begin() + N, ids.end(),
                   [&](int a, int b) {
                     return logp[(size_t)a] > logp[(size_t)b];
                   });
  ids.resize(N);
  std::sort(ids.begin(), ids.end(), [&](int a, int b) {
    if (logp[(size_t)a] != logp[(size_t)b])
      return logp[(size_t)a] > logp[(size_t)b];
    return a < b;
  });
  top_ids = ids;
  top_lp.resize(N);
  for (size_t i = 0; i < N; ++i) top_lp[i] = logp[(size_t)ids[i]];
}

}  // namespace Impl

// Top-k keep set via partial selection (nth_element, O(V) average, no full
// sort of the 128k vocab). Returns kept ids sorted by descending logit for
// deterministic test/order semantics. k <= 0 or k >= V keeps everything
// (descending order).
inline std::vector<int> top_k_keep(const float* logits, size_t V, int k) {
  std::vector<int> ids(V);
  std::iota(ids.begin(), ids.end(), 0);
  if (V == 0) return ids;
  if (k <= 0 || (size_t)k >= V) {
    std::sort(ids.begin(), ids.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });
    return ids;
  }
  std::nth_element(ids.begin(), ids.begin() + k, ids.end(),
                   [&](int a, int b) { return logits[a] > logits[b]; });
  ids.resize((size_t)k);
  std::sort(ids.begin(), ids.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
  return ids;
}

inline std::vector<int> top_k_keep(const std::vector<float>& logits, int k) {
  return top_k_keep(logits.data(), logits.size(), k);
}

// Nucleus (top-p) keep set: softmax over the given logits, then the smallest
// prefix (in descending-logit order) whose cumulative mass >= top_p.
// Always keeps at least the argmax. top_p >= 1 keeps all; top_p <= 0 keeps
// exactly the argmax. Returned in descending-logit order.
inline std::vector<int> top_p_keep(const float* logits, size_t V, float top_p) {
  std::vector<int> order(V);
  std::iota(order.begin(), order.end(), 0);
  if (V == 0) return order;
  if (!(top_p < 1.0f)) {
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });
    return order;
  }
  std::vector<float> vals(V);
  for (size_t i = 0; i < V; ++i) vals[i] = logits[i];
  std::vector<float> probs = Impl::softmax(vals);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
  if (!(top_p > 0.0f)) return std::vector<int>{order[0]};
  std::vector<int> kept;
  kept.reserve(V);
  double cum = 0.0;
  for (int id : order) {
    kept.push_back(id);
    cum += probs[(size_t)id];
    if (cum >= top_p) break;
  }
  return kept;
}

inline std::vector<int> top_p_keep(const std::vector<float>& logits,
                                   float top_p) {
  return top_p_keep(logits.data(), logits.size(), top_p);
}

// Min-p keep set: softmax over the given logits, keep ids with
// prob >= min_p * max_prob (always keeps the argmax). min_p <= 0 keeps all;
// min_p >= 1 keeps exactly the argmax. Order: descending logit.
inline std::vector<int> min_p_keep(const float* logits, size_t V, float min_p) {
  std::vector<int> order(V);
  std::iota(order.begin(), order.end(), 0);
  if (V == 0) return order;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
  if (!(min_p > 0.0f)) return order;
  std::vector<float> vals(V);
  for (size_t i = 0; i < V; ++i) vals[i] = logits[i];
  std::vector<float> probs = Impl::softmax(vals);
  if (!(min_p < 1.0f)) return std::vector<int>{order[0]};
  float pmax = probs[(size_t)order[0]];
  float thresh = min_p * pmax;
  std::vector<int> kept;
  kept.reserve(V);
  for (int id : order) {
    if (probs[(size_t)id] >= thresh)
      kept.push_back(id);
    else
      break;  // descending-logit order is descending-prob order
  }
  if (kept.empty()) kept.push_back(order[0]);
  return kept;
}

inline std::vector<int> min_p_keep(const std::vector<float>& logits,
                                   float min_p) {
  return min_p_keep(logits.data(), logits.size(), min_p);
}

// Host sampling over float logits (already on host), returning the sampled
// id plus optional top-N logprobs.
//
// Pipeline (every logit adjustment applies pre-softmax, on LOGITS):
//   repetition penalty (on `prior`)
//     -> logit_bias (additive per-id)
//     -> frequency / presence penalties (from occurrence counts in `prior`)
//     -> greedy short-circuit (argmax, RNG untouched)
//     -> temperature scale
//     -> top-k partial selection -> softmax -> top-p -> min-p ->
//        renormalize -> categorical draw with `rng`.
//
// When `opt.logprobs > 0`, the returned top-N entries come from the
// full-vocab temperature-scaled distribution: softmax over the
// penalty/bias-adjusted, temperature-scaled logits BEFORE top-k
// truncation (OpenAI reports logprobs over the final distribution, not
// the truncated candidate set). Greedy draws report the same
// distribution (temperature <= 0 treated as 1.0, i.e. unscaled — there
// is no meaningful divide-by-zero scaling); the RNG is still never
// touched on the greedy path.
//
// Neutral-by-default: frequency_penalty == 0, presence_penalty == 0,
// empty logit_bias, and logprobs == 0 reproduce the legacy pipeline
// bit-for-bit.
//
// Untruncated short-circuit: when nothing can remove a candidate
// (top_k <= 0 or >= V, top_p >= 1, min_p <= 0 -- i.e. the common
// OpenAI request that only sets `temperature`), the kept set is the whole
// vocabulary and the two full-vocab sorts below would only reorder the
// categorical walk. Sampling in index order instead is the same
// distribution with one uniform draw and O(V) work (measured on a
// 125k-vocab fixture: 14.4 ms -> 1.8 ms per host token, which matters
// because the server pays it per generated token next to a batch-1 model
// step).
// Trade-off, documented: the id drawn for a given seed differs from the
// sorted walk, so sampled (not greedy) output is not bit-reproducible
// across this change. Filtered paths are untouched.
inline SampleResult sample_host_logits_result(
    const float* logits, size_t V, const SampleOptions& opt,
    const std::vector<int>& prior, HostRNG& rng) {
  if (V == 0)
    throw std::invalid_argument("sample_host_logits_result: empty logits");
  std::vector<float> work(logits, logits + V);
  apply_repetition_penalty(work, prior, opt.repetition_penalty);
  apply_logit_bias(work, opt.logit_bias);
  apply_frequency_presence(work, prior, opt.frequency_penalty,
                           opt.presence_penalty);

  // Full-vocab logprobs distribution: penalty/bias-adjusted logits,
  // temperature-scaled when meaningful (temp > 0), pre-truncation.
  // temperature <= 0 (greedy) is treated as 1.0 (unscaled).
  auto full_logprobs = [&]() {
    std::vector<float> scaled = work;
    if (opt.temperature > 0.0f && opt.temperature != 1.0f)
      for (auto& v : scaled) v /= opt.temperature;
    return Impl::log_softmax(scaled);
  };

  if (opt.is_greedy()) {
    int best = 0;
    for (size_t v = 1; v < V; ++v)
      if (work[v] > work[best]) best = static_cast<int>(v);
    SampleResult r;
    r.id = best;
    if (opt.logprobs > 0) {
      std::vector<float> lp = full_logprobs();
      Impl::top_logprobs(lp, opt.logprobs, r.top_ids, r.logprobs);
    }
    return r;
  }

  float temp = opt.temperature;
  if (!(temp > 0.0f)) {  // non-greedy flag combo with bad temp: be greedy
    int best = 0;
    for (size_t v = 1; v < V; ++v)
      if (work[v] > work[best]) best = static_cast<int>(v);
    SampleResult r;
    r.id = best;
    if (opt.logprobs > 0) {
      std::vector<float> lp = Impl::log_softmax(work);
      Impl::top_logprobs(lp, opt.logprobs, r.top_ids, r.logprobs);
    }
    return r;
  }
  if (temp != 1.0f)
    for (auto& v : work) v /= temp;

  // Untruncated short-circuit (see the function comment): no top-k, no
  // top-p, no min-p means every id survives, so skip both full-vocab
  // sorts and draw straight from the index-order distribution.
  const bool untruncated =
      (opt.top_k <= 0 || (size_t)opt.top_k >= V) &&
      !(opt.top_p < 1.0f) && !(opt.min_p > 0.0f);
  if (untruncated) {
    const std::vector<float> probs = Impl::softmax(work);
    SampleResult r;
    r.id = Impl::categorical(probs, rng);
    if (opt.logprobs > 0) {
      // `work` is already temperature-scaled: log_softmax(work) is the
      // final full-vocab distribution.
      std::vector<float> lp = Impl::log_softmax(work);
      Impl::top_logprobs(lp, opt.logprobs, r.top_ids, r.logprobs);
    }
    return r;
  }

  // Top-k partial selection first: shrinks the sort below from 128k to k.
  std::vector<int> cand = top_k_keep(work.data(), V, opt.top_k);
  std::vector<float> clogits(cand.size());
  for (size_t i = 0; i < cand.size(); ++i) clogits[i] = work[(size_t)cand[i]];

  std::vector<float> probs = Impl::softmax(clogits);
  // Descending order over the candidate set (by logit == by prob).
  std::vector<int> order(cand.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return clogits[(size_t)a] > clogits[(size_t)b]; });

  // Nucleus filter on the candidate distribution.
  std::vector<int> kept;
  kept.reserve(order.size());
  if (!(opt.top_p < 1.0f)) {
    kept = order;
  } else if (!(opt.top_p > 0.0f)) {
    kept = {order[0]};
  } else {
    double cum = 0.0;
    for (int i : order) {
      kept.push_back(i);
      cum += probs[(size_t)i];
      if (cum >= opt.top_p) break;
    }
  }

  // Min-p filter relative to the kept max.
  if (opt.min_p > 0.0f && kept.size() > 1) {
    float pmax = 0.0f;
    for (int i : kept) pmax = std::max(pmax, probs[(size_t)i]);
    if (!(opt.min_p >= 1.0f)) {
      float thresh = opt.min_p * pmax;
      std::vector<int> narrowed;
      narrowed.reserve(kept.size());
      for (int i : kept)
        if (probs[(size_t)i] >= thresh) narrowed.push_back(i);
      if (!narrowed.empty()) kept = narrowed;
    } else {
      int arg = kept[0];
      for (int i : kept)
        if (probs[(size_t)i] > probs[(size_t)arg]) arg = i;
      kept = {arg};
    }
  }

  std::vector<float> final_w(kept.size());
  for (size_t i = 0; i < kept.size(); ++i) final_w[i] = probs[(size_t)kept[i]];
  int pick = Impl::categorical(final_w, rng);
  SampleResult r;
  r.id = cand[(size_t)kept[(size_t)pick]];
  if (opt.logprobs > 0) {
    // `work` is already temperature-scaled here: log_softmax(work) is the
    // final full-vocab distribution.
    std::vector<float> lp = Impl::log_softmax(work);
    Impl::top_logprobs(lp, opt.logprobs, r.top_ids, r.logprobs);
  }
  return r;
}

inline SampleResult sample_host_logits_result(
    const std::vector<float>& logits, const SampleOptions& opt,
    const std::vector<int>& prior, HostRNG& rng) {
  return sample_host_logits_result(logits.data(), logits.size(), opt, prior,
                                   rng);
}

// Legacy entry point: identical sampling, logprobs discarded. Signature
// UNCHANGED (Model::generate and all existing callers compile untouched;
// neutral new-option defaults reproduce the legacy draws bit-for-bit).
inline int sample_host_logits(const float* logits, size_t V,
                              const SampleOptions& opt,
                              const std::vector<int>& prior, HostRNG& rng) {
  return sample_host_logits_result(logits, V, opt, prior, rng).id;
}

inline int sample_host_logits(const std::vector<float>& logits,
                              const SampleOptions& opt,
                              const std::vector<int>& prior, HostRNG& rng) {
  return sample_host_logits(logits.data(), logits.size(), opt, prior, rng);
}

// Device argmax via parallel_reduce: no 128k host mirror + linear scan.
// Tie-break: smallest index (Kokkos MaxLoc join), matching the old scan.
template <class ExecSpace, class Scalar>
int device_argmax(const Kokkos::View<Scalar*, ExecSpace>& logits) {
  const size_t V = logits.extent(0);
  if (V == 0) throw std::invalid_argument("device_argmax: empty logits");
  using reducer_type = Kokkos::MaxLoc<float, int>;
  typename reducer_type::value_type result;
  result.val = -std::numeric_limits<float>::infinity();
  result.loc = -1;
  reducer_type reducer(result);
  auto lg = logits;
  Kokkos::parallel_reduce(
      "KokkosDNN::device_argmax", Kokkos::RangePolicy<ExecSpace>(0, V),
      KOKKOS_LAMBDA(size_t i,
                    typename reducer_type::value_type& upd) {
        float v = static_cast<float>(lg(i));
        if (v > upd.val) {
          upd.val = v;
          upd.loc = static_cast<int>(i);
        }
      },
      reducer);
  if (result.loc < 0) return 0;
  return result.loc;
}

namespace Impl {

// ---------------------------------------------------------------------------
// Device-side sampling pipeline
// ---------------------------------------------------------------------------
// Logit post-processing used to run on the host: every sampled decode step
// mirrored the whole V-entry logits row to the host (128k * 4 B = 500 KB of
// pageable traffic), converted it element by element, ran an O(V) FP64
// softmax plus a full-vocab sort/categorical walk and allocated two V-sized
// std::vectors per token. Measured on an RTX 3090 at V = 128000 with
// temperature 0.8 / top_p 0.95 that per-token step cost 13 ms; it now costs
// ~0.2 ms and the host sees a single int.
//
// Steps (each one Kokkos kernel, no backend-specific code):
//
//   adjust (penalties / bias) -> scale + max -> argmax             (greedy)
//   adjust -> scale + max -> exp/sum -> [histogram -> threshold]
//          -> [candidate gather for logprobs] -> chunk sums -> serial
//          super-chunk/chunk/element pick -> one int D2H           (sampled)
//
// The draw stays serial (one uniform double decides it), but only inside one
// chunk of kSampleChunk entries: two levels of parallel chunk sums narrow the
// search first, so the serial part is a few hundred adds instead of 128k.
//
// Properties preserved from the host reference (sample_host_logits_result):
//   * penalties apply pre-temperature, in host order: repetition penalty,
//     then additive logit_bias, then frequency/presence (see the folded
//     per-id record below);
//   * greedy draws never touch the RNG; a sampled draw consumes exactly one
//     uniform double, like Impl::categorical;
//   * top-k cuts at the k-th largest value, top-p at the nucleus crossing,
//     min-p at min_p times the maximum probability;
//   * logprobs are reported over the full, pre-truncation distribution.
//
// Documented differences:
//   * truncation boundaries are resolved on a log-probability grid that is
//     refined at the binding boundary (1/128 per bucket at level 1, then
//     1/(128 * kSampleFineBuckets) ~ 6e-5 log units at level 2), so a cut can
//     only differ from the host's by keeping ids whose weights are equal to
//     within ~6e-5 -- in practice only bit-identical ties, which the host
//     splits arbitrarily because its std::sort is not stable and which this
//     pipeline keeps together;
//   * the categorical walk runs in index order. The host does the same for
//     its untruncated short-circuit and sorts for filtered draws, so a
//     filtered draw with a fixed seed yields a different -- identically
//     distributed -- id;
//   * exp() is evaluated in FP32 on the device (host: FP64 exp stored to
//     float), i.e. the weights may differ in the last ulp;
//   * logit_bias and the frequency/presence penalties are pre-summed per id
//     (one rounding instead of two, ~1 ulp on the adjusted logit);
//   * an all-(-inf) row falls back to a uniform draw (the host's
//     untruncated path does the same; its filtered path returns index 0).
//
// The invariant checks live in tests/Test_Sampler.cpp: the device draw must
// stay inside the host's kept set (and reach every id of a deep nucleus), and
// the empirical frequencies must follow the reference distribution.

// One per-id logit adjustment, computed on the host from the token history.
// `pen` is the HF-style repetition penalty (0 = not applied, matching the
// host's penalty > 1 gate) and `add` folds logit_bias plus the
// frequency/presence penalties. Applying `pen` then `add` reproduces the
// host's per-id order.
struct SampleAdjustment {
  int id;
  float pen;
  float add;
};

// Truncation histogram, level 1: one bucket per 1/128 of log-probability
// below the row maximum (4096 buckets => probabilities from e^0 down to
// e^-32; below that the mass is far under FP32 resolution and is clamped).
// A 64-entry coarse histogram (64 fine buckets each) is accumulated alongside
// it so the single-thread threshold search scans 64+64 entries instead of
// 4096: on the GPU a serial global-memory loop is latency bound, and a
// 4096-element scan dominated the whole sampling step.
constexpr int kSampleHistBuckets = 4096;
constexpr float kSampleHistPerLog = 128.0f;
constexpr int kSampleCoarseBuckets = 64;
static_assert(kSampleHistBuckets % kSampleCoarseBuckets == 0,
              "coarse histogram must tile the fine one");
constexpr int kSampleFinePerCoarse =
    kSampleHistBuckets / kSampleCoarseBuckets;
// Truncation histogram, level 2: the single level-1 bucket that holds the
// binding truncation boundary is re-binned into this many sub-buckets, which
// resolves the cut to 1/(kSampleHistPerLog * kSampleFineBuckets) ~ 6e-5 log
// units (i.e. probabilities equal to ~6e-5), finer than the sampling noise.
constexpr int kSampleFineBuckets = 128;
// Draw granularity: two levels of per-chunk partial mass, then one serial
// pass over the selected chunk. Small chunks keep every thread busy
// (V/128 threads at 128k vocab) and bound the serial part to
// kSampleChunk + a few dozen chunk sums, so the draw stays deterministic
// without a latency-bound walk over the vocabulary.
constexpr int kSampleChunk = 128;
constexpr int kSampleChunkSuper = 32;  // chunks per second-level chunk
// Cap on the candidate window gathered for top-N logprobs; overflowing it
// falls back to the host reference, which is exact but mirrors the row.
constexpr int kSampleGatherCap = 4096;

// True when the options can change any logit (penalties need a non-empty
// token history, logit_bias does not).
inline bool sample_has_adjustments(const SampleOptions& opt,
                                   const std::vector<int>& prior) {
  if (!(opt.repetition_penalty > 1.0f) && opt.frequency_penalty == 0.0f &&
      opt.presence_penalty == 0.0f && opt.logit_bias.empty())
    return false;
  return !prior.empty() || !opt.logit_bias.empty();
}

// Scratch + kernels for one sampling site. All V-sized buffers are allocated
// once and reused across tokens (the Session owns one instance); the host
// per-token work is O(#prior tokens + #logit_bias entries), never O(V).
template <class ExecSpace>
class DeviceSampler {
 public:
  using ViewF = Kokkos::View<float*, ExecSpace>;
  using ViewI = Kokkos::View<int*, ExecSpace>;
  using ViewD = Kokkos::View<double*, ExecSpace>;
  using ViewRec = Kokkos::View<SampleAdjustment*, ExecSpace>;
  using HostRec = Kokkos::View<SampleAdjustment*, Kokkos::HostSpace>;
  using HostViewI = Kokkos::View<int*, Kokkos::HostSpace>;
  using HostViewF = Kokkos::View<float*, Kokkos::HostSpace>;
  using ScalarF = Kokkos::View<float, ExecSpace>;
  using ScalarI = Kokkos::View<int, ExecSpace>;
  using ScalarD = Kokkos::View<double, ExecSpace>;
  using ScalarIHost = Kokkos::View<int, Kokkos::HostSpace>;

  explicit DeviceSampler(std::size_t vocab_hint = 0)
      : m_("KokkosDNN::sample::max"),
        z_("KokkosDNN::sample::zsum"),
        logz_("KokkosDNN::sample::logz"),
        dthr_("KokkosDNN::sample::dthr"),
        gdthr_("KokkosDNN::sample::gdthr"),
        minp_("KokkosDNN::sample::minp"),
        mode_("KokkosDNN::sample::mode"),
        bind_("KokkosDNN::sample::bind"),
        above_("KokkosDNN::sample::above"),
        target_("KokkosDNN::sample::target"),
        mbefore_("KokkosDNN::sample::mbefore"),
        out_("KokkosDNN::sample::out"),
        gcount_("KokkosDNN::sample::gcount"),
        out_host_("KokkosDNN::sample::out_host"),
        gcount_host_("KokkosDNN::sample::gcount_host") {
    if (vocab_hint != 0) ensure(vocab_hint);
  }

  // Sample one token. `prior` is the token history the penalties look at
  // (prompt + generated so far). Returns the id plus, when
  // opt.logprobs > 0, the top-N pre-truncation logprobs.
  template <class Scalar>
  SampleResult sample(const Kokkos::View<Scalar*, ExecSpace>& logits,
                      const SampleOptions& opt, const std::vector<int>& prior,
                      HostRNG& rng) {
    const std::size_t V = logits.extent(0);
    if (V == 0)
      throw std::invalid_argument("sample_token_result: empty logits");
    const bool greedy = opt.is_greedy();
    // Pure greedy with neutral penalties/bias and no logprobs: the original
    // device path, untouched (no scratch, no mirror, 8-byte result).
    if (greedy && !sample_has_adjustments(opt, prior) && opt.logprobs <= 0)
      return SampleResult{device_argmax<ExecSpace, Scalar>(logits), {}, {}};

    ensure(V);
    const int nrec = build_adjustments(V, opt, prior);

    // --- adjusted logits (w_) + row max (m_) ------------------------------
    if (nrec > 0) {
      auto wd = w_;
      auto lgv = logits;
      Kokkos::parallel_for(
          "KokkosDNN::sample::copy", Kokkos::RangePolicy<ExecSpace>(0, V),
          KOKKOS_LAMBDA(std::size_t i) {
            wd(i) = static_cast<float>(lgv(i));
          });
      auto rec = rec_;
      Kokkos::parallel_for(
          "KokkosDNN::sample::adjust", Kokkos::RangePolicy<ExecSpace>(0, nrec),
          KOKKOS_LAMBDA(int j) {
            const SampleAdjustment a = rec(j);
            float v = wd(a.id);
            if (a.pen > 0.0f) {
              if (v > 0.0f)
                v /= a.pen;
              else if (v < 0.0f)
                v *= a.pen;
              // exact 0 stays 0 (host convention)
            }
            v += a.add;
            wd(a.id) = v;
          });
    }
    // Temperature scaling is monotone, so folding it in before the greedy
    // argmax keeps the same argmax while reporting the same logprobs.
    const bool scale = (opt.temperature > 0.0f) && (opt.temperature != 1.0f);
    const float temp = opt.temperature;
    {
      auto wd = w_;
      auto lgv = logits;
      auto mxdev = m_;
      const bool copy_in = (nrec == 0);
      Kokkos::parallel_reduce(
          "KokkosDNN::sample::scale", Kokkos::RangePolicy<ExecSpace>(0, V),
          KOKKOS_LAMBDA(std::size_t i, float& acc) {
            float v = copy_in ? static_cast<float>(lgv(i)) : wd(i);
            if (scale) v /= temp;
            wd(i) = v;
            if (v > acc) acc = v;
          },
          Kokkos::Max(mxdev));
    }

    const bool lpo = opt.logprobs > 0;
    // Greedy draws ignore every truncation filter (host contract).
    const bool kact =
        !greedy && (opt.top_k > 1) && (static_cast<std::size_t>(opt.top_k) < V);
    const bool pact = !greedy && (opt.top_p < 1.0f);
    const bool mact = !greedy && (opt.min_p > 0.0f);

    // Greedy without a logprob request is done: one reduce, one int back.
    if (greedy && !lpo)
      return SampleResult{device_argmax<ExecSpace, float>(w_), {}, {}};

    // --- unnormalized weights e_ = exp(w - m) and their sum (z_) ----------
    {
      auto wd = w_;
      auto ed = e_;
      auto mxdev = m_;
      auto zdev = z_;
      Kokkos::parallel_reduce(
          "KokkosDNN::sample::exp", Kokkos::RangePolicy<ExecSpace>(0, V),
          KOKKOS_LAMBDA(std::size_t i, double& acc) {
            const float ev = Kokkos::exp(wd(i) - mxdev());
            ed(i) = ev;
            acc += static_cast<double>(ev);
          },
          Kokkos::Sum(zdev));
    }

    // --- truncation: level-1 bucket histogram ----------------------------
    // Bucket 0 holds the row maximum (d == 0) and the index grows with
    // d = max - w, i.e. with falling probability. Level 1 gives the top-p
    // nucleus mass, the top-k bucket bracket and the logprob candidate
    // window; the bucket that actually binds is then refined one level down.
    const bool hist_needed = kact || pact || lpo;
    const int ntop = static_cast<int>(V) < opt.logprobs
                         ? static_cast<int>(V)
                         : opt.logprobs;
    const bool gather = lpo && ntop <= static_cast<int>(cap_);
    // The mass histogram is only needed for a top-p crossing, the count
    // histogram for top-k and for the logprob candidate window; each atomic
    // costs real time at 128k entries, so only the needed one is built.
    const bool need_mass = pact;
    const bool need_cnt = kact || gather;
    if (hist_needed) {
      if (need_mass) Kokkos::deep_copy(hist_mass_, 0.0f);
      if (need_cnt) Kokkos::deep_copy(hist_cnt_, 0);
      auto wd = w_;
      auto ed = e_;
      auto mxdev = m_;
      auto hm = hist_mass_;
      auto hc = hist_cnt_;
      constexpr int nb = kSampleHistBuckets;
      constexpr float per_log = kSampleHistPerLog;
      Kokkos::parallel_for(
          "KokkosDNN::sample::hist", Kokkos::RangePolicy<ExecSpace>(0, V),
          KOKKOS_LAMBDA(std::size_t i) {
            const float d = (mxdev() - wd(i)) * per_log;
            // Clamp high / NaN into the lowest-probability bucket.
            const int b = (!(d < static_cast<float>(nb - 1)))
                              ? (nb - 1)
                              : (d > 0.0f ? static_cast<int>(d) : 0);
            if (need_mass) Kokkos::atomic_add(&hm(b), ed(i));
            if (need_cnt) Kokkos::atomic_add(&hc(b), 1);
          });
      // Fold the fine histogram into the coarse one so the single-thread
      // threshold search below scans 64 + 64 entries instead of 4096.
      if (need_mass) Kokkos::deep_copy(hist_mass_coarse_, 0.0f);
      if (need_cnt) Kokkos::deep_copy(hist_cnt_coarse_, 0);
      auto hmc = hist_mass_coarse_;
      auto hcc = hist_cnt_coarse_;
      constexpr int nc = kSampleCoarseBuckets;
      constexpr int fpc = kSampleFinePerCoarse;
      Kokkos::parallel_for(
          "KokkosDNN::sample::hist_coarse",
          Kokkos::RangePolicy<ExecSpace>(0, nc), KOKKOS_LAMBDA(int c) {
            double m = 0.0;
            long n = 0;
            for (int f = 0; f < fpc; ++f) {
              if (need_mass) m += static_cast<double>(hm(c * fpc + f));
              if (need_cnt) n += static_cast<long>(hc(c * fpc + f));
            }
            if (need_mass) hmc(c) = static_cast<float>(m);
            if (need_cnt) hcc(c) = static_cast<int>(n);
          });
    }

    // --- truncation: level-1 thresholds (single thread) -------------------
    // Decides which boundary binds: the top-k bracket (count) or the top-p
    // crossing (mass). The other one is then a no-op at level 2.
    {
      auto hm = hist_mass_;
      auto hc = hist_cnt_;
      auto hmc = hist_mass_coarse_;
      auto hcc = hist_cnt_coarse_;
      auto zdev = z_;
      auto mode = mode_;
      auto bind = bind_;
      auto aboved = above_;
      auto mbefore = mbefore_;
      auto targetd = target_;
      auto gdthrd = gdthr_;
      auto logzd = logz_;
      auto gcountd = gcount_;
      const int ktop = opt.top_k;
      const float ptop = opt.top_p;
      constexpr int nb = kSampleHistBuckets;
      constexpr int nc = kSampleCoarseBuckets;
      constexpr int fpc = kSampleFinePerCoarse;
      constexpr float per_log = kSampleHistPerLog;
      Kokkos::parallel_for(
          "KokkosDNN::sample::scan", Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(int) {
            // top-k bracket: bucket holding the k-th largest weight.
            int ck = nb - 1;
            long above = 0;
            if (kact) {
              long acc = 0;
              int cc = 0;
              for (; cc < nc - 1; ++cc) {
                acc += static_cast<long>(hcc(cc));
                if (acc >= ktop) break;
              }
              long acc2 = acc - static_cast<long>(hcc(cc));
              const int fb = cc * fpc;
              int f = 0;
              for (; f < fpc - 1; ++f) {
                acc2 += static_cast<long>(hc(fb + f));
                if (acc2 >= ktop) break;
              }
              ck = fb + f;
              above = acc2 - static_cast<long>(hc(ck));
            }
            // top-p: smallest prefix (from the top) of the surviving mass
            // that reaches top_p, renormalized over the top-k candidate set
            // when top-k is active -- the host renormalizes there too.
            int cp = nb - 1;
            double before = 0.0;
            double z1 = 0.0;
            if (pact) {
              const int ck_coarse = ck / fpc;
              for (int c = 0; c < ck_coarse; ++c)
                z1 += static_cast<double>(hmc(c));
              for (int b = ck_coarse * fpc; b <= ck; ++b)
                z1 += static_cast<double>(hm(b));
              const double target = static_cast<double>(ptop) * z1;
              double cum = 0.0;
              int cc = 0;
              for (; cc < nc - 1; ++cc) {
                const double m = static_cast<double>(hmc(cc));
                if (cum + m >= target) break;
                cum += m;
              }
              const int fb = cc * fpc;
              int f = 0;
              for (; f < fpc - 1; ++f) {
                const double m = static_cast<double>(hm(fb + f));
                if (cum + m >= target) break;
                cum += m;
              }
              cp = fb + f;
              before = cum;
            }
            // Either boundary can bind; a top-p crossing strictly above the
            // top-k bracket wins, otherwise the top-k bracket does.
            int m = 0;
            int bnd = nb - 1;
            if (pact && cp < ck) {
              m = 2;
              bnd = cp;
            } else if (kact) {
              m = 1;
              bnd = ck;
            }
            mode() = m;
            bind() = bnd;
            aboved() = static_cast<int>(above);
            mbefore() = before;
            // Surviving mass: the top-p target is top_p times this.
            targetd() = z1;
            const double sum = zdev();
            logzd() =
                (sum > 0.0) ? Kokkos::log(static_cast<float>(sum)) : 0.0f;
            // top-N logprobs: bucket holding the N-th largest weight, so
            // every id that can be in the reported top-N is gathered.
            if (gather) {
              long acc = 0;
              int b = 0;
              for (; b < nb - 1; ++b) {
                acc += static_cast<long>(hc(b));
                if (acc >= ntop) break;
              }
              gdthrd() = (static_cast<float>(b) + 1.0f) / per_log;
            }
            gcountd() = 0;
          });
    }

    // --- truncation: level-2 refinement of the binding bucket -------------
    // Re-bin the ids that share the binding level-1 bucket (both counts and
    // mass, so either mode can be finished) into kSampleFineBuckets
    // sub-buckets spanning that one bucket. Exact unless several ids hold
    // bit-identical weights, which then stay together -- a tie the host
    // resolves arbitrarily because its std::sort is not stable.
    if (kact || pact) {
      Kokkos::deep_copy(hist_fine_, 0);
      Kokkos::deep_copy(hist_fine_mass_, 0.0f);
      auto wd = w_;
      auto ed = e_;
      auto mxdev = m_;
      auto hf = hist_fine_;
      auto hfm = hist_fine_mass_;
      auto bind = bind_;
      constexpr int nsub = kSampleFineBuckets;
      constexpr int nb = kSampleHistBuckets;
      constexpr float per_log = kSampleHistPerLog;
      Kokkos::parallel_for(
          "KokkosDNN::sample::hist_fine", Kokkos::RangePolicy<ExecSpace>(0, V),
          KOKKOS_LAMBDA(std::size_t i) {
            const float d = (mxdev() - wd(i)) * per_log;
            const int b = (!(d < static_cast<float>(nb - 1)))
                              ? (nb - 1)
                              : (d > 0.0f ? static_cast<int>(d) : 0);
            if (b != bind()) return;
            const float x =
                (d - static_cast<float>(bind())) * static_cast<float>(nsub);
            const int b2 = (!(x < static_cast<float>(nsub - 1)))
                               ? (nsub - 1)
                               : (x > 0.0f ? static_cast<int>(x) : 0);
            Kokkos::atomic_add(&hf(b2), 1);
            Kokkos::atomic_add(&hfm(b2), ed(i));
          });
    }

    // --- truncation: final threshold (single thread) ----------------------
    {
      auto hf = hist_fine_;
      auto hfm = hist_fine_mass_;
      auto mode = mode_;
      auto bind = bind_;
      auto aboved = above_;
      auto mbefore = mbefore_;
      auto targetd = target_;
      auto dthrd = dthr_;
      auto minpd = minp_;
      const int ktop = opt.top_k;
      const float ptop = opt.top_p;
      const float mmin = (opt.min_p < 1.0f) ? opt.min_p : 1.0f;
      // Nothing to truncate: keep the whole row (mode_/bind_ are only
      // meaningful after the level-1 scan ran, i.e. when this is true).
      const bool any_cut = kact || pact;
      constexpr int nsub = kSampleFineBuckets;
      constexpr float per_log = kSampleHistPerLog;
      Kokkos::parallel_for(
          "KokkosDNN::sample::cut", Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(int) {
            const int m = any_cut ? mode() : 0;
            const float base = static_cast<float>(bind());
            float dthr = Kokkos::Experimental::infinity_v<float>;
            if (m == 1) {
              // k-th largest: first sub-bucket whose cumulative count
              // reaches the remaining slots.
              const long need =
                  static_cast<long>(ktop) - static_cast<long>(aboved());
              long acc = 0;
              int j = 0;
              for (; j < nsub - 1; ++j) {
                acc += static_cast<long>(hf(j));
                if (acc >= need) break;
              }
              dthr = (base + (static_cast<float>(j) + 1.0f) /
                                 static_cast<float>(nsub)) /
                     per_log;
            } else if (m == 2) {
              // Nucleus crossing: first sub-bucket whose cumulative mass
              // reaches top_p of the surviving mass.
              const double target =
                  static_cast<double>(ptop) * targetd();
              double cum = mbefore();
              int j = 0;
              for (; j < nsub - 1; ++j) {
                cum += static_cast<double>(hfm(j));
                if (cum >= target) break;
              }
              dthr = (base + (static_cast<float>(j) + 1.0f) /
                                 static_cast<float>(nsub)) /
                     per_log;
            }
            dthrd() = dthr;
            minpd() = mact ? mmin : 0.0f;
          });
    }

    // --- candidate gather for top-N logprobs ------------------------------
    if (gather) {
      auto wd = w_;
      auto mxdev = m_;
      auto gdthrd = gdthr_;
      auto logzd = logz_;
      auto gidd = gid_;
      auto glpd = glp_;
      auto gcountd = gcount_;
      const int gcap = static_cast<int>(cap_);
      Kokkos::parallel_for(
          "KokkosDNN::sample::gather", Kokkos::RangePolicy<ExecSpace>(0, V),
          KOKKOS_LAMBDA(std::size_t i) {
            if ((mxdev() - wd(i)) < gdthrd()) {
              const int pos = Kokkos::atomic_fetch_add(&gcountd(), 1);
              if (pos < gcap) {
                gidd(pos) = static_cast<int>(i);
                glpd(pos) = static_cast<float>(static_cast<double>(wd(i)) -
                                               static_cast<double>(mxdev())) -
                            logzd();
              }
            }
          });
    }

    // --- draw ------------------------------------------------------------
    // One uniform double per sampled token, exactly like Impl::categorical
    // (u = rng * total); the multiply by the kept mass happens on device.
    const double r = greedy ? 0.0 : rng.next_uniform();
    int id = 0;
    if (greedy) {
      id = device_argmax<ExecSpace, float>(w_);
    } else {
      auto ed = e_;
      auto wd = w_;
      auto mxdev = m_;
      auto s1 = chunk_;
      auto s2 = chunk_super_;
      auto outd = out_;
      auto dthrd = dthr_;
      auto minpd = minp_;
      const std::size_t n1 = nchunk_;
      const std::size_t n2 = nsuper_;
      const int blk2 = chunk_per_super_;
      constexpr int csize = kSampleChunk;
      // Level 1: per-chunk partial mass (V / kSampleChunk parallel workers).
      Kokkos::parallel_for(
          "KokkosDNN::sample::chunk", Kokkos::RangePolicy<ExecSpace>(0, n1),
          KOKKOS_LAMBDA(int c) {
            const std::size_t lo = static_cast<std::size_t>(c) * csize;
            const std::size_t hi =
                (lo + csize < V) ? (lo + static_cast<std::size_t>(csize)) : V;
            const float dt = dthrd();
            const float th = minpd();
            const float mx = mxdev();
            double s = 0.0;
            for (std::size_t i = lo; i < hi; ++i) {
              const float ev = ed(i);
              if (ev > 0.0f && ev >= th && (mx - wd(i)) < dt)
                s += static_cast<double>(ev);
            }
            s1(c) = s;
          });
      // Level 2: fold kSampleChunkSuper level-1 chunks per worker.
      Kokkos::parallel_for(
          "KokkosDNN::sample::chunk_super", Kokkos::RangePolicy<ExecSpace>(0, n2),
          KOKKOS_LAMBDA(int c) {
            const std::size_t lo = static_cast<std::size_t>(c) * blk2;
            const std::size_t hi =
                (lo + blk2 < n1) ? (lo + static_cast<std::size_t>(blk2)) : n1;
            double s = 0.0;
            for (std::size_t j = lo; j < hi; ++j) s += s1(j);
            s2(c) = s;
          });
      // Serial pick: total, then the winning super-chunk, chunk and element.
      Kokkos::parallel_for(
          "KokkosDNN::sample::pick", Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(int) {
            const float dt = dthrd();
            const float th = minpd();
            const float mx = mxdev();
            double total = 0.0;
            for (std::size_t c = 0; c < n2; ++c) total += s2(c);
            int pick = -1;
            if (total > 0.0) {
              const double target = r * total;
              double acc = 0.0;
              std::size_t c = 0;
              for (; c < n2; ++c) {
                const double s = s2(c);
                if (target < acc + s) break;
                acc += s;
              }
              if (c >= n2) c = n2 - 1;
              const std::size_t lo1 = c * static_cast<std::size_t>(blk2);
              const std::size_t hi1 =
                  (lo1 + blk2 < n1) ? (lo1 + static_cast<std::size_t>(blk2)) : n1;
              std::size_t k = lo1;
              for (; k < hi1; ++k) {
                const double s = s1(k);
                if (target < acc + s) break;
                acc += s;
              }
              if (k >= hi1) k = hi1 - 1;
              const std::size_t lo = k * static_cast<std::size_t>(csize);
              const std::size_t hi =
                  (lo + csize < V) ? (lo + static_cast<std::size_t>(csize)) : V;
              int last = -1;
              for (std::size_t i = lo; i < hi; ++i) {
                const float ev = ed(i);
                if (ev > 0.0f && ev >= th && (mx - wd(i)) < dt) {
                  last = static_cast<int>(i);
                  acc += static_cast<double>(ev);
                  if (target < acc) {
                    pick = last;
                    break;
                  }
                }
              }
              if (pick < 0) {
                if (last >= 0) {
                  pick = last;  // rounding edge inside the chunk
                } else {
                  // rounding edge across chunks: last kept id in the row
                  for (std::size_t i = V; i-- > 0;) {
                    const float ev = ed(i);
                    if (ev > 0.0f && ev >= th && (mx - wd(i)) < dt) {
                      pick = static_cast<int>(i);
                      break;
                    }
                  }
                }
              }
            } else {
              // Degenerate row (no positive mass): uniform draw, i.e. what
              // the host's uniform softmax fallback would produce.
              const int ui = static_cast<int>(r * static_cast<double>(V));
              pick = (ui < static_cast<int>(V)) ? ui : static_cast<int>(V) - 1;
            }
            outd() = (pick >= 0) ? pick : 0;
          });
      Kokkos::deep_copy(out_host_, out_);
      id = out_host_();
    }

    SampleResult res;
    res.id = id;
    if (lpo) collect_logprobs(V, ntop, gather, res);
    return res;
  }

 private:
  void ensure(std::size_t vocab) {
    if (vocab == 0 || w_.extent(0) == vocab) return;
    w_ = ViewF("KokkosDNN::sample::logits", vocab);
    e_ = ViewF("KokkosDNN::sample::weights", vocab);
    if (hist_mass_.extent(0) !=
        static_cast<std::size_t>(kSampleHistBuckets)) {
      hist_mass_ = ViewF("KokkosDNN::sample::hist_mass", kSampleHistBuckets);
      hist_cnt_ = ViewI("KokkosDNN::sample::hist_cnt", kSampleHistBuckets);
      hist_mass_coarse_ =
          ViewF("KokkosDNN::sample::hist_mass_coarse", kSampleCoarseBuckets);
      hist_cnt_coarse_ =
          ViewI("KokkosDNN::sample::hist_cnt_coarse", kSampleCoarseBuckets);
      hist_fine_ = ViewI("KokkosDNN::sample::hist_fine", kSampleFineBuckets);
      hist_fine_mass_ =
          ViewF("KokkosDNN::sample::hist_fine_mass", kSampleFineBuckets);
    }
    nchunk_ = (vocab + kSampleChunk - 1) / kSampleChunk;
    chunk_per_super_ = (nchunk_ + kSampleChunkSuper - 1) / kSampleChunkSuper;
    nsuper_ = (nchunk_ + chunk_per_super_ - 1) / chunk_per_super_;
    chunk_ = ViewD("KokkosDNN::sample::chunks", nchunk_);
    chunk_super_ = ViewD("KokkosDNN::sample::chunk_super", nsuper_);
    cap_ = vocab < static_cast<std::size_t>(kSampleGatherCap)
               ? vocab
               : static_cast<std::size_t>(kSampleGatherCap);
    gid_ = ViewI("KokkosDNN::sample::gather_ids", cap_);
    glp_ = ViewF("KokkosDNN::sample::gather_lp", cap_);
    gid_host_ = HostViewI("KokkosDNN::sample::gather_ids_host", cap_);
    glp_host_ = HostViewF("KokkosDNN::sample::gather_lp_host", cap_);
    // Host-side dedupe/count scratch, stamped per call so a token never pays
    // an O(V) clear.
    stamp_.assign(vocab, 0u);
    cnt_.assign(vocab, 0);
    add_.assign(vocab, 0.0f);
    epoch_ = 0;
    touched_.clear();
    recs_.clear();
  }

  void ensure_recs(std::size_t n) {
    if (rec_.extent(0) >= n) return;
    rec_ = ViewRec("KokkosDNN::sample::adjustments", n);
    rec_host_ = HostRec("KokkosDNN::sample::adjustments_host", n);
  }

  // Fold the token history + logit_bias into at most one record per id and
  // upload them. O(#prior + #bias) host work.
  int build_adjustments(std::size_t vocab, const SampleOptions& opt,
                        const std::vector<int>& prior) {
    const bool rep = opt.repetition_penalty > 1.0f;
    const bool fpp =
        (opt.frequency_penalty != 0.0f) || (opt.presence_penalty != 0.0f);
    if (!rep && !fpp && opt.logit_bias.empty()) return 0;
    if (prior.empty() && opt.logit_bias.empty()) return 0;
    if (epoch_ == 0xFFFFFFFFu) {
      std::fill(stamp_.begin(), stamp_.end(), 0u);
      epoch_ = 0;
    }
    const std::uint32_t ep = ++epoch_;
    touched_.clear();
    for (int id : prior) {
      if (id < 0 || static_cast<std::size_t>(id) >= vocab) continue;
      const std::size_t k = static_cast<std::size_t>(id);
      if (stamp_[k] != ep) {
        stamp_[k] = ep;
        cnt_[k] = 0;
        add_[k] = 0.0f;
        touched_.push_back(id);
      }
      ++cnt_[k];
    }
    for (const auto& kv : opt.logit_bias) {
      const int id = kv.first;
      if (id < 0 || static_cast<std::size_t>(id) >= vocab) continue;
      const std::size_t k = static_cast<std::size_t>(id);
      if (stamp_[k] != ep) {
        stamp_[k] = ep;
        cnt_[k] = 0;
        add_[k] = 0.0f;
        touched_.push_back(id);
      }
      add_[k] += kv.second;
    }
    const float pen = rep ? opt.repetition_penalty : 0.0f;
    recs_.clear();
    for (int id : touched_) {
      const std::size_t k = static_cast<std::size_t>(id);
      // logit_bias is additive; the frequency/presence penalties subtract
      // (host: logits[id] -= freq * count + presence).
      float a = add_[k];
      if (cnt_[k] > 0)
        a -= opt.frequency_penalty * static_cast<float>(cnt_[k]) +
             opt.presence_penalty;
      if (pen <= 0.0f && a == 0.0f) continue;
      recs_.push_back(SampleAdjustment{id, pen, a});
    }
    if (recs_.empty()) return 0;
    ensure_recs(recs_.size());
    for (std::size_t i = 0; i < recs_.size(); ++i) rec_host_(i) = recs_[i];
    Kokkos::deep_copy(rec_, Kokkos::subview(rec_host_,
                                            std::make_pair(std::size_t(0),
                                                           recs_.size())));
    return static_cast<int>(recs_.size());
  }

  // Top-N logprobs of the full (pre-truncation) distribution. Uses the
  // gathered candidate window when it was wide enough; otherwise replays the
  // host reference over the already-computed logits (exact, rare).
  void collect_logprobs(std::size_t vocab, int ntop, bool gathered,
                        SampleResult& res) {
    if (gathered) {
      Kokkos::deep_copy(gcount_host_, gcount_);
      const int n = gcount_host_();
      if (n <= static_cast<int>(cap_)) {
        if (n > 0) {
          const auto rng_n = std::make_pair(std::size_t(0), std::size_t(n));
          Kokkos::deep_copy(Kokkos::subview(gid_host_, rng_n),
                            Kokkos::subview(gid_, rng_n));
          Kokkos::deep_copy(Kokkos::subview(glp_host_, rng_n),
                            Kokkos::subview(glp_, rng_n));
        }
        // Descending logprob, ties by smaller id: the documented order of
        // Impl::top_logprobs.
        std::vector<int> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
          if (glp_host_(a) != glp_host_(b)) return glp_host_(a) > glp_host_(b);
          return gid_host_(a) < gid_host_(b);
        });
        const std::size_t want = static_cast<std::size_t>(ntop);
        const std::size_t take =
            want < static_cast<std::size_t>(n) ? want
                                               : static_cast<std::size_t>(n);
        res.top_ids.resize(take);
        res.logprobs.resize(take);
        for (std::size_t i = 0; i < take; ++i) {
          const int j = order[i];
          res.top_ids[i] = gid_host_(j);
          res.logprobs[i] = glp_host_(j);
        }
        return;
      }
    }
    // Fallback: host reference over the device-computed (adjusted,
    // temperature-scaled) logits. Only for logprob requests whose candidate
    // window overflowed (mass ties) or N > the window.
    auto w_host = Kokkos::create_mirror_view(w_);
    Kokkos::deep_copy(w_host, w_);
    std::vector<float> fh(vocab);
    for (std::size_t i = 0; i < vocab; ++i) fh[i] = w_host(i);
    std::vector<float> lp = Impl::log_softmax(fh);
    Impl::top_logprobs(lp, ntop, res.top_ids, res.logprobs);
  }

  ViewF w_;  // adjusted, temperature-scaled logits
  ViewF e_;  // unnormalized weights exp(w - max)
  ViewD chunk_;        // per-chunk partial mass (level 1 of the draw)
  ViewD chunk_super_;  // per-super-chunk partial mass (level 2 of the draw)
  ViewF hist_mass_;
  ViewI hist_cnt_;
  ViewF hist_mass_coarse_;
  ViewI hist_cnt_coarse_;
  ViewI hist_fine_;
  ViewF hist_fine_mass_;
  ScalarF m_;      // row max of w_
  ScalarD z_;      // total unnormalized mass
  ScalarF logz_;   // log(z_), for the logprob report
  ScalarF dthr_;   // log-probability cutoff of the truncation threshold
  ScalarF gdthr_;  // log-probability cutoff of the logprob candidate window
  ScalarF minp_;   // min-p probability cutoff (0 = off)
  ScalarI mode_;   // 0 = no cut, 1 = top-k binds, 2 = top-p binds
  ScalarI bind_;   // level-1 bucket that binds (refined at level 2)
  ScalarI above_;  // ids above that bucket (top-k mode)
  ScalarD target_;  // surviving mass (top-p mode), scaled by top_p later
  ScalarD mbefore_;  // mass above that bucket (top-p mode)
  ScalarI out_;
  ScalarI gcount_;
  ScalarIHost out_host_;
  ScalarIHost gcount_host_;
  std::size_t nchunk_ = 0;
  std::size_t nsuper_ = 0;
  int chunk_per_super_ = 1;
  std::size_t cap_ = 0;
  ViewI gid_;
  ViewF glp_;
  HostViewI gid_host_;
  HostViewF glp_host_;
  ViewRec rec_;
  HostRec rec_host_;
  // Host-side per-token scratch (reused, never reallocated per token).
  std::vector<std::uint32_t> stamp_;
  std::vector<int> cnt_;
  std::vector<float> add_;
  std::vector<int> touched_;
  std::vector<SampleAdjustment> recs_;
  std::uint32_t epoch_ = 0;
};
}  // namespace Impl

// Device sampling with top-N logprobs.
//
// Fast path: greedy (temperature <= 0 or top_k == 1) with all penalties
// neutral (repetition_penalty == 1, frequency/presence == 0, empty
// logit_bias) and no logprobs request stays on device (parallel_reduce
// argmax, 8-byte result). Everything else runs the device pipeline described
// above and returns one int (plus the gathered candidate window when
// logprobs are requested). Scalar-agnostic; comparisons, weights and the
// truncation histogram are FP32, the normalizing sums FP64.
//
// This entry point allocates its scratch per call: hot decode loops should
// hold an Impl::DeviceSampler<ExecSpace> (Session and Model::generate do) so
// the V-sized buffers live for the whole request.
template <class ExecSpace, class Scalar>
SampleResult sample_token_result(
    const Kokkos::View<Scalar*, ExecSpace>& logits, const SampleOptions& opt,
    const std::vector<int>& prior, HostRNG& rng) {
  const std::size_t V = logits.extent(0);
  if (V == 0) throw std::invalid_argument("sample_token_result: empty logits");
  if (opt.is_greedy() && opt.logprobs <= 0 &&
      !Impl::sample_has_adjustments(opt, prior))
    return SampleResult{device_argmax<ExecSpace, Scalar>(logits), {}, {}};
  Impl::DeviceSampler<ExecSpace> sampler(V);
  return sampler.sample(logits, opt, prior, rng);
}

// Legacy entry point: identical sampling, logprobs discarded.
template <class ExecSpace, class Scalar>
int sample_token(const Kokkos::View<Scalar*, ExecSpace>& logits,
                 const SampleOptions& opt, const std::vector<int>& prior,
                 HostRNG& rng) {
  return sample_token_result<ExecSpace, Scalar>(logits, opt, prior, rng).id;
}

// Back-compatible greedy sampler: next_token = argmax(logits).
// Now runs on-device (no full-vocab deep_copy + host scan).
template <class ExecSpace, class Scalar>
int sample_greedy(const Kokkos::View<Scalar*, ExecSpace>& logits) {
  return device_argmax<ExecSpace, Scalar>(logits);
}

}  // namespace KokkosDNN

#endif
