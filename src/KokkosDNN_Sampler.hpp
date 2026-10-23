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

// Device sampling entry point.
//
// Fast path: greedy (temperature <= 0 or top_k == 1) with all penalties
// neutral (repetition_penalty == 1, frequency/presence == 0, empty
// logit_bias) and no logprobs request stays on device (parallel_reduce
// argmax, single-int result). Everything else mirrors the logits once
// and runs the host pipeline (penalties / bias / temperature / top-k
// partial selection / top-p / min-p / categorical, plus optional
// top-N logprobs). Scalar-agnostic; comparison and softmax are FP32.
template <class ExecSpace, class Scalar>
int sample_token(const Kokkos::View<Scalar*, ExecSpace>& logits,
                 const SampleOptions& opt, const std::vector<int>& prior,
                 HostRNG& rng) {
  const size_t V = logits.extent(0);
  if (V == 0) throw std::invalid_argument("sample_token: empty logits");
  const bool need_host =
      !opt.is_greedy() || (opt.repetition_penalty != 1.0f) ||
      (opt.frequency_penalty != 0.0f) || (opt.presence_penalty != 0.0f) ||
      (!opt.logit_bias.empty()) || (opt.logprobs > 0);
  if (!need_host) return device_argmax<ExecSpace, Scalar>(logits);
  auto h = Kokkos::create_mirror_view(logits);
  Kokkos::deep_copy(h, logits);
  std::vector<float> fh(V);
  for (size_t i = 0; i < V; ++i) fh[i] = static_cast<float>(h(i));
  return sample_host_logits(fh.data(), V, opt, prior, rng);
}

// Device sampling with top-N logprobs. Same routing as sample_token():
// neutral-options greedy stays on device (empty logprobs vectors, since
// the fast path requires opt.logprobs <= 0); everything else mirrors once
// and returns the full SampleResult (id + top logprobs). Signature
// mirrors sample_token() with the SampleResult return.
template <class ExecSpace, class Scalar>
SampleResult sample_token_result(
    const Kokkos::View<Scalar*, ExecSpace>& logits, const SampleOptions& opt,
    const std::vector<int>& prior, HostRNG& rng) {
  const size_t V = logits.extent(0);
  if (V == 0) throw std::invalid_argument("sample_token_result: empty logits");
  const bool need_host =
      !opt.is_greedy() || (opt.repetition_penalty != 1.0f) ||
      (opt.frequency_penalty != 0.0f) || (opt.presence_penalty != 0.0f) ||
      (!opt.logit_bias.empty()) || (opt.logprobs > 0);
  if (!need_host && opt.logprobs <= 0)
    return SampleResult{device_argmax<ExecSpace, Scalar>(logits), {}, {}};
  auto h = Kokkos::create_mirror_view(logits);
  Kokkos::deep_copy(h, logits);
  std::vector<float> fh(V);
  for (size_t i = 0; i < V; ++i) fh[i] = static_cast<float>(h(i));
  return sample_host_logits_result(fh.data(), V, opt, prior, rng);
}

// Back-compatible greedy sampler: next_token = argmax(logits).
// Now runs on-device (no full-vocab deep_copy + host scan).
template <class ExecSpace, class Scalar>
int sample_greedy(const Kokkos::View<Scalar*, ExecSpace>& logits) {
  return device_argmax<ExecSpace, Scalar>(logits);
}

}  // namespace KokkosDNN

#endif
