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

// Host sampling over float logits (already on host).
//
// Pipeline: repetition penalty (on `prior`) -> greedy short-circuit ->
// temperature scale -> top-k partial selection -> softmax -> top-p ->
// min-p -> renormalize -> categorical draw with `rng`.
//
// Greedy (temperature <= 0 or top_k == 1) returns argmax of the
// penalty-adjusted logits and never touches `rng`.
inline int sample_host_logits(const float* logits, size_t V,
                              const SampleOptions& opt,
                              const std::vector<int>& prior, HostRNG& rng) {
  if (V == 0) throw std::invalid_argument("sample_host_logits: empty logits");
  std::vector<float> work(logits, logits + V);
  apply_repetition_penalty(work, prior, opt.repetition_penalty);

  if (opt.is_greedy()) {
    int best = 0;
    for (size_t v = 1; v < V; ++v)
      if (work[v] > work[best]) best = static_cast<int>(v);
    return best;
  }

  float temp = opt.temperature;
  if (!(temp > 0.0f)) {  // non-greedy flag combo with bad temp: be greedy
    int best = 0;
    for (size_t v = 1; v < V; ++v)
      if (work[v] > work[best]) best = static_cast<int>(v);
    return best;
  }
  if (temp != 1.0f)
    for (auto& v : work) v /= temp;

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
  return cand[(size_t)kept[(size_t)pick]];
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
// Fast path: greedy (temperature <= 0 or top_k == 1) with no repetition
// penalty stays on device (parallel_reduce argmax, single-int result).
// Everything else mirrors the logits once and runs the host pipeline
// (penalty / temperature / top-k partial selection / top-p / min-p /
// categorical). Scalar-agnostic; comparison and softmax are FP32.
template <class ExecSpace, class Scalar>
int sample_token(const Kokkos::View<Scalar*, ExecSpace>& logits,
                 const SampleOptions& opt, const std::vector<int>& prior,
                 HostRNG& rng) {
  const size_t V = logits.extent(0);
  if (V == 0) throw std::invalid_argument("sample_token: empty logits");
  const bool need_host =
      !opt.is_greedy() || (opt.repetition_penalty != 1.0f);
  if (!need_host) return device_argmax<ExecSpace, Scalar>(logits);
  auto h = Kokkos::create_mirror_view(logits);
  Kokkos::deep_copy(h, logits);
  std::vector<float> fh(V);
  for (size_t i = 0; i < V; ++i) fh[i] = static_cast<float>(h(i));
  return sample_host_logits(fh.data(), V, opt, prior, rng);
}

// Back-compatible greedy sampler: next_token = argmax(logits).
// Now runs on-device (no full-vocab deep_copy + host scan).
template <class ExecSpace, class Scalar>
int sample_greedy(const Kokkos::View<Scalar*, ExecSpace>& logits) {
  return device_argmax<ExecSpace, Scalar>(logits);
}

}  // namespace KokkosDNN

#endif
