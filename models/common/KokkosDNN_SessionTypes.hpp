// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SESSION_TYPES_HPP
#define KOKKOSDNN_SESSION_TYPES_HPP

// Shared Session vocabulary (models/common).
//
// LFM25::SessionOptions/StepOutcome/CachedPrefillOutcome/ISession and
// their Glimmer counterparts were verbatim copies. Both now alias these
// Common types, so the server and tests can mix model backends behind one
// vocabulary. Per-arch Session classes keep their own prefill/step math
// (KV vs KV+conv state, merged vs split content+suffix forwards) and only
// differ in how they resolve defaults (eos field name, default stop
// strings); the turn state machine shape is shared.

#include <cstdint>
#include <string>
#include <vector>

#include "KokkosDNN_Sampler.hpp"
#include "KokkosDNN_Usage.hpp"

namespace KokkosDNN {
namespace Common {

struct SessionOptions {
  ::KokkosDNN::SampleOptions sampling;
  std::vector<int> stop_ids;              // empty -> per-arch default eos set
  std::vector<std::string> stop_strings;  // empty -> per-arch defaults
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

}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_SESSION_TYPES_HPP
