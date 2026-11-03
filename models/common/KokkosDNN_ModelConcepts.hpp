// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MODEL_CONCEPTS_HPP
#define KOKKOSDNN_MODEL_CONCEPTS_HPP

// Generic Config/Weights/Session/PrefixCache concepts for decoder-only
// models (models/common).
//
// These concepts describe the structural contract the shared generate()
// driver (KokkosDNN_Generate.hpp) and the per-arch Session/PrefixCache
// types rely on. They are intentionally narrow: per-arch Configs keep
// their own field names (eos vs eos_token, default stop sets) and
// per-arch Models keep their own forward() math; only the pieces the
// shared driver touches are constrained.
//
// A type models DecoderModel when it exposes:
//   - nested ViewInt (Kokkos::View<int*, ExecSpace> style token views),
//   - a `cfg` member whose type models DecoderConfig,
//   - a mutable `seq_len` position counter,
//   - `reset()` and `forward(ViewInt) -> rank-1 logits view`.
// PrefixCacheCoreConcept covers the storage-agnostic longest-prefix lookup
// shared via KokkosDNN_PrefixCacheCore.hpp; DecodingSessionConcept covers
// the server-facing turn API shared via KokkosDNN_SessionTypes.hpp.

#include <concepts>
#include <cstddef>
#include <string>
#include <vector>

namespace KokkosDNN {
namespace Common {

// Any config carrying the runtime context cap. Default stop ids stay
// per-arch ((field names differ: eos vs eos_token vs default_stop_ids),
// so they are deliberately NOT part of this concept.
template <class C>
concept DecoderConfig = requires(const C& c) {
  { c.max_seq_len } -> std::convertible_to<int>;
};

// Weights are opaque to the shared driver (per-arch block layouts differ).
// The concept only pins down copyability so Sessions/Models can hold them
// by value.
template <class W>
concept DecoderWeights = std::copyable<W>;

// Storage-agnostic longest-prefix lookup (see PrefixCacheCore).
template <class P>
concept PrefixCacheCoreConcept = requires(P& p, const P& cp,
                                          const std::vector<int>& ids,
                                          size_t max_len, size_t idx) {
  { cp.size() } -> std::convertible_to<size_t>;
  { cp.capacity() } -> std::convertible_to<size_t>;
  { cp.empty() } -> std::convertible_to<bool>;
  { p.clear() } -> std::same_as<void>;
  { cp.find(ids, max_len, max_len) } -> std::convertible_to<int>;
  { p.promote(idx) } -> std::same_as<void>;
};

// Minimal decoder-model surface used by Common::generate_impl().
template <class M>
concept DecoderModel = requires(M& m, const std::vector<int>& prompt) {
  typename M::ViewInt;
  requires DecoderConfig<std::decay_t<decltype(m.cfg)>>;
  { m.seq_len } -> std::convertible_to<size_t>;
  { m.reset() } -> std::same_as<void>;
};

// Server-facing turn API (see SessionTypes::ISession).
template <class S>
concept DecodingSession = requires(S& s, const std::vector<int>& ids) {
  { s.prefill(ids) } -> std::same_as<void>;
  { s.step() };
  { s.flush() } -> std::convertible_to<std::string>;
  { s.reset() } -> std::same_as<void>;
  { s.finished() } -> std::convertible_to<bool>;
};

}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_MODEL_CONCEPTS_HPP
