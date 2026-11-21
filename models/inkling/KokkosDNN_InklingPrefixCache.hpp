// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLING_PREFIX_CACHE_HPP
#define KOKKOSDNN_INKLING_PREFIX_CACHE_HPP

// Single-entry exact-prefix KV cache over Model (models/inkling).
//
// Agentic loops re-prefill the whole message history every turn: turn N+1's
// prompt starts with turn N's history plus new assistant/tool messages, so a
// full reset + forward(prompt) recomputes the shared prefix from scratch.
// PrefixCache keeps one resident snapshot of (KV prefix + conv history +
// last-token logits) keyed by its token sequence. A new prompt whose first
// L tokens exactly match the cached entry resumes from position L and
// forwards only the suffix, which is bit-identical to a full prefill: the
// chunked Model::forward() writes KV at [cpos, cpos+C) before reading the
// prefix [0, cpos+C) and carries stateful conv windows across chunks, so
// resuming mid-sequence reproduces single-shot prefill token-for-token.
// Unlike Glimmer there IS short-conv state to carry (4 convs per layer per
// pass slot, including the per-pass loop slots), so the snapshot covers
// the per-layer per-pass KV rows [0, len) plus every conv history window
// plus the stashed last-token logits.
//
// Policy: single entry (see Session::kPrefixCacheCapacity). Lookup hits
// when the cached entry is an exact prefix of the new prompt (and no
// longer than the caller's content length, so restores stay rewind-free).
// A miss replaces the entry. Sampling options/RNG never affect the
// snapshot (prefill is deterministic given tokens), so reuse is safe across
// requests with different sampling/stop/seed policies.
//
// Layout: PrefixCacheCore is the storage-agnostic half (token keys, LRU
// order, longest-prefix lookup). The PrefixCache subclass adds only view
// allocation and KV/conv copies in StoreSpace (default: the execution
// space's device memory). Cross-space copies ride on Kokkos::deep_copy, so
// the restore/save paths are backend-portable; on a host-only build
// StoreSpace collapses to the model space and no second copy ever exists.
//
// Threads/ownership: not thread-safe; the server guards its single Session
// (and hence this cache) with the single-flight mutex.

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "KokkosDNN_PrefixCacheCore.hpp"
#include "KokkosDNN_InklingModel.hpp"

namespace KokkosDNN {
namespace Inkling {

// Shared longest-prefix lookup (models/common): the per-arch
// PrefixCacheCore classes were verbatim copies. This alias keeps every
// existing Inkling::PrefixCacheCore name working.
using PrefixCacheCore = ::KokkosDNN::Common::PrefixCacheCore;

template <class ExecSpace, class Scalar = Kokkos::Experimental::bhalf_t,
          class StoreSpace = typename ExecSpace::memory_space,
          class ConfigT = Config,
          class WeightsT = Weights<ExecSpace, Scalar>>
class PrefixCache : public PrefixCacheCore {
 public:
  using ModelT = Model<ExecSpace, Scalar, ConfigT, WeightsT>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, StoreSpace>;
  using MatS = Kokkos::View<Scalar**, Kokkos::LayoutRight, StoreSpace>;
  using View1 = Kokkos::View<Scalar*, StoreSpace>;

  // Resident snapshot at one content prefix: the model state after
  // forwarding exactly `tokens` (`tokens` mirrors the base-class key at the
  // same index). Stores the per-layer per-pass KV rows [0, len), every
  // conv history window, plus the stashed last-token logits for the first
  // step().
  struct Entry {
    std::vector<int> tokens;
    // False for entries snapshotted after forwarding PAST the key (the
    // merged content+suffix prefill in Session::prefill_cached): the KV
    // rows are complete, so the entry is a valid resume point, but no
    // last-token logits were ever produced at this boundary. Such entries
    // can only serve suffixes (content < full), never an exact hit that
    // needs logits with no forward.
    bool has_logits = true;
    // [l][p] KV rows.
    std::vector<std::vector<View3>> k;
    std::vector<std::vector<View3>> v;
    // [l][p] conv history windows (k, v, pa, pm).
    struct ConvSnap {
      MatS k, v, pa, pm;
    };
    std::vector<std::vector<ConvSnap>> conv;
    View1 logits;  // last-token logits at tokens.size()
  };

  explicit PrefixCache(size_t capacity = 1) : PrefixCacheCore(capacity) {}

  void clear() {
    entries_.clear();
    PrefixCacheCore::clear();
  }

  // Restore entry `idx` into `model` (KV prefix, conv history, seq_len).
  // The entry's logits are NOT moved (the caller forwards the suffix and
  // takes fresh logits, or copies entry logits itself on exact hits).
  void restore(size_t idx, ModelT& model) const {
    const Entry& e = entries_.at(idx);
    const size_t n = e.tokens.size();
    const size_t nl = static_cast<size_t>(model.cfg.n_layers);
    if (e.k.size() != nl)
      throw std::runtime_error(
          "PrefixCache::restore: entry layer count does not match model");
    for (size_t l = 0; l < nl; ++l) {
      const size_t P = model.k_cache[l].size();
      if (e.k[l].size() != P)
        throw std::runtime_error(
            "PrefixCache::restore: entry pass count does not match model");
      for (size_t p = 0; p < P; ++p) {
        auto dst_k = Kokkos::subview(
            model.k_cache[l][p],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_k, e.k[l][p]);
        auto dst_v = Kokkos::subview(
            model.v_cache[l][p],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_v, e.v[l][p]);
        Kokkos::deep_copy(model.conv_[l][p].k, e.conv[l][p].k);
        Kokkos::deep_copy(model.conv_[l][p].v, e.conv[l][p].v);
        Kokkos::deep_copy(model.conv_[l][p].pa, e.conv[l][p].pa);
        Kokkos::deep_copy(model.conv_[l][p].pm, e.conv[l][p].pm);
      }
    }
    model.seq_len = n;
  }

  // Copy the entry's stashed last-token logits out (exact-hit path, where
  // no suffix forward runs to produce fresh logits). The destination may
  // live in any memory space; Kokkos::deep_copy bridges it. Only valid
  // when entry_has_logits(idx) is true.
  template <class DstView>
  void restore_logits(size_t idx, DstView& dst) const {
    Kokkos::deep_copy(dst, entries_.at(idx).logits);
  }

  // Logits width stashed under entry `idx` (for sizing `dst` above).
  size_t entry_logits_size(size_t idx) const {
    return entries_.at(idx).logits.extent(0);
  }

  // Whether entry `idx` can serve an exact hit without a forward (it was
  // snapshotted by a forward that ended exactly at the key boundary).
  bool entry_has_logits(size_t idx) const {
    return entries_.at(idx).has_logits;
  }

  // Mark entry `idx` most-recently-used (key and snapshot together).
  void promote(size_t idx) {
    if (idx == 0 || idx >= entries_.size()) return;
    Entry e = std::move(entries_[idx]);
    entries_.erase(entries_.begin() + static_cast<long>(idx));
    entries_.insert(entries_.begin(), std::move(e));
    PrefixCacheCore::promote(idx);
  }

  // Snapshot the model's current KV/conv state under `tokens`: rows [0, n)
  // must be fully written, i.e. model.seq_len >= tokens.size(). A snapshot
  // taken after forwarding past the key (the merged content+suffix prefill
  // in Session::prefill_cached) is valid for suffix resumes but stores no
  // logits; use the two-argument overload for that case. Exact duplicates
  // just promote; new keys push front and evict least-recently-used beyond
  // capacity.
  // `logits` is the last-token logits view returned by the forward that
  // reached this state (copied; the source aliases model scratch and may
  // live in any memory space).
  template <class LogitsView>
  void save(const std::vector<int>& tokens, const ModelT& model,
            const LogitsView& logits) {
    const int idx = save_state(tokens, model);
    if (idx < 0) return;
    Entry& e = entries_[static_cast<size_t>(idx)];
    e.has_logits = false;
    if (e.logits.extent(0) != logits.extent(0))
      e.logits = View1("KokkosDNN::Inkling::PrefixCache::logits",
                       logits.extent(0));
    Kokkos::deep_copy(e.logits, logits);
    e.has_logits = true;
  }

  // Snapshot without last-token logits (see above). The entry resumes
  // suffixes exactly like a logged snapshot; only an exact hit that would
  // otherwise skip the forward needs logits, and that path is unreachable
  // here because this overload is used only when the caller forwards a
  // suffix past the key.
  void save(const std::vector<int>& tokens, const ModelT& model) {
    (void)save_state(tokens, model);
  }

 private:
  // Shared snapshot/promote half of save(): returns the entry index (new
  // entry at 0, or an existing promoted one), or -1 when nothing was
  // stored. Callers fill in logits.
  int save_state(const std::vector<int>& tokens, const ModelT& model) {
    if (capacity() == 0 || tokens.empty()) return -1;
    // Every row in [0, n) must already be written. seq_len may be larger
    // when the caller forwarded a suffix past the snapshot boundary; those
    // later rows are simply not part of this entry.
    if (model.seq_len < tokens.size()) return -1;
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].tokens == tokens) {
        promote(i);
        return 0;
      }
    }
    Entry e;
    e.tokens = tokens;
    const size_t n = tokens.size();
    const size_t nl = static_cast<size_t>(model.cfg.n_layers);
    const size_t nkv = static_cast<size_t>(model.cfg.n_kv_heads);
    const size_t hd = static_cast<size_t>(model.cfg.head_dim);
    e.k.resize(nl);
    e.v.resize(nl);
    e.conv.resize(nl);
    for (size_t l = 0; l < nl; ++l) {
      const size_t P = model.k_cache[l].size();
      e.k[l].resize(P);
      e.v[l].resize(P);
      e.conv[l].resize(P);
      for (size_t p = 0; p < P; ++p) {
        e.k[l][p] = View3("KokkosDNN::Inkling::PrefixCache::k", n, nkv, hd);
        e.v[l][p] = View3("KokkosDNN::Inkling::PrefixCache::v", n, nkv, hd);
        e.conv[l][p].k =
            MatS("KokkosDNN::Inkling::PrefixCache::ck",
                 model.conv_[l][p].k.extent(0), model.conv_[l][p].k.extent(1));
        e.conv[l][p].v =
            MatS("KokkosDNN::Inkling::PrefixCache::cv",
                 model.conv_[l][p].v.extent(0), model.conv_[l][p].v.extent(1));
        e.conv[l][p].pa =
            MatS("KokkosDNN::Inkling::PrefixCache::cpa",
                 model.conv_[l][p].pa.extent(0),
                 model.conv_[l][p].pa.extent(1));
        e.conv[l][p].pm =
            MatS("KokkosDNN::Inkling::PrefixCache::cpm",
                 model.conv_[l][p].pm.extent(0),
                 model.conv_[l][p].pm.extent(1));
        auto src_k = Kokkos::subview(model.k_cache[l][p],
                                     std::make_pair(size_t(0), n), Kokkos::ALL,
                                     Kokkos::ALL);
        Kokkos::deep_copy(e.k[l][p], src_k);
        auto src_v = Kokkos::subview(model.v_cache[l][p],
                                     std::make_pair(size_t(0), n), Kokkos::ALL,
                                     Kokkos::ALL);
        Kokkos::deep_copy(e.v[l][p], src_v);
        Kokkos::deep_copy(e.conv[l][p].k, model.conv_[l][p].k);
        Kokkos::deep_copy(e.conv[l][p].v, model.conv_[l][p].v);
        Kokkos::deep_copy(e.conv[l][p].pa, model.conv_[l][p].pa);
        Kokkos::deep_copy(e.conv[l][p].pm, model.conv_[l][p].pm);
      }
    }
    entries_.insert(entries_.begin(), std::move(e));
    push_front_key(tokens);
    while (entries_.size() > capacity()) {
      entries_.pop_back();
      pop_back_key();
    }
    return 0;
  }

  // Snapshots parallel to the base-class keys (same order, same size).
  std::vector<Entry> entries_;
};

}  // namespace Inkling
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_INKLING_PREFIX_CACHE_HPP
