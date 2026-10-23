// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_LFM25_PREFIX_CACHE_HPP
#define KOKKOSDNN_LFM25_PREFIX_CACHE_HPP

// Single-entry exact-prefix KV cache over Model (models/lfm2_5).
//
// Agentic loops re-prefill the whole message history every turn: turn N+1's
// prompt starts with turn N's history plus new assistant/tool messages, so a
// full reset + forward(prompt) recomputes the shared prefix from scratch.
// PrefixCache keeps one resident snapshot of (KV prefix + conv state +
// last-token logits) keyed by its token sequence. A new prompt whose first
// L tokens exactly match the cached entry resumes from position L and
// forwards only the suffix, which is bit-identical to a full prefill: the
// chunked Model::forward() chains through the persistent conv state, and
// attention KV for [0, L) is restored verbatim.
//
// Why exact-prefix only (no truncation of longer entries): attention KV
// truncates freely, but the short-conv state at position L is the last K-1
// bx values before L and cannot be derived from the end-of-prompt state.
// Caching the content prefix (everything before the trailing generation
// prompt) as the key makes consecutive agentic turns hit exactly: the next
// turn's content starts with the previous turn's content, so the cached
// state length is the longest common prefix with no rewind needed.
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
// allocation and KV copies in StoreSpace (default: the execution space's
// device memory). HostSpace storage trades H2D/D2H copies on every
// restore/save for VRAM savings: one full-window entry at
// max_seq_len=131072 is ~2.5 GiB. Cross-space copies ride on
// Kokkos::deep_copy, so the restore/save paths are backend-portable; on a
// host-only build StoreSpace collapses to the model space and no second
// copy ever exists.
//
// Threads/ownership: not thread-safe; the server guards its single Session
// (and hence this cache) with the single-flight mutex.

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include "KokkosDNN_LFM25Model.hpp"

namespace KokkosDNN {
namespace LFM25 {

// Storage-agnostic half of the prefix cache: token keys in LRU order
// (front = most-recently-used) plus longest-prefix lookup. The view-owning
// subclass keeps its snapshots parallel to these keys.
class PrefixCacheCore {
 public:
  explicit PrefixCacheCore(size_t capacity = 1) : capacity_(capacity) {}

  size_t size() const { return keys_.size(); }
  size_t capacity() const { return capacity_; }
  bool empty() const { return keys_.empty(); }
  void clear() { keys_.clear(); }

  // Longest key that is an exact prefix of `ids` with length <= `max_len`.
  // Returns the key index, or -1 on miss; `out_len` is the reused length
  // (0 on miss). Restricting to `max_len` (the caller's content length)
  // keeps hits rewind-free: the restored state sits at or before the
  // content boundary, so the caller snapshots the content state by
  // forwarding forward only.
  int find(const std::vector<int>& ids, size_t max_len,
           size_t& out_len) const {
    int best = -1;
    size_t best_len = 0;
    for (size_t i = 0; i < keys_.size(); ++i) {
      const size_t n = keys_[i].size();
      if (n == 0 || n > max_len || n > ids.size() || n <= best_len) continue;
      bool match = true;
      for (size_t j = 0; j < n; ++j) {
        if (keys_[i][j] != ids[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        best = static_cast<int>(i);
        best_len = n;
      }
    }
    out_len = best_len;
    return best;
  }

  // Mark key `idx` most-recently-used.
  void promote(size_t idx) {
    if (idx == 0 || idx >= keys_.size()) return;
    std::vector<int> k = std::move(keys_[idx]);
    keys_.erase(keys_.begin() + static_cast<long>(idx));
    keys_.insert(keys_.begin(), std::move(k));
  }

 protected:
  // Push a new key front. The caller keeps its snapshots parallel (same
  // insert/erase positions) and owns capacity enforcement for both.
  void push_front_key(std::vector<int> key) {
    keys_.insert(keys_.begin(), std::move(key));
  }

  // Drop the least-recently-used key (the caller drops its snapshot too).
  void pop_back_key() {
    if (!keys_.empty()) keys_.pop_back();
  }

  const std::vector<int>& key(size_t idx) const { return keys_.at(idx); }

 private:
  std::vector<std::vector<int>> keys_;  // front = most-recently-used
  size_t capacity_;
};

template <class ExecSpace, class Scalar = DefaultScalar,
          class StoreSpace = typename ExecSpace::memory_space>
class PrefixCache : public PrefixCacheCore {
 public:
  using ModelT = Model<ExecSpace, Scalar>;
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, StoreSpace>;
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, StoreSpace>;
  using View1 = Kokkos::View<Scalar*, StoreSpace>;

  // Resident snapshot at one content prefix: the model state after
  // forwarding exactly `tokens` (KV rows [0, len), per-layer conv states,
  // and last-token logits for the first step()). `tokens` mirrors the
  // base-class key at the same index.
  struct Entry {
    std::vector<int> tokens;
    std::vector<View3> k;     // per-layer K cache prefix (empty: conv layer)
    std::vector<View3> v;     // per-layer V cache prefix (empty: conv layer)
    std::vector<MatR> conv;   // per-layer conv state (empty: attn layer)
    View1 logits;             // last-token logits at tokens.size()
  };

  explicit PrefixCache(size_t capacity = 1) : PrefixCacheCore(capacity) {}

  void clear() {
    entries_.clear();
    PrefixCacheCore::clear();
  }

  // Restore entry `idx` into `model` (KV prefix, conv states, seq_len).
  // The entry's logits are NOT moved (the caller forwards the suffix and
  // takes fresh logits, or copies entry logits itself on exact hits).
  void restore(size_t idx, ModelT& model) const {
    const Entry& e = entries_.at(idx);
    const size_t n = e.tokens.size();
    const size_t nl = static_cast<size_t>(model.cfg.n_layers);
    for (size_t l = 0; l < nl; ++l) {
      if (model.cfg.is_attention(static_cast<int>(l))) {
        auto dst_k = Kokkos::subview(
            model.k_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_k, e.k[l]);
        auto dst_v = Kokkos::subview(
            model.v_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_v, e.v[l]);
      } else if (e.conv[l].extent(0) != 0) {
        Kokkos::deep_copy(model.conv_state_[l], e.conv[l]);
      }
    }
    model.seq_len = n;
  }

  // Copy the entry's stashed last-token logits out (exact-hit path, where
  // no suffix forward runs to produce fresh logits). The destination may
  // live in any memory space; Kokkos::deep_copy bridges it.
  template <class DstView>
  void restore_logits(size_t idx, DstView& dst) const {
    Kokkos::deep_copy(dst, entries_.at(idx).logits);
  }

  // Logits width stashed under entry `idx` (for sizing `dst` above).
  size_t entry_logits_size(size_t idx) const {
    return entries_.at(idx).logits.extent(0);
  }

  // Mark entry `idx` most-recently-used (key and snapshot together).
  void promote(size_t idx) {
    if (idx == 0 || idx >= entries_.size()) return;
    Entry e = std::move(entries_[idx]);
    entries_.erase(entries_.begin() + static_cast<long>(idx));
    entries_.insert(entries_.begin(), std::move(e));
    PrefixCacheCore::promote(idx);
  }

  // Snapshot the model's current state under `tokens` (requires
  // model.seq_len == tokens.size()). Exact duplicates just promote; new
  // keys push front and evict least-recently-used beyond capacity.
  // `logits` is the last-token logits view returned by the forward that
  // reached this state (copied; the source aliases model scratch and may
  // live in any memory space).
  template <class LogitsView>
  void save(const std::vector<int>& tokens, const ModelT& model,
            const LogitsView& logits) {
    if (capacity() == 0 || tokens.empty()) return;
    if (model.seq_len != tokens.size()) return;  // misuse guard: no partial
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].tokens == tokens) {
        promote(i);
        return;
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
      if (model.cfg.is_attention(static_cast<int>(l))) {
        e.k[l] = View3("KokkosDNN::LFM25::PrefixCache::k", n, nkv, hd);
        e.v[l] = View3("KokkosDNN::LFM25::PrefixCache::v", n, nkv, hd);
        auto src_k = Kokkos::subview(
            model.k_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(e.k[l], src_k);
        auto src_v = Kokkos::subview(
            model.v_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(e.v[l], src_v);
      } else if (model.conv_state_[l].extent(0) != 0) {
        const size_t d0 = model.conv_state_[l].extent(0);
        const size_t d1 = model.conv_state_[l].extent(1);
        e.conv[l] = MatR("KokkosDNN::LFM25::PrefixCache::conv", d0, d1);
        Kokkos::deep_copy(e.conv[l], model.conv_state_[l]);
      }
    }
    e.logits = View1("KokkosDNN::LFM25::PrefixCache::logits",
                      logits.extent(0));
    Kokkos::deep_copy(e.logits, logits);
    entries_.insert(entries_.begin(), std::move(e));
    push_front_key(tokens);
    while (entries_.size() > capacity()) {
      entries_.pop_back();
      pop_back_key();
    }
  }

 private:
  // Snapshots parallel to the base-class keys (same order, same size).
  std::vector<Entry> entries_;
};

}  // namespace LFM25
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_LFM25_PREFIX_CACHE_HPP
