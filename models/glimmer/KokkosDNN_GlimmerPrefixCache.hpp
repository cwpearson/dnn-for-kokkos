// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GLIMMER_PREFIX_CACHE_HPP
#define KOKKOSDNN_GLIMMER_PREFIX_CACHE_HPP

// Single-entry exact-prefix KV cache over Model (models/glimmer).
//
// Agentic loops re-prefill the whole message history every turn: turn N+1's
// prompt starts with turn N's history plus new assistant/tool messages, so a
// full reset + forward(prompt) recomputes the shared prefix from scratch.
// PrefixCache keeps one resident snapshot of (KV prefix + last-token
// logits) keyed by its token sequence. A new prompt whose first L tokens
// exactly match the cached entry resumes from position L and forwards only
// the suffix, which is bit-identical to a full prefill: the chunked
// Model::forward() reads previously written KV (earlier chunks) plus its
// own prefix through the causal mask, so resuming mid-sequence reproduces
// single-shot prefill token-for-token. Unlike LFM25 there is no short-conv
// state to carry (Glimmer is dense attention only), so the snapshot is just
// the per-layer KV rows [0, len) — dense FP rows, or the symmetric-int8
// pools plus FP32 scale rows when Config::kv_int8 is set — plus the
// stashed last-token logits.
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
// device memory). Cross-space copies ride on Kokkos::deep_copy, so the
// restore/save paths are backend-portable; on a host-only build StoreSpace
// collapses to the model space and no second copy ever exists.
//
// Threads/ownership: not thread-safe; the server guards its single Session
// (and hence this cache) with the single-flight mutex.

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "KokkosDNN_GlimmerModel.hpp"

namespace KokkosDNN {
namespace Glimmer {

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
  using View3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, StoreSpace>;
  using View1 = Kokkos::View<Scalar*, StoreSpace>;
  // int8 KV snapshot views (quantized mode): per-token per-head symmetric
  // int8 values plus FP32 scales, mirroring Model::kq/vq/ks/vs pools.
  using ViewI8 = Kokkos::View<int8_t***, Kokkos::LayoutRight, StoreSpace>;
  using ViewScale = Kokkos::View<float**, Kokkos::LayoutRight, StoreSpace>;

  // Resident snapshot at one content prefix: the model state after
  // forwarding exactly `tokens` (`tokens` mirrors the base-class key at the
  // same index). Dense mode stores the per-layer KV rows [0, len) plus the
  // stashed last-token logits for the first step(); int8 mode
  // (Config::kv_int8) stores the quantized pools plus the per-token
  // per-head FP32 scale rows instead — the dense views stay empty and vice
  // versa (the two modes never coexist, matching Model::kv_alloc_fresh).
  struct Entry {
    std::vector<int> tokens;
    bool quant = false;  // snapshot mode: true = int8 pools + scales
    // False for entries snapshotted after forwarding PAST the key (the
    // merged content+suffix prefill in Session::prefill_cached): the KV
    // rows are complete, so the entry is a valid resume point, but no
    // last-token logits were ever produced at this boundary. Such entries
    // can only serve suffixes (content < full), never an exact hit that
    // needs logits with no forward.
    bool has_logits = true;
    std::vector<View3> k;  // per-layer dense K cache prefix
    std::vector<View3> v;  // per-layer dense V cache prefix
    std::vector<ViewI8> kq;  // per-layer int8 K pool prefix
    std::vector<ViewI8> vq;  // per-layer int8 V pool prefix
    std::vector<ViewScale> ks;  // per-layer K scale prefix
    std::vector<ViewScale> vs;  // per-layer V scale prefix
    View1 logits;          // last-token logits at tokens.size()
  };

  explicit PrefixCache(size_t capacity = 1) : PrefixCacheCore(capacity) {}

  void clear() {
    entries_.clear();
    PrefixCacheCore::clear();
  }

  // Restore entry `idx` into `model` (KV prefix, seq_len). The entry's
  // logits are NOT moved (the caller forwards the suffix and takes fresh
  // logits, or copies entry logits itself on exact hits). The entry's KV
  // mode must match the model's (dense vs Config::kv_int8); a mismatch
  // (e.g. weights/config swapped without clear_prefix_cache()) throws
  // rather than restoring rows into the wrong pools.
  void restore(size_t idx, ModelT& model) const {
    const Entry& e = entries_.at(idx);
    if (e.quant != model.cfg.use_quant_kv())
      throw std::runtime_error(
          "PrefixCache::restore: entry KV mode (int8/dense) does not match "
          "model config (call clear_prefix_cache() after changing kv_int8)");
    const size_t n = e.tokens.size();
    const size_t nl = static_cast<size_t>(model.cfg.n_layers);
    if (e.quant) {
      for (size_t l = 0; l < nl; ++l) {
        auto dst_kq = Kokkos::subview(
            model.kq_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_kq, e.kq[l]);
        auto dst_vq = Kokkos::subview(
            model.vq_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_vq, e.vq[l]);
        auto dst_ks = Kokkos::subview(model.ks_cache[l],
                                      std::make_pair(size_t(0), n),
                                      Kokkos::ALL);
        Kokkos::deep_copy(dst_ks, e.ks[l]);
        auto dst_vs = Kokkos::subview(model.vs_cache[l],
                                      std::make_pair(size_t(0), n),
                                      Kokkos::ALL);
        Kokkos::deep_copy(dst_vs, e.vs[l]);
      }
    } else {
      for (size_t l = 0; l < nl; ++l) {
        auto dst_k = Kokkos::subview(
            model.k_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_k, e.k[l]);
        auto dst_v = Kokkos::subview(
            model.v_cache[l],
            std::make_pair(size_t(0), n), Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(dst_v, e.v[l]);
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

  // Snapshot the model's current KV state under `tokens`: rows [0, n) must
  // be fully written, i.e. model.seq_len >= tokens.size(). A snapshot taken
  // after forwarding past the key (the merged content+suffix prefill in
  // Session::prefill_cached) is valid for suffix resumes but stores no
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
      e.logits = View1("KokkosDNN::Glimmer::PrefixCache::logits",
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
    e.quant = model.cfg.use_quant_kv();
    const size_t n = tokens.size();
    const size_t nl = static_cast<size_t>(model.cfg.n_layers);
    const size_t nkv = static_cast<size_t>(model.cfg.n_kv_heads);
    const size_t hd = static_cast<size_t>(model.cfg.head_dim);
    if (e.quant) {
      // Symmetric-int8 pools plus FP32 scale tables (no sharing, no paging;
      // one pair per layer, mirroring Model::kv_alloc_fresh).
      e.kq.resize(nl);
      e.vq.resize(nl);
      e.ks.resize(nl);
      e.vs.resize(nl);
      for (size_t l = 0; l < nl; ++l) {
        e.kq[l] = ViewI8("KokkosDNN::Glimmer::PrefixCache::kq", n, nkv, hd);
        e.vq[l] = ViewI8("KokkosDNN::Glimmer::PrefixCache::vq", n, nkv, hd);
        e.ks[l] = ViewScale("KokkosDNN::Glimmer::PrefixCache::ks", n, nkv);
        e.vs[l] = ViewScale("KokkosDNN::Glimmer::PrefixCache::vs", n, nkv);
        auto src_kq = Kokkos::subview(model.kq_cache[l],
                                      std::make_pair(size_t(0), n), Kokkos::ALL,
                                      Kokkos::ALL);
        Kokkos::deep_copy(e.kq[l], src_kq);
        auto src_vq = Kokkos::subview(model.vq_cache[l],
                                      std::make_pair(size_t(0), n), Kokkos::ALL,
                                      Kokkos::ALL);
        Kokkos::deep_copy(e.vq[l], src_vq);
        auto src_ks = Kokkos::subview(model.ks_cache[l],
                                      std::make_pair(size_t(0), n), Kokkos::ALL);
        Kokkos::deep_copy(e.ks[l], src_ks);
        auto src_vs = Kokkos::subview(model.vs_cache[l],
                                      std::make_pair(size_t(0), n), Kokkos::ALL);
        Kokkos::deep_copy(e.vs[l], src_vs);
      }
    } else {
      e.k.resize(nl);
      e.v.resize(nl);
      for (size_t l = 0; l < nl; ++l) {
        e.k[l] = View3("KokkosDNN::Glimmer::PrefixCache::k", n, nkv, hd);
        e.v[l] = View3("KokkosDNN::Glimmer::PrefixCache::v", n, nkv, hd);
        auto src_k = Kokkos::subview(model.k_cache[l],
                                     std::make_pair(size_t(0), n), Kokkos::ALL,
                                     Kokkos::ALL);
        Kokkos::deep_copy(e.k[l], src_k);
        auto src_v = Kokkos::subview(model.v_cache[l],
                                     std::make_pair(size_t(0), n), Kokkos::ALL,
                                     Kokkos::ALL);
        Kokkos::deep_copy(e.v[l], src_v);
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

}  // namespace Glimmer
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_GLIMMER_PREFIX_CACHE_HPP
