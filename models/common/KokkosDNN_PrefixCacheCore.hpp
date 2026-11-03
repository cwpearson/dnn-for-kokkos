// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_PREFIX_CACHE_CORE_HPP
#define KOKKOSDNN_PREFIX_CACHE_CORE_HPP

// Storage-agnostic half of the single-entry exact-prefix cache
// (models/common).
//
// LFM25::PrefixCacheCore and Glimmer::PrefixCacheCore were verbatim
// copies: token keys in LRU order (front = most-recently-used) plus
// longest-prefix lookup. Both now alias this Common type; the view-owning
// PrefixCache subclasses stay per-arch because their snapshots differ
// (LFM25 carries short-conv states alongside KV rows; Glimmer carries
// dense-FP or symmetric-int8 KV rows).

#include <cstddef>
#include <vector>

namespace KokkosDNN {
namespace Common {

// Token keys in LRU order (front = most-recently-used) plus longest-prefix
// lookup. The view-owning subclass keeps its snapshots parallel to these
// keys.
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

}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_PREFIX_CACHE_CORE_HPP
