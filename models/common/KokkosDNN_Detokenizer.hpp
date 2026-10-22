#ifndef KOKKOSDNN_DETOKENIZER_HPP
#define KOKKOSDNN_DETOKENIZER_HPP

// Streaming detokenizer with stop-string handling (models/common).
//
// Problem: server-SSE streaming must emit incremental text deltas per
// generated token. Decoding each id in isolation
// (`Tokenizer::decode({id})`) can split a UTF-8 codepoint across tokens,
// and a stop string (e.g. "<|im_end|>") can straddle chunk boundaries.
// Post-hoc truncation of the full text (what the file-based drivers do
// via find_stop_string/truncate_at_stop_strings) cannot produce correct
// incremental deltas.
//
// IncrementalDetokenizer fixes both by buffering, and is deliberately
// tokenizer-agnostic: the caller passes the already-decoded single-id
// piece (`Tokenizer::decode({id})`, skip_special_tokens default) and this
// class owns all buffering policy. The server must use this class and not
// reimplement its own buffering logic.
//
// Policy per push():
//   1. Append the piece to the buffer.
//   2. If any stop string occurs in the buffer, emit everything before the
//      earliest occurrence, latch stop state, and discard the stop match
//      and everything after it (sticky: later pushes/flush emit nothing
//      until reset()).
//   3. Otherwise emit the longest prefix that is BOTH valid-UTF-8-complete
//      (no trailing partial multi-byte sequence) AND stop-safe (no
//      non-empty suffix of the emitted prefix is a prefix of any stop
//      string, so a stop completing on a later push is never split).
//      The remainder stays buffered.
// flush() emits whatever is still buffered (end of stream: no more bytes
// are coming, so a held-back stop-prefix/partial-UTF-8 tail is data, not
// a signal). Stop lists are tiny, so all scans are naive.
//
// No Kokkos dependency: <string>/<vector>/<cstddef> only.

#include <cstddef>
#include <string>
#include <vector>

namespace KokkosDNN {
namespace Common {

class IncrementalDetokenizer {
 public:
  struct Step {
    std::string delta;   // newly releasable text (may be empty = "hold")
    bool stop_hit = false;
    std::string matched;  // stop string that matched (valid iff stop_hit)
  };

  explicit IncrementalDetokenizer(
      const std::vector<std::string>& stop_strings) {
    for (const auto& s : stop_strings)
      if (!s.empty()) stops_.push_back(s);
  }

  // piece = Tokenizer::decode({id}) for the just-sampled token id.
  // `id` is currently informational only (kept for API symmetry with the
  // decode call and for future EOS-id short-circuiting); all decisions are
  // made on the decoded bytes.
  Step push(int id, const std::string& piece) {
    (void)id;
    Step out;
    if (stopped_) {
      out.stop_hit = true;
      out.matched = matched_;
      return out;
    }
    buffer_ += piece;

    // Earliest stop-string occurrence wins (tie: longest match).
    size_t hit_pos = std::string::npos;
    size_t hit_idx = 0;
    for (size_t i = 0; i < stops_.size(); ++i) {
      const size_t p = buffer_.find(stops_[i]);
      if (p != std::string::npos &&
          (hit_pos == std::string::npos || p < hit_pos ||
           (p == hit_pos && stops_[i].size() > stops_[hit_idx].size()))) {
        hit_pos = p;
        hit_idx = i;
      }
    }
    if (hit_pos != std::string::npos) {
      out.delta = buffer_.substr(0, hit_pos);
      out.stop_hit = true;
      out.matched = stops_[hit_idx];
      stopped_ = true;
      matched_ = out.matched;
      buffer_.clear();
      return out;
    }

    const size_t safe = safe_prefix_len(buffer_, stops_);
    out.delta = buffer_.substr(0, safe);
    buffer_.erase(0, safe);
    return out;
  }

  // End of stream: emit whatever is still buffered. After a stop, the
  // buffer was discarded, so flush() returns "" until reset().
  std::string flush() {
    std::string tail = buffer_;
    buffer_.clear();
    return tail;
  }

  void reset() {
    buffer_.clear();
    stopped_ = false;
    matched_.clear();
  }

  const std::string& buffered() const { return buffer_; }
  bool stopped() const { return stopped_; }
  const std::string& matched() const { return matched_; }

 private:
  static bool is_continuation(unsigned char c) {
    return (c & 0xC0u) == 0x80u;
  }

  // Length of the trailing incomplete UTF-8 sequence (0 when the buffer
  // ends on a character boundary). Invalid lead bytes are treated as
  // complete (emitted) so corrupt input can never deadlock the stream.
  static size_t trailing_partial_utf8(const std::string& b) {
    const size_t n = b.size();
    if (n == 0) return 0;
    size_t cont = 0;
    while (cont < n && cont < 4 &&
           is_continuation(static_cast<unsigned char>(b[n - 1 - cont])))
      ++cont;
    if (cont >= 4) return 0;  // invalid run: do not hold back
    if (cont == n) return n;  // all continuation bytes: incomplete
    const unsigned char lead =
        static_cast<unsigned char>(b[n - 1 - cont]);
    size_t want = 0;
    if (lead < 0x80u)
      want = 1;
    else if (lead >= 0xC0u && lead < 0xE0u)
      want = 2;
    else if (lead >= 0xE0u && lead < 0xF0u)
      want = 3;
    else if (lead >= 0xF0u && lead < 0xF8u)
      want = 4;
    else
      return 0;  // stray continuation/invalid lead: emit
    const size_t have = 1 + cont;
    return (have < want) ? have : 0;
  }

  // True iff some non-empty suffix of b[0, len) is a prefix of any stop.
  // (A full in-buffer stop is excluded by the caller: push() checks for a
  // full match before calling this.)
  static bool has_stop_prefix_suffix(const std::string& b, size_t len,
                                     const std::vector<std::string>& stops) {
    for (const auto& s : stops) {
      const size_t max_k =
          len < s.size() ? len : s.size() - 1;  // proper prefixes only
      for (size_t k = 1; k <= max_k; ++k) {
        if (b.compare(len - k, k, s, 0, k) == 0) return true;
      }
    }
    return false;
  }

  // Largest L in [0, b.size()] such that [0, L) is UTF-8-complete and no
  // suffix of [0, L) is a stop-string prefix. L=0 always qualifies, so
  // the loop terminates.
  static size_t safe_prefix_len(const std::string& b,
                                const std::vector<std::string>& stops) {
    size_t len = b.size();
    size_t hold_utf8 = trailing_partial_utf8(b);
    if (hold_utf8 > len) hold_utf8 = len;
    len -= hold_utf8;
    while (len > 0) {
      if (len < b.size() &&
          is_continuation(static_cast<unsigned char>(b[len])))
        --len;  // would split a codepoint
      else if (has_stop_prefix_suffix(b, len, stops))
        --len;  // would split a potential stop string
      else
        break;
    }
    return len;
  }

  std::vector<std::string> stops_;
  std::string buffer_;
  bool stopped_ = false;
  std::string matched_;
};

}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_DETOKENIZER_HPP
