#ifndef KOKKOSDNN_USAGE_HPP
#define KOKKOSDNN_USAGE_HPP

// Token-usage counters and finish-reason vocabulary for the
// OpenAI-compatible server (models/common, header-only).
//
// The server must use these types (not its own ad-hoc structs/strings)
// so SSE chunks and the final usage payload share one vocabulary.
// No Kokkos dependency: <cstddef> only.

#include <cstddef>

namespace KokkosDNN {
namespace Common {

// Token counts for one completion request.
struct Usage {
  size_t prompt_tokens = 0;
  size_t completion_tokens = 0;
  size_t total() const { return prompt_tokens + completion_tokens; }
};

// Why generation stopped. Maps to the OpenAI `finish_reason` strings via
// to_string() below ("stop" / "length"; Error surfaces transport or
// inference failures that OpenAI has no code for as "error").
enum class FinishReason { Stop, Length, Error };

inline const char* to_string(FinishReason r) {
  switch (r) {
    case FinishReason::Stop: return "stop";
    case FinishReason::Length: return "length";
    case FinishReason::Error: return "error";
  }
  return "error";
}

}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_USAGE_HPP
