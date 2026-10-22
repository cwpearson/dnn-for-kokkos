#ifndef KOKKOSDNN_COMMON_HPP
#define KOKKOSDNN_COMMON_HPP

// Umbrella for models/common: shared model-support utilities extracted
// from models/* (minimal JSON helpers, BPE tokenizers, chat templates,
// samplers, streaming detokenizer + usage vocabulary below).
// No model-specific logic may live here; model libs (models/lfm2_5,
// models/moss_image) link KokkosDNN::common, never the reverse. See
// models/common/README.md.

#include "KokkosDNN_Detokenizer.hpp"
#include "KokkosDNN_Usage.hpp"

namespace KokkosDNN {
namespace Common {
// IncrementalDetokenizer (KokkosDNN_Detokenizer.hpp): streaming
// detokenizer with stop-string handling. The server feeds it the
// single-id decode piece per sampled token and emits Step::delta as SSE
// deltas; it must not implement its own buffering logic.
// Usage / FinishReason / to_string (KokkosDNN_Usage.hpp): token counts
// and the finish_reason vocabulary for SSE chunks ("stop"/"length").
}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_COMMON_HPP
