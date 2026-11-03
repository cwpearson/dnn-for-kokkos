// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_COMMON_HPP
#define KOKKOSDNN_COMMON_HPP

// Umbrella for models/common: shared model-support utilities extracted
// from models/* and examples/* (minimal JSON helpers, BPE tokenizers,
// chat templates, samplers, streaming detokenizer + usage vocabulary
// below, plus the shared generate/session/prefix-cache concepts).
// No model-specific logic may live here; model libs (models/lfm2_5,
// models/glimmer, ...) link KokkosDNN::common, never the reverse. See
// models/common/README.md.

#include "KokkosDNN_Detokenizer.hpp"
#include "KokkosDNN_Generate.hpp"
#include "KokkosDNN_ModelConcepts.hpp"
#include "KokkosDNN_PrefixCacheCore.hpp"
#include "KokkosDNN_SessionTypes.hpp"
#include "KokkosDNN_Usage.hpp"

namespace KokkosDNN {
namespace Common {
// IncrementalDetokenizer (KokkosDNN_Detokenizer.hpp): streaming
// detokenizer with stop-string handling. The server feeds it the
// single-id decode piece per sampled token and emits Step::delta as SSE
// deltas; it must not implement its own buffering logic.
// Usage / FinishReason / to_string (KokkosDNN_Usage.hpp): token counts
// and the finish_reason vocabulary for SSE chunks ("stop"/"length").
// generate_impl (KokkosDNN_Generate.hpp): shared prefill+decode driver;
// per-arch Models keep only forward() plus a thin wrapper resolving
// their default stop ids.
// SessionOptions / StepOutcome / ISession (KokkosDNN_SessionTypes.hpp):
// shared turn vocabulary; per-arch Sessions keep only their state math.
// PrefixCacheCore (KokkosDNN_PrefixCacheCore.hpp): shared longest-prefix
// lookup; per-arch PrefixCache subclasses keep only view snapshots.
// DecoderModel et al. (KokkosDNN_ModelConcepts.hpp): the structural
// contracts the shared driver relies on.
}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_COMMON_HPP
