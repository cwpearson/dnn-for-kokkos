#ifndef KOKKOSDNN_COMMON_HPP
#define KOKKOSDNN_COMMON_HPP

// Placeholder umbrella for models/common: the future home of shared
// model-support utilities extracted from models/* (minimal JSON helpers,
// BPE tokenizers, chat templates, samplers, detokenizers, stop-streamers).
// No model-specific logic may live here; model libs (models/lfm2_5,
// models/moss_image) link KokkosDNN::common, never the reverse. See
// models/common/README.md.

namespace KokkosDNN {
namespace Common {
// (empty scaffold; shared utilities land here as they are extracted)
}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_COMMON_HPP
