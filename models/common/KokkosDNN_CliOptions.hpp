// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_CLI_OPTIONS_HPP
#define KOKKOSDNN_CLI_OPTIONS_HPP

// Shared text-generate CLI parsing machinery (models/common). The five
// text drivers carried ~70-line argv loops with a verbatim-identical core
// (tokenizer/prompt/max-new/stop/stop-string/prefill-chunk/config/
// max-seq-len/fp32/sampling/seed/tiny/bos/weights); only --model and the
// chat/KV extras differ. Drivers keep model-specific flags locally and
// select the shared subset via ParseAllow. Unknown flags (or flags
// without their value) are left unconsumed, as before.

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "KokkosDNN_Sampler.hpp"

namespace KokkosDNN {
namespace Common {
namespace Cli {

namespace Impl {
// True when s opens a new "--flag" token: list flags consume tokens
// until the next flag (a value starting with "--" is never consumed).
inline bool is_flag(const char* s) {
  return s != nullptr && s[0] == '-' && s[1] == '-';
}
}  // namespace Impl

// Common text-generate options. Defaults mirror the historical driver
// locals; each driver installs its sampling preset before parsing.
struct TextGenOptions {
  std::vector<std::string> weights;
  std::string tokenizer_path;
  std::string prompt = "Hello, I am";
  int max_new = 16;
  std::vector<int> extra_stops;
  std::vector<std::string> stop_strings;
  int prefill_chunk = -1;
  std::string config_path;
  int max_seq_len = -1;
  bool fp32 = false;
  bool tiny = true;
  bool tiny_explicit = false;
  bool want_bos = true;
  bool chat_mode = false;
  std::string system_prompt;
  std::string role = "user";
  ::KokkosDNN::SampleOptions samp;
  bool seed_given = false;
};

// Shared subset a driver accepts (LFM2.5: no --messages; Gemma4: no chat).
struct ParseAllow {
  bool chat = true;      // --chat / --system / --role
  bool messages = true;  // --messages (implies chat)
};

// Try one argv token; true when recognized (list flags advance `i`),
// false leaves `i` for the driver's own branches. Numeric conversions
// keep the historical throwing stoi/stof/stoull.
inline bool try_parse_shared(const std::string& arg, char** argv, int& i,
                             int argc, TextGenOptions& opts,
                             const ParseAllow& allow = ParseAllow{}) {
  if (arg == "--weights" && i + 1 < argc) {
    while (i + 1 < argc && !Impl::is_flag(argv[i + 1]))
      opts.weights.push_back(argv[++i]);
    opts.tiny = false;
    return true;
  }
  if (arg == "--tokenizer" && i + 1 < argc) {
    opts.tokenizer_path = argv[++i];
    return true;
  }
  if (arg == "--prompt" && i + 1 < argc) {
    opts.prompt = argv[++i];
    return true;
  }
  if (arg == "--max-new" && i + 1 < argc) {
    opts.max_new = std::stoi(argv[++i]);
    return true;
  }
  if (arg == "--stop" && i + 1 < argc) {
    while (i + 1 < argc && !Impl::is_flag(argv[i + 1]))
      opts.extra_stops.push_back(std::stoi(argv[++i]));
    return true;
  }
  if (arg == "--tiny") {
    opts.tiny = true;
    opts.tiny_explicit = true;
    return true;
  }
  if (arg == "--add-bos") {
    opts.want_bos = true;
    return true;
  }
  if (arg == "--no-add-bos") {
    opts.want_bos = false;
    return true;
  }
  if (arg == "--stop-string" && i + 1 < argc) {
    while (i + 1 < argc && !Impl::is_flag(argv[i + 1]))
      opts.stop_strings.push_back(argv[++i]);
    return true;
  }
  if (arg == "--prefill-chunk" && i + 1 < argc) {
    opts.prefill_chunk = std::stoi(argv[++i]);
    return true;
  }
  if (arg == "--config" && i + 1 < argc) {
    opts.config_path = argv[++i];
    return true;
  }
  if (arg == "--max-seq-len" && i + 1 < argc) {
    opts.max_seq_len = std::stoi(argv[++i]);
    return true;
  }
  if (arg == "--fp32") {
    opts.fp32 = true;
    return true;
  }
  if (arg == "--temperature" && i + 1 < argc) {
    opts.samp.temperature = std::stof(argv[++i]);
    return true;
  }
  if (arg == "--top-k" && i + 1 < argc) {
    opts.samp.top_k = std::stoi(argv[++i]);
    return true;
  }
  if (arg == "--top-p" && i + 1 < argc) {
    opts.samp.top_p = std::stof(argv[++i]);
    return true;
  }
  if (arg == "--min-p" && i + 1 < argc) {
    opts.samp.min_p = std::stof(argv[++i]);
    return true;
  }
  if (arg == "--repetition-penalty" && i + 1 < argc) {
    opts.samp.repetition_penalty = std::stof(argv[++i]);
    return true;
  }
  if (arg == "--seed" && i + 1 < argc) {
    opts.samp.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
    opts.seed_given = true;
    return true;
  }
  if (allow.chat && arg == "--chat") {
    opts.chat_mode = true;
    return true;
  }
  if (allow.chat && arg == "--system" && i + 1 < argc) {
    opts.system_prompt = argv[++i];
    return true;
  }
  if (allow.chat && arg == "--role" && i + 1 < argc) {
    opts.role = argv[++i];
    return true;
  }
  if (allow.messages && arg == "--messages" && i + 1 < argc) {
    // Minimal alias: wrap the value as the user turn (implies chat).
    opts.prompt = argv[++i];
    opts.chat_mode = true;
    return true;
  }
  return false;
}

// Random seed unless --seed was given (the shared post-loop block).
inline void resolve_random_seed(TextGenOptions& opts) {
  if (opts.seed_given) return;
  std::random_device rd;
  opts.samp.seed =
      (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
}

}  // namespace Cli
}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_CLI_OPTIONS_HPP
