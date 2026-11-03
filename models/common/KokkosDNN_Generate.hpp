// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_GENERATE_HPP
#define KOKKOSDNN_GENERATE_HPP

// Shared prefill+decode generate() driver (models/common).
//
// Every decoder-only Model (models/lfm2_5, models/glimmer,
// examples/llama, examples/smollm2, examples/gemma4) used to carry a
// verbatim copy of this loop; only the default stop set differed
// ({cfg.eos} vs {cfg.eos_token} vs {eos,eom,eot} vs
// cfg.default_stop_ids). Per-arch Model headers now keep only their
// forward() math plus a thin generate() wrapper that resolves its
// defaults and delegates here.
//
// The driver owns its staging (one prompt H2D plus one persistent
// single-token device slot) and its DeviceSampler scratch, so per-arch
// Models do not need to expose their private tok_buf_/one_buf_ members.
// Behavior matches the historical loops exactly:
//   - reset() first, out starts as the prompt;
//   - stops are sorted+deduped; the prefill token is pushed (and streamed)
//     even when the prompt already fills the window, mirroring history;
//   - the decode loop checks is_stop / max_seq BEFORE forwarding, so a
//     stop token never triggers another forward;
//   - sampling runs on-device via a persistent DeviceSampler (the LFM25
//     choice); sample_token() constructs an equivalent fresh sampler per
//     token, so switching the other models to the persistent form is
//     bit-identical while removing one V-sized alloc per token.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "KokkosDNN_ModelConcepts.hpp"
#include "KokkosDNN_Sampler.hpp"

namespace KokkosDNN {
namespace Common {

using TokenCallback = std::function<void(int)>;

// Sort + dedupe a stop-id list (shared by generate() and Session).
inline std::vector<int> normalize_stop_ids(std::vector<int> stops) {
  std::sort(stops.begin(), stops.end());
  stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
  return stops;
}

// Prefill + sampled decode loop over any DecoderModel. `resolved_stops`
// must already contain the caller's defaults (empty means "no stop ids").
// Returns all ids (prompt + generated).
template <class ModelT>
  requires DecoderModel<ModelT>
std::vector<int> generate_impl(ModelT& model, const std::vector<int>& prompt,
                               int max_new,
                               const std::vector<int>& resolved_stops,
                               const SampleOptions& opt, HostRNG* rng = nullptr,
                               TokenCallback on_token = nullptr) {
  using ViewInt = typename ModelT::ViewInt;
  using ExecSpace = typename ViewInt::execution_space;
  using HostInt = typename ViewInt::host_mirror_type;

  model.reset();
  std::vector<int> out = prompt;
  std::vector<int> stops = normalize_stop_ids(resolved_stops);
  auto is_stop = [&](int id) {
    return std::binary_search(stops.begin(), stops.end(), id);
  };
  const int max_seq = model.cfg.max_seq_len;
  HostRNG local_rng(opt.seed);
  HostRNG& r = rng ? *rng : local_rng;

  // Prompt staging: a single host->device copy per generate().
  ViewInt tok_buf("KokkosDNN::Common::generate::toks", prompt.size());
  HostInt tok_host("KokkosDNN::Common::generate::toks_host", prompt.size());
  for (size_t i = 0; i < prompt.size(); ++i) tok_host(i) = prompt[i];
  Kokkos::deep_copy(tok_buf, tok_host);

  auto logits = model.forward(tok_buf);
  // One sampler for the whole run: the V-sized sampling scratch is
  // allocated once instead of per token.
  ::KokkosDNN::Impl::DeviceSampler<ExecSpace> sampler(logits.extent(0));
  int nxt = sampler.sample(logits, opt, out, r).id;
  if ((int)out.size() < max_seq) out.push_back(nxt);
  if (on_token) on_token(nxt);
  ViewInt one_buf("KokkosDNN::Common::generate::one", 1);
  for (int s = 1; s < max_new; ++s) {
    if (is_stop(nxt)) break;
    if ((int)out.size() >= max_seq) break;
    // Device-side single-token staging: no host mirror + deep_copy per
    // token. Hoisted to a local: device lambdas must not capture `model`.
    auto one = one_buf;
    const int tok = nxt;
    Kokkos::parallel_for(
        "KokkosDNN::Common::generate::fill_one",
        Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int i) {
          (void)i;
          one(0) = tok;
        });
    logits = model.forward(one_buf);
    nxt = sampler.sample(logits, opt, out, r).id;
    out.push_back(nxt);
    if (on_token) on_token(nxt);
  }
  return out;
}

}  // namespace Common
}  // namespace KokkosDNN

#endif  // KOKKOSDNN_GENERATE_HPP
