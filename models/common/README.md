# models/common

Shared model-support scaffold (header-only, `KokkosDNN::common`).
Every model library under `models/` links it; it must never link a model
library back (dependency direction: `models/* -> common -> core`).

## Status

`KokkosDNN_Common.hpp` umbrellas two header-only utilities (both
tokenizer-agnostic, no Kokkos dependency):

- `KokkosDNN_Detokenizer.hpp` — `Common::IncrementalDetokenizer`:
  streaming detokenizer with stop-string handling. Construct with the
  request's stop strings; per sampled token call
  `push(id, Tokenizer::decode({id}))` and emit `Step::delta` as the SSE
  delta. Partial UTF-8 tails and trailing stop-string prefixes are held
  back (not emitted) until completed or ruled out; a full stop match
  sets `Step::stop_hit` (with `Step::matched`), truncates before the
  match, and latches (sticky until `reset()`). `flush()` emits the
  remainder at end of stream; `buffered()` exposes the held-back bytes.
- `KokkosDNN_Usage.hpp` — `Common::Usage` (`prompt_tokens`,
  `completion_tokens`, `total()`) plus `Common::FinishReason`
  (`Stop`/`Length`/`Error`) with `to_string()` mapping to the OpenAI
  `finish_reason` strings (`"stop"`/`"length"`; `Error` -> `"error"`).

Rule: the server must use these headers for SSE deltas, stop truncation,
usage, and finish reasons — no own buffering/truncation logic.

## Shared generate/session/prefix-cache concepts

- `KokkosDNN_ModelConcepts.hpp` — generic `Common::DecoderConfig`,
  `Common::DecoderWeights`, `Common::DecoderModel`,
  `Common::PrefixCacheCoreConcept`, `Common::DecodingSession` concepts.
  Per-arch Configs keep their own field names (`eos` vs `eos_token` vs
  `default_stop_ids`); only the pieces the shared driver touches
  (`max_seq_len`, `reset()`/`forward()`/`seq_len`) are constrained.
- `KokkosDNN_Generate.hpp` — `Common::generate_impl(model, ...)` plus
  `Common::TokenCallback` and `Common::normalize_stop_ids()`. Every
  decoder-only Model keeps only its `forward()` math; its `generate()`
  wrapper resolves its default stop set and delegates here.
- `KokkosDNN_SessionTypes.hpp` — `Common::SessionOptions`,
  `Common::StepOutcome`, `Common::CachedPrefillOutcome`,
  `Common::ISession`. Per-arch Sessions alias these types and keep only
  their state math (KV vs KV+conv, split vs merged suffix forwards).
- `KokkosDNN_PrefixCacheCore.hpp` — `Common::PrefixCacheCore`, the
  storage-agnostic longest-prefix lookup. Per-arch `PrefixCacheCore`
  names alias it; view-owning `PrefixCache` subclasses stay per-arch
  (conv states vs dense/int8 KV rows differ).

## Planned extractions (from `models/*`, when duplicated a second time)

- Minimal JSON helpers (today each model carries its own flat parser)
- BPE tokenizer core (today: `KokkosDNN_LFM25Tokenizer`, `KokkosDNN_QwenTokenizer`)
- Chat templates / prompt formatting
- Samplers (today: `KokkosDNN_MossSampler`)

## Rules

- Header-only (`INTERFACE` target); no `.cpp` without discussion.
- No model-specific names, dimensions, or weights here.
- No sampling/chat orchestration logic — only reusable primitives.
  Policy stays in `models/<name>/`; the HTTP layer stays in `server/`.
