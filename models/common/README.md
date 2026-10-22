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
