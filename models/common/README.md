# models/common

Shared model-support scaffold (header-only, `KokkosDNN::common`).
Every model library under `models/` links it; it must never link a model
library back (dependency direction: `models/* -> common -> core`).

## Status

Scaffold only: `KokkosDNN_Common.hpp` is an empty-namespace placeholder.
No behavior has moved here yet.

## Planned extractions (from `models/*`, when duplicated a second time)

- Minimal JSON helpers (today each model carries its own flat parser)
- BPE tokenizer core (today: `KokkosDNN_LFM25Tokenizer`, `KokkosDNN_QwenTokenizer`)
- Chat templates / prompt formatting
- Samplers (today: `KokkosDNN_MossSampler`)
- Detokenizers / stop-string streamers (server-SSE support)

## Rules

- Header-only (`INTERFACE` target); no `.cpp` without discussion.
- No model-specific names, dimensions, or weights here.
- No sampling/chat orchestration logic — only reusable primitives.
  Policy stays in `models/<name>/`; the HTTP layer stays in `server/`.
