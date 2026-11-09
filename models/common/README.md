# models/common

Shared model-support scaffold (header-only, `KokkosDNN::common`).
Every model library under `models/` links it; it must never link a model
library back (dependency direction: `models/* -> common -> core`).

## Status

`KokkosDNN_Common.hpp` umbrellas header-only utilities (all
tokenizer-agnostic unless noted, no Kokkos dependency):

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

- `KokkosDNN_JsonLite.hpp` — `Common::JsonLite`: dependency-free
  `tokenizer.json` helpers (`read_file`/`skip_ws`/`utf8_encode_cp`/
  `json_parse_string`/`find_key_value`/`match_bracket`/`object_body`/
  `array_body`/`parse_number`). Single home for the blocks previously
  duplicated in every byte-level tokenizer; per-tokenizer code keeps
  only its regex tables and added-token handling.
  (Checkpoint container framing is NOT here: safetensors/ZIP header
  readers live in `src/KokkosDNN_Safetensors.hpp` +
  `src/KokkosDNN_Zip.hpp`, next to `src/KokkosDNN_GGUF.hpp`. Core
  cannot include `models/common` — the dependency direction is
  `models/* -> common -> core` — so the archive readers are
  self-contained in `src/` and `models/*` + `examples/*` include them
  from there. The former per-loader string scanners
  (`dit_parse_json_string`, `qwen21_parse_json_string`) are retired in
  favor of the central parser.)
- `KokkosDNN_BpeCore.hpp` — `Common::Bpe`: GPT-2 `bytes_to_unicode` table
  + inverse plus the greedy lowest-rank BPE merge loop
  (`bpe_piece_bytelevel`) shared by the byte-level tokenizers
  (Llama/SmolLM2/LFM25/Glimmer). SentencePiece-style tokenizers
  (Gemma4, Sana text encoder) keep their own byte-fallback paths.
- `KokkosDNN_ChatStop.hpp` — `Common::Chat`: `find_stop_string`,
  `truncate_at_stop_strings`, `ensure_bos`, plus generic
  `make_chat_messages<Message>` / `encode_rendered_chat<Tokenizer>`
  templates. Per-namespace `ChatMessage` types, renderers, and
  `encode_chat(ml)` wrappers stay (markup differs); they forward the
  shared logic here.
- Sampling stays central in `src/KokkosDNN_Sampler.hpp`
  (`is_greedy_sample`, `apply_*`, `top_*_keep`,
  `sample_host_logits[_result]`); model-local copies are deleted.

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
- `KokkosDNN_CliOptions.hpp` — `Common::Cli::TextGenOptions` plus
  `Common::Cli::try_parse_shared()` and
  `Common::Cli::resolve_random_seed()`. Shares the argv machinery (list
  and scalar consumers plus the common text-generate flags:
  tokenizer/prompt/max-new/stop/stop-string/prefill-chunk/config/
  max-seq-len/fp32/sampling/seed/tiny/bos/weights) across the
  `examples/*/*_generate` drivers; `--model` and other model-specific
  flags stay per driver (selected via `Common::Cli::ParseAllow`).
- `KokkosDNN_SessionTypes.hpp` — `Common::SessionOptions`,
  `Common::StepOutcome`, `Common::CachedPrefillOutcome`,
  `Common::ISession`. Per-arch Sessions alias these types and keep only
  their state math (KV vs KV+conv, split vs merged suffix forwards).
- `KokkosDNN_PrefixCacheCore.hpp` — `Common::PrefixCacheCore`, the
  storage-agnostic longest-prefix lookup. Per-arch `PrefixCacheCore`
  names alias it; view-owning `PrefixCache` subclasses stay per-arch
  (conv states vs dense/int8 KV rows differ).

## Planned extractions (from `models/*`, when duplicated a second time)

- Per-regex pre-tokenizers (Llama vs SmolLM2 vs LFM2 vs Glimmer Split
  patterns stay per-tokenizer; shared only when a fifth copy appears)
- Samplers (today: `KokkosDNN_MossSampler`, flow-Euler image samplers stay
  separate from the LLM `src/KokkosDNN_Sampler.hpp` pipeline)

## Completed extractions

- Safetensors/ZIP header readers (done): `src/KokkosDNN_Safetensors.hpp`
  (`Safetensors::parse_header`/`MappedFile`/`convert_chunk`/`stream_into`,
  F32/BF16) + `src/KokkosDNN_Zip.hpp` (`Zip::Archive`/`MappedFile`,
  stored entries) consolidate the loaders previously duplicated in
  `models/moss_image/KokkosDNN_MossImage{Dit,Encoder,Vae}.hpp`,
  `models/qwen_image_21/KokkosDNN_QwenImage21{Config,SafetensorsMap}.hpp`,
  and `examples/{moss_image,sana,kokoro}` (`KokkosDNN_MossZip.hpp`,
  `KokkosDNN_MossWeights.cpp`, `SanaWeightsPth.cpp`,
  `KokkosDNN_SanaWeights.cpp`, `KokkosDNN_KokoroPth.cpp`). Per-loader
  files keep only their expected-tensor inventories, exact/suffix match
  policies, pickle VMs, and device fill logic.

## Rules

- Header-only (`INTERFACE` target); no `.cpp` without discussion.
- No model-specific names, dimensions, or weights here.
- No sampling/chat orchestration logic — only reusable primitives.
  Policy stays in `models/<name>/`; the HTTP layer stays in `server/`.
