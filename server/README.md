# server/

OpenAI-compatible HTTP front-end for KokkosDNN (LFM2.5).
Enabled with `-DKokkosDNN_ENABLE_SERVER=ON` (default `OFF`, so the default
build is unaffected). Requires `KokkosDNN_ENABLE_MODEL_LFM25=ON` (default).

Fetching `cpp-httplib` + `nlohmann/json` via FetchContent touches the
network at configure time. For offline builds, pre-seed the sources and
pass `-DFETCHCONTENT_SOURCE_DIR_CPP_HTTPLIB=<path>` /
`-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=<path>`.

## Usage

```sh
cmake -B build -DKokkosDNN_ENABLE_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target kokkosdnn_server
./build/server/kokkosdnn_server --weights lfm2_5.gguf \
    --tokenizer /path/to/tokenizer.json [--config overlay.json] \
    [--max-seq-len 16384] [--model lfm2_5] [--port 8000] \
    [--prefix-cache-host]
```

`--help` / `--version` work without Kokkos weights. `--weights` and
`--tokenizer` are required to serve; `--config` is an optional model-config
JSON overlay on top of the baked-in `lfm2_5_24b_a2b` defaults.
`--max-seq-len N` overrides the context window (the default is 8192, and the
KV cache is allocated for N tokens), so prompts longer than 8k tokens need
either this flag or a config overlay.
`--prefix-cache-host` keeps the prefix-cache entry in host memory instead
of device memory (saves VRAM at the cost of host<->device copies per turn).

Startup logs the execution space, the device actually in use (name, compute
capability, SM count, memory, clocks) and the dispatch table (`cuBLAS
dispatch: ON/OFF`, `cuDNN SDPA dispatch: ON/OFF`). Check that block first
when throughput is below expectations: a MIG slice, a partially visible
device, or a build with the TPL dispatch compiled out all show up there.

## Docker

`server/Dockerfile` is a multi-stage CUDA build: the builder stage compiles
Kokkos (CUDA backend, `KOKKOS_VERSION` 5.2.1, `CUDA_ARCH` AMPERE80 — both
overridable via `--build-arg`) plus `kokkosdnn_server` with cuBLAS/cuDNN
dispatch on top of an `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` image
(same tags as the `tpl=ON` leg of `.github/workflows/cuda.yaml`); the
runtime stage copies only the server binary onto the matching
`cudnn-runtime` image. Build from the repo root:

```sh
docker build -f server/Dockerfile -t kokkosdnn-server:cuda .
docker run --gpus all -p 8000:8000 \
    -v /path/to/weights:/models:ro \
    kokkosdnn-server:cuda \
    --weights /models/lfm2_5.gguf --tokenizer /models/tokenizer.json
```

For a non-Ampere GPU, target it natively with
`--build-arg CUDA_ARCH=HOPPER90` (any `Kokkos_ARCH_*` suffix works). The
default `AMPERE80` build also carries compute_80 PTX, so it runs on Hopper
and Blackwell through the driver's PTX JIT; that works, but the kernels are
Ampere codegen and the JIT is repaid on every container start (no
`~/.nv/ComputeCache` in the image). Build HOPPER90 for a native sm_90 SASS
image on an H100.

## Routes

- `GET /v1/models` — `{object:"list", data:[{id, object:"model",
  created:0, owned_by:"kokkosdnn"}]}`
- `POST /v1/chat/completions` — chat completions; `stream:true` selects SSE
  (`Content-Type: text/event-stream`, per-step `data: {...}`, final
  stop-chunk with `finish_reason`, then `data: [DONE]`)
- `POST /v1/completions` — legacy alias: `{prompt: string}` becomes a single
  `user` message; same sampling/stop/seed/max-tokens handling, legacy
  `text_completion` envelope

## Supported request params

| Param | Mapping |
|---|---|
| `messages[]` | `role` in `system,user,assistant,tool` (`developer`→`user`, `function`→`tool`); string `content` (text-part arrays joined); assistant turns may carry `tool_calls` (validated, `content` may be `null`); rendered via `render_lfm25_chatml_with_tools` + `encode_chatml_with_tools` |
| `tools[]` | `[{type:"function", function:{name, description?, parameters?, strict?}}]` validated, then injected as `List of tools: [...]` JSON in the system prompt (native LFM2.5 tool format); `[]` = no-op |
| `tool_choice` | `"auto"` (default) \| `"none"` (no injection) \| `"required"` / `{"type":"function","function":{"name"}}` (best-effort `MUST call` system directive — prompted, not constrained) |
| `parallel_tool_calls` | `false` keeps only the first parsed call (default `true`) |
| `response_format` | `{"type":"text"}` (default) \| `{"type":"json_object"}` \| `{"type":"json_schema", json_schema:{...}}`: adds a JSON-only system instruction and strips ``` fences (non-streaming); the schema is recorded but **not enforced** (no grammar-constrained decoding — validate client-side) |
| `temperature, top_p, top_k, min_p, repetition_penalty` | `SampleOptions` verbatim (absent = session default; `temperature` 0 = greedy) |
| `frequency_penalty, presence_penalty` | `SampleOptions` verbatim |
| `logit_bias` | `{token_id: bias}` dict → `vector<pair<int,float>>` |
| `logprobs` (bool/int), `top_logprobs` (int) | top-N logprobs (`true` without a count = 5) |
| `stop` (string\|string[]) | integer-valued entries → `stop_ids` (always includes `cfg.eos`); rest → `stop_strings` |
| `max_tokens` / `max_completion_tokens` | completion-token cap, clamped to `max_seq_len - prompt_len` (latter wins when both present; default 256) |
| `seed` | per-request reseed (`prefill(ids, seed, true)`); absent = RNG stream continues |
| `stream` | SSE vs single JSON |
| `model` | must match `--model` or be absent, else 400 |
| unknown fields | ignored |

Agentic loop: with `tools` set, a turn that emits native
`<|tool_call_start|>[name(args)]<|tool_call_end|>` blocks returns
`finish_reason:"tool_calls"` with an OpenAI `tool_calls` payload (fresh
`call-kdnn-N` ids, `content` is the surrounding text or `null`) instead of
leaking marker text into `content`. Execute the functions, append the
assistant turn (with its `tool_calls`) plus one `tool`-role message per
result, and call again. Streaming emits content deltas live, then one
`tool_calls` delta chunk per call, then the `tool_calls` finish chunk.
`tool` history turns render natively, so multi-turn loops are exact.

Unsupported (400 `invalid_request_error`, never silently ignored):
`n>1`, legacy `functions` / `function_call` (retired; use
`tools`/`tool_choice`), unknown `tool_choice`/`response_format` shapes,
`tool_calls` on non-assistant messages, message-level `function_call`,
non-text content parts, prompts past `max_seq_len`. `tools`,
`tool_choice`, `response_format` are rejected on `POST /v1/completions`
(chat only).

Limits to know:
- `tool_choice:"required"`/named and `response_format` JSON modes are
  prompt-level hints, not hard guarantees: without constrained decoding
  the model can still answer in plain text. Treat a missing `tool_calls`
  (or non-JSON content) as a normal `stop` turn and retry/reprompt.
- Only `function` tools / `function` tool_choice targets are supported.
- `response_format` fence-stripping applies to non-streaming replies;
  streamed JSON deltas arrive unmodified.

## Errors

- `400 {"error":{"message", "type":"invalid_request_error"}}` — bad
  messages, unknown model id, over-context prompt, unsupported fields
- `500 {"error":{"message", "type":"server_error"}}` — inference exceptions
- SSE: headers are already sent, so mid-stream failures arrive as
  `data: {"error":...}` instead of an HTTP status

## Rules

- **No sampling/chat/detokenizer logic in `server/`** — that belongs to
  `models/` (shared primitives in `models/common/`). The server only
  translates HTTP <-> model-component calls (all generation runs through
  `LFM25::Session`).
- The server owns `Kokkos::initialize/finalize` for the serving process.
- Single-user, blocking: one global mutex guards all inference. A second
  concurrent request blocks until the in-flight request finishes (no 409;
  keep this in mind for multi-client use). One long-lived `Session` is
  reused across requests under that lock; per-request sampling/stop/seed
  options apply via `set_options()`, and a single-entry prefix cache of
  content states carries over across turns (see below).
- Prefix cache: each turn encodes the history twice — the full prompt and
  the cacheable content prefix (same encoding without the trailing
  generation prompt). When the new prompt starts with the cached content
  (the common agentic-loop shape: history plus new assistant/tool
  messages), only the suffix is prefilled; the turn is bit-identical to a
  full prefill. A miss replaces the entry. The per-turn log line reports
  `prefix_hit`/`prefix_reused`, and `prefill_ms` covers only the suffix
  forward, so the saving is measurable from the server logs. The entry
  lives in device memory by default; `--prefix-cache-host` keeps it in
  host memory instead (H2D/D2H copies per turn, ~20 KB per token for
  LFM2.5-24B at 128k context ≈ 2.5 GiB per full-window entry).
- A streaming client that disconnects stops the turn at the next token
  (detected from `sink.write()`); the abandoned turn is logged with
  `aborted=client-disconnect` instead of holding the inference lock for the
  remaining `max_tokens`.

## Observability

Every completed turn logs one line to stderr:

```
POST /v1/chat/completions -> 200 prompt_tokens=17 completion_tokens=64
  lock_ms=0.0 encode_ms=0.2 prefill_ms=261.4 decode_ms=1576.9 sse_ms=1.9
  decode_tok_s=40.6 prefix_hit=1 prefix_reused=12
```

`lock_ms` is time waiting for the inference mutex (non-zero means the
request queued behind another), `prefill_ms` is prompt processing (only
the uncached suffix when `prefix_hit=1`), `decode_ms` is the sum of model
steps, `sse_ms` is SSE framing plus socket writes, and `decode_tok_s` is
the rate a streaming client observes. `prefix_reused` is the reused
prefix length in tokens (0 on a miss). A GPU
that looks 100% busy in `nvidia-smi` while `decode_tok_s` is low means the
model step, not the transport, is the bottleneck.
