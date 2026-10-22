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
    [--model lfm2_5] [--port 8000]
```

`--help` / `--version` work without Kokkos weights. `--weights` and
`--tokenizer` are required to serve; `--config` is an optional model-config
JSON overlay on top of the baked-in `lfm2_5_24b_a2b` defaults.

## Docker

`server/Dockerfile` is a multi-stage CUDA build: the builder stage compiles
Kokkos (CUDA backend, `KOKKOS_VERSION` 5.2.1, `KOKKOS_ARCH` AMPERE80 — both
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
`--build-arg KOKKOS_ARCH=HOPPER90` (any `Kokkos_ARCH_*` suffix works).

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
| `messages[]` | `role` in `system,user,assistant,tool` (`developer`→`user`); string `content` (text-part arrays joined; `null` allowed on assistant turns with `tool_calls` and on `tool` results); assistant `tool_calls[]` (`{id?, function:{name, arguments}}`, arguments string or object); rendered via `render_lfm25_chatml` + `encode_chatml` |
| `tools[]` | OpenAI function tools (`{type:"function", function:{name, description?, parameters?}}`); bare-function JSON appended to the system prompt as `List of tools: [...]` (official LFM2.5 template) |
| `tool_choice` | `"auto"` (default) consider tools; `"none"` disables tool use; `"required"` / `{"type":"function","function":{"name"}}` accepted as a hint (rendered like `"auto"` — forcing a call would need constrained decoding, so the model still decides) |
| `messages[]` history | `tool` role blocks render as `<|im_start|>tool`; assistant `tool_calls` render as native `<|tool_call_start|>[name(arg='v', ...)]<|tool_call_end|>` |
| tool results | model-native tool blocks are parsed back: `finish_reason:"tool_calls"`, `message:{role, content, tool_calls:[{id,type:"function",function:{name,arguments}}]}` (`content` is `null` for pure calls); no tool blocks → plain `stop`/`length` reply |
| `stream` + tools | buffered: full generation first (tool calls are only known once the native block completes), then standard SSE `role` / `content` / per-call `tool_calls[{index,id,...}]` deltas + final usage chunk; without tools, per-token content streaming as before |
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

Unsupported (400 `invalid_request_error`, never silently ignored):
`n>1`, legacy `functions` / `function_call`, `response_format`,
`function` message roles, non-text content parts, prompts past
`max_seq_len`. (`tools` on the legacy `/v1/completions` route is also 400 —
use `/v1/chat/completions`.)

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
  keep this in mind for multi-client use). A fresh `Session` is built per
  request under that lock so per-request sampling/stop/seed options apply.
