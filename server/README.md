# server/

OpenAI-compatible HTTP front-end for KokkosDNN (multi-architecture:
LFM2.5 and Glimmer).
Enabled with `-DKokkosDNN_ENABLE_SERVER=ON` (default `OFF`, so the default
build is unaffected). Requires at least one model component
(`KokkosDNN_ENABLE_MODEL_LFM25=ON` (default) and/or
`KokkosDNN_ENABLE_MODEL_GLIMMER=ON`). Without the Glimmer component the
Glimmer backend builds as a stub whose `load()` reports the missing
target.

Fetching `cpp-httplib` + `nlohmann/json` via FetchContent touches the
network at configure time. For offline builds, pre-seed the sources and
pass `-DFETCHCONTENT_SOURCE_DIR_CPP_HTTPLIB=<path>` /
`-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=<path>`.

## Usage

```sh
cmake -B build -DKokkosDNN_ENABLE_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target kokkosdnn_server
./build/server/kokkosdnn_server --arch lfm2.5 --weights lfm2_5.gguf \
    --tokenizer /path/to/tokenizer.json [--config overlay.json] \
    [--max-seq-len 16384] [--model lfm2_5] [--port 8000] \
    [--prefix-cache-host]
# Glimmer (packed `.gguf` for 24GB GPUs, or sharded safetensors):
./build/server/kokkosdnn_server --arch glimmer \
    --weights /path/to/model.gguf \
    --tokenizer /path/to/tokenizer.json [--model glimmer-30b] \
    [--config config.json] [--max-seq-len 131072] [--kv-int8] \
    [--prefill-chunk 512] [--prefix-cache-host]
```

`--help` / `--version` work without Kokkos weights. `--arch` selects the
model architecture (`lfm2.5` (aliases `lfm2_5`, `lfm25`) or `glimmer`
(aliases `glimmer-30b`, `muse-glimmer[-30b]`, optional `darkc0de-` prefix),
case-insensitive). When `--arch` is absent the server sniffs `--config`
(LFM2.5 markers: `lfm2_moe` / `num_experts` / `hidden_act`; Glimmer markers:
`text_config` / `layer_rope_theta` / `qk_scale_factor`); an explicit
`--arch` that disagrees with a recognizable config is a startup error,
otherwise the default is `lfm2.5`. `--model` defaults to the arch default
(`lfm2_5` / `glimmer-30b`) when absent.
`--weights` is repeatable and accepts globs: LFM2.5 needs exactly one GGUF
file; Glimmer takes one `.gguf` file or one or more safetensors shards
(`model-*.safetensors` is expanded). `--tokenizer` is required to serve;
`--config` is an optional model-config JSON overlay on top of the arch
preset. `--max-seq-len N` overrides the context window (LFM2.5 default
8192; Glimmer default 8192, arch max 131072), so prompts longer than the
default need either this flag or a config overlay. Glimmer also takes
`--kv-int8` / `--kv-fp` / `--kv-quant int8|none` (KV-cache precision;
overrides the config-file `kv_int8`/`kv_quantized`/`kv_quant` keys, dense
FP by default, last flag wins) and `--prefill-chunk N` (`0` =
single-shot; overrides the config-file `prefill_chunk_size` and caps
prefill activations) — same spelling and precedence as the
`glimmer_generate` driver. `--prefix-cache-host` keeps the prefix-cache
entry in host memory instead of device memory (applies to both
architectures; saves VRAM at the cost of host<->device copies per turn).
The server runs the BF16 path only (there is no `--fp32` flag); Glimmer
weights load from one `.gguf` file (packed quant preserved: F32/F16/BF16
exact, Q8_0/Q4_K/Q5_K/Q6_K stay packed, ~17GB resident for the Q4_K_M
recipe) or sharded safetensors (~60 GiB BF16).

Startup logs the execution space, the device actually in use (name, compute
capability, SM count, memory, clocks) and the dispatch table (`cuBLAS
dispatch: ON/OFF`, `cuDNN SDPA dispatch: ON/OFF`). Check that block first
when throughput is below expectations: a MIG slice, a partially visible
device, or a build with the TPL dispatch compiled out all show up there.

## Docker

One image serves EITHER LFM2.5 OR Glimmer — the builder compiles both
model components and the checkpoint is volume-mounted at run time, so the
arch is selected per `docker run`, not per image build. (On a server build
that predates the `--arch lfm2.5|glimmer` selector, drop the `--arch` flag;
`--model` still selects the served id.)

`server/Dockerfile` is a multi-stage CUDA build: the builder stage compiles
Kokkos (CUDA backend, `KOKKOS_VERSION` 5.2.1, `CUDA_ARCH` AMPERE80 — both
overridable via `--build-arg`) plus `kokkosdnn_server` with cuBLAS/cuDNN
dispatch on top of an `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` image
(same tags as the `tpl=ON` leg of `.github/workflows/cuda.yaml`); the
runtime stage copies only the server binary (plus a tiny argument-
validating entrypoint) onto the matching `cudnn-runtime` image. Glimmer's
sharded-safetensors glob is expanded inside the binary (POSIX `glob(3)`),
so the runtime image needs no extra packages. Build from the repo root:

```sh
docker build -f server/Dockerfile -t kokkosdnn-server:cuda .
```

LFM2.5 (GGUF checkpoint):

```sh
docker run --gpus all -p 8000:8000 \
    -v /path/to/lfm:/models/lfm:ro \
    kokkosdnn-server:cuda \
    --weights /models/lfm/lfm2_5.gguf --tokenizer /models/lfm/tokenizer.json
```

Glimmer (sharded safetensors, ~60 GiB BF16 — quote the glob so the shell
passes it through; the server expands `model-*.safetensors` itself):

```sh
docker run --gpus all -p 8000:8000 \
    -v /path/to/glimmer:/models/glimmer:ro \
    kokkosdnn-server:cuda \
    --arch glimmer --model glimmer-30b \
    --weights '/models/glimmer/model-*.safetensors' \
    --tokenizer /models/glimmer/tokenizer.json \
    --config /models/glimmer/config-glimmer-30b-text.json
```

Glimmer from a single llama.cpp GGUF file instead (`--weights` ending in
`.gguf` selects the GGUF loader; F32/F16/BF16 exact, Q8_0/Q4_K/Q5_K/Q6_K
stay packed and dequantize on the fly — the Q4_K_M recipe is ~17GB
resident):

```sh
docker run --gpus all -p 8000:8000 \
    -v /path/to/glimmer:/models/glimmer:ro \
    kokkosdnn-server:cuda \
    --arch glimmer --model glimmer-30b \
    --weights /models/glimmer/model.gguf \
    --tokenizer /models/glimmer/tokenizer.json
```

Glimmer on a 24GB GPU (packed GGUF + int8 KV + chunk cap): `--kv-int8`
halves the KV cache (3432 MiB at `--max-seq-len 131072` vs ~6.5 GiB
dense; 214.5 MiB vs 416 MiB at the 8192 default) and `--prefill-chunk N`
caps prefill scratch (~440 MiB at the 2048 default, ~110 MiB at 512).
~17GB weights + 3432 MiB int8 KV + ~614 MiB dequant staging + ~440 MiB
scratch ≈ 21.4 GiB, fitting 24GB:

```sh
docker run --gpus all -p 8000:8000 \
    -v /path/to/glimmer:/models/glimmer:ro \
    kokkosdnn-server:cuda \
    --arch glimmer --model glimmer-30b \
    --weights /models/glimmer/model.gguf \
    --tokenizer /models/glimmer/tokenizer.json \
    --config /models/glimmer/config-glimmer-30b-text.json \
    --max-seq-len 131072 --kv-int8 --prefill-chunk 512
```

Both `-v` layouts may be mounted at once (`-v /path/to/lfm:/models/lfm:ro
-v /path/to/glimmer:/models/glimmer:ro`); only the `--weights` /
`--tokenizer` paths on the command line decide what is served. With no
args the container prints `kokkosdnn_server --help`; omitting `--weights`
or `--tokenizer` fails fast with a mount-oriented error from the
entrypoint. Note the ~60 GiB BF16 Glimmer weights need a large-GPU host —
for 24GB GPUs use the packed-GGUF + `--kv-int8` recipe above (a small
`--max-seq-len` also caps the KV cache); see
[glimmer](../examples/glimmer/README.md) for checkpoint download and
memory notes.

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

## Architectures

Route handlers are arch-agnostic: they parse OpenAI JSON once and drive the
`KokkosDNN::Server::ChatBackend` selected by `--arch`. Per-arch differences:

| Concern | `lfm2.5` (`BackendLFM25.cpp`) | `glimmer` (`BackendGlimmer.cpp`) |
|---|---|---|
| Chat template | LFM2.5-Instruct ChatML + `<think>` generation prompt, `List of tools:` system block, tool history turns | Muse-style `<\|begin_of_text\|>` + `<\|start\|>{role}<\|message\|>{content}<\|eot\|>` blocks with `to=self` (reasoning) / `to=<tool>` (ATEM tool calls) / `to=user` channels, tool catalogue + valid recipients in the system turn, `<|eom|>`/`<|eot|>` closers, bare `<|start|>assistant` generation prompt (upstream `chat_template.jinja`) |
| Default stops | ids `{cfg.eos}`, strings `["<\|im_end\|>"]` | ids `{eos, eot}`, strings `["<|eot|>"]` (`<|eom|>` ends a message while the turn continues, so it is structural, never a stop) |
| Tools | native `<\|tool_call_start\|>` grammar, full `tools`/`tool_choice` support | ATEM `<atem:invoke>` grammar, full `tools`/`tool_choice` support (one call per turn upstream; `parallel_tool_calls:false` keeps the first) |
| Weights | exactly one `--weights` GGUF | one `--weights` `.gguf` or sharded safetensors (globs expanded) |
| Reasoning channel | `<think>` trace → `reasoning_content` | `to=self` trace → `reasoning_content` |
| Prefix cache | single-entry content-state cache (exact-prefix reuse) | single-entry content-state cache (exact-prefix reuse; dense or int8 KV rows) |
| Response format | JSON-mode system directive + fence stripping | JSON-mode directive prepended to the system message (no fence stripping on streams, same as LFM25); no grammar enforcement |

### Adding a third architecture (3-step recipe)

1. New `server/BackendX.cpp` implementing `KokkosDNN::Server::ChatBackend`
   (load/render+encode/prefill_cached/step/flush/usage/stops/decode/stream
   parser) plus a file-static `register_backend("<name>", ...)` registrar —
   no handler or CLI edits needed. Non-public helpers go in `Impl`.
2. `server/CMakeLists.txt`: `KOKKOSDNN_SERVER_HAVE_X` compile definition +
   conditional `KokkosDNN::x` link (mirror the LFM25 block; stub the backend
   when the target is absent so `--arch` still resolves).
3. `server/test_server_parsing.cpp`: registry/defaults assertions plus a
   weight-free render test with a tiny config; `server/README.md`: a row in
   the table above.

## Supported request params

| Param | Mapping |
|---|---|
| `messages[]` | `role` in `system,user,assistant,tool` (`developer`→`user`, `function`→`tool`); string `content` (text-part arrays joined); assistant turns may carry `tool_calls` (validated, `content` may be `null`) plus `thinking`/`reasoning`/`reasoning_content` (trace replay, first non-empty wins); rendered via the arch backend (`render_lfm25_chatml_with_tools` / `render_glimmer_chatml_with_tools`) |
| `tools[]` | `[{type:"function", function:{name, description?, parameters?, strict?}}]` validated, then injected into the system prompt (arch-native tool format: LFM2.5 `List of tools: [...]` JSON, Glimmer ATEM catalogue); `[]` = no-op |
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

Agentic loop (LFM2.5): with `tools` set, a turn that emits native
`<|tool_call_start|>[name(args)]<|tool_call_end|>` blocks returns
`finish_reason:"tool_calls"` with an OpenAI `tool_calls` payload (fresh
`call-kdnn-N` ids, `content` is the surrounding text or `null`) instead of
leaking marker text into `content`. Execute the functions, append the
assistant turn (with its `tool_calls`) plus one `tool`-role message per
result, and call again. Streaming emits `reasoning_content` deltas for the
`<think>` trace and `content` deltas live, then one `tool_calls` delta
chunk per call, then the `tool_calls` finish chunk. `tool` history turns
render natively, so multi-turn loops are exact. A truncated turn (capped by
`max_tokens`) that ends inside a call block still returns the recoverable
calls instead of leaking call markup into `content`. A recovered
length-capped turn still reports `finish_reason:"tool_calls"` (the length
cap is hidden), so validate the arguments before executing — they may be
cut mid-list.
The parser also tolerates the LFM3 / qwen3-coder wrapper
`<tool_call>...</tool_call>` with `<function=name>` / `<parameter=k>v`
`</parameter>` bodies, including hybrids with the native form (an XML
opener closed by `<|tool_call_end|>`, pythonic items prefixed with
`function=`, or a degenerate `<function>name</function>` head), plus
`<thinking>...</thinking>` as a `<think>` alias, so a drifting turn still
becomes a tool call instead of leaking markup into the thinking trace.
Rendering always uses the canonical native markers.
Turn boundaries: the `<|im_end|>` terminator always ends the turn, even
with a custom `stop` list, and a length-capped turn ending mid-marker has
only its longest trailing partial marker head stripped (an earlier `<`
followed by `<` is content).

Thinking (LFM2.5-Instruct): LFM2.5-Instruct always reasons inside `<think>...</think>` first.
The generation prompt ends with an open `<think>` (matching the upstream
chat template), so raw completions normally open implicitly: the leading
segment up to the first `</think>` / tool-call start / `<|im_end|>` is
reasoning even with no explicit opener in the bytes. A turn that reaches
none of those terminators (cut by `max_tokens`, or a model that never
closes the trace) is reported entirely as reasoning, so `content` is
empty for that turn rather than the trace leaking into it.
`<thinking>` / `</thinking>` are accepted wherever the canonical
`<think>` / `</think>` pair is, and either closer ends either opener (a
model that mixes the two still splits correctly).
The trace never
appears in `content`: streaming sends it as
`reasoning_content` deltas (the DeepSeek/OpenRouter convention, which Zed
renders in its thinking trace), non-streaming as the message-level
`reasoning_content` field (omitted when empty). History assistant turns
accept `thinking` / `reasoning` / `reasoning_content` (first non-empty
wins, matching the upstream chat template) and re-render them into the
prompt's `<think>` block, so Zed's `interleaved_reasoning` replay keeps
its trace. An explicit `<think>` opener later in the turn only counts
before any non-whitespace content (tool-call blocks excluded, so thinking
after a tool call still counts); model text that merely
mentions the tag stays in `content`.

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
- Zed (OpenAI-compatible provider, chat-completions path) is supported;
  see [Zed (LFM2.5)](#zed-lfm25) and [Zed (Glimmer)](#zed-glimmer) for the
  provider settings this server expects (a couple of Zed defaults are
  wrong for a reasoning model and will visibly degrade multi-turn agent
  threads).

## Zed (LFM2.5)

Zed's `openai_compatible` provider works with the chat-completions path.
Add the server as a provider in `~/.config/zed/settings.json` (or via
Agent Settings → Add Provider, then fill in the capabilities below):

```json
{
  "language_models": {
    "openai_compatible": {
      "kokkosdnn": {
        "api_url": "http://127.0.0.1:8000/v1",
        "available_models": [
          {
            "name": "lfm2_5",
            "display_name": "LFM2.5 24B-A2B (local)",
            "max_tokens": 8192,
            "max_output_tokens": 4096,
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": true,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": true,
              "max_tokens_parameter": false
            }
          }
        ]
      }
    }
  }
}
```

Why these values (the defaults are not all right for this model):

- `interleaved_reasoning: true` is **required**. With Zed's default
  `false`, prior assistant `Thinking` content is not dropped and not sent
  as `reasoning_content` — it is replayed as **plain assistant text**
  (`crates/open_ai/src/completion.rs`), so the model sees its own trace
  without the template's `<think>` wrapper and imitates whatever markup it
  contained. That feedback loop is what makes a thread drift from the
  native format into other tool grammars. With `true`, Zed sends the trace
  back as `reasoning_content` and the server re-renders it inside the
  `<think>` block the chat template expects.
- `parallel_tool_calls: true` lets a multi-call turn through. With the
  default `false`, the request carries `parallel_tool_calls:false` and the
  server keeps only the first parsed call.
- `max_tokens` is Zed's **context window**, not an output cap; set it to
  the server's `--max-seq-len` (default `8192`). Zed compacts at 90% of
  this value, so a larger number lets the thread grow past the server's
  limit and every later request fails with an over-context 400.
- `max_output_tokens` is the output cap, sent as `max_completion_tokens`.
  Set it explicitly and generously: when unset, Zed sends no usable cap
  (the field is absent or `null`), and the server then falls back to 256
  completion tokens, which a model that always thinks first will often
  exhaust before it finishes a tool call (the server clamps the value to
  `max_seq_len - prompt_len` anyway).
- `chat_completions` must stay `true`; the server has no `/v1/responses`.
- `name` must match the server's `--model` (default `lfm2_5`), and
  `api_url` must end in `/v1` (Zed appends `/chat/completions`).
- `reasoning_effort` is optional and ignored by the server; setting a
  non-`none` value only enables Zed's thinking controls for the model.

After changing `interleaved_reasoning`, **start a new thread**: existing
threads still carry the old, polluted history and may keep reproducing
the old format.

## Zed (Glimmer)

Zed's `openai_compatible` provider works with the chat-completions path.
Add the server as a provider in `~/.config/zed/settings.json` (or via
Agent Settings → Add Provider, then fill in the capabilities below):

```json
{
  "language_models": {
    "openai_compatible": {
      "kokkosdnn": {
        "api_url": "http://127.0.0.1:8000/v1",
        "available_models": [
          {
            "name": "glimmer-30b",
            "display_name": "Glimmer 30B (local)",
            "max_tokens": 8192,
            "max_output_tokens": 4096,
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": true,
              "max_tokens_parameter": false
            }
          }
        ]
      }
    }
  }
}
```

Why these values (only `parallel_tool_calls` differs from the LFM2.5
settings above):

- `tools: true` is **required** for agentic threads. The server speaks the
  upstream ATEM tool protocol and exposes standard OpenAI `tool_calls`;
  with Zed's default `true` nothing changes, but flipping it to `false`
  (as an older revision of this section advised) silently disables tools
  and every tool-using thread degrades to plain text.
- `interleaved_reasoning: true` is **required**. With Zed's default
  `false`, prior assistant `Thinking` content is sent back as plain
  assistant text instead of `reasoning_content`, so the model sees its own
  trace outside the `to=self` channel the template expects (the same
  feedback loop documented for LFM2.5 above). With `true`, Zed sends the
  trace back as `reasoning_content` and the server re-renders it inside
  the `to=self` channel.
- `parallel_tool_calls: false` matches the upstream single-call norm: the
  model emits one tool call per turn, and the server keeps only the first
  parsed call when the request carries `parallel_tool_calls:false`.
- `max_tokens` is Zed's **context window**, not an output cap; set it to
  the server's `--max-seq-len` (default `8192`, arch max `131072`). Zed
  compacts at 90% of this value, so a larger number lets the thread grow
  past the server's limit and every later request fails with an
  over-context 400.
- `max_output_tokens` is the output cap, sent as `max_completion_tokens`.
  Set it explicitly and generously: when unset, Zed sends no usable cap
  (the field is absent or `null`), and the server then falls back to 256
  completion tokens (the server clamps the value to
  `max_seq_len - prompt_len` anyway).
- `chat_completions` must stay `true`; the server has no `/v1/responses`.
- `name` must match the server's `--model` (default `glimmer-30b`), and
  `api_url` must end in `/v1` (Zed appends `/chat/completions`).
- Do not set `reasoning_effort`: the server ignores it; setting a
  non-`none` value only enables Zed's thinking controls for the model.

## Errors

- `400 {"error":{"message", "type":"invalid_request_error"}}` — bad
  messages, unknown model id, over-context prompt, unsupported fields
- `500 {"error":{"message", "type":"server_error"}}` — inference exceptions
- SSE: headers are already sent, so mid-stream failures arrive as
  `data: {"error":...}` instead of an HTTP status

## Rules

- **No sampling/chat/detokenizer logic in `server/`** — that belongs to
  `models/` (shared primitives in `models/common/`). The server only
  translates HTTP <-> backend calls (all generation runs through the
  arch backend: `Glimmer::Session` for Glimmer, `LFM25::Session` for
  LFM2.5).
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
POST /v1/chat/completions -> 200 arch=lfm2.5 prompt_tokens=17 completion_tokens=64
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
