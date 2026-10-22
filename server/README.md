# server/

OpenAI-compatible HTTP front-end for KokkosDNN (stub status).
Enabled with `-DKokkosDNN_ENABLE_SERVER=ON` (default `OFF` while a stub,
so the default build is unaffected). Requires
`KokkosDNN_ENABLE_MODEL_LFM25=ON` (default).

## Status

`kokkosdnn_server --help/--version` works; otherwise the binary initializes
Kokkos, prints its baked-in configuration (model id, port, LFM2.5 context
bound), notes the routes as TODO, and exits. It binds no port, serves no
requests, and constructs no model session.

## Intended routes

- `GET /v1/models` — list served model ids
- `POST /v1/chat/completions` — chat completions (SSE streaming later)

## Rules

- **No sampling/chat/detokenizer logic in `server/`** — that belongs in
  `models/` (shared primitives migrate to `models/common/`). The server
  only translates HTTP <-> model-component calls.
- The server owns `Kokkos::initialize/finalize` for the serving process.
- HTTP/JSON dependencies (`cpp-httplib v0.15.3`, `nlohmann/json v3.11.3`)
  stay commented out in `CMakeLists.txt` until the first route lands, so
  configure never touches the network; see the TODO block there for pins
  and `FETCHCONTENT_SOURCE_DIR_*` offline overrides.
