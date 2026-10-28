# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# docker-entrypoint.sh: thin wrapper so one image serves EITHER LFM2.5 OR
# Glimmer with weights volume-mounted at runtime.
#
# The server binary owns all flag parsing (including the future `--arch
# lfm2.5|glimmer` selector from the --arch refactor workstream); this script
# only ensures a bare `docker run <image>` (no args) still prints help and
# that a serve invocation missing its checkpoint paths fails fast with a
# clear, mount-oriented error instead of a bare usage line.
#
# Back-compat: `docker run ... <image> --weights ... --tokenizer ... [...]`
# keeps working exactly as before — every argument is passed through to
# kokkosdnn_server untouched.
set -eu

BIN=/usr/local/bin/kokkosdnn_server

has_opt() {
  # has_opt NAME -- "$@" : true if NAME appears as an argument.
  _want=$1
  shift
  # Drop the literal "--" separator if the caller passed one.
  if [ "${1:-}" = "--" ]; then shift; fi
  for _a in "$@"; do
    if [ "$_a" = "$_want" ]; then return 0; fi
  done
  return 1
}

if [ "$#" -eq 0 ]; then
  # CMD is ["--help"], so this only triggers if the caller overrides CMD
  # with an empty command; print the arch-aware server help either way.
  exec "$BIN" --help
fi

if has_opt --help -- "$@" || has_opt -h -- "$@" || has_opt --version -- "$@"; then
  exec "$BIN" "$@"
fi

_missing=""
if ! has_opt --weights -- "$@"; then
  _missing="$_missing --weights"
fi
if ! has_opt --tokenizer -- "$@"; then
  _missing="$_missing --tokenizer"
fi
if [ -n "$_missing" ]; then
  cat >&2 <<EOF
error: missing required checkpoint flag(s):$_missing
Mount the weights into the container and pass both paths, e.g.:

  LFM2.5 (GGUF):
    docker run --gpus all -p 8000:8000 \\
      -v /path/to/lfm:/models/lfm:ro \\
      <image> --weights /models/lfm/lfm2_5.gguf \\
              --tokenizer /models/lfm/tokenizer.json

  Glimmer (sharded safetensors, ~60 GiB, shell glob expanded by the server):
    docker run --gpus all -p 8000:8000 \\
      -v /path/to/glimmer:/models/glimmer:ro \\
      <image> --arch glimmer --model glimmer-30b \\
              --weights '/models/glimmer/model-*.safetensors' \\
              --tokenizer /models/glimmer/tokenizer.json \\
              --config /models/glimmer/config-glimmer-30b-text.json

  Glimmer (single GGUF file):
    docker run --gpus all -p 8000:8000 \\
      -v /path/to/glimmer:/models/glimmer:ro \\
      <image> --arch glimmer --model glimmer-30b \\
              --weights /models/glimmer/model.gguf \\
              --tokenizer /models/glimmer/tokenizer.json

On server builds that predate --arch, drop the '--arch glimmer' flag and
serve Glimmer with --model glimmer-30b (LFM2.5 needs no --arch flag).
See 'kokkosdnn_server --help' for the full flag list.
EOF
  exit 2
fi

exec "$BIN" "$@"
