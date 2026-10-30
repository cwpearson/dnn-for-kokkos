#!/bin/sh
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
#
# Debug: `docker run -e KOKKOSDNN_GDB=1 ...` runs the server under
# `gdb --batch` so a crash dumps `thread apply all bt full` for every
# thread, plus the registers/disassembly that identify a null indirect
# call (see run_bin below).
#
# `docker run -e KOKKOSDNN_SANITIZER=1 ...` runs it under
# `compute-sanitizer` instead (memcheck by default) to catch out-of-bounds
# / misaligned / leaked device accesses and CUDA API errors. The two are
# MUTUALLY EXCLUSIVE and setting both is an error: compute-sanitizer
# injects into the target and drives it over its own frontend/target
# channel, and its `--mode attach` explicitly requires an application "to
# which no other tool is attached". Composing it with gdb yields results
# from neither tool. Knobs: KOKKOSDNN_SANITIZER_TOOL (default memcheck;
# racecheck/initcheck/synccheck also valid), KOKKOSDNN_SANITIZER_ARGS
# (extra raw compute-sanitizer options).
set -eu

# GPU sanity first: log driver/GPU state before anything else so the
# container logs always show what the CUDA runtime will see. A failure here
# means no usable GPU was injected (e.g. rootless podman needs
# --device nvidia.com/gpu=all, not --gpus all). Warn and continue so
# --help/--version still work on GPU-less hosts.
if ! nvidia-smi; then
  echo "warning: nvidia-smi failed; no usable NVIDIA GPU in this container" >&2
fi

BIN=/usr/local/bin/kokkosdnn_server

# is_on VALUE : true for a non-empty value other than "0".
is_on() {
  [ -n "${1:-}" ] && [ "$1" != "0" ]
}

# run_bin "$@" : exec the server, optionally under gdb or compute-sanitizer.
# The two debug modes are mutually exclusive (see the header comment): both
# take over the target process, so combining them reports garbage.
run_bin() {
  if is_on "${KOKKOSDNN_GDB:-}" && is_on "${KOKKOSDNN_SANITIZER:-}"; then
    cat >&2 <<'EOF'
error: KOKKOSDNN_GDB and KOKKOSDNN_SANITIZER are mutually exclusive.
compute-sanitizer injects into the target and drives it over its own
frontend/target channel, and its --mode attach requires an application to
which no other tool is attached; running it under gdb yields results from
neither. Pick one:
  -e KOKKOSDNN_GDB=1        crash backtrace + null-call registers
  -e KOKKOSDNN_SANITIZER=1  memcheck for device OOB/misaligned/leak
EOF
    exit 2
  fi

  # compute-sanitizer on a 30B model is slow (memcheck serializes and
  # instruments every access: expect 10-100x, i.e. minutes for a short
  # prompt). Source-level attribution additionally needs the binary built
  # with -lineinfo or -G; a RelWithDebInfo build gives it.
  if is_on "${KOKKOSDNN_SANITIZER:-}"; then
    if ! command -v compute-sanitizer >/dev/null 2>&1; then
      cat >&2 <<'EOF'
error: KOKKOSDNN_SANITIZER is set but compute-sanitizer is not installed.
It ships with the CUDA toolkit, not with the cudnn-runtime base image.
Rebuild with -DKokkosDNN_ENABLE_SERVER=ON and
  --build-arg INSTALL_COMPUTE_SANITIZER=ON
or run the builder (devel) image instead.
EOF
      exit 127
    fi
    _tool="${KOKKOSDNN_SANITIZER_TOOL:-memcheck}"
    echo "compute-sanitizer: tool=$_tool (expect 10-100x slowdown; use a short prompt)" >&2
    # KOKKOSDNN_SANITIZER_ARGS is intentionally word-split so callers can
    # pass several raw options, e.g. --force-blocking-launches --padding 32.
    # shellcheck disable=SC2086
    exec compute-sanitizer \
      --tool "$_tool" \
      --print-limit 25 \
      --error-exitcode 99 \
      ${KOKKOSDNN_SANITIZER_ARGS:-} \
      "$BIN" "$@"
  fi

  if is_on "${KOKKOSDNN_GDB:-}"; then
    if ! command -v gdb >/dev/null 2>&1; then
      echo "error: KOKKOSDNN_GDB is set but gdb is not installed in the image" >&2
      exit 127
    fi
    # --return-child-result propagates the server's exit code through gdb.
    # The handle lines keep image-stop signals (docker stop, Ctrl-C) and
    # SIGPIPE (httplib socket writes) flowing to the server instead of
    # stopping inside gdb. On normal exit the trailing bt prints
    # "No stack." and is harmless; on a crash it dumps all threads.
    #
    # Post-mortem capture beyond `bt full`: a SIGSEGV with `#0 0x0` means an
    # indirect call through a null pointer, and `bt` alone cannot say which
    # one (nvcc's generated lambda boilerplate carries no source lines).
    # `up` moves to the caller of the faulting frame, whose $pc is the return
    # address, so the disassembly around it shows the `call *%reg` that
    # faulted; `info registers` then shows whether that register (or the
    # memory operand) was zero. `info sharedlibrary` catches a missing or
    # version-skewed CUDA library (e.g. libnvJitLink, dlopen'd, so an ldd
    # check at build time cannot see it). Each -ex is independent: gdb
    # reports an error for one that does not apply and continues.
    exec gdb --batch --return-child-result \
      -ex "set pagination off" \
      -ex "set print frame-arguments all" \
      -ex "handle SIGINT nostop noprint pass" \
      -ex "handle SIGTERM nostop noprint pass" \
      -ex "handle SIGPIPE nostop noprint pass" \
      -ex run \
      -ex "thread apply all bt full" \
      -ex "info sharedlibrary" \
      -ex "info registers" \
      -ex up \
      -ex 'info symbol $pc' \
      -ex 'x/12i $pc-40' \
      --args "$BIN" "$@"
  fi
  exec "$BIN" "$@"
}

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
  run_bin --help
fi

if has_opt --help -- "$@" || has_opt -h -- "$@" || has_opt --version -- "$@"; then
  run_bin "$@"
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

run_bin "$@"
