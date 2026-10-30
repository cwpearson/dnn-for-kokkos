# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# kokkosdnn_server: CUDA-enabled inference server image (multi-stage).
#
# One image serves EITHER LFM2.5 OR Glimmer: the builder compiles every
# model component the server can dispatch to, and the checkpoint (GGUF or
# sharded safetensors) is volume-mounted at *run* time, so the arch is
# selected per `docker run` invocation, not per image build.
#
# Stage 1 (builder) compiles Kokkos (CUDA backend) + kokkosdnn_server in an
# NVIDIA CUDA *devel* image. Stage 2 (runtime) ships only the installed
# server binary on the matching *runtime* image, which provides the CUDA /
# cuBLAS / cuDNN shared libraries the binary loads. The two FROM lines must
# stay on the same CUDA/Ubuntu version so the userspace libraries match the
# driver ABI the container runs against.
#
# Build from the repository root:
#   docker build -f server/Dockerfile -t kokkosdnn-server:cuda .
#
# Run LFM2.5 (GGUF checkpoint + tokenizer.json mounted into the container):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/lfm:/models/lfm:ro \
#     kokkosdnn-server:cuda \
#     --weights /models/lfm/lfm2_5.gguf --tokenizer /models/lfm/tokenizer.json
#
# Run Glimmer (sharded safetensors, ~60 GiB BF16; quote the glob so the
# server — which expands it with POSIX glob(3), no extra image deps — sees
# it unexpanded; needs the --arch server build, else drop --arch):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/glimmer:/models/glimmer:ro \
#     kokkosdnn-server:cuda \
#     --arch glimmer --model glimmer-30b \
#     --weights '/models/glimmer/model-*.safetensors' \
#     --tokenizer /models/glimmer/tokenizer.json \
#     --config /models/glimmer/config-glimmer-30b-text.json
#
# Run Glimmer from a single llama.cpp GGUF file instead:
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/glimmer:/models/glimmer:ro \
#     kokkosdnn-server:cuda \
#     --arch glimmer --model glimmer-30b \
#     --weights /models/glimmer/model.gguf \
#     --tokenizer /models/glimmer/tokenizer.json
#
# With no args the image prints the server help (`--arch`-aware on new
# builds). Both -v layouts can be mounted at once; only the --weights /
# --tokenizer paths passed on the command line decide what is served.
#
# Crash backtraces: run with `-e KOKKOSDNN_GDB=1` to execute the server
# under `gdb --batch` (installed in the runtime image). A crash then dumps
# `thread apply all bt full` for every thread to the container logs; unset
# (or `0`) runs the binary directly. Example:
#   docker run --gpus all -p 8000:8000 -e KOKKOSDNN_GDB=1 \
#     -v /path/to/lfm:/models/lfm:ro \
#     kokkosdnn-server:cuda \
#     --weights /models/lfm/lfm2_5.gguf --tokenizer /models/lfm/tokenizer.json
#
# Cross-GPU note: CUDA_ARCH is a Kokkos CUDA arch name
# (--build-arg CUDA_ARCH=HOPPER90, default AMPERE80) passed straight to
# -DKokkos_ARCH_<arch>=ON. The default Ampere build also carries PTX, so
# the driver JIT-compiles it forward onto newer GPUs (Hopper/Blackwell);
# pass --build-arg CUDA_ARCH=HOPPER90 (etc.) to target a GPU natively.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG CUDA_ARCH=AMPERE80

# ---------------------------------------------------------------------------
# Builder: full CUDA toolkit + cuDNN headers, Kokkos + server compiled here.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

# Re-declare stage-local build args (ARGs reset at each FROM).
ARG KOKKOS_VERSION=5.2.1
ARG CUDA_ARCH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates cmake g++ git libstb-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Kokkos, mirroring .github/workflows/cuda.yaml: shallow release checkout
# (recursive: the build needs desul/mdspan), CUDA + Serial backends.
RUN git clone --depth 1 --branch ${KOKKOS_VERSION} \
      --recursive https://github.com/kokkos/kokkos.git kokkos-src \
 && cmake -S kokkos-src -B kokkos-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/kokkos \
      -DCMAKE_CXX_COMPILER=/src/kokkos-src/bin/nvcc_wrapper \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ARCH_${CUDA_ARCH}=ON \
      -DKokkos_ENABLE_TESTS=OFF \
 && cmake --build kokkos-build --parallel "$(nproc)" \
 && cmake --install kokkos-build

# Server only: tests/examples/benchmarks stay OFF to keep the image lean.
# TPL dispatch (cuBLAS GEMM, cuDNN SDPA) is ON — the cudnn-devel image above
# ships both. cudnn-frontend / cpp-httplib / nlohmann_json are fetched at
# configure time, so this step needs network access.
#
# Both model components are enabled so the one image can serve either arch:
# LFM2.5 (today's server) and Glimmer (the --arch server refactor / the
# models/glimmer promotion). -D flags for options that do not exist yet are
# harmless CMake cache entries, so -DKokkosDNN_ENABLE_MODEL_GLIMMER=ON is a
# no-op until that option lands and must NOT be removed as "unused".
COPY . /src/kokkosdnn
RUN cmake -S /src/kokkosdnn -B /src/kokkosdnn-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/src/kokkos-src/bin/nvcc_wrapper \
      -DCMAKE_INSTALL_PREFIX=/opt/kokkosdnn \
      -DKokkos_ROOT=/opt/kokkos \
      -DKokkosDNN_ENABLE_TESTS=OFF \
      -DKokkosDNN_ENABLE_EXAMPLES=OFF \
      -DKokkosDNN_ENABLE_BENCHMARKS=OFF \
      -DKokkosDNN_ENABLE_SERVER=ON \
      -DKokkosDNN_ENABLE_MODEL_LFM25=ON \
      -DKokkosDNN_ENABLE_MODEL_GLIMMER=ON \
      -DKokkosDNN_ENABLE_CUBLAS=ON \
      -DKokkosDNN_ENABLE_CUDNN=ON \
 && cmake --build /src/kokkosdnn-build --parallel "$(nproc)" \
 && cmake --install /src/kokkosdnn-build

# Loader check: every shared library the server needs must resolve inside
# this (devel) image. libcuda.so.1 is excluded: it is injected at container
# run time by the NVIDIA toolkit, so it cannot resolve during a GPU-less
# build. (--help is not executed here for the same reason: the loader would
# refuse to start the binary without libcuda. The image's default CMD is
# --help, so `docker run --gpus all ... kokkosdnn-server:cuda` smoke-tests
# startup on a GPU host.)
RUN test -x /opt/kokkosdnn/bin/kokkosdnn_server \
 && if ldd /opt/kokkosdnn/bin/kokkosdnn_server | grep -v "libcuda.so" | grep -q "not found"; then \
      ldd /opt/kokkosdnn/bin/kokkosdnn_server; \
      echo "kokkosdnn_server has unresolvable shared libraries"; exit 1; \
    fi

# ---------------------------------------------------------------------------
# Runtime: same CUDA/cuDNN version, runtime libraries only. The server
# binary is statically linked against Kokkos/kokkosdnn, so only this one
# file is copied over; libcudart/libcublas/libcudnn/libnvrtc resolve from
# the base image and libcuda.so.1 is injected by the NVIDIA runtime.
# Glimmer's sharded-safetensors glob expansion lives in the binary itself
# (POSIX glob(3) from libc — no extra packages needed).
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

COPY --from=builder /opt/kokkosdnn/bin/kokkosdnn_server /usr/local/bin/kokkosdnn_server
COPY server/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# httplib links OpenSSL/zlib, so make sure the runtime image carries them
# (tiny, no daemons). gdb is included so -e KOKKOSDNN_GDB=1 can run the
# server under `gdb --batch` with `thread apply all bt full` on crash (see
# docker-entrypoint.sh). All other NEEDED entries (libcudart/libcublas/
# libcudnn/libnvrtc, libstdc++) come from the base image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gdb libssl3 zlib1g \
 && rm -rf /var/lib/apt/lists/* \
 && chmod 755 /usr/local/bin/docker-entrypoint.sh \
 && mkdir -p /models/lfm /models/glimmer

EXPOSE 8000
# The entrypoint passes everything through to kokkosdnn_server, so
# `docker run ... <image> --weights ...` keeps working as before; with no
# args (CMD) it prints the server help. Validates --weights/--tokenizer
# early with a mount-oriented error (see the script). With -e
# KOKKOSDNN_GDB=1 the entrypoint execs the server under gdb --batch with
# `thread apply all bt full` on crash.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]

# Loader check against the runtime image (libcuda excluded: injected by the
# NVIDIA toolkit once the container runs with --gpus).
RUN if ldd /usr/local/bin/kokkosdnn_server | grep -v "libcuda.so" | grep -q "not found"; then \
      ldd /usr/local/bin/kokkosdnn_server; \
      echo "missing shared libraries in runtime image"; exit 1; \
    fi
