# kokkosdnn_server: CUDA-enabled inference server image (multi-stage).
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
# Run (needs the NVIDIA container toolkit and a CUDA-12.8-capable driver;
# mount the GGUF weights + tokenizer into the container):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/weights:/models:ro \
#     kokkosdnn-server:cuda \
#     --weights /models/lfm2_5.gguf --tokenizer /models/tokenizer.json
#
# Cross-GPU note: KOKKOS_ARCH selects the Kokkos CUDA arch flag
# (-DKokkos_ARCH_<arch>=ON). The default AMPERE80 build also carries PTX, so
# the driver JIT-compiles it forward onto newer GPUs (Hopper/Blackwell);
# pass --build-arg KOKKOS_ARCH=HOPPER90 (etc.) to target a GPU natively.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------
# Builder: full CUDA toolkit + cuDNN headers, Kokkos + server compiled here.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

# Re-declare stage-local build args (ARGs reset at each FROM).
ARG KOKKOS_VERSION=5.2.1
ARG KOKKOS_ARCH=AMPERE80

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
      -DKokkos_ARCH_${KOKKOS_ARCH}=ON \
      -DKokkos_ENABLE_TESTS=OFF \
 && cmake --build kokkos-build --parallel "$(nproc)" \
 && cmake --install kokkos-build

# Server only: tests/examples/benchmarks stay OFF to keep the image lean.
# TPL dispatch (cuBLAS GEMM, cuDNN SDPA) is ON — the cudnn-devel image above
# ships both. cudnn-frontend / cpp-httplib / nlohmann_json are fetched at
# configure time, so this step needs network access.
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
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

COPY --from=builder /opt/kokkosdnn/bin/kokkosdnn_server /usr/local/bin/kokkosdnn_server

# httplib links OpenSSL/zlib, so make sure the runtime image carries them
# (tiny, no daemons). All other NEEDED entries (libcudart/libcublas/
# libcudnn/libnvrtc, libstdc++) come from the base image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libssl3 zlib1g \
 && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
ENTRYPOINT ["kokkosdnn_server"]
CMD ["--help"]

# Loader check against the runtime image (libcuda excluded: injected by the
# NVIDIA toolkit once the container runs with --gpus).
RUN if ldd /usr/local/bin/kokkosdnn_server | grep -v "libcuda.so" | grep -q "not found"; then \
      ldd /usr/local/bin/kokkosdnn_server; \
      echo "missing shared libraries in runtime image"; exit 1; \
    fi
