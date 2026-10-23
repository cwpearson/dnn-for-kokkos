Getting started
===============

Requirements
------------

- CMake ≥ 3.22, C++20 compiler
- An installed Kokkos >= 5.2.1 (Serial is enough for tests; CUDA backend for GPU)
- Optional: CUDA Toolkit (for ``KokkosDNN_ENABLE_CUBLAS``), cuDNN (for
  ``KokkosDNN_ENABLE_CUDNN``)
- Supported platforms: Linux and macOS (tested in CI on ``ubuntu-latest`` /
  ``ubuntu-24.04`` and ``macos-latest``). Windows is untested and there are
  no shared-library export macros, so ``BUILD_SHARED_LIBS`` targets
  Linux/macOS.

Build
-----

.. code-block:: bash

   cmake -S . -B build \
     -DKokkos_ROOT=/path/to/kokkos-install \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel 8
   ctest --test-dir build --output-on-failure

Configure options
-----------------

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Effect
   * - ``KokkosDNN_ENABLE_TESTS``
     - ON
     - build ``tests/``
   * - ``KokkosDNN_ENABLE_EXAMPLES``
     - ON
     - build ``examples/``
   * - ``KokkosDNN_ENABLE_BENCHMARKS``
     - OFF
     - build ``benchmarks/kokkosdnn_bench`` (Google Benchmark, fetched if needed)
   * - ``KokkosDNN_ENABLE_CUBLAS``
     - OFF
     - GEMM via cuBLAS on CUDA spaces (incl. fused GEMM+bias via cublasLt)
   * - ``KokkosDNN_ENABLE_CUDNN``
     - OFF
     - attention + convolutions via cuDNN (with automatic native fallback;
       needs ``<cudnn.h>`` + ``cudnn-frontend``, fetched automatically)
   * - ``KokkosDNN_ENABLE_HIPBLAS``
     - OFF
     - GEMM via hipBLAS on HIP spaces
   * - ``KokkosDNN_ENABLE_MIOPEN``
     - OFF
     - convolutions via MIOpen on HIP spaces (attention path falls back to
       native; no fused SDPA in MIOpen)
   * - ``KokkosDNN_ENABLE_ACCELERATE``
     - OFF
     - GEMM/GEMV via Apple Accelerate ``cblas_sgemm`` on host spaces
       (float only; native fallback otherwise)
   * - ``KokkosDNN_ENABLE_TEST_DISCOVERY``
     - ON
     - register gtest tests via ``gtest_discover_tests`` (set OFF for
       GPU-less build-only CI, where test binaries cannot run at build time)
   * - ``KokkosDNN_ENABLE_IMAGE``
     - OFF
     - optional image component (``KokkosDNN::image``, stb-backed I/O + preprocessing)
   * - ``KokkosDNN_ENABLE_MODEL_LFM25``
     - ON
     - build LFM2.5 model component (``KokkosDNN::lfm25``)
   * - ``KokkosDNN_ENABLE_MODEL_MOSS``
     - ON
     - build MOSS-Image model component (``KokkosDNN::moss``)
   * - ``KokkosDNN_ENABLE_SERVER``
     - OFF
     - build OpenAI-compatible HTTP server (``kokkosdnn_server``; requires
       ``KokkosDNN_ENABLE_MODEL_LFM25=ON``)
   * - ``KokkosDNN_ENABLE_COMPILER_WARNINGS``
     - OFF
     - enable compiler warnings as errors (mirroring
       ``Kokkos_ENABLE_COMPILER_WARNINGS``)
   * - ``BUILD_SHARED_LIBS``
     - OFF
     - build shared libraries (Linux/macOS only; Windows is untested)

Building these docs
-------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build docs docs/_build/html
