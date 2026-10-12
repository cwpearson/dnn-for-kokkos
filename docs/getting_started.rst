Getting started
===============

Requirements
------------

- CMake ≥ 3.22, C++20 compiler
- An installed Kokkos (Serial is enough for tests; CUDA backend for GPU)
- Optional: CUDA Toolkit (for ``KokkosDNN_ENABLE_CUBLAS``), cuDNN (for
  ``KokkosDNN_ENABLE_CUDNN``)

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
     - GEMM via cuBLAS on CUDA spaces
   * - ``KokkosDNN_ENABLE_CUDNN``
     - OFF
     - attention via cuDNN SDPA (with automatic native fallback)
   * - ``KokkosDNN_ENABLE_IMAGE``
     - OFF
     - optional image component (``KokkosDNN::Image``, stb-backed I/O + preprocessing)

Building these docs
-------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build docs docs/_build/html
