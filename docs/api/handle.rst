Handle and configuration
========================

Header: ``src/KokkosDNN_Handle.hpp``.

.. cpp:class:: template <typename ExecSpace> DNNHandle

   Records which third-party dispatches are active (``use_cublas``,
   ``use_cudnn``, ``use_hipblas``, ``use_miopen``, ``use_accelerate``)
   and lazily caches the cuBLAS/cuDNN/hipBLAS/MIOpen handles
   (Accelerate is stateless). Non-copyable, movable. Native Kokkos
   paths ignore it, so call sites stay free of ``#ifdef``\ s.

   .. cpp:function:: void* cublas_raw() const

      Lazily created cached cuBLAS handle, or ``nullptr`` when unusable
      (non-CUDA space, TPL disabled, creation failure). Callers fall back
      to the native kernel then.

   .. cpp:function:: void* cudnn_raw() const

      Lazily created cached cuDNN handle, same ``nullptr`` contract as
      above.

.. cpp:function:: void print_configuration(std::ostream& os)

   Prints the library configuration (enabled TPLs, Kokkos backend info).
   Declared in ``src/KokkosDNN.hpp``, defined in
   ``src/KokkosDNN_PrintConfiguration.cpp``.

TPL dispatch notes
------------------

* ``src/tpl/cublas/`` — GEMM dispatch to cuBLAS; handle lifecycle in
  ``KokkosDNN_CuBLAS_Handle.hpp``.
* ``src/tpl/cudnn/`` — attention dispatch to cuDNN SDPA graph API with
  automatic native fallback; handle lifecycle in
  ``KokkosDNN_CuDNN_Handle.hpp``.
* ``src/tpl/accelerate/`` — GEMM/GEMV dispatch to Apple Accelerate
  ``cblas_sgemm`` on host spaces (float only, synchronous, no handle);
  ``KokkosDNN_CuBLAS_Handle.hpp``-style lifecycle not needed.
