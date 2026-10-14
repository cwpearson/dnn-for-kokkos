API reference
=============

All public entry points live in namespace ``KokkosDNN`` and are declared in
``src/KokkosDNN_*.hpp``. The umbrella header ``src/KokkosDNN.hpp`` pulls in
every core component (plus ``images/KokkosDNN_Image.hpp`` when
``KokkosDNN_ENABLE_IMAGE`` is on).

Conventions
-----------

* Every op is a function template on ``ExecSpace`` (any Kokkos execution
  space) and usually on ``Scalar`` (``float`` or
  ``Kokkos::Experimental::bhalf_t``). Accumulation inside kernels is FP32;
  results are cast back to ``Scalar``.
* 2-D data uses row-major views:
  ``Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>``.
* Most ops take a trailing ``label`` (kernel label for profiling).
* Ops that can dispatch to a CUDA library take
  ``const DNNHandle<ExecSpace>& handle`` as their first argument; on
  non-CUDA builds (or with the TPL disabled) the handle is ignored and the
  native kernel runs.
* Non-public helpers live in ``KokkosDNN::Impl`` and are not documented
  here; see the headers for tuning notes.

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/handle
   api/linalg
   api/normalization
   api/convolution
   api/attention
   api/diffusion
   api/embeddings
   api/activations
   api/training
   api/sampling
   api/image
