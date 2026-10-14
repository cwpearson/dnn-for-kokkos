Linear algebra
==============

GEMM / GEMV — header: ``src/KokkosDNN_Gemm.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemm(const DNNHandle<ExecSpace>& handle, Scalar alpha, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, Scalar beta, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label = "KokkosDNN::gemm")

   ``Y = alpha * X * W^T + beta * Y`` with ``X : [M, K]``, ``W : [N, K]``
   (HuggingFace Linear layout), ``Y : [M, N]``. Dispatch ladder on CUDA:
   cuBLAS (``sgemm`` for ``float``, ``cublasGemmEx`` BF16 I/O with FP32
   accumulate for ``bhalf_t``) → shared-memory tiled native kernel →
   naive MDRange kernel on non-CUDA backends.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemv(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label = "KokkosDNN::gemv")

   Rank-1 decode variant: ``y[n] = sum_k x[k] * W[n, k]``. Shares
   :cpp:func:`gemm`'s dispatch ladder (used by the LM head).

Bias add and fused GEMM+bias — header: ``src/KokkosDNN_GemmBias.hpp``.
The rank-1/rank-2 bias adds are ``KokkosDNN::Impl`` helpers
(``Impl::add_bias_2d`` / ``Impl::add_bias_1d``); the public entry points are
the fused ops below. An empty bias view (extent 0) selects the unbiased path.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemm_bias(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

   ``Y = X * W^T + b`` (bias empty → unbiased). GEMM part reuses
   :cpp:func:`gemm`'s dispatch ladder.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void gemv_bias(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

Dense layer — header: ``src/KokkosDNN_Dense.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void dense_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y, const std::string& label)

   ``Y = X * W^T + b`` (bias is required here; use :cpp:func:`gemm_bias`
   for the optional-bias form). Forward reuses :cpp:func:`gemm`.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void dense_backward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dY, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dX, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dW, const Kokkos::View<Scalar*, ExecSpace>& db, const std::string& label)
