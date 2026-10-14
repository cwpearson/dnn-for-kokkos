Normalization and softmax
=========================

RMSNorm — header: ``src/KokkosDNN_RMSNorm.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Rank-2 ``[T, D]`` form; a batched rank-3 ``[B, T, D]`` overload with the
   same per-row math also exists.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_row(const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_noscale(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_row_noscale(const Kokkos::View<Scalar*, ExecSpace>& x, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void rmsnorm_one_plus(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& w, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Gemma2 variant: ``y = x / sqrt(mean(x^2) + eps) * (1 + w)`` (zero-init
   weights, HF ``Gemma2RMSNorm``).

LayerNorm — header: ``src/KokkosDNN_LayerNorm.hpp``. TeamPolicy kernels (one
team per normalized row).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar*, ExecSpace>& beta, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_row(const Kokkos::View<Scalar*, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar*, ExecSpace>& beta, float eps, const Kokkos::View<Scalar*, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layernorm_noaffine_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float eps, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Affine-free LayerNorm, rank-2 ``[T, D]`` form; a rank-3 ``[B, T, D]``
   overload with the same per-row math also exists.

Softmax — header: ``src/KokkosDNN_Softmax.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void softmax_rows(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void softmax_rows_causal(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

LayerScale — header: ``src/KokkosDNN_LayerScale.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layerscale_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Elementwise ``y = x * gamma`` (batched, single-sequence, and residual
   overloads exist; see the header).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void layerscale_residual_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& residual, const Kokkos::View<Scalar*, ExecSpace>& gamma, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Rank-1 ``[D]`` and batched rank-3 ``[B, T, D]`` overloads exist for both
   entry points; see the header.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void scale_by_scalar(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, float scalar, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

   Uniform ``out = scalar * x`` (e.g. Gemma4 per-layer ``layer_scalar``);
   distinct from LayerScale's per-channel ``[D]`` multiply. Rank-1
   ``[D]`` overload also exists.
