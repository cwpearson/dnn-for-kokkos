Convolutions
============

Headers: ``src/KokkosDNN_Conv.hpp``,
``src/KokkosDNN_DepthwiseConv.hpp``. Token-row ``[T, C]`` views with
``T = H*W`` row-major, stride 1, symmetric zero padding, FP32
accumulation. Bias is ``[Cout]``/``[C]`` or empty (extent 0) for no bias.
Dense 3x3, depthwise, and grouped 1x1 dispatch to cuDNN convolution
forward on CUDA builds with ``KokkosDNN_ENABLE_CUDNN`` (FP32 stays FP32;
see the header's TPL-status note), MIOpen on HIP builds with
``KokkosDNN_ENABLE_MIOPEN``, and otherwise run the native lowering /
kernels. The ``handle`` is still taken for API uniformity.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void conv3x3_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H, size_t W, const std::string& label)

   Dense 3x3, pad 1: ``x [T, Cin]``, ``w [Cout, Cin, 3, 3]``,
   ``y [T, Cout]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void depthwise_conv_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t K, size_t H, size_t W, const std::string& label)

   Depthwise ``KxK`` (``K`` odd), pad ``K/2``: ``x/y [T, C]``,
   ``w [C, K, K]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void grouped_conv1x1_forward(const DNNHandle<ExecSpace>& handle, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t groups, const std::string& label)

   Grouped 1x1, no padding: ``x [T, Cin]``, ``w [Cout, Cg]``
   (``Cg = Cin / groups``), ``y [T, Cout]``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void depthwise_conv3x3_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H, size_t W, const std::string& label)

   Depthwise 3x3 convenience op (header
   ``src/KokkosDNN_DepthwiseConv.hpp``, no handle): ``x/y [T, C]``,
   ``w [C, 3, 3]``; a batched ``[B, T, C]`` overload also exists.
