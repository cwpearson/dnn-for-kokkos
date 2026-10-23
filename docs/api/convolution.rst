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

Causal depthwise Conv1D — header: ``src/KokkosDNN_Conv1D.hpp``.
Mamba-style sequence mixers (e.g. Qwen3-Next / Gemma-SSM): one
length-``K`` kernel per channel with ``K-1`` left zero padding, so
``y[t]`` only sees inputs ``[<=t]``. Scalar-agnostic, FP32
accumulation, portable kernels, host-friendly. Bias is ``[C]`` or
empty (extent 0) for none.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void depthwise_conv1d_causal_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   Full-sequence prefill: ``x/y [T, C]``, ``w [C, K]`` (``K >= 1``).

.. cpp:function:: template <typename ExecSpace, typename Scalar> void depthwise_conv1d_causal_step(const Kokkos::View<Scalar*, ExecSpace>& x_t, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cache, const Kokkos::View<Scalar*, ExecSpace>& y_t, const std::string& label)

   Single-token streaming decode: ``x_t/y_t [C]`` with an explicit
   ``cache [C, K-1]`` history (oldest-first, updated in place; for
   ``K == 1`` the cache has extent 0 and is untouched). Stepping over
   a whole sequence with a zero-initialized cache reproduces
   :cpp:func:`depthwise_conv1d_causal_forward` exactly. ``x_t`` and
   ``y_t`` must not alias.

Dense/general Conv1D — header: ``src/KokkosDNN_Conv1DGeneral.hpp``.
Dense 1D convolution and transposed 1D convolution over time-major
``[T, C]`` activations (the StyleTTS2/Kokoro convolutional stack:
plain, dilated, grouped, and transposed forms; weight-norm is fused
on the host at load time so every kernel sees plain dense weights).
Scalar-agnostic, FP32 accumulation, portable kernels. Bias is
``[Cout]`` or empty (extent 0) for none.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void conv1d_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t stride, size_t padding, size_t dilation, size_t groups, const std::string& label)

   ``y[t,o] = b[o] + sum w[o,i,k] * x[t*s + k*d - p, i]`` with
   ``w [Cout, Cg, K]`` (``Cg = Cin/groups``; ``groups=1`` dense,
   ``groups=Cin=Cout`` depthwise with ``w [C, 1, K]``) and
   ``Lout = floor((Lin + 2p - d*(K-1) - 1)/s + 1)``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void conv_transpose1d_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w, const Kokkos::View<Scalar*, ExecSpace>& b, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t stride, size_t padding, size_t output_padding, size_t groups, const std::string& label)

   PyTorch ``ConvTranspose1d`` semantics with
   ``w [Cin, Cout/groups, K]`` and
   ``Lout = (Lin-1)*s - 2p + K + op`` (``output_padding`` extends
   ``Lout`` with trailing bias-only rows).
