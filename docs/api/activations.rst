Activations
===========

SwiGLU — header: ``src/KokkosDNN_SwiGLU.hpp``.

.. cpp:function:: float silu(float x)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void swiglu(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

Pointwise activations — header: ``src/KokkosDNN_Activation.hpp``.

``gelu_forward`` (``GeluVariant::ExactErf`` / ``::Tanh``),
``silu_forward``, ``sigmoid_forward`` / ``sigmoid_backward``,
``tanh_forward`` / ``tanh_backward``, ``relu_forward`` /
``relu_backward``, ``logit_softcap_forward``, and ``gated_gelu_multiply``. All are
``template <class ExecSpace, class Scalar>`` over 2-D (and row-wise
1-D) views with a trailing label. ``silu_forward`` computes ``silu``
elementwise (see ``src/KokkosDNN_SwiGLU.hpp``); ``gated_gelu_multiply``
defaults to ``GeluVariant::Tanh`` (Gemma MLP).
