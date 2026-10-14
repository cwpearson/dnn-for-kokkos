Training: losses and optimizers
================================

Losses — header: ``src/KokkosDNN_Loss.hpp``. Binary cross-entropy with
logits over ``[B, Out]`` logits/targets (XOR dense outputs feed in
directly), FP32, mean reduction.

.. cpp:function:: template <typename ExecSpace, typename Scalar> float bce_with_logits_loss(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& targets, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void bce_with_logits_backward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& targets, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dlogits, const std::string& label)

Optimizers — header: ``src/KokkosDNN_Optimizer.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void sgd_step(const Kokkos::View<Scalar*, ExecSpace>& param, const Kokkos::View<Scalar*, ExecSpace>& grad, float lr, const std::string& label)

.. cpp:function:: template <typename ExecSpace, typename Scalar> void adam_step(const Kokkos::View<Scalar*, ExecSpace>& param, const Kokkos::View<Scalar*, ExecSpace>& grad, const Kokkos::View<float*, ExecSpace>& m, const Kokkos::View<float*, ExecSpace>& v, float lr, float beta1, float beta2, float eps, float t, const std::string& label)

   SGD (plain + momentum overloads) and Adam (bias-corrected, FP32
   moments). Rank-1 and rank-2 overloads exist; see the header.
