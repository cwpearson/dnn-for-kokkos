Diffusion conditioning
=======================

Header: ``src/KokkosDNN_Diffusion.hpp``.

.. cpp:function:: template <typename ExecSpace, typename Scalar> void adaln_modulate_forward(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x, const Kokkos::View<Scalar*, ExecSpace>& shift, const Kokkos::View<Scalar*, ExecSpace>& scale, const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, const std::string& label)

   AdaLN modulation ``y = x * (1 + scale) + shift``. Rank-3 ``[B, T, D]``
   overloads with ``[B, D]`` or ``[B, 1, D]`` shift/scale broadcast over
   tokens also exist.

.. cpp:function:: template <typename ExecSpace> void timestep_sinusoidal(const Kokkos::View<float*, ExecSpace>& t, const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& out, const std::string& label)

   Sinusoidal timestep embedding ``t [N] -> out [N, F]`` (``F`` even;
   cos/sin halves). Frequencies are recomputed in-kernel, no temporaries.
