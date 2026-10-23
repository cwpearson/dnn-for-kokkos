// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_SWIGLU_HPP
#define KOKKOSDNN_SWIGLU_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

namespace KokkosDNN {

KOKKOS_INLINE_FUNCTION float silu(float x) {
  return x / (1.0f + expf(-x));
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar silu_cast(Scalar x) {
  float f = static_cast<float>(x);
  return static_cast<Scalar>(f / (1.0f + expf(-f)));
}

// SwiGLU MLP activation: out[i,:] = silu(gate[i,:]) * up[i,:].
// gate, up, out: [tokens, ffn_dim]. Scalar-agnostic; computed in FP32.
template <class ExecSpace, class Scalar>
void swiglu(const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
            const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& up,
            const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
            const std::string& label = "KokkosDNN::swiglu") {
  const size_t T = gate.extent(0), F = gate.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, F}),
      KOKKOS_LAMBDA(size_t t, size_t f) {
        float g = static_cast<float>(gate(t, f));
        out(t, f) =
            static_cast<Scalar>((g / (1.0f + expf(-g))) *
                                static_cast<float>(up(t, f)));
      });
}

}  // namespace KokkosDNN

#endif
