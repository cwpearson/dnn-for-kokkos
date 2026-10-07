#ifndef KOKKOSDNN_SWIGLU_HPP
#define KOKKOSDNN_SWIGLU_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

namespace KokkosDNN {

KOKKOS_INLINE_FUNCTION float silu(float x) {
  return x / (1.0f + expf(-x));
}

// SwiGLU MLP activation: out[i,:] = silu(gate[i,:]) * up[i,:].
// gate, up, out: [tokens, ffn_dim].
template <class ExecSpace>
void swiglu(const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& gate,
            const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& up,
            const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& out,
            const std::string& label = "KokkosDNN::swiglu") {
  const size_t T = gate.extent(0), F = gate.extent(1);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, F}),
      KOKKOS_LAMBDA(size_t t, size_t f) {
        float g = gate(t, f);
        out(t, f) = (g / (1.0f + expf(-g))) * up(t, f);
      });
}

}  // namespace KokkosDNN

#endif
