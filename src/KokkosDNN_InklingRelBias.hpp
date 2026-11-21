// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_INKLINGRELBIAS_HPP
#define KOKKOSDNN_INKLINGRELBIAS_HPP

// Inkling2 loop hidden-conditioned relative position bias
// (thinkingmachines/Inkling-2-Mini text decoder). The caller supplies the
// GEMM-produced per-head relative states (linear: hidden -> H*d_rel,
// followed by Scaled or LogScaled L2Clip normalization, see
// KokkosDNN_InklingNorm.hpp); this header contributes the bank projection
// and distance gather:
//   rel_logits[q, h, e] = sum_r rel_states[q, h, r] * proj[r, e]
//   bias[q, k, h] = (0 <= d < E) ? rel_logits[q, h, clamp(d)] : 0,
//     d = rel_dist[q, k]
// All accumulation is FP32 with the result cast back to Scalar.
// Backend-portable: only parallel_for kernels; no backend-specific code.

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// Hidden-conditioned relative position bias gather.
// rel_states: [Tq, H, R] post-linear post-norm relative states (FP32 GEMM
//   outputs normalized with inkling_scaled_l2_clip /
//   inkling_log_scaled_l2_clip over the d_rel axis);
// proj: [R, E] trained bank of bias-vs-distance profiles (E = rel_extent);
// rel_dist: [Tq, Tk] integer query-minus-key distances;
// bias: [Tq, Tk, H] output. Distances outside [0, E) read as zero, matching
// Inkling2LoopRelativePositionBias (clamp-then-mask).
template <class ExecSpace, class Scalar, class DistView>
void inkling_relative_position_bias(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& rel_states,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& proj,
    const DistView& rel_dist, size_t rel_extent,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& bias,
    const std::string& label = "KokkosDNN::inkling_relative_position_bias") {
  static_assert(DistView::rank == 2,
                "inkling_relative_position_bias: rel_dist must be rank 2");
  const size_t Tq = rel_states.extent(0), H = rel_states.extent(1),
               R = rel_states.extent(2);
  const size_t Tk = rel_dist.extent(1);
  const size_t E = proj.extent(1);
  if (proj.extent(0) != R || rel_dist.extent(0) != Tq ||
      bias.extent(0) != Tq || bias.extent(1) != Tk || bias.extent(2) != H ||
      rel_extent != E)
    throw std::invalid_argument(
        "inkling_relative_position_bias: dimension mismatch");
  if (Tq == 0 || Tk == 0 || H == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                        {Tq, Tk, H}),
      KOKKOS_LAMBDA(size_t q, size_t k, size_t h) {
        const long d = static_cast<long>(rel_dist(q, k));
        if (d < 0 || d >= static_cast<long>(E)) {
          bias(q, k, h) = static_cast<Scalar>(0);
          return;
        }
        const size_t e = static_cast<size_t>(d);
        float acc = 0;
        for (size_t r = 0; r < R; ++r)
          acc += static_cast<float>(rel_states(q, h, r)) * proj(r, e);
        bias(q, k, h) = static_cast<Scalar>(acc);
      });
}

}  // namespace KokkosDNN

#endif
