#ifndef KOKKOSDNN_LOSS_HPP
#define KOKKOSDNN_LOSS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// Binary cross-entropy with logits, averaged over all elements:
//   loss = mean_b,o [ max(z,0) - z*t + log(1 + exp(-|z|)) ]
// with z = logits, t = targets in {0,1}. The split formulation keeps the
// forward pass stable for large |z| without a separate sigmoid. Backward:
//   dLogits = (sigmoid(z) - t) / N  (N = element count, matching the mean).
// Shapes are [B, Out] so dense outputs feed in directly; targets match.
// Scalar-agnostic; computed in FP32. For the XOR example B=4, Out=1.

template <class ExecSpace, class Scalar>
float bce_with_logits_loss(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& targets,
    const std::string& label = "KokkosDNN::bce_loss") {
  if (logits.extent(0) != targets.extent(0) ||
      logits.extent(1) != targets.extent(1))
    throw std::invalid_argument("bce_with_logits_loss: dimension mismatch");
  const size_t B = logits.extent(0), O = logits.extent(1);
  if (B == 0 || O == 0) return 0.0f;
  float total = 0;
  Kokkos::parallel_reduce(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, O}),
      KOKKOS_LAMBDA(size_t i, size_t j, float& acc) {
        float z = static_cast<float>(logits(i, j));
        float t = static_cast<float>(targets(i, j));
        float m = z > 0.0f ? z : 0.0f;
        acc += m - z * t + logf(1.0f + expf(-fabsf(z)));
      },
      total);
  return total / static_cast<float>(B * O);
}

template <class ExecSpace, class Scalar>
void bce_with_logits_backward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& logits,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& targets,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dlogits,
    const std::string& label = "KokkosDNN::bce_backward") {
  if (logits.extent(0) != targets.extent(0) ||
      logits.extent(1) != targets.extent(1) ||
      logits.extent(0) != dlogits.extent(0) ||
      logits.extent(1) != dlogits.extent(1))
    throw std::invalid_argument("bce_with_logits_backward: dimension mismatch");
  const size_t B = logits.extent(0), O = logits.extent(1);
  if (B == 0 || O == 0) return;
  const float inv = 1.0f / static_cast<float>(B * O);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, O}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float z = static_cast<float>(logits(i, j));
        float s = 1.0f / (1.0f + expf(-z));
        dlogits(i, j) = static_cast<Scalar>(
            (s - static_cast<float>(targets(i, j))) * inv);
      });
}

}  // namespace KokkosDNN

#endif
