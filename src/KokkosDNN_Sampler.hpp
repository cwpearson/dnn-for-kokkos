#ifndef KOKKOSDNN_SAMPLER_HPP
#define KOKKOSDNN_SAMPLER_HPP

#include <Kokkos_Core.hpp>
#include <string>

namespace KokkosDNN {

// Greedy sampler: next_token = argmax(logits). logits: [vocab].
// Returns host-side int. Deterministic; temperature/top-p sampling is
// future work (sample with RNG on host after deep_copy).
// Scalar-agnostic; comparison is done in FP32.
template <class ExecSpace, class Scalar>
int sample_greedy(const Kokkos::View<Scalar*, ExecSpace>& logits) {
  const size_t V = logits.extent(0);
  auto h = Kokkos::create_mirror_view(logits);
  Kokkos::deep_copy(h, logits);
  int best = 0;
  for (size_t v = 1; v < V; ++v)
    if (static_cast<float>(h(v)) > static_cast<float>(h(best)))
      best = static_cast<int>(v);
  return best;
}

}  // namespace KokkosDNN

#endif
