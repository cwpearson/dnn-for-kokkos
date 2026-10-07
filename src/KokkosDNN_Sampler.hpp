#ifndef KOKKOSDNN_SAMPLER_HPP
#define KOKKOSDNN_SAMPLER_HPP

#include <Kokkos_Core.hpp>
#include <string>

namespace KokkosDNN {

// Greedy sampler: next_token = argmax(logits). logits: [vocab].
// Returns host-side int. Deterministic; temperature/top-p sampling is
// future work (sample with RNG on host after deep_copy).
template <class ExecSpace>
int sample_greedy(const Kokkos::View<float*, ExecSpace>& logits) {
  const size_t V = logits.extent(0);
  auto h = Kokkos::create_mirror_view(logits);
  Kokkos::deep_copy(h, logits);
  int best = 0;
  for (size_t v = 1; v < V; ++v)
    if (h(v) > h(best)) best = static_cast<int>(v);
  return best;
}

}  // namespace KokkosDNN

#endif
