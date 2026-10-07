#ifndef KOKKOSDNN_ROPE_HPP
#define KOKKOSDNN_ROPE_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

namespace KokkosDNN {

// Precompute inverse frequencies: inv_freq[i] = theta^(-2i/head_dim),
// i in [0, head_dim/2). Matches HF Llama (rope_theta=500000 for 3.3 8B).
template <class ExecSpace>
Kokkos::View<float*, ExecSpace> rope_inv_freq(int head_dim, float theta,
                                              const std::string& label = "") {
  (void)label;
  Kokkos::View<float*, ExecSpace> f("rope_inv_freq", head_dim / 2);
  auto h = Kokkos::create_mirror_view(f);
  for (int i = 0; i < head_dim / 2; ++i)
    h(i) = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
  Kokkos::deep_copy(f, h);
  return f;
}

// GPT-NeoX-style rotary embedding applied to q and k in place:
// for each head, pairs (d, d+half) rotate by angle = pos * inv_freq[d].
// q, k: [tokens, n_heads, head_dim] (flattened last two dims as [T, H*D]).
// This matches HF `apply_rotary_pos_emb` (cat(freqs,freqs) convention).
template <class ExecSpace>
void rope_neoX_inplace(
    const Kokkos::View<float***, ExecSpace>& q,  // [T, Hq, D]
    const Kokkos::View<float***, ExecSpace>& k,  // [T, Hkv, D]
    const Kokkos::View<float*, ExecSpace>& inv_freq, int head_dim,
    const Kokkos::View<int*, ExecSpace>& positions,  // [T] absolute pos
    const std::string& label = "KokkosDNN::rope") {
  const size_t T  = q.extent(0), Hq = q.extent(1);
  const size_t Hk = k.extent(0) ? k.extent(1) : 0;
  const size_t Hd = inv_freq.extent(0);  // head_dim/2
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hq}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float pos = (float)positions(t);
        for (size_t d = 0; d < Hd; ++d) {
          float ang = pos * inv_freq(d);
          float c = cosf(ang), s = sinf(ang);
          float a = q(t, h, d), b = q(t, h, d + Hd);
          q(t, h, d)      = a * c - b * s;
          q(t, h, d + Hd) = a * s + b * c;
        }
      });
  Kokkos::parallel_for(
      label + "_k",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hk}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float pos = (float)positions(t);
        for (size_t d = 0; d < Hd; ++d) {
          float ang = pos * inv_freq(d);
          float c = cosf(ang), s = sinf(ang);
          float a = k(t, h, d), b = k(t, h, d + Hd);
          k(t, h, d)      = a * c - b * s;
          k(t, h, d + Hd) = a * s + b * c;
        }
      });
  (void)head_dim;
}

}  // namespace KokkosDNN

#endif
