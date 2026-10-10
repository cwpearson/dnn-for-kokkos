#ifndef KOKKOSDNN_ROPE_HPP
#define KOKKOSDNN_ROPE_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace KokkosDNN {

// Host reference for inverse frequencies, including Llama 3.1 llama3-type
// scaling (HF transformers LlamaModel._compute_llama3_parameters):
//   base inv = 1 / (theta^(2i/dim)); wavelen = 2*pi/inv;
//   low_w = orig/low; high_w = orig/high;
//   wavelen < high_w          -> keep inv;
//   wavelen > low_w           -> inv /= factor;
//   otherwise (medium range)  -> smooth = (orig/wavelen - low)/(high-low),
//                                inv = (1-smooth)*inv/factor + smooth*inv.
// Empty rope_type selects the plain NeoX path. Throws on unknown types or
// on llama3 with original_max <= 0.
inline std::vector<float> rope_inv_freq_host(
    int head_dim, float theta, const std::string& rope_type = "",
    double factor = 1.0, double low_freq = 1.0, double high_freq = 4.0,
    int original_max = 0) {
  std::vector<float> out(head_dim / 2);
  if (!rope_type.empty() && rope_type != "llama3")
    throw std::runtime_error("rope_inv_freq: unsupported rope_type '" +
                             rope_type + "' (supported: '' and 'llama3')");
  if (rope_type == "llama3" && original_max <= 0)
    throw std::invalid_argument(
        "rope_inv_freq: original_max must be > 0 for rope_type 'llama3'");
  const double pi = std::acos(-1.0);
  for (int i = 0; i < head_dim / 2; ++i) {
    double inv =
        1.0 / std::pow((double)theta, (2.0 * (double)i) / (double)head_dim);
    if (rope_type == "llama3") {
      const double wavelen = 2.0 * pi / inv;
      const double low_w   = (double)original_max / low_freq;
      const double high_w  = (double)original_max / high_freq;
      if (wavelen < high_w) {
        // high-frequency: keep.
      } else if (wavelen > low_w) {
        inv /= factor;
      } else {
        const double smooth = ((double)original_max / wavelen - low_freq) /
                              (high_freq - low_freq);
        inv = (1.0 - smooth) * inv / factor + smooth * inv;
      }
    }
    out[(size_t)i] = (float)inv;
  }
  return out;
}

// Llama3-aware overload: plain path when rope_type is empty.
template <class ExecSpace>
Kokkos::View<float*, ExecSpace> rope_inv_freq(
    int head_dim, float theta, const std::string& rope_type, double factor,
    double low_freq, double high_freq, int original_max,
    const std::string& label = "") {
  (void)label;
  std::vector<float> hv =
      rope_inv_freq_host(head_dim, theta, rope_type, factor, low_freq,
                         high_freq, original_max);
  Kokkos::View<float*, ExecSpace> f("rope_inv_freq", head_dim / 2);
  auto h = Kokkos::create_mirror_view(f);
  for (int i = 0; i < head_dim / 2; ++i) h(i) = hv[(size_t)i];
  Kokkos::deep_copy(f, h);
  return f;
}

// Precompute inverse frequencies: inv_freq[i] = theta^(-2i/head_dim),
// i in [0, head_dim/2). Matches HF Llama (rope_theta=500000 for 3.1 8B-Instruct).
template <class ExecSpace>
Kokkos::View<float*, ExecSpace> rope_inv_freq(int head_dim, float theta,
                                              const std::string& label = "") {
  return rope_inv_freq<ExecSpace>(head_dim, theta, "", 1.0, 1.0, 4.0, 0,
                                  label);
}

// GPT-NeoX-style rotary embedding applied to q and k in place:
// for each head, pairs (d, d+half) rotate by angle = pos * inv_freq[d].
// q, k: [tokens, n_heads, head_dim] (flattened last two dims as [T, H*D]).
// This matches HF `apply_rotary_pos_emb` (cat(freqs,freqs) convention).
// Scalar-agnostic over q/k (float, bhalf_t); inv_freq stays FP32 and the
// rotation is computed in FP32.
template <class ExecSpace, class Scalar>
void rope_neoX_inplace(
    const Kokkos::View<Scalar***, ExecSpace>& q,  // [T, Hq, D]
    const Kokkos::View<Scalar***, ExecSpace>& k,  // [T, Hkv, D]
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
          float a = static_cast<float>(q(t, h, d)),
                b = static_cast<float>(q(t, h, d + Hd));
          q(t, h, d)      = static_cast<Scalar>(a * c - b * s);
          q(t, h, d + Hd) = static_cast<Scalar>(a * s + b * c);
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
          float a = static_cast<float>(k(t, h, d)),
                b = static_cast<float>(k(t, h, d + Hd));
          k(t, h, d)      = static_cast<Scalar>(a * c - b * s);
          k(t, h, d + Hd) = static_cast<Scalar>(a * s + b * c);
        }
      });
  (void)head_dim;
}

}  // namespace KokkosDNN

#endif
