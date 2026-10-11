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

namespace Impl {

// 2D axial RoPE helpers (DINOv3 ViT). Public entry points below; these are
// implementation details (see AGENTS.md: non-public APIs live in Impl).

// Normalized center coordinate of patch index i among n patches along one
// axis: 2*(i+0.5)/n - 1 in [-1, +1]. Matches HF
// `get_patches_center_coordinates` (arange(0.5, n)/n shifted to [-1, +1]).
KOKKOS_INLINE_FUNCTION float rope2d_coord(int i, int n) {
  return 2.0f * ((float)i + 0.5f) / (float)n - 1.0f;
}

// Single-token axial rotation with the HF dinov3_vit quarter layout.
// Qh = head_dim/4. Lanes [0,Qh) and [2*Qh,3*Qh) form rotate_half pairs
// encoding the h coordinate; lanes [Qh,2*Qh) and [3*Qh,4*Qh) form pairs
// encoding the w coordinate. Angles carry the 2*pi scale from HF
// (angles = 2*pi*coord*inv_freq). x may alias y (in-place).
template <class XView, class YView, class FView>
KOKKOS_INLINE_FUNCTION void rope2d_rotate_token(const XView& x, const YView& y,
                                               size_t t, size_t h, size_t Qh,
                                               const FView& inv_freq, float ch,
                                               float cw) {
  using Scalar = typename YView::value_type;
  constexpr float two_pi = 6.283185307179586f;
  for (size_t d = 0; d < Qh; ++d) {
    const float ah = two_pi * ch * inv_freq(d);
    const float chh = cosf(ah), shh = sinf(ah);
    const float h1 = static_cast<float>(x(t, h, d));
    const float h2 = static_cast<float>(x(t, h, d + 2 * Qh));
    y(t, h, d) = static_cast<Scalar>(h1 * chh - h2 * shh);
    y(t, h, d + 2 * Qh) = static_cast<Scalar>(h1 * shh + h2 * chh);
    const float aw = two_pi * cw * inv_freq(d);
    const float cww = cosf(aw), sww = sinf(aw);
    const float w1 = static_cast<float>(x(t, h, d + Qh));
    const float w2 = static_cast<float>(x(t, h, d + 3 * Qh));
    y(t, h, d + Qh) = static_cast<Scalar>(w1 * cww - w2 * sww);
    y(t, h, d + 3 * Qh) = static_cast<Scalar>(w1 * sww + w2 * cww);
  }
}

// Range kernel shared by the in-place and out-of-place entry points (and by
// the q and k passes): rotates patch rows [t0, t0+Np) of x into y. Row-major
// grid mapping p = hi*Wp + wi. No-op when H == 0 or Np == 0.
template <class ExecSpace, class XView, class YView, class FView>
void rope2d_rotate_range(const XView& x, const YView& y,
                         const FView& inv_freq, size_t H, size_t Qh, size_t t0,
                         size_t Np, int Hp, int Wp, const std::string& label) {
  if (H == 0 || Np == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Np, H}),
      KOKKOS_LAMBDA(size_t p, size_t h) {
        const size_t t  = t0 + p;
        const int hi    = (int)(p / (size_t)Wp);
        const int wi    = (int)(p % (size_t)Wp);
        const float ch  = rope2d_coord(hi, Hp);
        const float cw  = rope2d_coord(wi, Wp);
        rope2d_rotate_token(x, y, t, h, Qh, inv_freq, ch, cw);
      });
}

// Host-side argument validation shared by the 2D entry points. q_T is the
// token count of q; k checks apply only when k holds tokens (empty k is
// allowed, mirroring rope_neoX_inplace). Throws std::invalid_argument.
inline void rope2d_check(int head_dim, size_t inv_n, size_t Dh_q, size_t Dh_k,
                         size_t q_T, size_t k_T, size_t k_tokens, int Hp,
                         int Wp, size_t prefix_skip) {
  if (head_dim <= 0 || head_dim % 4 != 0)
    throw std::invalid_argument(
        "rope2d: head_dim must be positive and divisible by 4 (got " +
        std::to_string(head_dim) + ")");
  if (inv_n != (size_t)(head_dim / 4))
    throw std::invalid_argument(
        "rope2d: inv_freq length must be head_dim/4 (got " +
        std::to_string(inv_n) + " for head_dim " + std::to_string(head_dim) +
        ")");
  if (Dh_q != (size_t)head_dim)
    throw std::invalid_argument("rope2d: q head dim mismatch (got " +
                                std::to_string(Dh_q) + ", expected " +
                                std::to_string(head_dim) + ")");
  if (k_tokens > 0 && Dh_k != (size_t)head_dim)
    throw std::invalid_argument("rope2d: k head dim mismatch (got " +
                                std::to_string(Dh_k) + ", expected " +
                                std::to_string(head_dim) + ")");
  if (Hp <= 0 || Wp <= 0)
    throw std::invalid_argument("rope2d: grid Hp, Wp must be positive");
  const size_t Np = (size_t)Hp * (size_t)Wp;
  if (q_T != prefix_skip + Np)
    throw std::invalid_argument(
        "rope2d: token count T must equal prefix_skip + Hp*Wp (got T=" +
        std::to_string(q_T) + ", prefix_skip=" + std::to_string(prefix_skip) +
        ", Hp*Wp=" + std::to_string(Np) + ")");
  if (k_tokens > 0 && k_T != q_T)
    throw std::invalid_argument("rope2d: q and k token counts differ (got " +
                                std::to_string(q_T) + " vs " +
                                std::to_string(k_T) + ")");
}

}  // namespace Impl

// Host reference for 2D axial inverse frequencies (DINOv3 ViT):
//   inv_freq[i] = theta^(-4*i/head_dim), i in [0, head_dim/4).
// Matches HF `DINOv3ViTRopePositionEmbedding` / `_init_weights`:
//   1 / base ** arange(0, 1, 4/head_dim)  (head_dim/4 entries).
// Note the exponent denominator: each axis owns head_dim/2 lanes, so the
// per-axis frequencies use dim head_dim/2, i.e. theta^(-2*i/(head_dim/2)).
// head_dim must be divisible by 4. Default theta=100 (dinov3 vitb16).
inline std::vector<float> rope2d_inv_freq_host(int head_dim,
                                              float theta = 100.0f) {
  if (head_dim <= 0 || head_dim % 4 != 0)
    throw std::invalid_argument(
        "rope2d_inv_freq: head_dim must be positive and divisible by 4 (got " +
        std::to_string(head_dim) + ")");
  std::vector<float> out((size_t)(head_dim / 4));
  for (int i = 0; i < head_dim / 4; ++i)
    out[(size_t)i] = (float)(1.0 / std::pow((double)theta,
                                            (4.0 * (double)i) /
                                                (double)head_dim));
  return out;
}

// Device mirror of rope2d_inv_freq_host.
template <class ExecSpace>
Kokkos::View<float*, ExecSpace> rope2d_inv_freq(int head_dim,
                                               float theta = 100.0f,
                                               const std::string& label = "") {
  (void)label;
  std::vector<float> hv = rope2d_inv_freq_host(head_dim, theta);
  Kokkos::View<float*, ExecSpace> f("rope2d_inv_freq", head_dim / 4);
  auto h = Kokkos::create_mirror_view(f);
  for (int i = 0; i < head_dim / 4; ++i) h(i) = hv[(size_t)i];
  Kokkos::deep_copy(f, h);
  return f;
}

// 2D axial rotary embedding applied to q and k in place (DINOv3 ViT).
// q, k: [T, H, Dh] with T = prefix_skip + Hp*Wp (tokens are 1 CLS +
//   registers + Np row-major patches, e.g. 5 + 196 = 201 @224px/16px).
// Only patch rows [prefix_skip, T) are rotated; prefix rows (CLS +
// register tokens) are untouched, matching HF `apply_rotary_pos_emb`
// (prefix/patch split by token count).
//
// Convention (HF dinov3_vit `modeling_dinov3_vit.py`):
//   * inv_freq has head_dim/4 entries, theta base (default 100).
//   * Per-patch normalized center coords (ch, cw) in [-1, +1]
//     (2*(i+0.5)/n - 1 per axis, row-major p = h*Wp + w), computed
//     dynamically from (Hp, Wp) each call -- no baked image size.
//   * Angles 2*pi*coord*inv_freq, tiled to quarter layout
//     [h(Dh/4), w(Dh/4), h(Dh/4), w(Dh/4)]; i.e. head_dim is NOT split
//     into contiguous halves per axis. Instead rotate_half over the full
//     head_dim pairs lane d with lane d+Dh/2, and the pair angle is the h
//     angle for d in [0, Dh/4) and the w angle for d in [Dh/4, Dh/2).
//     q_embed = q*cos + rotate_half(q)*sin with HF rotate_half
//     (cat(-x2, x1)), which this kernel expands explicitly.
//   * Identity at coord (0, 0) (odd-grid center patch): cos=1, sin=0.
// Scalar-agnostic over q/k (float, bhalf_t); inv_freq stays FP32 and the
// rotation is computed in FP32. Empty k (extent(0) == 0) rotates q only.
template <class ExecSpace, class Scalar, class QView, class KView>
void rope2d_axial_inplace(
    const QView& q,  // [T, Hq, Dh]
    const KView& k,  // [T, Hkv, Dh]
    const Kokkos::View<float*, ExecSpace>& inv_freq, int Hp, int Wp,
    size_t prefix_skip = 5,  // 1 CLS + 4 registers for dinov3 vitb16
    const std::string& label = "KokkosDNN::rope2d") {
  const int head_dim = (int)q.extent(2);
  Impl::rope2d_check(head_dim, inv_freq.extent(0), q.extent(2), k.extent(2),
                     q.extent(0), k.extent(0), k.extent(0), Hp, Wp,
                     prefix_skip);
  const size_t Qh = (size_t)(head_dim / 4);
  const size_t Np = (size_t)Hp * (size_t)Wp;
  Impl::rope2d_rotate_range<ExecSpace>(q, q, inv_freq, q.extent(1), Qh,
                                      prefix_skip, Np, Hp, Wp, label);
  Impl::rope2d_rotate_range<ExecSpace>(k, k, inv_freq, k.extent(1), Qh,
                                      prefix_skip, Np, Hp, Wp, label + "_k");
}

// Out-of-place 2D axial RoPE: qo/kO receive the rotated q/k (prefix rows
// copied through unchanged). Same convention as rope2d_axial_inplace.
// Shapes of qo/ko must match q/k. Inputs are not modified, so the same
// views may be passed as both input and output (degrades to in-place).
template <class ExecSpace, class Scalar, class QView, class KView,
          class QOView, class KOView>
void rope2d_axial_forward(
    const QView& q, const KView& k, const QOView& qo, const KOView& ko,
    const Kokkos::View<float*, ExecSpace>& inv_freq, int Hp, int Wp,
    size_t prefix_skip = 5,
    const std::string& label = "KokkosDNN::rope2d") {
  const int head_dim = (int)q.extent(2);
  Impl::rope2d_check(head_dim, inv_freq.extent(0), q.extent(2), k.extent(2),
                     q.extent(0), k.extent(0), k.extent(0), Hp, Wp,
                     prefix_skip);
  if (qo.extent(0) != q.extent(0) || qo.extent(1) != q.extent(1) ||
      qo.extent(2) != q.extent(2))
    throw std::invalid_argument("rope2d: qo shape must match q");
  if (ko.extent(0) != k.extent(0) || ko.extent(1) != k.extent(1) ||
      ko.extent(2) != k.extent(2))
    throw std::invalid_argument("rope2d: ko shape must match k");
  Kokkos::deep_copy(qo, q);
  if (k.extent(0) > 0) Kokkos::deep_copy(ko, k);
  const size_t Qh = (size_t)(head_dim / 4);
  const size_t Np = (size_t)Hp * (size_t)Wp;
  Impl::rope2d_rotate_range<ExecSpace>(qo, qo, inv_freq, qo.extent(1), Qh,
                                      prefix_skip, Np, Hp, Wp, label);
  Impl::rope2d_rotate_range<ExecSpace>(ko, ko, inv_freq, ko.extent(1), Qh,
                                      prefix_skip, Np, Hp, Wp, label + "_k");
}

}  // namespace KokkosDNN

#endif
