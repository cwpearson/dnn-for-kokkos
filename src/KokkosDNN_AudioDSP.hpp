#ifndef KOKKOSDNN_AUDIO_DSP_HPP
#define KOKKOSDNN_AUDIO_DSP_HPP

// Small audio DSP toolkit for vocoder-style example models (Kokoro /
// ISTFTNet): Hann window, STFT (magnitude/phase), inverse STFT, and 1D
// resampling matching PyTorch semantics.
//
//   fill_hann_window(w)            w[n] = 0.5 - 0.5*cos(2*pi*n/N) (fftbins)
//   stft_forward(x, win, hop, mag, phase)
//     center=True / reflect-pad convention (matches torch.stft with
//     center=True, pad_mode='reflect', normalized=False, onesided=True):
//     x length N is reflect-padded by P = W/2 on both sides, then framed
//     with window W and hop H. Frames F = 1 + N/H (must divide evenly).
//     mag/phase are [F, W/2+1] (forward DFT unscaled, like torch).
//   istft_inverse(mag, phase, win, hop, y)
//     overlap-add synthesis matching torch.istft (center=True): per-frame
//     IDFT (1/W scaled), re-windowed, divided by the sum-of-squares
//     envelope, center-trimmed to (F-1)*H samples.
//   upsample_nearest_1d / interpolate_linear_1d over time-major [T,C]:
//     nearest: integer repeat (matches F.interpolate(mode='nearest')).
//     linear: PyTorch align_corners=False convention,
//       src = (dst + 0.5) * (I/O) - 0.5, taps clamped to [0, I-1]
//       (matches F.interpolate(mode='linear', align_corners=False) for both
//       up and downsampling).
//
// All float (audio DSP is FP32 by nature). Portable parallel_for only.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#ifndef KOKKOSDNN_PI_F
#define KOKKOSDNN_PI_F 3.14159265358979323846f
#endif

namespace KokkosDNN {

// Hann window, fftbins convention: w[n] = 0.5 - 0.5*cos(2*pi*n/N).
template <class ExecSpace>
void fill_hann_window(const Kokkos::View<float*, ExecSpace>& w,
                      const std::string& label = "KokkosDNN::hann") {
  const size_t N = w.extent(0);
  if (N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t n) {
        w(n) = 0.5f - 0.5f * cosf(2.0f * KOKKOSDNN_PI_F *
                                  (float)n / (float)N);
      });
}

// Reflect-pad helper (host-callable index map): padded index j of the
// length-(N+2P) signal maps to x[reflect(j-P)] with torch 'reflect' mode
// (edge not duplicated: -1 -> 1, N -> N-2, ...).
KOKKOS_INLINE_FUNCTION
size_t reflect_index(long j, size_t n) {
  if (n <= 1) return 0;
  const long nn = (long)n;
  // Fold into [0, 2n-2): torch reflect mode mirrors with period 2n-2.
  long m = j % (2 * nn - 2);
  if (m < 0) m += 2 * nn - 2;
  return (size_t)(m < nn ? m : 2 * nn - 2 - m);
}

// STFT forward: x [N] -> mag/phase [F, B], B = W/2+1, F = 1 + N/hop.
// win [W] analysis window; hop H. Center/reflect convention as above.
template <class ExecSpace>
void stft_forward(const Kokkos::View<float*, ExecSpace>& x,
                  const Kokkos::View<float*, ExecSpace>& win, size_t hop,
                  const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& mag,
                  const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& ph,
                  const std::string& label = "KokkosDNN::stft") {
  const size_t N = x.extent(0), W = win.extent(0);
  if (W == 0 || W % 2 != 0 || hop == 0)
    throw std::invalid_argument("stft_forward: bad window/hop");
  if (N % hop != 0)
    throw std::invalid_argument("stft_forward: N must divide by hop");
  const size_t F = 1 + N / hop, B = W / 2 + 1;
  if (mag.extent(0) != F || mag.extent(1) != B || ph.extent(0) != F ||
      ph.extent(1) != B)
    throw std::invalid_argument("stft_forward: output shape mismatch");
  if (N == 0) return;
  const size_t P = W / 2;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {F, B}),
      KOKKOS_LAMBDA(size_t f, size_t k) {
        float re = 0.0f, im = 0.0f;
        for (size_t n = 0; n < W; ++n) {
          const long j = (long)(f * hop + n) - (long)P;
          const float s = x(reflect_index(j, N)) * win(n);
          const float ang =
              -2.0f * KOKKOSDNN_PI_F * (float)(k * n) / (float)W;
          re += s * cosf(ang);
          im += s * sinf(ang);
        }
        mag(f, k) = sqrtf(re * re + im * im);
        ph(f, k) = atan2f(im, re);
      });
}

// Inverse STFT: mag/phase [F, B] -> y [(F-1)*hop], re-window + envelope
// divide + center trim (matches torch.istft center=True, onesided).
template <class ExecSpace>
void istft_inverse(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& mag,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& ph,
    const Kokkos::View<float*, ExecSpace>& win, size_t hop,
    const Kokkos::View<float*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::istft") {
  const size_t F = mag.extent(0), B = mag.extent(1);
  const size_t W = win.extent(0);
  if (W == 0 || W % 2 != 0 || hop == 0 || B != W / 2 + 1)
    throw std::invalid_argument("istft_inverse: bad window/hop");
  if (ph.extent(0) != F || ph.extent(1) != B)
    throw std::invalid_argument("istft_inverse: phase shape mismatch");
  const size_t N = (F - 1) * hop;
  if (y.extent(0) != N)
    throw std::invalid_argument("istft_inverse: output shape mismatch");
  if (F == 0) return;
  // Per-output-sample synthesis (deterministic: no cross-thread
  // accumulation). Output sample i (of the trimmed signal) sits at full
  // position p = i + W/2; contributing frames satisfy
  // f*hop <= p < f*hop + W, i.e. at most ceil(W/hop) frames.
  // Per-frame sample via the real IDFT (conjugate-symmetric spectrum,
  // 1/W scaled), re-windowed; divided by the sum-of-squares envelope.
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(size_t i) {
        const size_t p = i + W / 2;
        // Frame range: f <= p/hop and f*hop + W > p.
        const size_t f_lo =
            (p + 1 >= W) ? (p + 1 - W + hop - 1) / hop : 0;
        size_t f_hi = p / hop;
        if (f_hi >= F) f_hi = F - 1;
        float num = 0.0f, den = 0.0f;
        for (size_t f = f_lo; f <= f_hi; ++f) {
          const size_t n = p - f * hop;  // 0 <= n < W by construction
          // x[n] = (1/W)(X0 + 2*sum_{k=1}^{B-2} |Xk|cos(2pi k n/W + ph)
          //              + |X_{B-1}|cos(pi n + ph))  (W even; bins 0 and
          //              B-1 are real).
          float acc = mag(f, 0) * cosf(ph(f, 0));
          for (size_t k = 1; k + 1 < B; ++k) {
            const float ang = 2.0f * KOKKOSDNN_PI_F * (float)(k * n) /
                              (float)W +
                              ph(f, k);
            acc += 2.0f * mag(f, k) * cosf(ang);
          }
          {
            const float ang = KOKKOSDNN_PI_F * (float)n + ph(f, B - 1);
            acc += mag(f, B - 1) * cosf(ang);
          }
          acc /= (float)W;
          const float wv = win(n);
          num += acc * wv;
          den += wv * wv;
        }
        y(i) = (den > 1e-10f) ? num / den : 0.0f;
      });
}

// Nearest-neighbor 1D upsample over time-major [T,C]: each input row is
// repeated `scale` times (matches F.interpolate(mode='nearest') for
// integer factors). Rank-2 and rank-1 variants.
template <class ExecSpace, class Scalar>
void upsample_nearest_1d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    size_t scale,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::upsample_nearest_1d") {
  const size_t T = x.extent(0), C = x.extent(1);
  if (scale == 0)
    throw std::invalid_argument("upsample_nearest_1d: scale must be > 0");
  if (y.extent(0) != T * scale || y.extent(1) != C)
    throw std::invalid_argument("upsample_nearest_1d: output shape mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                 {0, 0}, {T * scale, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) { y(t, c) = x(t / scale, c); });
}

template <class ExecSpace, class Scalar>
void upsample_nearest_1d(
    const Kokkos::View<Scalar*, ExecSpace>& x, size_t scale,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::upsample_nearest_1d") {
  const size_t T = x.extent(0);
  if (scale == 0)
    throw std::invalid_argument("upsample_nearest_1d: scale must be > 0");
  if (y.extent(0) != T * scale)
    throw std::invalid_argument("upsample_nearest_1d: output shape mismatch");
  if (T == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * scale),
      KOKKOS_LAMBDA(size_t t) { y(t) = x(t / scale); });
}

// Linear 1D resample over time-major [T,C], PyTorch align_corners=False
// convention (matches F.interpolate(mode='linear', align_corners=False)
// for both up and downsampling):
//   src(d) = (d + 0.5) * (I / O) - 0.5, taps clamped to [0, I-1].
template <class ExecSpace, class Scalar>
void interpolate_linear_1d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::interpolate_linear_1d") {
  const size_t I = x.extent(0), C = x.extent(1), O = y.extent(0);
  if (y.extent(1) != C)
    throw std::invalid_argument("interpolate_linear_1d: channel mismatch");
  if (I == 0 || O == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {O, C}),
      KOKKOS_LAMBDA(size_t d, size_t c) {
        float src = ((float)d + 0.5f) * ((float)I / (float)O) - 0.5f;
        if (src < 0.0f) src = 0.0f;
        if (src > (float)(I - 1)) src = (float)(I - 1);
        const size_t lo = (size_t)src;
        size_t hi = lo + 1;
        if (hi >= I) hi = I - 1;
        const float w = src - (float)lo;
        const float a = static_cast<float>(x(lo, c));
        const float b = static_cast<float>(x(hi, c));
        y(d, c) = static_cast<Scalar>(a + w * (b - a));
      });
}

}  // namespace KokkosDNN

#endif
