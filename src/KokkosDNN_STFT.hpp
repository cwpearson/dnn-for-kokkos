#ifndef KOKKOSDNN_STFT_HPP
#define KOKKOSDNN_STFT_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

// Torch-compatible STFT / iSTFT for the Kokoro ISTFTNet vocoder.
//
// Matches torch.stft / torch.istft with center=True, pad_mode="reflect",
// window=torch.hann_window(n_fft, periodic=True), normalized=False
// (forward unnormalized, inverse divides by n_fft), onesided=True:
//   forward : reflect-pad n_fft//2 each side, window, DFT -> mag/phase
//             [B, F, Frames], F = n_fft/2+1, Frames = 1 + T/hop.
//   inverse : IDFT (1/n scale), re-window, overlap-add, divide by the
//             sum-of-win^2 envelope, trim the center pad -> [B, T].
// Verified against torch.istft: OLA of (irfft*win) / sum(win^2) is exact.
// Naive O(F*N) DFT/IDFT per frame; n_fft=20 here so the cost is trivial.
// FP32 accumulation, scalar-agnostic I/O.

namespace KokkosDNN {

namespace Impl {
KOKKOS_INLINE_FUNCTION float hann_periodic(int n, int N) {
  const float pi = 3.14159265358979323846f;
  return 0.5f * (1.0f - cosf(2.0f * pi * static_cast<float>(n) /
                             static_cast<float>(N)));
}
}  // namespace Impl

// x [B,T] waveform -> mag, phase [B, F, Frames].
template <class ExecSpace, class Scalar>
void stft_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    int n_fft, int hop,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& mag,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& phase,
    const std::string& label = "KokkosDNN::stft") {
  const size_t B = x.extent(0), T = x.extent(1);
  const size_t F = (size_t)(n_fft / 2 + 1);
  if (n_fft <= 0 || hop <= 0) throw std::invalid_argument("stft: bad params");
  if (T == 0) throw std::invalid_argument("stft: empty input");
  const size_t Frames = 1 + T / (size_t)hop;
  if (mag.extent(0) != B || mag.extent(1) != F || mag.extent(2) != Frames ||
      phase.extent(0) != B || phase.extent(1) != F ||
      phase.extent(2) != Frames)
    throw std::invalid_argument("stft_forward: output mismatch");
  const int pad = n_fft / 2;
  const float pi = 3.14159265358979323846f;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, Frames, F}),
      KOKKOS_LAMBDA(size_t b, size_t f, size_t k) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < n_fft; ++n) {
          const long q = (long)(f * (size_t)hop + (size_t)n);
          const long p = q - pad;
          float v = 0.0f;
          if (p < 0) {
            const long idx = (long)pad - q;
            v = (float)x(b, (size_t)idx);
          } else if ((size_t)p >= T) {
            const long j = q - ((long)T + pad);
            const long idx = (long)T - 2 - j;
            v = (float)x(b, (size_t)idx);
          } else {
            v = (float)x(b, (size_t)p);
          }
          const float w = Impl::hann_periodic(n, n_fft);
          const float s = v * w;
          const float ang = 2.0f * pi * (float)(k * (size_t)n) / (float)n_fft;
          re += s * cosf(ang);
          im -= s * sinf(ang);
        }
        mag(b, k, f) = (Scalar)(sqrtf(re * re + im * im));
        phase(b, k, f) = (Scalar)(atan2f(im, re));
      });
}

// mag, phase [B, F, Frames] -> y [B, T] with Frames = 1 + T/hop.
template <class ExecSpace, class Scalar>
void istft_inverse(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& mag,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& phase,
    int n_fft, int hop,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::istft") {
  const size_t B = mag.extent(0), F = mag.extent(1), Frames = mag.extent(2);
  const size_t T = y.extent(1);
  if (y.extent(0) != B) throw std::invalid_argument("istft: batch mismatch");
  if ((size_t)(n_fft / 2 + 1) != F)
    throw std::invalid_argument("istft: F mismatch");
  if (1 + T / (size_t)hop != Frames)
    throw std::invalid_argument("istft_inverse: Frames mismatch");
  if (phase.extent(0) != B || phase.extent(1) != F ||
      phase.extent(2) != Frames)
    throw std::invalid_argument("istft_inverse: phase mismatch");
  const float pi = 3.14159265358979323846f;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, T}),
      KOKKOS_LAMBDA(size_t b, size_t t) {
        const long q = (long)t + n_fft / 2;
        long f0 = (q - n_fft + 1 + hop - 1) / hop;
        if (f0 < 0) f0 = 0;
        long f1 = q / hop;
        if (f1 >= (long)Frames) f1 = (long)Frames - 1;
        float num = 0.0f, den = 0.0f;
        for (long f = f0; f <= f1; ++f) {
          const int n = (int)(q - f * hop);
          float s = 0.0f;
          for (size_t k = 0; k < F; ++k) {
            const float m = (float)mag(b, k, (size_t)f);
            const float p = (float)phase(b, k, (size_t)f);
            const float ang = 2.0f * pi * (float)(k * (size_t)n) / (float)n_fft;
            float term = m * (cosf(p) * cosf(ang) - sinf(p) * sinf(ang));
            if (k > 0 && k + 1 < F) term *= 2.0f;
            s += term;
          }
          s /= (float)n_fft;
          const float w = Impl::hann_periodic(n, n_fft);
          num += s * w;
          den += w * w;
        }
        y(b, t) = (Scalar)(den > 1e-10f ? num / den : 0.0f);
      });
}

}  // namespace KokkosDNN

#endif
