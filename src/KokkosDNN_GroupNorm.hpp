#ifndef KOKKOSDNN_GROUPNORM_HPP
#define KOKKOSDNN_GROUPNORM_HPP

// GroupNorm over NCHW activations (diffusers ResNet/attention norm).
//
// Covers the FLUX.2-small VAE decoder (AutoencoderKLFlux2, MOSS-Image-SFT
// vae/): every ResNet block normalizes with GroupNorm(32 groups, eps 1e-6,
// affine weight+bias), the mid-block self-attention entry normalizes with
// GroupNorm(32 groups, eps 1e-6, affine), and the decoder head normalizes
// with GroupNorm(32 groups, eps 1e-6, affine) before SiLU + conv_out.
// Reference: diffusers 0.38.0 models/resnet.py (ResnetBlock2D norm1/norm2),
// models/autoencoders/vae.py (Decoder conv_norm_out), and
// models/attention_processor.py (Attention.group_norm).
//
// Math (PyTorch nn.GroupNorm, channels split into G contiguous groups):
//   for each (n, g): mean/var over the Cg*H*W elements
//     (Cg = C / G channels [g*Cg, (g+1)*Cg)),
//   y[n,c,h,w] = (x[n,c,h,w] - mean) / sqrt(var + eps) * w[c] + b[c].
// Scalar-agnostic (float, bhalf_t); statistics and the normalize/scale/
// shift accumulate in FP32, then cast to Scalar.
//
// Dispatch: native portable elementwise kernel (MDRangePolicy only, no TPL
// — matches repo policy that norm ops are native; cf. layernorm_forward,
// rmsnorm). No backend #ifdefs in the native path.
//
// In-place safety: statistics are fully reduced before any write in the
// same kernel launch, and each output element depends only on the
// same-index input plus its group's (already-final) statistics and the
// affine vectors, so x may alias out.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Impl {

// One (n, g) group normalized serially inside a single MDRange thread:
// group sizes in the VAE decoder are at most (384/32)*H*W elements, so a
// serial inner loop keeps the kernel portable (Serial-safe, no team
// scratch, no backend intrinsics) at the cost of occupancy the VAE
// decode path does not need (decode is memory-light vs the DiT).
template <class ExecSpace, class Scalar>
void groupnorm_nchw_impl(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, float eps, size_t num_groups,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label) {
  const size_t N = x.extent(0), C = x.extent(1), H = x.extent(2),
               W = x.extent(3);
  if (N == 0 || C == 0 || H == 0 || W == 0) return;
  const size_t Cg = C / num_groups;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, num_groups}),
      KOKKOS_LAMBDA(size_t n, size_t g) {
        double sum = 0.0;
        const size_t c0 = g * Cg;
        for (size_t c = 0; c < Cg; ++c)
          for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w)
              sum += static_cast<double>(x(n, c0 + c, h, w));
        const double count = static_cast<double>(Cg * H * W);
        const double mean = sum / count;
        double sq = 0.0;
        for (size_t c = 0; c < Cg; ++c)
          for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w) {
              const double d =
                  static_cast<double>(x(n, c0 + c, h, w)) - mean;
              sq += d * d;
            }
        const float inv =
            1.0f / sqrtf(static_cast<float>(sq / count) + eps);
        for (size_t c = 0; c < Cg; ++c) {
          const float w = static_cast<float>(weight(c0 + c));
          const float b = static_cast<float>(bias(c0 + c));
          for (size_t h = 0; h < H; ++h)
            for (size_t wdx = 0; wdx < W; ++wdx)
              y(n, c0 + c, h, wdx) = static_cast<Scalar>(
                  (static_cast<float>(x(n, c0 + c, h, wdx)) -
                   static_cast<float>(mean)) *
                      inv * w +
                  b);
        }
      });
}

}  // namespace Impl

// GroupNorm forward (affine):
//   y[n,c,h,w] = (x[n,c,h,w] - mean[n,g]) / sqrt(var[n,g] + eps)
//                * weight[c] + bias[c],  g = c / (C / num_groups).
// x, y: [N, C, H, W] (LayoutRight); weight, bias: [C] (diffusers affine
// GroupNorm always carries both). num_groups must be > 0 and divide C;
// eps must be finite and >= 0 (diffusers VAE uses 1e-6). In-place safe
// (x may alias y). Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void groupnorm_forward(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias, size_t num_groups,
    float eps,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::groupnorm_forward") {
  const size_t N = x.extent(0), C = x.extent(1), H = x.extent(2),
               W = x.extent(3);
  if (y.extent(0) != N || y.extent(1) != C || y.extent(2) != H ||
      y.extent(3) != W || weight.extent(0) != C || bias.extent(0) != C)
    throw std::invalid_argument("groupnorm_forward: dimension mismatch");
  if (num_groups == 0)
    throw std::invalid_argument("groupnorm_forward: num_groups must be > 0");
  if (C % num_groups != 0)
    throw std::invalid_argument(
        "groupnorm_forward: channels must be divisible by num_groups");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("groupnorm_forward: bad eps");
  Impl::groupnorm_nchw_impl<ExecSpace, Scalar>(x, weight, bias, eps,
                                               num_groups, y, label);
}

}  // namespace KokkosDNN

#endif
