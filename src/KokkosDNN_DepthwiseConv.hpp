#ifndef KOKKOSDNN_DEPTHWISECONV_HPP
#define KOKKOSDNN_DEPTHWISECONV_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

// Generic depthwise 3x3 convolution: stride 1, pad 1 (zero padding),
// groups = Cin (one 3x3 kernel per channel).
//
// Token views are LayoutRight [T, C] with T = H*W row-major (t = h*W + w),
// or batched [B, T, C]. Depthwise weights are [Cin, 3, 3] LayoutRight.
// Bias is [Cin] or empty (extent 0) for no bias.
// All accumulation is FP32, cast back to Scalar. Backend-portable: only
// parallel_for MDRange kernels; no backend-specific code.

namespace KokkosDNN {
namespace Impl {

// Depthwise 3x3, stride 1, pad 1 (zero padding), groups=Cin. Rank-2 token
// path: x/y [T, C] with T = H*W row-major, w [C, 3, 3], b [C] (or empty).
template <class ExecSpace, class Scalar>
void depthwise3x3_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t T = x.extent(0), C = x.extent(1);
  if (T == 0 || C == 0) return;
  const bool biased = (b.extent(0) != 0);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const size_t h = t / W, ww = t % W;
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = static_cast<int>(h) + kh - 1;
          if (ih < 0 || static_cast<size_t>(ih) >= H) continue;
          for (int kw = 0; kw < 3; ++kw) {
            const int iw = static_cast<int>(ww) + kw - 1;
            if (iw < 0 || static_cast<size_t>(iw) >= W) continue;
            const size_t s =
                static_cast<size_t>(ih) * W + static_cast<size_t>(iw);
            acc += static_cast<float>(x(s, c)) *
                   static_cast<float>(w(c, static_cast<size_t>(kh),
                                          static_cast<size_t>(kw)));
          }
        }
        y(t, c) = static_cast<Scalar>(acc);
      });
}

// Batched rank-3 variant: x/y [B, T, C], weights shared across the batch.
template <class ExecSpace, class Scalar>
void depthwise3x3_3d(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t B = x.extent(0), T = x.extent(1), C = x.extent(2);
  if (B == 0 || T == 0 || C == 0) return;
  const bool biased = (b.extent(0) != 0);
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, C}),
      KOKKOS_LAMBDA(size_t bb, size_t t, size_t c) {
        const size_t h = t / W, ww = t % W;
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = static_cast<int>(h) + kh - 1;
          if (ih < 0 || static_cast<size_t>(ih) >= H) continue;
          for (int kw = 0; kw < 3; ++kw) {
            const int iw = static_cast<int>(ww) + kw - 1;
            if (iw < 0 || static_cast<size_t>(iw) >= W) continue;
            const size_t s =
                static_cast<size_t>(ih) * W + static_cast<size_t>(iw);
            acc += static_cast<float>(x(bb, s, c)) *
                   static_cast<float>(w(c, static_cast<size_t>(kh),
                                          static_cast<size_t>(kw)));
          }
        }
        y(bb, t, c) = static_cast<Scalar>(acc);
      });
}

}  // namespace Impl

// Depthwise 3x3 convolution, stride 1, pad 1 (zero padding), groups=Cin:
//   y(t,c) = b(c) + sum_{kh,kw} w(c,kh,kw) * x_pad(h+kh-1, w+kw-1, c)
// with t = h*W + w. x/y [T, C] LayoutRight with T = H*W, w [C, 3, 3]
// LayoutRight, b [C] or empty (extent 0) for no bias. Scalar-agnostic;
// FP32 accumulation. Backend-portable MDRange kernel.
template <class ExecSpace, class Scalar>
void depthwise_conv3x3_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W,
    const std::string& label = "KokkosDNN::depthwise_conv3x3_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("depthwise_conv3x3_forward: H and W must be nonzero");
  const size_t T = x.extent(0), C = x.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (T != H * W || w.extent(0) != C || w.extent(1) != 3 || w.extent(2) != 3 ||
      y.extent(0) != T || y.extent(1) != C || (biased && b.extent(0) != C))
    throw std::invalid_argument("depthwise_conv3x3_forward: dimension mismatch");
  if (T == 0 || C == 0) return;
  Impl::depthwise3x3_2d<ExecSpace, Scalar>(x, w, b, y, H, W, label);
}

// Batched variant: x/y [B, T, C], weights shared across the batch.
template <class ExecSpace, class Scalar>
void depthwise_conv3x3_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W,
    const std::string& label = "KokkosDNN::depthwise_conv3x3_forward") {
  if (H == 0 || W == 0)
    throw std::invalid_argument("depthwise_conv3x3_forward: H and W must be nonzero");
  const size_t B = x.extent(0), T = x.extent(1), C = x.extent(2);
  const bool biased = (b.extent(0) != 0);
  if (T != H * W || w.extent(0) != C || w.extent(1) != 3 || w.extent(2) != 3 ||
      y.extent(0) != B || y.extent(1) != T || y.extent(2) != C ||
      (biased && b.extent(0) != C))
    throw std::invalid_argument("depthwise_conv3x3_forward: dimension mismatch");
  if (B == 0 || T == 0 || C == 0) return;
  Impl::depthwise3x3_3d<ExecSpace, Scalar>(x, w, b, y, H, W, label);
}

}  // namespace KokkosDNN

#endif
