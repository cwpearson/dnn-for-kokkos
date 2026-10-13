#ifndef KOKKOSDNN_CONV_HPP
#define KOKKOSDNN_CONV_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Handle.hpp"

// Spatial convolutions over token-row activations [T, C] with T = H*W
// row-major (t = h*W + w). That layout is exactly N=1 NHWC memory order,
// so the cuDNN dispatch below passes device pointers through zero-copy.
//
// Three operators, all stride 1 with symmetric zero padding:
//
//   conv3x3_forward         dense 3x3, pad 1, optional bias
//   depthwise_conv_forward  depthwise KxK (runtime odd K), pad K/2,
//                           optional bias
//   grouped_conv1x1_forward grouped 1x1, no padding, optional bias
//
// Scalar-agnostic (float, bhalf_t); accumulation is FP32, cast back to
// Scalar. The native kernels are backend-portable parallel_for only (no
// backend-specific code outside src/tpl).
//
// TPL status (investigated 2026-10, see task/p1-vae-cudnn-conv): there is
// deliberately NO cuDNN dispatch here. A dispatch attempt was built and
// reverted: the legacy cudnnConvolution* entry points live in the
// cudnn_cnn sublibrary, which is absent wherever only the loader plus
// cudnn_graph ship (this project's GPU container), failing with
// CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE; and the
// cudnn-frontend conv_fprop node at the pinned commit has no group-count
// knob (ruling out depthwise/grouped) while its only eligible dense
// engine on sm86 (engine 7, NVRTC-JIT) both silently requires
// C-innermost (KRSC) filters and computes FP32 I/O at TF32 precision
// (~6e-4 relative error vs native). Neither API is sound here, so all
// three operators run the native kernel. Revisit only with (a) full
// cuDNN sublibraries for a legacy grouped path, or (b) an eligible
// non-tensor-core dense engine (FORT is arch-gated to Hopper/Blackwell).
//
// Bias convention (matches gemm_bias / depthwise_conv3x3_forward): b is
// [Cout] (or [C] for depthwise), or empty (extent 0, e.g. a
// default-constructed view) for the unbiased path.

namespace KokkosDNN {
namespace Impl {

// Dense 3x3, stride 1, pad 1 (zero padding). Rank-2 token path: x [T,Cin]
// with T = H*W row-major, w [Cout,Cin,3,3], b [Cout] (or empty), y [T,Cout].
template <class ExecSpace, class Scalar>
void conv_dense3x3_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (T == 0 || Cin == 0 || Cout == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Cout}),
      KOKKOS_LAMBDA(size_t t, size_t o) {
        const size_t h = t / W, ww = t % W;
        float acc = biased ? static_cast<float>(b(o)) : 0.0f;
        for (size_t i = 0; i < Cin; ++i)
          for (int kh = 0; kh < 3; ++kh) {
            const int ih = static_cast<int>(h) + kh - 1;
            if (ih < 0 || static_cast<size_t>(ih) >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
              const int iw = static_cast<int>(ww) + kw - 1;
              if (iw < 0 || static_cast<size_t>(iw) >= W) continue;
              acc += static_cast<float>(
                         x(static_cast<size_t>(ih) * W + static_cast<size_t>(iw),
                           i)) *
                     static_cast<float>(w(o, i, static_cast<size_t>(kh),
                                          static_cast<size_t>(kw)));
            }
          }
        y(t, o) = static_cast<Scalar>(acc);
      });
}

// Depthwise KxK, stride 1, pad K/2 (zero padding), groups = C. Rank-2
// token path: x/y [T,C] with T = H*W row-major, w [C,K,K], b [C] (or
// empty). K must be odd (checked by the caller) so the padding is
// symmetric and the native kernel agrees with the cuDNN dispatch.
template <class ExecSpace, class Scalar>
void conv_depthwise_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t K,
    size_t H, size_t W, const std::string& label) {
  const size_t T = x.extent(0), C = x.extent(1);
  const int P = static_cast<int>(K / 2);
  const bool biased = (b.extent(0) != 0);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
      KOKKOS_LAMBDA(size_t t, size_t c) {
        const size_t h = t / W, ww = t % W;
        float acc = biased ? static_cast<float>(b(c)) : 0.0f;
        for (size_t kh = 0; kh < K; ++kh) {
          const int ih = static_cast<int>(h) + static_cast<int>(kh) - P;
          if (ih < 0 || static_cast<size_t>(ih) >= H) continue;
          for (size_t kw = 0; kw < K; ++kw) {
            const int iw = static_cast<int>(ww) + static_cast<int>(kw) - P;
            if (iw < 0 || static_cast<size_t>(iw) >= W) continue;
            acc += static_cast<float>(
                       x(static_cast<size_t>(ih) * W + static_cast<size_t>(iw),
                         c)) *
                   static_cast<float>(w(c, kh, kw));
          }
        }
        y(t, c) = static_cast<Scalar>(acc);
      });
}

// Grouped 1x1 convolution (no padding). x [T,Cin], w [Cout,Cg] (Cg =
// per-group input channels), b [Cout] (or empty), y [T,Cout]. Group g
// owns outputs [g*Co..(g+1)*Co) and inputs [g*Cg..(g+1)*Cg) with
// Co = Cout/groups.
template <class ExecSpace, class Scalar>
void conv_grouped1x1_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    size_t groups, const std::string& label) {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const size_t Cg = w.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (T == 0 || Cin == 0 || Cout == 0) return;
  const size_t Co = Cout / groups;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Cout}),
      KOKKOS_LAMBDA(size_t t, size_t o) {
        const size_t g = o / Co;
        float acc = biased ? static_cast<float>(b(o)) : 0.0f;
        for (size_t k = 0; k < Cg; ++k)
          acc += static_cast<float>(x(t, g * Cg + k)) *
                 static_cast<float>(w(o, k));
        y(t, o) = static_cast<Scalar>(acc);
      });
}

}  // namespace Impl

// Dense 3x3 convolution forward, stride 1, pad 1 (zero padding):
//   y(t,o) = b(o) + sum_{i,kh,kw} w(o,i,kh,kw) * x_pad(h+kh-1, w+kw-1, i)
// with t = h*W + w. x [T,Cin] with T = H*W, w [Cout,Cin,3,3], b [Cout]
// or empty for no bias, y [T,Cout]. Scalar-agnostic; FP32 accumulation.
//
// NOTE: the handle is currently unused (no cuDNN dispatch, see file
// docs); it is still taken for API uniformity with the other operators
// (and any future dispatch).
template <class ExecSpace, class Scalar>
void conv3x3_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W,
    const std::string& label = "KokkosDNN::conv3x3_forward") {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (w.extent(1) != Cin || w.extent(2) != 3 || w.extent(3) != 3 ||
      y.extent(0) != T || y.extent(1) != Cout ||
      (biased && b.extent(0) != Cout))
    throw std::invalid_argument("conv3x3_forward: dimension mismatch");
  if (H == 0 || W == 0 || T != H * W)
    throw std::invalid_argument("conv3x3_forward: H and W must be nonzero "
                                "with H*W == rows");
  if (T == 0 || Cin == 0 || Cout == 0) return;
  (void)handle;
  Impl::conv_dense3x3_2d<ExecSpace, Scalar>(x, w, b, y, H, W, label);
}

// Depthwise convolution forward, stride 1, symmetric zero padding, one
// KxK kernel per channel (groups = C):
//   y(t,c) = b(c) + sum_{kh,kw} w(c,kh,kw) * x_pad(h+kh-P, w+kw-P, c)
// with t = h*W + w and P = K/2. x/y [T,C] with T = H*W, w [C,K,K],
// b [C] or empty for no bias. K must be odd (even K has no symmetric
// padding, so the call is rejected). Scalar-agnostic; FP32 accumulation.
//
// NOTE: no cuDNN dispatch exists for this operator (see file docs), so
// the handle is currently unused; it is still taken for API uniformity
// with conv3x3_forward (and any future dispatch).
template <class ExecSpace, class Scalar>
void depthwise_conv_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y, size_t K,
    size_t H, size_t W,
    const std::string& label = "KokkosDNN::depthwise_conv_forward") {
  const size_t T = x.extent(0), C = x.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (K == 0 || K % 2 == 0)
    throw std::invalid_argument(
        "depthwise_conv_forward: K must be a positive odd kernel size");
  if (w.extent(0) != C || w.extent(1) != K || w.extent(2) != K ||
      y.extent(0) != T || y.extent(1) != C || (biased && b.extent(0) != C))
    throw std::invalid_argument("depthwise_conv_forward: dimension mismatch");
  if (H == 0 || W == 0 || T != H * W)
    throw std::invalid_argument("depthwise_conv_forward: H and W must be "
                                "nonzero with H*W == rows");
  if (T == 0 || C == 0) return;
  (void)handle;
  Impl::conv_depthwise_2d<ExecSpace, Scalar>(x, w, b, y, K, H, W, label);
}

// Grouped 1x1 convolution forward (pointwise, no padding):
//   y(t,o) = b(o) + sum_k w(o,k) * x(t, g*Cg+k),  g = o/(Cout/groups)
// x [T,Cin], w [Cout,Cg] with Cg = Cin/groups, b [Cout] or empty for no
// bias, y [T,Cout]. Cout and Cin must be divisible by groups.
// Scalar-agnostic; FP32 accumulation.
//
// NOTE: no cuDNN dispatch exists for this operator (see file docs), so
// the handle is currently unused; it is still taken for API uniformity
// with conv3x3_forward (and any future dispatch).
template <class ExecSpace, class Scalar>
void grouped_conv1x1_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& w,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& y,
    size_t groups,
    const std::string& label = "KokkosDNN::grouped_conv1x1_forward") {
  const size_t T = x.extent(0), Cin = x.extent(1), Cout = w.extent(0);
  const size_t Cg = w.extent(1);
  const bool biased = (b.extent(0) != 0);
  if (groups == 0)
    throw std::invalid_argument(
        "grouped_conv1x1_forward: groups must be nonzero");
  if (Cout % groups != 0 || Cin != groups * Cg || y.extent(0) != T ||
      y.extent(1) != Cout || (biased && b.extent(0) != Cout))
    throw std::invalid_argument(
        "grouped_conv1x1_forward: dimension mismatch");
  if (T == 0 || Cin == 0 || Cout == 0) return;
  (void)handle;
  Impl::conv_grouped1x1_2d<ExecSpace, Scalar>(x, w, b, y, groups, label);
}

}  // namespace KokkosDNN

#endif
