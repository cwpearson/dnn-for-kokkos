#ifndef KOKKOSDNN_GEMMBIAS_HPP
#define KOKKOSDNN_GEMMBIAS_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Impl {

// Broadcast-bias add for the rank-2 path: Y(i,j) += b(j) in FP32.
// Separate efficient device kernel (one element per thread), not a host
// loop. Called after gemm() so the cuBLAS/TPL dispatch inside gemm() is
// reused as-is (cuBLAS has no bias operand; same split as dense_forward).
template <class ExecSpace, class Scalar>
void add_bias_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const std::string& label) {
  const size_t M = Y.extent(0), N = Y.extent(1);
  if (M == 0 || N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {M, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        Y(i, j) = static_cast<Scalar>(static_cast<float>(Y(i, j)) +
                                      static_cast<float>(b(j)));
      });
}

// Rank-1 variant: y(j) += b(j) in FP32.
template <class ExecSpace, class Scalar>
void add_bias_1d(const Kokkos::View<Scalar*, ExecSpace>& y,
                 const Kokkos::View<Scalar*, ExecSpace>& b,
                 const std::string& label) {
  const size_t N = y.extent(0);
  if (N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t j) {
        y(j) = static_cast<Scalar>(static_cast<float>(y(j)) +
                                   static_cast<float>(b(j)));
      });
}

}  // namespace Impl

// Biased linear for inference: Y = X * W^T + b with
//   X : [M, K] activations (flatten leading [B, T] dims to [B*T, K] for
//       batched tokens; LayoutRight keeps rows contiguous so the reshape
//       is free when the view is contiguous),
//   W : [N, K] weights (HF Linear: [out_features, in_features], no
//       explicit transpose needed),
//   b : [N] bias broadcast over rows, or empty (extent 0, e.g.
//       default-constructed view) for the unbiased path (DINOv3 k proj),
//   Y : [M, N].
// Scalar-agnostic (float, half_t, bhalf_t); accumulation is FP32 and the
// result is cast back to Scalar.
// cuBLAS reuse: the GEMM part is a plain gemm() call (alpha=1, beta=0),
// so the full dispatch ladder in KokkosDNN_Gemm.hpp applies unchanged:
// cuBLAS sgemm / cublasGemmEx on CUDA for float/bhalf_t via the handle's
// cached handle, tiled fallback on CUDA, naive MDRange elsewhere. The
// bias broadcast is a separate device kernel afterwards (same split as
// dense_forward); existing gemm/gemv APIs are untouched.
template <class ExecSpace, class Scalar>
void gemm_bias(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "KokkosDNN::gemm_bias") {
  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (W.extent(1) != K || Y.extent(0) != M || Y.extent(1) != N ||
      (biased && b.extent(0) != N))
    throw std::invalid_argument("gemm_bias: dimension mismatch");
  if (M == 0 || N == 0 || K == 0) return;
  gemm(handle, static_cast<Scalar>(1), X, W, static_cast<Scalar>(0), Y, label);
  if (biased) Impl::add_bias_2d<ExecSpace, Scalar>(Y, b, label + "_bias");
}

// Rank-1 (single token) variant: y = W * x + b with x : [K], W : [N, K],
// b : [N] (or empty for unbiased), y : [N]. Reuses gemv()'s dispatch
// ladder the same way the rank-2 path reuses gemm().
template <class ExecSpace, class Scalar>
void gemv_bias(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar*, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar*, ExecSpace>& y,
    const std::string& label = "KokkosDNN::gemv_bias") {
  const size_t K = x.extent(0), N = W.extent(0);
  const bool biased = (b.extent(0) != 0);
  if (W.extent(1) != K || y.extent(0) != N || (biased && b.extent(0) != N))
    throw std::invalid_argument("gemv_bias: dimension mismatch");
  if (N == 0 || K == 0) return;
  gemv(handle, x, W, y, label);
  if (biased) Impl::add_bias_1d<ExecSpace, Scalar>(y, b, label + "_bias");
}

}  // namespace KokkosDNN

#endif
