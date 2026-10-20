#ifndef KOKKOSDNN_DENSE_HPP
#define KOKKOSDNN_DENSE_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {

// Fully-connected layer Y = X * W^T + b with
//   X : [B, In] activations, W : [Out, In] weights, b : [Out] bias,
//   Y : [B, Out].
// Forward delegates to gemm_bias (which requires a non-empty bias, as
// here), so it inherits the single-pass fusion: cublasLt BIAS epilogue on
// CUDA, bias folded into the native kernel write-back elsewhere. Backward
// is three explicit FP32-accumulation kernels (batch sizes here are tiny;
// no TPL needed):
//   db[o]    = sum_b dY[b,o]
//   dW[o,i]  = sum_b dY[b,o] * X[b,i]
//   dX[b,i]  = sum_o dY[b,o] * W[o,i]

template <class ExecSpace, class Scalar>
void dense_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label = "KokkosDNN::dense_forward") {
  const size_t B = X.extent(0), In = X.extent(1);
  if (W.extent(1) != In || b.extent(0) != W.extent(0) || Y.extent(0) != B ||
      Y.extent(1) != W.extent(0))
    throw std::invalid_argument("dense_forward: dimension mismatch");
  const size_t Out = W.extent(0);
  if (B == 0 || In == 0 || Out == 0) return;
  gemm_bias(handle, X, W, b, Y, label);
}

template <class ExecSpace, class Scalar>
void dense_backward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dY,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dX,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dW,
    const Kokkos::View<Scalar*, ExecSpace>& db,
    const std::string& label = "KokkosDNN::dense_backward") {
  const size_t B = X.extent(0), In = X.extent(1), Out = W.extent(0);
  if (W.extent(1) != In || dY.extent(0) != B || dY.extent(1) != Out ||
      dX.extent(0) != B || dX.extent(1) != In || dW.extent(0) != Out ||
      dW.extent(1) != In || db.extent(0) != Out)
    throw std::invalid_argument("dense_backward: dimension mismatch");
  if (B == 0 || In == 0 || Out == 0) return;

  Kokkos::parallel_for(
      label + "_bias", Kokkos::RangePolicy<ExecSpace>(0, Out),
      KOKKOS_LAMBDA(size_t o) {
        float s = 0;
        for (size_t i = 0; i < B; ++i) s += static_cast<float>(dY(i, o));
        db(o) = static_cast<Scalar>(s);
      });
  Kokkos::parallel_for(
      label + "_dW",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Out, In}),
      KOKKOS_LAMBDA(size_t o, size_t i) {
        float s = 0;
        for (size_t n = 0; n < B; ++n)
          s += static_cast<float>(dY(n, o)) * static_cast<float>(X(n, i));
        dW(o, i) = static_cast<Scalar>(s);
      });
  Kokkos::parallel_for(
      label + "_dX",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, In}),
      KOKKOS_LAMBDA(size_t n, size_t i) {
        float s = 0;
        for (size_t o = 0; o < Out; ++o)
          s += static_cast<float>(dY(n, o)) * static_cast<float>(W(o, i));
        dX(n, i) = static_cast<Scalar>(s);
      });
}

}  // namespace KokkosDNN

#endif
