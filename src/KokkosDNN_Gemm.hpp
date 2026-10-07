#ifndef KOKKOSDNN_GEMM_HPP
#define KOKKOSDNN_GEMM_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_Gemm_CuBLAS.hpp"

namespace KokkosDNN {

// Y = alpha * X * W^T + beta * Y
//   X : [M, K] row-major activations
//   W : [N, K] row-major weights (HF Linear: [out_features, in_features])
//   Y : [M, N]
// This is the only GEMM form the Llama PoC needs (no transposes at call
// sites). Scalar-agnostic (float, bhalf_t); accumulation is FP32 and the
// result is cast back to Scalar. When KokkosDNN_ENABLE_CUBLAS and ExecSpace
// is CUDA we route float to cublasSgemm; other scalars use native Kokkos.
template <class ExecSpace, class Scalar>
void gemm(const DNNHandle<ExecSpace>& handle, Scalar alpha,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
          Scalar beta,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
          const std::string& label = "KokkosDNN::gemm") {
  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
  if (W.extent(1) != K || Y.extent(0) != M || Y.extent(1) != N)
    throw std::invalid_argument("gemm: dimension mismatch");

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
  constexpr bool is_cuda =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaSpace> ||
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaUVMSpace>;
  if constexpr (is_cuda) {
    if constexpr (std::is_same_v<Scalar, float>) {
      if (handle.use_cublas) {
        Kokkos::fence();  // ensure prior Kokkos kernels completed
        bool ok = Impl::cublas_gemm_nt(
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            alpha, X.data(), W.data(), beta, Y.data());
        if (ok) return;
        // else fall through to native (e.g. cuBLAS failure)
      }
    }
  }
#else
  (void)handle;
#endif
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f  = static_cast<float>(beta);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {M, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k)
          acc += static_cast<float>(X(i, k)) * static_cast<float>(W(j, k));
        Y(i, j) = static_cast<Scalar>(alpha_f * acc +
                                      beta_f * static_cast<float>(Y(i, j)));
      });
}

// Rank-1 (single token) variant used in decode: y[n] = sum_k x[k]*W[n,k].
template <class ExecSpace, class Scalar>
void gemv(const DNNHandle<ExecSpace>&,
          const Kokkos::View<Scalar*, ExecSpace>& x,
          const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
          const Kokkos::View<Scalar*, ExecSpace>& y,
          const std::string& label = "KokkosDNN::gemv") {
  const size_t K = x.extent(0), N = W.extent(0);
  if (W.extent(1) != K || y.extent(0) != N)
    throw std::invalid_argument("gemv: dimension mismatch");
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k)
          acc += static_cast<float>(x(k)) * static_cast<float>(W(j, k));
        y(j) = static_cast<Scalar>(acc);
      });
}

}  // namespace KokkosDNN

#endif
