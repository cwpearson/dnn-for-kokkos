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
// sites). Native path is a straightforward ijk kernel; when
// KokkosDNN_ENABLE_CUBLAS and ExecSpace is CUDA we route to cublasSgemm.
template <class ExecSpace>
void gemm(const DNNHandle<ExecSpace>& handle, float alpha,
          const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& X,
          const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& W,
          float beta,
          const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& Y,
          const std::string& label = "KokkosDNN::gemm") {
  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(0);
  if (W.extent(1) != K || Y.extent(0) != M || Y.extent(1) != N)
    throw std::invalid_argument("gemm: dimension mismatch");

#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
  constexpr bool is_cuda =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaSpace> ||
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::CudaUVMSpace>;
  if constexpr (is_cuda) {
    if (handle.use_cublas) {
      Kokkos::fence();  // ensure prior Kokkos kernels completed
      bool ok = Impl::cublas_gemm_nt(
          static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
          alpha, X.data(), W.data(), beta, Y.data());
      if (ok) return;
      // else fall through to native (e.g. cuBLAS failure)
    }
  }
#else
  (void)handle;
#endif
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {M, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k) acc += X(i, k) * W(j, k);
        Y(i, j) = alpha * acc + beta * Y(i, j);
      });
}

// Rank-1 (single token) variant used in decode: y[n] = sum_k x[k]*W[n,k].
template <class ExecSpace>
void gemv(const DNNHandle<ExecSpace>&, const Kokkos::View<float*, ExecSpace>& x,
          const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& W,
          const Kokkos::View<float*, ExecSpace>& y,
          const std::string& label = "KokkosDNN::gemv") {
  const size_t K = x.extent(0), N = W.extent(0);
  if (W.extent(1) != K || y.extent(0) != N)
    throw std::invalid_argument("gemv: dimension mismatch");
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k) acc += x(k) * W(j, k);
        y(j) = acc;
      });
}

}  // namespace KokkosDNN

#endif
