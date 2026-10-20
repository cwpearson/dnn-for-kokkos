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
// loop. Used only by the TPL-split fallback in gemm_bias() below (cuBLAS
// sgemm / cublasGemmEx + bias kernel when the fused cublasLt epilogue is
// unavailable) and by the correctness tests as the split reference. The
// primary paths are single-pass fused (cublasLt BIAS epilogue on CUDA,
// bias folded into the native kernel write-back elsewhere).
//
// Flat 1D RangePolicy with j innermost (e = i*N + j): consecutive GPU
// threads step along j, the LayoutRight-contiguous dimension, so the
// Y read-modify-write is coalesced. A Rank<2> MDRangePolicy must NOT be
// used here: on CUDA its default tile steps the leftmost index with
// threadIdx.x (tile {64,4} steps i), i.e. consecutive threads would step
// along i (stride N) -- measured ~8x sector waste on this streaming
// kernel (nsys: bias adds alone cost ~48 ms of a VAE decode's GEMM
// traffic before this fix).
template <class ExecSpace, class Scalar>
void add_bias_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const std::string& label) {
  const size_t M = Y.extent(0), N = Y.extent(1);
  if (M == 0 || N == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, M * N),
      KOKKOS_LAMBDA(size_t e) {
        const size_t i = e / N, j = e % N;
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
//
// Single-pass fusion (no separate bias kernel on the primary paths):
//   * CUDA + cuBLAS, float/bhalf_t: cublasLt GEMM with the BIAS epilogue
//     (CUBLAS_COMPUTE_32F, so bf16 keeps FP32 accumulation; bias added in
//     FP32 before the single output cast). Falls back to cuBLAS
//     sgemm/cublasGemmEx + add_bias_2d when Lt cannot serve the shape.
//   * HIP + hipBLAS, float/bhalf_t: hipBLAS GEMM + add_bias_2d (unchanged
//     split; no hipBLASLt epilogue wired up -- no GEMM-speed regression).
//   * Native (TPL off/unavailable, other scalars): bias is folded into the
//     tiled GEMM (or M == 1 GEMV) write-back, so the native path is exactly
//     one kernel launch.
// The unbiased (empty-bias) path is a plain gemm() call, unchanged.
// Existing gemm/gemv APIs are untouched.
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
  if (!biased) {
    gemm(handle, static_cast<Scalar>(1), X, W, static_cast<Scalar>(0), Y,
         label);
    return;
  }
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
  // Backend named deliberately: cublasLt dispatch genuinely requires CUDA.
  constexpr bool is_cuda = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (is_cuda) {
    if (handle.use_cublas) {
      void* lt = handle.cublaslt_bias_state();
      if (lt != nullptr) {
        // No fence here: the Lt call re-selects the Kokkos CUDA stream on
        // every invocation, so it stays ordered with Kokkos kernels.
        bool ok = false;
        if constexpr (std::is_same_v<Scalar, float>) {
          ok = Impl::cublaslt_gemm_bias_nt_f32_with_state(
              lt, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), 1.0f, X.data(), W.data(), b.data(),
              Y.data());
        } else if constexpr (std::is_same_v<
                                 Scalar, Kokkos::Experimental::bhalf_t>) {
          ok = Impl::cublaslt_gemm_bias_nt_bf16_with_state(
              lt, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), 1.0f, X.data(), W.data(), b.data(),
              Y.data());
        }
        if (ok) return;
        // else fall through to the split TPL path below (keeps cuBLAS
        // GEMM speed; the bias kernel is today's behavior there).
      }
      // Split TPL fallback: the plain-gemm ladder (sgemm / cublasGemmEx)
      // plus the bias kernel. Only for scalars the ladder serves; other
      // scalars (e.g. half_t) go straight to the fused native path.
      if constexpr (std::is_same_v<Scalar, float> ||
                    std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
        if (handle.cublas_raw() != nullptr) {
          gemm(handle, static_cast<Scalar>(1), X, W, static_cast<Scalar>(0),
               Y, label);
          Impl::add_bias_2d<ExecSpace, Scalar>(Y, b, label + "_bias");
          return;
        }
      }
    }
  }
#endif
#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
  // Backend named deliberately: hipBLAS dispatch genuinely requires HIP.
  // Split only (no fused epilogue): today's behavior, no regression.
  constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
  if constexpr (is_hip) {
    if (handle.use_hipblas &&
        (std::is_same_v<Scalar, float> ||
         std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) &&
        handle.hipblas_raw() != nullptr) {
      gemm(handle, static_cast<Scalar>(1), X, W, static_cast<Scalar>(0), Y,
           label);
      Impl::add_bias_2d<ExecSpace, Scalar>(Y, b, label + "_bias");
      return;
    }
  }
#endif
  // Native fused single pass: rank-2 M == 1 reuses the team-per-row GEMV
  // kernel with fused bias (same working set), M > 1 uses the tiled GEMM
  // with bias folded into write-back.
  if (M == 1) {
    Kokkos::View<Scalar*, ExecSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        x1(X.data(), K), y1(Y.data(), N);
    Impl::gemv_native<ExecSpace, Scalar>(x1, W, 1.0f, 0.0f, y1, label,
                                         b.data());
    return;
  }
  Impl::gemm_tiled<ExecSpace, Scalar>(X, W, 1.0f, 0.0f, Y, label, b.data());
}

// Rank-1 (single token) variant: y = W * x + b with x : [K], W : [N, K],
// b : [N] (or empty for unbiased), y : [N]. Same single-pass fusion as
// the rank-2 path (cublasLt M == 1 bias epilogue on CUDA, bias folded
// into the native GEMV write-back elsewhere); unbiased reuses gemv().
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
  if (!biased) {
    gemv(handle, x, W, y, label);
    return;
  }
#if defined(KOKKOSDNN_ENABLE_CUBLAS) && defined(KOKKOS_ENABLE_CUDA)
  // Backend named deliberately: cublasLt dispatch genuinely requires CUDA.
  constexpr bool is_cuda_gv = std::is_same_v<ExecSpace, Kokkos::Cuda>;
  if constexpr (is_cuda_gv) {
    if (handle.use_cublas) {
      void* lt = handle.cublaslt_bias_state();
      if (lt != nullptr) {
        bool ok = false;
        if constexpr (std::is_same_v<Scalar, float>) {
          ok = Impl::cublaslt_gemm_bias_nt_f32_with_state(
              lt, 1, static_cast<int>(N), static_cast<int>(K), 1.0f, x.data(),
              W.data(), b.data(), y.data());
        } else if constexpr (std::is_same_v<
                                 Scalar, Kokkos::Experimental::bhalf_t>) {
          ok = Impl::cublaslt_gemm_bias_nt_bf16_with_state(
              lt, 1, static_cast<int>(N), static_cast<int>(K), 1.0f, x.data(),
              W.data(), b.data(), y.data());
        }
        if (ok) return;
      }
      if constexpr (std::is_same_v<Scalar, float> ||
                    std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
        if (handle.cublas_raw() != nullptr) {
          gemv(handle, x, W, y, label);
          Impl::add_bias_1d<ExecSpace, Scalar>(y, b, label + "_bias");
          return;
        }
      }
    }
  }
#endif
#if defined(KOKKOSDNN_ENABLE_HIPBLAS) && defined(KOKKOS_ENABLE_HIP)
  // Backend named deliberately: hipBLAS dispatch genuinely requires HIP.
  // Split only (no fused epilogue): today's behavior, no regression.
  constexpr bool is_hip_gv = std::is_same_v<ExecSpace, Kokkos::HIP>;
  if constexpr (is_hip_gv) {
    if (handle.use_hipblas &&
        (std::is_same_v<Scalar, float> ||
         std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) &&
        handle.hipblas_raw() != nullptr) {
      gemv(handle, x, W, y, label);
      Impl::add_bias_1d<ExecSpace, Scalar>(y, b, label + "_bias");
      return;
    }
  }
#endif
  // Native fused single pass (bias folded into the GEMV write-back).
  Impl::gemv_native<ExecSpace, Scalar>(x, W, 1.0f, 0.0f, y, label, b.data());
}

}  // namespace KokkosDNN

#endif
