#ifndef KOKKOSDNN_PATCHEMBED_HPP
#define KOKKOSDNN_PATCHEMBED_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Impl {

// Grid dims for a KxK stride-S conv over HxW (no padding, exact tiling).
// Throws when the kernel does not tile the image exactly; DINOv3 ViT-B/16
// (224/16) is the exact-divisibility case and fractional edge patches have
// no meaning for patch tokens.
inline void patchembed_grid(size_t H, size_t W, size_t K, size_t S,
                            size_t& Hp, size_t& Wp) {
  if (K == 0 || S == 0) throw std::invalid_argument("patchembed: K and S must be nonzero");
  if (H < K || W < K)
    throw std::invalid_argument("patchembed: image smaller than kernel");
  if ((H - K) % S != 0 || (W - K) % S != 0)
    throw std::invalid_argument("patchembed: (H-K)/(W-K) must be divisible by S");
  Hp = (H - K) / S + 1;
  Wp = (W - K) / S + 1;
}

// im2col gather for the patch embed: one row per (batch, patch), one column
// per (channel, kh, kw) in row-major order c*K*K + kh*K + kw, so the result
// multiplies directly against HF Conv2d weights reshaped to [D, C*K*K].
// Plain element copy (no accumulation), hence scalar-agnostic.
template <class ExecSpace, class Scalar>
void patchembed_im2col(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& images,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& patches,
    size_t Hp, size_t Wp, size_t K, size_t S,
    const std::string& label) {
  const size_t B = images.extent(0), C = images.extent(1);
  const size_t H = images.extent(2), W = images.extent(3);
  const size_t Np = Hp * Wp, CKK = C * K * K;
  if (B == 0 || Np == 0 || CKK == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B * Np, CKK}),
      KOKKOS_LAMBDA(size_t row, size_t col) {
        const size_t b  = row / Np;
        const size_t p  = row % Np;
        const size_t ph = p / Wp, pw = p % Wp;
        const size_t c  = col / (K * K);
        const size_t kk = col % (K * K);
        const size_t kh = kk / K, kw = kk % K;
        patches(row, col) = images(b, c, ph * S + kh, pw * S + kw);
      });
  (void)H;
  (void)W;
}

}  // namespace Impl

// ViT patch-embed forward as im2col + biased GEMM (no separate conv kernel):
//   patches = im2col(images)            : [B*Np, C*K*K]
//   flat    = patches * weight^T + bias : [B*Np, D]  (gemm_bias, FP32 accum)
// where Np = Hp*Wp patch tokens, Hp = (H-K)/S+1, Wp = (W-K)/S+1.
// Covers the DINOv3 ViT-B/16 case (C=3, D=768, K=16, S=16, non-overlapping)
// as well as general K/S (overlapping patches when S < K).
//
// Layouts (documented choice): images [B,C,H,W] LayoutRight (CHW-contiguous
// per batch, matching the HWC->CHW preprocessing the dinov3 example plans);
// weight [D, C*K*K] LayoutRight (HF Conv2d weight flattened per output
// channel, same [out_features, in_features] convention as gemm/gemm_bias);
// bias [D] broadcast over patches, or empty (extent 0) for unbiased;
// output [B,Np,D] LayoutRight patch tokens, contiguous (flattened in place
// as [B*Np, D] for the GEMM, so strided subviews are rejected).
// Scalar-agnostic; GEMM accumulation is FP32, cast back to Scalar.
template <class ExecSpace, class Scalar>
void patchembed_forward(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& images,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& weight,
    const Kokkos::View<Scalar*, ExecSpace>& bias,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& output,
    int K, int S,
    const std::string& label = "KokkosDNN::patchembed_forward") {
  if (K <= 0 || S <= 0)
    throw std::invalid_argument("patchembed_forward: K and S must be positive");
  const size_t B = images.extent(0), C = images.extent(1);
  const size_t H = images.extent(2), W = images.extent(3);
  const size_t D = weight.extent(0);
  const bool biased = (bias.extent(0) != 0);
  size_t Hp = 0, Wp = 0;
  Impl::patchembed_grid(H, W, static_cast<size_t>(K), static_cast<size_t>(S), Hp, Wp);
  const size_t Np = Hp * Wp;
  const size_t CKK = C * static_cast<size_t>(K) * static_cast<size_t>(K);
  if (weight.extent(1) != CKK || output.extent(0) != B ||
      output.extent(1) != Np || output.extent(2) != D ||
      (biased && bias.extent(0) != D))
    throw std::invalid_argument("patchembed_forward: dimension mismatch");
  if (!output.span_is_contiguous())
    throw std::invalid_argument("patchembed_forward: output must be contiguous");
  if (B == 0 || D == 0 || CKK == 0) return;

  // im2col scratch: one row per (batch, patch) token.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> patches(
      Kokkos::ViewAllocateWithoutInitializing(label + "_im2col"), B * Np, CKK);
  Impl::patchembed_im2col<ExecSpace, Scalar>(
      images, patches, Hp, Wp, static_cast<size_t>(K), static_cast<size_t>(S),
      label + "_im2col");

  // GEMM directly into output via a flattened [B*Np, D] alias (same
  // row-major strides as [B,Np,D] when contiguous, checked above), so no
  // copy back is needed. gemm_bias reuses gemm's full dispatch ladder
  // (cuBLAS on CUDA, tiled/naive native elsewhere) plus the bias kernel.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> flat(
      output.data(), B * Np, D);
  gemm_bias(handle, patches, weight, bias, flat, label + "_gemm");
}

}  // namespace KokkosDNN

#endif
