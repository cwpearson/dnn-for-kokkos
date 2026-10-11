#ifndef KOKKOSDNN_LAYERSCALE_HPP
#define KOKKOSDNN_LAYERSCALE_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace KokkosDNN {
namespace Impl {

// Dimension checks for the LayerScale helpers. All extents are passed
// explicitly so the thrown message can report the offending shapes.
inline void layerscale_check_batched(size_t B, size_t T, size_t D, size_t gammaD,
                                     size_t bB, size_t bT, size_t bD,
                                     size_t oB, size_t oT, size_t oD,
                                     const char* what) {
  const std::string w(what);
  if (gammaD != D)
    throw std::invalid_argument(w + ": gamma must be [D] (got " +
                                std::to_string(gammaD) + ", expected " +
                                std::to_string(D) + ")");
  if (bB != B || bT != T || bD != D)
    throw std::invalid_argument(w + ": branch must be [B,T,D] (got [" +
                                std::to_string(bB) + "," + std::to_string(bT) +
                                "," + std::to_string(bD) + "], expected [" +
                                std::to_string(B) + "," + std::to_string(T) +
                                "," + std::to_string(D) + "])");
  if (oB != B || oT != T || oD != D)
    throw std::invalid_argument(w + ": output must be [B,T,D] (got [" +
                                std::to_string(oB) + "," + std::to_string(oT) +
                                "," + std::to_string(oD) + "], expected [" +
                                std::to_string(B) + "," + std::to_string(T) +
                                "," + std::to_string(D) + "])");
}

inline void layerscale_check_single(size_t T, size_t D, size_t gammaD,
                                    size_t bT, size_t bD, size_t oT, size_t oD,
                                    const char* what) {
  const std::string w(what);
  if (gammaD != D)
    throw std::invalid_argument(w + ": gamma must be [D] (got " +
                                std::to_string(gammaD) + ", expected " +
                                std::to_string(D) + ")");
  if (bT != T || bD != D)
    throw std::invalid_argument(w + ": branch must be [T,D] (got [" +
                                std::to_string(bT) + "," + std::to_string(bD) +
                                "], expected [" + std::to_string(T) + "," +
                                std::to_string(D) + "])");
  if (oT != T || oD != D)
    throw std::invalid_argument(w + ": output must be [T,D] (got [" +
                                std::to_string(oT) + "," + std::to_string(oD) +
                                "], expected [" + std::to_string(T) + "," +
                                std::to_string(D) + "])");
}

inline void layerscale_check_vector(size_t D, size_t gammaD, size_t bD,
                                    size_t oD, const char* what) {
  const std::string w(what);
  if (gammaD != D)
    throw std::invalid_argument(w + ": gamma must be [D] (got " +
                                std::to_string(gammaD) + ", expected " +
                                std::to_string(D) + ")");
  if (bD != D)
    throw std::invalid_argument(w + ": branch must be [D] (got " +
                                std::to_string(bD) + ", expected " +
                                std::to_string(D) + ")");
  if (oD != D)
    throw std::invalid_argument(w + ": output must be [D] (got " +
                                std::to_string(oD) + ", expected " +
                                std::to_string(D) + ")");
}

}  // namespace Impl

// Per-block LayerScale (DINOv3 ViT / CaiT style) plus the fused residual add:
//
//   layerscale_forward:          out = gamma * branch
//   layerscale_residual_forward: out = x + gamma * branch
//
// gamma is per-channel [D] (learned, typically small init ~1e-4..1) and is
// broadcast over the batch and token dims. [B,T,D] batched form follows the
// repo token policy (LayoutRight, token dim is the middle extent, each
// (b, t) row a contiguous D-lane); [T,D] covers the single-sequence path
// and [D] the trivial rank-1 case.
//
// All arithmetic is accumulated in FP32, then cast back to Scalar, so the
// helpers are scalar-agnostic (float, double, bhalf_t, ...). Every entry
// point is in-place safe: each output element depends only on the
// same-index inputs plus gamma(d), so out may alias x and/or branch
// (including x == branch == out).
//
// DropPath (stochastic depth) is the inference identity: at eval time the
// survival probability is 1, so the residual branch is always kept and no
// stochastic kernel is needed. There is deliberately no DropPath op in this
// header; inference callers just use layerscale_residual_forward as is.
// (Training-time stochastic depth would be a separate op, out of scope for
// the inference library.)

// Standalone LayerScale multiply: out = gamma * branch, gamma [D]
// broadcast over [B,T,D].
template <class ExecSpace, class Scalar>
void layerscale_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& branch,
    const Kokkos::View<Scalar*, ExecSpace>& gamma,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::layerscale_forward") {
  const size_t B = branch.extent(0), T = branch.extent(1), D = branch.extent(2);
  Impl::layerscale_check_batched(B, T, D, gamma.extent(0), B, T, D,
                                 out.extent(0), out.extent(1), out.extent(2),
                                 "layerscale_forward");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                              {B, T, D}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t d) {
        out(b, t, d) = static_cast<Scalar>(static_cast<float>(branch(b, t, d)) *
                                           static_cast<float>(gamma(d)));
      });
}

// Single-sequence form: out = gamma * branch over [T,D].
template <class ExecSpace, class Scalar>
void layerscale_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& branch,
    const Kokkos::View<Scalar*, ExecSpace>& gamma,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::layerscale_forward") {
  const size_t T = branch.extent(0), D = branch.extent(1);
  Impl::layerscale_check_single(T, D, gamma.extent(0), T, D, out.extent(0),
                                out.extent(1), "layerscale_forward");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        out(t, d) = static_cast<Scalar>(static_cast<float>(branch(t, d)) *
                                        static_cast<float>(gamma(d)));
      });
}

// Rank-1 form: out = gamma * branch over [D].
template <class ExecSpace, class Scalar>
void layerscale_forward(
    const Kokkos::View<Scalar*, ExecSpace>& branch,
    const Kokkos::View<Scalar*, ExecSpace>& gamma,
    const Kokkos::View<Scalar*, ExecSpace>& out,
    const std::string& label = "KokkosDNN::layerscale_forward") {
  const size_t D = branch.extent(0);
  Impl::layerscale_check_vector(D, gamma.extent(0), D, out.extent(0),
                                "layerscale_forward");
  if (D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        out(d) = static_cast<Scalar>(static_cast<float>(branch(d)) *
                                     static_cast<float>(gamma(d)));
      });
}

// Fused LayerScale + residual: out = x + gamma * branch over [B,T,D].
// gamma = 1 degrades to a plain residual add; gamma = 0 returns x.
template <class ExecSpace, class Scalar>
void layerscale_residual_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& branch,
    const Kokkos::View<Scalar*, ExecSpace>& gamma,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::layerscale_residual_forward") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  Impl::layerscale_check_batched(B, T, D, gamma.extent(0), branch.extent(0),
                                 branch.extent(1), branch.extent(2),
                                 out.extent(0), out.extent(1), out.extent(2),
                                 "layerscale_residual_forward");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                              {B, T, D}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t d) {
        out(b, t, d) = static_cast<Scalar>(
            static_cast<float>(x(b, t, d)) +
            static_cast<float>(gamma(d)) * static_cast<float>(branch(b, t, d)));
      });
}

// Single-sequence form: out = x + gamma * branch over [T,D].
template <class ExecSpace, class Scalar>
void layerscale_residual_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& branch,
    const Kokkos::View<Scalar*, ExecSpace>& gamma,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::layerscale_residual_forward") {
  const size_t T = x.extent(0), D = x.extent(1);
  Impl::layerscale_check_single(T, D, gamma.extent(0), branch.extent(0),
                                branch.extent(1), out.extent(0), out.extent(1),
                                "layerscale_residual_forward");
  if (T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
      KOKKOS_LAMBDA(size_t t, size_t d) {
        out(t, d) = static_cast<Scalar>(
            static_cast<float>(x(t, d)) + static_cast<float>(gamma(d)) *
                                              static_cast<float>(branch(t, d)));
      });
}

// Rank-1 form: out = x + gamma * branch over [D].
template <class ExecSpace, class Scalar>
void layerscale_residual_forward(
    const Kokkos::View<Scalar*, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& branch,
    const Kokkos::View<Scalar*, ExecSpace>& gamma,
    const Kokkos::View<Scalar*, ExecSpace>& out,
    const std::string& label = "KokkosDNN::layerscale_residual_forward") {
  const size_t D = x.extent(0);
  Impl::layerscale_check_vector(D, gamma.extent(0), branch.extent(0),
                                out.extent(0), "layerscale_residual_forward");
  if (D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, D), KOKKOS_LAMBDA(size_t d) {
        out(d) = static_cast<Scalar>(static_cast<float>(x(d)) +
                                     static_cast<float>(gamma(d)) *
                                         static_cast<float>(branch(d)));
      });
}

}  // namespace KokkosDNN

#endif
