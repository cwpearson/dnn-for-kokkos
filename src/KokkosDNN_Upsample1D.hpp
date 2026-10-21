#ifndef KOKKOSDNN_UPSAMPLE1D_HPP
#define KOKKOSDNN_UPSAMPLE1D_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace KokkosDNN {

// 1D upsampling over [B, C, T] (LayoutRight).
// nearest_upsample1d: y[b,c,to] = x[b,c,to/scale] (integer scale, floor).
// linear_upsample1d: PyTorch align_corners=False linear interpolate to
//   explicit out_len. Matches F.interpolate(mode="linear",
//   align_corners=False) for the 1D case used by SineGen.

template <class ExecSpace, class Scalar>
void nearest_upsample1d(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    size_t scale,
    const std::string& label = "KokkosDNN::nearest_upsample1d") {
  const size_t B = x.extent(0), C = x.extent(1), Tin = x.extent(2);
  if (scale == 0) throw std::invalid_argument("nearest_upsample1d: scale==0");
  if (y.extent(0) != B || y.extent(1) != C || y.extent(2) != Tin * scale)
    throw std::invalid_argument("nearest_upsample1d: dimension mismatch");
  if (B == 0 || C == 0 || Tin == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, C, Tin * scale}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t to) {
        y(b, c, to) = x(b, c, to / scale);
      });
}

template <class ExecSpace, class Scalar>
void linear_upsample1d(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::linear_upsample1d") {
  const size_t B = x.extent(0), C = x.extent(1), Tin = x.extent(2);
  const size_t Tout = y.extent(2);
  if (y.extent(0) != B || y.extent(1) != C)
    throw std::invalid_argument("linear_upsample1d: dimension mismatch");
  if (B == 0 || C == 0 || Tin == 0 || Tout == 0) return;
  if (Tout == 1) {
    Kokkos::parallel_for(
        label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, C}),
        KOKKOS_LAMBDA(size_t b, size_t c) { y(b, c, 0) = x(b, c, 0); });
    return;
  }
  if (Tin == 1) {
    Kokkos::parallel_for(
        label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, C, Tout}),
        KOKKOS_LAMBDA(size_t b, size_t c, size_t to) {
          y(b, c, to) = x(b, c, 0);
        });
    return;
  }
  const float scale = static_cast<float>(Tin - 1) / static_cast<float>(Tout - 1);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, C, Tout}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t to) {
        const float src = static_cast<float>(to) * scale;
        size_t i0 = static_cast<size_t>(floorf(src));
        if (i0 >= Tin - 1) i0 = Tin - 2;
        float w = src - static_cast<float>(i0);
        if (w < 0.0f) w = 0.0f;
        if (w > 1.0f) w = 1.0f;
        const float v0 = static_cast<float>(x(b, c, i0));
        const float v1 = static_cast<float>(x(b, c, i0 + 1));
        y(b, c, to) = static_cast<Scalar>(v0 * (1.0f - w) + v1 * w);
      });
}

}  // namespace KokkosDNN

#endif
