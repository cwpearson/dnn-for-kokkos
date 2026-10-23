// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_IMAGE_TYPES_HPP
#define KOKKOSDNN_IMAGE_TYPES_HPP

#include <Kokkos_Core.hpp>

#include <cstdint>

namespace KokkosDNN::Image {

// Basic image metadata. `channels` is the native file channel count
// (e.g. 1 gray, 3 RGB, 4 RGBA) when returned by load_image_info.
struct ImageInfo {
  int width = 0;
  int height = 0;
  int channels = 0;
};

// Host-space HWC views. LayoutRight keeps the channel dim contiguous
// ([H,W,C], C stride 1) for simple memcpy-style packing.
using HostImageU8 = Kokkos::View<uint8_t***, Kokkos::LayoutRight, Kokkos::HostSpace>;  // [H,W,C] uint8
using HostImageF32 = Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>;    // [H,W,C] float 0..1

namespace Impl {
// Non-public helpers live here.
}  // namespace Impl

}  // namespace KokkosDNN::Image

#endif
