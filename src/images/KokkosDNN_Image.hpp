// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_IMAGE_HPP
#define KOKKOSDNN_IMAGE_HPP

// Umbrella header for the optional KokkosDNN image component
// (namespace KokkosDNN::Image, configure with -DKokkosDNN_ENABLE_IMAGE=ON,
// OFF by default; stb-backed I/O, no dependency from the core target).
//
// DINOv3 preprocessing entry points (see KokkosDNN_ImagePreprocess.hpp):
//   preprocess_dinov3<ExecSpace>(hwc_f32_01)        -> device CHW [3,size,size]
//   preprocess_dinov3<ExecSpace>(vector<hwc_f32>)   -> device NCHW [N,3,size,size]
// I/O entry points (see KokkosDNN_ImageIO.hpp):
//   load_image_u8 / load_image_f32 / decode_image_u8 / save_image_png / save_image_jpg
//   (+ WebP decode in load/decode_image_u8 and save_image_webp when built with
//   libwebp support; see Impl::webp_supported()).

#include "KokkosDNN_ImageTypes.hpp"
#include "KokkosDNN_ImageIO.hpp"
#include "KokkosDNN_ImagePreprocess.hpp"

#endif  // KOKKOSDNN_IMAGE_HPP
