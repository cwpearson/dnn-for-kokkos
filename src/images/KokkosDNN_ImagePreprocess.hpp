// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_IMAGE_PREPROCESS_HPP
#define KOKKOSDNN_IMAGE_PREPROCESS_HPP

// Image preprocessing for ViT-style encoders (DINOv2/DINOv3 path).
//
// Expected input: host HWC float image with values in [0, 1] (as produced by
// the I/O agent's decode path), channels last.
//
// DINOv3 ViT-B/16 @224px convention implemented by preprocess_dinov3():
//   1. resize so the short side is 256 (size*8/7, integer-safe: 224*8/7=256),
//   2. center-crop (or zero-pad, for tiny images) to size x size,
//   3. per-channel ImageNet normalize (mean {0.485,0.456,0.406},
//      std {0.229,0.224,0.225}),
//   4. HWC -> CHW transpose (+ optional NCHW batching).
// At 224px / patch 16 this yields 14x14=196 patch tokens (+1 CLS +4 registers
// = 201 tokens), matching examples/dinov3/README.md.
//
// All host preprocessing runs through Kokkos::parallel_for with labels (host
// execution space) so behavior is identical under any enabled host backend.
// Device-returning helpers stage on the host, then Kokkos::deep_copy to the
// target ExecSpace, so they work with any ExecSpace including Serial and
// plain HostSpace (CPU CI friendly). No file I/O in this header.

#if __has_include("KokkosDNN_ImageTypes.hpp")
#include "KokkosDNN_ImageTypes.hpp"
#else
// Fallback so this header compiles standalone before the I/O agent lands
// KokkosDNN_ImageTypes.hpp. Guarded by __has_include, so once the canonical
// header exists these declarations are never reached (no ODR clash).
#include <Kokkos_Core.hpp>
#include <cstdint>
namespace KokkosDNN::Image {
using HostImageU8 = Kokkos::View<uint8_t***, Kokkos::LayoutRight, Kokkos::HostSpace>;
using HostImageF32 = Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>;
struct ImageInfo {
  int height = 0;
  int width = 0;
  int channels = 0;
};
}  // namespace KokkosDNN::Image
#endif

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace KokkosDNN::Image {

// Device-side image aliases: CHW [C,H,W] single image, NCHW [N,C,H,W] batch.
template <class ExecSpace>
using DeviceImageCHW = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;
template <class ExecSpace>
using DeviceBatchNCHW = Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace>;

namespace Impl {

// ImageNet mean/std (RGB). Accessors avoid exposing raw globals.
KOKKOS_INLINE_FUNCTION
constexpr float imagenet_mean(int c) {
  return c == 0 ? 0.485f : (c == 1 ? 0.456f : 0.406f);
}
KOKKOS_INLINE_FUNCTION
constexpr float imagenet_std(int c) {
  return c == 0 ? 0.229f : (c == 1 ? 0.224f : 0.225f);
}

inline void validate_hwc(const HostImageF32& img, const char* what) {
  if (img.extent(0) == 0 || img.extent(1) == 0 || img.extent(2) == 0) {
    throw std::invalid_argument(std::string(what) + ": empty image dims [" +
                                std::to_string(img.extent(0)) + "," +
                                std::to_string(img.extent(1)) + "," +
                                std::to_string(img.extent(2)) + "]");
  }
}

inline void validate_3ch(const HostImageF32& img, const char* what) {
  validate_hwc(img, what);
  if (img.extent(2) != 3) {
    throw std::invalid_argument(std::string(what) + ": expected 3 channels, got " +
                                std::to_string(img.extent(2)));
  }
}

KOKKOS_INLINE_FUNCTION
float clamp_to(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Bilinear sample of channel c at fractional (sy, sx), align-corners=false:
// out-of-range coordinates are clamped (border replicate), matching the
// resize kernel below.
KOKKOS_INLINE_FUNCTION
float bilinear_sample(const HostImageF32& in, float sy, float sx, int c) {
  const float h = static_cast<float>(in.extent(0));
  const float w = static_cast<float>(in.extent(1));
  const float y = clamp_to(sy, 0.0f, h - 1.0f);
  const float x = clamp_to(sx, 0.0f, w - 1.0f);
  const size_t y0 = static_cast<size_t>(y);
  const size_t x0 = static_cast<size_t>(x);
  const size_t y1 = y0 + 1 < in.extent(0) ? y0 + 1 : y0;
  const size_t x1 = x0 + 1 < in.extent(1) ? x0 + 1 : x0;
  const float dy = y - static_cast<float>(y0);
  const float dx = x - static_cast<float>(x0);
  const float v00 = in(y0, x0, c);
  const float v01 = in(y0, x1, c);
  const float v10 = in(y1, x0, c);
  const float v11 = in(y1, x1, c);
  return (v00 * (1.0f - dx) + v01 * dx) * (1.0f - dy) +
         (v10 * (1.0f - dx) + v11 * dx) * dy;
}

}  // namespace Impl

// Bilinear resize, PyTorch/HF `interpolate(align_corners=false)` convention:
//   scale = in / out,  src = (dst + 0.5) * scale - 0.5  (per axis, clamped).
// Host implementation. Throws std::invalid_argument on empty input or
// non-positive out_h/out_w.
inline HostImageF32 resize_bilinear(const HostImageF32& in, int out_h, int out_w) {
  Impl::validate_hwc(in, "resize_bilinear");
  if (out_h <= 0 || out_w <= 0) {
    throw std::invalid_argument("resize_bilinear: out dims must be positive, got [" +
                                std::to_string(out_h) + "," + std::to_string(out_w) + "]");
  }
  const size_t in_h = in.extent(0), in_w = in.extent(1), ch = in.extent(2);
  const auto oh = static_cast<size_t>(out_h), ow = static_cast<size_t>(out_w);
  const float scale_h = static_cast<float>(in_h) / static_cast<float>(oh);
  const float scale_w = static_cast<float>(in_w) / static_cast<float>(ow);
  HostImageF32 out("KokkosDNN::Image::resize_bilinear", oh, ow, ch);
  Kokkos::parallel_for(
      "KokkosDNN::Image::resize_bilinear",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {oh, ow, ch}),
      KOKKOS_LAMBDA(size_t y, size_t x, size_t c) {
        const float sy = (static_cast<float>(y) + 0.5f) * scale_h - 0.5f;
        const float sx = (static_cast<float>(x) + 0.5f) * scale_w - 0.5f;
        out(y, x, c) = Impl::bilinear_sample(in, sy, sx, static_cast<int>(c));
      });
  Kokkos::fence();
  return out;
}

// Resize preserving aspect ratio so min(H, W) == short_side.
// If max_long_side > 0 the long side is additionally capped at max_long_side
// (scale = min(short_side/short, max_long_side/long), computed once from the
// original dims). DINOv2/v3 typical use: short side ~256. Throws on empty
// input or non-positive short_side.
inline HostImageF32 resize_short_side(const HostImageF32& in, int short_side,
                                      int max_long_side = 0) {
  Impl::validate_hwc(in, "resize_short_side");
  if (short_side <= 0) {
    throw std::invalid_argument("resize_short_side: short_side must be positive");
  }
  const size_t h = in.extent(0), w = in.extent(1);
  const size_t short_len = std::min(h, w), long_len = std::max(h, w);
  float scale = static_cast<float>(short_side) / static_cast<float>(short_len);
  if (max_long_side > 0) {
    scale = std::min(scale, static_cast<float>(max_long_side) / static_cast<float>(long_len));
  }
  const int new_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * scale)));
  const int new_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * scale)));
  if (new_h == static_cast<int>(h) && new_w == static_cast<int>(w)) {
    HostImageF32 copy("KokkosDNN::Image::resize_short_side_copy", h, w, in.extent(2));
    Kokkos::deep_copy(copy, in);
    return copy;
  }
  return resize_bilinear(in, new_h, new_w);
}

// Center crop that zero-pads (instead of throwing) when the input is smaller
// than the crop in either dimension. Centering uses truncation-toward-zero
// offsets (off = (in - crop) / 2); odd remainders bias one pixel toward
// bottom-right. Negative offsets pad top/left with zeros.
inline HostImageF32 center_crop_or_pad(const HostImageF32& in, int crop_h, int crop_w) {
  Impl::validate_hwc(in, "center_crop_or_pad");
  if (crop_h <= 0 || crop_w <= 0) {
    throw std::invalid_argument("center_crop_or_pad: crop dims must be positive");
  }
  const auto ih = static_cast<int>(in.extent(0));
  const auto iw = static_cast<int>(in.extent(1));
  const size_t ch = in.extent(2);
  const auto oh = static_cast<size_t>(crop_h), ow = static_cast<size_t>(crop_w);
  const int y_off = (ih - crop_h) / 2;  // may be negative -> pad top
  const int x_off = (iw - crop_w) / 2;  // may be negative -> pad left
  HostImageF32 out("KokkosDNN::Image::center_crop_or_pad", oh, ow, ch);
  Kokkos::parallel_for(
      "KokkosDNN::Image::center_crop_or_pad",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {oh, ow, ch}),
      KOKKOS_LAMBDA(size_t y, size_t x, size_t c) {
        const int sy = static_cast<int>(y) + y_off;
        const int sx = static_cast<int>(x) + x_off;
        out(y, x, c) = (sy >= 0 && sy < ih && sx >= 0 && sx < iw)
                           ? in(static_cast<size_t>(sy), static_cast<size_t>(sx), c)
                           : 0.0f;
      });
  Kokkos::fence();
  return out;
}

// Exact center crop to [crop_h, crop_w]. Throws std::invalid_argument if the
// input is smaller than the crop in either dimension (use center_crop_or_pad
// for tiny test images).
inline HostImageF32 center_crop(const HostImageF32& in, int crop_h, int crop_w) {
  Impl::validate_hwc(in, "center_crop");
  if (crop_h <= 0 || crop_w <= 0) {
    throw std::invalid_argument("center_crop: crop dims must be positive");
  }
  if (static_cast<size_t>(crop_h) > in.extent(0) ||
      static_cast<size_t>(crop_w) > in.extent(1)) {
    throw std::invalid_argument("center_crop: input [" + std::to_string(in.extent(0)) + "," +
                                std::to_string(in.extent(1)) + "] smaller than crop [" +
                                std::to_string(crop_h) + "," + std::to_string(crop_w) +
                                "]; use center_crop_or_pad");
  }
  return center_crop_or_pad(in, crop_h, crop_w);
}

// In-place per-channel normalize: img[y,x,c] = (img[y,x,c] - mean[c]) / std[c].
// Views are handles, so pass-by-value still modifies the caller's data.
// Requires 3 channels; throws on empty input, channel mismatch, or zero std.
inline void normalize(HostImageF32 img, const float mean[3], const float std[3]) {
  Impl::validate_3ch(img, "normalize");
  const float m0 = mean[0], m1 = mean[1], m2 = mean[2];
  const float s0 = std[0], s1 = std[1], s2 = std[2];
  if (s0 == 0.0f || s1 == 0.0f || s2 == 0.0f) {
    throw std::invalid_argument("normalize: std must be nonzero");
  }
  const size_t h = img.extent(0), w = img.extent(1);
  Kokkos::parallel_for(
      "KokkosDNN::Image::normalize",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0, 0}, {h, w}),
      KOKKOS_LAMBDA(size_t y, size_t x) {
        img(y, x, 0) = (img(y, x, 0) - m0) / s0;
        img(y, x, 1) = (img(y, x, 1) - m1) / s1;
        img(y, x, 2) = (img(y, x, 2) - m2) / s2;
      });
  Kokkos::fence();
}

// Const-correct overload returning a normalized copy; input is untouched.
inline HostImageF32 normalized_copy(const HostImageF32& img, const float mean[3],
                                    const float std[3]) {
  Impl::validate_3ch(img, "normalized_copy");
  HostImageF32 out("KokkosDNN::Image::normalized_copy", img.extent(0), img.extent(1),
                   img.extent(2));
  Kokkos::deep_copy(out, img);
  normalize(out, mean, std);
  return out;
}

// ImageNet normalize (mean {0.485,0.456,0.406}, std {0.229,0.224,0.225}):
// in-place variant plus a const-correct copy variant.
inline void imagenet_normalize(HostImageF32 img) {
  const float mean[3] = {Impl::imagenet_mean(0), Impl::imagenet_mean(1),
                         Impl::imagenet_mean(2)};
  const float std[3] = {Impl::imagenet_std(0), Impl::imagenet_std(1), Impl::imagenet_std(2)};
  normalize(img, mean, std);
}

inline HostImageF32 imagenet_normalized_copy(const HostImageF32& img) {
  const float mean[3] = {Impl::imagenet_mean(0), Impl::imagenet_mean(1),
                         Impl::imagenet_mean(2)};
  const float std[3] = {Impl::imagenet_std(0), Impl::imagenet_std(1), Impl::imagenet_std(2)};
  return normalized_copy(img, mean, std);
}

// Transpose host HWC -> device CHW [C,H,W] (host staging + deep_copy, so any
// ExecSpace works, including Serial/HostSpace). Throws on empty input.
template <class ExecSpace>
DeviceImageCHW<ExecSpace> hwc_to_chw(const HostImageF32& hwc) {
  Impl::validate_hwc(hwc, "hwc_to_chw");
  const size_t h = hwc.extent(0), w = hwc.extent(1), c = hwc.extent(2);
  Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace> tmp(
      "KokkosDNN::Image::hwc_to_chw_tmp", c, h, w);
  Kokkos::parallel_for(
      "KokkosDNN::Image::hwc_to_chw",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {c, h, w}),
      KOKKOS_LAMBDA(size_t cc, size_t y, size_t x) { tmp(cc, y, x) = hwc(y, x, cc); });
  Kokkos::fence();
  DeviceImageCHW<ExecSpace> out("KokkosDNN::Image::hwc_to_chw", c, h, w);
  Kokkos::deep_copy(out, tmp);
  Kokkos::fence();
  return out;
}

// Stack host HWC images into a device NCHW [N,C,H,W] batch for ViT input.
// Validates uniform shapes; throws on empty list or any shape mismatch.
template <class ExecSpace>
DeviceBatchNCHW<ExecSpace> batch_chw(const std::vector<HostImageF32>& hwcs) {
  if (hwcs.empty()) {
    throw std::invalid_argument("batch_chw: empty image list");
  }
  Impl::validate_hwc(hwcs[0], "batch_chw");
  const size_t n = hwcs.size();
  const size_t h = hwcs[0].extent(0), w = hwcs[0].extent(1), c = hwcs[0].extent(2);
  for (size_t i = 1; i < n; ++i) {
    Impl::validate_hwc(hwcs[i], "batch_chw");
    if (hwcs[i].extent(0) != h || hwcs[i].extent(1) != w || hwcs[i].extent(2) != c) {
      throw std::invalid_argument(
          "batch_chw: shape mismatch at index " + std::to_string(i) + " [" +
          std::to_string(hwcs[i].extent(0)) + "," + std::to_string(hwcs[i].extent(1)) + "," +
          std::to_string(hwcs[i].extent(2)) + "] vs [" + std::to_string(h) + "," +
          std::to_string(w) + "," + std::to_string(c) + "]");
    }
  }
  Kokkos::View<float****, Kokkos::LayoutRight, Kokkos::HostSpace> tmp(
      "KokkosDNN::Image::batch_chw_tmp", n, c, h, w);
  // Copy per image (each source view needs its own capture).
  for (size_t i = 0; i < n; ++i) {
    const HostImageF32 src = hwcs[i];
    Kokkos::parallel_for(
        "KokkosDNN::Image::batch_chw",
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>(
            {0, 0, 0}, {c, h, w}),
        KOKKOS_LAMBDA(size_t cc, size_t y, size_t x) { tmp(i, cc, y, x) = src(y, x, cc); });
  }
  Kokkos::fence();
  DeviceBatchNCHW<ExecSpace> out("KokkosDNN::Image::batch_chw", n, c, h, w);
  Kokkos::deep_copy(out, tmp);
  Kokkos::fence();
  return out;
}

// Stack already-transposed device CHW [C,H,W] views into NCHW [N,C,H,W].
// Same validation/throw contract as the HWC overload.
template <class ExecSpace>
DeviceBatchNCHW<ExecSpace> batch_chw(
    const std::vector<DeviceImageCHW<ExecSpace>>& chws) {
  if (chws.empty()) {
    throw std::invalid_argument("batch_chw: empty image list");
  }
  const size_t n = chws.size();
  const size_t c = chws[0].extent(0), h = chws[0].extent(1), w = chws[0].extent(2);
  if (c == 0 || h == 0 || w == 0) {
    throw std::invalid_argument("batch_chw: empty CHW image");
  }
  for (size_t i = 1; i < n; ++i) {
    if (chws[i].extent(0) != c || chws[i].extent(1) != h || chws[i].extent(2) != w) {
      throw std::invalid_argument("batch_chw: CHW shape mismatch at index " +
                                  std::to_string(i));
    }
  }
  DeviceBatchNCHW<ExecSpace> out("KokkosDNN::Image::batch_chw", n, c, h, w);
  for (size_t i = 0; i < n; ++i) {
    auto dst = Kokkos::subview(out, i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(dst, chws[i]);
  }
  Kokkos::fence();
  return out;
}

// Single-call DINOv3 preprocessing -> device CHW [C,size,size]:
//   resize_short_side(img, size*8/7)  [256 when size=224; integer-safe since
//     8/7 is applied as (size*8)/7]
//   -> center_crop(size,size), with zero-pad fallback for tiny images
//   -> ImageNet normalize -> hwc_to_chw<ExecSpace>().
// If do_center_crop=false, resize directly to size x size instead (square
// stretch; non-standard, kept for ablations). Throws on empty input or
// non-positive size.
template <class ExecSpace>
DeviceImageCHW<ExecSpace> preprocess_dinov3(const HostImageF32& img, int size = 224,
                                           bool do_center_crop = true) {
  Impl::validate_hwc(img, "preprocess_dinov3");
  if (size <= 0) {
    throw std::invalid_argument("preprocess_dinov3: size must be positive");
  }
  HostImageF32 work =
      do_center_crop
          ? center_crop_or_pad(resize_short_side(img, (size * 8) / 7), size, size)
          : resize_bilinear(img, size, size);
  imagenet_normalize(work);
  return hwc_to_chw<ExecSpace>(work);
}

// Batched DINOv3 preprocessing -> device NCHW [N,C,size,size]. Each image goes
// through the single-image pipeline on the host; the normalized HWC results
// are stacked with one batch_chw() call. Throws on empty list.
template <class ExecSpace>
DeviceBatchNCHW<ExecSpace> preprocess_dinov3(const std::vector<HostImageF32>& imgs,
                                            int size = 224, bool do_center_crop = true) {
  if (imgs.empty()) {
    throw std::invalid_argument("preprocess_dinov3: empty image list");
  }
  if (size <= 0) {
    throw std::invalid_argument("preprocess_dinov3: size must be positive");
  }
  std::vector<HostImageF32> staged;
  staged.reserve(imgs.size());
  for (const auto& img : imgs) {
    Impl::validate_hwc(img, "preprocess_dinov3");
    HostImageF32 work = do_center_crop ? center_crop_or_pad(
                                             resize_short_side(img, (size * 8) / 7), size, size)
                                       : resize_bilinear(img, size, size);
    imagenet_normalize(work);
    staged.push_back(work);
  }
  return batch_chw<ExecSpace>(staged);
}

}  // namespace KokkosDNN::Image

#endif  // KOKKOSDNN_IMAGE_PREPROCESS_HPP
