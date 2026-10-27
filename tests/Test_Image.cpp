// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Image component tests (requires -DKokkosDNN_ENABLE_IMAGE=ON).
#include "KokkosDNN_config.h"
#ifndef KOKKOSDNN_ENABLE_IMAGE
#error "Test_Image requires KokkosDNN_ENABLE_IMAGE=ON"
#endif

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "images/KokkosDNN_Image.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using KokkosDNN::Image::HostImageF32;
using KokkosDNN::Image::HostImageU8;

std::string tmp_png(const std::string& name) {
  return (std::filesystem::temp_directory_path() / ("kokkosdnn_" + name)).string();
}

HostImageU8 make_pattern_u8(size_t h, size_t w, size_t c) {
  HostImageU8 img("pattern", h, w, c);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      for (size_t cc = 0; cc < c; ++cc)
        img(y, x, cc) = static_cast<uint8_t>((y * w + x + cc * 17) % 256);
  return img;
}

}  // namespace

TEST(ImageIO, PngRoundTripU8) {
  namespace Img = KokkosDNN::Image;
  const std::string path = tmp_png("roundtrip.png");
  HostImageU8 src = make_pattern_u8(4, 5, 3);
  ASSERT_NO_THROW(Img::save_image_png(path, src));

  Img::ImageInfo info = Img::load_image_info(path);
  EXPECT_EQ(info.width, 5);
  EXPECT_EQ(info.height, 4);
  EXPECT_EQ(info.channels, 3);

  HostImageU8 got = Img::load_image_u8(path, 3);
  EXPECT_EQ(got.extent(0), 4u);
  EXPECT_EQ(got.extent(1), 5u);
  EXPECT_EQ(got.extent(2), 3u);
  for (size_t y = 0; y < 4; ++y)
    for (size_t x = 0; x < 5; ++x)
      for (size_t c = 0; c < 3; ++c) EXPECT_EQ(got(y, x, c), src(y, x, c));

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(ImageIO, JpgRoundTripDims) {
  namespace Img = KokkosDNN::Image;
  const std::string path = tmp_png("roundtrip.jpg");
  // MCU-aligned constant image: JPEG is lossy, so a flat field keeps the
  // roundtrip error tiny while still exercising the encode/decode path.
  HostImageU8 src("jpg_src", 16, 16, 3);
  for (size_t y = 0; y < 16; ++y)
    for (size_t x = 0; x < 16; ++x)
      for (size_t c = 0; c < 3; ++c) src(y, x, c) = 128;
  ASSERT_NO_THROW(Img::save_image_jpg(path, src, 100));

  Img::ImageInfo info = Img::load_image_info(path);
  EXPECT_EQ(info.width, 16);
  EXPECT_EQ(info.height, 16);
  EXPECT_EQ(info.channels, 3);

  HostImageU8 got = Img::load_image_u8(path, 3);
  EXPECT_EQ(got.extent(0), 16u);
  EXPECT_EQ(got.extent(1), 16u);
  EXPECT_EQ(got.extent(2), 3u);
  int max_abs_err = 0;
  for (size_t y = 0; y < 16; ++y)
    for (size_t x = 0; x < 16; ++x)
      for (size_t c = 0; c < 3; ++c)
        max_abs_err = std::max(max_abs_err, std::abs(static_cast<int>(got(y, x, c)) -
                                                     static_cast<int>(src(y, x, c))));
  EXPECT_LE(max_abs_err, 8);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(ImageIO, JpgGrayAndQualityValidation) {
  namespace Img = KokkosDNN::Image;
  HostImageU8 gray("jpg_gray", 8, 8, 1);
  for (size_t y = 0; y < 8; ++y)
    for (size_t x = 0; x < 8; ++x) gray(y, x, 0) = 200;
  const std::string path = tmp_png("gray.jpg");
  ASSERT_NO_THROW(Img::save_image_jpg(path, gray));
  HostImageU8 got = Img::load_image_u8(path, 1);
  EXPECT_EQ(got.extent(0), 8u);
  EXPECT_EQ(got.extent(1), 8u);
  EXPECT_EQ(got.extent(2), 1u);
  std::error_code ec;
  std::filesystem::remove(path, ec);

  EXPECT_THROW(Img::save_image_jpg(path, gray, 0), std::invalid_argument);
  EXPECT_THROW(Img::save_image_jpg(path, gray, 101), std::invalid_argument);
  HostImageU8 empty("jpg_empty", 0, 0, 0);
  EXPECT_THROW(Img::save_image_jpg(path, empty), std::invalid_argument);
}

TEST(ImageIO, F32Scaling) {
  namespace Img = KokkosDNN::Image;
  const std::string path = tmp_png("scaling.png");
  HostImageU8 src("src", 1, 3, 1);
  src(0, 0, 0) = 0;
  src(0, 1, 0) = 128;
  src(0, 2, 0) = 255;
  Img::save_image_png(path, src);
  HostImageF32 got = Img::load_image_f32(path, 0);  // native channels
  ASSERT_EQ(got.extent(2), 1u);
  EXPECT_NEAR(got(0, 0, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(got(0, 1, 0), 128.0f / 255.0f, 1e-6f);
  EXPECT_NEAR(got(0, 2, 0), 1.0f, 1e-6f);
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(ImagePreprocess, ResizeIdentityAndConstant1x1) {
  namespace Img = KokkosDNN::Image;
  HostImageF32 in("in", 3, 4, 2);
  for (size_t y = 0; y < 3; ++y)
    for (size_t x = 0; x < 4; ++x)
      for (size_t c = 0; c < 2; ++c) in(y, x, c) = 0.1f * y + 0.01f * x + c;
  HostImageF32 same = Img::resize_bilinear(in, 3, 4);
  for (size_t y = 0; y < 3; ++y)
    for (size_t x = 0; x < 4; ++x)
      for (size_t c = 0; c < 2; ++c) EXPECT_NEAR(same(y, x, c), in(y, x, c), 1e-6f);

  HostImageF32 one("one", 1, 1, 3);
  one(0, 0, 0) = 0.25f;
  one(0, 0, 1) = 0.5f;
  one(0, 0, 2) = 0.75f;
  HostImageF32 big = Img::resize_bilinear(one, 3, 3);
  for (size_t y = 0; y < 3; ++y)
    for (size_t x = 0; x < 3; ++x)
      for (size_t c = 0; c < 3; ++c) EXPECT_NEAR(big(y, x, c), one(0, 0, c), 1e-6f);
}

TEST(ImagePreprocess, CenterCropOffsets) {
  namespace Img = KokkosDNN::Image;
  HostImageF32 in("in", 4, 6, 1);
  for (size_t y = 0; y < 4; ++y)
    for (size_t x = 0; x < 6; ++x) in(y, x, 0) = static_cast<float>(y * 10 + x);
  // y_off = (4-2)/2 = 1, x_off = (6-2)/2 = 2.
  HostImageF32 out = Img::center_crop(in, 2, 2);
  EXPECT_NEAR(out(0, 0, 0), 12.0f, 1e-6f);
  EXPECT_NEAR(out(0, 1, 0), 13.0f, 1e-6f);
  EXPECT_NEAR(out(1, 0, 0), 22.0f, 1e-6f);
  EXPECT_NEAR(out(1, 1, 0), 23.0f, 1e-6f);
  EXPECT_THROW(Img::center_crop(in, 5, 5), std::invalid_argument);
  // Pad path accepts tiny inputs.
  HostImageF32 padded = Img::center_crop_or_pad(in, 6, 8);
  EXPECT_EQ(padded.extent(0), 6u);
  EXPECT_EQ(padded.extent(1), 8u);
}

TEST(ImagePreprocess, NormalizeAndImageNet) {
  namespace Img = KokkosDNN::Image;
  HostImageF32 in("in", 1, 2, 3);
  in(0, 0, 0) = 1.0f;
  in(0, 0, 1) = 2.0f;
  in(0, 0, 2) = 3.0f;
  in(0, 1, 0) = 4.0f;
  in(0, 1, 1) = 5.0f;
  in(0, 1, 2) = 6.0f;
  const float mean[3] = {1.0f, 2.0f, 3.0f};
  const float std[3] = {1.0f, 2.0f, 3.0f};
  HostImageF32 normed = Img::normalized_copy(in, mean, std);
  EXPECT_NEAR(normed(0, 0, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(normed(0, 0, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(normed(0, 0, 2), 0.0f, 1e-6f);
  EXPECT_NEAR(normed(0, 1, 0), 3.0f, 1e-6f);
  EXPECT_NEAR(normed(0, 1, 1), 1.5f, 1e-6f);
  EXPECT_NEAR(normed(0, 1, 2), 1.0f, 1e-6f);
  // Input untouched by the copy variant.
  EXPECT_NEAR(in(0, 1, 2), 6.0f, 1e-6f);

  // ImageNet: a pixel equal to the mean normalizes to ~0.
  HostImageF32 px("px", 1, 1, 3);
  px(0, 0, 0) = 0.485f;
  px(0, 0, 1) = 0.456f;
  px(0, 0, 2) = 0.406f;
  HostImageF32 nm = Img::imagenet_normalized_copy(px);
  EXPECT_NEAR(nm(0, 0, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(nm(0, 0, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(nm(0, 0, 2), 0.0f, 1e-6f);
}

TEST(ImagePreprocess, HwcToChwTranspose) {
  namespace Img = KokkosDNN::Image;
  HostImageF32 hwc("hwc", 2, 3, 2);
  for (size_t y = 0; y < 2; ++y)
    for (size_t x = 0; x < 3; ++x)
      for (size_t c = 0; c < 2; ++c) hwc(y, x, c) = static_cast<float>(c * 100 + y * 10 + x);
  auto chw = Img::hwc_to_chw<Exec>(hwc);
  ASSERT_EQ(chw.extent(0), 2u);
  ASSERT_EQ(chw.extent(1), 2u);
  ASSERT_EQ(chw.extent(2), 3u);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, chw);
  for (size_t c = 0; c < 2; ++c)
    for (size_t y = 0; y < 2; ++y)
      for (size_t x = 0; x < 3; ++x) EXPECT_NEAR(host(c, y, x), hwc(y, x, c), 1e-6f);
}

TEST(ImagePreprocess, BatchChwStackAndMismatch) {
  namespace Img = KokkosDNN::Image;
  HostImageF32 a("a", 2, 2, 1);
  HostImageF32 b("b", 2, 2, 1);
  for (size_t y = 0; y < 2; ++y)
    for (size_t x = 0; x < 2; ++x) {
      a(y, x, 0) = 1.0f;
      b(y, x, 0) = 2.0f;
    }
  auto batch = Img::batch_chw<Exec>({a, b});
  ASSERT_EQ(batch.extent(0), 2u);
  ASSERT_EQ(batch.extent(1), 1u);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, batch);
  EXPECT_NEAR(host(0, 0, 0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(host(1, 0, 1, 1), 2.0f, 1e-6f);

  HostImageF32 wrong("wrong", 3, 2, 1);
  EXPECT_THROW(Img::batch_chw<Exec>({a, wrong}), std::invalid_argument);
  EXPECT_THROW(Img::batch_chw<Exec>({}), std::invalid_argument);
}

TEST(ImagePreprocess, Dinov3SingleShape) {
  namespace Img = KokkosDNN::Image;
  // Small synthetic image: exercises resize-short-side + crop-or-pad path.
  HostImageF32 in("in", 8, 6, 3);
  for (size_t y = 0; y < 8; ++y)
    for (size_t x = 0; x < 6; ++x)
      for (size_t c = 0; c < 3; ++c)
        in(y, x, c) = static_cast<float>(y * 6 + x) / 48.0f + 0.01f * c;
  auto out = Img::preprocess_dinov3<Exec>(in);
  ASSERT_EQ(out.extent(0), 3u);
  ASSERT_EQ(out.extent(1), 224u);
  ASSERT_EQ(out.extent(2), 224u);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  for (size_t c = 0; c < 3; ++c)
    for (size_t s = 0; s < 224; s += 37)
      EXPECT_TRUE(std::isfinite(host(c, s, s))) << "non-finite at (" << c << "," << s << ")";
}
