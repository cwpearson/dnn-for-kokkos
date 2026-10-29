// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_IMAGE_IO_HPP
#define KOKKOSDNN_IMAGE_IO_HPP

// stb-based image I/O to Kokkos host views (JPEG/PNG/BMP/GIF/etc. for
// decode; PNG/JPEG for encode) plus WebP decode/encode via libwebp.
// All returned views are owning, deep copies in
// host space with HWC layout [H,W,C]. Decoding itself is third-party only
// (stb, libwebp); no hand-rolled decoders here. The stb implementation lives
// in the single translation unit KokkosDNN_Image.cpp, as do the WebP
// helpers (declared below, defined there) so no libwebp headers leak into
// this header or its consumers.

#include "KokkosDNN_ImageTypes.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// stb declarations only (implementation is in KokkosDNN_Image.cpp).
// stb warning suppression is handled at the build-system level: the stb
// include dir is exposed as SYSTEM (-isystem) via KokkosDNN::stb (see
// src/images/CMakeLists.txt), which silences third-party warnings
// portably on GCC/Clang without enumerating warning names here.
#include "stb_image.h"
#include "stb_image_write.h"

namespace KokkosDNN::Image {

namespace Impl {

// stb accepts 0 (keep native) or 1..4 (force). Anything else is ours to reject.
inline void validate_desired_channels(int desired_channels) {
  if (desired_channels < 0 || desired_channels > 4) {
    throw std::invalid_argument(
        "KokkosDNN::Image: desired_channels must be 0 (native) or 1..4");
  }
}

// WebP helpers, defined (non-inline) in KokkosDNN_Image.cpp so libwebp
// stays out of this header. When the build lacks libwebp,
// webp_supported() returns false and the rest throw std::runtime_error.
// Native WebP channels are 3 (no alpha) or 4 (alpha), matching the file.
bool webp_supported();
// Case-insensitive ".webp" extension check (routing hint only; the memory
// probe below is authoritative for buffers).
bool has_webp_extension(const std::string& path);
// Cheap non-throwing probe: true iff data holds a WebP bitstream.
bool is_webp_memory(const unsigned char* data, size_t len);
// Header-magic check that needs no libwebp: true iff the buffer starts
// with RIFF....WEBP. Used only for diagnostics when WebP support is not
// built in (decode_image_u8 has no filename to hint from).
inline bool looks_like_webp(const unsigned char* data, size_t len) {
  if (data == nullptr || len < 12) {
    return false;
  }
  return data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
         data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P';
}
// First up-to-16 bytes as lowercase hex (no libwebp/stb needed). Used only
// to identify unsupported uploads (e.g. HEIC/AVIF) in decode errors.
inline std::string format_magic_hex(const unsigned char* data, size_t len) {
  static const char* const kHex = "0123456789abcdef";
  if (data == nullptr || len == 0) {
    return "(empty)";
  }
  const size_t n = (len < 16) ? len : 16;
  std::string out;
  out.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    out.push_back(kHex[(data[i] >> 4) & 0x0F]);
    out.push_back(kHex[data[i] & 0x0F]);
  }
  return out;
}
ImageInfo webp_info_from_file(const std::string& path);
HostImageU8 webp_decode_file(const std::string& path, int desired_channels);
HostImageU8 webp_decode_memory(const unsigned char* data, size_t len,
                               int desired_channels);

}  // namespace Impl

// Probe dimensions without decoding pixels. Throws std::runtime_error if the
// file is missing or not a recognized image (WebP included when built with
// libwebp support; see Impl::webp_supported()).
inline ImageInfo load_image_info(const std::string& path) {
  int w = 0, h = 0, comp = 0;
  if (stbi_info(path.c_str(), &w, &h, &comp) != 0) {
    ImageInfo info;
    info.width = w;
    info.height = h;
    info.channels = comp;
    return info;
  }
  std::string stb_why = stbi_failure_reason() ? stbi_failure_reason() : "unknown error";
  if (Impl::webp_supported()) {
    try {
      return Impl::webp_info_from_file(path);
    } catch (const std::exception& e) {
      throw std::runtime_error("KokkosDNN::Image::load_image_info: cannot read '" + path +
                               "': stb says '" + stb_why + "'; webp says '" + e.what() +
                               "'");
    }
  }
  throw std::runtime_error("KokkosDNN::Image::load_image_info: cannot read '" + path +
                           "': " + stb_why +
                           " (WebP support not built in; configure with "
                           "-DKokkosDNN_ENABLE_IMAGE_WEBP=ON)");
}

// Decode an image file to an owning host HWC uint8 view. If desired_channels
// != 0 the output is forced to that many channels, else native is kept.
// WebP files route to the libwebp decoder when built with WebP support
// (by extension, with a content fallback for misnamed files).
// Throws std::runtime_error on missing file / decode failure.
inline HostImageU8 load_image_u8(const std::string& path, int desired_channels = 3) {
  Impl::validate_desired_channels(desired_channels);
  if (Impl::webp_supported() && Impl::has_webp_extension(path)) {
    return Impl::webp_decode_file(path, desired_channels);
  }
  int w = 0, h = 0, native = 0;
  unsigned char* pix = stbi_load(path.c_str(), &w, &h, &native, desired_channels);
  if (pix != nullptr) {
    const int c = (desired_channels != 0) ? desired_channels : native;
    HostImageU8 img("KokkosDNN::Image::load_u8", static_cast<size_t>(h),
                    static_cast<size_t>(w), static_cast<size_t>(c));
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int cc = 0; cc < c; ++cc)
          img(static_cast<size_t>(y), static_cast<size_t>(x), static_cast<size_t>(cc)) =
              pix[(static_cast<size_t>(y) * static_cast<size_t>(w) +
                   static_cast<size_t>(x)) *
                      static_cast<size_t>(c) +
                  static_cast<size_t>(cc)];
    stbi_image_free(pix);
    return img;
  }
  std::string stb_why = stbi_failure_reason() ? stbi_failure_reason() : "unknown error";
  if (Impl::webp_supported()) {
    try {
      return Impl::webp_decode_file(path, desired_channels);
    } catch (const std::exception& e) {
      throw std::runtime_error("KokkosDNN::Image::load_image_u8: cannot load '" + path +
                               "': stb says '" + stb_why + "'; webp says '" + e.what() +
                               "'");
    }
  }
  throw std::runtime_error("KokkosDNN::Image::load_image_u8: cannot load '" + path +
                           "': " + stb_why +
                           (Impl::has_webp_extension(path)
                                ? " (looks like WebP but WebP support is not built in; "
                                  "configure with -DKokkosDNN_ENABLE_IMAGE_WEBP=ON)"
                                : ""));
}

// Decode to float HWC in 0..1 (u8 / 255.0f). Same channel rules as load_image_u8.
inline HostImageF32 load_image_f32(const std::string& path, int desired_channels = 3) {
  HostImageU8 u8 = load_image_u8(path, desired_channels);
  const size_t h = u8.extent(0), w = u8.extent(1), c = u8.extent(2);
  HostImageF32 img("KokkosDNN::Image::load_f32", h, w, c);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      for (size_t cc = 0; cc < c; ++cc)
        img(y, x, cc) = static_cast<float>(u8(y, x, cc)) * (1.0f / 255.0f);
  return img;
}

// In-memory decode (e.g. for tests without files). Same channel rules;
// WebP buffers route to the libwebp decoder when built with WebP support.
// throws std::invalid_argument on empty input, std::runtime_error on failure.
inline HostImageU8 decode_image_u8(const unsigned char* data, size_t len,
                                   int desired_channels = 3) {
  Impl::validate_desired_channels(desired_channels);
  if (data == nullptr || len == 0) {
    throw std::invalid_argument("KokkosDNN::Image::decode_image_u8: empty input buffer");
  }
  if (len > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    throw std::invalid_argument("KokkosDNN::Image::decode_image_u8: input too large");
  }
  if (Impl::webp_supported() && Impl::is_webp_memory(data, len)) {
    return Impl::webp_decode_memory(data, len, desired_channels);
  }
  int w = 0, h = 0, native = 0;
  unsigned char* pix =
      stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &native, desired_channels);
  if (pix == nullptr) {
    std::string why = stbi_failure_reason() ? stbi_failure_reason() : "unknown error";
    if (Impl::webp_supported()) {
      // Probe false negatives (or a build where is_webp_memory disagrees
      // with the decoder) should still get a WebP attempt before giving
      // up, mirroring load_image_u8's content fallback for misnamed files.
      try {
        return Impl::webp_decode_memory(data, len, desired_channels);
      } catch (const std::exception& e) {
        throw std::runtime_error(
            "KokkosDNN::Image::decode_image_u8: decode failed: stb says '" + why +
            "'; webp says '" + e.what() + "'");
      }
    }
    if (Impl::looks_like_webp(data, len)) {
      throw std::runtime_error(
          "KokkosDNN::Image::decode_image_u8: decode failed: stb says '" + why +
          "' (looks like WebP but WebP support is not built in; configure with "
          "-DKokkosDNN_ENABLE_IMAGE_WEBP=ON)");
    }
    throw std::runtime_error(
        "KokkosDNN::Image::decode_image_u8: decode failed: " + why + " (len=" +
        std::to_string(len) + ", magic=" + Impl::format_magic_hex(data, len) +
        "; supported: JPEG/PNG/BMP/GIF + WebP when built with "
        "-DKokkosDNN_ENABLE_IMAGE_WEBP=ON; HEIC/HEIF/AVIF/JXL are not "
        "supported - convert to PNG/JPEG first)");
  }
  const int c = (desired_channels != 0) ? desired_channels : native;
  HostImageU8 img("KokkosDNN::Image::decode", static_cast<size_t>(h),
                  static_cast<size_t>(w), static_cast<size_t>(c));
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int cc = 0; cc < c; ++cc)
        img(static_cast<size_t>(y), static_cast<size_t>(x), static_cast<size_t>(cc)) =
            pix[(static_cast<size_t>(y) * static_cast<size_t>(w) +
                 static_cast<size_t>(x)) *
                    static_cast<size_t>(c) +
                static_cast<size_t>(cc)];
  stbi_image_free(pix);
  return img;
}

// Encode a host HWC uint8 view as PNG. Channels must be 1..4. The view may be
// strided; pixels are packed to a contiguous row-major buffer first.
// Throws std::invalid_argument on bad extents, std::runtime_error on write failure.
inline void save_image_png(const std::string& path, const HostImageU8& img) {
  const size_t h = img.extent(0), w = img.extent(1), c = img.extent(2);
  if (w == 0 || h == 0 || c == 0) {
    throw std::invalid_argument("KokkosDNN::Image::save_image_png: empty image view");
  }
  if (c < 1 || c > 4) {
    throw std::invalid_argument("KokkosDNN::Image::save_image_png: channels must be 1..4");
  }
  if (w > static_cast<size_t>((std::numeric_limits<int>::max)()) ||
      h > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    throw std::invalid_argument("KokkosDNN::Image::save_image_png: image too large");
  }
  std::vector<unsigned char> buf(w * h * c);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      for (size_t cc = 0; cc < c; ++cc) buf[(y * w + x) * c + cc] = img(y, x, cc);
  const int stride = static_cast<int>(w * c);
  if (stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h),
                     static_cast<int>(c), buf.data(), stride) == 0) {
    throw std::runtime_error("KokkosDNN::Image::save_image_png: failed to write '" + path +
                             "'");
  }
}

// Encode a host HWC uint8 view as JPEG. Channels must be 1..4 (alpha, if
// present, is ignored by the encoder). The view may be strided; pixels are
// packed to a contiguous row-major buffer first. Quality is the stb JPEG
// quality 1..100 (higher looks better but yields a bigger file); the default
// of 90 matches common encoder practice for a good size/quality tradeoff.
// JPEG is lossy: even at quality 100 a decode of the written file may differ
// slightly from the input. Throws std::invalid_argument on bad extents or
// out-of-range quality, std::runtime_error on write failure.
inline void save_image_jpg(const std::string& path, const HostImageU8& img,
                           int quality = 90) {
  const size_t h = img.extent(0), w = img.extent(1), c = img.extent(2);
  if (w == 0 || h == 0 || c == 0) {
    throw std::invalid_argument("KokkosDNN::Image::save_image_jpg: empty image view");
  }
  if (c < 1 || c > 4) {
    throw std::invalid_argument("KokkosDNN::Image::save_image_jpg: channels must be 1..4");
  }
  if (w > static_cast<size_t>((std::numeric_limits<int>::max)()) ||
      h > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    throw std::invalid_argument("KokkosDNN::Image::save_image_jpg: image too large");
  }
  if (quality < 1 || quality > 100) {
    throw std::invalid_argument("KokkosDNN::Image::save_image_jpg: quality must be 1..100");
  }
  std::vector<unsigned char> buf(w * h * c);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      for (size_t cc = 0; cc < c; ++cc) buf[(y * w + x) * c + cc] = img(y, x, cc);
  if (stbi_write_jpg(path.c_str(), static_cast<int>(w), static_cast<int>(h),
                     static_cast<int>(c), buf.data(), quality) == 0) {
    throw std::runtime_error("KokkosDNN::Image::save_image_jpg: failed to write '" + path +
                             "'");
  }
}

// Encode a host HWC uint8 view as WebP (libwebp). Channels must be 1
// (gray, expanded to RGB), 3 (RGB), or 4 (RGBA); 2-channel views are
// rejected. The view may be strided; pixels are packed to a contiguous
// row-major buffer first. Quality is the lossy quality 0..100 (default 90,
// matching save_image_jpg); lossless=true selects lossless encoding and the
// quality value is ignored. WebP is lossy by default: a decode of the
// written file may differ slightly from the input (use lossless=true for an
// exact roundtrip). Throws std::invalid_argument on bad extents, channel
// count, or out-of-range quality, and std::runtime_error on encode/write
// failure — including when the build lacks libwebp support (see
// Impl::webp_supported()).
void save_image_webp(const std::string& path, const HostImageU8& img, int quality = 90,
                     bool lossless = false);

}  // namespace KokkosDNN::Image

#endif
