// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Single stb implementation translation unit for libkokkosdnn_image.a.
//
// Headers under src/images/ (KokkosDNN_ImageIO.hpp) include stb_image.h /
// stb_image_write.h for declarations only. The IMPLEMENTATION macros are
// defined exactly once, here. "stb_image.h" resolves via the KokkosDNN::stb
// interface include dir (see src/images/CMakeLists.txt): that dir is the
// folder actually containing stb_image.h, whether the headers live at the
// top level of the stb repo or under a stb/ subdir, so no path tweaks here.
//
// This TU also defines the WebP helpers declared in KokkosDNN_ImageIO.hpp
// (KokkosDNN::Image::Impl::webp_* plus save_image_webp) so no libwebp
// headers leak into our installed headers. With
// -DKokkosDNN_ENABLE_IMAGE_WEBP=OFF (or no libwebp found and no network
// for the FetchContent fallback) KOKKOSDNN_IMAGE_HAVE_WEBP stays undefined:
// webp_supported() is false and the rest throw std::runtime_error.

// The IMPLEMENTATION macros must be defined before KokkosDNN_ImageIO.hpp:
// that header includes stb_image.h / stb_image_write.h, and stb's include
// guards make any second inclusion a no-op, so the first (and only)
// inclusion — the one below — has to carry the implementation.

// stb warning suppression is handled at the build-system level: the stb
// include dir is exposed as SYSTEM (-isystem, see src/images/CMakeLists.txt),
// so only third-party stb code is muted; our own code in this TU keeps
// normal warning flags.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

// Pull in the Impl::webp_* declarations (and stb declarations) defined here.
#include "KokkosDNN_ImageIO.hpp"

#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP
#include "webp/decode.h"
#include "webp/encode.h"
#endif

#include <climits>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace KokkosDNN::Image::Impl {

bool webp_supported() {
#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP
  return true;
#else
  return false;
#endif
}

bool has_webp_extension(const std::string& path) {
  const std::string::size_type pos = path.find_last_of('.');
  if (pos == std::string::npos || pos + 1 >= path.size()) {
    return false;
  }
  std::string ext = path.substr(pos + 1);
  for (char& ch : ext) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return ext == "webp";
}

namespace {

#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP

struct WebPFreeDeleter {
  void operator()(uint8_t* p) const noexcept { WebPFree(p); }
};

using WebPBuffer = std::unique_ptr<uint8_t[], WebPFreeDeleter>;

std::vector<unsigned char> read_file_bytes(const std::string& path, const char* what) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(std::string(what) + ": cannot open '" + path + "'");
  }
  in.seekg(0, std::ios::end);
  const std::streampos end = in.tellg();
  if (end < 0 || static_cast<uintmax_t>(end) > static_cast<uintmax_t>(INT_MAX)) {
    throw std::runtime_error(std::string(what) + ": cannot read '" + path + "'");
  }
  in.seekg(0, std::ios::beg);
  std::vector<unsigned char> bytes(static_cast<size_t>(end));
  if (!bytes.empty()) {
    in.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (!in) {
      throw std::runtime_error(std::string(what) + ": cannot read '" + path + "'");
    }
  }
  return bytes;
}

// Integer BT.601 luma, exact for gray pixels (Y(v,v,v) == v).
inline uint8_t rgb_to_gray(uint8_t r, uint8_t g, uint8_t b) {
  return static_cast<uint8_t>((77 * static_cast<int>(r) + 150 * static_cast<int>(g) +
                               29 * static_cast<int>(b)) >>
                              8);
}

#endif  // KOKKOSDNN_IMAGE_HAVE_WEBP

[[maybe_unused]] const char* const kNoWebP =
    "WebP support not built in (configure with -DKokkosDNN_ENABLE_IMAGE_WEBP=ON)";

}  // namespace

bool is_webp_memory(const unsigned char* data, size_t len) {
#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP
  if (data == nullptr || len == 0 || len > static_cast<size_t>(INT_MAX)) {
    return false;
  }
  int w = 0, h = 0;
  return WebPGetInfo(data, len, &w, &h) != 0;
#else
  (void)data;
  (void)len;
  return false;
#endif
}

ImageInfo webp_info_from_file(const std::string& path) {
#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP
  const char* const what = "KokkosDNN::Image::load_image_info";
  const std::vector<unsigned char> bytes = read_file_bytes(path, what);
  if (bytes.empty()) {
    throw std::runtime_error(std::string(what) + ": cannot read '" + path + "': empty file");
  }
  WebPBitstreamFeatures features{};
  if (WebPGetFeatures(bytes.data(), bytes.size(), &features) != VP8_STATUS_OK) {
    throw std::runtime_error(std::string(what) + ": cannot read '" + path +
                             "': not a valid WebP bitstream");
  }
  ImageInfo info;
  info.width = features.width;
  info.height = features.height;
  info.channels = (features.has_alpha != 0) ? 4 : 3;
  return info;
#else
  (void)path;
  throw std::runtime_error(
      "KokkosDNN::Image::load_image_info: cannot read WebP file: " + std::string(kNoWebP));
#endif
}

HostImageU8 webp_decode_memory(const unsigned char* data, size_t len, int desired_channels) {
  validate_desired_channels(desired_channels);
#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP
  const char* const what = "KokkosDNN::Image::decode_image_u8";
  if (data == nullptr || len == 0) {
    throw std::invalid_argument(std::string(what) + ": empty input buffer");
  }
  if (len > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument(std::string(what) + ": input too large");
  }
  WebPBitstreamFeatures features{};
  if (WebPGetFeatures(data, len, &features) != VP8_STATUS_OK) {
    throw std::runtime_error(std::string(what) + ": decode failed: not a valid WebP bitstream");
  }
  const int w = features.width, h = features.height;
  if (w <= 0 || h <= 0) {
    throw std::runtime_error(std::string(what) + ": decode failed: bad WebP dimensions");
  }
  const int native = (features.has_alpha != 0) ? 4 : 3;
  const int c = (desired_channels != 0) ? desired_channels : native;
  HostImageU8 img("KokkosDNN::Image::webp_decode", static_cast<size_t>(h),
                  static_cast<size_t>(w), static_cast<size_t>(c));
  const auto uw = static_cast<size_t>(w), uh = static_cast<size_t>(h);
  if (c == 4 || c == 2) {
    int dw = 0, dh = 0;
    WebPBuffer pix(WebPDecodeRGBA(data, len, &dw, &dh));
    if (pix == nullptr || dw != w || dh != h) {
      throw std::runtime_error(std::string(what) + ": WebP RGBA decode failed");
    }
    for (size_t y = 0; y < uh; ++y) {
      for (size_t x = 0; x < uw; ++x) {
        const size_t k = (y * uw + x) * 4;
        if (c == 4) {
          img(y, x, 0) = pix[k];
          img(y, x, 1) = pix[k + 1];
          img(y, x, 2) = pix[k + 2];
          img(y, x, 3) = pix[k + 3];
        } else {
          img(y, x, 0) = rgb_to_gray(pix[k], pix[k + 1], pix[k + 2]);
          img(y, x, 1) = pix[k + 3];
        }
      }
    }
  } else {
    int dw = 0, dh = 0;
    WebPBuffer pix(WebPDecodeRGB(data, len, &dw, &dh));
    if (pix == nullptr || dw != w || dh != h) {
      throw std::runtime_error(std::string(what) + ": WebP RGB decode failed");
    }
    for (size_t y = 0; y < uh; ++y) {
      for (size_t x = 0; x < uw; ++x) {
        const size_t k = (y * uw + x) * 3;
        if (c == 3) {
          img(y, x, 0) = pix[k];
          img(y, x, 1) = pix[k + 1];
          img(y, x, 2) = pix[k + 2];
        } else {  // c == 1
          img(y, x, 0) = rgb_to_gray(pix[k], pix[k + 1], pix[k + 2]);
        }
      }
    }
  }
  return img;
#else
  (void)data;
  (void)len;
  throw std::runtime_error(
      "KokkosDNN::Image::decode_image_u8: cannot decode WebP data: " + std::string(kNoWebP));
#endif
}

HostImageU8 webp_decode_file(const std::string& path, int desired_channels) {
  validate_desired_channels(desired_channels);
#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP
  const char* const what = "KokkosDNN::Image::load_image_u8";
  const std::vector<unsigned char> bytes = read_file_bytes(path, what);
  try {
    return webp_decode_memory(bytes.data(), bytes.size(), desired_channels);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(what) + ": cannot load '" + path + "': " + e.what());
  }
#else
  (void)path;
  throw std::runtime_error(
      "KokkosDNN::Image::load_image_u8: cannot load WebP file: " + std::string(kNoWebP));
#endif
}

}  // namespace KokkosDNN::Image::Impl

namespace KokkosDNN::Image {

void save_image_webp(const std::string& path, const HostImageU8& img, int quality,
                     bool lossless) {
#ifdef KOKKOSDNN_IMAGE_HAVE_WEBP
  const char* const what = "KokkosDNN::Image::save_image_webp";
  const size_t h = img.extent(0), w = img.extent(1), c = img.extent(2);
  if (w == 0 || h == 0 || c == 0) {
    throw std::invalid_argument(std::string(what) + ": empty image view");
  }
  if (c != 1 && c != 3 && c != 4) {
    throw std::invalid_argument(
        std::string(what) + ": channels must be 1, 3, or 4 (libwebp has no gray+alpha "
                            "encoding)");
  }
  if (w > static_cast<size_t>(INT_MAX) || h > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument(std::string(what) + ": image too large");
  }
  if (quality < 0 || quality > 100) {
    throw std::invalid_argument(std::string(what) + ": quality must be 0..100");
  }
  const int W = static_cast<int>(w), H = static_cast<int>(h);
  // Gray expands to RGB (triplicated); RGB/RGBA pack as-is. The view may be
  // strided, so pack to a contiguous row-major buffer first.
  const int comps = (c == 1) ? 3 : static_cast<int>(c);
  std::vector<unsigned char> buf(w * h * static_cast<size_t>(comps));
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      const size_t k = (y * w + x) * static_cast<size_t>(comps);
      if (c == 1) {
        const unsigned char v = img(y, x, 0);
        buf[k] = v;
        buf[k + 1] = v;
        buf[k + 2] = v;
      } else {
        for (size_t cc = 0; cc < c; ++cc) buf[k + cc] = img(y, x, cc);
      }
    }
  }
  const int stride = W * comps;
  uint8_t* out = nullptr;
  size_t out_size = 0;
  if (lossless) {
    out_size = (comps == 4) ? WebPEncodeLosslessRGBA(buf.data(), W, H, stride, &out)
                            : WebPEncodeLosslessRGB(buf.data(), W, H, stride, &out);
  } else {
    const float q = static_cast<float>(quality);
    out_size = (comps == 4) ? WebPEncodeRGBA(buf.data(), W, H, stride, q, &out)
                            : WebPEncodeRGB(buf.data(), W, H, stride, q, &out);
  }
  if (out_size == 0 || out == nullptr) {
    throw std::runtime_error(std::string(what) + ": failed to encode '" + path + "'");
  }
  Impl::WebPBuffer encoded(out);
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os) {
    throw std::runtime_error(std::string(what) + ": failed to write '" + path + "'");
  }
  os.write(reinterpret_cast<const char*>(encoded.get()),
           static_cast<std::streamsize>(out_size));
  os.close();
  if (!os) {
    throw std::runtime_error(std::string(what) + ": failed to write '" + path + "'");
  }
#else
  (void)path;
  (void)img;
  (void)quality;
  (void)lossless;
  throw std::runtime_error(
      "KokkosDNN::Image::save_image_webp: " + std::string(Impl::kNoWebP));
#endif
}

}  // namespace KokkosDNN::Image
