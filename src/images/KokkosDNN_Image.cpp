// Single stb implementation translation unit for libkokkosdnn_image.a.
//
// Headers under src/images/ (KokkosDNN_ImageIO.hpp) include stb_image.h /
// stb_image_write.h for declarations only. The IMPLEMENTATION macros are
// defined exactly once, here. "stb_image.h" resolves via the stb::stb
// interface include dir (see src/images/CMakeLists.txt): that dir is the
// folder actually containing stb_image.h, whether the headers live at the
// top level of the stb repo or under a stb/ subdir, so no path tweaks here.

// stb warning suppression is handled at the build-system level: the stb
// include dir is exposed as SYSTEM (-isystem, see src/images/CMakeLists.txt),
// so only third-party stb code is muted; our own code in this TU keeps
// normal warning flags.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
