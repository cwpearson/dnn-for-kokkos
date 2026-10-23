# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Optional KokkosDNN image component (KokkosDNN_ENABLE_IMAGE=ON only).
#
# Component headers (all present):
#   KokkosDNN_ImageTypes.hpp      - ImageInfo + host HWC view aliases
#   KokkosDNN_ImageIO.hpp         - stb-based JPEG/PNG/BMP decode/encode to Kokkos Views
#   KokkosDNN_ImagePreprocess.hpp - resize/crop/normalize, HWC->CHW, batching (DINOv3 path)
#   KokkosDNN_Image.hpp           - umbrella header including the above
#
# Header install is covered by the top-level
#   install(DIRECTORY src/ ... FILES_MATCHING PATTERN "*.hpp")
# rule, which will place these under ${CMAKE_INSTALL_INCLUDEDIR}/images.

# ---------------------------------------------------------------------------
# Dependency policy: prefer local compatible dependencies, fall back to
# FetchContent on a miss. Probe common system image libraries (QUIET, never
# fatal); the canonical image-I/O path is stb (header-only, JPEG/PNG/BMP).
# No fallback image-decoder implementations in our own code — third-party
# only for decoding.
# ---------------------------------------------------------------------------
find_package(JPEG QUIET)
find_package(PNG QUIET)
find_package(OpenCV QUIET)
if(JPEG_FOUND OR PNG_FOUND OR OpenCV_FOUND)
  message(STATUS "KokkosDNN image: system libs found:"
    " JPEG=${JPEG_FOUND} PNG=${PNG_FOUND} OpenCV=${OpenCV_FOUND}"
    " (informational only; canonical I/O path is stb)")
endif()

# Canonical path: stb (header-only). Look for a local stb_image.h first,
# including the default Homebrew (/opt/homebrew) and MacPorts (/opt/local)
# prefixes on macOS, which find_path does not search by default.
find_path(KokkosDNN_STB_INCLUDE_DIR stb_image.h
  HINTS $ENV{STB_ROOT} ${STB_ROOT} /usr/include /usr/local/include
    /opt/homebrew/include /opt/local/include
  PATH_SUFFIXES stb)
if(KokkosDNN_STB_INCLUDE_DIR)
  message(STATUS "KokkosDNN image: using local stb_image.h in ${KokkosDNN_STB_INCLUDE_DIR}")
else()
  message(STATUS "KokkosDNN image: no local stb_image.h; fetching stb via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    # Pinned master commit (2026-08-02). Override offline with
    # -DFETCHCONTENT_SOURCE_DIR_STB=<path to a local stb checkout>.
    GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
  )
  FetchContent_MakeAvailable(stb)
  set(KokkosDNN_STB_INCLUDE_DIR ${stb_SOURCE_DIR})
endif()
# Expose as an installable INTERFACE target (KokkosDNN::stb) carrying the
# stb headers. The include dir is exposed as SYSTEM (-isystem) so
# third-party stb warnings are silenced portably on GCC/Clang without
# per-warning pragmas (a blanket `#pragma ... ignored "-w"` is rejected by
# both compilers).
# The core umbrella (KokkosDNN.hpp) includes this component when
# KOKKOSDNN_ENABLE_IMAGE is defined, and KokkosDNN_ImageIO.hpp needs
# stb_image.h at compile time, so the stb dir is also exposed through the
# core target (build tree only) for existing core consumers. This is not a
# dependency on kokkosdnn_image itself: core never links it.
target_include_directories(kokkosdnn SYSTEM PUBLIC
  $<BUILD_INTERFACE:${KokkosDNN_STB_INCLUDE_DIR}>)
# Installable stb header target: part of the KokkosDNNTargets export set so
# installed KokkosDNN::image can link it. Headers are installed under
# <prefix>/include/kokkosdnn_stb so installed KokkosDNN_ImageIO.hpp (which
# does `#include "stb_image.h"`) keeps resolving for consumers.
add_library(kokkosdnn_stb INTERFACE)
add_library(KokkosDNN::stb ALIAS kokkosdnn_stb)
# NOTE: INTERFACE_SYSTEM_INCLUDE_DIRECTORIES alone adds nothing: it only
# marks entries of INTERFACE_INCLUDE_DIRECTORIES as SYSTEM. Both must list
# the dir for it to propagate as -isystem (setting only the SYSTEM property
# silently drops the include path and breaks the build).
set_target_properties(kokkosdnn_stb PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${KokkosDNN_STB_INCLUDE_DIR}>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kokkosdnn_stb>"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${KokkosDNN_STB_INCLUDE_DIR}>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kokkosdnn_stb>"
  EXPORT_NAME stb)
kokkosdnn_enable_compiler_warnings(kokkosdnn_stb)
install(TARGETS kokkosdnn_stb EXPORT KokkosDNNTargets)
foreach(_stb_header stb_image.h stb_image_write.h)
  if(NOT EXISTS "${KokkosDNN_STB_INCLUDE_DIR}/${_stb_header}")
    message(FATAL_ERROR
      "KokkosDNN image: ${_stb_header} not found in ${KokkosDNN_STB_INCLUDE_DIR}")
  endif()
endforeach()
install(FILES
  "${KokkosDNN_STB_INCLUDE_DIR}/stb_image.h"
  "${KokkosDNN_STB_INCLUDE_DIR}/stb_image_write.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kokkosdnn_stb)

# ---------------------------------------------------------------------------
# Library target (stb implementation TU + header-inline I/O/preprocess).
# ---------------------------------------------------------------------------
add_library(kokkosdnn_image
  KokkosDNN_Image.cpp
)
add_library(KokkosDNN::image ALIAS kokkosdnn_image)

target_include_directories(kokkosdnn_image PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(kokkosdnn_image PUBLIC Kokkos::kokkos KokkosDNN::stb)

# stb warnings are silenced via the SYSTEM (-isystem) include above, so this
# target keeps normal warning flags: only third-party stb code is muted,
# our own code (here and header-inline in consumers) is still checked.
set_target_properties(kokkosdnn_image PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON EXPORT_NAME image)
kokkosdnn_enable_compiler_warnings(kokkosdnn_image)

install(TARGETS kokkosdnn_image EXPORT KokkosDNNTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
