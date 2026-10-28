# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Optional KokkosDNN image component (KokkosDNN_ENABLE_IMAGE=ON only).
#
# Component headers (all present):
#   KokkosDNN_ImageTypes.hpp      - ImageInfo + host HWC view aliases
#   KokkosDNN_ImageIO.hpp         - stb-based JPEG/PNG/BMP decode, PNG/JPEG encode;
#                                   WebP decode/encode via libwebp (see below)
#   KokkosDNN_ImagePreprocess.hpp - resize/crop/normalize, HWC->CHW, batching (DINOv3 path)
#   KokkosDNN_Image.hpp           - umbrella header including the above
#
# Header install is covered by the top-level
#   install(DIRECTORY src/ ... FILES_MATCHING PATTERN "*.hpp")
# rule, which will place these under ${CMAKE_INSTALL_INCLUDEDIR}/images.
#
# WebP note: stb cannot decode/encode WebP, so WebP goes through libwebp.
# No libwebp headers leak into our installed headers: the WebP helpers are
# declared in KokkosDNN_ImageIO.hpp but defined in KokkosDNN_Image.cpp, and
# kokkosdnn_image links libwebp. Disable with
# -DKokkosDNN_ENABLE_IMAGE_WEBP=OFF (decode falls back to stb-only and
# save_image_webp throws std::runtime_error).

# ---------------------------------------------------------------------------
# Dependency policy: the canonical image-I/O path is stb (header-only,
# JPEG/PNG/BMP). Prefer a local stb_image.h, fall back to FetchContent on a
# miss. No fallback image-decoder implementations in our own code —
# third-party only for decoding.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Optional WebP support (decode + encode via libwebp).
# Preference order: system WebP package config (WebP::webp, re-found by
# consumers via find_dependency in KokkosDNNConfig.cmake.in), then a manual
# header/library lookup (absolute paths, exported with our targets, no
# consumer-side find needed), then FetchContent on a miss (slim: libraries
# only, no CLI tools/extras; installed into our export set). No fallback
# WebP implementations in our own code — third-party only.
# ---------------------------------------------------------------------------
set(KOKKOSDNN_IMAGE_HAVE_WEBP OFF)
set(KokkosDNN_IMAGE_USES_SYSTEM_WEBP OFF)
if(KokkosDNN_ENABLE_IMAGE_WEBP)
  find_package(WebP QUIET)
  if(TARGET WebP::webp)
    message(STATUS "KokkosDNN image: using system libwebp ${WebP_VERSION}")
    set(KOKKOSDNN_IMAGE_HAVE_WEBP ON)
    set(KokkosDNN_IMAGE_USES_SYSTEM_WEBP ON)
    target_link_libraries(kokkosdnn_image PUBLIC WebP::webp)
    kokkosdnn_suppress_third_party_warnings(WebP::webp)
  else()
    # WEBP_ROOT is intentionally read from the environment (and the
    # KokkosDNN_WEBP_ROOT cache variable) rather than a WEBP_ROOT CMake
    # variable, which CMake reserves for find_package search prefixes.
    find_path(KokkosDNN_WEBP_INCLUDE_DIR webp/encode.h
      HINTS $ENV{WEBP_ROOT} ${KokkosDNN_WEBP_ROOT} /usr /usr/local /opt/homebrew /opt/local
      PATH_SUFFIXES include)
    find_library(KokkosDNN_WEBP_LIBRARY webp
      HINTS $ENV{WEBP_ROOT} ${KokkosDNN_WEBP_ROOT} /usr /usr/local /opt/homebrew /opt/local
      PATH_SUFFIXES lib lib64)
    if(KokkosDNN_WEBP_INCLUDE_DIR AND KokkosDNN_WEBP_LIBRARY)
      message(STATUS "KokkosDNN image: using libwebp in ${KokkosDNN_WEBP_LIBRARY}")
      set(KOKKOSDNN_IMAGE_HAVE_WEBP ON)
      # Manual lookup: absolute paths baked into the targets file (same
      # approach as the CUDNN manual fallback in the top-level CMakeLists).
      # Includes are PRIVATE (our headers never mention webp/*.h); the
      # libraries are PUBLIC so consumers of the static kokkosdnn_image
      # resolve the WebP symbols at link time. No consumer-side find needed.
      target_include_directories(kokkosdnn_image SYSTEM PRIVATE
        ${KokkosDNN_WEBP_INCLUDE_DIR})
      target_link_libraries(kokkosdnn_image PUBLIC ${KokkosDNN_WEBP_LIBRARY})
      # libwebp's encoder also needs sharpyuv when statically linked; link
      # it too when present (shared libwebp already carries it).
      find_library(KokkosDNN_SHARPYUV_LIBRARY sharpyuv
        HINTS $ENV{WEBP_ROOT} ${KokkosDNN_WEBP_ROOT} /usr /usr/local /opt/homebrew /opt/local
        PATH_SUFFIXES lib lib64)
      if(KokkosDNN_SHARPYUV_LIBRARY)
        target_link_libraries(kokkosdnn_image PUBLIC ${KokkosDNN_SHARPYUV_LIBRARY})
      endif()
    else()
      message(STATUS "KokkosDNN image: no system libwebp; fetching libwebp via FetchContent")
      include(FetchContent)
      set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_LIBWEBPMUX OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
      set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
      FetchContent_Declare(
        libwebp
        GIT_REPOSITORY https://github.com/webmproject/libwebp.git
        # Pinned release tag v1.6.0. Override offline with
        # -DFETCHCONTENT_SOURCE_DIR_LIBWEBP=<path to a local libwebp checkout>.
        GIT_TAG v1.6.0
      )
      # Populate + EXCLUDE_FROM_ALL (instead of MakeAvailable): libwebp's
      # own install rules (headers, .pc files, WebPConfig.cmake) stay out
      # of our install tree; only the webp/sharpyuv libraries below are
      # exported alongside our targets.
      # CMP0169 (CMake >= 3.28) deprecates direct Populate; set OLD under
      # guard, mirroring the CMP0135 handling in the top-level CMakeLists.
      if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
      endif()
      FetchContent_Populate(libwebp)
      add_subdirectory(${libwebp_SOURCE_DIR} ${libwebp_BINARY_DIR} EXCLUDE_FROM_ALL)
      set(KOKKOSDNN_IMAGE_HAVE_WEBP ON)
      target_link_libraries(kokkosdnn_image PUBLIC webp)
      kokkosdnn_suppress_third_party_warnings(webp)
      kokkosdnn_suppress_third_party_warnings(sharpyuv)
      kokkosdnn_suppress_third_party_warnings(webpdecoder)
      # The fetched targets carry PUBLIC_HEADERs we deliberately do not
      # ship (our installed headers never include webp/*.h; consumers only
      # link). Clear them so install() skips headers without a dev warning.
      set_target_properties(webp sharpyuv PROPERTIES PUBLIC_HEADER "")
      install(TARGETS webp sharpyuv EXPORT KokkosDNNTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()
endif()
if(KOKKOSDNN_IMAGE_HAVE_WEBP)
  # PRIVATE: no libwebp headers appear in our installed headers (the WebP
  # helpers are declared in KokkosDNN_ImageIO.hpp but defined in
  # KokkosDNN_Image.cpp), so consumers never need this macro.
  target_compile_definitions(kokkosdnn_image PRIVATE KOKKOSDNN_IMAGE_HAVE_WEBP)
else()
  message(STATUS "KokkosDNN image: WebP support disabled")
endif()
# Propagate to the top-level scope for KokkosDNNConfig.cmake.in.
set(KOKKOSDNN_IMAGE_HAVE_WEBP "${KOKKOSDNN_IMAGE_HAVE_WEBP}" PARENT_SCOPE)
set(KokkosDNN_IMAGE_USES_SYSTEM_WEBP "${KokkosDNN_IMAGE_USES_SYSTEM_WEBP}" PARENT_SCOPE)

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
