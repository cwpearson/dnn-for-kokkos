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

# Canonical path: stb (header-only). Look for a local stb_image.h first.
find_path(KokkosDNN_STB_INCLUDE_DIR stb_image.h
  HINTS $ENV{STB_ROOT} ${STB_ROOT} /usr/include /usr/local/include
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
# Expose as an IMPORTED third-party target: imported targets linked PUBLICly
# do not need to belong to our install export set. Only the build-tree path
# is exposed; installing stb headers for a relocatable package is a TODO.
# The core umbrella (KokkosDNN.hpp) includes this component when
# KOKKOSDNN_ENABLE_IMAGE is defined, and KokkosDNN_ImageIO.hpp needs
# stb_image.h at compile time, so the stb dir is also exposed through the
# core target (build tree only) for existing core consumers. This is not a
# dependency on kokkosdnn_image itself: core never links it.
target_include_directories(kokkosdnn PUBLIC
  $<BUILD_INTERFACE:${KokkosDNN_STB_INCLUDE_DIR}>)
add_library(stb::stb INTERFACE IMPORTED GLOBAL)
set_target_properties(stb::stb PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${KokkosDNN_STB_INCLUDE_DIR}>")

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
target_link_libraries(kokkosdnn_image PUBLIC Kokkos::kokkos stb::stb)

if(KokkosDNN_ENABLE_COMPILER_WARNINGS)
  target_compile_options(kokkosdnn_image PRIVATE -Wall -Wextra -Werror)
endif()

set_target_properties(kokkosdnn_image PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

install(TARGETS kokkosdnn_image EXPORT KokkosDNNTragets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
