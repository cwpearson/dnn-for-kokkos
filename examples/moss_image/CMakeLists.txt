# MOSS-Image-SFT example: text-to-image DiT backbone inference built on top
# of the core library. Not part of libkokkosdnn; kept here as
# example/support code.

add_library(kokkosdnn_moss STATIC
  KokkosDNN_MossWeights.cpp
)
add_library(KokkosDNN::moss ALIAS kokkosdnn_moss)
target_link_libraries(kokkosdnn_moss PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_moss PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
if(KokkosDNN_ENABLE_IMAGE)
  target_link_libraries(kokkosdnn_moss PUBLIC KokkosDNN::image)
endif()
set_target_properties(kokkosdnn_moss PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# True-pipeline headers (Dit/Encoder/Vae) land in top-level src/ from
# parallel worktrees. They are already on the interface include path of
# KokkosDNN::kokkosdnn; probe here so configure logs which pipeline pieces
# are visible. This is status only: the driver selects them with
# __has_include and always builds with or without them (no hard dep).
find_file(MOSS_IMAGE_DIT_HEADER KokkosDNN_MossImageDit.hpp
  PATHS ${CMAKE_SOURCE_DIR}/src)
find_file(MOSS_IMAGE_ENCODER_HEADER KokkosDNN_MossImageEncoder.hpp
  PATHS ${CMAKE_SOURCE_DIR}/src)
find_file(MOSS_IMAGE_VAE_HEADER KokkosDNN_MossImageVae.hpp
  PATHS ${CMAKE_SOURCE_DIR}/src)
if(MOSS_IMAGE_DIT_HEADER AND NOT MOSS_IMAGE_DIT_HEADER MATCHES "-NOTFOUND$")
  message(STATUS "moss_image: KokkosDNN_MossImageDit.hpp visible")
else()
  message(STATUS "moss_image: KokkosDNN_MossImageDit.hpp absent (placeholder fallback)")
endif()
if(MOSS_IMAGE_ENCODER_HEADER AND NOT MOSS_IMAGE_ENCODER_HEADER MATCHES "-NOTFOUND$")
  message(STATUS "moss_image: KokkosDNN_MossImageEncoder.hpp visible")
else()
  message(STATUS "moss_image: KokkosDNN_MossImageEncoder.hpp absent (file/null fallback)")
endif()
if(MOSS_IMAGE_VAE_HEADER AND NOT MOSS_IMAGE_VAE_HEADER MATCHES "-NOTFOUND$")
  message(STATUS "moss_image: KokkosDNN_MossImageVae.hpp visible")
else()
  message(STATUS "moss_image: KokkosDNN_MossImageVae.hpp absent (preview fallback)")
endif()

add_executable(moss_image_generate moss_image_generate.cpp)
target_link_libraries(moss_image_generate PRIVATE KokkosDNN::moss)
set_target_properties(moss_image_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
