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

add_executable(moss_image_generate moss_image_generate.cpp)
target_link_libraries(moss_image_generate PRIVATE KokkosDNN::moss)
set_target_properties(moss_image_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
