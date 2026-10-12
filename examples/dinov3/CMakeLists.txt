# DINOv3 ViT example: image-encoder inference built on top of the core library.
# Not part of libkokkosdnn; kept here as example/support code.

add_library(kokkosdnn_dinov3 STATIC
  KokkosDNN_Dinov3Weights.cpp
)
add_library(KokkosDNN::dinov3 ALIAS kokkosdnn_dinov3)
target_link_libraries(kokkosdnn_dinov3 PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_dinov3 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
if(KokkosDNN_ENABLE_IMAGE)
  target_link_libraries(kokkosdnn_dinov3 PUBLIC KokkosDNN::image)
endif()
set_target_properties(kokkosdnn_dinov3 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(dinov3_encode dinov3_encode.cpp)
target_link_libraries(dinov3_encode PRIVATE KokkosDNN::dinov3)
set_target_properties(dinov3_encode PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_dinov3_smoke Test_Dinov3Smoke.cpp)
  target_link_libraries(test_dinov3_smoke PRIVATE KokkosDNN::dinov3 KokkosDNN::gtest_main)
  set_target_properties(test_dinov3_smoke PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_dinov3_smoke)

  add_executable(test_dinov3_golden Test_Dinov3Golden.cpp)
  target_link_libraries(test_dinov3_golden PRIVATE KokkosDNN::dinov3 KokkosDNN::gtest_main)
  set_target_properties(test_dinov3_golden PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_dinov3_golden)
endif()
