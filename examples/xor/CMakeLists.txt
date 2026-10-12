# XOR example: tiny 2-layer MLP trained with the KokkosDNN training ops.
# Header-only MLP (KokkosDNN_XorMLP.hpp) + training driver + gtest.

add_executable(xor_train xor_train.cpp)
target_link_libraries(xor_train PRIVATE KokkosDNN::kokkosdnn)
target_include_directories(xor_train PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(xor_train PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_xor Test_Xor.cpp)
  target_link_libraries(test_xor PRIVATE KokkosDNN::kokkosdnn KokkosDNN::gtest_main)
  target_include_directories(test_xor PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )
  set_target_properties(test_xor PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_xor)
endif()
