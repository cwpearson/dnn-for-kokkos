# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# XOR example: tiny 2-layer MLP trained with the KokkosDNN training ops.
# Header-only MLP (KokkosDNN_XorMLP.hpp) + training driver + gtest.

kokkosdnn_add_executable(xor_train xor_train.cpp)
target_link_libraries(xor_train PRIVATE KokkosDNN::kokkosdnn)
target_include_directories(xor_train PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_xor Test_Xor.cpp
    INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
endif()
