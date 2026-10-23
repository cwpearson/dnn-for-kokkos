# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Gemma4 E2B-it text example: model-specific inference built on top of the
# core library. Not part of libkokkosdnn; kept here as example/support code.
# Separate family from examples/llama and examples/smollm2 (config/weights/
# model/tokenizer are split-form duplicates in KokkosDNN::Gemma4; only the
# core src/ ops are shared).

add_library(kokkosdnn_gemma4 STATIC
  KokkosDNN_Gemma4Weights.cpp
  KokkosDNN_Gemma4Tokenizer.cpp
)
add_library(KokkosDNN::gemma4 ALIAS kokkosdnn_gemma4)
target_link_libraries(kokkosdnn_gemma4 PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_gemma4 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(kokkosdnn_gemma4 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_gemma4)

add_executable(gemma4_generate gemma4_generate.cpp)
target_link_libraries(gemma4_generate PRIVATE KokkosDNN::gemma4)
set_target_properties(gemma4_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(gemma4_generate)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_gemma4_generate Test_Gemma4Generate.cpp)
  target_link_libraries(test_gemma4_generate PRIVATE KokkosDNN::gemma4 KokkosDNN::gtest_main)
  set_target_properties(test_gemma4_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_gemma4_generate)
  kokkosdnn_discover_tests(test_gemma4_generate)
endif()
