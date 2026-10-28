# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Muse Glimmer 30B text example: model-specific inference built on top of
# the core library. Not part of libkokkosdnn; kept here as example/support
# code. Separate family from examples/llama, examples/smollm2, and
# examples/gemma4 (config/weights/model/tokenizer are split-form duplicates
# in KokkosDNN::Glimmer; only the core src/ ops are shared).

add_library(kokkosdnn_glimmer STATIC
  KokkosDNN_GlimmerConfig.cpp
  KokkosDNN_GlimmerWeights.cpp
  KokkosDNN_GlimmerTokenizer.cpp
)
add_library(KokkosDNN::glimmer ALIAS kokkosdnn_glimmer)
target_link_libraries(kokkosdnn_glimmer PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_glimmer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(kokkosdnn_glimmer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_glimmer)

add_executable(glimmer_generate glimmer_generate.cpp)
target_link_libraries(glimmer_generate PRIVATE KokkosDNN::glimmer)
set_target_properties(glimmer_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(glimmer_generate)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_glimmer_config Test_GlimmerConfig.cpp)
  target_link_libraries(test_glimmer_config PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  set_target_properties(test_glimmer_config PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_config)
  kokkosdnn_discover_tests(test_glimmer_config)

  add_executable(test_glimmer_generate Test_GlimmerGenerate.cpp)
  target_link_libraries(test_glimmer_generate PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  set_target_properties(test_glimmer_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_generate)
  kokkosdnn_discover_tests(test_glimmer_generate)

  add_executable(test_glimmer_tokenizer Test_GlimmerTokenizer.cpp)
  target_link_libraries(test_glimmer_tokenizer PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  set_target_properties(test_glimmer_tokenizer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_tokenizer)
  kokkosdnn_discover_tests(test_glimmer_tokenizer)
endif()
