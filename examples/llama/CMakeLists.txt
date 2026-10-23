# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Llama 3 example: model-specific inference built on top of the core library.
# Not part of libkokkosdnn; kept here as example/support code.

add_library(kokkosdnn_llama STATIC
  KokkosDNN_LlamaWeights.cpp
  KokkosDNN_Tokenizer.cpp
)
add_library(KokkosDNN::llama ALIAS kokkosdnn_llama)
target_link_libraries(kokkosdnn_llama PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_llama PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(kokkosdnn_llama PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_llama)

add_executable(llama_generate llama_generate.cpp)
target_link_libraries(llama_generate PRIVATE KokkosDNN::llama)
set_target_properties(llama_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(llama_generate)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_llama_generate Test_LlamaGenerate.cpp)
  target_link_libraries(test_llama_generate PRIVATE KokkosDNN::llama KokkosDNN::gtest_main)
  set_target_properties(test_llama_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_llama_generate)
  kokkosdnn_discover_tests(test_llama_generate)

  add_executable(test_safetensors Test_Safetensors.cpp)
  target_link_libraries(test_safetensors PRIVATE KokkosDNN::llama KokkosDNN::gtest_main)
  set_target_properties(test_safetensors PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_safetensors)
  kokkosdnn_discover_tests(test_safetensors)

  add_executable(test_tokenizer Test_Tokenizer.cpp)
  target_link_libraries(test_tokenizer PRIVATE KokkosDNN::llama KokkosDNN::gtest_main)
  set_target_properties(test_tokenizer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_tokenizer)
  kokkosdnn_discover_tests(test_tokenizer)

  add_executable(test_llama_golden Test_LlamaGolden.cpp)
  target_link_libraries(test_llama_golden PRIVATE KokkosDNN::llama KokkosDNN::gtest_main)
  set_target_properties(test_llama_golden PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_llama_golden)
  kokkosdnn_discover_tests(test_llama_golden)

  add_executable(test_llama_perf Test_LlamaPerf.cpp)
  target_link_libraries(test_llama_perf PRIVATE KokkosDNN::llama KokkosDNN::gtest_main)
  set_target_properties(test_llama_perf PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_llama_perf)
  kokkosdnn_discover_tests(test_llama_perf)

  add_executable(test_llama_kvcache Test_LlamaKVCache.cpp)
  target_link_libraries(test_llama_kvcache PRIVATE KokkosDNN::llama KokkosDNN::gtest_main)
  set_target_properties(test_llama_kvcache PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_llama_kvcache)
  kokkosdnn_discover_tests(test_llama_kvcache)
endif()
