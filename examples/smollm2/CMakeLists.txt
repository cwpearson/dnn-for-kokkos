# SmolLM2-135M example: model-specific inference built on top of the core
# library. Not part of libkokkosdnn; kept here as example/support code.
# Separate family from examples/llama (config/weights/model/tokenizer are
# split-form duplicates in KokkosDNN::SmolLM2; only the core src/ ops are
# shared).

add_library(kokkosdnn_smollm2 STATIC
  KokkosDNN_SmolLM2Weights.cpp
  KokkosDNN_SmolLM2Tokenizer.cpp
)
add_library(KokkosDNN::smollm2 ALIAS kokkosdnn_smollm2)
target_link_libraries(kokkosdnn_smollm2 PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_smollm2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(kokkosdnn_smollm2 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(smollm2_generate smollm2_generate.cpp)
target_link_libraries(smollm2_generate PRIVATE KokkosDNN::smollm2)
set_target_properties(smollm2_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_smollm2_generate Test_SmolLM2Generate.cpp)
  target_link_libraries(test_smollm2_generate PRIVATE KokkosDNN::smollm2 KokkosDNN::gtest_main)
  set_target_properties(test_smollm2_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  gtest_discover_tests(test_smollm2_generate)

  add_executable(test_smollm2_tokenizer Test_SmolLM2Tokenizer.cpp)
  target_link_libraries(test_smollm2_tokenizer PRIVATE KokkosDNN::smollm2 KokkosDNN::gtest_main)
  set_target_properties(test_smollm2_tokenizer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  gtest_discover_tests(test_smollm2_tokenizer)
endif()
