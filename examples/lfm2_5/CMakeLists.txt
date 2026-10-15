# LFM2.5-24B-A2B example: model-specific inference built on top of the core
# library. Not part of libkokkosdnn; kept here as example/support code.
# Separate family from examples/llama (config/weights/model/tokenizer are
# split-form duplicates in KokkosDNN::LFM25; only the core src/ ops are
# shared). Weights load from GGUF (Q4_K_M); --tiny needs no download.

add_library(kokkosdnn_lfm25 STATIC
  KokkosDNN_LFM25Weights.cpp
  KokkosDNN_LFM25Tokenizer.cpp
)
add_library(KokkosDNN::lfm25 ALIAS kokkosdnn_lfm25)
target_link_libraries(kokkosdnn_lfm25 PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_lfm25 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(kokkosdnn_lfm25 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(lfm2_5_generate lfm2_5_generate.cpp)
target_link_libraries(lfm2_5_generate PRIVATE KokkosDNN::lfm25)
set_target_properties(lfm2_5_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_lfm25_config Test_LFM25Config.cpp)
  target_link_libraries(test_lfm25_config PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_config PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_lfm25_config)

  add_executable(test_lfm25_generate Test_LFM25Generate.cpp)
  target_link_libraries(test_lfm25_generate PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_lfm25_generate)

  add_executable(test_lfm25_tokenizer Test_LFM25Tokenizer.cpp)
  target_link_libraries(test_lfm25_tokenizer PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_tokenizer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_lfm25_tokenizer)
endif()
