# LFM2.5-24B-A2B example driver: model implementation lives in
# models/lfm2_5 (KokkosDNN::lfm25); this directory only builds the
# lfm2_5_generate driver plus its tests.
# Separate family from examples/llama (config/weights/model/tokenizer are
# split-form duplicates in KokkosDNN::LFM25; only the core src/ ops are
# shared). Weights load from GGUF (Q4_K_M); --tiny needs no download.

add_executable(lfm2_5_generate lfm2_5_generate.cpp)
target_link_libraries(lfm2_5_generate PRIVATE KokkosDNN::lfm25)
set_target_properties(lfm2_5_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(lfm2_5_generate)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_lfm25_config Test_LFM25Config.cpp)
  target_link_libraries(test_lfm25_config PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_config PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_lfm25_config)
  kokkosdnn_discover_tests(test_lfm25_config)

  add_executable(test_lfm25_generate Test_LFM25Generate.cpp)
  target_link_libraries(test_lfm25_generate PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_lfm25_generate)
  kokkosdnn_discover_tests(test_lfm25_generate)

  add_executable(test_lfm25_tokenizer Test_LFM25Tokenizer.cpp)
  target_link_libraries(test_lfm25_tokenizer PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_tokenizer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_lfm25_tokenizer)
  kokkosdnn_discover_tests(test_lfm25_tokenizer)

  add_executable(test_lfm25_session Test_LFM25Session.cpp)
  target_link_libraries(test_lfm25_session PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_session PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_lfm25_session)
  kokkosdnn_discover_tests(test_lfm25_session)

  add_executable(test_lfm25_weights_gguf Test_LFM25WeightsGGUF.cpp)
  target_link_libraries(test_lfm25_weights_gguf PRIVATE KokkosDNN::lfm25 KokkosDNN::gtest_main)
  set_target_properties(test_lfm25_weights_gguf PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_lfm25_weights_gguf)
  kokkosdnn_discover_tests(test_lfm25_weights_gguf)
endif()
