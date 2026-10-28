# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Muse Glimmer 30B text example driver: model implementation lives in
# models/glimmer (KokkosDNN::glimmer); this directory only builds the
# glimmer_generate driver plus its tests.
# Separate family from examples/llama, examples/smollm2, and
# examples/gemma4 (config/weights/model/tokenizer are split-form duplicates
# in KokkosDNN::Glimmer; only the core src/ ops are shared).

add_executable(glimmer_generate glimmer_generate.cpp)
target_link_libraries(glimmer_generate PRIVATE KokkosDNN::glimmer)
set_target_properties(glimmer_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(glimmer_generate)

if(KokkosDNN_ENABLE_TESTS)
  # The config fixture (config-glimmer-30b-text.json) ships with the model
  # library; these tests locate it via the model source dir.
  add_executable(test_glimmer_config Test_GlimmerConfig.cpp)
  target_link_libraries(test_glimmer_config PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  target_compile_definitions(test_glimmer_config PRIVATE GLIMMER_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../models/glimmer")
  set_target_properties(test_glimmer_config PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_config)
  kokkosdnn_discover_tests(test_glimmer_config)

  add_executable(test_glimmer_generate Test_GlimmerGenerate.cpp)
  target_link_libraries(test_glimmer_generate PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  target_compile_definitions(test_glimmer_generate PRIVATE GLIMMER_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../models/glimmer")
  set_target_properties(test_glimmer_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_generate)
  kokkosdnn_discover_tests(test_glimmer_generate)

  add_executable(test_glimmer_session Test_GlimmerSession.cpp)
  target_link_libraries(test_glimmer_session PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  set_target_properties(test_glimmer_session PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_session)
  kokkosdnn_discover_tests(test_glimmer_session)

  add_executable(test_glimmer_tokenizer Test_GlimmerTokenizer.cpp)
  target_link_libraries(test_glimmer_tokenizer PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  set_target_properties(test_glimmer_tokenizer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_tokenizer)
  kokkosdnn_discover_tests(test_glimmer_tokenizer)

  add_executable(test_glimmer_weights_gguf Test_GlimmerWeightsGGUF.cpp)
  target_link_libraries(test_glimmer_weights_gguf PRIVATE KokkosDNN::glimmer KokkosDNN::gtest_main)
  set_target_properties(test_glimmer_weights_gguf PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(test_glimmer_weights_gguf)
  kokkosdnn_discover_tests(test_glimmer_weights_gguf)
endif()
