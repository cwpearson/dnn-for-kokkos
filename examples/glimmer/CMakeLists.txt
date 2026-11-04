# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Muse Glimmer 30B text example driver: model implementation lives in
# models/glimmer (KokkosDNN::glimmer); this directory only builds the
# glimmer_generate driver plus its tests.
# Separate family from examples/llama, examples/smollm2, and
# examples/gemma4 (config/weights/model/tokenizer are split-form duplicates
# in KokkosDNN::Glimmer; only the core src/ ops are shared).

kokkosdnn_add_executable(glimmer_generate glimmer_generate.cpp)
target_link_libraries(glimmer_generate PRIVATE KokkosDNN::glimmer)

if(KokkosDNN_ENABLE_TESTS)
  # The config fixture (config-glimmer-30b-text.json) ships with the model
  # library; these tests locate it via the model source dir.
  kokkosdnn_add_test(test_glimmer_config Test_GlimmerConfig.cpp
    LINK_LIBRARIES KokkosDNN::glimmer
    COMPILE_DEFINITIONS GLIMMER_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../models/glimmer")
  kokkosdnn_add_test(test_glimmer_generate Test_GlimmerGenerate.cpp
    LINK_LIBRARIES KokkosDNN::glimmer
    COMPILE_DEFINITIONS GLIMMER_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../models/glimmer")
  kokkosdnn_add_test(test_glimmer_session Test_GlimmerSession.cpp
    LINK_LIBRARIES KokkosDNN::glimmer)
  kokkosdnn_add_test(test_glimmer_tokenizer Test_GlimmerTokenizer.cpp
    LINK_LIBRARIES KokkosDNN::glimmer)
  kokkosdnn_add_test(test_glimmer_weights_gguf Test_GlimmerWeightsGGUF.cpp
    LINK_LIBRARIES KokkosDNN::glimmer)
  kokkosdnn_add_test(test_glimmer_quant_dispatch Test_GlimmerQuantDispatch.cpp
    LINK_LIBRARIES KokkosDNN::glimmer)
endif()
