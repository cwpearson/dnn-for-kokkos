# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# SmolLM2-135M example: model-specific inference built on top of the
# core library. Not part of libkokkosdnn; kept here as example/support code.
# Separate family from examples/llama (config/weights/model/tokenizer are
# split-form duplicates in KokkosDNN::SmolLM2; only the core src/ ops are
# shared).

kokkosdnn_add_compiled_model_library(kokkosdnn_smollm2
  EXPORT_NAME smollm2
  SOURCES
    KokkosDNN_SmolLM2Weights.cpp
    KokkosDNN_SmolLM2Tokenizer.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
  NO_INSTALL
)

kokkosdnn_add_executable(smollm2_generate smollm2_generate.cpp)
target_link_libraries(smollm2_generate PRIVATE KokkosDNN::smollm2)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_smollm2_generate Test_SmolLM2Generate.cpp LINK_LIBRARIES KokkosDNN::smollm2)
  kokkosdnn_add_test(test_smollm2_tokenizer Test_SmolLM2Tokenizer.cpp LINK_LIBRARIES KokkosDNN::smollm2)
endif()
