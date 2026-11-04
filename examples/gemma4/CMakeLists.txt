# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Gemma4 E2B-it text example: model-specific inference built on top of the
# core library. Not part of libkokkosdnn; kept here as example/support code.
# Separate family from examples/llama and examples/smollm2 (config/weights/
# model/tokenizer are split-form duplicates in KokkosDNN::Gemma4; only the
# core src/ ops are shared).

kokkosdnn_add_compiled_model_library(kokkosdnn_gemma4
  EXPORT_NAME gemma4
  SOURCES
    KokkosDNN_Gemma4Weights.cpp
    KokkosDNN_Gemma4Tokenizer.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
  NO_INSTALL
)

kokkosdnn_add_executable(gemma4_generate gemma4_generate.cpp)
target_link_libraries(gemma4_generate PRIVATE KokkosDNN::gemma4)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_gemma4_generate Test_Gemma4Generate.cpp LINK_LIBRARIES KokkosDNN::gemma4)
endif()
