# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Llama 3 example: model-specific inference built on top of the core library.
# Not part of libkokkosdnn; kept here as example/support code.

kokkosdnn_add_compiled_model_library(kokkosdnn_llama
  EXPORT_NAME llama
  SOURCES
    KokkosDNN_LlamaWeights.cpp
    KokkosDNN_Tokenizer.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
  NO_INSTALL
)

kokkosdnn_add_executable(llama_generate llama_generate.cpp)
target_link_libraries(llama_generate PRIVATE KokkosDNN::llama)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_llama_generate Test_LlamaGenerate.cpp LINK_LIBRARIES KokkosDNN::llama)
  kokkosdnn_add_test(test_safetensors Test_Safetensors.cpp LINK_LIBRARIES KokkosDNN::llama)
  kokkosdnn_add_test(test_tokenizer Test_Tokenizer.cpp LINK_LIBRARIES KokkosDNN::llama)
  kokkosdnn_add_test(test_llama_golden Test_LlamaGolden.cpp LINK_LIBRARIES KokkosDNN::llama)
  kokkosdnn_add_test(test_llama_perf Test_LlamaPerf.cpp LINK_LIBRARIES KokkosDNN::llama)
  kokkosdnn_add_test(test_llama_kvcache Test_LlamaKVCache.cpp LINK_LIBRARIES KokkosDNN::llama)
endif()
