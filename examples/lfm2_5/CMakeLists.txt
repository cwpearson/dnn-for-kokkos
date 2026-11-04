# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# LFM2.5-24B-A2B example driver: model implementation lives in
# models/lfm2_5 (KokkosDNN::lfm25); this directory only builds the
# lfm2_5_generate driver plus its tests.
# Separate family from examples/llama (config/weights/model/tokenizer are
# split-form duplicates in KokkosDNN::LFM25; only the core src/ ops are
# shared). Weights load from GGUF (Q4_K_M); --tiny needs no download.

kokkosdnn_add_executable(lfm2_5_generate lfm2_5_generate.cpp)
target_link_libraries(lfm2_5_generate PRIVATE KokkosDNN::lfm25)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_lfm25_config Test_LFM25Config.cpp LINK_LIBRARIES KokkosDNN::lfm25)
  kokkosdnn_add_test(test_lfm25_generate Test_LFM25Generate.cpp LINK_LIBRARIES KokkosDNN::lfm25)
  kokkosdnn_add_test(test_lfm25_tokenizer Test_LFM25Tokenizer.cpp LINK_LIBRARIES KokkosDNN::lfm25)
  kokkosdnn_add_test(test_lfm25_session Test_LFM25Session.cpp LINK_LIBRARIES KokkosDNN::lfm25)
  kokkosdnn_add_test(test_lfm25_weights_gguf Test_LFM25WeightsGGUF.cpp LINK_LIBRARIES KokkosDNN::lfm25)
endif()
