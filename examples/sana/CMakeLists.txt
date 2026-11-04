# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# SanaMS DiT example: text-to-image diffusion backbone inference built on
# top of the core library. Not part of libkokkosdnn; kept here as
# example/support code.

kokkosdnn_add_compiled_model_library(kokkosdnn_sana
  EXPORT_NAME sana
  SOURCES
    KokkosDNN_SanaWeights.cpp
    KokkosDNN_SanaVAE.cpp
    SanaWeightsPth.cpp
    KokkosDNN_SanaTextEncoder.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
  WITH_IMAGE
  NO_INSTALL
)

kokkosdnn_add_executable(sana_generate sana_generate.cpp)
target_link_libraries(sana_generate PRIVATE KokkosDNN::sana)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_sana_ops Test_SanaOpsAttention.cpp Test_SanaOpsConv.cpp
    LINK_LIBRARIES KokkosDNN::sana)
  kokkosdnn_add_test(test_sana_smoke Test_SanaSmoke.cpp
    LINK_LIBRARIES KokkosDNN::sana)
  kokkosdnn_add_test(test_sana_sampler Test_SanaSampler.cpp
    LINK_LIBRARIES KokkosDNN::sana)

  kokkosdnn_add_test(test_sana_vae Test_SanaVAE.cpp
    LINK_LIBRARIES KokkosDNN::sana
    COMPILE_DEFINITIONS SANA_WORKSPACE_DIR="${CMAKE_SOURCE_DIR}")

  kokkosdnn_add_test(test_sana_text_encoder Test_SanaTextEncoder.cpp
    LINK_LIBRARIES KokkosDNN::sana
    COMPILE_DEFINITIONS SANA_WORKSPACE_DIR="${CMAKE_SOURCE_DIR}")

  kokkosdnn_add_test(test_sana_golden Test_SanaGolden.cpp
    LINK_LIBRARIES KokkosDNN::sana
    COMPILE_DEFINITIONS SANA_WORKSPACE_DIR="${CMAKE_SOURCE_DIR}")
endif()
