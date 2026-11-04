# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# DINOv3 ViT example: image-encoder inference built on top of the core library.
# Not part of libkokkosdnn; kept here as example/support code.

kokkosdnn_add_compiled_model_library(kokkosdnn_dinov3
  EXPORT_NAME dinov3
  SOURCES KokkosDNN_Dinov3Weights.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn
  WITH_IMAGE
  NO_INSTALL
)

kokkosdnn_add_executable(dinov3_encode dinov3_encode.cpp)
target_link_libraries(dinov3_encode PRIVATE KokkosDNN::dinov3)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_dinov3_smoke Test_Dinov3Smoke.cpp LINK_LIBRARIES KokkosDNN::dinov3)
  kokkosdnn_add_test(test_dinov3_golden Test_Dinov3Golden.cpp LINK_LIBRARIES KokkosDNN::dinov3)
endif()
