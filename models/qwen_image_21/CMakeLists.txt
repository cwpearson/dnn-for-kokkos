# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Qwen-Image-2.1 model library: installable component (KokkosDNN::qwen_image_21).
# Header-only pipeline
# (KokkosDNN_QwenImage21{Config,Dit,Encoder,Vae,Vision}.hpp); there are no
# .cpp sources, hence INTERFACE. Text-to-image plus edit conditioning: the
# encoder (forward_joint) and vision tower consume joint text+vision
# prompts; the DiT/VAE remain text-to-image-only (no image-condition DiT
# input path, decode-only VAE).

kokkosdnn_add_header_only_model_library(kokkosdnn_qwen_image_21
  EXPORT_NAME qwen_image_21
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
  WITH_IMAGE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.inc"
)
