# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# MOSS-Image model library: installable component (KokkosDNN::moss).
# Header-only pipeline (KokkosDNN_MossImage{,Dit,Encoder,Vae}.hpp,
# KokkosDNN_QwenTokenizer.hpp + tables .inc live here); there are no .cpp
# sources, hence INTERFACE. The example-local support code
# (KokkosDNN_Moss{Config,Model,Sampler}.hpp, KokkosDNN_MossWeights.cpp)
# stays in examples/moss_image and compiles into the driver there.

kokkosdnn_add_header_only_model_library(kokkosdnn_moss
  EXPORT_NAME moss
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
  WITH_IMAGE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.inc"
)
