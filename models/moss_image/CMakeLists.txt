# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# MOSS-Image model library: installable component (KokkosDNN::moss).
# Compiled STATIC library like LFM25/Glimmer: the placeholder Moss geometry
# (KokkosDNN_Moss{Config,Model,Sampler}.hpp compiled via
# KokkosDNN_MossWeights.cpp, plus the KokkosDNN_MossZip.hpp ZIP-reader
# wrapper) lives here alongside the header-only true pipeline
# (KokkosDNN_MossImage{,Dit,Encoder,Vae}.hpp, KokkosDNN_QwenTokenizer.hpp +
# tables .inc). The driver (moss_image_generate) in examples/moss_image links
# this target and holds no model sources of its own.

kokkosdnn_add_compiled_model_library(kokkosdnn_moss
  EXPORT_NAME moss
  SOURCES
    KokkosDNN_MossWeights.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
  WITH_IMAGE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.inc"
)
