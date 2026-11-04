# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# LFM2.5 model library: installable component (KokkosDNN::lfm25).
# Mechanical promotion of the former examples/lfm2_5 library sources; no
# behavior change. Driver (lfm2_5_generate) and Test_*.cpp stay in
# examples/lfm2_5 and link this target.

kokkosdnn_add_compiled_model_library(kokkosdnn_lfm25
  EXPORT_NAME lfm25
  SOURCES
    KokkosDNN_LFM25Weights.cpp
    KokkosDNN_LFM25Tokenizer.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)
# Runtime data (not a compiler header): install the model config alongside
# other share/ payloads so <prefix>/include stays header-only.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config-lfm2_5-24b-a2b.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/KokkosDNN
)
