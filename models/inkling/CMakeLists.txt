# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Inkling 2 Mini text model library: installable component (KokkosDNN::inkling).

kokkosdnn_add_compiled_model_library(kokkosdnn_inkling
  EXPORT_NAME inkling
  SOURCES
    KokkosDNN_InklingConfig.cpp
    KokkosDNN_InklingWeights.cpp
    KokkosDNN_InklingTokenizer.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)
# Runtime data (not a compiler header): install the model config alongside
# other share/ payloads so <prefix>/include stays header-only.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config-inkling-2-mini.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/KokkosDNN
)
