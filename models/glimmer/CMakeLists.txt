# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Muse Glimmer 30B text model library: installable component (KokkosDNN::glimmer).
# Mechanical promotion of the former examples/glimmer library sources; no
# behavior change. Driver (glimmer_generate) and Test_*.cpp stay in
# examples/glimmer and link this target.

kokkosdnn_add_compiled_model_library(kokkosdnn_glimmer
  EXPORT_NAME glimmer
  SOURCES
    KokkosDNN_GlimmerConfig.cpp
    KokkosDNN_GlimmerWeights.cpp
    KokkosDNN_GlimmerTokenizer.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn KokkosDNN::common
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)
# Runtime data (not a compiler header): install the model config alongside
# other share/ payloads so <prefix>/include stays header-only.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config-glimmer-30b-text.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/KokkosDNN
)
