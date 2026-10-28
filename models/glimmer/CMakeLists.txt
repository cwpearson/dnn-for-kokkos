# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Muse Glimmer 30B text model library: installable component (KokkosDNN::glimmer).
# Mechanical promotion of the former examples/glimmer library sources; no
# behavior change. Driver (glimmer_generate) and Test_*.cpp stay in
# examples/glimmer and link this target.

add_library(kokkosdnn_glimmer STATIC
  KokkosDNN_GlimmerConfig.cpp
  KokkosDNN_GlimmerWeights.cpp
  KokkosDNN_GlimmerTokenizer.cpp
)
add_library(KokkosDNN::glimmer ALIAS kokkosdnn_glimmer)
target_link_libraries(kokkosdnn_glimmer PUBLIC KokkosDNN::kokkosdnn KokkosDNN::common)
target_include_directories(kokkosdnn_glimmer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(kokkosdnn_glimmer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON EXPORT_NAME glimmer)
kokkosdnn_enable_compiler_warnings(kokkosdnn_glimmer)

install(TARGETS kokkosdnn_glimmer EXPORT KokkosDNNTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
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
