# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# LFM2.5 model library: installable component (KokkosDNN::lfm25).
# Mechanical promotion of the former examples/lfm2_5 library sources; no
# behavior change. Driver (lfm2_5_generate) and Test_*.cpp stay in
# examples/lfm2_5 and link this target.

add_library(kokkosdnn_lfm25 STATIC
  KokkosDNN_LFM25Weights.cpp
  KokkosDNN_LFM25Tokenizer.cpp
)
add_library(KokkosDNN::lfm25 ALIAS kokkosdnn_lfm25)
target_link_libraries(kokkosdnn_lfm25 PUBLIC KokkosDNN::kokkosdnn KokkosDNN::common)
target_include_directories(kokkosdnn_lfm25 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(kokkosdnn_lfm25 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON EXPORT_NAME lfm25)
kokkosdnn_enable_compiler_warnings(kokkosdnn_lfm25)

install(TARGETS kokkosdnn_lfm25 EXPORT KokkosDNNTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.json"
)
