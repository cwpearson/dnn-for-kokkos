# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# models/common: shared model-support utilities (INTERFACE, header-only).
# KokkosDNN_Common.hpp umbrella + KokkosDNN_Detokenizer.hpp (streaming
# detokenizer with stop-string handling) + KokkosDNN_Usage.hpp (usage
# counters / finish reasons). Future home of further code extracted from
# models/*: minimal JSON helpers, BPE tokenizers, chat templates,
# samplers. See README.md.
# Every model lib links KokkosDNN::common; nothing here links a model lib.

add_library(kokkosdnn_common INTERFACE)
add_library(KokkosDNN::common ALIAS kokkosdnn_common)
target_link_libraries(kokkosdnn_common INTERFACE KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(kokkosdnn_common INTERFACE cxx_std_20)
kokkosdnn_enable_compiler_warnings(kokkosdnn_common)
set_target_properties(kokkosdnn_common PROPERTIES EXPORT_NAME common)

install(TARGETS kokkosdnn_common EXPORT KokkosDNNTargets)
install(FILES KokkosDNN_Common.hpp KokkosDNN_Detokenizer.hpp KokkosDNN_Usage.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(KokkosDNN_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
