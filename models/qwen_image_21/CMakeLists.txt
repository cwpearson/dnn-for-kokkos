# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Qwen-Image-2.1 model library: installable component (KokkosDNN::qwen_image_21).
# Header-only pipeline (KokkosDNN_QwenImage21{Config,Dit,Encoder,Vae}.hpp);
# there are no .cpp sources, hence INTERFACE. Text-to-image only.

add_library(kokkosdnn_qwen_image_21 INTERFACE)
add_library(KokkosDNN::qwen_image_21 ALIAS kokkosdnn_qwen_image_21)
target_link_libraries(kokkosdnn_qwen_image_21 INTERFACE
  KokkosDNN::kokkosdnn KokkosDNN::common)
target_include_directories(kokkosdnn_qwen_image_21 INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(KokkosDNN_ENABLE_IMAGE)
  target_link_libraries(kokkosdnn_qwen_image_21 INTERFACE KokkosDNN::image)
endif()
target_compile_features(kokkosdnn_qwen_image_21 INTERFACE cxx_std_20)
kokkosdnn_enable_compiler_warnings(kokkosdnn_qwen_image_21)
set_target_properties(kokkosdnn_qwen_image_21 PROPERTIES EXPORT_NAME qwen_image_21)

install(TARGETS kokkosdnn_qwen_image_21 EXPORT KokkosDNNTargets)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.inc"
)
