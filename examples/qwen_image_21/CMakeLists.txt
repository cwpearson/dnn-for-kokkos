# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Qwen-Image-2.1 text-to-image example driver: the true-pipeline model
# headers live in models/qwen_image_21 (KokkosDNN::qwen_image_21, owned by
# the model workstream) and are NOT required here. This directory builds
# only the qwen_image_21_generate driver, which probes for those headers
# via __has_include and otherwise runs the self-contained placeholder /
# file / null path, so configure and the --tiny smoke run always work.

add_executable(qwen_image_21_generate qwen_image_21_generate.cpp)
set_target_properties(qwen_image_21_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(TARGET kokkosdnn_qwen_image_21)
  target_link_libraries(qwen_image_21_generate PRIVATE KokkosDNN::qwen_image_21)
else()
  target_link_libraries(qwen_image_21_generate PRIVATE KokkosDNN::kokkosdnn)
endif()
if(TARGET kokkosdnn_image)
  target_link_libraries(qwen_image_21_generate PRIVATE KokkosDNN::image)
endif()
kokkosdnn_enable_compiler_warnings(qwen_image_21_generate)
