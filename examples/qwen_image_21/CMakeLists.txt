# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Qwen-Image-2.1 example drivers: text-to-image (generate, with an
# interactive REPL) and one-shot reference-image editing (edit). The
# true-pipeline model headers live in models/qwen_image_21
# (KokkosDNN::qwen_image_21, owned by the model workstream) and are NOT
# required here. Both drivers share qwen_image_21_common.hpp and probe for
# those headers via __has_include, otherwise running the self-contained
# placeholder / file / null path, so configure and the --tiny smoke runs
# always work.

function(qwen_image_21_add_driver name src)
  add_executable(${name} ${src})
  set_target_properties(${name} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  if(TARGET kokkosdnn_qwen_image_21)
    target_link_libraries(${name} PRIVATE KokkosDNN::qwen_image_21)
  else()
    target_link_libraries(${name} PRIVATE KokkosDNN::kokkosdnn)
  endif()
  if(TARGET kokkosdnn_image)
    target_link_libraries(${name} PRIVATE KokkosDNN::image)
  endif()
  kokkosdnn_enable_compiler_warnings(${name})
endfunction()

qwen_image_21_add_driver(qwen_image_21_generate qwen_image_21_generate.cpp)
target_link_libraries(qwen_image_21_generate PRIVATE KokkosDNN::example_readline)
qwen_image_21_add_driver(qwen_image_21_edit qwen_image_21_edit.cpp)
