# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Thinking Machines Inkling 2 Mini text example driver: model implementation
# lives in models/inkling (KokkosDNN::inkling target); this directory only
# builds the inkling_generate driver.
# The TARGET guard keeps configure green until the models/inkling component
# lands (sibling workstream); the parent examples/CMakeLists.txt option
# guard selects whether the driver is wanted at all.

if(TARGET KokkosDNN::inkling)
  kokkosdnn_add_executable(inkling_generate inkling_generate.cpp)
  target_link_libraries(inkling_generate PRIVATE KokkosDNN::inkling)
endif()
