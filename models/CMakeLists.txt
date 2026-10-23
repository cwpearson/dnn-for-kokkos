# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Model components: installable per-model libraries plus shared support.
# models/common is unconditional (every model lib links KokkosDNN::common);
# per-model subdirectories are guarded so
# -DKokkosDNN_ENABLE_MODEL_<NAME>=OFF drops them (and their example
# drivers, see examples/CMakeLists.txt).
add_subdirectory(common)
if(KokkosDNN_ENABLE_MODEL_LFM25)
  add_subdirectory(lfm2_5)
endif()
if(KokkosDNN_ENABLE_MODEL_MOSS)
  add_subdirectory(moss_image)
endif()
