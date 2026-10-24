# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
add_subdirectory(llama)
add_subdirectory(dinov3)
add_subdirectory(smollm2)
add_subdirectory(gemma4)
if(KokkosDNN_ENABLE_MODEL_LFM25)
  add_subdirectory(lfm2_5)
endif()
add_subdirectory(xor)
add_subdirectory(kokoro)
add_subdirectory(sana)
if(KokkosDNN_ENABLE_MODEL_MOSS)
  add_subdirectory(moss_image)
endif()
# Qwen-Image-2.1 text-to-image-only driver. The top-level
# KokkosDNN_ENABLE_MODEL_QWEN_IMAGE_21 option is owned by the model
# workstream; default ON here so the driver configures standalone, and the
# DEFINED guard keeps configure green when the option does not exist yet.
if(NOT DEFINED KokkosDNN_ENABLE_MODEL_QWEN_IMAGE_21)
  set(KokkosDNN_ENABLE_MODEL_QWEN_IMAGE_21 ON)
endif()
if(KokkosDNN_ENABLE_MODEL_QWEN_IMAGE_21)
  add_subdirectory(qwen_image_21)
endif()
