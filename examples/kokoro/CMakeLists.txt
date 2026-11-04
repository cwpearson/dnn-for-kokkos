# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Kokoro-82M TTS example: native Kokkos StyleTTS2 + ISTFTNet inference.
# Not part of libkokkosdnn; model-specific code lives here.

kokkosdnn_add_compiled_model_library(kokkosdnn_kokoro
  EXPORT_NAME kokoro
  SOURCES
    KokkosDNN_KokoroConfig.cpp
    KokkosDNN_KokoroPth.cpp
    KokkosDNN_KokoroWeights.cpp
  LINK_LIBRARIES KokkosDNN::kokkosdnn
  NO_INSTALL
)

kokkosdnn_add_executable(kokoro_tts kokoro_tts.cpp)
target_link_libraries(kokoro_tts PRIVATE KokkosDNN::kokoro)

if(KokkosDNN_ENABLE_TESTS)
  kokkosdnn_add_test(test_kokoro_smoke Test_KokoroSmoke.cpp LINK_LIBRARIES KokkosDNN::kokoro)
  kokkosdnn_add_test(test_kokoro_pth Test_KokoroPth.cpp LINK_LIBRARIES KokkosDNN::kokoro)
  kokkosdnn_add_test(test_kokoro_golden Test_KokoroGolden.cpp LINK_LIBRARIES KokkosDNN::kokoro)
endif()
