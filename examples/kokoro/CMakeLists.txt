# Kokoro-82M TTS example: native Kokkos StyleTTS2 + ISTFTNet inference.
# Not part of libkokkosdnn; model-specific code lives here.

add_library(kokkosdnn_kokoro STATIC
  KokkosDNN_KokoroConfig.cpp
  KokkosDNN_KokoroPth.cpp
  KokkosDNN_KokoroWeights.cpp
)
add_library(KokkosDNN::kokoro ALIAS kokkosdnn_kokoro)
target_link_libraries(kokkosdnn_kokoro PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_kokoro PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(kokkosdnn_kokoro PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(kokoro_tts kokoro_tts.cpp)
target_link_libraries(kokoro_tts PRIVATE KokkosDNN::kokoro)
set_target_properties(kokoro_tts PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_kokoro_smoke Test_KokoroSmoke.cpp)
  target_link_libraries(test_kokoro_smoke PRIVATE KokkosDNN::kokoro KokkosDNN::gtest_main)
  set_target_properties(test_kokoro_smoke PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_kokoro_smoke)

  add_executable(test_kokoro_pth Test_KokoroPth.cpp)
  target_link_libraries(test_kokoro_pth PRIVATE KokkosDNN::kokoro KokkosDNN::gtest_main)
  set_target_properties(test_kokoro_pth PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_kokoro_pth)

  add_executable(test_kokoro_golden Test_KokoroGolden.cpp)
  target_link_libraries(test_kokoro_golden PRIVATE KokkosDNN::kokoro KokkosDNN::gtest_main)
  set_target_properties(test_kokoro_golden PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_kokoro_golden)
endif()
