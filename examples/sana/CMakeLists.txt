# SanaMS DiT example: text-to-image diffusion backbone inference built on
# top of the core library. Not part of libkokkosdnn; kept here as
# example/support code.

add_library(kokkosdnn_sana STATIC
  KokkosDNN_SanaWeights.cpp
  KokkosDNN_SanaVAE.cpp
  SanaWeightsPth.cpp
  KokkosDNN_SanaTextEncoder.cpp
)
add_library(KokkosDNN::sana ALIAS kokkosdnn_sana)
target_link_libraries(kokkosdnn_sana PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_sana PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
if(KokkosDNN_ENABLE_IMAGE)
  target_link_libraries(kokkosdnn_sana PUBLIC KokkosDNN::image)
endif()
set_target_properties(kokkosdnn_sana PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(sana_generate sana_generate.cpp)
target_link_libraries(sana_generate PRIVATE KokkosDNN::sana)
set_target_properties(sana_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(KokkosDNN_ENABLE_TESTS)
  add_executable(test_sana_ops Test_SanaOpsAttention.cpp Test_SanaOpsConv.cpp)
  target_link_libraries(test_sana_ops PRIVATE KokkosDNN::sana KokkosDNN::gtest_main)
  set_target_properties(test_sana_ops PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_sana_ops)

  add_executable(test_sana_smoke Test_SanaSmoke.cpp)
  target_link_libraries(test_sana_smoke PRIVATE KokkosDNN::sana KokkosDNN::gtest_main)
  set_target_properties(test_sana_smoke PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_sana_smoke)

  add_executable(test_sana_sampler Test_SanaSampler.cpp)
  target_link_libraries(test_sana_sampler PRIVATE KokkosDNN::sana KokkosDNN::gtest_main)
  set_target_properties(test_sana_sampler PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_sana_sampler)

  add_executable(test_sana_vae Test_SanaVAE.cpp)
  target_link_libraries(test_sana_vae PRIVATE KokkosDNN::sana KokkosDNN::gtest_main)
  target_compile_definitions(test_sana_vae PRIVATE
    SANA_WORKSPACE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(test_sana_vae PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_sana_vae)

  add_executable(test_sana_text_encoder Test_SanaTextEncoder.cpp)
  target_link_libraries(test_sana_text_encoder PRIVATE KokkosDNN::sana KokkosDNN::gtest_main)
  # Workspace-absolute fallback so env-gated tests can locate the in-repo
  # Gemma/tokenizer paths under ctest (whose CWD is the build dir).
  target_compile_definitions(test_sana_text_encoder PRIVATE
    SANA_WORKSPACE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(test_sana_text_encoder PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_sana_text_encoder)

  add_executable(test_sana_golden Test_SanaGolden.cpp)
  target_link_libraries(test_sana_golden PRIVATE KokkosDNN::sana KokkosDNN::gtest_main)
  # Workspace-absolute fallback so the inventory test finds the in-repo
  # checkpoint under ctest (whose CWD is the build dir, not the repo root).
  target_compile_definitions(test_sana_golden PRIVATE
    SANA_WORKSPACE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(test_sana_golden PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_discover_tests(test_sana_golden)
endif()
