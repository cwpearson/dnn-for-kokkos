# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# kokkosdnn_server: OpenAI-compatible HTTP front-end.
# Thin binary: owns Kokkos::initialize/finalize, parses CLI flags, loads the
# LFM2.5 model (config + GGUF weights + tokenizer), and serves OpenAI routes
# via cpp-httplib. All inference, sampling, and chat logic belongs in
# models/* — the server only translates HTTP <-> model-component calls.
# Requires the LFM2.5 model component (server serves LFM25).

if(NOT KokkosDNN_ENABLE_MODEL_LFM25)
  message(FATAL_ERROR
    "KokkosDNN_ENABLE_SERVER requires KokkosDNN_ENABLE_MODEL_LFM25=ON"
    " (the server serves LFM2.5).")
endif()

# HTTP + JSON front-end deps. Pins: cpp-httplib v0.15.3, nlohmann/json
# v3.11.3. FetchContent touches the network at configure time; for offline
# builds pre-seed the sources and point at them with
# -DFETCHCONTENT_SOURCE_DIR_CPP_HTTPLIB=<path> /
# -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=<path>.
include(FetchContent)
FetchContent_Declare(
  cpp_httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.15.3
)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(cpp_httplib nlohmann_json)
# Third-party policy: header-only front-ends reach our TUs as SYSTEM, so
# they can never trip -Werror.
kokkosdnn_suppress_third_party_warnings(httplib)
kokkosdnn_suppress_third_party_warnings(nlohmann_json)

add_executable(kokkosdnn_server server_main.cpp)
target_link_libraries(kokkosdnn_server PRIVATE
  KokkosDNN::lfm25 KokkosDNN::common
  httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(kokkosdnn_server PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_server)

# Parsing unit tests (pure JSON helpers; no weights/tokenizer needed).
# NOTE: server/ is added before the top-level enable_testing(), so enable
# testing here for this directory scope to make the add_test() below stick.
if(KokkosDNN_ENABLE_TESTS)
  enable_testing()
  add_executable(kokkosdnn_server_parsing_tests test_server_parsing.cpp)
  target_link_libraries(kokkosdnn_server_parsing_tests PRIVATE
    KokkosDNN::lfm25 KokkosDNN::common
    httplib::httplib nlohmann_json::nlohmann_json)
  set_target_properties(kokkosdnn_server_parsing_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(kokkosdnn_server_parsing_tests)
  add_test(NAME kokkosdnn_server_parsing
    COMMAND kokkosdnn_server_parsing_tests)
endif()

install(TARGETS kokkosdnn_server EXPORT KokkosDNNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
