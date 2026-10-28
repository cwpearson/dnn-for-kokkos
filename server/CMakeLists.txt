# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# kokkosdnn_server: OpenAI-compatible HTTP front-end.
# Thin binary: owns Kokkos::initialize/finalize, parses CLI flags, loads one
# model backend (KokkosDNN::Server::ChatBackend; per-arch code in
# BackendLFM25.cpp / BackendGlimmer.cpp), and serves OpenAI routes via
# cpp-httplib. All inference, sampling, and chat logic belongs in
# models/* — the server only translates HTTP <-> backend calls.
# Requires at least one model component. The Glimmer backend additionally
# needs the models/glimmer component (KokkosDNN::glimmer target); without it
# BackendGlimmer.cpp compiles to a stub whose load() reports the missing
# component.

if(NOT KokkosDNN_ENABLE_MODEL_LFM25 AND NOT KokkosDNN_ENABLE_MODEL_GLIMMER)
  message(FATAL_ERROR
    "KokkosDNN_ENABLE_SERVER requires at least one model component "
    "(KokkosDNN_ENABLE_MODEL_LFM25=ON and/or KokkosDNN_ENABLE_MODEL_GLIMMER=ON).")
endif()
if(KokkosDNN_ENABLE_MODEL_GLIMMER AND NOT TARGET KokkosDNN::glimmer)
  message(FATAL_ERROR
    "KokkosDNN_ENABLE_MODEL_GLIMMER=ON but the KokkosDNN::glimmer target is "
    "missing (unset the option to build the server with the Glimmer backend "
    "stubbed out).")
endif()

# HTTP + JSON front-end deps. Pins: cpp-httplib v0.63.0, nlohmann/json
# v3.11.3. FetchContent touches the network at configure time; for offline
# builds pre-seed the sources and point at them with
# -DFETCHCONTENT_SOURCE_DIR_CPP_HTTPLIB=<path> /
# -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=<path>.
include(FetchContent)
FetchContent_Declare(
  cpp_httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.63.0
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

# Backend availability: per-arch compile definitions select the real
# implementation in Backend{ LFM25,Glimmer}.cpp; without them the TU
# compiles to a load()-time stub so the arch name still resolves.
set(KOKKOSDNN_SERVER_BACKEND_DEFS "")
set(KOKKOSDNN_SERVER_BACKEND_LIBS "")
if(KokkosDNN_ENABLE_MODEL_LFM25)
  list(APPEND KOKKOSDNN_SERVER_BACKEND_DEFS KOKKOSDNN_SERVER_HAVE_LFM25)
  list(APPEND KOKKOSDNN_SERVER_BACKEND_LIBS KokkosDNN::lfm25)
endif()
if(TARGET KokkosDNN::glimmer)
  list(APPEND KOKKOSDNN_SERVER_BACKEND_DEFS KOKKOSDNN_SERVER_HAVE_GLIMMER)
  list(APPEND KOKKOSDNN_SERVER_BACKEND_LIBS KokkosDNN::glimmer)
endif()

set(KOKKOSDNN_SERVER_BACKEND_SOURCES BackendLFM25.cpp BackendGlimmer.cpp)

add_executable(kokkosdnn_server server_main.cpp ${KOKKOSDNN_SERVER_BACKEND_SOURCES})
target_compile_definitions(kokkosdnn_server PRIVATE ${KOKKOSDNN_SERVER_BACKEND_DEFS})
target_link_libraries(kokkosdnn_server PRIVATE
  ${KOKKOSDNN_SERVER_BACKEND_LIBS} KokkosDNN::common
  httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(kokkosdnn_server PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(kokkosdnn_server)

# Parsing unit tests (pure JSON helpers; no weights/tokenizer needed).
# NOTE: server/ is added before the top-level enable_testing(), so enable
# testing here for this directory scope to make the add_test() below stick.
if(KokkosDNN_ENABLE_TESTS)
  enable_testing()
  # NOTE: test_server_parsing.cpp #includes server_main.cpp (NO_MAIN), so
  # only the backend TUs are compiled in addition — never server_main.cpp
  # itself (that would duplicate main() and every helper).
  add_executable(kokkosdnn_server_parsing_tests test_server_parsing.cpp ${KOKKOSDNN_SERVER_BACKEND_SOURCES})
  target_compile_definitions(kokkosdnn_server_parsing_tests PRIVATE ${KOKKOSDNN_SERVER_BACKEND_DEFS})
  target_link_libraries(kokkosdnn_server_parsing_tests PRIVATE
    ${KOKKOSDNN_SERVER_BACKEND_LIBS} KokkosDNN::common
    httplib::httplib nlohmann_json::nlohmann_json)
  set_target_properties(kokkosdnn_server_parsing_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(kokkosdnn_server_parsing_tests)
  add_test(NAME kokkosdnn_server_parsing
    COMMAND kokkosdnn_server_parsing_tests)
endif()

install(TARGETS kokkosdnn_server EXPORT KokkosDNNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
