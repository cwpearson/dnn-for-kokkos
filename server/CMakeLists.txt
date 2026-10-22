# kokkosdnn_server: OpenAI-compatible HTTP front-end (stub).
# Thin binary: owns Kokkos::initialize/finalize, parses --help/--version,
# prints its configuration, notes the REST routes as TODO. All inference,
# sampling, and chat logic belongs in models/* — the server must never grow
# model code of its own. Requires the LFM2.5 model component (stub serves
# LFM25); HTTP/JSON dependencies arrive when the stub grows routes (see
# the pinned-but-commented FetchContent block below).

if(NOT KokkosDNN_ENABLE_MODEL_LFM25)
  message(FATAL_ERROR
    "KokkosDNN_ENABLE_SERVER requires KokkosDNN_ENABLE_MODEL_LFM25=ON"
    " (the server stub serves LFM2.5).")
endif()

add_executable(kokkosdnn_server server_main.cpp)
target_link_libraries(kokkosdnn_server PRIVATE
  KokkosDNN::lfm25 KokkosDNN::common)
set_target_properties(kokkosdnn_server PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

install(TARGETS kokkosdnn_server EXPORT KokkosDNNTragets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---------------------------------------------------------------------------
# TODO (when the stub grows real routes): HTTP + JSON front-end deps.
# Pins: cpp-httplib v0.15.3, nlohmann/json v3.11.3. Kept commented so
# configure never touches the network while the server is a stub; uncomment
# together with the first route implementation. Offline override then via
# -DFETCHCONTENT_SOURCE_DIR_CPP_HTTPLIB=<path> /
# -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=<path>.
# ---------------------------------------------------------------------------
# include(FetchContent)
# FetchContent_Declare(
#   cpp_httplib
#   GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
#   GIT_TAG v0.15.3
# )
# FetchContent_Declare(
#   nlohmann_json
#   GIT_REPOSITORY https://github.com/nlohmann/json.git
#   GIT_TAG v3.11.3
# )
# FetchContent_MakeAvailable(cpp_httplib nlohmann_json)
# target_link_libraries(kokkosdnn_server PRIVATE httplib::httplib nlohmann_json::nlohmann_json)
