# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# KokkosDNN shared CMake helpers.
#
# Centralizes the per-target boilerplate that was previously repeated in
# every CMakeLists.txt (add_executable/add_library + CXX_STANDARD 20 +
# kokkosdnn_enable_compiler_warnings), the model-library shapes, and the
# TPL option/target lists. Include once from the top-level CMakeLists.txt
# before any bridging calls:
#   include(cmake/KokkosDNNHelpers.cmake)
#
# NOTE: this module does NOT include KokkosDNNCompilerWarnings.cmake
# itself. The warning helpers only need to be defined by the time the
# functions below are *called* (all call sites run after the top-level
# include of KokkosDNNCompilerWarnings.cmake), and the warning probe needs
# Kokkos_ENABLE_CUDA/HIP, which is only known after find_package(Kokkos).
# Keeping the include at its current top-level position preserves that.
include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# TPL option suffixes: the mixed-case options (KokkosDNN_ENABLE_*) and the
# ALL_CAPS config-header macros (KOKKOSDNN_ENABLE_*) share these suffixes.
# Used by kokkosdnn_bridge_tpl_options() below.
# ---------------------------------------------------------------------------
set(KOKKOSDNN_TPL_OPTION_SUFFIXES
  CUBLAS
  CUDNN
  HIPBLAS
  MIOPEN
  ACCELERATE
  IMAGE
  IMAGE_WEBP
)

# Third-party targets owned by each TPL option. Missing targets are a no-op
# for kokkosdnn_suppress_third_party_warnings(), so every spelling/sibling
# is listed unconditionally.
set(KOKKOSDNN_TPL_CUBLAS_TARGETS CUDA::cublas CUDA::cublasLt)
set(KOKKOSDNN_TPL_CUDNN_TARGETS cudnn_frontend CUDNN::cudnn CUDA::nvrtc CUDA::cudart)
set(KOKKOSDNN_TPL_HIPBLAS_TARGETS roc::hipblas)
set(KOKKOSDNN_TPL_MIOPEN_TARGETS MIOpen)
set(KOKKOSDNN_TPL_GTEST_TARGETS
  gtest gtest_main gmock gmock_main
  GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
set(KOKKOSDNN_TPL_BENCHMARK_TARGETS
  benchmark benchmark_main benchmark::benchmark benchmark::benchmark_main)

# ---------------------------------------------------------------------------
# Promote the ALL_CAPS -D spelling (a common mistake given the config-header
# macro spelling) to the canonical mixed-case option. Macro (not function)
# so the set() calls land in caller scope without PARENT_SCOPE plumbing.
# Must run before the find_package calls that consume the options.
# ---------------------------------------------------------------------------
macro(kokkosdnn_bridge_tpl_options)
  foreach(_suffix IN LISTS KOKKOSDNN_TPL_OPTION_SUFFIXES)
    if(DEFINED KOKKOSDNN_ENABLE_${_suffix} AND NOT KokkosDNN_ENABLE_${_suffix})
      set(KokkosDNN_ENABLE_${_suffix} ON)
    endif()
  endforeach()
  unset(_suffix)
endmacro()

# ---------------------------------------------------------------------------
# Project policy: warnings are enforced on first-party code only. Silence
# every third-party target owned by the enabled TPL options so their headers
# cannot trip -Werror in our TUs. Single call replaces the per-option
# if()/foreach() blocks at the top level.
# ---------------------------------------------------------------------------
function(kokkosdnn_suppress_tpl_warnings)
  if(KokkosDNN_ENABLE_CUBLAS)
    foreach(_t IN LISTS KOKKOSDNN_TPL_CUBLAS_TARGETS)
      kokkosdnn_suppress_third_party_warnings(${_t})
    endforeach()
  endif()
  if(KokkosDNN_ENABLE_CUDNN)
    foreach(_t IN LISTS KOKKOSDNN_TPL_CUDNN_TARGETS)
      kokkosdnn_suppress_third_party_warnings(${_t})
    endforeach()
  endif()
  if(KokkosDNN_ENABLE_HIPBLAS)
    foreach(_t IN LISTS KOKKOSDNN_TPL_HIPBLAS_TARGETS)
      kokkosdnn_suppress_third_party_warnings(${_t})
    endforeach()
  endif()
  if(KokkosDNN_ENABLE_MIOPEN)
    foreach(_t IN LISTS KOKKOSDNN_TPL_MIOPEN_TARGETS)
      kokkosdnn_suppress_third_party_warnings(${_t})
    endforeach()
  endif()
  unset(_t)
endfunction()

# ---------------------------------------------------------------------------
# Apply the project-standard language/warning policy to a first-party
# target: C++20 plus kokkosdnn_enable_compiler_warnings(). INTERFACE
# libraries cannot carry the CXX_STANDARD property, so they get the
# cxx_std_20 compile feature instead (same bar, propagates to consumers).
# ---------------------------------------------------------------------------
function(kokkosdnn_standardize_target target)
  if(NOT TARGET ${target})
    message(WARNING "kokkosdnn_standardize_target: no such target '${target}'")
    return()
  endif()
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    target_compile_features(${target} INTERFACE cxx_std_20)
  else()
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  endif()
  kokkosdnn_enable_compiler_warnings(${target})
  unset(_type)
endfunction()

# ---------------------------------------------------------------------------
# add_executable + project-standard properties. Link libraries, compile
# definitions, and test discovery stay at the call site (they vary per
# target); this only removes the repeated
# set_target_properties(CXX_STANDARD 20) + warnings pair.
#   kokkosdnn_add_executable(<name> <sources...>)
# ---------------------------------------------------------------------------
function(kokkosdnn_add_executable name)
  add_executable(${name} ${ARGN})
  kokkosdnn_standardize_target(${name})
endfunction()

# ---------------------------------------------------------------------------
# Wrapper around gtest_discover_tests that no-ops when discovery is
# disabled (build-only GPU runners). Defined here so tests/, examples/ and
# benchmarks/ can all use it; the call is deferred to the caller's
# configure time, by which point include(GoogleTest) has run whenever
# KokkosDNN_ENABLE_TESTS=ON.
# ---------------------------------------------------------------------------
function(kokkosdnn_discover_tests target)
  if(KokkosDNN_ENABLE_TEST_DISCOVERY)
    # DISCOVERY_TIMEOUT: discovery executes the test binary at build time.
    # The default 5 s is too tight on loaded CI runners (parallel links
    # starve the probe; macos-arm failed on test_lfm25_weights_gguf with
    # "Process terminated due to timeout"), so allow 60 s. Belt-and-braces
    # with GTestMain skipping Kokkos init for --gtest_list_tests probes.
    gtest_discover_tests(${target} DISCOVERY_TIMEOUT 60 ${ARGN})
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Unit test: executable + default core links + standard properties +
# discovery. The sources are the bare positional args (one or more .cpp
# files); extra link libraries and per-target include dirs/definitions go
# through keywords:
#   kokkosdnn_add_test(<name> <src...> [LINK_LIBRARIES a;b...]
#     [INCLUDE_DIRECTORIES d...] [COMPILE_DEFINITIONS d...])
# ---------------------------------------------------------------------------
function(kokkosdnn_add_test NAME)
  cmake_parse_arguments(PARSE_ARGV 1 _ARG "" "" "SOURCES;LINK_LIBRARIES;INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS")
  set(_srcs ${_ARG_SOURCES} ${_ARG_UNPARSED_ARGUMENTS})
  if(NOT _srcs)
    message(FATAL_ERROR "kokkosdnn_add_test(${NAME}): no sources given")
  endif()
  add_executable(${NAME} ${_srcs})
  target_link_libraries(${NAME} PRIVATE
    KokkosDNN::kokkosdnn KokkosDNN::gtest_main
    ${_ARG_LINK_LIBRARIES})
  if(_ARG_INCLUDE_DIRECTORIES)
    target_include_directories(${NAME} PRIVATE ${_ARG_INCLUDE_DIRECTORIES})
  endif()
  if(_ARG_COMPILE_DEFINITIONS)
    target_compile_definitions(${NAME} PRIVATE ${_ARG_COMPILE_DEFINITIONS})
  endif()
  kokkosdnn_standardize_target(${NAME})
  kokkosdnn_discover_tests(${NAME})
endfunction()

# ---------------------------------------------------------------------------
# Compiled (STATIC) model library: installable component with public
# headers in its source dir.
#   kokkosdnn_add_compiled_model_library(<target>
#     EXPORT_NAME <name> SOURCES <srcs...> [LINK_LIBRARIES a;b...]
#     [WITH_IMAGE] [NO_INSTALL])
# WITH_IMAGE additionally links KokkosDNN::image when
# KokkosDNN_ENABLE_IMAGE=ON (image-backed example/model libs such as sana
# and dinov3).
# ---------------------------------------------------------------------------
function(kokkosdnn_add_compiled_model_library TARGET)
  cmake_parse_arguments(PARSE_ARGV 1 _ARG "NO_INSTALL;WITH_IMAGE" "EXPORT_NAME" "SOURCES;LINK_LIBRARIES")
  if(NOT _ARG_EXPORT_NAME)
    message(FATAL_ERROR "kokkosdnn_add_compiled_model_library(${TARGET}): EXPORT_NAME required")
  endif()
  if(NOT _ARG_SOURCES)
    message(FATAL_ERROR "kokkosdnn_add_compiled_model_library(${TARGET}): SOURCES required")
  endif()
  add_library(${TARGET} STATIC ${_ARG_SOURCES})
  add_library(KokkosDNN::${_ARG_EXPORT_NAME} ALIAS ${TARGET})
  target_link_libraries(${TARGET} PUBLIC ${_ARG_LINK_LIBRARIES})
  target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  if(_ARG_WITH_IMAGE AND KokkosDNN_ENABLE_IMAGE)
    target_link_libraries(${TARGET} PUBLIC KokkosDNN::image)
  endif()
  kokkosdnn_standardize_target(${TARGET})
  set_target_properties(${TARGET} PROPERTIES EXPORT_NAME ${_ARG_EXPORT_NAME})
  if(NOT _ARG_NO_INSTALL)
    install(TARGETS ${TARGET} EXPORT KokkosDNNTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Header-only (INTERFACE) model library.
#   kokkosdnn_add_header_only_model_library(<target>
#     EXPORT_NAME <name> [LINK_LIBRARIES a;b...] [WITH_IMAGE] [NO_INSTALL])
# WITH_IMAGE additionally links KokkosDNN::image when
# KokkosDNN_ENABLE_IMAGE=ON (the image-backed pipeline headers). Header
# install rules stay at the call site (FILES list vs DIRECTORY globs).
# ---------------------------------------------------------------------------
function(kokkosdnn_add_header_only_model_library TARGET)
  cmake_parse_arguments(PARSE_ARGV 1 _ARG "WITH_IMAGE;NO_INSTALL" "EXPORT_NAME" "LINK_LIBRARIES")
  if(NOT _ARG_EXPORT_NAME)
    message(FATAL_ERROR "kokkosdnn_add_header_only_model_library(${TARGET}): EXPORT_NAME required")
  endif()
  add_library(${TARGET} INTERFACE)
  add_library(KokkosDNN::${_ARG_EXPORT_NAME} ALIAS ${TARGET})
  target_link_libraries(${TARGET} INTERFACE ${_ARG_LINK_LIBRARIES})
  target_include_directories(${TARGET} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  if(_ARG_WITH_IMAGE AND KokkosDNN_ENABLE_IMAGE)
    target_link_libraries(${TARGET} INTERFACE KokkosDNN::image)
  endif()
  kokkosdnn_standardize_target(${TARGET})
  set_target_properties(${TARGET} PROPERTIES EXPORT_NAME ${_ARG_EXPORT_NAME})
  if(NOT _ARG_NO_INSTALL)
    install(TARGETS ${TARGET} EXPORT KokkosDNNTargets)
  endif()
endfunction()
