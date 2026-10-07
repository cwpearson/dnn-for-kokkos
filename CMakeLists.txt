cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(KokkosDNN VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

# ---------------------------------------------------------------------------
# Options (kokkos-kernels style: KokkosDNN_ENABLE_*)
# ---------------------------------------------------------------------------
option(KokkosDNN_ENABLE_TESTS "Build unit tests" ON)
option(KokkosDNN_ENABLE_EXAMPLES "Build examples" ON)
option(KokkosDNN_ENABLE_BENCHMARKS "Build benchmarks (Google Benchmark)" OFF)
option(KokkosDNN_ENABLE_CUBLAS "Dispatch GEMM to cuBLAS when ExecSpace is CUDA" OFF)
option(KokkosDNN_ENABLE_CUDNN "Dispatch attention to cuDNN SDPA when ExecSpace is CUDA" OFF)
option(KokkosDNN_ENABLE_COMPILER_WARNINGS "Enable compiler warnings" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Kokkos REQUIRED)
if(Kokkos_COMPILE_LANGUAGE)
  enable_language(${Kokkos_COMPILE_LANGUAGE})
  if(Kokkos_COMPILE_LANGUAGE STREQUAL CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})
  endif()
endif()

if(KokkosDNN_ENABLE_CUBLAS)
  find_package(CUDAToolkit REQUIRED)
endif()

if(KokkosDNN_ENABLE_CUDNN)
  # cuDNN ships a CUDNNConfig.cmake in recent releases; fall back to manual find.
  find_package(CUDNN QUIET)
  if(NOT CUDNN_FOUND)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
      HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} /usr/local/cuda/include /usr/include)
    find_library(CUDNN_LIBRARY cudnn
      HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY CUDNN_INCLUDE_DIR)
  endif()
  # cudnn-frontend (header-only graph API) builds the SDPA node used by
  # src/KokkosDNN_Attention_CuDNN.cpp. Pinned to a commit tested against
  # cuDNN 9.x; override offline with -DFETCHCONTENT_SOURCE_DIR_CUDNN_FRONTEND=<path>.
  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    cudnn_frontend
    GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
    GIT_TAG b8e39555bbcc64b030dc7ea8d8f01c43ff88f483
  )
  FetchContent_MakeAvailable(cudnn_frontend)
  # The frontend's experimental engines need NVRTC at link time.
  find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Generated config header
# ---------------------------------------------------------------------------
# Options are mixed-case (KokkosDNN_ENABLE_*) but the config header and
# compile definitions use ALL_CAPS (KOKKOSDNN_ENABLE_*). CMake variables are
# case-sensitive, so #cmakedefine KOKKOSDNN_ENABLE_* would otherwise always
# see an undefined variable and emit /* #undef ... */ even when the option
# is ON. Bridge explicitly. Also accept the all-caps -D spelling (a common
# mistake given the macro name) by promoting it to the canonical option.
if(KOKKOSDNN_ENABLE_CUBLAS AND NOT KokkosDNN_ENABLE_CUBLAS)
  set(KokkosDNN_ENABLE_CUBLAS ON)
endif()
if(KOKKOSDNN_ENABLE_CUDNN AND NOT KokkosDNN_ENABLE_CUDNN)
  set(KokkosDNN_ENABLE_CUDNN ON)
endif()
set(KOKKOSDNN_ENABLE_CUBLAS ${KokkosDNN_ENABLE_CUBLAS})
set(KOKKOSDNN_ENABLE_CUDNN ${KokkosDNN_ENABLE_CUDNN})
configure_file(cmake/KokkosDNN_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNN_config.h @ONLY)

# ---------------------------------------------------------------------------
# Library: core DNN operators only (no model-specific code) + headers
# ---------------------------------------------------------------------------
add_library(kokkosdnn
  src/KokkosDNN_PrintConfiguration.cpp
  src/KokkosDNN_Attention_CuDNN.cpp
)
add_library(KokkosDNN::kokkosdnn ALIAS kokkosdnn)

target_include_directories(kokkosdnn PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(kokkosdnn PUBLIC Kokkos::kokkos)

if(KokkosDNN_ENABLE_CUBLAS)
  target_link_libraries(kokkosdnn PUBLIC CUDA::cublas)
endif()
if(KokkosDNN_ENABLE_CUDNN)
  if(TARGET CUDNN::cudnn)
    target_link_libraries(kokkosdnn PUBLIC CUDNN::cudnn)
  elseif(CUDNN_LIBRARY)
    target_include_directories(kokkosdnn PUBLIC ${CUDNN_INCLUDE_DIR})
    target_link_libraries(kokkosdnn PUBLIC ${CUDNN_LIBRARY})
  endif()
  target_link_libraries(kokkosdnn PUBLIC cudnn_frontend)
  # The frontend's experimental engines call NVRTC; link it so consumers
  # of libkokkosdnn (tests, examples, benchmarks) resolve nvrtc* symbols.
  target_link_libraries(kokkosdnn PUBLIC CUDA::nvrtc CUDA::cudart)
endif()

if(KokkosDNN_ENABLE_COMPILER_WARNINGS)
  target_compile_options(kokkosdnn PRIVATE -Wall -Wextra -Werror)
endif()

set_target_properties(kokkosdnn PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS kokkosdnn EXPORT KokkosDNNTragets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNN_config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT KokkosDNNTragets
  NAMESPACE KokkosDNN::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosDNN)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNNConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
configure_file(cmake/KokkosDNNConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNNConfig.cmake @ONLY)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNNConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNNConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosDNN)

# ---------------------------------------------------------------------------
# Tests / examples
# ---------------------------------------------------------------------------
if(KokkosDNN_ENABLE_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    # CMP0135 (CMake >= 3.24): use extraction time instead of archive
    # timestamps for FetchContent URLs, so URL changes trigger a rebuild.
    # Guarded for our cmake_minimum_required(VERSION 3.22) floor.
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/v1.14.0.zip
    )
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)
  # Shared GoogleTest main that initializes/finalizes Kokkos.
  add_library(kokkosdnn_gtest_main STATIC tests/GTestMain.cpp)
  add_library(KokkosDNN::gtest_main ALIAS kokkosdnn_gtest_main)
  target_link_libraries(kokkosdnn_gtest_main PUBLIC Kokkos::kokkos GTest::gtest)
  set_target_properties(kokkosdnn_gtest_main PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  add_subdirectory(tests)
endif()
if(KokkosDNN_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
if(KokkosDNN_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
