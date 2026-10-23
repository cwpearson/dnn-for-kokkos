cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(KokkosDNN VERSION 0.1.0 LANGUAGES CXX)

# Kokkos 5.x requires C++20. Enforce it globally so every target in this
# build (library, tests, examples, benchmarks) compiles in C++20 mode
# regardless of toolchain defaults (e.g. hipcc, nvcc_wrapper).
# Per-target CXX_STANDARD properties below are kept as belt-and-braces.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

# ---------------------------------------------------------------------------
# Options (kokkos-kernels style: KokkosDNN_ENABLE_*)
# ---------------------------------------------------------------------------
option(KokkosDNN_ENABLE_TESTS "Build unit tests" ON)
option(KokkosDNN_ENABLE_EXAMPLES "Build examples" ON)
option(KokkosDNN_ENABLE_BENCHMARKS "Build benchmarks (Google Benchmark)" OFF)
option(KokkosDNN_ENABLE_CUBLAS "Dispatch GEMM to cuBLAS when ExecSpace is CUDA" OFF)
option(KokkosDNN_ENABLE_CUDNN "Dispatch attention to cuDNN SDPA when ExecSpace is CUDA" OFF)
option(KokkosDNN_ENABLE_HIPBLAS "Dispatch GEMM to hipBLAS when ExecSpace is HIP" OFF)
option(KokkosDNN_ENABLE_MIOPEN "Dispatch conv/attention-helpers to MIOpen when ExecSpace is HIP" OFF)
option(KokkosDNN_ENABLE_ACCELERATE "Dispatch GEMM to Apple Accelerate (vecLib cblas_sgemm) on host execution spaces" OFF)
option(KokkosDNN_ENABLE_IMAGE "Build optional image I/O component (stb-based decode, JPEG/PNG/BMP)" OFF)
option(KokkosDNN_ENABLE_MODEL_LFM25 "Build LFM2.5 model component (KokkosDNN::lfm25)" ON)
option(KokkosDNN_ENABLE_MODEL_MOSS "Build MOSS-Image model component (KokkosDNN::moss)" ON)
# Server is a stub (binds no port, serves no routes yet), so it stays OFF by
# default to keep existing CI builds unaffected. -DKokkosDNN_ENABLE_SERVER=ON
# to build kokkosdnn_server (requires KokkosDNN_ENABLE_MODEL_LFM25=ON).
option(KokkosDNN_ENABLE_SERVER "Build OpenAI-compatible HTTP server stub" OFF)
option(KokkosDNN_ENABLE_COMPILER_WARNINGS "Enable compiler warnings (as errors, mirroring Kokkos_ENABLE_COMPILER_WARNINGS)" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
# gtest_discover_tests executes each test binary at build time. On GPU-less
# build-only CI (CUDA/ROCm runners have no GPU) the binaries abort inside
# Kokkos::initialize, failing the build. Pass
# -DKokkosDNN_ENABLE_TEST_DISCOVERY=OFF there; ctest then simply sees no
# tests, which matches the compile-coverage-only intent.
option(KokkosDNN_ENABLE_TEST_DISCOVERY "Register gtest tests via gtest_discover_tests (requires runnable test binaries)" ON)
# Wrapper around gtest_discover_tests that no-ops when discovery is
# disabled. Defined here (before any add_subdirectory) so tests/, examples/
# and benchmarks/ can all use it. The call is deferred to configure time of
# the caller, by which point include(GoogleTest) has run whenever
# KokkosDNN_ENABLE_TESTS=ON.
function(kokkosdnn_discover_tests TARGET)
  if(KokkosDNN_ENABLE_TEST_DISCOVERY)
    gtest_discover_tests(${TARGET} ${ARGN})
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Options are mixed-case (KokkosDNN_ENABLE_*) but the config header and
# compile definitions use ALL_CAPS (KOKKOSDNN_ENABLE_*). CMake variables are
# case-sensitive, so #cmakedefine KOKKOSDNN_ENABLE_* would otherwise always
# see an undefined variable and emit /* #undef ... */ even when the option
# is ON. Bridge explicitly. Also accept the all-caps -D spelling (a common
# mistake given the macro name) by promoting it to the canonical option.
# NOTE: this must run before the find_package calls below, so the promoted
# option value is what the dependency search sees.
# ---------------------------------------------------------------------------
if(KOKKOSDNN_ENABLE_CUBLAS AND NOT KokkosDNN_ENABLE_CUBLAS)
  set(KokkosDNN_ENABLE_CUBLAS ON)
endif()
if(KOKKOSDNN_ENABLE_CUDNN AND NOT KokkosDNN_ENABLE_CUDNN)
  set(KokkosDNN_ENABLE_CUDNN ON)
endif()
if(KOKKOSDNN_ENABLE_HIPBLAS AND NOT KokkosDNN_ENABLE_HIPBLAS)
  set(KokkosDNN_ENABLE_HIPBLAS ON)
endif()
if(KOKKOSDNN_ENABLE_MIOPEN AND NOT KokkosDNN_ENABLE_MIOPEN)
  set(KokkosDNN_ENABLE_MIOPEN ON)
endif()
if(KOKKOSDNN_ENABLE_ACCELERATE AND NOT KokkosDNN_ENABLE_ACCELERATE)
  set(KokkosDNN_ENABLE_ACCELERATE ON)
endif()
if(KOKKOSDNN_ENABLE_IMAGE AND NOT KokkosDNN_ENABLE_IMAGE)
  set(KokkosDNN_ENABLE_IMAGE ON)
endif()

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
  # src/tpl/cudnn/KokkosDNN_CuDNN_SDPA.cpp. Pinned to a commit tested against
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

if(KokkosDNN_ENABLE_HIPBLAS)
  # hipblas ships hipblas-config.cmake under ${ROCM_PATH}/lib/cmake/hipblas.
  # ROCM_PATH=/opt/rocm/core-7.14 is in the environment on our ROCm runners;
  # add it to the search prefix so a bare find_package works without -D flags.
  if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
  endif()
  find_package(hipblas REQUIRED)
endif()

if(KokkosDNN_ENABLE_MIOPEN)
  # MIOpen ships miopen-config.cmake in recent releases; fall back to manual find.
  find_package(miopen QUIET)
  if(NOT miopen_FOUND)
    if(DEFINED ENV{ROCM_PATH})
      list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
    endif()
    find_path(MIOPEN_INCLUDE_DIR miopen/miopen.h
      HINTS ${MIOPEN_ROOT} $ENV{MIOPEN_ROOT} $ENV{ROCM_PATH}/include /opt/rocm/include)
    find_library(MIOPEN_LIBRARY MIOpen
      HINTS ${MIOPEN_ROOT} $ENV{MIOPEN_ROOT} $ENV{ROCM_PATH}/lib /opt/rocm/lib)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(MIOpen DEFAULT_MSG MIOPEN_LIBRARY MIOPEN_INCLUDE_DIR)
  endif()
endif()

# ---------------------------------------------------------------------------
# Generated config header
# ---------------------------------------------------------------------------
# KOKKOSDNN_ENABLE_* spelling bridged to the canonical options above;
# here we just copy the values to the ALL_CAPS names configure_file sees.
set(KOKKOSDNN_ENABLE_CUBLAS ${KokkosDNN_ENABLE_CUBLAS})
set(KOKKOSDNN_ENABLE_CUDNN ${KokkosDNN_ENABLE_CUDNN})
set(KOKKOSDNN_ENABLE_HIPBLAS ${KokkosDNN_ENABLE_HIPBLAS})
set(KOKKOSDNN_ENABLE_MIOPEN ${KokkosDNN_ENABLE_MIOPEN})
set(KOKKOSDNN_ENABLE_ACCELERATE ${KokkosDNN_ENABLE_ACCELERATE})
set(KOKKOSDNN_ENABLE_IMAGE ${KokkosDNN_ENABLE_IMAGE})
configure_file(cmake/KokkosDNN_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNN_config.h @ONLY)

# ---------------------------------------------------------------------------
# Library: core DNN operators only (no model-specific code) + headers
# ---------------------------------------------------------------------------
add_library(kokkosdnn
  src/KokkosDNN_PrintConfiguration.cpp
  src/tpl/cudnn/KokkosDNN_CuDNN_SDPA.cpp
  src/tpl/cudnn/KokkosDNN_CuDNN_ConvForward.cpp
  src/tpl/miopen/KokkosDNN_MIOpen_SDPA.cpp
  src/tpl/miopen/KokkosDNN_MIOpen_ConvForward.cpp
)
add_library(KokkosDNN::kokkosdnn ALIAS kokkosdnn)

target_include_directories(kokkosdnn PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(kokkosdnn PUBLIC Kokkos::kokkos)

if(KokkosDNN_ENABLE_CUBLAS)
  # cublasLt backs the fused GEMM+bias epilogue in gemm_bias/gemv_bias
  # (src/tpl/cublas/KokkosDNN_CuBLAS_GemmBiasLt.hpp).
  target_link_libraries(kokkosdnn PUBLIC CUDA::cublas CUDA::cublasLt)
endif()
if(KokkosDNN_ENABLE_CUDNN)
  if(TARGET CUDNN::cudnn)
    target_link_libraries(kokkosdnn PUBLIC CUDNN::cudnn)
  elseif(CUDNN_LIBRARY)
    target_include_directories(kokkosdnn SYSTEM PUBLIC ${CUDNN_INCLUDE_DIR})
    target_link_libraries(kokkosdnn PUBLIC ${CUDNN_LIBRARY})
  endif()
  target_link_libraries(kokkosdnn PUBLIC cudnn_frontend)
  # The frontend's experimental engines call NVRTC; link it so consumers
  # of libkokkosdnn (tests, examples, benchmarks) resolve nvrtc* symbols.
  target_link_libraries(kokkosdnn PUBLIC CUDA::nvrtc CUDA::cudart)
endif()

if(KokkosDNN_ENABLE_HIPBLAS)
  target_link_libraries(kokkosdnn PUBLIC roc::hipblas)
endif()
if(KokkosDNN_ENABLE_MIOPEN)
  if(TARGET MIOpen)
    target_link_libraries(kokkosdnn PUBLIC MIOpen)
  elseif(MIOPEN_LIBRARY)
    target_include_directories(kokkosdnn SYSTEM PUBLIC ${MIOPEN_INCLUDE_DIR})
    target_link_libraries(kokkosdnn PUBLIC ${MIOPEN_LIBRARY})
  endif()
endif()
# Kokkos_ENABLE_COMPILER_WARNINGS only affects the Kokkos build itself (its
# flags are deliberately not exported downstream), so KokkosDNN carries its
# own equivalent: KokkosDNN_ENABLE_COMPILER_WARNINGS plus -Werror (/WX).
# The helper applies per-target PRIVATE/INTERFACE options so FetchContent
# third-party targets never inherit them.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/KokkosDNNCompilerWarnings.cmake)

# Third-party policy (warnings are enforced on first-party code only):
# silence every TPL target so its headers cannot trip -Werror in our TUs.
if(KokkosDNN_ENABLE_CUBLAS)
  kokkosdnn_suppress_third_party_warnings(CUDA::cublas)
  kokkosdnn_suppress_third_party_warnings(CUDA::cublasLt)
endif()
if(KokkosDNN_ENABLE_CUDNN)
  kokkosdnn_suppress_third_party_warnings(cudnn_frontend)
  kokkosdnn_suppress_third_party_warnings(CUDNN::cudnn)
  kokkosdnn_suppress_third_party_warnings(CUDA::nvrtc)
  kokkosdnn_suppress_third_party_warnings(CUDA::cudart)
endif()
if(KokkosDNN_ENABLE_HIPBLAS)
  kokkosdnn_suppress_third_party_warnings(roc::hipblas)
endif()
if(KokkosDNN_ENABLE_MIOPEN)
  kokkosdnn_suppress_third_party_warnings(MIOpen)
endif()

if(KokkosDNN_ENABLE_ACCELERATE)
  # Apple-only system framework (AMX/SME-backed sgemm, no install needed).
  # Fail loudly off-Apple so a stray -DKokkosDNN_ENABLE_ACCELERATE=ON on
  # Linux/Windows CI cannot silently build the native path instead.
  if(NOT APPLE)
    message(FATAL_ERROR "KokkosDNN_ENABLE_ACCELERATE requires Apple (Accelerate framework)")
  endif()
  find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
  target_link_libraries(kokkosdnn PUBLIC ${ACCELERATE_LIBRARY})
  # Select Accelerate's non-deprecated LAPACK 3.9.1+ CBLAS declarations (see
  # src/tpl/accelerate/KokkosDNN_Accelerate_GemmFloat.hpp). PUBLIC so every
  # consumer TU that reaches <Accelerate/...> sees it before the headers.
  target_compile_definitions(kokkosdnn PUBLIC ACCELERATE_NEW_LAPACK=1)
endif()

kokkosdnn_enable_compiler_warnings(kokkosdnn)

set_target_properties(kokkosdnn PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Optional image component (OFF by default; no dependency from core target)
# ---------------------------------------------------------------------------
if(KokkosDNN_ENABLE_IMAGE)
  add_subdirectory(src/images)
endif()

# ---------------------------------------------------------------------------
# GTest setup (must precede models/: models/*/tests register via
# kokkosdnn_discover_tests, which needs include(GoogleTest) to have run
# and KokkosDNN::gtest_main to exist).
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
  # Third-party policy: gtest builds warning-free (its own -w on every
  # compiled target) and its headers reach our test TUs as SYSTEM, so they
  # can never trip -Werror. Suppress all spellings/siblings: FetchContent
  # defines gtest+gtest_main (plus gmock+gmock_main when enabled), system
  # packages define GTest::gtest+GTest::gtest_main as IMPORTED (headers
  # still need SYSTEM treatment). Missing targets are a no-op.
  kokkosdnn_suppress_third_party_warnings(gtest)
  kokkosdnn_suppress_third_party_warnings(gtest_main)
  kokkosdnn_suppress_third_party_warnings(gmock)
  kokkosdnn_suppress_third_party_warnings(gmock_main)
  kokkosdnn_suppress_third_party_warnings(GTest::gtest)
  kokkosdnn_suppress_third_party_warnings(GTest::gtest_main)
  kokkosdnn_suppress_third_party_warnings(GTest::gmock)
  kokkosdnn_suppress_third_party_warnings(GTest::gmock_main)
  include(GoogleTest)
  # Shared GoogleTest main that initializes/finalizes Kokkos.
  add_library(kokkosdnn_gtest_main STATIC tests/GTestMain.cpp)
  add_library(KokkosDNN::gtest_main ALIAS kokkosdnn_gtest_main)
  target_link_libraries(kokkosdnn_gtest_main PUBLIC Kokkos::kokkos GTest::gtest)
  set_target_properties(kokkosdnn_gtest_main PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  kokkosdnn_enable_compiler_warnings(kokkosdnn_gtest_main)
endif()

# ---------------------------------------------------------------------------
# Model components (installable per-model libs + shared support).
# Must precede examples/ so thin drivers can link KokkosDNN::lfm25 /
# KokkosDNN::moss. Each model dir installs its own headers (see
# models/<name>/CMakeLists.txt); the core install() below covers src/ only.
# ---------------------------------------------------------------------------
add_subdirectory(models)
if(KokkosDNN_ENABLE_SERVER)
  add_subdirectory(server)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS kokkosdnn EXPORT KokkosDNNTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNN_config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT KokkosDNNTargets
  NAMESPACE KokkosDNN::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosDNN)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNNConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
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
  add_subdirectory(tests)
endif()
if(KokkosDNN_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
if(KokkosDNN_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
