cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(KokkosDNN VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

# ---------------------------------------------------------------------------
# Options (kokkos-kernels style: KokkosDNN_ENABLE_*)
# ---------------------------------------------------------------------------
option(KokkosDNN_ENABLE_TESTS "Build unit tests" ON)
option(KokkosDNN_ENABLE_EXAMPLES "Build examples" ON)
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
endif()

# ---------------------------------------------------------------------------
# Generated config header
# ---------------------------------------------------------------------------
configure_file(cmake/KokkosDNN_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosDNN_config.h @ONLY)

# ---------------------------------------------------------------------------
# Library: core DNN operators only (no model-specific code) + headers
# ---------------------------------------------------------------------------
add_library(kokkosdnn
  src/KokkosDNN_PrintConfiguration.cpp
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
  target_compile_definitions(kokkosdnn PUBLIC KOKKOSDNN_ENABLE_CUBLAS)
endif()
if(KokkosDNN_ENABLE_CUDNN)
  if(TARGET CUDNN::cudnn)
    target_link_libraries(kokkosdnn PUBLIC CUDNN::cudnn)
  elseif(CUDNN_LIBRARY)
    target_include_directories(kokkosdnn PUBLIC ${CUDNN_INCLUDE_DIR})
    target_link_libraries(kokkosdnn PUBLIC ${CUDNN_LIBRARY})
  endif()
  target_compile_definitions(kokkosdnn PUBLIC KOKKOSDNN_ENABLE_CUDNN)
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
  add_subdirectory(tests)
endif()
if(KokkosDNN_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
