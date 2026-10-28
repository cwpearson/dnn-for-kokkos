# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Standalone consumer of an INSTALLED KokkosDNN. This is a separate CMake
# project (configured with `cmake -S tests/consume`, never via
# add_subdirectory): it verifies find_package(KokkosDNN), every installed
# linkable target, and one header per installed component. Wired into CI
# (serial, CUDA, ROCm) after `cmake --install`.
#
# Regression coverage for the installed-consumer fixes: the missing
# cudnn_frontend IMPORTED target, PACKAGE_PREFIX_DIR clobbering across
# find_dependency on CMake <= 3.29, and the non-exported stb::stb target.
# CMake fails the generate step when a linked target is missing from the
# export or carries a non-existent INTERFACE_INCLUDE_DIRECTORIES (exactly
# those failure modes), so a successful configure+build IS the test. The
# image component is optional: installed moss/qwen_image_21 link
# KokkosDNN::image only when the package was built with
# KokkosDNN_ENABLE_IMAGE=ON.
cmake_minimum_required(VERSION 3.22)
project(KokkosDNNConsume CXX)

# No explicit find_package(Kokkos): KokkosDNNConfig.cmake pulls Kokkos via
# find_dependency, so this also verifies the transitive dependency path a
# minimal downstream project takes. Pass -DKokkos_ROOT=<kokkos-install>.
find_package(KokkosDNN REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE
  KokkosDNN::kokkosdnn
  KokkosDNN::common
  KokkosDNN::lfm25
  KokkosDNN::moss
  KokkosDNN::qwen_image_21)
if(TARGET KokkosDNN::image)
  target_link_libraries(consumer PRIVATE KokkosDNN::image)
  target_compile_definitions(consumer PRIVATE KOKKOSDNN_CONSUME_HAS_IMAGE)
endif()
set_target_properties(consumer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
