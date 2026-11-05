# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
cmake_minimum_required(VERSION 3.22)
project(graph_spike CXX CUDA)
# Standalone prototype: not referenced by the top-level CMakeLists.txt.
# Point Kokkos_ROOT (or CMAKE_PREFIX_PATH) at a CUDA Kokkos install.
find_package(Kokkos REQUIRED)
add_executable(graph_spike graph_spike.cpp)
target_link_libraries(graph_spike Kokkos::kokkos)
