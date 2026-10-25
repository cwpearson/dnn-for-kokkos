// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Shared GoogleTest main: initializes Kokkos around RUN_ALL_TESTS.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // gtest_discover_tests probes the binary with --gtest_list_tests at build
  // time. Skip Kokkos init/finalize for that probe so discovery stays fast
  // (its 5 s default timeout fired on loaded CI runners doing parallel
  // links) and never requires a functional Kokkos backend.
  const bool list_only = ::testing::GTEST_FLAG(list_tests);
  if (!list_only) {
    Kokkos::initialize(argc, argv);
  }
  const int result = RUN_ALL_TESTS();
  if (!list_only) {
    Kokkos::finalize();
  }
  return result;
}
