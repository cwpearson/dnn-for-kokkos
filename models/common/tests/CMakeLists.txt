# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# models/common/tests: unit tests for the header-only common utilities.
# Mirrors the examples/*/test pattern: link KokkosDNN::gtest_main (Kokkos
# init/finalize + GTest main) and register via kokkosdnn_discover_tests.

add_executable(test_common_detokenizer Test_Detokenizer.cpp)
target_link_libraries(test_common_detokenizer PRIVATE KokkosDNN::common KokkosDNN::gtest_main)
set_target_properties(test_common_detokenizer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(test_common_detokenizer)
kokkosdnn_discover_tests(test_common_detokenizer)

add_executable(test_common_usage Test_Usage.cpp)
target_link_libraries(test_common_usage PRIVATE KokkosDNN::common KokkosDNN::gtest_main)
set_target_properties(test_common_usage PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
kokkosdnn_enable_compiler_warnings(test_common_usage)
kokkosdnn_discover_tests(test_common_usage)
