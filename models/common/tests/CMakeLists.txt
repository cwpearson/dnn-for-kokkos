# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# models/common/tests: unit tests for the header-only common utilities.
# Mirrors the examples/*/test pattern: link KokkosDNN::gtest_main (Kokkos
# init/finalize + GTest main) and register via kokkosdnn_discover_tests
# (both wrapped by kokkosdnn_add_test in cmake/KokkosDNNHelpers.cmake).

kokkosdnn_add_test(test_common_detokenizer Test_Detokenizer.cpp
  LINK_LIBRARIES KokkosDNN::common)
kokkosdnn_add_test(test_common_usage Test_Usage.cpp
  LINK_LIBRARIES KokkosDNN::common)
kokkosdnn_add_test(test_common_generate Test_Generate.cpp
  LINK_LIBRARIES KokkosDNN::common)
kokkosdnn_add_test(test_common_session_types Test_SessionTypes.cpp
  LINK_LIBRARIES KokkosDNN::common)
kokkosdnn_add_test(test_common_cli_options Test_CliOptions.cpp
  LINK_LIBRARIES KokkosDNN::common)
