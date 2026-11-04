# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# MOSS-Image-SFT example driver: model pipeline headers live in
# models/moss_image (KokkosDNN::moss, header-only); this directory only
# builds the moss_image_generate driver. Example-local support code
# (KokkosDNN_Moss{Config,Model,Sampler}.hpp compiled via
# KokkosDNN_MossWeights.cpp) stays here and links into the driver.

kokkosdnn_add_executable(moss_image_generate moss_image_generate.cpp KokkosDNN_MossWeights.cpp)
target_link_libraries(moss_image_generate PRIVATE KokkosDNN::moss)

if(KokkosDNN_ENABLE_TESTS)
  # Synthetic-ZIP test for the .pth central-directory reader: no real
  # checkpoint needed. Links gtest_main + the header under test (which pulls
  # the central src/KokkosDNN_Zip.hpp reader via KokkosDNN::kokkosdnn).
  kokkosdnn_add_test(test_moss_zip Test_MossZip.cpp)
endif()
