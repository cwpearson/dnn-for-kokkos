# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# MOSS-Image-SFT example driver: model library lives in
# models/moss_image (KokkosDNN::moss, compiled STATIC like LFM25/Glimmer,
# including KokkosDNN_Moss{Config,Model,Sampler}.hpp +
# KokkosDNN_MossWeights.cpp); this directory only builds the
# moss_image_generate driver, which links that target.

kokkosdnn_add_executable(moss_image_generate moss_image_generate.cpp)
target_link_libraries(moss_image_generate PRIVATE KokkosDNN::moss)

if(KokkosDNN_ENABLE_TESTS)
  # Synthetic-ZIP test for the .pth central-directory reader: no real
  # checkpoint needed. Links the model target for KokkosDNN_MossZip.hpp
  # (which pulls the central src/KokkosDNN_Zip.hpp reader via
  # KokkosDNN::kokkosdnn).
  kokkosdnn_add_test(test_moss_zip Test_MossZip.cpp LINK_LIBRARIES KokkosDNN::moss)
endif()
