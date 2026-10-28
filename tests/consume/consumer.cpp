// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Installed-package smoke consumer: one include per installed component,
// linked against every installed linkable target (see CMakeLists.txt).
// A successful configure+build is the test (see CMakeLists.txt); on
// runners with a usable backend the binary also runs.
#include <Kokkos_Core.hpp>
#include <KokkosDNN.hpp>
#include <KokkosDNN_Common.hpp>
#include <KokkosDNN_LFM25Config.hpp>
#include <KokkosDNN_MossImageDit.hpp>
#include <KokkosDNN_QwenImage21Config.hpp>
#ifdef KOKKOSDNN_CONSUME_HAS_IMAGE
#include <images/KokkosDNN_Image.hpp>
#endif

#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    KokkosDNN::print_configuration(std::cout);
#ifdef KOKKOSDNN_CONSUME_HAS_IMAGE
    KokkosDNN::Image::ImageInfo info{2, 2, 3};
    std::cout << "image " << info.width << "x" << info.height << "x" << info.channels << "\n";
#endif
    std::cout << "lfm25 max seq " << KokkosDNN::LFM25::kMaxSupportedSeqLen << "\n";
    std::cout << "common finish "
              << KokkosDNN::Common::to_string(KokkosDNN::Common::FinishReason::Stop) << "\n";
    std::cout << "moss dit weights "
              << sizeof(KokkosDNN::MossImageDit::MossImageDitWeights<Kokkos::DefaultExecutionSpace,
                                                                    float>)
              << " bytes\n";
    std::cout << "qwen21 config " << sizeof(KokkosDNN::QwenImage21::Config) << " bytes\n";
  }
  Kokkos::finalize();
  return 0;
}
