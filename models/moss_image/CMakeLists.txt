# MOSS-Image model library: installable component (KokkosDNN::moss).
# Header-only pipeline (KokkosDNN_MossImage{,Dit,Encoder,Vae}.hpp,
# KokkosDNN_QwenTokenizer.hpp + tables .inc live here); there are no .cpp
# sources, hence INTERFACE. The example-local support code
# (KokkosDNN_Moss{Config,Model,Sampler}.hpp, KokkosDNN_MossWeights.cpp)
# stays in examples/moss_image and compiles into the driver there.

add_library(kokkosdnn_moss INTERFACE)
add_library(KokkosDNN::moss ALIAS kokkosdnn_moss)
target_link_libraries(kokkosdnn_moss INTERFACE
  KokkosDNN::kokkosdnn KokkosDNN::common)
target_include_directories(kokkosdnn_moss INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(KokkosDNN_ENABLE_IMAGE)
  target_link_libraries(kokkosdnn_moss INTERFACE KokkosDNN::image)
endif()
target_compile_features(kokkosdnn_moss INTERFACE cxx_std_20)
kokkosdnn_enable_compiler_warnings(kokkosdnn_moss)

install(TARGETS kokkosdnn_moss EXPORT KokkosDNNTragets)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.inc"
)
