# models/common: shared model-support utilities (INTERFACE, header-only).
# Currently a scaffold: only KokkosDNN_Common.hpp (placeholder). Future home
# of code extracted from models/*: minimal JSON helpers, BPE tokenizers,
# chat templates, samplers, detokenizers, stop-streamers. See README.md.
# Every model lib links KokkosDNN::common; nothing here links a model lib.

add_library(kokkosdnn_common INTERFACE)
add_library(KokkosDNN::common ALIAS kokkosdnn_common)
target_link_libraries(kokkosdnn_common INTERFACE KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(kokkosdnn_common INTERFACE cxx_std_20)

install(TARGETS kokkosdnn_common EXPORT KokkosDNNTragets)
install(FILES KokkosDNN_Common.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
