# Llama 3 example: model-specific inference built on top of the core library.
# Not part of libkokkosdnn; kept here as example/support code.

add_library(kokkosdnn_llama STATIC
  KokkosDNN_LlamaWeights.cpp
  KokkosDNN_Tokenizer.cpp
)
add_library(KokkosDNN::llama ALIAS kokkosdnn_llama)
target_link_libraries(kokkosdnn_llama PUBLIC KokkosDNN::kokkosdnn)
target_include_directories(kokkosdnn_llama PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(kokkosdnn_llama PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(llama_generate llama_generate.cpp)
target_link_libraries(llama_generate PRIVATE KokkosDNN::llama)
set_target_properties(llama_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(test_llama_generate Test_LlamaGenerate.cpp)
target_link_libraries(test_llama_generate PRIVATE KokkosDNN::llama)
set_target_properties(test_llama_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME test_llama_generate COMMAND test_llama_generate)

add_executable(test_safetensors Test_Safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE KokkosDNN::llama)
set_target_properties(test_safetensors PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME test_safetensors COMMAND test_safetensors)
