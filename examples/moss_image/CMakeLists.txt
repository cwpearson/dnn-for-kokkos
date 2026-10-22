# MOSS-Image-SFT example driver: model pipeline headers live in
# models/moss_image (KokkosDNN::moss, header-only); this directory only
# builds the moss_image_generate driver. Example-local support code
# (KokkosDNN_Moss{Config,Model,Sampler}.hpp compiled via
# KokkosDNN_MossWeights.cpp) stays here and links into the driver.

add_executable(moss_image_generate moss_image_generate.cpp KokkosDNN_MossWeights.cpp)
target_link_libraries(moss_image_generate PRIVATE KokkosDNN::moss)
set_target_properties(moss_image_generate PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
