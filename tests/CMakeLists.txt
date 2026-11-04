# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# kokkosdnn_add_test() lives in cmake/KokkosDNNHelpers.cmake (shared with
# examples/*/tests and models/common/tests).

kokkosdnn_add_test(test_gemm Test_Gemm.cpp)
kokkosdnn_add_test(test_gemmbias Test_GemmBias.cpp)
kokkosdnn_add_test(test_rmsnorm Test_RMSNorm.cpp)
kokkosdnn_add_test(test_layernorm Test_LayerNorm.cpp)
kokkosdnn_add_test(test_softmax Test_Softmax.cpp)
kokkosdnn_add_test(test_rope Test_RoPE.cpp)
kokkosdnn_add_test(test_rope2d Test_RoPE2D.cpp)
kokkosdnn_add_test(test_swiglu Test_SwiGLU.cpp)
kokkosdnn_add_test(test_attention Test_Attention.cpp)
kokkosdnn_add_test(test_attention_bidirectional Test_AttentionBidirectional.cpp)
kokkosdnn_add_test(test_gelu Test_Gelu.cpp)
kokkosdnn_add_test(test_silu Test_Silu.cpp)
kokkosdnn_add_test(test_diffusion Test_Diffusion.cpp)
kokkosdnn_add_test(test_patchembed Test_PatchEmbed.cpp)
kokkosdnn_add_test(test_depthwiseconv Test_DepthwiseConv.cpp)
kokkosdnn_add_test(test_conv Test_Conv.cpp)
kokkosdnn_add_test(test_tokens Test_Tokens.cpp)
kokkosdnn_add_test(test_layerscale Test_LayerScale.cpp)
kokkosdnn_add_test(test_sampler Test_Sampler.cpp)
kokkosdnn_add_test(test_gemmaops Test_GemmaOps.cpp)
kokkosdnn_add_test(test_gguf Test_GGUF.cpp)
kokkosdnn_add_test(test_moe Test_MoE.cpp)
kokkosdnn_add_test(test_moe_decode Test_MoeDecode.cpp)
kokkosdnn_add_test(test_conv1d Test_Conv1D.cpp)
kokkosdnn_add_test(test_conv1dgeneral Test_Conv1DGeneral.cpp)
kokkosdnn_add_test(test_lstm Test_LSTM.cpp)
kokkosdnn_add_test(test_adaptivenorm Test_AdaptiveNorm.cpp)
kokkosdnn_add_test(test_audiodsp Test_AudioDSP.cpp)
# MOSS-Image unit tests build only when the MOSS model component is enabled,
# mirroring models/CMakeLists.txt and examples/CMakeLists.txt (both guard
# their MOSS subdirectories on KokkosDNN_ENABLE_MODEL_MOSS).
# The model pipeline headers belong to the KokkosDNN::moss interface (its
# INTERFACE include dir + KokkosDNN::common link), so these tests link the
# model target instead of reaching into models/moss_image/ with a private
# include path.
if(KokkosDNN_ENABLE_MODEL_MOSS)
  kokkosdnn_add_test(test_mossimage Test_MossImageOps.cpp LINK_LIBRARIES KokkosDNN::moss)
  kokkosdnn_add_test(test_mossimageddit Test_MossImageDit.cpp LINK_LIBRARIES KokkosDNN::moss)
  kokkosdnn_add_test(test_mossimage_encoder Test_MossImageEncoder.cpp LINK_LIBRARIES KokkosDNN::moss)
  kokkosdnn_add_test(test_qwen_tokenizer Test_QwenTokenizer.cpp LINK_LIBRARIES KokkosDNN::moss)
  kokkosdnn_add_test(test_mossimagevae Test_MossImageVae.cpp LINK_LIBRARIES KokkosDNN::moss)
endif()

if(KokkosDNN_ENABLE_IMAGE)
  kokkosdnn_add_test(test_image Test_Image.cpp LINK_LIBRARIES KokkosDNN::image)
endif()

# Qwen-Image-2.1 vision tower unit test (random weights, no checkpoint).
if(KokkosDNN_ENABLE_MODEL_QWEN_IMAGE_21)
  kokkosdnn_add_test(test_qwen_image_21_vision Test_QwenImage21Vision.cpp
    LINK_LIBRARIES KokkosDNN::qwen_image_21)
  kokkosdnn_add_test(test_qwen_image_21_vae Test_QwenImage21Vae.cpp
    LINK_LIBRARIES KokkosDNN::qwen_image_21)
  kokkosdnn_add_test(test_qwen_image_21_safetensors_map
    Test_QwenImage21SafetensorsMap.cpp
    LINK_LIBRARIES KokkosDNN::qwen_image_21)
endif()
