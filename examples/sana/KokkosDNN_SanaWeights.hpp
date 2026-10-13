#ifndef KOKKOSDNN_SANA_WEIGHTS_HPP
#define KOKKOSDNN_SANA_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <string>
#include "KokkosDNN_SanaConfig.hpp"

namespace KokkosDNN {
namespace Sana {

// Default compute/storage scalar for the Sana example: float, the
// `Sana_600M_1024px_MultiLing.pth` checkpoint dtype (all float32).
using DefaultScalar = float;

// Layout convention (documented choice): 2D weights keep the PyTorch
// [out_features, in_features] row-major layout and are consumed directly by
// gemm_bias/gemm, whose W operand is already [N, K] = [out, in] (HF Linear
// convention, no transpose needed). The .pth loader therefore copies raw
// float32 storage verbatim (no host transpose); 4D conv weights with
// singleton spatial dims ([O, I, 1, 1]) are squeezed into [O, I] views and
// the depthwise weight ([H*2, 1, 3, 3]) into an [H*2, 3, 3] view, both
// without reordering elements.
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  // adaLN-single conditioning table [6, D]: shift/scale/gate x (msa, mlp).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> scale_shift_table;
  // LiteLA self-attention: fused qkv [3D, D] (no bias), proj [D, D] + bias.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_qkv_w;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_proj_w;
  Kokkos::View<Scalar*, ExecSpace> attn_proj_b;
  // Cross-attention: q [D, D] + bias, fused kv [2D, D] + bias [2D],
  // proj [D, D] + bias.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> cross_q_w;
  Kokkos::View<Scalar*, ExecSpace> cross_q_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> cross_kv_w;
  Kokkos::View<Scalar*, ExecSpace> cross_kv_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> cross_proj_w;
  Kokkos::View<Scalar*, ExecSpace> cross_proj_b;
  // GLUMBConv FFN (use_bias True,True,False; norms None; acts silu/silu/None):
  // inverted 1x1 [2H, D] + bias, depthwise 3x3 [2H, 3, 3] + bias (groups=2H),
  // point 1x1 [D, H] (no bias). H = mlp_hidden() (2880 at 600M).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mlp_inv_w;
  Kokkos::View<Scalar*, ExecSpace> mlp_inv_b;
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> mlp_dw_w;
  Kokkos::View<Scalar*, ExecSpace> mlp_dw_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mlp_pt_w;
};

// Full SanaMS DiT weights (device views in Scalar).
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  // x_embedder 1x1 conv (patch=1): proj [D, in_ch] + bias [D].
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x_proj_w;
  Kokkos::View<Scalar*, ExecSpace> x_proj_b;
  // t_embedder MLP: mlp0 [D, 256] + bias, mlp2 [D, D] + bias.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> t_mlp0_w, t_mlp2_w;
  Kokkos::View<Scalar*, ExecSpace> t_mlp0_b, t_mlp2_b;
  // t_block SiLU + Linear(D -> 6D): [6D, D] + bias [6D].
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> t_block_w;
  Kokkos::View<Scalar*, ExecSpace> t_block_b;
  // CaptionEmbedder: token table [maxlen, caption_channels], MLP
  // fc1 [D, caption_channels] + bias, fc2 [D, D] + bias (GELU-tanh).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> y_embedding;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> y_fc1_w, y_fc2_w;
  Kokkos::View<Scalar*, ExecSpace> y_fc1_b, y_fc2_b;
  // Scaled RMSNorm over caption embeds (scale 0.01 baked into values).
  Kokkos::View<Scalar*, ExecSpace> y_norm_w;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;
  // Final layer: adaLN table [2, D], linear [out_ch, D] + bias.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> final_table;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> final_w;
  Kokkos::View<Scalar*, ExecSpace> final_b;

  void allocate(const Config& cfg);
};

template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  cfg.validate();
  const size_t D = (size_t)cfg.hidden_size;
  const size_t Hh = (size_t)cfg.mlp_hidden();
  x_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_x_proj_w", D, (size_t)cfg.in_channels);
  x_proj_b = Kokkos::View<Scalar*, ExecSpace>("sana_x_proj_b", D);
  t_mlp0_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_t_mlp0_w", D, (size_t)cfg.t_freq_dim);
  t_mlp0_b = Kokkos::View<Scalar*, ExecSpace>("sana_t_mlp0_b", D);
  t_mlp2_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_t_mlp2_w", D, D);
  t_mlp2_b = Kokkos::View<Scalar*, ExecSpace>("sana_t_mlp2_b", D);
  t_block_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_t_block_w", 6 * D, D);
  t_block_b = Kokkos::View<Scalar*, ExecSpace>("sana_t_block_b", 6 * D);
  y_embedding = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_y_embedding", (size_t)cfg.model_max_length,
      (size_t)cfg.caption_channels);
  y_fc1_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_y_fc1_w", D, (size_t)cfg.caption_channels);
  y_fc1_b = Kokkos::View<Scalar*, ExecSpace>("sana_y_fc1_b", D);
  y_fc2_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_y_fc2_w", D, D);
  y_fc2_b = Kokkos::View<Scalar*, ExecSpace>("sana_y_fc2_b", D);
  y_norm_w = Kokkos::View<Scalar*, ExecSpace>("sana_y_norm_w", D);
  blocks.resize((size_t)cfg.depth);
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto& b = blocks[i];
    b.scale_shift_table =
        Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
            "sana_sst", 6, D);
    b.attn_qkv_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sana_qkv", 3 * D, D);
    b.attn_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sana_attn_proj", D, D);
    b.attn_proj_b = Kokkos::View<Scalar*, ExecSpace>("sana_attn_proj_b", D);
    b.cross_q_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sana_cross_q", D, D);
    b.cross_q_b = Kokkos::View<Scalar*, ExecSpace>("sana_cross_q_b", D);
    b.cross_kv_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sana_cross_kv", 2 * D, D);
    b.cross_kv_b = Kokkos::View<Scalar*, ExecSpace>("sana_cross_kv_b", 2 * D);
    b.cross_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sana_cross_proj", D, D);
    b.cross_proj_b =
        Kokkos::View<Scalar*, ExecSpace>("sana_cross_proj_b", D);
    b.mlp_inv_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sana_inv", 2 * Hh, D);
    b.mlp_inv_b = Kokkos::View<Scalar*, ExecSpace>("sana_inv_b", 2 * Hh);
    b.mlp_dw_w = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>(
        "sana_dw", 2 * Hh, 3, 3);
    b.mlp_dw_b = Kokkos::View<Scalar*, ExecSpace>("sana_dw_b", 2 * Hh);
    b.mlp_pt_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "sana_pt", D, Hh);
  }
  final_table = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_final_table", 2, D);
  final_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "sana_final_w", (size_t)cfg.out_channels, D);
  final_b = Kokkos::View<Scalar*, ExecSpace>("sana_final_b",
                                             (size_t)cfg.out_channels);
}

// Fill with small random values (for shape/forward tests without weights).
// LayerNorm/RMSNorm weights are ones, biases zero, adaLN tables zero
// (matches the adaLN-zero init: blocks start as identity).
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load the NVlabs/Sana `Sana_600M_1024px_MultiLing.pth` torch.save archive
// into device views (no libtorch, no new dependencies: minimal ZIP central-
// directory reader over an mmap'd mapping with bounded ifstream fallback,
// mirroring the safetensors loaders' file I/O approach; all entries in the
// released checkpoint are stored uncompressed).
//
// data.pkl is parsed with a lightweight key scanner: the state_dict holds
// 15 + 15*depth + 3 float32 tensors in a fixed order
// (pos_embed, x_embedder.proj.{weight,bias}, t_embedder.mlp.{0,2}.{weight,
// bias}, t_block.1.{weight,bias}, y_embedder.{y_embedding,y_proj.fc1/fc2.
// {weight,bias}}, attention_y_norm.weight, per block {scale_shift_table,
// attn.{qkv.weight,proj.{weight,bias}}, cross_attn.{q_linear.{weight,bias},
// kv_linear.{weight,bias},proj.{weight,bias}}, mlp.{inverted_conv.conv.
// {weight,bias},depth_conv.conv.{weight,bias},point_conv.conv.weight}},
// final_layer.{scale_shift_table,linear.{weight,bias}}); the Nth key maps to
// the Nth storage file `<prefix>/data/N` (raw float32 little-endian, verified
// via the archive `byteorder` entry), whose byte size must equal the expected
// numel * 4. 2D weights are copied verbatim in torch [out,in] row-major order
// (the gemm_bias W convention, so no transpose); 4D conv weights are squeezed
// (see the layout note above). pos_embed is size-checked but not stored
// (use_pe=false). Throws on key mismatch, count mismatch, non-float32
// storage, or shape/numel mismatch; y_embedding is explicitly confirmed as
// [model_max_length, caption_channels] ([300,2304] at 600M).
template <class ExecSpace, class Scalar = DefaultScalar>
void load_torch_pth(const Config& cfg, const std::string& pth_path,
                     Weights<ExecSpace, Scalar>& w);

}  // namespace Sana
}  // namespace KokkosDNN

#endif
