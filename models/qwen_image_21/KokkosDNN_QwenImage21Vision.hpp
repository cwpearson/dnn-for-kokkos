// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_QWENIMAGE21VISION_HPP
#define KOKKOSDNN_QWENIMAGE21VISION_HPP

// Qwen3-VL vision tower for Qwen-Image-2.1 edit conditioning (image -> joint
// text+vision embeddings). Ground truth (HF research: transformers
// modeling_qwen3_vl.py + the Qwen-Image-2.1-Turbo text_encoder checkpoint):
//   patch_embed Conv3d [H, C, T, P, P] (T=temporal_patch_size, P=patch_size,
//     stride == kernel, +bias) over block-major patches whose temporal axis
//     is duplicated (pixel rows [Np, C*T*P*P], Np = Hp*Wp patch rows);
//   learned pos_embed [side*side, H] resampled per image by separable
//     bilinear (align_corners=True, border clamp) to the patch grid;
//   axial 2D RoPE (H/W freq halves concatenated, full-head rotation);
//   depth full-attention MHA blocks (LayerNorm+bias, fused qkv+bias, proj
//     +bias, GELU-tanh MLP+bias, no window attention, no KV cache);
//   PatchMerger (biased LayerNorm -> group 2x2 -> Linear GELU(exact-erf)
//     Linear to out_hidden_size) plus one biased-norm merger per deepstack
//     layer (deepstack_visual_indexes). The merger output rows are the LLM
//     vision tokens (Nv = Np/merge^2, block-row-major); deepstack features
//     are added into text-tower hidden states at layers 0..2 (see
//     QwenImage21EncoderModel::forward_joint).
//
// Qwen-Image-2.1-Turbo values (text_encoder/config.json + safetensors
// header): depth 27, hidden 1152, 16 heads (hd 72), intermediate 4304,
// gelu_pytorch_tanh, patch 16, temporal 2, in 3, spatial_merge 2, out 4096,
// num_position_embeddings 2304 (side 48), deepstack [8, 16, 24], bf16.
// rope_parameters is absent from the checkpoint, so transformers fills its
// defaults (RotaryEmbeddingConfigMixin::default_theta = 10000.0 with
// Qwen3VLVisionConfig::default_rope_type = axial); that default theta value
// (10000.0, verified correct) is this header's default.
//
// Dispatch contract (no backend-gated code): every Linear goes through
// gemm_bias (cuBLAS ladder); attention is attention_mha_bidirectional
// (cuDNN SDPA ladder, else native); LayerNorm/GELU/RoPE/pos-gather/merge
// grouping are native backend-portable parallel_for kernels. All Impl
// helpers are elementwise/layout glue with no TPL expresses.
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "KokkosDNN_Activation.hpp"
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_LayerNorm.hpp"

#include "KokkosDNN_QwenImage21Config.hpp"

namespace KokkosDNN {
namespace QwenImage21Vision {

// Vision tower constants for Qwen-Image-2.1 (Qwen3-VL-8B visual).
struct Config {
  int depth = 27;              // ViT blocks
  int hidden_dim = 1152;       // patch embed dim
  int num_heads = 16;          // MHA heads (head_dim = 72)
  int intermediate_dim = 4304;  // MLP hidden (fc1/fc2 +bias, GELU-tanh)
  int patch_size = 16;         // spatial patch (driver packs these)
  int temporal_patch_size = 2;  // pixel rows duplicate the frame x2
  int in_channels = 3;         // RGB vision input ([-1,1] post-normalize)
  int spatial_merge_size = 2;  // 2x2 merge -> LLM vision tokens
  int out_hidden_size = 4096;  // merger output (== text hidden_dim)
  int num_position_embeddings = 2304;  // learned grid (side 48)
  std::array<int, 3> deepstack_visual_indexes = {8, 16, 24};
  double rope_theta = 10000.0;  // axial RoPE base (checkpoint default)
  double norm_eps = 1e-6;

  int head_dim() const { return (num_heads > 0) ? hidden_dim / num_heads : 0; }
  int num_grid_per_side() const {
    const int s = (int)std::llround(std::sqrt((double)num_position_embeddings));
    return (s * s == num_position_embeddings) ? s : 0;
  }
  // Pixel-row width the driver must pack: [Np, patch_dim].
  int patch_dim() const {
    return in_channels * temporal_patch_size * patch_size * patch_size;
  }

  static Config tiny_test() {
    Config c;
    c.depth = 3;
    c.hidden_dim = 32;
    c.num_heads = 4;  // head_dim 8
    c.intermediate_dim = 48;
    c.patch_size = 2;
    c.temporal_patch_size = 1;
    c.in_channels = 3;
    c.spatial_merge_size = 2;
    c.out_hidden_size = 16;
    c.num_position_embeddings = 16;  // side 4
    c.deepstack_visual_indexes = {0, 1, 2};
    return c;
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("QwenImage21Vision::Config::validate: " +
                                  m);
    };
    if (depth <= 0) bad("depth must be > 0");
    if (hidden_dim <= 0) bad("hidden_dim must be > 0");
    if (num_heads <= 0) bad("num_heads must be > 0");
    if (hidden_dim % num_heads != 0)
      bad("hidden_dim must be divisible by num_heads");
    if (head_dim() % 4 != 0)
      bad("head_dim must be divisible by 4 (axial H/W freq halves)");
    if (intermediate_dim <= 0) bad("intermediate_dim must be > 0");
    if (patch_size <= 0) bad("patch_size must be > 0");
    if (temporal_patch_size <= 0) bad("temporal_patch_size must be > 0");
    if (in_channels <= 0) bad("in_channels must be > 0");
    if (spatial_merge_size <= 0) bad("spatial_merge_size must be > 0");
    if (out_hidden_size <= 0) bad("out_hidden_size must be > 0");
    if (num_grid_per_side() == 0)
      bad("num_position_embeddings must be a perfect square");
    for (int l : deepstack_visual_indexes) {
      if (l < 0 || l >= depth) bad("deepstack_visual_indexes out of range");
    }
    if (!(rope_theta > 0) || !std::isfinite(rope_theta))
      bad("rope_theta must be positive and finite");
    if (!(norm_eps > 0) || !std::isfinite(norm_eps))
      bad("norm_eps must be positive and finite");
  }
};

// One expected checkpoint tensor: bare (suffix-matched) name + shape.
struct VisionTensorSpec {
  std::string name;
  int64_t d0 = 0;
  int64_t d1 = 1;
  bool matrix = false;
};

// Canonical Qwen3-VL visual weight list for a Config. Checkpoint keys live
// under model.visual.* (no infix to strip): patch_embed.proj (Conv3d
// [H, C, T, P, P] row-major, matching the driver's [C, T, P, P] pixel-row
// flatten so file bytes land verbatim), pos_embed, per-block
// attn.qkv/proj (+bias), norm1/norm2 (LayerNorm +bias), mlp.linear_fc1/fc2
// (+bias), merger (biased LayerNorm, fc1/fc2 +bias over
// hidden*merge^2), and one biased-norm merger per deepstack layer.
inline std::vector<VisionTensorSpec> qwen21_vision_expected_tensors(
    const Config& cfg) {
  cfg.validate();
  const int64_t H = cfg.hidden_dim, I = cfg.intermediate_dim,
                P = cfg.patch_size, T = cfg.temporal_patch_size,
                C = cfg.in_channels, M = (int64_t)cfg.spatial_merge_size,
                O = cfg.out_hidden_size;
  const int64_t G = H * M * M;  // merged width
  std::vector<VisionTensorSpec> ts;
  auto mat = [&](const std::string& n, int64_t r, int64_t c) {
    ts.push_back({n, r, c, true});
  };
  auto vec = [&](const std::string& n, int64_t r) {
    ts.push_back({n, r, 1, false});
  };
  mat("model.visual.patch_embed.proj.weight", H, C * T * P * P);
  vec("model.visual.patch_embed.proj.bias", H);
  mat("model.visual.pos_embed.weight", cfg.num_position_embeddings, H);
  for (int i = 0; i < cfg.depth; ++i) {
    const std::string a = "model.visual.blocks." + std::to_string(i) + ".";
    mat(a + "attn.qkv.weight", 3 * H, H);
    vec(a + "attn.qkv.bias", 3 * H);
    mat(a + "attn.proj.weight", H, H);
    vec(a + "attn.proj.bias", H);
    vec(a + "norm1.weight", H);
    vec(a + "norm1.bias", H);
    vec(a + "norm2.weight", H);
    vec(a + "norm2.bias", H);
    mat(a + "mlp.linear_fc1.weight", I, H);
    vec(a + "mlp.linear_fc1.bias", I);
    mat(a + "mlp.linear_fc2.weight", H, I);
    vec(a + "mlp.linear_fc2.bias", H);
  }
  const std::string m = "model.visual.merger.";
  vec(m + "norm.weight", H);
  vec(m + "norm.bias", H);
  mat(m + "linear_fc1.weight", G, G);
  vec(m + "linear_fc1.bias", G);
  mat(m + "linear_fc2.weight", O, G);
  vec(m + "linear_fc2.bias", O);
  for (size_t k = 0; k < cfg.deepstack_visual_indexes.size(); ++k) {
    const std::string d = "model.visual.deepstack_merger_list." +
                          std::to_string(k) + ".";
    vec(d + "norm.weight", G);
    vec(d + "norm.bias", G);
    mat(d + "linear_fc1.weight", G, G);
    vec(d + "linear_fc1.bias", G);
    mat(d + "linear_fc2.weight", O, G);
    vec(d + "linear_fc2.bias", O);
  }
  return ts;
}

namespace Impl {

// Host axial-RoPE inv_freq: inv[j] = theta^(-2j/Sd), j < Sd/2, Sd = Hd/2.
// Matches Qwen3VLVisionRotaryEmbedding.compute_axial_rope_parameters
// (base ^ (arange(0, spatial_dim, 2) / spatial_dim), spatial_dim = Hd/2).
inline std::vector<float> qwen21vit_rope_inv_host(int head_dim,
                                                  double theta) {
  const size_t Sd = static_cast<size_t>(head_dim) / 2;
  if (Sd % 2 != 0)
    throw std::invalid_argument(
        "qwen21vit_rope_inv_host: head_dim/2 must be even");
  std::vector<float> out(Sd / 2);
  for (size_t j = 0; j < Sd / 2; ++j)
    out[j] = static_cast<float>(1.0 / std::pow(theta,
                                               (2.0 * (double)j) /
                                                   (double)Sd));
  return out;
}

// Axial 2D RoPE, in place, full-head rotation pairing (a, b) = (x[d],
// x[d + Hd/2]): pair k < Hd/2 reads the H position for k < F and the W
// position otherwise (F = Hd/4 freqs per axis), matching the reference
// recomposition (cat([freq_h, freq_w]) duplicated to both halves) with
// rotate_half. x: [T, H, Hd]; pos_hw: [T, 2] int (h, w) block-major rows.
template <class ExecSpace, class XView, class PView, class FView>
void qwen21vit_rope_inplace(const XView& x, const PView& pos_hw,
                            const FView& inv_freq,
                            const std::string& label = "Qwen21Vit::rope") {
  const size_t T = x.extent(0), H = x.extent(1), D = x.extent(2);
  if (D % 4 != 0)
    throw std::invalid_argument("qwen21vit_rope_inplace: head_dim % 4 != 0");
  const size_t F = D / 4;
  if (inv_freq.extent(0) != F)
    throw std::invalid_argument("qwen21vit_rope_inplace: inv_freq dim");
  if (pos_hw.extent(0) != T || pos_hw.extent(1) != 2)
    throw std::invalid_argument("qwen21vit_rope_inplace: pos_hw must be [T,2]");
  if (T == 0 || H == 0) return;
  using Scalar = std::remove_const_t<typename XView::value_type>;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        const float ph = static_cast<float>(pos_hw(t, 0));
        const float pw = static_cast<float>(pos_hw(t, 1));
        for (size_t k = 0; k < D / 2; ++k) {
          const float pos = (k < F) ? ph : pw;
          const float ang = pos * inv_freq(k < F ? k : k - F);
          const float c = Kokkos::cosf(ang), s = Kokkos::sinf(ang);
          const float a = static_cast<float>(x(t, h, k));
          const float b = static_cast<float>(x(t, h, k + D / 2));
          x(t, h, k) = static_cast<Scalar>(a * c - b * s);
          x(t, h, k + D / 2) = static_cast<Scalar>(a * s + b * c);
        }
      });
}

// qkv [T,3C] -> q/k/v [T,C] split (row-major thirds).
template <class ExecSpace, class V2>
inline void qwen21vit_split_qkv(const V2& qkv, const V2& q, const V2& k,
                                const V2& v, const std::string& label) {
  const size_t T = qkv.extent(0), C = q.extent(1);
  if (qkv.extent(1) != 3 * C || k.extent(0) != T || k.extent(1) != C ||
      v.extent(0) != T || v.extent(1) != C)
    throw std::invalid_argument("qwen21vit_split_qkv: dimension mismatch");
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * C),
      KOKKOS_LAMBDA(size_t e) {
        const size_t r = e / C, c = e % C;
        q(r, c) = qkv(r, c);
        k(r, c) = qkv(r, c + C);
        v(r, c) = qkv(r, c + 2 * C);
      });
}

// Learned-grid resampling, host side: separable bilinear (align_corners =
// true, border clamp) of a side x side table to one (h, w) patch grid,
// rows in spatial-merge-block-major order (block_row, block_col, in_row,
// in_col), matching pixel_values packing and rope positions. Returns
// indices [Np, 4] into the flattened table + weights [Np, 4].
inline void qwen21vit_interp_pos_host(int h, int w, int side, int merge,
                                      std::vector<int>& indices,
                                      std::vector<float>& weights) {
  if (h <= 0 || w <= 0 || side <= 0 || merge <= 0)
    throw std::invalid_argument("qwen21vit_interp_pos_host: bad dims");
  if (h % merge != 0 || w % merge != 0)
    throw std::invalid_argument(
        "qwen21vit_interp_pos_host: grid must tile the merge");
  const size_t Np = (size_t)h * (size_t)w;
  indices.assign(Np * 4, 0);
  weights.assign(Np * 4, 0.0f);
  const int bw = w / merge;
  auto axis = [](float v, int n, int s, int taps[2], float wt[2]) {
    // align_corners: src = v * (s - 1) / max(n - 1, 1), border clamped.
    const float src =
        (n > 1) ? v * (float)(s - 1) / (float)(n - 1) : 0.0f;
    const float c =
        src < 0.0f ? 0.0f : (src > (float)(s - 1) ? (float)(s - 1) : src);
    const int f = (int)std::floor(c);
    const int f1 = (f + 1 <= s - 1) ? f + 1 : s - 1;
    const float d = c - (float)f;
    taps[0] = f;
    taps[1] = f1;
    wt[0] = 1.0f - d;
    wt[1] = d;
  };
  size_t p = 0;
  for (int bh = 0; bh < h / merge; ++bh)
    for (int bwid = 0; bwid < bw; ++bwid)
      for (int ih = 0; ih < merge; ++ih)
        for (int iw = 0; iw < merge; ++iw) {
          const int r = bh * merge + ih, c = bwid * merge + iw;
          int th[2], tw[2];
          float wh[2], ww[2];
          axis((float)r, h, side, th, wh);
          axis((float)c, w, side, tw, ww);
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
              indices[p * 4 + a * 2 + b] = th[a] * side + tw[b];
              weights[p * 4 + a * 2 + b] = wh[a] * ww[b];
            }
          ++p;
        }
}

// Block-major (h, w) positions for Np = h*w patch rows: row p holds the
// (r, c) patch in merge-block-major order (same order as pixel_values and
// the pos-interp rows above). Host side; mirrored to device by forward().
inline std::vector<int> qwen21vit_block_positions_host(int h, int w,
                                                       int merge) {
  if (h <= 0 || w <= 0 || merge <= 0 || h % merge != 0 || w % merge != 0)
    throw std::invalid_argument("qwen21vit_block_positions_host: bad dims");
  std::vector<int> out;
  out.reserve((size_t)h * (size_t)w * 2);
  for (int bh = 0; bh < h / merge; ++bh)
    for (int bw = 0; bw < w / merge; ++bw)
      for (int ih = 0; ih < merge; ++ih)
        for (int iw = 0; iw < merge; ++iw) {
          out.push_back(bh * merge + ih);
          out.push_back(bw * merge + iw);
        }
  return out;
}

}  // namespace Impl

// Vision weights: device blobs keyed by checkpoint tensor name.
template <class ExecSpace, class Scalar>
struct QwenImage21VisionWeights {
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;

  Config cfg;
  V2 patch_W;      // [H, C*T*P*P]
  V1 patch_b;      // [H]
  V2 pos_table;    // [side*side, H]
  struct Layer {
    V2 qkv_W;  // [3H, H]
    V1 qkv_b;  // [3H]
    V2 proj_W;  // [H, H]
    V1 proj_b;  // [H]
    V1 n1_w, n1_b, n2_w, n2_b;  // LayerNorm [H] +bias
    V2 fc1_W;  // [I, H]
    V1 fc1_b;  // [I]
    V2 fc2_W;  // [H, I]
    V1 fc2_b;  // [H]
  };
  std::vector<Layer> layers;
  // Main merger: biased LayerNorm over hidden (like the blocks).
  V1 merge_norm_w;  // [H]
  V1 merge_norm_b;  // [H] biased merger norm
  V2 merge_fc1_W;  // [G, G]
  V1 merge_fc1_b;  // [G]
  V2 merge_fc2_W;  // [O, G]
  V1 merge_fc2_b;  // [O]
  struct DeepMerger {
    V1 norm_w, norm_b;  // [G] biased postshuffle norm
    V2 fc1_W;  // [G, G]
    V1 fc1_b;  // [G]
    V2 fc2_W;  // [O, G]
    V1 fc2_b;  // [O]
  };
  std::vector<DeepMerger> deep;  // one per deepstack index

  void allocate(const Config& c) {
    c.validate();
    cfg = c;
    const size_t H = (size_t)c.hidden_dim, I = (size_t)c.intermediate_dim,
                 O = (size_t)c.out_hidden_size;
    const size_t P = (size_t)c.patch_dim();
    const size_t G = H * (size_t)c.spatial_merge_size *
                     (size_t)c.spatial_merge_size;
    patch_W = V2("qwen21vit_patch_W", H, P);
    patch_b = V1("qwen21vit_patch_b", H);
    pos_table =
        V2("qwen21vit_pos", (size_t)c.num_position_embeddings, H);
    layers.resize((size_t)c.depth);
    for (auto& l : layers) {
      l.qkv_W = V2("qwen21vit_qkv_W", 3 * H, H);
      l.qkv_b = V1("qwen21vit_qkv_b", 3 * H);
      l.proj_W = V2("qwen21vit_proj_W", H, H);
      l.proj_b = V1("qwen21vit_proj_b", H);
      l.n1_w = V1("qwen21vit_n1_w", H);
      l.n1_b = V1("qwen21vit_n1_b", H);
      l.n2_w = V1("qwen21vit_n2_w", H);
      l.n2_b = V1("qwen21vit_n2_b", H);
      l.fc1_W = V2("qwen21vit_fc1_W", I, H);
      l.fc1_b = V1("qwen21vit_fc1_b", I);
      l.fc2_W = V2("qwen21vit_fc2_W", H, I);
      l.fc2_b = V1("qwen21vit_fc2_b", H);
    }
    merge_norm_w = V1("qwen21vit_merge_norm_w", H);
    merge_norm_b = V1("qwen21vit_merge_norm_b", H);
    merge_fc1_W = V2("qwen21vit_merge_fc1_W", G, G);
    merge_fc1_b = V1("qwen21vit_merge_fc1_b", G);
    merge_fc2_W = V2("qwen21vit_merge_fc2_W", O, G);
    merge_fc2_b = V1("qwen21vit_merge_fc2_b", O);
    deep.resize(c.deepstack_visual_indexes.size());
    for (auto& d : deep) {
      d.norm_w = V1("qwen21vit_deep_norm_w", G);
      d.norm_b = V1("qwen21vit_deep_norm_b", G);
      d.fc1_W = V2("qwen21vit_deep_fc1_W", G, G);
      d.fc1_b = V1("qwen21vit_deep_fc1_b", G);
      d.fc2_W = V2("qwen21vit_deep_fc2_W", O, G);
      d.fc2_b = V1("qwen21vit_deep_fc2_b", O);
    }
  }

  void release() {
    patch_W = {};
    patch_b = {};
    pos_table = {};
    layers.clear();
    layers.shrink_to_fit();
    merge_norm_w = {};
    merge_norm_b = {};
    merge_fc1_W = {};
    merge_fc1_b = {};
    merge_fc2_W = {};
    merge_fc2_b = {};
    deep.clear();
    deep.shrink_to_fit();
  }

  size_t resident_bytes() const {
    size_t n = patch_W.size() + patch_b.size() + pos_table.size() +
               merge_norm_w.size() + merge_norm_b.size() +
               merge_fc1_W.size() + merge_fc1_b.size() + merge_fc2_W.size() +
               merge_fc2_b.size();
    for (const auto& l : layers)
      n += l.qkv_W.size() + l.qkv_b.size() + l.proj_W.size() +
           l.proj_b.size() + l.n1_w.size() + l.n1_b.size() + l.n2_w.size() +
           l.n2_b.size() + l.fc1_W.size() + l.fc1_b.size() + l.fc2_W.size() +
           l.fc2_b.size();
    for (const auto& d : deep)
      n += d.norm_w.size() + d.norm_b.size() + d.fc1_W.size() +
           d.fc1_b.size() + d.fc2_W.size() + d.fc2_b.size();
    return n * sizeof(Scalar);
  }

  // Load the visual tower from the text_encoder shard (model.visual.* only;
  // every other key is skipped, not an error, so the text loader can own
  // the same file). Throws on missing/duplicate/dtype/shape mismatch.
  // Pass a warmed QwenImage21::ShardCache to stream a staged shard from
  // host RAM instead of disk (repeat loads); the default nullptr reads
  // from disk, so existing one-argument callers are unchanged.
  void load_safetensors(const std::string& shard_path,
                        QwenImage21::ShardCache* cache = nullptr) {
    if (patch_W.extent(0) == 0)
      throw std::runtime_error(
          "QwenImage21VisionWeights::load_safetensors: allocate first");
    static_assert(std::is_same_v<Scalar, float> ||
                      std::is_same_v<Scalar,
                                     Kokkos::Experimental::bhalf_t>,
                  "QwenImage21VisionWeights::load_safetensors: Scalar must be "
                  "float or bhalf_t");
    const std::vector<VisionTensorSpec> specs =
        qwen21_vision_expected_tensors(cfg);
    std::map<std::string, bool> found;
    for (const auto& s : specs) found[s.name] = false;
    std::ifstream f;
    const QwenImage21::Impl::Qwen21ShardBlob* blob =
        (cache != nullptr) ? cache->find(shard_path) : nullptr;
    const std::map<std::string, QwenImage21::Impl::Qwen21SafeEntry> entries =
        (blob != nullptr)
            ? QwenImage21::Impl::qwen21_parse_shard_blob(blob->bytes,
                                                        shard_path)
            : QwenImage21::Impl::qwen21_open_shard(f, shard_path);
    for (const auto& kv : entries) {
      const std::string& full = kv.first;
      if (full.compare(0, 13, "model.visual.") != 0) continue;  // text tower
      const QwenImage21::Impl::Qwen21SafeEntry& en = kv.second;
      const VisionTensorSpec* spec = nullptr;
      for (const auto& s : specs) {
        if (full == s.name) {
          spec = &s;
          break;
        }
      }
      if (!spec)
        throw std::runtime_error(
            "QwenImage21VisionWeights::load_safetensors: unknown tensor '" +
            full + "' in '" + shard_path + "'");
      if (found[spec->name])
        throw std::runtime_error(
            "QwenImage21VisionWeights::load_safetensors: duplicate tensor '" +
            spec->name + "'");
      if (blob != nullptr) {
        load_one_tensor(*blob, full, en, *spec);
      } else {
        load_one_tensor(f, full, en, *spec);
      }
      found[spec->name] = true;
    }
    for (const auto& s : specs) {
      if (!found[s.name])
        throw std::runtime_error(
            "QwenImage21VisionWeights::load_safetensors: missing tensor '" +
            s.name + "'");
    }
  }

  // Allocate exactly the tensors forward() consumes (see
  // qwen21_vision_expected_tensors), zero-initialized. Lets tests exercise
  // forward() shapes without the checkpoint; call randomize() (or
  // load_safetensors) before running.
  void allocate_expected(const Config& c) {
    allocate(c);
    Kokkos::deep_copy(patch_W, 0);
    Kokkos::deep_copy(patch_b, 0);
    Kokkos::deep_copy(pos_table, 0);
    for (auto& l : layers) {
      Kokkos::deep_copy(l.qkv_W, 0);
      Kokkos::deep_copy(l.qkv_b, 0);
      Kokkos::deep_copy(l.proj_W, 0);
      Kokkos::deep_copy(l.proj_b, 0);
      Kokkos::deep_copy(l.n1_w, 0);
      Kokkos::deep_copy(l.n1_b, 0);
      Kokkos::deep_copy(l.n2_w, 0);
      Kokkos::deep_copy(l.n2_b, 0);
      Kokkos::deep_copy(l.fc1_W, 0);
      Kokkos::deep_copy(l.fc1_b, 0);
      Kokkos::deep_copy(l.fc2_W, 0);
      Kokkos::deep_copy(l.fc2_b, 0);
    }
    Kokkos::deep_copy(merge_norm_w, 0);
    Kokkos::deep_copy(merge_norm_b, 0);
    Kokkos::deep_copy(merge_fc1_W, 0);
    Kokkos::deep_copy(merge_fc1_b, 0);
    Kokkos::deep_copy(merge_fc2_W, 0);
    Kokkos::deep_copy(merge_fc2_b, 0);
    for (auto& d : deep) {
      Kokkos::deep_copy(d.norm_w, 0);
      Kokkos::deep_copy(d.norm_b, 0);
      Kokkos::deep_copy(d.fc1_W, 0);
      Kokkos::deep_copy(d.fc1_b, 0);
      Kokkos::deep_copy(d.fc2_W, 0);
      Kokkos::deep_copy(d.fc2_b, 0);
    }
  }

  // Smoke-test fill: N(0, 0.02) for projections/biases, 1 for norms (the
  // zero-bias merger slot is left at zeros, never randomized).
  void randomize(uint64_t seed) {
    cfg.validate();
    if (patch_W.extent(0) == 0)
      throw std::runtime_error(
          "QwenImage21VisionWeights::randomize: no tensors (allocate first)");
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    auto rnd2 = [&](V2& v) {
      auto h = Kokkos::create_mirror_view(v);
      for (size_t i = 0; i < h.extent(0); ++i)
        for (size_t j = 0; j < h.extent(1); ++j)
          h(i, j) = static_cast<Scalar>(dist(rng));
      Kokkos::deep_copy(v, h);
    };
    auto rnd1 = [&](V1& v, bool ones) {
      auto h = Kokkos::create_mirror_view(v);
      for (size_t i = 0; i < h.extent(0); ++i)
        h(i) = ones ? static_cast<Scalar>(1.0f)
                    : static_cast<Scalar>(dist(rng));
      Kokkos::deep_copy(v, h);
    };
    auto one2 = [&](V2& v) {
      auto h = Kokkos::create_mirror_view(v);
      for (size_t i = 0; i < h.extent(0); ++i)
        for (size_t j = 0; j < h.extent(1); ++j)
          h(i, j) = (i == j) ? static_cast<Scalar>(1.0f)
                             : static_cast<Scalar>(0.0f);
      Kokkos::deep_copy(v, h);
    };
    rnd2(patch_W);
    rnd1(patch_b, false);
    rnd2(pos_table);  // learned table: small random (smoke only)
    for (auto& l : layers) {
      one2(l.qkv_W);
      rnd1(l.qkv_b, false);
      one2(l.proj_W);
      rnd1(l.proj_b, false);
      rnd1(l.n1_w, true);
      rnd1(l.n1_b, false);
      rnd1(l.n2_w, true);
      rnd1(l.n2_b, false);
      rnd2(l.fc1_W);
      rnd1(l.fc1_b, false);
      rnd2(l.fc2_W);
      rnd1(l.fc2_b, false);
    }
    rnd1(merge_norm_w, true);
    rnd1(merge_norm_b, false);
    rnd2(merge_fc1_W);
    rnd1(merge_fc1_b, false);
    rnd2(merge_fc2_W);
    rnd1(merge_fc2_b, false);
    for (auto& d : deep) {
      rnd1(d.norm_w, true);
      rnd1(d.norm_b, false);
      rnd2(d.fc1_W);
      rnd1(d.fc1_b, false);
      rnd2(d.fc2_W);
      rnd1(d.fc2_b, false);
    }
  }

  // Resident estimate for the staged memory log.
  size_t num_tensors() const {
    size_t n = 3 + layers.size() * 12 + 5 + deep.size() * 6;
    return n;
  }

 private:
  // Stream one file tensor into its device view (shape-checked). Source is
  // either a disk stream or a staged shard image (overload resolution on
  // the first argument picks the matching qwen21_stream_into).
  template <class Source>
  void load_one_tensor(Source& src, const std::string& full,
                       const QwenImage21::Impl::Qwen21SafeEntry& en,
                       const VisionTensorSpec& s) {
    const size_t elsize = QwenImage21::Impl::qwen21_entry_elsize(en, full);
    const int64_t numel = QwenImage21::Impl::qwen21_entry_numel(en, full);
    const size_t want =
        s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
    if ((size_t)numel != want)
      throw std::runtime_error(
          "QwenImage21VisionWeights::load_safetensors: tensor '" + full +
          "' has numel " + std::to_string(numel) + ", expected " +
          std::to_string(want));
    if (en.end < en.begin || en.end - en.begin != want * elsize)
      throw std::runtime_error(
          "QwenImage21VisionWeights::load_safetensors: tensor '" + full +
          "' slice size mismatch");
    if (s.matrix) {
      V2* dst = resolve_mat(s.name);
      if (!dst || (*dst).extent(0) != (size_t)s.d0 ||
          (*dst).extent(1) != (size_t)s.d1)
        throw std::runtime_error(
            "QwenImage21VisionWeights::load_safetensors: view mismatch '" +
            full + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          src, en, want, elsize, (*dst).data());
    } else {
      V1* dst = resolve_vec(s.name);
      if (!dst || (*dst).extent(0) != (size_t)s.d0)
        throw std::runtime_error(
            "QwenImage21VisionWeights::load_safetensors: view mismatch '" +
            full + "'");
      QwenImage21::Impl::qwen21_stream_into<ExecSpace, Scalar>(
          src, en, want, elsize, (*dst).data());
    }
  }

  V2* resolve_mat(const std::string& name) {
    if (name == "model.visual.patch_embed.proj.weight") return &patch_W;
    if (name == "model.visual.pos_embed.weight") return &pos_table;
    if (name.compare(0, 20, "model.visual.blocks.") == 0) {
      size_t dot = name.find('.', 20);
      int li = std::stoi(name.substr(20, dot - 20));
      if (li < 0 || (size_t)li >= layers.size()) return nullptr;
      auto& l = layers[(size_t)li];
      std::string tail = name.substr(dot + 1);
      if (tail == "attn.qkv.weight") return &l.qkv_W;
      if (tail == "attn.proj.weight") return &l.proj_W;
      if (tail == "mlp.linear_fc1.weight") return &l.fc1_W;
      if (tail == "mlp.linear_fc2.weight") return &l.fc2_W;
      return nullptr;
    }
    const std::string mp = "model.visual.merger.";
    if (name.compare(0, mp.size(), mp) == 0) {
      std::string tail = name.substr(mp.size());
      if (tail == "linear_fc1.weight") return &merge_fc1_W;
      if (tail == "linear_fc2.weight") return &merge_fc2_W;
      return nullptr;
    }
    const std::string dp = "model.visual.deepstack_merger_list.";
    if (name.compare(0, dp.size(), dp) == 0) {
      size_t dot = name.find('.', dp.size());
      int ki = std::stoi(name.substr(dp.size(), dot - dp.size()));
      if (ki < 0 || (size_t)ki >= deep.size()) return nullptr;
      auto& d = deep[(size_t)ki];
      std::string tail = name.substr(dot + 1);
      if (tail == "linear_fc1.weight") return &d.fc1_W;
      if (tail == "linear_fc2.weight") return &d.fc2_W;
      return nullptr;
    }
    return nullptr;
  }

  V1* resolve_vec(const std::string& name) {
    if (name == "model.visual.patch_embed.proj.bias") return &patch_b;
    if (name.compare(0, 20, "model.visual.blocks.") == 0) {
      size_t dot = name.find('.', 20);
      int li = std::stoi(name.substr(20, dot - 20));
      if (li < 0 || (size_t)li >= layers.size()) return nullptr;
      auto& l = layers[(size_t)li];
      std::string tail = name.substr(dot + 1);
      if (tail == "attn.qkv.bias") return &l.qkv_b;
      if (tail == "attn.proj.bias") return &l.proj_b;
      if (tail == "norm1.weight") return &l.n1_w;
      if (tail == "norm1.bias") return &l.n1_b;
      if (tail == "norm2.weight") return &l.n2_w;
      if (tail == "norm2.bias") return &l.n2_b;
      if (tail == "mlp.linear_fc1.bias") return &l.fc1_b;
      if (tail == "mlp.linear_fc2.bias") return &l.fc2_b;
      return nullptr;
    }
    const std::string mp = "model.visual.merger.";
    if (name.compare(0, mp.size(), mp) == 0) {
      std::string tail = name.substr(mp.size());
      if (tail == "norm.weight") return &merge_norm_w;
      if (tail == "norm.bias") return &merge_norm_b;
      if (tail == "linear_fc1.bias") return &merge_fc1_b;
      if (tail == "linear_fc2.bias") return &merge_fc2_b;
      return nullptr;
    }
    const std::string dp = "model.visual.deepstack_merger_list.";
    if (name.compare(0, dp.size(), dp) == 0) {
      size_t dot = name.find('.', dp.size());
      int ki = std::stoi(name.substr(dp.size(), dot - dp.size()));
      if (ki < 0 || (size_t)ki >= deep.size()) return nullptr;
      auto& d = deep[(size_t)ki];
      std::string tail = name.substr(dot + 1);
      if (tail == "norm.weight") return &d.norm_w;
      if (tail == "norm.bias") return &d.norm_b;
      if (tail == "linear_fc1.bias") return &d.fc1_b;
      if (tail == "linear_fc2.bias") return &d.fc2_b;
      return nullptr;
    }
    return nullptr;
  }
};

// Vision model: owns Weights. forward() runs patch projection, learned-grid
// resampling, 27 axial-RoPE MHA blocks, the merger, and the 3 deepstack
// mergers, returning LLM vision tokens + per-deepstack features.
//
//   pixels [Np, P] host-packed block-major patch rows (P = C*T*Pp*Pp, the
//     temporal frame duplicated for single images, values in [-1,1]);
//   grid Hp x Wp patch dims (both multiples of spatial_merge_size);
//   merged [Nv, O] (Nv = Np/merge^2) LLM vision tokens;
//   deep[k] [Nv, O] deepstack features for text layers 0..2.
// Single image per forward (B=1); the reference packs one image per call.
template <class ExecSpace, class Scalar>
class QwenImage21VisionModel {
 public:
  using V1 = Kokkos::View<Scalar*, ExecSpace>;
  using V2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using I1 = Kokkos::View<int*, ExecSpace>;
  using Weights = QwenImage21VisionWeights<ExecSpace, Scalar>;

  explicit QwenImage21VisionModel(const Config& cfg = Config()) {
    cfg.validate();
    cfg_ = cfg;
    weights_.allocate(cfg_);
    auto hv = Impl::qwen21vit_rope_inv_host(cfg_.head_dim(), cfg_.rope_theta);
    rope_inv_ = Kokkos::View<float*, ExecSpace>("qwen21vit_rope_inv", hv.size());
    auto hm = Kokkos::create_mirror_view(rope_inv_);
    for (size_t i = 0; i < hv.size(); ++i) hm(i) = hv[i];
    Kokkos::deep_copy(rope_inv_, hm);
  }

  const Config& config() const { return cfg_; }
  Weights& weights() { return weights_; }
  const Weights& weights() const { return weights_; }

  void forward(const DNNHandle<ExecSpace>& handle, const V2& pixels, int Hp,
               int Wp, V2& merged, std::vector<V2>& deep) const {
    cfg_.validate();
    const size_t H = (size_t)cfg_.hidden_dim, O = (size_t)cfg_.out_hidden_size;
    const size_t Nq = (size_t)cfg_.num_heads, Hd = (size_t)cfg_.head_dim(),
                 I = (size_t)cfg_.intermediate_dim;
    const size_t M = (size_t)cfg_.spatial_merge_size;
    if (Hp <= 0 || Wp <= 0 || Hp % (int)M != 0 || Wp % (int)M != 0)
      throw std::invalid_argument(
          "QwenImage21VisionModel::forward: grid must tile the merge");
    const size_t Np = (size_t)Hp * (size_t)Wp;
    const size_t Nv = Np / (M * M);
    if (pixels.extent(0) != Np ||
        pixels.extent(1) != (size_t)cfg_.patch_dim())
      throw std::invalid_argument(
          "QwenImage21VisionModel::forward: pixels must be [Np, patch_dim]");
    if (merged.extent(0) != Nv || merged.extent(1) != O)
      throw std::invalid_argument(
          "QwenImage21VisionModel::forward: merged must be [Nv, out]");
    if (deep.size() != cfg_.deepstack_visual_indexes.size())
      throw std::invalid_argument(
          "QwenImage21VisionModel::forward: deep must hold one view per "
          "deepstack index");
    for (const auto& d : deep)
      if (d.extent(0) != Nv || d.extent(1) != O)
        throw std::invalid_argument(
            "QwenImage21VisionModel::forward: deep views must be [Nv, out]");
    if (weights_.patch_W.extent(0) == 0)
      throw std::logic_error(
          "QwenImage21VisionModel::forward: weights empty");
    if (Np == 0) return;
    const float eps = static_cast<float>(cfg_.norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(Hd));

    // Learned-grid resample (host) + block-major rope positions (host).
    std::vector<int> interp_idx;
    std::vector<float> interp_wt;
    Impl::qwen21vit_interp_pos_host(Hp, Wp, cfg_.num_grid_per_side(), (int)M,
                                    interp_idx, interp_wt);
    const std::vector<int> pos_hw =
        Impl::qwen21vit_block_positions_host(Hp, Wp, (int)M);
    I1 d_idx("qwen21vit_interp_idx", interp_idx.size());
    {
      auto h = Kokkos::create_mirror_view(d_idx);
      for (size_t i = 0; i < interp_idx.size(); ++i)
        h(i) = interp_idx[i];
      Kokkos::deep_copy(d_idx, h);
    }
    Kokkos::View<float*, ExecSpace> d_wt("qwen21vit_interp_wt",
                                         interp_wt.size());
    {
      auto h = Kokkos::create_mirror_view(d_wt);
      for (size_t i = 0; i < interp_wt.size(); ++i) h(i) = interp_wt[i];
      Kokkos::deep_copy(d_wt, h);
    }
    I1 d_pos("qwen21vit_pos_hw", pos_hw.size());
    {
      auto h = Kokkos::create_mirror_view(d_pos);
      for (size_t i = 0; i < pos_hw.size(); ++i) h(i) = pos_hw[i];
      Kokkos::deep_copy(d_pos, h);
    }
    Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        pos2(d_pos.data(), Np, 2);

    using UView3 = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                                Kokkos::MemoryUnmanaged>;
    V2 x("qwen21vit_x", Np, H);  // patch + pos stream
    V2 n("qwen21vit_n", Np, H);
    V2 qkv("qwen21vit_qkv", Np, 3 * H);
    V2 q("qwen21vit_q", Np, H), k("qwen21vit_k", Np, H),
        v("qwen21vit_v", Np, H);
    V2 ao("qwen21vit_ao", Np, H);
    V2 o("qwen21vit_o", Np, H);
    V2 f1("qwen21vit_f1", Np, I), f2("qwen21vit_f2", Np, H);
    UView3 q3(q.data(), Np, Nq, Hd);
    UView3 k3(k.data(), Np, Nq, Hd);
    UView3 v3(v.data(), Np, Nq, Hd);
    UView3 ao3(ao.data(), Np, Nq, Hd);

    // Patch projection + learned-grid add (fused gather-add below).
    KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, pixels, weights_.patch_W,
                                            weights_.patch_b, x,
                                            "Qwen21Vit::patch");
    {
      auto table = weights_.pos_table;
      Kokkos::parallel_for(
          "Qwen21Vit::pos_add",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Np, H}),
          KOKKOS_LAMBDA(size_t p, size_t d) {
            float acc = static_cast<float>(x(p, d));
            for (int t = 0; t < 4; ++t)
              acc += d_wt(p * 4 + t) *
                     static_cast<float>(table(d_idx(p * 4 + t), d));
            x(p, d) = static_cast<Scalar>(acc);
          });
    }

    for (int l = 0; l < cfg_.depth; ++l) {
      const auto& L = weights_.layers[(size_t)l];
      KokkosDNN::layernorm_forward<ExecSpace, Scalar>(x, L.n1_w, L.n1_b, eps,
                                                      n, "Qwen21Vit::n1");
      KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, n, L.qkv_W, L.qkv_b,
                                              qkv, "Qwen21Vit::qkv");
      Impl::qwen21vit_split_qkv<ExecSpace>(qkv, q, k, v, "Qwen21Vit::split");
      Impl::qwen21vit_rope_inplace<ExecSpace>(q3, pos2, rope_inv_,
                                             "Qwen21Vit::rope_q");
      Impl::qwen21vit_rope_inplace<ExecSpace>(k3, pos2, rope_inv_,
                                             "Qwen21Vit::rope_k");
      KokkosDNN::attention_mha_bidirectional<ExecSpace, Scalar>(
          handle, q3, k3, v3, ao3, scale, "Qwen21Vit::attn");
      KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, ao, L.proj_W, L.proj_b,
                                              o, "Qwen21Vit::proj");
      // Residual adds in FP32 (encoder idiom).
      Kokkos::parallel_for(
          "Qwen21Vit::res1",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Np, H}),
          KOKKOS_LAMBDA(size_t p, size_t d) {
            x(p, d) = static_cast<Scalar>(static_cast<float>(x(p, d)) +
                                          static_cast<float>(o(p, d)));
          });
      KokkosDNN::layernorm_forward<ExecSpace, Scalar>(x, L.n2_w, L.n2_b, eps,
                                                      n, "Qwen21Vit::n2");
      KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, n, L.fc1_W, L.fc1_b,
                                              f1, "Qwen21Vit::fc1");
      KokkosDNN::gelu_forward<ExecSpace, Scalar>(
          f1, f1, KokkosDNN::GeluVariant::Tanh, "Qwen21Vit::gelu");
      KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, f1, L.fc2_W, L.fc2_b,
                                              f2, "Qwen21Vit::fc2");
      Kokkos::parallel_for(
          "Qwen21Vit::res2",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Np, H}),
          KOKKOS_LAMBDA(size_t p, size_t d) {
            x(p, d) = static_cast<Scalar>(static_cast<float>(x(p, d)) +
                                          static_cast<float>(f2(p, d)));
          });
      // Deepstack taps: merge the post-block stream for the configured
      // layers (reference deepstack_merger_list order).
      for (size_t di = 0; di < cfg_.deepstack_visual_indexes.size(); ++di) {
        if (cfg_.deepstack_visual_indexes[di] != l) continue;
        merge_rows(handle, x, Np, H, M, weights_.deep[di].norm_w,
                   weights_.deep[di].norm_b, weights_.deep[di].fc1_W,
                   weights_.deep[di].fc1_b, weights_.deep[di].fc2_W,
                   weights_.deep[di].fc2_b, deep[di], /*pre_group_norm=*/false,
                   "Qwen21Vit::deep" + std::to_string(di));
      }
    }
    merge_rows(handle, x, Np, H, M, weights_.merge_norm_w,
               weights_.merge_norm_b, weights_.merge_fc1_W,
               weights_.merge_fc1_b, weights_.merge_fc2_W, weights_.merge_fc2_b,
               merged, /*pre_group_norm=*/true, "Qwen21Vit::merger");
    Kokkos::fence();
  }

 private:
  Config cfg_;
  Weights weights_;
  Kokkos::View<float*, ExecSpace> rope_inv_;  // [Hd/4] axial freqs

  // 2x2-spatial merge + Linear GELU(exact-erf) Linear to out_hidden_size:
  // x [Np, H] block-major rows -> y [Nv, O] (Nv = Np/M^2). The merge grouping
  // is a zero-copy re-view (consecutive rows form one MxM block). Norm
  // placement differs: the main merger norms the ungrouped [Np, H] stream
  // first (pre_group_norm), while deepstack mergers norm the grouped [Nv, G]
  // rows (postshuffle). Both norms are biased. Reference
  // Qwen3VLVisionPatchMerger.act_fn is nn.GELU() (exact erf) for both the
  // main merger and every deepstack merger; the ViT block MLPs stay
  // gelu_pytorch_tanh (see the Tanh call in forward()).
  void merge_rows(const DNNHandle<ExecSpace>& handle, const V2& x, size_t Np,
                  size_t H, size_t M, const V1& norm_w, const V1& norm_b,
                  const V2& fc1_W, const V1& fc1_b, const V2& fc2_W,
                  const V1& fc1_b2, V2& y, bool pre_group_norm,
                  const std::string& tag) const {
    const size_t Nv = Np / (M * M), G = H * M * M;
    using UV2 = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    V2 tmp("qwen21vit_merge_tmp", Nv, G);
    if (pre_group_norm) {
      V2 xn("qwen21vit_merge_xn", Np, H);
      KokkosDNN::layernorm_forward<ExecSpace, Scalar>(x, norm_w, norm_b,
                                                      (float)cfg_.norm_eps,
                                                      xn, tag + "_norm");
      UV2 xng(xn.data(), Nv, G);
      Kokkos::deep_copy(tmp, xng);
    } else {
      UV2 xg(const_cast<Scalar*>(x.data()), Nv, G);
      Kokkos::deep_copy(tmp, xg);
      KokkosDNN::layernorm_forward<ExecSpace, Scalar>(tmp, norm_w, norm_b,
                                                      (float)cfg_.norm_eps,
                                                      tmp, tag + "_norm");
    }
    V2 h1("qwen21vit_merge_h1", Nv, G);
    KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, tmp, fc1_W, fc1_b, h1,
                                            tag + "_fc1");
    KokkosDNN::gelu_forward<ExecSpace, Scalar>(
        h1, h1, KokkosDNN::GeluVariant::ExactErf, tag + "_gelu");
    KokkosDNN::gemm_bias<ExecSpace, Scalar>(handle, h1, fc2_W, fc1_b2, y,
                                            tag + "_fc2");
  }
};

}  // namespace QwenImage21Vision
}  // namespace KokkosDNN

#endif
