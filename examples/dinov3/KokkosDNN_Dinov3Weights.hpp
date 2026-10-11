#ifndef KOKKOSDNN_DINOV3_WEIGHTS_HPP
#define KOKKOSDNN_DINOV3_WEIGHTS_HPP

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include "KokkosDNN_Dinov3Config.hpp"

namespace KokkosDNN {
namespace Dinov3 {

// Default compute/storage scalar for the DINOv3 example: float, the
// `facebook/dinov3-vitb16-pretrain-lvd1689m` checkpoint dtype (all F32).
using DefaultScalar = float;

// One encoder block's weights (all row-major, device views). The DINOv3
// ViT attention uses biased q/v/o linears; k is unbiased, so there is no
// bk field (callers pass an empty bias view to gemm_bias for k).
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wq, wk, wv, wo;
  Kokkos::View<Scalar*, ExecSpace> bq, bv, bo;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> wup, wdown;
  Kokkos::View<Scalar*, ExecSpace> bup, bdown;
  Kokkos::View<Scalar*, ExecSpace> norm1_w, norm1_b, norm2_w, norm2_b;
  Kokkos::View<Scalar*, ExecSpace> gamma1, gamma2;
};

// Full model weights. Device views in Scalar; the safetensors loader
// converts on-disk F32/F16/BF16 to Scalar automatically.
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> patch_w, reg_tok;
  Kokkos::View<Scalar*, ExecSpace> patch_b, cls_tok;
  Kokkos::View<Scalar*, ExecSpace> final_norm_w, final_norm_b;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;

  void allocate(const Config& cfg);
};

// Allocate all views for a config (uninitialized contents).
template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  const size_t D   = (size_t)cfg.hidden_size;
  const size_t CKK = (size_t)cfg.num_channels * (size_t)cfg.patch_size *
                     (size_t)cfg.patch_size;
  const size_t R = (size_t)cfg.num_register_tokens;
  const size_t H = (size_t)cfg.intermediate_size;
  patch_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("patch_w",
                                                                  D, CKK);
  patch_b = Kokkos::View<Scalar*, ExecSpace>("patch_b", D);
  cls_tok = Kokkos::View<Scalar*, ExecSpace>("cls_tok", D);
  reg_tok =
      Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("reg_tok", R, D);
  final_norm_w = Kokkos::View<Scalar*, ExecSpace>("final_norm_w", D);
  final_norm_b = Kokkos::View<Scalar*, ExecSpace>("final_norm_b", D);
  blocks.resize((size_t)cfg.num_hidden_layers);
  for (size_t l = 0; l < blocks.size(); ++l) {
    auto& b  = blocks[l];
    b.wq     = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wq", D, D);
    b.bq  = Kokkos::View<Scalar*, ExecSpace>("bq", D);
    b.wk  = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wk", D, D);
    b.wv  = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wv", D, D);
    b.bv  = Kokkos::View<Scalar*, ExecSpace>("bv", D);
    b.wo  = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wo", D, D);
    b.bo  = Kokkos::View<Scalar*, ExecSpace>("bo", D);
    b.wup = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wup", H, D);
    b.bup   = Kokkos::View<Scalar*, ExecSpace>("bup", H);
    b.wdown = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "wdown", D, H);
    b.bdown   = Kokkos::View<Scalar*, ExecSpace>("bdown", D);
    b.norm1_w = Kokkos::View<Scalar*, ExecSpace>("norm1_w", D);
    b.norm1_b = Kokkos::View<Scalar*, ExecSpace>("norm1_b", D);
    b.norm2_w = Kokkos::View<Scalar*, ExecSpace>("norm2_w", D);
    b.norm2_b = Kokkos::View<Scalar*, ExecSpace>("norm2_b", D);
    b.gamma1  = Kokkos::View<Scalar*, ExecSpace>("gamma1", D);
    b.gamma2  = Kokkos::View<Scalar*, ExecSpace>("gamma2", D);
  }
}

// Fill with small random values (for shape/forward tests without weights).
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load the HF `facebook/dinov3-vitb16-pretrain-lvd1689m` safetensors
// checkpoint(s) into device views.
//   files : one or more .safetensors files (tensors are looked up across
//           all files; the ViT-B checkpoint is a single shard).
// On-disk dtypes F32/F16/BF16 are all converted to Scalar. Files are
// mapped read-only (mmap; bounded chunked reads where mmap is unavailable)
// and each tensor streams from the mapping through a small (~1 MiB) host
// staging buffer — no full-tensor/host-F32 expansion. Loading is lazy
// per-tensor in call order (each tensor converts/copies once, files stay
// mapped for the call). Throws on missing tensors or shape mismatch.
// Expected HF names:
//   embeddings.patch_embeddings.weight [D,C,K,K] (flattened to [D,C*K*K]
//     in c*K*K+kh*K+kw order, matching the patchembed_forward im2col
//     layout), embeddings.patch_embeddings.bias [D],
//   embeddings.cls_token [1,1,D] (-> [D]),
//   embeddings.register_tokens [1,R,D] (-> [R,D]),
//   layer.{l}.norm1.weight/bias, layer.{l}.attention.q_proj.weight/bias,
//   layer.{l}.attention.k_proj.weight (unbiased), layer.{l}.attention.
//   v_proj.weight/bias, layer.{l}.attention.o_proj.weight/bias,
//   layer.{l}.norm2.weight/bias, layer.{l}.mlp.up_proj.weight [H,D]/bias,
//   layer.{l}.mlp.down_proj.weight [D,H]/bias,
//   layer.{l}.layer_scale1.lambda1, layer.{l}.layer_scale2.lambda1,
//   norm.weight/bias (final LayerNorm).
// embeddings.mask_token exists in the checkpoint but is unused for
// inference and is not loaded.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_safetensors(const Config& cfg, const std::vector<std::string>& files,
                      Weights<ExecSpace, Scalar>& w);

// Raw FP32 host buffer (row-major) converted into a device matrix view.
template <class ExecSpace, class Scalar>
void copy_host_to_device(
    const std::vector<float>& host, size_t rows, size_t cols,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& dst);

}  // namespace Dinov3
}  // namespace KokkosDNN

#endif
