// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_DINOV3_MODEL_HPP
#define KOKKOSDNN_DINOV3_MODEL_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "KokkosDNN_Activation.hpp"
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Dinov3Config.hpp"
#include "KokkosDNN_Dinov3Weights.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_LayerNorm.hpp"
#include "KokkosDNN_LayerScale.hpp"
#include "KokkosDNN_PatchEmbed.hpp"
#include "KokkosDNN_RoPE.hpp"
#include "KokkosDNN_Tokens.hpp"

namespace KokkosDNN {
namespace Dinov3 {

// Single-sequence DINOv3 ViT image encoder. Owns the TPL handle, the cached
// 2D axial RoPE inv_freq (head_dim/4 entries), and persistent scratch, mirroring
// the structural pattern of Llama::Model (weights/handle/scratch + forward()).
//
// forward(pixels [C,H,W] LayoutRight) -> token matrix [T,D] LayoutRight with
// T = 1 + num_register_tokens + Np, Np = (H/patch)*(W/patch):
//   row 0              = CLS token,
//   rows [1, 1+R)      = register tokens,
//   rows [1+R, T)      = row-major patch tokens (p = hi*Wp + wi).
// cls_token() aliases row 0; patch_tokens() aliases the patch rows.
//
// Per-forward op sequence (one full pass, no KV-cache, no sampling):
//   patchembed_forward (im2col + gemm_bias, stride == patch_size)
//   tokens_assemble_single (cat [cls, registers, patches])
//   per block:
//     layernorm_forward (norm1) -> q/k/v via gemm_bias
//       (k uses an empty bias view: the k projection is unbiased)
//     reshape [T,D] -> [T,H,Dh] views
//     rope2d_axial_inplace (dynamic Hp/Wp, prefix_skip = 1+R, CLS/registers
//       untouched)
//     attention_mha_bidirectional (default scale = 1/sqrt(head_dim))
//     o_proj via gemm_bias on the [T,D] alias of the attention output
//     layerscale_residual_forward (gamma1)
//     layernorm_forward (norm2) -> up via gemm_bias -> gelu_forward
//       (ExactErf default = HF gelu) -> down via gemm_bias
//     layerscale_residual_forward (gamma2)
//   final layernorm_forward.
//
// Steady state performs no View allocations: all T-sized temporaries are
// capacity buffers carved via subview (grown on demand, never shrunk); the
// [1,Np,D] patchembed output is rebuilt only when the patch count changes.
// The returned view aliases persistent scratch: valid until the next
// forward() call.
//
// Single-sequence only (documented choice): patchembed offers a batched path
// and tokens/layerscale offer rank-3 overloads, but layernorm_forward and
// gemm_bias are rank-2-only, so a batched forward would need flatten+loop
// wrappers rather than falling out naturally. Not provided.
//
// Assumed sibling API (KokkosDNN_Dinov3Config.hpp / KokkosDNN_Dinov3Weights.hpp):
//   Config { patch_size, hidden_size, num_hidden_layers, num_attention_heads,
//            head_dim, intermediate_size, num_register_tokens, image_size,
//            num_channels, layer_norm_eps, rope_theta,
//            validate(), vitb16(), tiny_test() };
//   Weights<ExecSpace, Scalar> { patch_w [D,CKK], patch_b [D], cls_tok [D],
//            reg_tok [R,D], final_norm_w/b [D],
//            blocks[i] { wq,bq, wk, wv,bv, wo,bo, wup,bup, wdown,bdown,
//                        norm1_w,norm1_b, norm2_w,norm2_b, gamma1,gamma2 } };
//   DefaultScalar (= float); random_init(cfg, w); load_safetensors(cfg, files, w).
// The k projection is unbiased: the model passes its own empty (extent-0)
// bias view to gemm_bias rather than relying on a weights field for it, so it
// compiles whether or not the block struct carries a k-bias member.
template <class ExecSpace, class Scalar = DefaultScalar>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using ImgR = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using Seq3R = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<Scalar*, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;
  DNNHandle<ExecSpace> handle;
  Kokkos::View<float*, ExecSpace> inv_freq;  // 2D axial RoPE inv_freq (Dh/4)
  View1 no_bias_;  // empty bias (extent 0) => gemm_bias unbiased path (k proj)

  Model() = default;
  Model(const Config& c, const Weights<ExecSpace, Scalar>& w)
      : cfg(c), weights(w) {
    // Validate before renting any scratch: incompatible dims fail here, not
    // in a device kernel or a large allocation.
    cfg.validate();
    if ((size_t)cfg.num_attention_heads * (size_t)cfg.head_dim !=
        (size_t)cfg.hidden_size)
      throw std::invalid_argument(
          "Dinov3::Model: num_attention_heads * head_dim must equal "
          "hidden_size (packed [T,D] qkv buffers assume MHA layout)");
    inv_freq = rope2d_inv_freq<ExecSpace>(cfg.head_dim, cfg.rope_theta);
    // Pre-size scratch for the square cfg.image_size when it tiles exactly;
    // otherwise the first forward() sizes it for the actual image.
    const size_t P = (size_t)cfg.patch_size, S = (size_t)cfg.image_size;
    if (P > 0 && S >= P && S % P == 0) {
      const size_t g = S / P, Np = g * g;
      ensure_scratch(1 + (size_t)cfg.num_register_tokens + Np, Np);
    }
  }

  // Single-image encoder forward: pixels [C,H,W] -> [T,D] token matrix.
  // H and W must each tile patch_size exactly (patchembed has no padding).
  // The returned view aliases persistent scratch (see class docs).
  MatR forward(const ImgR& pixels) {
    const size_t C = pixels.extent(0);
    const size_t H = pixels.extent(1);
    const size_t W = pixels.extent(2);
    if (C != (size_t)cfg.num_channels)
      throw std::invalid_argument(
          "Dinov3::Model::forward: channel mismatch (got C=" +
          std::to_string(C) + ", expected " +
          std::to_string(cfg.num_channels) + ")");
    const size_t P = (size_t)cfg.patch_size;
    if (P == 0)
      throw std::invalid_argument(
          "Dinov3::Model::forward: patch_size is 0");
    if (H % P != 0 || W % P != 0)
      throw std::invalid_argument(
          "Dinov3::Model::forward: H and W must tile patch_size exactly "
          "(got H=" + std::to_string(H) + " W=" + std::to_string(W) +
          " patch=" + std::to_string(P) + ")");
    const size_t Hp = H / P, Wp = W / P, Np = Hp * Wp;
    const size_t R = (size_t)cfg.num_register_tokens;
    const size_t T = 1 + R + Np;
    ensure_scratch(T, Np);

    // Hoisted dims (device lambdas must not capture `this`/`cfg`).
    const size_t D = (size_t)cfg.hidden_size;
    const size_t DH = (size_t)cfg.head_dim;
    const float EPS = cfg.layer_norm_eps;
    const size_t NL = (size_t)cfg.num_hidden_layers;
    const size_t prefix_skip = 1 + R;
    const int Hpi = static_cast<int>(Hp), Wpi = static_cast<int>(Wp);
    auto inv = inv_freq;
    auto nobias = no_bias_;

    // Carve T-row working views out of the capacity buffers. Subviews share
    // the allocation, so none of this allocates in steady state.
    auto h = Kokkos::subview(h_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto n = Kokkos::subview(n_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto q2 = Kokkos::subview(q2_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto k2 = Kokkos::subview(k2_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto v2 = Kokkos::subview(v2_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto up = Kokkos::subview(up_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto down =
        Kokkos::subview(down_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto out =
        Kokkos::subview(out_, std::make_pair(size_t(0), T), Kokkos::ALL);
    auto q3 = Kokkos::subview(q3_, std::make_pair(size_t(0), T), Kokkos::ALL,
                              Kokkos::ALL);
    auto k3 = Kokkos::subview(k3_, std::make_pair(size_t(0), T), Kokkos::ALL,
                              Kokkos::ALL);
    auto v3 = Kokkos::subview(v3_, std::make_pair(size_t(0), T), Kokkos::ALL,
                              Kokkos::ALL);
    auto o3 = Kokkos::subview(o3_, std::make_pair(size_t(0), T), Kokkos::ALL,
                              Kokkos::ALL);

    // Unmanaged aliases over contiguous storage (no allocation, no copies):
    // single image as batch-1 NCHW for patchembed, patch tokens flattened
    // for token assembly, and the [T,H,Dh] attention output as [T,D] for the
    // o_proj GEMM (NH*Dh == D, checked in the constructor, so the alias is
    // the identity permutation).
    Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace> img4(
        pixels.data(), size_t(1), C, H, W);
    MatR patch2(patch3_.data(), Np, D);
    MatR oflat(o3.data(), T, D);

    patchembed_forward<ExecSpace, Scalar>(
        handle, img4, weights.patch_w, weights.patch_b, patch3_,
        static_cast<int>(P), static_cast<int>(P));
    tokens_assemble_single<ExecSpace, Scalar>(weights.cls_tok, weights.reg_tok,
                                              patch2, h);

    for (size_t l = 0; l < NL; ++l) {
      const auto& b = weights.blocks[l];
      layernorm_forward<ExecSpace, Scalar>(h, b.norm1_w, b.norm1_b, EPS, n);
      gemm_bias<ExecSpace, Scalar>(handle, n, b.wq, b.bq, q2);
      gemm_bias<ExecSpace, Scalar>(handle, n, b.wk, nobias, k2);
      gemm_bias<ExecSpace, Scalar>(handle, n, b.wv, b.bv, v2);
      Kokkos::parallel_for(
          "dinov3_reshape_q",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
          KOKKOS_LAMBDA(size_t t, size_t i) {
            q3(t, i / DH, i % DH) = q2(t, i);
          });
      Kokkos::parallel_for(
          "dinov3_reshape_kv",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
          KOKKOS_LAMBDA(size_t t, size_t i) {
            k3(t, i / DH, i % DH) = k2(t, i);
            v3(t, i / DH, i % DH) = v2(t, i);
          });
      rope2d_axial_inplace<ExecSpace, Scalar>(q3, k3, inv, Hpi, Wpi,
                                              prefix_skip);
      attention_mha_bidirectional<ExecSpace, Scalar>(handle, q3, k3, v3, o3);
      gemm_bias<ExecSpace, Scalar>(handle, oflat, b.wo, b.bo, n);
      layerscale_residual_forward<ExecSpace, Scalar>(h, n, b.gamma1, h);
      layernorm_forward<ExecSpace, Scalar>(h, b.norm2_w, b.norm2_b, EPS, n);
      gemm_bias<ExecSpace, Scalar>(handle, n, b.wup, b.bup, up);
      gelu_forward<ExecSpace, Scalar>(up, up);  // ExactErf default = HF gelu
      gemm_bias<ExecSpace, Scalar>(handle, up, b.wdown, b.bdown, down);
      layerscale_residual_forward<ExecSpace, Scalar>(h, down, b.gamma2, h);
    }
    layernorm_forward<ExecSpace, Scalar>(h, weights.final_norm_w,
                                         weights.final_norm_b, EPS, out);
    last_T_ = T;
    last_Hp_ = Hp;
    last_Wp_ = Wp;
    return out;
  }

  // Token count / patch grid of the most recent forward() (0 before the
  // first call).
  size_t token_count() const { return last_T_; }
  size_t grid_hp() const { return last_Hp_; }
  size_t grid_wp() const { return last_Wp_; }

  // CLS row (row 0) of the most recent forward(). Aliases persistent scratch:
  // valid until the next forward() call.
  Kokkos::View<Scalar*, ExecSpace> cls_token() const {
    return Kokkos::View<Scalar*, ExecSpace>(out_.data(),
                                            (size_t)cfg.hidden_size);
  }

  // Patch rows ([1+R, T), row-major) of the most recent forward(). Aliases
  // persistent scratch: valid until the next forward() call.
  MatR patch_tokens() const {
    const size_t D = (size_t)cfg.hidden_size;
    const size_t t0 = 1 + (size_t)cfg.num_register_tokens;
    const size_t nt = (last_T_ > t0) ? (last_T_ - t0) : 0;
    return MatR(out_.data() + t0 * D, nt, D);
  }

 private:
  // [1,Np,D] patchembed output, exact-sized (rebuilt when Np changes: a
  // ranged dim-1 subview of LayoutRight would be strided and patchembed
  // requires a contiguous output).
  Seq3R patch3_;
  // Grow-only T-capacity buffers (never shrunk; steady state = zero
  // allocations). q3_/k3_/v3_/o3_ are [T,NH,DH]; the rest are [T,D]/[T,I].
  MatR h_, n_, q2_, k2_, v2_, up_, down_, out_;
  Seq3R q3_, k3_, v3_, o3_;
  size_t cap_T_ = 0;
  size_t last_T_ = 0, last_Hp_ = 0, last_Wp_ = 0;

  void ensure_scratch(size_t T, size_t Np) {
    const size_t D = (size_t)cfg.hidden_size;
    const size_t NH = (size_t)cfg.num_attention_heads;
    const size_t DH = (size_t)cfg.head_dim;
    const size_t I = (size_t)cfg.intermediate_size;
    if (patch3_.extent(0) != 1 || patch3_.extent(1) != Np ||
        patch3_.extent(2) != D)
      patch3_ = Seq3R("dinov3_patch", size_t(1), Np, D);
    if (T <= cap_T_ && h_.extent(0) >= T && h_.extent(1) == D &&
        n_.extent(1) == D && q2_.extent(1) == D && k2_.extent(1) == D &&
        v2_.extent(1) == D && up_.extent(1) == I && down_.extent(1) == D &&
        out_.extent(0) >= T && out_.extent(1) == D && q3_.extent(1) == NH &&
        q3_.extent(2) == DH && k3_.extent(1) == NH && k3_.extent(2) == DH &&
        v3_.extent(1) == NH && v3_.extent(2) == DH && o3_.extent(1) == NH &&
        o3_.extent(2) == DH)
      return;
    h_ = MatR("dinov3_h", T, D);
    n_ = MatR("dinov3_n", T, D);
    q2_ = MatR("dinov3_q2", T, D);
    k2_ = MatR("dinov3_k2", T, D);
    v2_ = MatR("dinov3_v2", T, D);
    q3_ = Seq3R("dinov3_q3", T, NH, DH);
    k3_ = Seq3R("dinov3_k3", T, NH, DH);
    v3_ = Seq3R("dinov3_v3", T, NH, DH);
    o3_ = Seq3R("dinov3_o3", T, NH, DH);
    up_ = MatR("dinov3_up", T, I);
    down_ = MatR("dinov3_down", T, D);
    out_ = MatR("dinov3_out", T, D);
    cap_T_ = T;
  }
};

}  // namespace Dinov3
}  // namespace KokkosDNN

#endif
