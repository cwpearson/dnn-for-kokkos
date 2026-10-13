#ifndef KOKKOSDNN_SANAATTENTION_HPP
#define KOKKOSDNN_SANAATTENTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Impl {

// Weight-orientation helper for the fused Sana ops.
//
// The task text writes Sana linears in math form (x @ W with
// W : [in, out], e.g. qkv_w [D, 3D]), while the KokkosDNN GEMM family
// stores weights HF-Linear style ([out, in], e.g. qkv_w [3D, D]) and
// computes Y = X * W^T + b. Both spellings describe the same math, so
// the non-square Sana weights (qkv [D,3D]<->[3D,D], kv [D,2D]<->[2D,D])
// are auto-detected by extents: [3D,D]/[2D,D] takes the gemm/gemm_bias
// fast path (cuBLAS dispatch included), [D,3D]/[D,2D] runs an exact
// FP32 device GEMM with the indices swapped. Square weights
// (q_w, proj_w : [D, D]) are ambiguous by extents and always use the
// repo convention (HF layout, Y = X * W^T + b).
enum class SanaWeightOri { HF, Math };

inline SanaWeightOri sana_detect_ori(size_t e0, size_t e1, size_t in_dim,
                                     size_t out_dim, const char* what) {
  if (e0 == out_dim && e1 == in_dim) return SanaWeightOri::HF;
  if (e0 == in_dim && e1 == out_dim) return SanaWeightOri::Math;
  throw std::invalid_argument(what);
}

// Y = X * W + b with W : [K, N] math layout and optional bias
// (empty b means zeros). FP32 accumulation, cast back to Scalar.
// Used only for math-layout weights; HF-layout weights go through
// gemm()/gemm_bias() so the TPL dispatch ladder keeps applying.
template <class ExecSpace, class Scalar>
void sana_gemm_math(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& X,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& W,
    const Kokkos::View<Scalar*, ExecSpace>& b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& Y,
    const std::string& label) {
  const size_t M = X.extent(0), K = X.extent(1), N = W.extent(1);
  if (M == 0 || N == 0 || K == 0) return;
  const bool biased = (b.extent(0) != 0);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {M, N}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k)
          acc += static_cast<float>(X(i, k)) * static_cast<float>(W(k, j));
        if (biased) acc += static_cast<float>(b(j));
        Y(i, j) = static_cast<Scalar>(acc);
      });
}

// Rank-2 LiteLA body: x [T,D] -> out [T,D]. H/Dh, orientations, and
// weight shapes are validated by the public wrappers.
template <class ExecSpace, class Scalar>
void sana_litela_2d_body(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& qkv_w,
    SanaWeightOri qkv_ori,
    const Kokkos::View<Scalar*, ExecSpace>& q_norm_w,
    const Kokkos::View<Scalar*, ExecSpace>& k_norm_w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj_w,
    const Kokkos::View<Scalar*, ExecSpace>& proj_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    size_t H, float eps, float norm_eps, const std::string& label) {
  const size_t T = x.extent(0), D = x.extent(1);
  const size_t Dh = D / H;
  if (T == 0 || D == 0) return;
  const bool has_q = (q_norm_w.extent(0) != 0);
  const bool has_k = (k_norm_w.extent(0) != 0);

  // 1. qkv = x @ W (unbiased GEMM).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> qkv("KokkosDNN::sana_qkv",
                                                             T, 3 * D);
  if (qkv_ori == SanaWeightOri::HF) {
    gemm(handle, static_cast<Scalar>(1), x, qkv_w, static_cast<Scalar>(0), qkv,
         label + "_qkv");
  } else {
    Kokkos::View<Scalar*, ExecSpace> no_bias("KokkosDNN::sana_nobias", 0);
    sana_gemm_math<ExecSpace, Scalar>(x, qkv_w, no_bias, qkv, label + "_qkv");
  }

  // 2. Split q/k/v, RMSNorm (or Identity when the weight is empty),
  //    reshape to heads, ReLU q/k. RMSNorm is over the full D per token.
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> Q("KokkosDNN::sana_Q", T,
                                                            H, Dh),
      K("KokkosDNN::sana_K", T, H, Dh), V("KokkosDNN::sana_V", T, H, Dh);
  Kokkos::parallel_for(
      label + "_norm",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        float qss = 0, kss = 0;
        if (has_q)
          for (size_t d = 0; d < D; ++d) {
            const float v = static_cast<float>(qkv(t, d));
            qss += v * v;
          }
        if (has_k)
          for (size_t d = 0; d < D; ++d) {
            const float v = static_cast<float>(qkv(t, D + d));
            kss += v * v;
          }
        const float qinv =
            has_q ? 1.0f / sqrt(qss / static_cast<float>(D) + norm_eps) : 1.0f;
        const float kinv =
            has_k ? 1.0f / sqrt(kss / static_cast<float>(D) + norm_eps) : 1.0f;
        for (size_t e = 0; e < Dh; ++e) {
          const size_t idx = h * Dh + e;
          float qv = static_cast<float>(qkv(t, idx));
          float kv = static_cast<float>(qkv(t, D + idx));
          if (has_q) qv = qv * qinv * static_cast<float>(q_norm_w(idx));
          if (has_k) kv = kv * kinv * static_cast<float>(k_norm_w(idx));
          Q(t, h, e) = static_cast<Scalar>(qv > 0.0f ? qv : 0.0f);
          K(t, h, e) = static_cast<Scalar>(kv > 0.0f ? kv : 0.0f);
          V(t, h, e) = qkv(t, 2 * D + idx);
        }
      });

  // 3. Per-head statistics with the padded-ones row folded in:
  //    Ksum[e] = sum_n K[n,e] and VK[d,e] = sum_n V[n,d]*K[n,e].
  //    (The pad row V_pad = 1 contributes exactly Ksum to the VK block
  //    and Ksum to the denominator, i.e. the standard linear-attention
  //    denominator sum_e Ksum[e]*q[e].)
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> Ksum("KokkosDNN::sana_ks",
                                                             H, Dh);
  Kokkos::parallel_for(
      label + "_ksum",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {H, Dh}),
      KOKKOS_LAMBDA(size_t h, size_t e) {
        float s = 0;
        for (size_t n = 0; n < T; ++n) s += static_cast<float>(K(n, h, e));
        Ksum(h, e) = s;
      });
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> VK("KokkosDNN::sana_vk",
                                                             H, Dh, Dh);
  Kokkos::parallel_for(
      label + "_vk",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {H, Dh, Dh}),
      KOKKOS_LAMBDA(size_t h, size_t d, size_t e) {
        float s = 0;
        for (size_t n = 0; n < T; ++n)
          s += static_cast<float>(V(n, h, d)) * static_cast<float>(K(n, h, e));
        VK(h, d, e) = s;
      });

  // 4. out[n,d] = (sum_e VK[d,e] q[n,e]) / (sum_e Ksum[e] q[n,e] + eps).
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> O3("KokkosDNN::sana_O",
                                                             T, H, Dh);
  Kokkos::parallel_for(
      label + "_mix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {T, H, Dh}),
      KOKKOS_LAMBDA(size_t n, size_t h, size_t d) {
        float num = 0, den = eps;
        for (size_t e = 0; e < Dh; ++e) {
          const float q = static_cast<float>(Q(n, h, e));
          num += VK(h, d, e) * q;
          den += Ksum(h, e) * q;
        }
        O3(n, h, d) = static_cast<Scalar>(num / den);
      });

  // 5. Reshape [T,H,Dh] -> [T,D], then y = out @ proj_w + proj_b.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> flat("KokkosDNN::sana_flat",
                                                              T, D);
  Kokkos::parallel_for(
      label + "_flat",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t n, size_t h) {
        for (size_t e = 0; e < Dh; ++e) flat(n, h * Dh + e) = O3(n, h, e);
      });
  gemm_bias(handle, flat, proj_w, proj_b, out, label + "_proj");
}

// Rank-2 cross-attention body: x [T,D], cond [L,D] -> out [T,D].
template <class ExecSpace, class Scalar>
void sana_cross_2d_body(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& q_w,
    const Kokkos::View<Scalar*, ExecSpace>& q_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kv_w,
    SanaWeightOri kv_ori, const Kokkos::View<Scalar*, ExecSpace>& kv_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj_w,
    const Kokkos::View<Scalar*, ExecSpace>& proj_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    size_t H, const std::string& label) {
  const size_t T = x.extent(0), D = x.extent(1), L = cond.extent(0);
  const size_t Dh = D / H;
  if (T == 0 || D == 0) return;

  // 1. q = x @ Q + b, kv = cond @ KV + b, then split k (first D) / v.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> qlin("KokkosDNN::sana_ql",
                                                              T, D),
      kvlin("KokkosDNN::sana_kvl", L, 2 * D);
  gemm_bias(handle, x, q_w, q_b, qlin, label + "_q");
  if (kv_ori == SanaWeightOri::HF) {
    gemm_bias(handle, cond, kv_w, kv_b, kvlin, label + "_kv");
  } else {
    sana_gemm_math<ExecSpace, Scalar>(cond, kv_w, kv_b, kvlin, label + "_kv");
  }
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> q3("KokkosDNN::sana_q3",
                                                              T, H, Dh),
      k3("KokkosDNN::sana_k3", L, H, Dh), v3("KokkosDNN::sana_v3", L, H, Dh),
      o3("KokkosDNN::sana_o3", T, H, Dh);
  Kokkos::parallel_for(
      label + "_split_q",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        for (size_t e = 0; e < Dh; ++e) q3(t, h, e) = qlin(t, h * Dh + e);
      });
  Kokkos::parallel_for(
      label + "_split_kv",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {L, H}),
      KOKKOS_LAMBDA(size_t l, size_t h) {
        for (size_t e = 0; e < Dh; ++e) {
          k3(l, h, e) = kvlin(l, h * Dh + e);
          v3(l, h, e) = kvlin(l, D + h * Dh + e);
        }
      });

  // 2. Dense bidirectional softmax attention (no mask, no qk_norm),
  //    scale 1/sqrt(Dh). Reuses the streaming kernel (cuDNN fallback
  //    included) shared with the ViT/DINOv3 path.
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  attention_mha_bidirectional<ExecSpace, Scalar>(handle, q3, k3, v3, o3, scale,
                                                label + "_attn");

  // 3. Reshape + output projection.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> flat("KokkosDNN::sana_cflat",
                                                              T, D);
  Kokkos::parallel_for(
      label + "_flat",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t h) {
        for (size_t e = 0; e < Dh; ++e) flat(t, h * Dh + e) = o3(t, h, e);
      });
  gemm_bias(handle, flat, proj_w, proj_b, out, label + "_proj");
}

}  // namespace Impl

// Sana linear self-attention (LiteLA, NVlabs/Sana
// diffusion/model/nets/sana_blocks.py) for SanaMS_600M_P1_D28
// (hidden 1152, depth 28, self-attn linear heads = 36, head dim = 32;
// cross-attn heads = 16, head dim = 72).
//
// Per image, with x [T, D]:
//   qkv  = x @ qkv_w                      (unbiased GEMM, qkv_w has no bias)
//   q, k = RMSNorm(qkv[...]) or Identity  (empty q/k_norm_w selects
//                                          Identity: Sana 600M qk_norm=false)
//   v    = qkv rest (no norm)
//   reshape to [T, H, Dh], q = ReLU(q), k = ReLU(k)
//   Ksum[e]   = sum_n k[n, e]
//   VK[d, e]  = sum_n v[n, d] * k[n, e]   (V^T @ K per head; the padded
//                                          ones row contributes Ksum, i.e.
//                                          the denominator below)
//   o[n, d] = (sum_e VK[d,e] q[n,e]) / (sum_e Ksum[e] q[n,e] + eps)
//   out = o @ proj_w + proj_b            (via gemm_bias)
//
// Scalar-agnostic (float, bhalf_t); every accumulation is FP32 and the
// result is cast back to Scalar. Views are LayoutRight rank-2; norm and
// projection biases accept empty views (extent 0) for the unbiased path.
//
// Weight layout: non-square qkv_w accepts either HF-Linear [3D, D] or
// math [D, 3D] (auto-detected by extents); square proj_w [D, D] follows
// the repo HF convention (Y = X * W^T + b). num_heads == 0 selects the
// Sana self-attention rule H = D / 32 (Dh = 32); pass an explicit head
// count otherwise (tests use D = 8, H = 2). eps defaults to the LiteLA
// 1e-15; norm_eps (RMSNorm) defaults to 1e-6.
template <class ExecSpace, class Scalar>
void linear_attention_lite_la(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& qkv_w,
    const Kokkos::View<Scalar*, ExecSpace>& q_norm_w,
    const Kokkos::View<Scalar*, ExecSpace>& k_norm_w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj_w,
    const Kokkos::View<Scalar*, ExecSpace>& proj_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    size_t num_heads = 0, float eps = 1e-15f, float norm_eps = 1e-6f,
    const std::string& label = "KokkosDNN::sana_litela") {
  const size_t T = x.extent(0), D = x.extent(1);
  if (out.extent(0) != T || out.extent(1) != D)
    throw std::invalid_argument("linear_attention_lite_la: out mismatch");
  if (T == 0 || D == 0) return;
  size_t H = num_heads;
  if (H == 0) {
    if (D % 32 != 0)
      throw std::invalid_argument(
          "linear_attention_lite_la: D not divisible by 32, pass num_heads");
    H = D / 32;
  }
  if (H == 0 || D % H != 0)
    throw std::invalid_argument("linear_attention_lite_la: D % H != 0");
  const Impl::SanaWeightOri ori = Impl::sana_detect_ori(
      qkv_w.extent(0), qkv_w.extent(1), D, 3 * D,
      "linear_attention_lite_la: qkv_w must be [3D,D] or [D,3D]");
  const bool has_q = (q_norm_w.extent(0) != 0);
  const bool has_k = (k_norm_w.extent(0) != 0);
  if ((has_q && q_norm_w.extent(0) != D) || (has_k && k_norm_w.extent(0) != D))
    throw std::invalid_argument("linear_attention_lite_la: norm weight mismatch");
  if (proj_w.extent(0) != D || proj_w.extent(1) != D)
    throw std::invalid_argument("linear_attention_lite_la: proj_w mismatch");
  if (proj_b.extent(0) != 0 && proj_b.extent(0) != D)
    throw std::invalid_argument("linear_attention_lite_la: proj_b mismatch");
  Impl::sana_litela_2d_body(handle, x, qkv_w, ori, q_norm_w, k_norm_w, proj_w,
                            proj_b, out, H, eps, norm_eps, label);
}

// Batched LiteLA: x/out [B, T, D]; weights shared across the batch.
// Implemented as a per-image loop over the rank-2 body above.
template <class ExecSpace, class Scalar>
void linear_attention_lite_la(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& qkv_w,
    const Kokkos::View<Scalar*, ExecSpace>& q_norm_w,
    const Kokkos::View<Scalar*, ExecSpace>& k_norm_w,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj_w,
    const Kokkos::View<Scalar*, ExecSpace>& proj_b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    size_t num_heads = 0, float eps = 1e-15f, float norm_eps = 1e-6f,
    const std::string& label = "KokkosDNN::sana_litela") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (out.extent(0) != B || out.extent(1) != T || out.extent(2) != D)
    throw std::invalid_argument("linear_attention_lite_la: out mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  size_t H = num_heads;
  if (H == 0) {
    if (D % 32 != 0)
      throw std::invalid_argument(
          "linear_attention_lite_la: D not divisible by 32, pass num_heads");
    H = D / 32;
  }
  if (H == 0 || D % H != 0)
    throw std::invalid_argument("linear_attention_lite_la: D % H != 0");
  const Impl::SanaWeightOri ori = Impl::sana_detect_ori(
      qkv_w.extent(0), qkv_w.extent(1), D, 3 * D,
      "linear_attention_lite_la: qkv_w must be [3D,D] or [D,3D]");
  const bool has_q = (q_norm_w.extent(0) != 0);
  const bool has_k = (k_norm_w.extent(0) != 0);
  if ((has_q && q_norm_w.extent(0) != D) || (has_k && k_norm_w.extent(0) != D))
    throw std::invalid_argument("linear_attention_lite_la: norm weight mismatch");
  if (proj_w.extent(0) != D || proj_w.extent(1) != D)
    throw std::invalid_argument("linear_attention_lite_la: proj_w mismatch");
  if (proj_b.extent(0) != 0 && proj_b.extent(0) != D)
    throw std::invalid_argument("linear_attention_lite_la: proj_b mismatch");
  for (size_t b = 0; b < B; ++b) {
    Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> xb("KokkosDNN::sana_xb",
                                                               T, D),
        ob("KokkosDNN::sana_ob", T, D);
    Kokkos::parallel_for(
        label + "_gather",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { xb(t, d) = x(b, t, d); });
    Impl::sana_litela_2d_body(handle, xb, qkv_w, ori, q_norm_w, k_norm_w, proj_w,
                              proj_b, ob, H, eps, norm_eps, label);
    Kokkos::parallel_for(
        label + "_scatter",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { out(b, t, d) = ob(t, d); });
  }
}

// Sana multi-head cross attention (no qk_norm, no mask):
//   q      = x @ q_w + q_b              [T, D]
//   kv     = cond @ kv_w + kv_b         [L, 2D] -> split k (first D), v
//   reshape heads (H, Dh = D / H), softmax(Q K^T / sqrt(Dh)) V
//   (via attention_mha_bidirectional), out = attn @ proj_w + proj_b.
//
// x [T, D] image tokens, cond [L, D] text embeddings (batch-1 flattened
// for the rank-2 path). Scalar-agnostic (float, bhalf_t), FP32
// accumulation. Non-square kv_w accepts HF-Linear [2D, D] or math
// [D, 2D] (auto-detected); square q_w/proj_w [D, D] follow the repo HF
// convention. Empty bias views (extent 0) mean zeros. num_heads == 0
// selects the Sana cross-attention rule H = D / 72 (Dh = 72).
template <class ExecSpace, class Scalar>
void cross_attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& q_w,
    const Kokkos::View<Scalar*, ExecSpace>& q_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kv_w,
    const Kokkos::View<Scalar*, ExecSpace>& kv_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj_w,
    const Kokkos::View<Scalar*, ExecSpace>& proj_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    size_t num_heads = 0,
    const std::string& label = "KokkosDNN::sana_cross") {
  const size_t T = x.extent(0), D = x.extent(1), L = cond.extent(0);
  if (cond.extent(1) != D || out.extent(0) != T || out.extent(1) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: dim mismatch");
  if (T == 0 || D == 0) return;
  size_t H = num_heads;
  if (H == 0) {
    if (D % 72 != 0)
      throw std::invalid_argument(
          "cross_attention_mha_bidirectional: D not divisible by 72, pass "
          "num_heads");
    H = D / 72;
  }
  if (H == 0 || D % H != 0)
    throw std::invalid_argument("cross_attention_mha_bidirectional: D % H != 0");
  if (q_w.extent(0) != D || q_w.extent(1) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: q_w mismatch");
  if (q_b.extent(0) != 0 && q_b.extent(0) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: q_b mismatch");
  const Impl::SanaWeightOri ori = Impl::sana_detect_ori(
      kv_w.extent(0), kv_w.extent(1), D, 2 * D,
      "cross_attention_mha_bidirectional: kv_w must be [2D,D] or [D,2D]");
  if (kv_b.extent(0) != 0 && kv_b.extent(0) != 2 * D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: kv_b mismatch");
  if (proj_w.extent(0) != D || proj_w.extent(1) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: proj_w mismatch");
  if (proj_b.extent(0) != 0 && proj_b.extent(0) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: proj_b mismatch");
  Impl::sana_cross_2d_body(handle, x, cond, q_w, q_b, kv_w, ori, kv_b, proj_w,
                           proj_b, out, H, label);
}

// Batched cross attention: x/out [B, T, D], cond [B, L, D].
template <class ExecSpace, class Scalar>
void cross_attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& q_w,
    const Kokkos::View<Scalar*, ExecSpace>& q_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kv_w,
    const Kokkos::View<Scalar*, ExecSpace>& kv_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj_w,
    const Kokkos::View<Scalar*, ExecSpace>& proj_b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    size_t num_heads = 0,
    const std::string& label = "KokkosDNN::sana_cross") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  const size_t L = (cond.extent(0) == B) ? cond.extent(1) : 0;
  if (cond.extent(0) != B || cond.extent(2) != D || out.extent(0) != B ||
      out.extent(1) != T || out.extent(2) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: dim mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  size_t H = num_heads;
  if (H == 0) {
    if (D % 72 != 0)
      throw std::invalid_argument(
          "cross_attention_mha_bidirectional: D not divisible by 72, pass "
          "num_heads");
    H = D / 72;
  }
  if (H == 0 || D % H != 0)
    throw std::invalid_argument("cross_attention_mha_bidirectional: D % H != 0");
  if (q_w.extent(0) != D || q_w.extent(1) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: q_w mismatch");
  if (q_b.extent(0) != 0 && q_b.extent(0) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: q_b mismatch");
  const Impl::SanaWeightOri ori = Impl::sana_detect_ori(
      kv_w.extent(0), kv_w.extent(1), D, 2 * D,
      "cross_attention_mha_bidirectional: kv_w must be [2D,D] or [D,2D]");
  if (kv_b.extent(0) != 0 && kv_b.extent(0) != 2 * D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: kv_b mismatch");
  if (proj_w.extent(0) != D || proj_w.extent(1) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: proj_w mismatch");
  if (proj_b.extent(0) != 0 && proj_b.extent(0) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: proj_b mismatch");
  for (size_t b = 0; b < B; ++b) {
    Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> xb("KokkosDNN::sana_xb",
                                                               T, D),
        cb("KokkosDNN::sana_cb", L, D), ob("KokkosDNN::sana_ob", T, D);
    Kokkos::parallel_for(
        label + "_gather_x",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { xb(t, d) = x(b, t, d); });
    Kokkos::parallel_for(
        label + "_gather_c",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {L, D}),
        KOKKOS_LAMBDA(size_t l, size_t d) { cb(l, d) = cond(b, l, d); });
    Impl::sana_cross_2d_body(handle, xb, cb, q_w, q_b, kv_w, ori, kv_b, proj_w,
                             proj_b, ob, H, label);
    Kokkos::parallel_for(
        label + "_scatter",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { out(b, t, d) = ob(t, d); });
  }
}

// Batched cross attention with one shared conditioning sequence:
// x/out [B, T, D], cond [L, D] broadcast to every batch element.
template <class ExecSpace, class Scalar>
void cross_attention_mha_bidirectional(
    const DNNHandle<ExecSpace>& handle,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& q_w,
    const Kokkos::View<Scalar*, ExecSpace>& q_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& kv_w,
    const Kokkos::View<Scalar*, ExecSpace>& kv_b,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& proj_w,
    const Kokkos::View<Scalar*, ExecSpace>& proj_b,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    size_t num_heads = 0,
    const std::string& label = "KokkosDNN::sana_cross") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  const size_t L = cond.extent(0);
  if (cond.extent(1) != D || out.extent(0) != B || out.extent(1) != T ||
      out.extent(2) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: dim mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  size_t H = num_heads;
  if (H == 0) {
    if (D % 72 != 0)
      throw std::invalid_argument(
          "cross_attention_mha_bidirectional: D not divisible by 72, pass "
          "num_heads");
    H = D / 72;
  }
  if (H == 0 || D % H != 0)
    throw std::invalid_argument("cross_attention_mha_bidirectional: D % H != 0");
  if (q_w.extent(0) != D || q_w.extent(1) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: q_w mismatch");
  if (q_b.extent(0) != 0 && q_b.extent(0) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: q_b mismatch");
  const Impl::SanaWeightOri ori = Impl::sana_detect_ori(
      kv_w.extent(0), kv_w.extent(1), D, 2 * D,
      "cross_attention_mha_bidirectional: kv_w must be [2D,D] or [D,2D]");
  if (kv_b.extent(0) != 0 && kv_b.extent(0) != 2 * D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: kv_b mismatch");
  if (proj_w.extent(0) != D || proj_w.extent(1) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: proj_w mismatch");
  if (proj_b.extent(0) != 0 && proj_b.extent(0) != D)
    throw std::invalid_argument("cross_attention_mha_bidirectional: proj_b mismatch");
  for (size_t b = 0; b < B; ++b) {
    Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> xb("KokkosDNN::sana_xb",
                                                               T, D),
        ob("KokkosDNN::sana_ob", T, D);
    Kokkos::parallel_for(
        label + "_gather_x",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { xb(t, d) = x(b, t, d); });
    Impl::sana_cross_2d_body(handle, xb, cond, q_w, q_b, kv_w, ori, kv_b, proj_w,
                             proj_b, ob, H, label);
    Kokkos::parallel_for(
        label + "_scatter",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
        KOKKOS_LAMBDA(size_t t, size_t d) { out(b, t, d) = ob(t, d); });
  }
}

}  // namespace KokkosDNN

#endif
