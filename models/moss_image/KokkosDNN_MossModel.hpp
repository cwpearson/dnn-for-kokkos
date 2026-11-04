// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_MOSS_MODEL_HPP
#define KOKKOSDNN_MOSS_MODEL_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "KokkosDNN_Activation.hpp"
#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_LayerNorm.hpp"
#include "KokkosDNN_RMSNorm.hpp"
#include "KokkosDNN_MossConfig.hpp"
#include "KokkosDNN_MossWeights.hpp"

namespace KokkosDNN {
namespace Moss {
namespace Impl {

// Upper bound for head dims handled by the joint-attention path.
inline constexpr size_t kMaxHeadDim = 128;

// Fused bias-add + SiLU, in-place: m[r,d] = silu(m[r,d] + bias[d]).
template <class ExecSpace, class Scalar, class Mat, class Vec>
void bias_add_silu_inplace(const Mat& m, const Vec& bias,
                            const std::string& label) {
  const size_t R = m.extent(0), D = m.extent(1);
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t r, size_t d) {
        const float v =
            static_cast<float>(m(r, d)) + static_cast<float>(bias(d));
        m(r, d) = static_cast<Scalar>(v / (1.0f + expf(-v)));
      });
}

// Gated residual: x[r,d] += gate[b,d] * v[r,d], b = r / T. In-place on x.
template <class ExecSpace, class Scalar, class Mat, class GMat>
void gated_residual(const Mat& x, const Mat& v, const GMat& gate, size_t T,
                    const std::string& label) {
  const size_t R = x.extent(0), D = x.extent(1);
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t r, size_t d) {
        const size_t b = (T == 0) ? 0 : r / T;
        x(r, d) = static_cast<Scalar>(
            static_cast<float>(x(r, d)) +
            static_cast<float>(gate(b, d)) * static_cast<float>(v(r, d)));
      });
}

// Fused bias-add + gated residual: tok += gate*(proj+bias).
template <class ExecSpace, class Scalar, class Mat, class Vec, class GMat>
void bias_add_gated_residual(const Mat& proj, const Vec& bias, const Mat& tok,
                              const GMat& gate, size_t T,
                              const std::string& label) {
  const size_t R = proj.extent(0), D = proj.extent(1);
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t r, size_t d) {
        const size_t b = (T == 0) ? 0 : r / T;
        tok(r, d) = static_cast<Scalar>(
            static_cast<float>(tok(r, d)) +
            static_cast<float>(gate(b, d)) *
                (static_cast<float>(proj(r, d)) +
                 static_cast<float>(bias(d))));
      });
}

// Per-block adaLN-single conditioning: mod[b,k,d] = t0[b,k,d] + table[k,d].
template <class ExecSpace, class Scalar, class Mat3, class Mat>
void add_block_table(const Mat3& t0, const Mat& table, const Mat3& mod,
                     const std::string& label) {
  const size_t B = t0.extent(0), K = t0.extent(1), D = t0.extent(2);
  if (B == 0 || K == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, K, D}),
      KOKKOS_LAMBDA(size_t b, size_t k, size_t d) {
        mod(b, k, d) = static_cast<Scalar>(
            static_cast<float>(t0(b, k, d)) + static_cast<float>(table(k, d)));
      });
}

// Sinusoidal timestep embedding with optional normalization divisor:
// ts = t[b] for norm_scale == 1 else t[b]/norm_scale.
template <class ExecSpace, class Scalar, class TView, class Mat>
void timestep_embed(const TView& t, double norm_scale, const Mat& emb,
                    const std::string& label) {
  const size_t B = t.extent(0), F = emb.extent(1), H = F / 2;
  if (B == 0 || F == 0) return;
  const float ns = static_cast<float>(norm_scale);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, H}),
      KOKKOS_LAMBDA(size_t b, size_t i) {
        const float raw = static_cast<float>(t(b));
        const float ts = (ns == 1.0f) ? raw : raw / ns;
        const float fr = expf(-9.210340371976184f * static_cast<float>(i) /
                              static_cast<float>(H));
        const float a = ts * fr;
        emb(b, i) = static_cast<Scalar>(cosf(a));
        emb(b, i + H) = static_cast<Scalar>(sinf(a));
      });
}

// Patchify latents [B, C, H, W] to rows [B*T, C*P*P] in torch flatten order:
// col = (c*P+kh)*P+kw.
template <class ExecSpace, class Scalar, class Lat, class Mat>
void patchify(const Lat& x, const Mat& rows, int patch,
              const std::string& label) {
  const size_t B = x.extent(0), C = x.extent(1);
  const size_t H = x.extent(2), W = x.extent(3);
  if (B == 0 || C == 0) return;
  const size_t P = (size_t)patch;
  const size_t Hp = H / P, Wp = W / P;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {B, Hp, Wp, C}),
      KOKKOS_LAMBDA(size_t b, size_t hh, size_t ww, size_t c) {
        const size_t t = hh * Wp + ww;
        for (size_t kh = 0; kh < P; ++kh)
          for (size_t kw = 0; kw < P; ++kw) {
            const size_t col = (c * P + kh) * P + kw;
            rows(b * (Hp * Wp) + t, col) =
                x(b, c, hh * P + kh, ww * P + kw);
          }
      });
}

// Unpatchify rows [B*Hp*Wp, Oc*P*P] -> [B, Oc, H, W] (inverse of patchify).
template <class ExecSpace, class Scalar, class Mat, class Lat>
void unpatchify(const Mat& rows, const Lat& img, int patch,
                const std::string& label) {
  const size_t B = img.extent(0), Oc = img.extent(1);
  const size_t H = img.extent(2), W = img.extent(3);
  if (B == 0 || Oc == 0) return;
  const size_t P = (size_t)patch;
  const size_t Hp = H / P, Wp = W / P;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {B, Hp, Wp, Oc}),
      KOKKOS_LAMBDA(size_t b, size_t hh, size_t ww, size_t c) {
        const size_t t = hh * Wp + ww;
        for (size_t kh = 0; kh < P; ++kh)
          for (size_t kw = 0; kw < P; ++kw) {
            const size_t col = (c * P + kh) * P + kw;
            img(b, c, hh * P + kh, ww * P + kw) =
                static_cast<Scalar>(rows(b * (Hp * Wp) + t, col));
          }
      });
}

// Fused bias-add + unpatchify: img = rows @ + bias, one pass saved.
template <class ExecSpace, class Scalar, class Mat, class Lat, class Vec>
void bias_add_unpatchify(const Mat& rows, const Vec& bias, const Lat& img,
                         int patch, const std::string& label) {
  const size_t B = img.extent(0), Oc = img.extent(1);
  const size_t H = img.extent(2), W = img.extent(3);
  if (B == 0 || Oc == 0) return;
  const size_t P = (size_t)patch;
  const size_t Hp = H / P, Wp = W / P;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {B, Hp, Wp, Oc}),
      KOKKOS_LAMBDA(size_t b, size_t hh, size_t ww, size_t c) {
        const size_t t = hh * Wp + ww;
        for (size_t kh = 0; kh < P; ++kh)
          for (size_t kw = 0; kw < P; ++kw) {
            const size_t col = (c * P + kh) * P + kw;
            img(b, c, hh * P + kh, ww * P + kw) = static_cast<Scalar>(
                static_cast<float>(rows(b * (Hp * Wp) + t, col)) +
                static_cast<float>(bias(col)));
          }
      });
}

}  // namespace Impl

// MOSS DiT noise predictor (single forward, no sampler).
//
// Owns the TPL handle and persistent grow-only scratch. All GEMMs go
// through gemm/gemm_bias (cuBLAS dispatch inherited on CUDA) and the joint
// self+cross attention goes through attention_mha_bidirectional (cuDNN SDPA
// dispatch inherited on CUDA); elementwise/norm ops use the core library.
//
// forward(x [B,C,H,W] latents, t [B] timesteps, y [B,L,Cc] caption
// embeddings, out [B,Cout,H,W], skip_layer):
//   x: patchify (P=2) -> linear -> tokens [B*T, D]
//   t: sinusoidal embed (t_freq_dim) -> MLP(SiLU) -> t;
//      t0 = Linear(SiLU(t)) [B,6,D] (adaLN-single conditioning)
//   y: fc -> RMSNorm -> [B*L, D]
//   per block (skipped entirely when (int)l == skip_layer, i.e. identity):
//     adaLN-modulate(LN no-affine) -> joint attention over
//       queries = tokens, keys/values = tokens ++ caption -> gated residual
//     adaLN-modulate(LN no-affine) -> FFN (fc1 -> GELU-tanh -> fc2)
//       -> gated residual
//   final: adaLN-modulate(LN no-affine, [2,D] table + t) -> linear ->
//     unpatchify -> [B,Cout,H,W] noise prediction.
//
// skip_layer implements skip-layer guidance on the negative branch: the
// sampler runs the unconditional forward with skip_layer = N (default 22,
// i.e. that block is identity) while the conditional forward uses -1.
template <class ExecSpace, class Scalar = DefaultScalar>
class Model {
 public:
  using MatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using Mat3R = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using Vec = Kokkos::View<Scalar*, ExecSpace>;
  using LatR = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;
  using TokR = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
  using UMatR = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UMat3R = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UVec = Kokkos::View<Scalar*, ExecSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using TVec = Kokkos::View<float*, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;
  DNNHandle<ExecSpace> handle;
  Vec ones_;    // [D] ones for affine-free LayerNorm
  Vec zeros_;   // [D] zeros for affine-free LayerNorm
  Vec nobias_;  // empty (extent 0) => gemm_bias unbiased path

  Model() = default;
  Model(const Config& c, const Weights<ExecSpace, Scalar>& w)
      : cfg(c), weights(w) {
    cfg.validate();
    if ((size_t)cfg.head_dim() > Impl::kMaxHeadDim)
      throw std::invalid_argument(
          "Moss::Model: head dim exceeds joint-attention kernel limit");
    ones_ = Vec("moss_ones", (size_t)cfg.hidden_size);
    zeros_ = Vec("moss_zeros", (size_t)cfg.hidden_size);
    Kokkos::deep_copy(ones_, static_cast<Scalar>(1.0f));
    Kokkos::deep_copy(zeros_, static_cast<Scalar>(0.0f));
  }

  // DiT forward: see class docs. t holds raw diffusion timesteps (float);
  // y holds caption embeddings of dim caption_channels, L <= maxlen.
  // out must be [B, out_channels, H, W]. H and W must tile patch_size.
  // skip_layer >= 0 replaces that block with identity (SLG negative pass).
  void forward(LatR x, TVec t, TokR y, LatR out, int skip_layer = -1) {
    const size_t B = x.extent(0);
    const size_t C = x.extent(1);
    const size_t H = x.extent(2), W = x.extent(3);
    if (B == 0) throw std::invalid_argument("Moss::Model::forward: B == 0");
    if (C != (size_t)cfg.in_channels)
      throw std::invalid_argument(
          "Moss::Model::forward: in_channels mismatch (got C=" +
          std::to_string(C) + ", expected " + std::to_string(cfg.in_channels) +
          ")");
    if (t.extent(0) != B)
      throw std::invalid_argument(
          "Moss::Model::forward: timestep batch mismatch");
    if (y.extent(0) != B)
      throw std::invalid_argument(
          "Moss::Model::forward: caption batch mismatch");
    const size_t L = y.extent(1);
    if (L == 0 || L > (size_t)cfg.model_max_length)
      throw std::invalid_argument(
          "Moss::Model::forward: caption length out of range");
    if (y.extent(2) != (size_t)cfg.caption_channels)
      throw std::invalid_argument(
          "Moss::Model::forward: caption_channels mismatch");
    if (out.extent(0) != B || out.extent(1) != (size_t)cfg.out_channels ||
        out.extent(2) != H || out.extent(3) != W)
      throw std::invalid_argument("Moss::Model::forward: out shape mismatch");
    const size_t P = (size_t)cfg.patch_size;
    if (H % P != 0 || W % P != 0)
      throw std::invalid_argument(
          "Moss::Model::forward: H and W must tile patch_size");
    if (skip_layer >= cfg.depth)
      throw std::invalid_argument(
          "Moss::Model::forward: skip_layer out of range");
    const size_t T = (H / P) * (W / P);
    ensure_scratch(B, T, L);

    // Hoisted dims (device lambdas must not capture `this`/`cfg`).
    const size_t D = (size_t)cfg.hidden_size;
    const size_t nH = (size_t)cfg.num_heads;
    const size_t hd = D / nH;
    const float EPS = static_cast<float>(cfg.norm_eps);
    const float REPS = static_cast<float>(cfg.rms_eps);
    const float CSCALE = 1.0f / std::sqrt(static_cast<float>(hd));
    const double TNS = cfg.timestep_norm_scale;
    const size_t NL = (size_t)cfg.depth;
    const size_t Cc = (size_t)cfg.caption_channels;
    const size_t R = B * T, Ry = B * L, Rj = B * (T + L);

    auto tok = sub_rows(tok_, 0, R);    // [R, D] token stream
    auto n1 = sub_rows(n1_, 0, R);     // [R, D] norm/mod buffer
    auto att = sub_rows(att_, 0, R);   // [R, D] attn/mlp out buffer
    auto qkv = sub_rows(qkv_, 0, R);   // [R, 3D] token qkv
    auto qq = sub_rows(qq_, 0, R);     // [R, D] token queries (Q third)
    auto ckv = sub_rows(ckv_, 0, Ry);  // [Ry, 2D] caption kv
    auto kj = sub_rows(kj_, 0, Rj);    // [Rj, D] joint keys
    auto vj = sub_rows(vj_, 0, Rj);    // [Rj, D] joint values
    auto fc = sub_rows(fc_, 0, R);     // [R, Hh] FFN hidden
    auto lin = sub_rows(lin_, 0, R);   // [R, Oc*P*P] final rows
    auto xin = sub_rows(xin_, 0, R);   // [R, C*P*P] patchified input
    auto tf = sub_rows(tf_, 0, B);     // [B, F]
    auto te = sub_rows(te_, 0, B);     // [B, D]
    auto t0 = sub_rows3(t0_, 0, B);    // [B, 6, D] t_block output
    auto mod = sub_rows3(mod_, 0, B);  // [B, 6, D] table+t0 per block
    auto yp = sub_rows(yp_, 0, Ry);    // [Ry, D] caption (normed)
    auto yn = sub_rows(yn_, 0, Ry);    // [Ry, D] caption (fc out)
    auto nobias = nobias_;
    auto ones = ones_, zeros = zeros_;

    // -- input embedding --
    Impl::patchify<ExecSpace, Scalar>(x, xin, cfg.patch_size, "moss_patchify");
    gemm_bias<ExecSpace, Scalar>(handle, xin, weights.x_proj_w, weights.x_proj_b,
                                 tok);

    // -- timestep conditioning --
    Impl::timestep_embed<ExecSpace, Scalar>(t, TNS, tf, "moss_tfreq");
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), tf,
                            weights.t_mlp0_w, static_cast<Scalar>(0.0f), te,
                            "moss_tmlp0");
    Impl::bias_add_silu_inplace<ExecSpace, Scalar>(te, weights.t_mlp0_b,
                                                  "moss_tsilu");
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), te,
                            weights.t_mlp2_w, static_cast<Scalar>(0.0f), te,
                            "moss_tmlp2");
    Impl::bias_add_silu_inplace<ExecSpace, Scalar>(te, weights.t_mlp2_b,
                                                  "moss_tsilu2");
    {
      UMatR t0f(t0.data(), B, 6 * D);
      gemm_bias<ExecSpace, Scalar>(handle, te, weights.t_block_w,
                                   weights.t_block_b, t0f);
    }

    // -- caption projection: fc -> RMSNorm --
    {
      UMatR yflat(y.data(), Ry, Cc);
      gemm_bias<ExecSpace, Scalar>(handle, yflat, weights.cap_proj_w,
                                   weights.cap_proj_b, yn);
      rmsnorm<ExecSpace, Scalar>(yn, weights.cap_norm_w, REPS, yp);
    }

    // -- transformer blocks --
    for (size_t l = 0; l < NL; ++l) {
      if ((int)l == skip_layer) continue;  // SLG: identity on this branch
      const auto& bw = weights.blocks[l];
      // adaLN-single: mod = table + t0, chunked into six lanes.
      Impl::add_block_table<ExecSpace, Scalar>(t0, bw.scale_shift_table, mod,
                                               "moss_add_table");
      auto s_att = Kokkos::subview(mod, Kokkos::ALL(), 0, Kokkos::ALL());
      auto c_att = Kokkos::subview(mod, Kokkos::ALL(), 1, Kokkos::ALL());
      auto g_att = Kokkos::subview(mod, Kokkos::ALL(), 2, Kokkos::ALL());
      auto s_mlp = Kokkos::subview(mod, Kokkos::ALL(), 3, Kokkos::ALL());
      auto c_mlp = Kokkos::subview(mod, Kokkos::ALL(), 4, Kokkos::ALL());
      auto g_mlp = Kokkos::subview(mod, Kokkos::ALL(), 5, Kokkos::ALL());

      // Joint self+cross attention: queries from modulated tokens,
      // keys/values from tokens ++ caption (single bidirectional kernel).
      layernorm_forward<ExecSpace, Scalar>(tok, ones, zeros, EPS, n1);
      {
        // modulate n1 in place: n1 *= (1 + c_att) + s_att per batch row.
        const size_t RR = n1.extent(0);
        Kokkos::parallel_for(
            "moss_attn_mod",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {RR, D}),
            KOKKOS_LAMBDA(size_t r, size_t d) {
              const size_t b = (T == 0) ? 0 : r / T;
              n1(r, d) = static_cast<Scalar>(
                  static_cast<float>(n1(r, d)) *
                      (1.0f + static_cast<float>(c_att(b, d))) +
                  static_cast<float>(s_att(b, d)));
            });
      }
      gemm_bias<ExecSpace, Scalar>(handle, n1, bw.attn_qkv_w, nobias, qkv);
      gemm_bias<ExecSpace, Scalar>(handle, yp, bw.attn_cap_kv_w, nobias, ckv);
      // Pack joint K/V rows: [T tokens | L caption] per batch element.
      Kokkos::parallel_for(
          "moss_pack_kv",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, D}),
          KOKKOS_LAMBDA(size_t b, size_t d) {
            for (size_t ti = 0; ti < T; ++ti) {
              kj(b * (T + L) + ti, d) = qkv(b * T + ti, D + d);          // K
              vj(b * (T + L) + ti, d) = qkv(b * T + ti, 2 * D + d);      // V
            }
            for (size_t li = 0; li < L; ++li) {
              kj(b * (T + L) + T + li, d) = ckv(b * L + li, d);          // K
              vj(b * (T + L) + T + li, d) = ckv(b * L + li, D + d);      // V
            }
          });
      for (size_t b = 0; b < B; ++b) {
        // Q rows are the first D-wide third of qkv. They are repacked into
        // the dense [R, D] qq buffer first: qkv rows have stride 3D, so an
        // unmanaged [T, nH, hd] alias directly over qkv would read with the
        // wrong row stride (nH*hd == D != 3D). kj_/vj_ rows are dense [*, D]
        // (nH*hd == D), so k3/v3/o3 aliases below are exact.
        Kokkos::parallel_for(
            "moss_q_pack",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, D}),
            KOKKOS_LAMBDA(size_t ti, size_t d) {
              qq(b * T + ti, d) = qkv(b * T + ti, d);
            });
        UMat3R q3(qq.data() + (b * T) * D, T, nH, hd);
        UMat3R k3(kj.data() + (b * (T + L)) * D, T + L, nH, hd);
        UMat3R v3(vj.data() + (b * (T + L)) * D, T + L, nH, hd);
        UMat3R o3(att.data() + (b * T) * D, T, nH, hd);
        attention_mha_bidirectional<ExecSpace, Scalar>(handle, q3, k3, v3, o3,
                                                      CSCALE);
      }
      gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), att,
                              bw.attn_proj_w, static_cast<Scalar>(0.0f), n1,
                              "moss_attnproj");
      Impl::bias_add_gated_residual<ExecSpace, Scalar>(
          n1, bw.attn_proj_b, tok, g_att, T, "moss_gate1");

      // FFN path: modulate(LN) -> fc1 -> GELU-tanh -> fc2 -> gated residual.
      layernorm_forward<ExecSpace, Scalar>(tok, ones, zeros, EPS, n1);
      {
        const size_t RR = n1.extent(0);
        Kokkos::parallel_for(
            "moss_mlp_mod",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {RR, D}),
            KOKKOS_LAMBDA(size_t r, size_t d) {
              const size_t b = (T == 0) ? 0 : r / T;
              n1(r, d) = static_cast<Scalar>(
                  static_cast<float>(n1(r, d)) *
                      (1.0f + static_cast<float>(c_mlp(b, d))) +
                  static_cast<float>(s_mlp(b, d)));
            });
      }
      gemm_bias<ExecSpace, Scalar>(handle, n1, bw.mlp_fc1_w, bw.mlp_fc1_b, fc);
      gelu_forward<ExecSpace, Scalar>(fc, fc, GeluVariant::Tanh, "moss_gelu");
      gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), fc,
                              bw.mlp_fc2_w, static_cast<Scalar>(0.0f), att,
                              "moss_fc2");
      Impl::bias_add_gated_residual<ExecSpace, Scalar>(
          att, bw.mlp_fc2_b, tok, g_mlp, T, "moss_gate2");
    }

    // -- final layer: modulate(LN, table + t) -> linear -> unpatchify --
    layernorm_forward<ExecSpace, Scalar>(tok, ones, zeros, EPS, n1);
    {
      const size_t RR = n1.extent(0);
      auto ftab = weights.final_table;
      Kokkos::parallel_for(
          "moss_final_mod",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {RR, D}),
          KOKKOS_LAMBDA(size_t r, size_t d) {
            const size_t b = r / T;
            const float sh =
                static_cast<float>(ftab(0, d)) + static_cast<float>(te(b, d));
            const float sc =
                static_cast<float>(ftab(1, d)) + static_cast<float>(te(b, d));
            n1(r, d) = static_cast<Scalar>(static_cast<float>(n1(r, d)) *
                                               (1.0f + sc) +
                                           sh);
          });
    }
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), n1,
                            weights.final_w, static_cast<Scalar>(0.0f), lin,
                            "moss_final");
    Impl::bias_add_unpatchify<ExecSpace, Scalar>(
        lin, weights.final_b, out, cfg.patch_size, "moss_unpatch");
  }

 private:
  // Grow-only capacity buffers (never shrunk; steady state = zero
  // allocations). tok_/n1_/att_ [Rcap, D]; qkv_ [Rcap, 3D];
  // qq_ [Rcap, D] repacked queries; ckv_ [Rycap, 2D];
  // kj_/vj_ [RBcap, D] joint K/V rows (B*(T+L));
  // fc_ [Rcap, Hh]; lin_ [Rcap, Oc*P*P]; xin_ [Rcap, Ic*P*P];
  // tf_ [Bcap, F]; te_ [Bcap, D]; t0_ [Bcap, 6, D];
  // yp_/yn_ [Rycap, D]; mod_ [Bcap, 6, D].
  MatR tok_, n1_, att_, qkv_, qq_, ckv_, kj_, vj_, fc_, lin_, xin_;
  MatR tf_, te_, yp_, yn_;
  Mat3R t0_, mod_;
  size_t cap_B_ = 0, cap_T_ = 0, cap_L_ = 0;

  static MatR sub_rows(const MatR& v, size_t r0, size_t r1) {
    return Kokkos::subview(v, std::make_pair(r0, r1), Kokkos::ALL());
  }
  static Mat3R sub_rows3(const Mat3R& v, size_t r0, size_t r1) {
    return Kokkos::subview(v, std::make_pair(r0, r1), Kokkos::ALL(),
                           Kokkos::ALL());
  }

  void ensure_scratch(size_t B, size_t T, size_t L) {
    if (B <= cap_B_ && T <= cap_T_ && L <= cap_L_) return;
    cap_B_ = std::max(cap_B_, B);
    cap_T_ = std::max(cap_T_, T);
    cap_L_ = std::max(cap_L_, L);
    const size_t D = (size_t)cfg.hidden_size;
    const size_t Hh = (size_t)cfg.mlp_hidden();
    const size_t F = (size_t)cfg.t_freq_dim;
    const size_t P = (size_t)cfg.patch_size;
    const size_t Oc = (size_t)cfg.out_channels;
    const size_t Ic = (size_t)cfg.in_channels;
    const size_t R = cap_B_ * cap_T_, Ry = cap_B_ * cap_L_;
    const size_t Rj = cap_B_ * (cap_T_ + cap_L_);
    tok_ = MatR("moss_tok", R, D);
    n1_ = MatR("moss_n1", R, D);
    att_ = MatR("moss_att", R, D);
    qkv_ = MatR("moss_qkv", R, 3 * D);
    qq_ = MatR("moss_qq", R, D);
    ckv_ = MatR("moss_ckv", Ry, 2 * D);
    kj_ = MatR("moss_kj", Rj, D);
    vj_ = MatR("moss_vj", Rj, D);
    fc_ = MatR("moss_fc", R, Hh);
    lin_ = MatR("moss_lin", R, Oc * P * P);
    xin_ = MatR("moss_xin", R, Ic * P * P);
    tf_ = MatR("moss_tf", cap_B_, F);
    te_ = MatR("moss_te", cap_B_, D);
    t0_ = Mat3R("moss_t0", cap_B_, 6, D);
    mod_ = Mat3R("moss_mod", cap_B_, 6, D);
    yp_ = MatR("moss_yp", Ry, D);
    yn_ = MatR("moss_yn", Ry, D);
  }
};

}  // namespace Moss
}  // namespace KokkosDNN

#endif
