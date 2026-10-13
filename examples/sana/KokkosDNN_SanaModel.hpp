#ifndef KOKKOSDNN_SANA_MODEL_HPP
#define KOKKOSDNN_SANA_MODEL_HPP

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
#include "KokkosDNN_SanaConfig.hpp"
#include "KokkosDNN_SanaWeights.hpp"
#include "KokkosDNN_SwiGLU.hpp"

namespace KokkosDNN {
namespace Sana {
namespace Impl {

// Upper bound for head dims handled by the LiteLA kernel's stack
// accumulators (600M uses 32; cross-attention uses 72).
inline constexpr size_t kMaxHeadDim = 128;

// y[r,d] = x[r,d] * (1 + scale[b,d]) + shift[b,d], b = r / T, with
// shift/scale as [B, D] lanes (any layout; subviews of the t0 table).
// In-place safe on x/y.
template <class ExecSpace, class Scalar, class Mat, class GMat>
void modulate(const Mat& x, const Mat& y, const GMat& shift,
              const GMat& scale, size_t T, const std::string& label) {
  const size_t R = x.extent(0), D = x.extent(1);
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t r, size_t d) {
        const size_t b = (T == 0) ? 0 : r / T;
        float v        = static_cast<float>(x(r, d));
        y(r, d) = static_cast<Scalar>(v * (1.0f + static_cast<float>(scale(b, d))) +
                                      static_cast<float>(shift(b, d)));
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

// In-place SiLU (t_embedder/t_block activations, GLUMBConv acts).
template <class ExecSpace, class Scalar, class Mat>
void silu_inplace(const Mat& m, const std::string& label) {
  const size_t R = m.extent(0), D = m.extent(1);
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t r, size_t d) {
        float v = static_cast<float>(m(r, d));
        m(r, d) = static_cast<Scalar>(v / (1.0f + expf(-v)));
      });
}

// Sinusoidal timestep embedding: emb[b,:] = [cos(ts*f), sin(ts*f)] with
// f[i] = exp(-ln(10000)*i/(F/2)), ts = trunc(t[b])/norm_scale for
// norm_scale == 1 (SanaMS applies timestep.long() in that case) else
// t[b]/norm_scale (SanaMS keeps fractions when a scale is configured).
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
        const float ts  = (ns == 1.0f) ? truncf(raw) : raw / ns;
        const float fr = expf(-9.210340371976184f * static_cast<float>(i) /
                              static_cast<float>(H));
        const float a = ts * fr;
        emb(b, i)     = static_cast<Scalar>(cosf(a));
        emb(b, i + H) = static_cast<Scalar>(sinf(a));
      });
}

// LiteLA linear self-attention for one batch element (qk_norm=false, no
// rotary embedding in the 600M image setting). qkv is the fused [T, 3D]
// projection (q | k | v thirds, matching torch qkv.unbind); o is
// [T, nH, hd]. Column thirds are indexed explicitly because the [T, 3D]
// rows are NOT reinterpretable as a dense [T, nH, hd] view (row stride 3D
// != D); o aliases persistent scratch and is dense.
// out[h,n] = (V K^T Q)[h,n] / (sum(K^T Q)[h,n] + eps), i.e. the PAD_VAL=1
// normalization of sana_blocks.LiteLA.attn_matmul without materializing
// the pad row. ReLU kernels are fused on the Q/K loads.
template <class ExecSpace, class Scalar, class Mat, class V3>
void litela_batch_qkv(const Mat& qkv, const V3& o, size_t nH, size_t hd,
                      float eps, const std::string& label) {
  const size_t T = qkv.extent(0), D = qkv.extent(1) / 3;
  if (T == 0 || D == 0) return;
  if (hd > kMaxHeadDim)
    throw std::invalid_argument("sana: LiteLA head dim exceeds kernel limit");
  if (nH * hd != D || o.extent(0) != T || o.extent(1) != nH ||
      o.extent(2) != hd)
    throw std::invalid_argument("sana: LiteLA shape mismatch");
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nH, T}),
      KOKKOS_LAMBDA(size_t h, size_t n) {
        float qv[kMaxHeadDim];
        float num[kMaxHeadDim];
        for (size_t e = 0; e < hd; ++e) {
          const float x = static_cast<float>(qkv(n, h * hd + e));
          qv[e]         = (x > 0.0f) ? x : 0.0f;
          num[e]        = 0.0f;
        }
        float den = 0.0f;
        for (size_t j = 0; j < T; ++j) {
          float s = 0.0f;
          for (size_t e = 0; e < hd; ++e) {
            const float y = static_cast<float>(qkv(j, D + h * hd + e));
            s += ((y > 0.0f) ? y : 0.0f) * qv[e];
          }
          den += s;
          for (size_t d = 0; d < hd; ++d)
            num[d] += static_cast<float>(qkv(j, 2 * D + h * hd + d)) * s;
        }
        const float inv = 1.0f / (den + eps);
        for (size_t d = 0; d < hd; ++d)
          o(n, h, d) = static_cast<Scalar>(num[d] * inv);
      });
}

// Depthwise 3x3 conv (groups=C, stride 1, pad 1, dilation 1) + fused bias
// over flattened token rows: row r = batch (r / T), spatial ((r % T) / Wp,
// (r % T) % Wp). y and x are [R, C]; w is [C, 3, 3].
template <class ExecSpace, class Scalar, class Mat, class W3, class Vec>
void depthwise3x3(const Mat& x, const W3& w, const Vec& bias, const Mat& y,
                  size_t T, size_t Hp, size_t Wp, const std::string& label) {
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t r, size_t c) {
        const size_t hw = r % T;
        const size_t h  = hw / Wp;
        const size_t wc = hw % Wp;
        float acc       = 0.0f;
        for (size_t kh = 0; kh < 3; ++kh) {
          // Padded input row h-1+kh is valid iff h+kh in [1, Hp].
          if (h + kh < 1 || h + kh > Hp) continue;
          for (size_t kw = 0; kw < 3; ++kw) {
            if (wc + kw < 1 || wc + kw > Wp) continue;
            const size_t rr =
                (r / T) * T + (h + kh - 1) * Wp + (wc + kw - 1);
            acc += static_cast<float>(x(rr, c)) *
                   static_cast<float>(w(c, kh, kw));
          }
        }
        y(r, c) = static_cast<Scalar>(acc + static_cast<float>(bias(c)));
      });
}

// GLU gating: out[r,h] = x[r,h] * silu(x[r,Hh+h]) over [R, 2Hh] -> [R, Hh].
template <class ExecSpace, class Scalar, class Mat>
void glu_chunk_silu(const Mat& x, const Mat& out, const std::string& label) {
  const size_t R = x.extent(0), Hh = out.extent(1);
  if (R == 0 || Hh == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, Hh}),
      KOKKOS_LAMBDA(size_t r, size_t h) {
        const float a = static_cast<float>(x(r, h));
        const float g = static_cast<float>(x(r, Hh + h));
        out(r, h) = static_cast<Scalar>(a * (g / (1.0f + expf(-g))));
      });
}

// Per-block adaLN-single conditioning: mod[b,k,d] = t0[b,k,d] + table[k,d]
// (torch: (scale_shift_table[None] + t.reshape(B,6,-1)).chunk(6, dim=1)).
// t0 is the shared [B,6,D] t_block output; table is the block's learned
// [6,D] base. mod aliases persistent scratch (recomputed per block).
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

// Flatten latents [B, C, H, W] to rows [B*T, C] (patch=1, T = H*W).
template <class ExecSpace, class Scalar, class Lat, class Mat>
void flatten_latents(const Lat& x, const Mat& flat, const std::string& label) {
  const size_t B = x.extent(0), C = x.extent(1);
  const size_t H = x.extent(2), W = x.extent(3);
  if (B == 0 || C == 0) return;
  const size_t T = H * W;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, C}),
      KOKKOS_LAMBDA(size_t b, size_t hw, size_t c) {
        flat(b * T + hw, c) = x(b, c, hw / W, hw % W);
      });
}

// DiT unpatchify: rows [B*Hp*Wp, Oc*P*P] -> [B, Oc, H, W].
template <class ExecSpace, class Scalar, class Mat, class Lat>
void unpatchify(const Mat& rows, const Lat& img, int patch,
                const std::string& label) {
  const size_t B = img.extent(0), Oc = img.extent(1);
  const size_t H = img.extent(2), W = img.extent(3);
  if (B == 0 || Oc == 0) return;
  const size_t P  = (size_t)patch;
  const size_t Hp = H / P, Wp = W / P;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {B, Hp, Wp, Oc}),
      KOKKOS_LAMBDA(size_t b, size_t hh, size_t ww, size_t c) {
        const size_t t = hh * Wp + ww;
        for (size_t kh = 0; kh < P; ++kh)
          for (size_t kw = 0; kw < P; ++kw) {
            const size_t col = c * P * P + kh * P + kw;
            img(b, c, hh * P + kh, ww * P + kw) =
                static_cast<Scalar>(rows(b * (Hp * Wp) + t, col));
          }
      });
}

}  // namespace Impl

// SanaMS DiT noise predictor (single forward, no sampler).
// Owns the TPL handle and persistent grow-only scratch, mirroring the
// structural pattern of Dinov3::Model (weights/handle/scratch + forward).
//
// Dependency note (SanaCore / SanaAttention / SanaConv): those core ops now
// exist in src/ and are mathematically equivalent to the Impl kernels below
// (silu/timestep/rmsnorm/modulate <-> SanaCore, LiteLA/cross bodies <->
// SanaAttention, depthwise/GLU/unpatchify <-> SanaConv), but they allocate
// temporaries per call. This model instead uses purpose-built kernels that
// write into persistent scratch so that steady state performs no View
// allocations (see class docs); the building blocks reused from the core
// library are gemm_bias/gemm, attention_mha_bidirectional, layernorm_
// forward, rmsnorm, and gelu_forward (Tanh).
//
// forward(x [B,C,H,W] latents, t [B] timesteps, y [B,L,Ccap] caption
// embeddings, out [B,Cout,H,W]):
//   x_embedder 1x1 (= per-pixel linear, patch 1) -> tokens [B*T, D]
//   t: sinusoidal embed (t_freq_dim) -> MLP(SiLU) -> t;
//      t0 = Linear(SiLU(t)) [B,6,D] (adaLN-single conditioning)
//   y: fc1 -> GELU-tanh -> fc2 -> scaled RMSNorm -> [B*L, D]
//   per block: adaLN-modulate(LN no-affine) -> LiteLA -> gated residual;
//     x += cross_attn(x, y) (unmodulated, ungated);
//     adaLN-modulate(LN no-affine) -> GLUMBConv -> gated residual
//   final: adaLN-modulate(LN no-affine, [2,D] table + t) -> linear ->
//     unpatchify -> [B,Cout,H,W] noise prediction.
//
// Steady state performs no View allocations: temporaries are capacity
// buffers carved via subview (grown on demand, never shrunk); per-batch
// head views and adaLN lanes are unmanaged aliases over those buffers.
// Views are passed by value (shallow, ref-counted) so unmanaged aliases
// over caller data can be built without const casts.
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
    if (cfg.patch_size != 1)
      throw std::invalid_argument(
          "Sana::Model: only patch_size == 1 is implemented (the "
          "SanaMS_600M_P1_D28 setting)");
    if ((size_t)cfg.cross_head_dim() > Impl::kMaxHeadDim ||
        (size_t)cfg.linear_head_dim > Impl::kMaxHeadDim)
      throw std::invalid_argument(
          "Sana::Model: head dim exceeds LiteLA kernel limit");
    ones_  = Vec("sana_ones", (size_t)cfg.hidden_size);
    zeros_ = Vec("sana_zeros", (size_t)cfg.hidden_size);
    Kokkos::deep_copy(ones_, static_cast<Scalar>(1.0f));
    Kokkos::deep_copy(zeros_, static_cast<Scalar>(0.0f));
  }

  // DiT forward: see class docs. t holds raw diffusion timesteps (float);
  // y holds caption embeddings of dim caption_channels, L <= maxlen.
  // out must be [B, out_channels, H, W].
  void forward(LatR x, TVec t, TokR y, LatR out) {
    const size_t B = x.extent(0);
    const size_t C = x.extent(1);
    const size_t H = x.extent(2), W = x.extent(3);
    if (B == 0) throw std::invalid_argument("Sana::Model::forward: B == 0");
    if (C != (size_t)cfg.in_channels)
      throw std::invalid_argument(
          "Sana::Model::forward: in_channels mismatch (got C=" +
          std::to_string(C) + ", expected " + std::to_string(cfg.in_channels) +
          ")");
    if (t.extent(0) != B)
      throw std::invalid_argument(
          "Sana::Model::forward: timestep batch mismatch");
    if (y.extent(0) != B)
      throw std::invalid_argument(
          "Sana::Model::forward: caption batch mismatch");
    const size_t L = y.extent(1);
    if (L == 0 || L > (size_t)cfg.model_max_length)
      throw std::invalid_argument(
          "Sana::Model::forward: caption length out of range");
    if (y.extent(2) != (size_t)cfg.caption_channels)
      throw std::invalid_argument(
          "Sana::Model::forward: caption_channels mismatch");
    if (out.extent(0) != B || out.extent(1) != (size_t)cfg.out_channels ||
        out.extent(2) != H || out.extent(3) != W)
      throw std::invalid_argument("Sana::Model::forward: out shape mismatch");
    const size_t P = (size_t)cfg.patch_size;
    if (H % P != 0 || W % P != 0)
      throw std::invalid_argument(
          "Sana::Model::forward: H and W must tile patch_size");
    const size_t T  = (H / P) * (W / P);
    const size_t Hp = H / P, Wp = W / P;
    ensure_scratch(B, T, L);

    // Hoisted dims (device lambdas must not capture `this`/`cfg`).
    const size_t D   = (size_t)cfg.hidden_size;
    const size_t Hh  = (size_t)cfg.mlp_hidden();
    const size_t nHl = (size_t)cfg.linear_num_heads;
    const size_t hdl = (size_t)cfg.linear_head_dim;
    const size_t nHc = (size_t)cfg.num_heads;
    const size_t hdc = D / nHc;
    const float EPS  = static_cast<float>(cfg.norm_eps);
    const float REPS = static_cast<float>(cfg.rms_eps);
    const float LEPS = static_cast<float>(cfg.litela_eps);
    const float CSCALE =
        1.0f / std::sqrt(static_cast<float>(hdc));
    const double TNS = cfg.timestep_norm_scale;
    const size_t NL  = (size_t)cfg.depth;
    const size_t Cc  = (size_t)cfg.caption_channels;
    const size_t R   = B * T, Ry = B * L;

    auto tok = sub_rows(tok_, 0, R);    // [R, D] token stream
    auto n1  = sub_rows(n1_, 0, R);     // [R, D] norm/mod buffer
    auto att = sub_rows(att_, 0, R);    // [R, D] attn/mlp out buffer
    auto qkv = sub_rows(qkv_, 0, R);    // [R, 3D]
    auto cq  = sub_rows(cq_, 0, R);     // [R, D] cross q
    auto inv = sub_rows(inv_, 0, R);    // [R, 2Hh] GLU pre-activation
    auto m2  = sub_rows(m2_, 0, R);     // [R, 2Hh] depthwise out
    auto glu = sub_rows(glu_, 0, R);    // [R, Hh] GLU output
    auto lin = sub_rows(lin_, 0, R);    // [R, Oc]
    auto xin = sub_rows(xin_, 0, R);    // [R, C] (C == in_channels)
    auto tf  = sub_rows(tf_, 0, B);     // [B, F]
    auto te  = sub_rows(te_, 0, B);     // [B, D]
    auto t0  = sub_rows3(t0_, 0, B);    // [B, 6, D] t_block output
    auto mod = sub_rows3(mod_, 0, B);    // [B, 6, D] table+t0 per block
    auto yp  = sub_rows(yp_, 0, Ry);    // [Ry, D] caption (normed)
    auto yn  = sub_rows(yn_, 0, Ry);    // [Ry, D] caption (fc2 out)
    auto ckk = sub_rows(ckk_, 0, Ry);   // [Ry, D] cross keys
    auto cvv = sub_rows(cvv_, 0, Ry);   // [Ry, D] cross values
    auto nobias = nobias_;
    auto ones = ones_, zeros = zeros_;

    // -- input embedding --
    Impl::flatten_latents<ExecSpace, Scalar>(x, xin, "sana_flatten");
    gemm_bias<ExecSpace, Scalar>(handle, xin, weights.x_proj_w,
                                 weights.x_proj_b, tok);

    // -- timestep conditioning --
    Impl::timestep_embed<ExecSpace, Scalar>(t, TNS, tf, "sana_tfreq");
    gemm_bias<ExecSpace, Scalar>(handle, tf, weights.t_mlp0_w,
                                 weights.t_mlp0_b, te);
    Impl::silu_inplace<ExecSpace, Scalar>(te, "sana_tsilu");
    gemm_bias<ExecSpace, Scalar>(handle, te, weights.t_mlp2_w,
                                 weights.t_mlp2_b, te);
    Impl::silu_inplace<ExecSpace, Scalar>(te, "sana_tsilu2");
    {
      UMatR t0f(t0.data(), B, 6 * D);
      gemm_bias<ExecSpace, Scalar>(handle, te, weights.t_block_w,
                                   weights.t_block_b, t0f);
    }

    // -- caption projection: fc1 -> GELU-tanh -> fc2 -> scaled RMSNorm --
    {
      UMatR yflat(y.data(), Ry, Cc);
      gemm_bias<ExecSpace, Scalar>(handle, yflat, weights.y_fc1_w,
                                   weights.y_fc1_b, yp);
      gelu_forward<ExecSpace, Scalar>(yp, yp, GeluVariant::Tanh);
      gemm_bias<ExecSpace, Scalar>(handle, yp, weights.y_fc2_w,
                                   weights.y_fc2_b, yn);
      rmsnorm<ExecSpace, Scalar>(yn, weights.y_norm_w, REPS, yp);
    }

    // -- transformer blocks --
    for (size_t l = 0; l < NL; ++l) {
      const auto& bw = weights.blocks[l];
      // adaLN-single: mod = scale_shift_table[None] + t0, chunked into the
      // six shift/scale/gate lanes (deduced layout).
      Impl::add_block_table<ExecSpace, Scalar>(t0, bw.scale_shift_table, mod,
                                               "sana_add_table");
      auto s_msa = Kokkos::subview(mod, Kokkos::ALL(), 0, Kokkos::ALL());
      auto c_msa = Kokkos::subview(mod, Kokkos::ALL(), 1, Kokkos::ALL());
      auto g_msa = Kokkos::subview(mod, Kokkos::ALL(), 2, Kokkos::ALL());
      auto s_mlp = Kokkos::subview(mod, Kokkos::ALL(), 3, Kokkos::ALL());
      auto c_mlp = Kokkos::subview(mod, Kokkos::ALL(), 4, Kokkos::ALL());
      auto g_mlp = Kokkos::subview(mod, Kokkos::ALL(), 5, Kokkos::ALL());

      // Self path: modulate(LN(x)) -> LiteLA -> gated residual.
      layernorm_forward<ExecSpace, Scalar>(tok, ones, zeros, EPS, n1);
      Impl::modulate<ExecSpace, Scalar>(n1, n1, s_msa, c_msa, T, "sana_mod1");
      gemm_bias<ExecSpace, Scalar>(handle, n1, bw.attn_qkv_w, nobias, qkv);
      // Per-batch LiteLA over the fused [T, 3D] rows, writing directly
      // into att rows (nHl*hdl == D, so the [T, nHl, hdl] output alias
      // over att's [T, D] rows is dense and valid).
      for (size_t b = 0; b < B; ++b) {
        UMatR qkvb(qkv.data() + (b * T) * 3 * D, T, 3 * D);
        UMat3R o(att.data() + (b * T) * D, T, nHl, hdl);
        Impl::litela_batch_qkv<ExecSpace, Scalar>(qkvb, o, nHl, hdl, LEPS,
                                                  "sana_litela");
      }
      gemm_bias<ExecSpace, Scalar>(handle, att, bw.attn_proj_w,
                                   bw.attn_proj_b, n1);
      Impl::gated_residual<ExecSpace, Scalar>(tok, n1, g_msa, T, "sana_gate1");

      // Cross path: x += proj(attn(q(x), kv(y))) (unmodulated, ungated).
      // kv is split into two GEMMs (k rows [0,D), v rows [D,2D) of kv_w;
      // row ranges are contiguous, so the unmanaged aliases are dense)
      // because the fused [L, 2D] rows cannot alias dense [L, H, Dh] views.
      gemm_bias<ExecSpace, Scalar>(handle, tok, bw.cross_q_w, bw.cross_q_b,
                                   cq);
      {
        UMatR kvw_k(bw.cross_kv_w.data(), D, D);
        UMatR kvw_v(bw.cross_kv_w.data() + D * D, D, D);
        UVec kvb_k(bw.cross_kv_b.data(), D);
        UVec kvb_v(bw.cross_kv_b.data() + D, D);
        UMatR ypf(yp.data(), Ry, D);
        gemm_bias<ExecSpace, Scalar>(handle, ypf, kvw_k, kvb_k, ckk);
        gemm_bias<ExecSpace, Scalar>(handle, ypf, kvw_v, kvb_v, cvv);
      }
      for (size_t b = 0; b < B; ++b) {
        UMat3R q3(cq.data() + (b * T) * D, T, nHc, hdc);
        UMat3R k3(ckk.data() + (b * L) * D, L, nHc, hdc);
        UMat3R v3(cvv.data() + (b * L) * D, L, nHc, hdc);
        UMat3R o3(n1.data() + (b * T) * D, T, nHc, hdc);
        attention_mha_bidirectional<ExecSpace, Scalar>(handle, q3, k3, v3, o3,
                                                       CSCALE);
      }
      gemm_bias<ExecSpace, Scalar>(handle, n1, bw.cross_proj_w,
                                   bw.cross_proj_b, att);
      {
        const size_t RR = att.extent(0), DD = att.extent(1);
        Kokkos::parallel_for(
            "sana_cross_res",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                              {RR, DD}),
            KOKKOS_LAMBDA(size_t r, size_t d) {
              tok(r, d) = static_cast<Scalar>(static_cast<float>(tok(r, d)) +
                                              static_cast<float>(att(r, d)));
            });
      }

      // FFN path: modulate(LN(x)) -> GLUMBConv -> gated residual.
      layernorm_forward<ExecSpace, Scalar>(tok, ones, zeros, EPS, n1);
      Impl::modulate<ExecSpace, Scalar>(n1, n1, s_mlp, c_mlp, T, "sana_mod2");
      gemm_bias<ExecSpace, Scalar>(handle, n1, bw.mlp_inv_w, bw.mlp_inv_b,
                                   inv);
      Impl::silu_inplace<ExecSpace, Scalar>(inv, "sana_inv_silu");
      Impl::depthwise3x3<ExecSpace, Scalar>(inv, bw.mlp_dw_w, bw.mlp_dw_b,
                                            m2, T, Hp, Wp, "sana_dw");
      Impl::glu_chunk_silu<ExecSpace, Scalar>(m2, glu, "sana_glu");
      gemm_bias<ExecSpace, Scalar>(handle, glu, bw.mlp_pt_w, nobias, att);
      Impl::gated_residual<ExecSpace, Scalar>(tok, att, g_mlp, T,
                                              "sana_gate2");
    }

    // -- final layer: modulate(LN, table + t) -> linear -> unpatchify --
    layernorm_forward<ExecSpace, Scalar>(tok, ones, zeros, EPS, n1);
    {
      // shift[d] = table[0,d] + t[b,d]; scale[d] = table[1,d] + t[b,d].
      const size_t RR = n1.extent(0);
      auto ftab       = weights.final_table;
      Kokkos::parallel_for(
          "sana_final_mod",
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
    gemm_bias<ExecSpace, Scalar>(handle, n1, weights.final_w, weights.final_b,
                                 lin);
    Impl::unpatchify<ExecSpace, Scalar>(lin, out, cfg.patch_size,
                                        "sana_unpatch");
  }

 private:
  // Grow-only capacity buffers (never shrunk; steady state = zero
  // allocations). tok_/n1_/att_ [Rcap, D]; qkv_ [Rcap, 3D]; cq_ [Rcap, D];
  // inv_/m2_ [Rcap, 2Hh]; glu_ [Rcap, Hh]; lin_ [Rcap, Oc];
  // xin_ [Rcap, Ic]; tf_ [Bcap, F]; te_ [Bcap, D]; t0_ [Bcap, 6, D];
  // yp_/yn_ [Rycap, D]; ckk_/cvv_ [Rycap, D] (split cross k/v);
  // mod_ [Bcap, 6, D] (per-block table + t0 conditioning).
  MatR tok_, n1_, att_, qkv_, cq_, inv_, m2_, glu_, lin_, xin_;
  MatR tf_, te_, yp_, yn_, ckk_, cvv_;
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
    const size_t D  = (size_t)cfg.hidden_size;
    const size_t Hh = (size_t)cfg.mlp_hidden();
    const size_t F  = (size_t)cfg.t_freq_dim;
    const size_t R = cap_B_ * cap_T_, Ry = cap_B_ * cap_L_;
    tok_ = MatR("sana_tok", R, D);
    n1_  = MatR("sana_n1", R, D);
    att_ = MatR("sana_att", R, D);
    qkv_ = MatR("sana_qkv", R, 3 * D);
    cq_  = MatR("sana_cq", R, D);
    inv_ = MatR("sana_inv", R, 2 * Hh);
    m2_  = MatR("sana_m2", R, 2 * Hh);
    glu_ = MatR("sana_glu", R, Hh);
    lin_ = MatR("sana_lin", R, (size_t)cfg.out_channels);
    xin_ = MatR("sana_xin", R, (size_t)cfg.in_channels);
    tf_  = MatR("sana_tf", cap_B_, F);
    te_  = MatR("sana_te", cap_B_, D);
    t0_  = Mat3R("sana_t0", cap_B_, 6, D);
    mod_ = Mat3R("sana_mod", cap_B_, 6, D);
    yp_  = MatR("sana_yp", Ry, D);
    yn_  = MatR("sana_yn", Ry, D);
    ckk_ = MatR("sana_ckk", Ry, D);
    cvv_ = MatR("sana_cvv", Ry, D);
  }
};

}  // namespace Sana
}  // namespace KokkosDNN

#endif
