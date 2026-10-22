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

// Team functor for the fused layernorm+modulate kernel below (one team per
// normalized row). Mirrors Impl::LayerNormAffineFunctor in
// src/KokkosDNN_LayerNorm.hpp exactly -- same TeamThreadRange two-pass FP32
// reductions (mean, then variance with eps) -- but the write-back applies the
// adaLN modulation y = norm(x)*(1+scale[b,d]) + shift[b,d] with b = r / T.
// Named functor (not nested lambdas): nvcc rejects an extended lambda inside
// another extended lambda. Templated on view types so LayoutStride subview
// lanes (shift/scale) are accepted. Op order in the write-back matches the
// two-pass layernorm_forward(ones, zeros) + modulate sequence in FP32
// (norm, then n*(1+s)+sh), so results agree with the unfused form to ~1e-7.
template <class ExecSpace, class Scalar, class Mat, class GMat>
struct LayerNormModulatedFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Mat x, y;
  GMat shift, scale;
  float eps;
  size_t D, T;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& member) const {
    const auto X  = x;
    const auto Y  = y;
    const auto SH = shift;
    const auto SC = scale;
    const float EPEPS  = eps;
    const size_t DD    = D;
    const size_t TT    = T;
    const size_t r     = static_cast<size_t>(member.league_rank());
    const size_t b     = (TT == 0) ? 0 : r / TT;
    const float Dn     = static_cast<float>(DD);
    float sum          = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, DD),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          acc += static_cast<float>(X(r, d));
        },
        sum);
    const float mean = sum / Dn;
    float sq         = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, DD),
        KOKKOS_LAMBDA(size_t d, float& acc) {
          float c = static_cast<float>(X(r, d)) - mean;
          acc += c * c;
        },
        sq);
    const float inv = 1.0f / sqrt(sq / Dn + EPEPS);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, DD), KOKKOS_LAMBDA(size_t d) {
          const float n = (static_cast<float>(X(r, d)) - mean) * inv;
          const float s = static_cast<float>(SC(b, d));
          const float h = static_cast<float>(SH(b, d));
          Y(r, d)       = static_cast<Scalar>(n * (1.0f + s) + h);
        });
  }
};

// Fused layernorm (no affine) + adaLN modulate in ONE pass:
//   y[r,d] = norm(x)[r,d]*(1+scale[b,d]) + shift[b,d], b = r / T,
// with shift/scale as [B, D] lanes (any layout). Replaces the
// layernorm_forward(tok, ones, zeros) + modulate(n1, n1, shift, scale) pair,
// removing one full read-modify-write pass over the [R, D] activations.
// x and y may alias (same in-place safety as the core layernorm: reductions
// only read x; writes start after the reductions complete).
template <class ExecSpace, class Scalar, class Mat, class GMat>
void layernorm_modulated(const Mat& x, const Mat& y, const GMat& shift,
                         const GMat& scale, float eps, size_t T,
                         const std::string& label) {
  const size_t R = x.extent(0), D = x.extent(1);
  if (R == 0 || D == 0) return;
  if (y.extent(0) != R || y.extent(1) != D)
    throw std::invalid_argument("sana: layernorm_modulated shape mismatch");
  if (shift.extent(1) != D || scale.extent(1) != D)
    throw std::invalid_argument("sana: layernorm_modulated lane mismatch");
  if (T != 0 && shift.extent(0) * T < R)
    throw std::invalid_argument("sana: layernorm_modulated batch mismatch");
  if (T != 0 && scale.extent(0) * T < R)
    throw std::invalid_argument("sana: layernorm_modulated batch mismatch");
  LayerNormModulatedFunctor<ExecSpace, Scalar, Mat, GMat> f{x, y, shift, scale,
                                                            eps, D, T};
  Kokkos::parallel_for(label, Kokkos::TeamPolicy<ExecSpace>(R, Kokkos::AUTO),
                       f);
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
// Flattened Rank-1 mapping (idx -> (r,d)) for near-roofline streaming;
// numerics identical to the MDRange form (one independent op per element).
template <class ExecSpace, class Scalar, class Mat>
void silu_inplace(const Mat& m, const std::string& label) {
  const size_t R = m.extent(0), D = m.extent(1);
  if (R == 0 || D == 0) return;
  const size_t N = R * D;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(size_t idx) {
        const size_t r = idx / D;
        const size_t d = idx % D;
        float v = static_cast<float>(m(r, d));
        m(r, d) = static_cast<Scalar>(v / (1.0f + expf(-v)));
      });
}

// Fused bias-add + SiLU, in-place: m[r,d] = silu(m[r,d] + bias[d]).
// Replaces the gemm_bias(..., bias, Y) + silu_inplace(Y) pair with
// gemm(...) + this kernel, removing one full read-modify-write pass over
// Y. FP32 op order matches the two-pass form ((x+b) then SiLU), so results
// are bit-identical for float Scalar (no intermediate rounding exists).
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

// Fused bias-add + tanh-GELU, in-place: m[r,d] = gelu_tanh(m[r,d]+bias[d]).
// Replaces gemm_bias(..., bias, Y) + gelu_forward(Y, Y, Tanh) with gemm(...)
// + this kernel (one pass saved). The formula replicates
// Impl::gelu_tanh_approx in src/KokkosDNN_Activation.hpp exactly:
//   0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// with (x+b) evaluated first in FP32, as in the two-pass form.
template <class ExecSpace, class Scalar, class Mat, class Vec>
void bias_add_gelu_tanh_inplace(const Mat& m, const Vec& bias,
                                const std::string& label) {
  const size_t R = m.extent(0), D = m.extent(1);
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t r, size_t d) {
        const float v =
            static_cast<float>(m(r, d)) + static_cast<float>(bias(d));
        const float c = 0.7978845608028654f;  // sqrt(2/pi)
        const float u = c * (v + 0.044715f * v * v * v);
        m(r, d) = static_cast<Scalar>(0.5f * v * (1.0f + tanhf(u)));
      });
}

// Fused bias-add + residual: tok[r,d] += proj[r,d] + bias[d].
// Replaces gemm_bias(..., bias, proj) + (tok += proj) with gemm(proj) +
// this kernel (one pass saved; same FP32 order: (proj+bias) then add).
template <class ExecSpace, class Scalar, class Mat, class Vec>
void bias_add_residual(const Mat& proj, const Vec& bias, const Mat& tok,
                       const std::string& label) {
  const size_t R = proj.extent(0), D = proj.extent(1);
  if (R == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, D}),
      KOKKOS_LAMBDA(size_t r, size_t d) {
        const float pb =
            static_cast<float>(proj(r, d)) + static_cast<float>(bias(d));
        tok(r, d) = static_cast<Scalar>(static_cast<float>(tok(r, d)) + pb);
      });
}

// Fused bias-add + gated residual: tok[r,d] += gate[b,d]*(proj[r,d]+bias[d]),
// b = r / T. Replaces gemm_bias(..., bias, proj) + gated_residual(tok, proj,
// gate) with gemm(proj) + this kernel (one pass saved; same FP32 order).
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

// Fused bias-add + unpatchify: img[b,c,hh*P+kh,ww*P+kw] = rows[t,col]+bias[col]
// (t = b*(Hp*Wp)+hh*Wp+ww, col = c*P*P+kh*P+kw). Replaces gemm_bias(...,
// bias, rows) + unpatchify(rows, img) with gemm(rows) + this kernel (one
// pass saved; same FP32 order: (row+bias) then store).
template <class ExecSpace, class Scalar, class Mat, class Lat, class Vec>
void bias_add_unpatchify(const Mat& rows, const Vec& bias, const Lat& img,
                         int patch, const std::string& label) {
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
            img(b, c, hh * P + kh, ww * P + kw) = static_cast<Scalar>(
                static_cast<float>(rows(b * (Hp * Wp) + t, col)) +
                static_cast<float>(bias(col)));
          }
      });
}
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
//
// Factored O(T*hd^2) formulation: per head, the shared key statistics
//   Ksum[e]   = sum_j reluK[j,e]
//   KV[d,e]   = sum_j V[j,d]*reluK[j,e]
// are precomputed once over keys (stage 1: one team per head streams keys
// once via team scratch, each member owning one e-column in registers;
// hd > 32 falls back to one (head,e) thread with a kMaxHeadDim stack
// column), then each query is a cheap hd x hd matvec
// (stage 2): num[d] = sum_e KV[d,e]*reluQ[e],
// den = sum_e Ksum[e]*reluQ[e], out = num/(den+eps). FP32 accumulation,
// ReLU placement, and eps handling match the unfactored O(T^2*hd) form up
// to floating-point summation order.
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
  // Stage 1: per-head shared statistics over keys. One TEAM per head streams
  // the T keys ONCE: team threads cooperatively load each K row-segment (hd)
  // and V row-segment (hd) into team scratch (2*hd floats), then each member
  // accumulates its owned e-column (Ksum scalar + VK column of hd floats in
  // registers) over the streamed keys. hd <= 32 keeps the per-member VK
  // column (<=32 floats) on-chip; larger hd (up to kMaxHeadDim) falls back
  // to the original per-(h,e) MDRange formulation.
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> Ksum("sana_litela_ksum",
                                                             nH, hd);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> KV("sana_litela_kv",
                                                            nH, hd, hd);
  constexpr size_t kTeamMaxHd = 32;
  if (hd <= kTeamMaxHd) {
    // Stage-1 performance notes (RTX 3090, nH=36, T=1024, hd=32): the
    // previous form launched one team per head with Kokkos::AUTO team size
    // (1024 threads on CUDA, of which only hd do work) and 2*hd team scratch
    // (K+V row segments), paying ~2*T wide-team barriers per launch at
    // single-block-per-SM occupancy (36 teams over 82 SMs). This form instead:
    // (a) caps the team at one warp (32), clamped at runtime via
    // team_size_max for backends that cannot field a warp (Serial runs
    // size 1) -- no backend branches; (b) splits the T-key
    // stream into nChunk chunks per head (288 teams at T=1024) for occupancy;
    // (c) shrinks scratch to a single hd-float V row-segment with K loaded
    // directly by its owner thread, fusing the K/V row loads under one
    // barrier; (d) drops the outer column-tile loop to a single tile on wide
    // teams (tiles remain only for team_size==1 correctness). Keys stream
    // once per (head, chunk); a tiny follow-up reduction folds chunks.
    size_t nChunk = 1;
    if (T >= 256)
      nChunk = 8;
    else if (T >= 64)
      nChunk = 4;
    if (nChunk > T) nChunk = T;
    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = typename team_policy::member_type;
    // V row-segment scratch only (hd floats); K is owner-loaded from global.
    const size_t shBytes = hd * sizeof(float);
    // Portable team size (one warp where the backend can field it): every
    // loop below strides on the runtime team size, so clamping the tuned
    // size via team_size_max (scratch accounted) stays correct everywhere
    // (Serial runs team_size 1) with no backend branches.
    constexpr int kStage1Team = 32;
    const size_t league = nH * nChunk;
    if (nChunk == 1) {
      auto stage1_body = KOKKOS_LAMBDA(const member_type& member) {
            const size_t h   = static_cast<size_t>(member.league_rank());
            const size_t tid = static_cast<size_t>(member.team_rank());
            const size_t tsz = static_cast<size_t>(member.team_size());
            float* vbuf =
                static_cast<float*>(member.team_scratch(0).get_shmem(shBytes));
            const size_t kbase = D + h * hd;
            const size_t vbase = 2 * D + h * hd;
            for (size_t e0 = 0; e0 < hd; e0 += tsz) {
              const size_t eo  = e0 + tid;
              const bool owned = eo < hd;
              float ksum_local   = 0.0f;
              float vk_local[32] = {};
              for (size_t j = 0; j < T; ++j) {
                // Cooperative V row-segment load (strided, coalesced); K is
                // owner-loaded below so one barrier covers the shared state.
                for (size_t e = tid; e < hd; e += tsz)
                  vbuf[e] = static_cast<float>(qkv(j, vbase + e));
                member.team_barrier();
                if (owned) {
                  const float y = static_cast<float>(qkv(j, kbase + eo));
                  const float rk = (y > 0.0f) ? y : 0.0f;
                  ksum_local += rk;
                  for (size_t d = 0; d < hd; ++d) vk_local[d] += vbuf[d] * rk;
                }
                member.team_barrier();
              }
              if (owned) {
                Ksum(h, eo) = ksum_local;
                for (size_t d = 0; d < hd; ++d) KV(h, d, eo) = vk_local[d];
              }
            }
      };
      team_policy probe1(1, 1);
      probe1.set_scratch_size(0, Kokkos::PerTeam(shBytes));
      const int tsize =
          std::max(1, std::min(kStage1Team, probe1.team_size_max(
                                                stage1_body,
                                                Kokkos::ParallelForTag{})));
      team_policy policy(static_cast<int>(league), tsize);
      policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));
      Kokkos::parallel_for(label, policy, stage1_body);
    } else {
      Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> Kpart(
          "sana_litela_kpart", league, hd);
      Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> KVpart(
          "sana_litela_kvpart", league, hd, hd);
      auto stage1_chunked_body = KOKKOS_LAMBDA(const member_type& member) {
            const size_t lr  = static_cast<size_t>(member.league_rank());
            const size_t h    = lr / nChunk;
            const size_t c    = lr % nChunk;
            const size_t j0   = (c * T) / nChunk;
            const size_t j1   = ((c + 1) * T) / nChunk;
            const size_t tid  = static_cast<size_t>(member.team_rank());
            const size_t tsz  = static_cast<size_t>(member.team_size());
            float* vbuf =
                static_cast<float*>(member.team_scratch(0).get_shmem(shBytes));
            const size_t kbase = D + h * hd;
            const size_t vbase = 2 * D + h * hd;
            for (size_t e0 = 0; e0 < hd; e0 += tsz) {
              const size_t eo  = e0 + tid;
              const bool owned = eo < hd;
              float ksum_local   = 0.0f;
              float vk_local[32] = {};
              for (size_t j = j0; j < j1; ++j) {
                for (size_t e = tid; e < hd; e += tsz)
                  vbuf[e] = static_cast<float>(qkv(j, vbase + e));
                member.team_barrier();
                if (owned) {
                  const float y = static_cast<float>(qkv(j, kbase + eo));
                  const float rk = (y > 0.0f) ? y : 0.0f;
                  ksum_local += rk;
                  for (size_t d = 0; d < hd; ++d) vk_local[d] += vbuf[d] * rk;
                }
                member.team_barrier();
              }
              if (owned) {
                Kpart(lr, eo) = ksum_local;
                for (size_t d = 0; d < hd; ++d) KVpart(lr, d, eo) = vk_local[d];
              }
            }
      };
      team_policy probe1c(1, 1);
      probe1c.set_scratch_size(0, Kokkos::PerTeam(shBytes));
      const int tsize =
          std::max(1, std::min(kStage1Team, probe1c.team_size_max(
                                                 stage1_chunked_body,
                                                 Kokkos::ParallelForTag{})));
      team_policy policy(static_cast<int>(league), tsize);
      policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));
      Kokkos::parallel_for(label, policy, stage1_chunked_body);
      // Fold chunk partials (chunking only changes fp32 summation order).
      Kokkos::parallel_for(
          label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nH, hd}),
          KOKKOS_LAMBDA(size_t h, size_t e) {
            float s = 0.0f;
            for (size_t c = 0; c < nChunk; ++c)
              s += Kpart(h * nChunk + c, e);
            Ksum(h, e) = s;
            for (size_t d = 0; d < hd; ++d) {
              float a = 0.0f;
              for (size_t c = 0; c < nChunk; ++c)
                a += KVpart(h * nChunk + c, d, e);
              KV(h, d, e) = a;
            }
          });
    }
  } else {
    Kokkos::parallel_for(
        label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nH, hd}),
        KOKKOS_LAMBDA(size_t h, size_t e) {
          float kvcol[kMaxHeadDim];
          for (size_t d = 0; d < hd; ++d) kvcol[d] = 0.0f;
          float ksum      = 0.0f;
          const size_t kc = D + h * hd + e;
          const size_t vb = 2 * D + h * hd;
          for (size_t j = 0; j < T; ++j) {
            const float y  = static_cast<float>(qkv(j, kc));
            const float rk = (y > 0.0f) ? y : 0.0f;
            ksum += rk;
            for (size_t d = 0; d < hd; ++d)
              kvcol[d] += static_cast<float>(qkv(j, vb + d)) * rk;
          }
          Ksum(h, e) = ksum;
          for (size_t d = 0; d < hd; ++d) KV(h, d, e) = kvcol[d];
        });
  }
  // Stage 2: per-query hd x hd matvec against the shared statistics.
  // Performance form: the previous (h,n) MDRange re-read each head's FULL
  // KV (hd x hd) + Ksum (hd) from GLOBAL memory per query (~147MB/call at
  // 600M: 36*1024 threads x 4KB each for 150KB of unique data). Now teams
  // over (head, query-chunk): each team loads its head's KV+Ksum into team
  // scratch ONCE, barriers, then team threads stride over the chunk's
  // queries computing den/num from scratch. Chunking (up to 8) keeps
  // 144-288 teams for occupancy (36 heads alone under-fill 82 SMs).
  // Summation order per query matches the MDRange form (e ascending), so
  // results agree up to fp32 chunking-level differences (well within 1e-4).
  if (hd <= kTeamMaxHd) {
    size_t nChunkQ = 1;
    if (T >= 256)
      nChunkQ = 8;
    else if (T >= 64)
      nChunkQ = 4;
    if (nChunkQ > T) nChunkQ = T;
    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = typename team_policy::member_type;
    const size_t shBytes2 = (hd * hd + hd) * sizeof(float);
    // Portable team size (tuned width where the backend can field it; every
    // loop below strides on the runtime team size, so the clamped size stays
    // correct -- Serial runs team_size 1 -- with no backend branches).
    constexpr int kStage2Team = 128;
    const size_t league2 = nH * nChunkQ;
    auto stage2_body = KOKKOS_LAMBDA(const member_type& member) {
          const size_t lr  = static_cast<size_t>(member.league_rank());
          const size_t tid = static_cast<size_t>(member.team_rank());
          const size_t tsz = static_cast<size_t>(member.team_size());
          const size_t h   = lr / nChunkQ;
          const size_t c   = lr % nChunkQ;
          const size_t j0  = (c * T) / nChunkQ;
          const size_t j1  = ((c + 1) * T) / nChunkQ;
          float* sh =
              static_cast<float*>(member.team_scratch(0).get_shmem(shBytes2));
          float* shKV = sh;
          float* shK  = sh + hd * hd;
          // Cooperative load of this head's KV + Ksum ONCE per team.
          const size_t nElem = hd * hd + hd;
          for (size_t idx = tid; idx < nElem; idx += tsz) {
            if (idx < hd * hd) {
              const size_t d = idx / hd;
              const size_t e = idx % hd;
              shKV[idx]      = KV(h, d, e);
            } else {
              const size_t e = idx - hd * hd;
              shK[e]         = Ksum(h, e);
            }
          }
          member.team_barrier();
          // Team threads cooperatively process the chunk's queries.
          for (size_t n = j0 + tid; n < j1; n += tsz) {
            float qv[32];
            for (size_t e = 0; e < hd; ++e) {
              const float x = static_cast<float>(qkv(n, h * hd + e));
              qv[e]         = (x > 0.0f) ? x : 0.0f;
            }
            float den = 0.0f;
            for (size_t e = 0; e < hd; ++e) den += shK[e] * qv[e];
            const float inv = 1.0f / (den + eps);
            for (size_t d = 0; d < hd; ++d) {
              float acc      = 0.0f;
              const float* r = shKV + d * hd;
              for (size_t e = 0; e < hd; ++e) acc += r[e] * qv[e];
              o(n, h, d) = static_cast<Scalar>(acc * inv);
            }
          }
    };
    team_policy probe2(1, 1);
    probe2.set_scratch_size(0, Kokkos::PerTeam(shBytes2));
    const int tsize2 =
        std::max(1, std::min(kStage2Team, probe2.team_size_max(
                                              stage2_body,
                                              Kokkos::ParallelForTag{})));
    team_policy policy2(static_cast<int>(league2), tsize2);
    policy2.set_scratch_size(0, Kokkos::PerTeam(shBytes2));
    Kokkos::parallel_for(label, policy2, stage2_body);
  } else {
    Kokkos::parallel_for(
        label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nH, T}),
        KOKKOS_LAMBDA(size_t h, size_t n) {
          float qv[kMaxHeadDim];
          float num[kMaxHeadDim];
          for (size_t e = 0; e < hd; ++e) {
            const float x = static_cast<float>(qkv(n, h * hd + e));
            qv[e]         = (x > 0.0f) ? x : 0.0f;
          }
          float den = 0.0f;
          for (size_t e = 0; e < hd; ++e) den += Ksum(h, e) * qv[e];
          for (size_t d = 0; d < hd; ++d) {
            float acc = 0.0f;
            for (size_t e = 0; e < hd; ++e) acc += KV(h, d, e) * qv[e];
            num[d] = acc;
          }
          const float inv = 1.0f / (den + eps);
          for (size_t d = 0; d < hd; ++d)
            o(n, h, d) = static_cast<Scalar>(num[d] * inv);
        });
  }
}

// Depthwise 3x3 conv (groups=C, stride 1, pad 1, dilation 1) + fused bias
// over flattened token rows: row r = batch (r / T), spatial ((r % T) / Wp,
// (r % T) % Wp). y and x are [R, C] with R = B*T, T = Hp*Wp; w is [C, 3, 3].
//
// Tiling: one team per (batch, output h-row, channel tile, w tile). The
// team cooperatively loads the up-to-3 input h-rows its outputs need
// (each Wp-token h-row contributes a (Wt+2)-wide window per w-tile,
// zero-filled where padding/borders apply) into team scratch, so the 9
// taps of neighboring outputs are served from on-chip shared memory
// instead of 9 scattered global loads each. Within a tile, consecutive
// threads own consecutive channels, so the cooperative loads hit
// contiguous channel data (coalesced; one transaction serves the warp's
// channels) and every per-element bounds check is hoisted: kh-row validity
// is computed once per team and w validity once per (kh, s) load row, so
// the compute loop indexes scratch with no branches at all. Out-of-bound
// taps contribute 0.0f * weight (exact no-op in FP32), hence accumulation
// matches the naive form tap-for-tap (bit-identical for float Scalar).
template <class ExecSpace, class Scalar, class Mat, class W3, class Vec>
void depthwise3x3(const Mat& x, const W3& w, const Vec& bias, const Mat& y,
                  size_t T, size_t Hp, size_t Wp, const std::string& label) {
  const size_t R = x.extent(0), C = x.extent(1);
  if (R == 0 || C == 0 || T == 0 || Hp == 0 || Wp == 0) return;
  const size_t B = R / T;
  if (B == 0) return;
  // Tile widths: Tc channels with one CUDA thread per channel, and Wt
  // tokens of Wp per w-tile. Scratch per team is 3*(Wt+2)*Tc floats
  // (~15 KB at 128x8), independent of Hp/Wp/C so large maps only grow the
  // league, not shared usage; at ~15 KB several teams stay resident per SM
  // (occupancy) while each input byte serves Wt outputs from scratch.
  constexpr size_t Tc = 128;
  constexpr size_t WtMax = 8;
  const size_t Wt = (Wp < WtMax) ? Wp : WtMax;
  const size_t nCt = (C + Tc - 1) / Tc;
  const size_t nWt = (Wp + Wt - 1) / Wt;
  const size_t league = B * Hp * nCt * nWt;
  const size_t shElems = 3 * (WtMax + 2) * Tc;
  const size_t shBytes = shElems * sizeof(float);
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename team_policy::member_type;
  // Portable team size (tuned width where the backend can field it; every
  // channel loop below strides on the runtime team size, so the clamped
  // size stays correct -- Serial runs team_size 1 -- with no backend
  // branches).
  constexpr int kConvTeam = 128;
  auto conv_body = KOKKOS_LAMBDA(const member_type& member) {
        const size_t lr  = static_cast<size_t>(member.league_rank());
        const size_t tid = static_cast<size_t>(member.team_rank());
        const size_t tsz = static_cast<size_t>(member.team_size());
        const size_t b   = lr / (Hp * nCt * nWt);
        const size_t rem = lr % (Hp * nCt * nWt);
        const size_t h   = rem / (nCt * nWt);
        const size_t ct  = (rem / nWt) % nCt;
        const size_t wt  = rem % nWt;
        const size_t c0  = ct * Tc;
        const size_t cT  = ((c0 + Tc) < C) ? Tc : (C - c0);  // live channels
        const size_t w0  = wt * Wt;
        size_t w1        = w0 + Wt;
        if (w1 > Wp) w1 = Wp;
        const size_t Wn = w1 - w0;  // live outputs in w
        float* sh =
            static_cast<float*>(member.team_scratch(0).get_shmem(shBytes));
        const size_t rowStride = WtMax + 2;
        const size_t bBase     = b * T;
        // Cooperative load of the 3 input h-rows x (Wt+2)-wide windows.
        // kh-row validity is hoisted per kh; w validity per (kh, s) row, so
        // the innermost channel loop is a straight contiguous stream.
        for (size_t kh = 0; kh < 3; ++kh) {
          const long ih   = static_cast<long>(h) + static_cast<long>(kh) - 1;
          const bool rok  = (ih >= 0) && (ih < static_cast<long>(Hp));
          const size_t ib = rok ? (bBase + static_cast<size_t>(ih) * Wp) : 0;
          for (size_t s = 0; s < Wt + 2; ++s) {
            const long ws = static_cast<long>(w0) + static_cast<long>(s) - 1;
            const bool ok =
                rok && (ws >= 0) && (ws < static_cast<long>(Wp));
            float* dst = sh + (kh * rowStride + s) * Tc;
            if (ok) {
              const size_t rr = ib + static_cast<size_t>(ws);
              for (size_t tc = tid; tc < cT; tc += tsz)
                dst[tc] = static_cast<float>(x(rr, c0 + tc));
            } else {
              for (size_t tc = tid; tc < cT; tc += tsz) dst[tc] = 0.0f;
            }
          }
        }
        member.team_barrier();
        // One thread per channel: weights/bias hoisted to registers, reused
        // over the tile's Wn outputs; scratch indexed branch-free.
        for (size_t tc = tid; tc < cT; tc += tsz) {
          const size_t c = c0 + tc;
          float f[3][3];
          for (size_t kh = 0; kh < 3; ++kh)
            for (size_t kw = 0; kw < 3; ++kw)
              f[kh][kw] = static_cast<float>(w(c, kh, kw));
          const float bs = static_cast<float>(bias(c));
          for (size_t wl = 0; wl < Wn; ++wl) {
            float acc = 0.0f;
            for (size_t kh = 0; kh < 3; ++kh) {
              const float* srow = sh + (kh * rowStride + wl) * Tc + tc;
              acc += srow[0] * f[kh][0];
              acc += srow[Tc] * f[kh][1];
              acc += srow[2 * Tc] * f[kh][2];
            }
            y(bBase + h * Wp + w0 + wl, c) =
                static_cast<Scalar>(acc + bs);
          }
        }
  };
  team_policy probe(1, 1);
  probe.set_scratch_size(0, Kokkos::PerTeam(shBytes));
  const int tsize =
      std::max(1, std::min(kConvTeam, probe.team_size_max(
                                          conv_body,
                                          Kokkos::ParallelForTag{})));
  team_policy policy(static_cast<int>(league), tsize);
  policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));
  Kokkos::parallel_for(label, policy, conv_body);
}

// GLU gating: out[r,h] = x[r,h] * silu(x[r,Hh+h]) over [R, 2Hh] -> [R, Hh].
// Flattened Rank-1 mapping (idx -> (r,h)); numerics identical (per-element).
template <class ExecSpace, class Scalar, class Mat>
void glu_chunk_silu(const Mat& x, const Mat& out, const std::string& label) {
  const size_t R = x.extent(0), Hh = out.extent(1);
  if (R == 0 || Hh == 0) return;
  const size_t N = R * Hh;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(size_t idx) {
        const size_t r = idx / Hh;
        const size_t h = idx % Hh;
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
// Dependency note: generic elementwise/norm/diffusion/conv ops live in src/
// (silu/timestep/rmsnorm/modulate <-> Activation/Diffusion/RMSNorm,
// depthwise/GLU/unpatchify <-> DepthwiseConv/PatchEmbed) and the fused Sana
// reference ops (LiteLA/cross/GLUMBConv) live in KokkosDNN_SanaOps.hpp, but
// they allocate temporaries per call. This model instead uses purpose-built
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
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), tf,
                            weights.t_mlp0_w, static_cast<Scalar>(0.0f), te,
                            "sana_tmlp0");
    Impl::bias_add_silu_inplace<ExecSpace, Scalar>(te, weights.t_mlp0_b,
                                                  "sana_tsilu");
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), te,
                            weights.t_mlp2_w, static_cast<Scalar>(0.0f), te,
                            "sana_tmlp2");
    Impl::bias_add_silu_inplace<ExecSpace, Scalar>(te, weights.t_mlp2_b,
                                                  "sana_tsilu2");
    {
      UMatR t0f(t0.data(), B, 6 * D);
      gemm_bias<ExecSpace, Scalar>(handle, te, weights.t_block_w,
                                   weights.t_block_b, t0f);
    }

    // -- caption projection: fc1 -> GELU-tanh -> fc2 -> scaled RMSNorm --
    {
      UMatR yflat(y.data(), Ry, Cc);
      gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), yflat,
                              weights.y_fc1_w, static_cast<Scalar>(0.0f), yp,
                              "sana_yfc1");
      Impl::bias_add_gelu_tanh_inplace<ExecSpace, Scalar>(yp, weights.y_fc1_b,
                                                         "sana_yact");
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
      Impl::layernorm_modulated<ExecSpace, Scalar>(tok, n1, s_msa, c_msa, EPS,
                                                  T, "sana_lnmod1");
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
      gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), att,
                              bw.attn_proj_w, static_cast<Scalar>(0.0f), n1,
                              "sana_attnproj");
      Impl::bias_add_gated_residual<ExecSpace, Scalar>(
          n1, bw.attn_proj_b, tok, g_msa, T, "sana_gate1");

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
      gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), n1,
                              bw.cross_proj_w, static_cast<Scalar>(0.0f), att,
                              "sana_crossproj");
      Impl::bias_add_residual<ExecSpace, Scalar>(att, bw.cross_proj_b, tok,
                                                "sana_cross_res");

      // FFN path: modulate(LN(x)) -> GLUMBConv -> gated residual.
      Impl::layernorm_modulated<ExecSpace, Scalar>(tok, n1, s_mlp, c_mlp, EPS,
                                                  T, "sana_lnmod2");
      gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), n1,
                              bw.mlp_inv_w, static_cast<Scalar>(0.0f), inv,
                              "sana_inv");
      Impl::bias_add_silu_inplace<ExecSpace, Scalar>(inv, bw.mlp_inv_b,
                                                    "sana_inv_silu");
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
    gemm<ExecSpace, Scalar>(handle, static_cast<Scalar>(1.0f), n1,
                            weights.final_w, static_cast<Scalar>(0.0f), lin,
                            "sana_final");
    Impl::bias_add_unpatchify<ExecSpace, Scalar>(
        lin, weights.final_b, out, cfg.patch_size, "sana_unpatch");
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
