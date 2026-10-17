#ifndef KOKKOSDNN_MOSSIMAGE_HPP
#define KOKKOSDNN_MOSSIMAGE_HPP

// Portable Kokkos ops for MOSS-Image-SFT (OpenMOSS-Team/MOSS-Image-SFT,
// ~5B text-to-image, same arch as MOSS-Image-Base): Qwen3.5 text encoder,
// MMDiT flow transformer (joint video+text self-attention), FLUX.2-small
// VAE decoder, flow-matching Euler sampler (CFG 3, shift 5, negative-branch
// skip layer 22).
//
// Grounded dimensions (transformer/config.json, text_encoder/config.json,
// vae/config.json, scheduler/scheduler_config.json @ main):
//   transformer: dim=3072, ffn_dim=8192, num_heads=24 (head_dim=128),
//     num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128,
//     rope_base=10000. Modulation Linear(dim -> 6*dim) slices
//     (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f); head
//     Linear(dim -> 2*dim) slices (shift, scale).
//   scheduler: shift=5.0, training_shift=1.0, 1000 train timesteps.
//   pipeline defaults: guidance_scale=3.0, uncond_skip_layers=(22,).
//   text encoder: Qwen3.5 (hidden 4096, 32 layers) -> passthrough (open gap,
//     see below); the in-checkpoint text refiner (Linear 4096->3072 +
//     2 _TextBlocks) is covered by the ops below.
//   VAE: AutoencoderKLFlux2, latent_channels=32, decode-only path with
//     per-channel BN-stat denormalization and a x2 depth-to-space
//     (inverse of patch_size [2,2] space-to-depth), scale_factor 16.
//
// Dispatch contract (no hand-written cublas/cudnn calls anywhere here):
//   * All linears (q/k/v/o, ffn_up_gate, ffn_down, time-embed MLP,
//     shared_block_modulation, video_head_modulation, text proj,
//     final linear) go through KokkosDNN::gemm / gemm_bias, which own the
//     cuBLAS ladder (sgemm / cublasGemmEx on CUDA+CUBLAS, portable tiled
//     or naive native kernels otherwise).
//   * Joint attention goes through attention_mha_bidirectional, which owns
//     the cuDNN ladder (try_cudnn_sdpa_bidir on CUDA+CUDNN for float /
//     bhalf_t, native streaming FlashAttention-style kernel otherwise).
//   * Everything in this header is a native portable elementwise kernel
//     (MDRangePolicy / RangePolicy only): RMSNorm+modulation fusion, gated
//     residuals, per-head QK RMSNorm, RoPE build/apply, SiTU activation,
//     CFG/SLG combiners, VAE denormalize/depth-to-space. Same rationale as
//     rmsnorm / adaln_modulate_forward: no TPL expresses these, so native
//     == dispatched by construction.
//
// All ops are templated <ExecSpace, Scalar>, accumulate in FP32, cast back
// to Scalar. Internals live in KokkosDNN::MossImage::Impl. No backend
// #ifdefs in native paths.

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Diffusion.hpp"
#include "KokkosDNN_Gemm.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_RMSNorm.hpp"

namespace KokkosDNN {
namespace MossImage {

// Model / sampler constants for OpenMOSS-Team/MOSS-Image-SFT.
struct Config {
  int dim = 3072;            // transformer width
  int ffn_dim = 8192;        // FFN intermediate dim
  int num_heads = 24;        // joint-attention heads (head_dim = 128)
  int num_layers = 42;       // MMDiT blocks
  int text_dim = 4096;       // Qwen3.5 hidden size (encoder passthrough)
  int freq_dim = 256;        // timestep sinusoidal dim (even)
  int latent_channels = 128;  // video token dim (patchified latents)
  double rope_base = 10000.0;
  double shift = 5.0;             // scheduler inference shift
  double guidance_scale = 3.0;    // default CFG scale
  std::vector<int> skip_layers = {22};  // default negative-branch skips

  int head_dim() const { return (num_heads > 0) ? dim / num_heads : 0; }

  static Config moss_image_sft() { return Config{}; }

  // Tiny config for unit tests (no checkpoint needed).
  static Config tiny_test() {
    Config c;
    c.dim = 32;
    c.ffn_dim = 64;
    c.num_heads = 4;  // head_dim 8
    c.num_layers = 2;
    c.text_dim = 16;
    c.freq_dim = 8;
    c.latent_channels = 8;
    c.skip_layers = {1};
    return c;
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("MossImage::Config::validate: " + m);
    };
    if (dim <= 0) bad("dim must be > 0");
    if (ffn_dim <= 0) bad("ffn_dim must be > 0");
    if (num_heads <= 0) bad("num_heads must be > 0");
    if (dim % num_heads != 0) bad("dim must be divisible by num_heads");
    if (head_dim() % 8 != 0) bad("head_dim must be divisible by 8 (RoPE)");
    if (num_layers <= 0) bad("num_layers must be > 0");
    if (text_dim <= 0) bad("text_dim must be > 0");
    if (freq_dim <= 0 || freq_dim % 2 != 0)
      bad("freq_dim must be a positive even number");
    if (latent_channels <= 0) bad("latent_channels must be > 0");
    if (!(rope_base > 0) || !std::isfinite(rope_base))
      bad("rope_base must be positive and finite");
    if (!(shift > 0) || !std::isfinite(shift))
      bad("shift must be positive and finite");
    if (!std::isfinite(guidance_scale) || guidance_scale < 0)
      bad("guidance_scale must be finite and nonnegative");
    for (int l : skip_layers)
      if (l < 0 || l >= num_layers)
        bad("skip_layers must hold block indices in [0, num_layers)");
  }
};

namespace Impl {

// Device-callable SiTU scalar (MOSS DiT/TextBlock FFN activation):
//   situ(h, g) = (4 * tanh(g/4) * sigmoid(g)) * (25 * tanh(h/25)).
// FP32 in/out; callers cast to/from Scalar.
KOKKOS_INLINE_FUNCTION float moss_situ_scalar(float h, float g) {
  const float sig = 1.0f / (1.0f + expf(-g));
  return (4.0f * tanhf(g / 4.0f) * sig) * (25.0f * tanhf(h / 25.0f));
}

}  // namespace Impl

// Shift-aware timestep embedding: raw t in [0,1] is first mapped through
// the MOSS/SD3 schedule shift ts = shift*t / (1 + (shift-1)*t) (same
// mapping as MOSSImageScheduler's grid()), then embedded with the shared
// timestep_sinusoidal op (cos in [0,H), sin in [H,F), FP32).
// The transformer itself consumes already-shifted sigmas, so this helper
// covers scheduler-side schedule construction and any caller holding raw
// [0,1] times. shift must be positive and finite.
// Dispatch: elementwise native (shift map) + timestep_sinusoidal (native
// elementwise, no TPL).
template <class ExecSpace>
void moss_timestep_shifted(
    const Kokkos::View<float*, ExecSpace>& t, float shift,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_timestep_shifted") {
  if (!(shift > 0.0f) || !std::isfinite(shift))
    throw std::invalid_argument("moss_timestep_shifted: shift must be > 0");
  const size_t N = t.extent(0);
  if (out.extent(0) != N)
    throw std::invalid_argument("moss_timestep_shifted: dimension mismatch");
  if (N == 0) return;
  Kokkos::View<float*, ExecSpace> ts("KokkosDNN::moss_ts", N);
  Kokkos::parallel_for(
      label + "_map", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(size_t i) {
        const float ti = t(i);
        ts(i) = shift * ti / (1.0f + (shift - 1.0f) * ti);
      });
  timestep_sinusoidal<ExecSpace>(ts, out, label + "_embed");
}

// Fused RMSNorm + AdaLN-zero modulation for one MMDiT sub-block:
//   y[b,t,d] = rmsnorm(x[b,t,:]; w, eps)[d] * (1 + scale[b,d]) + shift[b,d]
// x, y: [B, T, D]; w: [D]; shift, scale: [B, D] (one row of the 6-way
// shared_block_modulation slice: att-{shift,scale} or ffn-{shift,scale}).
// Same FP32 RMSNorm math as rmsnorm(); the residual gate multiplies
// outside (see moss_gate_residual_forward). In-place safe (x may alias y).
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_rmsnorm_modulate_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& shift,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& scale,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label = "KokkosDNN::moss_rmsnorm_modulate") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (y.extent(0) != B || y.extent(1) != T || y.extent(2) != D ||
      w.extent(0) != D || shift.extent(0) != B || shift.extent(1) != D ||
      scale.extent(0) != B || scale.extent(1) != D)
    throw std::invalid_argument("moss_rmsnorm_modulate_forward: dim mismatch");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("moss_rmsnorm_modulate_forward: bad eps");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {B, T}),
      KOKKOS_LAMBDA(size_t b, size_t t) {
        float ss = 0;
        for (size_t d = 0; d < D; ++d) {
          float v = static_cast<float>(x(b, t, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / static_cast<float>(D) + eps);
        for (size_t d = 0; d < D; ++d) {
          float n = static_cast<float>(x(b, t, d)) * inv *
                    static_cast<float>(w(d));
          y(b, t, d) = static_cast<Scalar>(
              n * (1.0f + static_cast<float>(scale(b, d))) +
              static_cast<float>(shift(b, d)));
        }
      });
}

// AdaLN-zero gated residual: out = x + gate[b,d] * branch, all [B,T,D] /
// [B,D]. Covers the attention residual (* gate_a), the FFN residual
// (* gate_f), and skipped-block semantics (a skipped block is the identity,
// i.e. gate 0 with any branch). In-place safe (out may alias x/branch).
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_gate_residual_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& branch,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_gate_residual") {
  const size_t B = x.extent(0), T = x.extent(1), D = x.extent(2);
  if (branch.extent(0) != B || branch.extent(1) != T || branch.extent(2) != D ||
      out.extent(0) != B || out.extent(1) != T || out.extent(2) != D ||
      gate.extent(0) != B || gate.extent(1) != D)
    throw std::invalid_argument("moss_gate_residual_forward: dim mismatch");
  if (B == 0 || T == 0 || D == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, D}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t d) {
        out(b, t, d) = static_cast<Scalar>(
            static_cast<float>(x(b, t, d)) +
            static_cast<float>(gate(b, d)) *
                static_cast<float>(branch(b, t, d)));
      });
}

// Batched per-head QK RMSNorm (MMDiT joint-attention q_norm / k_norm and
// the in-checkpoint text-refiner q_norm / k_norm):
//   out[b,s,h,:] = q[b,s,h,:] / rms(q[b,s,h,:], eps) * w[:]
// q, out: [B, S, H, Dh]; w: [Dh] (shared across heads, as in the
// checkpoint). Same FP32 math as qk_rmsnorm3d(), extended to a batch of
// joint sequences. eps = 1e-6 in the checkpoint. In-place safe.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_qk_rmsnorm_forward(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& q,
    const Kokkos::View<Scalar*, ExecSpace>& w, float eps,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_qk_rmsnorm") {
  const size_t B = q.extent(0), S = q.extent(1), H = q.extent(2),
               Dh = q.extent(3);
  if (out.extent(0) != B || out.extent(1) != S || out.extent(2) != H ||
      out.extent(3) != Dh || w.extent(0) != Dh)
    throw std::invalid_argument("moss_qk_rmsnorm_forward: dim mismatch");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("moss_qk_rmsnorm_forward: bad eps");
  if (B == 0 || S == 0 || H == 0 || Dh == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, S, H}),
      KOKKOS_LAMBDA(size_t b, size_t s, size_t h) {
        float ss = 0;
        for (size_t d = 0; d < Dh; ++d) {
          float v = static_cast<float>(q(b, s, h, d));
          ss += v * v;
        }
        float inv = 1.0f / sqrt(ss / static_cast<float>(Dh) + eps);
        for (size_t d = 0; d < Dh; ++d)
          out(b, s, h, d) = static_cast<Scalar>(
              static_cast<float>(q(b, s, h, d)) * inv *
              static_cast<float>(w(d)));
      });
}

// MOSS 4-coordinate RoPE tensor build: out [1, S, Dh] with
// S = height*width + text_length, Dh % 8 == 0 (num_bands = Dh/8).
// Image token (r, c): coords (0, text_length, r - (H-1)/2, c - (W-1)/2);
// text token i: (0, i, 0, 0). bands[k] = 1 / base^(k/num_bands).
// Layout matches the reference (sin block then cos block per coordinate):
//   out[0, s, m*2*nb + k]      = sin(coord[m] * bands[k])
//   out[0, s, m*2*nb + nb + k] = cos(coord[m] * bands[k]).
// FP32 throughout (reference builds fp32 bands on CPU and fp32 phase).
// Dispatch: native elementwise (no TPL).
template <class ExecSpace>
void moss_rope_build_forward(
    int height, int width, int text_length, float base,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_rope_build") {
  if (height <= 0 || width <= 0 || text_length < 0)
    throw std::invalid_argument("moss_rope_build_forward: bad geometry");
  if (!(base > 0.0f) || !std::isfinite(base))
    throw std::invalid_argument("moss_rope_build_forward: bad base");
  const size_t S = static_cast<size_t>(height) * static_cast<size_t>(width) +
                   static_cast<size_t>(text_length);
  const size_t Dh = out.extent(2);
  if (out.extent(0) != 1 || out.extent(1) != S)
    throw std::invalid_argument("moss_rope_build_forward: dim mismatch");
  if (Dh == 0 || Dh % 8 != 0)
    throw std::invalid_argument(
        "moss_rope_build_forward: head_dim must be a nonzero multiple of 8");
  if (S == 0) return;
  const size_t nb = Dh / 8;
  const size_t img = static_cast<size_t>(height) * static_cast<size_t>(width);
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {S, nb}),
      KOKKOS_LAMBDA(size_t s, size_t k) {
        const float band =
            1.0f / powf(base, static_cast<float>(k) / static_cast<float>(nb));
        float coord[4];
        if (s < img) {
          const size_t r = s / static_cast<size_t>(width);
          const size_t c = s % static_cast<size_t>(width);
          coord[0] = 0.0f;
          coord[1] = static_cast<float>(text_length);
          coord[2] = static_cast<float>(r) -
                     static_cast<float>(height - 1) / 2.0f;
          coord[3] =
              static_cast<float>(c) - static_cast<float>(width - 1) / 2.0f;
        } else {
          coord[0] = 0.0f;
          coord[1] = static_cast<float>(s - img);
          coord[2] = 0.0f;
          coord[3] = 0.0f;
        }
        for (int m = 0; m < 4; ++m) {
          const float ph = coord[m] * band;
          out(0, s, m * 2 * nb + k) = sinf(ph);
          out(0, s, m * 2 * nb + nb + k) = cosf(ph);
        }
      });
}

// NeoX-style half-rotation with a pipeline-layout rope tensor:
// halves sn/cs = rope[0, s, :Dh/2] / rope[0, s, Dh/2:] are each doubled to
// full width (S_full = cat(sn, sn), C_full = cat(cs, cs)), then with
// x = [a, b] halves: out = x * C_full + cat(-b, a) * S_full.
// x, out: [B, S, H, Dh]; rope: [1, S, Dh] from moss_rope_build_forward.
// Apply separately to q and k (same rope). In-place safe.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_rope_apply_forward(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& rope,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_rope_apply") {
  const size_t B = x.extent(0), S = x.extent(1), H = x.extent(2),
               Dh = x.extent(3);
  if (out.extent(0) != B || out.extent(1) != S || out.extent(2) != H ||
      out.extent(3) != Dh || rope.extent(0) != 1 || rope.extent(1) != S ||
      rope.extent(2) != Dh)
    throw std::invalid_argument("moss_rope_apply_forward: dim mismatch");
  if (Dh == 0 || Dh % 2 != 0)
    throw std::invalid_argument(
        "moss_rope_apply_forward: head_dim must be even");
  if (B == 0 || S == 0 || H == 0) return;
  const size_t Hh = Dh / 2;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, S, H}),
      KOKKOS_LAMBDA(size_t b, size_t s, size_t h) {
        for (size_t d = 0; d < Hh; ++d) {
          const float sn = rope(0, s, d), cs = rope(0, s, d + Hh);
          const float a = static_cast<float>(x(b, s, h, d));
          const float bv = static_cast<float>(x(b, s, h, d + Hh));
          out(b, s, h, d) = static_cast<Scalar>(a * cs - bv * sn);
          out(b, s, h, d + Hh) = static_cast<Scalar>(bv * cs + a * sn);
        }
      });
}

// Joint video+text self-attention for one MMDiT block: dense bidirectional
// attention over the concatenated [video, valid-text] sequence
// (per-sample text padding is truncated before concat, so no mask is
// needed). q/k are RoPE'd + QK-normed beforehand (moss_rope_apply_forward
// + moss_qk_rmsnorm_forward); v is the plain projection. Thin wrapper over
// attention_mha_bidirectional so the TPL ladder stays in one place.
// Dispatch: float/bhalf_t try cuDNN SDPA first on CUDA+CUDNN builds
// (try_cudnn_sdpa_bidir), native streaming kernel otherwise / on failure.
template <class ExecSpace, class Scalar, class QView, class KView, class VView,
          class OView>
void moss_joint_attention_forward(
    const DNNHandle<ExecSpace>& handle, const QView& q, const KView& k,
    const VView& v, const OView& out, float scale,
    const std::string& label = "KokkosDNN::moss_joint_attn") {
  attention_mha_bidirectional<ExecSpace, Scalar>(handle, q, k, v, out, scale,
                                                 label);
}

// Classifier-free guidance combiner (flow-matching velocity space, FP32):
//   out = uncond + guidance * (cond - uncond).
// guidance == 1 selects cond; 0 selects uncond; the MOSS default is 3.0.
// guidance must be finite and >= 0. Rank-2 [N, D] overload covers flattened
// token batches; rank-4 [B, C, H, W] covers latent velocity grids.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_cfg_combine_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& uncond,
    float guidance,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_cfg") {
  const size_t N = cond.extent(0), D = cond.extent(1);
  if (uncond.extent(0) != N || uncond.extent(1) != D || out.extent(0) != N ||
      out.extent(1) != D)
    throw std::invalid_argument("moss_cfg_combine_forward: dim mismatch");
  if (!std::isfinite(guidance) || guidance < 0.0f)
    throw std::invalid_argument("moss_cfg_combine_forward: bad guidance");
  if (N == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, D}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        const float c = static_cast<float>(cond(i, j));
        const float u = static_cast<float>(uncond(i, j));
        out(i, j) = static_cast<Scalar>(u + guidance * (c - u));
      });
}

template <class ExecSpace, class Scalar>
void moss_cfg_combine_forward(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& uncond,
    float guidance,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_cfg") {
  const size_t B = cond.extent(0), C = cond.extent(1), H = cond.extent(2),
               W = cond.extent(3);
  if (uncond.extent(0) != B || uncond.extent(1) != C ||
      uncond.extent(2) != H || uncond.extent(3) != W || out.extent(0) != B ||
      out.extent(1) != C || out.extent(2) != H || out.extent(3) != W)
    throw std::invalid_argument("moss_cfg_combine_forward: dim mismatch");
  if (!std::isfinite(guidance) || guidance < 0.0f)
    throw std::invalid_argument("moss_cfg_combine_forward: bad guidance");
  if (B == 0 || C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                       {B, C, H, W}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
        const float cc = static_cast<float>(cond(b, c, h, w));
        const float u = static_cast<float>(uncond(b, c, h, w));
        out(b, c, h, w) = static_cast<Scalar>(u + guidance * (cc - u));
      });
}

// Skip-layer guidance correction (Diffusers SkipLayerGuidance form):
//   out = base + slg_scale * (cond - cond_skip).
// With guidance != 1, base is the CFG-combined prediction; with
// guidance == 1 the caller passes base == cond_skip, giving
// cond_skip + slg_scale * (cond - cond_skip). slg_scale == 1 recovers
// cond. slg_scale must be finite and >= 0 (0 disables the correction).
// Rank-2 [N, D] and rank-4 [B, C, H, W] overloads.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_slg_correct_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& base,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& cond_skip,
    float slg_scale,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_slg") {
  const size_t N = base.extent(0), D = base.extent(1);
  if (cond.extent(0) != N || cond.extent(1) != D ||
      cond_skip.extent(0) != N || cond_skip.extent(1) != D ||
      out.extent(0) != N || out.extent(1) != D)
    throw std::invalid_argument("moss_slg_correct_forward: dim mismatch");
  if (!std::isfinite(slg_scale) || slg_scale < 0.0f)
    throw std::invalid_argument("moss_slg_correct_forward: bad slg_scale");
  if (N == 0 || D == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, D}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        out(i, j) = static_cast<Scalar>(
            static_cast<float>(base(i, j)) +
            slg_scale * (static_cast<float>(cond(i, j)) -
                         static_cast<float>(cond_skip(i, j))));
      });
}

template <class ExecSpace, class Scalar>
void moss_slg_correct_forward(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& base,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& cond,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& cond_skip,
    float slg_scale,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_slg") {
  const size_t B = base.extent(0), C = base.extent(1), H = base.extent(2),
               W = base.extent(3);
  if (cond.extent(0) != B || cond.extent(1) != C || cond.extent(2) != H ||
      cond.extent(3) != W || cond_skip.extent(0) != B ||
      cond_skip.extent(1) != C || cond_skip.extent(2) != H ||
      cond_skip.extent(3) != W || out.extent(0) != B || out.extent(1) != C ||
      out.extent(2) != H || out.extent(3) != W)
    throw std::invalid_argument("moss_slg_correct_forward: dim mismatch");
  if (!std::isfinite(slg_scale) || slg_scale < 0.0f)
    throw std::invalid_argument("moss_slg_correct_forward: bad slg_scale");
  if (B == 0 || C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                       {B, C, H, W}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
        out(b, c, h, w) = static_cast<Scalar>(
            static_cast<float>(base(b, c, h, w)) +
            slg_scale * (static_cast<float>(cond(b, c, h, w)) -
                         static_cast<float>(cond_skip(b, c, h, w))));
      });
}

// Host-side skip-layer validation (negative-branch skips, default {22}):
// throws unless every index is an integer block index in [0, depth).
// Mirrors the reference _validate_skip_layers (order-free set semantics;
// the returned vector preserves input order).
inline std::vector<int> moss_checked_skip_layers(const std::vector<int>& layers,
                                                 int depth) {
  if (depth <= 0)
    throw std::invalid_argument(
        "moss_checked_skip_layers: depth must be > 0");
  for (int l : layers)
    if (l < 0 || l >= depth)
      throw std::invalid_argument(
          "moss_checked_skip_layers: index out of range");
  return layers;
}

// FLUX.2-small VAE latent denormalization (decode path, differs from the
// Sana DC-AE path): out = z * sqrt(var + eps) + mean, per channel, from
// the VAE BatchNorm running stats (running_var / running_mean, eps =
// batch_norm_eps = 1e-4). z, out: [B, C, H, W]; mean/var: [C]. The
// denormalized grid then passes through moss_vae_depth_to_space_2x before
// vae.decode. In-place safe. Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_vae_denormalize_forward(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& z,
    const Kokkos::View<float*, ExecSpace>& mean,
    const Kokkos::View<float*, ExecSpace>& var, float eps,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_vae_denorm") {
  const size_t B = z.extent(0), C = z.extent(1), H = z.extent(2),
               W = z.extent(3);
  if (out.extent(0) != B || out.extent(1) != C || out.extent(2) != H ||
      out.extent(3) != W || mean.extent(0) != C || var.extent(0) != C)
    throw std::invalid_argument("moss_vae_denormalize_forward: dim mismatch");
  if (!(eps >= 0.0f) || !std::isfinite(eps))
    throw std::invalid_argument("moss_vae_denormalize_forward: bad eps");
  if (B == 0 || C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                       {B, C, H, W}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t h, size_t w) {
        const float rstd = sqrtf(var(c) + eps);
        out(b, c, h, w) = static_cast<Scalar>(static_cast<float>(z(b, c, h, w)) *
                                                  rstd +
                                              mean(c));
      });
}

// Inverse of the FLUX.2 patch_size [2,2] space-to-depth: [B, C, H, W] ->
// [B, C/4, 2H, 2W] with out[b, co, 2h+i, 2w+j] = z[b, co*4+i*2+j, h, w].
// C must be divisible by 4. Pure data movement, exact for any Scalar.
// Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_vae_depth_to_space_2x(
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& z,
    const Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_vae_d2s") {
  const size_t B = z.extent(0), C = z.extent(1), H = z.extent(2),
               W = z.extent(3);
  if (C % 4 != 0)
    throw std::invalid_argument(
        "moss_vae_depth_to_space_2x: channels must be divisible by 4");
  if (out.extent(0) != B || out.extent(1) != C / 4 || out.extent(2) != 2 * H ||
      out.extent(3) != 2 * W)
    throw std::invalid_argument("moss_vae_depth_to_space_2x: dim mismatch");
  if (B == 0 || C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                       {B, C / 4, H, W}),
      KOKKOS_LAMBDA(size_t b, size_t co, size_t h, size_t w) {
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j)
            out(b, co, 2 * h + i, 2 * w + j) =
                z(b, co * 4 + i * 2 + j, h, w);
      });
}

// SiTU FFN activation: out = situ(hidden, gate) elementwise (see
// Impl::moss_situ_scalar). hidden/gate/out: [N, F] (rank-2, e.g. flattened
// tokens) or [B, T, F] (rank-3). FP32, cast back to Scalar; in-place safe
// with respect to hidden/gate. Dispatch: native elementwise (no TPL).
template <class ExecSpace, class Scalar>
void moss_situ_forward(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& hidden,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_situ") {
  const size_t N = hidden.extent(0), F = hidden.extent(1);
  if (gate.extent(0) != N || gate.extent(1) != F || out.extent(0) != N ||
      out.extent(1) != F)
    throw std::invalid_argument("moss_situ_forward: dim mismatch");
  if (N == 0 || F == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, F}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        out(i, j) = static_cast<Scalar>(Impl::moss_situ_scalar(
            static_cast<float>(hidden(i, j)), static_cast<float>(gate(i, j))));
      });
}

template <class ExecSpace, class Scalar>
void moss_situ_forward(
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& hidden,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& gate,
    const Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& out,
    const std::string& label = "KokkosDNN::moss_situ") {
  const size_t B = hidden.extent(0), T = hidden.extent(1), F = hidden.extent(2);
  if (gate.extent(0) != B || gate.extent(1) != T || gate.extent(2) != F ||
      out.extent(0) != B || out.extent(1) != T || out.extent(2) != F)
    throw std::invalid_argument("moss_situ_forward: dim mismatch");
  if (B == 0 || T == 0 || F == 0) return;
  Kokkos::parallel_for(
      label,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, F}),
      KOKKOS_LAMBDA(size_t b, size_t t, size_t f) {
        out(b, t, f) = static_cast<Scalar>(Impl::moss_situ_scalar(
            static_cast<float>(hidden(b, t, f)),
            static_cast<float>(gate(b, t, f))));
      });
}

}  // namespace MossImage
}  // namespace KokkosDNN

#endif
