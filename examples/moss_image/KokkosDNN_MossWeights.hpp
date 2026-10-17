#ifndef KOKKOSDNN_MOSS_WEIGHTS_HPP
#define KOKKOSDNN_MOSS_WEIGHTS_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "KokkosDNN_MossConfig.hpp"

namespace KokkosDNN {
namespace Moss {

// Default compute/storage scalar for the MOSS example: float.
using DefaultScalar = float;

// Layout convention (documented choice, mirrors the Sana example): 2D
// weights keep the PyTorch [out_features, in_features] row-major layout and
// are consumed directly by gemm_bias/gemm, whose W operand is already
// [N, K] = [out, in] (no transpose needed). The loaders copy raw float32
// storage verbatim (no host transpose); 4D patch-embed conv weights with
// shape [D, in_ch, patch, patch] are loaded into [D, in_ch*patch*patch]
// row-major views (flattened in torch order), matching the patchify order
// used by the model (kh outer, kw inner, channel innermost flattened the
// same way: col = (c*P+kh)*P+kw).
template <class ExecSpace, class Scalar = DefaultScalar>
struct BlockWeights {
  // adaLN-single conditioning table [6, D]: shift/scale/gate x (attn, mlp).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> scale_shift_table;
  // Joint self+cross attention: fused token QKV [3D, D] (no bias),
  // caption K/V projection [2D, D] (no bias), output proj [D, D] + bias.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_qkv_w;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_cap_kv_w;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> attn_proj_w;
  Kokkos::View<Scalar*, ExecSpace> attn_proj_b;
  // FFN (GELU-tanh): fc1 [H, D] + bias, fc2 [D, H] + bias.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mlp_fc1_w;
  Kokkos::View<Scalar*, ExecSpace> mlp_fc1_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> mlp_fc2_w;
  Kokkos::View<Scalar*, ExecSpace> mlp_fc2_b;

  // Drop every device allocation (staged execution: release the encoder or
  // DiT weights before the next phase is loaded so a 24GB GPU never holds
  // both at once). Assigning {} releases the View's reference; the caller
  // must Kokkos::fence() before loading the next phase.
  void release() {
    scale_shift_table = {};
    attn_qkv_w = {};
    attn_cap_kv_w = {};
    attn_proj_w = {};
    attn_proj_b = {};
    mlp_fc1_w = {};
    mlp_fc1_b = {};
    mlp_fc2_w = {};
    mlp_fc2_b = {};
  }
};

// Full MOSS DiT weights (device views in Scalar).
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  // Patch embed: proj [D, in_ch*P*P] + bias [D].
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> x_proj_w;
  Kokkos::View<Scalar*, ExecSpace> x_proj_b;
  // Timestep MLP: mlp0 [D, F] + bias, mlp2 [D, D] + bias.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> t_mlp0_w, t_mlp2_w;
  Kokkos::View<Scalar*, ExecSpace> t_mlp0_b, t_mlp2_b;
  // t_block SiLU + Linear(D -> 6D): [6D, D] + bias [6D].
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> t_block_w;
  Kokkos::View<Scalar*, ExecSpace> t_block_b;
  // Caption projection: fc [D, caption_channels] + bias, RMSNorm weight.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> cap_proj_w;
  Kokkos::View<Scalar*, ExecSpace> cap_proj_b;
  Kokkos::View<Scalar*, ExecSpace> cap_norm_w;
  std::vector<BlockWeights<ExecSpace, Scalar>> blocks;
  // Final layer: adaLN table [2, D], linear [out_ch*P*P, D] + bias
  // (one logit per unpatchified pixel-channel).
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> final_table;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> final_w;
  Kokkos::View<Scalar*, ExecSpace> final_b;

  void allocate(const Config& cfg);

  // Drop every device allocation (see BlockWeights::release). Clears the
  // per-block vector as well, so resident DiT bytes return to ~0 before the
  // next phase is loaded.
  void release() {
    x_proj_w = {};
    x_proj_b = {};
    t_mlp0_w = {};
    t_mlp2_w = {};
    t_mlp0_b = {};
    t_mlp2_b = {};
    t_block_w = {};
    t_block_b = {};
    cap_proj_w = {};
    cap_proj_b = {};
    cap_norm_w = {};
    for (auto& b : blocks) b.release();
    blocks.clear();
    blocks.shrink_to_fit();
    final_table = {};
    final_w = {};
    final_b = {};
  }

  // Resident device bytes currently held by these views (rank sizes times
  // sizeof(Scalar); 0 for released views). Host-side estimate only.
  size_t resident_bytes() const {
    size_t n = 0;
    auto acc2 = [&](const auto& v) {
      n += v.extent(0) * v.extent(1);
    };
    auto acc1 = [&](const auto& v) { n += v.extent(0); };
    acc2(x_proj_w);
    acc1(x_proj_b);
    acc2(t_mlp0_w);
    acc2(t_mlp2_w);
    acc1(t_mlp0_b);
    acc1(t_mlp2_b);
    acc2(t_block_w);
    acc1(t_block_b);
    acc2(cap_proj_w);
    acc1(cap_proj_b);
    acc1(cap_norm_w);
    for (const auto& b : blocks) {
      n += b.scale_shift_table.extent(0) * b.scale_shift_table.extent(1);
      n += b.attn_qkv_w.extent(0) * b.attn_qkv_w.extent(1);
      n += b.attn_cap_kv_w.extent(0) * b.attn_cap_kv_w.extent(1);
      n += b.attn_proj_w.extent(0) * b.attn_proj_w.extent(1);
      n += b.attn_proj_b.extent(0);
      n += b.mlp_fc1_w.extent(0) * b.mlp_fc1_w.extent(1);
      n += b.mlp_fc1_b.extent(0);
      n += b.mlp_fc2_w.extent(0) * b.mlp_fc2_w.extent(1);
      n += b.mlp_fc2_b.extent(0);
    }
    acc2(final_table);
    acc2(final_w);
    acc1(final_b);
    return n * sizeof(Scalar);
  }
};

template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  cfg.validate();
  const size_t D = (size_t)cfg.hidden_size;
  const size_t Hh = (size_t)cfg.mlp_hidden();
  const size_t P = (size_t)cfg.patch_size;
  x_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "moss_x_proj_w", D, (size_t)cfg.in_channels * P * P);
  x_proj_b = Kokkos::View<Scalar*, ExecSpace>("moss_x_proj_b", D);
  t_mlp0_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "moss_t_mlp0_w", D, (size_t)cfg.t_freq_dim);
  t_mlp0_b = Kokkos::View<Scalar*, ExecSpace>("moss_t_mlp0_b", D);
  t_mlp2_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "moss_t_mlp2_w", D, D);
  t_mlp2_b = Kokkos::View<Scalar*, ExecSpace>("moss_t_mlp2_b", D);
  t_block_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "moss_t_block_w", 6 * D, D);
  t_block_b = Kokkos::View<Scalar*, ExecSpace>("moss_t_block_b", 6 * D);
  cap_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "moss_cap_proj_w", D, (size_t)cfg.caption_channels);
  cap_proj_b = Kokkos::View<Scalar*, ExecSpace>("moss_cap_proj_b", D);
  cap_norm_w = Kokkos::View<Scalar*, ExecSpace>("moss_cap_norm_w", D);
  blocks.resize((size_t)cfg.depth);
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto& b = blocks[i];
    b.scale_shift_table =
        Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("moss_sst", 6,
                                                               D);
    b.attn_qkv_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "moss_qkv", 3 * D, D);
    b.attn_cap_kv_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "moss_cap_kv", 2 * D, D);
    b.attn_proj_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "moss_attn_proj", D, D);
    b.attn_proj_b = Kokkos::View<Scalar*, ExecSpace>("moss_attn_proj_b", D);
    b.mlp_fc1_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "moss_fc1", Hh, D);
    b.mlp_fc1_b = Kokkos::View<Scalar*, ExecSpace>("moss_fc1_b", Hh);
    b.mlp_fc2_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
        "moss_fc2", D, Hh);
    b.mlp_fc2_b = Kokkos::View<Scalar*, ExecSpace>("moss_fc2_b", D);
  }
  final_table = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "moss_final_table", 2, D);
  final_w = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(
      "moss_final_w", (size_t)cfg.out_channels * P * P, D);
  final_b = Kokkos::View<Scalar*, ExecSpace>(
      "moss_final_b", (size_t)cfg.out_channels * P * P);
}

// Fill with small random values (for shape/forward tests without weights).
// LayerNorm/RMSNorm weights are ones, biases zero, adaLN tables zero
// (adaLN-zero init: blocks start as identity).
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load a safetensors checkpoint into device views (no new dependencies:
// minimal header-JSON scanner + raw float32 LE copies, mirroring the Sana
// safetensors loaders). Bytes stream from the file through bounded HostSpace
// chunks with a single deep_copy per tensor (no 2x device residency);
// Scalar may be float or Kokkos::Experimental::bhalf_t (FP32 accumulation
// is unchanged). Expected (suffix-matched, prefix-agnostic) keys:
//
//   patch_embed.proj.{weight,bias}
//   time_embed.mlp.0.{weight,bias}, time_embed.mlp.2.{weight,bias},
//   time_embed.block.{weight,bias}
//   cap_embed.proj.{weight,bias}, cap_embed.norm.weight
//   blocks.{i}.scale_shift_table
//   blocks.{i}.attn.qkv.weight, blocks.{i}.attn.cap_kv.weight,
//   blocks.{i}.attn.proj.{weight,bias}
//   blocks.{i}.mlp.fc1.{weight,bias}, blocks.{i}.mlp.fc2.{weight,bias}
//   final.scale_shift_table, final.linear.{weight,bias}
//
// Throws on missing keys, non-float32 dtype, or shape/numel mismatch.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_safetensors(const Config& cfg, const std::string& st_path,
                      Weights<ExecSpace, Scalar>& w);

// Load a torch.save (.pth/.pt/.bin) checkpoint into device views (no
// libtorch: minimal ZIP central-directory reader + key scanner, mirroring
// the Sana .pth loader). Maps the same canonical key list as
// load_safetensors positionally: the Nth expected tensor maps to the Nth
// float32 storage (data/{N}), with the data.pkl key order validated as a
// suffix match. Throws with guidance on mismatch.
template <class ExecSpace, class Scalar = DefaultScalar>
void load_torch_pth(const Config& cfg, const std::string& pth_path,
                    Weights<ExecSpace, Scalar>& w);

// Load raw float32 caption features [L, Cc] (row-major) for --text-weights
// files that are already extracted features rather than checkpoints.
std::vector<float> load_caption_raw(const std::string& path, int expect_L,
                                    int expect_Cc);

// ---- staged-execution memory budget --------------------------------------
// Real-checkpoint resident budgets (bf16, documented in the README 24GB fit
// table). The encode phase holds the text encoder, the sample phase holds
// the DiT; the VAE decode (~0.25GB) joins the sample phase. Staging keeps
// the peak at max(16.8, 10.2) + 0.25 + activations instead of 27+ GB.
inline constexpr double kEncoderBudgetBytes = 16.8e9;  // Qwen text encoder
inline constexpr double kDitBudgetBytes = 10.2e9;      // MMDiT flow transformer
inline constexpr double kVaeBudgetBytes = 0.25e9;      // FLUX.2-small VAE decode
inline constexpr double kGpuBudgetBytes = 24.0e9;      // 24GB card

// Exact DiT weight bytes for a Config at a given scalar size (no allocation;
// mirrors Weights::allocate extents). Useful for pre-load logging.
inline size_t dit_weights_bytes(const Config& cfg, size_t scalar_bytes) {
  const size_t D = (size_t)cfg.hidden_size;
  const size_t Hh = (size_t)cfg.mlp_hidden();
  const size_t P = (size_t)cfg.patch_size;
  size_t n = 0;
  n += D * (size_t)cfg.in_channels * P * P + D;            // patch embed
  n += D * (size_t)cfg.t_freq_dim + D;                     // t mlp0
  n += D * D + D;                                          // t mlp2
  n += 6 * D * D + 6 * D;                                  // t block
  n += D * (size_t)cfg.caption_channels + D + D;           // cap proj+bias+norm
  const size_t per_block = 6 * D + 3 * D * D + 2 * D * D + D * D + D +
                           Hh * D + Hh + D * Hh + D;
  n += per_block * (size_t)cfg.depth;
  n += 2 * D + (size_t)cfg.out_channels * P * P * D +
       (size_t)cfg.out_channels * P * P;  // final
  return n * scalar_bytes;
}

// Steady-state DiT forward activation estimate (B=1): model scratch
// (tok/n1/att/qq, qkv, ckv, joint kj/vj, fc, lin/xin, caption yp/yn, t
// buffers) plus the Euler loop's x/vc/vu/t/y grids. One step's worth; the
// sampler reuses the same buffers every step.
inline size_t dit_activation_bytes(const Config& cfg, int h, int w,
                                   size_t scalar_bytes) {
  const size_t D = (size_t)cfg.hidden_size;
  const size_t Hh = (size_t)cfg.mlp_hidden();
  const size_t P = (size_t)cfg.patch_size;
  const size_t T = (size_t)(h / (int)P) * (size_t)(w / (int)P);
  const size_t L = (size_t)cfg.model_max_length;
  const size_t R = T, Ry = L, Rj = T + L;
  size_t n = 0;
  n += 4 * R * D;                 // tok, n1, att, qq
  n += R * 3 * D;                 // qkv
  n += Ry * 2 * D;                // ckv
  n += 2 * Rj * D;                // kj, vj
  n += R * Hh;                    // fc
  n += R * (size_t)cfg.out_channels * P * P;      // lin
  n += R * (size_t)cfg.in_channels * P * P;       // xin
  n += 2 * Ry * D;                // yp, yn
  n += (size_t)cfg.t_freq_dim + D + 2 * 6 * D;    // tf, te, t0, mod
  const size_t C = (size_t)cfg.in_channels;
  n += 3 * C * (size_t)h * (size_t)w;  // euler x, vc, vu
  n += 2 * L * (size_t)cfg.caption_channels;  // euler y cond/uncond (fp32)
  return n * scalar_bytes;
}

// One-line per-phase resident estimate used by the driver between staged
// phases (call after Kokkos::fence()).
inline void print_phase_memory(const char* phase, size_t weights_bytes,
                               size_t activation_bytes) {
  const double w = (double)weights_bytes / 1e9;
  const double a = (double)activation_bytes / 1e9;
  const double t = w + a;
  std::printf("[memory phase=%s] weights=%.3f GB + activations~%.3f GB = %.3f "
              "GB resident (24GB budget: %s)\n",
              phase, w, a, t, t < 24.0 ? "FITS" : "EXCEEDS");
  std::fflush(stdout);
}

}  // namespace Moss
}  // namespace KokkosDNN

#endif
