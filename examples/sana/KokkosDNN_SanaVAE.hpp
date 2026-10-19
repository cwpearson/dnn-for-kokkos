#ifndef KOKKOSDNN_SANA_VAE_HPP
#define KOKKOSDNN_SANA_VAE_HPP

// DC-AE f32c32 VAE decoder for the Sana example (native, FP32).
//
// Checkpoint (researched 2026-10, see README decoder section):
//   HF repo  mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers
//   file     diffusion_pytorch_model.safetensors (F32, ~1.2 GB)
//   config   in_channels=3, latent_channels=32,
//            decoder_block_out_channels=[128,256,512,512,1024,1024],
//            decoder_layers_per_block=[3,3,3,3,3,3],
//            decoder_block_types=[ResBlock x3, EfficientViTBlock x3],
//            decoder_qkv_multiscales=[(),(),(),(5,),(5,),(5,)],
//            upsample_block_type="interpolate", decoder_norm_types="rms_norm",
//            decoder_act_fns="silu", decoder_in_shortcut=True,
//            decoder_conv_act_fn="relu", scaling_factor=0.41407.
//   The sibling raw efficientvit repo (mit-han-lab/dc-ae-f32c32-sana-1.0,
//   model.safetensors) holds identical math under different key names
//   (decoder.project_in / decoder.stages.* / decoder.project_out); this
//   loader accepts the diffusers key layout only ("decoder.conv_in" /
//   "decoder.up_blocks.i.j.*" / "decoder.norm_out" / "decoder.conv_out").
//
// Decoder dataflow (diffusers AutoencoderDC::Decoder, PR #9708; Sana
// vae_decode divides latents by scaling_factor, then clamps x*0.5+0.5):
//   x = latent / 0.41407, [32,H,W]
//   h = conv3x3(32->1024) + repeat_interleave(x,32)      // in_shortcut
//   for stage i = 5..0 (1024ch @HxW, ..., 128ch @32Hx32W):
//     h = blocks[i] (3x ResBlock | 3x EfficientViTBlock, +residual each)
//     if i > 0: h = upsample(h)  // nearest2x -> conv3x3 -> +pixel_shuffle2x(
//                                //   repeat_interleave(h_in, 4*Cout/Cin))
//   y = conv3x3(128->3, +bias) after RMSNorm(1e-5, w+b) + ReLU.
//   5 upsamples => 32x spatial compression: [32,32,32] -> [3,1024,1024].
//
// Block details (all FP32 here):
// - ResBlock(C): conv3x3+bias -> SiLU -> conv3x3 no-bias ->
//     RMSNorm(eps 1e-5, affine weight+bias, over channels) -> +residual.
// - EfficientViTBlock(C) = SanaMultiscaleLinearAttention + GLUMBConv:
//     attn: q,k,v = Linear(C->C, no bias); U=[q;k;v] (3C); M = grouped1x1
//       (groups=3*heads, 32ch/group, no bias) after depthwise5x5 pad2
//       (no bias) on U; V=[U;M] (6C) viewed as G=2*heads groups of D=32
//       (q in [:,0:32], k in [:,32:64], v in [:,64:96]); ReLU on q,k;
//       linear attention when N=H*W > 32 else quadratic attention
//       (eps 1e-15); concat -> [N,2C]; Linear(2C->C, no bias) ->
//       RMSNorm(w+b) -> +residual.
//     glumb: Linear(C->8C,+bias) -> SiLU -> depthwise3x3(+bias) ->
//       split a*SiLU(g) (4C) -> Linear(4C->C, no bias) -> RMSNorm(w+b) ->
//       +residual. (expand_ratio 4.)
//
// Activations are [T,C] token rows (T=H*W row-major, t=h*W+w); 1x1 convs
// and linears reuse KokkosDNN::gemm_bias (cuBLAS ladder intact); spatial
// compute (dense 3x3, depthwise, grouped 1x1) uses the promoted core ops
// in KokkosDNN_Conv.hpp (cuDNN-dispatched on CUDA builds, native
// otherwise; see the TPL-status note there). Remaining kernels below are
// example-local. Backend-portable parallel_for only.

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_Conv.hpp"
#include "KokkosDNN_Handle.hpp"

namespace KokkosDNN {
namespace Sana {
namespace VAE {

using Mat = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
// Device-side aliases are spelled out per template (ExecSpace-dependent).

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Config {
  int latent_channels = 32;
  int out_channels = 3;  // RGB
  std::vector<int> stage_widths = {128, 256, 512, 512, 1024, 1024};
  std::vector<int> stage_depths = {3, 3, 3, 3, 3, 3};
  // Per-stage block kind: "ResBlock" or "EfficientViTBlock".
  std::vector<std::string> stage_blocks = {
      "ResBlock", "ResBlock", "ResBlock",
      "EfficientViTBlock", "EfficientViTBlock", "EfficientViTBlock"};
  int attention_head_dim = 32;
  int multiscale_kernel = 5;  // qkv depthwise kernel (pad = k/2)
  float scaling_factor = 0.41407f;  // latents are divided by this
  float norm_eps = 1e-5f;           // RMSNorm epsilon
  float attn_eps = 1e-15f;          // linear/quadratic attention epsilon
  int glumb_expand = 4;             // GLUMBConv expand ratio

  static Config dcae_f32c32_sana() { return Config{}; }

  // Miniature 2-stage config for host-side smoke tests (no checkpoint):
  // latent 4ch 2x2 -> 4x4 RGB, exercises conv_in shortcut, one upsample,
  // one ResBlock stage, one EfficientViTBlock stage (quadratic path: N=4).
  static Config tiny_test() {
    Config c;
    c.latent_channels = 4;
    c.out_channels = 3;
    c.stage_widths = {16, 32};
    c.stage_depths = {1, 1};
    c.stage_blocks = {"ResBlock", "EfficientViTBlock"};
    return c;
  }

  int num_stages() const { return (int)stage_widths.size(); }
  // Spatial compression: one 2x upsample per stage boundary.
  int compression_ratio() const {
    int r = 1;
    for (int i = 1; i < num_stages(); ++i) r *= 2;
    return r;
  }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("Sana::VAE::Config::validate: " + m);
    };
    const int S = num_stages();
    if (S < 1) bad("need at least one stage");
    if ((int)stage_depths.size() != S) bad("stage_depths size mismatch");
    if ((int)stage_blocks.size() != S) bad("stage_blocks size mismatch");
    if (latent_channels <= 0) bad("latent_channels must be > 0");
    if (out_channels <= 0) bad("out_channels must be > 0");
    if (attention_head_dim != 32)
      bad("attention_head_dim must be 32 (proj_out groups assume 32ch)");
    if (multiscale_kernel != 5) bad("multiscale_kernel must be 5");
    if (!(scaling_factor > 0) || !std::isfinite(scaling_factor))
      bad("scaling_factor must be positive and finite");
    if (!(norm_eps > 0) || !std::isfinite(norm_eps))
      bad("norm_eps must be positive and finite");
    if (!(attn_eps > 0) || !std::isfinite(attn_eps))
      bad("attn_eps must be positive and finite");
    if (glumb_expand != 4) bad("glumb_expand must be 4");
    for (int i = 0; i < S; ++i) {
      if (stage_widths[i] <= 0) bad("stage_widths must be > 0");
      if (stage_depths[i] < 0) bad("stage_depths must be >= 0");
      const std::string& b = stage_blocks[i];
      if (b != "ResBlock" && b != "EfficientViTBlock")
        bad("stage_blocks[" + std::to_string(i) + "] must be ResBlock or "
            "EfficientViTBlock, got '" + b + "'");
      if (b == "EfficientViTBlock" && stage_widths[i] % attention_head_dim != 0)
        bad("EfficientViTBlock width must be a multiple of 32");
    }
    if (stage_widths[S - 1] % latent_channels != 0)
      bad("last stage width must be divisible by latent_channels "
          "(conv_in in_shortcut repeat)");
    for (int i = 1; i < S; ++i) {
      // Up shortcut: repeat_interleave(x, r) then pixel_shuffle(2) needs
      // r = 4*Cout/Cin integral.
      if ((4 * stage_widths[i - 1]) % stage_widths[i] != 0)
        bad("4*widths[" + std::to_string(i - 1) + "] must be divisible by "
            "widths[" + std::to_string(i) + "] (upsample shortcut repeat)");
    }
  }
};

// ---------------------------------------------------------------------------
// Expected-tensor inventory (diffusers key layout). Used by the loader and
// by the header-only golden test.
// ---------------------------------------------------------------------------
struct TensorSpec {
  std::string name;
  std::vector<size_t> shape;  // on-disk row-major shape
};

inline std::vector<TensorSpec> expected_decoder_tensors(const Config& cfg) {
  cfg.validate();
  const int S = cfg.num_stages();
  std::vector<TensorSpec> out;
  const int CL = cfg.latent_channels;
  const int Wlast = cfg.stage_widths[S - 1];
  out.push_back({"decoder.conv_in.weight",
                 {(size_t)Wlast, (size_t)CL, 3, 3}});
  out.push_back({"decoder.conv_in.bias", {(size_t)Wlast}});
  for (int i = 0; i < S; ++i) {
    const int C = cfg.stage_widths[i];
    const std::string p =
        "decoder.up_blocks." + std::to_string(i) + ".";
    int j = 0;
    if (i < S - 1) {
      const int Cin = cfg.stage_widths[i + 1];
      out.push_back({p + "0.conv.weight", {(size_t)C, (size_t)Cin, 3, 3}});
      out.push_back({p + "0.conv.bias", {(size_t)C}});
      j = 1;
    }
    for (int d = 0; d < cfg.stage_depths[i]; ++d, ++j) {
      const std::string q = p + std::to_string(j) + ".";
      if (cfg.stage_blocks[i] == "ResBlock") {
        out.push_back({q + "conv1.weight", {(size_t)C, (size_t)C, 3, 3}});
        out.push_back({q + "conv1.bias", {(size_t)C}});
        out.push_back({q + "conv2.weight", {(size_t)C, (size_t)C, 3, 3}});
        out.push_back({q + "norm.weight", {(size_t)C}});
        out.push_back({q + "norm.bias", {(size_t)C}});
      } else {
        const size_t H = 4 * (size_t)C;  // GLU hidden = expand(4)*C
        out.push_back({q + "attn.to_q.weight", {(size_t)C, (size_t)C}});
        out.push_back({q + "attn.to_k.weight", {(size_t)C, (size_t)C}});
        out.push_back({q + "attn.to_v.weight", {(size_t)C, (size_t)C}});
        out.push_back({q + "attn.to_qkv_multiscale.0.proj_in.weight",
                       {3 * (size_t)C, 1, 5, 5}});
        out.push_back({q + "attn.to_qkv_multiscale.0.proj_out.weight",
                       {3 * (size_t)C, 32, 1, 1}});
        out.push_back({q + "attn.to_out.weight", {(size_t)C, 2 * (size_t)C}});
        out.push_back({q + "attn.norm_out.weight", {(size_t)C}});
        out.push_back({q + "attn.norm_out.bias", {(size_t)C}});
        out.push_back({q + "conv_out.conv_inverted.weight",
                       {2 * H, (size_t)C, 1, 1}});
        out.push_back({q + "conv_out.conv_inverted.bias", {2 * H}});
        out.push_back({q + "conv_out.conv_depth.weight", {2 * H, 1, 3, 3}});
        out.push_back({q + "conv_out.conv_depth.bias", {2 * H}});
        out.push_back({q + "conv_out.conv_point.weight",
                       {(size_t)C, H, 1, 1}});
        out.push_back({q + "conv_out.norm.weight", {(size_t)C}});
        out.push_back({q + "conv_out.norm.bias", {(size_t)C}});
      }
    }
  }
  const int W0 = cfg.stage_widths[0];
  out.push_back({"decoder.norm_out.weight", {(size_t)W0}});
  out.push_back({"decoder.norm_out.bias", {(size_t)W0}});
  out.push_back({"decoder.conv_out.weight",
                 {(size_t)cfg.out_channels, (size_t)W0, 3, 3}});
  out.push_back({"decoder.conv_out.bias", {(size_t)cfg.out_channels}});
  return out;
}

// ---------------------------------------------------------------------------
// Weights (device views; float = checkpoint dtype)
// ---------------------------------------------------------------------------
template <class ExecSpace>
struct Weights {
  using Mat2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using Mat3 = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;
  using Mat4 = Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace>;
  using Vec = Kokkos::View<float*, ExecSpace>;

  struct ResW {
    Mat4 conv1_w;  // [C,C,3,3]
    Vec conv1_b;   // [C]
    Mat4 conv2_w;  // [C,C,3,3], no bias
    Vec norm_w, norm_b;
  };
  struct EViTW {
    Mat2 to_q, to_k, to_v;  // [C,C], no bias
    Mat3 ms_in;             // [3C,5,5] depthwise, no bias
    Mat2 ms_out;            // [3C,32] grouped 1x1 (groups=3*heads), no bias
    Mat2 attn_out;          // [C,2C], no bias
    Vec attn_nw, attn_nb;
    Mat2 gl_inv;  // [8C,C] (checkpoint [8C,C,1,1] flattened)
    Vec gl_inv_b;
    Mat3 gl_dw;  // [8C,3,3] depthwise (+bias)
    Vec gl_dw_b;
    Mat2 gl_pt;  // [C,4C] (checkpoint [C,4C,1,1] flattened)
    Vec gl_nw, gl_nb;
  };
  struct Stage {
    Mat4 up_w;  // [Cout,Cin,3,3]; empty when !has_up
    Vec up_b;
    bool has_up = false;
    std::vector<ResW> res;
    std::vector<EViTW> evit;
  };

  Mat4 conv_in_w;
  Vec conv_in_b;
  std::vector<Stage> stages;
  Vec norm_out_w, norm_out_b;
  Mat4 conv_out_w;
  Vec conv_out_b;

  void allocate(const Config& cfg) {
    cfg.validate();
    const int S = cfg.num_stages();
    conv_in_w = Mat4("vae_conv_in_w", cfg.stage_widths[S - 1],
                     cfg.latent_channels, 3, 3);
    conv_in_b = Vec("vae_conv_in_b", cfg.stage_widths[S - 1]);
    stages.resize((size_t)S);
    for (int i = 0; i < S; ++i) {
      const int C = cfg.stage_widths[i];
      const size_t H = 4 * (size_t)C;
      Stage& st = stages[(size_t)i];
      st.has_up = (i < S - 1);
      if (st.has_up) {
        const int Cin = cfg.stage_widths[i + 1];
        st.up_w = Mat4("vae_up_w", C, Cin, 3, 3);
        st.up_b = Vec("vae_up_b", C);
      }
      if (cfg.stage_blocks[i] == "ResBlock") {
        st.res.resize((size_t)cfg.stage_depths[i]);
        for (auto& r : st.res) {
          r.conv1_w = Mat4("vae_res_c1w", C, C, 3, 3);
          r.conv1_b = Vec("vae_res_c1b", C);
          r.conv2_w = Mat4("vae_res_c2w", C, C, 3, 3);
          r.norm_w = Vec("vae_res_nw", C);
          r.norm_b = Vec("vae_res_nb", C);
        }
      } else {
        st.evit.resize((size_t)cfg.stage_depths[i]);
        for (auto& e : st.evit) {
          e.to_q = Mat2("vae_evit_q", C, C);
          e.to_k = Mat2("vae_evit_k", C, C);
          e.to_v = Mat2("vae_evit_v", C, C);
          e.ms_in = Mat3("vae_evit_msin", 3 * (size_t)C, 5, 5);
          e.ms_out = Mat2("vae_evit_msout", 3 * (size_t)C, 32);
          e.attn_out = Mat2("vae_evit_o", C, 2 * (size_t)C);
          e.attn_nw = Vec("vae_evit_nw", C);
          e.attn_nb = Vec("vae_evit_nb", C);
          e.gl_inv = Mat2("vae_evit_gi", 2 * H, C);
          e.gl_inv_b = Vec("vae_evit_gib", 2 * H);
          e.gl_dw = Mat3("vae_evit_gdw", 2 * H, 3, 3);
          e.gl_dw_b = Vec("vae_evit_gdwb", 2 * H);
          e.gl_pt = Mat2("vae_evit_gp", C, H);
          e.gl_nw = Vec("vae_evit_gnw", C);
          e.gl_nb = Vec("vae_evit_gnb", C);
        }
      }
    }
    norm_out_w = Vec("vae_norm_out_w", cfg.stage_widths[0]);
    norm_out_b = Vec("vae_norm_out_b", cfg.stage_widths[0]);
    conv_out_w =
        Mat4("vae_conv_out_w", cfg.out_channels, cfg.stage_widths[0], 3, 3);
    conv_out_b = Vec("vae_conv_out_b", cfg.out_channels);
  }
};

// Fill with small random values (smoke tests only, no checkpoint needed):
// mats ~ N(0, 0.02), biases ~ N(0, 0.02), norm weights = 1.
template <class ExecSpace>
void random_init(const Config& cfg, Weights<ExecSpace>& w, uint64_t seed);

// Strict F32 loader for the diffusers-layout safetensors file. Throws on
// missing file, bad magic, non-F32 dtype, or any shape/content mismatch.
template <class ExecSpace>
void load_safetensors(const Config& cfg, const std::string& path,
                      Weights<ExecSpace>& w);

namespace Impl {

// ---- elementwise helpers (tokens [T,C]) ----
template <class ExecSpace>
void silu_inplace(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& v,
    const std::string& label) {
  const size_t R = v.extent(0), C = v.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        const float x = v(i, j);
        v(i, j) = x / (1.0f + expf(-x));
      });
}

template <class ExecSpace>
void relu_inplace(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& v,
    const std::string& label) {
  const size_t R = v.extent(0), C = v.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) {
        v(i, j) = v(i, j) > 0.0f ? v(i, j) : 0.0f;
      });
}

// y(t,c) += x(t,c) (residual add; x and y may alias only if same view).
template <class ExecSpace>
void add_inplace(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const std::string& label) {
  const size_t R = y.extent(0), C = y.extent(1);
  if (R == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
      KOKKOS_LAMBDA(size_t i, size_t j) { y(i, j) += x(i, j); });
}

// RMSNorm with affine weight AND bias (diffusers RMSNorm, eps 1e-5):
//   y(t,c) = x(t,c)/sqrt(mean_c(x^2)+eps) * w(c) + b(c)
template <class ExecSpace>
void rmsnorm_wb(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float*, ExecSpace>& w,
    const Kokkos::View<float*, ExecSpace>& b, float eps,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label) {
  const size_t T = x.extent(0), C = x.extent(1);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T),
      KOKKOS_LAMBDA(size_t t) {
        float ss = 0.0f;
        for (size_t c = 0; c < C; ++c) ss += x(t, c) * x(t, c);
        const float inv = 1.0f / sqrtf(ss / (float)C + eps);
        for (size_t c = 0; c < C; ++c)
          y(t, c) = x(t, c) * inv * w(c) + b(c);
      });
}

// ---- nearest-neighbor 2x upsample (tokens) ----
template <class ExecSpace>
void nearest_upsample2x(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, const std::string& label) {
  const size_t C = x.extent(1), H2 = 2 * H, W2 = 2 * W;
  if (H == 0 || W == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {H2, W2}),
      KOKKOS_LAMBDA(size_t h2, size_t w2) {
        const size_t s = (h2 / 2) * W + (w2 / 2), d = h2 * W2 + w2;
        for (size_t c = 0; c < C; ++c) y(d, c) = x(s, c);
      });
}

// ---- repeat_interleave(channels) + pixel_shuffle(2) shortcut ----
// x [H*W,Cin] -> y [4H*W,Cout], repeats r = 4*Cout/Cin:
//   tmp = repeat_interleave(x, r) [H,W,r*Cin] (channel tiles),
//   y = pixel_shuffle(tmp, 2) [2H,2W,Cout].
template <class ExecSpace>
void repeat_pixel_shuffle2x(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y, size_t H,
    size_t W, size_t Cout, const std::string& label) {
  const size_t Cin = x.extent(1);
  const size_t r = 4 * Cout / Cin;
  const size_t H2 = 2 * H, W2 = 2 * W;
  if (H == 0 || W == 0 || Cin == 0 || Cout == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                              {H2, W2, Cout}),
      KOKKOS_LAMBDA(size_t h2, size_t w2, size_t c) {
        // pixel_shuffle inverse: tmp channel = c*4 + (h2%2)*2 + (w2%2)
        const size_t tc = c * 4 + (h2 % 2) * 2 + (w2 % 2);
        // repeat_interleave inverse: src channel = tc / r
        y(h2 * W2 + w2, c) = x((h2 / 2) * W + (w2 / 2), tc / r);
      });
}

// ---- conv_in shortcut add: y [T,Cout] += repeat_interleave(x, r) ----
template <class ExecSpace>
void add_repeat_inplace(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& y,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const std::string& label) {
  const size_t T = y.extent(0), Cout = y.extent(1), Cin = x.extent(1);
  const size_t r = Cout / Cin;
  if (T == 0 || Cin == 0 || Cout == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Cout}),
      KOKKOS_LAMBDA(size_t t, size_t o) { y(t, o) += x(t, o / r); });
}

// GLU gate: h(t,j) = a(t,j) * SiLU(g(t,j)); a/g are halves of u [T,2Hid].
template <class ExecSpace>
void silu_gate(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& u,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& h, size_t Hid,
    const std::string& label) {
  const size_t T = h.extent(0);
  if (T == 0 || Hid == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, Hid}),
      KOKKOS_LAMBDA(size_t t, size_t j) {
        const float a = u(t, j), g = u(t, j + Hid);
        h(t, j) = a * (g / (1.0f + expf(-g)));
      });
}

// ---- Sana multiscale linear/quadratic attention core ----
// feats [N,6C]: channels [0,3C) direct qkv, [3C,6C) multiscale qkv.
// Viewed as G=2*heads groups of D=32 with q=chn[0:32], k=[32:64],
// v=[64:96] of each 96-wide group (diffusers reshape+chunk). ReLU on q,k.
// out [N,2C]: out[n][g*32+d].
template <class ExecSpace>
void multiscale_attention(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& feats,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& out,
    size_t num_heads, float eps, const std::string& label) {
  const size_t N = feats.extent(0);
  const size_t G = 2 * num_heads;
  constexpr size_t D = 32;
  if (N == 0 || G == 0) return;
  const bool linear = (N > D);
  if (linear) {
    // S[g] = Vp @ K^T [33,32] (Vp = [v;1]), O[g] = S @ Q [33,N].
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> S("vae_attn_S", G,
                                                             D + 1, D);
    Kokkos::parallel_for(
        label + "_gram", Kokkos::RangePolicy<ExecSpace>(0, G),
        KOKKOS_LAMBDA(size_t g) {
          float s00[D + 1][D];
          for (size_t a = 0; a <= D; ++a)
            for (size_t d = 0; d < D; ++d) s00[a][d] = 0.0f;
          for (size_t n = 0; n < N; ++n) {
            float k[D], v[D];
            for (size_t d = 0; d < D; ++d) {
              const size_t base = g * 96;
              float kk = feats(n, base + 32 + d);
              kk = kk > 0.0f ? kk : 0.0f;
              k[d] = kk;
              v[d] = feats(n, base + 64 + d);
            }
            for (size_t a = 0; a < D; ++a)
              for (size_t d = 0; d < D; ++d) s00[a][d] += v[a] * k[d];
            for (size_t d = 0; d < D; ++d) s00[D][d] += k[d];  // ones row
          }
          for (size_t a = 0; a <= D; ++a)
            for (size_t d = 0; d < D; ++d) S(g, a, d) = s00[a][d];
        });
    Kokkos::parallel_for(
        label + "_apply",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {G, N}),
        KOKKOS_LAMBDA(size_t g, size_t n) {
          float q[D];
          for (size_t d = 0; d < D; ++d) {
            float qq = feats(n, g * 96 + d);
            q[d] = qq > 0.0f ? qq : 0.0f;
          }
          float den = eps;
          for (size_t d = 0; d < D; ++d) den += S(g, D, d) * q[d];
          for (size_t a = 0; a < D; ++a) {
            float acc = 0.0f;
            for (size_t d = 0; d < D; ++d) acc += S(g, a, d) * q[d];
            out(n, g * D + a) = acc / den;
          }
        });
  } else {
    // Quadratic: S[g] = K^T Q [N,N], col-normalize, O[g] = V S.
    Kokkos::parallel_for(
        label + "_quad",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {G, N}),
        KOKKOS_LAMBDA(size_t g, size_t i) {
          float srow[64];  // N <= 32; 64 is a safe static bound
          float vsum[32];
          for (size_t d = 0; d < D; ++d) vsum[d] = 0.0f;
          // scores(i,j) = <k_j, q_i>, normalized over j
          float denom = eps;
          for (size_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) {
              float qq = feats(i, g * 96 + d);
              float kk = feats(j, g * 96 + 32 + d);
              qq = qq > 0.0f ? qq : 0.0f;
              kk = kk > 0.0f ? kk : 0.0f;
              s += kk * qq;
            }
            srow[j] = s;
            denom += s;
          }
          for (size_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (size_t j = 0; j < N; ++j)
              acc += feats(j, g * 96 + 64 + d) * (srow[j] / denom);
            vsum[d] = acc;
          }
          for (size_t d = 0; d < D; ++d) out(i, g * D + d) = vsum[d];
        });
  }
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// Decoder: latents [C,H,W] host row-major -> RGB [3,oH,oW] host row-major.
// ---------------------------------------------------------------------------
template <class ExecSpace>
class Decoder {
 public:
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using View1 = Kokkos::View<float*, ExecSpace>;

  Decoder(const Config& cfg, const Weights<ExecSpace>& w) : cfg_(cfg), w_(w) {
    cfg_.validate();
  }

  // oH = H*ratio, oW = W*ratio. Output range ~[-1,1] (diffusers sample).
  std::vector<float> decode(const std::vector<float>& lat, size_t H,
                            size_t W) const;

  const Config& config() const { return cfg_; }

 private:
  Config cfg_;
  Weights<ExecSpace> w_;
  DNNHandle<ExecSpace> handle_;

  View2 res_block(const typename Weights<ExecSpace>::ResW& r, const View2& x,
                  size_t C, size_t H, size_t W, const std::string& tag) const;
  View2 evit_block(const typename Weights<ExecSpace>::EViTW& e, const View2& x,
                   size_t C, size_t H, size_t W,
                   const std::string& tag) const;
};

// u8 RGB interleaved [H*W*3] from decoder float output [3,H,W].
inline std::vector<unsigned char> to_u8_rgb(const std::vector<float>& rgb) {
  std::vector<unsigned char> out(rgb.size());
  for (size_t i = 0; i < rgb.size(); ++i) {
    const float v = rgb[i] * 0.5f + 0.5f;
    const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    out[i] = (unsigned char)std::lround(c * 255.0f);
  }
  return out;
}

}  // namespace VAE
}  // namespace Sana
}  // namespace KokkosDNN

#endif
