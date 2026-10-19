#ifndef KOKKOSDNN_MOSSIMAGEVAE_HPP
#define KOKKOSDNN_MOSSIMAGEVAE_HPP

// FLUX.2-small VAE decoder for the MOSS-Image-SFT full pipeline
// (OpenMOSS-Team/MOSS-Image-SFT, vae/diffusion_pytorch_model.safetensors,
// single ~250 MB FP32 file; weights NOT vendored).
//
// Ground truth (pinned, copies in reference/):
//   reference/vae/config.json ......................... AutoencoderKLFlux2
//     in/out 3, latent 32, encoder block_out [128,256,512,512], decoder
//     [96,192,384,384], 2 layers/block, GroupNorm 32 groups, SiLU,
//     mid-block attention, quant + post-quant conv, patch [2,2], sample
//     1024, force_upcast true, batch_norm_eps 1e-4.
//   reference/modeling_autoencoderkl_flux2.py ......... diffusers 0.38.0
//     AutoencoderKLFlux2: BN(4*32 = 128ch, affine=False) running stats,
//     post_quant_conv (Conv2d 32->32, 1x1), _decode = post_quant + Decoder.
//   reference/vae_ref.py (Decoder/UpSample/UNetMidBlock2D), resnet.py
//     (ResnetBlock2D), unet_2d_blocks.py (UpDecoderBlock2D),
//     attention_processor.py (Attention + AttnProcessor), upsampling.py
//     (Upsample2D).
//
// Decode path (pipeline __call__, MOSS_ARCH_INVESTIGATION.md section 5):
//   latents [B,128,H,W] (DiT latent_channels = 128, FP32)
//   -> BN-stat denormalize: z*sqrt(running_var + 1e-4) + running_mean
//      (moss_vae_denormalize_forward, already in KokkosDNN_MossImage.hpp)
//   -> 2x2 depth-to-space de-interleave -> [B,32,2H,2W]
//      (moss_vae_depth_to_space_2x, already in KokkosDNN_MossImage.hpp)
//   -> post_quant_conv 1x1 (32->32, +bias)
//   -> Decoder: conv_in 3x3 (32->384, +bias); mid block (ResNet, 1-head
//      self-attention over 384ch, ResNet); 4 UpDecoderBlocks with 3
//      ResNets each (384->384, 384->384, 384->192, 192->96) and
//      nearest-2x + conv3x3 upsamplers after the first three; head
//      GroupNorm(32) + SiLU + conv_out 3x3 (96->3, +bias)
//   -> RGB [B,3,16H,16W] (2 * 2^3 = 16 = VAE scale factor 16, so
//      1024px <-> 64x64x128 latents). FP32 throughout (force_upcast).
//
// Block details (all biases present unless noted; dropout is 0 at
// inference; temb is None so time_emb_proj is absent):
//   ResNet (ResnetBlock2D, eps 1e-6): GN(32) -> SiLU -> conv3x3+bias ->
//     GN(32) -> SiLU -> conv3x3 (no bias) -> +shortcut (1x1 conv +bias
//     iff in != out) -> /1.0.
//   Mid attention (Attention, heads=1, dim_head=384, eps 1e-6, bias=True,
//     out_bias=True, residual, scale 1/sqrt(384)): GN(32, affine) ->
//     q/k/v Linear -> dense softmax(QK^T/sqrt(d))V (FP32) -> out Linear
//     (+bias) -> +residual -> /1.0.
//   Upsampler (Upsample2D use_conv, bias=True): nearest-2x -> conv3x3+bias.
//
// Dispatch contract (no hand-written cublas/cudnn calls anywhere here):
//   * 1x1 convs, conv shortcuts, attention qkv/out projections go through
//     gemm_bias, which owns the cuBLAS ladder.
//   * Mid-block attention goes through attention_mha_bidirectional, which
//     owns the cuDNN ladder (float path: fused SDPA graph, TPL math
//     fallback past the graph's head-dim limit, native streaming kernel
//     otherwise / on failure).
//   * conv3x3_forward owns the im2col + gemm_bias lowering (cuBLAS
//     ladder on CUDA, MIOpen path on HIP builds; see KokkosDNN_Conv.hpp
//     TPL note).
//   * GroupNorm goes through groupnorm_forward, SiLU/upsample/residual/
//     denormalize/depth-to-space are native elementwise (no TPL expresses
//     them, so native == dispatched by construction — same rationale as
//     the MossImage elementwise ops).
// All Impl helpers are backend-portable parallel_for only. No backend
// #ifdefs in native paths.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "KokkosDNN_Attention.hpp"
#include "KokkosDNN_Conv.hpp"
#include "KokkosDNN_GemmBias.hpp"
#include "KokkosDNN_GroupNorm.hpp"
#include "KokkosDNN_Handle.hpp"
#include "KokkosDNN_MossImage.hpp"

namespace KokkosDNN {
namespace MossImage {
namespace Vae {

using Mat2H = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Config {
  int dit_latent_channels = 128;  // pipeline/DiT latent width (BN width)
  int latent_channels = 32;       // decoder input width (after d2s)
  int out_channels = 3;           // RGB
  std::vector<int> decoder_widths = {96, 192, 384, 384};
  int layers_per_block = 2;  // ResNets per up block = layers_per_block + 1
  int num_groups = 32;       // GroupNorm groups
  float bn_eps = 1e-4f;       // batch_norm_eps (denormalize)
  float gn_eps = 1e-6f;       // ResNet/attention/head GroupNorm eps
  int scale_factor = 16;      // VAE scale factor (2 * 2^3)

  static Config moss_image_sft() { return Config{}; }

  // Miniature config for host-side smoke tests (no checkpoint):
  // 4x4x8 latents -> d2s 8x8x2 -> mid @8ch -> up 8ch -> up 4ch -> RGB
  // [B,3,16,16]. Exercises denorm, d2s, post-quant, conv_in, mid ResNets
  // + attention, a same-width and a shrinking up block (1x1 shortcut),
  // upsamplers, head norm/conv. Mid width 8 = 1 head of dim 8.
  static Config tiny_test() {
    Config c;
    c.dit_latent_channels = 8;
    c.latent_channels = 2;
    c.decoder_widths = {4, 8};
    c.layers_per_block = 1;
    c.num_groups = 2;
    c.scale_factor = 4;  // 2 (d2s) * 2^1 (one upsample)
    return c;
  }

  int num_up_blocks() const { return (int)decoder_widths.size(); }
  int mid_width() const { return decoder_widths.back(); }
  int resnets_per_up_block() const { return layers_per_block + 1; }

  void validate() const {
    auto bad = [](const std::string& m) {
      throw std::invalid_argument("MossImage::Vae::Config::validate: " + m);
    };
    if (num_up_blocks() < 1) bad("need at least one up block");
    if (layers_per_block < 0) bad("layers_per_block must be >= 0");
    if (dit_latent_channels <= 0) bad("dit_latent_channels must be > 0");
    if (latent_channels <= 0) bad("latent_channels must be > 0");
    if (out_channels <= 0) bad("out_channels must be > 0");
    if (dit_latent_channels != 4 * latent_channels)
      bad("dit_latent_channels must equal 4 * latent_channels (2x2 d2s)");
    if (num_groups <= 0) bad("num_groups must be > 0");
    for (int w : decoder_widths) {
      if (w <= 0) bad("decoder widths must be > 0");
      if (w % num_groups != 0)
        bad("decoder widths must be divisible by num_groups");
    }
    if (!(bn_eps >= 0.0f) || !std::isfinite(bn_eps)) bad("bad bn_eps");
    if (!(gn_eps >= 0.0f) || !std::isfinite(gn_eps)) bad("bad gn_eps");
    if (scale_factor != (2 << (num_up_blocks() - 1)))
      bad("scale_factor must equal 2 * 2^(num_up_blocks - 1)");
  }
};

// ---------------------------------------------------------------------------
// Expected-tensor inventory (diffusers key layout, F32). Used by the loader
// (fail-fast presence check) and by tests.
// ---------------------------------------------------------------------------
struct TensorSpec {
  std::string name;
  std::vector<size_t> shape;  // on-disk row-major shape
};

inline std::vector<TensorSpec> expected_decoder_tensors(const Config& cfg) {
  cfg.validate();
  std::vector<TensorSpec> out;
  const size_t DL = (size_t)cfg.dit_latent_channels;
  const size_t LC = (size_t)cfg.latent_channels;
  const size_t MW = (size_t)cfg.mid_width();
  out.push_back({"bn.running_mean", {DL}});
  out.push_back({"bn.running_var", {DL}});
  out.push_back({"post_quant_conv.weight", {LC, LC, 1, 1}});
  out.push_back({"post_quant_conv.bias", {LC}});
  out.push_back({"decoder.conv_in.weight", {MW, LC, 3, 3}});
  out.push_back({"decoder.conv_in.bias", {MW}});
  auto res_keys = [&](const std::string& p, size_t cin, size_t cout,
                      bool shortcut) {
    out.push_back({p + "norm1.weight", {cin}});
    out.push_back({p + "norm1.bias", {cin}});
    out.push_back({p + "conv1.weight", {cout, cin, 3, 3}});
    out.push_back({p + "conv1.bias", {cout}});
    out.push_back({p + "norm2.weight", {cout}});
    out.push_back({p + "norm2.bias", {cout}});
    out.push_back({p + "conv2.weight", {cout, cout, 3, 3}});
    out.push_back({p + "conv2.bias", {cout}});
    if (shortcut) {
      out.push_back({p + "conv_shortcut.weight", {cout, cin, 1, 1}});
      out.push_back({p + "conv_shortcut.bias", {cout}});
    }
  };
  res_keys("decoder.mid_block.resnets.0.", MW, MW, false);
  out.push_back({"decoder.mid_block.attentions.0.group_norm.weight", {MW}});
  out.push_back({"decoder.mid_block.attentions.0.group_norm.bias", {MW}});
  for (const char* n : {"to_q", "to_k", "to_v"}) {
    out.push_back(
        {"decoder.mid_block.attentions.0." + std::string(n) + ".weight",
         {MW, MW}});
    out.push_back(
        {"decoder.mid_block.attentions.0." + std::string(n) + ".bias", {MW}});
  }
  out.push_back({"decoder.mid_block.attentions.0.to_out.0.weight", {MW, MW}});
  out.push_back({"decoder.mid_block.attentions.0.to_out.0.bias", {MW}});
  res_keys("decoder.mid_block.resnets.1.", MW, MW, false);
  const int NB = cfg.num_up_blocks();
  for (int i = 0; i < NB; ++i) {
    // UpDecoderBlock2D(i): first resnet in = prev width, out = widths[i];
    // remaining resnets are widths[i] -> widths[i]. Reversed layout:
    // widths index of block i is NB-1-i.
    const size_t cout = (size_t)cfg.decoder_widths[NB - 1 - i];
    const size_t cin_first =
        (i == 0) ? MW : (size_t)cfg.decoder_widths[NB - i];
    for (int j = 0; j < cfg.resnets_per_up_block(); ++j) {
      const std::string p = "decoder.up_blocks." + std::to_string(i) +
                            ".resnets." + std::to_string(j) + ".";
      const size_t cin = (j == 0) ? cin_first : cout;
      res_keys(p, cin, cout, cin != cout);
    }
    if (i < NB - 1) {
      const std::string p =
          "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.conv.";
      out.push_back({p + "weight", {cout, cout, 3, 3}});
      out.push_back({p + "bias", {cout}});
    }
  }
  const size_t W0 = (size_t)cfg.decoder_widths[0];
  out.push_back({"decoder.conv_norm_out.weight", {W0}});
  out.push_back({"decoder.conv_norm_out.bias", {W0}});
  out.push_back({"decoder.conv_out.weight",
                 {(size_t)cfg.out_channels, W0, 3, 3}});
  out.push_back({"decoder.conv_out.bias", {(size_t)cfg.out_channels}});
  return out;
}

// ---------------------------------------------------------------------------
// Weights (device views, float = checkpoint dtype). 1x1 conv / linear
// weights are stored [out, in] (HF Linear layout, gemm_bias-ready), so a
// checkpoint [o,i,1,1] tensor copies in flat with no transpose.
// ---------------------------------------------------------------------------
template <class ExecSpace>
struct Weights {
  using Vec = Kokkos::View<float*, ExecSpace>;
  using Mat2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using Mat4 = Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace>;

  struct ResW {
    Vec gn1_w, gn1_b;
    Mat4 c1_w;
    Vec c1_b;
    Vec gn2_w, gn2_b;
    Mat4 c2_w;
    Vec c2_b;  // conv2 bias (nn.Conv2d default bias=True)
    Mat2 sc_w;  // [cout, cin], empty when cin == cout
    Vec sc_b;
    size_t cin = 0, cout = 0;
    bool has_shortcut() const { return sc_w.extent(0) != 0; }
  };
  struct MidAttnW {
    Vec gn_w, gn_b;
    Mat2 q_w, k_w, v_w;
    Vec q_b, k_b, v_b;
    Mat2 o_w;
    Vec o_b;
  };
  struct UpStage {
    std::vector<ResW> res;
    Mat4 up_w;  // empty when !has_up (final block)
    Vec up_b;
    bool has_up = false;
  };

  Vec bn_mean, bn_var;  // [dit_latent_channels]
  Mat2 pq_w;            // [latent, latent]
  Vec pq_b;
  Mat4 conv_in_w;  // [mid, latent, 3, 3]
  Vec conv_in_b;
  ResW mid_a, mid_b;  // mid-block ResNets (mid -> mid)
  MidAttnW mid_attn;
  std::vector<UpStage> ups;
  Vec nout_w, nout_b;
  Mat4 cout_w;  // [out_channels, widths[0], 3, 3]
  Vec cout_b;

  void allocate(const Config& cfg) {
    cfg.validate();
    const size_t DL = (size_t)cfg.dit_latent_channels;
    const size_t LC = (size_t)cfg.latent_channels;
    const size_t MW = (size_t)cfg.mid_width();
    bn_mean = Vec("vae_bn_mean", DL);
    bn_var = Vec("vae_bn_var", DL);
    pq_w = Mat2("vae_pq_w", LC, LC);
    pq_b = Vec("vae_pq_b", LC);
    conv_in_w = Mat4("vae_conv_in_w", MW, LC, 3, 3);
    conv_in_b = Vec("vae_conv_in_b", MW);
    alloc_res(mid_a, MW, MW, "vae_mid_a");
    alloc_res(mid_b, MW, MW, "vae_mid_b");
    mid_attn.gn_w = Vec("vae_mid_attn_gn_w", MW);
    mid_attn.gn_b = Vec("vae_mid_attn_gn_b", MW);
    mid_attn.q_w = Mat2("vae_mid_attn_q", MW, MW);
    mid_attn.k_w = Mat2("vae_mid_attn_k", MW, MW);
    mid_attn.v_w = Mat2("vae_mid_attn_v", MW, MW);
    mid_attn.q_b = Vec("vae_mid_attn_qb", MW);
    mid_attn.k_b = Vec("vae_mid_attn_kb", MW);
    mid_attn.v_b = Vec("vae_mid_attn_vb", MW);
    mid_attn.o_w = Mat2("vae_mid_attn_o", MW, MW);
    mid_attn.o_b = Vec("vae_mid_attn_ob", MW);
    const int NB = cfg.num_up_blocks();
    ups.resize((size_t)NB);
    for (int i = 0; i < NB; ++i) {
      const size_t cout = (size_t)cfg.decoder_widths[NB - 1 - i];
      const size_t cin_first =
          (i == 0) ? MW : (size_t)cfg.decoder_widths[NB - i];
      UpStage& st = ups[(size_t)i];
      st.has_up = (i < NB - 1);
      if (st.has_up) {
        st.up_w = Mat4("vae_up_w", cout, cout, 3, 3);
        st.up_b = Vec("vae_up_b", cout);
      }
      st.res.resize((size_t)cfg.resnets_per_up_block());
      for (int j = 0; j < cfg.resnets_per_up_block(); ++j) {
        const size_t cin = (j == 0) ? cin_first : cout;
        alloc_res(st.res[(size_t)j], cin, cout,
                  "vae_up" + std::to_string(i) + "r" + std::to_string(j));
      }
    }
    const size_t W0 = (size_t)cfg.decoder_widths[0];
    nout_w = Vec("vae_nout_w", W0);
    nout_b = Vec("vae_nout_b", W0);
    cout_w = Mat4("vae_cout_w", (size_t)cfg.out_channels, W0, 3, 3);
    cout_b = Vec("vae_cout_b", (size_t)cfg.out_channels);
  }

  void release() { *this = Weights<ExecSpace>{}; }

  size_t resident_bytes() const {
    size_t n = 0;
    auto acc = [&](auto& v) { n += v.size() * sizeof(float); };
    acc(bn_mean);
    acc(bn_var);
    acc(pq_w);
    acc(pq_b);
    acc(conv_in_w);
    acc(conv_in_b);
    auto acc_res = [&](const ResW& r) {
      acc(r.gn1_w);
      acc(r.gn1_b);
      acc(r.c1_w);
      acc(r.c1_b);
      acc(r.gn2_w);
      acc(r.gn2_b);
      acc(r.c2_w);
      acc(r.c2_b);
      acc(r.sc_w);
      acc(r.sc_b);
    };
    acc_res(mid_a);
    acc_res(mid_b);
    acc(mid_attn.gn_w);
    acc(mid_attn.gn_b);
    acc(mid_attn.q_w);
    acc(mid_attn.k_w);
    acc(mid_attn.v_w);
    acc(mid_attn.q_b);
    acc(mid_attn.k_b);
    acc(mid_attn.v_b);
    acc(mid_attn.o_w);
    acc(mid_attn.o_b);
    for (auto& st : ups) {
      for (auto& r : st.res) acc_res(r);
      acc(st.up_w);
      acc(st.up_b);
    }
    acc(nout_w);
    acc(nout_b);
    acc(cout_w);
    acc(cout_b);
    return n;
  }

 private:
  static void alloc_res(ResW& r, size_t cin, size_t cout,
                        const std::string& tag) {
    r.cin = cin;
    r.cout = cout;
    r.gn1_w = Vec(tag + "_gn1w", cin);
    r.gn1_b = Vec(tag + "_gn1b", cin);
    r.c1_w = Mat4(tag + "_c1w", cout, cin, 3, 3);
    r.c1_b = Vec(tag + "_c1b", cout);
    r.gn2_w = Vec(tag + "_gn2w", cout);
    r.gn2_b = Vec(tag + "_gn2b", cout);
    r.c2_w = Mat4(tag + "_c2w", cout, cout, 3, 3);
    r.c2_b = Vec(tag + "_c2b", cout);
    if (cin != cout) {
      r.sc_w = Mat2(tag + "_scw", cout, cin);
      r.sc_b = Vec(tag + "_scb", cout);
    }
  }
};

// Fill with small random values (smoke tests only, no checkpoint needed):
// mats ~ N(0, 0.02), biases ~ N(0, 0.02), norm weights = 1, BN var = 1.
template <class ExecSpace>
void random_init(const Config& cfg, Weights<ExecSpace>& w, uint64_t seed) {
  w.allocate(cfg);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  auto fill = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.size(); ++i) h.data()[i] = dist(rng);
    Kokkos::deep_copy(v, h);
  };
  auto ones = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.size(); ++i) h.data()[i] = 1.0f;
    Kokkos::deep_copy(v, h);
  };
  fill(w.bn_mean);
  ones(w.bn_var);
  fill(w.pq_w);
  fill(w.pq_b);
  fill(w.conv_in_w);
  fill(w.conv_in_b);
  auto fill_res = [&](typename Weights<ExecSpace>::ResW& r) {
    ones(r.gn1_w);
    fill(r.gn1_b);
    fill(r.c1_w);
    fill(r.c1_b);
    ones(r.gn2_w);
    fill(r.gn2_b);
    fill(r.c2_w);
    fill(r.c2_b);
    if (r.has_shortcut()) {
      fill(r.sc_w);
      fill(r.sc_b);
    }
  };
  fill_res(w.mid_a);
  fill_res(w.mid_b);
  ones(w.mid_attn.gn_w);
  fill(w.mid_attn.gn_b);
  fill(w.mid_attn.q_w);
  fill(w.mid_attn.k_w);
  fill(w.mid_attn.v_w);
  fill(w.mid_attn.q_b);
  fill(w.mid_attn.k_b);
  fill(w.mid_attn.v_b);
  fill(w.mid_attn.o_w);
  fill(w.mid_attn.o_b);
  for (auto& st : w.ups) {
    for (auto& r : st.res) fill_res(r);
    if (st.has_up) {
      fill(st.up_w);
      fill(st.up_b);
    }
  }
  ones(w.nout_w);
  fill(w.nout_b);
  fill(w.cout_w);
  fill(w.cout_b);
}

// Strict F32 streaming loader for the diffusers-layout safetensors file
// (vae/diffusion_pytorch_model.safetensors, single ~250 MB file). Reads the
// JSON header once, then streams one tensor at a time (HostSpace chunk +
// exactly 1 deep_copy per tensor), so peak host RAM is one tensor, not the
// whole file. Throws on missing file, bad magic, non-F32 dtype, or any
// shape/content mismatch.
template <class ExecSpace>
void load_safetensors(const Config& cfg, const std::string& path,
                       Weights<ExecSpace>& w);

namespace Impl {

// ---- minimal safetensors reader (F32 decoder tensors only) ----
inline uint64_t read_le64(std::ifstream& f) {
  unsigned char b[8];
  f.read(reinterpret_cast<char*>(b), 8);
  if (!f) throw std::runtime_error("vae safetensors: truncated header size");
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

struct SafeEntry {
  std::string dtype;
  std::vector<size_t> shape;
  size_t begin = 0, end = 0;
};

inline size_t safe_numel(const std::vector<size_t>& s) {
  size_t n = 1;
  for (size_t v : s) n *= v;
  return n;
}

// Parse the {"dtype","shape","data_offsets"} object following `"name"`.
inline SafeEntry parse_entry(const std::string& js, size_t key_pos) {
  SafeEntry e;
  const size_t ob = js.find('{', key_pos);
  if (ob == std::string::npos)
    throw std::runtime_error("vae safetensors: malformed header");
  int depth = 0;
  size_t cb = std::string::npos;
  bool in_str = false, esc = false;
  for (size_t i = ob; i < js.size(); ++i) {
    const char c = js[i];
    if (in_str) {
      if (esc)
        esc = false;
      else if (c == '\\')
        esc = true;
      else if (c == '"')
        in_str = false;
    } else if (c == '"') {
      in_str = true;
    } else if (c == '{') {
      ++depth;
    } else if (c == '}') {
      if (--depth == 0) {
        cb = i;
        break;
      }
    }
  }
  if (cb == std::string::npos)
    throw std::runtime_error("vae safetensors: unbalanced header JSON");
  const std::string obj = js.substr(ob, cb - ob + 1);
  auto str_val = [&](const char* key) {
    const std::string q = std::string("\"") + key + "\"";
    const size_t p = obj.find(q);
    if (p == std::string::npos) return std::string();
    const size_t colon = obj.find(':', p + q.size());
    const size_t a = obj.find('"', colon + 1);
    const size_t b = obj.find('"', a + 1);
    return obj.substr(a + 1, b - a - 1);
  };
  auto num_list = [&](const char* key) {
    std::vector<size_t> v;
    const std::string q = std::string("\"") + key + "\"";
    const size_t p = obj.find(q);
    if (p == std::string::npos) return v;
    const size_t a = obj.find('[', p + q.size());
    const size_t b = obj.find(']', a + 1);
    const std::string inner = obj.substr(a + 1, b - a - 1);
    size_t i = 0;
    while (i < inner.size()) {
      while (i < inner.size() && !(inner[i] >= '0' && inner[i] <= '9')) ++i;
      size_t j = i;
      while (j < inner.size() && inner[j] >= '0' && inner[j] <= '9') ++j;
      if (j > i) v.push_back(std::stoull(inner.substr(i, j - i)));
      i = j;
    }
    return v;
  };
  e.dtype = str_val("dtype");
  e.shape = num_list("shape");
  const std::vector<size_t> off = num_list("data_offsets");
  if (off.size() == 2) {
    e.begin = off[0];
    e.end = off[1];
  }
  return e;
}

struct SafeFile {
  std::string path;
  std::string header;
  size_t data_start = 0;

  explicit SafeFile(const std::string& p) : path(p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("vae: cannot open '" + p + "'");
    const long long size = (long long)f.tellg();
    if (size < 8)
      throw std::runtime_error("vae: file too small to be safetensors: '" +
                               p + "'");
    f.seekg(0, std::ios::beg);
    const uint64_t n = read_le64(f);
    if (n > (uint64_t)size || n > (uint64_t)(1 << 26))
      throw std::runtime_error("vae: bad safetensors header size in '" + p +
                               "' (not a safetensors file?)");
    header.resize((size_t)n);
    f.read(header.data(), (std::streamsize)n);
    if (!f) throw std::runtime_error("vae: truncated JSON in '" + p + "'");
    data_start = (size_t)(8 + n);
  }

  SafeEntry find(const std::string& name) const {
    const std::string q = "\"" + name + "\"";
    const size_t p = header.find(q);
    if (p == std::string::npos)
      throw std::runtime_error("vae: tensor '" + name + "' not found in '" +
                               path +
                               "' (need the FLUX.2-small VAE file: "
                               "OpenMOSS-Team/MOSS-Image-SFT/vae/"
                               "diffusion_pytorch_model.safetensors)");
    return parse_entry(header, p);
  }

  // Read one F32 tensor (exact shape) into a host buffer (row-major).
  void read_f32(const SafeEntry& e, const std::string& name,
                const std::vector<size_t>& want,
                std::vector<float>& out) const {
    if (e.dtype != "F32")
      throw std::runtime_error("vae: tensor '" + name + "' has dtype '" +
                               e.dtype + "', need F32");
    if (e.shape != want) {
      auto sh = [](const std::vector<size_t>& s) {
        std::string o = "[";
        for (size_t i = 0; i < s.size(); ++i) {
          if (i) o += ",";
          o += std::to_string(s[i]);
        }
        return o + "]";
      };
      throw std::runtime_error("vae: shape mismatch for '" + name +
                               "': file has " + sh(e.shape) + ", expected " +
                               sh(want));
    }
    if (e.end < e.begin)
      throw std::runtime_error("vae: bad data_offsets for '" + name + "'");
    const size_t n = safe_numel(want);
    if (e.end - e.begin != n * 4)
      throw std::runtime_error("vae: byte-size mismatch for '" + name + "'");
    out.resize(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("vae: cannot reopen '" + path + "'");
    f.seekg((std::streamoff)(data_start + e.begin));
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n * 4));
    if (!f) throw std::runtime_error("vae: truncated data for '" + name + "'");
  }
};

// ---- elementwise glue (native portable kernels; no TPL expresses these) ----

// In-place SiLU over a contiguous [C,H,W] view (SiLU(x) = x*sigmoid(x),
// FP32, cast back). Same math as silu_forward; kept header-local so the
// decoder can stay in NCHW layout without flattening through view casts.
//
// Flat 1D RangePolicy with w innermost: consecutive threads step along w,
// the LayoutRight-contiguous dimension, so both the read and the write
// are coalesced. (A Rank<3> MDRangePolicy steps the leftmost index with
// consecutive threads on CUDA, i.e. channel-strided ~8x sector waste on
// both streams here -- measured ~4x on the 512x512 layers.)
template <class ExecSpace, class V3>
void silu_inplace_3d(const V3& v, const std::string& label) {
  const size_t C = v.extent(0), H = v.extent(1), W = v.extent(2);
  if (C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C * H * W),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = e / (H * W), rem = e % (H * W);
        const size_t h = rem / W, w = rem % W;
        const float x = static_cast<float>(v(c, h, w));
        v(c, h, w) = static_cast<typename V3::value_type>(
            x / (1.0f + expf(-x)));
      });
}

// y += x over [C,H,W] (residual add; x and y must not alias). Same flat,
// w-innermost mapping as silu_inplace_3d: all three streams coalesced.
template <class ExecSpace, class V3>
void add_3d(const V3& y, const V3& x, const std::string& label) {
  const size_t C = y.extent(0), H = y.extent(1), W = y.extent(2);
  if (C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C * H * W),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = e / (H * W), rem = e % (H * W);
        const size_t h = rem / W, w = rem % W;
        y(c, h, w) = static_cast<typename V3::value_type>(
            static_cast<float>(y(c, h, w)) + static_cast<float>(x(c, h, w)));
      });
}

// Nearest-neighbor 2x upsample: y [C,2H,2W] from x [C,H,W]. Same flat,
// w-innermost mapping: consecutive threads read (mostly) the same input
// element pair and write consecutive outputs -- coalesced.
template <class ExecSpace, class V3>
void nearest_upsample2x_3d(const V3& x, const V3& y, const std::string& label) {
  const size_t C = x.extent(0), H = x.extent(1), W = x.extent(2);
  if (C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C * 2 * H * 2 * W),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = e / (2 * H * 2 * W), rem = e % (2 * H * 2 * W);
        const size_t h2 = rem / (2 * W), w2 = rem % (2 * W);
        y(c, h2, w2) = x(c, h2 / 2, w2 / 2);
      });
}

// 32x32 scratch-tiled transpose between [C,H,W] and [T=H*W,C] token rows.
// A direct kernel always strides on one side (channel stride H*W vs token
// row stride C); staging a 32-channel x 32-position tile in scratch makes
// both the NCHW and the token stream coalesced. Portable: TeamPolicy +
// team scratch only; the team size is probed at runtime (same pattern as
// Impl::gemv_native) so team-incapable backends still run correctly.
// The [32][33] pad avoids shared-memory bank conflicts on the transposed
// side; it is just a size, not backend-gated code.
template <class ExecSpace>
struct TransposeTileFunctor {
  using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using V3 = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  using V2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace,
                         Kokkos::MemoryUnmanaged>;
  V3 nchw;  // [C,H,W]
  V2 tok;   // [T,C] with T = H*W
  size_t H, W, shBytes;
  bool to_tokens;  // true: nchw -> tok; false: tok -> nchw

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& team) const {
    constexpr int TT = 32;
    const int Tk = team.team_size();
    const int rank = team.team_rank();
    float* sh = static_cast<float*>(team.team_scratch(0).get_shmem(shBytes));
    // tile[dc][dt]: dc = channel within tile, dt = position within tile.
    auto at = [&](int dc, int dt) -> float& { return sh[dc * (TT + 1) + dt]; };
    const size_t C = nchw.extent(0), T = H * W;
    const size_t nCT = (C + TT - 1) / TT, nTT = (T + TT - 1) / TT;
    const size_t tile = static_cast<size_t>(team.league_rank());
    const size_t tc = tile / nTT, tt = tile % nTT;
    const size_t c0 = tc * TT, t0 = tt * TT;
    if (to_tokens) {
      // Load NCHW tile coalesced (consecutive threads: consecutive dt =
      // consecutive w within a row), then store tokens coalesced
      // (consecutive threads: consecutive dc = consecutive c in a row).
      for (int q = rank; q < TT * TT; q += Tk) {
        const int dc = q / TT, dt = q % TT;
        const size_t t = t0 + static_cast<size_t>(dt);
        float v = 0.0f;
        if (c0 + static_cast<size_t>(dc) < C && t < T)
          v = nchw(c0 + static_cast<size_t>(dc), t / W, t % W);
        at(dc, dt) = v;
      }
      team.team_barrier();
      for (int q = rank; q < TT * TT; q += Tk) {
        const int dc = q % TT, dt = q / TT;
        const size_t t = t0 + static_cast<size_t>(dt);
        if (c0 + static_cast<size_t>(dc) < C && t < T)
          tok(t, c0 + static_cast<size_t>(dc)) = at(dc, dt);
      }
    } else {
      // Mirror image: load tokens coalesced, store NCHW coalesced.
      for (int q = rank; q < TT * TT; q += Tk) {
        const int dc = q % TT, dt = q / TT;
        const size_t t = t0 + static_cast<size_t>(dt);
        float v = 0.0f;
        if (c0 + static_cast<size_t>(dc) < C && t < T)
          v = tok(t, c0 + static_cast<size_t>(dc));
        at(dc, dt) = v;
      }
      team.team_barrier();
      for (int q = rank; q < TT * TT; q += Tk) {
        const int dc = q / TT, dt = q % TT;
        const size_t t = t0 + static_cast<size_t>(dt);
        if (c0 + static_cast<size_t>(dc) < C && t < T)
          nchw(c0 + static_cast<size_t>(dc), t / W, t % W) = at(dc, dt);
      }
    }
  }
};

// Shared launcher for the tiled transposes below.
template <class ExecSpace>
void transpose_tile_launch(
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace,
                       Kokkos::MemoryUnmanaged>& nchw,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace,
                       Kokkos::MemoryUnmanaged>& tok,
    size_t H, size_t W, bool to_tokens, const std::string& label) {
  constexpr int TT = 32;
  const size_t C = nchw.extent(0), T = H * W;
  if (C == 0 || T == 0) return;
  using Functor = TransposeTileFunctor<ExecSpace>;
  const size_t shBytes = static_cast<size_t>(TT) * (TT + 1) * sizeof(float);
  int team_size = 128;
  {
    Kokkos::TeamPolicy<ExecSpace> probe(1, 1);
    probe.set_scratch_size(0, Kokkos::PerTeam(shBytes));
    Functor f{nchw, tok, H, W, shBytes, to_tokens};
    int mx = probe.team_size_max(f, Kokkos::ParallelForTag{});
    if (mx < team_size) team_size = mx;
    if (team_size < 1) team_size = 1;
  }
  const size_t nCT = (C + TT - 1) / TT, nTT = (T + TT - 1) / TT;
  Kokkos::TeamPolicy<ExecSpace> policy(static_cast<int>(nCT * nTT), team_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(shBytes));
  Functor f{nchw, tok, H, W, shBytes, to_tokens};
  Kokkos::parallel_for(label, policy, f);
}

// [C,H,W] -> [T=H*W, C] token rows (t = h*W + w), the layout the core
// conv/linear ops consume.
// Token-layout entries writing into caller-provided buffers (no
// allocation): the same tiled transpose as below, but the output lives in
// persistent scratch. Templated on view types so both managed views and
// unmanaged scratch carves bind. Fully overwrite the output (every
// in-bounds (t, c) is written by exactly one tile), so the destination
// may be uninitialized.
template <class ExecSpace, class V3, class V2>
void to_tokens_into(const V3& x, const V2& tok, const std::string& label) {
  const size_t C = x.extent(0), H = x.extent(1), W = x.extent(2);
  if (C == 0 || H == 0 || W == 0) return;
  using V3u = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace,
                           Kokkos::MemoryUnmanaged>;
  using V2u = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace,
                           Kokkos::MemoryUnmanaged>;
  V3u xu(x.data(), C, H, W);
  V2u tu(tok.data(), H * W, C);
  transpose_tile_launch<ExecSpace>(xu, tu, H, W, true, label + "_k");
}

// [T,C] token rows -> [C,H,W] with T == H*W, into a caller-provided
// buffer (no allocation). Same overwrite contract as to_tokens_into.
template <class ExecSpace, class V2, class V3>
void to_nchw_into(const V2& tok, const V3& x, const std::string& label) {
  const size_t C = x.extent(0), H = x.extent(1), W = x.extent(2);
  if (C == 0 || H == 0 || W == 0) return;
  using V3u = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace,
                           Kokkos::MemoryUnmanaged>;
  using V2u = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace,
                           Kokkos::MemoryUnmanaged>;
  V3u xu(x.data(), C, H, W);
  V2u tu(const_cast<float*>(tok.data()), H * W, C);
  transpose_tile_launch<ExecSpace>(xu, tu, H, W, false, label + "_k");
}

// y = x elementwise copy over [C,H,W] (async stream-ordered kernel).
// Replaces a two-arg Kokkos::deep_copy on the hot path: deep_copy without
// an execution-space argument blocks the host (fence) on CUDA, chopping
// the decode into fully-drained segments, while this kernel stays ordered
// with the surrounding launches on the same stream. Same flat
// w-innermost mapping as the other elementwise glue; bitwise exact.
template <class ExecSpace, class VY, class VX>
void copy_3d(const VY& y, const VX& x, const std::string& label) {
  const size_t C = y.extent(0), H = y.extent(1), W = y.extent(2);
  if (C == 0 || H == 0 || W == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, C * H * W),
      KOKKOS_LAMBDA(size_t e) {
        const size_t c = e / (H * W), rem = e % (H * W);
        const size_t h = rem / W, w = rem % W;
        y(c, h, w) = x(c, h, w);
      });
}

// y += x over [T,C] token rows (residual add in token layout; x and y
// must not alias). Same math as add_3d (one FP32 add per element, no
// accumulation order), so bitwise identical to the NCHW residual while
// letting the decoder stay in token layout across the add. Portable flat
// kernel, exact.
template <class ExecSpace, class VY, class VX>
void add_2d(const VY& y, const VX& x, const std::string& label) {
  const size_t T = y.extent(0), C = y.extent(1);
  if (T == 0 || C == 0) return;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecSpace>(0, T * C),
      KOKKOS_LAMBDA(size_t e) {
        const size_t t = e / C, c = e % C;
        y(t, c) = static_cast<typename VY::value_type>(
            static_cast<float>(y(t, c)) + static_cast<float>(x(t, c)));
      });
}

// [C,H,W] -> [T=H*W, C] token rows (t = h*W + w), the layout the core
// conv/linear ops consume.
template <class ExecSpace>
Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> to_tokens(
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& x,
    const std::string& label) {
  const size_t C = x.extent(0), H = x.extent(1), W = x.extent(2);
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> t(label, H * W, C);
  to_tokens_into<ExecSpace>(x, t, label);
  return t;
}

// [T,C] token rows -> [C,H,W] with T == H*W.
template <class ExecSpace>
Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> to_nchw(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& t, size_t C,
    size_t H, size_t W, const std::string& label) {
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> x(label, C, H, W);
  to_nchw_into<ExecSpace>(t, x, label);
  return x;
}

// GroupNorm over a single-sample [C,H,W] view via an unmanaged [1,C,H,W]
// wrap (same memory, no copy); delegates to the core op. The scratch
// variant takes a caller-provided FP32 buffer ([2*(C*H+num_groups)] for a
// single sample, managed or unmanaged) instead of allocating per call;
// same kernels, same order, same numerics.
template <class ExecSpace, class V3>
void groupnorm_3d_ws(
    const V3& x, const Kokkos::View<float*, ExecSpace>& weight,
    const Kokkos::View<float*, ExecSpace>& bias, size_t num_groups,
    float eps, const V3& y, const Kokkos::View<float*, ExecSpace>& scratch,
    const std::string& label) {
  using V4 = Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  V4 xu(x.data(), 1, x.extent(0), x.extent(1), x.extent(2));
  V4 yu(y.data(), 1, y.extent(0), y.extent(1), y.extent(2));
  ::KokkosDNN::Impl::groupnorm_nchw_with_scratch<ExecSpace, float>(
      xu, weight, bias, eps, num_groups, yu, scratch, label);
}

// GroupNorm over a single-sample [C,H,W] view via an unmanaged [1,C,H,W]
// wrap (same memory, no copy); delegates to the core op.
template <class ExecSpace>
void groupnorm_3d(
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float*, ExecSpace>& weight,
    const Kokkos::View<float*, ExecSpace>& bias, size_t num_groups,
    float eps,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& y,
    const std::string& label) {
  using V4 = Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  V4 xu(x.data(), 1, x.extent(0), x.extent(1), x.extent(2));
  V4 yu(y.data(), 1, y.extent(0), y.extent(1), y.extent(2));
  groupnorm_forward<ExecSpace, float>(xu, weight, bias, num_groups, eps, yu,
                                      label);
}

}  // namespace Impl

template <class ExecSpace>
void load_safetensors(const Config& cfg, const std::string& path,
                       Weights<ExecSpace>& w) {
  cfg.validate();
  w.allocate(cfg);
  Impl::SafeFile sf(path);
  std::vector<float> host;
  auto get = [&](const std::string& name, const std::vector<size_t>& shape) {
    sf.read_f32(sf.find(name), name, shape, host);
    return host;
  };
  // Copy host floats into a device view via its host mirror (ranks differ,
  // but every view is contiguous row-major, so a flat copy is exact).
  auto put = [&](const std::vector<float>& src, auto& dst) {
    auto h = Kokkos::create_mirror_view(dst);
    if (h.size() != src.size())
      throw std::runtime_error("vae: host/device size mismatch");
    std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
    Kokkos::deep_copy(dst, h);
  };
  {
    // Fail fast on inventory drift before streaming ~250 MB of weights.
    for (const auto& t : expected_decoder_tensors(cfg)) (void)sf.find(t.name);
  }
  const size_t DL = (size_t)cfg.dit_latent_channels;
  const size_t LC = (size_t)cfg.latent_channels;
  const size_t MW = (size_t)cfg.mid_width();
  put(get("bn.running_mean", {DL}), w.bn_mean);
  put(get("bn.running_var", {DL}), w.bn_var);
  put(get("post_quant_conv.weight", {LC, LC, 1, 1}), w.pq_w);
  put(get("post_quant_conv.bias", {LC}), w.pq_b);
  put(get("decoder.conv_in.weight", {MW, LC, 3, 3}), w.conv_in_w);
  put(get("decoder.conv_in.bias", {MW}), w.conv_in_b);
  auto get_res = [&](const std::string& p, typename Weights<ExecSpace>::ResW& r,
                     size_t cin, size_t cout, bool shortcut) {
    put(get(p + "norm1.weight", {cin}), r.gn1_w);
    put(get(p + "norm1.bias", {cin}), r.gn1_b);
    put(get(p + "conv1.weight", {cout, cin, 3, 3}), r.c1_w);
    put(get(p + "conv1.bias", {cout}), r.c1_b);
    put(get(p + "norm2.weight", {cout}), r.gn2_w);
    put(get(p + "norm2.bias", {cout}), r.gn2_b);
    put(get(p + "conv2.weight", {cout, cout, 3, 3}), r.c2_w);
    put(get(p + "conv2.bias", {cout}), r.c2_b);
    if (shortcut) {
      put(get(p + "conv_shortcut.weight", {cout, cin, 1, 1}), r.sc_w);
      put(get(p + "conv_shortcut.bias", {cout}), r.sc_b);
    }
  };
  get_res("decoder.mid_block.resnets.0.", w.mid_a, MW, MW, false);
  {
    const std::string p = "decoder.mid_block.attentions.0.";
    put(get(p + "group_norm.weight", {MW}), w.mid_attn.gn_w);
    put(get(p + "group_norm.bias", {MW}), w.mid_attn.gn_b);
    put(get(p + "to_q.weight", {MW, MW}), w.mid_attn.q_w);
    put(get(p + "to_q.bias", {MW}), w.mid_attn.q_b);
    put(get(p + "to_k.weight", {MW, MW}), w.mid_attn.k_w);
    put(get(p + "to_k.bias", {MW}), w.mid_attn.k_b);
    put(get(p + "to_v.weight", {MW, MW}), w.mid_attn.v_w);
    put(get(p + "to_v.bias", {MW}), w.mid_attn.v_b);
    put(get(p + "to_out.0.weight", {MW, MW}), w.mid_attn.o_w);
    put(get(p + "to_out.0.bias", {MW}), w.mid_attn.o_b);
  }
  get_res("decoder.mid_block.resnets.1.", w.mid_b, MW, MW, false);
  const int NB = cfg.num_up_blocks();
  for (int i = 0; i < NB; ++i) {
    const size_t cout = (size_t)cfg.decoder_widths[NB - 1 - i];
    const size_t cin_first =
        (i == 0) ? MW : (size_t)cfg.decoder_widths[NB - i];
    for (int j = 0; j < cfg.resnets_per_up_block(); ++j) {
      const size_t cin = (j == 0) ? cin_first : cout;
      get_res("decoder.up_blocks." + std::to_string(i) + ".resnets." +
                  std::to_string(j) + ".",
              w.ups[(size_t)i].res[(size_t)j], cin, cout, cin != cout);
    }
    if (w.ups[(size_t)i].has_up) {
      const std::string p =
          "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.conv.";
      put(get(p + "weight", {cout, cout, 3, 3}), w.ups[(size_t)i].up_w);
      put(get(p + "bias", {cout}), w.ups[(size_t)i].up_b);
    }
  }
  const size_t W0 = (size_t)cfg.decoder_widths[0];
  put(get("decoder.conv_norm_out.weight", {W0}), w.nout_w);
  put(get("decoder.conv_norm_out.bias", {W0}), w.nout_b);
  put(get("decoder.conv_out.weight",
          {(size_t)cfg.out_channels, W0, 3, 3}),
      w.cout_w);
  put(get("decoder.conv_out.bias", {(size_t)cfg.out_channels}), w.cout_b);
}

namespace Impl {

// ---------------------------------------------------------------------------
// Persistent per-model decode scratch (grow-only, weights-adjacent
// lifetime: owned by DecoderModel). Same pattern as the encoder's
// LinScratch/FullScratch in KokkosDNN_MossImageEncoder.hpp: buffers are
// grown only when a larger (B, H, W) geometry arrives; every decode carves
// metadata-only unmanaged subviews out of them, so steady state performs
// zero allocations. All buffers use AllocateWithoutInitializing: every
// region is fully overwritten before any read (same contract as the
// previous per-call owning views, several of which already relied on it).
// ---------------------------------------------------------------------------
struct ScratchSizes {
  size_t nmax = 0;  // max NCHW/token stage elements (C*H*W)
  size_t cmax = 0;  // max im2col column elements over non-skinny 3x3 convs
  size_t gmax = 0;  // max GroupNorm scratch floats over all GN calls
};

// Maxima over every decode stage for one (H, W) latent geometry. Mirrors
// the decode() stage walk below (same widths, same spatial growth), plus
// the exact skinny/blocking rule from Impl::conv_dense3x3_2d so the col
// budget covers the same row-blocks the lowering takes.
inline ScratchSizes decoder_scratch_sizes(const Config& cfg, size_t H,
                                          size_t W) {
  ScratchSizes s;
  const size_t LC = (size_t)cfg.latent_channels;
  const size_t MW = (size_t)cfg.mid_width();
  const size_t ng = (size_t)cfg.num_groups;
  const size_t H2 = 2 * H, W2 = 2 * W;
  auto stage = [&](size_t C, size_t SH, size_t SW) {
    const size_t T = SH * SW;
    s.nmax = std::max(s.nmax, C * T);
    s.gmax = std::max(s.gmax, 2 * (C * SH + ng));
  };
  auto conv3 = [&](size_t Cin, size_t Cout, size_t T) {
    // Token pairs for this conv (input + output rows).
    s.nmax = std::max(s.nmax, T * Cin);
    s.nmax = std::max(s.nmax, T * Cout);
    const size_t K9 = Cin * 9;
    // Skinny path takes no im2col (same rule as conv_dense3x3_2d).
    if (Cout <= ::KokkosDNN::Impl::ConvSkinnyMaxCout &&
        K9 <= ::KokkosDNN::Impl::ConvSkinnyMaxK9)
      return;
    const size_t per_row = K9 * sizeof(float);
    size_t budget_rows = ::KokkosDNN::Impl::ConvIm2colBudgetBytes / per_row;
    if (budget_rows < 1) budget_rows = 1;
    const size_t Tm = (T < budget_rows) ? T : budget_rows;
    s.cmax = std::max(s.cmax, Tm * K9);
  };
  const size_t T0 = H2 * W2;
  stage(LC, H2, W2);  // depth-to-space sample
  stage(MW, H2, W2);  // conv_in out / mid block
  conv3(LC, MW, T0);  // conv_in
  for (int r = 0; r < 4; ++r) conv3(MW, MW, T0);  // mid ResNet pair
  // Mid attention q/k/v/out/proj + scores plane are sized in ensure().
  const int NB = cfg.num_up_blocks();
  size_t SH = H2, SW = W2;
  for (int i = 0; i < NB; ++i) {
    const size_t cout = (size_t)cfg.decoder_widths[(size_t)NB - 1 - (size_t)i];
    const size_t cin_first =
        (i == 0) ? MW : (size_t)cfg.decoder_widths[(size_t)NB - (size_t)i];
    const size_t T = SH * SW;
    stage(cin_first, SH, SW);
    stage(cout, SH, SW);
    for (int j = 0; j < cfg.resnets_per_up_block(); ++j) {
      const size_t cin = (j == 0) ? cin_first : cout;
      conv3(cin, cout, T);
      conv3(cout, cout, T);
      if (cin != cout) {
        // 1x1 shortcut: token pair only, no im2col.
        s.nmax = std::max(s.nmax, T * cin);
        s.nmax = std::max(s.nmax, T * cout);
      }
    }
    if (i < NB - 1) {
      SH *= 2;
      SW *= 2;
      stage(cout, SH, SW);  // upsampler output (= next block input)
      conv3(cout, cout, SH * SW);
    }
  }
  stage((size_t)cfg.decoder_widths[0], SH, SW);  // head norm / conv_out in
  conv3((size_t)cfg.decoder_widths[0], (size_t)cfg.out_channels, SH * SW);
  return s;
}

template <class ExecSpace>
struct DecoderScratch {
  using F1 = Kokkos::View<float*, ExecSpace>;
  using F2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using F4 = Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace>;

  F4 dn;    // [B,DL,H,W] denormalized latents (batched, write-first)
  F4 d2s;   // [B,LC,2H,2W] depth-to-space latents (batched, write-first)
  F1 n0, n1;      // NCHW stage ping-pong (flat, [nmax] each)
  F1 t0, t1, t2;  // token ping-pong + shortcut (flat, [nmax] each)
  F1 col;   // im2col columns (flat, [cmax])
  F1 gn;    // GroupNorm scratch (flat, [gmax])
  F2 aq, ak, av, ao, ap;  // mid-attention q/k/v/out/proj [Tmid,Cmid]
  F1 attn;  // cuDNN math scores plane [Tmid*Tmid]
  size_t B = 0, H = 0, W = 0;

  void ensure(const Config& cfg, size_t Bn, size_t Hn, size_t Wn) {
    if (Bn <= B && Hn <= H && Wn <= W && dn.extent(0) != 0) return;
    if (Bn > B) B = Bn;
    if (Hn > H) H = Hn;
    if (Wn > W) W = Wn;
    cfg.validate();
    const size_t DL = (size_t)cfg.dit_latent_channels;
    const size_t LC = (size_t)cfg.latent_channels;
    const size_t MW = (size_t)cfg.mid_width();
    const ScratchSizes sz = decoder_scratch_sizes(cfg, H, W);
    const size_t Tmid = (2 * H) * (2 * W);
    dn = F4(Kokkos::ViewAllocateWithoutInitializing("vae_sc_dn"), B, DL, H,
            W);
    d2s = F4(Kokkos::ViewAllocateWithoutInitializing("vae_sc_d2s"), B, LC,
             2 * H, 2 * W);
    n0 = F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_n0"), sz.nmax);
    n1 = F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_n1"), sz.nmax);
    t0 = F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_t0"), sz.nmax);
    t1 = F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_t1"), sz.nmax);
    t2 = F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_t2"), sz.nmax);
    col = F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_col"), sz.cmax);
    gn = F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_gn"), sz.gmax);
    aq = F2(Kokkos::ViewAllocateWithoutInitializing("vae_sc_aq"), Tmid, MW);
    ak = F2(Kokkos::ViewAllocateWithoutInitializing("vae_sc_ak"), Tmid, MW);
    av = F2(Kokkos::ViewAllocateWithoutInitializing("vae_sc_av"), Tmid, MW);
    ao = F2(Kokkos::ViewAllocateWithoutInitializing("vae_sc_ao"), Tmid, MW);
    ap = F2(Kokkos::ViewAllocateWithoutInitializing("vae_sc_ap"), Tmid, MW);
    attn =
        F1(Kokkos::ViewAllocateWithoutInitializing("vae_sc_attn"), Tmid * Tmid);
  }

  void release() { *this = DecoderScratch{}; }

  size_t resident_bytes() const {
    size_t n = 0;
    auto acc = [&](auto& v) { n += v.size() * sizeof(float); };
    acc(dn);
    acc(d2s);
    acc(n0);
    acc(n1);
    acc(t0);
    acc(t1);
    acc(t2);
    acc(col);
    acc(gn);
    acc(aq);
    acc(ak);
    acc(av);
    acc(ao);
    acc(ap);
    acc(attn);
    return n;
  }
};

}  // namespace Impl

// ---------------------------------------------------------------------------
// DecoderModel: FP32 VAE decoder.
// decode(latents [B,128,H,W] -> rgb [B,3,16H,16W]).
//
// Steady-state allocation discipline: every per-forward temporary lives in
// Impl::DecoderScratch (owned by the model, grown once per geometry), so a
// warm decoder performs zero allocations per decode. See DecoderScratch
// for the rationale (cudaMallocAsync + fence per >= 40 kB temporary plus
// ~38 GB of default-init memset traffic otherwise). Token layout is kept
// across consecutive token-legal ops (post-quant -> conv_in tokens feed
// the 3x3 directly; conv outputs feed residual adds in token space),
// transposing to NCHW only where a spatial op needs it (GroupNorm,
// upsample, the final RGB write). Same kernels, shapes, and launch order
// as the allocating path, so numerics are unchanged.
// ---------------------------------------------------------------------------
template <class ExecSpace>
class DecoderModel {
 public:
  using View4 =
      Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace>;
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>;
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using Vec = Kokkos::View<float*, ExecSpace>;
  using U3 = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  using U2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryUnmanaged>;
  using U1 = Kokkos::View<float*, ExecSpace, Kokkos::MemoryUnmanaged>;
  using Mat2 = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using Mat4 = Kokkos::View<float****, Kokkos::LayoutRight, ExecSpace>;

  DecoderModel(const Config& cfg, const Weights<ExecSpace>& w)
      : cfg_(cfg), w_(w) {
    cfg_.validate();
  }

  const Config& config() const { return cfg_; }

  // Bytes held by the persistent decode scratch (grows to the largest
  // geometry decoded so far).
  size_t scratch_bytes() const { return sc_.resident_bytes(); }

  // Full decode: denorm + depth-to-space (batched core ops), then the
  // post-quant / Decoder chain per sample. rgb must be [B,3,16H,16W].
  // Zero steady-state allocations after the first decode of a geometry
  // (scratch is grown in ensure(), never per layer).
  void decode(const View4& latents, View4& rgb) const {
    cfg_.validate();
    const size_t B = latents.extent(0);
    const size_t DL = latents.extent(1);
    const size_t H = latents.extent(2), W = latents.extent(3);
    if (DL != (size_t)cfg_.dit_latent_channels)
      throw std::invalid_argument(
          "MossImage::Vae::DecoderModel::decode: latents must have " +
          std::to_string(cfg_.dit_latent_channels) + " channels");
    if (H == 0 || W == 0)
      throw std::invalid_argument(
          "MossImage::Vae::DecoderModel::decode: H,W must be > 0");
    const size_t oH = (size_t)cfg_.scale_factor * H,
                 oW = (size_t)cfg_.scale_factor * W;
    if (rgb.extent(0) != B || rgb.extent(1) != (size_t)cfg_.out_channels ||
        rgb.extent(2) != oH || rgb.extent(3) != oW)
      throw std::invalid_argument(
          "MossImage::Vae::DecoderModel::decode: rgb must be [B,3,SH,SW] "
          "with S = scale_factor");
    if (B == 0) return;

    Impl::DecoderScratch<ExecSpace>& sc = sc_;
    sc.ensure(cfg_, B, H, W);
    const size_t LC = (size_t)cfg_.latent_channels;
    const size_t MW = (size_t)cfg_.mid_width();
    const size_t SH0 = 2 * H, SW0 = 2 * W, T0 = SH0 * SW0;
    float* Nbuf[2] = {sc.n0.data(), sc.n1.data()};
    float* Tbuf[3] = {sc.t0.data(), sc.t1.data(), sc.t2.data()};
    const U1 col(sc.col.data(), sc.col.extent(0));

    // Pipeline glue (batched, core ops): BN-stat denormalize then 2x2
    // depth-to-space, [B,128,H,W] -> [B,32,2H,2W]. Both fully overwrite
    // the persistent outputs.
    MossImage::moss_vae_denormalize_forward<ExecSpace, float>(
        latents, w_.bn_mean, w_.bn_var, cfg_.bn_eps, sc.dn, "vae_denorm");
    MossImage::moss_vae_depth_to_space_2x<ExecSpace, float>(sc.dn, sc.d2s,
                                                           "vae_d2s");

    for (size_t b = 0; b < B; ++b) {
      auto d2s_b = Kokkos::subview(sc.d2s, b, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      U3 h0(Nbuf[0], LC, SH0, SW0);
      Impl::copy_3d<ExecSpace>(h0, d2s_b, "vae_h0");
      // Post-quant 1x1, token-threaded straight into conv_in: the token
      // rows skip the NCHW round-trip the allocating path took.
      U2 pq_in(Tbuf[0], T0, LC), pq_out(Tbuf[1], T0, LC);
      Impl::to_tokens_into<ExecSpace>(h0, pq_in, "vae_pq");
      linear1x1_tok(pq_in, w_.pq_w, w_.pq_b, pq_out, "vae_pq");
      int cur = 1;
      {
        U2 ci_out(Tbuf[2], T0, MW);
        conv3x3_tok(pq_out, w_.conv_in_w, w_.conv_in_b, ci_out, SH0, SW0,
                    col, "vae_conv_in");
        U3 h1(Nbuf[cur], MW, SH0, SW0);
        Impl::to_nchw_into<ExecSpace>(ci_out, h1, "vae_conv_in");
      }
      size_t cC = MW, cH = SH0, cW = SW0;
      resnet_stage(Nbuf[cur], cC, Nbuf[1 - cur], MW, cH, cW, w_.mid_a, col,
                   "vae_mid_a");
      cur = 1 - cur;
      mid_attention_stage(Nbuf[cur], Nbuf[1 - cur], cH, cW, "vae_mid_attn");
      cur = 1 - cur;
      resnet_stage(Nbuf[cur], cC, Nbuf[1 - cur], MW, cH, cW, w_.mid_b, col,
                   "vae_mid_b");
      cur = 1 - cur;
      const int NB = cfg_.num_up_blocks();
      for (int i = 0; i < NB; ++i) {
        const auto& st = w_.ups[(size_t)i];
        const std::string tag = "vae_up" + std::to_string(i);
        const size_t cout =
            (size_t)cfg_.decoder_widths[(size_t)NB - 1 - (size_t)i];
        for (size_t j = 0; j < st.res.size(); ++j) {
          const size_t cin = (j == 0) ? cC : cout;
          resnet_stage(Nbuf[cur], cin, Nbuf[1 - cur], cout, cH, cW,
                       st.res[j], col,
                       tag + "r" + std::to_string(j));
          cur = 1 - cur;
          cC = cout;
        }
        if (st.has_up) {
          U3 xu(Nbuf[cur], cC, cH, cW);
          cH *= 2;
          cW *= 2;
          U3 xup(Nbuf[1 - cur], cC, cH, cW);
          Impl::nearest_upsample2x_3d<ExecSpace>(xu, xup, tag + "_nn");
          cur = 1 - cur;
          // Upsampler conv runs in place on the upsampled buffer: tokens
          // out of / back into the same NCHW (stream-ordered, same as the
          // in-place GroupNorm contract).
          U3 xc(Nbuf[cur], cC, cH, cW);
          U2 u0(Tbuf[0], cH * cW, cC), u1(Tbuf[1], cH * cW, cC);
          Impl::to_tokens_into<ExecSpace>(xc, u0, tag + "_conv");
          conv3x3_tok(u0, st.up_w, st.up_b, u1, cH, cW, col, tag + "_conv");
          Impl::to_nchw_into<ExecSpace>(u1, xc, tag + "_conv");
        }
      }
      {
        U3 hh(Nbuf[cur], cC, cH, cW);
        groupnorm_now(hh, w_.nout_w, w_.nout_b, "vae_norm_out");
        Impl::silu_inplace_3d<ExecSpace>(hh, "vae_head_silu");
        U2 o0(Tbuf[0], cH * cW, cC),
            o1(Tbuf[1], cH * cW, (size_t)cfg_.out_channels);
        Impl::to_tokens_into<ExecSpace>(hh, o0, "vae_conv_out");
        conv3x3_tok(o0, w_.cout_w, w_.cout_b, o1, cH, cW, col,
                    "vae_conv_out");
        // Conv-out lands directly in the caller's RGB (no staging copy).
        auto rgb_b =
            Kokkos::subview(rgb, b, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        Impl::to_nchw_into<ExecSpace>(o1, rgb_b, "vae_conv_out");
      }
    }
  }

 private:
  Config cfg_;
  Weights<ExecSpace> w_;
  DNNHandle<ExecSpace> handle_;
  mutable Impl::DecoderScratch<ExecSpace> sc_;

  // GroupNorm + SiLU are in-place on the stage buffer; the persistent GN
  // scratch is carved per call (exact prefix, no allocation).
  void groupnorm_now(const U3& v, const Vec& w, const Vec& b,
                     const std::string& tag) const {
    const size_t need =
        2 * (v.extent(0) * v.extent(1) + (size_t)cfg_.num_groups);
    Vec gs = Kokkos::subview(sc_.gn, std::make_pair(size_t{0}, need));
    Impl::groupnorm_3d_ws<ExecSpace>(v, w, b, (size_t)cfg_.num_groups,
                                     cfg_.gn_eps, v, gs, tag);
  }

  // 1x1 conv / linear over token rows via the core biased-GEMM op
  // (cuBLAS ladder intact): y = x*W^T + b. No transposes, no allocation.
  void linear1x1_tok(const U2& x, const Mat2& wt, const Vec& b, const U2& y,
                     const std::string& tag) const {
    gemm_bias<ExecSpace, float>(handle_, x, wt, b, y, tag);
  }

  // 3x3 stride-1 pad-1 conv over token rows with a caller-provided im2col
  // buffer (no allocation). Same dispatch as conv3x3_forward: MIOpen fast
  // path on HIP builds when enabled, skinny team kernel for tiny Cout,
  // otherwise the same row-blocked im2col + gemm_bias lowering with the
  // same block sizes -- same kernels, same shapes, same numerics.
  void conv3x3_tok(const U2& x, const Mat4& wt, const Vec& b, const U2& y,
                   size_t H, size_t W, const U1& col,
                   const std::string& tag) const {
    const size_t T = x.extent(0), Cin = x.extent(1), Cout = wt.extent(0);
    const bool biased = (b.extent(0) != 0);
    if (wt.extent(1) != Cin || wt.extent(2) != 3 || wt.extent(3) != 3 ||
        y.extent(0) != T || y.extent(1) != Cout ||
        (biased && b.extent(0) != Cout))
      throw std::invalid_argument(
          "MossImage::Vae::DecoderModel::conv3x3_tok: dimension mismatch");
    if (H == 0 || W == 0 || T != H * W)
      throw std::invalid_argument(
          "MossImage::Vae::DecoderModel::conv3x3_tok: H and W must be "
          "nonzero with H*W == rows");
    if (T == 0 || Cin == 0 || Cout == 0) return;
#if defined(KOKKOSDNN_ENABLE_MIOPEN) && defined(KOKKOS_ENABLE_HIP)
    constexpr bool is_hip = std::is_same_v<ExecSpace, Kokkos::HIP>;
    if constexpr (is_hip) {
      if (handle_.use_miopen &&
          ::KokkosDNN::Impl::try_miopen_conv3x3_forward<float>(
              handle_, x, wt, b, y, H, W))
        return;
    } else {
      (void)handle_;
    }
#else
    (void)handle_;
#endif
    const size_t K9 = Cin * 9;
    if (Cout <= ::KokkosDNN::Impl::ConvSkinnyMaxCout &&
        K9 <= ::KokkosDNN::Impl::ConvSkinnyMaxK9) {
      ::KokkosDNN::Impl::conv_dense3x3_skinny<ExecSpace, float>(x, wt, b, y,
                                                               H, W, tag);
      return;
    }
    // Row-block the columns exactly like Impl::conv_dense3x3_2d (same Tm
    // per layer, so the same GEMM shapes run).
    size_t Tm = T;
    const size_t per_row = K9 * sizeof(float);
    if (per_row != 0) {
      size_t budget_rows =
          ::KokkosDNN::Impl::ConvIm2colBudgetBytes / per_row;
      if (budget_rows < 1) budget_rows = 1;
      if (Tm > budget_rows) Tm = budget_rows;
    }
    using Mat2u = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace,
                               Kokkos::MemoryUnmanaged>;
    // Zero-copy [Cout, Cin*9] flatten of the dense LayoutRight weight.
    Mat2u w2d(wt.data(), Cout, K9);
    for (size_t r0 = 0; r0 < T; r0 += Tm) {
      const size_t cur = (r0 + Tm <= T) ? Tm : T - r0;
      Mat2u col_cur(col.data(), cur, K9);
      Mat2u y_cur(const_cast<float*>(y.data()) + r0 * Cout, cur, Cout);
      ::KokkosDNN::Impl::conv3x3_im2col_block<ExecSpace, float>(
          x, col_cur, r0, H, W, tag + "_im2col");
      ::KokkosDNN::gemm_bias<ExecSpace, float>(handle_, col_cur, w2d, b,
                                              y_cur, tag + "_gemm");
    }
  }

  // One decode stage (x [Cin,H,W] -> y [Cout,H,W]) on the ping-pong
  // buffers: GN -> SiLU -> conv3x3 -> GN -> SiLU -> conv3x3 -> +shortcut
  // (1x1 iff Cin != Cout) in token space -> NCHW. The residual add stays
  // in token layout (add_2d, bitwise identical to the NCHW add), so only
  // the GN inputs/outputs touch NCHW.
  void resnet_stage(float* xbuf, size_t Cin, float* ybuf, size_t Cout,
                    size_t H, size_t W,
                    const typename Weights<ExecSpace>::ResW& r, const U1& col,
                    const std::string& tag) const {
    Impl::DecoderScratch<ExecSpace>& sc = sc_;
    const size_t T = H * W;
    U3 x(xbuf, Cin, H, W);
    U3 y1(ybuf, Cin, H, W);
    Impl::copy_3d<ExecSpace>(y1, x, tag + "_copy");
    groupnorm_now(y1, r.gn1_w, r.gn1_b, tag + "_n1");
    Impl::silu_inplace_3d<ExecSpace>(y1, tag + "_s1");
    U2 a0(sc.t0.data(), T, Cin), a1(sc.t1.data(), T, Cout);
    Impl::to_tokens_into<ExecSpace>(y1, a0, tag + "_c1");
    conv3x3_tok(a0, r.c1_w, r.c1_b, a1, H, W, col, tag + "_c1");
    // Stage 2 reuses the y buffer at Cout width (stream-ordered after the
    // stage-1 transpose consumed it, same in-place discipline as GN).
    U3 y2(ybuf, Cout, H, W);
    Impl::to_nchw_into<ExecSpace>(a1, y2, tag + "_c1");
    groupnorm_now(y2, r.gn2_w, r.gn2_b, tag + "_n2");
    Impl::silu_inplace_3d<ExecSpace>(y2, tag + "_s2");
    U2 b0(sc.t0.data(), T, Cout), b1(sc.t1.data(), T, Cout);
    Impl::to_tokens_into<ExecSpace>(y2, b0, tag + "_c2");
    conv3x3_tok(b0, r.c2_w, r.c2_b, b1, H, W, col, tag + "_c2");
    if (r.has_shortcut()) {
      U2 s0(sc.t0.data(), T, Cin), s2(sc.t2.data(), T, Cout);
      Impl::to_tokens_into<ExecSpace>(x, s0, tag + "_sc");
      linear1x1_tok(s0, r.sc_w, r.sc_b, s2, tag + "_sc");
      Impl::add_2d<ExecSpace>(b1, s2, tag + "_add");
      Impl::to_nchw_into<ExecSpace>(b1, y2, tag + "_add");
    } else {
      // No shortcut: add in NCHW (x is alive in its own stage buffer),
      // skipping the token round-trip of the residual.
      Impl::to_nchw_into<ExecSpace>(b1, y2, tag + "_add");
      Impl::add_3d<ExecSpace>(y2, x, tag + "_add");
    }
  }

  void mid_attention_stage(float* xbuf, float* ybuf, size_t H, size_t W,
                           const std::string& tag) const {
    Impl::DecoderScratch<ExecSpace>& sc = sc_;
    const size_t C = w_.mid_attn.gn_w.extent(0), T = H * W;
    U3 x(xbuf, C, H, W), y(ybuf, C, H, W);
    const auto& a = w_.mid_attn;
    Impl::copy_3d<ExecSpace>(y, x, tag + "_copy");
    groupnorm_now(y, a.gn_w, a.gn_b, tag + "_gn");
    U2 xt(sc.t0.data(), T, C);
    Impl::to_tokens_into<ExecSpace>(y, xt, tag);
    gemm_bias<ExecSpace, float>(handle_, xt, a.q_w, a.q_b, sc.aq,
                                tag + "_q");
    gemm_bias<ExecSpace, float>(handle_, xt, a.k_w, a.k_b, sc.ak,
                                tag + "_k");
    gemm_bias<ExecSpace, float>(handle_, xt, a.v_w, a.v_b, sc.av,
                                tag + "_v");
    // Single head: reinterpret [T,C] token rows as [T,1,C] triples
    // (same memory; LayoutRight keeps D contiguous, as the op requires).
    using V3u = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace,
                             Kokkos::MemoryUnmanaged>;
    V3u qu(sc.aq.data(), T, 1, C), ku(sc.ak.data(), T, 1, C),
        vu(sc.av.data(), T, 1, C);
    V3u ou(sc.ao.data(), T, 1, C);
    // Persistent scores plane replaces the math fallback's per-call
    // [T,T] allocation; same op sequence and numerics.
    attention_mha_bidirectional<ExecSpace, float>(
        handle_, qu, ku, vu, ou, 0.0f, tag + "_attn", sc.attn.data(),
        sc.attn.extent(0));
    gemm_bias<ExecSpace, float>(handle_, sc.ao, a.o_w, a.o_b, sc.ap,
                                tag + "_out");
    // Residual in token space (xt buffer is free: q/k/v are materialized
    // in the dedicated attention buffers).
    Impl::to_tokens_into<ExecSpace>(x, xt, tag + "_res");
    Impl::add_2d<ExecSpace>(sc.ap, xt, tag + "_res");
    Impl::to_nchw_into<ExecSpace>(sc.ap, y, tag + "_res");
  }
};

}  // namespace Vae
}  // namespace MossImage
}  // namespace KokkosDNN

#endif
