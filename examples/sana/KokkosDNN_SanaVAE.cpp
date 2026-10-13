#include "KokkosDNN_SanaVAE.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <random>

namespace KokkosDNN {
namespace Sana {
namespace VAE {
namespace {

// ---- minimal safetensors reader (F32 decoder tensors only) ----
uint64_t read_le64(std::ifstream& f) {
  unsigned char b[8];
  f.read(reinterpret_cast<char*>(b), 8);
  if (!f) throw std::runtime_error("vae safetensors: truncated header size");
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

struct Entry {
  std::string dtype;
  std::vector<size_t> shape;
  size_t begin = 0, end = 0;
};

size_t numel(const std::vector<size_t>& s) {
  size_t n = 1;
  for (size_t v : s) n *= v;
  return n;
}

// Parse the {"dtype","shape","data_offsets"} object that follows the
// `"name"` key at position key_pos. Shapes promote to size_t; numbers are
// bare JSON integers (no overflow risk at these sizes).
Entry parse_entry(const std::string& js, size_t key_pos) {
  Entry e;
  const size_t ob = js.find('{', key_pos);
  if (ob == std::string::npos)
    throw std::runtime_error("vae safetensors: malformed header");
  int depth = 0;
  size_t cb = std::string::npos;
  bool in_str = false, esc = false;
  for (size_t i = ob; i < js.size(); ++i) {
    const char c = js[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
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
      while (i < inner.size() &&
             !(inner[i] >= '0' && inner[i] <= '9'))
        ++i;
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
    // Header JSON is KiBs for real checkpoints; anything absurd (or past
    // EOF) means this is not a safetensors file.
    if (n > (uint64_t)size || n > (uint64_t)(1 << 26))
      throw std::runtime_error("vae: bad safetensors header size in '" + p +
                               "' (not a safetensors file?)");
    header.resize((size_t)n);
    f.read(header.data(), (std::streamsize)n);
    if (!f) throw std::runtime_error("vae: truncated JSON in '" + p + "'");
    data_start = (size_t)(8 + n);
  }

  Entry find(const std::string& name) const {
    const std::string q = "\"" + name + "\"";
    const size_t p = header.find(q);
    if (p == std::string::npos)
      throw std::runtime_error("vae: tensor '" + name +
                               "' not found in '" + path +
                               "' (need the diffusers-layout DC-AE file: "
                               "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers/"
                               "diffusion_pytorch_model.safetensors)");
    return parse_entry(header, p);
  }

  // Read one F32 tensor (exact shape) into a host buffer (row-major).
  void read_f32(const Entry& e, const std::string& name,
                const std::vector<size_t>& want, std::vector<float>& out) const {
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
    const size_t n = numel(want);
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

}  // namespace

template <class ExecSpace>
void random_init(const Config& cfg, Weights<ExecSpace>& w, uint64_t seed) {
  w.allocate(cfg);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  auto fill = [&](auto& v, float c0, float c1) {
    auto h = Kokkos::create_mirror_view(v);
    const size_t n = h.size();
    for (size_t i = 0; i < n; ++i)
      h.data()[i] = c0 + dist(rng) * c1;
    Kokkos::deep_copy(v, h);
  };
  auto ones = [&](auto& v) {
    auto h = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < h.size(); ++i) h.data()[i] = 1.0f;
    Kokkos::deep_copy(v, h);
  };
  // Rank-agnostic fill via raw span (all views are contiguous).
  auto fill_any = [&](auto& v) { fill(v, 0.0f, 1.0f); };
  fill_any(w.conv_in_w);
  fill_any(w.conv_in_b);
  for (auto& st : w.stages) {
    if (st.has_up) {
      fill_any(st.up_w);
      fill_any(st.up_b);
    }
    for (auto& r : st.res) {
      fill_any(r.conv1_w);
      fill_any(r.conv1_b);
      fill_any(r.conv2_w);
      ones(r.norm_w);
      fill_any(r.norm_b);
    }
    for (auto& e : st.evit) {
      fill_any(e.to_q);
      fill_any(e.to_k);
      fill_any(e.to_v);
      fill_any(e.ms_in);
      fill_any(e.ms_out);
      fill_any(e.attn_out);
      ones(e.attn_nw);
      fill_any(e.attn_nb);
      fill_any(e.gl_inv);
      fill_any(e.gl_inv_b);
      fill_any(e.gl_dw);
      fill_any(e.gl_dw_b);
      fill_any(e.gl_pt);
      ones(e.gl_nw);
      fill_any(e.gl_nb);
    }
  }
  ones(w.norm_out_w);
  fill_any(w.norm_out_b);
  fill_any(w.conv_out_w);
  fill_any(w.conv_out_b);
}

template <class ExecSpace>
void load_safetensors(const Config& cfg, const std::string& path,
                      Weights<ExecSpace>& w) {
  cfg.validate();
  w.allocate(cfg);
  SafeFile sf(path);
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
    const auto spec = expected_decoder_tensors(cfg);
    // Fail fast on inventory drift before streaming ~0.5 GB of weights.
    for (const auto& t : spec) (void)sf.find(t.name);
  }
  put(get("decoder.conv_in.weight",
          {(size_t)cfg.stage_widths.back(), (size_t)cfg.latent_channels, 3, 3}),
      w.conv_in_w);
  put(get("decoder.conv_in.bias", {(size_t)cfg.stage_widths.back()}),
      w.conv_in_b);
  const int S = cfg.num_stages();
  for (int i = 0; i < S; ++i) {
    const int C = cfg.stage_widths[i];
    const std::string p = "decoder.up_blocks." + std::to_string(i) + ".";
    int j = 0;
    if (i < S - 1) {
      const int Cin = cfg.stage_widths[i + 1];
      put(get(p + "0.conv.weight", {(size_t)C, (size_t)Cin, 3, 3}),
          w.stages[(size_t)i].up_w);
      put(get(p + "0.conv.bias", {(size_t)C}), w.stages[(size_t)i].up_b);
      j = 1;
    }
    for (int d = 0; d < cfg.stage_depths[i]; ++d, ++j) {
      const std::string q = p + std::to_string(j) + ".";
      if (cfg.stage_blocks[i] == "ResBlock") {
        auto& r = w.stages[(size_t)i].res[(size_t)d];
        put(get(q + "conv1.weight", {(size_t)C, (size_t)C, 3, 3}), r.conv1_w);
        put(get(q + "conv1.bias", {(size_t)C}), r.conv1_b);
        put(get(q + "conv2.weight", {(size_t)C, (size_t)C, 3, 3}), r.conv2_w);
        put(get(q + "norm.weight", {(size_t)C}), r.norm_w);
        put(get(q + "norm.bias", {(size_t)C}), r.norm_b);
      } else {
        auto& e = w.stages[(size_t)i].evit[(size_t)d];
        const size_t H = 4 * (size_t)C;
        put(get(q + "attn.to_q.weight", {(size_t)C, (size_t)C}), e.to_q);
        put(get(q + "attn.to_k.weight", {(size_t)C, (size_t)C}), e.to_k);
        put(get(q + "attn.to_v.weight", {(size_t)C, (size_t)C}), e.to_v);
        put(get(q + "attn.to_qkv_multiscale.0.proj_in.weight",
                {3 * (size_t)C, 1, 5, 5}),
            e.ms_in);
        put(get(q + "attn.to_qkv_multiscale.0.proj_out.weight",
                {3 * (size_t)C, 32, 1, 1}),
            e.ms_out);
        put(get(q + "attn.to_out.weight", {(size_t)C, 2 * (size_t)C}),
            e.attn_out);
        put(get(q + "attn.norm_out.weight", {(size_t)C}), e.attn_nw);
        put(get(q + "attn.norm_out.bias", {(size_t)C}), e.attn_nb);
        put(get(q + "conv_out.conv_inverted.weight", {2 * H, (size_t)C, 1, 1}),
            e.gl_inv);
        put(get(q + "conv_out.conv_inverted.bias", {2 * H}), e.gl_inv_b);
        put(get(q + "conv_out.conv_depth.weight", {2 * H, 1, 3, 3}), e.gl_dw);
        put(get(q + "conv_out.conv_depth.bias", {2 * H}), e.gl_dw_b);
        put(get(q + "conv_out.conv_point.weight", {(size_t)C, H, 1, 1}),
            e.gl_pt);
        put(get(q + "conv_out.norm.weight", {(size_t)C}), e.gl_nw);
        put(get(q + "conv_out.norm.bias", {(size_t)C}), e.gl_nb);
      }
    }
  }
  put(get("decoder.norm_out.weight", {(size_t)cfg.stage_widths[0]}),
      w.norm_out_w);
  put(get("decoder.norm_out.bias", {(size_t)cfg.stage_widths[0]}),
      w.norm_out_b);
  put(get("decoder.conv_out.weight",
          {(size_t)cfg.out_channels, (size_t)cfg.stage_widths[0], 3, 3}),
      w.conv_out_w);
  put(get("decoder.conv_out.bias", {(size_t)cfg.out_channels}), w.conv_out_b);
}

// ---------------------------------------------------------------------------
// Decoder forward
// ---------------------------------------------------------------------------
template <class ExecSpace>
typename Decoder<ExecSpace>::View2 Decoder<ExecSpace>::res_block(
    const typename Weights<ExecSpace>::ResW& r, const View2& x, size_t C,
    size_t H, size_t W, const std::string& tag) const {
  const size_t T = H * W;
  View2 c1(tag + "_c1", T, C), c2(tag + "_c2", T, C);
  Impl::conv3x3<ExecSpace>(x, r.conv1_w, r.conv1_b, c1, H, W, tag + "_conv1");
  Impl::silu_inplace<ExecSpace>(c1, tag + "_silu");
  typename Weights<ExecSpace>::Vec no_bias;
  Impl::conv3x3<ExecSpace>(c1, r.conv2_w, no_bias, c2, H, W, tag + "_conv2");
  Impl::rmsnorm_wb<ExecSpace>(c2, r.norm_w, r.norm_b, cfg_.norm_eps, c2,
                              tag + "_norm");
  Impl::add_inplace<ExecSpace>(c2, x, tag + "_res");
  return c2;
}

template <class ExecSpace>
typename Decoder<ExecSpace>::View2 Decoder<ExecSpace>::evit_block(
    const typename Weights<ExecSpace>::EViTW& e, const View2& x, size_t C,
    size_t H, size_t W, const std::string& tag) const {
  const size_t T = H * W;
  const size_t heads = C / 32;
  typename Weights<ExecSpace>::Vec no_bias;
  // -- attention branch --
  View2 q(tag + "_q", T, C), k(tag + "_k", T, C), v(tag + "_v", T, C);
  gemm_bias(handle_, x, e.to_q, no_bias, q, tag + "_toq");
  gemm_bias(handle_, x, e.to_k, no_bias, k, tag + "_tok");
  gemm_bias(handle_, x, e.to_v, no_bias, v, tag + "_tov");
  // U = [q;k;v] as [T,3C] token concat.
  View2 u(tag + "_u", T, 3 * C);
  {
    auto uh = Kokkos::create_mirror_view(u);
    auto qh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
    auto kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k);
    auto vh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c) {
        uh(t, c) = qh(t, c);
        uh(t, C + c) = kh(t, c);
        uh(t, 2 * C + c) = vh(t, c);
      }
    Kokkos::deep_copy(u, uh);
  }
  View2 m5(tag + "_m5", T, 3 * C), m(tag + "_m", T, 3 * C);
  Impl::depthwise<ExecSpace, 5>(u, e.ms_in, no_bias, m5, H, W, tag + "_msdw");
  Impl::grouped_conv1x1<ExecSpace>(m5, e.ms_out, m, 3 * heads, tag + "_mspt");
  View2 feats(tag + "_feats", T, 6 * C);
  {
    auto fh = Kokkos::create_mirror_view(feats);
    auto uh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
    auto mh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m);
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < 3 * C; ++c) {
        fh(t, c) = uh(t, c);
        fh(t, 3 * C + c) = mh(t, c);
      }
    Kokkos::deep_copy(feats, fh);
  }
  View2 att(tag + "_att", T, 2 * C);
  Impl::multiscale_attention<ExecSpace>(feats, att, heads, cfg_.attn_eps,
                                        tag + "_msa");
  View2 ao(tag + "_ao", T, C);
  gemm_bias(handle_, att, e.attn_out, no_bias, ao, tag + "_o");
  Impl::rmsnorm_wb<ExecSpace>(ao, e.attn_nw, e.attn_nb, cfg_.norm_eps, ao,
                              tag + "_onorm");
  Impl::add_inplace<ExecSpace>(ao, x, tag + "_ores");
  // -- GLUMBConv branch (expand 4: hidden H=4C, inverted 2H=8C) --
  const size_t Hid = 4 * C;
  View2 gi(tag + "_gi", T, 2 * Hid);
  gemm_bias(handle_, ao, e.gl_inv, e.gl_inv_b, gi, tag + "_ginv");
  Impl::silu_inplace<ExecSpace>(gi, tag + "_gsilu");
  View2 gd(tag + "_gd", T, 2 * Hid);
  Impl::depthwise<ExecSpace, 3>(gi, e.gl_dw, e.gl_dw_b, gd, H, W, tag + "_gdw");
  View2 gh(tag + "_gh", T, Hid);
  Impl::silu_gate<ExecSpace>(gd, gh, Hid, tag + "_gate");
  View2 gp(tag + "_gp", T, C);
  gemm_bias(handle_, gh, e.gl_pt, no_bias, gp, tag + "_gpt");
  Impl::rmsnorm_wb<ExecSpace>(gp, e.gl_nw, e.gl_nb, cfg_.norm_eps, gp,
                              tag + "_gnorm");
  Impl::add_inplace<ExecSpace>(gp, ao, tag + "_gres");
  return gp;
}

template <class ExecSpace>
std::vector<float> Decoder<ExecSpace>::decode(const std::vector<float>& lat,
                                              size_t H, size_t W) const {
  const int S = cfg_.num_stages();
  const size_t CL = (size_t)cfg_.latent_channels;
  if (lat.size() != CL * H * W)
    throw std::invalid_argument("Sana::VAE::Decoder::decode: latents hold " +
                                std::to_string(lat.size()) + " floats, need " +
                                std::to_string(CL * H * W));
  if (H == 0 || W == 0)
    throw std::invalid_argument("Sana::VAE::Decoder::decode: H,W must be > 0");
  // Latent [C,H,W] host -> scaled tokens [T,C].
  View2 h("vae_lat", H * W, CL);
  {
    auto hh = Kokkos::create_mirror_view(h);
    const float s = cfg_.scaling_factor;
    for (size_t c = 0; c < CL; ++c)
      for (size_t t = 0; t < H * W; ++t)
        hh(t, c) = lat[(c * H + t / W) * W + t % W] / s;
    Kokkos::deep_copy(h, hh);
  }
  // conv_in + in_shortcut.
  size_t C = (size_t)cfg_.stage_widths[S - 1];
  size_t cH = H, cW = W;
  {
    View2 y("vae_conv_in", H * W, C);
    Impl::conv3x3<ExecSpace>(h, w_.conv_in_w, w_.conv_in_b, y, H, W,
                             "vae_conv_in");
    Impl::add_repeat_inplace<ExecSpace>(y, h, "vae_in_shortcut");
    h = y;
  }
  // Stages 5..0 with 2x upsamples between them.
  for (int i = S - 1; i >= 0; --i) {
    C = (size_t)cfg_.stage_widths[i];
    const auto& st = w_.stages[(size_t)i];
    const std::string tag = "vae_s" + std::to_string(i);
    for (size_t d = 0; d < st.res.size(); ++d)
      h = res_block(st.res[d], h, C, cH, cW, tag + "r" + std::to_string(d));
    for (size_t d = 0; d < st.evit.size(); ++d)
      h = evit_block(st.evit[d], h, C, cH, cW, tag + "e" + std::to_string(d));
    if (i > 0) {
      // Upsample out of stage i into stage i-1: diffusers up_blocks[i-1]
      // starts with the DCUpBlock (widths[i] -> widths[i-1]).
      const auto& upst = w_.stages[(size_t)(i - 1)];
      const size_t Cout = (size_t)cfg_.stage_widths[i - 1];
      View2 up("vae_up", 4 * cH * cW, Cout);
      View2 nn("vae_nn", 4 * cH * cW, C);
      Impl::nearest_upsample2x<ExecSpace>(h, nn, cH, cW, tag + "_nn");
      Impl::conv3x3<ExecSpace>(nn, upst.up_w, upst.up_b, up, 2 * cH, 2 * cW,
                               tag + "_upconv");
      View2 sh("vae_sh", 4 * cH * cW, Cout);
      Impl::repeat_pixel_shuffle2x<ExecSpace>(h, sh, cH, cW, Cout,
                                             tag + "_upsh");
      Impl::add_inplace<ExecSpace>(up, sh, tag + "_upadd");
      h = up;
      cH *= 2;
      cW *= 2;
    }
  }
  // Head: RMSNorm -> ReLU -> conv3x3 -> RGB.
  Impl::rmsnorm_wb<ExecSpace>(h, w_.norm_out_w, w_.norm_out_b, cfg_.norm_eps, h,
                              "vae_norm_out");
  Impl::relu_inplace<ExecSpace>(h, "vae_head_relu");
  View2 rgb_tok("vae_rgb", cH * cW, (size_t)cfg_.out_channels);
  Impl::conv3x3<ExecSpace>(h, w_.conv_out_w, w_.conv_out_b, rgb_tok, cH, cW,
                           "vae_conv_out");
  // Tokens -> host [3,oH,oW] row-major.
  std::vector<float> rgb((size_t)cfg_.out_channels * cH * cW);
  auto rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rgb_tok);
  for (size_t c = 0; c < (size_t)cfg_.out_channels; ++c)
    for (size_t t = 0; t < cH * cW; ++t)
      rgb[(c * cH + t / cW) * cW + t % cW] = rh(t, c);
  return rgb;
}

// Explicit instantiations (float = checkpoint dtype).
template void random_init<Kokkos::Serial>(const Config&, Weights<Kokkos::Serial>&,
                                         uint64_t);
template void load_safetensors<Kokkos::Serial>(const Config&, const std::string&,
                                              Weights<Kokkos::Serial>&);
template class Decoder<Kokkos::Serial>;
#ifdef KOKKOS_ENABLE_OPENMP
template void random_init<Kokkos::OpenMP>(const Config&, Weights<Kokkos::OpenMP>&,
                                         uint64_t);
template void load_safetensors<Kokkos::OpenMP>(const Config&, const std::string&,
                                              Weights<Kokkos::OpenMP>&);
template class Decoder<Kokkos::OpenMP>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template void random_init<Kokkos::Cuda>(const Config&, Weights<Kokkos::Cuda>&,
                                       uint64_t);
template void load_safetensors<Kokkos::Cuda>(const Config&, const std::string&,
                                            Weights<Kokkos::Cuda>&);
template class Decoder<Kokkos::Cuda>;
#endif

}  // namespace VAE
}  // namespace Sana
}  // namespace KokkosDNN
