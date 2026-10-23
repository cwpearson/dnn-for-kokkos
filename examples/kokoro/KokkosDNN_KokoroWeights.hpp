// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_KOKORO_WEIGHTS_HPP
#define KOKKOSDNN_KOKORO_WEIGHTS_HPP

// Kokoro-82M weight structures + torch.save loader.
//
// Layout convention (matches the PyTorch checkpoint, row-major, verbatim):
// Linear weights stay [out, in] and are consumed by gemm (the HF Linear
// convention, no transpose); Conv1d weights stay [Cout, Cin, K] and
// ConvTranspose1d weights stay [Cin, Cout/groups, K]. weight_norm
// (g, v) pairs are fused on the host at load time into plain dense weights
// (w = g * v / ||v|| over all-but-axis-0), so every device kernel below
// sees plain dense weights. All views are float (checkpoint dtype).
//
// Loader strictness: every expected tensor must be present with exactly the
// expected shape and row-major-contiguous strides; extra archive tensors
// (e.g. bert pooler, unused in KModel.forward) are ignored. Throws on any
// mismatch, so a config/weight mismatch fails loudly at load time.

#include <Kokkos_Core.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "KokkosDNN_KokoroConfig.hpp"

namespace KokkosDNN {
namespace Kokoro {

// Fused dense 1D conv (plain Conv1d or weight_norm-fused): w [Cout,Cin,K]
// (groups fold into Cout/Cin, see conv1d_forward), b [Cout] or empty.
template <class ExecSpace, class Scalar = DefaultScalar>
struct Conv1DW {
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> w;
  Kokkos::View<Scalar*, ExecSpace> b;  // empty == no bias
};

// Fused dense transposed 1D conv: w [Cin,Cout/groups,K], b [Cout] or empty.
template <class ExecSpace, class Scalar = DefaultScalar>
struct ConvT1DW {
  Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> w;
  Kokkos::View<Scalar*, ExecSpace> b;
};

// Single-layer BiLSTM weights, dir 0 = fwd, 1 = rev:
// w_ih [2,4H,D], w_hh [2,4H,H], b_ih/b_hh [2,4H].
template <class ExecSpace, class Scalar = DefaultScalar>
struct LstmW {
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> w_ih, w_hh;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> b_ih, b_hh;
};

// AdainResBlk1d weights (StyleTTS2 decoder / prosody F0-N stacks):
// conv1 (din->dout k3), conv2 (dout->dout k3), optional conv1x1 shortcut
// (din->dout k1, no bias, iff din != dout), optional depthwise-transpose
// pool (din ch, k3 s2, iff upsample), AdaIN fc pairs (norm1 over din,
// norm2 over dout: fc [2C, style]).
template <class ExecSpace, class Scalar = DefaultScalar>
struct AdainBlkW {
  int din = 0, dout = 0;
  bool up = false;
  Conv1DW<ExecSpace, Scalar> conv1, conv2, conv1x1;  // conv1x1.w empty unless din != dout
  ConvT1DW<ExecSpace, Scalar> pool;                  // pool.w empty unless up
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> n1w, n2w;
  Kokkos::View<float*, ExecSpace> n1b, n2b;
};

// ISTFTNet AdaINResBlock1 weights (generator resblocks / noise_res):
// 3x (convs1 k, dilation d[j]) + 3x (convs2 k, d=1), 6 AdaIN fc pairs,
// Snake alphas. ch = channels, k = kernel, dil[3] = dilations.
template <class ExecSpace, class Scalar = DefaultScalar>
struct ResBlkW {
  int ch = 0, k = 3;
  int dil[3] = {1, 3, 5};
  std::array<Conv1DW<ExecSpace, Scalar>, 3> c1, c2;
  std::array<Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>, 3> a1w, a2w;
  std::array<Kokkos::View<float*, ExecSpace>, 3> a1b, a2b;
  std::array<Kokkos::View<float*, ExecSpace>, 3> al1, al2;  // Snake alpha [ch]
};

// PL-BERT Albert weights (one shared layer, applied plbert_layers times).
template <class ExecSpace, class Scalar = DefaultScalar>
struct AlbertW {
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> word, pos, tokentype;
  Kokkos::View<Scalar*, ExecSpace> ln_w, ln_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> map_w;  // [768, 128]
  Kokkos::View<Scalar*, ExecSpace> map_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> q_w, k_w, v_w, dense_w;
  Kokkos::View<Scalar*, ExecSpace> q_b, k_b, v_b, dense_b;
  Kokkos::View<Scalar*, ExecSpace> attn_ln_w, attn_ln_b;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> ffn_w, ffn_out_w;
  Kokkos::View<Scalar*, ExecSpace> ffn_b, ffn_out_b;
  Kokkos::View<Scalar*, ExecSpace> full_ln_w, full_ln_b;
};

// Full Kokoro-82M weights (device views).
template <class ExecSpace, class Scalar = DefaultScalar>
struct Weights {
  AlbertW<ExecSpace, Scalar> bert;
  // bert_encoder Linear 768 -> hidden.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> enc_w;
  Kokkos::View<Scalar*, ExecSpace> enc_b;
  // DurationEncoder: n_layer x (BiLSTM(hidden+style -> hidden/2) + AdaLN fc).
  std::vector<LstmW<ExecSpace, Scalar>> dur_lstm;
  std::vector<Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>> dur_fcw;
  std::vector<Kokkos::View<float*, ExecSpace>> dur_fcb;
  // Predictor head: lstm, duration_proj, shared lstm, F0/N stacks + projs.
  LstmW<ExecSpace, Scalar> pred_lstm, pred_shared;
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> dur_proj_w;
  Kokkos::View<Scalar*, ExecSpace> dur_proj_b;
  std::vector<AdainBlkW<ExecSpace, Scalar>> f0, nstack;
  Conv1DW<ExecSpace, Scalar> f0_proj, n_proj;
  // TextEncoder: embedding, n_text... 3x (conv k5 + LayerNorm), BiLSTM.
  Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> text_embed;
  struct TextConvW {
    Conv1DW<ExecSpace, Scalar> conv;
    Kokkos::View<Scalar*, ExecSpace> gamma, beta;
  };
  std::vector<TextConvW> text_cnn;
  LstmW<ExecSpace, Scalar> text_lstm;
  // Decoder: F0/N downsample convs, encode + 4 decode Adain blocks, asr_res,
  // ISTFTNet generator.
  Conv1DW<ExecSpace, Scalar> f0_conv, n_conv, asr_res;
  AdainBlkW<ExecSpace, Scalar> enc_blk;
  std::vector<AdainBlkW<ExecSpace, Scalar>> dec_blk;  // 4
  struct GeneratorW {
    std::array<ConvT1DW<ExecSpace, Scalar>, 2> ups;
    std::array<ResBlkW<ExecSpace, Scalar>, 6> rb;   // 256ch k3/7/11, 128ch k3/7/11
    std::array<ResBlkW<ExecSpace, Scalar>, 2> nres;  // 256ch k7, 128ch k11
    Conv1DW<ExecSpace, Scalar> nc0, nc1;             // plain (no WN)
    Conv1DW<ExecSpace, Scalar> post;
    Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> lin_w;  // [1,9]
    Kokkos::View<float*, ExecSpace> lin_b;
  } gen;

  void allocate(const Config& cfg);
};

// Host-side raw voice pack: rows [510, 256] (squeezed from [510,1,256]).
struct VoicePack {
  std::vector<float> rows;  // [n_rows * 256]
  int n_rows = 0;           // 510 for af_heart
};

template <class ExecSpace, class Scalar>
void Weights<ExecSpace, Scalar>::allocate(const Config& cfg) {
  cfg.validate();
  const size_t HD = (size_t)cfg.hidden_dim, SD = (size_t)cfg.style_dim;
  const size_t PH = (size_t)cfg.plbert_hidden;
  auto mat2 = [&](const char*, size_t a, size_t b) {
    return Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>("k", a, b);
  };
  auto vec = [&](const char*, size_t a) {
    return Kokkos::View<Scalar*, ExecSpace>("k", a);
  };
  auto fmat2 = [&](const char*, size_t a, size_t b) {
    return Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>("k", a, b);
  };
  auto fvec = [&](const char*, size_t a) {
    return Kokkos::View<float*, ExecSpace>("k", a);
  };
  auto mat3 = [&](const char*, size_t a, size_t b, size_t c) {
    return Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>("k", a, b,
                                                                   c);
  };
  (void)mat2;
  (void)vec;
  // --- Albert ---
  bert.word = mat2("", (size_t)cfg.n_token, (size_t)cfg.plbert_embed);
  bert.pos = mat2("", (size_t)cfg.max_position, (size_t)cfg.plbert_embed);
  bert.tokentype = mat2("", 2, (size_t)cfg.plbert_embed);
  bert.ln_w = vec("", (size_t)cfg.plbert_embed);
  bert.ln_b = vec("", (size_t)cfg.plbert_embed);
  bert.map_w = mat2("", PH, (size_t)cfg.plbert_embed);
  bert.map_b = vec("", PH);
  bert.q_w = mat2("", PH, PH);
  bert.q_b = vec("", PH);
  bert.k_w = mat2("", PH, PH);
  bert.k_b = vec("", PH);
  bert.v_w = mat2("", PH, PH);
  bert.v_b = vec("", PH);
  bert.dense_w = mat2("", PH, PH);
  bert.dense_b = vec("", PH);
  bert.attn_ln_w = vec("", PH);
  bert.attn_ln_b = vec("", PH);
  bert.ffn_w = mat2("", (size_t)cfg.plbert_intermediate, PH);
  bert.ffn_b = vec("", (size_t)cfg.plbert_intermediate);
  bert.ffn_out_w = mat2("", PH, (size_t)cfg.plbert_intermediate);
  bert.ffn_out_b = vec("", PH);
  bert.full_ln_w = vec("", PH);
  bert.full_ln_b = vec("", PH);
  // --- trunk ---
  enc_w = mat2("", HD, PH);
  enc_b = vec("", HD);
  dur_lstm.resize((size_t)cfg.n_layer);
  dur_fcw.resize((size_t)cfg.n_layer);
  dur_fcb.resize((size_t)cfg.n_layer);
  const size_t DD = HD + SD;  // duration LSTM input dim
  const size_t DH = HD / 2;   // per-dir hidden
  auto alloc_lstm = [&](LstmW<ExecSpace, Scalar>& l, size_t d, size_t h) {
    l.w_ih = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>("k", 2,
                                                                    4 * h, d);
    l.w_hh = Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>("k", 2,
                                                                    4 * h, h);
    l.b_ih = fmat2("", 2, 4 * h);
    l.b_hh = fmat2("", 2, 4 * h);
  };
  for (int i = 0; i < cfg.n_layer; ++i) {
    alloc_lstm(dur_lstm[(size_t)i], DD, DH);
    dur_fcw[(size_t)i] = fmat2("", 2 * HD, SD);
    dur_fcb[(size_t)i] = fvec("", 2 * HD);
  }
  alloc_lstm(pred_lstm, DD, DH);
  alloc_lstm(pred_shared, DD, DH);
  dur_proj_w = mat2("", (size_t)cfg.max_dur, HD);
  dur_proj_b = vec("", (size_t)cfg.max_dur);
  auto alloc_adain = [&](AdainBlkW<ExecSpace, Scalar>& b, int din, int dout,
                         bool up) {
    b.din = din;
    b.dout = dout;
    b.up = up;
    b.conv1.w = mat3("", (size_t)dout, (size_t)din, 3);
    b.conv1.b = vec("", (size_t)dout);
    b.conv2.w = mat3("", (size_t)dout, (size_t)dout, 3);
    b.conv2.b = vec("", (size_t)dout);
    if (din != dout) {
      b.conv1x1.w = mat3("", (size_t)dout, (size_t)din, 1);
      b.conv1x1.b = Kokkos::View<Scalar*, ExecSpace>();
    }
    if (up) {
      b.pool.w = mat3("", (size_t)din, 1, 3);
      b.pool.b = vec("", (size_t)din);
    }
    b.n1w = fmat2("", (size_t)2 * (size_t)din, SD);
    b.n1b = fvec("", (size_t)2 * (size_t)din);
    b.n2w = fmat2("", (size_t)2 * (size_t)dout, SD);
    b.n2b = fvec("", (size_t)2 * (size_t)dout);
  };
  f0.resize(3);
  nstack.resize(3);
  // F0/N stacks mirror the reference topology at any width: full channels,
  // then half channels + x2 upsample, then half channels.
  const int fdin[3] = {(int)HD, (int)HD, (int)HD / 2};
  const int fdout[3] = {(int)HD, (int)HD / 2, (int)HD / 2};
  for (int i = 0; i < 3; ++i) {
    const bool up = (i == 1);
    int din = fdin[i];
    int dout = fdout[i];
    alloc_adain(f0[(size_t)i], din, dout, up);
    alloc_adain(nstack[(size_t)i], din, dout, up);
  }
  f0_proj.w = mat3("", 1, HD / 2, 1);
  f0_proj.b = vec("", 1);
  n_proj.w = mat3("", 1, HD / 2, 1);
  n_proj.b = vec("", 1);
  // --- TextEncoder ---
  text_embed = mat2("", (size_t)cfg.n_token, HD);
  text_cnn.resize(3);
  for (auto& tc : text_cnn) {
    tc.conv.w = mat3("", HD, HD, (size_t)cfg.text_kernel);
    tc.conv.b = vec("", HD);
    tc.gamma = vec("", HD);
    tc.beta = vec("", HD);
  }
  alloc_lstm(text_lstm, HD, DH);
  // NOTE: text LSTM hidden is hidden/2 per dir only when hidden is even;
  // tiny HD=32 -> 16. The loader validates against cfg, so mismatched
  // checkpoints fail loudly.
  // --- Decoder ---
  f0_conv.w = mat3("", 1, 1, 3);
  f0_conv.b = vec("", 1);
  n_conv.w = mat3("", 1, 1, 3);
  n_conv.b = vec("", 1);
  asr_res.w = mat3("", 64, HD, 1);
  asr_res.b = vec("", 64);
  alloc_adain(enc_blk, (int)HD + 2, (int)cfg.decode_channels[0], false);
  dec_blk.resize(4);
  for (int i = 0; i < 4; ++i) {
    const int din = (int)cfg.decode_channels[0] + 64 + 2;
    const int dout = (int)cfg.decode_channels[i];
    alloc_adain(dec_blk[(size_t)i], din, dout,
                cfg.decode_upsample_last && i == 3);
  }
  // --- Generator ---
  const int UC = cfg.upsample_initial_channel;
  gen.ups[0].w = mat3("", (size_t)UC, (size_t)(UC / 2), 20);
  gen.ups[0].b = vec("", (size_t)(UC / 2));
  gen.ups[1].w = mat3("", (size_t)(UC / 2), (size_t)(UC / 4), 12);
  gen.ups[1].b = vec("", (size_t)(UC / 4));
  auto alloc_res = [&](ResBlkW<ExecSpace, Scalar>& r, int ch, int k,
                       const int* dil) {
    r.ch = ch;
    r.k = k;
    for (int j = 0; j < 3; ++j) {
      r.dil[j] = dil ? dil[j] : 1;
      r.c1[(size_t)j].w = mat3("", (size_t)ch, (size_t)ch, (size_t)k);
      r.c1[(size_t)j].b = vec("", (size_t)ch);
      r.c2[(size_t)j].w = mat3("", (size_t)ch, (size_t)ch, (size_t)k);
      r.c2[(size_t)j].b = vec("", (size_t)ch);
      r.a1w[(size_t)j] = fmat2("", (size_t)2 * (size_t)ch, SD);
      r.a1b[(size_t)j] = fvec("", (size_t)2 * (size_t)ch);
      r.a2w[(size_t)j] = fmat2("", (size_t)2 * (size_t)ch, SD);
      r.a2b[(size_t)j] = fvec("", (size_t)2 * (size_t)ch);
      r.al1[(size_t)j] = fvec("", (size_t)ch);
      r.al2[(size_t)j] = fvec("", (size_t)ch);
    }
  };
  const int kd[3] = {3, 7, 11}, dd[3] = {1, 3, 5};
  for (int i = 0; i < 3; ++i) alloc_res(gen.rb[(size_t)i], UC / 2, kd[i], dd);
  for (int i = 0; i < 3; ++i) alloc_res(gen.rb[(size_t)3 + (size_t)i], UC / 4, kd[i], dd);
  alloc_res(gen.nres[0], UC / 2, 7, dd);
  alloc_res(gen.nres[1], UC / 4, 11, dd);
  const int NW = cfg.gen_n_fft + 2;
  gen.nc0.w = mat3("", (size_t)(UC / 2), (size_t)NW, 12);
  gen.nc0.b = vec("", (size_t)(UC / 2));
  gen.nc1.w = mat3("", (size_t)(UC / 4), (size_t)NW, 1);
  gen.nc1.b = vec("", (size_t)(UC / 4));
  gen.post.w = mat3("", (size_t)NW, (size_t)(UC / 4), 7);
  gen.post.b = vec("", (size_t)NW);
  gen.lin_w = fmat2("", 1, 9);
  gen.lin_b = fvec("", 1);
}

// Fill with small random values (smoke tests without weights). LayerNorm
// scales are ones, biases/AdaIN fcs/alin tables zero (identity-ish start),
// Snake alphas one.
template <class ExecSpace, class Scalar = DefaultScalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed = 42);

// Load kokoro-v1_0.pth into device views (fuses weight_norm on the host).
template <class ExecSpace, class Scalar = DefaultScalar>
void load_pth_weights(const Config& cfg, const std::string& pth_path,
                      Weights<ExecSpace, Scalar>& w);

// Load a voice pack (voices/*.pt) to host rows [n_rows, 256].
VoicePack load_voice(const std::string& voice_path);

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
