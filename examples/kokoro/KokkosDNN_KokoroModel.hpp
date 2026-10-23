// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_KOKORO_MODEL_HPP
#define KOKKOSDNN_KOKORO_MODEL_HPP

// Kokoro-82M inference: KModel.forward(phonemes, ref_s, speed) ported to
// Kokkos (batch size 1, float). Reference: kokoro/model.py + modules.py +
// istftnet.py (pip package v0.7.x).
//
// Data layout: activations are time-major [T, C] LayoutRight; Linear weights
// stay [out, in] (consumed by gemm_bias, the HF convention, no transpose);
// Conv1d weights stay [Cout, Cin, K]; ConvTranspose1d weights stay
// [Cin, Cout/groups, K]. All device math is FP32-accumulated.
//
// Entry point: Model::synthesize(ids, style, speed, seed, &debug) returns
// audio samples at 24 kHz plus (optionally) deterministic intermediates
// (durations, F0/N curves, aligned ASR features) for golden tests. The only
// stochastic stage is SineGen (host mt19937_64 stream, draw order mirrors
// torch: 9 uniforms, N*9 normals, N normals); Kokoro audio therefore differs
// from the torch reference by RNG stream but is deterministic given `seed`.
//
// Masks: none. The driver always passes the exact token count (no padding),
// so the Albert attention mask is all-valid and every LSTM runs full length
// (matching pack/pad semantics with lengths == T).

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_KokoroConfig.hpp"
#include "KokkosDNN_KokoroWeights.hpp"

namespace KokkosDNN {
namespace Kokoro {

// Deterministic intermediates (host copies) for golden tests.
struct Intermediates {
  std::vector<int> pred_dur;  // [T] int durations (incl. BOS/EOS pads)
  int frames = 0;             // F = sum(pred_dur)
  std::vector<float> f0;      // [2F] Hz-ish F0 curve
  std::vector<float> ncurve;  // [2F] noise curve
  std::vector<float> asr;     // [F * hidden] aligned text features
};

template <class ExecSpace, class Scalar = DefaultScalar>
struct Model {
  using Mat = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>;
  using Vec = Kokkos::View<Scalar*, ExecSpace>;
  using FMat = Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>;
  using FVec = Kokkos::View<float*, ExecSpace>;
  using IVec = Kokkos::View<int*, ExecSpace>;

  Config cfg;
  Weights<ExecSpace, Scalar> weights;
  DNNHandle<ExecSpace> handle;

  Model() = default;
  Model(const Config& c, const Weights<ExecSpace, Scalar>& w)
      : cfg(c), weights(w) {
    cfg.validate();
  }

  // Transpose helper (row-major [R,C] -> [C,R]).
  static Mat transpose2d(const Mat& a) {
    const size_t R = a.extent(0), C = a.extent(1);
    Mat b("kokoro::T", C, R);
    Kokkos::parallel_for(
        "kokoro::transpose",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {R, C}),
        KOKKOS_LAMBDA(size_t i, size_t j) { b(j, i) = a(i, j); });
    return b;
  }
  // -- PL-BERT Albert encoder: ids [T] -> [T, plbert_hidden] -------------
  void albert_encode(const IVec& ids, const Mat& out) {
    const size_t T = ids.extent(0), E = (size_t)cfg.plbert_embed,
                 PH = (size_t)cfg.plbert_hidden, NH = (size_t)cfg.plbert_heads;
    const size_t HDd = PH / NH;
    // Embeddings: word + position + token_type(zeros).
    Mat emb("albert::emb", T, E);
    embedding_gather<ExecSpace, Scalar>(weights.bert.word, ids, emb);
    IVec pos("albert::pos", T), zero("albert::zero", T);
    Kokkos::parallel_for(
        "albert::idx", Kokkos::RangePolicy<ExecSpace>(0, T),
        KOKKOS_LAMBDA(size_t t) {
          pos(t) = (int)t;
          zero(t) = 0;
        });
    Mat pe("albert::pe", T, E), te("albert::te", T, E);
    embedding_gather<ExecSpace, Scalar>(weights.bert.pos, pos, pe);
    embedding_gather<ExecSpace, Scalar>(weights.bert.tokentype, zero, te);
    Mat h("albert::h", T, E);
    Kokkos::parallel_for(
        "albert::sum",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, E}),
        KOKKOS_LAMBDA(size_t t, size_t e) {
          h(t, e) = emb(t, e) + pe(t, e) + te(t, e);
        });
    layernorm_forward<ExecSpace, Scalar>(h, weights.bert.ln_w,
                                         weights.bert.ln_b, cfg.plbert_eps, h);
    // embedding_hidden_mapping_in: [T,E] -> [T,PH].
    Mat x("albert::x", T, PH);
    gemm_bias<ExecSpace, Scalar>(handle, h, weights.bert.map_w,
                                 weights.bert.map_b, x);
    // 12x shared layer.
    Mat q("albert::q", T, PH), k("albert::k", T, PH), v("albert::v", T, PH);
    Mat attn_flat("albert::af", T, PH), ff("albert::ff", T, PH);
    Mat inter("albert::inter", T, (size_t)cfg.plbert_intermediate);
    using QV = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace,
                            Kokkos::MemoryUnmanaged>;
    for (int l = 0; l < cfg.plbert_layers; ++l) {
      gemm_bias<ExecSpace, Scalar>(handle, x, weights.bert.q_w,
                                   weights.bert.q_b, q);
      gemm_bias<ExecSpace, Scalar>(handle, x, weights.bert.k_w,
                                   weights.bert.k_b, k);
      gemm_bias<ExecSpace, Scalar>(handle, x, weights.bert.v_w,
                                   weights.bert.v_b, v);
      QV q3(q.data(), T, NH, HDd), k3(k.data(), T, NH, HDd),
          v3(v.data(), T, NH, HDd), o3(attn_flat.data(), T, NH, HDd);
      attention_mha_bidirectional<ExecSpace, Scalar>(
          handle, q3, k3, v3, o3, 1.0f / std::sqrt((float)HDd), "albert::attn");
      Mat dense("albert::dense", T, PH);
      gemm_bias<ExecSpace, Scalar>(handle, attn_flat, weights.bert.dense_w,
                                   weights.bert.dense_b, dense);
      // h = LN(x + dense).
      Kokkos::parallel_for(
          "albert::res1",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, PH}),
          KOKKOS_LAMBDA(size_t t, size_t d) { dense(t, d) += x(t, d); });
      layernorm_forward<ExecSpace, Scalar>(dense, weights.bert.attn_ln_w,
                                           weights.bert.attn_ln_b,
                                           cfg.plbert_eps, x);
      gemm_bias<ExecSpace, Scalar>(handle, x, weights.bert.ffn_w,
                                   weights.bert.ffn_b, inter);
      gelu_forward<ExecSpace, Scalar>(inter, inter, GeluVariant::Tanh);
      gemm_bias<ExecSpace, Scalar>(handle, inter, weights.bert.ffn_out_w,
                                   weights.bert.ffn_out_b, ff);
      Kokkos::parallel_for(
          "albert::res2",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, PH}),
          KOKKOS_LAMBDA(size_t t, size_t d) { ff(t, d) += x(t, d); });
      layernorm_forward<ExecSpace, Scalar>(ff, weights.bert.full_ln_w,
                                           weights.bert.full_ln_b,
                                           cfg.plbert_eps, x);
    }
    Kokkos::deep_copy(out, x);
  }

  // -- DurationEncoder: d_en [T,HD] + style -> [T,HD+SD] -------------------
  void duration_encode(const Mat& d_en, const FVec& style, const Mat& out) {
    const size_t T = d_en.extent(0), HD = (size_t)cfg.hidden_dim,
                 SD = (size_t)cfg.style_dim;
    Mat xs("dur::xs", T, HD + SD);
    Kokkos::parallel_for(
        "dur::cat",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, HD}),
        KOKKOS_LAMBDA(size_t t, size_t d) {
          xs(t, d) = d_en(t, d);
          if (d < SD) xs(t, HD + d) = static_cast<Scalar>(style(d));
        });
    // n_layer x (BiLSTM -> AdaLN -> concat style).
    IVec lens("dur::len", 1);
    Kokkos::deep_copy(lens, (int)T);
    Mat h("dur::h", T, HD), n("dur::n", T, HD);
    for (int i = 0; i < cfg.n_layer; ++i) {
      auto& lw = weights.dur_lstm[(size_t)i];
      lstm_bidirectional<ExecSpace, Scalar>(xs, lens, lw.w_ih, lw.w_hh,
                                           lw.b_ih, lw.b_hh, h);
      adaln_forward<ExecSpace, Scalar>(h, style, weights.dur_fcw[(size_t)i],
                                       weights.dur_fcb[(size_t)i], 1e-5f, n);
      Kokkos::parallel_for(
          "dur::cat2",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, HD}),
          KOKKOS_LAMBDA(size_t t, size_t d) {
            xs(t, d) = n(t, d);
            if (d < SD) xs(t, HD + d) = static_cast<Scalar>(style(d));
          });
    }
    Kokkos::deep_copy(out, xs);
  }

  // -- TextEncoder: ids [T] -> [T,HD] --------------------------------------
  void text_encode(const IVec& ids, const Mat& out) {
    const size_t T = ids.extent(0), HD = (size_t)cfg.hidden_dim;
    const int K = cfg.text_kernel;
    const size_t P = (size_t)(K - 1) / 2;
    Mat x("text::x", T, HD), y("text::y", T, HD);
    embedding_gather<ExecSpace, Scalar>(weights.text_embed, ids, x);
    for (int i = 0; i < 3; ++i) {
      auto& tc = weights.text_cnn[(size_t)i];
      conv1d_forward<ExecSpace, Scalar>(x, tc.conv.w, tc.conv.b, y, 1, P, 1,
                                       1);
      layernorm_forward<ExecSpace, Scalar>(y, tc.gamma, tc.beta, 1e-5f, x);
      leaky_relu_forward<ExecSpace, Scalar>(x, 0.2f, x);
    }
    IVec lens("text::len", 1);
    Kokkos::deep_copy(lens, (int)T);
    lstm_bidirectional<ExecSpace, Scalar>(x, lens, weights.text_lstm.w_ih,
                                         weights.text_lstm.w_hh,
                                         weights.text_lstm.b_ih,
                                         weights.text_lstm.b_hh, out);
  }

  // -- AdainResBlk1d: x [T,din] -> [T,dout] (up: 2T) ------------------------
  void adain_block(const Mat& x, const FVec& style,
                   const AdainBlkW<ExecSpace, Scalar>& b, const Mat& out) {
    const size_t T = x.extent(0);
    const size_t Tout = b.up ? 2 * T : T;
    Mat r("adain::r", T, (size_t)b.din), u("adain::u", Tout, (size_t)b.din);
    adain1d_forward<ExecSpace, Scalar>(x, style, b.n1w, b.n1b, 1e-5f, r);
    leaky_relu_forward<ExecSpace, Scalar>(r, 0.2f, r);
    if (b.up) {
      conv_transpose1d_forward<ExecSpace, Scalar>(
          r, b.pool.w, b.pool.b, u, 2, 1, 1, (size_t)b.din);
    } else {
      Kokkos::deep_copy(u, r);
    }
    Mat c1("adain::c1", Tout, (size_t)b.dout);
    conv1d_forward<ExecSpace, Scalar>(u, b.conv1.w, b.conv1.b, c1, 1, 1, 1,
                                     1);
    Mat r2("adain::r2", Tout, (size_t)b.dout);
    adain1d_forward<ExecSpace, Scalar>(c1, style, b.n2w, b.n2b, 1e-5f, r2);
    leaky_relu_forward<ExecSpace, Scalar>(r2, 0.2f, r2);
    Mat c2("adain::c2", Tout, (size_t)b.dout);
    conv1d_forward<ExecSpace, Scalar>(r2, b.conv2.w, b.conv2.b, c2, 1, 1, 1,
                                     1);
    // Shortcut: nearest x2 + conv1x1 when dims differ.
    Mat sc("adain::sc", Tout, (size_t)b.dout);
    if (b.din != b.dout) {
      Mat xu("adain::xu", Tout, (size_t)b.din);
      if (b.up)
        upsample_nearest_1d<ExecSpace, Scalar>(x, 2, xu);
      else
        Kokkos::deep_copy(xu, x);
      conv1d_forward<ExecSpace, Scalar>(xu, b.conv1x1.w, b.conv1x1.b, sc, 1,
                                       0, 1, 1);
    } else if (b.up) {
      upsample_nearest_1d<ExecSpace, Scalar>(x, 2, sc);
    } else {
      Kokkos::deep_copy(sc, x);
    }
    const float inv = 1.0f / std::sqrt(2.0f);
    Kokkos::parallel_for(
        "adain::mix",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {Tout, (size_t)b.dout}),
        KOKKOS_LAMBDA(size_t t, size_t d) {
          out(t, d) = static_cast<Scalar>(
              (static_cast<float>(c2(t, d)) + static_cast<float>(sc(t, d))) *
              inv);
        });
  }

  // -- ISTFTNet AdaINResBlock1: x [T,ch] -> [T,ch] ---------------------------
  void res_block(const Mat& x, const FVec& style,
                 const ResBlkW<ExecSpace, Scalar>& b, const Mat& out) {
    const size_t T = x.extent(0), C = (size_t)b.ch;
    Mat cur("res::cur", T, C), xt("res::xt", T, C), a("res::a", T, C),
        z("res::z", T, C);
    Kokkos::deep_copy(cur, x);
    for (int j = 0; j < 3; ++j) {
      const int K = b.k, D = b.dil[j];
      const size_t P = (size_t)D * (size_t)(K - 1) / 2;
      const size_t P2 = (size_t)(K - 1) / 2;  // convs2: dilation 1
      adain1d_forward<ExecSpace, Scalar>(cur, style, b.a1w[(size_t)j],
                                         b.a1b[(size_t)j], 1e-5f, xt);
      snake_forward<ExecSpace, Scalar>(xt, b.al1[(size_t)j], a);
      conv1d_forward<ExecSpace, Scalar>(a, b.c1[(size_t)j].w,
                                       b.c1[(size_t)j].b, xt, 1, P,
                                       (size_t)D, 1);
      adain1d_forward<ExecSpace, Scalar>(xt, style, b.a2w[(size_t)j],
                                         b.a2b[(size_t)j], 1e-5f, a);
      snake_forward<ExecSpace, Scalar>(a, b.al2[(size_t)j], xt);
      conv1d_forward<ExecSpace, Scalar>(xt, b.c2[(size_t)j].w,
                                       b.c2[(size_t)j].b, z, 1, P2, 1, 1);
      Kokkos::parallel_for(
          "res::add",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, C}),
          KOKKOS_LAMBDA(size_t t, size_t d) {
            cur(t, d) = z(t, d) + cur(t, d);
          });
    }
    Kokkos::deep_copy(out, cur);
  }


  // -- SineGen (source module excitation), host-RNG version -----------------
  // f0 [N] -> sine_merge [N]. Draw order mirrors torch (9 uniforms, N*9
  // normals, N normals) from a host mt19937_64(seed).
  // NOTE: voiced threshold is 10 Hz (SourceModuleHnNSF passes
  // voiced_threshod=10 to SineGen), not 0: uv = (f0 > 10).
  void sinegen(const Vec& f0, uint64_t seed, const Vec& merge_out) {
    const size_t N = f0.extent(0);
    const float sr = 24000.0f;
    auto hf0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> rand_ini(9);
    for (int j = 0; j < 9; ++j) rand_ini[(size_t)j] = uni(rng);
    rand_ini[0] = 0.0f;  // no phase noise on the fundamental
    std::vector<float> noise1(N * 9), noise2(N);
    for (size_t i = 0; i < N * 9; ++i) noise1[i] = nrm(rng);
    for (size_t i = 0; i < N; ++i) noise2[i] = nrm(rng);
    // rad_values[n,j] = pymod1(f0[n]*(j+1)/sr), first frame += rand_ini.
    std::vector<float> rad(N * 9);
    for (size_t n = 0; n < N; ++n) {
      const float f = hf0(n);
      for (int j = 0; j < 9; ++j) {
        float v = f * (float)(j + 1) / sr;
        v -= std::floor(v);  // torch remainder (sign of divisor)
        if (n == 0) v += rand_ini[(size_t)j];
        rad[n * 9 + (size_t)j] = v;
      }
    }
    // Downsample x1/300 (linear), cumsum phase x2pi, scale x300, upsample.
    const size_t M = N / 300;  // N is a multiple of 300 by construction
    std::vector<float> dn(M * 9, 0.0f);
    for (size_t d = 0; d < M; ++d)
      for (int j = 0; j < 9; ++j) {
        float src = ((float)d + 0.5f) * 300.0f - 0.5f;
        if (src < 0.0f) src = 0.0f;
        if (src > (float)(N - 1)) src = (float)(N - 1);
        const size_t lo = (size_t)src;
        size_t hi = lo + 1;
        if (hi >= N) hi = N - 1;
        const float wgt = src - (float)lo;
        dn[d * 9 + (size_t)j] =
            rad[lo * 9 + (size_t)j] * (1.0f - wgt) + rad[hi * 9 + (size_t)j] * wgt;
      }
    std::vector<float> sines(N * 9);
    {
      // phase[m,j] = cumsum(dn)[m,j] * 2pi, scaled x300, then linear up.
      std::vector<float> acc(9, 0.0f), up(M * 9);
      for (size_t d = 0; d < M; ++d)
        for (int j = 0; j < 9; ++j) {
          acc[(size_t)j] += dn[d * 9 + (size_t)j];
          up[d * 9 + (size_t)j] = acc[(size_t)j] * 300.0f;
        }
      for (size_t n = 0; n < N; ++n)
        for (int j = 0; j < 9; ++j) {
          float src = ((float)n + 0.5f) * ((float)M / (float)N) - 0.5f;
          if (src < 0.0f) src = 0.0f;
          if (src > (float)(M - 1)) src = (float)(M - 1);
          const size_t lo = (size_t)src;
          size_t hi = lo + 1;
          if (hi >= M) hi = M - 1;
          const float wgt = src - (float)lo;
          const float p = up[lo * 9 + (size_t)j] * (1.0f - wgt) +
                          up[hi * 9 + (size_t)j] * wgt;
          sines[n * 9 + (size_t)j] = std::sin(2.0f * KOKKOSDNN_PI_F * p);
        }
    }
    // sine_waves = sines*uv*0.1 + noise*amp; merge = tanh(lin).
    // (torch SineGen: _f02sine output scaled by sine_amp=0.1, noise drawn
    // randn then scaled by uv*0.003 + (1-uv)*0.1/3.)
    auto hlin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    weights.gen.lin_w);
    auto hlinb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     weights.gen.lin_b);
    std::vector<float> merged(N);
    for (size_t n = 0; n < N; ++n) {
      const float f = hf0(n);
      const float uv = (f > 10.0f) ? 1.0f : 0.0f;
      const float namp = uv * 0.003f + (1.0f - uv) * (0.1f / 3.0f);
      float acc = hlinb(0);
      for (int j = 0; j < 9; ++j) {
        const float sw = sines[n * 9 + (size_t)j] * uv * 0.1f +
                         noise1[n * 9 + (size_t)j] * namp;
        acc += hlin(0, (size_t)j) * sw;
      }
      merged[n] = std::tanh(acc);
    }
    // Drawn to mirror torch's draw order (rand_ini, sine noise, uv noise);
    // the uv-branch noise itself is unused downstream, matching the
    // reference Generator (which only consumes sine_merge).
    (void)noise2;
    auto hm = Kokkos::create_mirror_view(merge_out);
    for (size_t n = 0; n < N; ++n) hm(n) = (Scalar)merged[n];
    Kokkos::deep_copy(merge_out, hm);
  }

  // -- Full synthesis ---------------------------------------------------------
  // ids: phoneme ids WITHOUT pads ([0, *ids, 0] built inside, like KModel).
  // style: host [256] voice row; speed: duration divisor; seed: SineGen RNG.
  // Returns audio samples; fills dbg when non-null.
  std::vector<float> synthesize(const std::vector<int>& ids,
                                const std::vector<float>& style, float speed,
                                uint64_t seed,
                                Intermediates* dbg = nullptr) {
    const size_t HD = (size_t)cfg.hidden_dim, SD = (size_t)cfg.style_dim;
    const size_t T = ids.size() + 2;
    if (T > (size_t)cfg.max_position)
      throw std::invalid_argument("synthesize: phoneme sequence too long");
    if (style.size() != 2 * SD)
      throw std::invalid_argument("synthesize: style must be 256-dim");
    // Device ids [0, *ids, 0].
    IVec dids("kokoro::ids", T);
    {
      auto h = Kokkos::create_mirror_view(dids);
      h(0) = 0;
      for (size_t i = 0; i < ids.size(); ++i) h(i + 1) = ids[i];
      h(T - 1) = 0;
      Kokkos::deep_copy(dids, h);
    }
    FVec dstyle("kokoro::style", SD), sdec("kokoro::sdec", SD);
    {
      auto h = Kokkos::create_mirror_view(dstyle);
      auto hd = Kokkos::create_mirror_view(sdec);
      for (size_t i = 0; i < SD; ++i) {
        hd(i) = style[i];
        h(i) = style[SD + i];
      }
      Kokkos::deep_copy(dstyle, h);
      Kokkos::deep_copy(sdec, hd);
    }
    // 1. Albert + bert_encoder.
    Mat bert("kokoro::bert", T, (size_t)cfg.plbert_hidden);
    albert_encode(dids, bert);
    Mat d_en("kokoro::den", T, HD);
    gemm_bias<ExecSpace, Scalar>(handle, bert, weights.enc_w, weights.enc_b,
                                 d_en);
    // 2. DurationEncoder + predictor LSTM + duration_proj -> pred_dur.
    Mat dmat("kokoro::d", T, HD + SD);
    duration_encode(d_en, dstyle, dmat);
    IVec lensT("kokoro::lenT", 1);
    Kokkos::deep_copy(lensT, (int)T);
    Mat dpre("kokoro::dpre", T, HD);
    lstm_bidirectional<ExecSpace, Scalar>(dmat, lensT, weights.pred_lstm.w_ih,
                                         weights.pred_lstm.w_hh,
                                         weights.pred_lstm.b_ih,
                                         weights.pred_lstm.b_hh, dpre);
    Mat dur_logits("kokoro::dur", T, (size_t)cfg.max_dur);
    gemm_bias<ExecSpace, Scalar>(handle, dpre, weights.dur_proj_w,
                                 weights.dur_proj_b, dur_logits);
    Mat dur_sig("kokoro::dursig", T, (size_t)cfg.max_dur);
    sigmoid_forward<ExecSpace, Scalar>(dur_logits, dur_sig);
    auto hdur = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dur_sig);
    std::vector<int> pred_dur(T);
    size_t F = 0;
    for (size_t t = 0; t < T; ++t) {
      double s = 0.0;
      for (size_t c = 0; c < (size_t)cfg.max_dur; ++c) s += hdur(t, c);
      int d = (int)std::lround(s / (double)speed);
      if (d < 1) d = 1;
      pred_dur[t] = d;
      F += (size_t)d;
    }
    // 3. Alignment: alnT [F,T] one-hot-transposed (host-built).
    Mat alnT("kokoro::alnT", F, T);
    {
      auto h = Kokkos::create_mirror_view(alnT);
      Kokkos::deep_copy(h, (Scalar)0);
      size_t f = 0;
      for (size_t t = 0; t < T; ++t)
        for (int k = 0; k < pred_dur[t]; ++k) h(f++, t) = (Scalar)1;
      Kokkos::deep_copy(alnT, h);
    }
    // 4. Shared LSTM + F0/N stacks -> curves [2F].
    Mat enT("kokoro::enT", F, HD + SD);
    gemm<ExecSpace, Scalar>(handle, (Scalar)1, alnT, transpose2d(dmat),
                           (Scalar)0, enT);
    IVec lensF("kokoro::lenF", 1);
    Kokkos::deep_copy(lensF, (int)F);
    Mat sh("kokoro::sh", F, HD);
    lstm_bidirectional<ExecSpace, Scalar>(enT, lensF, weights.pred_shared.w_ih,
                                         weights.pred_shared.w_hh,
                                         weights.pred_shared.b_ih,
                                         weights.pred_shared.b_hh, sh);
    // F0/N blocks change width mid-stack (512->256 + upsample), so run
    // them explicitly.
    Mat f0x("kokoro::f0x", F, HD), f0y("kokoro::f0y", F, HD);
    Kokkos::deep_copy(f0x, sh);
    adain_block(f0x, dstyle, weights.f0[0], f0y);
    Mat f0u("kokoro::f0u", 2 * F, HD / 2);
    adain_block(f0y, dstyle, weights.f0[1], f0u);
    Mat f0z("kokoro::f0z", 2 * F, HD / 2);
    adain_block(f0u, dstyle, weights.f0[2], f0z);
    Mat f0c("kokoro::f0c", 2 * F, 1);
    conv1d_forward<ExecSpace, Scalar>(f0z, weights.f0_proj.w,
                                     weights.f0_proj.b, f0c, 1, 0, 1, 1);
    Mat nx("kokoro::nx", F, HD), ny("kokoro::ny", F, HD);
    Kokkos::deep_copy(nx, sh);
    adain_block(nx, dstyle, weights.nstack[0], ny);
    Mat nu("kokoro::nu", 2 * F, HD / 2);
    adain_block(ny, dstyle, weights.nstack[1], nu);
    Mat nz("kokoro::nz", 2 * F, HD / 2);
    adain_block(nu, dstyle, weights.nstack[2], nz);
    Mat nc("kokoro::nc", 2 * F, 1);
    conv1d_forward<ExecSpace, Scalar>(nz, weights.n_proj.w, weights.n_proj.b,
                                     nc, 1, 0, 1, 1);
    const size_t F2 = 2 * F;
    // 5. TextEncoder + alignment -> asrT [F,HD].
    Mat ten("kokoro::ten", T, HD);
    text_encode(dids, ten);
    Mat asrT("kokoro::asrT", F, HD);
    gemm<ExecSpace, Scalar>(handle, (Scalar)1, alnT, transpose2d(ten),
                           (Scalar)0, asrT);
    // 6. Decoder conv stack.
    Vec f0v("kokoro::f0v", F2), ncv("kokoro::ncv", F2);
    Kokkos::parallel_for(
        "kokoro::squeeze", Kokkos::RangePolicy<ExecSpace>(0, F2),
        KOKKOS_LAMBDA(size_t i) {
          f0v(i) = f0c(i, 0);
          ncv(i) = nc(i, 0);
        });
    Mat f0d("kokoro::f0d", F, 1), ncd("kokoro::ncd", F, 1);
    conv1d_forward<ExecSpace, Scalar>(f0c, weights.f0_conv.w,
                                     weights.f0_conv.b, f0d, 2, 1, 1, 1);
    conv1d_forward<ExecSpace, Scalar>(nc, weights.n_conv.w, weights.n_conv.b,
                                     ncd, 2, 1, 1, 1);
    Mat xcat("kokoro::xcat", F, HD + 2);
    Kokkos::parallel_for(
        "kokoro::cat",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {F, HD}),
        KOKKOS_LAMBDA(size_t f, size_t d) {
          xcat(f, d) = asrT(f, d);
          if (d == 0) {
            xcat(f, HD) = f0d(f, 0);
            xcat(f, HD + 1) = ncd(f, 0);
          }
        });
    Mat xe("kokoro::xe", F, (size_t)cfg.decode_channels[0]);
    adain_block(xcat, sdec, weights.enc_blk, xe);
    Mat ares("kokoro::ares", F, 64);
    conv1d_forward<ExecSpace, Scalar>(asrT, weights.asr_res.w,
                                     weights.asr_res.b, ares, 1, 0, 1, 1);
    // 4 decode blocks: input = [cur, asr_res, F0, N] ([F, DIN]);
    // block 3 upsamples to [2F, decode_channels[3]].
    const size_t C0 = (size_t)cfg.decode_channels[0];
    const size_t DIN = C0 + 64 + 2;
    Mat cur("kokoro::cur", F, C0);
    Kokkos::deep_copy(cur, xe);
    for (int i = 0; i < 4; ++i) {
      auto& b = weights.dec_blk[(size_t)i];
      const size_t DO = (size_t)b.dout;
      Mat inp("kokoro::din", F, DIN);
      {
        Mat prv = cur;
        Kokkos::parallel_for(
            "kokoro::catd",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                             {F, DIN}),
            KOKKOS_LAMBDA(size_t f, size_t d) {
              float v;
              if (d < C0)
                v = static_cast<float>(prv(f, d));
              else if (d < C0 + 64)
                v = static_cast<float>(ares(f, d - C0));
              else if (d == DIN - 2)
                v = static_cast<float>(f0d(f, 0));
              else
                v = static_cast<float>(ncd(f, 0));
              inp(f, d) = static_cast<Scalar>(v);
            });
      }
      Mat outb("kokoro::dout", b.up ? 2 * F : F, DO);
      adain_block(inp, sdec, b, outb);
      cur = outb;
    }
    // cur is now [F2, UC] (decode.3 output; UC == decode_channels[3]).
    const int UC = cfg.upsample_initial_channel;
    const size_t N = F2 * 300;  // audio-rate samples for SineGen
    // 7. Generator: harmonic source from SineGen(f0 upsampled x300).
    Vec f0up("kokoro::f0up", N);
    upsample_nearest_1d<ExecSpace, Scalar>(f0v, 300, f0up);
    Vec mergev("kokoro::merge", N);
    sinegen(f0up, seed, mergev);
    const size_t NW = (size_t)cfg.gen_n_fft, HOP = (size_t)cfg.gen_hop;
    const size_t Fh = 1 + N / HOP;
    FVec win("kokoro::win", NW);
    fill_hann_window<ExecSpace>(win);
    Mat mag("kokoro::mag", Fh, NW / 2 + 1), pha("kokoro::pha", Fh, NW / 2 + 1);
    stft_forward<ExecSpace>(mergev, win, HOP, mag, pha);
    Mat har("kokoro::har", Fh, NW + 2);
    Kokkos::parallel_for(
        "kokoro::harcat",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {Fh, NW / 2 + 1}),
        KOKKOS_LAMBDA(size_t f, size_t k) {
          har(f, k) = mag(f, k);
          har(f, NW / 2 + 1 + k) = pha(f, k);
        });
    Mat gx("kokoro::gx", F2, (size_t)UC);
    Kokkos::deep_copy(gx, cur);
    for (int i = 0; i < 2; ++i) {
      leaky_relu_forward<ExecSpace, Scalar>(gx, 0.1f, gx);
      // Noise branch: plain conv on har + AdaINResBlock.
      const size_t nch = (size_t)(UC / (i == 0 ? 2 : 4));
      auto& ncvw = (i == 0) ? weights.gen.nc0 : weights.gen.nc1;
      // Output rows: i==0 stride-6 conv (20F rows == ups0 length),
      // i==1 k=1 conv over har (Fh rows) matching the padded length.
      Mat xs("kokoro::xs", (i == 0) ? 20 * F : Fh, nch);
      if (i == 0)
        conv1d_forward<ExecSpace, Scalar>(har, ncvw.w, ncvw.b, xs, 6, 3, 1, 1);
      else
        conv1d_forward<ExecSpace, Scalar>(har, ncvw.w, ncvw.b, xs, 1, 0, 1, 1);
      Mat xsn("kokoro::xsn", xs.extent(0), nch);
      res_block(xs, sdec, weights.gen.nres[(size_t)i], xsn);
      // Upsample x, reflection-pad on the last stage, add noise branch.
      // Transpose pads are (k-u)//2 per the reference (5 for ups0 k20/u10,
      // 3 for ups1 k12/u6).
      auto& upw = weights.gen.ups[(size_t)i];
      const size_t xlen = gx.extent(0);
      const size_t ustride = (i == 0) ? 10 : 6;
      const size_t upad = (i == 0) ? 5 : 3;
      const size_t ukern = (i == 0) ? 20 : 12;
      const size_t ulen = (xlen - 1) * ustride - 2 * upad + ukern;
      Mat xu("kokoro::xu", ulen, nch);
      conv_transpose1d_forward<ExecSpace, Scalar>(gx, upw.w, upw.b, xu,
                                                 ustride, upad, 0, 1);
      Mat xa = xu;
      if (i == 1) {
        // ReflectionPad1d(1, 0): y[0] = x[1], y[1:] = x.
        const size_t L = xu.extent(0), C = xu.extent(1);
        xa = Mat("kokoro::xap", L + 1, C);
        Kokkos::parallel_for(
            "kokoro::refpad",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                {0, 0}, {L + 1, C}),
            KOKKOS_LAMBDA(size_t t, size_t dd) {
              xa(t, dd) = (t == 0) ? xu(1, dd) : xu(t - 1, dd);
            });
      }
      Mat xn("kokoro::xn", xa.extent(0), nch);
      Kokkos::parallel_for(
          "kokoro::xadd",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
              {0, 0}, {xa.extent(0), nch}),
          KOKKOS_LAMBDA(size_t t, size_t dd) {
            xn(t, dd) = xa(t, dd) + xsn(t, dd);
          });
      // Mean of the 3 resblocks (each sees the same input, like the
      // reference `xs = sum(blocks(x)) / 3`).
      Mat acc("kokoro::acc", xn.extent(0), nch);
      Mat rb("kokoro::rb", xn.extent(0), nch);
      res_block(xn, sdec, weights.gen.rb[(size_t)i * 3], rb);
      Kokkos::deep_copy(acc, rb);
      for (int j = 1; j < 3; ++j) {
        res_block(xn, sdec, weights.gen.rb[(size_t)i * 3 + (size_t)j], rb);
        Kokkos::parallel_for(
            "kokoro::rbadd",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                {0, 0}, {xn.extent(0), nch}),
            KOKKOS_LAMBDA(size_t t, size_t dd) {
              acc(t, dd) = acc(t, dd) + rb(t, dd);
            });
      }
      Mat gxn("kokoro::gx2", acc.extent(0), nch);
      Kokkos::parallel_for(
          "kokoro::rbdiv",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
              {0, 0}, {acc.extent(0), nch}),
          KOKKOS_LAMBDA(size_t t, size_t dd) {
            gxn(t, dd) = acc(t, dd) / 3.0f;
          });
      gx = gxn;
    }
    {
    }
    // Final leaky uses the DEFAULT slope (0.01): the reference calls
    // F.leaky_relu(x) with no slope here (in-loop uses 0.1).
    leaky_relu_forward<ExecSpace, Scalar>(gx, 0.01f, gx);
    Mat post("kokoro::post", gx.extent(0), NW + 2);
    conv1d_forward<ExecSpace, Scalar>(gx, weights.gen.post.w,
                                     weights.gen.post.b, post, 1, 3, 1, 1);
    const size_t Fp = post.extent(0), B2 = NW / 2 + 1;
    Mat spec("kokoro::spec", Fp, B2), ph2("kokoro::ph2", Fp, B2);
    Kokkos::parallel_for(
        "kokoro::expsin",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {Fp, B2}),
        KOKKOS_LAMBDA(size_t f, size_t k) {
          spec(f, k) = static_cast<Scalar>(expf(static_cast<float>(post(f, k))));
          ph2(f, k) = static_cast<Scalar>(
              sinf(static_cast<float>(post(f, B2 + k))));
        });
    Vec audio_v("kokoro::audio", (Fp - 1) * HOP);
    istft_inverse<ExecSpace>(spec, ph2, win, HOP, audio_v);
    auto haudio =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), audio_v);
    std::vector<float> audio(haudio.extent(0));
    for (size_t i = 0; i < audio.size(); ++i) audio[i] = (float)haudio(i);

    if (dbg) {
      dbg->pred_dur = pred_dur;
      dbg->frames = (int)F;
      auto hf0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f0v);
      auto hnc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ncv);
      dbg->f0.assign(hf0.data(), hf0.data() + F2);
      dbg->ncurve.assign(hnc.data(), hnc.data() + F2);
      auto hasr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), asrT);
      dbg->asr.resize(F * HD);
      for (size_t f = 0; f < F; ++f)
        for (size_t d = 0; d < HD; ++d) dbg->asr[f * HD + d] = (float)hasr(f, d);
    }
    return audio;
  }
};

}  // namespace Kokoro
}  // namespace KokkosDNN

#endif
