#include "KokkosDNN_KokoroModel.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "KokkosDNN.hpp"

namespace KokkosDNN {
namespace Kokoro {
namespace {

template <class ExecSpace>
Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> btc_to_bct(
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& a) {
  const size_t B = a.extent(0), T = a.extent(1), C = a.extent(2);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> b("t", B, C, T);
  Kokkos::parallel_for(
      "t", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, T, C}),
      KOKKOS_LAMBDA(size_t bb, size_t t, size_t c) { b(bb, c, t) = a(bb, t, c); });
  return b;
}

template <class ExecSpace>
Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> bct_to_btc(
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& a) {
  const size_t B = a.extent(0), C = a.extent(1), T = a.extent(2);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> b("t", B, T, C);
  Kokkos::parallel_for(
      "t", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, C, T}),
      KOKKOS_LAMBDA(size_t bb, size_t c, size_t t) { b(bb, t, c) = a(bb, c, t); });
  return b;
}

template <class ExecSpace>
Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> expand_style(
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& s, size_t T) {
  const size_t B = s.extent(0), S = s.extent(1);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> o("sexp", B, S, T);
  Kokkos::parallel_for(
      "sexp", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, S, T}),
      KOKKOS_LAMBDA(size_t b, size_t ss, size_t t) { o(b, ss, t) = s(b, ss); });
  return o;
}

template <class ExecSpace>
Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> cat_channel(
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& a,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& b) {
  const size_t B = a.extent(0), C1 = a.extent(1), C2 = b.extent(1),
               T = a.extent(2);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> o("cat", B, C1 + C2, T);
  Kokkos::parallel_for(
      "cat", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {B, C1 + C2, T}),
      KOKKOS_LAMBDA(size_t bb, size_t c, size_t t) {
        o(bb, c, t) = (c < C1) ? a(bb, c, t) : b(bb, c - C1, t);
      });
  return o;
}

// ALBERT shared layer on [T,H] (bidirectional, eval).
template <class ExecSpace>
void albert_layer(
    const DNNHandle<ExecSpace>& h,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& x,
    const Weights<ExecSpace, float>& w, int nheads,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& out) {
  const std::string AL = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
  const size_t T = x.extent(0), H = x.extent(1);
  const size_t D = H / (size_t)nheads;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> Q("Q", T, H);
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> K("K", T, H);
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> V("V", T, H);
  gemm_bias(h, x, w.v2(AL + "attention.query.weight"),
            w.v1(AL + "attention.query.bias"), Q, "alq");
  gemm_bias(h, x, w.v2(AL + "attention.key.weight"),
            w.v1(AL + "attention.key.bias"), K, "alk");
  gemm_bias(h, x, w.v2(AL + "attention.value.weight"),
            w.v1(AL + "attention.value.bias"), V, "alv");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> S("S", (size_t)nheads,
                                                            T, T);
  const float scale = 1.0f / sqrtf((float)D);
  Kokkos::parallel_for(
      "S", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {(size_t)nheads, T, T}),
      KOKKOS_LAMBDA(size_t hd, size_t i, size_t j) {
        float acc = 0;
        for (size_t d = 0; d < D; ++d)
          acc += (float)Q(i, hd * D + d) * (float)K(j, hd * D + d);
        S(hd, i, j) = acc * scale;
      });
  Kokkos::parallel_for(
      "Sm", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {(size_t)nheads, T}),
      KOKKOS_LAMBDA(size_t hd, size_t i) {
        float m = (float)S(hd, i, 0);
        for (size_t j = 1; j < T; ++j) m = fmaxf(m, (float)S(hd, i, j));
        float s = 0;
        for (size_t j = 0; j < T; ++j) {
          float e = expf((float)S(hd, i, j) - m);
          S(hd, i, j) = e;
          s += e;
        }
        for (size_t j = 0; j < T; ++j) S(hd, i, j) /= s;
      });
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> A("A", T, H);
  Kokkos::parallel_for(
      "A", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {T, (size_t)nheads, D}),
      KOKKOS_LAMBDA(size_t i, size_t hd, size_t d) {
        float acc = 0;
        for (size_t j = 0; j < T; ++j)
          acc += (float)S(hd, i, j) * (float)V(j, hd * D + d);
        A(i, hd * D + d) = acc;
      });
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> O("O", T, H);
  gemm_bias(h, A, w.v2(AL + "attention.dense.weight"),
            w.v1(AL + "attention.dense.bias"), O, "alo");
  Kokkos::parallel_for(
      "res", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t i, size_t j) { O(i, j) += x(i, j); });
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> An("An", T, H);
  layernorm_forward<ExecSpace, float>(O, w.v1(AL + "attention.LayerNorm.weight"),
                                      w.v1(AL + "attention.LayerNorm.bias"),
                                      1e-12f, An, "aln");
  const size_t PI = w.v2(AL + "ffn.weight").extent(0);
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> F("F", T, PI);
  gemm_bias(h, An, w.v2(AL + "ffn.weight"), w.v1(AL + "ffn.bias"), F, "ffn");
  gelu_forward<ExecSpace, float>(F, F, GeluVariant::Tanh, "gelu");
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> Fo("Fo", T, H);
  gemm_bias(h, F, w.v2(AL + "ffn_output.weight"), w.v1(AL + "ffn_output.bias"),
            Fo, "ffno");
  Kokkos::parallel_for(
      "res2", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t i, size_t j) { Fo(i, j) += An(i, j); });
  layernorm_forward<ExecSpace, float>(Fo, w.v1(AL + "full_layer_layer_norm.weight"),
                                      w.v1(AL + "full_layer_layer_norm.bias"),
                                      1e-12f, out, "fln");
}

// AdainResBlk (predictor F0/N + decoder encode/decode), LeakyReLU 0.2.
template <class ExecSpace>
Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> adain_blk(
    const DNNHandle<ExecSpace>& h,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& style,
    const Weights<ExecSpace, float>& w, const std::string& p, bool up) {
  const size_t B = x.extent(0), Cin = x.extent(1);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> h1("h1", B, Cin,
                                                             x.extent(2));
  adain_forward<ExecSpace, float>(h, x, w.v2(p + ".norm1.fc.weight"),
                               w.v1(p + ".norm1.fc.bias"), style,
                               h1, 1e-5f, "n1");
  leaky_relu_forward<ExecSpace, float>(h1, h1, 0.2f, "a1");
  size_t T2 = h1.extent(2);
  if (up) T2 = 2 * T2;
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> h1u("h1u", B, Cin,
                                                              T2);
  if (up) {
    convtranspose1d_forward<ExecSpace, float>(h1, w.v3(p + ".pool.weight"),
                                              w.v1(p + ".pool.bias"), h1u, 2,
                                              1, 1, Cin, 1, "pool");
  } else {
    Kokkos::deep_copy(h1u, h1);
  }
  const size_t Cout = w.v3(p + ".conv2.weight").extent(0);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> c1("c1", B, Cout, T2);
  conv1d_forward<ExecSpace, float>(h1u, w.v3(p + ".conv1.weight"),
                                   w.v1(p + ".conv1.bias"), c1, 1, 1, 1, 1,
                                   "c1");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> h2("h2", B, Cout, T2);
  adain_forward<ExecSpace, float>(h, c1, w.v2(p + ".norm2.fc.weight"),
                            w.v1(p + ".norm2.fc.bias"), style,
                            h2, 1e-5f, "n2");
  leaky_relu_forward<ExecSpace, float>(h2, h2, 0.2f, "a2");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> c2("c2", B, Cout, T2);
  conv1d_forward<ExecSpace, float>(h2, w.v3(p + ".conv2.weight"),
                                   w.v1(p + ".conv2.bias"), c2, 1, 1, 1, 1,
                                   "c2");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> sc("sc", B, Cout,
                                                              T2);
  const bool has_c1x1 = w.has_v3(p + ".conv1x1.weight");
  Kokkos::View<float*, ExecSpace> nb;
  if (up) {
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> xu("xu", B, Cin,
                                                                T2);
    nearest_upsample1d<ExecSpace, float>(x, xu, (size_t)2, "up");
    if (has_c1x1) {
      conv1d_forward<ExecSpace, float>(xu, w.v3(p + ".conv1x1.weight"), nb, sc,
                                       1, 0, 1, 1, "c1x1");
    } else {
      Kokkos::deep_copy(sc, xu);
    }
  } else if (has_c1x1) {
    conv1d_forward<ExecSpace, float>(x, w.v3(p + ".conv1x1.weight"), nb, sc,
                                     1, 0, 1, 1, "c1x1");
  } else {
    Kokkos::deep_copy(sc, x);
  }
  Kokkos::parallel_for(
      "add", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {B, Cout, T2}),
      KOKKOS_LAMBDA(size_t b, size_t c, size_t t) {
        sc(b, c, t) = (c2(b, c, t) + sc(b, c, t)) * 0.7071067811865475f;
      });
  return sc;
}

// Generator AdaINResBlock1 branch set (Snake), 3 branches dils {1,3,5}.
template <class ExecSpace>
Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> gen_resblk(
    const DNNHandle<ExecSpace>& h,
    const Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace>& x,
    const Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace>& style,
    const Weights<ExecSpace, float>& w, const std::string& p, size_t K,
    const int dils[3]) {
  auto y = x;
  const size_t B = x.extent(0), C = x.extent(1), T = x.extent(2);
  for (int j = 0; j < 3; ++j) {
    const std::string sj = std::to_string(j);
    const size_t pad = (size_t)dils[j] * (K - 1) / 2;
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> h1("h1", B, C, T);
    adain_forward<ExecSpace, float>(h, y, w.v2(p + ".adain1." + sj + ".fc.weight"),
                                    w.v1(p + ".adain1." + sj + ".fc.bias"),
                                    style, h1, 1e-5f, "ga1");
    snake_forward<ExecSpace, float>(h1, w.v3(p + ".alpha1." + sj), h1, "sn1");
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> c1("c1", B, C, T);
    conv1d_forward<ExecSpace, float>(h1, w.v3(p + ".convs1." + sj + ".weight"),
                                     w.v1(p + ".convs1." + sj + ".bias"), c1,
                                     1, pad, (size_t)dils[j], 1, "gc1");
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> h2("h2", B, C, T);
    adain_forward<ExecSpace, float>(h, c1, w.v2(p + ".adain2." + sj + ".fc.weight"),
                                    w.v1(p + ".adain2." + sj + ".fc.bias"),
                                    style, h2, 1e-5f, "ga2");
    snake_forward<ExecSpace, float>(h2, w.v3(p + ".alpha2." + sj), h2, "sn2");
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> c2("c2", B, C, T);
    conv1d_forward<ExecSpace, float>(h2, w.v3(p + ".convs2." + sj + ".weight"),
                                     w.v1(p + ".convs2." + sj + ".bias"), c2,
                                     1, (K - 1) / 2, 1, 1, "gc2");
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> ny("ny", B, C, T);
    Kokkos::parallel_for(
        "radd", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {B, C, T}),
        KOKKOS_LAMBDA(size_t b, size_t c, size_t t) {
          ny(b, c, t) = c2(b, c, t) + y(b, c, t);
        });
    y = ny;
  }
  return y;
}

}  // namespace

template <class ExecSpace>
SynthOutput synthesize(const Config& cfg, const DNNHandle<ExecSpace>& handle,
                       const Weights<ExecSpace, float>& w, const Vocab& vocab,
                       const std::string& phonemes,
                       const Kokkos::View<float*, ExecSpace>& ref_s, float speed) {
  cfg.validate();
  if (speed <= 0) throw std::invalid_argument("synthesize: speed must be > 0");
  std::vector<int> ids;
  ids.push_back(0);
  for (size_t i = 0; i < phonemes.size();) {
    unsigned char c = phonemes[i];
    size_t len = 1;
    if ((c & 0x80) == 0) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else len = 4;
    std::string ch = phonemes.substr(i, len);
    auto it = vocab.find(ch);
    if (it != vocab.end()) ids.push_back(it->second);
    i += len;
  }
  ids.push_back(0);
  const size_t T = ids.size();
  if (T > (size_t)cfg.max_tokens)
    throw std::runtime_error("synthesize: phoneme sequence too long");
  Kokkos::View<int*, ExecSpace> toks("toks", T);
  {
    auto h = Kokkos::create_mirror_view(toks);
    for (size_t i = 0; i < T; ++i) h(i) = ids[i];
    Kokkos::deep_copy(toks, h);
  }
  const size_t E = 128, PH = (size_t)cfg.plbert_hidden_size,
               H = (size_t)cfg.hidden_dim, S = (size_t)cfg.style_dim;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> we("we", T, E),
      pe("pe", T, E), te("te", T, E);
  {
    auto tbl = w.v2("bert.embeddings.word_embeddings.weight");
    Kokkos::parallel_for(
        "we", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, E}),
        KOKKOS_LAMBDA(size_t t, size_t e) { we(t, e) = tbl(toks(t), e); });
  }
  {
    auto tbl = w.v2("bert.embeddings.position_embeddings.weight");
    Kokkos::parallel_for(
        "pe", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, E}),
        KOKKOS_LAMBDA(size_t t, size_t e) { pe(t, e) = tbl(t, e); });
  }
  {
    auto tbl = w.v2("bert.embeddings.token_type_embeddings.weight");
    Kokkos::parallel_for(
        "te", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, E}),
        KOKKOS_LAMBDA(size_t t, size_t e) { te(t, e) = tbl(0, e); });
  }
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> emb("emb", T, E);
  Kokkos::parallel_for(
      "emb", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, E}),
      KOKKOS_LAMBDA(size_t t, size_t e) { emb(t, e) = we(t, e) + pe(t, e) + te(t, e); });
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> embn("embn", T, E);
  layernorm_forward<ExecSpace, float>(emb, w.v1("bert.embeddings.LayerNorm.weight"), w.v1("bert.embeddings.LayerNorm.bias"), 1e-12f, embn, "eln");
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> hid("hid", T, PH);
  gemm_bias(handle, embn, w.v2("bert.encoder.embedding_hidden_mapping_in.weight"), w.v1("bert.encoder.embedding_hidden_mapping_in.bias"), hid, "map");
  for (int l = 0; l < cfg.plbert_layers; ++l) {
    Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> nxt("nxt", T, PH);
    albert_layer<ExecSpace>(handle, hid, w, cfg.plbert_heads, nxt);
    hid = nxt;
  }
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> be("be", T, H);
  gemm_bias(handle, hid, w.v2("bert_encoder.weight"), w.v1("bert_encoder.bias"), be, "benc");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> den("den", 1, H, T);
  Kokkos::parallel_for(
      "den", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t c) { den(0, c, t) = be(t, c); });
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> style("style", 1, S);
  Kokkos::parallel_for(
      "st", Kokkos::RangePolicy<ExecSpace>(0, S),
      KOKKOS_LAMBDA(size_t s) { style(0, s) = ref_s(s); });
  // Voice split (KModel): ref_s[:, :128] conditions the decoder/generator,
  // ref_s[:, 128:] conditions the prosody predictor.
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> style_p("style_p", 1, S);
  Kokkos::parallel_for(
      "stp", Kokkos::RangePolicy<ExecSpace>(0, S),
      KOKKOS_LAMBDA(size_t s) { style_p(0, s) = ref_s(S + s); });
  auto xd = den;
  {
    auto sexp0 = expand_style<ExecSpace>(style_p, T);
    xd = cat_channel<ExecSpace>(xd, sexp0);
  }
  for (int l = 0; l < 3; ++l) {
    const std::string P = "predictor.text_encoder.lstms." + std::to_string(2 * l) + ".";
    auto xbtc = bct_to_btc<ExecSpace>(xd);
    Kokkos::View<int*, ExecSpace> lens("lens", 1);
    Kokkos::deep_copy(lens, (int)T);
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> lo("lo", 1, T, H);
    lstm_forward_bidir<ExecSpace, float>(handle, xbtc, w.v2(P + "weight_ih_l0"), w.v2(P + "weight_hh_l0"), w.v1(P + "bias_ih_l0"), w.v1(P + "bias_hh_l0"), w.v2(P + "weight_ih_l0_reverse"), w.v2(P + "weight_hh_l0_reverse"), w.v1(P + "bias_ih_l0_reverse"), w.v1(P + "bias_hh_l0_reverse"), lo, lens, "dur_lstm");
    auto lob = btc_to_bct<ExecSpace>(lo);
    const std::string F = "predictor.text_encoder.lstms." + std::to_string(2 * l + 1) + ".fc.";
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> la("la", 1, lob.extent(1), T);
    adalayernorm_forward<ExecSpace, float>(handle, lob, w.v2(F + "weight"), w.v1(F + "bias"), style_p, la, 1e-5f, "daln");
    xd = cat_channel<ExecSpace>(la, expand_style<ExecSpace>(style_p, T));
  }
  auto d_btc = bct_to_btc<ExecSpace>(xd);
  Kokkos::View<int*, ExecSpace> lens1("lens1", 1);
  Kokkos::deep_copy(lens1, (int)T);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> lstm_o("lstm_o", 1, T, H);
  lstm_forward_bidir<ExecSpace, float>(handle, d_btc, w.v2("predictor.lstm.weight_ih_l0"), w.v2("predictor.lstm.weight_hh_l0"), w.v1("predictor.lstm.bias_ih_l0"), w.v1("predictor.lstm.bias_hh_l0"), w.v2("predictor.lstm.weight_ih_l0_reverse"), w.v2("predictor.lstm.weight_hh_l0_reverse"), w.v1("predictor.lstm.bias_ih_l0_reverse"), w.v1("predictor.lstm.bias_hh_l0_reverse"), lstm_o, lens1, "plstm");
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> lflat("lf", T, H), dflat("df", T, (size_t)cfg.max_dur);
  Kokkos::parallel_for(
      "lf", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t c) { lflat(t, c) = lstm_o(0, t, c); });
  gemm_bias(handle, lflat, w.v2("predictor.duration_proj.linear_layer.weight"), w.v1("predictor.duration_proj.linear_layer.bias"), dflat, "dproj");
  std::vector<int> pred_dur(T);
  {
    auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dflat);
    for (size_t t = 0; t < T; ++t) {
      double s = 0;
      for (int k = 0; k < cfg.max_dur; ++k) s += 1.0 / (1.0 + std::exp(-(double)hd(t, k)));
      s /= speed;
      pred_dur[t] = std::max(1, (int)std::lround(s));
    }
  }
  size_t Tf = 0;
  for (int d : pred_dur) Tf += (size_t)d;
  if (Tf > (size_t)cfg.max_frames)
    throw std::runtime_error("synthesize: regulated frames exceed max_frames");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> en("en", 1, H + S, Tf);
  {
    auto db = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_btc);
    auto eh = Kokkos::create_mirror_view(en);
    size_t f = 0;
    for (size_t t = 0; t < T; ++t)
      for (int r = 0; r < pred_dur[t]; ++r, ++f)
        for (size_t c = 0; c < H + S; ++c) eh(0, c, f) = db(0, t, c);
    Kokkos::deep_copy(en, eh);
  }
  auto en_btc = bct_to_btc<ExecSpace>(en);
  Kokkos::View<int*, ExecSpace> lensf("lensf", 1);
  Kokkos::deep_copy(lensf, (int)Tf);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> sh_o("sho", 1, Tf, H);
  lstm_forward_bidir<ExecSpace, float>(
      handle, en_btc, w.v2("predictor.shared.weight_ih_l0"), w.v2("predictor.shared.weight_hh_l0"),
      w.v1("predictor.shared.bias_ih_l0"), w.v1("predictor.shared.bias_hh_l0"),
      w.v2("predictor.shared.weight_ih_l0_reverse"), w.v2("predictor.shared.weight_hh_l0_reverse"),
      w.v1("predictor.shared.bias_ih_l0_reverse"), w.v1("predictor.shared.bias_hh_l0_reverse"),
      sh_o, lensf, "shared");
  auto sh_bct = btc_to_bct<ExecSpace>(sh_o);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> f0x = sh_bct, nx = sh_bct;
  for (int i = 0; i < 3; ++i) f0x = adain_blk<ExecSpace>(handle, f0x, style_p, w, "predictor.F0." + std::to_string(i), i == 1);
  for (int i = 0; i < 3; ++i) nx = adain_blk<ExecSpace>(handle, nx, style_p, w, "predictor.N." + std::to_string(i), i == 1);
  const size_t T2f = f0x.extent(2);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> f0p("f0p", 1, 1, T2f), np("np", 1, 1, T2f);
  conv1d_forward<ExecSpace, float>(f0x, w.v3("predictor.F0_proj.weight"), w.v1("predictor.F0_proj.bias"), f0p, 1, 0, 1, 1, "f0p");
  conv1d_forward<ExecSpace, float>(nx, w.v3("predictor.N_proj.weight"), w.v1("predictor.N_proj.bias"), np, 1, 0, 1, 1, "np");
  std::vector<float> f0_curve(T2f), n_curve(T2f);
  {
    auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f0p);
    auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, np);
    for (size_t i = 0; i < T2f; ++i) {
      f0_curve[i] = a(0, 0, i);
      n_curve[i] = b(0, 0, i);
    }
  }
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> temb("temb", T, H);
  {
    auto tbl = w.v2("text_encoder.embedding.weight");
    Kokkos::parallel_for(
        "te2", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({(size_t)0, (size_t)0}, {T, H}),
        KOKKOS_LAMBDA(size_t t, size_t c) { temb(t, c) = tbl(toks(t), c); });
  }
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> tx("tx", 1, H, T);
  Kokkos::parallel_for(
      "tx", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({(size_t)0, (size_t)0}, {T, H}),
      KOKKOS_LAMBDA(size_t t, size_t c) { tx(0, c, t) = temb(t, c); });
  for (int i = 0; i < 3; ++i) {
    const std::string P = "text_encoder.cnn." + std::to_string(i) + ".";
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> co("co", 1, H, T);
    conv1d_forward<ExecSpace, float>(tx, w.v3(P + "0.weight"), w.v1(P + "0.bias"), co, 1, 2, 1, 1, "tc");
    Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> co2("co2", T, H), cn2("cn2", T, H);
    Kokkos::parallel_for(
        "rsh", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({(size_t)0, (size_t)0}, {T, H}),
        KOKKOS_LAMBDA(size_t t, size_t c) { co2(t, c) = co(0, c, t); });
    layernorm_forward<ExecSpace, float>(co2, w.v1(P + "1.gamma"), w.v1(P + "1.beta"), 1e-5f, cn2, "tln");
    Kokkos::parallel_for(
        "wr", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({(size_t)0, (size_t)0}, {T, H}),
        KOKKOS_LAMBDA(size_t t, size_t c) { co(0, c, t) = cn2(t, c); });
    leaky_relu_forward<ExecSpace, float>(co, co, 0.2f, "tl");
    tx = co;
  }
  auto tx_btc = bct_to_btc<ExecSpace>(tx);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> tx_o("txo", 1, T, H);
  lstm_forward_bidir<ExecSpace, float>(
      handle, tx_btc, w.v2("text_encoder.lstm.weight_ih_l0"), w.v2("text_encoder.lstm.weight_hh_l0"),
      w.v1("text_encoder.lstm.bias_ih_l0"), w.v1("text_encoder.lstm.bias_hh_l0"),
      w.v2("text_encoder.lstm.weight_ih_l0_reverse"), w.v2("text_encoder.lstm.weight_hh_l0_reverse"),
      w.v1("text_encoder.lstm.bias_ih_l0_reverse"), w.v1("text_encoder.lstm.bias_hh_l0_reverse"),
      tx_o, lens1, "tlstm");
  auto ten = btc_to_bct<ExecSpace>(tx_o);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> asr("asr", 1, H, Tf);
  {
    auto th = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ten);
    auto ah = Kokkos::create_mirror_view(asr);
    size_t f = 0;
    for (size_t t = 0; t < T; ++t)
      for (int r = 0; r < pred_dur[t]; ++r, ++f)
        for (size_t c = 0; c < H; ++c) ah(0, c, f) = th(0, c, t);
    Kokkos::deep_copy(asr, ah);
  }
  // Decoder: F0/N stride-2 convs, encode + 4 decode blocks (last upsamples).
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> f0c("f0c", 1, 1, Tf), nc("nc", 1, 1, Tf);
  {
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> f0in("f0in", 1, 1, T2f), nin("nin", 1, 1, T2f);
    auto fa = Kokkos::create_mirror_view(f0in);
    auto fb = Kokkos::create_mirror_view(nin);
    for (size_t i = 0; i < T2f; ++i) {
      fa(0, 0, i) = f0_curve[i];
      fb(0, 0, i) = n_curve[i];
    }
    Kokkos::deep_copy(f0in, fa);
    Kokkos::deep_copy(nin, fb);
    conv1d_forward<ExecSpace, float>(f0in, w.v3("decoder.F0_conv.weight"), w.v1("decoder.F0_conv.bias"), f0c, 2, 1, 1, 1, "f0c");
    conv1d_forward<ExecSpace, float>(nin, w.v3("decoder.N_conv.weight"), w.v1("decoder.N_conv.bias"), nc, 2, 1, 1, 1, "nc");
  }
  auto dec_x = cat_channel<ExecSpace>(asr, f0c);
  dec_x = cat_channel<ExecSpace>(dec_x, nc);
  auto enc_o = adain_blk<ExecSpace>(handle, dec_x, style, w, "decoder.encode", false);
  const size_t AW = (size_t)cfg.upsample_initial_channel / 8;
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> asr_r("asrr", 1, AW, Tf);
  conv1d_forward<ExecSpace, float>(asr, w.v3("decoder.asr_res.0.weight"), w.v1("decoder.asr_res.0.bias"), asr_r, 1, 0, 1, 1, "asrr");
  auto xdec = enc_o;
  auto f0_stage = f0c;
  auto n_stage = nc;
  for (int i = 0; i < 4; ++i) {
    const std::string P = "decoder.decode." + std::to_string(i);
    auto cat = cat_channel<ExecSpace>(xdec, asr_r);
    if (cat.extent(2) != f0_stage.extent(2)) {
      Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> f2("f2", 1, 1, cat.extent(2)), n2("n2", 1, 1, cat.extent(2));
      nearest_upsample1d<ExecSpace, float>(f0_stage, f2, (size_t)2, "fup");
      nearest_upsample1d<ExecSpace, float>(n_stage, n2, (size_t)2, "nup");
      f0_stage = f2;
      n_stage = n2;
      Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> ar2("ar2", 1, AW, cat.extent(2));
      nearest_upsample1d<ExecSpace, float>(asr_r, ar2, (size_t)2, "aup");
      asr_r = ar2;
      cat = cat_channel<ExecSpace>(xdec, asr_r);
    }
    cat = cat_channel<ExecSpace>(cat, f0_stage);
    cat = cat_channel<ExecSpace>(cat, n_stage);
    xdec = adain_blk<ExecSpace>(handle, cat, style, w, P, i == 3);
  }
  // Generator (deterministic: zero-init phase, no random noise).
  const size_t Twave = 300 * T2f;
  if (Twave > (size_t)cfg.max_samples)
    throw std::runtime_error("synthesize: waveform exceeds max_samples");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> f0u("f0u", 1, 1, Twave);
  {
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> f02("f02", 1, 1, T2f);
    auto h = Kokkos::create_mirror_view(f02);
    for (size_t i = 0; i < T2f; ++i) h(0, 0, i) = f0_curve[i];
    Kokkos::deep_copy(f02, h);
    nearest_upsample1d<ExecSpace, float>(f02, f0u, 300, "f0up");
  }
  const int HDIM = 9;
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> sine9("sine9", 1, Twave, HDIM);
  {
    auto fu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f0u);
    auto sh = Kokkos::create_mirror_view(sine9);
    // Deterministic noise (fixed seed) matching torch's levels:
    // voiced N(0, 0.003), unvoiced N(0, 0.1/3). Torch draws randn (random);
    // we use a seeded PRNG so output is reproducible.
    std::mt19937_64 nrng(0);
    std::normal_distribution<double> ndist(0.0, 1.0);
    double ph[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t t = 0; t < Twave; ++t) {
      const double f0 = fu(0, 0, t);
      const double uv = (f0 > 10.0) ? 1.0 : 0.0;
      const double namp = uv * 0.003 + (1.0 - uv) * (0.1 / 3.0);
      for (int k = 0; k < HDIM; ++k) {
        ph[k] += 2.0 * 3.141592653589793 * f0 * (k + 1) / 24000.0;
        const double nz = ndist(nrng) * namp;
        sh(0, t, k) = (float)(std::sin(ph[k]) * 0.1 * uv + nz);
      }
    }
    Kokkos::deep_copy(sine9, sh);
  }
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> sflat("sf", Twave, HDIM), mflat("mf", Twave, 1);
  Kokkos::parallel_for(
      "sf", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({(size_t)0, (size_t)0}, {Twave, (size_t)HDIM}),
      KOKKOS_LAMBDA(size_t t, size_t k) { sflat(t, k) = sine9(0, t, k); });
  gemm_bias(handle, sflat, w.v2("decoder.generator.m_source.l_linear.weight"), w.v1("decoder.generator.m_source.l_linear.bias"), mflat, "src_lin");
  Kokkos::parallel_for(
      "tanh", Kokkos::RangePolicy<ExecSpace>(0, Twave),
      KOKKOS_LAMBDA(size_t t) { mflat(t, 0) = tanhf(mflat(t, 0)); });
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> har_src("har", 1, Twave);
  Kokkos::parallel_for(
      "har", Kokkos::RangePolicy<ExecSpace>(0, Twave),
      KOKKOS_LAMBDA(size_t t) { har_src(0, t) = mflat(t, 0); });
  const int NF = 20, HOP = 5, FF = NF / 2 + 1;
  const size_t Frames = 1 + Twave / HOP;
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> mag("mag", 1, FF, Frames), pha("pha", 1, FF, Frames);
  stft_forward<ExecSpace, float>(har_src, NF, HOP, mag, pha, "stft");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> har("har2", 1, 2 * FF, Frames);
  Kokkos::parallel_for(
      "har2", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {(size_t)1, (size_t)FF, Frames}),
      KOKKOS_LAMBDA(size_t b, size_t k, size_t f) {
        har(b, k, f) = mag(b, k, f);
        har(b, (size_t)FF + k, f) = pha(b, k, f);
      });
  auto xg = xdec;
  const int dils[3] = {1, 3, 5};
  const size_t ups_rate[2] = {10, 6};
  const size_t ups_pad[2] = {5, 3};
  for (int u = 0; u < 2; ++u) {
    leaky_relu_forward<ExecSpace, float>(xg, xg, 0.1f, "gpre");
    const std::string NZ = "decoder.generator.noise_convs." + std::to_string(u);
    const size_t nCout = w.v3(NZ + ".weight").extent(0);
    const size_t nT = (u == 0) ? 20 * Tf : Frames;
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> xsrc("xsrc", 1, nCout, nT);
    if (u == 0) {
      conv1d_forward<ExecSpace, float>(har, w.v3(NZ + ".weight"), w.v1(NZ + ".bias"), xsrc, 6, 3, 1, 1, "nzc0");
    } else {
      conv1d_forward<ExecSpace, float>(har, w.v3(NZ + ".weight"), w.v1(NZ + ".bias"), xsrc, 1, 0, 1, 1, "nzc1");
    }
    const std::string NR = "decoder.generator.noise_res." + std::to_string(u);
    const size_t NK = (u == 0) ? 7 : 11;
    xsrc = gen_resblk<ExecSpace>(handle, xsrc, style, w, NR, NK, dils);
    const std::string UP = "decoder.generator.ups." + std::to_string(u);
    const size_t Tin = xg.extent(2);
    const size_t Tout = Tin * ups_rate[u];
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> xu("xu", 1, nCout, Tout);
    convtranspose1d_forward<ExecSpace, float>(xg, w.v3(UP + ".weight"), w.v1(UP + ".bias"), xu, ups_rate[u], ups_pad[u], 0, 1, 1, "up");
    if (u == 1) {
      Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> xp("xp", 1, nCout, Tout + 1);
      Kokkos::parallel_for(
          "rpad", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {(size_t)1, nCout, Tout + 1}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t t) {
            xp(b, c, t) = (t == 0) ? xu(b, c, 1) : xu(b, c, t - 1);
          });
      xu = xp;
    }
    Kokkos::parallel_for(
        "gadd", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {(size_t)1, nCout, xu.extent(2)}),
        KOKKOS_LAMBDA(size_t b, size_t c, size_t t) { xu(b, c, t) += xsrc(b, c, t); });
    xg = xu;
    Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> xg_acc("xacc", 1, nCout, xu.extent(2));
    Kokkos::deep_copy(xg_acc, 0.0f);
    for (int k = 0; k < 3; ++k) {
      const std::string RB = "decoder.generator.resblocks." + std::to_string(u * 3 + k);
      const size_t RK = (k == 0) ? 3 : (k == 1 ? 7 : 11);
      auto yk = gen_resblk<ExecSpace>(handle, xu, style, w, RB, RK, dils);
      Kokkos::parallel_for(
          "avg", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {(size_t)1, nCout, yk.extent(2)}),
          KOKKOS_LAMBDA(size_t b, size_t c, size_t t) { xg_acc(b, c, t) += yk(b, c, t); });
    }
    Kokkos::parallel_for(
        "div3", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {(size_t)1, nCout, xg_acc.extent(2)}),
        KOKKOS_LAMBDA(size_t b, size_t c, size_t t) { xg_acc(b, c, t) /= 3.0f; });
    xg = xg_acc;
  }
  // Generator final activation uses torch's default leaky slope (0.01),
  // unlike the 0.1 used inside the upsample loop.
  leaky_relu_forward<ExecSpace, float>(xg, xg, 0.01f, "gpost");
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> post("post", 1, 22, xg.extent(2));
  conv1d_forward<ExecSpace, float>(xg, w.v3("decoder.generator.conv_post.weight"), w.v1("decoder.generator.conv_post.bias"), post, 1, 3, 1, 1, "post");
  const size_t PF = post.extent(2);
  Kokkos::View<float***, Kokkos::LayoutRight, ExecSpace> spec("spec", 1, 11, PF), ph("ph", 1, 11, PF);
  Kokkos::parallel_for(
      "split", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({(size_t)0, (size_t)0, (size_t)0}, {(size_t)1, (size_t)11, PF}),
      KOKKOS_LAMBDA(size_t b, size_t k, size_t f) {
        spec(b, k, f) = expf((float)post(b, k, f));
        ph(b, k, f) = sinf((float)post(b, 11 + k, f));
      });
  const size_t Nout = (PF - 1) * 5;
  Kokkos::View<float**, Kokkos::LayoutRight, ExecSpace> wav("wav", 1, Nout);
  istft_inverse<ExecSpace, float>(spec, ph, NF, HOP, wav, "istft");
  SynthOutput out;
  out.durations = pred_dur;
  out.audio.resize(Nout);
  {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, wav);
    for (size_t i = 0; i < Nout; ++i) out.audio[i] = h(0, i);
  }
  return out;
}

template SynthOutput synthesize<Kokkos::DefaultExecutionSpace>(
    const Config&, const DNNHandle<Kokkos::DefaultExecutionSpace>&,
    const Weights<Kokkos::DefaultExecutionSpace, float>&, const Vocab&,
    const std::string&, const Kokkos::View<float*, Kokkos::DefaultExecutionSpace>&,
    float);

}  // namespace Kokoro
}  // namespace KokkosDNN
