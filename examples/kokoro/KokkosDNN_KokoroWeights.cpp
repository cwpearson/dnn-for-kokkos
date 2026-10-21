#include "KokkosDNN_KokoroWeights.hpp"

#include <cmath>
#include <cstring>
#include <random>

#include "KokkosDNN_KokoroPth.hpp"

namespace KokkosDNN {
namespace Kokoro {

namespace {

template <class ExecSpace, class Scalar>
void fill_random_1(Kokkos::View<Scalar*, ExecSpace>& v,
                   std::mt19937_64& rng) {
  auto h = Kokkos::create_mirror_view(v);
  std::uniform_real_distribution<float> d(-0.1f, 0.1f);
  for (size_t i = 0; i < h.extent(0); ++i) h(i) = static_cast<Scalar>(d(rng));
  Kokkos::deep_copy(v, h);
}

template <class ExecSpace, class Scalar>
void fill_random_2(
    Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>& v,
    std::mt19937_64& rng) {
  auto h = Kokkos::create_mirror_view(v);
  std::uniform_real_distribution<float> d(-0.1f, 0.1f);
  for (size_t i = 0; i < h.extent(0); ++i)
    for (size_t j = 0; j < h.extent(1); ++j)
      h(i, j) = static_cast<Scalar>(d(rng));
  Kokkos::deep_copy(v, h);
}

template <class ExecSpace, class Scalar>
void fill_random_3(
    Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>& v,
    std::mt19937_64& rng) {
  auto h = Kokkos::create_mirror_view(v);
  std::uniform_real_distribution<float> d(-0.1f, 0.1f);
  for (size_t i = 0; i < h.extent(0); ++i)
    for (size_t j = 0; j < h.extent(1); ++j)
      for (size_t k = 0; k < h.extent(2); ++k)
        h(i, j, k) = static_cast<Scalar>(d(rng));
  Kokkos::deep_copy(v, h);
}

}  // namespace

template <class ExecSpace, class Scalar>
void random_init(const Config& cfg, Weights<ExecSpace, Scalar>& w,
                 uint64_t seed) {
  cfg.validate();
  w.v1_.clear();
  w.v2_.clear();
  w.v3_.clear();
  std::mt19937_64 rng(seed);
  const size_t H = (size_t)cfg.hidden_dim, S = (size_t)cfg.style_dim;
  const size_t PH = (size_t)cfg.plbert_hidden_size,
               PI = (size_t)cfg.plbert_intermediate;
  const size_t V = (size_t)cfg.n_token, E = 128;
  const size_t HD = H / 2;
  const auto mk1 = [&](const std::string& n, size_t d) {
    w.v1_[n] = Kokkos::View<Scalar*, ExecSpace>(n, d);
    fill_random_1(w.v1_[n], rng);
  };
  const auto mk2 = [&](const std::string& n, size_t a, size_t b) {
    w.v2_[n] = Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace>(n, a, b);
    fill_random_2(w.v2_[n], rng);
  };
  const auto mk3 = [&](const std::string& n, size_t a, size_t b, size_t c) {
    w.v3_[n] =
        Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>(n, a, b, c);
    fill_random_3(w.v3_[n], rng);
  };
  mk2("bert.embeddings.word_embeddings.weight", V, E);
  mk2("bert.embeddings.position_embeddings.weight", (size_t)cfg.plbert_max_pos, E);
  mk2("bert.embeddings.token_type_embeddings.weight", 2, E);
  mk1("bert.embeddings.LayerNorm.weight", E);
  mk1("bert.embeddings.LayerNorm.bias", E);
  mk2("bert.encoder.embedding_hidden_mapping_in.weight", PH, E);
  mk1("bert.encoder.embedding_hidden_mapping_in.bias", PH);
  const std::string AL = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
  mk2(AL + "attention.query.weight", PH, PH);
  mk1(AL + "attention.query.bias", PH);
  mk2(AL + "attention.key.weight", PH, PH);
  mk1(AL + "attention.key.bias", PH);
  mk2(AL + "attention.value.weight", PH, PH);
  mk1(AL + "attention.value.bias", PH);
  mk2(AL + "attention.dense.weight", PH, PH);
  mk1(AL + "attention.dense.bias", PH);
  mk1(AL + "attention.LayerNorm.weight", PH);
  mk1(AL + "attention.LayerNorm.bias", PH);
  mk2(AL + "ffn.weight", PI, PH);
  mk1(AL + "ffn.bias", PI);
  mk2(AL + "ffn_output.weight", PH, PI);
  mk1(AL + "ffn_output.bias", PH);
  mk1(AL + "full_layer_layer_norm.weight", PH);
  mk1(AL + "full_layer_layer_norm.bias", PH);
  mk2("bert_encoder.weight", H, PH);
  mk1("bert_encoder.bias", H);
  for (int l = 0; l < 3; ++l) {
    const std::string P = "predictor.text_encoder.lstms." + std::to_string(2 * l) + ".";
    mk2(P + "weight_ih_l0", 4 * HD, H + S);
    mk2(P + "weight_hh_l0", 4 * HD, HD);
    mk1(P + "bias_ih_l0", 4 * HD);
    mk1(P + "bias_hh_l0", 4 * HD);
    mk2(P + "weight_ih_l0_reverse", 4 * HD, H + S);
    mk2(P + "weight_hh_l0_reverse", 4 * HD, HD);
    mk1(P + "bias_ih_l0_reverse", 4 * HD);
    mk1(P + "bias_hh_l0_reverse", 4 * HD);
    const std::string F = "predictor.text_encoder.lstms." + std::to_string(2 * l + 1) + ".fc.";
    mk2(F + "weight", 2 * H, S);
    mk1(F + "bias", 2 * H);
  }
  const auto mk_lstm = [&](const std::string& p, size_t In, size_t Hd) {
    mk2(p + ".weight_ih_l0", 4 * Hd, In);
    mk2(p + ".weight_hh_l0", 4 * Hd, Hd);
    mk1(p + ".bias_ih_l0", 4 * Hd);
    mk1(p + ".bias_hh_l0", 4 * Hd);
    mk2(p + ".weight_ih_l0_reverse", 4 * Hd, In);
    mk2(p + ".weight_hh_l0_reverse", 4 * Hd, Hd);
    mk1(p + ".bias_ih_l0_reverse", 4 * Hd);
    mk1(p + ".bias_hh_l0_reverse", 4 * Hd);
  };
  mk_lstm("predictor.lstm", H + S, HD);
  mk2("predictor.duration_proj.linear_layer.weight", (size_t)cfg.max_dur, H);
  mk1("predictor.duration_proj.linear_layer.bias", (size_t)cfg.max_dur);
  mk_lstm("predictor.shared", H + S, HD);
  const auto mk_adain = [&](const std::string& p, size_t Cin, size_t Cout) {
    mk3(p + ".conv1.weight", Cout, Cin, 3);
    mk1(p + ".conv1.bias", Cout);
    mk3(p + ".conv2.weight", Cout, Cout, 3);
    mk1(p + ".conv2.bias", Cout);
    mk2(p + ".norm1.fc.weight", 2 * Cin, S);
    mk1(p + ".norm1.fc.bias", 2 * Cin);
    mk2(p + ".norm2.fc.weight", 2 * Cout, S);
    mk1(p + ".norm2.fc.bias", 2 * Cout);
    if (Cout != Cin) mk3(p + ".conv1x1.weight", Cout, Cin, 1);
  };
  mk_adain("predictor.F0.0", H, H);
  mk_adain("predictor.F0.1", H, HD);
  mk3("predictor.F0.1.pool.weight", H, 1, 3);
  mk1("predictor.F0.1.pool.bias", H);
  mk3("predictor.F0.1.conv1x1.weight", HD, H, 1);
  mk_adain("predictor.F0.2", HD, HD);
  mk_adain("predictor.N.0", H, H);
  mk_adain("predictor.N.1", H, HD);
  mk3("predictor.N.1.pool.weight", H, 1, 3);
  mk1("predictor.N.1.pool.bias", H);
  mk3("predictor.N.1.conv1x1.weight", HD, H, 1);
  mk_adain("predictor.N.2", HD, HD);
  mk3("predictor.F0_proj.weight", 1, HD, 1);
  mk1("predictor.F0_proj.bias", 1);
  mk3("predictor.N_proj.weight", 1, HD, 1);
  mk1("predictor.N_proj.bias", 1);
  mk2("text_encoder.embedding.weight", V, H);
  for (int i = 0; i < 3; ++i) {
    const std::string P = "text_encoder.cnn." + std::to_string(i) + ".";
    mk3(P + "0.weight", H, H, 5);
    mk1(P + "0.bias", H);
    mk1(P + "1.gamma", H);
    mk1(P + "1.beta", H);
  }
  mk_lstm("text_encoder.lstm", H, HD);
  const size_t U = (size_t)cfg.upsample_initial_channel;
  const size_t DW = 2 * U, DL = U, AW = U / 8;
  const size_t ENC_IN = H + 2;
  mk_adain("decoder.encode", ENC_IN, DW);
  mk3("decoder.encode.conv1x1.weight", DW, ENC_IN, 1);
  const size_t DEC_IN = DW + AW + 2;
  for (int i = 0; i < 3; ++i) {
    const std::string P = "decoder.decode." + std::to_string(i);
    mk_adain(P, DEC_IN, DW);
    mk3(P + ".conv1x1.weight", DW, DEC_IN, 1);
  }
  mk_adain("decoder.decode.3", DEC_IN, DL);
  mk3("decoder.decode.3.conv1x1.weight", DL, DEC_IN, 1);
  mk3("decoder.decode.3.pool.weight", DEC_IN, 1, 3);
  mk1("decoder.decode.3.pool.bias", DEC_IN);
  mk3("decoder.F0_conv.weight", 1, 1, 3);
  mk1("decoder.F0_conv.bias", 1);
  mk3("decoder.N_conv.weight", 1, 1, 3);
  mk1("decoder.N_conv.bias", 1);
  mk3("decoder.asr_res.0.weight", AW, H, 1);
  mk1("decoder.asr_res.0.bias", AW);
  mk2("decoder.generator.m_source.l_linear.weight", 1, 9);
  mk1("decoder.generator.m_source.l_linear.bias", 1);
  mk3("decoder.generator.noise_convs.0.weight", U / 2, 22, 12);
  mk1("decoder.generator.noise_convs.0.bias", U / 2);
  mk3("decoder.generator.noise_convs.1.weight", U / 4, 22, 1);
  mk1("decoder.generator.noise_convs.1.bias", U / 4);
  const auto mk_genblk = [&](const std::string& p, size_t C, size_t K) {
    for (int j = 0; j < 3; ++j) {
      const std::string sj = std::to_string(j);
      mk3(p + ".convs1." + sj + ".weight", C, C, K);
      mk1(p + ".convs1." + sj + ".bias", C);
      mk3(p + ".convs2." + sj + ".weight", C, C, K);
      mk1(p + ".convs2." + sj + ".bias", C);
      mk2(p + ".adain1." + sj + ".fc.weight", 2 * C, S);
      mk1(p + ".adain1." + sj + ".fc.bias", 2 * C);
      mk2(p + ".adain2." + sj + ".fc.weight", 2 * C, S);
      mk1(p + ".adain2." + sj + ".fc.bias", 2 * C);
      mk3(p + ".alpha1." + sj, 1, C, 1);
      mk3(p + ".alpha2." + sj, 1, C, 1);
      {
        auto ah = Kokkos::create_mirror_view(w.v3_[p + ".alpha1." + sj]);
        auto bh = Kokkos::create_mirror_view(w.v3_[p + ".alpha2." + sj]);
        for (size_t c = 0; c < C; ++c) {
          ah(0, c, 0) = static_cast<Scalar>(1);
          bh(0, c, 0) = static_cast<Scalar>(1);
        }
        Kokkos::deep_copy(w.v3_[p + ".alpha1." + sj], ah);
        Kokkos::deep_copy(w.v3_[p + ".alpha2." + sj], bh);
      }
      // Snake has 1/alpha: keep random-init alphas at exactly 1 so the
      // tiny smoke test cannot divide by a near-zero random value.
      {
        auto ah = Kokkos::create_mirror_view(w.v3_[p + ".alpha1." + sj]);
        auto bh = Kokkos::create_mirror_view(w.v3_[p + ".alpha2." + sj]);
        for (size_t c = 0; c < C; ++c) {
          ah(0, c, 0) = static_cast<Scalar>(1);
          bh(0, c, 0) = static_cast<Scalar>(1);
        }
        Kokkos::deep_copy(w.v3_[p + ".alpha1." + sj], ah);
        Kokkos::deep_copy(w.v3_[p + ".alpha2." + sj], bh);
      }
    }
  };
  mk_genblk("decoder.generator.noise_res.0", U / 2, 7);
  mk_genblk("decoder.generator.noise_res.1", U / 4, 11);
  mk3("decoder.generator.ups.0.weight", U, U / 2, 20);
  mk1("decoder.generator.ups.0.bias", U / 2);
  mk3("decoder.generator.ups.1.weight", U / 2, U / 4, 12);
  mk1("decoder.generator.ups.1.bias", U / 4);
  const int rks[3] = {3, 7, 11};
  for (int u = 0; u < 2; ++u) {
    const size_t C = (u == 0) ? U / 2 : U / 4;
    for (int k = 0; k < 3; ++k)
      mk_genblk("decoder.generator.resblocks." + std::to_string(u * 3 + k),
                C, (size_t)rks[k]);
  }
  mk3("decoder.generator.conv_post.weight", 22, U / 4, 7);
  mk1("decoder.generator.conv_post.bias", 22);
}

template <class ExecSpace, class Scalar>
void load_kokoro_pth(const Config& cfg, const std::string& pth_path,
                      Weights<ExecSpace, Scalar>& w) {
  cfg.validate();
  KokoroArchive ar = open_kokoro_pth(pth_path);
  std::map<std::string, KokoroTensorMeta> by_name;
  for (const auto& t : ar.tensors) by_name[t.name] = t;
  std::map<int, std::vector<float>> storage_cache;
  const auto get_storage = [&](int sid) -> const std::vector<float>& {
    auto it = storage_cache.find(sid);
    if (it != storage_cache.end()) return it->second;
    std::vector<float> v = read_kokoro_storage(ar, sid);
    storage_cache[sid] = std::move(v);
    return storage_cache[sid];
  };
  w.v1_.clear();
  w.v2_.clear();
  w.v3_.clear();
  std::map<std::string, bool> fused_done;
  for (const auto& t : ar.tensors) {
    const std::string& n = t.name;
    if (n.size() >= 9 && n.compare(n.size() - 9, 9, ".weight_g") == 0) {
      const std::string base = n.substr(0, n.size() - 9);
      if (fused_done.count(base)) continue;
      fused_done[base] = true;
      auto git = by_name.find(base + ".weight_g");
      auto vit = by_name.find(base + ".weight_v");
      if (git == by_name.end() || vit == by_name.end())
        throw std::runtime_error("Kokoro load: missing norm pair for " + base);
      const auto& gm = git->second, vm = vit->second;
      const std::vector<float>& gd = get_storage(gm.storage_id);
      const std::vector<float>& vd = get_storage(vm.storage_id);
      if (gm.shape.size() != 3 || vm.shape.size() != 3)
        throw std::runtime_error("Kokoro load: bad norm shapes for " + base);
      const size_t O = (size_t)vm.shape[0], I = (size_t)vm.shape[1],
                   K = (size_t)vm.shape[2];
      std::vector<float> fused(O * I * K);
      for (size_t o = 0; o < O; ++o) {
        const float g = gd[(size_t)gm.storage_offset + o];
        double nrm = 0;
        for (size_t i = 0; i < I; ++i)
          for (size_t k = 0; k < K; ++k) {
            const float v = vd[(size_t)vm.storage_offset + (o * I + i) * K + k];
            nrm += (double)v * (double)v;
          }
        nrm = std::sqrt(nrm) + 1e-12;
        const float s = g / (float)nrm;
        for (size_t i = 0; i < I; ++i)
          for (size_t k = 0; k < K; ++k)
            fused[(o * I + i) * K + k] =
                s * vd[(size_t)vm.storage_offset + (o * I + i) * K + k];
      }
      Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> vw(base, O, I, K);
      auto h = Kokkos::create_mirror_view(vw);
      for (size_t o = 0; o < O; ++o)
        for (size_t i = 0; i < I; ++i)
          for (size_t k = 0; k < K; ++k)
            h(o, i, k) = static_cast<Scalar>(fused[(o * I + i) * K + k]);
      Kokkos::deep_copy(vw, h);
      w.v3_[base + ".weight"] = vw;
    }
  }
  for (const auto& t : ar.tensors) {
    const std::string& n = t.name;
    if (n.size() >= 9 && n.compare(n.size() - 9, 9, ".weight_g") == 0) continue;
    if (n.size() >= 9 && n.compare(n.size() - 9, 9, ".weight_v") == 0) continue;
    const std::vector<float>& d = get_storage(t.storage_id);
    const size_t off = (size_t)t.storage_offset;
    if (t.shape.size() == 1) {
      const size_t D = (size_t)t.shape[0];
      Kokkos::View<Scalar*, ExecSpace> v(n, D);
      auto h = Kokkos::create_mirror_view(v);
      for (size_t i = 0; i < D; ++i) h(i) = static_cast<Scalar>(d[off + i]);
      Kokkos::deep_copy(v, h);
      w.v1_[n] = v;
    } else if (t.shape.size() == 2) {
      const size_t A = (size_t)t.shape[0], Bn = (size_t)t.shape[1];
      Kokkos::View<Scalar**, Kokkos::LayoutRight, ExecSpace> v(n, A, Bn);
      auto h = Kokkos::create_mirror_view(v);
      for (size_t i = 0; i < A; ++i)
        for (size_t j = 0; j < Bn; ++j)
          h(i, j) = static_cast<Scalar>(d[off + i * Bn + j]);
      Kokkos::deep_copy(v, h);
      w.v2_[n] = v;
    } else if (t.shape.size() == 3) {
      const size_t A = (size_t)t.shape[0], Bn = (size_t)t.shape[1],
                   C = (size_t)t.shape[2];
      Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace> v(n, A, Bn, C);
      auto h = Kokkos::create_mirror_view(v);
      for (size_t i = 0; i < A; ++i)
        for (size_t j = 0; j < Bn; ++j)
          for (size_t k = 0; k < C; ++k)
            h(i, j, k) = static_cast<Scalar>(d[off + (i * Bn + j) * C + k]);
      Kokkos::deep_copy(v, h);
      w.v3_[n] = v;
    } else {
      throw std::runtime_error("Kokoro load: unsupported rank for '" + n + "'");
    }
  }
}

template <class ExecSpace, class Scalar>
void load_voice_row(const std::string& voice_path, int idx,
                    Kokkos::View<Scalar*, ExecSpace>& ref_s) {
  KokoroArchive ar = open_voice_pt(voice_path);
  if (ar.tensors.size() != 1)
    throw std::runtime_error("Kokoro voice: expected 1 tensor in " + voice_path);
  const auto& m = ar.tensors[0];
  if (m.shape.size() != 3 || m.shape[1] != 1 || m.shape[2] != 256)
    throw std::runtime_error("Kokoro voice: expected [N,1,256] in " + voice_path);
  const size_t N = (size_t)m.shape[0];
  if (N == 0) throw std::runtime_error("Kokoro voice: empty pack");
  int pick = idx;
  if (pick < 0) pick = 0;
  if ((size_t)pick >= N) pick = (int)N - 1;
  std::vector<float> d = read_kokoro_storage(ar, m.storage_id);
  const size_t off = (size_t)m.storage_offset + (size_t)pick * 256;
  ref_s = Kokkos::View<Scalar*, ExecSpace>("ref_s", 256);
  auto h = Kokkos::create_mirror_view(ref_s);
  for (size_t i = 0; i < 256; ++i) h(i) = static_cast<Scalar>(d[off + i]);
  Kokkos::deep_copy(ref_s, h);
}

template void random_init<Kokkos::DefaultExecutionSpace, float>(
    const Config&, Weights<Kokkos::DefaultExecutionSpace, float>&, uint64_t);
template void load_kokoro_pth<Kokkos::DefaultExecutionSpace, float>(
    const Config&, const std::string&,
    Weights<Kokkos::DefaultExecutionSpace, float>&);
template void load_voice_row<Kokkos::DefaultExecutionSpace, float>(
    const std::string&, int,
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace>&);

}  // namespace Kokoro
}  // namespace KokkosDNN
