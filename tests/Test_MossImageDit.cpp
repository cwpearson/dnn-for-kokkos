// TRUE MOSS-Image DiT: scheduler goldens, safetensors loader roundtrip,
// forward smoke, skip-layer identity, bhalf smoke.
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "KokkosDNN.hpp"

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using bhalf = Kokkos::Experimental::bhalf_t;
using Dit = KokkosDNN::MossImageDit::Config;

// ---- scheduler -----------------------------------------------------------

TEST(MossDitScheduler, HardcodedGoldens) {
  // Values from an independent pure-python mirror of
  // MOSSImageScheduler::set_timesteps (grid + lower_bound snap +
  // unique_consecutive); endpoints hit the training grid exactly.
  {
    auto s = KokkosDNN::MossImageDit::moss_flow_grid_timesteps(5, 5.0, 1.0,
                                                               1000);
    const std::vector<double> want{1.0, 0.938, 0.835, 0.63, 0.024};
    ASSERT_EQ(s.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i)
      EXPECT_NEAR(s[i], want[i], 1e-9) << "i=" << i;
  }
  {
    auto s =
        KokkosDNN::MossImageDit::moss_flow_grid_timesteps(8, 5.0, 1.0, 100);
    const std::vector<double> want{1.0, 0.97, 0.93, 0.88,
                                   0.81, 0.7,  0.53, 0.2};
    ASSERT_EQ(s.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i)
      EXPECT_NEAR(s[i], want[i], 1e-9) << "i=" << i;
  }
  // Single step collapses to sigma 1.
  {
    auto s = KokkosDNN::MossImageDit::moss_flow_grid_timesteps(1, 5.0, 1.0,
                                                               1000);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_DOUBLE_EQ(s[0], 1.0);
  }
  EXPECT_THROW(KokkosDNN::MossImageDit::moss_flow_grid_timesteps(0, 5.0, 1.0,
                                                                 1000),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::MossImageDit::moss_flow_grid_timesteps(5, 0.0, 1.0,
                                                                 1000),
               std::invalid_argument);
  EXPECT_THROW(KokkosDNN::MossImageDit::moss_flow_grid_timesteps(5, 5.0, -1.0,
                                                                 1000),
               std::invalid_argument);
}

TEST(MossDitScheduler, InTestRecomputeAndMonotone) {
  // Independent re-derivation of the same formula inside the test.
  const int steps = 12, total = 1000;
  const double shift = 5.0, tsh = 1.0;
  auto grid = [&](int count, double s) {
    const double lower = s / (total + s - 1.0);
    std::vector<double> g((size_t)count);
    for (int i = 0; i < count; ++i) {
      const double v =
          (count == 1) ? 1.0 : 1.0 + (lower - 1.0) * i / (count - 1);
      g[(size_t)i] = s * v / (1.0 + (s - 1.0) * v);
    }
    return g;
  };
  std::vector<double> sig = grid(steps, shift), tr = grid(total, tsh);
  std::reverse(tr.begin(), tr.end());
  std::vector<double> want;
  for (double sg : sig) {
    size_t u = 0;
    while (u + 1 < tr.size() && tr[u + 1] < sg) ++u;  // linear lower_bound
    if (u >= tr.size()) u = tr.size() - 1;
    size_t l = (u > 0) ? u - 1 : 0;
    // Snap: linear scan for the true nearest (ties -> lower index).
    size_t best = 0;
    for (size_t k = 0; k < tr.size(); ++k) {
      double db = std::fabs(sg - tr[k]), dn = std::fabs(sg - tr[best]);
      if (db < dn || (db == dn && k < best)) best = k;
    }
    (void)l;
    (void)u;
    if (want.empty() || tr[best] != want.back()) want.push_back(tr[best]);
  }
  auto got =
      KokkosDNN::MossImageDit::moss_flow_grid_timesteps(steps, shift, tsh,
                                                        total);
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < want.size(); ++i)
    EXPECT_NEAR(got[i], want[i], 1e-12) << "i=" << i;
  for (size_t i = 1; i < got.size(); ++i)
    EXPECT_LE(got[i], got[i - 1]) << "must be non-increasing at " << i;
}

// ---- synthetic safetensors writer (test-local, F32/BF16) ------------------

struct SynthTensor {
  std::string name, dtype;  // "F32" or "BF16"
  std::vector<int64_t> shape;
  std::vector<float> vals;  // host float values (converted on write)
};

void write_synth_safetensors(const std::string& path,
                             const std::vector<SynthTensor>& ts) {
  std::string js = "{";
  std::vector<char> payload;
  for (size_t i = 0; i < ts.size(); ++i) {
    const auto& t = ts[i];
    size_t n = 1;
    for (auto d : t.shape) n *= (size_t)d;
    if (t.vals.size() != n) throw std::logic_error("bad synth tensor");
    const size_t el = (t.dtype == "F32") ? 4 : 2;
    if (i) js += ",";
    js += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t k = 0; k < t.shape.size(); ++k) {
      if (k) js += ",";
      js += std::to_string(t.shape[k]);
    }
    js += "],\"data_offsets\":[" + std::to_string(payload.size()) + "," +
          std::to_string(payload.size() + n * el) + "]}";
    size_t base = payload.size();
    payload.resize(base + n * el);
    if (t.dtype == "F32") {
      std::memcpy(payload.data() + base, t.vals.data(), n * 4);
    } else {
      uint16_t* dst = reinterpret_cast<uint16_t*>(payload.data() + base);
      for (size_t k = 0; k < n; ++k)
        dst[k] =
            KokkosDNN::MossImageDit::Impl::dit_float_to_bf16(t.vals[k]);
    }
  }
  js += "}";
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t hn = js.size();
  char lb[8];
  for (int i = 0; i < 8; ++i) lb[i] = (char)((hn >> (8 * i)) & 0xFF);
  f.write(lb, 8);
  f.write(js.data(), (std::streamsize)js.size());
  f.write(payload.data(), (std::streamsize)payload.size());
  f.flush();
  if (!f) throw std::runtime_error("write failed " + path);
}

std::string dit_tmp_path(const char* tag) {
  return "/tmp/mossdit_" + std::string(tag) + "_" +
         std::to_string((long long)::getpid()) + ".safetensors";
}

// Deterministic bf16-exact values (multiples of 0.25, |v| < 16).
std::vector<float> dit_synth_vals(size_t n, size_t t) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = (float)(((i * 7 + t * 13) % 64)) * 0.25f - 4.0f;
  return v;
}

// Read back one rank-1 / rank-2 view to host floats for comparison.
template <class View>
std::vector<float> to_floats(const View& v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  std::vector<float> o(v.size());
  if constexpr (View::rank == 1) {
    for (size_t i = 0; i < v.extent(0); ++i)
      o[i] = static_cast<float>(h(i));
  } else {
    size_t n = 0;
    for (size_t i = 0; i < v.extent(0); ++i)
      for (size_t j = 0; j < v.extent(1); ++j)
        o[n++] = static_cast<float>(h(i, j));
  }
  return o;
}

TEST(MossDitLoader, RoundTripTiny) {
  auto cfg = Dit::tiny_test();
  ASSERT_NO_THROW(cfg.validate());
  auto specs = KokkosDNN::MossImageDit::moss_dit_expected_tensors(cfg);
  // Real layout has 462 weight tensors (+ rope.bands = 463 on disk).
  EXPECT_EQ(specs.size(), 62u);
  std::vector<SynthTensor> ts;
  for (size_t k = 0; k < specs.size(); ++k) {
    const auto& s = specs[k];
    size_t n = s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
    std::vector<int64_t> shape =
        s.matrix ? std::vector<int64_t>{s.d0, s.d1} : std::vector<int64_t>{s.d0};
    ts.push_back({"transformer." + s.name, (k % 2) ? "BF16" : "F32", shape,
                  dit_synth_vals(n, k)});
  }
  // rope.bands must be skipped, not rejected.
  ts.push_back({"rope.bands", "F32", {1}, {1.0f}});
  const std::string path = dit_tmp_path("tiny");
  write_synth_safetensors(path, ts);

  // Float weights.
  {
    KokkosDNN::MossImageDit::MossImageDitWeights<Exec, float> w;
    w.allocate(cfg);
    EXPECT_NO_THROW(w.load_safetensors(path));
    EXPECT_GT(w.resident_bytes(), 0u);
    // Spot-check every tensor through the public views.
    size_t checked = 0;
    auto chk1 = [&](const auto& v, size_t k, const std::string& what) {
      auto got = to_floats(v);
      ASSERT_EQ(got.size(), ts[k].vals.size()) << what;
      for (size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], ts[k].vals[i], 1e-6) << what << " i=" << i;
      ++checked;
    };
    // Rebuild the spec->tensor index the writer used (same order).
    for (size_t k = 0; k < specs.size(); ++k) {
      const auto& s = specs[k];
      const std::string& n = s.name;
      if (n == "video_embedder.weight")
        chk1(w.video_W, k, n);
      else if (n == "video_embedder.bias")
        chk1(w.video_b, k, n);
      else if (n == "text_embedder.proj.weight")
        chk1(w.text_proj_W, k, n);
      else if (n == "text_embedder.proj.bias")
        chk1(w.text_proj_b, k, n);
      else if (n == "time_embedding.0.weight")
        chk1(w.time0_W, k, n);
      else if (n == "time_embedding.0.bias")
        chk1(w.time0_b, k, n);
      else if (n == "time_embedding.2.weight")
        chk1(w.time2_W, k, n);
      else if (n == "time_embedding.2.bias")
        chk1(w.time2_b, k, n);
      else if (n == "shared_block_modulation.weight")
        chk1(w.shared_W, k, n);
      else if (n == "shared_block_modulation.bias")
        chk1(w.shared_b, k, n);
      else if (n == "video_head_modulation.weight")
        chk1(w.head_W, k, n);
      else if (n == "video_head_modulation.bias")
        chk1(w.head_b, k, n);
      else if (n == "final_norm_video.weight")
        chk1(w.final_norm, k, n);
      else if (n == "final_linear_video.weight")
        chk1(w.final_W, k, n);
      else if (n.rfind("text_embedder.blocks.", 0) == 0 ||
               n.rfind("layers.", 0) == 0) {
        // Structural presence: resolved by the loader without error; value
        // coverage below samples one block of each kind explicitly.
        ++checked;
      } else {
        FAIL() << "unhandled spec " << n;
      }
    }
    // Explicit value checks for one text block + one main block.
    {
      size_t k = 0;
      for (; k < specs.size(); ++k)
        if (specs[k].name == "text_embedder.blocks.0.qkv.weight") break;
      ASSERT_LT(k, specs.size());
      auto got = to_floats(w.text_blocks[0].qkv_W);
      ASSERT_EQ(got.size(), ts[k].vals.size());
      for (size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], ts[k].vals[i], 1e-6);
      for (; k < specs.size(); ++k)
        if (specs[k].name == "layers.1.proj.ffn_up_gate.weight") break;
      ASSERT_LT(k, specs.size());
      auto got2 = to_floats(w.blocks[1].up_gate_W);
      ASSERT_EQ(got2.size(), ts[k].vals.size());
      for (size_t i = 0; i < got2.size(); ++i)
        EXPECT_NEAR(got2[i], ts[k].vals[i], 1e-6);
      checked += 2;
    }
    EXPECT_EQ(checked, specs.size() + 2);
    w.release();
    EXPECT_EQ(w.resident_bytes(), 0u);
  }

  // BFloat16 weights from the same mixed-dtype file.
  {
    KokkosDNN::MossImageDit::MossImageDitWeights<Exec, bhalf> w;
    w.allocate(cfg);
    EXPECT_NO_THROW(w.load_safetensors(path));
    auto got = to_floats(w.video_W);
    size_t k = 0;
    for (; k < specs.size(); ++k)
      if (specs[k].name == "video_embedder.weight") break;
    ASSERT_LT(k, specs.size());
    ASSERT_EQ(got.size(), ts[k].vals.size());
    for (size_t i = 0; i < got.size(); ++i)
      EXPECT_NEAR(got[i], ts[k].vals[i], 1e-3) << "i=" << i;
    auto gotb = to_floats(w.blocks[1].up_gate_W);
    size_t k2 = 0;
    for (; k2 < specs.size(); ++k2)
      if (specs[k2].name == "layers.1.proj.ffn_up_gate.weight") break;
    ASSERT_LT(k2, specs.size());
    for (size_t i = 0; i < gotb.size(); ++i)
      EXPECT_NEAR(gotb[i], ts[k2].vals[i], 1e-3) << "i=" << i;
  }
  std::remove(path.c_str());
}

TEST(MossDitLoader, Errors) {
  auto cfg = Dit::tiny_test();
  auto specs = KokkosDNN::MossImageDit::moss_dit_expected_tensors(cfg);
  KokkosDNN::MossImageDit::MossImageDitWeights<Exec, float> w;
  w.allocate(cfg);
  EXPECT_THROW(w.load_safetensors("/nonexistent_xyz.safetensors"),
               std::runtime_error);
  EXPECT_THROW(w.load_safetensors(std::vector<std::string>{}),
               std::invalid_argument);
  // Missing tensor: file with only the first spec.
  {
    size_t n = (size_t)specs[0].d0 * (size_t)specs[0].d1;
    std::vector<SynthTensor> ts{
        {specs[0].name, "F32", {specs[0].d0, specs[0].d1},
         dit_synth_vals(n, 0)}};
    const std::string p = dit_tmp_path("miss");
    write_synth_safetensors(p, ts);
    EXPECT_THROW(w.load_safetensors(p), std::runtime_error);
    std::remove(p.c_str());
  }
  // Wrong shape on one tensor.
  {
    std::vector<SynthTensor> ts;
    for (size_t k = 0; k < specs.size(); ++k) {
      const auto& s = specs[k];
      size_t n = s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
      std::vector<int64_t> shape = s.matrix
                                       ? std::vector<int64_t>{s.d0, s.d1}
                                       : std::vector<int64_t>{s.d0};
      if (k == 3) shape.back() += 1;  // corrupt one entry's shape
      size_t nn = 1;
      for (auto d : shape) nn *= (size_t)d;
      ts.push_back({s.name, "F32", shape, dit_synth_vals(nn, k)});
    }
    const std::string p = dit_tmp_path("badshape");
    write_synth_safetensors(p, ts);
    EXPECT_THROW(w.load_safetensors(p), std::runtime_error);
    std::remove(p.c_str());
  }
  // Unknown tensor name is rejected (strict, no silent layout drift).
  {
    std::vector<SynthTensor> ts;
    for (size_t k = 0; k < specs.size(); ++k) {
      const auto& s = specs[k];
      size_t n = s.matrix ? (size_t)s.d0 * (size_t)s.d1 : (size_t)s.d0;
      std::vector<int64_t> shape = s.matrix
                                       ? std::vector<int64_t>{s.d0, s.d1}
                                       : std::vector<int64_t>{s.d0};
      ts.push_back({s.name, "F32", shape, dit_synth_vals(n, k)});
    }
    ts.push_back({"some_random_tier.weight", "F32", {4}, {1, 2, 3, 4}});
    const std::string p = dit_tmp_path("unknown");
    write_synth_safetensors(p, ts);
    EXPECT_THROW(w.load_safetensors(p), std::runtime_error);
    std::remove(p.c_str());
  }
  // Real-config spec count is 462 weights (+ rope.bands = 463 on disk).
  {
    auto real = KokkosDNN::MossImageDit::moss_dit_expected_tensors(
        Dit::moss_image_sft());
    EXPECT_EQ(real.size(), 462u);
    // Byte-exact cross-check against the HF weight-map total_size
    // (10164617280, verified in the fetched transformer index): all DiT
    // tensors are bf16 except fp32 rope.bands [16] (16*4 = 64 bytes).
    int64_t numel = 0;
    for (const auto& s : real)
      numel += s.matrix ? s.d0 * s.d1 : s.d0;
    EXPECT_EQ(numel * 2 + 64, 10164617280LL);
  }
}

// ---- forward --------------------------------------------------------------

template <class Scalar>
Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> dit_rand4(int B, int C,
                                                              int H, int W,
                                                              float lo,
                                                              float hi,
                                                              uint64_t& s) {
  Kokkos::View<Scalar****, Kokkos::LayoutRight, Exec> v("t", B, C, H, W);
  auto h = Kokkos::create_mirror_view(v);
  uint64_t x = s;
  auto rnd = [&]() {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return lo + (hi - lo) * ((x >> 11) * (1.0f / 9007199254740992.0f));
  };
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) h(b, c, i, j) = (Scalar)rnd();
  s = x;
  Kokkos::deep_copy(v, h);
  return v;
}

TEST(MossDitForward, SmokeAndDeterminism) {
  auto cfg = Dit::tiny_test();  // dim 32, heads 4, layers 2, ffn 64
  KokkosDNN::MossImageDit::MossImageDitModel<Exec, float> model(cfg);
  model.weights().random_init(11);
  EXPECT_GT(model.weights().resident_bytes(), 0u);
  const int B = 1, H = 2, W = 2, S = 6;
  uint64_t s = 1234;
  auto lat = dit_rand4<float>(B, cfg.latent_channels, H, W, -0.5f, 0.5f, s);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> enc("e", B, S,
                                                        cfg.text_dim);
  {
    auto h = Kokkos::create_mirror_view(enc);
    for (int i = 0; i < S; ++i)
      for (int j = 0; j < cfg.text_dim; ++j)
        h(0, i, j) = 0.1f * (i + 1) - 0.05f * j;
    Kokkos::deep_copy(enc, h);
  }
  Kokkos::View<unsigned char**, Kokkos::LayoutRight, Exec> mask("m", B, S);
  Kokkos::deep_copy(mask, (unsigned char)1);
  Kokkos::View<float*, Exec> t("t", B);
  Kokkos::deep_copy(t, 0.5f);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> out("o", B,
                                                         cfg.latent_channels,
                                                         H, W);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> out2("o2", B,
                                                          cfg.latent_channels,
                                                          H, W);
  EXPECT_NO_THROW(model.forward(lat, t, enc, mask, out));
  Kokkos::fence();
  EXPECT_GT(model.scratch_bytes(), 0u);
  auto ho = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(ho, out);
  for (int c = 0; c < cfg.latent_channels; ++c)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        EXPECT_TRUE(std::isfinite(ho(0, c, i, j))) << c << i << j;
  EXPECT_NO_THROW(model.forward(lat, t, enc, mask, out2));
  Kokkos::fence();
  auto ho2 = Kokkos::create_mirror_view(out2);
  Kokkos::deep_copy(ho2, out2);
  for (int c = 0; c < cfg.latent_channels; ++c)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        EXPECT_FLOAT_EQ(ho2(0, c, i, j), ho(0, c, i, j));
  // Bad geometry / mask / skips throw.
  EXPECT_THROW(model.forward(lat, t, enc, mask, out, {2}),
               std::invalid_argument);
  Kokkos::View<unsigned char**, Kokkos::LayoutRight, Exec> badmask("bm", B, S);
  Kokkos::deep_copy(badmask, (unsigned char)0);
  EXPECT_THROW(model.forward(lat, t, enc, badmask, out),
               std::invalid_argument);
}

TEST(MossDitForward, SkipLayersAreIdentity) {
  auto cfg = Dit::tiny_test();
  KokkosDNN::MossImageDit::MossImageDitModel<Exec, float> model(cfg);
  model.weights().random_init(7);
  auto& W = model.weights();
  const int B = 1, H = 2, Wd = 2, S = 5, HW = H * Wd;
  const int D = cfg.dim, C = cfg.latent_channels, Fr = cfg.freq_dim;
  uint64_t s = 999;
  auto lat = dit_rand4<float>(B, C, H, Wd, -0.5f, 0.5f, s);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> enc("e", B, S,
                                                        cfg.text_dim);
  {
    auto h = Kokkos::create_mirror_view(enc);
    for (int i = 0; i < S; ++i)
      for (int j = 0; j < cfg.text_dim; ++j)
        h(0, i, j) = 0.07f * (i + 1) + 0.03f * j - 0.2f;
    Kokkos::deep_copy(enc, h);
  }
  Kokkos::View<unsigned char**, Kokkos::LayoutRight, Exec> mask("m", B, S);
  Kokkos::deep_copy(mask, (unsigned char)1);
  Kokkos::View<float*, Exec> t("t", B);
  Kokkos::deep_copy(t, 0.7f);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> out_all("oa", B, C, H,
                                                             Wd);
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> out_none("on", B, C, H,
                                                              Wd);
  model.forward(lat, t, enc, mask, out_all, {0, 1});
  model.forward(lat, t, enc, mask, out_none, {});
  Kokkos::fence();
  auto ha = Kokkos::create_mirror_view(out_all);
  Kokkos::deep_copy(ha, out_all);
  auto hn = Kokkos::create_mirror_view(out_none);
  Kokkos::deep_copy(hn, out_none);
  // Skipping nothing vs everything must differ (random gates non-degenerate).
  double maxd = 0;
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < Wd; ++j)
        maxd = std::max(maxd, (double)std::fabs(hn(0, c, i, j) - ha(0, c, i, j)));
  EXPECT_GT(maxd, 1e-4) << "blocks look degenerate";

  // Manual head-only path (video slice passes skipped blocks unchanged):
  // video_embedder -> final RMSNorm -> head adaLN -> final Linear.
  KokkosDNN::DNNHandle<Exec> handle;
  using V2 = Kokkos::View<float**, Kokkos::LayoutRight, Exec>;
  // Time: same float64 host sinusoid the model uses.
  V2 temb("te", B, Fr);
  {
    auto h = Kokkos::create_mirror_view(temb);
    for (int j = 0; j < Fr / 2; ++j) {
      double freq = std::pow(10000.0, -(double)j / (Fr / 2));
      double ph = 0.7 * freq;
      h(0, j) = (float)std::cos(ph);
      h(0, j + Fr / 2) = (float)std::sin(ph);
    }
    Kokkos::deep_copy(temb, h);
  }
  V2 h1("h1", B, D), h2("h2", B, D), m2("m2", B, 2 * D);
  KokkosDNN::gemm_bias<Exec, float>(handle, temb, W.time0_W, W.time0_b, h1,
                                    "t0");
  KokkosDNN::silu_forward<Exec, float>(h1, h1, "silu");
  KokkosDNN::gemm_bias<Exec, float>(handle, h1, W.time2_W, W.time2_b, h2,
                                    "t2");
  KokkosDNN::gemm_bias<Exec, float>(handle, h2, W.head_W, W.head_b, m2,
                                    "hm");
  V2 hdsh("hs", B, D), hdsc("hc", B, D);
  KokkosDNN::MossImageDit::Impl::dit_split2<Exec, float>(m2, hdsh, hdsc,
                                                         "sp");
  V2 vid_in("vi", B * HW, C), vid("v", B * HW, D);
  KokkosDNN::MossImageDit::Impl::dit_transpose_bchw_to_rows<Exec, float>(
      lat, vid_in, "tr");
  KokkosDNN::gemm_bias<Exec, float>(handle, vid_in, W.video_W, W.video_b, vid,
                                    "ve");
  V2 nr("nr", HW, D), fout("fo", HW, C);
  auto vid0 = Kokkos::subview(vid, std::pair<size_t, size_t>(0, HW),
                              Kokkos::ALL());
  KokkosDNN::rmsnorm<Exec, float>(vid0, W.final_norm, 1e-6f, nr, "fn");
  auto shb = Kokkos::subview(hdsh, 0, Kokkos::ALL());
  auto scb = Kokkos::subview(hdsc, 0, Kokkos::ALL());
  KokkosDNN::adaln_modulate_forward<Exec, float>(nr, shb, scb, nr, "fm");
  KokkosDNN::gemm<Exec, float>(handle, 1.0f, nr, W.final_W, 0.0f, fout, "fl");
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> want("w", B, C, H, Wd);
  auto want0 = Kokkos::subview(want, 0, Kokkos::ALL(), Kokkos::ALL(),
                               Kokkos::ALL());
  KokkosDNN::MossImageDit::Impl::dit_transpose_rows_to_chw<Exec, float>(
      fout, H, Wd, want0, "tro");
  Kokkos::fence();
  auto hw = Kokkos::create_mirror_view(want);
  Kokkos::deep_copy(hw, want);
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < Wd; ++j)
        EXPECT_NEAR(ha(0, c, i, j), hw(0, c, i, j), 1e-4)
            << "skip-all != head-only at (" << c << "," << i << "," << j
            << ")";
}

TEST(MossDitForward, BatchSamplesAreIndependent) {
  auto cfg = Dit::tiny_test();
  KokkosDNN::MossImageDit::MossImageDitModel<Exec, float> model(cfg);
  model.weights().random_init(3);
  const int H = 2, W = 2, S = 6;
  const int C = cfg.latent_channels;
  // B=2 with different valid lengths (6 vs 3) and timesteps.
  uint64_t s = 555;
  auto lat2 = dit_rand4<float>(2, C, H, W, -0.5f, 0.5f, s);
  Kokkos::View<float***, Kokkos::LayoutRight, Exec> enc2("e2", 2, S,
                                                         cfg.text_dim);
  {
    auto h = Kokkos::create_mirror_view(enc2);
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < S; ++i)
        for (int j = 0; j < cfg.text_dim; ++j)
          h(b, i, j) = 0.05f * (b + 1) * (i + 1) - 0.02f * j;
    Kokkos::deep_copy(enc2, h);
  }
  Kokkos::View<unsigned char**, Kokkos::LayoutRight, Exec> mask2("m2", 2, S);
  {
    auto h = Kokkos::create_mirror_view(mask2);
    for (int i = 0; i < S; ++i) h(0, i) = 1;
    for (int i = 0; i < S; ++i) h(1, i) = (unsigned char)(i < 3 ? 1 : 0);
    Kokkos::deep_copy(mask2, h);
  }
  Kokkos::View<float*, Exec> t2("t2", 2);
  {
    auto h = Kokkos::create_mirror_view(t2);
    h(0) = 0.4f;
    h(1) = 0.8f;
    Kokkos::deep_copy(t2, h);
  }
  Kokkos::View<float****, Kokkos::LayoutRight, Exec> out2("o2", 2, C, H, W);
  model.forward(lat2, t2, enc2, mask2, out2);
  Kokkos::fence();
  // Same samples run as B=1 must match to FP tolerance (no cross-sample
  // leakage). This is EXPECT_NEAR, not EXPECT_FLOAT_EQ, by construction:
  // the time-path and video-embedder GEMMs run batched ([B,...]/[B*HW,...])
  // in the B=2 call but single-sample ([1,...]/[HW,...]) in the B=1 calls.
  // cuBLAS selects kernels / accumulation order per shape, so the two shapes
  // round differently by a few ulps even for the same sample row (cuBLAS
  // documents shape-dependent rounding; the ~8-ulp diff seen on CUDA+TPL is
  // appropriate TPL numerics, not leakage, which would be O(1e-2+) given
  // the samples' different timesteps/masks/encodings). The native Kokkos
  // GEMM is per-row deterministic, so Serial stays bit-exact well inside
  // this bound.
  for (int b = 0; b < 2; ++b) {
    Kokkos::View<float****, Kokkos::LayoutRight, Exec> lat1("l1", 1, C, H, W);
    auto sl = Kokkos::subview(lat2, b, Kokkos::ALL(), Kokkos::ALL(),
                              Kokkos::ALL());
    auto dl = Kokkos::subview(lat1, 0, Kokkos::ALL(), Kokkos::ALL(),
                              Kokkos::ALL());
    Kokkos::deep_copy(dl, sl);
    Kokkos::View<float***, Kokkos::LayoutRight, Exec> enc1("e1", 1, S,
                                                           cfg.text_dim);
    auto se = Kokkos::subview(enc2, b, Kokkos::ALL(), Kokkos::ALL());
    auto de = Kokkos::subview(enc1, 0, Kokkos::ALL(), Kokkos::ALL());
    Kokkos::deep_copy(de, se);
    Kokkos::View<unsigned char**, Kokkos::LayoutRight, Exec> mask1("m1", 1, S);
    auto sm = Kokkos::subview(mask2, b, Kokkos::ALL());
    auto dm = Kokkos::subview(mask1, 0, Kokkos::ALL());
    Kokkos::deep_copy(dm, sm);
    Kokkos::View<float*, Exec> t1("t1", 1);
    {
      auto h = Kokkos::create_mirror_view(t2);
      Kokkos::deep_copy(h, t2);
      Kokkos::deep_copy(t1, h(b));
    }
    Kokkos::View<float****, Kokkos::LayoutRight, Exec> out1("o1", 1, C, H, W);
    model.forward(lat1, t1, enc1, mask1, out1);
    Kokkos::fence();
    auto h2 = Kokkos::create_mirror_view(out2);
    Kokkos::deep_copy(h2, out2);
    auto h1 = Kokkos::create_mirror_view(out1);
    Kokkos::deep_copy(h1, out1);
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
          EXPECT_NEAR(h1(0, c, i, j), h2(b, c, i, j), 1e-5f)
              << "sample " << b << " differs batched vs single";
  }
}

TEST(MossDitForward, BHalfSmoke) {
  auto cfg = Dit::tiny_test();
  KokkosDNN::MossImageDit::MossImageDitModel<Exec, bhalf> model(cfg);
  model.weights().random_init(5);
  const int B = 1, H = 2, W = 2, S = 4;
  Kokkos::View<bhalf****, Kokkos::LayoutRight, Exec> lat("l", B,
                                                         cfg.latent_channels,
                                                         H, W);
  Kokkos::deep_copy(lat, bhalf(0.25f));
  Kokkos::View<bhalf***, Kokkos::LayoutRight, Exec> enc("e", B, S,
                                                        cfg.text_dim);
  Kokkos::deep_copy(enc, bhalf(-0.125f));
  Kokkos::View<unsigned char**, Kokkos::LayoutRight, Exec> mask("m", B, S);
  Kokkos::deep_copy(mask, (unsigned char)1);
  Kokkos::View<float*, Exec> t("t", B);
  Kokkos::deep_copy(t, 0.3f);
  Kokkos::View<bhalf****, Kokkos::LayoutRight, Exec> out("o", B,
                                                         cfg.latent_channels,
                                                         H, W);
  EXPECT_NO_THROW(model.forward(lat, t, enc, mask, out));
  Kokkos::fence();
  auto ho = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(ho, out);
  for (int c = 0; c < cfg.latent_channels; ++c)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        EXPECT_TRUE(std::isfinite(static_cast<float>(ho(0, c, i, j))));
}

}  // namespace
