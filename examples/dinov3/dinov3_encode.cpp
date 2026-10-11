#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_Dinov3Config.hpp"
#include "KokkosDNN_Dinov3Model.hpp"
#include "KokkosDNN_Dinov3Weights.hpp"

#ifdef KOKKOSDNN_ENABLE_IMAGE
#if __has_include("KokkosDNN_Image.hpp")
#include "KokkosDNN_Image.hpp"
#elif __has_include("images/KokkosDNN_Image.hpp")
#include "images/KokkosDNN_Image.hpp"
#else
#error "KOKKOSDNN_ENABLE_IMAGE is set but no KokkosDNN_Image header was found"
#endif
#endif

// PoC driver: tiny random model by default (no download); real weights via
// --weights. Image input uses KokkosDNN::Image preprocessing when the build
// enables KOKKOSDNN_ENABLE_IMAGE, otherwise (or when --image is absent) a
// deterministic synthetic gradient image is encoded.
//   dinov3_encode [--tiny] [--config config.json]
//                 [--weights model.safetensors ...] [--image img.jpg]
//                 [--fp32] [--print-tokens N]
//
// --tiny forces the tiny_test() config + random weights (the default when no
// --weights are given). With --weights the vitb16() base is used and the
// checkpoint is loaded via load_safetensors. --config overlays an HF
// config.json onto the selected base when Config::from_json_file is
// available, and is ignored with a warning otherwise. --fp32 selects float
// compute (the Dinov3 DefaultScalar is already float, so today this is a
// no-op kept for parity with llama_generate). --print-tokens N prints the
// first N CLS values (default 8).
namespace D3 = KokkosDNN::Dinov3;

template <class Scalar>
int run_model(D3::Config cfg, const std::vector<std::string>& wfiles,
              bool tiny, const std::string& image_path, int print_tokens) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using ImgR = Kokkos::View<Scalar***, Kokkos::LayoutRight, Exec>;

  D3::Weights<Exec, Scalar> w;
  if (tiny || wfiles.empty()) {
    D3::random_init(cfg, w);
    std::printf("[%srandom weights] hidden=%d layers=%d heads=%d regs=%d\n",
                tiny ? "tiny " : "", cfg.hidden_size, cfg.num_hidden_layers,
                cfg.num_attention_heads, cfg.num_register_tokens);
  } else {
    D3::load_safetensors(cfg, wfiles, w);
    std::printf("[loaded %zu shard(s)]\n", wfiles.size());
  }

  D3::Model<Exec, Scalar> model(cfg, w);

  ImgR pixels;
  bool from_file = false;
#ifdef KOKKOSDNN_ENABLE_IMAGE
  if (!image_path.empty()) {
    KokkosDNN::Image::HostImageF32 hwc =
        KokkosDNN::Image::load_image_f32(image_path, 3);
    Kokkos::View<float***, Kokkos::LayoutRight, Exec> chw =
        KokkosDNN::Image::preprocess_dinov3<Exec>(hwc, cfg.image_size);
    pixels = ImgR("dinov3_pixels", chw.extent(0), chw.extent(1),
                  chw.extent(2));
    if constexpr (std::is_same_v<Scalar, float>) {
      Kokkos::deep_copy(pixels, chw);
    } else {
      auto dst = pixels;
      Kokkos::parallel_for(
          "dinov3_cast_pixels",
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>(
              {0, 0, 0}, {chw.extent(0), chw.extent(1), chw.extent(2)}),
          KOKKOS_LAMBDA(size_t c, size_t y, size_t x) {
            dst(c, y, x) = static_cast<Scalar>(chw(c, y, x));
          });
    }
    from_file = true;
    std::printf("[image %s] CHW [%zu,%zu,%zu]\n", image_path.c_str(),
                pixels.extent(0), pixels.extent(1), pixels.extent(2));
  }
#else
  if (!image_path.empty())
    std::fprintf(stderr,
                 "warning: --image ignored (built without "
                 "KokkosDNN_ENABLE_IMAGE; using synthetic pixels)\n");
#endif
  if (!from_file) {
    // Deterministic synthetic pixels (gradient + per-channel offset), sized
    // from the config so the tiny smoke path needs no downloads at all.
    const size_t C = (size_t)cfg.num_channels, S = (size_t)cfg.image_size;
    pixels = ImgR("dinov3_synthetic", C, S, S);
    auto dst = pixels;
    Kokkos::parallel_for(
        "dinov3_synthetic",
        Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>({0, 0, 0}, {C, S, S}),
        KOKKOS_LAMBDA(size_t c, size_t y, size_t x) {
          float v = static_cast<float>((y * S + x) % 256) * (1.0f / 255.0f) +
                    0.01f * static_cast<float>(c);
          dst(c, y, x) = static_cast<Scalar>(v);
        });
    std::printf("[synthetic pixels] gradient CHW [%zu,%zu,%zu]\n", C, S, S);
  }
  Kokkos::fence();

  auto out = model.forward(pixels);
  Kokkos::fence();
  auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  const size_t T = host.extent(0), D = host.extent(1);
  double n2 = 0.0;
  for (size_t d = 0; d < D; ++d) {
    const double v = static_cast<double>(host(0, d));
    n2 += v * v;
  }
  std::printf("tokens: [T=%zu, D=%zu] (row 0 = CLS, %d registers, grid %zux%zu)\n",
              T, D, cfg.num_register_tokens, model.grid_hp(), model.grid_wp());
  std::printf("CLS L2 norm: %.6f\n", std::sqrt(n2));
  size_t K = print_tokens < 0 ? 0 : (size_t)print_tokens;
  if (K > D) K = D;
  std::printf("CLS[0..%zu]:", K);
  for (size_t k = 0; k < K; ++k)
    std::printf(" %.6f", static_cast<double>(host(0, k)));
  std::printf("\n");
  Kokkos::fence();
  return 0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::vector<std::string> wfiles;
    std::string config_path, image_path;
    bool tiny = true;
    bool fp32 = false;
    int print_tokens = 8;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--weights" && i + 1 < argc) {
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          wfiles.push_back(argv[++i]);
        tiny = false;
      } else if (a == "--tiny") {
        tiny = true;
      } else if (a == "--config" && i + 1 < argc) {
        config_path = argv[++i];
      } else if (a == "--image" && i + 1 < argc) {
        image_path = argv[++i];
      } else if (a == "--fp32") {
        fp32 = true;
      } else if (a == "--print-tokens" && i + 1 < argc) {
        print_tokens = std::stoi(argv[++i]);
      } else if (a == "--help" || a == "-h") {
        std::printf(
            "usage: dinov3_encode [--tiny] [--config config.json]\n"
            "                    [--weights model.safetensors ...] [--image "
            "img.jpg]\n"
            "                    [--fp32] [--print-tokens N]\n");
        Kokkos::finalize();
        return 0;
      } else {
        std::fprintf(stderr, "warning: ignoring unknown arg '%s'\n",
                     a.c_str());
      }
    }

    KokkosDNN::print_configuration(std::cout);
    D3::Config cfg = tiny ? D3::Config::tiny_test() : D3::Config::vitb16();
    std::printf("[model %s]\n", tiny ? "tiny_test" : "vitb16");
    if (!config_path.empty()) {
      // The sibling Config may or may not provide the HF JSON overlay yet;
      // use it when present, warn otherwise (C++20 requires-clause keeps
      // this compiling in both worlds).
      if constexpr (requires {
                      D3::Config::from_json_file(config_path, cfg);
                    }) {
        try {
          cfg = D3::Config::from_json_file(config_path, cfg);
        } catch (const std::exception& e) {
          std::fprintf(stderr, "error: --config '%s': %s\n",
                       config_path.c_str(), e.what());
          Kokkos::finalize();
          return 1;
        }
      } else {
        std::fprintf(stderr,
                     "warning: --config '%s' ignored "
                     "(Config::from_json_file unavailable)\n",
                     config_path.c_str());
      }
    }
    try {
      cfg.validate();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: invalid model config: %s\n", e.what());
      Kokkos::finalize();
      return 1;
    }
    std::printf("[arch] patch=%d hidden=%d layers=%d heads=%d head_dim=%d "
                "mlp=%d regs=%d image=%d channels=%d eps=%g theta=%g\n",
                cfg.patch_size, cfg.hidden_size, cfg.num_hidden_layers,
                cfg.num_attention_heads, cfg.head_dim, cfg.intermediate_size,
                cfg.num_register_tokens, cfg.image_size, cfg.num_channels,
                (double)cfg.layer_norm_eps, (double)cfg.rope_theta);
    int rc;
    if (fp32) {
      std::printf("[dtype fp32]\n");
      rc = run_model<float>(cfg, wfiles, tiny, image_path, print_tokens);
    } else {
      std::printf("[dtype default]\n");
      rc = run_model<D3::DefaultScalar>(cfg, wfiles, tiny, image_path,
                                        print_tokens);
    }
    Kokkos::finalize();
    return rc;
  }
}
