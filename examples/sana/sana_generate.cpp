// Sana text-to-image driver (Task E): FlowEuler sampling around the SanaMS
// DiT example model.
//
//   sana_generate [--tiny] [--weights Sana_600M_1024px/checkpoints/...pth]
//                 [--prompt "text"] [--prompt-embeds feats.f32]
//                 [--steps 18] [--guidance 5.0] [--flow-shift 4.0]
//                 [--seed 42] [--out sana_out.png] [--fp32]
//
// Text encoding: Gemma is NOT bundled, so --prompt alone cannot condition
// the model; without --prompt-embeds the run falls back to the checkpoint's
// null embedding (y_embedding row block = unconditional) and the prompt is
// logged only. --prompt-embeds takes precomputed Gemma caption features as
// raw float32 LE, [1,1,300,2304] (trailing [300,2304] / [1,300,2304] also
// accepted) matching the Sana pipeline's caption_embs layout.
//
// VAE decoding: DC-AE weights are NOT bundled, so there is no latent
// decoder here. The driver writes a decoded PNG *preview* (channel-mean ->
// grayscale, nearest-neighbor 32x to 1024x1024) plus the raw fp32 latents
// (<stem>.f32, [C,H,W] row-major). --vae-safetensors is a stub: passing it
// is an error (documented in README.md).
//
// --tiny runs a random-weights smoke model on 4x4 latents (no downloads).
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_SanaConfig.hpp"
#include "KokkosDNN_SanaModel.hpp"
#include "KokkosDNN_SanaSampler.hpp"
#include "KokkosDNN_SanaWeights.hpp"

#ifdef KOKKOSDNN_ENABLE_IMAGE
#if __has_include("KokkosDNN_Image.hpp")
#include "KokkosDNN_Image.hpp"
#elif __has_include("images/KokkosDNN_Image.hpp")
#include "images/KokkosDNN_Image.hpp"
#else
#error "KOKKOSDNN_ENABLE_IMAGE is set but no KokkosDNN_Image header was found"
#endif
#endif

namespace {

namespace SN = KokkosDNN::Sana;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = SN::DefaultScalar;  // float (the checkpoint dtype)

// Host copy of a [R,C] device matrix, row-major.
std::vector<float> mat_to_host(
    const Kokkos::View<Scalar**, Kokkos::LayoutRight, Exec>& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  std::vector<float> out(h.extent(0) * h.extent(1));
  size_t k = 0;
  for (size_t i = 0; i < h.extent(0); ++i)
    for (size_t j = 0; j < h.extent(1); ++j) out[k++] = (float)h(i, j);
  return out;
}

void save_raw_f32(const std::string& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f.write(reinterpret_cast<const char*>(v.data()),
          (long long)(v.size() * sizeof(float)));
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

// Grayscale preview: channel-mean -> min/max normalize -> u8 -> nearest 32x.
std::vector<unsigned char> render_preview(const std::vector<float>& lat,
                                          size_t C, size_t H, size_t W,
                                          size_t& oH, size_t& oW) {
  std::vector<float> gray(H * W, 0.0f);
  for (size_t h = 0; h < H; ++h)
    for (size_t w = 0; w < W; ++w) {
      double acc = 0.0;
      for (size_t c = 0; c < C; ++c)
        acc += lat[(c * H + h) * W + w];
      gray[h * W + w] = (float)(acc / (double)C);
    }
  float mn = gray[0], mx = gray[0];
  for (float v : gray) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  const float span = (mx > mn) ? (mx - mn) : 1.0f;
  constexpr size_t kUp = 32;  // VAE downsample rate (DC-AE f32c32)
  oH = H * kUp;
  oW = W * kUp;
  std::vector<unsigned char> img(oH * oW);
  for (size_t h = 0; h < oH; ++h)
    for (size_t w = 0; w < oW; ++w) {
      const float v = gray[(h / kUp) * W + (w / kUp)];
      img[h * oW + w] =
          (unsigned char)(std::lround(255.0f * (v - mn) / span));
    }
  return img;
}

void save_ppm(const std::string& path, const std::vector<unsigned char>& gray,
              size_t H, size_t W) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f << "P5\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char*>(gray.data()),
          (long long)gray.size());
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

std::string with_extension(const std::string& path, const std::string& ext) {
  const size_t dot = path.find_last_of('.');
  const size_t sep = path.find_last_of("/\\");
  if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
    return path + ext;
  return path.substr(0, dot) + ext;
}

std::vector<float> read_raw_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open '" + path + "'");
  const long long n = (long long)f.tellg();
  if (n <= 0 || n % 4 != 0)
    throw std::runtime_error("bad float32 file '" + path + "'");
  f.seekg(0, std::ios::beg);
  std::vector<float> out((size_t)n / 4);
  f.read(reinterpret_cast<char*>(out.data()), n);
  if (!f) throw std::runtime_error("short read of '" + path + "'");
  return out;
}

void usage() {
  std::printf(
      "usage: sana_generate [--tiny] [--weights model.pth]\n"
      "                    [--prompt \"text\"] [--prompt-embeds feats.f32]\n"
      "                    [--steps N] [--guidance G] [--flow-shift S]\n"
      "                    [--seed N] [--out img.png] [--fp32]\n"
      "\n"
      "  --tiny         random-weights smoke model, 4x4 latents (no files)\n"
      "  --weights      torch.save checkpoint (default:\n"
      "                 Sana_600M_1024px/checkpoints/"
      "Sana_600M_1024px_MultiLing.pth)\n"
      "  --prompt-embeds raw float32 Gemma features [1,1,300,2304]\n"
      "  --vae-safetensors <f>  unsupported stub (errors out; no DC-AE here)\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights =
      "Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth";
  std::string prompt = "a cyberpunk cat";
  std::string prompt_embeds, out = "sana_out.png", vae_stub;
  int steps = 18;
  float guidance = 5.0f, flow_shift = 4.0f;
  uint64_t seed = 42;
  bool tiny = false, fp32 = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto need = [&](std::string& dst) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
        return false;
      }
      dst = argv[++i];
      return true;
    };
    if (a == "--weights") {
      if (!need(weights)) return 1;
      tiny = false;
    } else if (a == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (a == "--prompt-embeds" && i + 1 < argc) {
      prompt_embeds = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
    } else if (a == "--guidance" && i + 1 < argc) {
      guidance = std::stof(argv[++i]);
    } else if (a == "--flow-shift" && i + 1 < argc) {
      flow_shift = std::stof(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      seed = (uint64_t)std::stoull(argv[++i]);
    } else if (a == "--vae-safetensors" && i + 1 < argc) {
      vae_stub = argv[++i];
    } else if (a == "--tiny") {
      tiny = true;
    } else if (a == "--fp32") {
      fp32 = true;
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "warning: ignoring unknown arg '%s'\n", a.c_str());
    }
  }
  (void)fp32;  // float is the only (and checkpoint-native) dtype.

  if (!vae_stub.empty()) {
    std::fprintf(stderr,
                 "error: --vae-safetensors '%s' is an unsupported stub: this "
                 "example bundles no DC-AE decoder (see README.md). It writes "
                 "a channel-mean preview plus raw latents instead.\n",
                 vae_stub.c_str());
    return 1;
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    KokkosDNN::print_configuration(std::cout);
    SN::Config cfg = tiny ? SN::Config::tiny_test() : SN::Config::sana600m_1024();
    const int H = tiny ? 4 : 32, W = tiny ? 4 : 32;
    const size_t C = (size_t)cfg.in_channels, Cc = (size_t)cfg.caption_channels;
    const size_t L = (size_t)cfg.model_max_length;
    std::printf("[dtype fp32]\n[model %s]\n", tiny ? "tiny_test" : "sana600m_1024");

    SN::Weights<Exec, Scalar> w;
    if (tiny) {
      SN::random_init(cfg, w, seed);
      std::printf("[tiny random weights] hidden=%d depth=%d latent=%dx%d\n",
                  cfg.hidden_size, cfg.depth, H, W);
    } else {
      std::printf("[loading %s]\n", weights.c_str());
      try {
        SN::load_torch_pth(cfg, weights, w);
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) +
                                 " (need the 600M .pth or --tiny)");
      }
      std::printf("[loaded] depth=%d hidden=%d\n", cfg.depth, cfg.hidden_size);
    }

    // Captions: Gemma features when given, else the checkpoint null block.
    std::vector<float> cond, uncond = mat_to_host(w.y_embedding);
    if (!prompt_embeds.empty() && tiny) {
      std::fprintf(stderr,
                   "warning: --prompt-embeds ignored in --tiny mode (caption "
                   "geometry %zux%zu != tiny %zux%zu); using null embedding\n",
                   L, Cc, L, Cc);
      prompt_embeds.clear();
    }
    if (!prompt_embeds.empty()) {
      cond = read_raw_f32(prompt_embeds);
      if (cond.size() != L * Cc)
        throw std::runtime_error(
            "--prompt-embeds '" + prompt_embeds + "' holds " +
            std::to_string(cond.size()) + " floats, expected " +
            std::to_string(L * Cc) + " ([1,1,300,2304] raw float32)");
      std::printf("[caption %s] %zux%zu features\n", prompt_embeds.c_str(), L,
                  Cc);
    } else {
      cond = uncond;
      std::printf("[caption null] prompt \"%s\" has no Gemma encoder here; "
                  "using y_embedding unconditional (%zux%zu)\n",
                  prompt.c_str(), L, Cc);
    }

    std::vector<float> x_init = SN::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    std::printf("[sample] steps=%d guidance=%.3g shift=%.3g seed=%llu\n",
                steps, (double)guidance, (double)flow_shift,
                (unsigned long long)seed);
    SN::FlowEulerConfig s;
    s.steps = steps;
    s.guidance_scale = guidance;
    s.flow_shift = flow_shift;
    s.seed = seed;
    SN::Model<Exec, Scalar> model(cfg, w);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> lat = SN::run_flow_euler(
        model, cfg, x_init, cond, uncond, s, H, W,
        [](int i, float t, float sg) {
          std::printf("  step %3d t=%9.3f sigma=%.6f\n", i, (double)t,
                      (double)sg);
          std::fflush(stdout);
        });
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("[done] %.2fs total, %.2fs/step\n", secs,
                secs / (double)std::max(steps, 1));

    double mean = 0.0, var = 0.0, mn = lat[0], mx = lat[0];
    for (float v : lat) {
      mean += v;
      mn = std::min(mn, (double)v);
      mx = std::max(mx, (double)v);
    }
    mean /= (double)lat.size();
    for (float v : lat) var += (v - mean) * (v - mean);
    var /= (double)lat.size();
    std::printf("[latents %zux%zux%zu] mean=%.6f std=%.6f min=%.4f max=%.4f\n",
                C, (size_t)H, (size_t)W, mean, std::sqrt(var), mn, mx);

    const std::string lat_path = with_extension(out, ".f32");
    save_raw_f32(lat_path, lat);
    std::printf("[wrote %s] raw fp32 latents\n", lat_path.c_str());

    size_t oH = 0, oW = 0;
    std::vector<unsigned char> preview =
        render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
#ifdef KOKKOSDNN_ENABLE_IMAGE
    {
      Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::HostSpace>
          img("sana_preview", oH, oW, 1);
      for (size_t h = 0; h < oH; ++h)
        for (size_t ww = 0; ww < oW; ++ww) img(h, ww, 0) = preview[h * oW + ww];
      KokkosDNN::Image::save_image_png(out, img);
      std::printf("[wrote %s] %zux%zu grayscale preview (PNG)\n", out.c_str(),
                  oH, oW);
    }
#else
    {
      const std::string ppm = with_extension(out, ".ppm");
      if (ppm != out)
        std::printf("[note] built without KokkosDNN_ENABLE_IMAGE: writing PPM "
                    "to %s instead of PNG %s\n",
                    ppm.c_str(), out.c_str());
      save_ppm(ppm, preview, oH, oW);
      std::printf("[wrote %s] %zux%zu grayscale preview (PPM)\n", ppm.c_str(),
                  oH, oW);
    }
#endif
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
