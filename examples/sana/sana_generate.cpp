// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// Sana text-to-image driver (Task E): FlowEuler sampling around the SanaMS
// DiT example model.
//
//   sana_generate [--tiny] [--weights Sana_600M_1024px/checkpoints/...pth]
//                 [--prompt "text"] [--prompt-embeds feats.f32]
//                 [--text-weights gemma-2-2b-it/model.safetensors]
//                 [--tokenizer tokenizer.json] [--text-config config.json]
//                 [--steps 18] [--guidance 5.0] [--flow-shift 4.0]
//                 [--seed S] [--out sana_out.png | sana_out.jpg] [--fp32]
//
// Text encoding: --prompt is encoded with the native Gemma-2-2B-IT text
// encoder (KokkosDNN_SanaTextEncoder.{hpp,cpp}) when --text-weights points
// at a gemma-2-2b-it safetensors file (or directory). The encoder replicates
// the Sana pipeline (tokenize with padding/truncation to 300 tokens,
// Gemma2Model last-hidden states, select_index gather to [300,2304]); the
// unconditional branch encodes "" the same way. Without --text-weights,
// --prompt alone cannot condition the model: the run falls back to the
// checkpoint's null embedding (y_embedding row block = unconditional) and
// the prompt is logged only. --prompt-embeds overrides everything
// (precomputed Gemma caption features as raw float32 LE, [1,1,300,2304];
// trailing [300,2304] / [1,300,2304] also accepted) matching the Sana
// pipeline's caption_embs layout.
//
// VAE decoding: with --vae-safetensors pointing at the diffusers-layout
// DC-AE checkpoint (mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers,
// diffusion_pytorch_model.safetensors), latents [32,32,32] decode to true
// 1024x1024 RGB via KokkosDNN::Sana::VAE (on by default; pass
// --no-vae-decode for the preview). Without it the driver writes a
// decoded image *preview* (channel-mean -> grayscale, nearest-neighbor 32x
// to 1024x1024; PNG, or JPEG when --out ends in .jpg/.jpeg).
//
// --tiny runs a random-weights smoke model on 4x4 latents (no downloads).
#include <Kokkos_Core.hpp>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "KokkosDNN.hpp"
#include "KokkosDNN_SanaConfig.hpp"
#include "KokkosDNN_SanaModel.hpp"
#include "KokkosDNN_SanaSampler.hpp"
#include "KokkosDNN_SanaTextEncoder.hpp"
#include "KokkosDNN_SanaVAE.hpp"
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

[[maybe_unused]] std::string with_extension(const std::string& path,
    const std::string& ext) {
  const size_t dot = path.find_last_of('.');
  const size_t sep = path.find_last_of("/\\");
  if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
    return path + ext;
  return path.substr(0, dot) + ext;
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

#ifndef KOKKOSDNN_ENABLE_IMAGE
// PPM fallback writers: only the non-image build paths (below) use these;
// guard them so image-enabled builds do not see unused functions.
void save_ppm_rgb(const std::string& path,
                  const std::vector<unsigned char>& rgb, size_t H, size_t W) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f << "P6\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char*>(rgb.data()),
          (long long)rgb.size());
  if (!f) throw std::runtime_error("short write to '" + path + "'");
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
#endif

#ifdef KOKKOSDNN_ENABLE_IMAGE
// Output format follows the --out extension: ".jpg"/".jpeg" (case
// insensitive) selects lossy JPEG, anything else selects lossless PNG.
bool out_is_jpeg(const std::string& path) {
  const size_t dot = path.find_last_of('.');
  const size_t sep = path.find_last_of("/\\");
  if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
    return false;
  std::string ext = path.substr(dot);
  for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
  return ext == ".jpg" || ext == ".jpeg";
}
#endif

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
      "                    [--text-weights gemma.safetensors]\n"
      "                    [--tokenizer tok.json] [--text-config cfg.json]\n"
      "                    [--steps N] [--guidance G] [--flow-shift S]\n"
       "                    [--seed N] [--out img.png|img.jpg] [--fp32]\n"
       "                    [--vae-safetensors dc-ae.safetensors]\n"
       "\n"
       "  --tiny         random-weights smoke model, 4x4 latents (no files)\n"
       "  --weights      torch.save checkpoint (default:\n"
       "                 Sana_600M_1024px/checkpoints/"
       "Sana_600M_1024px_MultiLing.pth)\n"
       "  --prompt-embeds raw float32 Gemma features [1,1,300,2304]\n"
       "  --text-weights gemma-2-2b-it .safetensors file (or directory) to\n"
       "                 encode --prompt natively (--gemma-weights is an\n"
       "                 alias); needs tokenizer.json + config.json beside\n"
       "                 it unless --tokenizer/--text-config override them\n"
       "  --tokenizer    tokenizer.json override (--gemma-tokenizer alias)\n"
       "  --text-config  Gemma config.json override\n"
       "  --chi-prompt   complex-human-instruction prefix (Sana chi_prompt)\n"
       "  --vae-safetensors <f>  DC-AE decoder weights (diffusers layout:\n"
       "                 mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers/\n"
       "                 diffusion_pytorch_model.safetensors)\n"
       "  --vae-decode   (default ON) run the real VAE decode when\n"
       "                 --vae-safetensors is given (else a grayscale preview).\n"
       "                 Pass --no-vae-decode for the preview\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights =
      "Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth";
  std::string prompt = "a cyberpunk cat";
  std::string prompt_embeds, out = "sana_out.png", vae_path;
  std::string text_weights, tokenizer_path, text_config_path, chi_prompt;
  int steps = 18;
  float guidance = 5.0f, flow_shift = 4.0f;
  uint64_t seed = 42;
  bool seed_given = false;
  bool tiny = false;
  bool vae_decode = true;
  // --fp32 is accepted for forward compatibility; float is the only (and
  // checkpoint-native) dtype, so the flag is currently a no-op.
  [[maybe_unused]] bool fp32 = false;

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
    } else if ((a == "--text-weights" || a == "--gemma-weights") &&
               i + 1 < argc) {
      text_weights = argv[++i];
    } else if ((a == "--tokenizer" || a == "--gemma-tokenizer") &&
               i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (a == "--text-config" && i + 1 < argc) {
      text_config_path = argv[++i];
    } else if (a == "--chi-prompt" && i + 1 < argc) {
      chi_prompt = argv[++i];
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
      seed_given = true;
    } else if (a == "--vae-safetensors" && i + 1 < argc) {
      vae_path = argv[++i];
    } else if (a == "--vae-decode") {
      vae_decode = true;
    } else if (a == "--no-vae-decode") {
      vae_decode = false;
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

  if (!seed_given) {
    std::random_device rd;
    seed = (uint64_t(rd()) << 32) | uint64_t(rd());
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

    // Captions: --prompt-embeds wins; else the native Gemma-2-2B-IT encoder
    // when --text-weights is given; else the checkpoint null block.
    namespace SGT = KokkosDNN::SanaGemma;
    std::vector<float> cond, uncond = mat_to_host(w.y_embedding);
    if ((!prompt_embeds.empty() || !text_weights.empty()) && tiny) {
      std::fprintf(stderr,
                   "warning: --prompt-embeds/--text-weights ignored in --tiny "
                   "mode (caption geometry %zux%zu != tiny %zux%zu); using "
                   "null embedding\n",
                   L, Cc, L, Cc);
      prompt_embeds.clear();
      text_weights.clear();
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
    } else if (!text_weights.empty()) {
      const std::vector<std::string> gemma_files =
          SGT::collect_safetensors(text_weights);
      std::string gemma_cfg_path = text_config_path.empty()
                                       ? SGT::sibling_file(text_weights,
                                                           "config.json")
                                       : text_config_path;
      std::string gemma_tok_path = tokenizer_path.empty()
                                       ? SGT::sibling_file(text_weights,
                                                           "tokenizer.json")
                                       : tokenizer_path;
      if (gemma_tok_path.empty())
        throw std::runtime_error(
            "--text-weights needs a tokenizer.json beside the weights or via "
            "--tokenizer");
      SGT::Config gc = gemma_cfg_path.empty()
                           ? SGT::Config::gemma2_2b_it()
                           : SGT::Config::from_json_file(gemma_cfg_path);
      if (gc.hidden_size != (int)Cc)
        throw std::runtime_error(
            "Gemma hidden_size " + std::to_string(gc.hidden_size) +
            " != caption_channels " + std::to_string(Cc));
      std::printf("[gemma loading %s + %s]\n", text_weights.c_str(),
                  gemma_tok_path.c_str());
      SGT::Tokenizer gtok(gemma_tok_path);
      SGT::Weights<Exec, Scalar> gw;
      SGT::load_safetensors(gc, gemma_files, gw);
      SGT::Model<Exec, Scalar> gm(gc, gw);
      cond = SGT::encode_prompt(gc, gtok, gm, prompt, (int)L, chi_prompt);
      // Faithful to the pipeline: the unconditional branch is Gemma(""),
      // not the checkpoint null block, once the encoder is available.
      uncond = SGT::encode_prompt(gc, gtok, gm, "", (int)L, "");
      std::printf("[caption gemma] prompt \"%s\" -> %zux%zu features\n",
                  prompt.c_str(), L, Cc);
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

    if (!vae_decode && !vae_path.empty())
      std::printf("[vae] preview requested (--no-vae-decode); skipping the "
                  "real VAE decode of %s\n",
                  vae_path.c_str());
    else if (vae_decode && vae_path.empty())
      std::printf("[vae] VAE decode needs --vae-safetensors (else grayscale "
                  "preview); writing a preview for this run\n");
    if (!vae_path.empty() && vae_decode) {
      // True DC-AE decode: latents [32,H,W] -> RGB 32x (1024x1024 full).
      if (tiny)
        throw std::runtime_error(
            "--vae-safetensors needs the full 32-channel model latents, not "
            "--tiny");
      if (C != 32)
        throw std::runtime_error("--vae-safetensors needs 32-channel latents");
      std::printf("[loading %s]\n", vae_path.c_str());
      SN::VAE::Config vc = SN::VAE::Config::dcae_f32c32_sana();
      SN::VAE::Weights<Exec> vw;
      SN::VAE::load_safetensors(vc, vae_path, vw);
      SN::VAE::Decoder<Exec> vdec(vc, vw);
      const auto t0d = std::chrono::steady_clock::now();
      std::vector<float> rgb =
          vdec.decode(lat, (size_t)H, (size_t)W);  // [3,oH,oW] ~[-1,1]
      const double dsecs = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0d)
                               .count();
      const size_t oH = (size_t)H * 32, oW = (size_t)W * 32;
      std::printf("[vae decode] %.2fs, %zux%zu RGB\n", dsecs, oH, oW);
      std::vector<unsigned char> u8 = SN::VAE::to_u8_rgb(rgb);
      std::vector<unsigned char> hwc(oH * oW * 3);
      for (size_t c = 0; c < 3; ++c)
        for (size_t i = 0; i < oH * oW; ++i)
          hwc[i * 3 + c] = u8[c * oH * oW + i];
#ifdef KOKKOSDNN_ENABLE_IMAGE
      {
        Kokkos::View<unsigned char***, Kokkos::LayoutRight,
                       Kokkos::HostSpace>
            img("sana_rgb", oH, oW, 3);
        for (size_t h = 0; h < oH; ++h)
          for (size_t ww = 0; ww < oW; ++ww)
            for (size_t c = 0; c < 3; ++c)
              img(h, ww, c) = hwc[(h * oW + ww) * 3 + c];
        if (out_is_jpeg(out)) {
          KokkosDNN::Image::save_image_jpg(out, img);
          std::printf("[wrote %s] %zux%zu RGB (JPEG)\n", out.c_str(), oH, oW);
        } else {
          KokkosDNN::Image::save_image_png(out, img);
          std::printf("[wrote %s] %zux%zu RGB (PNG)\n", out.c_str(), oH, oW);
        }
      }
#else
      {
        const std::string ppm = with_extension(out, ".ppm");
        if (ppm != out)
          std::printf("[note] built without KokkosDNN_ENABLE_IMAGE: writing "
                      "PPM to %s instead of %s\n",
                      ppm.c_str(), out.c_str());
        save_ppm_rgb(ppm, hwc, oH, oW);
        std::printf("[wrote %s] %zux%zu RGB (PPM)\n", ppm.c_str(), oH, oW);
      }
#endif
    } else {
      size_t oH = 0, oW = 0;
      std::vector<unsigned char> preview =
          render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
#ifdef KOKKOSDNN_ENABLE_IMAGE
    {
      Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::HostSpace>
          img("sana_preview", oH, oW, 1);
      for (size_t h = 0; h < oH; ++h)
        for (size_t ww = 0; ww < oW; ++ww) img(h, ww, 0) = preview[h * oW + ww];
      if (out_is_jpeg(out)) {
        KokkosDNN::Image::save_image_jpg(out, img);
        std::printf("[wrote %s] %zux%zu grayscale preview (JPEG)\n", out.c_str(),
                    oH, oW);
      } else {
        KokkosDNN::Image::save_image_png(out, img);
        std::printf("[wrote %s] %zux%zu grayscale preview (PNG)\n", out.c_str(),
                    oH, oW);
      }
    }
#else
    {
      const std::string ppm = with_extension(out, ".ppm");
      if (ppm != out)
        std::printf("[note] built without KokkosDNN_ENABLE_IMAGE: writing PPM to %s "
                    "instead of %s\n",
                    ppm.c_str(), out.c_str());
      save_ppm(ppm, preview, oH, oW);
      std::printf("[wrote %s] %zux%zu grayscale preview (PPM)\n", ppm.c_str(),
                  oH, oW);
      }
#endif
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
