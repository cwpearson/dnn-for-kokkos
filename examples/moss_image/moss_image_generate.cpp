// MOSS-Image-SFT text-to-image driver: FlowEuler sampling with CFG + SLG
// around the MOSS example DiT model.
//
//   moss_image_generate [--tiny] [--weights model.safetensors]
//                       [--prompt "text"] [--text-weights feats.f32]
//                       [--prompt-embeds-in feats.f32]
//                       [--height H] [--width W]
//                       [--steps 50] [--cfg 3.0] [--shift 5.0]
//                       [--skip-layers 22] [--seed 42]
//                       [--vae-weights vae.safetensors] [--out img.png]
//                       [--staged [--prompt-embeds-out embeds.f32]]
//                       [--encode-only [--prompt-embeds-out embeds.f32]]
//
// Text conditioning: --text-weights (alias --prompt-embeds-in) points at raw
// float32 LE caption features [L, Cc] (L <= model_max_length,
// Cc == caption_channels); [1,L,Cc] / [1,1,L,Cc] flattened layouts are also
// accepted (last block wins). Without caption features the run uses a null
// (zeros) embedding and the prompt is logged only: the MOSS text encoder is
// not ported yet (see README). --vae-weights is accepted for forward
// compatibility but the MOSS VAE decoder is not ported yet: the driver
// validates the path exists and writes a channel-mean grayscale preview plus
// the raw fp32 latents (<stem>.f32, [C,H,W] row-major) either way.
//
// Staged execution (24GB fit): --encode-only runs just the encoder/embed
// stage and exits before any DiT weight is loaded; --staged runs encode ->
// writes embeds -> releases encoder views -> Kokkos::fence() -> loads the
// DiT -> samples, so the 16.8GB encoder and the 10.2GB DiT are never
// resident at once. Each phase logs its resident estimate (weights bytes +
// activation estimate).
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
#include "KokkosDNN_MossConfig.hpp"
#include "KokkosDNN_MossModel.hpp"
#include "KokkosDNN_MossSampler.hpp"
#include "KokkosDNN_MossWeights.hpp"

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

namespace MO = KokkosDNN::Moss;
using Exec = Kokkos::DefaultExecutionSpace;
using Scalar = MO::DefaultScalar;  // float (the checkpoint dtype)

void save_raw_f32(const std::string& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
  f.write(reinterpret_cast<const char*>(v.data()),
          (long long)(v.size() * sizeof(float)));
  if (!f) throw std::runtime_error("short write to '" + path + "'");
}

// Grayscale preview: channel-mean -> min/max normalize -> u8 -> nearest 8x.
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
  constexpr size_t kUp = 8;  // nominal VAE downsample stand-in (preview only)
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

std::string with_extension(const std::string& path, const std::string& ext) {
  const size_t dot = path.find_last_of('.');
  const size_t sep = path.find_last_of("/\\");
  if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
    return path + ext;
  return path.substr(0, dot) + ext;
}

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

// Encode stage: owns the caption-embedding state that the text encoder
// produces and hands to the DiT sample phase.
//
// Current coverage: the Qwen text encoder itself is NOT ported yet, so
// run() resolves embeddings from (a) a raw float32 LE file
// (--text-weights / --prompt-embeds-in, e.g. Python-produced), or (b) a
// null (zeros) embedding while the prompt is logged only. The load ->
// device-stage -> file-round-trip -> release plumbing around that choice
// is real and demonstrable (see --encode-only / --staged), and stays
// unchanged when the encoder lands.
//
// TODO(qwen-port): inside run(), replace the file/null branch with:
//   1. tokenize(prompt) -> ids[M] (Qwen tokenizer, M <= model_max_length);
//   2. stream text_encoder/*.safetensors into device views via HostSpace
//      chunks + one deep_copy per tensor (same pattern as the DiT loaders
//      in KokkosDNN_MossWeights.cpp; FP32 accumulate, cast to Scalar);
//   3. run the Qwen forward on-device -> hidden [M,4096], then the
//      checkpoint's text refiner (Linear 4096->3072 + 2 _TextBlocks) ->
//      caption features [L,Cc];
//   4. deep_copy the result into embeds/device below.
//   The future port must also call release() + Kokkos::fence() before the
//   DiT load (staged path already does), and extend print_phase_memory's
//   encode line with the ~16.8GB encoder budget (kEncoderBudgetBytes).
struct EncoderStage {
  std::vector<float> embeds;  // host [L*Cc] row-major caption features
  Kokkos::View<float**, Kokkos::LayoutRight, Exec> device;  // [L,Cc] device copy
  size_t L = 0, Cc = 0;

  static EncoderStage run(const MO::Config& cfg, const std::string& prompt,
                          const std::string& caption_path, bool tiny) {
    EncoderStage st;
    st.L = (size_t)cfg.model_max_length;
    st.Cc = (size_t)cfg.caption_channels;
    if (!caption_path.empty()) {
      try {
        st.embeds = MO::load_caption_raw(caption_path, (int)st.L, (int)st.Cc);
        std::printf("[caption %s] %zux%zu features\n", caption_path.c_str(),
                    st.L, st.Cc);
      } catch (const std::exception& e) {
        if (tiny) {
          // A full-geometry file cannot condition the tiny smoke model;
          // fall back to null instead of failing the smoke run.
          std::fprintf(stderr,
                       "warning: caption file '%s' unusable here (%s); using "
                       "null embedding\n",
                       caption_path.c_str(), e.what());
          st.embeds.assign(st.L * st.Cc, 0.0f);
        } else {
          throw;
        }
      }
    } else {
      st.embeds.assign(st.L * st.Cc, 0.0f);
      std::printf("[caption null] prompt \"%s\" has no features here; using "
                  "zeros (%zux%zu)\n",
                  prompt.c_str(), st.L, st.Cc);
    }
    // Device-stage the embeds (HostSpace + single deep_copy), the same
    // handoff the future on-device encoder output will use.
    st.device = Kokkos::View<float**, Kokkos::LayoutRight, Exec>(
        "moss_prompt_embeds", st.L, st.Cc);
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> stage(
        "moss_prompt_embeds_host", st.L, st.Cc);
    std::memcpy(stage.data(), st.embeds.data(),
                st.embeds.size() * sizeof(float));
    Kokkos::deep_copy(st.device, stage);
    Kokkos::fence();
    return st;
  }

  void save(const std::string& path) const { save_raw_f32(path, embeds); }

  size_t device_bytes() const {
    return device.extent(0) * device.extent(1) * sizeof(float);
  }

  // Drop the device view only (host embeds survive for the sample phase).
  void release_device() { device = {}; }

  // Full teardown: mirrors Weights::release (device + host).
  void release() {
    device = {};
    embeds.clear();
    embeds.shrink_to_fit();
    L = Cc = 0;
  }
};

void log_encode_memory(const EncoderStage& st) {
  MO::print_phase_memory("encode", st.device_bytes(), st.device_bytes());
  std::printf("[budget] full Qwen encoder projects to %.1f GB; it is "
              "released before the DiT load, so the staged peak stays under "
              "24GB (see README fit table)\n",
              MO::kEncoderBudgetBytes / 1e9);
  std::fflush(stdout);
}

void usage() {
  std::printf(
      "usage: moss_image_generate [--tiny] [--weights model.safetensors]\n"
      "                    [--prompt \"text\"] [--text-weights feats.f32]\n"
      "                    [--prompt-embeds-in feats.f32]\n"
      "                    [--height H] [--width W]\n"
      "                    [--steps N] [--cfg G] [--shift S]\n"
      "                    [--skip-layers N] [--seed N]\n"
      "                    [--vae-weights vae.safetensors] [--out img.png]\n"
      "                    [--staged [--prompt-embeds-out embeds.f32]]\n"
      "                    [--encode-only [--prompt-embeds-out embeds.f32]]\n"
      "\n"
      "  --tiny         random-weights smoke model, 4x4 latents (no files)\n"
      "  --weights      DiT checkpoint (.safetensors, or torch.save .pth)\n"
      "                 (default: moss_image_sft.safetensors)\n"
      "  --prompt       caption text (logged; needs --text-weights to\n"
      "                 condition the model, else null embedding is used)\n"
      "  --text-weights raw float32 LE caption features [L,Cc]\n"
      "                 (L <= maxlen, Cc == caption_channels)\n"
      "  --prompt-embeds-in alias of --text-weights (wins if both differ)\n"
      "  --prompt-embeds-out write the encode-stage embeds to a raw F32 file\n"
      "  --staged       encode -> write embeds -> release encoder -> fence\n"
      "                 -> load DiT -> sample (encoder/DiT never co-resident)\n"
      "  --encode-only  run the encode/embed stage, optionally write embeds,\n"
      "                 then exit WITHOUT loading DiT weights\n"
      "  --height/--width latent H/W, each a multiple of 16 (default 64)\n"
      "  --steps        Euler steps (default 50)\n"
      "  --cfg/--guidance CFG scale, <=1 disables guidance (default 3.0)\n"
      "  --shift/--flow-shift flow-matching sigma shift (default 5.0)\n"
      "  --skip-layers/--skip-layer SLG layer skipped on the uncond branch\n"
      "                 (-1 disables; default 22)\n"
      "  --seed         latent init seed (default 42)\n"
      "  --vae-weights  reserved (validated, preview written regardless)\n"
      "  --out          PNG with KokkosDNN_ENABLE_IMAGE=ON, else PPM;\n"
      "                 raw F32 latents always written to <stem>.f32\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "moss_image_sft.safetensors";
  std::string prompt = "a red panda reading a newspaper";
  std::string text_weights, prompt_embeds_in, prompt_embeds_out;
  std::string vae_weights, out = "moss_out.png";
  int height = 64, width = 64;
  int steps = 50;
  float guidance = 3.0f, flow_shift = 5.0f;
  int skip_layers = 22;
  uint64_t seed = 42;
  bool tiny = false;
  bool staged = false, encode_only = false;

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
    } else if (a == "--text-weights" && i + 1 < argc) {
      text_weights = argv[++i];
    } else if (a == "--prompt-embeds-in" && i + 1 < argc) {
      prompt_embeds_in = argv[++i];
    } else if (a == "--prompt-embeds-out" && i + 1 < argc) {
      prompt_embeds_out = argv[++i];
    } else if (a == "--vae-weights" && i + 1 < argc) {
      vae_weights = argv[++i];
    } else if (a == "--height" && i + 1 < argc) {
      height = std::stoi(argv[++i]);
    } else if (a == "--width" && i + 1 < argc) {
      width = std::stoi(argv[++i]);
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
    } else if ((a == "--cfg" || a == "--guidance") && i + 1 < argc) {
      guidance = std::stof(argv[++i]);
    } else if ((a == "--shift" || a == "--flow-shift") && i + 1 < argc) {
      flow_shift = std::stof(argv[++i]);
    } else if ((a == "--skip-layers" || a == "--skip-layer") && i + 1 < argc) {
      skip_layers = std::stoi(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      seed = (uint64_t)std::stoull(argv[++i]);
    } else if (a == "--tiny") {
      tiny = true;
    } else if (a == "--staged") {
      staged = true;
    } else if (a == "--encode-only") {
      encode_only = true;
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "warning: ignoring unknown arg '%s'\n", a.c_str());
    }
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    KokkosDNN::print_configuration(std::cout);
    MO::Config cfg = tiny ? MO::Config::tiny_test() : MO::Config::moss_sft();
    const int H = tiny ? 4 : height, W = tiny ? 4 : width;
    if (!tiny) {
      if (H <= 0 || W <= 0 || H % 16 != 0 || W % 16 != 0)
        throw std::runtime_error(
            "--height/--width must be positive multiples of 16 (got " +
            std::to_string(height) + "x" + std::to_string(width) + ")");
      if (H % cfg.patch_size != 0 || W % cfg.patch_size != 0)
        throw std::runtime_error(
            "--height/--width must tile patch_size=" +
            std::to_string(cfg.patch_size));
    }
    const size_t C = (size_t)cfg.in_channels, Cc = (size_t)cfg.caption_channels;
    const size_t L = (size_t)cfg.model_max_length;
    std::printf("[model %s]\n", tiny ? "tiny_test" : "moss-sft");
    if (tiny && skip_layers >= cfg.depth) {
      std::fprintf(stderr,
                   "warning: --skip-layers %d >= tiny depth %d; disabling SLG "
                   "(-1) for the smoke run\n",
                   skip_layers, cfg.depth);
      skip_layers = -1;
    }

    // Resolve the caption source: --prompt-embeds-in is an alias of
    // --text-weights and wins when both are given with different values.
    std::string caption_src = text_weights;
    if (!prompt_embeds_in.empty()) {
      if (!text_weights.empty() && prompt_embeds_in != text_weights)
        std::fprintf(stderr,
                     "warning: --prompt-embeds-in wins over --text-weights\n");
      caption_src = prompt_embeds_in;
    }

    // ---- Phase 1: encode (no DiT weights are loaded yet) ----
    EncoderStage enc = EncoderStage::run(cfg, prompt, caption_src, tiny);
    log_encode_memory(enc);

    std::vector<float> cond;
    if (encode_only) {
      if (!prompt_embeds_out.empty()) {
        enc.save(prompt_embeds_out);
        std::printf("[wrote %s] %zux%zu prompt embeds (raw F32)\n",
                    prompt_embeds_out.c_str(), L, Cc);
      } else {
        std::printf("[encode-only] no --prompt-embeds-out; embeds discarded\n");
      }
      enc.release();
      Kokkos::fence();
      MO::print_phase_memory("encode-released", 0, 0);
      std::printf("[encode-only] exiting before DiT load (encoder and DiT "
                  "never co-resident)\n");
      Kokkos::finalize();
      return 0;
    }

    if (staged && !prompt_embeds_out.empty()) {
      // File handoff: the embeds file is the only state crossing phases.
      enc.save(prompt_embeds_out);
      std::printf("[wrote %s] %zux%zu prompt embeds (raw F32)\n",
                  prompt_embeds_out.c_str(), L, Cc);
      std::printf("[staged] releasing encoder views before DiT load\n");
      enc.release();
      Kokkos::fence();
      MO::print_phase_memory("encode-released", 0, 0);
      cond = MO::load_caption_raw(prompt_embeds_out, (int)L, (int)Cc);
      std::printf("[staged] re-read embeds from %s for the sample phase\n",
                  prompt_embeds_out.c_str());
    } else {
      if (!prompt_embeds_out.empty()) {
        enc.save(prompt_embeds_out);
        std::printf("[wrote %s] %zux%zu prompt embeds (raw F32)\n",
                    prompt_embeds_out.c_str(), L, Cc);
      }
      cond = enc.embeds;
      if (staged) {
        std::printf("[staged] releasing encoder device views before DiT load\n");
        enc.release_device();
        Kokkos::fence();
        MO::print_phase_memory("encode-released", 0, 0);
      } else {
        std::printf("[note] single-phase run: encoder embeds stay live "
                    "across the DiT load (use --staged to release them)\n");
      }
    }

    // ---- Phase 2: DiT load ----
    MO::Weights<Exec, Scalar> w;
    if (tiny) {
      MO::random_init(cfg, w, seed);
      std::printf("[tiny random weights] hidden=%d depth=%d latent=%dx%d\n",
                  cfg.hidden_size, cfg.depth, H, W);
    } else {
      std::printf("[loading %s]\n", weights.c_str());
      if (!file_exists(weights))
        throw std::runtime_error("weights file '" + weights +
                                 "' not found (need a MOSS checkpoint or "
                                 "--tiny). See README for the expected key "
                                 "layout.");
      try {
        if (weights.size() >= 11 &&
            weights.compare(weights.size() - 11, 11, ".safetensors") == 0)
          MO::load_safetensors(cfg, weights, w);
        else
          MO::load_torch_pth(cfg, weights, w);
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) +
                                 " (need MOSS weights or --tiny)");
      }
      std::printf("[loaded] depth=%d hidden=%d\n", cfg.depth, cfg.hidden_size);
    }
    MO::print_phase_memory("sample", w.resident_bytes(),
                           MO::dit_activation_bytes(cfg, H, W,
                                                    sizeof(Scalar)));
    std::printf("[budget] full-checkpoint projection: DiT %.1f GB + VAE %.2f "
                "GB + activations vs 24GB (see README fit table)\n",
                MO::kDitBudgetBytes / 1e9, MO::kVaeBudgetBytes / 1e9);

    std::vector<float> uncond(L * Cc, 0.0f);

    if (!vae_weights.empty()) {
      if (!file_exists(vae_weights))
        throw std::runtime_error("vae weights file '" + vae_weights +
                                 "' not found");
      std::printf("[vae %s] MOSS VAE decode is not ported yet; writing a "
                  "grayscale preview instead\n",
                  vae_weights.c_str());
    }

    std::vector<float> x_init = MO::sample_gaussian(C * (size_t)H * (size_t)W, seed);
    std::printf("[sample] H=%d W=%d steps=%d cfg=%.3g shift=%.3g "
                "skip_layers=%d seed=%llu\n",
                H, W, steps, (double)guidance, (double)flow_shift, skip_layers,
                (unsigned long long)seed);
    MO::FlowEulerConfig s;
    s.steps = steps;
    s.guidance_scale = guidance;
    s.flow_shift = flow_shift;
    s.skip_layers = skip_layers;
    s.seed = seed;
    MO::Model<Exec, Scalar> model(cfg, w);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> lat = MO::run_flow_euler(
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

    std::string lat_path = with_extension(out, ".f32");
    if (!prompt_embeds_out.empty() && lat_path == prompt_embeds_out) {
      // The sample-phase latents dump would otherwise clobber the staged
      // embeds handoff file; redirect it.
      lat_path = with_extension(out, "_latents.f32");
      std::fprintf(stderr,
                   "warning: latents dump would overwrite --prompt-embeds-out; "
                   "redirecting to '%s'\n",
                   lat_path.c_str());
    }
    save_raw_f32(lat_path, lat);
    std::printf("[wrote %s] raw fp32 latents\n", lat_path.c_str());

    size_t oH = 0, oW = 0;
    std::vector<unsigned char> preview =
        render_preview(lat, C, (size_t)H, (size_t)W, oH, oW);
#ifdef KOKKOSDNN_ENABLE_IMAGE
    {
      Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::HostSpace>
          img("moss_preview", oH, oW, 1);
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
